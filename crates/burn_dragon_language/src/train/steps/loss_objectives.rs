//! Token regularization, rollout objectives, and forward/TBPTT loss composition.

use super::*;

impl<B: BackendTrait> LanguageTrainModel<B> {
    pub(super) fn logit_entropy_floor_loss(
        &self,
        log_probs: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
    ) -> Option<Tensor<B, 1>> {
        let [batch, time, vocab] = log_probs.shape().dims();
        if batch == 0 || time == 0 || vocab == 0 {
            return None;
        }
        let token_count = batch * time;
        let flat_log_probs = log_probs.reshape([token_count, vocab]);
        let flat_probs = flat_log_probs.clone().exp();
        let weight = self.logit_entropy_floor_weight();
        let target_entropy_bits = self.logit_entropy_floor.target_entropy_bits;
        let marginal_weight = self.logit_marginal_entropy_floor_weight();
        let target_marginal_entropy_bits = self.logit_entropy_floor.target_marginal_entropy_bits;
        let target_coverage_weight = self.logit_target_coverage_weight();
        let mut total = if weight > f32::EPSILON && target_entropy_bits > f32::EPSILON {
            entropy_floor_loss_from_flat_log_probs(
                flat_log_probs.clone(),
                flat_probs.clone(),
                target_entropy_bits,
            )
            .map(|loss| loss.mul_scalar(weight))
        } else {
            None
        };
        let marginal_probs = (marginal_weight > f32::EPSILON
            || target_coverage_weight > f32::EPSILON)
            .then(|| flat_probs.mean_dim(0));
        if marginal_weight > f32::EPSILON
            && target_marginal_entropy_bits > f32::EPSILON
            && let Some(loss) = marginal_entropy_floor_loss_from_marginal(
                marginal_probs
                    .as_ref()
                    .expect("marginal probabilities")
                    .clone(),
                target_marginal_entropy_bits,
            )
            .map(|loss| loss.mul_scalar(marginal_weight))
        {
            total = Some(match total {
                Some(accumulated) => accumulated + loss,
                None => loss,
            });
        }
        if target_coverage_weight > f32::EPSILON
            && let Some(loss) = target_marginal_coverage_loss_from_marginal(
                marginal_probs.expect("marginal probabilities"),
                targets,
                self.logit_entropy_floor.target_coverage_epsilon,
            )
            .map(|loss| loss.mul_scalar(target_coverage_weight))
        {
            total = Some(match total {
                Some(accumulated) => accumulated + loss,
                None => loss,
            });
        }
        total
    }

    pub(super) fn greedy_rollout_entropy_floor_weight(&self) -> f32 {
        Self::scheduled_weight(
            self.greedy_rollout_unlikelihood.enabled,
            self.greedy_rollout_unlikelihood.entropy_floor_weight,
            self.greedy_rollout_unlikelihood.warmup_steps,
            self.greedy_rollout_unlikelihood.ramp_steps,
            self.gradient_scale_step.load(Ordering::Relaxed),
        )
    }

    pub(super) fn greedy_rollout_entropy_floor_loss(
        &self,
        log_probs: Tensor<B, 3>,
    ) -> Option<Tensor<B, 1>> {
        let weight = self.greedy_rollout_entropy_floor_weight();
        let target_entropy_bits = self.greedy_rollout_unlikelihood.target_entropy_bits;
        if weight <= f32::EPSILON || target_entropy_bits <= f32::EPSILON {
            return None;
        }
        entropy_floor_loss_from_log_probs(log_probs, target_entropy_bits)
            .map(|loss| loss.mul_scalar(weight))
    }

    pub(super) fn greedy_rollout_unlikelihood_loss(
        &self,
        clean_inputs: Tensor<B, 2, Int>,
    ) -> Option<Tensor<B, 1>> {
        let step_index = self.gradient_scale_step.load(Ordering::Relaxed);
        let config = &self.greedy_rollout_unlikelihood;
        if config.recovery_only && !self.greedy_rollout_recovery_active.load(Ordering::Relaxed) {
            return None;
        }
        let weight = self.greedy_rollout_unlikelihood_weight();
        let margin_weight = self.greedy_rollout_unlikelihood_margin_weight();
        let cycle_weight = self.greedy_rollout_cycle_weight();
        let cycle_margin_weight = self.greedy_rollout_cycle_margin_weight();
        let entropy_floor_weight = self.greedy_rollout_entropy_floor_weight();
        let recovery_weight = Self::scheduled_weight(
            config.enabled,
            config.recovery_weight,
            config.warmup_steps,
            config.ramp_steps,
            step_index,
        );
        let sequence_recovery_weight = Self::scheduled_weight(
            config.enabled,
            config.sequence_recovery_weight,
            config.warmup_steps,
            config.ramp_steps,
            step_index,
        );
        if (weight <= f32::EPSILON
            && margin_weight <= f32::EPSILON
            && cycle_weight <= f32::EPSILON
            && cycle_margin_weight <= f32::EPSILON
            && recovery_weight <= f32::EPSILON
            && sequence_recovery_weight <= f32::EPSILON
            && entropy_floor_weight <= f32::EPSILON)
            || self.pipeline_enabled()
            || self.model.uses_factorized_language_head()
            || !step_index.is_multiple_of(config.every_steps)
        {
            return None;
        }
        let [batch_size, block_size] = clean_inputs.shape().dims();
        let prompt_batch = batch_size.min(config.batch_prompts.max(1));
        let prompt_tokens = block_size.min(config.prompt_tokens.max(1));
        if prompt_batch == 0 || prompt_tokens == 0 {
            return None;
        }
        let prompt_start =
            rollout_prompt_start(step_index, config.every_steps, block_size, prompt_tokens);
        let prompt = clean_inputs.clone().slice([
            0..prompt_batch,
            prompt_start..(prompt_start + prompt_tokens),
        ]);
        let mut state = self.model.init_state();
        let logits = self.model.forward_with_state(prompt.clone(), &mut state);
        let [_, time, vocab] = logits.shape().dims::<3>();
        if time == 0 || vocab == 0 {
            return None;
        }
        let needs_step_log_probs = weight > f32::EPSILON
            || cycle_weight > f32::EPSILON
            || recovery_weight > f32::EPSILON
            || entropy_floor_weight > f32::EPSILON;
        let needs_step_logits = needs_step_log_probs
            || margin_weight > f32::EPSILON
            || cycle_margin_weight > f32::EPSILON;
        let mut last_logits = logits
            .slice_dim(1, (time - 1)..time)
            .reshape([prompt_batch, vocab]);
        if !needs_step_logits {
            last_logits = last_logits.detach();
            state.detach_in_place();
        }
        let history_tokens = config.history_tokens.max(1);
        let mut history = Vec::with_capacity(history_tokens);
        for offset in 0..prompt_tokens.min(history_tokens) {
            let start = prompt_tokens - 1 - offset;
            history.push(prompt.clone().slice([0..prompt_batch, start..(start + 1)]));
        }
        let mut total_loss: Option<Tensor<B, 1>> = None;
        let mut total_hits: Option<Tensor<B, 1>> = None;
        let mut total_margin: Option<Tensor<B, 1>> = None;
        let mut total_margin_hits: Option<Tensor<B, 1>> = None;
        let mut total_cycle: Option<Tensor<B, 1>> = None;
        let mut total_cycle_hits: Option<Tensor<B, 1>> = None;
        let mut total_cycle_margin: Option<Tensor<B, 1>> = None;
        let mut total_cycle_margin_hits: Option<Tensor<B, 1>> = None;
        let mut total_recovery: Option<Tensor<B, 1>> = None;
        let mut recovery_steps = 0usize;
        let mut generated_tokens = Vec::with_capacity(config.rollout_tokens);
        let mut total_entropy_floor: Option<Tensor<B, 1>> = None;
        let mut entropy_floor_steps = 0usize;
        for rollout_index in 0..config.rollout_tokens {
            let step_logits =
                needs_step_logits.then(|| last_logits.clone().reshape([prompt_batch, 1, vocab]));
            let step_log_probs = needs_step_log_probs.then(|| {
                log_probs_from_logits(
                    step_logits
                        .as_ref()
                        .expect("step logits are required for rollout log-probs")
                        .clone(),
                )
            });
            if let Some(entropy_loss) = step_log_probs
                .as_ref()
                .and_then(|log_probs| self.greedy_rollout_entropy_floor_loss(log_probs.clone()))
            {
                total_entropy_floor = Some(match total_entropy_floor {
                    Some(accumulated) => accumulated + entropy_loss,
                    None => entropy_loss,
                });
                entropy_floor_steps = entropy_floor_steps.saturating_add(1);
            }
            let next = last_logits.clone().argmax(1).reshape([prompt_batch, 1]);
            let mut repeat_mask = next.clone().equal(
                history
                    .first()
                    .expect("greedy rollout history should not be empty")
                    .clone(),
            );
            for previous in history.iter().skip(1) {
                repeat_mask = repeat_mask.bool_or(next.clone().equal(previous.clone()));
            }
            let repeat_mask = repeat_mask.int();
            let cycle_mask =
                cycle_repeat_mask(&next, &history, config.cycle_min_lag, config.cycle_max_lag);
            if weight > f32::EPSILON {
                let next_log_probs = selected_token_log_probs(
                    step_log_probs
                        .as_ref()
                        .expect("step log-probs are required for rollout unlikelihood")
                        .clone(),
                    next.clone(),
                );
                let next_prob = next_log_probs
                    .exp()
                    .clamp_min(0.0)
                    .clamp_max(1.0 - config.epsilon);
                let unlikelihood = next_prob
                    .mul_scalar(-1.0)
                    .add_scalar(1.0)
                    .clamp_min(config.epsilon)
                    .log()
                    .mul_scalar(-1.0);
                let repeat_weight = repeat_mask.clone().float();
                let step_loss = (unlikelihood * repeat_weight.clone()).sum().reshape([1]);
                let step_hits = repeat_weight.sum().reshape([1]);
                total_loss = Some(match total_loss {
                    Some(accumulated) => accumulated + step_loss,
                    None => step_loss,
                });
                total_hits = Some(match total_hits {
                    Some(accumulated) => accumulated + step_hits,
                    None => step_hits,
                });
            }
            if cycle_weight > f32::EPSILON
                && let Some(cycle_mask) = cycle_mask.clone()
            {
                let next_log_probs = selected_token_log_probs(
                    step_log_probs
                        .as_ref()
                        .expect("step log-probs are required for rollout cycle unlikelihood")
                        .clone(),
                    next.clone(),
                );
                let next_prob = next_log_probs
                    .exp()
                    .clamp_min(0.0)
                    .clamp_max(1.0 - config.epsilon);
                let unlikelihood = next_prob
                    .mul_scalar(-1.0)
                    .add_scalar(1.0)
                    .clamp_min(config.epsilon)
                    .log()
                    .mul_scalar(-1.0);
                let cycle_weight_tensor = cycle_mask.float();
                let step_cycle = (unlikelihood * cycle_weight_tensor.clone())
                    .sum()
                    .reshape([1]);
                let step_hits = cycle_weight_tensor.sum().reshape([1]);
                total_cycle = Some(match total_cycle {
                    Some(accumulated) => accumulated + step_cycle,
                    None => step_cycle,
                });
                total_cycle_hits = Some(match total_cycle_hits {
                    Some(accumulated) => accumulated + step_hits,
                    None => step_hits,
                });
            }
            if margin_weight > f32::EPSILON {
                let repeat_weight = repeat_mask.float();
                let step_logits = step_logits
                    .as_ref()
                    .expect("step logits are required for rollout margin");
                let next_logits = selected_token_logits(step_logits.clone(), next.clone());
                let mean_logits = step_logits.clone().mean_dim(2).reshape([prompt_batch, 1]);
                let margin_penalty =
                    activation::softplus(next_logits - mean_logits + config.margin, 1.0);
                let step_margin = (margin_penalty * repeat_weight.clone()).sum().reshape([1]);
                let step_hits = repeat_weight.sum().reshape([1]);
                total_margin = Some(match total_margin {
                    Some(accumulated) => accumulated + step_margin,
                    None => step_margin,
                });
                total_margin_hits = Some(match total_margin_hits {
                    Some(accumulated) => accumulated + step_hits,
                    None => step_hits,
                });
            }
            if cycle_margin_weight > f32::EPSILON
                && let Some(cycle_mask) = cycle_mask
            {
                let cycle_weight_tensor = cycle_mask.float();
                let step_logits = step_logits
                    .as_ref()
                    .expect("step logits are required for rollout cycle margin");
                let next_logits = selected_token_logits(step_logits.clone(), next.clone());
                let mean_logits = step_logits.clone().mean_dim(2).reshape([prompt_batch, 1]);
                let margin_penalty =
                    activation::softplus(next_logits - mean_logits + config.margin, 1.0);
                let step_margin = (margin_penalty * cycle_weight_tensor.clone())
                    .sum()
                    .reshape([1]);
                let step_hits = cycle_weight_tensor.sum().reshape([1]);
                total_cycle_margin = Some(match total_cycle_margin {
                    Some(accumulated) => accumulated + step_margin,
                    None => step_margin,
                });
                total_cycle_margin_hits = Some(match total_cycle_margin_hits {
                    Some(accumulated) => accumulated + step_hits,
                    None => step_hits,
                });
            }
            let target_pos = prompt_start + prompt_tokens + rollout_index;
            if recovery_weight > f32::EPSILON && target_pos < block_size {
                let recovery_target = clean_inputs
                    .clone()
                    .slice([0..prompt_batch, target_pos..(target_pos + 1)]);
                let recovery_loss = selected_token_log_probs(
                    step_log_probs
                        .as_ref()
                        .expect("step log-probs are required for rollout recovery")
                        .clone(),
                    recovery_target,
                )
                .mul_scalar(-1.0)
                .mean()
                .reshape([1]);
                total_recovery = Some(match total_recovery {
                    Some(accumulated) => accumulated + recovery_loss,
                    None => recovery_loss,
                });
                recovery_steps = recovery_steps.saturating_add(1);
            }
            generated_tokens.push(next.clone());
            let logits = self.model.forward_with_state(next.clone(), &mut state);
            let [_, time, vocab] = logits.shape().dims::<3>();
            if time == 0 || vocab == 0 {
                break;
            }
            last_logits = logits
                .slice_dim(1, (time - 1)..time)
                .reshape([prompt_batch, vocab]);
            if !needs_step_logits {
                last_logits = last_logits.detach();
                state.detach_in_place();
            }
            history.insert(0, next);
            if history.len() > history_tokens {
                history.pop();
            }
        }
        let mut loss = total_loss.map(|loss| {
            loss.div(
                total_hits
                    .expect("greedy rollout hit accumulator should exist")
                    .clamp_min(1.0),
            )
            .mul_scalar(weight)
        });
        if let Some(margin) = total_margin {
            let margin = margin
                .div(
                    total_margin_hits
                        .expect("greedy rollout margin hit accumulator should exist")
                        .clamp_min(1.0),
                )
                .mul_scalar(margin_weight);
            loss = Some(match loss {
                Some(accumulated) => accumulated + margin,
                None => margin,
            });
        }
        if let Some(cycle) = total_cycle {
            let cycle = cycle
                .div(
                    total_cycle_hits
                        .expect("greedy rollout cycle hit accumulator should exist")
                        .clamp_min(1.0),
                )
                .mul_scalar(cycle_weight);
            loss = Some(match loss {
                Some(accumulated) => accumulated + cycle,
                None => cycle,
            });
        }
        if let Some(cycle_margin) = total_cycle_margin {
            let cycle_margin = cycle_margin
                .div(
                    total_cycle_margin_hits
                        .expect("greedy rollout cycle margin hit accumulator should exist")
                        .clamp_min(1.0),
                )
                .mul_scalar(cycle_margin_weight);
            loss = Some(match loss {
                Some(accumulated) => accumulated + cycle_margin,
                None => cycle_margin,
            });
        }
        if recovery_steps > 0
            && let Some(recovery) = total_recovery
        {
            let recovery = recovery.mul_scalar(recovery_weight / recovery_steps as f32);
            loss = Some(match loss {
                Some(accumulated) => accumulated + recovery,
                None => recovery,
            });
        }
        if sequence_recovery_weight > f32::EPSILON
            && !generated_tokens.is_empty()
            && prompt_start + prompt_tokens < block_size
        {
            let available_targets = generated_tokens
                .len()
                .min(block_size - prompt_start - prompt_tokens);
            if available_targets > 0 {
                let generated = Tensor::cat(
                    generated_tokens
                        .into_iter()
                        .take(available_targets)
                        .collect(),
                    1,
                );
                let recovery_inputs = Tensor::cat(vec![prompt.clone(), generated], 1);
                let recovery_logits = self.model.forward(recovery_inputs);
                let [_, recovery_time, recovery_vocab] = recovery_logits.shape().dims::<3>();
                let logit_start = prompt_tokens.saturating_sub(1);
                let logit_end = (logit_start + available_targets).min(recovery_time);
                let used_targets = logit_end.saturating_sub(logit_start);
                if used_targets > 0 && recovery_vocab > 0 {
                    let recovery_targets = clean_inputs.clone().slice([
                        0..prompt_batch,
                        (prompt_start + prompt_tokens)
                            ..(prompt_start + prompt_tokens + used_targets),
                    ]);
                    let recovery_log_probs = log_probs_from_logits(recovery_logits.slice([
                        0..prompt_batch,
                        logit_start..logit_end,
                        0..recovery_vocab,
                    ]));
                    let sequence_recovery =
                        selected_token_log_probs(recovery_log_probs, recovery_targets)
                            .mul_scalar(-1.0)
                            .mean()
                            .reshape([1])
                            .mul_scalar(sequence_recovery_weight);
                    loss = Some(match loss {
                        Some(accumulated) => accumulated + sequence_recovery,
                        None => sequence_recovery,
                    });
                }
            }
        }
        if entropy_floor_steps > 0
            && let Some(entropy_floor) = total_entropy_floor
        {
            let entropy_floor = entropy_floor.mul_scalar(1.0 / entropy_floor_steps as f32);
            loss = Some(match loss {
                Some(accumulated) => accumulated + entropy_floor,
                None => entropy_floor,
            });
        }
        loss
    }

    pub(super) fn corrupt_causal_inputs(&self, inputs: Tensor<B, 2, Int>) -> Tensor<B, 2, Int> {
        let probability = self.causal_input_corruption_probability();
        if probability <= f32::EPSILON {
            return inputs;
        }
        let shape = inputs.shape();
        let device = inputs.device();
        let mask = Tensor::<B, 2>::random(
            shape.clone(),
            TensorDistribution::Uniform(0.0, 1.0),
            &device,
        )
        .lower_elem(probability);
        let replacements = if let Some(token_id) = self.input_corruption.replacement_token_id {
            Tensor::<B, 2, Int>::full(shape, i64::from(token_id), &device)
        } else {
            let vocab_size = self.input_vocab_size.max(1);
            Tensor::<B, 2>::random(
                shape,
                TensorDistribution::Uniform(0.0, vocab_size as f64),
                &device,
            )
            .clamp_min(0.0)
            .clamp_max(vocab_size.saturating_sub(1) as f32)
            .int()
        };
        inputs.mask_where(mask, replacements)
    }

    pub(super) fn truncate_reprompt_tokens(
        mut tokens: Vec<i64>,
        max_len: usize,
        truncation: RepromptTruncation,
    ) -> Vec<i64> {
        if tokens.len() <= max_len {
            return tokens;
        }
        match truncation {
            RepromptTruncation::Right => tokens.split_off(tokens.len() - max_len),
            RepromptTruncation::Left => {
                tokens.truncate(max_len);
                tokens
            }
            RepromptTruncation::Error => {
                panic!(
                    "teacher-conditioned reprompt length {} exceeds max_reprompt_len {}",
                    tokens.len(),
                    max_len
                )
            }
        }
    }

    pub(super) fn rollout_score_batch(
        &self,
        generator_model: &DragonModel<B>,
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
        config: RolloutScoreConfig,
    ) -> ObjectiveScoreBatch<B> {
        let [batch_size, block_size] = inputs.shape().dims();
        let device = inputs.device();
        let completion_len = config
            .max_completion_tokens
            .max(1)
            .min(block_size.saturating_sub(1).max(1));
        let prompt_len = block_size.saturating_sub(completion_len).max(1);
        let score_len = prompt_len + completion_len - 1;
        let group_size = config.group_size.max(1);

        let input_tokens = inputs
            .to_data()
            .convert::<i64>()
            .into_vec::<i64>()
            .expect("objective rollout inputs to host tokens");
        let target_tokens = targets
            .to_data()
            .convert::<i64>()
            .into_vec::<i64>()
            .expect("objective rollout targets to host tokens");

        let total_rows = batch_size * group_size;
        let mut student_inputs = Vec::with_capacity(total_rows * score_len);
        let mut student_targets = Vec::with_capacity(total_rows * score_len);
        let mut teacher_inputs = Vec::with_capacity(total_rows * score_len);
        let mut teacher_targets = Vec::with_capacity(total_rows * score_len);
        let mut mask = Vec::with_capacity(total_rows * score_len);

        for batch_idx in 0..batch_size {
            let row_start = batch_idx * block_size;
            let prompt = input_tokens[row_start..row_start + prompt_len].to_vec();
            let completion_start = prompt_len.saturating_sub(1);
            let golden_completion = target_tokens
                [row_start + completion_start..row_start + completion_start + completion_len]
                .to_vec();
            for _ in 0..group_size {
                let generated = crate::generation::generate_tokens(
                    generator_model,
                    prompt.clone(),
                    &device,
                    crate::generation::GenerationSettings {
                        max_new_tokens: Some(completion_len),
                        temperature: config.temperature,
                        top_k: config.top_k,
                        strategy: crate::generation::ContextStrategy::Infinite,
                        stop_on_token: None,
                    },
                    None,
                )
                .expect("objective rollout generation should succeed");
                let completion = generated[prompt_len..prompt_len + completion_len].to_vec();
                let mut teacher_sequence = prompt.clone();
                teacher_sequence.extend_from_slice(&golden_completion);
                teacher_sequence.extend_from_slice(&completion);
                let teacher_sequence = Self::truncate_reprompt_tokens(
                    teacher_sequence,
                    config.max_reprompt_len.max(score_len + 1),
                    config.reprompt_truncation,
                );

                student_inputs.extend_from_slice(&generated[..score_len]);
                student_targets.extend_from_slice(&generated[1..score_len + 1]);
                teacher_inputs.extend_from_slice(
                    &teacher_sequence
                        [teacher_sequence.len() - (score_len + 1)..teacher_sequence.len() - 1],
                );
                teacher_targets.extend_from_slice(
                    &teacher_sequence[teacher_sequence.len() - score_len..teacher_sequence.len()],
                );
                let loss_start = prompt_len.saturating_sub(1)
                    + config.num_loss_tokens_to_skip.min(completion_len);
                for position in 0..score_len {
                    mask.push((position >= loss_start) as i64);
                }
            }
        }

        ObjectiveScoreBatch {
            student_inputs: Tensor::<B, 2, Int>::from_data(
                TensorData::new(student_inputs, [total_rows, score_len]),
                &device,
            ),
            student_targets: Tensor::<B, 2, Int>::from_data(
                TensorData::new(student_targets, [total_rows, score_len]),
                &device,
            ),
            teacher_inputs: Tensor::<B, 2, Int>::from_data(
                TensorData::new(teacher_inputs, [total_rows, score_len]),
                &device,
            ),
            teacher_targets: Tensor::<B, 2, Int>::from_data(
                TensorData::new(teacher_targets, [total_rows, score_len]),
                &device,
            ),
            mask: Tensor::<B, 2, Int>::from_data(
                TensorData::new(mask, [total_rows, score_len]),
                &device,
            ),
        }
    }

    pub(super) fn objective_loss(
        &self,
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
    ) -> Tensor<B, 1>
    where
        B: AutodiffBackend,
    {
        assert!(
            !(self.pipeline_enabled() && self.tbptt_persist_across_steps),
            "pipeline objective execution does not support persistent stream state"
        );
        self.assert_flat_logits_for_rollout_objective();
        match &self.objective {
            TrainingObjectiveConfig::NextToken => unreachable!("next_token uses the CE fast path"),
            TrainingObjectiveConfig::Sdft(config) => self.sdft_loss(inputs, targets, config),
            TrainingObjectiveConfig::Sdpo(config) => self.sdpo_loss(inputs, targets, config),
            TrainingObjectiveConfig::SdftSdpo(config) => {
                self.composite_sdft_sdpo_loss(inputs, targets, config)
            }
        }
    }

    pub(super) fn sdft_loss(
        &self,
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
        config: &SdftObjectiveConfig,
    ) -> Tensor<B, 1>
    where
        B: AutodiffBackend,
    {
        let teacher = self.current_teacher_model();
        let generator_model = if config.generate_from_teacher {
            &teacher
        } else {
            &self.model
        };
        let rollout = self.rollout_score_batch(
            generator_model,
            inputs,
            targets,
            RolloutScoreConfig {
                max_completion_tokens: config.max_completion_tokens,
                group_size: 1,
                temperature: config.temperature,
                top_k: config.top_k,
                num_loss_tokens_to_skip: config.num_loss_tokens_to_skip,
                max_reprompt_len: usize::MAX,
                reprompt_truncation: RepromptTruncation::Right,
            },
        );
        let student_hidden = self.forward_hidden_for_objective(rollout.student_inputs);
        let teacher_hidden = teacher.forward_hidden(rollout.teacher_inputs);
        self_distillation_loss_from_logits(
            self.model.logits_from_hidden(student_hidden),
            teacher.logits_from_hidden(teacher_hidden).detach(),
            Some(rollout.mask),
            config.kl,
        )
    }

    pub(super) fn sdpo_loss(
        &self,
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
        config: &SdpoObjectiveConfig,
    ) -> Tensor<B, 1>
    where
        B: AutodiffBackend,
    {
        let teacher = self.current_teacher_model();
        let rollout = self.rollout_score_batch(
            &self.model,
            inputs,
            targets,
            RolloutScoreConfig {
                max_completion_tokens: config.max_completion_tokens,
                group_size: config.group_size,
                temperature: config.temperature,
                top_k: config.top_k,
                num_loss_tokens_to_skip: 0,
                max_reprompt_len: config.max_reprompt_len,
                reprompt_truncation: config.reprompt_truncation,
            },
        );
        let mask = rollout.mask;
        let student_hidden = self.forward_hidden_for_objective(rollout.student_inputs);
        let teacher_hidden = teacher.forward_hidden(rollout.teacher_inputs);
        let student_logits = self.model.logits_from_hidden(student_hidden);
        let teacher_logits = teacher.logits_from_hidden(teacher_hidden).detach();
        let student_log_probs = log_probs_from_logits(student_logits);
        let teacher_log_probs = log_probs_from_logits(teacher_logits);
        let new_token_log_probs =
            selected_token_log_probs(student_log_probs.clone(), rollout.student_targets);
        let old_token_log_probs = new_token_log_probs.clone().detach();
        let mut per_token_loss = self_distillation_per_token_from_log_probs(
            student_log_probs,
            teacher_log_probs,
            SelfDistillationKlKind::from_sdpo_alpha(config.alpha),
        );
        if let Some(max_ratio) = config.is_clip.filter(|value| *value > 0.0) {
            let log_ratio = (new_token_log_probs - old_token_log_probs)
                .clamp_min(-20.0)
                .clamp_max(20.0);
            let ratio = log_ratio.exp().clamp_max(max_ratio);
            per_token_loss = per_token_loss * ratio;
        }
        masked_token_mean(per_token_loss, Some(mask))
    }

    pub(super) fn composite_sdft_sdpo_loss(
        &self,
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
        config: &SdftSdpoObjectiveConfig,
    ) -> Tensor<B, 1>
    where
        B: AutodiffBackend,
    {
        let sdft_weight = config.sdft_weight.max(0.0);
        let sdpo_weight = config.sdpo_weight.max(0.0);
        let weight_sum = (sdft_weight + sdpo_weight).max(1.0e-6);
        self.sdft_loss(inputs.clone(), targets.clone(), &config.sdft)
            .mul_scalar(sdft_weight / weight_sum)
            + self
                .sdpo_loss(inputs, targets, &config.sdpo)
                .mul_scalar(sdpo_weight / weight_sum)
    }

    pub(super) fn forward_loss_with_pipeline(
        &self,
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
        loss_mask: Option<Tensor<B, 2, Int>>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> (Tensor<B, 1>, Tensor<B, 3>, Tensor<B, 3>) {
        let plan = self
            .pipeline_plan
            .as_ref()
            .expect("forward_loss_with_pipeline requires a pipeline plan");
        assert!(
            !self.tbptt_persist_across_steps,
            "pipeline execution does not support tbptt_persist_across_steps"
        );
        assert!(
            self.tbptt_chunk_size.is_none(),
            "pipeline execution does not support tbptt chunking"
        );

        let [batch_size, _block_size] = inputs.shape().dims();
        let ranges = split_microbatch_ranges(batch_size, plan.microbatches)
            .expect("pipeline execution requires batch_size >= microbatches");
        let chunk_inputs = ranges
            .iter()
            .map(|range| Self::slice_batch(inputs.clone(), range.start, range.end))
            .collect::<Vec<_>>();
        let chunk_targets = ranges
            .iter()
            .map(|range| Self::slice_batch(targets.clone(), range.start, range.end))
            .collect::<Vec<_>>();
        let chunk_loss_masks = ranges
            .iter()
            .map(|range| {
                loss_mask
                    .clone()
                    .map(|mask| Self::slice_batch(mask, range.start, range.end))
            })
            .collect::<Vec<_>>();
        let chunk_masks = ranges
            .iter()
            .map(|range| {
                summary_event_mask
                    .clone()
                    .map(|mask| Self::slice_batch(mask, range.start, range.end))
            })
            .collect::<Vec<_>>();
        let factorized_head = self.model.uses_factorized_language_head();

        let mut chunk_states = (0..plan.microbatches)
            .map(|_| self.model.init_state_ephemeral())
            .collect::<Vec<_>>();
        let mut pipeline_states = vec![None; plan.microbatches];

        for event in plan.events.iter().filter(|event| {
            matches!(
                event.kind,
                burn_dragon_train::train::pipeline::PipelineEventKind::Forward
            )
        }) {
            let microbatch_id = event.microbatch_id;
            if pipeline_states[microbatch_id].is_none() {
                pipeline_states[microbatch_id] = Some(
                    self.model
                        .begin_language_pipeline(chunk_inputs[microbatch_id].clone()),
                );
            }
            let assignment = plan.assignment(event.virtual_stage_id).clone();
            let state = &mut chunk_states[microbatch_id];
            let stage_state = pipeline_states[microbatch_id]
                .take()
                .expect("microbatch stage state");
            pipeline_states[microbatch_id] =
                Some(self.model.forward_language_pipeline_stage_with_state(
                    stage_state,
                    state,
                    assignment.layer_range.clone(),
                    chunk_masks[microbatch_id].clone(),
                ));
        }

        let mut total_loss: Option<Tensor<B, 1>> = None;
        let mut hidden_chunks = Vec::with_capacity(plan.microbatches);
        let mut logits_chunks = Vec::with_capacity(plan.microbatches);
        for microbatch_id in 0..plan.microbatches {
            let hidden = self.model.finish_language_pipeline_hidden_with_state(
                pipeline_states[microbatch_id]
                    .take()
                    .expect("pipeline state after scheduled forward"),
                &mut chunk_states[microbatch_id],
            );
            let weight = ranges[microbatch_id].len() as f32 / batch_size as f32;
            let chunk_loss = self
                .language_loss_from_hidden(
                    hidden.clone(),
                    chunk_targets[microbatch_id].clone(),
                    chunk_loss_masks[microbatch_id].clone(),
                )
                .mul_scalar(weight);
            total_loss = Some(match total_loss {
                Some(accumulated) => accumulated + chunk_loss,
                None => chunk_loss,
            });
            if !factorized_head {
                logits_chunks.push(self.model.logits_from_hidden(hidden.clone()));
            }
            hidden_chunks.push(hidden);
        }

        (
            total_loss.expect("pipeline forward should produce at least one microbatch loss"),
            Tensor::cat(hidden_chunks, 0),
            if logits_chunks.is_empty() {
                let device = inputs.device();
                Tensor::<B, 3>::zeros([batch_size, 0, 1], &device)
            } else {
                Tensor::cat(logits_chunks, 0)
            },
        )
    }

    pub(super) fn forward_loss_with_tbptt(
        &self,
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
        chunk_size: usize,
        state: &mut ModelState<B>,
    ) -> (Tensor<B, 1>, u128) {
        let [batch_size, block_size] = inputs.shape().dims();
        debug_assert!(chunk_size > 0 && chunk_size < block_size);

        let mut total_loss: Option<Tensor<B, 1>> = None;
        let mut total_forward_ns = 0u128;

        for start in (0..block_size).step_by(chunk_size) {
            let end = (start + chunk_size).min(block_size);
            let chunk_inputs = Self::slice_tokens(inputs.clone(), batch_size, start, end);
            let chunk_targets = Self::slice_tokens(targets.clone(), batch_size, start, end);
            let chunk_summary_event_mask = summary_event_mask
                .clone()
                .map(|mask| Self::slice_tokens(mask, batch_size, start, end));

            let chunk_forward_start = Instant::now();
            let logits = if let Some(mask) = chunk_summary_event_mask {
                self.model
                    .forward_with_state_and_summary_event_mask(chunk_inputs, mask, state)
            } else {
                self.model.forward_with_state(chunk_inputs, state)
            };
            total_forward_ns += chunk_forward_start.elapsed().as_nanos();

            let chunk_weight = (end - start) as f32 / block_size as f32;
            let chunk_loss =
                language_model_loss::<B>(logits, chunk_targets).mul_scalar(chunk_weight);
            total_loss = Some(match total_loss {
                Some(accumulated) => accumulated + chunk_loss,
                None => chunk_loss,
            });

            if end < block_size {
                state.detach_in_place();
            }
        }

        (
            total_loss.expect("tbptt forward should produce at least one loss chunk"),
            total_forward_ns,
        )
    }
}
