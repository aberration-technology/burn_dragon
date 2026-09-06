//! Ruliad answer contracts, hard negatives, policy utilities, and field binding.

use super::*;

impl<B: BackendTrait> LanguageTrainModel<B> {
    pub(super) fn ruliad_answer_ranking_weight(&self) -> f32 {
        let config = self.ruliad_supervision.answer_ranking;
        if config.enabled {
            config.weight.max(0.0)
        } else {
            0.0
        }
    }

    pub(super) fn ruliad_answer_ranking_loss_from_logits(
        &self,
        logits: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
        loss_mask: Option<Tensor<B, 2, Int>>,
    ) -> Option<Tensor<B, 1>> {
        let config = self.ruliad_supervision.answer_ranking;
        let weight = self.ruliad_answer_ranking_weight();
        if weight <= f32::EPSILON {
            return None;
        }
        let mask = loss_mask?;
        let [batch, time, vocab] = logits.shape().dims();
        if batch == 0 || time == 0 || vocab <= 1 {
            return None;
        }
        let offset = (config.corrupt_offset % vocab as i64).max(1);
        let corrupt_targets = targets
            .clone()
            .add_scalar(offset)
            .remainder_scalar(vocab as i64);
        let oracle_logits = selected_token_logits(logits.clone(), targets);
        let corrupt_logits = selected_token_logits(logits, corrupt_targets);
        let penalty =
            activation::softplus(corrupt_logits - oracle_logits + config.margin.max(0.0), 1.0);
        Some(masked_token_mean(penalty, Some(mask)).mul_scalar(weight))
    }

    pub(super) fn ruliad_answer_denoising_weight(&self) -> f32 {
        let config = self.ruliad_supervision.answer_denoising;
        if config.enabled {
            config.weight.max(0.0)
        } else {
            0.0
        }
    }

    pub(super) fn ruliad_structured_answer_recovery_weight(&self) -> f32 {
        let config = self.ruliad_supervision.answer_denoising;
        if !config.enabled
            || config.structured_recovery_weight <= f32::EPSILON
            || config.structured_recovery_every_steps == 0
        {
            return 0.0;
        }
        let step_index = self.gradient_scale_step.load(Ordering::Relaxed);
        if step_index < config.structured_recovery_start_after_steps {
            return 0.0;
        }
        if !step_index.is_multiple_of(config.structured_recovery_every_steps) {
            return 0.0;
        }
        config.structured_recovery_weight
    }

    pub(super) fn ruliad_answer_denoising_loss(
        &self,
        clean_inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
        loss_mask: Option<Tensor<B, 2, Int>>,
    ) -> Option<Tensor<B, 1>> {
        let config = self.ruliad_supervision.answer_denoising;
        let weight = self.ruliad_answer_denoising_weight();
        if weight <= f32::EPSILON || self.pipeline_enabled() {
            return None;
        }
        let mask = loss_mask?;
        let prefix_mask = answer_prefix_input_mask(mask.clone());
        let corrupted_inputs =
            self.corrupt_ruliad_answer_prefix_inputs(clean_inputs, prefix_mask, config);
        let hidden = self.model.forward_hidden(corrupted_inputs);
        Some(
            self.language_loss_from_hidden(hidden, targets, Some(mask))
                .mul_scalar(weight),
        )
    }

    pub(super) fn corrupt_ruliad_answer_prefix_inputs(
        &self,
        inputs: Tensor<B, 2, Int>,
        prefix_mask: Tensor<B, 2, Int>,
        config: RuliadAnswerDenoisingConfig,
    ) -> Tensor<B, 2, Int> {
        let probability = config.probability.clamp(0.0, 1.0);
        if probability <= f32::EPSILON {
            return inputs;
        }
        let [batch, time] = inputs.shape().dims();
        if batch == 0 || time == 0 || self.input_vocab_size <= 1 {
            return inputs;
        }
        let vocab = self.input_vocab_size as i64;
        let offset = (config.corrupt_offset % vocab).max(1);
        let mut mask = prefix_mask.equal_elem(1);
        if probability < 1.0 {
            let device = inputs.device();
            let keep = Tensor::<B, 2>::random(
                [batch, time],
                TensorDistribution::Uniform(0.0, 1.0),
                &device,
            )
            .lower_elem(probability);
            mask = mask.bool_and(keep);
        }
        let replacements = inputs.clone().add_scalar(offset).remainder_scalar(vocab);
        inputs.mask_where(mask, replacements)
    }

    pub(super) fn ruliad_answer_contract_weight(&self) -> f32 {
        let config = self.ruliad_supervision.answer_contract;
        if !config.enabled || config.weight <= f32::EPSILON || config.every_steps == 0 {
            return 0.0;
        }
        let step_index = self.gradient_scale_step.load(Ordering::Relaxed);
        if step_index < config.start_after_steps {
            return 0.0;
        }
        if !step_index.is_multiple_of(config.every_steps) {
            return 0.0;
        }
        config.weight
    }

    pub(super) fn ruliad_answer_contract_loss(
        &self,
        policy_batch: &crate::dataset::RuliadPolicyBatch,
        device: &B::Device,
        block_size: usize,
    ) -> Option<Tensor<B, 1>> {
        let config = self.ruliad_supervision.answer_contract;
        let weight = self.ruliad_answer_contract_weight();
        if weight <= f32::EPSILON || policy_batch.samples.is_empty() || self.pipeline_enabled() {
            return None;
        }
        let tokenizer =
            burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer::from_config(
                &policy_batch.tokenization,
            )
            .ok()?;
        let completion_budget = config
            .max_completion_tokens
            .max(1)
            .min(block_size.saturating_sub(1).max(1));
        let max_rows = config.max_rows_per_step.max(1);
        let prompt_schema_max_rows = if config.prompt_schema_max_rows_per_step == 0 {
            max_rows
        } else {
            config.prompt_schema_max_rows_per_step
        }
        .max(1);

        #[derive(Clone)]
        struct ContractRow {
            inputs: Vec<i64>,
            targets: Vec<i64>,
            mask: Vec<f32>,
            premature_close_mask: Vec<f32>,
        }

        // A sequence terminator may only be penalized as one event when the
        // tokenizer represents it with one structural token. Penalizing each
        // byte in `[/R*]` independently suppresses common answer characters.
        let close_token_ids = policy_batch.stop_token_id.into_iter().collect::<Vec<_>>();
        let premature_close_weight = config.premature_close_unlikelihood_weight;
        let mut rows = Vec::<ContractRow>::new();
        let mut sample_groups = 0usize;
        let mut prompt_schema_sample_groups = 0usize;
        let mut contract_tokens = 0usize;
        let mut prompt_schema_value_tokens = 0usize;
        let mut prompt_schema_rows = 0usize;
        let mut schema_tokens = 0usize;
        let mut schema_start_tokens = 0usize;
        let mut value_tokens = 0usize;
        let mut other_tokens = 0usize;
        let mut premature_close_tokens = 0usize;
        for sample in policy_batch.samples.iter() {
            if rows.len() >= max_rows {
                break;
            }
            let mut prompt = sample.prompt_tokens.clone();
            if prompt.is_empty() || sample.item.expected_answer.trim().is_empty() {
                continue;
            }
            let Some((oracle_completion, _oracle_text, _truncated)) =
                Self::ruliad_oracle_completion_tokens(&tokenizer, sample, completion_budget)
            else {
                continue;
            };
            prompt = Self::ruliad_trim_prompt_for_completion(
                &prompt,
                oracle_completion.len(),
                block_size,
            );
            let Some((inputs, targets, _default_mask)) =
                Self::ruliad_policy_row_from_completion(&prompt, &oracle_completion)
            else {
                continue;
            };
            let completion_start = prompt.len().saturating_sub(1).min(targets.len());
            let schema_mask = Self::ruliad_answer_schema_completion_mask(
                &tokenizer,
                &sample.item.expected_answer,
                oracle_completion.len(),
            );
            let schema_start_mask = Self::ruliad_answer_schema_start_completion_mask(
                &tokenizer,
                &sample.item.expected_answer,
                oracle_completion.len(),
            );
            let value_mask = Self::ruliad_answer_value_completion_mask(
                &tokenizer,
                &sample.item.expected_answer,
                oracle_completion.len(),
            );
            let mut mask = vec![0.0f32; targets.len()];
            let mut premature_close_mask = vec![0.0f32; targets.len()];
            let mut active_tokens = 0usize;
            for completion_index in 0..oracle_completion.len() {
                let target_index = completion_start.saturating_add(completion_index);
                if target_index >= mask.len() {
                    continue;
                }
                let schema_token = schema_mask.get(completion_index).copied().unwrap_or(false);
                let schema_start_token = schema_start_mask
                    .get(completion_index)
                    .copied()
                    .unwrap_or(false);
                let token_weight = if schema_token {
                    schema_tokens = schema_tokens.saturating_add(1);
                    if schema_start_token {
                        schema_start_tokens = schema_start_tokens.saturating_add(1);
                        config
                            .schema_token_weight
                            .max(config.schema_start_token_weight)
                    } else {
                        config.schema_token_weight
                    }
                } else if value_mask.get(completion_index).copied().unwrap_or(false) {
                    value_tokens = value_tokens.saturating_add(1);
                    config.value_token_weight
                } else {
                    other_tokens = other_tokens.saturating_add(1);
                    config.other_token_weight
                };
                if token_weight > f32::EPSILON {
                    mask[target_index] = token_weight;
                    active_tokens = active_tokens.saturating_add(1);
                }
            }
            if premature_close_weight > f32::EPSILON && !close_token_ids.is_empty() {
                let answer_token_len = tokenizer
                    .encode_payload(sample.item.expected_answer.trim())
                    .len()
                    .min(oracle_completion.len());
                for completion_index in 0..answer_token_len {
                    let target_index = completion_start.saturating_add(completion_index);
                    if let Some(slot) = premature_close_mask.get_mut(target_index)
                        && *slot <= f32::EPSILON
                    {
                        *slot = 1.0;
                        premature_close_tokens = premature_close_tokens.saturating_add(1);
                    }
                }
            }
            if active_tokens == 0 {
                continue;
            }
            contract_tokens = contract_tokens.saturating_add(active_tokens);
            sample_groups = sample_groups.saturating_add(1);
            rows.push(ContractRow {
                inputs,
                targets,
                mask,
                premature_close_mask,
            });
        }
        let oracle_rows = rows.len();
        if config.prompt_schema_value_weight > f32::EPSILON {
            let field_rows_by_sample = policy_batch
                .samples
                .iter()
                .filter_map(|sample| {
                    let prompt = sample.prompt_tokens.clone();
                    if prompt.is_empty() || sample.item.expected_answer.trim().is_empty() {
                        return None;
                    }
                    let field_rows = Self::ruliad_prompt_schema_value_completion_rows(
                        &tokenizer,
                        &prompt,
                        &sample.item.expected_answer,
                        sample.item.document_close_marker(),
                        completion_budget,
                        block_size,
                        prompt_schema_max_rows,
                    );
                    (!field_rows.is_empty()).then_some(field_rows)
                })
                .collect::<Vec<_>>();
            let selected_rows =
                take_rows_round_robin(&field_rows_by_sample, prompt_schema_max_rows);
            prompt_schema_sample_groups = selected_rows
                .iter()
                .map(|(sample_index, _)| *sample_index)
                .collect::<HashSet<_>>()
                .len();
            for (_sample_index, (inputs, targets, mask, active_tokens)) in selected_rows {
                if active_tokens == 0 {
                    continue;
                }
                let mask = mask
                    .into_iter()
                    .map(|value| {
                        if value > f32::EPSILON {
                            config.prompt_schema_value_weight
                        } else {
                            0.0
                        }
                    })
                    .collect::<Vec<_>>();
                let premature_close_mask = vec![0.0f32; targets.len()];
                contract_tokens = contract_tokens.saturating_add(active_tokens);
                prompt_schema_value_tokens =
                    prompt_schema_value_tokens.saturating_add(active_tokens);
                prompt_schema_rows = prompt_schema_rows.saturating_add(1);
                rows.push(ContractRow {
                    inputs,
                    targets,
                    mask,
                    premature_close_mask,
                });
            }
        }
        let skip_reason = rows
            .is_empty()
            .then(|| "no_answer_contract_rows".to_string());
        self.write_ruliad_answer_contract_telemetry(RuliadAnswerContractTelemetry {
            version: 1,
            step_index: self.gradient_scale_step.load(Ordering::Relaxed),
            policy_batch_present: true,
            skip_reason,
            sample_groups,
            prompt_schema_sample_groups,
            oracle_rows,
            prompt_schema_rows,
            contract_tokens,
            prompt_schema_value_tokens,
            schema_tokens,
            schema_start_tokens,
            value_tokens,
            other_tokens,
            premature_close_tokens,
            answer_contract_weight: weight,
            premature_close_unlikelihood_weight: premature_close_weight,
            max_completion_tokens: completion_budget,
            max_rows_per_step: max_rows,
            prompt_schema_max_rows_per_step: prompt_schema_max_rows,
        });
        if rows.is_empty() {
            return None;
        }

        let max_len = rows.iter().map(|row| row.inputs.len()).max()?.max(1);
        let row_count = rows.len();
        let mut input_values = vec![0i64; row_count * max_len];
        let mut target_values = vec![0i64; row_count * max_len];
        let mut mask_values = vec![0.0f32; row_count * max_len];
        let mut premature_close_mask_values = vec![0.0f32; row_count * max_len];
        for (row_index, row) in rows.iter().enumerate() {
            let offset = row_index * max_len;
            let len = row.inputs.len().min(max_len);
            input_values[offset..offset + len].copy_from_slice(&row.inputs[..len]);
            target_values[offset..offset + len].copy_from_slice(&row.targets[..len]);
            mask_values[offset..offset + len].copy_from_slice(&row.mask[..len]);
            premature_close_mask_values[offset..offset + len]
                .copy_from_slice(&row.premature_close_mask[..len]);
        }
        let inputs = Tensor::<B, 2, Int>::from_data(
            TensorData::new(input_values, [row_count, max_len]),
            device,
        );
        let targets = Tensor::<B, 2, Int>::from_data(
            TensorData::new(target_values, [row_count, max_len]),
            device,
        );
        let mask =
            Tensor::<B, 2>::from_data(TensorData::new(mask_values, [row_count, max_len]), device);
        let logits = self.model.forward(inputs);
        let log_probs = log_probs_from_logits(logits);
        let token_log_probs = selected_token_log_probs(log_probs.clone(), targets);
        let active = mask.clone().sum().reshape([1]).clamp_min(1.0);
        let mut loss = (token_log_probs * mask)
            .sum()
            .reshape([1])
            .div(active)
            .mul_scalar(-weight);
        if premature_close_weight > f32::EPSILON
            && premature_close_tokens > 0
            && !close_token_ids.is_empty()
        {
            let close_mask = Tensor::<B, 2>::from_data(
                TensorData::new(premature_close_mask_values, [row_count, max_len]),
                device,
            );
            let close_active = close_mask.clone().sum().reshape([1]).clamp_min(1.0);
            let mut close_loss: Option<Tensor<B, 1>> = None;
            let close_token_count = close_token_ids.len().max(1) as f32;
            for close_token_id in close_token_ids {
                let close_targets = Tensor::<B, 2, Int>::from_data(
                    TensorData::new(
                        vec![close_token_id; row_count * max_len],
                        [row_count, max_len],
                    ),
                    device,
                );
                let token_loss =
                    unlikelihood_from_log_probs(log_probs.clone(), close_targets, 1.0e-6);
                let masked = (token_loss * close_mask.clone())
                    .sum()
                    .reshape([1])
                    .div(close_active.clone());
                close_loss = Some(match close_loss {
                    Some(accumulated) => accumulated + masked,
                    None => masked,
                });
            }
            if let Some(close_loss) = close_loss {
                loss = loss
                    + close_loss
                        .div_scalar(close_token_count)
                        .mul_scalar(premature_close_weight);
            }
        }
        Some(loss)
    }

    pub(super) fn ruliad_answer_contract_auxiliary_loss(
        &self,
        policy_batch: Option<&crate::dataset::RuliadPolicyBatch>,
        device: &B::Device,
        block_size: usize,
    ) -> Option<Tensor<B, 1>> {
        let contract_weight = self.ruliad_answer_contract_weight();
        if contract_weight <= f32::EPSILON {
            return None;
        }
        if let Some(policy_batch) = policy_batch {
            self.ruliad_answer_contract_loss(policy_batch, device, block_size)
        } else {
            self.write_ruliad_answer_contract_skip("missing_policy_batch", contract_weight);
            None
        }
    }

    pub(super) fn ruliad_structured_answer_recovery_loss(
        &self,
        policy_batch: &crate::dataset::RuliadPolicyBatch,
        device: &B::Device,
        block_size: usize,
    ) -> Option<Tensor<B, 1>> {
        let config = self.ruliad_supervision.answer_denoising;
        let weight = self.ruliad_structured_answer_recovery_weight();
        if weight <= f32::EPSILON || policy_batch.samples.is_empty() || self.pipeline_enabled() {
            return None;
        }
        let tokenizer =
            burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer::from_config(
                &policy_batch.tokenization,
            )
            .ok()?;
        let completion_budget = config
            .structured_recovery_max_completion_tokens
            .max(1)
            .min(block_size.saturating_sub(1).max(1));
        let prompt_budget = block_size.saturating_sub(completion_budget).max(1);

        #[derive(Clone)]
        struct RecoveryRow {
            inputs: Vec<i64>,
            targets: Vec<i64>,
            mask: Vec<i64>,
        }

        let mut rows = Vec::<RecoveryRow>::new();
        let mut sample_groups = 0usize;
        let mut field_negative_recovery_rows = 0usize;
        let mut template_negative_recovery_rows = 0usize;
        let mut schema_negative_recovery_rows = 0usize;
        for sample in policy_batch.samples.iter() {
            let mut prompt = sample.prompt_tokens.clone();
            if prompt.is_empty() {
                continue;
            }
            if prompt.len() > prompt_budget {
                prompt = prompt[prompt.len() - prompt_budget..].to_vec();
            }
            let Some((oracle_completion, _oracle_text, _truncated)) =
                Self::ruliad_oracle_completion_tokens(&tokenizer, sample, completion_budget)
            else {
                continue;
            };
            let Some((oracle_inputs, oracle_targets, oracle_mask)) =
                Self::ruliad_policy_row_from_completion(&prompt, &oracle_completion)
            else {
                continue;
            };
            let completion_start = prompt.len().saturating_sub(1).min(oracle_inputs.len());
            let mut sample_rows = 0usize;
            for (negative, negative_kind) in Self::ruliad_structured_negative_answers_with_schema(
                &sample.item.expected_answer,
                config.structured_recovery_negative_count,
                config.structured_recovery_template_negative_count,
                config.structured_recovery_schema_negative_count,
            ) {
                let Some((negative_completion, _negative_text)) =
                    Self::ruliad_completion_tokens_from_answer(
                        &tokenizer,
                        &negative,
                        sample.item.document_close_marker(),
                        completion_budget,
                    )
                else {
                    continue;
                };
                let mut corrupted_inputs = oracle_inputs.clone();
                for (index, value) in corrupted_inputs
                    .iter_mut()
                    .enumerate()
                    .skip(completion_start)
                {
                    let negative_index = index - completion_start;
                    if let Some(negative_token) = negative_completion.get(negative_index) {
                        *value = *negative_token;
                    }
                }
                rows.push(RecoveryRow {
                    inputs: corrupted_inputs,
                    targets: oracle_targets.clone(),
                    mask: oracle_mask
                        .iter()
                        .map(|value| if *value > 0.0 { 1 } else { 0 })
                        .collect(),
                });
                sample_rows = sample_rows.saturating_add(1);
                match negative_kind {
                    RuliadStructuredNegativeKind::FieldMutation => {
                        field_negative_recovery_rows =
                            field_negative_recovery_rows.saturating_add(1);
                    }
                    RuliadStructuredNegativeKind::TemplateCollapse => {
                        template_negative_recovery_rows =
                            template_negative_recovery_rows.saturating_add(1);
                    }
                    RuliadStructuredNegativeKind::SchemaCollapse => {
                        schema_negative_recovery_rows =
                            schema_negative_recovery_rows.saturating_add(1);
                    }
                }
            }
            if sample_rows > 0 {
                sample_groups = sample_groups.saturating_add(1);
            }
        }
        self.write_ruliad_structured_recovery_telemetry(RuliadStructuredRecoveryTelemetry {
            version: 1,
            step_index: self.gradient_scale_step.load(Ordering::Relaxed),
            policy_batch_present: true,
            skip_reason: None,
            sample_groups,
            recovery_rows: rows.len(),
            field_negative_recovery_rows,
            template_negative_recovery_rows,
            schema_negative_recovery_rows,
            structured_recovery_weight: weight,
            structured_recovery_max_completion_tokens: completion_budget,
        });
        if rows.is_empty() {
            return None;
        }

        let max_len = rows.iter().map(|row| row.inputs.len()).max()?.max(1);
        let row_count = rows.len();
        let mut input_values = vec![0i64; row_count * max_len];
        let mut target_values = vec![0i64; row_count * max_len];
        let mut mask_values = vec![0i64; row_count * max_len];
        for (row_index, row) in rows.into_iter().enumerate() {
            let offset = row_index * max_len;
            let len = row.inputs.len().min(max_len);
            input_values[offset..offset + len].copy_from_slice(&row.inputs[..len]);
            target_values[offset..offset + len].copy_from_slice(&row.targets[..len]);
            mask_values[offset..offset + len].copy_from_slice(&row.mask[..len]);
        }
        let inputs = Tensor::<B, 2, Int>::from_data(
            TensorData::new(input_values, [row_count, max_len]),
            device,
        );
        let targets = Tensor::<B, 2, Int>::from_data(
            TensorData::new(target_values, [row_count, max_len]),
            device,
        );
        let mask = Tensor::<B, 2, Int>::from_data(
            TensorData::new(mask_values, [row_count, max_len]),
            device,
        );
        let hidden = self.model.forward_hidden(inputs);
        Some(
            self.language_loss_from_hidden(hidden, targets, Some(mask))
                .mul_scalar(weight),
        )
    }

    pub(super) fn ruliad_structured_answer_recovery_auxiliary_loss(
        &self,
        policy_batch: Option<&crate::dataset::RuliadPolicyBatch>,
        device: &B::Device,
        block_size: usize,
    ) -> Option<Tensor<B, 1>> {
        let recovery_weight = self.ruliad_structured_answer_recovery_weight();
        if recovery_weight <= f32::EPSILON {
            return None;
        }
        if let Some(policy_batch) = policy_batch {
            self.ruliad_structured_answer_recovery_loss(policy_batch, device, block_size)
        } else {
            self.write_ruliad_structured_recovery_skip("missing_policy_batch", recovery_weight);
            None
        }
    }

    pub(super) fn write_ruliad_structured_recovery_skip(&self, reason: &str, weight: f32) {
        self.write_ruliad_structured_recovery_telemetry(RuliadStructuredRecoveryTelemetry {
            version: 1,
            step_index: self.gradient_scale_step.load(Ordering::Relaxed),
            policy_batch_present: false,
            skip_reason: Some(reason.to_string()),
            sample_groups: 0,
            recovery_rows: 0,
            field_negative_recovery_rows: 0,
            template_negative_recovery_rows: 0,
            schema_negative_recovery_rows: 0,
            structured_recovery_weight: weight,
            structured_recovery_max_completion_tokens: self
                .ruliad_supervision
                .answer_denoising
                .structured_recovery_max_completion_tokens,
        });
    }

    pub(super) fn ruliad_verifier_reward_weight(&self) -> f32 {
        let config = self.ruliad_supervision.verifier_reward;
        if !config.enabled || config.weight <= f32::EPSILON || config.every_steps == 0 {
            return 0.0;
        }
        let step_index = self.gradient_scale_step.load(Ordering::Relaxed);
        if step_index < config.start_after_steps {
            return 0.0;
        }
        if !step_index.is_multiple_of(config.every_steps) {
            return 0.0;
        }
        config.weight
    }

    pub(super) fn ruliad_structured_contrast_weight(&self) -> f32 {
        let config = self.ruliad_supervision.verifier_reward;
        if !config.enabled
            || config.structured_contrast_weight <= f32::EPSILON
            || config.structured_contrast_every_steps == 0
        {
            return 0.0;
        }
        let step_index = self.gradient_scale_step.load(Ordering::Relaxed);
        if step_index < config.structured_contrast_start_after_steps {
            return 0.0;
        }
        if !step_index.is_multiple_of(config.structured_contrast_every_steps) {
            return 0.0;
        }
        config.structured_contrast_weight
    }

    pub(super) fn ruliad_field_binding_contrast_weight(&self) -> f32 {
        let config = self.ruliad_supervision.verifier_reward;
        if !config.enabled
            || config.field_binding_contrast_weight <= f32::EPSILON
            || config.field_binding_contrast_every_steps == 0
        {
            return 0.0;
        }
        let step_index = self.gradient_scale_step.load(Ordering::Relaxed);
        if step_index < config.field_binding_contrast_start_after_steps {
            return 0.0;
        }
        if !step_index.is_multiple_of(config.field_binding_contrast_every_steps) {
            return 0.0;
        }
        config.field_binding_contrast_weight
    }

    pub(super) fn ruliad_verifier_rollout_feedback_active(&self) -> bool {
        let config = self.ruliad_supervision.verifier_reward;
        if !config.enabled
            || (config.rollout_imitation_weight <= f32::EPSILON
                && config.rollout_recovery_weight <= f32::EPSILON)
            || config.rollout_imitation_every_steps == 0
        {
            return false;
        }
        let step_index = self.gradient_scale_step.load(Ordering::Relaxed);
        if step_index < config.rollout_imitation_start_after_steps {
            return false;
        }
        if !step_index.is_multiple_of(config.rollout_imitation_every_steps) {
            return false;
        }
        true
    }

    pub(super) fn ruliad_proof_policy_dagger_weight(&self) -> f32 {
        self.ruliad_proof_policy_dagger_weight_at_step(
            self.gradient_scale_step.load(Ordering::Relaxed),
        )
    }

    pub(super) fn ruliad_proof_policy_dagger_weight_at_step(&self, step_index: usize) -> f32 {
        let config = self.ruliad_supervision.proof_policy;
        if !config.enabled || config.weight <= f32::EPSILON || config.every_steps == 0 {
            return 0.0;
        }
        if step_index < config.start_after_steps || !step_index.is_multiple_of(config.every_steps) {
            return 0.0;
        }
        config.weight
    }

    pub(super) fn mix_ruliad_policy_seed(mut value: u64) -> u64 {
        value ^= value >> 30;
        value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value ^= value >> 27;
        value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    pub(super) fn ruliad_vpo_scalarizations(
        &self,
        sample_index: usize,
        count: usize,
        config: crate::config::train::RuliadVerifierRewardConfig,
    ) -> Vec<[f32; burn_dragon_universality::ruliad::RULIAD_VERIFIER_REWARD_VECTOR_DIM]> {
        let step_index = self.gradient_scale_step.load(Ordering::Relaxed) as u64;
        let seed = Self::mix_ruliad_policy_seed(
            step_index
                ^ (sample_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
                ^ (count as u64).wrapping_mul(0xd1b5_4a32_d192_ed03),
        );
        let mut rng = StdRng::seed_from_u64(seed);
        let mut scalarizations = Vec::with_capacity(count);
        for _ in 0..count {
            let mut weights =
                [0.0f32; burn_dragon_universality::ruliad::RULIAD_VERIFIER_REWARD_VECTOR_DIM];
            let mut sum = 0.0f32;
            for weight in weights.iter_mut() {
                let draw = -rng.gen_range(f32::MIN_POSITIVE..1.0).ln();
                *weight = draw;
                sum += draw;
            }
            if !sum.is_finite() || sum <= f32::EPSILON {
                let uniform = 1.0 / weights.len() as f32;
                weights.fill(uniform);
            } else {
                for weight in weights.iter_mut() {
                    *weight /= sum;
                }
            }
            Self::constrain_ruliad_vpo_scalarization(&mut weights, config);
            scalarizations.push(weights);
        }
        scalarizations
    }

    pub(super) fn constrain_ruliad_vpo_scalarization(
        weights: &mut [f32; burn_dragon_universality::ruliad::RULIAD_VERIFIER_REWARD_VECTOR_DIM],
        config: crate::config::train::RuliadVerifierRewardConfig,
    ) {
        const CORRECTNESS_AXES: &[usize] = &[0, 1, 2, 3, 4];
        const SCHEMA_QUALITY_AXES: &[usize] = &[6];
        const HEALTH_AXES: &[usize] = &[8, 9];
        const COMPACTNESS_AXIS: usize = 5;
        let original = *weights;
        let correctness_floor = config.vpo_correctness_mass_floor.clamp(0.0, 1.0);
        let schema_floor = config
            .vpo_schema_quality_mass_floor
            .clamp(0.0, 1.0 - correctness_floor);
        let health_floor = config
            .vpo_completion_health_mass_floor
            .clamp(0.0, 1.0 - correctness_floor - schema_floor);
        let residual_mass = (1.0 - correctness_floor - schema_floor - health_floor).max(0.0);
        for (weight, original_weight) in weights.iter_mut().zip(original) {
            *weight = original_weight * residual_mass;
        }
        Self::add_weighted_group_mass(weights, &original, CORRECTNESS_AXES, correctness_floor);
        Self::add_weighted_group_mass(weights, &original, SCHEMA_QUALITY_AXES, schema_floor);
        Self::add_weighted_group_mass(weights, &original, HEALTH_AXES, health_floor);
        let compactness_max = config.vpo_compactness_max_weight.clamp(0.0, 1.0);
        if weights[COMPACTNESS_AXIS] > compactness_max {
            let excess = weights[COMPACTNESS_AXIS] - compactness_max;
            weights[COMPACTNESS_AXIS] = compactness_max;
            Self::add_uniform_mass(weights, CORRECTNESS_AXES, excess * 0.60);
            Self::add_uniform_mass(weights, SCHEMA_QUALITY_AXES, excess * 0.25);
            Self::add_uniform_mass(weights, HEALTH_AXES, excess * 0.15);
        }
        Self::renormalize_scalarization(weights);
    }

    pub(super) fn add_weighted_group_mass(
        weights: &mut [f32; burn_dragon_universality::ruliad::RULIAD_VERIFIER_REWARD_VECTOR_DIM],
        original: &[f32; burn_dragon_universality::ruliad::RULIAD_VERIFIER_REWARD_VECTOR_DIM],
        axes: &[usize],
        mass: f32,
    ) {
        if axes.is_empty() || mass <= f32::EPSILON {
            return;
        }
        let group_mass = axes.iter().map(|axis| original[*axis]).sum::<f32>();
        if group_mass <= f32::EPSILON {
            Self::add_uniform_mass(weights, axes, mass);
            return;
        }
        for axis in axes {
            weights[*axis] += mass * original[*axis] / group_mass;
        }
    }

    pub(super) fn add_uniform_mass(
        weights: &mut [f32; burn_dragon_universality::ruliad::RULIAD_VERIFIER_REWARD_VECTOR_DIM],
        axes: &[usize],
        mass: f32,
    ) {
        if axes.is_empty() || mass <= f32::EPSILON {
            return;
        }
        let share = mass / axes.len() as f32;
        for axis in axes {
            weights[*axis] += share;
        }
    }

    pub(super) fn renormalize_scalarization(
        weights: &mut [f32; burn_dragon_universality::ruliad::RULIAD_VERIFIER_REWARD_VECTOR_DIM],
    ) {
        let sum = weights.iter().copied().sum::<f32>();
        if sum <= f32::EPSILON || !sum.is_finite() {
            let uniform = 1.0 / weights.len() as f32;
            weights.fill(uniform);
            return;
        }
        for weight in weights.iter_mut() {
            *weight = (*weight / sum).max(0.0);
        }
    }

    pub(super) fn ruliad_vpo_independent_utilities_with_telemetry(
        &self,
        scores: &[burn_dragon_universality::ruliad::RuliadReasoningScore],
        scalarizations: &[
            [f32; burn_dragon_universality::ruliad::RULIAD_VERIFIER_REWARD_VECTOR_DIM]
        ],
        telemetry: &mut RuliadPolicyRewardTelemetryAccumulator,
    ) -> Vec<f32> {
        let mut utilities = vec![0.0f32; scores.len()];
        if scores.is_empty() || scalarizations.is_empty() {
            return utilities;
        }
        let vectors = scores
            .iter()
            .map(burn_dragon_universality::ruliad::ruliad_verifier_reward_vector)
            .collect::<Vec<_>>();
        for weights in scalarizations {
            telemetry.record_vpo_scalarization(weights);
            let mut best_index = 0usize;
            let mut best_value = f32::NEG_INFINITY;
            for (index, vector) in vectors.iter().copied().enumerate() {
                let value = vector.scalarize(weights);
                if value > best_value {
                    best_index = index;
                    best_value = value;
                }
            }
            if best_value.is_finite() {
                utilities[best_index] += best_value;
            }
        }
        let scale = scalarizations.len() as f32;
        for utility in utilities.iter_mut() {
            *utility /= scale;
        }
        utilities
    }

    pub(super) fn ruliad_score_has_policy_correctness_signal(
        score: &burn_dragon_universality::ruliad::RuliadReasoningScore,
        min_partial_progress_ppm: usize,
        min_completion_quality_ppm: usize,
    ) -> bool {
        if score.completion_quality_ppm < min_completion_quality_ppm {
            return false;
        }
        matches!(
            score.status,
            burn_dragon_universality::ruliad::RuliadAnswerStatus::VerifierMatch
                | burn_dragon_universality::ruliad::RuliadAnswerStatus::SemanticMatch
        ) || (score.status == burn_dragon_universality::ruliad::RuliadAnswerStatus::Partial
            && score.partial_progress_ppm >= min_partial_progress_ppm)
    }

    pub(super) fn ruliad_score_has_rollout_recovery_signal(
        score: &burn_dragon_universality::ruliad::RuliadReasoningScore,
        min_partial_progress_ppm: usize,
        min_completion_quality_ppm: usize,
    ) -> bool {
        if score.completion_quality_ppm < min_completion_quality_ppm {
            return false;
        }
        match score.status {
            burn_dragon_universality::ruliad::RuliadAnswerStatus::Partial => {
                score.partial_progress_ppm >= min_partial_progress_ppm
            }
            burn_dragon_universality::ruliad::RuliadAnswerStatus::SchemaValidWrong => true,
            burn_dragon_universality::ruliad::RuliadAnswerStatus::Malformed
            | burn_dragon_universality::ruliad::RuliadAnswerStatus::Missing => true,
            burn_dragon_universality::ruliad::RuliadAnswerStatus::VerifierMatch
            | burn_dragon_universality::ruliad::RuliadAnswerStatus::SemanticMatch => false,
        }
    }

    pub(super) fn ruliad_score_passes_policy_positive_advantage_gate(
        score: &burn_dragon_universality::ruliad::RuliadReasoningScore,
        config: crate::config::train::RuliadVerifierRewardConfig,
    ) -> bool {
        Self::ruliad_score_has_policy_correctness_signal(
            score,
            config.positive_advantage_min_partial_progress_ppm,
            config.positive_advantage_min_completion_quality_ppm,
        )
    }

    pub(super) fn constrain_ruliad_policy_advantages(
        scores: &[burn_dragon_universality::ruliad::RuliadReasoningScore],
        advantages: &mut [f32],
        config: crate::config::train::RuliadVerifierRewardConfig,
    ) -> bool {
        if !config.positive_advantage_requires_correctness {
            return true;
        }
        let mut has_correctness_candidate = false;
        for score in scores {
            if Self::ruliad_score_passes_policy_positive_advantage_gate(score, config) {
                has_correctness_candidate = true;
                break;
            }
        }
        if !has_correctness_candidate {
            return false;
        }
        for (score, advantage) in scores.iter().zip(advantages.iter_mut()) {
            if *advantage > 0.0
                && !Self::ruliad_score_passes_policy_positive_advantage_gate(score, config)
            {
                *advantage = 0.0;
            }
        }
        advantages
            .iter()
            .any(|advantage| advantage.abs() > f32::EPSILON)
    }

    pub(super) fn write_ruliad_policy_telemetry(&self, telemetry: RuliadPolicyRewardTelemetry) {
        let Some(path) = self.ruliad_policy_telemetry_path.as_ref() else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let Ok(line) = serde_json::to_string(&telemetry) else {
            return;
        };
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.as_ref())
        {
            let _ = writeln!(file, "{line}");
        }
    }

    pub(super) fn write_ruliad_answer_contract_telemetry(
        &self,
        telemetry: RuliadAnswerContractTelemetry,
    ) {
        let Some(path) = self.ruliad_answer_contract_telemetry_path.as_ref() else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let Ok(line) = serde_json::to_string(&telemetry) else {
            return;
        };
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.as_ref())
        {
            let _ = writeln!(file, "{line}");
        }
    }

    pub(super) fn write_ruliad_answer_contract_skip(&self, reason: &str, weight: f32) {
        self.write_ruliad_answer_contract_telemetry(RuliadAnswerContractTelemetry {
            version: 1,
            step_index: self.gradient_scale_step.load(Ordering::Relaxed),
            policy_batch_present: false,
            skip_reason: Some(reason.to_string()),
            sample_groups: 0,
            prompt_schema_sample_groups: 0,
            oracle_rows: 0,
            prompt_schema_rows: 0,
            contract_tokens: 0,
            prompt_schema_value_tokens: 0,
            schema_tokens: 0,
            schema_start_tokens: 0,
            value_tokens: 0,
            other_tokens: 0,
            premature_close_tokens: 0,
            answer_contract_weight: weight,
            premature_close_unlikelihood_weight: self
                .ruliad_supervision
                .answer_contract
                .premature_close_unlikelihood_weight,
            max_completion_tokens: self
                .ruliad_supervision
                .answer_contract
                .max_completion_tokens,
            max_rows_per_step: self.ruliad_supervision.answer_contract.max_rows_per_step,
            prompt_schema_max_rows_per_step: {
                let contract = self.ruliad_supervision.answer_contract;
                if contract.prompt_schema_max_rows_per_step == 0 {
                    contract.max_rows_per_step
                } else {
                    contract.prompt_schema_max_rows_per_step
                }
            },
        });
    }

    pub(super) fn write_ruliad_structured_contrast_telemetry(
        &self,
        telemetry: RuliadStructuredContrastTelemetry,
    ) {
        let Some(path) = self.ruliad_structured_contrast_telemetry_path.as_ref() else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let Ok(line) = serde_json::to_string(&telemetry) else {
            return;
        };
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.as_ref())
        {
            let _ = writeln!(file, "{line}");
        }
    }

    pub(super) fn write_ruliad_structured_contrast_skip(&self, reason: &str, weight: f32) {
        self.write_ruliad_structured_contrast_telemetry(RuliadStructuredContrastTelemetry {
            version: 1,
            step_index: self.gradient_scale_step.load(Ordering::Relaxed),
            skip_reason: Some(reason.to_string()),
            sample_groups: 0,
            oracle_completion_rows: 0,
            field_negative_completion_rows: 0,
            template_negative_completion_rows: 0,
            schema_negative_completion_rows: 0,
            generated_attractor_negative_completion_rows: 0,
            contrast_pairs: 0,
            contrast_discriminative_tokens: 0,
            structured_contrast_weight: weight,
            structured_contrast_margin: self
                .ruliad_supervision
                .verifier_reward
                .structured_contrast_margin,
        });
    }

    pub(super) fn write_ruliad_field_binding_contrast_telemetry(
        &self,
        telemetry: RuliadFieldBindingContrastTelemetry,
    ) {
        let Some(path) = self.ruliad_field_binding_contrast_telemetry_path.as_ref() else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let Ok(line) = serde_json::to_string(&telemetry) else {
            return;
        };
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.as_ref())
        {
            let _ = writeln!(file, "{line}");
        }
    }

    pub(super) fn write_ruliad_field_binding_contrast_skip(&self, reason: &str, weight: f32) {
        self.write_ruliad_field_binding_contrast_telemetry(RuliadFieldBindingContrastTelemetry {
            version: 3,
            objective: RULIAD_FIELD_BINDING_OBJECTIVE,
            step_index: self.gradient_scale_step.load(Ordering::Relaxed),
            skip_reason: Some(reason.to_string()),
            sample_groups: 0,
            oracle_prompt_count: 0,
            prompt_pairs: 0,
            contrast_pairs: 0,
            candidate_pairs: 0,
            filtered_presented_action_candidates: 0,
            contrast_discriminative_tokens: 0,
            negative_pool_size: 0,
            replay_pool_size: 0,
            replay_contrast_pairs: 0,
            generated_attractor_pool_size: 0,
            generated_attractor_negative_pool_size: 0,
            generated_attractor_contrast_pairs: 0,
            rank_metric_pairs: 0,
            rank_metric_tokens: 0,
            logit_margin_mean: None,
            positive_token_fraction: None,
            margin_satisfied_token_fraction: None,
            exact_pair_rank_fraction: None,
            exact_pair_margin_fraction: None,
            sequence_rank_metric_pairs: 0,
            sequence_log_probability_margin_mean: None,
            positive_sequence_fraction: None,
            sequence_margin_satisfied_fraction: None,
            field_binding_contrast_weight: weight,
            field_binding_contrast_margin: self
                .ruliad_supervision
                .verifier_reward
                .field_binding_contrast_margin,
            field_binding_contrast_pair_weight: self
                .ruliad_supervision
                .verifier_reward
                .field_binding_contrast_pair_weight,
        });
    }

    pub(super) fn write_ruliad_generated_attractor_telemetry(
        &self,
        telemetry: RuliadGeneratedAttractorReplayTelemetry,
    ) {
        let Some(path) = self.ruliad_generated_attractor_telemetry_path.as_ref() else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let Ok(line) = serde_json::to_string(&telemetry) else {
            return;
        };
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.as_ref())
        {
            let _ = writeln!(file, "{line}");
        }
    }

    pub(super) fn ruliad_generated_attractor_summary(
        &self,
    ) -> RuliadGeneratedAttractorReplaySummary {
        let config = self.ruliad_supervision.verifier_reward;
        self.ruliad_generated_attractor_replay
            .lock()
            .map(|replay| replay.summary(config.generated_attractor_replay_min_count.max(1)))
            .unwrap_or_default()
    }

    pub(super) fn ruliad_generated_attractor_replay_skip_reason(
        &self,
        summary: &RuliadGeneratedAttractorReplaySummary,
        selected_candidate_rows: usize,
    ) -> Option<String> {
        let config = self.ruliad_supervision.verifier_reward;
        if config.generated_attractor_replay_capacity == 0 || selected_candidate_rows > 0 {
            return None;
        }
        summary
            .diversity_skip_reason(
                config
                    .generated_attractor_replay_min_distinct_answers
                    .max(1),
                config.generated_attractor_replay_max_dominant_fraction,
            )
            .map(str::to_string)
    }

    pub(super) fn write_ruliad_structured_recovery_telemetry(
        &self,
        telemetry: RuliadStructuredRecoveryTelemetry,
    ) {
        let Some(path) = self.ruliad_structured_recovery_telemetry_path.as_ref() else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let Ok(line) = serde_json::to_string(&telemetry) else {
            return;
        };
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.as_ref())
        {
            let _ = writeln!(file, "{line}");
        }
    }

    pub(super) fn write_ruliad_verifier_rollout_telemetry(
        &self,
        telemetry: RuliadVerifierRolloutImitationTelemetry,
    ) {
        let Some(path) = self.ruliad_verifier_rollout_telemetry_path.as_ref() else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let Ok(line) = serde_json::to_string(&telemetry) else {
            return;
        };
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.as_ref())
        {
            let _ = writeln!(file, "{line}");
        }
    }

    pub(super) fn write_ruliad_proof_policy_dagger_telemetry(
        &self,
        telemetry: RuliadProofPolicyDaggerTelemetry,
    ) {
        let Some(path) = self.ruliad_proof_policy_telemetry_path.as_ref() else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let Ok(line) = serde_json::to_string(&telemetry) else {
            return;
        };
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.as_ref())
        {
            let _ = writeln!(file, "{line}");
        }
    }

    pub(crate) fn ruliad_policy_row_from_completion(
        prompt: &[i64],
        completion: &[i64],
    ) -> Option<(Vec<i64>, Vec<i64>, Vec<f32>)> {
        if completion.is_empty() {
            return None;
        }
        let mut sequence = prompt.to_vec();
        sequence.extend_from_slice(completion);
        if sequence.len() < 2 {
            return None;
        }
        let input_len = sequence.len() - 1;
        let inputs = sequence[..input_len].to_vec();
        let targets = sequence[1..].to_vec();
        let mut mask = vec![0.0f32; input_len];
        let completion_start = prompt.len().saturating_sub(1).min(input_len);
        for value in mask.iter_mut().skip(completion_start) {
            *value = 1.0;
        }
        Some((inputs, targets, mask))
    }

    pub(super) fn ruliad_policy_row_from_completion_token(
        prompt: &[i64],
        completion: &[i64],
        completion_token_index: usize,
    ) -> Option<(Vec<i64>, Vec<i64>, Vec<f32>)> {
        if completion_token_index >= completion.len() {
            return None;
        }
        let (inputs, targets, mut mask) =
            Self::ruliad_policy_row_from_completion(prompt, completion)?;
        mask.fill(0.0);
        let target_index = prompt
            .len()
            .saturating_sub(1)
            .saturating_add(completion_token_index);
        *mask.get_mut(target_index)? = 1.0;
        Some((inputs, targets, mask))
    }

    pub(super) fn ruliad_trim_prompt_for_completion(
        prompt: &[i64],
        completion_len: usize,
        block_size: usize,
    ) -> Vec<i64> {
        if prompt.is_empty() {
            return Vec::new();
        }
        let max_prompt_len = block_size.saturating_sub(completion_len.max(1)).max(1);
        if prompt.len() > max_prompt_len {
            prompt[prompt.len() - max_prompt_len..].to_vec()
        } else {
            prompt.to_vec()
        }
    }

    pub(super) fn ruliad_oracle_completion_tokens(
        tokenizer: &burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer,
        sample: &crate::dataset::RuliadPolicySample,
        completion_budget: usize,
    ) -> Option<(Vec<i64>, String, bool)> {
        let answer = sample.item.expected_answer.trim();
        if answer.is_empty() || completion_budget == 0 {
            return None;
        }
        let full_completion = format!("{answer}\n{}", sample.item.document_close_marker());
        let mut payload_tokens = tokenizer.encode_payload(&full_completion);
        let truncated = payload_tokens.len() > completion_budget;
        payload_tokens.truncate(completion_budget);
        if payload_tokens.is_empty() {
            return None;
        }
        let completion_text = tokenizer.decode_payload(&payload_tokens, true);
        let completion = payload_tokens
            .into_iter()
            .map(i64::from)
            .collect::<Vec<_>>();
        Some((completion, completion_text, truncated))
    }

    pub(super) fn record_ruliad_generated_attractor(
        &self,
        sample: &crate::dataset::RuliadPolicySample,
        completion_text: &str,
        score: &burn_dragon_universality::ruliad::RuliadReasoningScore,
        step_index: usize,
    ) -> bool {
        let config = self.ruliad_supervision.verifier_reward;
        if config.generated_attractor_replay_capacity == 0 {
            return false;
        }
        let extracted =
            burn_dragon_universality::ruliad::extract_ruliad_completion(completion_text);
        let Some(answer) = extracted.answer.map(|answer| answer.trim().to_string()) else {
            return false;
        };
        if answer.is_empty() || answer == sample.item.expected_answer.trim() {
            return false;
        }
        let Some(contract) = Self::ruliad_answer_contract(&answer) else {
            return false;
        };
        let key = RuliadGeneratedAttractorKey {
            family: sample.item.family.clone(),
            task_kind: sample.item.task_kind.clone(),
            contract,
            answer,
        };
        self.ruliad_generated_attractor_replay
            .lock()
            .map(|mut replay| {
                replay.record(
                    key,
                    score.status,
                    step_index,
                    config.generated_attractor_replay_capacity,
                )
            })
            .unwrap_or(false)
    }

    pub(super) fn ruliad_generated_attractor_candidates_for_sample(
        &self,
        sample: &crate::dataset::RuliadPolicySample,
    ) -> Vec<RuliadGeneratedAttractorEntry> {
        let config = self.ruliad_supervision.verifier_reward;
        if config.generated_attractor_replay_capacity == 0
            || config.generated_attractor_replay_max_candidates == 0
        {
            return Vec::new();
        }
        let Some(expected_contract) = Self::ruliad_answer_contract(&sample.item.expected_answer)
        else {
            return Vec::new();
        };
        self.ruliad_generated_attractor_replay
            .lock()
            .map(|replay| {
                replay.candidates_for(RuliadGeneratedAttractorQuery {
                    family: &sample.item.family,
                    task_kind: &sample.item.task_kind,
                    expected_contract: &expected_contract,
                    expected_answer: sample.item.expected_answer.trim(),
                    min_count: config.generated_attractor_replay_min_count.max(1),
                    max_candidates: config.generated_attractor_replay_max_candidates,
                    min_distinct_answers: config
                        .generated_attractor_replay_min_distinct_answers
                        .max(1),
                    max_dominant_fraction: config.generated_attractor_replay_max_dominant_fraction,
                })
            })
            .unwrap_or_default()
    }

    pub(super) fn ruliad_structured_negative_answers(answer: &str, count: usize) -> Vec<String> {
        Self::ruliad_structured_negative_answers_with_templates(answer, count, 0)
            .into_iter()
            .map(|(answer, _kind)| answer)
            .collect()
    }

    pub(super) fn ruliad_model_proof_step_negative_answers(
        answer: &str,
        mutation_count: usize,
        template_count: usize,
    ) -> Option<Vec<(String, RuliadStructuredNegativeKind)>> {
        use burn_dragon_universality::ruliad::{
            RuliadProofSource, RuliadProofStep, RuliadRewriteDirection,
        };

        let (goal, step) = burn_dragon_universality::ruliad::wire::decode_model_proof_step(answer)?;

        let mut negatives = Vec::with_capacity(mutation_count.saturating_add(template_count));
        let mut template_rows = 0usize;
        for index in 0..template_count.saturating_add(4) {
            if template_rows >= template_count {
                break;
            }
            let candidate = burn_dragon_universality::ruliad::wire::encode_model_proof_step(
                index,
                &RuliadProofStep {
                    source: RuliadProofSource::Axiom {
                        id: format!("r{index}"),
                    },
                    direction: if index.is_multiple_of(2) {
                        RuliadRewriteDirection::Forward
                    } else {
                        RuliadRewriteDirection::Reverse
                    },
                    path: (!index.is_multiple_of(3))
                        .then_some(vec![0])
                        .unwrap_or_default(),
                },
            );
            let previous_len = negatives.len();
            Self::push_ruliad_negative_answer(
                &mut negatives,
                answer,
                candidate,
                RuliadStructuredNegativeKind::TemplateCollapse,
            );
            template_rows =
                template_rows.saturating_add(usize::from(negatives.len() > previous_len));
        }

        for index in 0..mutation_count {
            let mut candidate_goal = goal;
            let mut candidate_step = step.clone();
            let field_count = 4;
            let delta = index / field_count + 1;
            match index % field_count {
                0 => {
                    candidate_goal = candidate_goal.saturating_add(delta);
                }
                1 => match &mut candidate_step.source {
                    RuliadProofSource::Axiom { id } => {
                        let mutated = Self::mutate_ruliad_answer_value(id, delta);
                        *id = mutated
                            .strip_suffix("_wrong")
                            .map(|prefix| format!("{prefix}x"))
                            .unwrap_or(mutated);
                    }
                    RuliadProofSource::Lemma { goal } => {
                        *goal = goal.saturating_add(delta);
                    }
                },
                2 => {
                    candidate_step.direction = match candidate_step.direction {
                        RuliadRewriteDirection::Forward => RuliadRewriteDirection::Reverse,
                        RuliadRewriteDirection::Reverse => RuliadRewriteDirection::Forward,
                    };
                }
                3 => {
                    if candidate_step.path.is_empty() {
                        candidate_step.path.push(delta.saturating_sub(1));
                    } else {
                        let path_index = (index / field_count) % candidate_step.path.len();
                        let value = candidate_step.path.get_mut(path_index)?;
                        *value = value.saturating_add(delta);
                    }
                }
                _ => unreachable!(),
            }
            Self::push_ruliad_negative_answer(
                &mut negatives,
                answer,
                burn_dragon_universality::ruliad::wire::encode_model_proof_step(
                    candidate_goal,
                    &candidate_step,
                ),
                RuliadStructuredNegativeKind::FieldMutation,
            );
        }
        Some(negatives)
    }

    pub(super) fn ruliad_structured_negative_answers_with_templates(
        answer: &str,
        mutation_count: usize,
        template_count: usize,
    ) -> Vec<(String, RuliadStructuredNegativeKind)> {
        let answer = answer.trim();
        if answer.is_empty() || (mutation_count == 0 && template_count == 0) {
            return Vec::new();
        }
        if let Some(negatives) =
            Self::ruliad_model_proof_step_negative_answers(answer, mutation_count, template_count)
        {
            return negatives;
        }
        let fields = answer
            .split(';')
            .filter_map(|part| {
                let (key, value) = part.split_once('=')?;
                let key = key.trim();
                if key.is_empty() {
                    return None;
                }
                Some((key.to_string(), value.trim().to_string()))
            })
            .collect::<Vec<_>>();
        if fields.is_empty() {
            return (0..mutation_count.max(1))
                .map(|_| {
                    (
                        format!("{answer}_wrong"),
                        RuliadStructuredNegativeKind::FieldMutation,
                    )
                })
                .take(mutation_count.max(template_count))
                .collect();
        }

        let mut negatives = Vec::with_capacity(mutation_count + template_count);
        for template in Self::ruliad_template_collapse_negative_answers(answer, &fields) {
            if negatives.len() >= template_count {
                break;
            }
            Self::push_ruliad_negative_answer(
                &mut negatives,
                answer,
                template,
                RuliadStructuredNegativeKind::TemplateCollapse,
            );
        }

        for index in 0..mutation_count {
            let mutate_index = index % fields.len();
            let mut candidate = fields.clone();
            let mutated = Self::mutate_ruliad_answer_value(&candidate[mutate_index].1, index + 1);
            candidate[mutate_index].1 = mutated;
            let text = candidate
                .into_iter()
                .map(|(key, value)| format!("{key}={value}"))
                .collect::<Vec<_>>()
                .join(";");
            Self::push_ruliad_negative_answer(
                &mut negatives,
                answer,
                text,
                RuliadStructuredNegativeKind::FieldMutation,
            );
        }
        negatives
    }

    pub(super) fn ruliad_structured_negative_answers_with_schema(
        answer: &str,
        mutation_count: usize,
        template_count: usize,
        schema_count: usize,
    ) -> Vec<(String, RuliadStructuredNegativeKind)> {
        let mut negatives = Self::ruliad_structured_negative_answers_with_templates(
            answer,
            mutation_count,
            template_count,
        );
        if schema_count == 0 {
            return negatives;
        }
        negatives.reserve(schema_count);
        for schema_negative in Self::ruliad_schema_collapse_negative_answers(answer)
            .into_iter()
            .take(schema_count)
        {
            Self::push_ruliad_negative_answer(
                &mut negatives,
                answer,
                schema_negative,
                RuliadStructuredNegativeKind::SchemaCollapse,
            );
        }
        negatives
    }

    pub(super) fn push_ruliad_negative_answer(
        negatives: &mut Vec<(String, RuliadStructuredNegativeKind)>,
        answer: &str,
        candidate: String,
        kind: RuliadStructuredNegativeKind,
    ) {
        let candidate = candidate.trim();
        if candidate.is_empty() || candidate == answer {
            return;
        }
        if negatives
            .iter()
            .any(|(existing, _existing_kind)| existing == candidate)
        {
            return;
        }
        negatives.push((candidate.to_string(), kind));
    }

    pub(super) fn ruliad_template_collapse_negative_answers(
        answer: &str,
        fields: &[(String, String)],
    ) -> Vec<String> {
        let has_key = |key: &str| fields.iter().any(|(candidate, _)| candidate == key);
        let mut templates = Vec::<String>::new();
        let mut push = |candidate: &str| {
            if candidate != answer && !templates.iter().any(|existing| existing == candidate) {
                templates.push(candidate.to_string());
            }
        };

        if has_key("ok") && has_key("l") && has_key("r") {
            push("ok=1;l=5;r=5");
            push("ok=1;l=1;r=1");
            push("ok=0;l=0;r=0");
            push("ok=1;l=0;r=0");
        } else if has_key("ok") {
            push("ok=0");
            push("ok=1");
        }

        if has_key("acc") {
            push("acc=0");
            push("acc=1");
        }

        if has_key("xlen") && has_key("xalpha") && has_key("xcounts") && has_key("xedge") {
            push("xlen=13;xalpha=abc;nfcounts=1,1,0;nfedge=ba");
            push("xlen=1;xalpha=01;xcounts=1,1;xedge=00");
            push("xlen=10;xalpha=01;xcounts=10,10;xedge=00");
            push("xlen=21;xalpha=01;xcounts=10,11;xedge=00");
            push("xlen=64;xalpha=01;xcounts=32,32;xedge=00");
        }

        if has_key("nflen") && has_key("nfalpha") && has_key("nfcounts") && has_key("nfedge") {
            push("nflen=5;nfalpha=abc;nfcounts=1,1,0;nfedge=ba");
            push("nflen=1;nfalpha=01;nfcounts=1,1;nfedge=00");
            push("nflen=10;nfalpha=01;nfcounts=10,10;nfedge=00");
            push("nflen=21;nfalpha=01;nfcounts=10,11;nfedge=00");
            push("nflen=64;nfalpha=01;nfcounts=32,32;nfedge=00");
        }

        templates
    }

    pub(super) fn ruliad_template_collapse_negative_answers_from_answer(
        answer: &str,
    ) -> Vec<String> {
        let answer = answer.trim();
        if answer.is_empty() {
            return Vec::new();
        }
        let fields = answer
            .split(';')
            .filter_map(|part| {
                let (key, value) = part.split_once('=')?;
                let key = key.trim();
                if key.is_empty() {
                    return None;
                }
                Some((key.to_string(), value.trim().to_string()))
            })
            .collect::<Vec<_>>();
        if fields.is_empty() {
            return Vec::new();
        }
        Self::ruliad_template_collapse_negative_answers(answer, &fields)
    }

    pub(super) fn ruliad_schema_collapse_negative_answers(answer: &str) -> Vec<String> {
        let answer = answer.trim();
        if answer.is_empty() {
            return Vec::new();
        }
        let fields = answer
            .split(';')
            .filter_map(|part| {
                let (key, value) = part.split_once('=')?;
                let key = key.trim();
                if key.is_empty() {
                    return None;
                }
                Some((key.to_string(), value.trim().to_string()))
            })
            .collect::<Vec<_>>();
        let keys = fields
            .iter()
            .map(|(key, _value)| key.as_str())
            .collect::<Vec<_>>();
        let values = fields
            .iter()
            .map(|(_key, value)| value.as_str())
            .collect::<Vec<_>>();
        let mut negatives = Vec::<String>::new();
        let mut push = |candidate: String| {
            if candidate != answer && !negatives.iter().any(|existing| existing == &candidate) {
                negatives.push(candidate);
            }
        };
        if fields.len() > 1 {
            push(
                fields[..fields.len() - 1]
                    .iter()
                    .map(|(key, value)| format!("{key}={value}"))
                    .collect::<Vec<_>>()
                    .join(";"),
            );
            push(format!("{}={}", fields[0].0, fields[0].1));
        }
        if keys == ["xlen", "xalpha", "xcounts", "xedge"] && values.len() == 4 {
            push(format!(
                "xlen={};nfalpha={};nfcounts={};xedge={}",
                values[0], values[1], values[2], values[3]
            ));
            push(format!(
                "nflen={};nfalpha={};nfcounts={};nfedge={}",
                values[0], values[1], values[2], values[3]
            ));
        } else if keys == ["nflen", "nfalpha", "nfcounts", "nfedge"] && values.len() == 4 {
            push(format!(
                "nflen={};xalpha={};xcounts={};nfedge={}",
                values[0], values[1], values[2], values[3]
            ));
            push(format!(
                "xlen={};xalpha={};xcounts={};xedge={}",
                values[0], values[1], values[2], values[3]
            ));
        } else if keys == ["ok", "l", "r"] && values.len() == 3 {
            push(format!("ok={}", values[0]));
        } else if keys == ["ok"] && values.len() == 1 {
            push(format!("ok={};l=1;r=1", values[0]));
        }
        for prototype in Self::ruliad_cross_contract_prototype_negatives(&keys) {
            push(prototype);
        }
        negatives
    }

    pub(super) fn ruliad_cross_contract_prototype_negatives(keys: &[&str]) -> Vec<String> {
        let contract = keys.join(",");
        let mut prototypes = match contract.as_str() {
            "xlen,xalpha,xcounts,xedge" | "nflen,nfalpha,nfcounts,nfedge" => {
                vec!["ok=1;l=1;r=1", "ok=0;l=0;r=0", "acc=1", "acc=0"]
            }
            "ok,l,r" | "ok" => vec![
                "xlen=1;xalpha=01;xcounts=1,0;xedge=00",
                "nflen=1;nfalpha=ABC;nfcounts=1,0,0;nfedge=AA",
                "acc=1",
                "acc=0",
            ],
            "acc" => vec![
                "ok=1;l=1;r=1",
                "xlen=1;xalpha=01;xcounts=1,0;xedge=00",
                "nflen=1;nfalpha=ABC;nfcounts=1,0,0;nfedge=AA",
            ],
            _ => Vec::new(),
        };
        prototypes.drain(..).map(str::to_string).collect::<Vec<_>>()
    }

    pub(super) fn mutate_ruliad_answer_value(value: &str, delta: usize) -> String {
        let delta = delta.max(1) as u64;
        if value == "0" {
            return "1".to_string();
        }
        if value == "1" {
            return "0".to_string();
        }
        if value.len() > 1 && value.bytes().all(|byte| byte == b'0' || byte == b'1') {
            let mut bytes = value.as_bytes().to_vec();
            let index = delta as usize % bytes.len();
            bytes[index] = if bytes[index] == b'0' { b'1' } else { b'0' };
            return String::from_utf8(bytes).unwrap_or_else(|_| value.to_string());
        }

        let mut output = String::with_capacity(value.len() + 4);
        let bytes = value.as_bytes();
        let mut index = 0usize;
        let mut mutated_any = false;
        while index < bytes.len() {
            if bytes[index].is_ascii_digit() {
                let start = index;
                while index < bytes.len() && bytes[index].is_ascii_digit() {
                    index += 1;
                }
                let text = &value[start..index];
                let width = text.len();
                let modulus = 10u64.saturating_pow(width.min(18) as u32).max(2);
                let parsed = text.parse::<u64>().unwrap_or(0);
                let mut next = (parsed + delta) % modulus;
                if next == parsed {
                    next = (next + 1) % modulus;
                }
                output.push_str(&format!("{next:0width$}"));
                mutated_any = true;
            } else {
                output.push(bytes[index] as char);
                index += 1;
            }
        }
        if mutated_any {
            output
        } else {
            format!("{value}_wrong")
        }
    }

    pub(super) fn ruliad_completion_tokens_from_answer(
        tokenizer: &burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer,
        answer: &str,
        close_marker: &str,
        completion_budget: usize,
    ) -> Option<(Vec<i64>, String)> {
        if answer.trim().is_empty() || completion_budget == 0 {
            return None;
        }
        let full_completion = format!("{}\n{close_marker}", answer.trim());
        let mut payload_tokens = tokenizer.encode_payload(&full_completion);
        payload_tokens.truncate(completion_budget);
        if payload_tokens.is_empty() {
            return None;
        }
        let completion_text = tokenizer.decode_payload(&payload_tokens, true);
        let completion = payload_tokens
            .into_iter()
            .map(i64::from)
            .collect::<Vec<_>>();
        Some((completion, completion_text))
    }

    pub(super) fn ruliad_answer_value_completion_mask(
        tokenizer: &burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer,
        answer: &str,
        completion_len: usize,
    ) -> Vec<bool> {
        let answer = answer.trim();
        if answer.is_empty() || completion_len == 0 {
            return vec![false; completion_len];
        }
        let full_completion = format!("{answer}\n[/R2]");
        let mut mask = vec![false; completion_len];
        if burn_dragon_universality::ruliad::wire::decode_model_proof_step(answer).is_some() {
            let mut segment_start = 0usize;
            for (segment_index, segment) in answer.split('|').enumerate() {
                let value_offset = match segment_index {
                    0 => 1,
                    1 => 2,
                    2 | 3 => 0,
                    _ => return vec![false; completion_len],
                };
                let value_start = segment_start.saturating_add(value_offset);
                let value_end = segment_start.saturating_add(segment.len());
                if value_start < value_end {
                    let prefix_tokens = tokenizer
                        .encode_payload(&full_completion[..value_start])
                        .len();
                    let value_tokens = tokenizer
                        .encode_payload(&full_completion[value_start..value_end])
                        .len();
                    for index in prefix_tokens..prefix_tokens.saturating_add(value_tokens) {
                        if let Some(slot) = mask.get_mut(index) {
                            *slot = true;
                        }
                    }
                }
                segment_start = value_end.saturating_add(1);
            }
            return mask;
        }
        let bytes = answer.as_bytes();
        let mut field_start = 0usize;
        while field_start < answer.len() {
            let field_end = bytes[field_start..]
                .iter()
                .position(|byte| *byte == b';')
                .map(|offset| field_start + offset)
                .unwrap_or(answer.len());
            let field = &answer[field_start..field_end];
            if let Some(eq_offset) = field.find('=') {
                let mut value_start = field_start + eq_offset + 1;
                while value_start < field_end
                    && answer.as_bytes()[value_start].is_ascii_whitespace()
                {
                    value_start += 1;
                }
                let mut value_end = field_end;
                while value_end > value_start
                    && answer.as_bytes()[value_end - 1].is_ascii_whitespace()
                {
                    value_end -= 1;
                }
                if value_start < value_end {
                    let prefix_tokens = tokenizer
                        .encode_payload(&full_completion[..value_start])
                        .len();
                    let value_tokens = tokenizer
                        .encode_payload(&full_completion[value_start..value_end])
                        .len();
                    for index in prefix_tokens..prefix_tokens.saturating_add(value_tokens) {
                        if let Some(slot) = mask.get_mut(index) {
                            *slot = true;
                        }
                    }
                }
            }
            field_start = field_end.saturating_add(1);
        }
        mask
    }

    pub(super) fn ruliad_answer_key_completion_mask(
        tokenizer: &burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer,
        answer: &str,
        completion_len: usize,
    ) -> Vec<bool> {
        let answer = answer.trim();
        if answer.is_empty() || completion_len == 0 {
            return vec![false; completion_len];
        }
        let full_completion = format!("{answer}\n[/R2]");
        let mut mask = vec![false; completion_len];
        let bytes = answer.as_bytes();
        let mut field_start = 0usize;
        while field_start < answer.len() {
            let field_end = bytes[field_start..]
                .iter()
                .position(|byte| *byte == b';')
                .map(|offset| field_start + offset)
                .unwrap_or(answer.len());
            let field = &answer[field_start..field_end];
            if let Some(eq_offset) = field.find('=') {
                let mut key_start = field_start;
                while key_start < field_start + eq_offset
                    && answer.as_bytes()[key_start].is_ascii_whitespace()
                {
                    key_start += 1;
                }
                let mut key_end = field_start + eq_offset;
                while key_end > key_start && answer.as_bytes()[key_end - 1].is_ascii_whitespace() {
                    key_end -= 1;
                }
                if key_start < key_end {
                    let prefix_tokens = tokenizer
                        .encode_payload(&full_completion[..key_start])
                        .len();
                    let key_tokens = tokenizer
                        .encode_payload(&full_completion[key_start..key_end])
                        .len();
                    for index in prefix_tokens..prefix_tokens.saturating_add(key_tokens) {
                        if let Some(slot) = mask.get_mut(index) {
                            *slot = true;
                        }
                    }
                }
            }
            field_start = field_end.saturating_add(1);
        }
        mask
    }

    pub(super) fn ruliad_answer_schema_completion_mask(
        tokenizer: &burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer,
        answer: &str,
        completion_len: usize,
    ) -> Vec<bool> {
        let answer = answer.trim();
        if answer.is_empty() || completion_len == 0 {
            return vec![false; completion_len];
        }
        let full_completion = format!("{answer}\n[/R2]");
        let mut mask = vec![false; completion_len];
        let bytes = answer.as_bytes();
        for (byte_index, byte) in bytes.iter().enumerate() {
            let active = byte.is_ascii_alphabetic() || *byte == b'=' || *byte == b';';
            if !active {
                continue;
            }
            let prefix_tokens = tokenizer
                .encode_payload(&full_completion[..byte_index])
                .len();
            let token_count = tokenizer
                .encode_payload(&full_completion[byte_index..byte_index + 1])
                .len();
            for index in prefix_tokens..prefix_tokens.saturating_add(token_count) {
                if let Some(slot) = mask.get_mut(index) {
                    *slot = true;
                }
            }
        }
        mask
    }

    pub(super) fn ruliad_answer_schema_start_completion_mask(
        tokenizer: &burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer,
        answer: &str,
        completion_len: usize,
    ) -> Vec<bool> {
        let answer = answer.trim();
        if answer.is_empty() || completion_len == 0 {
            return vec![false; completion_len];
        }
        let full_completion = format!("{answer}\n[/R2]");
        let mut mask = vec![false; completion_len];
        let bytes = answer.as_bytes();
        let mut field_start = 0usize;
        while field_start < answer.len() {
            let field_end = bytes[field_start..]
                .iter()
                .position(|byte| *byte == b';')
                .map(|offset| field_start + offset)
                .unwrap_or(answer.len());
            let field = &answer[field_start..field_end];
            if let Some(eq_offset) = field.find('=') {
                let mut key_start = field_start;
                while key_start < field_start + eq_offset
                    && answer.as_bytes()[key_start].is_ascii_whitespace()
                {
                    key_start += 1;
                }
                if key_start < field_start + eq_offset {
                    let first = answer.as_bytes()[key_start];
                    if first.is_ascii_alphabetic() || first == b'_' {
                        let prefix_tokens = tokenizer
                            .encode_payload(&full_completion[..key_start])
                            .len();
                        let token_count = tokenizer
                            .encode_payload(&full_completion[key_start..key_start + 1])
                            .len();
                        for index in prefix_tokens..prefix_tokens.saturating_add(token_count) {
                            if let Some(slot) = mask.get_mut(index) {
                                *slot = true;
                            }
                        }
                    }
                }
            }
            field_start = field_end.saturating_add(1);
        }
        mask
    }

    pub(super) fn ruliad_answer_contract(answer: &str) -> Option<String> {
        if burn_dragon_universality::ruliad::wire::decode_model_proof_step(answer).is_some() {
            return Some("proof_action_step".to_string());
        }
        let mut keys = Vec::<String>::new();
        for part in answer.trim().split(';') {
            let (key, _value) = part.split_once('=')?;
            let key = key.trim();
            if key.is_empty() {
                return None;
            }
            keys.push(key.to_string());
        }
        (!keys.is_empty()).then(|| keys.join(";"))
    }

    pub(super) fn ruliad_answer_fields(answer: &str) -> Option<Vec<(String, String)>> {
        let mut fields = Vec::<(String, String)>::new();
        for part in answer.trim().split(';') {
            let (key, value) = part.split_once('=')?;
            let key = key.trim();
            let value = value.trim();
            if key.is_empty() || value.is_empty() {
                return None;
            }
            fields.push((key.to_string(), value.to_string()));
        }
        (!fields.is_empty()).then_some(fields)
    }

    pub(super) fn ruliad_prompt_schema_value_completion_rows(
        tokenizer: &burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer,
        base_prompt: &[i64],
        answer: &str,
        close_marker: &str,
        completion_budget: usize,
        block_size: usize,
        max_rows: usize,
    ) -> Vec<RuliadPromptSchemaValueRow> {
        if base_prompt.is_empty() || completion_budget == 0 || block_size < 4 || max_rows == 0 {
            return Vec::new();
        }
        let fields = if burn_dragon_universality::ruliad::wire::decode_model_proof_step(answer)
            .is_some()
        {
            let parts = answer.trim().split('|').collect::<Vec<_>>();
            if parts.len() != 4 {
                return Vec::new();
            }
            let Some(goal) = parts[0].strip_prefix('g') else {
                return Vec::new();
            };
            let (source_schema, source_value) = if let Some(source) = parts[1].strip_prefix("a:") {
                ("a:", source)
            } else if let Some(source) = parts[1].strip_prefix("l:") {
                ("l:", source)
            } else {
                return Vec::new();
            };
            vec![
                ("g".to_string(), goal.to_string(), "|".to_string()),
                (
                    format!("g{goal}|{source_schema}"),
                    source_value.to_string(),
                    "|".to_string(),
                ),
                (
                    format!("g{goal}|{}|", parts[1]),
                    parts[2].to_string(),
                    "|".to_string(),
                ),
                (
                    format!("g{goal}|{}|{}|", parts[1], parts[2]),
                    parts[3].to_string(),
                    format!("\n{close_marker}"),
                ),
            ]
        } else {
            let Some(answer_fields) = Self::ruliad_answer_fields(answer) else {
                return Vec::new();
            };
            let mut fields = Vec::with_capacity(answer_fields.len());
            let mut prior = String::new();
            let field_count = answer_fields.len();
            for (index, (key, value)) in answer_fields.into_iter().enumerate() {
                let close = if index + 1 == field_count {
                    format!("\n{close_marker}")
                } else {
                    ";".to_string()
                };
                fields.push((format!("{prior}{key}="), value.clone(), close));
                prior.push_str(&key);
                prior.push('=');
                prior.push_str(&value);
                prior.push(';');
            }
            fields
        };
        let row_completion_budget = completion_budget.min(block_size.saturating_sub(2).max(1));
        let mut rows = Vec::<RuliadPromptSchemaValueRow>::new();
        for (schema_prefix, value, close) in fields {
            if rows.len() >= max_rows {
                break;
            }
            let mut completion_tokens = tokenizer.encode_payload(&format!("{value}{close}"));
            completion_tokens.truncate(row_completion_budget);
            if completion_tokens.is_empty() {
                continue;
            }
            let mut schema_prefix_tokens = tokenizer.encode_payload(&schema_prefix);
            let prefix_budget = block_size
                .saturating_sub(completion_tokens.len())
                .saturating_sub(1);
            if prefix_budget == 0 {
                continue;
            }
            if schema_prefix_tokens.len() > prefix_budget {
                schema_prefix_tokens =
                    schema_prefix_tokens[schema_prefix_tokens.len() - prefix_budget..].to_vec();
            }
            let prompt_budget = block_size
                .saturating_sub(schema_prefix_tokens.len())
                .saturating_sub(completion_tokens.len())
                .max(1);
            let mut prompt = if base_prompt.len() > prompt_budget {
                base_prompt[base_prompt.len() - prompt_budget..].to_vec()
            } else {
                base_prompt.to_vec()
            };
            prompt.extend(schema_prefix_tokens.into_iter().map(i64::from));
            let completion = completion_tokens
                .into_iter()
                .map(i64::from)
                .collect::<Vec<_>>();
            if let Some((inputs, targets, mask)) =
                Self::ruliad_policy_row_from_completion(&prompt, &completion)
            {
                let active_tokens = mask.iter().filter(|value| **value > f32::EPSILON).count();
                if active_tokens > 0 {
                    rows.push((inputs, targets, mask, active_tokens));
                }
            }
        }
        rows
    }

    pub(super) fn ruliad_field_binding_rank_stats(
        logit_margin: Tensor<B, 2>,
        mask_values: &[i64],
        row_count: usize,
        max_len: usize,
        required_margin: f64,
    ) -> RuliadFieldBindingRankStats {
        let Ok(margin_values) = logit_margin.to_data().convert::<f32>().into_vec::<f32>() else {
            return RuliadFieldBindingRankStats::default();
        };
        if margin_values.len() != mask_values.len() || mask_values.len() != row_count * max_len {
            return RuliadFieldBindingRankStats::default();
        }

        let mut token_count = 0usize;
        let mut positive_token_count = 0usize;
        let mut margin_token_count = 0usize;
        let mut margin_sum = 0.0f64;
        let mut pair_count = 0usize;
        let mut exact_pair_rank_count = 0usize;
        let mut exact_pair_margin_count = 0usize;

        for row_index in 0..row_count {
            let row_offset = row_index * max_len;
            let mut row_tokens = 0usize;
            let mut row_positive = 0usize;
            let mut row_margin = 0usize;
            for column in 0..max_len {
                let index = row_offset + column;
                if mask_values[index] == 0 {
                    continue;
                }
                let margin = margin_values[index] as f64;
                if !margin.is_finite() {
                    continue;
                }
                row_tokens = row_tokens.saturating_add(1);
                token_count = token_count.saturating_add(1);
                margin_sum += margin;
                if margin > 0.0 {
                    row_positive = row_positive.saturating_add(1);
                    positive_token_count = positive_token_count.saturating_add(1);
                }
                if margin >= required_margin {
                    row_margin = row_margin.saturating_add(1);
                    margin_token_count = margin_token_count.saturating_add(1);
                }
            }
            if row_tokens > 0 {
                pair_count = pair_count.saturating_add(1);
                if row_positive == row_tokens {
                    exact_pair_rank_count = exact_pair_rank_count.saturating_add(1);
                }
                if row_margin == row_tokens {
                    exact_pair_margin_count = exact_pair_margin_count.saturating_add(1);
                }
            }
        }

        let token_denominator = token_count as f64;
        let pair_denominator = pair_count as f64;
        RuliadFieldBindingRankStats {
            pairs: pair_count,
            tokens: token_count,
            logit_margin_mean: (token_count > 0).then_some(margin_sum / token_denominator),
            positive_token_fraction: (token_count > 0)
                .then_some(positive_token_count as f64 / token_denominator),
            margin_satisfied_token_fraction: (token_count > 0)
                .then_some(margin_token_count as f64 / token_denominator),
            exact_pair_rank_fraction: (pair_count > 0)
                .then_some(exact_pair_rank_count as f64 / pair_denominator),
            exact_pair_margin_fraction: (pair_count > 0)
                .then_some(exact_pair_margin_count as f64 / pair_denominator),
        }
    }

    pub(super) fn ruliad_field_binding_sequence_rank_stats(
        log_probability_margin: Tensor<B, 1>,
        required_margin: f64,
    ) -> RuliadFieldBindingSequenceRankStats {
        let Ok(margins) = log_probability_margin
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
        else {
            return RuliadFieldBindingSequenceRankStats::default();
        };
        let finite = margins
            .into_iter()
            .map(f64::from)
            .filter(|margin| margin.is_finite())
            .collect::<Vec<_>>();
        if finite.is_empty() {
            return RuliadFieldBindingSequenceRankStats::default();
        }
        let denominator = finite.len() as f64;
        RuliadFieldBindingSequenceRankStats {
            pairs: finite.len(),
            log_probability_margin_mean: Some(finite.iter().sum::<f64>() / denominator),
            positive_sequence_fraction: Some(
                finite.iter().filter(|margin| **margin > 0.0).count() as f64 / denominator,
            ),
            margin_satisfied_sequence_fraction: Some(
                finite
                    .iter()
                    .filter(|margin| **margin >= required_margin)
                    .count() as f64
                    / denominator,
            ),
        }
    }
}
