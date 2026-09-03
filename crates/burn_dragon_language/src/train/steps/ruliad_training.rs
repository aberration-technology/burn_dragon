//! Verifier, contrastive, rollout, DAgger, and policy training objectives.

use super::*;

pub(super) struct RuliadProofPolicyObjective<B: Backend> {
    pub(super) loss: Tensor<B, 1>,
    pub(super) semantic_states: usize,
    pub(super) decision_rows: usize,
    pub(super) padded_tokens: usize,
}

impl<B: BackendTrait> LanguageTrainModel<B> {
    pub(super) fn ruliad_field_binding_contrast_loss(
        &self,
        policy_batch: &crate::dataset::RuliadPolicyBatch,
        device: &B::Device,
        block_size: usize,
    ) -> Option<Tensor<B, 1>> {
        let config = self.ruliad_supervision.verifier_reward;
        let weight = self.ruliad_field_binding_contrast_weight();
        let step_index = self.gradient_scale_step.load(Ordering::Relaxed);
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

        #[derive(Clone)]
        struct EligibleSample {
            source_index: usize,
            prompt: Vec<i64>,
            answer: String,
            family: String,
            task_kind: String,
            contract: String,
            presented_action_answers: Option<HashSet<String>>,
            oracle_completion: Vec<i64>,
            value_mask: Vec<bool>,
            schema_mask: Vec<bool>,
        }

        #[derive(Clone)]
        struct NegativeSample {
            current_source_index: Option<usize>,
            answer: String,
            family: String,
            task_kind: String,
            contract: String,
            oracle_completion: Vec<i64>,
            from_replay: bool,
            from_generated_attractor: bool,
            schema_negative: bool,
        }

        #[derive(Clone)]
        struct ContrastCandidate {
            oracle_index: usize,
            negative_index: usize,
            negative_source_index: Option<usize>,
            negative_answer: String,
            discriminative_tokens: usize,
            from_replay: bool,
            from_generated_attractor: bool,
            schema_negative: bool,
        }

        #[derive(Clone)]
        struct ContrastRow {
            prompt: Vec<i64>,
            oracle_completion: Vec<i64>,
            negative_completion: Vec<i64>,
            inputs: Vec<i64>,
            oracle_targets: Vec<i64>,
            negative_targets: Vec<i64>,
            mask: Vec<i64>,
            discriminative_tokens: usize,
            source_index: usize,
            negative_source_index: Option<usize>,
            from_replay: bool,
            from_generated_attractor: bool,
        }

        let mut eligible = Vec::<EligibleSample>::new();
        for (source_index, sample) in policy_batch.samples.iter().enumerate() {
            let answer = sample.item.expected_answer.trim();
            if answer.is_empty() {
                continue;
            }
            let Some(contract) = Self::ruliad_answer_contract(answer) else {
                continue;
            };
            let prompt = sample.prompt_tokens.clone();
            if prompt.is_empty() {
                continue;
            }
            let Some((oracle_completion, _oracle_text, _truncated)) =
                Self::ruliad_oracle_completion_tokens(&tokenizer, sample, completion_budget)
            else {
                continue;
            };
            let value_mask = Self::ruliad_answer_value_completion_mask(
                &tokenizer,
                answer,
                oracle_completion.len(),
            );
            let schema_mask = Self::ruliad_answer_schema_completion_mask(
                &tokenizer,
                answer,
                oracle_completion.len(),
            );
            if !value_mask.iter().any(|active| *active) && !schema_mask.iter().any(|active| *active)
            {
                continue;
            }
            eligible.push(EligibleSample {
                source_index,
                prompt,
                answer: answer.to_string(),
                family: sample.item.family.clone(),
                task_kind: sample.item.task_kind.clone(),
                contract,
                presented_action_answers:
                    burn_dragon_universality::ruliad::ruliad_presented_action_answers(&sample.item)
                        .map(|answers| answers.into_iter().collect()),
                oracle_completion,
                value_mask,
                schema_mask,
            });
        }

        let replay_capacity = config.field_binding_contrast_replay_capacity;
        let replay_snapshot = if replay_capacity > 0 {
            self.ruliad_field_binding_replay
                .lock()
                .map(|replay| replay.iter().cloned().collect::<Vec<_>>())
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let replay_pool_size = replay_snapshot.len();
        let mut negative_pool = eligible
            .iter()
            .map(|sample| NegativeSample {
                current_source_index: Some(sample.source_index),
                answer: sample.answer.clone(),
                family: sample.family.clone(),
                task_kind: sample.task_kind.clone(),
                contract: sample.contract.clone(),
                oracle_completion: sample.oracle_completion.clone(),
                from_replay: false,
                from_generated_attractor: false,
                schema_negative: false,
            })
            .collect::<Vec<_>>();
        negative_pool.extend(replay_snapshot.into_iter().map(|sample| NegativeSample {
            current_source_index: None,
            answer: sample.answer,
            family: sample.family,
            task_kind: sample.task_kind,
            contract: sample.contract,
            oracle_completion: sample.oracle_completion,
            from_replay: true,
            from_generated_attractor: false,
            schema_negative: false,
        }));
        let generated_attractor_snapshot = eligible
            .iter()
            .flat_map(|sample| {
                self.ruliad_generated_attractor_candidates_for_sample(
                    &policy_batch.samples[sample.source_index],
                )
            })
            .collect::<Vec<_>>();
        let generated_attractor_pool_size = generated_attractor_snapshot.len();
        let mut seen_generated_attractors = HashSet::<(String, String, String, String)>::new();
        for entry in generated_attractor_snapshot {
            let key = (
                entry.key.answer.clone(),
                entry.key.family.clone(),
                entry.key.task_kind.clone(),
                entry.key.contract.clone(),
            );
            if !seen_generated_attractors.insert(key) {
                continue;
            }
            let Some((oracle_completion, _completion_text)) =
                Self::ruliad_completion_tokens_from_answer(
                    &tokenizer,
                    &entry.key.answer,
                    burn_dragon_universality::ruliad::RULIAD_V2_DOCUMENT_CLOSE_MARKER,
                    completion_budget,
                )
            else {
                continue;
            };
            negative_pool.push(NegativeSample {
                current_source_index: None,
                answer: entry.key.answer,
                family: entry.key.family,
                task_kind: entry.key.task_kind,
                contract: entry.key.contract,
                oracle_completion,
                from_replay: false,
                from_generated_attractor: true,
                schema_negative: false,
            });
        }
        let mut seen_template_negatives = HashSet::<(String, String, String, String)>::new();
        for sample in eligible.iter() {
            for answer in
                Self::ruliad_template_collapse_negative_answers_from_answer(&sample.answer)
            {
                if answer == sample.answer {
                    continue;
                }
                let Some(contract) = Self::ruliad_answer_contract(&answer) else {
                    continue;
                };
                if contract != sample.contract {
                    continue;
                }
                let key = (
                    answer.clone(),
                    sample.family.clone(),
                    sample.task_kind.clone(),
                    contract.clone(),
                );
                if !seen_template_negatives.insert(key) {
                    continue;
                }
                let Some((oracle_completion, _completion_text)) =
                    Self::ruliad_completion_tokens_from_answer(
                        &tokenizer,
                        &answer,
                        burn_dragon_universality::ruliad::RULIAD_V2_DOCUMENT_CLOSE_MARKER,
                        completion_budget,
                    )
                else {
                    continue;
                };
                negative_pool.push(NegativeSample {
                    current_source_index: None,
                    answer,
                    family: sample.family.clone(),
                    task_kind: sample.task_kind.clone(),
                    contract,
                    oracle_completion,
                    from_replay: false,
                    from_generated_attractor: false,
                    schema_negative: false,
                });
            }
            for answer in Self::ruliad_schema_collapse_negative_answers(&sample.answer) {
                if answer == sample.answer {
                    continue;
                }
                let Some(contract) = Self::ruliad_answer_contract(&answer) else {
                    continue;
                };
                let key = (
                    answer.clone(),
                    sample.family.clone(),
                    sample.task_kind.clone(),
                    contract.clone(),
                );
                if !seen_template_negatives.insert(key) {
                    continue;
                }
                let Some((oracle_completion, _completion_text)) =
                    Self::ruliad_completion_tokens_from_answer(
                        &tokenizer,
                        &answer,
                        burn_dragon_universality::ruliad::RULIAD_V2_DOCUMENT_CLOSE_MARKER,
                        completion_budget,
                    )
                else {
                    continue;
                };
                negative_pool.push(NegativeSample {
                    current_source_index: None,
                    answer,
                    family: sample.family.clone(),
                    task_kind: sample.task_kind.clone(),
                    contract,
                    oracle_completion,
                    from_replay: false,
                    from_generated_attractor: false,
                    schema_negative: true,
                });
            }
        }
        let generated_attractor_negative_pool_size = negative_pool
            .iter()
            .filter(|negative| negative.from_generated_attractor)
            .count();
        let negative_pool_size = negative_pool.len();

        let max_pairs = config.field_binding_contrast_max_pairs.max(1);
        let mut candidates_by_oracle = (0..eligible.len())
            .map(|_| Vec::<ContrastCandidate>::new())
            .collect::<Vec<_>>();
        let mut candidate_pairs = 0usize;
        let mut filtered_presented_action_candidates = 0usize;
        for (oracle_index, oracle) in eligible.iter().enumerate() {
            for (negative_index, negative) in negative_pool.iter().enumerate() {
                if negative.current_source_index == Some(oracle.source_index)
                    || oracle.answer == negative.answer
                    || oracle.family != negative.family
                    || oracle.task_kind != negative.task_kind
                    || (!negative.schema_negative && oracle.contract != negative.contract)
                {
                    continue;
                }
                if !negative.schema_negative
                    && oracle
                        .presented_action_answers
                        .as_ref()
                        .is_some_and(|answers| answers.contains(negative.answer.trim()))
                {
                    filtered_presented_action_candidates =
                        filtered_presented_action_candidates.saturating_add(1);
                    continue;
                }
                let diff_len = oracle
                    .oracle_completion
                    .len()
                    .min(negative.oracle_completion.len());
                let mut discriminative_tokens = 0usize;
                for completion_index in 0..diff_len {
                    let active = if negative.schema_negative {
                        oracle
                            .schema_mask
                            .get(completion_index)
                            .copied()
                            .unwrap_or(false)
                    } else {
                        oracle
                            .value_mask
                            .get(completion_index)
                            .copied()
                            .unwrap_or(false)
                    };
                    if active
                        && oracle.oracle_completion[completion_index]
                            != negative.oracle_completion[completion_index]
                    {
                        discriminative_tokens = discriminative_tokens.saturating_add(1);
                    }
                }
                if discriminative_tokens == 0 {
                    continue;
                }
                candidates_by_oracle[oracle_index].push(ContrastCandidate {
                    oracle_index,
                    negative_index,
                    negative_source_index: negative.current_source_index,
                    negative_answer: negative.answer.clone(),
                    discriminative_tokens,
                    from_replay: negative.from_replay,
                    from_generated_attractor: negative.from_generated_attractor,
                    schema_negative: negative.schema_negative,
                });
                candidate_pairs = candidate_pairs.saturating_add(1);
            }
        }
        let candidate_priority = |candidate: &ContrastCandidate| {
            if candidate.from_generated_attractor {
                0usize
            } else if candidate.schema_negative {
                2
            } else {
                1
            }
        };
        for candidates in candidates_by_oracle.iter_mut() {
            candidates.sort_by(|left, right| {
                candidate_priority(left)
                    .cmp(&candidate_priority(right))
                    .then_with(|| right.discriminative_tokens.cmp(&left.discriminative_tokens))
                    .then_with(|| left.from_replay.cmp(&right.from_replay))
                    .then_with(|| left.negative_index.cmp(&right.negative_index))
            });
        }
        // Spend the bounded auxiliary batch across prompts before taking a second negative for any
        // prompt. A global top-k here repeatedly trained only the rows with the longest byte-level
        // answer differences and left most prompts without a binding gradient.
        let mut selected_candidates = Vec::<ContrastCandidate>::new();
        let mut rank = 0usize;
        while selected_candidates.len() < max_pairs {
            let mut selected_this_round = 0usize;
            for candidates in candidates_by_oracle.iter() {
                if let Some(candidate) = candidates.get(rank) {
                    selected_candidates.push(candidate.clone());
                    selected_this_round = selected_this_round.saturating_add(1);
                    if selected_candidates.len() == max_pairs {
                        break;
                    }
                }
            }
            if selected_this_round == 0 {
                break;
            }
            rank = rank.saturating_add(1);
        }
        let mut rows = Vec::<ContrastRow>::new();
        for candidate in selected_candidates {
            let oracle = &eligible[candidate.oracle_index];
            let Some((negative_completion, _negative_text)) =
                Self::ruliad_completion_tokens_from_answer(
                    &tokenizer,
                    &candidate.negative_answer,
                    policy_batch.samples[oracle.source_index]
                        .item
                        .document_close_marker(),
                    completion_budget,
                )
            else {
                continue;
            };
            let prompt = Self::ruliad_trim_prompt_for_completion(
                &oracle.prompt,
                oracle
                    .oracle_completion
                    .len()
                    .max(negative_completion.len()),
                block_size,
            );
            let Some((mut inputs, mut oracle_targets, _oracle_mask)) =
                Self::ruliad_policy_row_from_completion(&prompt, &oracle.oracle_completion)
            else {
                continue;
            };
            let completion_start = prompt.len().saturating_sub(1).min(oracle_targets.len());
            let diff_len = oracle
                .oracle_completion
                .len()
                .min(negative_completion.len());
            let mut negative_targets = oracle_targets.clone();
            let mut mask = vec![0i64; oracle_targets.len()];
            let mut first_discriminative_token = None;
            for (completion_index, (&oracle_token, &negative_token)) in oracle
                .oracle_completion
                .iter()
                .zip(&negative_completion)
                .take(diff_len)
                .enumerate()
            {
                let target_index = completion_start.saturating_add(completion_index);
                let active = if candidate.schema_negative {
                    oracle
                        .schema_mask
                        .get(completion_index)
                        .copied()
                        .unwrap_or(false)
                } else {
                    oracle
                        .value_mask
                        .get(completion_index)
                        .copied()
                        .unwrap_or(false)
                };
                if active && target_index < negative_targets.len() && oracle_token != negative_token
                {
                    negative_targets[target_index] = negative_token;
                    mask[target_index] = 1;
                    first_discriminative_token = Some(completion_index);
                    break;
                }
            }
            let Some(first_discriminative_token) = first_discriminative_token else {
                continue;
            };
            let causal_len = completion_start
                .saturating_add(first_discriminative_token)
                .saturating_add(1);
            inputs.truncate(causal_len);
            oracle_targets.truncate(causal_len);
            negative_targets.truncate(causal_len);
            mask.truncate(causal_len);
            rows.push(ContrastRow {
                prompt,
                oracle_completion: oracle.oracle_completion.clone(),
                negative_completion,
                inputs,
                oracle_targets,
                negative_targets,
                mask,
                discriminative_tokens: 1,
                source_index: oracle.source_index,
                negative_source_index: candidate.negative_source_index,
                from_replay: candidate.from_replay,
                from_generated_attractor: candidate.from_generated_attractor,
            });
        }

        if replay_capacity > 0
            && !eligible.is_empty()
            && let Ok(mut replay) = self.ruliad_field_binding_replay.lock()
        {
            for sample in eligible.iter() {
                replay.push_back(RuliadFieldBindingReplaySample {
                    answer: sample.answer.clone(),
                    family: sample.family.clone(),
                    task_kind: sample.task_kind.clone(),
                    contract: sample.contract.clone(),
                    oracle_completion: sample.oracle_completion.clone(),
                });
            }
            while replay.len() > replay_capacity {
                replay.pop_front();
            }
        }

        if rows.is_empty() {
            let replay_summary = self.ruliad_generated_attractor_summary();
            self.write_ruliad_generated_attractor_telemetry(
                RuliadGeneratedAttractorReplayTelemetry {
                    version: 1,
                    step_index,
                    source: "field_binding".to_string(),
                    skip_reason: self
                        .ruliad_generated_attractor_replay_skip_reason(
                            &replay_summary,
                            generated_attractor_negative_pool_size,
                        )
                        .or_else(|| Some("no_counterfactual_pairs".to_string())),
                    observed_completion_rows: 0,
                    recorded_attractor_rows: 0,
                    selected_candidate_rows: generated_attractor_negative_pool_size,
                    selected_field_binding_pairs: 0,
                    replay_pool_size: replay_summary.pool_size,
                    active_attractor_count: replay_summary.active_count,
                    active_observation_count: replay_summary.active_observation_count,
                    distinct_answer_count: replay_summary.distinct_answers,
                    dominant_answer_count: replay_summary.dominant_count,
                    dominant_answer_fraction: replay_summary.dominant_fraction(),
                    min_count: config.generated_attractor_replay_min_count.max(1),
                    max_candidates: config.generated_attractor_replay_max_candidates,
                    min_distinct_answers: config
                        .generated_attractor_replay_min_distinct_answers
                        .max(1),
                    max_dominant_fraction: config.generated_attractor_replay_max_dominant_fraction,
                },
            );
            self.write_ruliad_field_binding_contrast_telemetry(
                RuliadFieldBindingContrastTelemetry {
                    version: 3,
                    objective: RULIAD_FIELD_BINDING_OBJECTIVE,
                    step_index,
                    skip_reason: Some("no_counterfactual_pairs".to_string()),
                    sample_groups: eligible.len(),
                    oracle_prompt_count: 0,
                    prompt_pairs: 0,
                    contrast_pairs: 0,
                    candidate_pairs,
                    filtered_presented_action_candidates,
                    contrast_discriminative_tokens: 0,
                    negative_pool_size,
                    replay_pool_size,
                    replay_contrast_pairs: 0,
                    generated_attractor_pool_size,
                    generated_attractor_negative_pool_size,
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
                    field_binding_contrast_margin: config.field_binding_contrast_margin,
                    field_binding_contrast_pair_weight: config.field_binding_contrast_pair_weight,
                },
            );
            return None;
        }

        let mut participating_samples = HashSet::<usize>::new();
        let mut oracle_prompts = HashSet::<usize>::new();
        for row in rows.iter() {
            oracle_prompts.insert(row.source_index);
            participating_samples.insert(row.source_index);
            if let Some(negative_source_index) = row.negative_source_index {
                participating_samples.insert(negative_source_index);
            }
        }
        let replay_contrast_pairs = rows.iter().filter(|row| row.from_replay).count();
        let generated_attractor_contrast_pairs = rows
            .iter()
            .filter(|row| row.from_generated_attractor)
            .count();
        let contrast_discriminative_tokens = rows
            .iter()
            .map(|row| row.discriminative_tokens)
            .sum::<usize>();

        let max_len = rows.iter().map(|row| row.inputs.len()).max()?.max(1);
        let row_count = rows.len();
        let mut input_values = vec![0i64; row_count * max_len];
        let mut oracle_target_values = vec![0i64; row_count * max_len];
        let mut negative_target_values = vec![0i64; row_count * max_len];
        let mut mask_values = vec![0i64; row_count * max_len];
        for (row_index, row) in rows.iter().enumerate() {
            let offset = row_index * max_len;
            let len = row.inputs.len().min(max_len);
            input_values[offset..offset + len].copy_from_slice(&row.inputs[..len]);
            oracle_target_values[offset..offset + len].copy_from_slice(&row.oracle_targets[..len]);
            negative_target_values[offset..offset + len]
                .copy_from_slice(&row.negative_targets[..len]);
            mask_values[offset..offset + len].copy_from_slice(&row.mask[..len]);
        }
        let inputs = Tensor::<B, 2, Int>::from_data(
            TensorData::new(input_values, [row_count, max_len]),
            device,
        );
        let oracle_targets = Tensor::<B, 2, Int>::from_data(
            TensorData::new(oracle_target_values, [row_count, max_len]),
            device,
        );
        let negative_targets = Tensor::<B, 2, Int>::from_data(
            TensorData::new(negative_target_values, [row_count, max_len]),
            device,
        );
        let mask = Tensor::<B, 2, Int>::from_data(
            TensorData::new(mask_values.clone(), [row_count, max_len]),
            device,
        );
        let logits = self.model.forward(inputs);
        let oracle_logits = selected_token_logits(logits.clone(), oracle_targets);
        let negative_logits = selected_token_logits(logits, negative_targets);
        let logit_margin = oracle_logits.clone() - negative_logits.clone();
        let contrast_margin = config.field_binding_contrast_margin.max(0.0);
        let should_collect_rank_metric = config.field_binding_contrast_rank_metric_every_steps > 0
            && step_index.is_multiple_of(config.field_binding_contrast_rank_metric_every_steps);
        let rank_stats = should_collect_rank_metric.then(|| {
            Self::ruliad_field_binding_rank_stats(
                logit_margin.clone(),
                &mask_values,
                row_count,
                max_len,
                contrast_margin as f64,
            )
        });
        let pair_weight = config.field_binding_contrast_pair_weight.max(0.0);
        let sequence_log_probability_margin = if pair_weight > f32::EPSILON {
            let prompts = rows
                .iter()
                .map(|row| row.prompt.clone())
                .collect::<Vec<_>>();
            let candidates = rows
                .iter()
                .map(|row| {
                    vec![
                        row.oracle_completion.clone(),
                        row.negative_completion.clone(),
                    ]
                })
                .collect::<Vec<_>>();
            let scores = crate::train::ruliad_policy::sequence_completion_score_tensor(
                &self.model,
                &prompts,
                &candidates,
                device,
            )
            .ok()?;
            if scores.group_sizes.iter().any(|group_size| *group_size != 2) {
                return None;
            }
            let scores = scores.mean_log_scores.reshape([row_count, 2]);
            let oracle_scores = scores
                .clone()
                .slice([0..row_count, 0..1])
                .reshape([row_count]);
            let negative_scores = scores.slice([0..row_count, 1..2]).reshape([row_count]);
            Some(oracle_scores - negative_scores)
        } else {
            None
        };
        let sequence_rank_stats = should_collect_rank_metric
            .then(|| {
                sequence_log_probability_margin.clone().map(|margin| {
                    Self::ruliad_field_binding_sequence_rank_stats(margin, contrast_margin as f64)
                })
            })
            .flatten();
        self.write_ruliad_field_binding_contrast_telemetry(RuliadFieldBindingContrastTelemetry {
            version: 3,
            objective: RULIAD_FIELD_BINDING_OBJECTIVE,
            step_index,
            skip_reason: None,
            sample_groups: participating_samples.len(),
            oracle_prompt_count: oracle_prompts.len(),
            prompt_pairs: row_count,
            contrast_pairs: row_count,
            candidate_pairs,
            filtered_presented_action_candidates,
            contrast_discriminative_tokens,
            negative_pool_size,
            replay_pool_size,
            replay_contrast_pairs,
            generated_attractor_pool_size,
            generated_attractor_negative_pool_size,
            generated_attractor_contrast_pairs,
            rank_metric_pairs: rank_stats.as_ref().map(|stats| stats.pairs).unwrap_or(0),
            rank_metric_tokens: rank_stats.as_ref().map(|stats| stats.tokens).unwrap_or(0),
            logit_margin_mean: rank_stats
                .as_ref()
                .and_then(|stats| stats.logit_margin_mean),
            positive_token_fraction: rank_stats
                .as_ref()
                .and_then(|stats| stats.positive_token_fraction),
            margin_satisfied_token_fraction: rank_stats
                .as_ref()
                .and_then(|stats| stats.margin_satisfied_token_fraction),
            exact_pair_rank_fraction: rank_stats
                .as_ref()
                .and_then(|stats| stats.exact_pair_rank_fraction),
            exact_pair_margin_fraction: rank_stats
                .as_ref()
                .and_then(|stats| stats.exact_pair_margin_fraction),
            sequence_rank_metric_pairs: sequence_rank_stats
                .as_ref()
                .map(|stats| stats.pairs)
                .unwrap_or(0),
            sequence_log_probability_margin_mean: sequence_rank_stats
                .as_ref()
                .and_then(|stats| stats.log_probability_margin_mean),
            positive_sequence_fraction: sequence_rank_stats
                .as_ref()
                .and_then(|stats| stats.positive_sequence_fraction),
            sequence_margin_satisfied_fraction: sequence_rank_stats
                .as_ref()
                .and_then(|stats| stats.margin_satisfied_sequence_fraction),
            field_binding_contrast_weight: weight,
            field_binding_contrast_margin: config.field_binding_contrast_margin,
            field_binding_contrast_pair_weight: config.field_binding_contrast_pair_weight,
        });
        let replay_summary = self.ruliad_generated_attractor_summary();
        self.write_ruliad_generated_attractor_telemetry(RuliadGeneratedAttractorReplayTelemetry {
            version: 1,
            step_index,
            source: "field_binding".to_string(),
            skip_reason: self.ruliad_generated_attractor_replay_skip_reason(
                &replay_summary,
                generated_attractor_negative_pool_size,
            ),
            observed_completion_rows: 0,
            recorded_attractor_rows: 0,
            selected_candidate_rows: generated_attractor_negative_pool_size,
            selected_field_binding_pairs: generated_attractor_contrast_pairs,
            replay_pool_size: replay_summary.pool_size,
            active_attractor_count: replay_summary.active_count,
            active_observation_count: replay_summary.active_observation_count,
            distinct_answer_count: replay_summary.distinct_answers,
            dominant_answer_count: replay_summary.dominant_count,
            dominant_answer_fraction: replay_summary.dominant_fraction(),
            min_count: config.generated_attractor_replay_min_count.max(1),
            max_candidates: config.generated_attractor_replay_max_candidates,
            min_distinct_answers: config
                .generated_attractor_replay_min_distinct_answers
                .max(1),
            max_dominant_fraction: config.generated_attractor_replay_max_dominant_fraction,
        });
        let token_loss = masked_token_mean(
            activation::softplus(
                negative_logits.clone() - oracle_logits.clone() + contrast_margin,
                1.0,
            ),
            Some(mask.clone()),
        );
        let loss = sequence_log_probability_margin.map_or(token_loss.clone(), |margin| {
            token_loss
                + activation::softplus(margin.mul_scalar(-1.0) + contrast_margin, 1.0)
                    .mean()
                    .reshape([1])
                    .mul_scalar(pair_weight)
        });
        Some(loss.mul_scalar(weight))
    }

    pub(super) fn ruliad_structured_answer_contrast_loss(
        &self,
        policy_batch: &crate::dataset::RuliadPolicyBatch,
        device: &B::Device,
        block_size: usize,
    ) -> Option<Tensor<B, 1>> {
        let config = self.ruliad_supervision.verifier_reward;
        let weight = self.ruliad_structured_contrast_weight();
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

        #[derive(Clone)]
        struct ContrastRow {
            inputs: Vec<i64>,
            oracle_targets: Vec<i64>,
            negative_targets: Vec<i64>,
            mask: Vec<i64>,
            discriminative_tokens: usize,
        }

        let mut rows = Vec::<ContrastRow>::new();
        let mut oracle_completion_rows = 0usize;
        let mut field_negative_completion_rows = 0usize;
        let mut template_negative_completion_rows = 0usize;
        let mut schema_negative_completion_rows = 0usize;
        let mut generated_attractor_negative_completion_rows = 0usize;
        let mut sample_groups = 0usize;
        for sample in policy_batch.samples.iter() {
            let mut prompt = sample.prompt_tokens.clone();
            if prompt.is_empty() {
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
            let value_mask = Self::ruliad_answer_value_completion_mask(
                &tokenizer,
                &sample.item.expected_answer,
                oracle_completion.len(),
            );
            let schema_mask = Self::ruliad_answer_schema_completion_mask(
                &tokenizer,
                &sample.item.expected_answer,
                oracle_completion.len(),
            );
            if !value_mask.iter().any(|active| *active) && !schema_mask.iter().any(|active| *active)
            {
                continue;
            }
            let Some((inputs, oracle_targets, _oracle_mask)) =
                Self::ruliad_policy_row_from_completion(&prompt, &oracle_completion)
            else {
                continue;
            };
            let completion_start = prompt.len().saturating_sub(1).min(oracle_targets.len());
            oracle_completion_rows = oracle_completion_rows.saturating_add(1);
            let mut sample_pair_count = 0usize;

            for (negative, negative_kind) in Self::ruliad_structured_negative_answers_with_schema(
                &sample.item.expected_answer,
                config.structured_negative_count,
                config.structured_template_negative_count,
                config.structured_schema_negative_count,
            ) {
                let Some((completion, _completion_text)) =
                    Self::ruliad_completion_tokens_from_answer(
                        &tokenizer,
                        &negative,
                        sample.item.document_close_marker(),
                        completion_budget,
                    )
                else {
                    continue;
                };
                let diff_len = oracle_completion.len().min(completion.len());
                let mut negative_targets = oracle_targets.clone();
                let mut mask = vec![0i64; oracle_targets.len()];
                let mut discriminative_tokens = 0usize;
                for completion_index in 0..diff_len {
                    let target_index = completion_start.saturating_add(completion_index);
                    let active = match negative_kind {
                        RuliadStructuredNegativeKind::SchemaCollapse => {
                            schema_mask.get(completion_index).copied().unwrap_or(false)
                        }
                        RuliadStructuredNegativeKind::FieldMutation
                        | RuliadStructuredNegativeKind::TemplateCollapse => {
                            value_mask.get(completion_index).copied().unwrap_or(false)
                        }
                    };
                    if active
                        && target_index < negative_targets.len()
                        && oracle_completion[completion_index] != completion[completion_index]
                    {
                        negative_targets[target_index] = completion[completion_index];
                        mask[target_index] = 1;
                        discriminative_tokens = discriminative_tokens.saturating_add(1);
                    }
                }
                if discriminative_tokens == 0 {
                    continue;
                }
                rows.push(ContrastRow {
                    inputs: inputs.clone(),
                    oracle_targets: oracle_targets.clone(),
                    negative_targets,
                    mask,
                    discriminative_tokens,
                });
                sample_pair_count = sample_pair_count.saturating_add(1);
                match negative_kind {
                    RuliadStructuredNegativeKind::FieldMutation => {
                        field_negative_completion_rows =
                            field_negative_completion_rows.saturating_add(1);
                    }
                    RuliadStructuredNegativeKind::TemplateCollapse => {
                        template_negative_completion_rows =
                            template_negative_completion_rows.saturating_add(1);
                    }
                    RuliadStructuredNegativeKind::SchemaCollapse => {
                        schema_negative_completion_rows =
                            schema_negative_completion_rows.saturating_add(1);
                    }
                }
            }
            let expected_contract = Self::ruliad_answer_contract(&sample.item.expected_answer);
            for entry in self.ruliad_generated_attractor_candidates_for_sample(sample) {
                let Some((completion, _completion_text)) =
                    Self::ruliad_completion_tokens_from_answer(
                        &tokenizer,
                        &entry.key.answer,
                        sample.item.document_close_marker(),
                        completion_budget,
                    )
                else {
                    continue;
                };
                let schema_negative = expected_contract
                    .as_ref()
                    .is_some_and(|contract| contract != &entry.key.contract);
                let diff_len = oracle_completion.len().min(completion.len());
                let mut negative_targets = oracle_targets.clone();
                let mut mask = vec![0i64; oracle_targets.len()];
                let mut discriminative_tokens = 0usize;
                for completion_index in 0..diff_len {
                    let target_index = completion_start.saturating_add(completion_index);
                    let active = if schema_negative {
                        schema_mask.get(completion_index).copied().unwrap_or(false)
                    } else {
                        value_mask.get(completion_index).copied().unwrap_or(false)
                    };
                    if active
                        && target_index < negative_targets.len()
                        && oracle_completion[completion_index] != completion[completion_index]
                    {
                        negative_targets[target_index] = completion[completion_index];
                        mask[target_index] = 1;
                        discriminative_tokens = discriminative_tokens.saturating_add(1);
                    }
                }
                if discriminative_tokens == 0 {
                    continue;
                }
                rows.push(ContrastRow {
                    inputs: inputs.clone(),
                    oracle_targets: oracle_targets.clone(),
                    negative_targets,
                    mask,
                    discriminative_tokens,
                });
                sample_pair_count = sample_pair_count.saturating_add(1);
                generated_attractor_negative_completion_rows =
                    generated_attractor_negative_completion_rows.saturating_add(1);
            }
            sample_groups = sample_groups.saturating_add(usize::from(sample_pair_count > 0));
        }
        if rows.is_empty() {
            self.write_ruliad_structured_contrast_telemetry(RuliadStructuredContrastTelemetry {
                version: 1,
                step_index: self.gradient_scale_step.load(Ordering::Relaxed),
                skip_reason: Some("no_field_value_pairs".to_string()),
                sample_groups,
                oracle_completion_rows,
                field_negative_completion_rows,
                template_negative_completion_rows,
                schema_negative_completion_rows,
                generated_attractor_negative_completion_rows,
                contrast_pairs: 0,
                contrast_discriminative_tokens: 0,
                structured_contrast_weight: weight,
                structured_contrast_margin: config.structured_contrast_margin,
            });
            return None;
        }
        let contrast_discriminative_tokens = rows
            .iter()
            .map(|row| row.discriminative_tokens)
            .sum::<usize>();
        self.write_ruliad_structured_contrast_telemetry(RuliadStructuredContrastTelemetry {
            version: 1,
            step_index: self.gradient_scale_step.load(Ordering::Relaxed),
            skip_reason: None,
            sample_groups,
            oracle_completion_rows,
            field_negative_completion_rows,
            template_negative_completion_rows,
            schema_negative_completion_rows,
            generated_attractor_negative_completion_rows,
            contrast_pairs: rows.len(),
            contrast_discriminative_tokens,
            structured_contrast_weight: weight,
            structured_contrast_margin: config.structured_contrast_margin,
        });

        let max_len = rows.iter().map(|row| row.inputs.len()).max()?.max(1);
        let row_count = rows.len();
        let mut input_values = vec![0i64; row_count * max_len];
        let mut oracle_target_values = vec![0i64; row_count * max_len];
        let mut negative_target_values = vec![0i64; row_count * max_len];
        let mut mask_values = vec![0i64; row_count * max_len];
        for (row_index, row) in rows.into_iter().enumerate() {
            let offset = row_index * max_len;
            let len = row.inputs.len().min(max_len);
            input_values[offset..offset + len].copy_from_slice(&row.inputs[..len]);
            oracle_target_values[offset..offset + len].copy_from_slice(&row.oracle_targets[..len]);
            negative_target_values[offset..offset + len]
                .copy_from_slice(&row.negative_targets[..len]);
            mask_values[offset..offset + len].copy_from_slice(&row.mask[..len]);
        }
        let inputs = Tensor::<B, 2, Int>::from_data(
            TensorData::new(input_values, [row_count, max_len]),
            device,
        );
        let oracle_targets = Tensor::<B, 2, Int>::from_data(
            TensorData::new(oracle_target_values, [row_count, max_len]),
            device,
        );
        let negative_targets = Tensor::<B, 2, Int>::from_data(
            TensorData::new(negative_target_values, [row_count, max_len]),
            device,
        );
        let mask = Tensor::<B, 2, Int>::from_data(
            TensorData::new(mask_values, [row_count, max_len]),
            device,
        );
        let logits = self.model.forward(inputs);
        let oracle_logits = selected_token_logits(logits.clone(), oracle_targets);
        let negative_logits = selected_token_logits(logits, negative_targets);
        Some(
            masked_token_mean(
                activation::softplus(
                    negative_logits - oracle_logits + config.structured_contrast_margin.max(0.0),
                    1.0,
                ),
                Some(mask),
            )
            .mul_scalar(weight),
        )
    }

    pub(super) fn ruliad_verifier_rollout_imitation_loss(
        &self,
        policy_batch: &crate::dataset::RuliadPolicyBatch,
        device: &B::Device,
        block_size: usize,
    ) -> Option<Tensor<B, 1>> {
        let config = self.ruliad_supervision.verifier_reward;
        if !self.ruliad_verifier_rollout_feedback_active()
            || policy_batch.samples.is_empty()
            || self.pipeline_enabled()
        {
            return None;
        }
        let imitation_weight = config.rollout_imitation_weight.max(0.0);
        let recovery_weight = config.rollout_recovery_weight.max(0.0);
        let tokenizer =
            burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer::from_config(
                &policy_batch.tokenization,
            )
            .ok()?;
        let completion_budget = config
            .max_completion_tokens
            .max(1)
            .min(block_size.saturating_sub(1).max(1));
        let prompt_budget = block_size.saturating_sub(completion_budget).max(1);
        let max_rows = config.rollout_imitation_max_rows_per_step.max(1);
        let step_index = self.gradient_scale_step.load(Ordering::Relaxed);

        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        enum RolloutFeedbackKind {
            Imitation,
            Recovery,
        }

        #[derive(Clone)]
        struct RolloutFeedbackRow {
            inputs: Vec<i64>,
            targets: Vec<i64>,
            mask: Vec<f32>,
            weight: f32,
            kind: RolloutFeedbackKind,
        }

        let mut rows = Vec::<RolloutFeedbackRow>::new();
        let mut sample_groups = 0usize;
        let mut generated_completion_rows = 0usize;
        let mut recorded_attractor_rows = 0usize;
        let mut verifier_match_rows = 0usize;
        let mut semantic_match_rows = 0usize;
        let mut partial_rows = 0usize;
        let mut schema_wrong_rows = 0usize;
        let mut malformed_rows = 0usize;
        let mut missing_rows = 0usize;
        let mut recovery_partial_rows = 0usize;
        let mut recovery_schema_wrong_rows = 0usize;
        let mut recovery_malformed_rows = 0usize;
        let mut recovery_missing_rows = 0usize;
        let mut field_accuracy_sum = 0.0f64;
        let mut partial_progress_sum = 0.0f64;
        let mut completion_quality_sum = 0.0f64;

        'samples: for sample in policy_batch.samples.iter() {
            let mut prompt = sample.prompt_tokens.clone();
            if prompt.is_empty() {
                continue;
            }
            if prompt.len() > prompt_budget {
                prompt = prompt[prompt.len() - prompt_budget..].to_vec();
            }
            let oracle_row = if recovery_weight > f32::EPSILON {
                Self::ruliad_oracle_completion_tokens(&tokenizer, sample, completion_budget)
                    .and_then(|(oracle_completion, _oracle_text, _truncated)| {
                        Self::ruliad_policy_row_from_completion(&prompt, &oracle_completion)
                    })
            } else {
                None
            };
            let mut generated_for_sample = 0usize;
            for group_index in 0..config.group_size.max(1) {
                if rows.len() >= max_rows {
                    break 'samples;
                }
                let rollout_seed = Self::mix_ruliad_policy_seed(
                    (step_index as u64).rotate_left(17)
                        ^ (sample.item.sample_index as u64).rotate_left(7)
                        ^ group_index as u64,
                );
                let generated = crate::generation::generate_tokens_seeded(
                    &self.model,
                    prompt.clone(),
                    device,
                    crate::generation::GenerationSettings {
                        max_new_tokens: Some(completion_budget),
                        temperature: config.temperature,
                        top_k: Some(config.top_k),
                        strategy: crate::generation::ContextStrategy::Infinite,
                        stop_on_token: policy_batch.stop_token_id,
                    },
                    rollout_seed,
                    None,
                )
                .ok()?;
                if generated.len() <= prompt.len() {
                    continue;
                }
                let completion = generated[prompt.len()..].to_vec();
                if completion.is_empty() {
                    continue;
                }
                let completion_tokens = completion
                    .iter()
                    .filter_map(|token| u32::try_from(*token).ok())
                    .collect::<Vec<_>>();
                let completion_text = tokenizer.decode_payload(&completion_tokens, true);
                let score = burn_dragon_universality::ruliad::score_ruliad_item_completion(
                    &sample.item,
                    Some(&completion_text),
                );
                generated_completion_rows = generated_completion_rows.saturating_add(1);
                recorded_attractor_rows = recorded_attractor_rows.saturating_add(usize::from(
                    self.record_ruliad_generated_attractor(
                        sample,
                        &completion_text,
                        &score,
                        step_index,
                    ),
                ));
                generated_for_sample = generated_for_sample.saturating_add(1);
                match score.status {
                    burn_dragon_universality::ruliad::RuliadAnswerStatus::VerifierMatch => {
                        verifier_match_rows = verifier_match_rows.saturating_add(1)
                    }
                    burn_dragon_universality::ruliad::RuliadAnswerStatus::SemanticMatch => {
                        semantic_match_rows = semantic_match_rows.saturating_add(1)
                    }
                    burn_dragon_universality::ruliad::RuliadAnswerStatus::Partial => {
                        partial_rows = partial_rows.saturating_add(1)
                    }
                    burn_dragon_universality::ruliad::RuliadAnswerStatus::SchemaValidWrong => {
                        schema_wrong_rows = schema_wrong_rows.saturating_add(1)
                    }
                    burn_dragon_universality::ruliad::RuliadAnswerStatus::Malformed => {
                        malformed_rows = malformed_rows.saturating_add(1)
                    }
                    burn_dragon_universality::ruliad::RuliadAnswerStatus::Missing => {
                        missing_rows = missing_rows.saturating_add(1)
                    }
                }
                let field_accuracy = if score.expected_field_count == 0 {
                    0.0
                } else {
                    score.correct_field_count as f64 / score.expected_field_count as f64
                };
                field_accuracy_sum += field_accuracy;
                partial_progress_sum += score.partial_progress_ppm as f64 / 1_000_000.0;
                completion_quality_sum += score.completion_quality_ppm as f64 / 1_000_000.0;
                let has_imitation_signal = Self::ruliad_score_has_policy_correctness_signal(
                    &score,
                    config.rollout_imitation_min_partial_progress_ppm,
                    config.rollout_imitation_min_completion_quality_ppm,
                );
                let has_recovery_signal = Self::ruliad_score_has_rollout_recovery_signal(
                    &score,
                    config.rollout_imitation_min_partial_progress_ppm,
                    config.rollout_imitation_min_completion_quality_ppm,
                );
                if !has_imitation_signal && !has_recovery_signal {
                    continue;
                }
                let is_correct = matches!(
                    score.status,
                    burn_dragon_universality::ruliad::RuliadAnswerStatus::VerifierMatch
                        | burn_dragon_universality::ruliad::RuliadAnswerStatus::SemanticMatch
                );
                if imitation_weight > f32::EPSILON
                    && let Some((inputs, targets, mask)) =
                        Self::ruliad_policy_row_from_completion(&prompt, &completion)
                {
                    rows.push(RolloutFeedbackRow {
                        inputs,
                        targets,
                        mask,
                        weight: imitation_weight,
                        kind: RolloutFeedbackKind::Imitation,
                    });
                }
                if recovery_weight > f32::EPSILON
                    && !is_correct
                    && has_recovery_signal
                    && let Some((oracle_inputs, oracle_targets, oracle_mask)) = oracle_row.as_ref()
                {
                    let completion_start = prompt.len().saturating_sub(1).min(oracle_inputs.len());
                    let mut corrupted_inputs = oracle_inputs.clone();
                    for (index, value) in corrupted_inputs
                        .iter_mut()
                        .enumerate()
                        .skip(completion_start)
                    {
                        let completion_index = index - completion_start;
                        if let Some(generated_token) = completion.get(completion_index) {
                            *value = *generated_token;
                        }
                    }
                    rows.push(RolloutFeedbackRow {
                        inputs: corrupted_inputs,
                        targets: oracle_targets.clone(),
                        mask: oracle_mask.clone(),
                        weight: recovery_weight,
                        kind: RolloutFeedbackKind::Recovery,
                    });
                    match score.status {
                        burn_dragon_universality::ruliad::RuliadAnswerStatus::Partial => {
                            recovery_partial_rows = recovery_partial_rows.saturating_add(1)
                        }
                        burn_dragon_universality::ruliad::RuliadAnswerStatus::SchemaValidWrong => {
                            recovery_schema_wrong_rows =
                                recovery_schema_wrong_rows.saturating_add(1)
                        }
                        burn_dragon_universality::ruliad::RuliadAnswerStatus::Malformed => {
                            recovery_malformed_rows = recovery_malformed_rows.saturating_add(1)
                        }
                        burn_dragon_universality::ruliad::RuliadAnswerStatus::Missing => {
                            recovery_missing_rows = recovery_missing_rows.saturating_add(1)
                        }
                        burn_dragon_universality::ruliad::RuliadAnswerStatus::VerifierMatch
                        | burn_dragon_universality::ruliad::RuliadAnswerStatus::SemanticMatch => {}
                    }
                }
            }
            sample_groups += usize::from(generated_for_sample > 0);
        }

        let rate_ppm = |count: usize| -> usize {
            count
                .saturating_mul(1_000_000)
                .checked_div(generated_completion_rows)
                .unwrap_or_default()
        };
        let verifier_rate_ppm = rate_ppm(verifier_match_rows.saturating_add(semantic_match_rows));
        let schema_wrong_rate_ppm = rate_ppm(schema_wrong_rows);
        let malformed_rate_ppm = rate_ppm(malformed_rows);
        let candidate_completion_rows = rows.len();
        let health_gate_passed = generated_completion_rows > 0
            && verifier_rate_ppm >= config.rollout_imitation_min_verifier_rate_ppm
            && schema_wrong_rate_ppm <= config.rollout_imitation_max_schema_wrong_rate_ppm
            && malformed_rate_ppm <= config.rollout_imitation_max_malformed_rate_ppm;
        if !health_gate_passed {
            rows.retain(|row| row.kind == RolloutFeedbackKind::Recovery);
        }
        let accepted_imitation_rows = rows
            .iter()
            .filter(|row| row.kind == RolloutFeedbackKind::Imitation)
            .count();
        let accepted_recovery_rows = rows
            .iter()
            .filter(|row| row.kind == RolloutFeedbackKind::Recovery)
            .count();
        let skip_reason = if generated_completion_rows == 0 {
            Some("no_generated_completion".to_string())
        } else if candidate_completion_rows == 0 {
            Some("no_candidate_completion".to_string())
        } else if rows.is_empty() && !health_gate_passed {
            Some("rollout_health_gate".to_string())
        } else if rows.is_empty() {
            Some("no_accepted_completion".to_string())
        } else {
            None
        };

        let denominator = generated_completion_rows.max(1) as f64;
        self.write_ruliad_verifier_rollout_telemetry(RuliadVerifierRolloutImitationTelemetry {
            version: 1,
            step_index,
            skip_reason,
            sample_groups,
            generated_completion_rows,
            candidate_completion_rows,
            accepted_completion_rows: rows.len(),
            accepted_imitation_rows,
            accepted_recovery_rows,
            health_gate_passed,
            verifier_rate_ppm,
            schema_wrong_rate_ppm,
            malformed_rate_ppm,
            verifier_match_rows,
            semantic_match_rows,
            partial_rows,
            schema_wrong_rows,
            malformed_rows,
            missing_rows,
            recovery_partial_rows,
            recovery_schema_wrong_rows,
            recovery_malformed_rows,
            recovery_missing_rows,
            field_accuracy_mean: field_accuracy_sum / denominator,
            partial_progress_mean: partial_progress_sum / denominator,
            completion_quality_mean: completion_quality_sum / denominator,
            rollout_imitation_weight: imitation_weight,
            rollout_recovery_weight: recovery_weight,
            max_completion_tokens: completion_budget,
        });
        let replay_summary = self.ruliad_generated_attractor_summary();
        self.write_ruliad_generated_attractor_telemetry(RuliadGeneratedAttractorReplayTelemetry {
            version: 1,
            step_index,
            source: "rollout".to_string(),
            skip_reason: (generated_completion_rows == 0).then(|| "no_generated_rows".to_string()),
            observed_completion_rows: generated_completion_rows,
            recorded_attractor_rows,
            selected_candidate_rows: 0,
            selected_field_binding_pairs: 0,
            replay_pool_size: replay_summary.pool_size,
            active_attractor_count: replay_summary.active_count,
            active_observation_count: replay_summary.active_observation_count,
            distinct_answer_count: replay_summary.distinct_answers,
            dominant_answer_count: replay_summary.dominant_count,
            dominant_answer_fraction: replay_summary.dominant_fraction(),
            min_count: config.generated_attractor_replay_min_count.max(1),
            max_candidates: config.generated_attractor_replay_max_candidates,
            min_distinct_answers: config
                .generated_attractor_replay_min_distinct_answers
                .max(1),
            max_dominant_fraction: config.generated_attractor_replay_max_dominant_fraction,
        });

        if rows.is_empty() {
            return None;
        }
        let max_len = rows.iter().map(|row| row.inputs.len()).max()?.max(1);
        let row_count = rows.len();
        let mut input_values = vec![0i64; row_count * max_len];
        let mut target_values = vec![0i64; row_count * max_len];
        let mut active_mask_values = vec![0.0f32; row_count * max_len];
        let mut weighted_mask_values = vec![0.0f32; row_count * max_len];
        for (row_index, row) in rows.into_iter().enumerate() {
            let offset = row_index * max_len;
            let len = row.inputs.len().min(max_len);
            input_values[offset..offset + len].copy_from_slice(&row.inputs[..len]);
            target_values[offset..offset + len].copy_from_slice(&row.targets[..len]);
            for (mask_index, value) in row.mask.iter().copied().take(len).enumerate() {
                active_mask_values[offset + mask_index] = value;
                weighted_mask_values[offset + mask_index] = value * row.weight;
            }
        }
        let inputs = Tensor::<B, 2, Int>::from_data(
            TensorData::new(input_values, [row_count, max_len]),
            device,
        );
        let targets = Tensor::<B, 2, Int>::from_data(
            TensorData::new(target_values, [row_count, max_len]),
            device,
        );
        let active_mask = Tensor::<B, 2>::from_data(
            TensorData::new(active_mask_values, [row_count, max_len]),
            device,
        );
        let weighted_mask = Tensor::<B, 2>::from_data(
            TensorData::new(weighted_mask_values, [row_count, max_len]),
            device,
        );
        let logits = self.model.forward(inputs);
        let log_probs = log_probs_from_logits(logits);
        let token_log_probs = selected_token_log_probs(log_probs, targets);
        let active = active_mask.sum().reshape([1]).clamp_min(1.0);
        Some(
            (token_log_probs * weighted_mask)
                .sum()
                .reshape([1])
                .div(active)
                .mul_scalar(-1.0),
        )
    }

    pub(super) fn ruliad_proof_policy_dagger_loss(
        &self,
        policy_batch: &crate::dataset::RuliadPolicyBatch,
        device: &B::Device,
        block_size: usize,
    ) -> Option<Tensor<B, 1>>
    where
        B: AutodiffBackend,
    {
        self.ruliad_proof_policy_dagger_loss_at_step(
            policy_batch,
            device,
            block_size,
            self.gradient_scale_step.load(Ordering::Relaxed),
        )
    }

    pub(super) fn ruliad_proof_policy_dagger_loss_at_step(
        &self,
        policy_batch: &crate::dataset::RuliadPolicyBatch,
        device: &B::Device,
        block_size: usize,
        step_index: usize,
    ) -> Option<Tensor<B, 1>>
    where
        B: AutodiffBackend,
    {
        self.ruliad_proof_policy_objective_at_step(policy_batch, device, block_size, step_index)
            .map(|objective| objective.loss)
    }

    pub(super) fn ruliad_proof_policy_objective(
        &self,
        policy_batch: &crate::dataset::RuliadPolicyBatch,
        device: &B::Device,
        block_size: usize,
    ) -> Option<RuliadProofPolicyObjective<B>>
    where
        B: AutodiffBackend,
    {
        self.ruliad_proof_policy_objective_at_step(
            policy_batch,
            device,
            block_size,
            self.gradient_scale_step.load(Ordering::Relaxed),
        )
    }

    pub(super) fn ruliad_proof_policy_objective_at_step(
        &self,
        policy_batch: &crate::dataset::RuliadPolicyBatch,
        device: &B::Device,
        block_size: usize,
        step_index: usize,
    ) -> Option<RuliadProofPolicyObjective<B>>
    where
        B: AutodiffBackend,
    {
        let config = self.ruliad_supervision.proof_policy_for_step(step_index);
        let weight = self.ruliad_proof_policy_dagger_weight_at_step(step_index);
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
        let effective_mode = config.effective_mode(step_index);
        let semantic_row_budget = config.semantic_rows_per_update();
        let base_semantic_row_budget = config.base_semantic_rows_per_update();
        let batch_plan = RuliadProofPolicyBatchPlan::new(
            effective_mode,
            base_semantic_row_budget,
            config.rollout_steps,
            config.stratified_difficulty_levels,
        );
        let trajectory_budget = batch_plan.trajectory_budget();
        let sampling_model_started = Instant::now();
        let sampling_model = (batch_plan.dagger_trajectory_budget > 0).then(|| {
            self.model
                .valid()
                .materialize_random_scaffold_for_inference()
        });
        let sampling_model_materialize_ms =
            sampling_model_started.elapsed().as_micros() as f64 / 1_000.0;

        #[derive(Clone)]
        enum ExpertRowObjective {
            PresentationIndex {
                inputs: Vec<i64>,
                branch_position: usize,
                candidate_target_tokens: Vec<i64>,
                equivalent_target_tokens: Vec<i64>,
            },
            SemanticStep {
                prompt: Vec<i64>,
                candidate_completions: Vec<Vec<i64>>,
                equivalent_indices: Vec<usize>,
            },
        }

        #[derive(Clone)]
        struct ExpertRow {
            objective: ExpertRowObjective,
            presentation_weight: f32,
        }

        struct PrefixBranchRow {
            inputs: Vec<i64>,
            branch_position: usize,
            candidate_target_tokens: Vec<i64>,
            equivalent_target_tokens: Vec<i64>,
            weight: f32,
        }

        let mut rows = Vec::<ExpertRow>::new();
        let mut visited_prompts = HashSet::<Vec<i64>>::new();
        let mut available_sample_groups = 0usize;
        let mut sample_groups = 0usize;
        let mut nonzero_start_trajectories = 0usize;
        let mut start_step_sum = 0usize;
        let mut visited_states = 0usize;
        let mut semantic_state_rows = 0usize;
        let mut base_semantic_state_rows = 0usize;
        let mut counterfactual_semantic_state_rows = 0usize;
        let mut counterfactual_target_shortfall = 0usize;
        let mut static_expert_rows = 0usize;
        let mut dagger_expert_rows = 0usize;
        let mut model_visited_expert_rows = 0usize;
        let mut model_valid_actions = 0usize;
        let mut model_invalid_actions = 0usize;
        let mut model_expert_equivalent_actions = 0usize;
        let mut model_off_expert_actions = 0usize;
        let mut repeated_states = 0usize;
        let mut model_backtracks = 0usize;
        let mut model_scoring_batches = 0usize;
        let mut maximum_model_scoring_batch_rows = 0usize;
        let mut model_scoring_padded_tokens = 0usize;
        let mut rollout_cpu_prepare_ms = 0.0f64;
        let mut model_scoring_ms = 0.0f64;
        let mut difficulty_sample_groups = BTreeMap::<usize, usize>::new();
        let mut difficulty_visited_states = BTreeMap::<usize, usize>::new();
        let mut difficulty_expert_rows = BTreeMap::<usize, usize>::new();
        let mut expert_selected_index_histogram = BTreeMap::<usize, usize>::new();
        let mut expert_equivalent_index_histogram = BTreeMap::<usize, usize>::new();
        let mut model_selected_index_histogram = BTreeMap::<usize, usize>::new();
        let mut candidate_target_tokens = 0usize;
        let mut equivalent_target_tokens = 0usize;
        let mut supervised_action_tokens = 0usize;
        let mut rollout_depth_reached = 0usize;
        let mut presentation_budget_exhausted = false;

        struct DaggerTrajectory {
            sample_index: usize,
            difficulty_level: usize,
            is_dagger: bool,
            max_depth: usize,
            answer_contract: burn_dragon_universality::ruliad::RuliadProofActionAnswerContract,
            state: burn_dragon_universality::ruliad::RuliadProofPolicyState,
        }

        struct DaggerExpansion {
            trajectory_index: usize,
            actions: burn_dragon_universality::ruliad::RuliadProofActionSet,
            presentations: Vec<DaggerScoringPresentation>,
        }

        struct DaggerScoringPresentation {
            rotation: usize,
            prompt: Vec<i64>,
            candidate_completions: Vec<Vec<i64>>,
            answer_contract: burn_dragon_universality::ruliad::RuliadProofActionAnswerContract,
        }

        struct PreparedExpertState {
            canonical_prompt: Vec<i64>,
            presentation_rows: Vec<ExpertRow>,
            scoring_presentations: Vec<DaggerScoringPresentation>,
            presentation_selected_indices: Vec<usize>,
            presentation_equivalent_indices: Vec<Vec<usize>>,
        }

        let prepare_expert_state = |
            problem: &burn_dragon_universality::ruliad::RuliadProofProblem,
            actions: &burn_dragon_universality::ruliad::RuliadProofActionSet,
            presentation_index: usize,
            scoring_contract: burn_dragon_universality::ruliad::RuliadProofActionAnswerContract,
            base_rotations: Option<&[usize]>,
        | -> Option<PreparedExpertState> {
            let rotations = crate::train::ruliad_policy::target_group_presentation_rotations(
                config.candidate_symmetry,
                actions.selected_index,
                actions.candidates.len(),
                presentation_index,
                base_rotations,
            )
            .ok()?;
            let canonical_prompt = tokenizer
                .encode_payload(
                    &burn_dragon_universality::ruliad::ruliad_proof_action_prompt(
                        problem, actions,
                    )
                    .ok()?,
                )
                .into_iter()
                .map(i64::from)
                .collect::<Vec<_>>();
            let presentation_weight = 1.0 / rotations.len().max(1) as f32;
            let mut presentation_rows = Vec::<ExpertRow>::with_capacity(rotations.len());
            let mut scoring_presentations =
                Vec::<DaggerScoringPresentation>::with_capacity(rotations.len());
            let mut presentation_selected_indices = Vec::<usize>::with_capacity(rotations.len());
            let mut presentation_equivalent_indices =
                Vec::<Vec<usize>>::with_capacity(rotations.len());
            for rotation in rotations {
                let presented_actions = actions.rotate_left(rotation).ok()?;
                let prompt_text =
                    burn_dragon_universality::ruliad::ruliad_proof_action_prompt(
                        problem,
                        &presented_actions,
                    )
                    .ok()?;
                let candidate_completions = (0..presented_actions.candidates.len())
                    .map(|candidate_index| {
                        let answer = burn_dragon_universality::ruliad::proof_action_answer(
                            &presented_actions,
                            candidate_index,
                            scoring_contract,
                        )
                        .ok()?;
                        let mut completion = tokenizer
                            .encode_payload(&answer)
                            .into_iter()
                            .map(i64::from)
                            .collect::<Vec<_>>();
                        if scoring_contract
                            == burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep
                            && let Some(stop_token_id) = policy_batch.stop_token_id
                            && completion.last().copied() != Some(stop_token_id)
                        {
                            completion.push(stop_token_id);
                        }
                        Some(completion)
                    })
                    .collect::<Option<Vec<_>>>()?;
                if candidate_completions.iter().any(|completion| {
                    completion.is_empty() || completion.len() > completion_budget
                }) {
                    return None;
                }
                let expert_completion = candidate_completions
                    .get(presented_actions.selected_index)
                    .cloned()?;
                if presented_actions.equivalent_indices.is_empty()
                    || presented_actions
                        .equivalent_indices
                        .iter()
                        .any(|index| *index >= candidate_completions.len())
                {
                    return None;
                }
                let prompt = tokenizer
                    .encode_payload(&prompt_text)
                    .into_iter()
                    .map(i64::from)
                    .collect::<Vec<_>>();
                let prompt = Self::ruliad_trim_prompt_for_completion(
                    &prompt,
                    candidate_completions
                        .iter()
                        .map(Vec::len)
                        .max()
                        .unwrap_or(expert_completion.len()),
                    block_size,
                );
                if prompt.is_empty() {
                    return None;
                }
                let objective = match scoring_contract {
                    burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::PresentationIndex => {
                        let branch_token_index = crate::train::ruliad_policy::candidate_branch_index(
                            &candidate_completions,
                        )
                        .ok()?;
                        let equivalent_tokens = presented_actions
                            .equivalent_indices
                            .iter()
                            .filter_map(|candidate_index| candidate_completions.get(*candidate_index))
                            .filter_map(|completion| completion.get(branch_token_index).copied())
                            .collect::<std::collections::BTreeSet<_>>()
                            .into_iter()
                            .collect::<Vec<_>>();
                        let candidate_tokens = candidate_completions
                            .iter()
                            .filter_map(|completion| completion.get(branch_token_index).copied())
                            .collect::<std::collections::BTreeSet<_>>()
                            .into_iter()
                            .collect::<Vec<_>>();
                        if equivalent_tokens.is_empty()
                            || candidate_tokens.len() != candidate_completions.len()
                        {
                            return None;
                        }
                        let (inputs, targets, mask) =
                            Self::ruliad_policy_row_from_completion_token(
                                &prompt,
                                &expert_completion,
                                branch_token_index,
                            )?;
                        let branch_position = mask.iter().position(|value| *value > 0.0)?;
                        debug_assert_eq!(
                            targets[branch_position],
                            expert_completion[branch_token_index]
                        );
                        ExpertRowObjective::PresentationIndex {
                            inputs,
                            branch_position,
                            candidate_target_tokens: candidate_tokens,
                            equivalent_target_tokens: equivalent_tokens,
                        }
                    }
                    burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep => {
                        ExpertRowObjective::SemanticStep {
                            prompt: prompt.clone(),
                            candidate_completions: candidate_completions.clone(),
                            equivalent_indices: presented_actions.equivalent_indices.clone(),
                        }
                    }
                };
                presentation_selected_indices.push(presented_actions.selected_index);
                presentation_equivalent_indices
                    .push(presented_actions.equivalent_indices.clone());
                presentation_rows.push(ExpertRow {
                    objective,
                    presentation_weight,
                });
                scoring_presentations.push(DaggerScoringPresentation {
                    rotation,
                    prompt,
                    candidate_completions,
                    answer_contract: scoring_contract,
                });
            }
            (!presentation_rows.is_empty() && !scoring_presentations.is_empty()).then_some(
                PreparedExpertState {
                    canonical_prompt,
                    presentation_rows,
                    scoring_presentations,
                    presentation_selected_indices,
                    presentation_equivalent_indices,
                },
            )
        };

        let state_prepare_started = Instant::now();
        let mut trajectories = Vec::<DaggerTrajectory>::new();
        let mut answer_contract = None;
        for (sample_index, sample) in policy_batch.samples.iter().enumerate() {
            let Some(burn_dragon_universality::RuliadSampleSpec::FormalProof {
                problem,
                certificate,
                proof_step_index,
                action_answer_contract,
                task: burn_dragon_universality::RuliadTaskKind::SelectProofAction,
                ..
            }) = sample.item.spec.as_ref()
            else {
                continue;
            };
            available_sample_groups = available_sample_groups.saturating_add(1);
            if trajectories.len() >= trajectory_budget {
                continue;
            }
            let scoring_contract = match config.scoring {
                crate::config::RuliadProofPolicyScoring::CompletionLikelihood => {
                    *action_answer_contract
                }
                crate::config::RuliadProofPolicyScoring::SemanticEnergy
                | crate::config::RuliadProofPolicyScoring::ResidualEnergy => {
                    burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep
                }
            };
            if answer_contract.is_some_and(|contract| contract != scoring_contract) {
                continue;
            }
            answer_contract.get_or_insert(scoring_contract);
            let difficulty_level = sample.item.difficulty_level.unwrap_or(0);
            sample_groups = sample_groups.saturating_add(1);
            *difficulty_sample_groups
                .entry(difficulty_level)
                .or_default() += 1;
            let start_step = proof_step_index.unwrap_or_default();
            nonzero_start_trajectories =
                nonzero_start_trajectories.saturating_add(usize::from(start_step > 0));
            start_step_sum = start_step_sum.saturating_add(start_step);
            let Ok(state) =
                burn_dragon_universality::ruliad::RuliadProofPolicyState::from_certificate_prefix(
                    problem,
                    certificate,
                    start_step,
                )
            else {
                continue;
            };
            let trajectory_index = trajectories.len();
            let (is_dagger, max_depth) = if trajectory_index < batch_plan.static_row_budget {
                (false, 1)
            } else {
                let dagger_index = trajectory_index - batch_plan.static_row_budget;
                (true, batch_plan.dagger_depth(dagger_index))
            };
            trajectories.push(DaggerTrajectory {
                sample_index,
                difficulty_level,
                is_dagger,
                max_depth,
                answer_contract: scoring_contract,
                state,
            });
        }
        let state_prepare_ms = state_prepare_started.elapsed().as_micros() as f64 / 1_000.0;

        for rollout_depth in 0..batch_plan.rollout_steps {
            if presentation_budget_exhausted
                || base_semantic_state_rows >= base_semantic_row_budget
                || trajectories
                    .iter()
                    .all(|item| rollout_depth >= item.max_depth || item.state.solved())
            {
                break;
            }
            let wave_prepare_started = Instant::now();
            let states_before_wave = base_semantic_state_rows;
            let mut expansions = Vec::<DaggerExpansion>::new();
            for (trajectory_index, trajectory) in trajectories.iter_mut().enumerate() {
                if rollout_depth >= trajectory.max_depth
                    || trajectory.state.solved()
                    || base_semantic_state_rows >= base_semantic_row_budget
                {
                    continue;
                }
                let sample = &policy_batch.samples[trajectory.sample_index];
                let Some(burn_dragon_universality::RuliadSampleSpec::FormalProof {
                    problem, ..
                }) = sample.item.spec.as_ref()
                else {
                    continue;
                };
                let actions = match trajectory.state.action_set(problem, config.candidates) {
                    Ok(actions) => actions,
                    Err(_) if trajectory.state.backtrack() => {
                        model_backtracks = model_backtracks.saturating_add(1);
                        continue;
                    }
                    Err(_) => {
                        model_invalid_actions = model_invalid_actions.saturating_add(1);
                        continue;
                    }
                };
                let Some(mut original_state) = prepare_expert_state(
                    problem,
                    &actions,
                    semantic_state_rows,
                    trajectory.answer_contract,
                    None,
                ) else {
                    model_invalid_actions = model_invalid_actions.saturating_add(1);
                    continue;
                };
                visited_states = visited_states.saturating_add(1);
                *difficulty_visited_states
                    .entry(trajectory.difficulty_level)
                    .or_default() += 1;

                // Counterfactual targets are supervision only. The model rollout below still
                // scores and applies the original formal transition.
                let target_group_rotations = original_state
                    .scoring_presentations
                    .iter()
                    .map(|presentation| presentation.rotation)
                    .collect::<Vec<_>>();
                let scoring_presentations =
                    std::mem::take(&mut original_state.scoring_presentations);
                let mut prepared_states = vec![original_state];
                let counterfactual_indices =
                    crate::train::ruliad_policy::counterfactual_candidate_indices(
                        &actions,
                        config.counterfactual_targets_per_state,
                        actions
                            .selected_index
                            .saturating_add(base_semantic_state_rows)
                            .saturating_add(1),
                    );
                let mut group_shortfall = config
                    .counterfactual_targets_per_state
                    .saturating_sub(counterfactual_indices.len());
                for candidate_index in counterfactual_indices {
                    let Some((counterfactual_problem, counterfactual_actions)) =
                        burn_dragon_universality::ruliad::counterfactual_proof_action_target(
                            problem,
                            &actions,
                            candidate_index,
                        )
                        .ok()
                    else {
                        group_shortfall = group_shortfall.saturating_add(1);
                        continue;
                    };
                    let Some(counterfactual_state) = prepare_expert_state(
                        &counterfactual_problem,
                        &counterfactual_actions,
                        semantic_state_rows.saturating_add(prepared_states.len()),
                        trajectory.answer_contract,
                        Some(&target_group_rotations),
                    ) else {
                        group_shortfall = group_shortfall.saturating_add(1);
                        continue;
                    };
                    prepared_states.push(counterfactual_state);
                }
                counterfactual_target_shortfall =
                    counterfactual_target_shortfall.saturating_add(group_shortfall);
                let complete_target_group = group_shortfall == 0
                    && prepared_states.len() == config.target_variants_per_state();
                let presentation_rows = prepared_states
                    .iter()
                    .map(|state| state.presentation_rows.len())
                    .sum::<usize>();
                if complete_target_group
                    && rows.len().saturating_add(presentation_rows)
                        > config.max_presentation_rows_per_update
                {
                    presentation_budget_exhausted = true;
                    break;
                }
                let unique_target_group = complete_target_group
                    && prepared_states
                        .iter()
                        .all(|state| !visited_prompts.contains(&state.canonical_prompt));
                if unique_target_group {
                    let variants_added = prepared_states.len();
                    for state in prepared_states {
                        visited_prompts.insert(state.canonical_prompt);
                        for selected_index in state.presentation_selected_indices {
                            *expert_selected_index_histogram
                                .entry(selected_index)
                                .or_default() += 1;
                        }
                        for equivalent_indices in state.presentation_equivalent_indices {
                            for candidate_index in equivalent_indices {
                                *expert_equivalent_index_histogram
                                    .entry(candidate_index)
                                    .or_default() += 1;
                            }
                        }
                        for row in &state.presentation_rows {
                            match &row.objective {
                                ExpertRowObjective::PresentationIndex {
                                    candidate_target_tokens: candidate_tokens,
                                    equivalent_target_tokens: equivalent_tokens,
                                    ..
                                } => {
                                    supervised_action_tokens =
                                        supervised_action_tokens.saturating_add(1);
                                    candidate_target_tokens = candidate_target_tokens
                                        .saturating_add(candidate_tokens.len());
                                    equivalent_target_tokens = equivalent_target_tokens
                                        .saturating_add(equivalent_tokens.len());
                                }
                                ExpertRowObjective::SemanticStep {
                                    candidate_completions,
                                    equivalent_indices,
                                    ..
                                } => {
                                    let candidate_tokens =
                                        candidate_completions.iter().map(Vec::len).sum::<usize>();
                                    let equivalent_tokens = equivalent_indices
                                        .iter()
                                        .filter_map(|index| candidate_completions.get(*index))
                                        .map(Vec::len)
                                        .sum::<usize>();
                                    supervised_action_tokens =
                                        supervised_action_tokens.saturating_add(candidate_tokens);
                                    candidate_target_tokens =
                                        candidate_target_tokens.saturating_add(candidate_tokens);
                                    equivalent_target_tokens =
                                        equivalent_target_tokens.saturating_add(equivalent_tokens);
                                }
                            }
                        }
                        rows.extend(state.presentation_rows);
                    }
                    semantic_state_rows = semantic_state_rows.saturating_add(variants_added);
                    base_semantic_state_rows = base_semantic_state_rows.saturating_add(1);
                    counterfactual_semantic_state_rows = counterfactual_semantic_state_rows
                        .saturating_add(variants_added.saturating_sub(1));
                    static_expert_rows = static_expert_rows.saturating_add(
                        variants_added.saturating_mul(usize::from(!trajectory.is_dagger)),
                    );
                    dagger_expert_rows = dagger_expert_rows.saturating_add(
                        variants_added.saturating_mul(usize::from(trajectory.is_dagger)),
                    );
                    model_visited_expert_rows = model_visited_expert_rows
                        .saturating_add(variants_added.saturating_mul(usize::from(
                            trajectory.is_dagger && rollout_depth > 0,
                        )));
                    *difficulty_expert_rows
                        .entry(trajectory.difficulty_level)
                        .or_default() += variants_added;
                }
                if trajectory.is_dagger && rollout_depth.saturating_add(1) < trajectory.max_depth {
                    expansions.push(DaggerExpansion {
                        trajectory_index,
                        actions,
                        presentations: scoring_presentations,
                    });
                }
            }
            // The last supervised wave is already represented in `rows`. Scoring it cannot
            // produce another training row once the row budget is full, so avoid a synchronized
            // inference forward that only changes diagnostic terminal state.
            if presentation_budget_exhausted || base_semantic_state_rows >= base_semantic_row_budget
            {
                expansions.clear();
            }
            rollout_cpu_prepare_ms += wave_prepare_started.elapsed().as_micros() as f64 / 1_000.0;
            if base_semantic_state_rows > states_before_wave || !expansions.is_empty() {
                rollout_depth_reached = rollout_depth_reached.max(rollout_depth.saturating_add(1));
            }
            if expansions.is_empty() {
                break;
            }
            let scoring_presentations = expansions
                .iter()
                .enumerate()
                .flat_map(|(expansion_index, expansion)| {
                    expansion
                        .presentations
                        .iter()
                        .map(move |presentation| (expansion_index, presentation))
                })
                .collect::<Vec<_>>();
            let prompts = scoring_presentations
                .iter()
                .map(|(_, presentation)| presentation.prompt.clone())
                .collect::<Vec<_>>();
            let candidates = scoring_presentations
                .iter()
                .map(|(_, presentation)| presentation.candidate_completions.clone())
                .collect::<Vec<_>>();
            model_scoring_batches = model_scoring_batches.saturating_add(1);
            maximum_model_scoring_batch_rows =
                maximum_model_scoring_batch_rows.max(scoring_presentations.len());
            let Some(scoring_contract) = scoring_presentations
                .first()
                .map(|(_, presentation)| presentation.answer_contract)
            else {
                break;
            };
            if scoring_presentations
                .iter()
                .any(|(_, presentation)| presentation.answer_contract != scoring_contract)
            {
                model_invalid_actions = model_invalid_actions.saturating_add(expansions.len());
                break;
            }
            let scoring_max_len = scoring_presentations
                .iter()
                .filter_map(|(_, presentation)| match scoring_contract {
                    burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::PresentationIndex => {
                        crate::train::ruliad_policy::candidate_branch_index(
                            &presentation.candidate_completions,
                        )
                        .ok()
                        .map(|prefix_len| presentation.prompt.len().saturating_add(prefix_len))
                    }
                    burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep => {
                        presentation
                            .candidate_completions
                            .iter()
                            .map(Vec::len)
                            .max()
                            .map(|completion_len| {
                                presentation
                                    .prompt
                                    .len()
                                    .saturating_add(completion_len)
                                    .saturating_sub(1)
                            })
                    }
                })
                .max()
                .unwrap_or_default();
            model_scoring_padded_tokens = model_scoring_padded_tokens
                .saturating_add(scoring_max_len.saturating_mul(scoring_presentations.len()));
            let model_scoring_started = Instant::now();
            let Some(sampling_model) = sampling_model.as_ref() else {
                break;
            };
            let Ok(score_rows) =
                crate::train::ruliad_policy::proof_action_scores_batch_with_normalization(
                    sampling_model,
                    &prompts,
                    &candidates,
                    scoring_contract,
                    config.scoring,
                    config.normalization,
                    device,
                )
            else {
                model_invalid_actions = model_invalid_actions.saturating_add(expansions.len());
                break;
            };
            model_scoring_ms += model_scoring_started.elapsed().as_micros() as f64 / 1_000.0;
            let mut scores_by_expansion = (0..expansions.len())
                .map(|_| Vec::<(usize, Vec<f32>)>::new())
                .collect::<Vec<_>>();
            for ((expansion_index, presentation), scores) in
                scoring_presentations.iter().zip(score_rows)
            {
                scores_by_expansion[*expansion_index].push((presentation.rotation, scores));
            }
            drop(scoring_presentations);
            for (expansion, presentation_scores) in expansions.into_iter().zip(scores_by_expansion)
            {
                let Ok(scores) = crate::train::ruliad_policy::semantic_action_log_probs(
                    &presentation_scores,
                    expansion.actions.candidates.len(),
                ) else {
                    model_invalid_actions = model_invalid_actions.saturating_add(1);
                    continue;
                };
                let Some(candidate_index) =
                    crate::train::ruliad_policy::best_candidate_index(&scores)
                else {
                    model_invalid_actions = model_invalid_actions.saturating_add(1);
                    continue;
                };
                *model_selected_index_histogram
                    .entry(candidate_index)
                    .or_default() += 1;
                if expansion.actions.is_equivalent_index(candidate_index) {
                    model_expert_equivalent_actions =
                        model_expert_equivalent_actions.saturating_add(1);
                } else {
                    model_off_expert_actions = model_off_expert_actions.saturating_add(1);
                }
                match trajectories[expansion.trajectory_index]
                    .state
                    .apply(&expansion.actions, candidate_index)
                {
                    Ok(repeated) => {
                        model_valid_actions = model_valid_actions.saturating_add(1);
                        repeated_states = repeated_states.saturating_add(usize::from(repeated));
                    }
                    Err(_) => {
                        model_invalid_actions = model_invalid_actions.saturating_add(1);
                    }
                }
            }
        }
        let solved_proofs = trajectories
            .iter()
            .filter(|trajectory| trajectory.state.solved())
            .count();

        let mut prefix_branch_rows = Vec::<PrefixBranchRow>::new();
        if config.normalization == crate::config::RuliadProofPolicyNormalization::PrefixConditional
            && answer_contract
                == Some(
                    burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep,
                )
        {
            for row in &rows {
                let ExpertRowObjective::SemanticStep {
                    prompt,
                    candidate_completions,
                    equivalent_indices,
                } = &row.objective
                else {
                    return None;
                };
                let branches = crate::train::ruliad_policy::semantic_candidate_trie_branches(
                    candidate_completions,
                    equivalent_indices,
                )
                .ok()?;
                let branch_weight = row.presentation_weight / branches.len().max(1) as f32;
                for branch in branches {
                    let mut inputs = prompt.clone();
                    inputs.extend(branch.prefix);
                    let branch_position = inputs.len().checked_sub(1)?;
                    prefix_branch_rows.push(PrefixBranchRow {
                        inputs,
                        branch_position,
                        candidate_target_tokens: branch.candidate_tokens,
                        equivalent_target_tokens: branch.equivalent_tokens,
                        weight: branch_weight,
                    });
                }
            }
        }
        let prefix_candidate_tokens = prefix_branch_rows
            .iter()
            .map(|row| row.candidate_target_tokens.len())
            .sum::<usize>();
        let prefix_equivalent_tokens = prefix_branch_rows
            .iter()
            .map(|row| row.equivalent_target_tokens.len())
            .sum::<usize>();

        debug_assert!(rows.len() <= config.max_presentation_rows_per_update);
        self.write_ruliad_proof_policy_dagger_telemetry(RuliadProofPolicyDaggerTelemetry {
            version: RULIAD_PROOF_POLICY_TELEMETRY_VERSION,
            answer_contract: answer_contract.unwrap_or_default().label(),
            objective: match config.scoring {
                crate::config::RuliadProofPolicyScoring::SemanticEnergy => {
                    if config.counterfactual_targets_per_state > 0 {
                        "semantic_sequence_energy_counterfactual_v1"
                    } else {
                        "semantic_sequence_energy_v1"
                    }
                }
                crate::config::RuliadProofPolicyScoring::ResidualEnergy => {
                    if config.counterfactual_targets_per_state > 0 {
                        "autoregressive_residual_energy_counterfactual_v1"
                    } else {
                        "autoregressive_residual_energy_v1"
                    }
                }
                crate::config::RuliadProofPolicyScoring::CompletionLikelihood => {
                    match config.normalization {
                        crate::config::RuliadProofPolicyNormalization::CandidateConditional => {
                            if config.counterfactual_targets_per_state > 0 {
                                "candidate_normalized_counterfactual_v1"
                            } else {
                                "candidate_normalized_equivalent_v1"
                            }
                        }
                        crate::config::RuliadProofPolicyNormalization::PrefixConditional => {
                            if config.counterfactual_targets_per_state > 0 {
                                "prefix_conditional_counterfactual_v1"
                            } else {
                                "prefix_conditional_equivalent_v1"
                            }
                        }
                        crate::config::RuliadProofPolicyNormalization::VocabularyMarginal => {
                            "vocabulary_marginal_equivalent_v1"
                        }
                    }
                }
            },
            gradient_scope: match config.gradient_scope {
                crate::config::RuliadProofPolicyGradientScope::FullModel => "full_model",
                crate::config::RuliadProofPolicyGradientScope::ScoreHeadOnly => "score_head_only",
                crate::config::RuliadProofPolicyGradientScope::LanguageHeadOnly => {
                    "language_head_only"
                }
            },
            presentation_risk: match config.presentation_risk {
                crate::config::RuliadProofPolicyPresentationRisk::Mean => "mean",
                crate::config::RuliadProofPolicyPresentationRisk::Worst => "worst",
            },
            configured_mode: match config.mode {
                crate::config::RuliadProofPolicyTrainingMode::StaticExpert => "static_expert",
                crate::config::RuliadProofPolicyTrainingMode::Dagger => "dagger",
                crate::config::RuliadProofPolicyTrainingMode::StaticThenPairedDagger => {
                    "static_then_paired_dagger"
                }
            },
            mode: match effective_mode {
                crate::config::RuliadProofPolicyEffectiveMode::StaticExpert => "static_expert",
                crate::config::RuliadProofPolicyEffectiveMode::Dagger => "dagger",
                crate::config::RuliadProofPolicyEffectiveMode::PairedDagger => "paired_dagger",
            },
            candidate_symmetry: match config.candidate_symmetry {
                crate::config::RuliadProofPolicyCandidateSymmetry::Canonical => "canonical",
                crate::config::RuliadProofPolicyCandidateSymmetry::BalancedRotation => {
                    "balanced_rotation"
                }
                crate::config::RuliadProofPolicyCandidateSymmetry::CyclicOrbitAverage => {
                    "cyclic_orbit_average"
                }
            },
            step_index,
            skip_reason: rows
                .is_empty()
                .then(|| "no_formal_policy_states".to_string()),
            available_sample_groups,
            sample_groups,
            nonzero_start_trajectories,
            mean_start_step: start_step_sum as f64 / sample_groups.max(1) as f64,
            visited_states,
            semantic_state_rows,
            base_semantic_state_rows,
            counterfactual_semantic_state_rows,
            counterfactual_target_shortfall,
            expert_rows: semantic_state_rows,
            static_expert_rows,
            dagger_expert_rows,
            model_visited_expert_rows,
            supervised_action_tokens,
            supervised_presentation_rows: rows.len(),
            mean_presentations_per_state: rows.len() as f64 / semantic_state_rows.max(1) as f64,
            model_valid_actions,
            model_invalid_actions,
            model_expert_equivalent_actions,
            model_off_expert_actions,
            repeated_states,
            model_backtracks,
            solved_proofs,
            model_scoring_batches,
            maximum_model_scoring_batch_rows,
            model_scoring_padded_tokens,
            sampling_model_materialize_ms,
            state_prepare_ms,
            rollout_cpu_prepare_ms,
            model_scoring_ms,
            difficulty_sample_groups,
            difficulty_visited_states,
            difficulty_expert_rows,
            expert_selected_index_histogram,
            expert_equivalent_index_histogram,
            model_selected_index_histogram,
            candidate_target_tokens,
            equivalent_target_tokens,
            mean_candidate_targets_per_row: candidate_target_tokens as f64
                / rows.len().max(1) as f64,
            mean_equivalent_targets_per_row: equivalent_target_tokens as f64
                / rows.len().max(1) as f64,
            prefix_branch_rows: prefix_branch_rows.len(),
            prefix_candidate_tokens,
            prefix_equivalent_tokens,
            weight,
            rollout_steps: batch_plan.rollout_steps,
            rollout_depth_reached,
            configured_rollout_steps: config.rollout_steps,
            trajectory_budget,
            semantic_row_budget,
            base_semantic_row_budget,
            configured_counterfactual_targets_per_state: config.counterfactual_targets_per_state,
            target_variants_per_state: config.target_variants_per_state(),
            max_rows_per_update: config.max_rows_per_update,
            max_presentation_rows_per_update: config.max_presentation_rows_per_update,
        });
        if rows.is_empty() {
            return None;
        }
        if semantic_state_rows == 0 || !rows.len().is_multiple_of(semantic_state_rows) {
            return None;
        }
        let presentation_group_size = rows.len() / semantic_state_rows;
        let row_count = rows.len();
        let row_weights = Tensor::<B, 1>::from_data(
            TensorData::new(
                rows.iter()
                    .map(|row| row.presentation_weight)
                    .collect::<Vec<_>>(),
                [row_count],
            ),
            device,
        );
        match answer_contract? {
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::PresentationIndex => {
                let max_len = rows
                    .iter()
                    .filter_map(|row| match &row.objective {
                        ExpertRowObjective::PresentationIndex { inputs, .. } => Some(inputs.len()),
                        ExpertRowObjective::SemanticStep { .. } => None,
                    })
                    .max()?
                    .max(1);
                let mut input_values = vec![0i64; row_count * max_len];
                let mut branch_positions = Vec::with_capacity(row_count);
                for (row_index, row) in rows.iter().enumerate() {
                    let ExpertRowObjective::PresentationIndex {
                        inputs,
                        branch_position,
                        ..
                    } = &row.objective
                    else {
                        return None;
                    };
                    let offset = row_index * max_len;
                    let len = inputs.len().min(max_len);
                    input_values[offset..offset + len].copy_from_slice(&inputs[..len]);
                    branch_positions.push(*branch_position);
                }
                let inputs = Tensor::<B, 2, Int>::from_data(
                    TensorData::new(input_values, [row_count, max_len]),
                    device,
                );
                let branch_logits = crate::train::ruliad_policy::logits_at_sequence_positions(
                    &self.model,
                    inputs,
                    &branch_positions,
                    device,
                )
                .ok()?;
                let [_, vocab] = branch_logits.shape().dims::<2>();
                let branch_logits = branch_logits.reshape([row_count, 1, vocab]);
                let mut candidate_mask_values =
                    vec![0.0f32; row_count.saturating_mul(vocab)];
                let mut equivalent_mask_values =
                    vec![0.0f32; row_count.saturating_mul(vocab)];
                for (row_index, row) in rows.iter().enumerate() {
                    let ExpertRowObjective::PresentationIndex {
                        candidate_target_tokens,
                        equivalent_target_tokens,
                        ..
                    } = &row.objective
                    else {
                        return None;
                    };
                    for token in candidate_target_tokens {
                        let token = usize::try_from(*token).ok()?;
                        if token >= vocab {
                            return None;
                        }
                        candidate_mask_values[row_index * vocab + token] = 1.0;
                    }
                    for token in equivalent_target_tokens {
                        let token = usize::try_from(*token).ok()?;
                        if token >= vocab {
                            return None;
                        }
                        equivalent_mask_values[row_index * vocab + token] = 1.0;
                    }
                }
                let candidate_mask = Tensor::<B, 3>::from_data(
                    TensorData::new(candidate_mask_values, [row_count, 1, vocab]),
                    device,
                );
                let equivalent_mask = Tensor::<B, 3>::from_data(
                    TensorData::new(equivalent_mask_values, [row_count, 1, vocab]),
                    device,
                );
                Some(RuliadProofPolicyObjective {
                    loss: grouped_verifier_equivalent_action_loss(
                        branch_logits,
                        candidate_mask,
                        equivalent_mask,
                        row_weights,
                        config.normalization,
                        config.presentation_risk,
                        presentation_group_size,
                        weight,
                    ),
                    semantic_states: semantic_state_rows,
                    decision_rows: row_count,
                    padded_tokens: row_count.saturating_mul(max_len),
                })
            }
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep => {
                if config.normalization
                    == crate::config::RuliadProofPolicyNormalization::PrefixConditional
                {
                    let branch_row_count = prefix_branch_rows.len();
                    if branch_row_count == 0 {
                        return None;
                    }
                    let max_len = prefix_branch_rows
                        .iter()
                        .map(|row| row.inputs.len())
                        .max()?
                        .max(1);
                    let mut input_values = vec![0i64; branch_row_count * max_len];
                    let mut branch_positions = Vec::with_capacity(branch_row_count);
                    let branch_weights = prefix_branch_rows
                        .iter()
                        .map(|row| row.weight)
                        .collect::<Vec<_>>();
                    for (row_index, row) in prefix_branch_rows.iter().enumerate() {
                        let offset = row_index * max_len;
                        input_values[offset..offset + row.inputs.len()]
                            .copy_from_slice(&row.inputs);
                        branch_positions.push(row.branch_position);
                    }
                    let inputs = Tensor::<B, 2, Int>::from_data(
                        TensorData::new(input_values, [branch_row_count, max_len]),
                        device,
                    );
                    let branch_logits =
                        crate::train::ruliad_policy::logits_at_sequence_positions(
                            &self.model,
                            inputs,
                            &branch_positions,
                            device,
                        )
                        .ok()?;
                    let [_, vocab] = branch_logits.shape().dims::<2>();
                    let mut candidate_mask_values =
                        vec![0.0f32; branch_row_count.saturating_mul(vocab)];
                    let mut equivalent_mask_values =
                        vec![0.0f32; branch_row_count.saturating_mul(vocab)];
                    for (row_index, row) in prefix_branch_rows.iter().enumerate() {
                        for token in &row.candidate_target_tokens {
                            let token = usize::try_from(*token).ok()?;
                            if token >= vocab {
                                return None;
                            }
                            candidate_mask_values[row_index * vocab + token] = 1.0;
                        }
                        for token in &row.equivalent_target_tokens {
                            let token = usize::try_from(*token).ok()?;
                            if token >= vocab {
                                return None;
                            }
                            equivalent_mask_values[row_index * vocab + token] = 1.0;
                        }
                    }
                    let candidate_mask = Tensor::<B, 3>::from_data(
                        TensorData::new(candidate_mask_values, [branch_row_count, 1, vocab]),
                        device,
                    );
                    let equivalent_mask = Tensor::<B, 3>::from_data(
                        TensorData::new(equivalent_mask_values, [branch_row_count, 1, vocab]),
                        device,
                    );
                    let row_weights = Tensor::<B, 1>::from_data(
                        TensorData::new(branch_weights, [branch_row_count]),
                        device,
                    );
                    return Some(RuliadProofPolicyObjective {
                        loss: grouped_verifier_equivalent_action_loss(
                            branch_logits.reshape([branch_row_count, 1, vocab]),
                            candidate_mask,
                            equivalent_mask,
                            row_weights,
                            crate::config::RuliadProofPolicyNormalization::CandidateConditional,
                            crate::config::RuliadProofPolicyPresentationRisk::Mean,
                            1,
                            weight,
                        ),
                        semantic_states: semantic_state_rows,
                        decision_rows: branch_row_count,
                        padded_tokens: branch_row_count.saturating_mul(max_len),
                    });
                }
                let mut prompts = Vec::with_capacity(row_count);
                let mut candidates = Vec::with_capacity(row_count);
                let mut equivalent_indices = Vec::with_capacity(row_count);
                for row in &rows {
                    let ExpertRowObjective::SemanticStep {
                        prompt,
                        candidate_completions,
                        equivalent_indices: row_equivalent_indices,
                    } = &row.objective
                    else {
                        return None;
                    };
                    prompts.push(prompt.clone());
                    candidates.push(candidate_completions.clone());
                    equivalent_indices.push(row_equivalent_indices.clone());
                }
                let candidate_count = candidates.first()?.len();
                if candidate_count < 2
                    || candidates.iter().any(|group| group.len() != candidate_count)
                {
                    return None;
                }
                let (mean_log_scores, sum_log_scores, group_sizes) = match config.scoring {
                    crate::config::RuliadProofPolicyScoring::CompletionLikelihood => {
                        let scores =
                            crate::train::ruliad_policy::sequence_completion_score_tensor_with_gradient_scope(
                                &self.model,
                                &prompts,
                                &candidates,
                                config.gradient_scope,
                                device,
                            )
                            .ok()?;
                        (
                            scores.mean_log_scores,
                            scores.sum_log_scores,
                            scores.group_sizes,
                        )
                    }
                    crate::config::RuliadProofPolicyScoring::SemanticEnergy => {
                        let (scores, group_sizes) =
                            crate::train::ruliad_policy::sequence_energy_score_tensor_with_gradient_scope(
                                &self.model,
                                &prompts,
                                &candidates,
                                config.gradient_scope,
                                device,
                            )
                            .ok()?;
                        (scores.clone(), scores, group_sizes)
                    }
                    crate::config::RuliadProofPolicyScoring::ResidualEnergy => {
                        let (scores, group_sizes) =
                            crate::train::ruliad_policy::sequence_residual_energy_score_tensor_with_gradient_scope(
                                &self.model,
                                &prompts,
                                &candidates,
                                config.gradient_scope,
                                device,
                            )
                            .ok()?;
                        (scores.clone(), scores, group_sizes)
                    }
                };
                if group_sizes
                    .iter()
                    .any(|group_size| *group_size != candidate_count)
                {
                    return None;
                }
                let mut equivalent_mask_values =
                    vec![0.0f32; row_count.saturating_mul(candidate_count)];
                for (row_index, indices) in equivalent_indices.iter().enumerate() {
                    for index in indices {
                        if *index >= candidate_count {
                            return None;
                        }
                        equivalent_mask_values[row_index * candidate_count + *index] = 1.0;
                    }
                }
                let equivalent_mask = Tensor::<B, 2>::from_data(
                    TensorData::new(equivalent_mask_values, [row_count, candidate_count]),
                    device,
                );
                let padded_tokens = prompts
                    .iter()
                    .zip(candidates.iter())
                    .map(|(prompt, completions)| {
                        let max_completion = completions.iter().map(Vec::len).max().unwrap_or(0);
                        prompt
                            .len()
                            .saturating_add(max_completion)
                            .saturating_mul(completions.len())
                    })
                    .sum();
                Some(RuliadProofPolicyObjective {
                    loss: grouped_verifier_equivalent_sequence_loss(
                        mean_log_scores.reshape([row_count, candidate_count]),
                        sum_log_scores.reshape([row_count, candidate_count]),
                        equivalent_mask,
                        row_weights,
                        GroupedVerifierSequenceLossConfig {
                            normalization: config.normalization,
                            presentation_risk: config.presentation_risk,
                            presentation_group_size,
                            weight,
                        },
                    ),
                    semantic_states: semantic_state_rows,
                    decision_rows: row_count,
                    padded_tokens,
                })
            }
        }
    }

    pub(super) fn ruliad_verifier_policy_loss(
        &self,
        policy_batch: &crate::dataset::RuliadPolicyBatch,
        device: &B::Device,
        block_size: usize,
    ) -> Option<Tensor<B, 1>>
    where
        B: AutodiffBackend,
    {
        let config = self.ruliad_supervision.verifier_reward;
        let weight = self.ruliad_verifier_reward_weight();
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
        let prompt_budget = block_size.saturating_sub(completion_budget).max(1);
        let group_size = config.group_size.max(2);

        #[derive(Clone)]
        struct PolicyRow {
            inputs: Vec<i64>,
            targets: Vec<i64>,
            mask: Vec<f32>,
            advantage: f32,
        }

        let mut rows = Vec::new();
        let mut telemetry = RuliadPolicyRewardTelemetryAccumulator::new(
            config.mode,
            self.gradient_scale_step.load(Ordering::Relaxed),
        );
        let mut observed_generated_rows = 0usize;
        let mut recorded_attractor_rows = 0usize;
        let sampling_model = self
            .model
            .valid()
            .materialize_random_scaffold_for_inference();
        for sample in policy_batch.samples.iter() {
            let mut prompt = sample.prompt_tokens.clone();
            if prompt.is_empty() {
                continue;
            }
            if prompt.len() > prompt_budget {
                prompt = prompt[prompt.len() - prompt_budget..].to_vec();
            }
            let configured_structured_negatives = if config.include_structured_negative_candidates {
                config
                    .structured_negative_count
                    .saturating_add(config.structured_template_negative_count)
                    .saturating_add(config.structured_schema_negative_count)
            } else {
                0
            };
            let generated_attractor_candidates =
                self.ruliad_generated_attractor_candidates_for_sample(sample);
            let mut group_rows = Vec::with_capacity(
                group_size
                    + usize::from(config.include_oracle_candidate)
                    + configured_structured_negatives
                    + generated_attractor_candidates.len(),
            );
            let mut scores = Vec::with_capacity(
                group_size
                    + usize::from(config.include_oracle_candidate)
                    + configured_structured_negatives
                    + generated_attractor_candidates.len(),
            );
            if config.include_oracle_candidate
                && let Some((oracle_completion, oracle_completion_text, oracle_truncated)) =
                    Self::ruliad_oracle_completion_tokens(&tokenizer, sample, completion_budget)
                && let Some(row) =
                    Self::ruliad_policy_row_from_completion(&prompt, &oracle_completion)
            {
                let score = burn_dragon_universality::ruliad::score_ruliad_item_completion(
                    &sample.item,
                    Some(&oracle_completion_text),
                );
                telemetry.record_oracle_candidate(oracle_truncated);
                scores.push(score);
                group_rows.push(row);
            }
            if config.include_structured_negative_candidates {
                for (negative, _negative_kind) in
                    Self::ruliad_structured_negative_answers_with_schema(
                        &sample.item.expected_answer,
                        config.structured_negative_count,
                        config.structured_template_negative_count,
                        config.structured_schema_negative_count,
                    )
                {
                    let Some((completion, completion_text)) =
                        Self::ruliad_completion_tokens_from_answer(
                            &tokenizer,
                            &negative,
                            sample.item.document_close_marker(),
                            completion_budget,
                        )
                    else {
                        continue;
                    };
                    let Some(row) = Self::ruliad_policy_row_from_completion(&prompt, &completion)
                    else {
                        continue;
                    };
                    let score = burn_dragon_universality::ruliad::score_ruliad_item_completion(
                        &sample.item,
                        Some(&completion_text),
                    );
                    telemetry.record_structured_negative_candidate();
                    scores.push(score);
                    group_rows.push(row);
                }
            }
            for entry in generated_attractor_candidates {
                let Some((completion, completion_text)) =
                    Self::ruliad_completion_tokens_from_answer(
                        &tokenizer,
                        &entry.key.answer,
                        sample.item.document_close_marker(),
                        completion_budget,
                    )
                else {
                    continue;
                };
                let Some(row) = Self::ruliad_policy_row_from_completion(&prompt, &completion)
                else {
                    continue;
                };
                let score = burn_dragon_universality::ruliad::score_ruliad_item_completion(
                    &sample.item,
                    Some(&completion_text),
                );
                telemetry.record_generated_attractor_candidate();
                scores.push(score);
                group_rows.push(row);
            }
            for _ in 0..group_size {
                let generated = crate::generation::generate_tokens(
                    &sampling_model,
                    prompt.clone(),
                    device,
                    crate::generation::GenerationSettings {
                        max_new_tokens: Some(completion_budget),
                        temperature: config.temperature,
                        top_k: Some(config.top_k),
                        strategy: crate::generation::ContextStrategy::Infinite,
                        stop_on_token: policy_batch.stop_token_id,
                    },
                    None,
                )
                .ok()?;
                if generated.len() <= prompt.len() {
                    continue;
                }
                let completion = generated[prompt.len()..].to_vec();
                if completion.is_empty() {
                    continue;
                }
                let completion_tokens = completion
                    .iter()
                    .filter_map(|token| u32::try_from(*token).ok())
                    .collect::<Vec<_>>();
                let completion_text = tokenizer.decode_payload(&completion_tokens, true);
                let score = burn_dragon_universality::ruliad::score_ruliad_item_completion(
                    &sample.item,
                    Some(&completion_text),
                );
                observed_generated_rows = observed_generated_rows.saturating_add(1);
                recorded_attractor_rows = recorded_attractor_rows.saturating_add(usize::from(
                    self.record_ruliad_generated_attractor(
                        sample,
                        &completion_text,
                        &score,
                        telemetry.step_index,
                    ),
                ));
                scores.push(score);
                if let Some(row) = Self::ruliad_policy_row_from_completion(&prompt, &completion) {
                    group_rows.push(row);
                }
            }
            if group_rows.is_empty() || scores.len() != group_rows.len() {
                continue;
            }
            telemetry.record_vectors(&scores);
            let rewards = match config.mode {
                crate::config::train::RuliadVerifierRewardMode::Scalar => scores
                    .iter()
                    .map(|score| {
                        burn_dragon_universality::ruliad::ruliad_verifier_reward(
                            score,
                            config.reward,
                        )
                    })
                    .collect::<Vec<_>>(),
                crate::config::train::RuliadVerifierRewardMode::VpoIndependent => {
                    let scalarizations = self.ruliad_vpo_scalarizations(
                        sample.item.sample_index,
                        config.vpo_scalarizations.max(1),
                        config,
                    );
                    self.ruliad_vpo_independent_utilities_with_telemetry(
                        &scores,
                        &scalarizations,
                        &mut telemetry,
                    )
                }
            };
            let mut advantages = burn_dragon_universality::ruliad::normalized_advantages(
                &rewards,
                config.advantage_epsilon,
            );
            if !Self::constrain_ruliad_policy_advantages(&scores, &mut advantages, config) {
                telemetry.record_gated_group(rewards.len());
                continue;
            }
            telemetry.record_rewards_and_advantages(&rewards, &advantages, config.clip_range);
            rows.extend(group_rows.into_iter().zip(advantages).map(
                |((inputs, targets, mask), advantage)| PolicyRow {
                    inputs,
                    targets,
                    mask,
                    advantage: advantage.clamp(-config.clip_range, config.clip_range),
                },
            ));
        }
        let replay_summary = self.ruliad_generated_attractor_summary();
        self.write_ruliad_generated_attractor_telemetry(RuliadGeneratedAttractorReplayTelemetry {
            version: 1,
            step_index: telemetry.step_index,
            source: "policy".to_string(),
            skip_reason: (observed_generated_rows == 0)
                .then(|| "no_generated_rows".to_string())
                .or_else(|| {
                    self.ruliad_generated_attractor_replay_skip_reason(
                        &replay_summary,
                        telemetry.generated_attractor_completion_rows,
                    )
                }),
            observed_completion_rows: observed_generated_rows,
            recorded_attractor_rows,
            selected_candidate_rows: telemetry.generated_attractor_completion_rows,
            selected_field_binding_pairs: 0,
            replay_pool_size: replay_summary.pool_size,
            active_attractor_count: replay_summary.active_count,
            active_observation_count: replay_summary.active_observation_count,
            distinct_answer_count: replay_summary.distinct_answers,
            dominant_answer_count: replay_summary.dominant_count,
            dominant_answer_fraction: replay_summary.dominant_fraction(),
            min_count: config.generated_attractor_replay_min_count.max(1),
            max_candidates: config.generated_attractor_replay_max_candidates,
            min_distinct_answers: config
                .generated_attractor_replay_min_distinct_answers
                .max(1),
            max_dominant_fraction: config.generated_attractor_replay_max_dominant_fraction,
        });
        if rows.is_empty() {
            if telemetry.has_observations() {
                telemetry.mark_skipped("positive_advantage_gate");
                if let Some(telemetry) = telemetry.finish() {
                    self.write_ruliad_policy_telemetry(telemetry);
                }
            }
            return None;
        }
        if let Some(max_clip_fraction) = config.max_advantage_clip_fraction {
            let clip_fraction = telemetry.advantage_clip_fraction();
            if clip_fraction > f64::from(max_clip_fraction) {
                telemetry.mark_skipped(format!("advantage_clip_fraction>{max_clip_fraction:.6}"));
                if let Some(telemetry) = telemetry.finish() {
                    self.write_ruliad_policy_telemetry(telemetry);
                }
                return None;
            }
        }
        if let Some(telemetry) = telemetry.finish() {
            self.write_ruliad_policy_telemetry(telemetry);
        }
        let max_len = rows.iter().map(|row| row.inputs.len()).max()?.max(1);
        let row_count = rows.len();
        let mut input_values = vec![0i64; row_count * max_len];
        let mut target_values = vec![0i64; row_count * max_len];
        let mut mask_values = vec![0.0f32; row_count * max_len];
        let mut advantage_values = vec![0.0f32; row_count * max_len];
        for (row_index, row) in rows.into_iter().enumerate() {
            let offset = row_index * max_len;
            let len = row.inputs.len().min(max_len);
            input_values[offset..offset + len].copy_from_slice(&row.inputs[..len]);
            target_values[offset..offset + len].copy_from_slice(&row.targets[..len]);
            mask_values[offset..offset + len].copy_from_slice(&row.mask[..len]);
            for value in advantage_values[offset..offset + len].iter_mut() {
                *value = row.advantage;
            }
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
        let advantages = Tensor::<B, 2>::from_data(
            TensorData::new(advantage_values, [row_count, max_len]),
            device,
        );
        let logits = self.model.forward(inputs.clone());
        let log_probs = log_probs_from_logits(logits);
        let token_log_probs = selected_token_log_probs(log_probs.clone(), targets);
        let active = mask.clone().sum().reshape([1]).clamp_min(1.0);
        let mut loss = (token_log_probs * advantages * mask.clone())
            .sum()
            .reshape([1])
            .div(active)
            .mul_scalar(-weight);
        if config.kl_weight > f32::EPSILON && self.teacher_model.is_some() {
            let teacher_log_probs =
                log_probs_from_logits(self.current_teacher_model().forward(inputs).detach());
            let [rows, time, _vocab] = log_probs.shape().dims();
            let per_token_kl = (log_probs.clone().exp() * (log_probs - teacher_log_probs))
                .sum_dim(2)
                .reshape([rows, time]);
            let active = mask.clone().sum().reshape([1]).clamp_min(1.0);
            let kl_loss = (per_token_kl * mask)
                .sum()
                .reshape([1])
                .div(active)
                .mul_scalar(config.kl_weight);
            loss = loss + kl_loss;
        }
        Some(loss)
    }
}
