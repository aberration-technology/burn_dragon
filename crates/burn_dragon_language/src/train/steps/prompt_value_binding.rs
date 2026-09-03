//! Prompt-conditioned value binding as a scheduled primary training objective.

use super::*;
use crate::train::local_predictive_coding;

pub(super) struct RuliadPromptValueBindingStepInput<B: Backend> {
    pub policy_batch: Option<Arc<crate::dataset::RuliadPolicyBatch>>,
    pub stream_inputs: Tensor<B, 2, Int>,
    pub summary_event_mask: Option<Tensor<B, 2, Int>>,
    pub reset_stream_state: bool,
    pub block_size: usize,
    pub schedule_step_index: usize,
    pub profiling: bool,
}

impl<B: AutodiffBackend> LanguageTrainModel<B> {
    pub(super) fn ruliad_prompt_value_binding_step(
        &self,
        input: RuliadPromptValueBindingStepInput<B>,
    ) -> Option<TrainOutput<LanguageModelTrainItem<B>>> {
        let RuliadPromptValueBindingStepInput {
            policy_batch,
            stream_inputs,
            summary_event_mask,
            reset_stream_state,
            block_size,
            schedule_step_index,
            profiling,
        } = input;
        let prepared = policy_batch.as_deref().and_then(|policy_batch| {
            self.prepare_ruliad_prompt_value_binding_batch(
                policy_batch,
                &stream_inputs.device(),
                block_size,
            )
        });
        let Some(prepared) = prepared else {
            self.write_ruliad_prompt_value_binding_telemetry(RuliadPromptValueBindingTelemetry {
                version: 1,
                step_index: schedule_step_index,
                algorithm: self.ruliad_prompt_value_binding_algorithm(),
                skip_reason: Some("missing_or_empty_policy_batch"),
                sample_groups: 0,
                rows: 0,
                active_tokens: 0,
                padded_tokens: 0,
                global_backward_calls: 0,
            });
            return None;
        };
        let PreparedRuliadPromptValueBindingBatch {
            inputs,
            targets,
            loss_mask,
            sample_groups,
            rows,
            active_tokens,
            padded_tokens,
        } = prepared;
        let [structured_batch_size, structured_sequence_len] = inputs.shape().dims::<2>();

        if matches!(self.training_algorithm, TrainingAlgorithm::PredictiveCoding) {
            let step = local_predictive_coding::local_predictive_coding_train_step(
                &self.model,
                inputs,
                targets,
                Some(loss_mask),
                &self.local_predictive_coding,
                &self.local_predictive_coding_profile,
            );
            debug_assert_eq!(step.report.global_backward_calls, 0);
            self.write_ruliad_prompt_value_binding_telemetry(RuliadPromptValueBindingTelemetry {
                version: 1,
                step_index: schedule_step_index,
                algorithm: "predictive_coding",
                skip_reason: None,
                sample_groups,
                rows,
                active_tokens,
                padded_tokens,
                global_backward_calls: step.report.global_backward_calls,
            });
            if profiling {
                crate::train::profile::record_local_learning_step(step.report.elapsed_ns);
                crate::train::profile::record_structured_terminal(
                    rows,
                    structured_batch_size.saturating_mul(structured_sequence_len),
                );
            }
            self.advance_prompt_value_binding_stream(
                stream_inputs,
                summary_event_mask,
                reset_stream_state,
                profiling,
            );
            return Some(TrainOutput {
                grads: self.apply_gradient_scale_schedule(step.grads),
                item: LanguageModelTrainItem::new(step.loss),
            });
        }

        let forward_started = profiling.then(Instant::now);
        let logits = self.model.forward(inputs);
        let loss = self.language_loss_from_logits(logits, targets, Some(loss_mask));
        let forward_ns = forward_started
            .map(|started| started.elapsed().as_nanos())
            .unwrap_or_default();
        let backward_started = profiling.then(Instant::now);
        let grads = loss.backward();
        let backward_ns = backward_started
            .map(|started| started.elapsed().as_nanos())
            .unwrap_or_default();
        self.write_ruliad_prompt_value_binding_telemetry(RuliadPromptValueBindingTelemetry {
            version: 1,
            step_index: schedule_step_index,
            algorithm: "backpropagation",
            skip_reason: None,
            sample_groups,
            rows,
            active_tokens,
            padded_tokens,
            global_backward_calls: 1,
        });
        self.local_predictive_coding_profile
            .record_global_structured_terminal(
                sample_groups,
                rows,
                forward_ns.saturating_add(backward_ns),
            );
        if profiling {
            crate::train::profile::record_train_step(forward_ns, backward_ns);
            crate::train::profile::record_structured_terminal(
                rows,
                structured_batch_size.saturating_mul(structured_sequence_len),
            );
        }
        self.advance_prompt_value_binding_stream(
            stream_inputs,
            summary_event_mask,
            reset_stream_state,
            profiling,
        );
        Some(TrainOutput {
            grads: self.apply_gradient_scale_schedule(GradientsParams::from_grads(grads, self)),
            item: LanguageModelTrainItem::new(loss),
        })
    }

    fn advance_prompt_value_binding_stream(
        &self,
        inputs: Tensor<B, 2, Int>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
        reset_stream_state: bool,
        profiling: bool,
    ) {
        let started = profiling.then(Instant::now);
        self.advance_stream_state_without_update(inputs, summary_event_mask, reset_stream_state);
        if let Some(started) = started {
            crate::train::profile::record_stream_advance(started.elapsed().as_nanos());
        }
    }

    fn ruliad_prompt_value_binding_algorithm(&self) -> &'static str {
        if matches!(self.training_algorithm, TrainingAlgorithm::PredictiveCoding) {
            "predictive_coding"
        } else {
            "backpropagation"
        }
    }
}

impl<B: BackendTrait> LanguageTrainModel<B> {
    pub(super) fn prepare_ruliad_prompt_value_binding_batch(
        &self,
        policy_batch: &crate::dataset::RuliadPolicyBatch,
        device: &B::Device,
        block_size: usize,
    ) -> Option<PreparedRuliadPromptValueBindingBatch<B>> {
        let config = self.ruliad_supervision.prompt_value_binding;
        if !config.enabled || policy_batch.samples.is_empty() || block_size < 4 {
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
            .min(block_size.saturating_sub(2).max(1));
        let row_groups = policy_batch
            .samples
            .iter()
            .filter_map(|sample| {
                if sample.prompt_tokens.is_empty() || sample.item.expected_answer.trim().is_empty()
                {
                    return None;
                }
                let rows = Self::ruliad_prompt_schema_value_completion_rows(
                    &tokenizer,
                    &sample.prompt_tokens,
                    &sample.item.expected_answer,
                    sample.item.document_close_marker(),
                    completion_budget,
                    block_size,
                    config.max_rows_per_step,
                );
                (!rows.is_empty()).then_some(rows)
            })
            .collect::<Vec<_>>();
        let selected = take_rows_round_robin(&row_groups, config.max_rows_per_step);
        if selected.is_empty() {
            return None;
        }
        let sample_groups = selected
            .iter()
            .map(|(sample_index, _)| *sample_index)
            .collect::<HashSet<_>>()
            .len();
        let rows = selected.len();
        let max_len = selected
            .iter()
            .map(|(_, (inputs, _, _, _))| inputs.len())
            .max()?
            .max(1);
        let mut input_values = vec![0_i64; rows * max_len];
        let mut target_values = vec![0_i64; rows * max_len];
        let mut mask_values = vec![0_i64; rows * max_len];
        let mut active_tokens = 0usize;
        let mut populated_tokens = 0usize;
        for (row_index, (_, (inputs, targets, mask, _))) in selected.into_iter().enumerate() {
            let len = inputs.len().min(max_len);
            let offset = row_index * max_len;
            input_values[offset..offset + len].copy_from_slice(&inputs[..len]);
            target_values[offset..offset + len].copy_from_slice(&targets[..len]);
            for (column, weight) in mask.into_iter().take(len).enumerate() {
                if weight > f32::EPSILON {
                    mask_values[offset + column] = 1;
                    active_tokens = active_tokens.saturating_add(1);
                }
            }
            populated_tokens = populated_tokens.saturating_add(len);
        }
        if active_tokens == 0 {
            return None;
        }
        Some(PreparedRuliadPromptValueBindingBatch {
            inputs: Tensor::from_data(TensorData::new(input_values, [rows, max_len]), device),
            targets: Tensor::from_data(TensorData::new(target_values, [rows, max_len]), device),
            loss_mask: Tensor::from_data(TensorData::new(mask_values, [rows, max_len]), device),
            sample_groups,
            rows,
            active_tokens,
            padded_tokens: rows
                .saturating_mul(max_len)
                .saturating_sub(populated_tokens),
        })
    }

    pub(super) fn write_ruliad_prompt_value_binding_telemetry(
        &self,
        telemetry: RuliadPromptValueBindingTelemetry,
    ) {
        let Some(path) = self.ruliad_prompt_value_binding_telemetry_path.as_ref() else {
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
}
