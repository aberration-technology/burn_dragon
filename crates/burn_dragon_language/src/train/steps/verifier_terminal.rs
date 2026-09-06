//! Joint language-model and formally verified action-set terminals.

use super::*;
use crate::train::local_predictive_coding;

impl<B: BackendTrait> LanguageTrainModel<B> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn joint_backprop_verifier_terminal_step(
        &self,
        policy_batch: Option<&crate::dataset::RuliadPolicyBatch>,
        clean_inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
        loss_mask: Option<Tensor<B, 2, Int>>,
        supervised_token_count: Option<usize>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
        reset_stream_state: bool,
        block_size: usize,
        schedule_step_index: usize,
        profiling: bool,
    ) -> Option<TrainOutput<LanguageModelTrainItem<B>>>
    where
        B: AutodiffBackend,
    {
        let policy_batch = policy_batch?;
        let policy = self
            .ruliad_supervision
            .proof_policy_for_step(schedule_step_index);
        let dynamic_policy = !matches!(
            policy.effective_mode(schedule_step_index),
            crate::config::RuliadProofPolicyEffectiveMode::StaticExpert
        );
        let sampling_model = dynamic_policy.then(|| {
            self.model
                .valid()
                .materialize_random_scaffold_for_inference()
        });
        let prepared =
            local_predictive_coding::prepare_ruliad_verifier_terminal_at_step::<B::InnerBackend>(
                sampling_model.as_ref(),
                policy_batch,
                policy,
                block_size,
                self.model.vocab_size(),
                schedule_step_index,
                &clean_inputs.device(),
            )?;
        self.write_ruliad_proof_policy_dagger_telemetry(
            RuliadProofPolicyDaggerTelemetry::from_verifier_panel(
                &prepared.stats,
                policy,
                schedule_step_index,
                prepared.decision_rows,
            )
            .with_policy_sampling(Some(policy_batch)),
        );
        let prepared = local_predictive_coding::lift_ruliad_verifier_terminal::<B>(prepared);
        let semantic_states = prepared.semantic_states;
        let decision_rows = prepared.decision_rows;
        let [structured_batch_size, structured_sequence_len] = prepared.inputs.shape().dims::<2>();

        let started = Instant::now();
        let structured_hidden = self.model.forward_hidden(prepared.inputs);
        let structured_loss = prepared
            .criterion
            .verifier_autodiff_loss_from_hidden(&self.model, structured_hidden)?;

        if self.tbptt_persist_across_steps {
            let structured_backward_started = Instant::now();
            let structured_grads = structured_loss.clone().backward();
            let mut backward_ns = structured_backward_started.elapsed().as_nanos();
            let mut accumulator = GradientsAccumulator::new();
            accumulator.accumulate(self, GradientsParams::from_grads(structured_grads, self));

            let primary_loss = if supervised_token_count == Some(0) {
                let device = clean_inputs.device();
                self.advance_stream_state_without_update(
                    clean_inputs,
                    summary_event_mask,
                    reset_stream_state,
                );
                Tensor::zeros([1], &device)
            } else {
                let primary_inputs = self.corrupt_causal_inputs(clean_inputs.clone());
                let mut primary_state = self.load_step_state(reset_stream_state, block_size);
                let primary_hidden = if let Some(mask) = summary_event_mask {
                    self.model.forward_hidden_with_state_and_summary_event_mask(
                        primary_inputs,
                        mask,
                        &mut primary_state,
                    )
                } else {
                    self.model
                        .forward_hidden_with_state(primary_inputs, &mut primary_state)
                };
                let primary_loss = self.next_token_loss_from_hidden(
                    primary_hidden,
                    targets,
                    clean_inputs,
                    loss_mask,
                    None,
                );
                let primary_backward_started = Instant::now();
                let primary_grads = primary_loss.clone().backward();
                backward_ns =
                    backward_ns.saturating_add(primary_backward_started.elapsed().as_nanos());
                accumulator.accumulate(self, GradientsParams::from_grads(primary_grads, self));
                self.store_step_state(primary_state);
                primary_loss
            };
            let forward_ns = started.elapsed().as_nanos().saturating_sub(backward_ns);
            let loss = primary_loss + structured_loss;
            self.local_predictive_coding_profile
                .record_global_structured_terminal(
                    semantic_states,
                    decision_rows,
                    started.elapsed().as_nanos(),
                );
            if profiling {
                crate::train::profile::record_train_step(forward_ns, backward_ns);
                crate::train::profile::record_structured_terminal(
                    decision_rows,
                    structured_batch_size.saturating_mul(structured_sequence_len),
                );
            }
            return Some(TrainOutput {
                grads: self.apply_gradient_scale_schedule(accumulator.grads()),
                item: LanguageModelTrainItem::new(loss),
            });
        }

        let primary_inputs = self.corrupt_causal_inputs(clean_inputs.clone());
        let mut primary_state = self.model.init_state();
        let primary_hidden = if let Some(mask) = summary_event_mask {
            self.model.forward_hidden_with_state_and_summary_event_mask(
                primary_inputs,
                mask,
                &mut primary_state,
            )
        } else {
            self.model
                .forward_hidden_with_state(primary_inputs, &mut primary_state)
        };
        let primary_loss = self.next_token_loss_from_hidden(
            primary_hidden,
            targets,
            clean_inputs,
            loss_mask,
            None,
        );
        let forward_ns = started.elapsed().as_nanos();
        let loss = primary_loss + structured_loss;
        let backward_started = Instant::now();
        let grads = loss.backward();
        let backward_ns = backward_started.elapsed().as_nanos();

        self.local_predictive_coding_profile
            .record_global_structured_terminal(
                semantic_states,
                decision_rows,
                started.elapsed().as_nanos(),
            );
        if profiling {
            crate::train::profile::record_train_step(forward_ns, backward_ns);
            crate::train::profile::record_structured_terminal(
                decision_rows,
                structured_batch_size.saturating_mul(structured_sequence_len),
            );
        }
        Some(TrainOutput {
            grads: self.apply_gradient_scale_schedule(GradientsParams::from_grads(grads, self)),
            item: LanguageModelTrainItem::new(loss),
        })
    }
}
