//! Autodiff training-step dispatch and optimizer integration.

use super::*;
use crate::train::local_predictive_coding;

fn uses_shared_ruliad_verifier_terminal(
    policy: crate::config::RuliadProofPolicyTrainingConfig,
) -> bool {
    policy.scoring == crate::config::RuliadProofPolicyScoring::CompletionLikelihood
        && policy.normalization == crate::config::RuliadProofPolicyNormalization::PrefixConditional
        && policy.gradient_scope == crate::config::RuliadProofPolicyGradientScope::FullModel
}

impl<B: BackendTrait> LanguageTrainModel<B> {
    fn report_ruliad_verifier_terminal_skip(
        &self,
        policy_batch: Option<&crate::dataset::RuliadPolicyBatch>,
        policy: crate::config::RuliadProofPolicyTrainingConfig,
        step_index: usize,
        reason: &'static str,
    ) {
        self.write_ruliad_proof_policy_dagger_telemetry(RuliadProofPolicyDaggerTelemetry::skipped(
            policy_batch,
            policy,
            step_index,
            reason,
        ));
        self.local_predictive_coding_profile
            .record_structured_terminal_skip();
        assert!(
            !policy.require_scheduled_update,
            "required Ruliad proof-policy update failed at step {step_index}: {reason}"
        );
    }
}

impl<B: AutodiffBackend> TrainStep for LanguageTrainModel<B> {
    type Input = SequenceBatch<B>;
    type Output = LanguageModelTrainItem<B>;

    fn step(&self, batch: SequenceBatch<B>) -> TrainOutput<LanguageModelTrainItem<B>> {
        let prof_enabled = crate::train::profile::enabled();
        let step_index = self.gradient_scale_step.load(Ordering::Relaxed);
        let schedule_step_index = batch.absolute_step.unwrap_or(step_index);
        let detail_prof_enabled = prof_enabled && crate::train::profile::detail_due(step_index);
        let memory_prof_enabled = prof_enabled && crate::train::profile::memory_enabled();
        let forward_start = prof_enabled.then(Instant::now);
        B::seed(
            &batch.inputs.device(),
            stochastic_step_seed(self.stochastic_seed, step_index, STOCHASTIC_STREAM_MAIN),
        );
        let ruliad_policy_batch = batch.ruliad_policy_batch.clone();
        if prof_enabled {
            let [source_batch_size, source_sequence_len] = batch.inputs.shape().dims::<2>();
            crate::train::profile::record_source_batch(source_batch_size, source_sequence_len);
        }
        if self.uses_parallel_adjoint_predictive_coding() {
            return self.parallel_adjoint_predictive_coding_step(batch);
        }
        if self.uses_two_phase_dkp_predictive_coding() {
            return self.stage_dkp_predictive_coding_step(batch);
        }
        if self.uses_incremental_predictive_coding() {
            return self.stage_incremental_predictive_coding_step(batch);
        }
        let clean_inputs = batch.inputs;
        let targets = batch.targets;
        let loss_mask = batch.loss_mask;
        let known_supervised_token_count = batch.supervised_token_count;
        let summary_event_mask = batch.summary_event_mask;
        let reset_stream_state = batch.reset_stream_state;
        let [_, block_size] = clean_inputs.shape().dims::<2>();
        let verifier_terminal_due = local_predictive_coding::verifier_terminal_due(
            self.local_predictive_coding.terminal_criterion,
            self.ruliad_supervision.proof_policy,
            schedule_step_index,
        );
        if matches!(self.training_algorithm, TrainingAlgorithm::Backpropagation)
            && verifier_terminal_due
            && local_predictive_coding::verifier_terminal_preserves_primary(
                self.local_predictive_coding.terminal_criterion,
            )
        {
            if let Some(output) = self.joint_backprop_verifier_terminal_step(
                ruliad_policy_batch.as_deref(),
                clean_inputs.clone(),
                targets.clone(),
                loss_mask.clone(),
                known_supervised_token_count,
                summary_event_mask.clone(),
                reset_stream_state,
                block_size,
                schedule_step_index,
                prof_enabled,
            ) {
                return output;
            }
            let terminal_policy = self
                .ruliad_supervision
                .proof_policy_for_step(schedule_step_index);
            self.report_ruliad_verifier_terminal_skip(
                ruliad_policy_batch.as_deref(),
                terminal_policy,
                schedule_step_index,
                if ruliad_policy_batch.is_some() {
                    "unencodable_or_empty_verifier_panel"
                } else {
                    "missing_policy_batch"
                },
            );
        }
        if matches!(self.training_algorithm, TrainingAlgorithm::Backpropagation)
            && !local_predictive_coding::verifier_terminal_preserves_primary(
                self.local_predictive_coding.terminal_criterion,
            )
            && local_predictive_coding::verifier_terminal_due(
                self.local_predictive_coding.terminal_criterion,
                self.ruliad_supervision.proof_policy,
                schedule_step_index,
            )
        {
            let terminal_policy = self
                .ruliad_supervision
                .proof_policy_for_step(schedule_step_index);
            let shared_prefix_terminal = uses_shared_ruliad_verifier_terminal(terminal_policy);
            if !shared_prefix_terminal {
                let started = Instant::now();
                let forward_started = prof_enabled.then(Instant::now);
                let objective = ruliad_policy_batch.as_deref().and_then(|policy_batch| {
                    self.ruliad_proof_policy_objective_at_step(
                        policy_batch,
                        &clean_inputs.device(),
                        block_size,
                        schedule_step_index,
                    )
                });
                let forward_ns = forward_started
                    .map(|started| started.elapsed().as_nanos())
                    .unwrap_or_default();
                if let Some(objective) = objective {
                    let backward_started = prof_enabled.then(Instant::now);
                    let grads = objective.loss.backward();
                    let backward_ns = backward_started
                        .map(|started| started.elapsed().as_nanos())
                        .unwrap_or_default();
                    self.local_predictive_coding_profile
                        .record_global_structured_terminal(
                            objective.semantic_states,
                            objective.decision_rows,
                            started.elapsed().as_nanos(),
                        );
                    let stream_advance_started = prof_enabled.then(Instant::now);
                    self.advance_stream_state_without_update(
                        clean_inputs,
                        summary_event_mask,
                        reset_stream_state,
                    );
                    if prof_enabled {
                        crate::train::profile::record_train_step(forward_ns, backward_ns);
                        crate::train::profile::record_structured_terminal(
                            objective.decision_rows,
                            objective.padded_tokens,
                        );
                        crate::train::profile::record_stream_advance(
                            stream_advance_started
                                .expect("profiling start exists")
                                .elapsed()
                                .as_nanos(),
                        );
                    }
                    return TrainOutput {
                        grads: self.apply_gradient_scale_schedule(GradientsParams::from_grads(
                            grads, self,
                        )),
                        item: LanguageModelTrainItem::new(objective.loss),
                    };
                }
                self.report_ruliad_verifier_terminal_skip(
                    ruliad_policy_batch.as_deref(),
                    terminal_policy,
                    schedule_step_index,
                    if ruliad_policy_batch.is_some() {
                        "unencodable_or_empty_verifier_panel"
                    } else {
                        "missing_policy_batch"
                    },
                );
            }
            if shared_prefix_terminal {
                let started = Instant::now();
                let dynamic_policy = !matches!(
                    self.ruliad_supervision
                        .proof_policy
                        .effective_mode(schedule_step_index),
                    crate::config::RuliadProofPolicyEffectiveMode::StaticExpert
                );
                let sampling_model = dynamic_policy.then(|| {
                    self.model
                        .valid()
                        .materialize_random_scaffold_for_inference()
                });
                let prepared = ruliad_policy_batch.as_deref().and_then(|policy_batch| {
                    local_predictive_coding::prepare_ruliad_verifier_terminal_at_step::<
                        B::InnerBackend,
                    >(
                        sampling_model.as_ref(),
                        policy_batch,
                        terminal_policy,
                        block_size,
                        self.model.vocab_size(),
                        schedule_step_index,
                        &clean_inputs.device(),
                    )
                });
                if let Some(prepared) = prepared {
                    self.write_ruliad_proof_policy_dagger_telemetry(
                        RuliadProofPolicyDaggerTelemetry::from_verifier_panel(
                            &prepared.stats,
                            terminal_policy,
                            schedule_step_index,
                            prepared.decision_rows,
                        )
                        .with_policy_sampling(ruliad_policy_batch.as_deref()),
                    );
                    let prepared =
                        local_predictive_coding::lift_ruliad_verifier_terminal::<B>(prepared);
                    let semantic_states = prepared.semantic_states;
                    let decision_rows = prepared.decision_rows;
                    let [structured_batch_size, structured_sequence_len] =
                        prepared.inputs.shape().dims::<2>();
                    let forward_started = prof_enabled.then(Instant::now);
                    let logits = self.model.forward(prepared.inputs);
                    let loss = prepared
                        .criterion
                        .verifier_autodiff_loss(logits)
                        .expect("Ruliad verifier preparation must produce a verifier criterion");
                    let forward_ns = forward_started
                        .map(|started| started.elapsed().as_nanos())
                        .unwrap_or_default();
                    let backward_started = prof_enabled.then(Instant::now);
                    let grads = loss.backward();
                    let backward_ns = backward_started
                        .map(|started| started.elapsed().as_nanos())
                        .unwrap_or_default();
                    self.local_predictive_coding_profile
                        .record_global_structured_terminal(
                            semantic_states,
                            decision_rows,
                            started.elapsed().as_nanos(),
                        );
                    let stream_advance_started = prof_enabled.then(Instant::now);
                    self.advance_stream_state_without_update(
                        clean_inputs,
                        summary_event_mask,
                        reset_stream_state,
                    );
                    if prof_enabled {
                        crate::train::profile::record_train_step(forward_ns, backward_ns);
                        crate::train::profile::record_structured_terminal(
                            decision_rows,
                            structured_batch_size.saturating_mul(structured_sequence_len),
                        );
                        crate::train::profile::record_stream_advance(
                            stream_advance_started
                                .expect("profiling start exists")
                                .elapsed()
                                .as_nanos(),
                        );
                    }
                    return TrainOutput {
                        grads: self.apply_gradient_scale_schedule(GradientsParams::from_grads(
                            grads, self,
                        )),
                        item: LanguageModelTrainItem::new(loss),
                    };
                }
                self.report_ruliad_verifier_terminal_skip(
                    ruliad_policy_batch.as_deref(),
                    terminal_policy,
                    schedule_step_index,
                    if ruliad_policy_batch.is_some() {
                        "unencodable_or_empty_verifier_panel"
                    } else {
                        "missing_policy_batch"
                    },
                );
            }
        }
        if self
            .ruliad_supervision
            .prompt_value_binding
            .active_at_step(schedule_step_index)
            && !verifier_terminal_due
        {
            let input = super::prompt_value_binding::RuliadPromptValueBindingStepInput {
                policy_batch: ruliad_policy_batch.clone(),
                stream_inputs: clean_inputs.clone(),
                summary_event_mask: summary_event_mask.clone(),
                reset_stream_state,
                block_size,
                schedule_step_index,
                profiling: prof_enabled,
            };
            if let Some(output) = self.ruliad_prompt_value_binding_step(input) {
                return output;
            }
        }
        // A context-only streaming chunk still carries an independently scheduled
        // verifier terminal. Do not let the zero-token fast path silently drop
        // that objective (and bypass its required-delivery assertion).
        let predictive_coding_verifier_due =
            matches!(self.training_algorithm, TrainingAlgorithm::PredictiveCoding)
                && verifier_terminal_due;
        if self.objective.is_next_token()
            && known_supervised_token_count == Some(0)
            && !predictive_coding_verifier_due
        {
            let device = clean_inputs.device();
            let stream_advance_started = prof_enabled.then(Instant::now);
            self.advance_stream_state_without_update(
                clean_inputs,
                summary_event_mask,
                reset_stream_state,
            );
            if let Some(started) = stream_advance_started {
                crate::train::profile::record_stream_advance(started.elapsed().as_nanos());
            }
            return TrainOutput {
                grads: GradientsParams::new(),
                item: LanguageModelTrainItem::new(Tensor::zeros([1], &device)),
            };
        }
        if matches!(self.training_algorithm, TrainingAlgorithm::PredictiveCoding) {
            if local_predictive_coding::verifier_terminal_due(
                self.local_predictive_coding.terminal_criterion,
                self.ruliad_supervision.proof_policy,
                schedule_step_index,
            ) {
                let dynamic_policy = !matches!(
                    self.ruliad_supervision
                        .proof_policy
                        .effective_mode(schedule_step_index),
                    crate::config::RuliadProofPolicyEffectiveMode::StaticExpert
                );
                let sampling_model = dynamic_policy.then(|| {
                    self.model
                        .valid()
                        .materialize_random_scaffold_for_inference()
                });
                let prepared = ruliad_policy_batch.as_deref().and_then(|policy_batch| {
                    local_predictive_coding::prepare_ruliad_verifier_terminal_at_step::<
                        B::InnerBackend,
                    >(
                        sampling_model.as_ref(),
                        policy_batch,
                        self.ruliad_supervision
                            .proof_policy_for_step(schedule_step_index),
                        block_size,
                        self.model.vocab_size(),
                        schedule_step_index,
                        &clean_inputs.device(),
                    )
                });
                if let Some(prepared) = prepared {
                    self.write_ruliad_proof_policy_dagger_telemetry(
                        RuliadProofPolicyDaggerTelemetry::from_verifier_panel(
                            &prepared.stats,
                            self.ruliad_supervision
                                .proof_policy_for_step(schedule_step_index),
                            schedule_step_index,
                            prepared.decision_rows,
                        )
                        .with_policy_sampling(ruliad_policy_batch.as_deref()),
                    );
                    let semantic_rows = prepared.decision_rows;
                    let [structured_batch_size, structured_sequence_len] =
                        prepared.inputs.shape().dims::<2>();
                    let step = local_predictive_coding::local_predictive_coding_verifier_train_step(
                        &self.model,
                        prepared,
                        &self.local_predictive_coding,
                        &self.local_predictive_coding_profile,
                    );
                    debug_assert_eq!(step.report.global_backward_calls, 0);
                    let preserves_primary =
                        local_predictive_coding::verifier_terminal_preserves_primary(
                            self.local_predictive_coding.terminal_criterion,
                        );
                    if preserves_primary {
                        let primary = if self
                            .local_predictive_coding
                            .temporal_credit
                            .carries_temporal_credit()
                            && known_supervised_token_count != Some(0)
                        {
                            let chunk_size = self
                                .effective_tbptt_chunk_size(block_size)
                                .expect("validated exact temporal credit requires a TBPTT chunk");
                            self.primary_exact_window_predictive_coding_step(
                                clean_inputs,
                                targets,
                                loss_mask,
                                reset_stream_state,
                                chunk_size,
                            )
                        } else {
                            self.primary_local_predictive_coding_step(
                                clean_inputs,
                                targets,
                                loss_mask,
                                known_supervised_token_count,
                                summary_event_mask,
                                reset_stream_state,
                                block_size,
                            )
                        };
                        let mut accumulator = GradientsAccumulator::new();
                        accumulator.accumulate(self, primary.grads);
                        accumulator.accumulate(self, step.grads);
                        if prof_enabled {
                            crate::train::profile::record_local_learning_step(
                                step.report.elapsed_ns,
                            );
                            crate::train::profile::record_structured_terminal(
                                semantic_rows,
                                structured_batch_size.saturating_mul(structured_sequence_len),
                            );
                        }
                        return TrainOutput {
                            grads: self.apply_gradient_scale_schedule(accumulator.grads()),
                            item: LanguageModelTrainItem::new(primary.loss + step.loss),
                        };
                    }
                    if prof_enabled {
                        crate::train::profile::record_local_learning_step(step.report.elapsed_ns);
                        crate::train::profile::record_structured_terminal(
                            semantic_rows,
                            structured_batch_size.saturating_mul(structured_sequence_len),
                        );
                    }
                    let stream_advance_started = prof_enabled.then(Instant::now);
                    self.advance_stream_state_without_update(
                        clean_inputs,
                        summary_event_mask,
                        reset_stream_state,
                    );
                    if prof_enabled {
                        crate::train::profile::record_stream_advance(
                            stream_advance_started
                                .expect("profiling start exists")
                                .elapsed()
                                .as_nanos(),
                        );
                    }
                    return TrainOutput {
                        grads: self.apply_gradient_scale_schedule(step.grads),
                        item: LanguageModelTrainItem::new(step.loss),
                    };
                }
                self.report_ruliad_verifier_terminal_skip(
                    ruliad_policy_batch.as_deref(),
                    self.ruliad_supervision
                        .proof_policy_for_step(schedule_step_index),
                    schedule_step_index,
                    if ruliad_policy_batch.is_some() {
                        "unencodable_or_empty_verifier_panel"
                    } else {
                        "missing_policy_batch"
                    },
                );
            }
            let chunk_size = self.effective_tbptt_chunk_size(block_size);
            if self
                .local_predictive_coding
                .temporal_credit
                .carries_temporal_credit()
                && known_supervised_token_count != Some(0)
                && let Some(chunk_size) = chunk_size
            {
                let primary = self.primary_exact_window_predictive_coding_step(
                    clean_inputs,
                    targets,
                    loss_mask,
                    reset_stream_state,
                    chunk_size,
                );
                return TrainOutput {
                    grads: self.apply_gradient_scale_schedule(primary.grads),
                    item: LanguageModelTrainItem::new(primary.loss),
                };
            }
            let primary = self.primary_local_predictive_coding_step(
                clean_inputs,
                targets,
                loss_mask,
                known_supervised_token_count,
                summary_event_mask,
                reset_stream_state,
                block_size,
            );
            return TrainOutput {
                grads: self.apply_gradient_scale_schedule(primary.grads),
                item: LanguageModelTrainItem::new(primary.loss),
            };
        }
        if !self.objective.is_next_token() {
            self.update_teacher_runtime();
            let loss = self.objective_loss(clean_inputs, targets);
            let grads = loss.backward();
            return TrainOutput {
                grads: self.apply_gradient_scale_schedule(GradientsParams::from_grads(grads, self)),
                item: LanguageModelTrainItem::new(loss),
            };
        }
        if self.latent_reasoning.enabled {
            self.update_teacher_runtime();
        }
        let inputs = self.corrupt_causal_inputs(clean_inputs.clone());
        let clean_inputs_for_aux = clean_inputs.clone();
        let step_device = memory_prof_enabled.then(|| inputs.device());
        let step_memory_before = step_device
            .as_ref()
            .and_then(|device| device_memory_usage_safe::<B>(device));
        let [_batch_size, block_size] = inputs.shape().dims();
        let tbptt_chunk_size = self.effective_tbptt_chunk_size(block_size);
        let factorized_head = self.model.uses_factorized_language_head();
        // State inference needs gradients only for recurrent-state leaves. Build
        // one current-weight detached parameter view per train step and reuse it
        // across all corrected chunks.
        let predictive_coding_model_needed = tbptt_chunk_size.is_some_and(|chunk_size| {
            let chunks_per_step = block_size.div_ceil(chunk_size.max(1));
            (0..chunks_per_step).any(|chunk_index| {
                self.predictive_coding_active_for_chunk(step_index, chunk_index, chunks_per_step)
            })
        });
        let predictive_coding_model =
            predictive_coding_model_needed.then(|| detach_teacher_model(&self.model));
        let recurrent_teacher = self.recurrent_teacher_model();
        let (recurrent_teacher, recurrent_teacher_emits_logits) = match recurrent_teacher {
            Some((teacher, emit_logits)) => (Some(teacher), emit_logits),
            None => (None, false),
        };
        let mut recurrent_teacher_state = recurrent_teacher
            .as_ref()
            .map(|teacher| teacher.init_state());
        let probe_inputs = detail_prof_enabled.then(|| inputs.clone());
        let probe_summary_event_mask = detail_prof_enabled
            .then(|| summary_event_mask.clone())
            .flatten();
        let mut step_state = self.load_step_state(reset_stream_state, block_size);
        let (loss, probe_hidden, probe_logits, forward_ns) = if self.pipeline_enabled() {
            let forward_start = Instant::now();
            let (loss, hidden, logits) = self.forward_loss_with_pipeline(
                inputs,
                targets.clone(),
                loss_mask.clone(),
                summary_event_mask,
            );
            step_state = self.model.init_state();
            (
                loss,
                Some(hidden),
                (!factorized_head).then_some(logits),
                forward_start.elapsed().as_nanos(),
            )
        } else if let Some(chunk_size) = tbptt_chunk_size {
            let use_tbptt_block_backward = self.tbptt_credit_window_chunks == 1
                && if self.predictive_coding.enabled {
                    matches!(
                        self.predictive_coding.backward_mode,
                        PredictiveCodingBackwardMode::Block
                    )
                } else {
                    detail_prof_enabled
                };
            if use_tbptt_block_backward {
                let [batch_size, block_size] = inputs.shape().dims();
                let mut hidden_chunks = Vec::new();
                let mut logits_chunks = Vec::new();
                let mut teacher_logits_chunks = Vec::new();
                let mut total_forward_ns = 0u128;
                let mut predictive_coding_step_report = PredictiveCodingChunkReport::default();
                let chunks_per_step = block_size.div_ceil(chunk_size);
                for (chunk_index, start) in (0..block_size).step_by(chunk_size).enumerate() {
                    let end = (start + chunk_size).min(block_size);
                    let chunk_inputs = Self::slice_tokens(inputs.clone(), batch_size, start, end);
                    let chunk_summary_event_mask = summary_event_mask
                        .clone()
                        .map(|mask| Self::slice_tokens(mask, batch_size, start, end));
                    if self.predictive_coding_active_for_chunk(
                        step_index,
                        chunk_index,
                        chunks_per_step,
                    ) && matches!(
                        self.predictive_coding.observation_contract,
                        PredictiveCodingObservationContract::OracleNextTokenNegativeControl
                    ) {
                        let chunk_targets =
                            Self::slice_tokens(targets.clone(), batch_size, start, end);
                        let chunk_loss_mask = loss_mask
                            .clone()
                            .map(|mask| Self::slice_tokens(mask, batch_size, start, end));
                        let (corrected_state, report) = self
                            .correct_state_with_oracle_predictive_coding_using_model(
                                predictive_coding_model
                                    .as_ref()
                                    .expect("enabled predictive-coding model"),
                                step_state,
                                chunk_inputs.clone(),
                                chunk_targets,
                                chunk_loss_mask,
                                chunk_summary_event_mask.clone(),
                            );
                        step_state = corrected_state;
                        if self.predictive_coding.sync_diagnostics {
                            report.record();
                        } else {
                            predictive_coding_step_report.accumulate_unsynced(report);
                        }
                    }
                    let chunk_teacher_logits = if let (Some(teacher), Some(teacher_state)) =
                        (recurrent_teacher.as_ref(), recurrent_teacher_state.as_mut())
                    {
                        Self::teacher_forward_with_state(
                            teacher,
                            recurrent_teacher_emits_logits,
                            chunk_inputs.clone(),
                            chunk_summary_event_mask.clone(),
                            teacher_state,
                        )
                    } else {
                        None
                    };
                    let chunk_forward_start = Instant::now();
                    let hidden = if let Some(mask) = chunk_summary_event_mask.clone() {
                        self.model.forward_hidden_with_state_and_summary_event_mask(
                            chunk_inputs,
                            mask,
                            &mut step_state,
                        )
                    } else {
                        self.model
                            .forward_hidden_with_state(chunk_inputs, &mut step_state)
                    };
                    total_forward_ns += chunk_forward_start.elapsed().as_nanos();
                    hidden_chunks.push(hidden);
                    if detail_prof_enabled && !factorized_head {
                        logits_chunks.push(
                            self.model
                                .logits_from_hidden(hidden_chunks.last().expect("hidden").clone()),
                        );
                    }
                    if let Some(logits) = chunk_teacher_logits {
                        teacher_logits_chunks.push(logits);
                    }
                    if end < block_size {
                        step_state.detach_in_place();
                        if let Some(teacher_state) = recurrent_teacher_state.as_mut() {
                            teacher_state.detach_in_place();
                        }
                    }
                }
                if predictive_coding_step_report.has_activity() {
                    predictive_coding_step_report.record();
                }
                let hidden = Tensor::cat(hidden_chunks, 1);
                let teacher_logits = (!teacher_logits_chunks.is_empty())
                    .then(|| Tensor::cat(teacher_logits_chunks, 1));
                let loss = self.next_token_loss_from_hidden(
                    hidden.clone(),
                    targets.clone(),
                    clean_inputs.clone(),
                    loss_mask.clone(),
                    teacher_logits,
                );
                let loss = self.add_latent_rho_memory_auxiliary_loss(loss, &step_state);
                let loss = self.add_latent_dragon_state_auxiliary_loss(
                    loss,
                    &step_state,
                    recurrent_teacher_state.as_ref(),
                );
                let logits = (!factorized_head && !logits_chunks.is_empty())
                    .then(|| Tensor::cat(logits_chunks, 1));
                (
                    loss,
                    detail_prof_enabled.then_some(hidden),
                    logits,
                    total_forward_ns,
                )
            } else {
                let [batch_size, block_size] = inputs.shape().dims();
                let mut total_forward_ns = 0u128;
                let mut total_backward_ns = 0u128;
                let mut total_loss: Option<Tensor<B, 1>> = None;
                let mut window_loss: Option<Tensor<B, 1>> = None;
                let mut accumulator = GradientsAccumulator::new();
                let mut predictive_coding_step_report = PredictiveCodingChunkReport::default();
                let chunks_per_step = block_size.div_ceil(chunk_size);
                let credit_window_chunks = self.tbptt_credit_window_chunks.max(1);
                let total_supervised_tokens = supervised_token_count(
                    loss_mask.clone(),
                    batch_size,
                    block_size,
                    &targets.device(),
                )
                .clamp_min(1.0);

                for (chunk_index, start) in (0..block_size).step_by(chunk_size).enumerate() {
                    let end = (start + chunk_size).min(block_size);
                    let chunk_inputs = Self::slice_tokens(inputs.clone(), batch_size, start, end);
                    let chunk_clean_inputs =
                        Self::slice_tokens(clean_inputs.clone(), batch_size, start, end);
                    let chunk_targets = Self::slice_tokens(targets.clone(), batch_size, start, end);
                    let chunk_loss_mask = loss_mask
                        .clone()
                        .map(|mask| Self::slice_tokens(mask, batch_size, start, end));
                    let chunk_summary_event_mask = summary_event_mask
                        .clone()
                        .map(|mask| Self::slice_tokens(mask, batch_size, start, end));
                    let predictive_coding_active = self.predictive_coding_active_for_chunk(
                        step_index,
                        chunk_index,
                        chunks_per_step,
                    );
                    let observed_pc_entry = (predictive_coding_active
                        && matches!(
                            self.predictive_coding.observation_contract,
                            PredictiveCodingObservationContract::ObservedPrefix
                        ))
                    .then(|| step_state.detached_clone())
                    .filter(|state| {
                        Self::predictive_coding_state_has_latents(
                            state,
                            self.predictive_coding.state_scope,
                        )
                    });
                    if predictive_coding_active
                        && matches!(
                            self.predictive_coding.observation_contract,
                            PredictiveCodingObservationContract::OracleNextTokenNegativeControl
                        )
                    {
                        let (corrected_state, report) = self
                            .correct_state_with_oracle_predictive_coding_using_model(
                                predictive_coding_model
                                    .as_ref()
                                    .expect("enabled predictive-coding model"),
                                step_state,
                                chunk_inputs.clone(),
                                chunk_targets.clone(),
                                chunk_loss_mask.clone(),
                                chunk_summary_event_mask.clone(),
                            );
                        step_state = corrected_state;
                        if self.predictive_coding.sync_diagnostics {
                            report.record();
                        } else {
                            predictive_coding_step_report.accumulate_unsynced(report);
                        }
                    }
                    let chunk_teacher_logits = if let (Some(teacher), Some(teacher_state)) =
                        (recurrent_teacher.as_ref(), recurrent_teacher_state.as_mut())
                    {
                        Self::teacher_forward_with_state(
                            teacher,
                            recurrent_teacher_emits_logits,
                            chunk_inputs.clone(),
                            chunk_summary_event_mask.clone(),
                            teacher_state,
                        )
                    } else {
                        None
                    };

                    let chunk_forward_start = Instant::now();
                    let chunk_loss_parts = if let Some(mask) = chunk_summary_event_mask.clone() {
                        let hidden = self.model.forward_hidden_with_state_and_summary_event_mask(
                            chunk_inputs,
                            mask,
                            &mut step_state,
                        );
                        self.next_token_loss_parts_from_hidden(
                            hidden,
                            chunk_targets.clone(),
                            chunk_clean_inputs.clone(),
                            chunk_loss_mask.clone(),
                            chunk_teacher_logits,
                        )
                    } else {
                        let hidden = self
                            .model
                            .forward_hidden_with_state(chunk_inputs, &mut step_state);
                        self.next_token_loss_parts_from_hidden(
                            hidden,
                            chunk_targets.clone(),
                            chunk_clean_inputs.clone(),
                            chunk_loss_mask.clone(),
                            chunk_teacher_logits,
                        )
                    };
                    let chunk_weight = (end - start) as f32 / block_size as f32;
                    let mut chunk_loss = chunk_loss_parts
                        .tbptt_weighted(total_supervised_tokens.clone(), chunk_weight);
                    if let Some(auxiliary) = self.latent_rho_memory_auxiliary_loss(&step_state) {
                        chunk_loss = chunk_loss + auxiliary.mul_scalar(chunk_weight);
                    }
                    if let Some(auxiliary) = self.latent_dragon_state_auxiliary_loss(
                        &step_state,
                        recurrent_teacher_state.as_ref(),
                    ) {
                        chunk_loss = chunk_loss + auxiliary.mul_scalar(chunk_weight);
                    }
                    total_forward_ns += chunk_forward_start.elapsed().as_nanos();

                    if let Some(entry_state) = observed_pc_entry {
                        let (corrected_state, mut report) = self
                            .correct_state_from_observed_prefix_using_model(
                                predictive_coding_model
                                    .as_ref()
                                    .expect("enabled predictive-coding model"),
                                entry_state,
                                chunk_clean_inputs,
                                chunk_loss_mask,
                                chunk_summary_event_mask,
                            );
                        if report.chunks_corrected > 0 {
                            if matches!(
                                self.predictive_coding.parameter_update,
                                PredictiveCodingParameterUpdate::Optimizer
                            ) {
                                let (constraint, components) = self
                                    .predictive_coding_amortization_constraint(
                                        &step_state,
                                        &corrected_state,
                                    );
                                report.amortization_components = components;
                                if let Some(constraint) = constraint {
                                    if self.predictive_coding.sync_diagnostics {
                                        report.amortization_loss = Some(scalar_tensor_to_f64(
                                            constraint.clone().detach().inner(),
                                        ));
                                    }
                                    chunk_loss = chunk_loss + constraint.mul_scalar(chunk_weight);
                                }
                            } else {
                                // This explicitly non-learning control retains online state
                                // inference so it remains distinct from the AdamW baseline.
                                step_state = corrected_state;
                            }
                        }
                        if self.predictive_coding.sync_diagnostics {
                            report.record();
                        } else {
                            predictive_coding_step_report.accumulate_unsynced(report);
                        }
                    }

                    total_loss = Some(match total_loss {
                        Some(accumulated) => accumulated + chunk_loss.clone().detach(),
                        None => chunk_loss.clone().detach(),
                    });

                    window_loss = Some(match window_loss {
                        Some(accumulated) => accumulated + chunk_loss,
                        None => chunk_loss,
                    });
                    let window_complete =
                        end == block_size || (chunk_index + 1).is_multiple_of(credit_window_chunks);
                    if window_complete {
                        let window_backward_start = Instant::now();
                        let window_grads = window_loss
                            .take()
                            .expect("TBPTT credit window must contain a loss")
                            .backward();
                        total_backward_ns += window_backward_start.elapsed().as_nanos();
                        accumulator
                            .accumulate(self, GradientsParams::from_grads(window_grads, self));
                    }

                    if window_complete && end < block_size {
                        step_state.detach_in_place();
                        if let Some(teacher_state) = recurrent_teacher_state.as_mut() {
                            teacher_state.detach_in_place();
                        }
                    }
                }
                if predictive_coding_step_report.has_activity() {
                    predictive_coding_step_report.record();
                }

                if let Some(contract_loss) = self.ruliad_answer_contract_auxiliary_loss(
                    ruliad_policy_batch.as_deref(),
                    &targets.device(),
                    block_size,
                ) {
                    total_loss = Some(match total_loss {
                        Some(accumulated) => accumulated + contract_loss.clone().detach(),
                        None => contract_loss.clone().detach(),
                    });
                    let contract_grads = contract_loss.backward();
                    accumulator.accumulate(self, GradientsParams::from_grads(contract_grads, self));
                }

                if let Some(recovery_loss) = self.ruliad_structured_answer_recovery_auxiliary_loss(
                    ruliad_policy_batch.as_deref(),
                    &targets.device(),
                    block_size,
                ) {
                    total_loss = Some(match total_loss {
                        Some(accumulated) => accumulated + recovery_loss.clone().detach(),
                        None => recovery_loss.clone().detach(),
                    });
                    let recovery_grads = recovery_loss.backward();
                    accumulator.accumulate(self, GradientsParams::from_grads(recovery_grads, self));
                }

                let field_binding_weight = self.ruliad_field_binding_contrast_weight();
                if field_binding_weight > f32::EPSILON {
                    let field_binding_loss =
                        if let Some(policy_batch) = ruliad_policy_batch.as_deref() {
                            self.ruliad_field_binding_contrast_loss(
                                policy_batch,
                                &targets.device(),
                                block_size,
                            )
                        } else {
                            self.write_ruliad_field_binding_contrast_skip(
                                "missing_policy_batch",
                                field_binding_weight,
                            );
                            None
                        };
                    if let Some(field_binding_loss) = field_binding_loss {
                        total_loss = Some(match total_loss {
                            Some(accumulated) => accumulated + field_binding_loss.clone().detach(),
                            None => field_binding_loss.clone().detach(),
                        });
                        let field_binding_grads = field_binding_loss.backward();
                        accumulator.accumulate(
                            self,
                            GradientsParams::from_grads(field_binding_grads, self),
                        );
                    }
                }

                self.store_step_state(step_state);

                let step_memory_after_forward = step_device
                    .as_ref()
                    .and_then(|device| device_memory_usage_safe::<B>(device));
                if prof_enabled {
                    crate::train::profile::record_train_step(total_forward_ns, total_backward_ns);
                    if let (Some(before), Some(after_forward), Some(device)) = (
                        step_memory_before,
                        step_memory_after_forward,
                        step_device.as_ref(),
                    ) {
                        let after_backward =
                            device_memory_usage_safe::<B>(device).unwrap_or(after_forward);
                        crate::train::profile::record_train_step_memory(
                            before.reserved_bytes,
                            before.in_use_bytes,
                            after_forward.reserved_bytes,
                            after_forward.in_use_bytes,
                            after_backward.reserved_bytes,
                            after_backward.in_use_bytes,
                        );
                    }
                }

                return TrainOutput {
                    grads: self.apply_gradient_scale_schedule(accumulator.grads()),
                    item: LanguageModelTrainItem::new(
                        total_loss
                            .expect("tbptt train step should produce at least one loss chunk"),
                    ),
                };
            }
        } else if detail_prof_enabled {
            if let Some(summary_event_mask) = summary_event_mask {
                let teacher_logits = if let (Some(teacher), Some(teacher_state)) =
                    (recurrent_teacher.as_ref(), recurrent_teacher_state.as_mut())
                {
                    Self::teacher_forward_with_state(
                        teacher,
                        recurrent_teacher_emits_logits,
                        inputs.clone(),
                        Some(summary_event_mask.clone()),
                        teacher_state,
                    )
                } else {
                    None
                };
                let hidden = self.model.forward_hidden_with_state_and_summary_event_mask(
                    inputs,
                    summary_event_mask,
                    &mut step_state,
                );
                let forward_ns = forward_start
                    .map(|start| start.elapsed().as_nanos())
                    .unwrap_or_default();
                let loss = self.next_token_loss_from_hidden(
                    hidden.clone(),
                    targets.clone(),
                    clean_inputs.clone(),
                    loss_mask.clone(),
                    teacher_logits,
                );
                let loss = self.add_latent_rho_memory_auxiliary_loss(loss, &step_state);
                let loss = self.add_latent_dragon_state_auxiliary_loss(
                    loss,
                    &step_state,
                    recurrent_teacher_state.as_ref(),
                );
                let logits =
                    (!factorized_head).then(|| self.model.logits_from_hidden(hidden.clone()));
                (loss, Some(hidden), logits, forward_ns)
            } else {
                let teacher_logits = if let (Some(teacher), Some(teacher_state)) =
                    (recurrent_teacher.as_ref(), recurrent_teacher_state.as_mut())
                {
                    Self::teacher_forward_with_state(
                        teacher,
                        recurrent_teacher_emits_logits,
                        inputs.clone(),
                        None,
                        teacher_state,
                    )
                } else {
                    None
                };
                let hidden = self
                    .model
                    .forward_hidden_with_state(inputs, &mut step_state);
                let forward_ns = forward_start
                    .map(|start| start.elapsed().as_nanos())
                    .unwrap_or_default();
                let loss = self.next_token_loss_from_hidden(
                    hidden.clone(),
                    targets.clone(),
                    clean_inputs.clone(),
                    loss_mask.clone(),
                    teacher_logits,
                );
                let loss = self.add_latent_rho_memory_auxiliary_loss(loss, &step_state);
                let loss = self.add_latent_dragon_state_auxiliary_loss(
                    loss,
                    &step_state,
                    recurrent_teacher_state.as_ref(),
                );
                let logits =
                    (!factorized_head).then(|| self.model.logits_from_hidden(hidden.clone()));
                (loss, Some(hidden), logits, forward_ns)
            }
        } else {
            let teacher_logits = if let (Some(teacher), Some(teacher_state)) =
                (recurrent_teacher.as_ref(), recurrent_teacher_state.as_mut())
            {
                Self::teacher_forward_with_state(
                    teacher,
                    recurrent_teacher_emits_logits,
                    inputs.clone(),
                    summary_event_mask.clone(),
                    teacher_state,
                )
            } else {
                None
            };
            let hidden = if let Some(summary_event_mask) = summary_event_mask {
                self.model.forward_hidden_with_state_and_summary_event_mask(
                    inputs,
                    summary_event_mask,
                    &mut step_state,
                )
            } else {
                self.model
                    .forward_hidden_with_state(inputs, &mut step_state)
            };
            let forward_ns = forward_start
                .map(|start| start.elapsed().as_nanos())
                .unwrap_or_default();
            let loss = self.next_token_loss_from_hidden(
                hidden,
                targets.clone(),
                clean_inputs.clone(),
                loss_mask.clone(),
                teacher_logits,
            );
            let loss = self.add_latent_rho_memory_auxiliary_loss(loss, &step_state);
            let loss = self.add_latent_dragon_state_auxiliary_loss(
                loss,
                &step_state,
                recurrent_teacher_state.as_ref(),
            );
            (loss, None, None, forward_ns)
        };
        let auxiliary_objective_start = prof_enabled.then(Instant::now);
        let loss = if let Some(rollout_loss) =
            self.greedy_rollout_unlikelihood_loss(clean_inputs_for_aux)
        {
            loss + rollout_loss
        } else {
            loss
        };
        let loss = if let Some(contract_loss) = self.ruliad_answer_contract_auxiliary_loss(
            ruliad_policy_batch.as_deref(),
            &targets.device(),
            block_size,
        ) {
            loss + contract_loss
        } else {
            loss
        };
        let loss = if let Some(recovery_loss) = self
            .ruliad_structured_answer_recovery_auxiliary_loss(
                ruliad_policy_batch.as_deref(),
                &targets.device(),
                block_size,
            ) {
            loss + recovery_loss
        } else {
            loss
        };
        let contrast_weight = self.ruliad_structured_contrast_weight();
        let loss = if contrast_weight > f32::EPSILON {
            if let Some(policy_batch) = ruliad_policy_batch.as_deref() {
                if let Some(contrast_loss) = self.ruliad_structured_answer_contrast_loss(
                    policy_batch,
                    &targets.device(),
                    block_size,
                ) {
                    loss + contrast_loss
                } else {
                    loss
                }
            } else {
                self.write_ruliad_structured_contrast_skip("missing_policy_batch", contrast_weight);
                loss
            }
        } else {
            loss
        };
        let field_binding_weight = self.ruliad_field_binding_contrast_weight();
        let loss = if field_binding_weight > f32::EPSILON {
            if let Some(policy_batch) = ruliad_policy_batch.as_deref() {
                if let Some(field_binding_loss) = self.ruliad_field_binding_contrast_loss(
                    policy_batch,
                    &targets.device(),
                    block_size,
                ) {
                    loss + field_binding_loss
                } else {
                    loss
                }
            } else {
                self.write_ruliad_field_binding_contrast_skip(
                    "missing_policy_batch",
                    field_binding_weight,
                );
                loss
            }
        } else {
            loss
        };
        let loss = if let Some(policy_batch) = ruliad_policy_batch.as_deref()
            && let Some(rollout_imitation_loss) = self.ruliad_verifier_rollout_imitation_loss(
                policy_batch,
                &targets.device(),
                block_size,
            ) {
            loss + rollout_imitation_loss
        } else {
            loss
        };
        B::seed(
            &targets.device(),
            stochastic_step_seed(
                self.stochastic_seed,
                step_index,
                STOCHASTIC_STREAM_PROOF_POLICY,
            ),
        );
        let proof_policy_start = prof_enabled.then(Instant::now);
        let loss = if let Some(policy_batch) = ruliad_policy_batch.as_deref()
            && let Some(proof_policy_loss) = self.ruliad_proof_policy_dagger_loss_at_step(
                policy_batch,
                &targets.device(),
                block_size,
                schedule_step_index,
            ) {
            loss + proof_policy_loss
        } else {
            loss
        };
        let proof_policy_ns = proof_policy_start
            .map(|start| start.elapsed().as_nanos())
            .unwrap_or_default();
        B::seed(
            &targets.device(),
            stochastic_step_seed(
                self.stochastic_seed,
                step_index,
                STOCHASTIC_STREAM_VERIFIER_POLICY,
            ),
        );
        let loss = if let Some(policy_batch) = ruliad_policy_batch.as_deref()
            && let Some(policy_loss) =
                self.ruliad_verifier_policy_loss(policy_batch, &targets.device(), block_size)
        {
            loss + policy_loss
        } else {
            loss
        };
        let auxiliary_objective_ns = auxiliary_objective_start
            .map(|start| start.elapsed().as_nanos())
            .unwrap_or_default();
        self.store_step_state(step_state);
        let step_memory_after_forward = step_device
            .as_ref()
            .and_then(|device| device_memory_usage_safe::<B>(device));

        let probe_targets = (prof_enabled && detail_prof_enabled).then(|| targets.clone());
        let probe_logits = if prof_enabled && detail_prof_enabled {
            probe_logits.clone().map(|logits| logits.detach())
        } else {
            None
        };
        let probe_hidden = probe_hidden.map(|hidden| hidden.detach());

        let loss_backward_start = prof_enabled.then(Instant::now);
        let grads = loss.backward();
        let loss_backward_ns = loss_backward_start
            .map(|start| start.elapsed().as_nanos())
            .unwrap_or_default();

        if prof_enabled {
            crate::train::profile::record_auxiliary_objectives(
                auxiliary_objective_ns,
                proof_policy_ns,
            );
            crate::train::profile::record_train_step(forward_ns, loss_backward_ns);
            if let (Some(before), Some(after_forward), Some(device)) = (
                step_memory_before,
                step_memory_after_forward,
                step_device.as_ref(),
            ) {
                let after_backward = device_memory_usage_safe::<B>(device).unwrap_or(after_forward);
                crate::train::profile::record_train_step_memory(
                    before.reserved_bytes,
                    before.in_use_bytes,
                    after_forward.reserved_bytes,
                    after_forward.in_use_bytes,
                    after_backward.reserved_bytes,
                    after_backward.in_use_bytes,
                );
            }
            if detail_prof_enabled {
                let mut embed_probe_ns = 0;
                let mut first_layer_forward_probe_ns = 0;
                let mut first_layer_probe_ns = 0;
                let mut logits_loss_probe_ns = 0;
                let mut hidden_logits_loss_probe_ns = 0;
                let mut hidden_model_forward_probe_ns = 0;
                let mut hidden_model_probe_ns = 0;
                if let Some(probe_inputs) = probe_inputs.clone() {
                    let embed_start = Instant::now();
                    let probe_embedded = self.model.embed_tokens(probe_inputs);
                    let embed_loss = probe_embedded.clone().tanh().powf_scalar(2.0).mean();
                    let _embed_grads = embed_loss.backward();
                    let _ = B::sync(&probe_embedded.device());
                    embed_probe_ns = embed_start.elapsed().as_nanos();

                    let first_layer_forward_start = Instant::now();
                    let first_layer_forward_hidden = self
                        .model
                        .forward_hidden_prefix_layers_from_embedded_for_profile(
                            probe_embedded.clone().detach(),
                            1,
                            probe_summary_event_mask.clone(),
                        );
                    let _ = B::sync(&first_layer_forward_hidden.device());
                    first_layer_forward_probe_ns = first_layer_forward_start.elapsed().as_nanos();

                    let first_layer_start = Instant::now();
                    let probe_embedded_leaf = probe_embedded.detach().require_grad();
                    let first_layer_hidden = self
                        .model
                        .forward_hidden_prefix_layers_from_embedded_for_profile(
                            probe_embedded_leaf.clone(),
                            1,
                            probe_summary_event_mask.clone(),
                        );
                    let first_layer_loss =
                        first_layer_hidden.clone().tanh().powf_scalar(2.0).mean();
                    let _first_layer_grads = first_layer_loss.backward();
                    let _ = B::sync(&probe_embedded_leaf.device());
                    first_layer_probe_ns = first_layer_start.elapsed().as_nanos();
                }
                if let (Some(probe_targets), Some(probe_logits), Some(probe_hidden)) =
                    (probe_targets, probe_logits, probe_hidden)
                {
                    let hidden_model_forward_start = Instant::now();
                    let probe_hidden_forward = if let Some(mask) = probe_summary_event_mask.clone()
                    {
                        let mut probe_hidden_forward_state = self.model.init_state();
                        self.model
                            .forward_with_hidden_and_state_and_summary_event_mask(
                                probe_inputs
                                    .clone()
                                    .expect("probe inputs for hidden forward probe"),
                                mask,
                                &mut probe_hidden_forward_state,
                            )
                            .0
                    } else {
                        self.model
                            .forward_with_hidden(
                                probe_inputs
                                    .clone()
                                    .expect("probe inputs for hidden forward probe"),
                            )
                            .0
                    };
                    let _ = B::sync(&probe_hidden_forward.device());
                    hidden_model_forward_probe_ns = hidden_model_forward_start.elapsed().as_nanos();

                    let logits_only_start = Instant::now();
                    let probe_logits_leaf = probe_logits.require_grad();
                    let logits_only_loss =
                        language_model_loss::<B>(probe_logits_leaf.clone(), probe_targets.clone());
                    let logits_only_grads = logits_only_loss.backward();
                    let _ = probe_logits_leaf
                        .grad(&logits_only_grads)
                        .expect("probe logits grad")
                        .sum()
                        .into_data();
                    logits_loss_probe_ns = logits_only_start.elapsed().as_nanos();

                    let hidden_logits_start = Instant::now();
                    let probe_hidden_leaf = probe_hidden.require_grad();
                    let hidden_logits_loss = language_model_loss::<B>(
                        self.model.logits_from_hidden(probe_hidden_leaf.clone()),
                        probe_targets,
                    );
                    let hidden_logits_grads = hidden_logits_loss.backward();
                    let _ = probe_hidden_leaf
                        .grad(&hidden_logits_grads)
                        .expect("probe hidden grad")
                        .sum()
                        .into_data();
                    hidden_logits_loss_probe_ns = hidden_logits_start.elapsed().as_nanos();
                }
                if let Some(probe_inputs) = probe_inputs {
                    let hidden_model_start = Instant::now();
                    let probe_hidden_model =
                        if let Some(summary_event_mask) = probe_summary_event_mask {
                            let mut probe_state = self.model.init_state();
                            self.model
                                .forward_with_hidden_and_state_and_summary_event_mask(
                                    probe_inputs,
                                    summary_event_mask,
                                    &mut probe_state,
                                )
                                .0
                        } else {
                            self.model.forward_with_hidden(probe_inputs).0
                        };
                    let hidden_model_loss =
                        probe_hidden_model.clone().tanh().powf_scalar(2.0).mean();
                    let _hidden_model_grads = hidden_model_loss.backward();
                    let _ = B::sync(&probe_hidden_model.device());
                    hidden_model_probe_ns = hidden_model_start.elapsed().as_nanos();
                }
                crate::train::profile::record_detail_probe(
                    embed_probe_ns,
                    first_layer_forward_probe_ns,
                    first_layer_probe_ns,
                    logits_loss_probe_ns,
                    hidden_logits_loss_probe_ns,
                    hidden_model_forward_probe_ns,
                    hidden_model_probe_ns,
                );
            }
        }

        TrainOutput {
            grads: self.apply_gradient_scale_schedule(GradientsParams::from_grads(grads, self)),
            item: LanguageModelTrainItem::new(loss),
        }
    }

    fn optimize<B2, O>(self, optim: &mut O, lr: f64, grads: GradientsParams) -> Self
    where
        B2: AutodiffBackend,
        O: Optimizer<Self, B2>,
        Self: AutodiffModule<B2>,
    {
        if self.uses_two_phase_dkp_predictive_coding() {
            return self.optimize_dkp_predictive_coding::<B2, O>(optim, lr);
        }
        if self.uses_incremental_predictive_coding() {
            return self.optimize_incremental_predictive_coding::<B2, O>(optim, lr);
        }
        self.local_predictive_coding_profile
            .record_optimizer_updates(1);
        optim.step(lr, self, grads)
    }
}

#[cfg(test)]
mod dispatch_tests {
    use super::*;

    #[test]
    fn full_model_prefix_counterfactual_policy_uses_shared_verifier_terminal() {
        let policy = crate::config::RuliadProofPolicyTrainingConfig {
            scoring: crate::config::RuliadProofPolicyScoring::CompletionLikelihood,
            normalization: crate::config::RuliadProofPolicyNormalization::PrefixConditional,
            gradient_scope: crate::config::RuliadProofPolicyGradientScope::FullModel,
            counterfactual_targets_per_state: 1,
            ..crate::config::RuliadProofPolicyTrainingConfig::default()
        };
        assert!(uses_shared_ruliad_verifier_terminal(policy));

        assert!(!uses_shared_ruliad_verifier_terminal(
            crate::config::RuliadProofPolicyTrainingConfig {
                scoring: crate::config::RuliadProofPolicyScoring::SemanticEnergy,
                ..policy
            }
        ));
        assert!(!uses_shared_ruliad_verifier_terminal(
            crate::config::RuliadProofPolicyTrainingConfig {
                gradient_scope: crate::config::RuliadProofPolicyGradientScope::LanguageHeadOnly,
                ..policy
            }
        ));
    }
}
