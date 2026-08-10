//! Local predictive-coding staging, optimizer ownership, and context routing.

use super::*;
use crate::train::local_predictive_coding;

pub(crate) struct PredictiveContextTrainStep<B: AutodiffBackend> {
    pub output: TrainOutput<LanguageModelTrainItem<B>>,
    pub terminal_state: Option<ModelState<B>>,
}

impl<B: AutodiffBackend> LanguageTrainModel<B> {
    /// Execute fixed-prediction local PC over bounded exact recurrent-credit
    /// windows. Each window retains at most `window_chunks` plain-backend
    /// traces, reverses them once, and explicitly detaches its oldest state.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn exact_window_predictive_coding_step(
        &self,
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
        loss_mask: Option<Tensor<B, 2, Int>>,
        reset_stream_state: bool,
        chunk_size: usize,
    ) -> TrainOutput<LanguageModelTrainItem<B>>
    where
        B::Device: 'static,
        B::FloatTensorPrimitive: 'static,
    {
        debug_assert!(
            self.local_predictive_coding
                .temporal_credit
                .carries_temporal_credit()
        );
        debug_assert!(matches!(
            self.local_predictive_coding.solver,
            LocalPredictiveCodingSolver::FixedPrediction
        ));
        let [batch_size, block_size] = inputs.shape().dims::<2>();
        let chunk_starts = (0..block_size).step_by(chunk_size).collect::<Vec<_>>();
        let window_chunks = self.local_predictive_coding.temporal_credit.window_chunks;
        let executor = local_predictive_coding::FixedPredictionTemporalExecutor::new(&self.model);
        let mut state = self.load_step_state(reset_stream_state, block_size);
        let mut accumulator = GradientsAccumulator::new();
        let mut total_loss: Option<Tensor<B, 1>> = None;
        let mut total_supervised_tokens: Option<Tensor<B, 1>> = None;
        let mut total_elapsed_ns = 0u128;

        for window in chunk_starts.chunks(window_chunks) {
            let mut prepared = Vec::with_capacity(window.len());
            for &start in window {
                let end = (start + chunk_size).min(block_size);
                let chunk = executor.prepare(
                    Self::slice_tokens(inputs.clone(), batch_size, start, end),
                    Self::slice_tokens(targets.clone(), batch_size, start, end),
                    loss_mask
                        .clone()
                        .map(|mask| Self::slice_tokens(mask, batch_size, start, end)),
                    state,
                    &self.local_predictive_coding,
                );
                state = chunk.terminal_state();
                prepared.push(chunk);
            }

            let mut future_rho_adjoints = None;
            for chunk in prepared.into_iter().rev() {
                let mut derivatives = executor.finish(
                    chunk,
                    future_rho_adjoints.take(),
                    &self.local_predictive_coding,
                    &self.local_predictive_coding_profile,
                );
                debug_assert_eq!(derivatives.report.global_backward_calls, 0);
                future_rho_adjoints = Some(std::mem::take(&mut derivatives.initial_rho_adjoints));
                accumulator.accumulate(self, derivatives.grads);
                let supervised_tokens = derivatives.supervised_tokens;
                let weighted_loss = derivatives.loss * supervised_tokens.clone();
                total_loss = Some(match total_loss {
                    Some(accumulated) => accumulated + weighted_loss,
                    None => weighted_loss,
                });
                total_supervised_tokens = Some(match total_supervised_tokens {
                    Some(accumulated) => accumulated + supervised_tokens,
                    None => supervised_tokens,
                });
                total_elapsed_ns = total_elapsed_ns.saturating_add(derivatives.report.elapsed_ns);
            }
            // Dropping the oldest adjoint is the explicit bounded-window
            // truncation contract. The causal state itself continues forward.
        }

        self.store_step_state(state);
        if crate::train::profile::enabled() {
            crate::train::profile::record_local_learning_step(total_elapsed_ns);
        }
        let supervised_tokens = total_supervised_tokens
            .expect("exact-window local PC requires at least one chunk")
            .clamp_min(1.0);
        let mut grads = accumulator.grads();
        rescale_gradients_by_device_scalar::<B, _>(
            self,
            &mut grads,
            supervised_tokens.clone().inner(),
            true,
        );
        let loss = total_loss.expect("exact-window local PC requires at least one chunk")
            / supervised_tokens;
        TrainOutput {
            grads: self.apply_gradient_scale_schedule(grads),
            item: LanguageModelTrainItem::new(loss),
        }
    }

    pub(super) fn stage_incremental_predictive_coding_step(
        &self,
        batch: SequenceBatch<B>,
    ) -> TrainOutput<LanguageModelTrainItem<B>>
    where
        B::Device: 'static,
        B::FloatTensorPrimitive: 'static,
    {
        let profile_started = crate::train::profile::enabled().then(burn_dragon_time::Instant::now);
        let [batch_size, block_size] = batch.inputs.shape().dims::<2>();
        let chunk_size = self
            .effective_tbptt_chunk_size(block_size)
            .unwrap_or(block_size);
        let initial_state = self.load_step_state(batch.reset_stream_state, block_size);
        let mut metric_state = initial_state.clone();
        let mut total_loss: Option<Tensor<B, 1>> = None;
        let mut total_supervised_tokens: Option<Tensor<B, 1>> = None;
        let mut first_chunk = None;

        for start in (0..block_size).step_by(chunk_size) {
            let end = (start + chunk_size).min(block_size);
            let chunk = local_predictive_coding::prepare_incremental_predictive_coding_chunk(
                &self.model,
                Self::slice_tokens(batch.inputs.clone(), batch_size, start, end),
                Self::slice_tokens(batch.targets.clone(), batch_size, start, end),
                batch
                    .loss_mask
                    .clone()
                    .map(|mask| Self::slice_tokens(mask, batch_size, start, end)),
                metric_state,
                &self.local_predictive_coding,
            );
            metric_state = chunk.terminal_state.clone();
            let supervised_tokens = chunk.supervised_tokens.clone();
            let weighted_loss = chunk.loss.clone() * supervised_tokens.clone();
            total_loss = Some(match total_loss {
                Some(accumulated) => accumulated + weighted_loss,
                None => weighted_loss,
            });
            total_supervised_tokens = Some(match total_supervised_tokens {
                Some(accumulated) => accumulated + supervised_tokens,
                None => supervised_tokens,
            });
            if first_chunk.is_none() {
                first_chunk = Some(chunk);
            }
        }

        let pending = IncrementalPredictiveCodingPendingBatch {
            batch,
            initial_state,
            first_chunk: first_chunk.expect("incremental PC requires at least one chunk"),
        };
        let mut runtime = self
            .incremental_predictive_coding_runtime
            .inner
            .lock()
            .expect("incremental predictive-coding runtime lock poisoned");
        assert!(
            runtime.is_none(),
            "incremental predictive-coding batch was not consumed by the optimizer"
        );
        *runtime = Some(Box::new(pending));
        drop(runtime);
        if let Some(started) = profile_started {
            crate::train::profile::record_local_learning(started.elapsed().as_nanos());
        }

        let supervised_tokens = total_supervised_tokens
            .expect("incremental PC requires at least one chunk")
            .clamp_min(1.0);
        TrainOutput {
            grads: GradientsParams::new(),
            item: LanguageModelTrainItem::new(
                total_loss.expect("incremental PC requires at least one chunk") / supervised_tokens,
            ),
        }
    }

    pub(super) fn optimize_incremental_predictive_coding<B2, O>(
        mut self,
        optim: &mut O,
        lr: f64,
    ) -> Self
    where
        B2: AutodiffBackend,
        O: Optimizer<Self, B2>,
        Self: AutodiffModule<B2>,
        B::Device: 'static,
        B::FloatTensorPrimitive: 'static,
    {
        let pending = self
            .incremental_predictive_coding_runtime
            .inner
            .lock()
            .expect("incremental predictive-coding runtime lock poisoned")
            .take()
            .expect("incremental predictive-coding optimizer step has no staged batch")
            .downcast::<IncrementalPredictiveCodingPendingBatch<B>>()
            .unwrap_or_else(|_| panic!("incremental predictive-coding backend type mismatch"));
        let IncrementalPredictiveCodingPendingBatch {
            batch,
            initial_state,
            first_chunk,
        } = *pending;
        let config = self.local_predictive_coding.clone();
        let [batch_size, block_size] = batch.inputs.shape().dims::<2>();
        let chunk_size = self
            .effective_tbptt_chunk_size(block_size)
            .unwrap_or(block_size);
        let mut state = initial_state;
        let mut first_chunk = Some(first_chunk);
        let parameter_lr = lr * config.incremental_parameter_step_scale;
        let mut outer_local_learning_ns = 0u128;

        for start in (0..block_size).step_by(chunk_size) {
            let mut local_started = Instant::now();
            let mut local_learning_ns = 0u128;
            let end = (start + chunk_size).min(block_size);
            let mut chunk = if start == 0 {
                first_chunk
                    .take()
                    .expect("staged incremental PC first chunk")
            } else {
                local_predictive_coding::prepare_incremental_predictive_coding_chunk(
                    &self.model,
                    Self::slice_tokens(batch.inputs.clone(), batch_size, start, end),
                    Self::slice_tokens(batch.targets.clone(), batch_size, start, end),
                    batch
                        .loss_mask
                        .clone()
                        .map(|mask| Self::slice_tokens(mask, batch_size, start, end)),
                    state,
                    &config,
                )
            };
            state = chunk.terminal_state.clone();
            let energy_before = config.sync_diagnostics.then(|| {
                local_predictive_coding::incremental_predictive_coding_energy(
                    &self.model,
                    &chunk,
                    &config,
                )
            });
            let mut local_vjp_calls = 0usize;
            let mut gradient_tensors = 0usize;
            for _ in 0..config.inference.steps {
                local_vjp_calls = local_vjp_calls.saturating_add(
                    local_predictive_coding::incremental_predictive_coding_infer(
                        &self.model,
                        &mut chunk,
                        &config,
                    ),
                );
                let derivatives =
                    local_predictive_coding::incremental_predictive_coding_parameter_derivatives(
                        &self.model,
                        &chunk,
                        &config,
                    );
                local_vjp_calls = local_vjp_calls.saturating_add(derivatives.local_vjp_calls);
                gradient_tensors = gradient_tensors.saturating_add(derivatives.gradient_tensors);
                let grads = self.apply_gradient_scale_schedule(derivatives.grads);
                local_learning_ns =
                    local_learning_ns.saturating_add(local_started.elapsed().as_nanos());
                let optimizer_started =
                    crate::train::profile::enabled().then(burn_dragon_time::Instant::now);
                self = optim.step(parameter_lr, self, grads);
                if let Some(started) = optimizer_started {
                    crate::train::profile::record_optimizer(started.elapsed().as_nanos());
                }
                self.local_predictive_coding_profile
                    .record_optimizer_updates(1);
                local_started = Instant::now();
            }
            let energy_after = config.sync_diagnostics.then(|| {
                local_predictive_coding::incremental_predictive_coding_energy(
                    &self.model,
                    &chunk,
                    &config,
                )
            });
            local_learning_ns =
                local_learning_ns.saturating_add(local_started.elapsed().as_nanos());
            let report = local_predictive_coding::LocalPredictiveCodingStepReport {
                solver: config.solver,
                inference_steps: config.inference.steps,
                dual_steps: 0,
                factors: self.model.predictive_coding_layer_count() + 1,
                local_vjp_calls,
                temporal_state_vjp_calls: 0,
                fused_temporal_vjp_calls: 0,
                global_backward_calls: 0,
                gradient_tensors,
                direct_forward_updates: 0,
                feedback_parameter_updates: 0,
                adjoint_teacher_updates: 0,
                adjoint_local_updates: 0,
                parameter_updates: config.inference.steps,
                energy_before,
                energy_after,
                grad_norm_mean: None,
                grad_norm_max: None,
                delta_rms_mean: None,
                clip_fraction_mean: None,
                constraint_rms: None,
                dual_rms: None,
                composite_signal_rms: None,
                elapsed_ns: local_learning_ns,
            };
            local_predictive_coding::validate_step_execution_contract(&config, &report);
            self.local_predictive_coding_profile.record(report);
            outer_local_learning_ns = outer_local_learning_ns.saturating_add(report.elapsed_ns);
        }
        if crate::train::profile::enabled() {
            crate::train::profile::record_local_learning_step(outer_local_learning_ns);
        }
        self.store_step_state(state);
        self
    }

    pub(super) fn parallel_adjoint_predictive_coding_step(
        &self,
        batch: SequenceBatch<B>,
    ) -> TrainOutput<LanguageModelTrainItem<B>>
    where
        B::Device: 'static,
        B::FloatTensorPrimitive: 'static,
    {
        let [batch_size, block_size] = batch.inputs.shape().dims::<2>();
        let chunk_size = self
            .effective_tbptt_chunk_size(block_size)
            .unwrap_or(block_size);
        let mut state = self.load_step_state(batch.reset_stream_state, block_size);
        let feedback_state = self.dkp_feedback_for_checkpoint();
        let mut feedback = feedback_state.as_ref().map(|state| state.feedback.clone());
        let mut feedback_updates = feedback_state.map_or(0, |state| state.updates);
        let mut accumulator = GradientsAccumulator::new();
        let mut total_loss: Option<Tensor<B, 1>> = None;
        let mut total_supervised_tokens: Option<Tensor<B, 1>> = None;
        let mut total_elapsed_ns = 0u128;

        for start in (0..block_size).step_by(chunk_size) {
            let end = (start + chunk_size).min(block_size);
            let mut derivatives =
                local_predictive_coding::parallel_adjoint_predictive_coding_train_step(
                    &self.model,
                    Self::slice_tokens(batch.inputs.clone(), batch_size, start, end),
                    Self::slice_tokens(batch.targets.clone(), batch_size, start, end),
                    batch
                        .loss_mask
                        .clone()
                        .map(|mask| Self::slice_tokens(mask, batch_size, start, end)),
                    state,
                    feedback,
                    feedback_updates,
                    &self.local_predictive_coding,
                    &self.local_predictive_coding_profile,
                );
            state = derivatives.terminal_state;
            feedback = derivatives.dkp_feedback.take();
            if feedback.is_some() {
                feedback_updates = feedback_updates.saturating_add(1);
            }
            let supervised_tokens = derivatives.supervised_tokens.clone();
            rescale_gradients_by_device_scalar::<B, _>(
                self,
                &mut derivatives.grads,
                supervised_tokens.clone().inner(),
                false,
            );
            accumulator.accumulate(self, derivatives.grads);
            let weighted_loss = derivatives.loss * supervised_tokens.clone();
            total_loss = Some(match total_loss {
                Some(accumulated) => accumulated + weighted_loss,
                None => weighted_loss,
            });
            total_supervised_tokens = Some(match total_supervised_tokens {
                Some(accumulated) => accumulated + supervised_tokens,
                None => supervised_tokens,
            });
            total_elapsed_ns = total_elapsed_ns.saturating_add(derivatives.report.elapsed_ns);
        }
        self.restore_dkp_feedback_from_checkpoint(feedback.map(|feedback| {
            local_predictive_coding::DkpFeedbackState {
                feedback,
                updates: feedback_updates,
            }
        }));
        self.store_step_state(state);
        if crate::train::profile::enabled() {
            crate::train::profile::record_local_learning_step(total_elapsed_ns);
        }
        let supervised_tokens = total_supervised_tokens
            .expect("parallel adjoint requires at least one chunk")
            .clamp_min(1.0);
        let mut grads = accumulator.grads();
        rescale_gradients_by_device_scalar::<B, _>(
            self,
            &mut grads,
            supervised_tokens.clone().inner(),
            true,
        );
        TrainOutput {
            grads: self.apply_gradient_scale_schedule(grads),
            item: LanguageModelTrainItem::new(
                total_loss.expect("parallel adjoint requires at least one chunk")
                    / supervised_tokens,
            ),
        }
    }

    pub(super) fn stage_dkp_predictive_coding_step(
        &self,
        batch: SequenceBatch<B>,
    ) -> TrainOutput<LanguageModelTrainItem<B>>
    where
        B::Device: 'static,
        B::FloatTensorPrimitive: 'static,
    {
        let profile_started = crate::train::profile::enabled().then(burn_dragon_time::Instant::now);
        let [batch_size, block_size] = batch.inputs.shape().dims::<2>();
        let chunk_size = self
            .effective_tbptt_chunk_size(block_size)
            .unwrap_or(block_size);
        let initial_state = self.load_step_state(batch.reset_stream_state, block_size);
        let mut metric_state = initial_state.clone();
        let feedback_state = self.dkp_feedback_for_checkpoint();
        let mut feedback = feedback_state.as_ref().map(|state| state.feedback.clone());
        let feedback_updates = feedback_state.map_or(0, |state| state.updates);
        let mut total_loss: Option<Tensor<B, 1>> = None;
        let mut total_supervised_tokens: Option<Tensor<B, 1>> = None;
        let mut first_chunk = None;

        for start in (0..block_size).step_by(chunk_size) {
            let end = (start + chunk_size).min(block_size);
            let inputs = Self::slice_tokens(batch.inputs.clone(), batch_size, start, end);
            let targets = Self::slice_tokens(batch.targets.clone(), batch_size, start, end);
            let loss_mask = batch
                .loss_mask
                .clone()
                .map(|mask| Self::slice_tokens(mask, batch_size, start, end));
            let (loss, supervised_tokens) = if start == 0 {
                let chunk = local_predictive_coding::prepare_dkp_predictive_coding_chunk(
                    &self.model,
                    inputs,
                    targets,
                    loss_mask,
                    metric_state,
                    feedback.clone(),
                    feedback_updates,
                    &self.local_predictive_coding,
                );
                metric_state = chunk.terminal_state.clone();
                if feedback.is_none() {
                    feedback = Some(Tensor::<B, 3>::from_inner(chunk.feedback.clone()));
                }
                let loss = chunk.loss.clone();
                let supervised_tokens = chunk.supervised_tokens.clone();
                first_chunk = Some(chunk);
                (loss, supervised_tokens)
            } else {
                let observation = local_predictive_coding::observe_dkp_predictive_coding_chunk(
                    &self.model,
                    inputs,
                    targets,
                    loss_mask,
                    metric_state,
                );
                metric_state = observation.terminal_state;
                (observation.loss, observation.supervised_tokens)
            };
            let weighted_loss = loss * supervised_tokens.clone();
            total_loss = Some(match total_loss {
                Some(accumulated) => accumulated + weighted_loss,
                None => weighted_loss,
            });
            total_supervised_tokens = Some(match total_supervised_tokens {
                Some(accumulated) => accumulated + supervised_tokens,
                None => supervised_tokens,
            });
        }

        let pending = DkpPredictiveCodingPendingBatch {
            batch,
            initial_state,
            first_chunk: first_chunk.expect("DKP requires at least one chunk"),
        };
        let mut runtime = self
            .dkp_predictive_coding_runtime
            .inner
            .lock()
            .expect("DKP runtime lock poisoned");
        assert!(
            runtime.is_none(),
            "DKP batch was not consumed by the optimizer"
        );
        *runtime = Some(Box::new(pending));
        drop(runtime);
        if let Some(started) = profile_started {
            crate::train::profile::record_local_learning(started.elapsed().as_nanos());
        }

        let supervised_tokens = total_supervised_tokens
            .expect("DKP requires at least one chunk")
            .clamp_min(1.0);
        TrainOutput {
            grads: GradientsParams::new(),
            item: LanguageModelTrainItem::new(
                total_loss.expect("DKP requires at least one chunk") / supervised_tokens,
            ),
        }
    }

    pub(super) fn optimize_dkp_predictive_coding<B2, O>(mut self, optim: &mut O, lr: f64) -> Self
    where
        B2: AutodiffBackend,
        O: Optimizer<Self, B2>,
        Self: AutodiffModule<B2>,
        B::Device: 'static,
        B::FloatTensorPrimitive: 'static,
    {
        let pending = self
            .dkp_predictive_coding_runtime
            .inner
            .lock()
            .expect("DKP runtime lock poisoned")
            .take()
            .expect("DKP optimizer step has no staged batch")
            .downcast::<DkpPredictiveCodingPendingBatch<B>>()
            .unwrap_or_else(|_| panic!("DKP backend type mismatch"));
        let DkpPredictiveCodingPendingBatch {
            batch,
            initial_state,
            first_chunk,
        } = *pending;
        let config = self.local_predictive_coding.clone();
        let [batch_size, block_size] = batch.inputs.shape().dims::<2>();
        let chunk_size = self
            .effective_tbptt_chunk_size(block_size)
            .unwrap_or(block_size);
        let mut state = initial_state;
        let mut first_chunk = Some(first_chunk);
        let preliminary_lr = lr * config.direct_feedback.preliminary_step_size as f64;
        let mut elapsed_ns = 0u128;

        for start in (0..block_size).step_by(chunk_size) {
            let started = Instant::now();
            let end = (start + chunk_size).min(block_size);
            let mut chunk = if start == 0 {
                first_chunk.take().expect("staged first DKP chunk")
            } else {
                let feedback_state = self.dkp_feedback_for_checkpoint();
                local_predictive_coding::prepare_dkp_predictive_coding_chunk(
                    &self.model,
                    Self::slice_tokens(batch.inputs.clone(), batch_size, start, end),
                    Self::slice_tokens(batch.targets.clone(), batch_size, start, end),
                    batch
                        .loss_mask
                        .clone()
                        .map(|mask| Self::slice_tokens(mask, batch_size, start, end)),
                    state,
                    feedback_state
                        .as_ref()
                        .map(|feedback| feedback.feedback.clone()),
                    feedback_state.map_or(0, |feedback| feedback.updates),
                    &config,
                )
            };
            state = chunk.terminal_state.clone();
            let preliminary_grads = self.apply_gradient_scale_schedule(std::mem::replace(
                &mut chunk.preliminary_grads,
                GradientsParams::new(),
            ));
            self = optim.step(preliminary_lr, self, preliminary_grads);
            self.local_predictive_coding_profile
                .record_optimizer_updates(1);

            let mut derivatives = local_predictive_coding::finish_dkp_predictive_coding_chunk(
                &self.model,
                chunk,
                &config,
                &self.local_predictive_coding_profile,
            );
            if let Some(feedback) = derivatives.dkp_feedback.take() {
                self.store_dkp_feedback(feedback);
            }
            let grads = self.apply_gradient_scale_schedule(derivatives.grads);
            self = optim.step(lr, self, grads);
            self.local_predictive_coding_profile
                .record_optimizer_updates(1);
            elapsed_ns = elapsed_ns.saturating_add(started.elapsed().as_nanos());
        }
        if crate::train::profile::enabled() {
            crate::train::profile::record_local_learning_step(elapsed_ns);
        }
        self.store_step_state(state);
        self
    }

    pub(super) fn uses_two_phase_dkp_predictive_coding(&self) -> bool {
        matches!(self.training_algorithm, TrainingAlgorithm::PredictiveCoding)
            && matches!(
                self.local_predictive_coding.solver,
                LocalPredictiveCodingSolver::DirectKolenPollack
            )
    }

    pub(crate) fn uses_amortized_adjoint_predictive_coding(&self) -> bool {
        matches!(self.training_algorithm, TrainingAlgorithm::PredictiveCoding)
            && matches!(
                self.local_predictive_coding.solver,
                LocalPredictiveCodingSolver::AmortizedAdjoint
            )
    }

    pub(crate) fn uses_parallel_adjoint_predictive_coding(&self) -> bool {
        matches!(self.training_algorithm, TrainingAlgorithm::PredictiveCoding)
            && matches!(
                self.local_predictive_coding.terminal_criterion,
                crate::config::LocalPredictiveCodingTerminalCriterion::NextToken
            )
            && matches!(
                self.local_predictive_coding.solver,
                LocalPredictiveCodingSolver::AmortizedAdjoint
                    | LocalPredictiveCodingSolver::FirstOrderAdjoint
            )
    }

    pub(crate) fn uses_local_pc_feedback_state(&self) -> bool {
        self.uses_two_phase_dkp_predictive_coding()
            || self.uses_amortized_adjoint_predictive_coding()
    }

    pub(crate) fn uses_incremental_predictive_coding(&self) -> bool {
        matches!(self.training_algorithm, TrainingAlgorithm::PredictiveCoding)
            && matches!(
                self.local_predictive_coding.learning_schedule,
                burn_pc::PcLearningSchedule::Incremental
            )
    }

    pub(crate) fn predictive_context_probe_loss(
        &self,
        batch: &SequenceBatch<B>,
        neuron_mask: Tensor<B, 4>,
        activity_mask: Tensor<B, 4>,
        probe_tokens: usize,
    ) -> Tensor<B::InnerBackend, 1>
    where
        B::Device: 'static,
        B::FloatTensorPrimitive: 'static,
    {
        let [batch_size, block_size] = batch.inputs.shape().dims();
        let time = probe_tokens.min(block_size).max(1);
        let inputs = Self::slice_tokens(batch.inputs.clone(), batch_size, 0, time).inner();
        let targets = Self::slice_tokens(batch.targets.clone(), batch_size, 0, time).inner();
        let loss_mask = batch
            .loss_mask
            .clone()
            .map(|mask| Self::slice_tokens(mask, batch_size, 0, time).inner());
        let plain = self.model.valid();
        let logits = plain
            .predictive_coding_forward_with_subnetwork_masks(
                inputs,
                neuron_mask.inner(),
                activity_mask.inner(),
            )
            .expect("validated predictive context masks");
        burn_dragon_core::objective::masked_token_mean(
            plain.language_token_losses_from_logits(logits, targets),
            loss_mask,
        )
    }

    pub(crate) fn predictive_context_train_step(
        &self,
        batch: SequenceBatch<B>,
        neuron_mask: Tensor<B, 4>,
        activity_mask: Tensor<B, 4>,
        initial_state: Option<ModelState<B>>,
    ) -> PredictiveContextTrainStep<B>
    where
        B::Device: 'static,
        B::FloatTensorPrimitive: 'static,
    {
        let step_index = self.gradient_scale_step.load(Ordering::Relaxed);
        B::seed(
            &batch.inputs.device(),
            stochastic_step_seed(self.stochastic_seed, step_index, STOCHASTIC_STREAM_MAIN),
        );
        let [batch_size, block_size] = batch.inputs.shape().dims::<2>();
        let chunk_size = self.effective_tbptt_chunk_size(block_size);
        if chunk_size.is_none() {
            let initial_state = if self.tbptt_persist_across_steps {
                Some(initial_state.unwrap_or_else(|| self.model.init_state()))
            } else {
                initial_state
            };
            let step = local_predictive_coding::local_predictive_coding_train_step_with_state_and_context_masks(
                &self.model,
                batch.inputs,
                batch.targets,
                batch.loss_mask,
                initial_state,
                local_predictive_coding::LocalPredictiveCodingContextMasks {
                    neuron: Some(neuron_mask),
                    activity: Some(activity_mask),
                },
                &self.local_predictive_coding,
                &self.local_predictive_coding_profile,
            );
            debug_assert_eq!(step.report.global_backward_calls, 0);
            if crate::train::profile::enabled() {
                crate::train::profile::record_local_learning_step(step.report.elapsed_ns);
            }
            return PredictiveContextTrainStep {
                output: TrainOutput {
                    grads: self.apply_gradient_scale_schedule(step.grads),
                    item: LanguageModelTrainItem::new(step.loss),
                },
                terminal_state: self
                    .tbptt_persist_across_steps
                    .then_some(step.terminal_state),
            };
        }

        let mut state = initial_state.unwrap_or_else(|| self.model.init_state());
        let mut accumulator = GradientsAccumulator::new();
        let mut total_loss: Option<Tensor<B, 1>> = None;
        let mut total_supervised_tokens: Option<Tensor<B, 1>> = None;
        let mut total_elapsed_ns = 0u128;
        let chunk_size = chunk_size.expect("checked predictive context chunk size");
        for start in (0..block_size).step_by(chunk_size) {
            let end = (start + chunk_size).min(block_size);
            let chunk_inputs = Self::slice_tokens(batch.inputs.clone(), batch_size, start, end);
            let chunk_targets = Self::slice_tokens(batch.targets.clone(), batch_size, start, end);
            let chunk_loss_mask = batch
                .loss_mask
                .clone()
                .map(|mask| Self::slice_tokens(mask, batch_size, start, end));
            let mut step = local_predictive_coding::local_predictive_coding_train_step_with_state_and_context_masks(
                &self.model,
                chunk_inputs,
                chunk_targets,
                chunk_loss_mask,
                Some(state),
                local_predictive_coding::LocalPredictiveCodingContextMasks {
                    neuron: Some(neuron_mask.clone()),
                    activity: Some(activity_mask.clone()),
                },
                &self.local_predictive_coding,
                &self.local_predictive_coding_profile,
            );
            debug_assert_eq!(step.report.global_backward_calls, 0);
            state = step.terminal_state;
            let supervised_tokens = step.supervised_tokens;
            rescale_gradients_by_device_scalar::<B, _>(
                self,
                &mut step.grads,
                supervised_tokens.clone().inner(),
                false,
            );
            accumulator.accumulate(self, step.grads);
            let weighted_loss = step.loss * supervised_tokens.clone();
            total_loss = Some(match total_loss {
                Some(accumulated) => accumulated + weighted_loss,
                None => weighted_loss,
            });
            total_supervised_tokens = Some(match total_supervised_tokens {
                Some(accumulated) => accumulated + supervised_tokens,
                None => supervised_tokens,
            });
            total_elapsed_ns = total_elapsed_ns.saturating_add(step.report.elapsed_ns);
        }
        if crate::train::profile::enabled() {
            crate::train::profile::record_local_learning_step(total_elapsed_ns);
        }
        let supervised_tokens = total_supervised_tokens
            .expect("predictive context TBPTT requires at least one chunk")
            .clamp_min(1.0);
        let mut grads = accumulator.grads();
        rescale_gradients_by_device_scalar::<B, _>(
            self,
            &mut grads,
            supervised_tokens.clone().inner(),
            true,
        );
        let loss = total_loss.expect("predictive context TBPTT requires at least one chunk")
            / supervised_tokens;
        PredictiveContextTrainStep {
            output: TrainOutput {
                grads: self.apply_gradient_scale_schedule(grads),
                item: LanguageModelTrainItem::new(loss),
            },
            terminal_state: self.tbptt_persist_across_steps.then_some(state),
        }
    }
}
