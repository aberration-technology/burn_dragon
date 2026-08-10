//! Model construction, runtime state, validation probes, and shared loss helpers.

use super::*;
use crate::train::local_predictive_coding;

impl<B: BackendTrait> LanguageTrainModel<B> {
    pub fn new(model: DragonModel<B>) -> Self {
        Self {
            input_vocab_size: model.vocab_size(),
            model,
            tbptt_chunk_size: None,
            tbptt_credit_window_chunks: 1,
            pipeline_plan: None,
            tbptt_persist_across_steps: false,
            retain_ephemeral_terminal_sequence_state: false,
            objective: TrainingObjectiveConfig::NextToken,
            input_corruption: CausalInputCorruptionConfig::default(),
            logit_entropy_floor: LogitEntropyFloorConfig::default(),
            repeat_unlikelihood: RepeatUnlikelihoodConfig::default(),
            greedy_rollout_unlikelihood: GreedyRolloutUnlikelihoodConfig::default(),
            dynamics_anchor: DynamicsAnchorConfig::default(),
            predictive_coding: PredictiveCodingConfig::default(),
            training_algorithm: TrainingAlgorithm::Auto,
            local_predictive_coding: LocalPredictiveCodingConfig::default(),
            local_predictive_coding_profile:
                local_predictive_coding::LocalPredictiveCodingProfile::default(),
            incremental_predictive_coding_runtime: PipelineRuntimeCell::default(),
            dkp_predictive_coding_runtime: PipelineRuntimeCell::default(),
            dkp_feedback_bank: PipelineRuntimeCell::default(),
            latent_reasoning: LatentReasoningTrainingConfig::default(),
            ruliad_supervision: RuliadSupervisionConfig::default(),
            latent_reasoning_capability_gate_open: Arc::new(AtomicBool::new(false)),
            greedy_rollout_recovery_active: Arc::new(AtomicBool::new(false)),
            teacher_model: None,
            teacher_runtime: PipelineRuntimeCell::default(),
            streaming_state: PipelineRuntimeCell::default(),
            gradient_scale_schedule: GradientScaleSchedule::default(),
            gradient_scale_step: Arc::new(AtomicUsize::new(0)),
            stochastic_seed: 0,
            ruliad_policy_telemetry_path: None,
            ruliad_structured_recovery_telemetry_path: None,
            ruliad_answer_contract_telemetry_path: None,
            ruliad_structured_contrast_telemetry_path: None,
            ruliad_field_binding_contrast_telemetry_path: None,
            ruliad_field_binding_replay: Arc::new(Mutex::new(VecDeque::new())),
            ruliad_generated_attractor_replay: Arc::new(Mutex::new(
                RuliadGeneratedAttractorReplay::default(),
            )),
            ruliad_generated_attractor_telemetry_path: None,
            ruliad_verifier_rollout_telemetry_path: None,
            ruliad_proof_policy_telemetry_path: None,
        }
    }

    pub fn with_tbptt_chunk_size(mut self, tbptt_chunk_size: Option<usize>) -> Self {
        self.tbptt_chunk_size = tbptt_chunk_size;
        self
    }

    pub fn with_tbptt_credit_window_chunks(mut self, window_chunks: usize) -> Self {
        assert!(window_chunks > 0, "TBPTT credit window must be positive");
        self.tbptt_credit_window_chunks = window_chunks;
        self
    }

    pub(crate) fn map_model(mut self, f: impl FnOnce(DragonModel<B>) -> DragonModel<B>) -> Self {
        self.model = f(self.model);
        self
    }

    pub(crate) fn materialize_random_scaffold_for_inference(mut self) -> Self {
        self.model = self.model.materialize_random_scaffold_for_inference();
        self.teacher_model = self
            .teacher_model
            .map(DragonModel::materialize_random_scaffold_for_inference);
        self
    }

    pub fn with_pipeline_plan(mut self, pipeline_plan: Option<PipelinePlan>) -> Self {
        self.pipeline_plan = pipeline_plan;
        self
    }

    pub fn with_tbptt_persist_across_steps(mut self, enabled: bool) -> Self {
        self.tbptt_persist_across_steps = enabled;
        self
    }

    pub fn with_ephemeral_terminal_sequence_state_retention(mut self, retain: bool) -> Self {
        self.retain_ephemeral_terminal_sequence_state = retain;
        self
    }

    pub fn with_training_objective(mut self, objective: TrainingObjectiveConfig) -> Self {
        self.teacher_model =
            (!objective.is_next_token()).then(|| detach_teacher_model(&self.model));
        *self
            .teacher_runtime
            .inner
            .lock()
            .expect("teacher model runtime lock poisoned") = self
            .teacher_model
            .clone()
            .map(|model| Box::new(TeacherModelRuntime::new(model)) as Box<dyn Any + Send>);
        self.objective = objective;
        self
    }

    /// Applies the objective and auxiliary-loss portion of a language training contract.
    ///
    /// Keep this path shared by local, distributed, and peer-to-peer executors so that
    /// changing the launch mode cannot silently change what the model is optimizing.
    pub fn with_training_objectives(self, training: &TrainingHyperparameters) -> Self {
        self.with_stochastic_seed(training.seed)
            .with_training_objective(training.objective.clone())
            .with_input_corruption(training.input_corruption.clone())
            .with_logit_entropy_floor(training.logit_entropy_floor.clone())
            .with_repeat_unlikelihood(training.repeat_unlikelihood.clone())
            .with_greedy_rollout_unlikelihood(training.greedy_rollout_unlikelihood.clone())
            .with_dynamics_anchor(training.dynamics_anchor.clone())
            .with_predictive_coding(training.predictive_coding.clone())
            .with_training_algorithm(training.algorithm)
            .with_local_predictive_coding(training.local_predictive_coding.clone())
            .with_latent_reasoning(training.latent_reasoning.clone())
            .with_ruliad_supervision(training.ruliad_supervision)
    }

    pub fn with_stochastic_seed(mut self, seed: u64) -> Self {
        self.stochastic_seed = seed;
        self
    }

    /// Applies the complete launch-independent language training contract.
    pub fn with_training_configuration(
        self,
        training: &TrainingHyperparameters,
        total_steps: usize,
    ) -> Self
    where
        B: AutodiffBackend,
    {
        self.with_training_objectives(training)
            .with_tbptt_chunk_size(training.tbptt_chunk_size)
            .with_tbptt_credit_window_chunks(training.tbptt_credit_window_chunks)
            .with_tbptt_persist_across_steps(training.tbptt_persist_across_steps)
            .with_ephemeral_terminal_sequence_state_retention(
                training.retain_ephemeral_terminal_sequence_state,
            )
            .with_continual_backprop(&training.continual_backprop)
            .with_gradient_scale_schedule(training, total_steps)
    }

    pub fn with_input_corruption(mut self, config: CausalInputCorruptionConfig) -> Self {
        self.input_corruption = config;
        self
    }

    pub fn with_logit_entropy_floor(mut self, config: LogitEntropyFloorConfig) -> Self {
        self.logit_entropy_floor = config;
        self
    }

    pub fn with_repeat_unlikelihood(mut self, config: RepeatUnlikelihoodConfig) -> Self {
        self.repeat_unlikelihood = config;
        self
    }

    pub fn with_greedy_rollout_unlikelihood(
        mut self,
        config: GreedyRolloutUnlikelihoodConfig,
    ) -> Self {
        self.greedy_rollout_unlikelihood = config;
        self
    }

    pub fn with_dynamics_anchor(mut self, config: DynamicsAnchorConfig) -> Self {
        self.dynamics_anchor = config;
        if self.dynamics_anchor.enabled && self.dynamics_anchor.weight > f32::EPSILON {
            let teacher_model = self
                .teacher_model
                .clone()
                .unwrap_or_else(|| detach_teacher_model(&self.model));
            let teacher_model = detach_teacher_model(&teacher_model);
            self.teacher_model = Some(teacher_model.clone());
            let mut runtime = self
                .teacher_runtime
                .inner
                .lock()
                .expect("teacher model runtime lock poisoned");
            if runtime.is_none() {
                *runtime = Some(Box::new(TeacherModelRuntime::new(teacher_model)));
            }
        }
        self
    }

    pub fn with_predictive_coding(mut self, config: PredictiveCodingConfig) -> Self {
        self.predictive_coding = config;
        self
    }

    pub fn with_training_algorithm(mut self, algorithm: TrainingAlgorithm) -> Self {
        self.training_algorithm = algorithm;
        self
    }

    pub fn with_local_predictive_coding(mut self, config: LocalPredictiveCodingConfig) -> Self {
        self.local_predictive_coding = config;
        self
    }

    pub fn local_predictive_coding_profile(
        &self,
    ) -> local_predictive_coding::LocalPredictiveCodingProfile {
        self.local_predictive_coding_profile.clone()
    }

    pub fn with_latent_reasoning(mut self, config: LatentReasoningTrainingConfig) -> Self {
        self.latent_reasoning = config;
        if self.latent_reasoning.enabled
            && (matches!(
                self.latent_reasoning.target_encoder,
                crate::config::LatentReasoningTargetEncoder::EmaTeacher
            ) || self.latent_reasoning.dragon_state.enabled)
        {
            let teacher_model = self
                .teacher_model
                .clone()
                .unwrap_or_else(|| detach_teacher_model(&self.model));
            let teacher_model = detach_teacher_model(&teacher_model);
            self.teacher_model = Some(teacher_model.clone());
            let mut runtime = self
                .teacher_runtime
                .inner
                .lock()
                .expect("teacher model runtime lock poisoned");
            if runtime.is_none() {
                *runtime = Some(Box::new(TeacherModelRuntime::new(teacher_model)));
            }
        }
        self
    }

    pub fn with_ruliad_supervision(mut self, config: RuliadSupervisionConfig) -> Self {
        if config.verifier_reward.enabled && config.verifier_reward.kl_weight > f32::EPSILON {
            let teacher_model = detach_teacher_model(&self.model);
            self.teacher_model = Some(teacher_model.clone());
            let mut runtime = self
                .teacher_runtime
                .inner
                .lock()
                .expect("teacher model runtime lock poisoned");
            if runtime.is_none() {
                *runtime = Some(Box::new(TeacherModelRuntime::new(teacher_model)));
            }
        }
        self.ruliad_supervision = config;
        self
    }

    pub fn with_ruliad_policy_telemetry_path(mut self, path: Option<PathBuf>) -> Self {
        self.ruliad_policy_telemetry_path = path.map(Arc::new);
        self
    }

    pub fn with_ruliad_structured_recovery_telemetry_path(mut self, path: Option<PathBuf>) -> Self {
        self.ruliad_structured_recovery_telemetry_path = path.map(Arc::new);
        self
    }

    pub fn with_ruliad_answer_contract_telemetry_path(mut self, path: Option<PathBuf>) -> Self {
        self.ruliad_answer_contract_telemetry_path = path.map(Arc::new);
        self
    }

    pub fn with_ruliad_structured_contrast_telemetry_path(mut self, path: Option<PathBuf>) -> Self {
        self.ruliad_structured_contrast_telemetry_path = path.map(Arc::new);
        self
    }

    pub fn with_ruliad_field_binding_contrast_telemetry_path(
        mut self,
        path: Option<PathBuf>,
    ) -> Self {
        self.ruliad_field_binding_contrast_telemetry_path = path.map(Arc::new);
        self
    }

    pub fn with_ruliad_generated_attractor_telemetry_path(mut self, path: Option<PathBuf>) -> Self {
        self.ruliad_generated_attractor_telemetry_path = path.map(Arc::new);
        self
    }

    pub fn with_ruliad_verifier_rollout_telemetry_path(mut self, path: Option<PathBuf>) -> Self {
        self.ruliad_verifier_rollout_telemetry_path = path.map(Arc::new);
        self
    }

    pub fn with_ruliad_proof_policy_telemetry_path(mut self, path: Option<PathBuf>) -> Self {
        self.ruliad_proof_policy_telemetry_path = path.map(Arc::new);
        self
    }

    pub fn set_recovery_auxiliary_active(&self, active: bool) {
        self.greedy_rollout_recovery_active
            .store(active, Ordering::Relaxed);
    }

    pub fn set_latent_reasoning_capability_gate_open(&self, open: bool) {
        self.latent_reasoning_capability_gate_open
            .store(open, Ordering::Relaxed);
    }

    pub fn with_gradient_scale_schedule(
        mut self,
        training: &TrainingHyperparameters,
        total_steps: usize,
    ) -> Self {
        self.gradient_scale_schedule =
            GradientScaleSchedule::from_training(&self.model, training, total_steps);
        self
    }

    pub fn gradient_scale_step_index(&self) -> usize {
        self.gradient_scale_step
            .load(Ordering::Relaxed)
            .saturating_sub(1)
    }

    pub fn with_neuron_scale_stabilization(
        mut self,
        old_latent_total: usize,
        new_latent_total: usize,
        config: &NeuronScalingStabilizationConfig,
    ) -> Self {
        let start_step_index = self.gradient_scale_step_index().saturating_add(1);
        self.gradient_scale_schedule = self
            .gradient_scale_schedule
            .with_neuron_scale_stabilization(
                &self.model,
                old_latent_total,
                new_latent_total,
                start_step_index,
                config,
            );
        self
    }

    pub fn continual_backprop_target_lr_scale(&self) -> f32 {
        let step_index = self
            .gradient_scale_step
            .load(Ordering::Relaxed)
            .saturating_sub(1);
        self.gradient_scale_schedule
            .shared_lowrank_target_lr_scale_for_step_index(step_index)
    }

    pub(super) fn apply_gradient_scale_schedule(
        &self,
        mut grads: GradientsParams,
    ) -> GradientsParams
    where
        B: AutodiffBackend,
    {
        let step = self.gradient_scale_step.fetch_add(1, Ordering::Relaxed) + 1;
        let step_index = step.saturating_sub(1);
        let extra_scale = self
            .gradient_scale_schedule
            .backbone_grad_scale
            .filter(|_| step <= self.gradient_scale_schedule.backbone_grad_scale_steps);
        scale_gradients_by_schedule::<B, _>(
            self,
            &mut grads,
            self.gradient_scale_schedule.param_scale_rules.as_ref(),
            step_index,
            self.gradient_scale_schedule.backbone_param_ids.as_ref(),
            extra_scale,
            self.gradient_scale_schedule
                .neuron_scale_stabilization
                .as_ref(),
        );
        grads
    }

    pub(super) fn effective_tbptt_chunk_size(&self, block_size: usize) -> Option<usize> {
        self.tbptt_chunk_size
            .filter(|chunk_size| *chunk_size > 0 && *chunk_size < block_size)
    }

    pub(super) fn can_elide_terminal_sequence_state(&self, block_size: usize) -> bool {
        !self.retain_ephemeral_terminal_sequence_state
            && !self.tbptt_persist_across_steps
            && self.effective_tbptt_chunk_size(block_size).is_none()
            && !self.pipeline_enabled()
            && !self.predictive_coding.enabled
            && !(self.latent_reasoning.enabled
                && (self.latent_reasoning.dragon_state.enabled
                    || (self.latent_reasoning.sigreg.enabled
                        && matches!(
                            self.latent_reasoning.sigreg.target,
                            crate::config::LatentReasoningSigRegTarget::RhoMemorySlots
                                | crate::config::LatentReasoningSigRegTarget::HiddenAndRhoMemorySlots
                        ))))
            && self.model.supports_terminal_sequence_state_elision()
    }

    pub(super) fn load_step_state(
        &self,
        reset_stream_state: bool,
        block_size: usize,
    ) -> ModelState<B> {
        if !self.tbptt_persist_across_steps {
            return if self.can_elide_terminal_sequence_state(block_size) {
                self.model.init_state_stateless()
            } else {
                self.model.init_state_ephemeral()
            };
        }
        let mut runtime = self
            .streaming_state
            .inner
            .lock()
            .expect("streaming TBPTT state lock poisoned");
        if reset_stream_state {
            *runtime = None;
        }
        runtime
            .take()
            .and_then(|state| state.downcast::<ModelState<B>>().ok().map(|state| *state))
            .unwrap_or_else(|| self.model.init_state())
    }

    pub(super) fn store_step_state(&self, mut state: ModelState<B>) {
        if !self.tbptt_persist_across_steps {
            return;
        }
        state.detach_in_place();
        *self
            .streaming_state
            .inner
            .lock()
            .expect("streaming TBPTT state lock poisoned") = Some(Box::new(state));
    }

    /// Consume a scheduled auxiliary-update batch without applying its language
    /// loss. The auxiliary panel is a different sequence, so it cannot own the
    /// persistent stream state; advancing the original batch preserves the
    /// loader/model continuity contract on verifier-only update steps.
    pub(super) fn advance_stream_state_without_update(
        &self,
        inputs: Tensor<B, 2, Int>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
        reset_stream_state: bool,
    ) {
        if !self.tbptt_persist_across_steps {
            return;
        }
        let [batch_size, block_size] = inputs.shape().dims::<2>();
        let chunk_size = self
            .effective_tbptt_chunk_size(block_size)
            .unwrap_or(block_size);
        let inference_model = detach_teacher_model(&self.model);
        let mut state = self.load_step_state(reset_stream_state, block_size);
        for start in (0..block_size).step_by(chunk_size) {
            let end = (start + chunk_size).min(block_size);
            let chunk_inputs = Self::slice_tokens(inputs.clone(), batch_size, start, end);
            let chunk_summary_event_mask = summary_event_mask
                .clone()
                .map(|mask| Self::slice_tokens(mask, batch_size, start, end));
            let hidden = if let Some(mask) = chunk_summary_event_mask {
                inference_model.forward_hidden_with_state_and_summary_event_mask(
                    chunk_inputs,
                    mask,
                    &mut state,
                )
            } else {
                inference_model.forward_hidden_with_state(chunk_inputs, &mut state)
            };
            drop(hidden);
            state.detach_in_place();
        }
        self.store_step_state(state);
    }

    pub(crate) fn streaming_state_for_checkpoint(&self) -> Option<ModelState<B>> {
        self.streaming_state
            .inner
            .lock()
            .expect("streaming TBPTT state lock poisoned")
            .as_ref()
            .and_then(|state| state.downcast_ref::<ModelState<B>>().cloned())
            .map(|mut state| {
                state.detach_in_place();
                state
            })
    }

    pub(crate) fn gradient_scale_step_for_checkpoint(&self) -> usize {
        self.gradient_scale_step.load(Ordering::Relaxed)
    }

    pub(in crate::train) fn dkp_feedback_for_checkpoint(
        &self,
    ) -> Option<local_predictive_coding::DkpFeedbackState<B>> {
        self.dkp_feedback_bank
            .inner
            .lock()
            .expect("DKP feedback lock poisoned")
            .as_ref()
            .and_then(|state| {
                state
                    .downcast_ref::<local_predictive_coding::DkpFeedbackState<B>>()
                    .cloned()
            })
    }

    pub(in crate::train) fn restore_dkp_feedback_from_checkpoint(
        &self,
        state: Option<local_predictive_coding::DkpFeedbackState<B>>,
    ) {
        *self
            .dkp_feedback_bank
            .inner
            .lock()
            .expect("DKP feedback lock poisoned") =
            state.map(|state| Box::new(state) as Box<dyn Any + Send>);
    }

    pub(super) fn store_dkp_feedback(&self, feedback: Tensor<B, 3>) {
        let updates = self
            .dkp_feedback_for_checkpoint()
            .map_or(1, |state| state.updates.saturating_add(1));
        self.restore_dkp_feedback_from_checkpoint(Some(
            local_predictive_coding::DkpFeedbackState { feedback, updates },
        ));
    }

    pub(crate) fn predictive_coding_checkpoint_manifest(
        &self,
    ) -> Option<burn_pc::PcCheckpointManifest> {
        matches!(self.training_algorithm, TrainingAlgorithm::PredictiveCoding).then(|| {
            local_predictive_coding::dragon_predictive_coding_checkpoint_manifest(
                self.model.predictive_coding_layer_count(),
                &self.local_predictive_coding,
            )
            .expect("validated Dragon predictive-coding program identity")
        })
    }

    pub(crate) fn restore_gradient_scale_step_from_checkpoint(&self, step: usize) {
        self.gradient_scale_step.store(step, Ordering::Relaxed);
    }

    pub(crate) fn teacher_model_for_checkpoint(&self) -> Option<(DragonModel<B>, usize)> {
        self.teacher_model.as_ref()?;
        let runtime = self
            .teacher_runtime
            .inner
            .lock()
            .expect("teacher model runtime lock poisoned");
        let runtime = runtime
            .as_ref()
            .and_then(|runtime| runtime.downcast_ref::<TeacherModelRuntime<B>>());
        Some(runtime.map_or_else(
            || {
                (
                    self.teacher_model
                        .clone()
                        .expect("checked teacher model presence"),
                    0,
                )
            },
            |runtime| (runtime.model.clone(), runtime.update_count),
        ))
    }

    pub(crate) fn restore_teacher_model_from_checkpoint(
        &self,
        model: DragonModel<B>,
        update_count: usize,
    ) {
        *self
            .teacher_runtime
            .inner
            .lock()
            .expect("teacher model runtime lock poisoned") = Some(Box::new(TeacherModelRuntime {
            model,
            update_count,
        }));
    }

    pub(crate) fn restore_streaming_state_from_checkpoint(
        &self,
        mut state: ModelState<B>,
    ) -> Result<(), String> {
        let expected_layers = self.model.init_state().layers.len();
        if state.layers.len() != expected_layers {
            return Err(format!(
                "runtime-state checkpoint has {} layers, expected {expected_layers}",
                state.layers.len()
            ));
        }
        state.detach_in_place();
        *self
            .streaming_state
            .inner
            .lock()
            .expect("streaming TBPTT state lock poisoned") = Some(Box::new(state));
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn peek_step_state_for_test(&self) -> Option<ModelState<B>> {
        self.streaming_state
            .inner
            .lock()
            .expect("streaming TBPTT state lock poisoned")
            .as_ref()
            .and_then(|state| state.downcast_ref::<ModelState<B>>().cloned())
    }

    pub(crate) fn slice_tokens(
        tensor: Tensor<B, 2, Int>,
        batch_size: usize,
        start: usize,
        end: usize,
    ) -> Tensor<B, 2, Int> {
        tensor.slice([0..batch_size, start..end])
    }

    pub(super) fn slice_batch(
        tensor: Tensor<B, 2, Int>,
        batch_start: usize,
        batch_end: usize,
    ) -> Tensor<B, 2, Int> {
        let [_batch_size, block_size] = tensor.shape().dims();
        tensor.slice([batch_start..batch_end, 0..block_size])
    }

    pub(super) fn pipeline_enabled(&self) -> bool {
        self.pipeline_plan.is_some()
    }

    pub(super) fn language_loss_from_hidden(
        &self,
        hidden: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
        loss_mask: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 1> {
        self.language_loss_from_hidden_for_latent_step(
            hidden,
            targets,
            loss_mask,
            self.model.latent_reasoning_config().max_steps,
        )
    }

    pub(super) fn language_loss_from_hidden_for_latent_step(
        &self,
        hidden: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
        loss_mask: Option<Tensor<B, 2, Int>>,
        step: usize,
    ) -> Tensor<B, 1> {
        self.language_loss_with_supervised_tokens_from_hidden_for_latent_step(
            hidden, targets, loss_mask, step,
        )
        .0
    }

    pub(super) fn language_loss_with_supervised_tokens_from_hidden_for_latent_step(
        &self,
        hidden: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
        loss_mask: Option<Tensor<B, 2, Int>>,
        step: usize,
    ) -> (Tensor<B, 1>, Tensor<B, 1>) {
        masked_token_mean_with_count(
            self.model
                .language_token_losses_from_hidden_for_latent_step(hidden, targets, step),
            loss_mask,
        )
    }

    pub(super) fn language_loss_from_logits(
        &self,
        logits: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
        loss_mask: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 1> {
        if let Some(mask) = loss_mask {
            return masked_token_mean(
                self.model
                    .language_token_losses_from_logits(logits, targets),
                Some(mask),
            );
        }
        self.model.language_loss_from_logits(logits, targets)
    }

    pub(super) fn forward_hidden_with_pipeline_for_objective(
        &self,
        inputs: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let plan = self
            .pipeline_plan
            .as_ref()
            .expect("pipeline objective forward requires a pipeline plan");
        let [batch_size, _block_size] = inputs.shape().dims();
        let ranges = split_microbatch_ranges(batch_size, plan.microbatches)
            .expect("pipeline objective execution requires batch_size >= microbatches");
        let chunk_inputs = ranges
            .iter()
            .map(|range| Self::slice_batch(inputs.clone(), range.start, range.end))
            .collect::<Vec<_>>();

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
                    None,
                ));
        }

        let mut hidden_chunks = Vec::with_capacity(plan.microbatches);
        for microbatch_id in 0..plan.microbatches {
            hidden_chunks.push(
                self.model.finish_language_pipeline_hidden_with_state(
                    pipeline_states[microbatch_id]
                        .take()
                        .expect("pipeline state after scheduled forward"),
                    &mut chunk_states[microbatch_id],
                ),
            );
        }
        Tensor::cat(hidden_chunks, 0)
    }

    pub(super) fn forward_hidden_for_objective(&self, inputs: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        if self.pipeline_enabled() {
            self.forward_hidden_with_pipeline_for_objective(inputs)
        } else {
            self.model.forward_hidden(inputs)
        }
    }

    pub(super) fn current_teacher_model(&self) -> DragonModel<B> {
        let runtime = self
            .teacher_runtime
            .inner
            .lock()
            .expect("teacher model runtime lock poisoned");
        if let Some(runtime) = runtime
            .as_ref()
            .and_then(|runtime| runtime.downcast_ref::<TeacherModelRuntime<B>>())
        {
            return runtime.model.clone();
        }
        self.teacher_model
            .clone()
            .unwrap_or_else(|| self.model.clone())
    }

    pub(super) fn objective_teacher_update_rate(&self) -> f32 {
        let objective_rate = match &self.objective {
            TrainingObjectiveConfig::NextToken => 0.0,
            TrainingObjectiveConfig::Sdft(config) => config.teacher_update_rate,
            TrainingObjectiveConfig::Sdpo(config) => config.teacher_update_rate,
            TrainingObjectiveConfig::SdftSdpo(config) => {
                let sdft_weight = config.sdft_weight.max(0.0);
                let sdpo_weight = config.sdpo_weight.max(0.0);
                let weight_sum = sdft_weight + sdpo_weight;
                if weight_sum <= f32::EPSILON {
                    0.0
                } else {
                    (config.sdft.teacher_update_rate * sdft_weight
                        + config.sdpo.teacher_update_rate * sdpo_weight)
                        / weight_sum
                }
            }
        };
        let anchor_rate =
            if self.dynamics_anchor.enabled && self.dynamics_anchor.weight > f32::EPSILON {
                self.dynamics_anchor.teacher_update_rate.clamp(0.0, 1.0)
            } else {
                0.0
            };
        let latent_rate = if self.latent_reasoning.enabled
            && (matches!(
                self.latent_reasoning.target_encoder,
                crate::config::LatentReasoningTargetEncoder::EmaTeacher
            ) || self.latent_reasoning.dragon_state.enabled)
        {
            self.latent_reasoning.teacher_update_rate.clamp(0.0, 1.0)
        } else {
            0.0
        };
        objective_rate.max(anchor_rate).max(latent_rate)
    }

    pub(super) fn update_teacher_runtime(&self) {
        let rate = self.objective_teacher_update_rate().clamp(0.0, 1.0);
        if rate <= f32::EPSILON {
            return;
        };
        let mut runtime = self
            .teacher_runtime
            .inner
            .lock()
            .expect("teacher model runtime lock poisoned");
        if runtime.is_none() {
            *runtime = Some(Box::new(TeacherModelRuntime::new(
                self.teacher_model
                    .clone()
                    .unwrap_or_else(|| self.model.clone()),
            )));
        }
        let runtime = runtime
            .as_mut()
            .and_then(|runtime| runtime.downcast_mut::<TeacherModelRuntime<B>>())
            .expect("teacher runtime backend type must match learner backend");
        runtime.model = ema_blend_model(&runtime.model, &self.model, rate);
        runtime.update_count = runtime.update_count.saturating_add(1);
    }

    pub(crate) fn validation_loss_and_output_degeneracy(
        &self,
        batch: SequenceBatch<B>,
        probe_tokens: usize,
        eos_id: Option<i64>,
    ) -> (Tensor<B, 1>, Option<OutputDegeneracyStats>) {
        let output = <Self as ValidStep>::step(self, batch.clone());
        let loss_value: LossValue<B> = output.adapt();
        let stats = self.output_degeneracy_for_batch(batch, probe_tokens, eos_id);
        (loss_value.value(), stats)
    }

    pub(crate) fn validation_loss_and_output_degeneracy_with_subnetwork_masks(
        &self,
        batch: SequenceBatch<B>,
        neuron_mask: Tensor<B, 4>,
        activity_mask: Tensor<B, 4>,
        probe_tokens: usize,
        eos_id: Option<i64>,
    ) -> (Tensor<B, 1>, Option<OutputDegeneracyStats>)
    where
        B::Device: 'static,
        B::FloatTensorPrimitive: 'static,
    {
        let logits = self
            .model
            .predictive_coding_forward_with_subnetwork_masks(
                batch.inputs.clone(),
                neuron_mask.clone(),
                activity_mask.clone(),
            )
            .expect("validated predictive context masks");
        let loss = masked_token_mean(
            self.model
                .language_token_losses_from_logits(logits, batch.targets.clone()),
            batch.loss_mask.clone(),
        );
        let stats = self.output_degeneracy_for_batch_with_subnetwork_masks(
            batch,
            probe_tokens,
            eos_id,
            neuron_mask,
            activity_mask,
        );
        (loss, stats)
    }

    pub(crate) fn latent_reasoning_step_diagnostics(
        &self,
        batch: SequenceBatch<B>,
    ) -> Option<LatentReasoningStepDiagnostics> {
        if !self.model.latent_reasoning_enabled()
            || self.pipeline_enabled()
            || self.model.uses_factorized_language_head()
        {
            return None;
        }
        let raw = self.model.forward_hidden_raw(batch.inputs);
        let output = self.model.reason_hidden(raw.clone());
        if output.step_hiddens.is_empty() {
            return None;
        }
        let raw_loss = scalar_tensor_to_f64(self.language_loss_from_hidden_for_latent_step(
            raw.clone(),
            batch.targets.clone(),
            batch.loss_mask.clone(),
            0,
        ));
        let raw_entropy_bits = self.hidden_entropy_bits_for_latent_step(raw.clone(), 0);
        let final_loss = scalar_tensor_to_f64(self.language_loss_from_hidden_for_latent_step(
            output.final_hidden.clone(),
            batch.targets.clone(),
            batch.loss_mask.clone(),
            output.steps_used,
        ));
        let final_entropy_bits = self
            .hidden_entropy_bits_for_latent_step(output.final_hidden.clone(), output.steps_used);
        let final_delta_rms = Self::tensor_delta_rms(raw.clone(), output.final_hidden.clone());
        let final_raw_cosine = Self::tensor_cosine(raw.clone(), output.final_hidden.clone());
        let mut previous = raw.clone();
        let mut previous_ce = raw_loss;
        let mut step_loss = Vec::with_capacity(output.step_hiddens.len());
        let mut step_ce_delta = Vec::with_capacity(output.step_hiddens.len());
        let mut step_ce_monotonic_violation_rate = Vec::with_capacity(output.step_hiddens.len());
        let mut step_entropy_bits = Vec::with_capacity(output.step_hiddens.len());
        let mut step_delta_rms = Vec::with_capacity(output.step_hiddens.len());
        let mut step_raw_cosine = Vec::with_capacity(output.step_hiddens.len());
        let mut step_energy_mean = Vec::with_capacity(output.energies.len());
        let mut step_energy_delta = Vec::with_capacity(output.energies.len());
        let mut step_energy_monotonic_violation_rate = Vec::with_capacity(output.energies.len());
        let mut previous_energy = self.model.latent_energy_from_hidden(raw.clone());
        for (index, hidden) in output.step_hiddens.into_iter().enumerate() {
            let step = index.saturating_add(1);
            let loss = scalar_tensor_to_f64(self.language_loss_from_hidden_for_latent_step(
                hidden.clone(),
                batch.targets.clone(),
                batch.loss_mask.clone(),
                step,
            ));
            let ce_delta = loss - previous_ce;
            step_loss.push(loss);
            step_ce_delta.push(ce_delta);
            step_ce_monotonic_violation_rate.push(f64::from(ce_delta > 1.0e-6));
            previous_ce = loss;
            step_entropy_bits.push(self.hidden_entropy_bits_for_latent_step(hidden.clone(), step));
            step_delta_rms.push(Self::tensor_delta_rms(previous.clone(), hidden.clone()));
            step_raw_cosine.push(Self::tensor_cosine(raw.clone(), hidden.clone()));
            previous = hidden;
            if let Some(energy) = output.energies.get(index) {
                step_energy_mean.push(scalar_tensor_to_f64(energy.clone().mean().reshape([1])));
                if let Some(prev_energy) = previous_energy.as_ref() {
                    let energy_delta = energy.clone() - prev_energy.clone();
                    step_energy_delta.push(scalar_tensor_to_f64(
                        energy_delta.clone().mean().reshape([1]),
                    ));
                    let violations = energy_delta.greater_elem(0.0).float().mean().reshape([1]);
                    step_energy_monotonic_violation_rate.push(scalar_tensor_to_f64(violations));
                }
                previous_energy = Some(energy.clone());
            }
        }
        let best_energy_step = step_energy_mean
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, value)| value.is_finite())
            .min_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
            .map(|(index, _)| index.saturating_add(1));
        Some(LatentReasoningStepDiagnostics {
            raw_loss,
            final_loss,
            raw_entropy_bits,
            final_entropy_bits,
            final_delta_rms,
            final_raw_cosine,
            step_loss,
            step_ce_delta,
            step_ce_monotonic_violation_rate,
            step_entropy_bits,
            step_delta_rms,
            step_raw_cosine,
            step_energy_mean,
            step_energy_delta,
            step_energy_monotonic_violation_rate,
            best_energy_step,
        })
    }

    pub(super) fn hidden_entropy_bits(&self, hidden: Tensor<B, 3>) -> f64 {
        self.hidden_entropy_bits_for_latent_step(
            hidden,
            self.model.latent_reasoning_config().max_steps,
        )
    }

    pub(super) fn hidden_entropy_bits_for_latent_step(
        &self,
        hidden: Tensor<B, 3>,
        step: usize,
    ) -> f64 {
        let logits = self.model.logits_from_hidden_for_latent_step(hidden, step);
        let [batch, time, vocab] = logits.shape().dims::<3>();
        if batch == 0 || time == 0 || vocab == 0 {
            return 0.0;
        }
        let log_probs = activation::log_softmax(logits.reshape([batch * time, vocab]), 1);
        let entropy = (log_probs.clone().exp() * log_probs)
            .sum_dim(1)
            .mean()
            .mul_scalar(-1.0 / std::f32::consts::LN_2);
        scalar_tensor_to_f64(entropy.reshape([1]))
    }

    pub(super) fn tensor_delta_rms(lhs: Tensor<B, 3>, rhs: Tensor<B, 3>) -> f64 {
        scalar_tensor_to_f64((rhs - lhs).powf_scalar(2.0).mean().sqrt().reshape([1]))
    }

    pub(super) fn tensor_cosine(lhs: Tensor<B, 3>, rhs: Tensor<B, 3>) -> f64 {
        let dot = scalar_tensor_to_f64((lhs.clone() * rhs.clone()).mean().reshape([1]));
        let lhs_rms = scalar_tensor_to_f64(lhs.powf_scalar(2.0).mean().sqrt().reshape([1]));
        let rhs_rms = scalar_tensor_to_f64(rhs.powf_scalar(2.0).mean().sqrt().reshape([1]));
        let denom = (lhs_rms * rhs_rms).max(1.0e-12);
        dot / denom
    }

    pub(super) fn output_degeneracy_for_batch(
        &self,
        batch: SequenceBatch<B>,
        probe_tokens: usize,
        eos_id: Option<i64>,
    ) -> Option<OutputDegeneracyStats> {
        self.output_degeneracy_for_batch_impl(batch, probe_tokens, eos_id, None)
    }

    pub(super) fn output_degeneracy_for_batch_with_subnetwork_masks(
        &self,
        batch: SequenceBatch<B>,
        probe_tokens: usize,
        eos_id: Option<i64>,
        neuron_mask: Tensor<B, 4>,
        activity_mask: Tensor<B, 4>,
    ) -> Option<OutputDegeneracyStats>
    where
        B::Device: 'static,
        B::FloatTensorPrimitive: 'static,
    {
        self.output_degeneracy_for_batch_impl(
            batch,
            probe_tokens,
            eos_id,
            Some((neuron_mask, activity_mask)),
        )
    }

    pub(super) fn output_degeneracy_for_batch_impl(
        &self,
        batch: SequenceBatch<B>,
        probe_tokens: usize,
        eos_id: Option<i64>,
        context_masks: Option<(Tensor<B, 4>, Tensor<B, 4>)>,
    ) -> Option<OutputDegeneracyStats>
    where
        B::Device: 'static,
        B::FloatTensorPrimitive: 'static,
    {
        if probe_tokens == 0
            || self.pipeline_enabled()
            || self.model.uses_factorized_language_head()
        {
            return None;
        }
        let [batch_size, block_size] = batch.inputs.shape().dims::<2>();
        if batch_size == 0 || block_size == 0 {
            return None;
        }
        let probe_batch = batch_size.min(4);
        let generated_tokens = probe_tokens.max(1);
        let prompt_time = block_size.min(probe_tokens.clamp(1, 32));
        let prompt_available = block_size.saturating_sub(prompt_time);
        let device = batch.inputs.device();
        let mut accumulator = OutputDegeneracyAccumulator::new(eos_id);
        for prompt_index in 0..probe_batch {
            let prompt_start =
                validation_degeneracy_prompt_start(prompt_index, probe_batch, prompt_available);
            let inputs = batch.inputs.clone().slice([
                prompt_index..prompt_index + 1,
                prompt_start..(prompt_start + prompt_time),
            ]);
            let prompt_tokens = inputs
                .clone()
                .to_data()
                .convert::<i64>()
                .into_vec::<i64>()
                .expect("validation degeneracy prompt tokens");
            accumulator.record_prompt_tokens(prompt_tokens);
            let summary_event_mask = batch.summary_event_mask.clone().map(|mask| {
                mask.slice([
                    prompt_index..prompt_index + 1,
                    prompt_start..(prompt_start + prompt_time),
                ])
            });
            let mut state = self.model.init_state();
            let logits = if let Some((neuron_mask, activity_mask)) = context_masks.as_ref() {
                self.model
                    .predictive_coding_forward_with_subnetwork_masks_and_state(
                        inputs,
                        neuron_mask.clone(),
                        activity_mask.clone(),
                        &mut state,
                    )
                    .expect("validated predictive context masks")
            } else if let Some(mask) = summary_event_mask {
                self.model
                    .forward_with_state_and_summary_event_mask(inputs, mask, &mut state)
            } else {
                self.model.forward_with_state(inputs, &mut state)
            };
            let [_, time, vocab] = logits.shape().dims::<3>();
            if time == 0 || vocab == 0 {
                continue;
            }
            let mut last_logits = logits.slice_dim(1, (time - 1)..time).reshape([vocab]);
            for _ in 0..generated_tokens {
                let Some(step) = output_degeneracy_step_from_logits(last_logits.clone()) else {
                    continue;
                };
                accumulator.record(step);
                let next = step.argmax as i64;
                accumulator.record_generated_token(next);
                let next_tensor =
                    Tensor::<B, 2, Int>::from_data(TensorData::new(vec![next], [1, 1]), &device);
                let logits = match context_masks.as_ref() {
                    Some((neuron_mask, activity_mask)) => self
                        .model
                        .predictive_coding_forward_with_subnetwork_masks_and_state(
                            next_tensor,
                            neuron_mask.clone(),
                            activity_mask.clone(),
                            &mut state,
                        )
                        .expect("validated predictive context masks"),
                    None => self.model.forward_with_state(next_tensor, &mut state),
                };
                let [_, time, vocab] = logits.shape().dims::<3>();
                if time == 0 || vocab == 0 {
                    break;
                }
                last_logits = logits.slice_dim(1, (time - 1)..time).reshape([vocab]);
            }
        }
        Some(accumulator.finish()).filter(|stats| stats.token_count > 0)
    }

    #[cfg(test)]
    pub(super) fn teacher_update_count_for_test(&self) -> Option<usize> {
        self.teacher_runtime
            .inner
            .lock()
            .expect("teacher model runtime lock poisoned")
            .as_ref()
            .and_then(|runtime| runtime.downcast_ref::<TeacherModelRuntime<B>>())
            .map(|runtime| runtime.update_count)
    }

    pub(super) fn assert_flat_logits_for_rollout_objective(&self) {
        assert_flat_logits_for_rollout_objective(
            &self.objective,
            self.model.uses_factorized_language_head(),
        );
    }

    pub(super) fn causal_input_corruption_probability(&self) -> f32 {
        if !self.input_corruption.enabled {
            return 0.0;
        }
        let probability = self.input_corruption.probability.clamp(0.0, 1.0);
        if probability <= f32::EPSILON {
            return 0.0;
        }
        let step_index = self.gradient_scale_step.load(Ordering::Relaxed);
        if step_index < self.input_corruption.warmup_steps {
            return 0.0;
        }
        if self.input_corruption.ramp_steps == 0 {
            return probability;
        }
        let ramp_step = step_index
            .saturating_sub(self.input_corruption.warmup_steps)
            .saturating_add(1);
        let ramp = (ramp_step as f32 / self.input_corruption.ramp_steps as f32).clamp(0.0, 1.0);
        probability * ramp
    }

    pub(super) fn scheduled_weight(
        enabled: bool,
        weight: f32,
        warmup_steps: usize,
        ramp_steps: usize,
        step_index: usize,
    ) -> f32 {
        if !enabled || weight <= f32::EPSILON {
            return 0.0;
        }
        if step_index < warmup_steps {
            return 0.0;
        }
        if ramp_steps == 0 {
            return weight;
        }
        let ramp_step = step_index.saturating_sub(warmup_steps).saturating_add(1);
        let ramp = (ramp_step as f32 / ramp_steps as f32).clamp(0.0, 1.0);
        weight * ramp
    }

    pub(super) fn scheduled_weight_on_cadence(
        enabled: bool,
        weight: f32,
        warmup_steps: usize,
        ramp_steps: usize,
        every_steps: usize,
        step_index: usize,
    ) -> f32 {
        if every_steps > 1 && !step_index.is_multiple_of(every_steps) {
            return 0.0;
        }
        Self::scheduled_weight(enabled, weight, warmup_steps, ramp_steps, step_index)
    }

    pub(super) fn repeat_unlikelihood_weight(&self) -> f32 {
        Self::scheduled_weight_on_cadence(
            self.repeat_unlikelihood.enabled,
            self.repeat_unlikelihood.weight,
            self.repeat_unlikelihood.warmup_steps,
            self.repeat_unlikelihood.ramp_steps,
            self.repeat_unlikelihood.every_steps,
            self.gradient_scale_step.load(Ordering::Relaxed),
        )
    }

    pub(super) fn repeat_cycle_weight(&self) -> f32 {
        Self::scheduled_weight_on_cadence(
            self.repeat_unlikelihood.enabled,
            self.repeat_unlikelihood.cycle_weight,
            self.repeat_unlikelihood.warmup_steps,
            self.repeat_unlikelihood.ramp_steps,
            self.repeat_unlikelihood.every_steps,
            self.gradient_scale_step.load(Ordering::Relaxed),
        )
    }

    pub(super) fn repeat_cycle_margin_weight(&self) -> f32 {
        Self::scheduled_weight_on_cadence(
            self.repeat_unlikelihood.enabled,
            self.repeat_unlikelihood.cycle_margin_weight,
            self.repeat_unlikelihood.warmup_steps,
            self.repeat_unlikelihood.ramp_steps,
            self.repeat_unlikelihood.every_steps,
            self.gradient_scale_step.load(Ordering::Relaxed),
        )
    }

    pub(super) fn logit_entropy_floor_weight(&self) -> f32 {
        Self::scheduled_weight_on_cadence(
            self.logit_entropy_floor.enabled,
            self.logit_entropy_floor.weight,
            self.logit_entropy_floor.warmup_steps,
            self.logit_entropy_floor.ramp_steps,
            self.logit_entropy_floor.every_steps,
            self.gradient_scale_step.load(Ordering::Relaxed),
        )
    }

    pub(super) fn logit_marginal_entropy_floor_weight(&self) -> f32 {
        Self::scheduled_weight_on_cadence(
            self.logit_entropy_floor.enabled,
            self.logit_entropy_floor.marginal_weight,
            self.logit_entropy_floor.warmup_steps,
            self.logit_entropy_floor.ramp_steps,
            self.logit_entropy_floor.every_steps,
            self.gradient_scale_step.load(Ordering::Relaxed),
        )
    }

    pub(super) fn logit_target_coverage_weight(&self) -> f32 {
        Self::scheduled_weight_on_cadence(
            self.logit_entropy_floor.enabled,
            self.logit_entropy_floor.target_coverage_weight,
            self.logit_entropy_floor.warmup_steps,
            self.logit_entropy_floor.ramp_steps,
            self.logit_entropy_floor.every_steps,
            self.gradient_scale_step.load(Ordering::Relaxed),
        )
    }

    pub(super) fn dynamics_anchor_weight(&self) -> f32 {
        Self::scheduled_weight_on_cadence(
            self.dynamics_anchor.enabled,
            self.dynamics_anchor.weight,
            self.dynamics_anchor.warmup_steps,
            self.dynamics_anchor.ramp_steps,
            self.dynamics_anchor.every_steps,
            self.gradient_scale_step.load(Ordering::Relaxed),
        )
    }

    pub(super) fn dynamics_anchor_teacher_model(&self) -> Option<DragonModel<B>> {
        if self.dynamics_anchor_weight() <= f32::EPSILON
            || self.pipeline_enabled()
            || self.model.uses_factorized_language_head()
        {
            return None;
        }
        Some(self.current_teacher_model())
    }

    pub(super) fn latent_dragon_state_consistency_active(&self) -> bool {
        self.latent_reasoning.enabled
            && self.latent_reasoning.dragon_state.enabled
            && !self.pipeline_enabled()
    }

    pub(super) fn recurrent_teacher_model(&self) -> Option<(DragonModel<B>, bool)> {
        if let Some(teacher) = self.dynamics_anchor_teacher_model() {
            return Some((teacher, true));
        }
        self.latent_dragon_state_consistency_active()
            .then(|| (self.current_teacher_model(), false))
    }

    pub(super) fn dynamics_anchor_mask(
        &self,
        loss_mask: Option<Tensor<B, 2, Int>>,
    ) -> Option<Tensor<B, 2, Int>> {
        match self.dynamics_anchor.mask {
            DynamicsAnchorMask::AllTokens => None,
            DynamicsAnchorMask::TargetTokens => loss_mask,
            DynamicsAnchorMask::ContextTokens => loss_mask.map(|mask| mask.equal_elem(0).int()),
        }
    }

    pub(super) fn dynamics_anchor_loss_from_log_probs(
        &self,
        student_log_probs: Tensor<B, 3>,
        teacher_logits: Tensor<B, 3>,
        loss_mask: Option<Tensor<B, 2, Int>>,
    ) -> Option<Tensor<B, 1>> {
        let weight = self.dynamics_anchor_weight();
        if weight <= f32::EPSILON {
            return None;
        }
        let teacher_log_probs = log_probs_from_logits(teacher_logits.detach());
        let per_token = self_distillation_per_token_from_log_probs(
            student_log_probs,
            teacher_log_probs,
            self.dynamics_anchor.kl,
        );
        Some(masked_token_mean(per_token, self.dynamics_anchor_mask(loss_mask)).mul_scalar(weight))
    }

    pub(super) fn teacher_logits_with_state(
        teacher: &DragonModel<B>,
        inputs: Tensor<B, 2, Int>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
        state: &mut ModelState<B>,
    ) -> Tensor<B, 3> {
        if let Some(mask) = summary_event_mask {
            teacher.forward_with_state_and_summary_event_mask(inputs, mask, state)
        } else {
            teacher.forward_with_state(inputs, state)
        }
        .detach()
    }

    pub(super) fn teacher_forward_with_state(
        teacher: &DragonModel<B>,
        emit_logits: bool,
        inputs: Tensor<B, 2, Int>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
        state: &mut ModelState<B>,
    ) -> Option<Tensor<B, 3>> {
        if emit_logits {
            return Some(Self::teacher_logits_with_state(
                teacher,
                inputs,
                summary_event_mask,
                state,
            ));
        }
        if let Some(mask) = summary_event_mask {
            teacher.forward_hidden_with_state_and_summary_event_mask(inputs, mask, state);
        } else {
            teacher.forward_hidden_with_state(inputs, state);
        }
        None
    }

    pub(super) fn predictive_coding_active_for_chunk(
        &self,
        step_index: usize,
        chunk_index: usize,
        chunks_per_step: usize,
    ) -> bool {
        if !self.predictive_coding.enabled || step_index < self.predictive_coding.warmup_steps {
            return false;
        }
        predictive_coding_chunk_due(
            self.predictive_coding.observation_contract,
            step_index,
            chunk_index,
            chunks_per_step,
            self.predictive_coding.apply_every_chunks,
        )
    }

    pub(super) fn predictive_coding_inference_config(&self) -> burn_pc::PcInferenceConfig {
        self.predictive_coding.inference_config()
    }

    pub(super) fn predictive_coding_state_has_latents(
        state: &ModelState<B>,
        scope: PredictiveCodingStateScope,
    ) -> bool {
        let mut state = state.clone();
        let mut mapper = PredictiveCodingPresenceMapper::default();
        map_predictive_coding_state(&mut state, scope, &mut mapper);
        mapper.present
    }

    pub(super) fn attach_predictive_coding_state_latents(
        state: &mut ModelState<B>,
        scope: PredictiveCodingStateScope,
    ) -> bool {
        let mut mapper = PredictiveCodingAttachMapper::default();
        map_predictive_coding_state(state, scope, &mut mapper);
        mapper.attached
    }

    pub(super) fn update_predictive_coding_state_latents(
        state: &mut ModelState<B>,
        grads: &B::Gradients,
        config: &burn_pc::PcInferenceConfig,
        sync_diagnostics: bool,
        scope: PredictiveCodingStateScope,
    ) -> PredictiveCodingTensorUpdateStats
    where
        B: AutodiffBackend,
    {
        let mut mapper = PredictiveCodingUpdateMapper::<B> {
            grads,
            config,
            sync_diagnostics,
            stats: PredictiveCodingTensorUpdateStats::default(),
        };
        map_predictive_coding_state(state, scope, &mut mapper);
        mapper.stats
    }

    pub(super) fn predictive_coding_oracle_energy_with_state(
        &self,
        inference_model: &DragonModel<B>,
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
        loss_mask: Option<Tensor<B, 2, Int>>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
        state: &mut ModelState<B>,
    ) -> Tensor<B, 1> {
        let hidden = if let Some(mask) = summary_event_mask {
            inference_model.forward_hidden_with_state_and_summary_event_mask(inputs, mask, state)
        } else {
            inference_model.forward_hidden_with_state(inputs, state)
        };
        self.language_loss_from_hidden(hidden, targets, loss_mask)
    }

    pub(super) fn predictive_coding_amortization_constraint(
        &self,
        student: &ModelState<B>,
        teacher: &ModelState<B>,
    ) -> (Option<Tensor<B, 1>>, usize) {
        let mut total = None;
        let mut components = 0usize;
        let step_index = self.gradient_scale_step.load(Ordering::Relaxed);
        let constraint = PredictiveCodingAmortizationConstraint {
            sample_axis: 2,
            max_slots: self.predictive_coding.amortization_max_state_slots.max(1),
            sample_offset: stochastic_step_seed(
                self.stochastic_seed,
                step_index,
                STOCHASTIC_STREAM_PC_AMORTIZATION,
            ) as usize,
            tolerance: self.predictive_coding.amortization_tolerance.max(0.0),
            eps: self.predictive_coding.eps.max(1.0e-12),
        };
        let scope = self.predictive_coding.state_scope;
        let student = predictive_coding_state_snapshot(student, scope);
        let teacher = predictive_coding_state_snapshot(teacher, scope);
        let mut sample_indices = PredictiveCodingSampleIndexCache::new();
        debug_assert_eq!(student.rank3.len(), teacher.rank3.len());
        debug_assert_eq!(student.rank4.len(), teacher.rank4.len());
        for ((student_name, student), (teacher_name, teacher)) in
            student.rank3.iter().zip(&teacher.rank3)
        {
            debug_assert_eq!(student_name, teacher_name);
            accumulate_predictive_coding_amortization_constraint(
                &mut total,
                &mut components,
                student,
                teacher,
                constraint,
                &mut sample_indices,
            );
        }
        for ((student_name, student), (teacher_name, teacher)) in
            student.rank4.iter().zip(&teacher.rank4)
        {
            debug_assert_eq!(student_name, teacher_name);
            accumulate_predictive_coding_amortization_constraint(
                &mut total,
                &mut components,
                student,
                teacher,
                constraint,
                &mut sample_indices,
            );
        }
        (
            total.map(|loss| loss.div_scalar(components.max(1) as f32)),
            components,
        )
    }

    pub(super) fn correct_state_with_oracle_predictive_coding_using_model(
        &self,
        inference_model: &DragonModel<B>,
        state: ModelState<B>,
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
        loss_mask: Option<Tensor<B, 2, Int>>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> (ModelState<B>, PredictiveCodingChunkReport)
    where
        B: AutodiffBackend,
    {
        let start = Instant::now();
        let mut report = PredictiveCodingChunkReport {
            chunks_seen: 1,
            ..PredictiveCodingChunkReport::default()
        };
        let state_scope = self.predictive_coding.state_scope;
        if !Self::predictive_coding_state_has_latents(&state, state_scope) {
            report.skipped_empty_state = 1;
            report.elapsed_ns = start.elapsed().as_nanos();
            return (state.detached_clone(), report);
        }

        let config = self.predictive_coding_inference_config();
        let sync_diagnostics = self.predictive_coding.sync_diagnostics;
        let mut corrected = state.detached_clone();
        let mut update_stats = PredictiveCodingTensorUpdateStats::default();
        for step in 0..config.steps {
            if !Self::attach_predictive_coding_state_latents(&mut corrected, state_scope) {
                report.skipped_empty_state = report.skipped_empty_state.saturating_add(1);
                break;
            }
            let mut inference_state = corrected.clone();
            let energy = self.predictive_coding_oracle_energy_with_state(
                inference_model,
                inputs.clone(),
                targets.clone(),
                loss_mask.clone(),
                summary_event_mask.clone(),
                &mut inference_state,
            );
            if sync_diagnostics && step == 0 {
                report.energy_before = Some(scalar_tensor_to_f64(energy.clone().detach().inner()));
            }
            let grads = energy.backward();
            let step_stats = Self::update_predictive_coding_state_latents(
                &mut corrected,
                &grads,
                &config,
                sync_diagnostics,
                state_scope,
            );
            if step_stats.tensor_count == 0 {
                report.skipped_empty_state = report.skipped_empty_state.saturating_add(1);
                corrected.detach_in_place();
                break;
            }
            update_stats.tensor_count = update_stats
                .tensor_count
                .saturating_add(step_stats.tensor_count);
            update_stats.diagnostic_count = update_stats
                .diagnostic_count
                .saturating_add(step_stats.diagnostic_count);
            update_stats.grad_norm_sum += step_stats.grad_norm_sum;
            update_stats.grad_norm_max = update_stats.grad_norm_max.max(step_stats.grad_norm_max);
            update_stats.delta_rms_sum += step_stats.delta_rms_sum;
            update_stats.clip_fraction_sum += step_stats.clip_fraction_sum;
            report.inference_steps = report.inference_steps.saturating_add(1);
            corrected.detach_in_place();
        }

        if report.inference_steps > 0 {
            if sync_diagnostics {
                let mut post_state = corrected.clone();
                let post_energy = self.predictive_coding_oracle_energy_with_state(
                    inference_model,
                    inputs,
                    targets,
                    loss_mask,
                    summary_event_mask,
                    &mut post_state,
                );
                report.energy_after = Some(scalar_tensor_to_f64(post_energy.detach().inner()));
            }
            report.chunks_corrected = 1;
            report.grad_norm_mean = update_stats.grad_norm_mean();
            report.grad_norm_max = update_stats.grad_norm_max();
            report.delta_rms_mean = update_stats.delta_rms_mean();
            report.clip_fraction_mean = update_stats.clip_fraction_mean();
        }
        report.elapsed_ns = start.elapsed().as_nanos();
        (corrected, report)
    }

    pub(super) fn correct_state_with_oracle_predictive_coding(
        &self,
        state: ModelState<B>,
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
        loss_mask: Option<Tensor<B, 2, Int>>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> (ModelState<B>, PredictiveCodingChunkReport)
    where
        B: AutodiffBackend,
    {
        let inference_model = detach_teacher_model(&self.model);
        self.correct_state_with_oracle_predictive_coding_using_model(
            &inference_model,
            state,
            inputs,
            targets,
            loss_mask,
            summary_event_mask,
        )
    }

    pub(super) fn replay_observed_prefix(
        &self,
        inference_model: &DragonModel<B>,
        mut state: ModelState<B>,
        observed_inputs: Tensor<B, 2, Int>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> ModelState<B> {
        if let Some(mask) = summary_event_mask {
            inference_model.forward_hidden_with_state_and_summary_event_mask(
                observed_inputs,
                mask,
                &mut state,
            );
        } else {
            inference_model.forward_hidden_with_state(observed_inputs, &mut state);
        }
        state.detach_in_place();
        state
    }

    /// Corrects the state entering an already-observed token span, then replays
    /// that span to produce state for subsequent predictions. No next-token
    /// targets are accepted by this API.
    pub(super) fn correct_state_from_observed_prefix_using_model(
        &self,
        inference_model: &DragonModel<B>,
        state: ModelState<B>,
        observed_inputs: Tensor<B, 2, Int>,
        observed_loss_mask: Option<Tensor<B, 2, Int>>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> (ModelState<B>, PredictiveCodingChunkReport)
    where
        B: AutodiffBackend,
    {
        let start = Instant::now();
        let mut report = PredictiveCodingChunkReport {
            chunks_seen: 1,
            ..PredictiveCodingChunkReport::default()
        };
        let state_scope = self.predictive_coding.state_scope;
        if !Self::predictive_coding_state_has_latents(&state, state_scope) {
            report.skipped_empty_state = 1;
            let replayed = self.replay_observed_prefix(
                inference_model,
                state,
                observed_inputs,
                summary_event_mask,
            );
            report.elapsed_ns = start.elapsed().as_nanos();
            return (replayed, report);
        }
        let [batch_size, observed_length] = observed_inputs.shape().dims();
        if observed_length < 2 {
            report.skipped_empty_state = 1;
            let replayed = self.replay_observed_prefix(
                inference_model,
                state,
                observed_inputs,
                summary_event_mask,
            );
            report.elapsed_ns = start.elapsed().as_nanos();
            return (replayed, report);
        }

        let energy_inputs =
            Self::slice_tokens(observed_inputs.clone(), batch_size, 0, observed_length - 1);
        let energy_targets =
            Self::slice_tokens(observed_inputs.clone(), batch_size, 1, observed_length);
        let energy_loss_mask = observed_loss_mask
            .clone()
            .map(|mask| Self::slice_tokens(mask, batch_size, 0, observed_length - 1));
        let energy_summary_mask = summary_event_mask
            .clone()
            .map(|mask| Self::slice_tokens(mask, batch_size, 0, observed_length - 1));

        let config = self.predictive_coding_inference_config();
        let sync_diagnostics = self.predictive_coding.sync_diagnostics;
        let mut corrected_entry = state.detached_clone();
        let mut update_stats = PredictiveCodingTensorUpdateStats::default();
        for step in 0..config.steps {
            if !Self::attach_predictive_coding_state_latents(&mut corrected_entry, state_scope) {
                report.skipped_empty_state = report.skipped_empty_state.saturating_add(1);
                break;
            }
            let mut inference_state = corrected_entry.clone();
            let hidden = if let Some(mask) = energy_summary_mask.clone() {
                inference_model.forward_hidden_with_state_and_summary_event_mask(
                    energy_inputs.clone(),
                    mask,
                    &mut inference_state,
                )
            } else {
                inference_model
                    .forward_hidden_with_state(energy_inputs.clone(), &mut inference_state)
            };
            let energy = self.language_loss_from_hidden(
                hidden,
                energy_targets.clone(),
                energy_loss_mask.clone(),
            );
            if sync_diagnostics && step == 0 {
                report.energy_before = Some(scalar_tensor_to_f64(energy.clone().detach().inner()));
            }
            let grads = energy.backward();
            let step_stats = Self::update_predictive_coding_state_latents(
                &mut corrected_entry,
                &grads,
                &config,
                sync_diagnostics,
                state_scope,
            );
            if step_stats.tensor_count == 0 {
                report.skipped_empty_state = report.skipped_empty_state.saturating_add(1);
                corrected_entry.detach_in_place();
                break;
            }
            update_stats.tensor_count = update_stats
                .tensor_count
                .saturating_add(step_stats.tensor_count);
            update_stats.diagnostic_count = update_stats
                .diagnostic_count
                .saturating_add(step_stats.diagnostic_count);
            update_stats.grad_norm_sum += step_stats.grad_norm_sum;
            update_stats.grad_norm_max = update_stats.grad_norm_max.max(step_stats.grad_norm_max);
            update_stats.delta_rms_sum += step_stats.delta_rms_sum;
            update_stats.clip_fraction_sum += step_stats.clip_fraction_sum;
            report.inference_steps = report.inference_steps.saturating_add(1);
            corrected_entry.detach_in_place();
        }

        if report.inference_steps == 0 {
            let replayed = self.replay_observed_prefix(
                inference_model,
                state,
                observed_inputs,
                summary_event_mask,
            );
            report.elapsed_ns = start.elapsed().as_nanos();
            return (replayed, report);
        }
        if sync_diagnostics {
            let mut post_state = corrected_entry.clone();
            let hidden = if let Some(mask) = energy_summary_mask {
                inference_model.forward_hidden_with_state_and_summary_event_mask(
                    energy_inputs,
                    mask,
                    &mut post_state,
                )
            } else {
                inference_model.forward_hidden_with_state(energy_inputs, &mut post_state)
            };
            let post_energy =
                self.language_loss_from_hidden(hidden, energy_targets, energy_loss_mask);
            report.energy_after = Some(scalar_tensor_to_f64(post_energy.detach().inner()));
        }

        let replayed = self.replay_observed_prefix(
            inference_model,
            corrected_entry,
            observed_inputs,
            summary_event_mask,
        );
        report.chunks_corrected = 1;
        report.grad_norm_mean = update_stats.grad_norm_mean();
        report.grad_norm_max = update_stats.grad_norm_max();
        report.delta_rms_mean = update_stats.delta_rms_mean();
        report.clip_fraction_mean = update_stats.clip_fraction_mean();
        report.elapsed_ns = start.elapsed().as_nanos();
        (replayed, report)
    }

    pub(super) fn correct_state_from_observed_prefix(
        &self,
        state: ModelState<B>,
        observed_inputs: Tensor<B, 2, Int>,
        observed_loss_mask: Option<Tensor<B, 2, Int>>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> (ModelState<B>, PredictiveCodingChunkReport)
    where
        B: AutodiffBackend,
    {
        let inference_model = detach_teacher_model(&self.model);
        self.correct_state_from_observed_prefix_using_model(
            &inference_model,
            state,
            observed_inputs,
            observed_loss_mask,
            summary_event_mask,
        )
    }

    pub(super) fn greedy_rollout_unlikelihood_weight(&self) -> f32 {
        Self::scheduled_weight(
            self.greedy_rollout_unlikelihood.enabled,
            self.greedy_rollout_unlikelihood.weight,
            self.greedy_rollout_unlikelihood.warmup_steps,
            self.greedy_rollout_unlikelihood.ramp_steps,
            self.gradient_scale_step.load(Ordering::Relaxed),
        )
    }

    pub(super) fn greedy_rollout_unlikelihood_margin_weight(&self) -> f32 {
        Self::scheduled_weight(
            self.greedy_rollout_unlikelihood.enabled,
            self.greedy_rollout_unlikelihood.margin_weight,
            self.greedy_rollout_unlikelihood.warmup_steps,
            self.greedy_rollout_unlikelihood.ramp_steps,
            self.gradient_scale_step.load(Ordering::Relaxed),
        )
    }

    pub(super) fn greedy_rollout_cycle_weight(&self) -> f32 {
        Self::scheduled_weight(
            self.greedy_rollout_unlikelihood.enabled,
            self.greedy_rollout_unlikelihood.cycle_weight,
            self.greedy_rollout_unlikelihood.warmup_steps,
            self.greedy_rollout_unlikelihood.ramp_steps,
            self.gradient_scale_step.load(Ordering::Relaxed),
        )
    }

    pub(super) fn greedy_rollout_cycle_margin_weight(&self) -> f32 {
        Self::scheduled_weight(
            self.greedy_rollout_unlikelihood.enabled,
            self.greedy_rollout_unlikelihood.cycle_margin_weight,
            self.greedy_rollout_unlikelihood.warmup_steps,
            self.greedy_rollout_unlikelihood.ramp_steps,
            self.gradient_scale_step.load(Ordering::Relaxed),
        )
    }

    pub(super) fn next_token_loss_from_logits(
        &self,
        logits: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
        clean_inputs: Tensor<B, 2, Int>,
        loss_mask: Option<Tensor<B, 2, Int>>,
        dynamics_teacher_logits: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 1> {
        self.next_token_loss_parts_from_logits(
            logits,
            targets,
            clean_inputs,
            loss_mask,
            dynamics_teacher_logits,
        )
        .total()
    }

    pub(super) fn next_token_loss_parts_from_logits(
        &self,
        logits: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
        clean_inputs: Tensor<B, 2, Int>,
        loss_mask: Option<Tensor<B, 2, Int>>,
        dynamics_teacher_logits: Option<Tensor<B, 3>>,
    ) -> NextTokenLossParts<B> {
        let [batch_size, time, vocab] = logits.shape().dims();
        let log_probs = log_probs_from_logits(logits.clone());
        let (primary, supervised_tokens) = masked_token_mean_with_count(
            selected_token_log_probs(log_probs.clone(), targets.clone()).mul_scalar(-1.0),
            loss_mask.clone(),
        );
        let mut parts = NextTokenLossParts::new(primary, supervised_tokens);
        if let Some(answer_ranking_loss) = self.ruliad_answer_ranking_loss_from_logits(
            logits.clone(),
            targets.clone(),
            loss_mask.clone(),
        ) {
            parts.add_auxiliary(answer_ranking_loss);
        }
        let weight = self.repeat_unlikelihood_weight();
        let cycle_weight = self.repeat_cycle_weight();
        let cycle_margin_weight = self.repeat_cycle_margin_weight();
        let needs_lagged_aux = weight > f32::EPSILON
            || cycle_weight > f32::EPSILON
            || cycle_margin_weight > f32::EPSILON;
        if needs_lagged_aux {
            if weight > f32::EPSILON {
                let mut total_loss: Option<Tensor<B, 1>> = None;
                let mut total_hits: Option<Tensor<B, 1>> = None;
                for lag in self.repeat_unlikelihood_lags(time) {
                    let Some((lag_log_probs, lag_targets, history_tokens)) =
                        lagged_prediction_tensors(
                            log_probs.clone(),
                            targets.clone(),
                            clean_inputs.clone(),
                            lag,
                            batch_size,
                            time,
                            vocab,
                        )
                    else {
                        continue;
                    };
                    let repeat_weight = history_tokens.clone().not_equal(lag_targets).int().float();
                    let unlikelihood = unlikelihood_from_log_probs(
                        lag_log_probs,
                        history_tokens,
                        self.repeat_unlikelihood.epsilon,
                    );
                    let lag_loss = (unlikelihood * repeat_weight.clone()).sum().reshape([1]);
                    let lag_hits = repeat_weight.sum().reshape([1]);
                    total_loss = Some(match total_loss {
                        Some(accumulated) => accumulated + lag_loss,
                        None => lag_loss,
                    });
                    total_hits = Some(match total_hits {
                        Some(accumulated) => accumulated + lag_hits,
                        None => lag_hits,
                    });
                }
                if let Some(total_loss) = total_loss {
                    parts.add_auxiliary(
                        total_loss
                            .div(
                                total_hits
                                    .expect("repeat unlikelihood hit accumulator")
                                    .clamp_min(1.0),
                            )
                            .mul_scalar(weight),
                    );
                }
            }
            if cycle_weight > f32::EPSILON || cycle_margin_weight > f32::EPSILON {
                let mut total_cycle: Option<Tensor<B, 1>> = None;
                let mut total_cycle_hits: Option<Tensor<B, 1>> = None;
                let mut total_cycle_margin: Option<Tensor<B, 1>> = None;
                let mut total_cycle_margin_hits: Option<Tensor<B, 1>> = None;
                let mean_logits_by_position = (cycle_margin_weight > f32::EPSILON)
                    .then(|| logits.clone().mean_dim(2).reshape([batch_size, time]));
                for lag in self.repeat_cycle_lags(time) {
                    let Some((lag_log_probs, lag_targets, history_tokens)) =
                        lagged_prediction_tensors(
                            log_probs.clone(),
                            targets.clone(),
                            clean_inputs.clone(),
                            lag,
                            batch_size,
                            time,
                            vocab,
                        )
                    else {
                        continue;
                    };
                    let cycle_weight_tensor =
                        history_tokens.clone().not_equal(lag_targets).int().float();
                    if cycle_weight > f32::EPSILON {
                        let unlikelihood = unlikelihood_from_log_probs(
                            lag_log_probs,
                            history_tokens.clone(),
                            self.repeat_unlikelihood.epsilon,
                        );
                        let lag_loss = (unlikelihood * cycle_weight_tensor.clone())
                            .sum()
                            .reshape([1]);
                        let lag_hits = cycle_weight_tensor.clone().sum().reshape([1]);
                        total_cycle = Some(match total_cycle {
                            Some(accumulated) => accumulated + lag_loss,
                            None => lag_loss,
                        });
                        total_cycle_hits = Some(match total_cycle_hits {
                            Some(accumulated) => accumulated + lag_hits,
                            None => lag_hits,
                        });
                    }
                    if cycle_margin_weight > f32::EPSILON {
                        let start = lag.saturating_sub(1);
                        let lag_logits =
                            logits.clone().slice([0..batch_size, start..time, 0..vocab]);
                        let history_logits =
                            selected_token_logits(lag_logits.clone(), history_tokens);
                        let mean_logits = mean_logits_by_position
                            .as_ref()
                            .expect("cycle margin mean logits")
                            .clone()
                            .slice([0..batch_size, start..time]);
                        let margin_penalty = activation::softplus(
                            history_logits - mean_logits + self.repeat_unlikelihood.cycle_margin,
                            1.0,
                        );
                        let lag_margin = (margin_penalty * cycle_weight_tensor.clone())
                            .sum()
                            .reshape([1]);
                        let lag_hits = cycle_weight_tensor.sum().reshape([1]);
                        total_cycle_margin = Some(match total_cycle_margin {
                            Some(accumulated) => accumulated + lag_margin,
                            None => lag_margin,
                        });
                        total_cycle_margin_hits = Some(match total_cycle_margin_hits {
                            Some(accumulated) => accumulated + lag_hits,
                            None => lag_hits,
                        });
                    }
                }
                if let Some(total_cycle) = total_cycle {
                    parts.add_auxiliary(
                        total_cycle
                            .div(
                                total_cycle_hits
                                    .expect("repeat cycle hit accumulator")
                                    .clamp_min(1.0),
                            )
                            .mul_scalar(cycle_weight),
                    );
                }
                if let Some(total_cycle_margin) = total_cycle_margin {
                    parts.add_auxiliary(
                        total_cycle_margin
                            .div(
                                total_cycle_margin_hits
                                    .expect("repeat cycle margin hit accumulator")
                                    .clamp_min(1.0),
                            )
                            .mul_scalar(cycle_margin_weight),
                    );
                }
            }
        }
        if let Some(entropy_floor_loss) =
            self.logit_entropy_floor_loss(log_probs.clone(), targets.clone())
        {
            parts.add_auxiliary(entropy_floor_loss);
        }
        if let Some(teacher_logits) = dynamics_teacher_logits
            && let Some(anchor_loss) =
                self.dynamics_anchor_loss_from_log_probs(log_probs, teacher_logits, loss_mask)
        {
            parts.add_auxiliary(anchor_loss);
        }
        parts
    }

    pub(super) fn repeat_unlikelihood_lags(&self, time: usize) -> Vec<usize> {
        if time == 0 {
            return Vec::new();
        }
        let mut lags = Vec::with_capacity(self.repeat_unlikelihood.history_lags.len() + 1);
        lags.push(1);
        lags.extend(self.repeat_unlikelihood.history_lags.iter().copied());
        lags.retain(|lag| (1..=time).contains(lag));
        lags.sort_unstable();
        lags.dedup();
        lags
    }

    pub(super) fn repeat_cycle_lags(&self, time: usize) -> Vec<usize> {
        if time == 0
            || self.repeat_unlikelihood.cycle_min_lag == 0
            || self.repeat_unlikelihood.cycle_max_lag < self.repeat_unlikelihood.cycle_min_lag
        {
            return Vec::new();
        }
        let min_lag = self.repeat_unlikelihood.cycle_min_lag.min(time);
        let max_lag = self.repeat_unlikelihood.cycle_max_lag.min(time);
        if max_lag < min_lag {
            return Vec::new();
        }
        let available = max_lag - min_lag + 1;
        let budget = self
            .repeat_unlikelihood
            .cycle_lags_per_step
            .max(1)
            .min(available);
        if budget == available {
            return (min_lag..=max_lag).collect();
        }
        let step_index = self.gradient_scale_step.load(Ordering::Relaxed);
        let mut lags = Vec::with_capacity(budget);
        let stride = (available / budget).max(1);
        let offset = step_index % available;
        for index in 0..budget {
            let relative = (offset + index * stride) % available;
            lags.push(min_lag + relative);
        }
        lags.sort_unstable();
        lags.dedup();
        lags
    }

    pub(super) fn next_token_loss_from_hidden(
        &self,
        hidden: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
        clean_inputs: Tensor<B, 2, Int>,
        loss_mask: Option<Tensor<B, 2, Int>>,
        dynamics_teacher_logits: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 1> {
        self.next_token_loss_parts_from_hidden(
            hidden,
            targets,
            clean_inputs,
            loss_mask,
            dynamics_teacher_logits,
        )
        .total()
    }

    pub(super) fn next_token_loss_parts_from_hidden(
        &self,
        hidden: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
        clean_inputs: Tensor<B, 2, Int>,
        loss_mask: Option<Tensor<B, 2, Int>>,
        dynamics_teacher_logits: Option<Tensor<B, 3>>,
    ) -> NextTokenLossParts<B> {
        if (self.repeat_unlikelihood_weight() <= f32::EPSILON
            && self.repeat_cycle_weight() <= f32::EPSILON
            && self.repeat_cycle_margin_weight() <= f32::EPSILON
            && self.logit_entropy_floor_weight() <= f32::EPSILON
            && self.logit_marginal_entropy_floor_weight() <= f32::EPSILON
            && self.logit_target_coverage_weight() <= f32::EPSILON
            && self.ruliad_answer_ranking_weight() <= f32::EPSILON
            && self.ruliad_answer_denoising_weight() <= f32::EPSILON
            && dynamics_teacher_logits.is_none())
            || self.model.uses_factorized_language_head()
        {
            let (primary, supervised_tokens) = self
                .language_loss_with_supervised_tokens_from_hidden_for_latent_step(
                    hidden.clone(),
                    targets.clone(),
                    loss_mask.clone(),
                    self.model.latent_reasoning_config().max_steps,
                );
            let mut parts = NextTokenLossParts::new(primary, supervised_tokens);
            if let Some(aux) = self.latent_reasoning_auxiliary_loss(
                hidden,
                clean_inputs.clone(),
                Some(targets.clone()),
                loss_mask.clone(),
            ) {
                parts.add_auxiliary(aux);
            }
            if let Some(denoising) =
                self.ruliad_answer_denoising_loss(clean_inputs, targets, loss_mask)
            {
                parts.add_auxiliary(denoising);
            }
            return parts;
        }
        let logits = self.model.logits_from_hidden(hidden.clone());
        let mut parts = self.next_token_loss_parts_from_logits(
            logits,
            targets.clone(),
            clean_inputs.clone(),
            loss_mask.clone(),
            dynamics_teacher_logits,
        );
        if let Some(aux) = self.latent_reasoning_auxiliary_loss(
            hidden,
            clean_inputs.clone(),
            Some(targets.clone()),
            loss_mask.clone(),
        ) {
            parts.add_auxiliary(aux);
        }
        if let Some(denoising) = self.ruliad_answer_denoising_loss(clean_inputs, targets, loss_mask)
        {
            parts.add_auxiliary(denoising);
        }
        parts
    }
}
