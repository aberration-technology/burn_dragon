use anyhow::{Result, anyhow};
use std::collections::HashSet;

use burn_dragon_core::{
    DragonConfig, DragonResidualScalingKind, LanguageHeadConfig, ResidualConnectorKind,
    RotaryEmbedding, SequenceMemorySystem, SequenceTrainingExecutor,
    objective::validate_training_objective_config,
};
use burn_dragon_train::{
    LearningRateScheduleConfig, OptimizerKind, ParallelismKind, PipelineCommunicationKind,
    PipelineScheduleKind, TensorParallelPartitionKind, train::pipeline::TrainingLaunchMode,
};

use super::{
    DatasetSourceConfig, PredictiveCodingBackwardMode, PredictiveCodingMode,
    PredictiveCodingObservationContract, RuliadVerifierRewardMode, SequenceBatchingMode,
    TrainingAlgorithm, TrainingConfig,
};
use crate::tokenizer::TokenizerKind;

mod execution;
mod model;
mod ruliad;
mod runtime;

#[cfg(test)]
mod tests;

impl TrainingConfig {
    pub fn resolved_training_algorithm(&self) -> TrainingAlgorithm {
        match self.training.algorithm {
            TrainingAlgorithm::Auto => match self.optimizer.name {
                OptimizerKind::Eggroll => TrainingAlgorithm::Eggroll,
                OptimizerKind::Adamw | OptimizerKind::PredictiveCoding => {
                    TrainingAlgorithm::Backpropagation
                }
            },
            explicit => explicit,
        }
    }

    pub fn validate(&self) -> Result<()> {
        self.validate_runtime_contracts()?;
        self.validate_execution_contracts()?;
        self.validate_ruliad_contracts()?;
        self.validate_dataset_model_contracts()?;
        self.validate_training_algorithm()?;
        Ok(())
    }

    fn validate_training_algorithm(&self) -> Result<()> {
        let algorithm = self.resolved_training_algorithm();
        let verifier_terminal = matches!(
            self.training.local_predictive_coding.terminal_criterion,
            crate::config::LocalPredictiveCodingTerminalCriterion::RuliadVerifierSet
        );
        if verifier_terminal {
            if !matches!(
                algorithm,
                TrainingAlgorithm::Backpropagation | TrainingAlgorithm::PredictiveCoding
            ) {
                return Err(anyhow!(
                    "ruliad_verifier_set requires training.algorithm=backpropagation or predictive_coding"
                ));
            }
            self.validate_ruliad_verifier_terminal(matches!(
                algorithm,
                TrainingAlgorithm::PredictiveCoding
            ))?;
        }
        match algorithm {
            TrainingAlgorithm::Auto => unreachable!("training algorithm must resolve"),
            TrainingAlgorithm::Backpropagation => {
                if matches!(self.optimizer.name, OptimizerKind::Eggroll) {
                    return Err(anyhow!(
                        "training.algorithm=backpropagation is incompatible with optimizer.name=eggroll"
                    ));
                }
            }
            TrainingAlgorithm::Eggroll => {
                if !matches!(self.optimizer.name, OptimizerKind::Eggroll) {
                    return Err(anyhow!(
                        "training.algorithm=eggroll requires optimizer.name=eggroll"
                    ));
                }
            }
            TrainingAlgorithm::PredictiveCoding => self.validate_local_predictive_coding()?,
        }
        Ok(())
    }

    fn validate_ruliad_verifier_terminal(&self, require_local_solver: bool) -> Result<()> {
        let pc = &self.training.local_predictive_coding;
        let policy = self.training.ruliad_supervision.proof_policy;
        let semantic_refresh = self
            .training
            .ruliad_supervision
            .proof_policy_semantic_refresh;
        if require_local_solver
            && !matches!(
                pc.solver,
                crate::config::LocalPredictiveCodingSolver::FixedPrediction
                    | crate::config::LocalPredictiveCodingSolver::ErrorEquilibrium
                    | crate::config::LocalPredictiveCodingSolver::AugmentedLagrangian
            )
        {
            return Err(anyhow!(
                "training.local_predictive_coding.terminal_criterion=ruliad_verifier_set currently requires solver=fixed_prediction, error_equilibrium, or augmented_lagrangian"
            ));
        }
        if require_local_solver && semantic_refresh.enabled {
            return Err(anyhow!(
                "predictive-coding ruliad_verifier_set does not yet support semantic-energy refresh terminals"
            ));
        }
        if !policy.enabled
            || !matches!(
                policy.presentation_risk,
                crate::config::RuliadProofPolicyPresentationRisk::Mean
            )
            || (policy.weight - 1.0).abs() > f32::EPSILON
            || policy.stratified_difficulty_levels == 0
        {
            return Err(anyhow!(
                "ruliad_verifier_set requires an enabled proof policy with mean presentation risk, weight=1, and stratified_difficulty_levels>0"
            ));
        }
        let completion_terminal = policy.scoring
            == crate::config::RuliadProofPolicyScoring::CompletionLikelihood
            && policy.gradient_scope == crate::config::RuliadProofPolicyGradientScope::FullModel
            && policy.normalization
                == crate::config::RuliadProofPolicyNormalization::PrefixConditional;
        let global_semantic_terminal = !require_local_solver
            && policy.scoring.uses_sequence_score_head()
            && matches!(
                policy.gradient_scope,
                crate::config::RuliadProofPolicyGradientScope::FullModel
                    | crate::config::RuliadProofPolicyGradientScope::ScoreHeadOnly
            )
            && policy.normalization
                == crate::config::RuliadProofPolicyNormalization::CandidateConditional;
        let exact_local_semantic_terminal = require_local_solver
            && pc.solver == crate::config::LocalPredictiveCodingSolver::FixedPrediction
            && policy.scoring == crate::config::RuliadProofPolicyScoring::SemanticEnergy
            && policy.gradient_scope == crate::config::RuliadProofPolicyGradientScope::FullModel
            && policy.normalization
                == crate::config::RuliadProofPolicyNormalization::CandidateConditional;
        if !completion_terminal && !global_semantic_terminal && !exact_local_semantic_terminal {
            return Err(anyhow!(
                "ruliad_verifier_set requires full-model prefix-conditional deployed-decoder completion likelihood; global-backprop candidate-conditional semantic/residual energy with full-model or score-head-only gradients; or fixed-prediction PC candidate-conditional semantic energy with full-model gradients"
            ));
        }
        Ok(())
    }

    fn validate_local_predictive_coding(&self) -> Result<()> {
        let pc = &self.training.local_predictive_coding;
        if self.training.tbptt_credit_window_chunks != 1 {
            return Err(anyhow!(
                "training.algorithm=predictive_coding requires training.tbptt_credit_window_chunks=1; configure local_predictive_coding.temporal_credit for local rho credit"
            ));
        }
        pc.inference
            .validate("training.local_predictive_coding.inference")?;
        pc.augmented_lagrangian
            .validate("training.local_predictive_coding.augmented_lagrangian")?;
        pc.temporal_credit
            .validate("training.local_predictive_coding.temporal_credit")?;
        pc.direct_feedback.validate()?;
        pc.amortized_adjoint.validate()?;
        pc.tied_consensus.validate()?;
        if pc.temporal_credit.carries_temporal_credit()
            && !matches!(
                pc.solver,
                crate::config::LocalPredictiveCodingSolver::FixedPrediction
            )
        {
            return Err(anyhow!(
                "training.local_predictive_coding.temporal_credit.mode=exact_window currently requires solver=fixed_prediction"
            ));
        }
        if pc.temporal_credit.carries_temporal_credit() && self.training.tbptt_chunk_size.is_none()
        {
            return Err(anyhow!(
                "training.local_predictive_coding.temporal_credit.mode=exact_window requires training.tbptt_chunk_size"
            ));
        }
        if pc.temporal_credit.carries_temporal_credit()
            && self.training.predictive_context_routing.enabled
        {
            return Err(anyhow!(
                "exact-window temporal credit does not yet compose with predictive_context_routing"
            ));
        }
        if pc.amortized_adjoint.enabled
            && !matches!(
                pc.solver,
                crate::config::LocalPredictiveCodingSolver::DirectKolenPollack
                    | crate::config::LocalPredictiveCodingSolver::AmortizedAdjoint
            )
        {
            return Err(anyhow!(
                "training.local_predictive_coding.amortized_adjoint requires a direct-feedback solver"
            ));
        }
        if matches!(
            pc.solver,
            crate::config::LocalPredictiveCodingSolver::AmortizedAdjoint
        ) && !pc.amortized_adjoint.enabled
        {
            return Err(anyhow!(
                "training.local_predictive_coding.solver=amortized_adjoint requires amortized_adjoint.enabled=true"
            ));
        }
        if matches!(
            pc.amortized_adjoint.predictor,
            burn_pc::PcAdjointPredictorKind::ResidualConditioned
        ) && !matches!(
            pc.solver,
            crate::config::LocalPredictiveCodingSolver::AmortizedAdjoint
        ) {
            return Err(anyhow!(
                "amortized_adjoint.predictor=residual_conditioned requires solver=amortized_adjoint"
            ));
        }
        if matches!(
            pc.amortized_adjoint.predictor,
            burn_pc::PcAdjointPredictorKind::ResidualConditioned
        ) && (pc.direct_feedback.signal_scale - 1.0).abs() > f32::EPSILON
        {
            return Err(anyhow!(
                "residual-conditioned adjoints require direct_feedback.signal_scale=1 so the identity credit path is preserved"
            ));
        }
        if matches!(
            pc.adjoint_conditioning,
            crate::config::LocalPredictiveCodingAdjointConditioning::TerminalDisplacement
        ) && !matches!(
            pc.amortized_adjoint.predictor,
            burn_pc::PcAdjointPredictorKind::ResidualConditioned
        ) {
            return Err(anyhow!(
                "adjoint_conditioning=terminal_displacement requires amortized_adjoint.predictor=residual_conditioned"
            ));
        }
        if pc.prediction_precision <= 0.0 || !pc.prediction_precision.is_finite() {
            return Err(anyhow!(
                "training.local_predictive_coding.prediction_precision must be finite and > 0"
            ));
        }
        if matches!(
            pc.solver,
            crate::config::LocalPredictiveCodingSolver::AugmentedLagrangian
        ) && (pc.prediction_precision - 1.0).abs() > f32::EPSILON
        {
            return Err(anyhow!(
                "solver=augmented_lagrangian requires prediction_precision=1; use augmented_lagrangian.penalty for the PC-ALM constraint penalty"
            ));
        }
        if pc.incremental_parameter_step_scale <= 0.0
            || !pc.incremental_parameter_step_scale.is_finite()
        {
            return Err(anyhow!(
                "training.local_predictive_coding.incremental_parameter_step_scale must be finite and > 0"
            ));
        }
        if matches!(
            pc.learning_schedule,
            burn_pc::PcLearningSchedule::Incremental
        ) && matches!(
            pc.solver,
            crate::config::LocalPredictiveCodingSolver::ErrorEquilibrium
                | crate::config::LocalPredictiveCodingSolver::FixedPrediction
                | crate::config::LocalPredictiveCodingSolver::LayerLocalPrediction
                | crate::config::LocalPredictiveCodingSolver::DirectKolenPollack
                | crate::config::LocalPredictiveCodingSolver::AmortizedAdjoint
                | crate::config::LocalPredictiveCodingSolver::FirstOrderAdjoint
                | crate::config::LocalPredictiveCodingSolver::AugmentedLagrangian
        ) {
            return Err(anyhow!(
                "training.local_predictive_coding.learning_schedule=incremental requires solver=synchronous_equilibrium or reverse_gauss_seidel"
            ));
        }
        if matches!(
            pc.solver,
            crate::config::LocalPredictiveCodingSolver::DirectKolenPollack
                | crate::config::LocalPredictiveCodingSolver::AmortizedAdjoint
        ) && pc.direct_feedback.forward_weight_decay > f32::EPSILON
        {
            return Err(anyhow!(
                "training.local_predictive_coding.direct_feedback.forward_weight_decay must be 0 for direct-feedback solvers because the outer optimizer owns forward-parameter decay"
            ));
        }
        if matches!(pc.parameterization, burn_pc::PcParameterizationKind::MuPc)
            && !matches!(
                pc.solver,
                crate::config::LocalPredictiveCodingSolver::ErrorEquilibrium
            )
        {
            return Err(anyhow!(
                "training.local_predictive_coding.parameterization=mu_pc currently requires solver=error_equilibrium so its shared-depth reduction cannot silently change another control"
            ));
        }
        if matches!(
            pc.learning_schedule,
            burn_pc::PcLearningSchedule::Incremental
        ) && self.training.predictive_context_routing.enabled
        {
            return Err(anyhow!(
                "incremental local predictive coding does not yet compose with training.predictive_context_routing; routed optimizers require an explicit per-context incremental schedule"
            ));
        }
        if matches!(
            pc.solver,
            crate::config::LocalPredictiveCodingSolver::DirectKolenPollack
                | crate::config::LocalPredictiveCodingSolver::AmortizedAdjoint
        ) && self.training.predictive_context_routing.enabled
        {
            return Err(anyhow!(
                "direct-feedback solvers do not yet compose with predictive_context_routing because both require optimizer-owned state"
            ));
        }
        if matches!(
            pc.solver,
            crate::config::LocalPredictiveCodingSolver::AmortizedAdjoint
                | crate::config::LocalPredictiveCodingSolver::FirstOrderAdjoint
        ) && !matches!(
            pc.factor_reduction,
            crate::config::PredictiveCodingFactorReduction::Sum
        ) {
            return Err(anyhow!(
                "parallel adjoint solvers require factor_reduction=sum so terminal credit is not depth averaged"
            ));
        }
        if matches!(
            pc.solver,
            crate::config::LocalPredictiveCodingSolver::LayerLocalPrediction
        ) && !matches!(
            pc.factor_reduction,
            crate::config::PredictiveCodingFactorReduction::Mean
        ) {
            return Err(anyhow!(
                "training.local_predictive_coding.solver=layer_local_prediction requires factor_reduction=mean so the auxiliary readout update is invariant to shared depth"
            ));
        }
        if matches!(
            pc.solver,
            crate::config::LocalPredictiveCodingSolver::LayerLocalPrediction
                | crate::config::LocalPredictiveCodingSolver::FirstOrderAdjoint
        ) && pc.sync_diagnostics
        {
            return Err(anyhow!(
                "the selected feed-forward adjoint solver does not define equilibrium-energy diagnostics; set sync_diagnostics=false"
            ));
        }
        if !matches!(self.optimizer.name, OptimizerKind::Adamw) {
            return Err(anyhow!(
                "training.algorithm=predictive_coding currently requires optimizer.name=adamw; AdamW is only the local-derivative update transform"
            ));
        }
        if self.training.predictive_coding.enabled {
            return Err(anyhow!(
                "training.algorithm=predictive_coding cannot be combined with the historical training.predictive_coding recurrent-state replay auxiliary"
            ));
        }
        if !self.training.objective.is_next_token() {
            return Err(anyhow!(
                "training.algorithm=predictive_coding currently requires the next-token objective"
            ));
        }
        if self.parallel.mode != ParallelismKind::Single || self.parallel.pipeline.enabled {
            return Err(anyhow!(
                "training.algorithm=predictive_coding currently requires local single-process execution"
            ));
        }
        if self.training.gradient_accumulation_steps != 1 {
            return Err(anyhow!(
                "training.algorithm=predictive_coding currently requires training.gradient_accumulation_steps=1"
            ));
        }
        if self.training.continual_backprop.enabled || self.training.neuron_scaling.enabled {
            return Err(anyhow!(
                "training.algorithm=predictive_coding does not yet compose with continual_backprop or neuron_scaling"
            ));
        }
        if self.training.input_corruption.enabled
            || self.training.dynamics_anchor.enabled
            || self.training.latent_reasoning.enabled
            || self.training.logit_entropy_floor.enabled
            || self.training.repeat_unlikelihood.enabled
            || self.training.greedy_rollout_unlikelihood.enabled
        {
            return Err(anyhow!(
                "training.algorithm=predictive_coding supports only local prediction factors and its configured terminal factor; input corruption and global auxiliary losses must be disabled"
            ));
        }
        let ruliad = self.training.ruliad_supervision;
        let verifier_terminal = matches!(
            pc.terminal_criterion,
            crate::config::LocalPredictiveCodingTerminalCriterion::RuliadVerifierSet
        );
        if ruliad.answer_ranking.enabled
            || ruliad.answer_denoising.enabled
            || ruliad.answer_contract.enabled
            || ruliad.verifier_reward.enabled
            || (ruliad.proof_policy.enabled && !verifier_terminal)
        {
            return Err(anyhow!(
                "training.algorithm=predictive_coding supports only the explicit ruliad_verifier_set terminal program; other Ruliad auxiliary objectives remain unsupported"
            ));
        }
        if self.training.gdpo.is_some() {
            return Err(anyhow!(
                "training.algorithm=predictive_coding does not yet support training.gdpo"
            ));
        }

        let mut model = crate::inference::build_model_config(&self.model, self.training.block_size);
        if let Some(sequence_kernel) = self.training.sequence_kernel_override {
            model.sequence_kernel = sequence_kernel;
        }
        if model.random_scaffold.enabled
            || model.dropout != 0.0
            || model.y_neuron_recurrence.enabled
            || model.hierarchical_dragon.enabled
            || model.clocked_slow_memory.enabled
            || model.summary_memory.enabled
            || model.latent_reasoning.enabled
            || model.rollout_fast_steps_per_slow_step != 1
            || model.tie_input_output_embeddings
            || !model.language_head.uses_flat_token_logits()
            || model.latent_fanout_schedule.is_some()
        {
            return Err(anyhow!(
                "training.algorithm=predictive_coding selected a model outside the current analytic local-VJP contract (including dropout=0)"
            ));
        }
        if model.resolved_residual_connector_kind() != ResidualConnectorKind::Vanilla {
            return Err(anyhow!(
                "training.algorithm=predictive_coding currently requires model.residual_connector=vanilla"
            ));
        }
        if matches!(pc.parameterization, burn_pc::PcParameterizationKind::MuPc)
            && !matches!(
                model.initialization.residual_scaling.kind,
                DragonResidualScalingKind::DepthScaled
            )
        {
            return Err(anyhow!(
                "training.local_predictive_coding.parameterization=mu_pc requires model.initialization.residual_scaling.kind=depth_scaled"
            ));
        }
        if model.sequence_kernel.memory_system != SequenceMemorySystem::LinearAttention
            || !matches!(
                model.sequence_kernel.executor,
                SequenceTrainingExecutor::Reference
                    | SequenceTrainingExecutor::DenseScoreShortContext
            )
            || model.fused_kernels.rotary_embedding != RotaryEmbedding::Alibi
        {
            return Err(anyhow!(
                "training.algorithm=predictive_coding currently requires linear attention with the reference or dense-score executor, and ALiBi"
            ));
        }
        self.validate_predictive_context_routing()?;
        Ok(())
    }

    fn validate_predictive_context_routing(&self) -> Result<()> {
        let routing = &self.training.predictive_context_routing;
        if !routing.enabled {
            return Ok(());
        }
        routing.bank.validate().map_err(anyhow::Error::msg)?;
        if routing.probe_every_steps == 0 {
            return Err(anyhow!(
                "training.predictive_context_routing.probe_every_steps must be > 0"
            ));
        }
        if routing.probe_tokens == 0 {
            return Err(anyhow!(
                "training.predictive_context_routing.probe_tokens must be > 0"
            ));
        }
        if routing.novelty_confirmations == 0 {
            return Err(anyhow!(
                "training.predictive_context_routing.novelty_confirmations must be > 0"
            ));
        }
        if !(0.0..=1.0).contains(&routing.active_fraction)
            || routing.active_fraction <= f32::EPSILON
            || !routing.active_fraction.is_finite()
        {
            return Err(anyhow!(
                "training.predictive_context_routing.active_fraction must be finite and in (0, 1]"
            ));
        }
        if !matches!(
            self.training.local_predictive_coding.solver,
            crate::config::LocalPredictiveCodingSolver::FixedPrediction
                | crate::config::LocalPredictiveCodingSolver::LayerLocalPrediction
        ) {
            return Err(anyhow!(
                "training.predictive_context_routing requires a feed-forward local solver: fixed_prediction or layer_local_prediction"
            ));
        }
        if self.training.dynamics.enabled
            || self.training.neuron_scaling.enabled
            || self.training.continual_backprop.enabled
        {
            return Err(anyhow!(
                "training.predictive_context_routing owns bounded context optimizer/state lifecycles and cannot be combined with dynamics, neuron_scaling, or continual_backprop"
            ));
        }
        if self.training.gradient_accumulation_steps != 1
            || self
                .training
                .target_effective_batch_size
                .is_some_and(|target| target > self.training.batch_size)
        {
            return Err(anyhow!(
                "training.predictive_context_routing requires one microbatch per optimizer step; set gradient_accumulation_steps=1 and target_effective_batch_size <= batch_size"
            ));
        }
        if !matches!(
            self.training.validation.sampling,
            crate::config::TrainingValidationSampling::FixedHoldout
        ) {
            return Err(anyhow!(
                "training.predictive_context_routing requires fixed_holdout validation"
            ));
        }
        if self.training.ruliad_policy_probe.enabled {
            return Err(anyhow!(
                "training.predictive_context_routing does not yet support hidden-state Ruliad policy probes; disable training.ruliad_policy_probe so validation cannot bypass the selected subnetwork"
            ));
        }
        if self.training.events.ruliad_contract_probe_enabled {
            return Err(anyhow!(
                "training.predictive_context_routing does not yet support constrained Ruliad contract decoding; disable training.events.ruliad_contract_probe_enabled so validation cannot bypass the selected subnetwork"
            ));
        }
        Ok(())
    }
}
