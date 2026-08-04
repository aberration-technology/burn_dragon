use anyhow::{Result, anyhow};
use std::collections::HashSet;

use burn_dragon_core::{
    DragonConfig, LanguageHeadConfig, ResidualConnectorKind,
    objective::validate_training_objective_config,
};
use burn_dragon_train::{
    LearningRateScheduleConfig, OptimizerKind, ParallelismKind, PipelineCommunicationKind,
    PipelineScheduleKind, TensorParallelPartitionKind, train::pipeline::TrainingLaunchMode,
};

use super::{
    DatasetSourceConfig, PredictiveCodingBackwardMode, PredictiveCodingMode,
    PredictiveCodingObservationContract, PredictiveCodingParameterUpdate, RuliadVerifierRewardMode,
    SequenceBatchingMode, TrainingConfig,
};
use crate::tokenizer::TokenizerKind;

impl TrainingConfig {
    pub fn validate(&self) -> Result<()> {
        if self.training.block_size == 0 {
            return Err(anyhow!("training.block_size must be > 0"));
        }
        if self
            .model
            .sequence_score_head
            .is_some_and(|head| head.enabled && head.projection_dim == 0)
        {
            return Err(anyhow!(
                "model.sequence_score_head.projection_dim must be > 0 when enabled"
            ));
        }
        if let Some(tbptt_chunk_size) = self.training.tbptt_chunk_size {
            if tbptt_chunk_size == 0 {
                return Err(anyhow!("training.tbptt_chunk_size must be > 0 when set"));
            }
            if tbptt_chunk_size > self.training.block_size {
                return Err(anyhow!(
                    "training.tbptt_chunk_size must be <= training.block_size (got {} > {})",
                    tbptt_chunk_size,
                    self.training.block_size
                ));
            }
        }
        if let Some(min_logical_block_size) = self.training.min_logical_block_size
            && min_logical_block_size == 0
        {
            return Err(anyhow!(
                "training.min_logical_block_size must be > 0 when set"
            ));
        }
        if self.training.tbptt_persist_across_steps && self.training.tbptt_chunk_size.is_none() {
            return Err(anyhow!(
                "training.tbptt_persist_across_steps requires training.tbptt_chunk_size"
            ));
        }
        if self.training.tbptt_persist_across_steps
            && self.training.sequence_batching == SequenceBatchingMode::Random
        {
            return Err(anyhow!(
                "training.sequence_batching=random is incompatible with training.tbptt_persist_across_steps=true"
            ));
        }
        if self.training.sequence_state_probe.enabled {
            if self.training.sequence_state_probe.paired_batches == 0 {
                return Err(anyhow!(
                    "training.sequence_state_probe.paired_batches must be > 0 when enabled"
                ));
            }
            if self.training.sequence_state_probe.max_rho_slots < 2 {
                return Err(anyhow!(
                    "training.sequence_state_probe.max_rho_slots must be >= 2 when enabled"
                ));
            }
        }
        if self.training.batch_size == 0 {
            return Err(anyhow!("training.batch_size must be > 0"));
        }
        if self.training.gradient_accumulation_steps == 0 {
            return Err(anyhow!("training.gradient_accumulation_steps must be > 0"));
        }
        if !self.training.validation.execution.is_local() {
            if self.parallel.mode != ParallelismKind::Single {
                return Err(anyhow!(
                    "training.validation.execution=external_evaluator currently requires parallel.mode=single"
                ));
            }
            if self.training.gates.enabled {
                return Err(anyhow!(
                    "training.validation.execution=external_evaluator requires training.gates.enabled=false; the external evaluator owns promotion gates"
                ));
            }
            if self.training.dynamics.enabled {
                return Err(anyhow!(
                    "training.validation.execution=external_evaluator requires training.dynamics.enabled=false; local dynamics depend on validation results"
                ));
            }
            if self.training.neuron_scaling.enabled {
                return Err(anyhow!(
                    "training.validation.execution=external_evaluator requires training.neuron_scaling.enabled=false; local scaling depends on validation results"
                ));
            }
            if self.training.events.ruliad_correctness_probe_items > 0 {
                return Err(anyhow!(
                    "training.validation.execution=external_evaluator requires training.events.ruliad_correctness_probe_items=0"
                ));
            }
            if self.training.events.source_weighted_validation_batches > 0 {
                return Err(anyhow!(
                    "training.validation.execution=external_evaluator requires training.events.source_weighted_validation_batches=0"
                ));
            }
            if self.training.ruliad_policy_probe.enabled {
                return Err(anyhow!(
                    "training.validation.execution=external_evaluator requires training.ruliad_policy_probe.enabled=false"
                ));
            }
            if self
                .training
                .latent_reasoning
                .start_after_capability_gate_passed
            {
                return Err(anyhow!(
                    "training.validation.execution=external_evaluator is incompatible with training.latent_reasoning.start_after_capability_gate_passed=true"
                ));
            }
        }
        if self.training.auto_batch_size.enabled {
            let auto_batch = &self.training.auto_batch_size;
            if auto_batch.min_batch_size == 0 {
                return Err(anyhow!(
                    "training.auto_batch_size.min_batch_size must be > 0 when enabled"
                ));
            }
            if matches!(auto_batch.max_batch_size, Some(0)) {
                return Err(anyhow!(
                    "training.auto_batch_size.max_batch_size must be > 0 when set"
                ));
            }
            if matches!(auto_batch.max_probe_batch_size, Some(0)) {
                return Err(anyhow!(
                    "training.auto_batch_size.max_probe_batch_size must be > 0 when set"
                ));
            }
            if let Some(max_batch_size) = auto_batch.max_batch_size
                && max_batch_size < auto_batch.min_batch_size
            {
                return Err(anyhow!(
                    "training.auto_batch_size.max_batch_size must be >= min_batch_size"
                ));
            }
            if let Some(max_probe_batch_size) = auto_batch.max_probe_batch_size
                && max_probe_batch_size < auto_batch.min_batch_size
            {
                return Err(anyhow!(
                    "training.auto_batch_size.max_probe_batch_size must be >= min_batch_size"
                ));
            }
            if auto_batch.probe_steps == 0 {
                return Err(anyhow!(
                    "training.auto_batch_size.probe_steps must be > 0 when enabled"
                ));
            }
            if !auto_batch.scale_memory_exponent.is_finite()
                || auto_batch.scale_memory_exponent < 0.0
            {
                return Err(anyhow!(
                    "training.auto_batch_size.scale_memory_exponent must be finite and >= 0"
                ));
            }
            if !auto_batch.max_system_memory_fraction.is_finite()
                || !(0.0..=0.9).contains(&auto_batch.max_system_memory_fraction)
                || auto_batch.max_system_memory_fraction == 0.0
            {
                return Err(anyhow!(
                    "training.auto_batch_size.max_system_memory_fraction must be finite and in (0, 0.9]"
                ));
            }
            if !auto_batch.probe_safety_margin.is_finite() || auto_batch.probe_safety_margin < 1.0 {
                return Err(anyhow!(
                    "training.auto_batch_size.probe_safety_margin must be finite and >= 1"
                ));
            }
            if self.parallel.pipeline.enabled
                && let Some(max_batch_size) = auto_batch.max_batch_size
                && max_batch_size < self.parallel.pipeline.microbatches
            {
                return Err(anyhow!(
                    "training.auto_batch_size.max_batch_size must be >= parallel.pipeline.microbatches when pipeline is enabled"
                ));
            }
        }
        if self.training.neuron_scaling.enabled {
            if self.parallel.mode != ParallelismKind::Single {
                return Err(anyhow!(
                    "training.neuron_scaling.enabled currently requires parallel.mode=single"
                ));
            }
            if self.training.neuron_scaling.max_latent_total == 0 {
                return Err(anyhow!(
                    "training.neuron_scaling.max_latent_total must be > 0"
                ));
            }
            if self.training.neuron_scaling.max_scale_events == 0 {
                return Err(anyhow!(
                    "training.neuron_scaling.max_scale_events must be > 0"
                ));
            }
            if self.training.neuron_scaling.capacity_patience_epochs == 0 {
                return Err(anyhow!(
                    "training.neuron_scaling.capacity_patience_epochs must be > 0"
                ));
            }
            if self
                .training
                .neuron_scaling
                .stabilization
                .new_slice_lr_scale
                < 0.0
                || !self
                    .training
                    .neuron_scaling
                    .stabilization
                    .new_slice_lr_scale
                    .is_finite()
            {
                return Err(anyhow!(
                    "training.neuron_scaling.stabilization.new_slice_lr_scale must be finite and >= 0"
                ));
            }
            if self
                .training
                .neuron_scaling
                .stabilization
                .base_lr_scale_after_ramp
                < 0.0
                || !self
                    .training
                    .neuron_scaling
                    .stabilization
                    .base_lr_scale_after_ramp
                    .is_finite()
            {
                return Err(anyhow!(
                    "training.neuron_scaling.stabilization.base_lr_scale_after_ramp must be finite and >= 0"
                ));
            }
        }
        if self.training.events.flush_every_steps == 0 {
            return Err(anyhow!("training.events.flush_every_steps must be > 0"));
        }
        if self.training.events.ruliad_correctness_probe_tokens == 0 {
            return Err(anyhow!(
                "training.events.ruliad_correctness_probe_tokens must be > 0"
            ));
        }
        if self.training.events.ruliad_correctness_probe_hard_token_cap
            < self.training.events.ruliad_correctness_probe_tokens
        {
            return Err(anyhow!(
                "training.events.ruliad_correctness_probe_hard_token_cap must be >= ruliad_correctness_probe_tokens"
            ));
        }
        if self.training.ruliad_probe_generation.enabled {
            let generation = self.training.ruliad_probe_generation;
            if generation.max_batch_rows == 0 {
                return Err(anyhow!(
                    "training.ruliad_probe_generation.max_batch_rows must be > 0 when enabled"
                ));
            }
            if generation.minimum_batch_rows == 0
                || generation.minimum_batch_rows > generation.max_batch_rows
            {
                return Err(anyhow!(
                    "training.ruliad_probe_generation.minimum_batch_rows must be in 1..=max_batch_rows when enabled"
                ));
            }
            if generation.maximum_prompt_position_span == 0 {
                return Err(anyhow!(
                    "training.ruliad_probe_generation.maximum_prompt_position_span must be > 0 when enabled"
                ));
            }
            if generation.device_buffer_tokens == 0 {
                return Err(anyhow!(
                    "training.ruliad_probe_generation.device_buffer_tokens must be > 0 when enabled"
                ));
            }
            if generation.max_in_flight_rows == 0 {
                return Err(anyhow!(
                    "training.ruliad_probe_generation.max_in_flight_rows must be > 0 when enabled"
                ));
            }
        }
        if self.training.ruliad_policy_probe.enabled {
            if self.training.ruliad_policy_probe.scoring
                == crate::config::RuliadProofPolicyScoring::SemanticEnergy
                && !self
                    .model
                    .sequence_score_head
                    .is_some_and(|head| head.enabled)
            {
                return Err(anyhow!(
                    "training.ruliad_policy_probe.scoring=semantic_energy requires model.sequence_score_head.enabled=true"
                ));
            }
            if self.training.ruliad_policy_probe.every_epochs == 0 {
                return Err(anyhow!(
                    "training.ruliad_policy_probe.every_epochs must be > 0 when enabled"
                ));
            }
            if self
                .training
                .ruliad_policy_probe
                .closed_loop_every_epochs
                .is_some_and(|every_epochs| every_epochs == 0)
            {
                return Err(anyhow!(
                    "training.ruliad_policy_probe.closed_loop_every_epochs must be > 0 when set"
                ));
            }
            if self.training.ruliad_policy_probe.items == 0 {
                return Err(anyhow!(
                    "training.ruliad_policy_probe.items must be > 0 when enabled"
                ));
            }
            if self.training.ruliad_policy_probe.max_steps == 0 {
                return Err(anyhow!(
                    "training.ruliad_policy_probe.max_steps must be > 0 when enabled"
                ));
            }
            if self.training.ruliad_policy_probe.candidates < 2 {
                return Err(anyhow!(
                    "training.ruliad_policy_probe.candidates must be >= 2 when enabled"
                ));
            }
            if self.training.ruliad_policy_probe.beam_width == 0 {
                return Err(anyhow!(
                    "training.ruliad_policy_probe.beam_width must be > 0 when enabled"
                ));
            }
            if self.training.ruliad_policy_probe.scoring_batch_rows == 0 {
                return Err(anyhow!(
                    "training.ruliad_policy_probe.scoring_batch_rows must be > 0 when enabled"
                ));
            }
            if self.training.ruliad_policy_probe.scoring_token_budget == 0 {
                return Err(anyhow!(
                    "training.ruliad_policy_probe.scoring_token_budget must be > 0 when enabled"
                ));
            }
            if self.training.ruliad_policy_probe.scoring_pipeline_depth == 0 {
                return Err(anyhow!(
                    "training.ruliad_policy_probe.scoring_pipeline_depth must be > 0 when enabled"
                ));
            }
            if self
                .training
                .ruliad_policy_probe
                .stratified_difficulty_levels
                > self.training.ruliad_policy_probe.items
            {
                return Err(anyhow!(
                    "training.ruliad_policy_probe.stratified_difficulty_levels must be <= training.ruliad_policy_probe.items"
                ));
            }
            let gate = self.training.ruliad_policy_probe.promotion_gate;
            if gate.enabled {
                if gate.minimum_items == 0 {
                    return Err(anyhow!(
                        "training.ruliad_policy_probe.promotion_gate.minimum_items must be > 0 when enabled"
                    ));
                }
                if gate.minimum_items > self.training.ruliad_policy_probe.items {
                    return Err(anyhow!(
                        "training.ruliad_policy_probe.promotion_gate.minimum_items must be <= training.ruliad_policy_probe.items"
                    ));
                }
                for (name, value) in [
                    ("minimum_solve_rate", gate.minimum_solve_rate),
                    (
                        "minimum_goal_completion_rate",
                        gate.minimum_goal_completion_rate,
                    ),
                    ("minimum_valid_action_rate", gate.minimum_valid_action_rate),
                    (
                        "maximum_invalid_action_rate",
                        gate.maximum_invalid_action_rate,
                    ),
                    (
                        "maximum_repeated_state_rate",
                        gate.maximum_repeated_state_rate,
                    ),
                    ("maximum_backtrack_rate", gate.maximum_backtrack_rate),
                ] {
                    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                        return Err(anyhow!(
                            "training.ruliad_policy_probe.promotion_gate.{name} must be finite and in [0, 1]"
                        ));
                    }
                }
            }
        }
        if self.training.input_corruption.enabled {
            if !(0.0..=1.0).contains(&self.training.input_corruption.probability)
                || !self.training.input_corruption.probability.is_finite()
            {
                return Err(anyhow!(
                    "training.input_corruption.probability must be finite and in [0, 1]"
                ));
            }
            if let Some(token_id) = self.training.input_corruption.replacement_token_id {
                let vocab_size = self.dataset.tokenizer.vocab_size();
                if vocab_size > 0 && token_id as usize >= vocab_size {
                    return Err(anyhow!(
                        "training.input_corruption.replacement_token_id must be < resolved vocab_size"
                    ));
                }
            }
        }
        if self.training.logit_entropy_floor.enabled {
            if self.training.logit_entropy_floor.weight < 0.0
                || !self.training.logit_entropy_floor.weight.is_finite()
            {
                return Err(anyhow!(
                    "training.logit_entropy_floor.weight must be finite and >= 0"
                ));
            }
            if self.training.logit_entropy_floor.marginal_weight < 0.0
                || !self
                    .training
                    .logit_entropy_floor
                    .marginal_weight
                    .is_finite()
            {
                return Err(anyhow!(
                    "training.logit_entropy_floor.marginal_weight must be finite and >= 0"
                ));
            }
            if self.training.logit_entropy_floor.target_coverage_weight < 0.0
                || !self
                    .training
                    .logit_entropy_floor
                    .target_coverage_weight
                    .is_finite()
            {
                return Err(anyhow!(
                    "training.logit_entropy_floor.target_coverage_weight must be finite and >= 0"
                ));
            }
            if self.training.logit_entropy_floor.target_entropy_bits < 0.0
                || !self
                    .training
                    .logit_entropy_floor
                    .target_entropy_bits
                    .is_finite()
            {
                return Err(anyhow!(
                    "training.logit_entropy_floor.target_entropy_bits must be finite and >= 0"
                ));
            }
            if self
                .training
                .logit_entropy_floor
                .target_marginal_entropy_bits
                < 0.0
                || !self
                    .training
                    .logit_entropy_floor
                    .target_marginal_entropy_bits
                    .is_finite()
            {
                return Err(anyhow!(
                    "training.logit_entropy_floor.target_marginal_entropy_bits must be finite and >= 0"
                ));
            }
            if self.training.logit_entropy_floor.target_coverage_epsilon <= 0.0
                || self.training.logit_entropy_floor.target_coverage_epsilon >= 1.0
                || !self
                    .training
                    .logit_entropy_floor
                    .target_coverage_epsilon
                    .is_finite()
            {
                return Err(anyhow!(
                    "training.logit_entropy_floor.target_coverage_epsilon must be finite and in (0, 1)"
                ));
            }
            if self.training.logit_entropy_floor.every_steps == 0 {
                return Err(anyhow!(
                    "training.logit_entropy_floor.every_steps must be > 0"
                ));
            }
        }
        if self.training.repeat_unlikelihood.enabled {
            if self.training.repeat_unlikelihood.weight < 0.0
                || !self.training.repeat_unlikelihood.weight.is_finite()
            {
                return Err(anyhow!(
                    "training.repeat_unlikelihood.weight must be finite and >= 0"
                ));
            }
            if self.training.repeat_unlikelihood.cycle_weight < 0.0
                || !self.training.repeat_unlikelihood.cycle_weight.is_finite()
            {
                return Err(anyhow!(
                    "training.repeat_unlikelihood.cycle_weight must be finite and >= 0"
                ));
            }
            if self.training.repeat_unlikelihood.cycle_margin_weight < 0.0
                || !self
                    .training
                    .repeat_unlikelihood
                    .cycle_margin_weight
                    .is_finite()
            {
                return Err(anyhow!(
                    "training.repeat_unlikelihood.cycle_margin_weight must be finite and >= 0"
                ));
            }
            if self.training.repeat_unlikelihood.cycle_margin < 0.0
                || !self.training.repeat_unlikelihood.cycle_margin.is_finite()
            {
                return Err(anyhow!(
                    "training.repeat_unlikelihood.cycle_margin must be finite and >= 0"
                ));
            }
            if self.training.repeat_unlikelihood.cycle_min_lag == 0 {
                return Err(anyhow!(
                    "training.repeat_unlikelihood.cycle_min_lag must be > 0"
                ));
            }
            if self.training.repeat_unlikelihood.cycle_max_lag
                < self.training.repeat_unlikelihood.cycle_min_lag
            {
                return Err(anyhow!(
                    "training.repeat_unlikelihood.cycle_max_lag must be >= cycle_min_lag"
                ));
            }
            if self.training.repeat_unlikelihood.cycle_lags_per_step == 0
                && (self.training.repeat_unlikelihood.cycle_weight > f32::EPSILON
                    || self.training.repeat_unlikelihood.cycle_margin_weight > f32::EPSILON)
            {
                return Err(anyhow!(
                    "training.repeat_unlikelihood.cycle_lags_per_step must be > 0 when cycle weights are enabled"
                ));
            }
            if self.training.repeat_unlikelihood.every_steps == 0 {
                return Err(anyhow!(
                    "training.repeat_unlikelihood.every_steps must be > 0"
                ));
            }
            if self.training.repeat_unlikelihood.epsilon <= 0.0
                || self.training.repeat_unlikelihood.epsilon >= 1.0
                || !self.training.repeat_unlikelihood.epsilon.is_finite()
            {
                return Err(anyhow!(
                    "training.repeat_unlikelihood.epsilon must be finite and in (0, 1)"
                ));
            }
            if self.training.repeat_unlikelihood.history_lags.contains(&0) {
                return Err(anyhow!(
                    "training.repeat_unlikelihood.history_lags must contain only positive lags"
                ));
            }
        }
        if self.training.greedy_rollout_unlikelihood.enabled {
            if self.training.greedy_rollout_unlikelihood.weight < 0.0
                || !self.training.greedy_rollout_unlikelihood.weight.is_finite()
            {
                return Err(anyhow!(
                    "training.greedy_rollout_unlikelihood.weight must be finite and >= 0"
                ));
            }
            if self.training.greedy_rollout_unlikelihood.margin_weight < 0.0
                || !self
                    .training
                    .greedy_rollout_unlikelihood
                    .margin_weight
                    .is_finite()
            {
                return Err(anyhow!(
                    "training.greedy_rollout_unlikelihood.margin_weight must be finite and >= 0"
                ));
            }
            if self.training.greedy_rollout_unlikelihood.margin < 0.0
                || !self.training.greedy_rollout_unlikelihood.margin.is_finite()
            {
                return Err(anyhow!(
                    "training.greedy_rollout_unlikelihood.margin must be finite and >= 0"
                ));
            }
            if self.training.greedy_rollout_unlikelihood.recovery_weight < 0.0
                || !self
                    .training
                    .greedy_rollout_unlikelihood
                    .recovery_weight
                    .is_finite()
            {
                return Err(anyhow!(
                    "training.greedy_rollout_unlikelihood.recovery_weight must be finite and >= 0"
                ));
            }
            if self
                .training
                .greedy_rollout_unlikelihood
                .sequence_recovery_weight
                < 0.0
                || !self
                    .training
                    .greedy_rollout_unlikelihood
                    .sequence_recovery_weight
                    .is_finite()
            {
                return Err(anyhow!(
                    "training.greedy_rollout_unlikelihood.sequence_recovery_weight must be finite and >= 0"
                ));
            }
            if self
                .training
                .greedy_rollout_unlikelihood
                .entropy_floor_weight
                < 0.0
                || !self
                    .training
                    .greedy_rollout_unlikelihood
                    .entropy_floor_weight
                    .is_finite()
            {
                return Err(anyhow!(
                    "training.greedy_rollout_unlikelihood.entropy_floor_weight must be finite and >= 0"
                ));
            }
            if self
                .training
                .greedy_rollout_unlikelihood
                .target_entropy_bits
                < 0.0
                || !self
                    .training
                    .greedy_rollout_unlikelihood
                    .target_entropy_bits
                    .is_finite()
            {
                return Err(anyhow!(
                    "training.greedy_rollout_unlikelihood.target_entropy_bits must be finite and >= 0"
                ));
            }
            if self.training.greedy_rollout_unlikelihood.cycle_weight < 0.0
                || !self
                    .training
                    .greedy_rollout_unlikelihood
                    .cycle_weight
                    .is_finite()
            {
                return Err(anyhow!(
                    "training.greedy_rollout_unlikelihood.cycle_weight must be finite and >= 0"
                ));
            }
            if self
                .training
                .greedy_rollout_unlikelihood
                .cycle_margin_weight
                < 0.0
                || !self
                    .training
                    .greedy_rollout_unlikelihood
                    .cycle_margin_weight
                    .is_finite()
            {
                return Err(anyhow!(
                    "training.greedy_rollout_unlikelihood.cycle_margin_weight must be finite and >= 0"
                ));
            }
            if self.training.greedy_rollout_unlikelihood.cycle_min_lag == 0 {
                return Err(anyhow!(
                    "training.greedy_rollout_unlikelihood.cycle_min_lag must be > 0"
                ));
            }
            if self.training.greedy_rollout_unlikelihood.cycle_max_lag
                < self.training.greedy_rollout_unlikelihood.cycle_min_lag
            {
                return Err(anyhow!(
                    "training.greedy_rollout_unlikelihood.cycle_max_lag must be >= cycle_min_lag"
                ));
            }
            if self.training.greedy_rollout_unlikelihood.every_steps == 0 {
                return Err(anyhow!(
                    "training.greedy_rollout_unlikelihood.every_steps must be > 0"
                ));
            }
            if self.training.greedy_rollout_unlikelihood.prompt_tokens == 0 {
                return Err(anyhow!(
                    "training.greedy_rollout_unlikelihood.prompt_tokens must be > 0"
                ));
            }
            if self.training.greedy_rollout_unlikelihood.rollout_tokens == 0 {
                return Err(anyhow!(
                    "training.greedy_rollout_unlikelihood.rollout_tokens must be > 0"
                ));
            }
            if self.training.greedy_rollout_unlikelihood.history_tokens == 0 {
                return Err(anyhow!(
                    "training.greedy_rollout_unlikelihood.history_tokens must be > 0"
                ));
            }
            if self.training.greedy_rollout_unlikelihood.batch_prompts == 0 {
                return Err(anyhow!(
                    "training.greedy_rollout_unlikelihood.batch_prompts must be > 0"
                ));
            }
            if self.training.greedy_rollout_unlikelihood.epsilon <= 0.0
                || self.training.greedy_rollout_unlikelihood.epsilon >= 1.0
                || !self
                    .training
                    .greedy_rollout_unlikelihood
                    .epsilon
                    .is_finite()
            {
                return Err(anyhow!(
                    "training.greedy_rollout_unlikelihood.epsilon must be finite and in (0, 1)"
                ));
            }
        }
        if self.training.dynamics_anchor.enabled {
            if self.training.dynamics_anchor.weight < 0.0
                || !self.training.dynamics_anchor.weight.is_finite()
            {
                return Err(anyhow!(
                    "training.dynamics_anchor.weight must be finite and >= 0"
                ));
            }
            if !(0.0..=1.0).contains(&self.training.dynamics_anchor.teacher_update_rate)
                || !self
                    .training
                    .dynamics_anchor
                    .teacher_update_rate
                    .is_finite()
            {
                return Err(anyhow!(
                    "training.dynamics_anchor.teacher_update_rate must be finite and in [0, 1]"
                ));
            }
            if self.training.dynamics_anchor.every_steps == 0 {
                return Err(anyhow!("training.dynamics_anchor.every_steps must be > 0"));
            }
        }
        if self.training.predictive_coding.enabled {
            let pc = &self.training.predictive_coding;
            if !matches!(pc.mode, PredictiveCodingMode::RecurrentState) {
                return Err(anyhow!(
                    "training.predictive_coding.mode currently supports only recurrent_state"
                ));
            }
            pc.inference_config()
                .validate("training.predictive_coding")?;
            if pc.apply_every_chunks == 0 {
                return Err(anyhow!(
                    "training.predictive_coding.apply_every_chunks must be > 0"
                ));
            }
            if pc.amortization_tolerance < 0.0 || !pc.amortization_tolerance.is_finite() {
                return Err(anyhow!(
                    "training.predictive_coding.amortization_tolerance must be finite and >= 0"
                ));
            }
            if pc.amortization_max_state_slots == 0 {
                return Err(anyhow!(
                    "training.predictive_coding.amortization_max_state_slots must be > 0"
                ));
            }
            if matches!(
                pc.observation_contract,
                PredictiveCodingObservationContract::OracleNextTokenNegativeControl
            ) && !pc.allow_oracle_target_leak
            {
                return Err(anyhow!(
                    "training.predictive_coding.observation_contract=oracle_next_token_negative_control requires allow_oracle_target_leak=true"
                ));
            }
            if matches!(
                pc.observation_contract,
                PredictiveCodingObservationContract::ObservedPrefix
            ) && matches!(pc.backward_mode, PredictiveCodingBackwardMode::Block)
            {
                return Err(anyhow!(
                    "training.predictive_coding.observation_contract=observed_prefix requires backward_mode=chunked so correction follows the observed chunk"
                ));
            }
            if self.training.tbptt_chunk_size.is_none() {
                return Err(anyhow!(
                    "training.predictive_coding.enabled requires training.tbptt_chunk_size"
                ));
            }
            if !self.training.objective.is_next_token() {
                return Err(anyhow!(
                    "training.predictive_coding.enabled currently requires next-token training.objective"
                ));
            }
            if self.parallel.mode != ParallelismKind::Single {
                return Err(anyhow!(
                    "training.predictive_coding.enabled currently requires parallel.mode=single"
                ));
            }
            if self.parallel.pipeline.enabled {
                return Err(anyhow!(
                    "training.predictive_coding.enabled does not support parallel.pipeline.enabled"
                ));
            }
        }
        let latent = &self.training.latent_reasoning;
        if latent.eval_step_sweep.contains(&0) {
            return Err(anyhow!(
                "training.latent_reasoning.eval_step_sweep must contain only positive step counts"
            ));
        }
        if self.training.latent_reasoning.enabled {
            if latent.every_steps == 0 {
                return Err(anyhow!(
                    "training.latent_reasoning.every_steps must be > 0 when enabled"
                ));
            }
            if latent.jepa_every_steps == Some(0) {
                return Err(anyhow!(
                    "training.latent_reasoning.jepa_every_steps must be > 0 when set"
                ));
            }
            if latent.jepa_future_offsets.is_empty()
                && !latent.next_latent.enabled
                && !latent.dragon_state.enabled
                && !latent.energy_model.enabled
                && !latent.step_contract.enabled
                && !latent.sigreg.enabled
            {
                return Err(anyhow!(
                    "training.latent_reasoning.jepa_future_offsets must not be empty unless next_latent, dragon_state, energy_model, step_contract, or sigreg is enabled"
                ));
            }
            if latent.jepa_future_offsets.contains(&0) {
                return Err(anyhow!(
                    "training.latent_reasoning.jepa_future_offsets must contain only positive offsets"
                ));
            }
            if !latent.teacher_update_rate.is_finite()
                || !(0.0..=1.0).contains(&latent.teacher_update_rate)
            {
                return Err(anyhow!(
                    "training.latent_reasoning.teacher_update_rate must be finite and in [0, 1]"
                ));
            }
            if latent.constraint_balancer.normalized_aux_scale < 0.0
                || !latent.constraint_balancer.normalized_aux_scale.is_finite()
            {
                return Err(anyhow!(
                    "training.latent_reasoning.constraint_balancer.normalized_aux_scale must be finite and >= 0"
                ));
            }
            if latent.next_latent.enabled {
                if latent.next_latent.every_steps == Some(0) {
                    return Err(anyhow!(
                        "training.latent_reasoning.next_latent.every_steps must be > 0 when set"
                    ));
                }
                if latent.next_latent.horizon == 0 {
                    return Err(anyhow!(
                        "training.latent_reasoning.next_latent.horizon must be > 0 when enabled"
                    ));
                }
                if !latent.next_latent.regression_weight.is_finite()
                    || latent.next_latent.regression_weight < 0.0
                {
                    return Err(anyhow!(
                        "training.latent_reasoning.next_latent.regression_weight must be finite and >= 0"
                    ));
                }
                if !latent.next_latent.token_kl_weight.is_finite()
                    || latent.next_latent.token_kl_weight < 0.0
                {
                    return Err(anyhow!(
                        "training.latent_reasoning.next_latent.token_kl_weight must be finite and >= 0"
                    ));
                }
                if !latent.next_latent.smooth_l1_beta.is_finite()
                    || latent.next_latent.smooth_l1_beta <= 0.0
                {
                    return Err(anyhow!(
                        "training.latent_reasoning.next_latent.smooth_l1_beta must be finite and > 0"
                    ));
                }
            }
            if latent.step_contract.enabled {
                if latent.step_contract.every_steps == Some(0) {
                    return Err(anyhow!(
                        "training.latent_reasoning.step_contract.every_steps must be > 0 when set"
                    ));
                }
                if latent.step_contract.max_rollout_steps_for_loss == 0 {
                    return Err(anyhow!(
                        "training.latent_reasoning.step_contract.max_rollout_steps_for_loss must be > 0 when enabled"
                    ));
                }
                if !latent.step_contract.ce_weight.is_finite()
                    || latent.step_contract.ce_weight < 0.0
                {
                    return Err(anyhow!(
                        "training.latent_reasoning.step_contract.ce_weight must be finite and >= 0"
                    ));
                }
                if !latent.step_contract.token_kl_weight.is_finite()
                    || latent.step_contract.token_kl_weight < 0.0
                {
                    return Err(anyhow!(
                        "training.latent_reasoning.step_contract.token_kl_weight must be finite and >= 0"
                    ));
                }
                if !latent.step_contract.monotonic_ce_weight.is_finite()
                    || latent.step_contract.monotonic_ce_weight < 0.0
                {
                    return Err(anyhow!(
                        "training.latent_reasoning.step_contract.monotonic_ce_weight must be finite and >= 0"
                    ));
                }
                if !latent.step_contract.contractive_weight.is_finite()
                    || latent.step_contract.contractive_weight < 0.0
                {
                    return Err(anyhow!(
                        "training.latent_reasoning.step_contract.contractive_weight must be finite and >= 0"
                    ));
                }
                if latent.step_contract.ce_weight <= f32::EPSILON
                    && latent.step_contract.token_kl_weight <= f32::EPSILON
                    && latent.step_contract.monotonic_ce_weight <= f32::EPSILON
                    && latent.step_contract.contractive_weight <= f32::EPSILON
                {
                    return Err(anyhow!(
                        "training.latent_reasoning.step_contract requires at least one positive loss weight"
                    ));
                }
                if !latent.step_contract.ce_tolerance.is_finite()
                    || latent.step_contract.ce_tolerance < 0.0
                {
                    return Err(anyhow!(
                        "training.latent_reasoning.step_contract.ce_tolerance must be finite and >= 0"
                    ));
                }
                if !latent.step_contract.trust_radius.is_finite()
                    || latent.step_contract.trust_radius < 0.0
                {
                    return Err(anyhow!(
                        "training.latent_reasoning.step_contract.trust_radius must be finite and >= 0"
                    ));
                }
            }
            if latent.dragon_state.enabled {
                if latent.dragon_state.every_steps == Some(0) {
                    return Err(anyhow!(
                        "training.latent_reasoning.dragon_state.every_steps must be > 0 when set"
                    ));
                }
                if !latent.dragon_state.rho_weight.is_finite()
                    || latent.dragon_state.rho_weight < 0.0
                {
                    return Err(anyhow!(
                        "training.latent_reasoning.dragon_state.rho_weight must be finite and >= 0"
                    ));
                }
                if !latent.dragon_state.rho_energy_weight.is_finite()
                    || latent.dragon_state.rho_energy_weight < 0.0
                {
                    return Err(anyhow!(
                        "training.latent_reasoning.dragon_state.rho_energy_weight must be finite and >= 0"
                    ));
                }
                if latent.dragon_state.rho_weight <= f32::EPSILON
                    && latent.dragon_state.rho_energy_weight <= f32::EPSILON
                {
                    return Err(anyhow!(
                        "training.latent_reasoning.dragon_state requires rho_weight or rho_energy_weight > 0"
                    ));
                }
                if !latent.dragon_state.smooth_l1_beta.is_finite()
                    || latent.dragon_state.smooth_l1_beta <= 0.0
                {
                    return Err(anyhow!(
                        "training.latent_reasoning.dragon_state.smooth_l1_beta must be finite and > 0"
                    ));
                }
                if latent.dragon_state.max_rho_slots < 2 {
                    return Err(anyhow!(
                        "training.latent_reasoning.dragon_state.max_rho_slots must be >= 2"
                    ));
                }
                if self.parallel.pipeline.enabled {
                    return Err(anyhow!(
                        "training.latent_reasoning.dragon_state.enabled does not support parallel.pipeline.enabled yet"
                    ));
                }
            }
            if latent.energy_model.enabled {
                if latent.energy_model.every_steps == Some(0) {
                    return Err(anyhow!(
                        "training.latent_reasoning.energy_model.every_steps must be > 0 when set"
                    ));
                }
                if !latent.energy_model.contrastive_weight.is_finite()
                    || latent.energy_model.contrastive_weight < 0.0
                {
                    return Err(anyhow!(
                        "training.latent_reasoning.energy_model.contrastive_weight must be finite and >= 0"
                    ));
                }
                if !latent.energy_model.monotonic_weight.is_finite()
                    || latent.energy_model.monotonic_weight < 0.0
                {
                    return Err(anyhow!(
                        "training.latent_reasoning.energy_model.monotonic_weight must be finite and >= 0"
                    ));
                }
                if !latent.energy_model.contractive_weight.is_finite()
                    || latent.energy_model.contractive_weight < 0.0
                {
                    return Err(anyhow!(
                        "training.latent_reasoning.energy_model.contractive_weight must be finite and >= 0"
                    ));
                }
                if latent.energy_model.contrastive_weight <= f32::EPSILON
                    && latent.energy_model.monotonic_weight <= f32::EPSILON
                    && latent.energy_model.contractive_weight <= f32::EPSILON
                {
                    return Err(anyhow!(
                        "training.latent_reasoning.energy_model requires at least one positive loss weight"
                    ));
                }
                if !latent.energy_model.margin.is_finite() || latent.energy_model.margin < 0.0 {
                    return Err(anyhow!(
                        "training.latent_reasoning.energy_model.margin must be finite and >= 0"
                    ));
                }
                if !latent.energy_model.monotonic_tolerance.is_finite()
                    || latent.energy_model.monotonic_tolerance < 0.0
                {
                    return Err(anyhow!(
                        "training.latent_reasoning.energy_model.monotonic_tolerance must be finite and >= 0"
                    ));
                }
                if !latent.energy_model.trust_radius.is_finite()
                    || latent.energy_model.trust_radius < 0.0
                {
                    return Err(anyhow!(
                        "training.latent_reasoning.energy_model.trust_radius must be finite and >= 0"
                    ));
                }
                if latent.energy_model.max_rollout_steps_for_loss == 0 {
                    return Err(anyhow!(
                        "training.latent_reasoning.energy_model.max_rollout_steps_for_loss must be > 0 when enabled"
                    ));
                }
            }
            if latent.sigreg.enabled {
                if latent.sigreg.every_steps == Some(0) {
                    return Err(anyhow!(
                        "training.latent_reasoning.sigreg.every_steps must be > 0 when set"
                    ));
                }
                if latent.sigreg.min_variance < 0.0 || !latent.sigreg.min_variance.is_finite() {
                    return Err(anyhow!(
                        "training.latent_reasoning.sigreg.min_variance must be finite and >= 0"
                    ));
                }
                if latent.sigreg.mean_tolerance < 0.0 || !latent.sigreg.mean_tolerance.is_finite() {
                    return Err(anyhow!(
                        "training.latent_reasoning.sigreg.mean_tolerance must be finite and >= 0"
                    ));
                }
                if latent.sigreg.max_rho_slots < 2 {
                    return Err(anyhow!(
                        "training.latent_reasoning.sigreg.max_rho_slots must be >= 2"
                    ));
                }
            }
        }
        if self.training.events.source_selection_every_steps == 0 {
            return Err(anyhow!(
                "training.events.source_selection_every_steps must be > 0"
            ));
        }
        if self.training.events.continual_backprop_every_steps == 0 {
            return Err(anyhow!(
                "training.events.continual_backprop_every_steps must be > 0"
            ));
        }
        if self.training.events.degeneracy_probe_every_epochs == 0 {
            return Err(anyhow!(
                "training.events.degeneracy_probe_every_epochs must be > 0"
            ));
        }
        if self.training.gates.plateau_patience_epochs == 0 {
            return Err(anyhow!(
                "training.gates.plateau_patience_epochs must be > 0"
            ));
        }
        if self.training.gates.validation_regression_patience_epochs == 0 {
            return Err(anyhow!(
                "training.gates.validation_regression_patience_epochs must be > 0"
            ));
        }
        if self.training.gates.source_entropy_patience == 0 {
            return Err(anyhow!(
                "training.gates.source_entropy_patience must be > 0"
            ));
        }
        if self.training.gates.difficulty_patience == 0 {
            return Err(anyhow!("training.gates.difficulty_patience must be > 0"));
        }
        if self.training.gates.degeneracy_patience == 0 {
            return Err(anyhow!("training.gates.degeneracy_patience must be > 0"));
        }
        if self.training.gates.capability_zero_verifier_patience_epochs == 0 {
            return Err(anyhow!(
                "training.gates.capability_zero_verifier_patience_epochs must be > 0"
            ));
        }
        if self.training.gates.capability_regression_patience_epochs == 0 {
            return Err(anyhow!(
                "training.gates.capability_regression_patience_epochs must be > 0"
            ));
        }
        if self.training.gates.degeneracy_entropy_min_bits < 0.0
            || !self.training.gates.degeneracy_entropy_min_bits.is_finite()
        {
            return Err(anyhow!(
                "training.gates.degeneracy_entropy_min_bits must be finite and >= 0"
            ));
        }
        if self.training.gates.capability_output_entropy_min_bits < 0.0
            || !self
                .training
                .gates
                .capability_output_entropy_min_bits
                .is_finite()
        {
            return Err(anyhow!(
                "training.gates.capability_output_entropy_min_bits must be finite and >= 0"
            ));
        }
        if !(0.0..=1.0).contains(&self.training.gates.capability_schema_wrong_max_rate)
            || !self
                .training
                .gates
                .capability_schema_wrong_max_rate
                .is_finite()
        {
            return Err(anyhow!(
                "training.gates.capability_schema_wrong_max_rate must be finite and in [0, 1]"
            ));
        }
        if !(0.0..=1.0).contains(&self.training.gates.capability_malformed_max_rate)
            || !self
                .training
                .gates
                .capability_malformed_max_rate
                .is_finite()
        {
            return Err(anyhow!(
                "training.gates.capability_malformed_max_rate must be finite and in [0, 1]"
            ));
        }
        if !(0.0..=1.0).contains(&self.training.gates.capability_missing_max_rate)
            || !self.training.gates.capability_missing_max_rate.is_finite()
        {
            return Err(anyhow!(
                "training.gates.capability_missing_max_rate must be finite and in [0, 1]"
            ));
        }
        if !(0.0..=1.0).contains(&self.training.gates.capability_completion_health_min_rate)
            || !self
                .training
                .gates
                .capability_completion_health_min_rate
                .is_finite()
        {
            return Err(anyhow!(
                "training.gates.capability_completion_health_min_rate must be finite and in [0, 1]"
            ));
        }
        if !(0.0..=1.0).contains(&self.training.gates.capability_distinct_2_min_fraction)
            || !self
                .training
                .gates
                .capability_distinct_2_min_fraction
                .is_finite()
        {
            return Err(anyhow!(
                "training.gates.capability_distinct_2_min_fraction must be finite and in [0, 1]"
            ));
        }
        if !(0.0..=1.0).contains(&self.training.gates.capability_answer_distinct_min_fraction)
            || !self
                .training
                .gates
                .capability_answer_distinct_min_fraction
                .is_finite()
        {
            return Err(anyhow!(
                "training.gates.capability_answer_distinct_min_fraction must be finite and in [0, 1]"
            ));
        }
        if !(0.0..=1.0).contains(
            &self
                .training
                .gates
                .capability_field_value_distinct_ratio_min,
        ) || !self
            .training
            .gates
            .capability_field_value_distinct_ratio_min
            .is_finite()
        {
            return Err(anyhow!(
                "training.gates.capability_field_value_distinct_ratio_min must be finite and in [0, 1]"
            ));
        }
        if !(0.0..=1.0).contains(&self.training.gates.capability_field_value_dominance_max)
            || !self
                .training
                .gates
                .capability_field_value_dominance_max
                .is_finite()
        {
            return Err(anyhow!(
                "training.gates.capability_field_value_dominance_max must be finite and in [0, 1]"
            ));
        }
        if !(0.0..=1.0).contains(&self.training.gates.degeneracy_max_probability_max)
            || !self
                .training
                .gates
                .degeneracy_max_probability_max
                .is_finite()
        {
            return Err(anyhow!(
                "training.gates.degeneracy_max_probability_max must be finite and in [0, 1]"
            ));
        }
        if !(0.0..=1.0).contains(&self.training.gates.degeneracy_argmax_unique_min_fraction)
            || !self
                .training
                .gates
                .degeneracy_argmax_unique_min_fraction
                .is_finite()
        {
            return Err(anyhow!(
                "training.gates.degeneracy_argmax_unique_min_fraction must be finite and in [0, 1]"
            ));
        }
        if !(0.0..=1.0).contains(&self.training.gates.degeneracy_distinct_2_min_fraction)
            || !self
                .training
                .gates
                .degeneracy_distinct_2_min_fraction
                .is_finite()
        {
            return Err(anyhow!(
                "training.gates.degeneracy_distinct_2_min_fraction must be finite and in [0, 1]"
            ));
        }
        if !(0.0..=1.0).contains(&self.training.gates.degeneracy_repetition_max_fraction)
            || !self
                .training
                .gates
                .degeneracy_repetition_max_fraction
                .is_finite()
        {
            return Err(anyhow!(
                "training.gates.degeneracy_repetition_max_fraction must be finite and in [0, 1]"
            ));
        }
        if !(0.0..=1.0).contains(&self.training.gates.degeneracy_eos_max_fraction)
            || !self.training.gates.degeneracy_eos_max_fraction.is_finite()
        {
            return Err(anyhow!(
                "training.gates.degeneracy_eos_max_fraction must be finite and in [0, 1]"
            ));
        }
        if !(0.0..=1.0).contains(&self.training.gates.degeneracy_period_2_max_fraction)
            || !self
                .training
                .gates
                .degeneracy_period_2_max_fraction
                .is_finite()
        {
            return Err(anyhow!(
                "training.gates.degeneracy_period_2_max_fraction must be finite and in [0, 1]"
            ));
        }
        if !(0.0..=1.0).contains(&self.training.gates.degeneracy_period_3_max_fraction)
            || !self
                .training
                .gates
                .degeneracy_period_3_max_fraction
                .is_finite()
        {
            return Err(anyhow!(
                "training.gates.degeneracy_period_3_max_fraction must be finite and in [0, 1]"
            ));
        }
        if !(0.0..=1.0).contains(&self.training.gates.degeneracy_period_2_to_16_max_fraction)
            || !self
                .training
                .gates
                .degeneracy_period_2_to_16_max_fraction
                .is_finite()
        {
            return Err(anyhow!(
                "training.gates.degeneracy_period_2_to_16_max_fraction must be finite and in [0, 1]"
            ));
        }
        if !(0.0..=1.0).contains(&self.training.gates.degeneracy_period_2_to_64_max_fraction)
            || !self
                .training
                .gates
                .degeneracy_period_2_to_64_max_fraction
                .is_finite()
        {
            return Err(anyhow!(
                "training.gates.degeneracy_period_2_to_64_max_fraction must be finite and in [0, 1]"
            ));
        }
        if self.parallel.world_size == 0 {
            return Err(anyhow!("parallel.world_size must be > 0"));
        }
        if self.parallel.data.size == 0 {
            return Err(anyhow!("parallel.data.size must be > 0"));
        }
        let collective_globals = (
            self.parallel.data.collective_num_nodes,
            self.parallel.data.collective_global_address.as_ref(),
            self.parallel.data.collective_node_address.as_ref(),
            self.parallel.data.collective_data_service_port,
        );
        match collective_globals {
            (None, None, None, None) => {}
            (Some(num_nodes), Some(global_address), Some(node_address), Some(port)) => {
                if num_nodes == 0 {
                    return Err(anyhow!(
                        "parallel.data.collective_num_nodes must be > 0 when set"
                    ));
                }
                if global_address.trim().is_empty() {
                    return Err(anyhow!(
                        "parallel.data.collective_global_address must not be empty when set"
                    ));
                }
                if node_address.trim().is_empty() {
                    return Err(anyhow!(
                        "parallel.data.collective_node_address must not be empty when set"
                    ));
                }
                if port == 0 {
                    return Err(anyhow!(
                        "parallel.data.collective_data_service_port must be > 0 when set"
                    ));
                }
            }
            _ => {
                return Err(anyhow!(
                    "parallel.data collective global settings must either all be set or all be omitted"
                ));
            }
        }
        if self.parallel.tensor.size == 0 {
            return Err(anyhow!("parallel.tensor.size must be > 0"));
        }
        let pipeline_stage_multiplier = if self.parallel.pipeline.enabled {
            self.parallel.pipeline.stage_count.max(1)
        } else {
            1
        };
        let expected_world_size = self
            .parallel
            .data
            .size
            .checked_mul(self.parallel.tensor.size)
            .and_then(|value| value.checked_mul(pipeline_stage_multiplier))
            .ok_or_else(|| anyhow!("parallel size configuration overflow"))?;
        if self.parallel.mode != ParallelismKind::Single
            && expected_world_size != self.parallel.world_size
        {
            return Err(anyhow!(
                "parallel.data.size * parallel.tensor.size * pipeline_stage_multiplier must equal parallel.world_size (got {} * {} * {} != {})",
                self.parallel.data.size,
                self.parallel.tensor.size,
                pipeline_stage_multiplier,
                self.parallel.world_size
            ));
        }
        match self.parallel.mode {
            ParallelismKind::Single => {
                if self.parallel.world_size != 1
                    || self.parallel.data.size != 1
                    || self.parallel.tensor.size != 1
                {
                    return Err(anyhow!(
                        "parallel.mode=single requires parallel.world_size=1, parallel.data.size=1, and parallel.tensor.size=1"
                    ));
                }
                if self.parallel.fsdp.enabled {
                    return Err(anyhow!(
                        "parallel.fsdp.enabled must be false when parallel.mode=single"
                    ));
                }
            }
            ParallelismKind::Ddp => {
                if self.parallel.world_size < 2 {
                    return Err(anyhow!(
                        "parallel.mode=ddp requires parallel.world_size >= 2"
                    ));
                }
                if self.parallel.tensor.size != 1 {
                    return Err(anyhow!(
                        "parallel.mode=ddp requires parallel.tensor.size = 1"
                    ));
                }
                if self.parallel.data.size * pipeline_stage_multiplier != self.parallel.world_size {
                    return Err(anyhow!(
                        "parallel.mode=ddp requires parallel.data.size * pipeline_stage_multiplier = parallel.world_size"
                    ));
                }
                if self.parallel.fsdp.enabled {
                    return Err(anyhow!(
                        "parallel.fsdp.enabled must be false when parallel.mode=ddp"
                    ));
                }
            }
            ParallelismKind::Fsdp => {
                if self.parallel.world_size < 2 {
                    return Err(anyhow!(
                        "parallel.mode=fsdp requires parallel.world_size >= 2"
                    ));
                }
                if self.parallel.tensor.size != 1 {
                    return Err(anyhow!(
                        "parallel.mode=fsdp requires parallel.tensor.size = 1"
                    ));
                }
                if self.parallel.data.size * pipeline_stage_multiplier != self.parallel.world_size {
                    return Err(anyhow!(
                        "parallel.mode=fsdp requires parallel.data.size * pipeline_stage_multiplier = parallel.world_size"
                    ));
                }
                if !self.parallel.fsdp.enabled {
                    return Err(anyhow!(
                        "parallel.fsdp.enabled must be true when parallel.mode=fsdp"
                    ));
                }
            }
            ParallelismKind::TensorParallelNeuron => {
                if self.parallel.world_size < 2 {
                    return Err(anyhow!(
                        "parallel.mode=tensor_parallel_neuron requires parallel.world_size >= 2"
                    ));
                }
                if self.parallel.data.size != 1 {
                    return Err(anyhow!(
                        "parallel.mode=tensor_parallel_neuron requires parallel.data.size = 1"
                    ));
                }
                if self.parallel.tensor.size * pipeline_stage_multiplier != self.parallel.world_size
                {
                    return Err(anyhow!(
                        "parallel.mode=tensor_parallel_neuron requires parallel.tensor.size * pipeline_stage_multiplier = parallel.world_size"
                    ));
                }
                if self.parallel.fsdp.enabled {
                    return Err(anyhow!(
                        "parallel.fsdp.enabled must be false when parallel.mode=tensor_parallel_neuron"
                    ));
                }
            }
            ParallelismKind::Hybrid2D => {
                if self.parallel.world_size < 4 {
                    return Err(anyhow!(
                        "parallel.mode=hybrid_2d requires parallel.world_size >= 4"
                    ));
                }
                if self.parallel.data.size < 2 || self.parallel.tensor.size < 2 {
                    return Err(anyhow!(
                        "parallel.mode=hybrid_2d requires parallel.data.size >= 2 and parallel.tensor.size >= 2"
                    ));
                }
            }
        }
        if self.parallel.pipeline.enabled {
            if self.parallel.pipeline.stage_count == 0 {
                return Err(anyhow!(
                    "parallel.pipeline.stage_count must be > 0 when pipeline is enabled"
                ));
            }
            if self.parallel.pipeline.virtual_stages_per_rank == 0 {
                return Err(anyhow!(
                    "parallel.pipeline.virtual_stages_per_rank must be > 0 when pipeline is enabled"
                ));
            }
            if self.parallel.pipeline.microbatches == 0 {
                return Err(anyhow!(
                    "parallel.pipeline.microbatches must be > 0 when pipeline is enabled"
                ));
            }
            if self.parallel.pipeline.microbatches > self.training.batch_size {
                return Err(anyhow!(
                    "parallel.pipeline.microbatches must be <= training.batch_size (got {} > {})",
                    self.parallel.pipeline.microbatches,
                    self.training.batch_size
                ));
            }
            if self.parallel.mode != ParallelismKind::Single
                && self.parallel.pipeline.stage_count > self.parallel.world_size
            {
                return Err(anyhow!(
                    "parallel.pipeline.stage_count must be <= parallel.world_size (got {} > {})",
                    self.parallel.pipeline.stage_count,
                    self.parallel.world_size
                ));
            }
            if self.parallel.pipeline.virtual_stages_per_rank > self.parallel.pipeline.stage_count {
                return Err(anyhow!(
                    "parallel.pipeline.virtual_stages_per_rank must be <= parallel.pipeline.stage_count (got {} > {})",
                    self.parallel.pipeline.virtual_stages_per_rank,
                    self.parallel.pipeline.stage_count
                ));
            }
            if matches!(
                self.parallel.pipeline.schedule,
                PipelineScheduleKind::Interleaved1f1b
            ) && self.parallel.pipeline.microbatches < self.parallel.pipeline.stage_count
            {
                return Err(anyhow!(
                    "parallel.pipeline.microbatches must be >= parallel.pipeline.stage_count for interleaved_1f1b (got {} < {})",
                    self.parallel.pipeline.microbatches,
                    self.parallel.pipeline.stage_count
                ));
            }
            if self.parallel.pipeline.cache.max_inflight_microbatches == 0 {
                return Err(anyhow!(
                    "parallel.pipeline.cache.max_inflight_microbatches must be > 0 when pipeline is enabled"
                ));
            }
        } else if self.parallel.pipeline.cache.enabled {
            return Err(anyhow!(
                "parallel.pipeline.cache.enabled requires parallel.pipeline.enabled"
            ));
        }
        if self.parallel.pipeline.cache.enabled
            && self.parallel.pipeline.communication != PipelineCommunicationKind::BlockResidualCache
        {
            return Err(anyhow!(
                "parallel.pipeline.cache.enabled requires parallel.pipeline.communication = \"block_residual_cache\""
            ));
        }
        if self.parallel.pipeline.enabled
            && self.parallel.pipeline.communication == PipelineCommunicationKind::BlockResidualCache
            && self.model.residual_connector != Some(ResidualConnectorKind::BlockAttentionResidual)
        {
            return Err(anyhow!(
                "parallel.pipeline.communication = \"block_residual_cache\" requires model.residual_connector = \"block_attention_residual\""
            ));
        }
        if matches!(self.training.target_effective_batch_size, Some(0)) {
            return Err(anyhow!(
                "training.target_effective_batch_size must be > 0 when set"
            ));
        }
        if self.training.max_iters == 0 {
            return Err(anyhow!("training.max_iters must be > 0"));
        }
        if self.training.checkpoint_interval_iters == 0 {
            return Err(anyhow!("training.checkpoint_interval_iters must be > 0"));
        }
        if self.training.log_frequency == 0 {
            return Err(anyhow!("training.log_frequency must be > 0"));
        }
        if self.training.init_checkpoint_epoch.is_some()
            && self.training.init_checkpoint_path.is_none()
        {
            return Err(anyhow!(
                "training.init_checkpoint_epoch requires training.init_checkpoint_path"
            ));
        }
        if self.training.init_transfer.backbone_blend_alpha.is_some()
            && self.training.init_checkpoint_path.is_none()
        {
            return Err(anyhow!(
                "training.init_transfer.backbone_blend_alpha requires training.init_checkpoint_path"
            ));
        }
        if self
            .training
            .init_transfer
            .interface_checkpoint_path
            .is_some()
            && self.training.init_checkpoint_path.is_none()
        {
            return Err(anyhow!(
                "training.init_transfer.interface_checkpoint_path requires training.init_checkpoint_path"
            ));
        }
        if self
            .training
            .init_transfer
            .interface_checkpoint_epoch
            .is_some()
            && self
                .training
                .init_transfer
                .interface_checkpoint_path
                .is_none()
        {
            return Err(anyhow!(
                "training.init_transfer.interface_checkpoint_epoch requires training.init_transfer.interface_checkpoint_path"
            ));
        }
        if (self
            .training
            .init_transfer
            .preserve_interface_input_embedding
            || self.training.init_transfer.preserve_interface_output_head
            || self
                .training
                .init_transfer
                .interface_output_head_blend_alpha
                .is_some())
            && self
                .training
                .init_transfer
                .interface_checkpoint_path
                .is_none()
        {
            return Err(anyhow!(
                "training.init_transfer.preserve_interface_input_embedding, training.init_transfer.preserve_interface_output_head, and training.init_transfer.interface_output_head_blend_alpha require training.init_transfer.interface_checkpoint_path"
            ));
        }
        if self
            .training
            .init_transfer
            .interface_output_head_blend_alpha
            .is_some()
            && self.training.init_transfer.preserve_interface_output_head
        {
            return Err(anyhow!(
                "training.init_transfer.interface_output_head_blend_alpha cannot be combined with training.init_transfer.preserve_interface_output_head"
            ));
        }
        if self.training.init_transfer.decoder_blend_alpha.is_some()
            && self.training.init_checkpoint_path.is_none()
        {
            return Err(anyhow!(
                "training.init_transfer.decoder_blend_alpha requires training.init_checkpoint_path"
            ));
        }
        if self.training.init_transfer.norm_blend_alpha.is_some()
            && self.training.init_checkpoint_path.is_none()
        {
            return Err(anyhow!(
                "training.init_transfer.norm_blend_alpha requires training.init_checkpoint_path"
            ));
        }
        if (self.training.init_transfer.backbone_grad_scale.is_some()
            || self
                .training
                .init_transfer
                .backbone_grad_scale_steps
                .is_some())
            && self.training.init_checkpoint_path.is_none()
        {
            return Err(anyhow!(
                "training.init_transfer.backbone_grad_scale and training.init_transfer.backbone_grad_scale_steps require training.init_checkpoint_path"
            ));
        }
        if self.training.init_transfer.fresh_top_layers.is_some()
            && self.training.init_checkpoint_path.is_none()
        {
            return Err(anyhow!(
                "training.init_transfer.fresh_top_layers requires training.init_checkpoint_path"
            ));
        }
        if self.training.init_transfer.preserve_fresh_decoder
            && self.training.init_checkpoint_path.is_none()
        {
            return Err(anyhow!(
                "training.init_transfer.preserve_fresh_decoder requires training.init_checkpoint_path"
            ));
        }
        if self.training.init_transfer.preserve_fresh_norm
            && self.training.init_checkpoint_path.is_none()
        {
            return Err(anyhow!(
                "training.init_transfer.preserve_fresh_norm requires training.init_checkpoint_path"
            ));
        }
        if self.training.init_transfer.match_fresh_rms
            && self.training.init_checkpoint_path.is_none()
        {
            return Err(anyhow!(
                "training.init_transfer.match_fresh_rms requires training.init_checkpoint_path"
            ));
        }
        if let Some(alpha) = self.training.init_transfer.backbone_blend_alpha
            && !(0.0..=1.0).contains(&alpha)
        {
            return Err(anyhow!(
                "training.init_transfer.backbone_blend_alpha must be in [0, 1]"
            ));
        }
        if let Some(alpha) = self.training.init_transfer.decoder_blend_alpha
            && !(0.0..=1.0).contains(&alpha)
        {
            return Err(anyhow!(
                "training.init_transfer.decoder_blend_alpha must be in [0, 1]"
            ));
        }
        if let Some(alpha) = self.training.init_transfer.norm_blend_alpha
            && !(0.0..=1.0).contains(&alpha)
        {
            return Err(anyhow!(
                "training.init_transfer.norm_blend_alpha must be in [0, 1]"
            ));
        }
        if let Some(alpha) = self
            .training
            .init_transfer
            .interface_output_head_blend_alpha
            && !(0.0..=1.0).contains(&alpha)
        {
            return Err(anyhow!(
                "training.init_transfer.interface_output_head_blend_alpha must be in [0, 1]"
            ));
        }
        if self.training.continual_backprop.enabled {
            if !(0.0..1.0).contains(&self.training.continual_backprop.utility_decay) {
                return Err(anyhow!(
                    "training.continual_backprop.utility_decay must be in [0, 1)"
                ));
            }
            if self.training.continual_backprop.replacement_rate <= 0.0
                || !self
                    .training
                    .continual_backprop
                    .replacement_rate
                    .is_finite()
            {
                return Err(anyhow!(
                    "training.continual_backprop.replacement_rate must be finite and > 0"
                ));
            }
            if self.training.continual_backprop.maturity_steps == 0 {
                return Err(anyhow!(
                    "training.continual_backprop.maturity_steps must be > 0"
                ));
            }
            if self.training.continual_backprop.sample_interval_steps == 0 {
                return Err(anyhow!(
                    "training.continual_backprop.sample_interval_steps must be > 0"
                ));
            }
            if self.training.continual_backprop.replace_interval_steps == 0 {
                return Err(anyhow!(
                    "training.continual_backprop.replace_interval_steps must be > 0"
                ));
            }
            if self.training.continual_backprop.utility_epsilon <= 0.0
                || !self.training.continual_backprop.utility_epsilon.is_finite()
            {
                return Err(anyhow!(
                    "training.continual_backprop.utility_epsilon must be finite and > 0"
                ));
            }
            if self.training.continual_backprop.lr_coupling_power < 0.0
                || !self
                    .training
                    .continual_backprop
                    .lr_coupling_power
                    .is_finite()
            {
                return Err(anyhow!(
                    "training.continual_backprop.lr_coupling_power must be finite and >= 0"
                ));
            }
            if self
                .training
                .continual_backprop
                .max_replacements_per_interval
                == 0
            {
                return Err(anyhow!(
                    "training.continual_backprop.max_replacements_per_interval must be > 0"
                ));
            }
        }
        let mut seen_module_lr_targets = HashSet::new();
        for entry in &self.training.module_lr_scales {
            if entry.scale <= 0.0 || !entry.scale.is_finite() {
                return Err(anyhow!(
                    "training.module_lr_scales[{:#?}] scale must be finite and > 0",
                    entry.target
                ));
            }
            if let Some(schedule) = &entry.schedule {
                if schedule.final_scale <= 0.0 || !schedule.final_scale.is_finite() {
                    return Err(anyhow!(
                        "training.module_lr_scales[{:#?}].schedule.final_scale must be finite and > 0",
                        entry.target
                    ));
                }
                if !schedule.start_fraction.is_finite()
                    || !(0.0..=1.0).contains(&schedule.start_fraction)
                {
                    return Err(anyhow!(
                        "training.module_lr_scales[{:#?}].schedule.start_fraction must be finite and in [0, 1]",
                        entry.target
                    ));
                }
                if !schedule.end_fraction.is_finite()
                    || !(0.0..=1.0).contains(&schedule.end_fraction)
                {
                    return Err(anyhow!(
                        "training.module_lr_scales[{:#?}].schedule.end_fraction must be finite and in [0, 1]",
                        entry.target
                    ));
                }
                if schedule.end_fraction < schedule.start_fraction {
                    return Err(anyhow!(
                        "training.module_lr_scales[{:#?}].schedule.end_fraction must be >= start_fraction",
                        entry.target
                    ));
                }
            }
            if !seen_module_lr_targets.insert(entry.target) {
                return Err(anyhow!(
                    "training.module_lr_scales contains duplicate target {:?}",
                    entry.target
                ));
            }
        }
        if self.training.init_transfer.backbone_grad_scale.is_some()
            ^ self
                .training
                .init_transfer
                .backbone_grad_scale_steps
                .is_some()
        {
            return Err(anyhow!(
                "training.init_transfer.backbone_grad_scale and training.init_transfer.backbone_grad_scale_steps must be set together"
            ));
        }
        if let Some(scale) = self.training.init_transfer.backbone_grad_scale
            && !(0.0..=1.0).contains(&scale)
        {
            return Err(anyhow!(
                "training.init_transfer.backbone_grad_scale must be in [0, 1]"
            ));
        }
        if matches!(
            self.training.init_transfer.backbone_grad_scale_steps,
            Some(0)
        ) {
            return Err(anyhow!(
                "training.init_transfer.backbone_grad_scale_steps must be > 0 when set"
            ));
        }
        if matches!(self.training.init_transfer.fresh_top_layers, Some(0)) {
            return Err(anyhow!(
                "training.init_transfer.fresh_top_layers must be > 0 when set"
            ));
        }
        match self.training.launch_mode {
            TrainingLaunchMode::Fresh => {
                if self.training.resume_run_dir.is_some()
                    || self.training.resume_checkpoint_epoch.is_some()
                    || self.training.init_checkpoint_path.is_some()
                    || self.training.init_checkpoint_epoch.is_some()
                    || self.training.init_transfer != Default::default()
                {
                    return Err(anyhow!(
                        "training.launch_mode = \"fresh\" requires resume and init checkpoint settings to all be unset"
                    ));
                }
            }
            TrainingLaunchMode::ResumeExactRun => {
                if self.training.resume_run_dir.is_none() {
                    return Err(anyhow!(
                        "training.launch_mode = \"resume_exact_run\" requires training.resume_run_dir"
                    ));
                }
                if self.training.init_checkpoint_path.is_some()
                    || self.training.init_checkpoint_epoch.is_some()
                    || self.training.init_transfer != Default::default()
                {
                    return Err(anyhow!(
                        "training.launch_mode = \"resume_exact_run\" cannot be combined with init checkpoint or init transfer settings"
                    ));
                }
            }
            TrainingLaunchMode::ResumeLatestCheckpointIfPresent => {
                if self.training.resume_run_dir.is_some() {
                    return Err(anyhow!(
                        "training.launch_mode = \"resume_latest_checkpoint_if_present\" cannot be combined with training.resume_run_dir"
                    ));
                }
                if self.training.resume_checkpoint_epoch.is_some() {
                    return Err(anyhow!(
                        "training.launch_mode = \"resume_latest_checkpoint_if_present\" cannot be combined with training.resume_checkpoint_epoch"
                    ));
                }
            }
            TrainingLaunchMode::InitFromCheckpoint => {
                if self.training.init_checkpoint_path.is_none() {
                    return Err(anyhow!(
                        "training.launch_mode = \"init_from_checkpoint\" requires training.init_checkpoint_path"
                    ));
                }
                if self.training.resume_run_dir.is_some()
                    || self.training.resume_checkpoint_epoch.is_some()
                {
                    return Err(anyhow!(
                        "training.launch_mode = \"init_from_checkpoint\" cannot be combined with training.resume_run_dir or training.resume_checkpoint_epoch"
                    ));
                }
            }
        }
        if self.wgpu.training.startup_autotune.enabled {
            let autotune = &self.wgpu.training.startup_autotune;
            if autotune.target_device_memory_mb == 0 {
                return Err(anyhow!(
                    "wgpu.training.startup_autotune.target_device_memory_mb must be > 0 when enabled"
                ));
            }
            if autotune.min_batch_size == 0 {
                return Err(anyhow!(
                    "wgpu.training.startup_autotune.min_batch_size must be > 0 when enabled"
                ));
            }
            if matches!(autotune.max_batch_size, Some(0)) {
                return Err(anyhow!(
                    "wgpu.training.startup_autotune.max_batch_size must be > 0 when set"
                ));
            }
            if autotune.probe_steps == 0 {
                return Err(anyhow!(
                    "wgpu.training.startup_autotune.probe_steps must be > 0 when enabled"
                ));
            }
            if let Some(max_batch_size) = autotune.max_batch_size
                && max_batch_size < autotune.min_batch_size
            {
                return Err(anyhow!(
                    "wgpu.training.startup_autotune.max_batch_size must be >= min_batch_size"
                ));
            }
        }
        if let Some(epochs) = self.training.epochs
            && epochs == 0
        {
            return Err(anyhow!("training.epochs must be > 0"));
        }
        self.optimizer.validate()?;
        if matches!(self.optimizer.name, OptimizerKind::PredictiveCoding) {
            if !self.training.predictive_coding.enabled {
                return Err(anyhow!(
                    "optimizer.name=predictive_coding requires training.predictive_coding.enabled"
                ));
            }
            if matches!(
                self.training.predictive_coding.parameter_update,
                PredictiveCodingParameterUpdate::StateOnlyControl
            ) {
                return Err(anyhow!(
                    "optimizer.name=predictive_coding requires training.predictive_coding.parameter_update=optimizer"
                ));
            }
            if matches!(
                self.training.predictive_coding.backward_mode,
                PredictiveCodingBackwardMode::Block
            ) {
                return Err(anyhow!(
                    "optimizer.name=predictive_coding currently requires training.predictive_coding.backward_mode=chunked"
                ));
            }
            if self.training.continual_backprop.enabled {
                return Err(anyhow!(
                    "optimizer.name=predictive_coding does not yet support training.continual_backprop.enabled"
                ));
            }
            if self.training.neuron_scaling.enabled {
                return Err(anyhow!(
                    "optimizer.name=predictive_coding does not yet support training.neuron_scaling.enabled"
                ));
            }
            if self.training.tbptt_persist_across_steps {
                return Err(anyhow!(
                    "optimizer.name=predictive_coding does not yet support training.tbptt_persist_across_steps"
                ));
            }
        }
        if matches!(self.optimizer.name, OptimizerKind::Eggroll) {
            if self.optimizer.eggroll.gradient_learning_rate.is_some() {
                return Err(anyhow!(
                    "optimizer.name=eggroll does not support optimizer.eggroll.gradient_learning_rate; choose pure EGGROLL or optimizer.name=adamw"
                ));
            }
            if self.optimizer.eggroll.gradient_weight_decay.is_some() {
                return Err(anyhow!(
                    "optimizer.name=eggroll does not support optimizer.eggroll.gradient_weight_decay; choose pure EGGROLL or optimizer.name=adamw"
                ));
            }
            if self.parallel.mode != ParallelismKind::Single {
                return Err(anyhow!(
                    "optimizer.name=eggroll currently requires parallel.mode=single"
                ));
            }
            if self.parallel.pipeline.enabled {
                return Err(anyhow!(
                    "optimizer.name=eggroll does not yet support parallel.pipeline.enabled"
                ));
            }
            if self.training.gradient_accumulation_steps != 1 {
                return Err(anyhow!(
                    "optimizer.name=eggroll currently requires training.gradient_accumulation_steps = 1"
                ));
            }
            if self.training.continual_backprop.enabled {
                return Err(anyhow!(
                    "optimizer.name=eggroll does not yet support training.continual_backprop.enabled"
                ));
            }
            if self.training.neuron_scaling.enabled {
                return Err(anyhow!(
                    "optimizer.name=eggroll does not yet support training.neuron_scaling.enabled"
                ));
            }
            if self.training.predictive_coding.enabled {
                return Err(anyhow!(
                    "optimizer.name=eggroll does not support training.predictive_coding.enabled; use optimizer.name=predictive_coding for the PC optimizer path"
                ));
            }
            if self.training.tbptt_persist_across_steps {
                return Err(anyhow!(
                    "optimizer.name=eggroll does not yet support training.tbptt_persist_across_steps"
                ));
            }
            if !self.training.objective.is_next_token() {
                return Err(anyhow!(
                    "optimizer.name=eggroll currently supports only the next-token training objective"
                ));
            }
            if self.training.ruliad_supervision.uses_target_loss_mask() {
                return Err(anyhow!(
                    "optimizer.name=eggroll does not yet support masked training.ruliad_supervision={:?}; use optimizer.name=adamw or disable ruliad target masks",
                    self.training.ruliad_supervision.mode
                ));
            }
            if self.training.ruliad_supervision.verifier_reward.enabled {
                return Err(anyhow!(
                    "optimizer.name=eggroll does not yet support training.ruliad_supervision.verifier_reward.enabled"
                ));
            }
            if self.training.ruliad_supervision.proof_policy.enabled {
                return Err(anyhow!(
                    "optimizer.name=eggroll does not yet support training.ruliad_supervision.proof_policy.enabled"
                ));
            }
        }
        if self.training.ruliad_supervision.answer_ranking.enabled {
            let ranking = self.training.ruliad_supervision.answer_ranking;
            if !ranking.weight.is_finite() || ranking.weight < 0.0 {
                return Err(anyhow!(
                    "training.ruliad_supervision.answer_ranking.weight must be finite and non-negative"
                ));
            }
            if !ranking.margin.is_finite() || ranking.margin < 0.0 {
                return Err(anyhow!(
                    "training.ruliad_supervision.answer_ranking.margin must be finite and non-negative"
                ));
            }
            if ranking.corrupt_offset <= 0 {
                return Err(anyhow!(
                    "training.ruliad_supervision.answer_ranking.corrupt_offset must be positive"
                ));
            }
            if !self.training.ruliad_supervision.uses_answer_target_mask() {
                return Err(anyhow!(
                    "training.ruliad_supervision.answer_ranking.enabled requires training.ruliad_supervision.mode to use answer target masks"
                ));
            }
        }
        if !(1..=16).contains(&self.training.ruliad_supervision.answer_value_token_weight) {
            return Err(anyhow!(
                "training.ruliad_supervision.answer_value_token_weight must be in [1, 16]"
            ));
        }
        if !(1..=16).contains(&self.training.ruliad_supervision.answer_close_marker_weight) {
            return Err(anyhow!(
                "training.ruliad_supervision.answer_close_marker_weight must be in [1, 16]"
            ));
        }
        if !(1..=16).contains(&self.training.ruliad_supervision.answer_schema_token_weight) {
            return Err(anyhow!(
                "training.ruliad_supervision.answer_schema_token_weight must be in [1, 16]"
            ));
        }
        if !(1..=16).contains(
            &self
                .training
                .ruliad_supervision
                .answer_schema_start_token_weight,
        ) {
            return Err(anyhow!(
                "training.ruliad_supervision.answer_schema_start_token_weight must be in [1, 16]"
            ));
        }
        if self.training.ruliad_supervision.answer_contract.enabled {
            let contract = self.training.ruliad_supervision.answer_contract;
            if !contract.weight.is_finite() || contract.weight < 0.0 {
                return Err(anyhow!(
                    "training.ruliad_supervision.answer_contract.weight must be finite and non-negative"
                ));
            }
            if contract.weight > 0.0 {
                if contract.every_steps == 0 {
                    return Err(anyhow!(
                        "training.ruliad_supervision.answer_contract.every_steps must be positive when weight > 0"
                    ));
                }
                if contract.max_completion_tokens == 0 {
                    return Err(anyhow!(
                        "training.ruliad_supervision.answer_contract.max_completion_tokens must be positive when weight > 0"
                    ));
                }
                if contract.max_rows_per_step == 0 {
                    return Err(anyhow!(
                        "training.ruliad_supervision.answer_contract.max_rows_per_step must be positive when weight > 0"
                    ));
                }
                for (name, value) in [
                    ("schema_token_weight", contract.schema_token_weight),
                    (
                        "schema_start_token_weight",
                        contract.schema_start_token_weight,
                    ),
                    ("value_token_weight", contract.value_token_weight),
                    ("other_token_weight", contract.other_token_weight),
                    (
                        "prompt_schema_value_weight",
                        contract.prompt_schema_value_weight,
                    ),
                    (
                        "premature_close_unlikelihood_weight",
                        contract.premature_close_unlikelihood_weight,
                    ),
                ] {
                    if !value.is_finite() || value < 0.0 {
                        return Err(anyhow!(
                            "training.ruliad_supervision.answer_contract.{name} must be finite and non-negative"
                        ));
                    }
                }
                if contract.schema_token_weight <= f32::EPSILON
                    && contract.value_token_weight <= f32::EPSILON
                    && contract.other_token_weight <= f32::EPSILON
                    && contract.prompt_schema_value_weight <= f32::EPSILON
                {
                    return Err(anyhow!(
                        "training.ruliad_supervision.answer_contract requires at least one positive token weight when weight > 0"
                    ));
                }
                if !self.training.ruliad_supervision.uses_answer_target_mask() {
                    return Err(anyhow!(
                        "training.ruliad_supervision.answer_contract.enabled requires training.ruliad_supervision.mode to use answer target masks"
                    ));
                }
                if self.parallel.pipeline.enabled {
                    return Err(anyhow!(
                        "training.ruliad_supervision.answer_contract.enabled does not yet support parallel.pipeline.enabled"
                    ));
                }
            }
        }
        if self.training.ruliad_supervision.answer_denoising.enabled {
            let denoising = self.training.ruliad_supervision.answer_denoising;
            if !denoising.weight.is_finite() || denoising.weight < 0.0 {
                return Err(anyhow!(
                    "training.ruliad_supervision.answer_denoising.weight must be finite and non-negative"
                ));
            }
            if !denoising.probability.is_finite() || !(0.0..=1.0).contains(&denoising.probability) {
                return Err(anyhow!(
                    "training.ruliad_supervision.answer_denoising.probability must be finite and in [0, 1]"
                ));
            }
            if denoising.corrupt_offset <= 0 {
                return Err(anyhow!(
                    "training.ruliad_supervision.answer_denoising.corrupt_offset must be positive"
                ));
            }
            if !denoising.structured_recovery_weight.is_finite()
                || denoising.structured_recovery_weight < 0.0
            {
                return Err(anyhow!(
                    "training.ruliad_supervision.answer_denoising.structured_recovery_weight must be finite and non-negative"
                ));
            }
            if denoising.structured_recovery_weight > 0.0 {
                if denoising.structured_recovery_every_steps == 0 {
                    return Err(anyhow!(
                        "training.ruliad_supervision.answer_denoising.structured_recovery_every_steps must be positive when structured_recovery_weight > 0"
                    ));
                }
                if denoising.structured_recovery_max_completion_tokens == 0 {
                    return Err(anyhow!(
                        "training.ruliad_supervision.answer_denoising.structured_recovery_max_completion_tokens must be positive when structured_recovery_weight > 0"
                    ));
                }
                if denoising.structured_recovery_negative_count == 0
                    && denoising.structured_recovery_template_negative_count == 0
                    && denoising.structured_recovery_schema_negative_count == 0
                {
                    return Err(anyhow!(
                        "training.ruliad_supervision.answer_denoising.structured_recovery_negative_count, structured_recovery_template_negative_count, or structured_recovery_schema_negative_count must be positive when structured_recovery_weight > 0"
                    ));
                }
            }
            if !self.training.ruliad_supervision.uses_answer_target_mask() {
                return Err(anyhow!(
                    "training.ruliad_supervision.answer_denoising.enabled requires training.ruliad_supervision.mode to use answer target masks"
                ));
            }
            if self.parallel.pipeline.enabled {
                return Err(anyhow!(
                    "training.ruliad_supervision.answer_denoising.enabled does not yet support parallel.pipeline.enabled"
                ));
            }
        }
        if self.training.ruliad_supervision.proof_policy.enabled {
            let proof_policy = self.training.ruliad_supervision.proof_policy;
            if proof_policy.scoring == crate::config::RuliadProofPolicyScoring::SemanticEnergy {
                if !self
                    .model
                    .sequence_score_head
                    .is_some_and(|head| head.enabled)
                {
                    return Err(anyhow!(
                        "training.ruliad_supervision.proof_policy.scoring=semantic_energy requires model.sequence_score_head.enabled=true"
                    ));
                }
                if proof_policy.normalization
                    != crate::config::RuliadProofPolicyNormalization::CandidateConditional
                {
                    return Err(anyhow!(
                        "training.ruliad_supervision.proof_policy.scoring=semantic_energy requires normalization=candidate_conditional"
                    ));
                }
            }
            match proof_policy.gradient_scope {
                crate::config::RuliadProofPolicyGradientScope::ScoreHeadOnly
                    if proof_policy.scoring
                        != crate::config::RuliadProofPolicyScoring::SemanticEnergy =>
                {
                    return Err(anyhow!(
                        "training.ruliad_supervision.proof_policy.gradient_scope=score_head_only requires scoring=semantic_energy"
                    ));
                }
                crate::config::RuliadProofPolicyGradientScope::LanguageHeadOnly
                    if proof_policy.scoring
                        != crate::config::RuliadProofPolicyScoring::CompletionLikelihood =>
                {
                    return Err(anyhow!(
                        "training.ruliad_supervision.proof_policy.gradient_scope=language_head_only requires scoring=completion_likelihood"
                    ));
                }
                crate::config::RuliadProofPolicyGradientScope::LanguageHeadOnly
                    if self.model.tie_input_output_embeddings.unwrap_or(false) =>
                {
                    return Err(anyhow!(
                        "training.ruliad_supervision.proof_policy.gradient_scope=language_head_only requires model.tie_input_output_embeddings=false"
                    ));
                }
                crate::config::RuliadProofPolicyGradientScope::LanguageHeadOnly
                    if self
                        .model
                        .language_head
                        .as_ref()
                        .is_some_and(|head| !head.uses_flat_token_logits()) =>
                {
                    return Err(anyhow!(
                        "training.ruliad_supervision.proof_policy.gradient_scope=language_head_only requires model.language_head.type=standard_token_classification"
                    ));
                }
                crate::config::RuliadProofPolicyGradientScope::LanguageHeadOnly
                    if self
                        .model
                        .latent_reasoning
                        .as_ref()
                        .is_some_and(|latent| latent.step_conditioned_decoder) =>
                {
                    return Err(anyhow!(
                        "training.ruliad_supervision.proof_policy.gradient_scope=language_head_only requires model.latent_reasoning.step_conditioned_decoder=false"
                    ));
                }
                crate::config::RuliadProofPolicyGradientScope::FullModel
                | crate::config::RuliadProofPolicyGradientScope::ScoreHeadOnly
                | crate::config::RuliadProofPolicyGradientScope::LanguageHeadOnly => {}
            }
            if !proof_policy.weight.is_finite() || proof_policy.weight <= 0.0 {
                return Err(anyhow!(
                    "training.ruliad_supervision.proof_policy.weight must be finite and positive when enabled"
                ));
            }
            if proof_policy.every_steps == 0 {
                return Err(anyhow!(
                    "training.ruliad_supervision.proof_policy.every_steps must be positive when enabled"
                ));
            }
            if proof_policy.rollout_steps == 0 {
                return Err(anyhow!(
                    "training.ruliad_supervision.proof_policy.rollout_steps must be positive when enabled"
                ));
            }
            if proof_policy.mode
                == crate::config::RuliadProofPolicyTrainingMode::StaticThenPairedDagger
            {
                if proof_policy.dagger_start_after_steps <= proof_policy.start_after_steps {
                    return Err(anyhow!(
                        "training.ruliad_supervision.proof_policy.dagger_start_after_steps must exceed start_after_steps for static_then_paired_dagger"
                    ));
                }
                if !proof_policy
                    .dagger_start_after_steps
                    .is_multiple_of(proof_policy.every_steps)
                {
                    return Err(anyhow!(
                        "training.ruliad_supervision.proof_policy.dagger_start_after_steps must align with every_steps for static_then_paired_dagger"
                    ));
                }
                if proof_policy.max_rows_per_update < 2 {
                    return Err(anyhow!(
                        "training.ruliad_supervision.proof_policy.max_rows_per_update must be at least 2 for static_then_paired_dagger"
                    ));
                }
                if proof_policy.rollout_steps > 1 {
                    let dagger_rows = proof_policy.base_semantic_rows_per_update() / 2;
                    let maximum_stratified_trajectories = dagger_rows / 2;
                    if dagger_rows < 2 {
                        return Err(anyhow!(
                            "training.ruliad_supervision.proof_policy row budgets must fit an initial and model-visited DAgger state for static_then_paired_dagger"
                        ));
                    }
                    if proof_policy.stratified_difficulty_levels > maximum_stratified_trajectories {
                        return Err(anyhow!(
                            "training.ruliad_supervision.proof_policy.stratified_difficulty_levels exceeds the paired DAgger trajectory budget after reserving one model-visited state per trajectory"
                        ));
                    }
                }
            }
            if proof_policy.max_rows_per_update == 0 {
                return Err(anyhow!(
                    "training.ruliad_supervision.proof_policy.max_rows_per_update must be positive when enabled"
                ));
            }
            if proof_policy.max_presentation_rows_per_update == 0 {
                return Err(anyhow!(
                    "training.ruliad_supervision.proof_policy.max_presentation_rows_per_update must be positive when enabled"
                ));
            }
            if proof_policy.candidates < 2 {
                return Err(anyhow!(
                    "training.ruliad_supervision.proof_policy.candidates must be at least 2 when enabled"
                ));
            }
            if proof_policy.counterfactual_targets_per_state > 0 {
                let semantic_energy =
                    proof_policy.scoring == crate::config::RuliadProofPolicyScoring::SemanticEnergy;
                let isolated_completion = proof_policy.scoring
                    == crate::config::RuliadProofPolicyScoring::CompletionLikelihood
                    && proof_policy.gradient_scope
                        == crate::config::RuliadProofPolicyGradientScope::LanguageHeadOnly
                    && proof_policy.normalization
                        == crate::config::RuliadProofPolicyNormalization::CandidateConditional;
                if !semantic_energy && !isolated_completion {
                    return Err(anyhow!(
                        "training.ruliad_supervision.proof_policy.counterfactual_targets_per_state requires scoring=semantic_energy or completion_likelihood with gradient_scope=language_head_only and normalization=candidate_conditional"
                    ));
                }
            }
            if proof_policy.counterfactual_targets_per_state >= proof_policy.candidates {
                return Err(anyhow!(
                    "training.ruliad_supervision.proof_policy.counterfactual_targets_per_state must be less than candidates"
                ));
            }
            if proof_policy.presentation_risk
                == crate::config::RuliadProofPolicyPresentationRisk::Worst
                && proof_policy.candidate_symmetry
                    != crate::config::RuliadProofPolicyCandidateSymmetry::CyclicOrbitAverage
            {
                return Err(anyhow!(
                    "training.ruliad_supervision.proof_policy.presentation_risk=worst requires candidate_symmetry=cyclic_orbit_average"
                ));
            }
            if proof_policy.normalization
                == crate::config::RuliadProofPolicyNormalization::PrefixConditional
                && proof_policy.presentation_risk
                    != crate::config::RuliadProofPolicyPresentationRisk::Mean
            {
                return Err(anyhow!(
                    "training.ruliad_supervision.proof_policy.normalization=prefix_conditional requires presentation_risk=mean"
                ));
            }
            if proof_policy.semantic_rows_per_update() == 0 {
                return Err(anyhow!(
                    "training.ruliad_supervision.proof_policy row budgets must fit one complete target-variant presentation group"
                ));
            }
            if proof_policy.mode
                == crate::config::RuliadProofPolicyTrainingMode::StaticThenPairedDagger
                && proof_policy.base_semantic_rows_per_update() < 2
            {
                return Err(anyhow!(
                    "training.ruliad_supervision.proof_policy presentation budget must fit at least 2 base semantic rows for static_then_paired_dagger"
                ));
            }
            if proof_policy.max_completion_tokens == 0 {
                return Err(anyhow!(
                    "training.ruliad_supervision.proof_policy.max_completion_tokens must be positive when enabled"
                ));
            }
            if self.parallel.pipeline.enabled {
                return Err(anyhow!(
                    "training.ruliad_supervision.proof_policy.enabled does not yet support parallel.pipeline.enabled"
                ));
            }
        }
        if self.training.ruliad_supervision.verifier_reward.enabled {
            let verifier_reward = self.training.ruliad_supervision.verifier_reward;
            if !verifier_reward.weight.is_finite() || verifier_reward.weight < 0.0 {
                return Err(anyhow!(
                    "training.ruliad_supervision.verifier_reward.weight must be finite and non-negative"
                ));
            }
            if verifier_reward.max_completion_tokens == 0 {
                return Err(anyhow!(
                    "training.ruliad_supervision.verifier_reward.max_completion_tokens must be positive"
                ));
            }
            let policy_reward_enabled = verifier_reward.weight > 0.0;
            let structured_contrast_enabled = verifier_reward.structured_contrast_weight > 0.0;
            let field_binding_contrast_enabled =
                verifier_reward.field_binding_contrast_weight > 0.0;
            let rollout_imitation_enabled = verifier_reward.rollout_imitation_weight > 0.0
                || verifier_reward.rollout_recovery_weight > 0.0;
            let generated_attractor_replay_enabled =
                verifier_reward.generated_attractor_replay_capacity > 0;
            if verifier_reward.include_structured_negative_candidates
                && verifier_reward.structured_negative_count == 0
                && verifier_reward.structured_template_negative_count == 0
                && verifier_reward.structured_schema_negative_count == 0
            {
                return Err(anyhow!(
                    "training.ruliad_supervision.verifier_reward.structured_negative_count, structured_template_negative_count, or structured_schema_negative_count must be positive when include_structured_negative_candidates is true"
                ));
            }
            if !verifier_reward.structured_contrast_weight.is_finite()
                || verifier_reward.structured_contrast_weight < 0.0
            {
                return Err(anyhow!(
                    "training.ruliad_supervision.verifier_reward.structured_contrast_weight must be finite and non-negative"
                ));
            }
            if verifier_reward.structured_contrast_weight > 0.0 {
                if verifier_reward.structured_contrast_every_steps == 0 {
                    return Err(anyhow!(
                        "training.ruliad_supervision.verifier_reward.structured_contrast_every_steps must be positive when structured_contrast_weight > 0"
                    ));
                }
                if verifier_reward.structured_negative_count == 0
                    && verifier_reward.structured_template_negative_count == 0
                    && verifier_reward.structured_schema_negative_count == 0
                {
                    return Err(anyhow!(
                        "training.ruliad_supervision.verifier_reward.structured_negative_count, structured_template_negative_count, or structured_schema_negative_count must be positive when structured_contrast_weight > 0"
                    ));
                }
                if !verifier_reward.structured_contrast_margin.is_finite()
                    || verifier_reward.structured_contrast_margin < 0.0
                {
                    return Err(anyhow!(
                        "training.ruliad_supervision.verifier_reward.structured_contrast_margin must be finite and non-negative"
                    ));
                }
            }
            if !verifier_reward.field_binding_contrast_weight.is_finite()
                || verifier_reward.field_binding_contrast_weight < 0.0
            {
                return Err(anyhow!(
                    "training.ruliad_supervision.verifier_reward.field_binding_contrast_weight must be finite and non-negative"
                ));
            }
            if field_binding_contrast_enabled {
                if verifier_reward.field_binding_contrast_every_steps == 0 {
                    return Err(anyhow!(
                        "training.ruliad_supervision.verifier_reward.field_binding_contrast_every_steps must be positive when field_binding_contrast_weight > 0"
                    ));
                }
                if !verifier_reward.field_binding_contrast_margin.is_finite()
                    || verifier_reward.field_binding_contrast_margin < 0.0
                {
                    return Err(anyhow!(
                        "training.ruliad_supervision.verifier_reward.field_binding_contrast_margin must be finite and non-negative"
                    ));
                }
                if !verifier_reward
                    .field_binding_contrast_pair_weight
                    .is_finite()
                    || verifier_reward.field_binding_contrast_pair_weight < 0.0
                {
                    return Err(anyhow!(
                        "training.ruliad_supervision.verifier_reward.field_binding_contrast_pair_weight must be finite and non-negative"
                    ));
                }
                if verifier_reward.field_binding_contrast_max_pairs == 0 {
                    return Err(anyhow!(
                        "training.ruliad_supervision.verifier_reward.field_binding_contrast_max_pairs must be positive when field_binding_contrast_weight > 0"
                    ));
                }
                if verifier_reward.field_binding_contrast_rank_metric_every_steps == 0 {
                    return Err(anyhow!(
                        "training.ruliad_supervision.verifier_reward.field_binding_contrast_rank_metric_every_steps must be positive when field_binding_contrast_weight > 0"
                    ));
                }
            }
            if generated_attractor_replay_enabled {
                if !policy_reward_enabled && !rollout_imitation_enabled {
                    return Err(anyhow!(
                        "training.ruliad_supervision.verifier_reward.generated_attractor_replay_capacity requires verifier_reward.weight > 0 or rollout_imitation_weight/rollout_recovery_weight > 0 so generated attractors can be observed"
                    ));
                }
                if !structured_contrast_enabled && !field_binding_contrast_enabled {
                    return Err(anyhow!(
                        "training.ruliad_supervision.verifier_reward.generated_attractor_replay_capacity requires structured_contrast_weight > 0 or field_binding_contrast_weight > 0 so generated attractors can be replayed as negatives"
                    ));
                }
                if verifier_reward.generated_attractor_replay_min_count == 0 {
                    return Err(anyhow!(
                        "training.ruliad_supervision.verifier_reward.generated_attractor_replay_min_count must be positive when generated_attractor_replay_capacity > 0"
                    ));
                }
                if verifier_reward.generated_attractor_replay_max_candidates == 0 {
                    return Err(anyhow!(
                        "training.ruliad_supervision.verifier_reward.generated_attractor_replay_max_candidates must be positive when generated_attractor_replay_capacity > 0"
                    ));
                }
                if verifier_reward.generated_attractor_replay_min_distinct_answers == 0 {
                    return Err(anyhow!(
                        "training.ruliad_supervision.verifier_reward.generated_attractor_replay_min_distinct_answers must be positive when generated_attractor_replay_capacity > 0"
                    ));
                }
                if !verifier_reward
                    .generated_attractor_replay_max_dominant_fraction
                    .is_finite()
                    || verifier_reward.generated_attractor_replay_max_dominant_fraction <= 0.0
                    || verifier_reward.generated_attractor_replay_max_dominant_fraction > 1.0
                {
                    return Err(anyhow!(
                        "training.ruliad_supervision.verifier_reward.generated_attractor_replay_max_dominant_fraction must be finite and in (0, 1] when generated_attractor_replay_capacity > 0"
                    ));
                }
            }
            if !verifier_reward.rollout_imitation_weight.is_finite()
                || verifier_reward.rollout_imitation_weight < 0.0
            {
                return Err(anyhow!(
                    "training.ruliad_supervision.verifier_reward.rollout_imitation_weight must be finite and non-negative"
                ));
            }
            if !verifier_reward.rollout_recovery_weight.is_finite()
                || verifier_reward.rollout_recovery_weight < 0.0
            {
                return Err(anyhow!(
                    "training.ruliad_supervision.verifier_reward.rollout_recovery_weight must be finite and non-negative"
                ));
            }
            if rollout_imitation_enabled {
                if verifier_reward.rollout_imitation_every_steps == 0 {
                    return Err(anyhow!(
                        "training.ruliad_supervision.verifier_reward.rollout_imitation_every_steps must be positive when rollout_imitation_weight > 0"
                    ));
                }
                if verifier_reward.rollout_imitation_min_partial_progress_ppm > 1_000_000 {
                    return Err(anyhow!(
                        "training.ruliad_supervision.verifier_reward.rollout_imitation_min_partial_progress_ppm must be <= 1000000"
                    ));
                }
                if verifier_reward.rollout_imitation_min_completion_quality_ppm > 1_000_000 {
                    return Err(anyhow!(
                        "training.ruliad_supervision.verifier_reward.rollout_imitation_min_completion_quality_ppm must be <= 1000000"
                    ));
                }
                if verifier_reward.rollout_imitation_min_verifier_rate_ppm > 1_000_000 {
                    return Err(anyhow!(
                        "training.ruliad_supervision.verifier_reward.rollout_imitation_min_verifier_rate_ppm must be <= 1000000"
                    ));
                }
                if verifier_reward.rollout_imitation_max_schema_wrong_rate_ppm > 1_000_000 {
                    return Err(anyhow!(
                        "training.ruliad_supervision.verifier_reward.rollout_imitation_max_schema_wrong_rate_ppm must be <= 1000000"
                    ));
                }
                if verifier_reward.rollout_imitation_max_malformed_rate_ppm > 1_000_000 {
                    return Err(anyhow!(
                        "training.ruliad_supervision.verifier_reward.rollout_imitation_max_malformed_rate_ppm must be <= 1000000"
                    ));
                }
                if verifier_reward.rollout_imitation_max_rows_per_step == 0 {
                    return Err(anyhow!(
                        "training.ruliad_supervision.verifier_reward.rollout_imitation_max_rows_per_step must be positive when rollout_imitation_weight > 0"
                    ));
                }
            }
            if !policy_reward_enabled
                && !structured_contrast_enabled
                && !field_binding_contrast_enabled
                && !rollout_imitation_enabled
                && !generated_attractor_replay_enabled
            {
                return Err(anyhow!(
                    "training.ruliad_supervision.verifier_reward.enabled requires verifier_reward.weight > 0, structured_contrast_weight > 0, field_binding_contrast_weight > 0, rollout_imitation_weight > 0, rollout_recovery_weight > 0, or generated_attractor_replay_capacity > 0"
                ));
            }
            if policy_reward_enabled {
                if verifier_reward.group_size < 2 {
                    return Err(anyhow!(
                        "training.ruliad_supervision.verifier_reward.group_size must be at least 2"
                    ));
                }
                if verifier_reward.every_steps == 0 {
                    return Err(anyhow!(
                        "training.ruliad_supervision.verifier_reward.every_steps must be positive when verifier_reward.weight > 0"
                    ));
                }
                if !verifier_reward.temperature.is_finite() || verifier_reward.temperature <= 0.0 {
                    return Err(anyhow!(
                        "training.ruliad_supervision.verifier_reward.temperature must be finite and positive"
                    ));
                }
                if verifier_reward.top_k == 0 {
                    return Err(anyhow!(
                        "training.ruliad_supervision.verifier_reward.top_k must be positive"
                    ));
                }
                if !verifier_reward.kl_weight.is_finite() || verifier_reward.kl_weight < 0.0 {
                    return Err(anyhow!(
                        "training.ruliad_supervision.verifier_reward.kl_weight must be finite and non-negative"
                    ));
                }
                if !verifier_reward.clip_range.is_finite() || verifier_reward.clip_range <= 0.0 {
                    return Err(anyhow!(
                        "training.ruliad_supervision.verifier_reward.clip_range must be finite and positive"
                    ));
                }
                if let Some(max_clip_fraction) = verifier_reward.max_advantage_clip_fraction
                    && (!max_clip_fraction.is_finite() || !(0.0..=1.0).contains(&max_clip_fraction))
                {
                    return Err(anyhow!(
                        "training.ruliad_supervision.verifier_reward.max_advantage_clip_fraction must be finite and in [0, 1] when set"
                    ));
                }
                if verifier_reward.positive_advantage_min_partial_progress_ppm > 1_000_000 {
                    return Err(anyhow!(
                        "training.ruliad_supervision.verifier_reward.positive_advantage_min_partial_progress_ppm must be <= 1000000"
                    ));
                }
                if verifier_reward.positive_advantage_min_completion_quality_ppm > 1_000_000 {
                    return Err(anyhow!(
                        "training.ruliad_supervision.verifier_reward.positive_advantage_min_completion_quality_ppm must be <= 1000000"
                    ));
                }
                if !verifier_reward.advantage_epsilon.is_finite()
                    || verifier_reward.advantage_epsilon <= 0.0
                {
                    return Err(anyhow!(
                        "training.ruliad_supervision.verifier_reward.advantage_epsilon must be finite and positive"
                    ));
                }
                if matches!(
                    verifier_reward.mode,
                    RuliadVerifierRewardMode::VpoIndependent
                ) {
                    if verifier_reward.vpo_scalarizations == 0 {
                        return Err(anyhow!(
                            "training.ruliad_supervision.verifier_reward.vpo_scalarizations must be positive when mode=\"vpo_independent\""
                        ));
                    }
                    if !verifier_reward.vpo_correctness_mass_floor.is_finite()
                        || !(0.0..=1.0).contains(&verifier_reward.vpo_correctness_mass_floor)
                    {
                        return Err(anyhow!(
                            "training.ruliad_supervision.verifier_reward.vpo_correctness_mass_floor must be finite and in [0, 1]"
                        ));
                    }
                    if !verifier_reward.vpo_completion_health_mass_floor.is_finite()
                        || !(0.0..=1.0).contains(&verifier_reward.vpo_completion_health_mass_floor)
                    {
                        return Err(anyhow!(
                            "training.ruliad_supervision.verifier_reward.vpo_completion_health_mass_floor must be finite and in [0, 1]"
                        ));
                    }
                    if !verifier_reward.vpo_schema_quality_mass_floor.is_finite()
                        || !(0.0..=1.0).contains(&verifier_reward.vpo_schema_quality_mass_floor)
                    {
                        return Err(anyhow!(
                            "training.ruliad_supervision.verifier_reward.vpo_schema_quality_mass_floor must be finite and in [0, 1]"
                        ));
                    }
                    if verifier_reward.vpo_correctness_mass_floor
                        + verifier_reward.vpo_completion_health_mass_floor
                        + verifier_reward.vpo_schema_quality_mass_floor
                        > 1.0 + f32::EPSILON
                    {
                        return Err(anyhow!(
                            "training.ruliad_supervision.verifier_reward VPO mass floors must sum to <= 1"
                        ));
                    }
                    if !verifier_reward.vpo_compactness_max_weight.is_finite()
                        || !(0.0..=1.0).contains(&verifier_reward.vpo_compactness_max_weight)
                    {
                        return Err(anyhow!(
                            "training.ruliad_supervision.verifier_reward.vpo_compactness_max_weight must be finite and in [0, 1]"
                        ));
                    }
                }
            }
            let reward_weights = verifier_reward.reward;
            for (field, value) in [
                ("verifier_match", reward_weights.verifier_match),
                ("semantic_match", reward_weights.semantic_match),
                ("partial_progress", reward_weights.partial_progress),
                ("field_accuracy", reward_weights.field_accuracy),
                ("certificate_prefix", reward_weights.certificate_prefix),
                ("compactness", reward_weights.compactness),
                ("malformed_penalty", reward_weights.malformed_penalty),
                ("missing_penalty", reward_weights.missing_penalty),
                ("schema_wrong_penalty", reward_weights.schema_wrong_penalty),
                (
                    "hash_canary_wrong_penalty",
                    reward_weights.hash_canary_wrong_penalty,
                ),
            ] {
                if !value.is_finite() {
                    return Err(anyhow!(
                        "training.ruliad_supervision.verifier_reward.reward.{field} must be finite"
                    ));
                }
            }
            if !matches!(
                self.dataset.source,
                DatasetSourceConfig::UniversalityRuliad { .. }
            ) {
                return Err(anyhow!(
                    "training.ruliad_supervision.verifier_reward.enabled requires dataset.type=\"universality_ruliad\""
                ));
            }
            if self.parallel.pipeline.enabled {
                return Err(anyhow!(
                    "training.ruliad_supervision.verifier_reward.enabled does not yet support parallel.pipeline.enabled"
                ));
            }
            if policy_reward_enabled && self.training.tbptt_chunk_size.is_some() {
                return Err(anyhow!(
                    "training.ruliad_supervision.verifier_reward.weight > 0 does not yet support training.tbptt_chunk_size"
                ));
            }
            if policy_reward_enabled && self.training.tbptt_persist_across_steps {
                return Err(anyhow!(
                    "training.ruliad_supervision.verifier_reward.weight > 0 does not yet support training.tbptt_persist_across_steps"
                ));
            }
            if structured_contrast_enabled && self.training.tbptt_chunk_size.is_some() {
                return Err(anyhow!(
                    "training.ruliad_supervision.verifier_reward.structured_contrast_weight > 0 does not yet support training.tbptt_chunk_size"
                ));
            }
            if structured_contrast_enabled && self.training.tbptt_persist_across_steps {
                return Err(anyhow!(
                    "training.ruliad_supervision.verifier_reward.structured_contrast_weight > 0 does not yet support training.tbptt_persist_across_steps"
                ));
            }
            if rollout_imitation_enabled && self.training.tbptt_chunk_size.is_some() {
                return Err(anyhow!(
                    "training.ruliad_supervision.verifier_reward.rollout_imitation_weight > 0 does not yet support training.tbptt_chunk_size"
                ));
            }
            if rollout_imitation_enabled && self.training.tbptt_persist_across_steps {
                return Err(anyhow!(
                    "training.ruliad_supervision.verifier_reward.rollout_imitation_weight > 0 does not yet support training.tbptt_persist_across_steps"
                ));
            }
            if !self.training.objective.is_next_token() {
                return Err(anyhow!(
                    "training.ruliad_supervision.verifier_reward.enabled currently supports only the next-token training objective"
                ));
            }
        }
        if self.training.ruliad_supervision.uses_target_loss_mask()
            && !matches!(
                self.dataset.source,
                DatasetSourceConfig::UniversalityRuliad { .. }
            )
        {
            return Err(anyhow!(
                "training.ruliad_supervision.mode={:?} requires dataset.type=\"universality_ruliad\"",
                self.training.ruliad_supervision.mode
            ));
        }
        if self.training.source_selection_state_path.is_some()
            && !matches!(
                self.dataset.source,
                DatasetSourceConfig::UniversalityRuliad { .. }
            )
        {
            return Err(anyhow!(
                "training.source_selection_state_path requires dataset.type=\"universality_ruliad\""
            ));
        }
        if !(0.0 < self.dataset.train_split_ratio && self.dataset.train_split_ratio <= 1.0) {
            return Err(anyhow!(
                "dataset.train_split_ratio must be in (0, 1] (got {})",
                self.dataset.train_split_ratio
            ));
        }
        if let Some(validation) = &self.dataset.validation
            && let Some(train_split_ratio) = validation.train_split_ratio
            && !(0.0 < train_split_ratio && train_split_ratio <= 1.0)
        {
            return Err(anyhow!(
                "dataset.validation.train_split_ratio must be in (0, 1] when set (got {})",
                train_split_ratio
            ));
        }
        if let Some(max_tokens) = self.generation.max_tokens
            && max_tokens <= 0
        {
            return Err(anyhow!("generation.max_tokens must be > 0"));
        }
        if self.generation.temperature <= 0.0 {
            return Err(anyhow!("generation.temperature must be > 0"));
        }
        if let Some(top_k) = self.generation.top_k
            && top_k == 0
        {
            return Err(anyhow!("generation.top_k must be > 0"));
        }

        validate_dataset_source(
            &self.dataset.source,
            &self.dataset.tokenizer.kind,
            false,
            "dataset",
        )?;
        if let Some(validation) = &self.dataset.validation {
            validate_dataset_source(
                &validation.source,
                &self.dataset.tokenizer.kind,
                true,
                "dataset.validation",
            )?;
        }

        if let Some(gdpo) = &self.training.gdpo
            && gdpo.enabled
        {
            return Err(anyhow!(
                "training.gdpo.enabled is a legacy objective flag and cannot be combined with training.objective; use training.objective.type = \"sdpo\" for self-distilled policy optimization"
            ));
        }

        validate_training_objective_config(&self.training.objective)?;

        if let Some(n_layer) = self.model.n_layer
            && n_layer == 0
        {
            return Err(anyhow!("model.n_layer must be > 0 when set"));
        }
        if let Some(n_embd) = self.model.n_embd
            && n_embd == 0
        {
            return Err(anyhow!("model.n_embd must be > 0 when set"));
        }
        if let Some(n_head) = self.model.n_head
            && n_head == 0
        {
            return Err(anyhow!("model.n_head must be > 0 when set"));
        }
        let mut resolved_model = DragonConfig::default();
        if let Some(language_head) = &self.model.language_head {
            resolved_model.language_head = language_head.clone();
        }
        if let Some(tie_input_output_embeddings) = self.model.tie_input_output_embeddings {
            resolved_model.tie_input_output_embeddings = tie_input_output_embeddings;
        }
        if resolved_model.tie_input_output_embeddings
            && !matches!(
                resolved_model.language_head,
                LanguageHeadConfig::StandardTokenClassification
            )
        {
            return Err(anyhow!(
                "model.tie_input_output_embeddings requires model.language_head.type = \"standard_token_classification\""
            ));
        }
        if let Some(n_layer) = self.model.n_layer {
            resolved_model.n_layer = n_layer;
        }
        if let Some(n_embd) = self.model.n_embd {
            resolved_model.n_embd = n_embd;
        }
        if let Some(n_head) = self.model.n_head {
            resolved_model.n_head = n_head;
        }
        if let Some(multiplier) = self.model.mlp_internal_dim_multiplier
            && multiplier == 0
        {
            return Err(anyhow!(
                "model.mlp_internal_dim_multiplier must be > 0 when set"
            ));
        }
        if let Some(multiplier) = self.model.mlp_internal_dim_multiplier {
            resolved_model.mlp_internal_dim_multiplier = multiplier;
        }
        if let Some(latent_total) = self.model.latent_total {
            if latent_total == 0 {
                return Err(anyhow!("model.latent_total must be > 0 when set"));
            }
            let resolved_n_embd = resolved_model.n_embd;
            if latent_total % resolved_n_embd != 0 {
                return Err(anyhow!(
                    "model.latent_total must be divisible by model.n_embd (got latent_total={} n_embd={})",
                    latent_total,
                    resolved_n_embd
                ));
            }
            if let Some(multiplier) = self.model.mlp_internal_dim_multiplier
                && multiplier * resolved_n_embd != latent_total
            {
                return Err(anyhow!(
                    "model.latent_total and model.mlp_internal_dim_multiplier disagree (latent_total={} n_embd={} multiplier={})",
                    latent_total,
                    resolved_n_embd,
                    multiplier
                ));
            }
            resolved_model.mlp_internal_dim_multiplier = latent_total / resolved_model.n_embd;
        }
        if let Some(initialization) = &self.model.initialization {
            initialization.validate().map_err(anyhow::Error::msg)?;
            resolved_model.initialization = initialization.clone();
        }
        if let Some(random_scaffold) = &self.model.random_scaffold {
            random_scaffold
                .validate_for_model(
                    resolved_model.n_embd,
                    resolved_model.n_head,
                    resolved_model.latent_total(),
                )
                .map_err(|message| anyhow!("model.random_scaffold {message}"))?;
            resolved_model.random_scaffold = random_scaffold.clone();
        }
        if let Some(sequence_kernel) = self.model.sequence_kernel {
            sequence_kernel
                .validate()
                .map_err(|message| anyhow!("model.sequence_kernel {message}"))?;
            resolved_model.sequence_kernel = sequence_kernel;
        }
        if let Some(sequence_kernel) = self.training.sequence_kernel_override {
            sequence_kernel
                .validate()
                .map_err(|message| anyhow!("training.sequence_kernel_override {message}"))?;
        }
        if let Some(mamba) = &self.model.mamba {
            let memory_system = self
                .training
                .sequence_kernel_override
                .unwrap_or(
                    self.model
                        .sequence_kernel
                        .unwrap_or(resolved_model.sequence_kernel),
                )
                .memory_system;
            mamba
                .validate(memory_system, resolved_model.n_embd)
                .map_err(|message| anyhow!("model.mamba {message}"))?;
            resolved_model.mamba = mamba.clone();
        }
        if let Some(gated_deltanet2) = &self.model.gated_deltanet2 {
            gated_deltanet2
                .validate(
                    resolved_model.n_head,
                    resolved_model.n_embd,
                    resolved_model.latent_per_head(),
                )
                .map_err(|message| anyhow!("model.gated_deltanet2 {message}"))?;
            resolved_model.gated_deltanet2 = gated_deltanet2.clone();
        }
        if matches!(
            self.training
                .sequence_kernel_override
                .unwrap_or(resolved_model.sequence_kernel)
                .memory_system,
            burn_dragon_core::SequenceMemorySystem::Mamba3StateSpaceDuality
        ) {
            resolved_model
                .mamba
                .validate(
                    resolved_model.sequence_kernel.memory_system,
                    resolved_model.n_embd,
                )
                .map_err(|message| anyhow!("resolved model.mamba {message}"))?;
        }
        if matches!(
            self.training
                .sequence_kernel_override
                .unwrap_or(resolved_model.sequence_kernel)
                .memory_system,
            burn_dragon_core::SequenceMemorySystem::GatedDeltaNet2
        ) {
            resolved_model
                .gated_deltanet2
                .validate(
                    resolved_model.n_head,
                    resolved_model.n_embd,
                    resolved_model.latent_per_head(),
                )
                .map_err(|message| anyhow!("resolved model.gated_deltanet2 {message}"))?;
        }
        if resolved_model.latent_total() % self.parallel.tensor.size != 0 {
            return Err(anyhow!(
                "resolved model.latent_total must be divisible by parallel.tensor.size (got latent_total={} tensor_size={})",
                resolved_model.latent_total(),
                self.parallel.tensor.size
            ));
        }
        if self.training.dynamics_anchor.enabled
            && !self.training.auto_batch_size.enabled
            && self.training.batch_size > 1
            && resolved_model.latent_total() >= 65_536
        {
            return Err(anyhow!(
                "training.dynamics_anchor with fixed training.batch_size > 1 is not allowed for resolved model.latent_total >= 65536; use training.batch_size=1 or enable training.auto_batch_size"
            ));
        }
        if self.training.predictive_coding.enabled
            && !self.training.auto_batch_size.enabled
            && self.training.batch_size > 1
            && resolved_model.latent_total() >= 16_384
        {
            return Err(anyhow!(
                "training.predictive_coding with fixed training.batch_size > 1 is not allowed for resolved model.latent_total >= 16384; use training.batch_size=1 or enable training.auto_batch_size"
            ));
        }
        if self.training.neuron_scaling.enabled {
            if resolved_model.random_scaffold.enabled {
                return Err(anyhow!(
                    "training.neuron_scaling is not compatible with model.random_scaffold.enabled; scaffold growth history must be represented by a new model revision"
                ));
            }
            let max_latent_total = self.training.neuron_scaling.max_latent_total;
            if max_latent_total < resolved_model.latent_total() {
                return Err(anyhow!(
                    "training.neuron_scaling.max_latent_total must be >= resolved model.latent_total (got max={} current={})",
                    max_latent_total,
                    resolved_model.latent_total()
                ));
            }
            if !max_latent_total.is_multiple_of(resolved_model.n_embd) {
                return Err(anyhow!(
                    "training.neuron_scaling.max_latent_total must be divisible by model.n_embd (got max={} n_embd={})",
                    max_latent_total,
                    resolved_model.n_embd
                ));
            }
            if !max_latent_total.is_multiple_of(resolved_model.n_head) {
                return Err(anyhow!(
                    "training.neuron_scaling.max_latent_total must be divisible by model.n_head (got max={} n_head={})",
                    max_latent_total,
                    resolved_model.n_head
                ));
            }
        }
        if resolved_model.random_scaffold.enabled
            && matches!(
                self.optimizer.name,
                burn_dragon_train::OptimizerKind::Eggroll
            )
        {
            return Err(anyhow!(
                "optimizer.name=eggroll does not yet support adapter-parameter population evaluation for model.random_scaffold; use optimizer.name=adamw for the paper-faithful local baseline"
            ));
        }
        if resolved_model.random_scaffold.enabled && self.training.continual_backprop.enabled {
            return Err(anyhow!(
                "training.continual_backprop is not compatible with model.random_scaffold because feature replacement mutates the immutable scaffold"
            ));
        }
        if resolved_model.random_scaffold.enabled && self.training.init_checkpoint_path.is_some() {
            return Err(anyhow!(
                "training.init_checkpoint_path transfer is not supported for model.random_scaffold; resume an exact scaffold checkpoint or start a fresh revision"
            ));
        }
        if matches!(
            self.parallel.tensor.partition,
            TensorParallelPartitionKind::HeadAligned
        ) && self.parallel.tensor.size > resolved_model.n_head
        {
            return Err(anyhow!(
                "parallel.tensor.partition=head_aligned requires parallel.tensor.size <= model.n_head (got tensor_size={} n_head={})",
                self.parallel.tensor.size,
                resolved_model.n_head
            ));
        }
        if let Some(schedule) = &self.model.latent_fanout_schedule
            && let Err(message) = resolved_model.validate_latent_fanout_schedule(schedule)
        {
            return Err(anyhow!(message));
        }
        if let Some(latent_reasoning) = &self.model.latent_reasoning {
            latent_reasoning.validate().map_err(anyhow::Error::msg)?;
            resolved_model.latent_reasoning = latent_reasoning.clone();
        }
        if let Some(next_latent_transition) = &self.model.next_latent_transition {
            next_latent_transition
                .validate()
                .map_err(anyhow::Error::msg)?;
            resolved_model.next_latent_transition = next_latent_transition.clone();
        }
        if let Some(hierarchical_dragon) = &self.model.hierarchical_dragon {
            resolved_model.hierarchical_dragon = hierarchical_dragon.clone();
            if hierarchical_dragon.enabled {
                if matches!(hierarchical_dragon.last_layers, Some(0)) {
                    return Err(anyhow!(
                        "model.hierarchical_dragon.last_layers must be > 0 when set"
                    ));
                }
                if hierarchical_dragon.fast_cycles == 0 {
                    return Err(anyhow!(
                        "model.hierarchical_dragon.fast_cycles must be > 0 when enabled"
                    ));
                }
                if hierarchical_dragon.slow_cycles == 0 {
                    return Err(anyhow!(
                        "model.hierarchical_dragon.slow_cycles must be > 0 when enabled"
                    ));
                }
                if !hierarchical_dragon.slow_to_fast_scale.is_finite()
                    || hierarchical_dragon.slow_to_fast_scale < 0.0
                {
                    return Err(anyhow!(
                        "model.hierarchical_dragon.slow_to_fast_scale must be finite and >= 0 when enabled"
                    ));
                }
                if !hierarchical_dragon.fast_to_slow_scale.is_finite()
                    || hierarchical_dragon.fast_to_slow_scale < 0.0
                {
                    return Err(anyhow!(
                        "model.hierarchical_dragon.fast_to_slow_scale must be finite and >= 0 when enabled"
                    ));
                }
                if resolved_model.y_neuron_recurrence.enabled {
                    return Err(anyhow!(
                        "model.hierarchical_dragon is not yet supported together with model.y_neuron_recurrence"
                    ));
                }
                if resolved_model.clocked_slow_memory.enabled {
                    return Err(anyhow!(
                        "model.hierarchical_dragon is not supported together with model.clocked_slow_memory"
                    ));
                }
                if self.parallel.pipeline.enabled {
                    return Err(anyhow!(
                        "model.hierarchical_dragon is not yet supported together with parallel.pipeline.enabled"
                    ));
                }
            }
        }
        let latent_jepa_can_run = self.training.latent_reasoning.enabled
            && self
                .training
                .latent_reasoning
                .jepa_future_offsets
                .iter()
                .any(|offset| *offset < self.training.block_size);
        if latent_jepa_can_run && !resolved_model.latent_reasoning.enabled {
            return Err(anyhow!(
                "training.latent_reasoning JEPA offsets within training.block_size require model.latent_reasoning.enabled"
            ));
        }
        if !self.training.latent_reasoning.eval_step_sweep.is_empty()
            && !resolved_model.latent_reasoning.enabled
        {
            return Err(anyhow!(
                "training.latent_reasoning.eval_step_sweep requires model.latent_reasoning.enabled"
            ));
        }
        if self.training.latent_reasoning.enabled
            && self.training.latent_reasoning.next_latent.enabled
            && !resolved_model.next_latent_transition.enabled
        {
            return Err(anyhow!(
                "training.latent_reasoning.next_latent.enabled requires model.next_latent_transition.enabled"
            ));
        }
        if self.training.latent_reasoning.enabled
            && self.training.latent_reasoning.energy_model.enabled
        {
            if !resolved_model.latent_reasoning.enabled {
                return Err(anyhow!(
                    "training.latent_reasoning.energy_model.enabled requires model.latent_reasoning.enabled"
                ));
            }
            if !resolved_model.latent_reasoning.energy_head {
                return Err(anyhow!(
                    "training.latent_reasoning.energy_model.enabled requires model.latent_reasoning.energy_head=true"
                ));
            }
        }
        if self.training.latent_reasoning.enabled
            && self.training.latent_reasoning.step_contract.enabled
            && !resolved_model.latent_reasoning.enabled
        {
            return Err(anyhow!(
                "training.latent_reasoning.step_contract.enabled requires model.latent_reasoning.enabled"
            ));
        }
        if self.training.latent_reasoning.enabled
            && self.training.latent_reasoning.next_latent.enabled
            && self.training.latent_reasoning.next_latent.token_kl_weight > f32::EPSILON
            && !resolved_model.language_head.uses_flat_token_logits()
        {
            return Err(anyhow!(
                "training.latent_reasoning.next_latent.token_kl_weight requires a flat token language head"
            ));
        }
        if self.training.latent_reasoning.enabled
            && self.training.latent_reasoning.step_contract.enabled
            && self.training.latent_reasoning.step_contract.token_kl_weight > f32::EPSILON
            && !resolved_model.language_head.uses_flat_token_logits()
        {
            return Err(anyhow!(
                "training.latent_reasoning.step_contract.token_kl_weight requires a flat token language head"
            ));
        }
        if let Some(dropout) = self.model.dropout
            && dropout < 0.0
        {
            return Err(anyhow!("model.dropout must be >= 0"));
        }
        if let Some(block_size) = self.model.block_size
            && block_size == 0
        {
            return Err(anyhow!("model.block_size must be > 0 when set"));
        }
        if let Some(rollout_fast_steps) = self.model.rollout_fast_steps_per_slow_step
            && !DragonConfig::is_valid_rollout_fast_steps(rollout_fast_steps)
        {
            return Err(anyhow!(
                "model.rollout_fast_steps_per_slow_step must be one of {:?} when set (got {})",
                DragonConfig::SUPPORTED_ROLLOUT_FAST_STEPS,
                rollout_fast_steps
            ));
        }
        if let Some(y_neuron_recurrence) = &self.model.y_neuron_recurrence
            && y_neuron_recurrence.enabled
        {
            if y_neuron_recurrence.carry_in_scale < 0.0 {
                return Err(anyhow!(
                    "model.y_neuron_recurrence.carry_in_scale must be >= 0 when enabled"
                ));
            }
            if matches!(y_neuron_recurrence.last_layers, Some(0)) {
                return Err(anyhow!(
                    "model.y_neuron_recurrence.last_layers must be > 0 when set"
                ));
            }
            if y_neuron_recurrence.chunk_tokens == 0 {
                return Err(anyhow!(
                    "model.y_neuron_recurrence.chunk_tokens must be > 0 when enabled"
                ));
            }
            if !(0.0..=1.0).contains(&y_neuron_recurrence.state_decay) {
                return Err(anyhow!(
                    "model.y_neuron_recurrence.state_decay must be in [0, 1] when enabled"
                ));
            }
            if y_neuron_recurrence.state_update_scale <= 0.0 {
                return Err(anyhow!(
                    "model.y_neuron_recurrence.state_update_scale must be > 0 when enabled"
                ));
            }
            if matches!(y_neuron_recurrence.state_rms_cap, Some(value) if value <= 0.0) {
                return Err(anyhow!(
                    "model.y_neuron_recurrence.state_rms_cap must be > 0 when set"
                ));
            }
        }
        if let Some(clocked_slow_memory) = &self.model.clocked_slow_memory
            && clocked_slow_memory.enabled
        {
            if matches!(clocked_slow_memory.last_layers, Some(0)) {
                return Err(anyhow!(
                    "model.clocked_slow_memory.last_layers must be > 0 when set"
                ));
            }
            if clocked_slow_memory.chunk_tokens == 0 {
                return Err(anyhow!(
                    "model.clocked_slow_memory.chunk_tokens must be > 0 when enabled"
                ));
            }
            if clocked_slow_memory.residual_scale <= 0.0 {
                return Err(anyhow!(
                    "model.clocked_slow_memory.residual_scale must be > 0 when enabled"
                ));
            }
            if matches!(self.model.y_neuron_recurrence.as_ref(), Some(value) if value.enabled) {
                return Err(anyhow!(
                    "model.clocked_slow_memory is not yet supported together with model.y_neuron_recurrence"
                ));
            }
        }
        if let Some(summary_memory) = &self.model.summary_memory
            && summary_memory.enabled
        {
            if matches!(summary_memory.last_layers, Some(0)) {
                return Err(anyhow!(
                    "model.summary_memory.last_layers must be > 0 when set"
                ));
            }
            if summary_memory.chunk_tokens == 0 {
                return Err(anyhow!(
                    "model.summary_memory.chunk_tokens must be > 0 when enabled"
                ));
            }
            if summary_memory.residual_scale <= 0.0 {
                return Err(anyhow!(
                    "model.summary_memory.residual_scale must be > 0 when enabled"
                ));
            }
            if !(0.0..=1.0).contains(&summary_memory.state_decay) {
                return Err(anyhow!(
                    "model.summary_memory.state_decay must be in [0, 1] when enabled"
                ));
            }
            if summary_memory.state_update_scale <= 0.0 {
                return Err(anyhow!(
                    "model.summary_memory.state_update_scale must be > 0 when enabled"
                ));
            }
            if summary_memory.surprise_gate_threshold < 0.0 {
                return Err(anyhow!(
                    "model.summary_memory.surprise_gate_threshold must be >= 0 when enabled"
                ));
            }
            if summary_memory.surprise_gate_sharpness <= 0.0 {
                return Err(anyhow!(
                    "model.summary_memory.surprise_gate_sharpness must be > 0 when enabled"
                ));
            }
            if matches!(
                summary_memory.write_trigger_text.as_ref(),
                Some(value) if value.trim().is_empty()
            ) {
                return Err(anyhow!(
                    "model.summary_memory.write_trigger_text must not be empty when set"
                ));
            }
            if matches!(
                summary_memory.write_trigger_token_ids.as_ref(),
                Some(value) if value.is_empty()
            ) {
                return Err(anyhow!(
                    "model.summary_memory.write_trigger_token_ids must not be empty when set"
                ));
            }
            if matches!(self.model.y_neuron_recurrence.as_ref(), Some(value) if value.enabled) {
                return Err(anyhow!(
                    "model.summary_memory is not yet supported together with model.y_neuron_recurrence"
                ));
            }
        }
        if let Some(mhc) = &self.model.mhc
            && mhc.enabled
        {
            if mhc.num_streams == 0 {
                return Err(anyhow!("model.mhc.num_streams must be > 0 when enabled"));
            }
            if mhc.num_views == 0 {
                return Err(anyhow!("model.mhc.num_views must be > 0 when enabled"));
            }
            if matches!(mhc.last_layers, Some(0)) {
                return Err(anyhow!("model.mhc.last_layers must be > 0 when set"));
            }
            if mhc.mhc_tau <= 0.0 {
                return Err(anyhow!("model.mhc.mhc_tau must be > 0 when enabled"));
            }
        }
        if let Some(attention_residual) = &self.model.attention_residual
            && attention_residual.enabled
        {
            if attention_residual.num_heads == 0 {
                return Err(anyhow!(
                    "model.attention_residual.num_heads must be > 0 when enabled"
                ));
            }
            if matches!(attention_residual.last_layers, Some(0)) {
                return Err(anyhow!(
                    "model.attention_residual.last_layers must be > 0 when set"
                ));
            }
            if matches!(attention_residual.history_window, Some(0)) {
                return Err(anyhow!(
                    "model.attention_residual.history_window must be > 0 when set"
                ));
            }
        }
        if let Some(block_attention_residual) = &self.model.block_attention_residual
            && block_attention_residual.enabled
        {
            if block_attention_residual.num_heads == 0 {
                return Err(anyhow!(
                    "model.block_attention_residual.num_heads must be > 0 when enabled"
                ));
            }
            if matches!(block_attention_residual.last_layers, Some(0)) {
                return Err(anyhow!(
                    "model.block_attention_residual.last_layers must be > 0 when set"
                ));
            }
            if block_attention_residual.layers_per_block == 0 {
                return Err(anyhow!(
                    "model.block_attention_residual.layers_per_block must be > 0 when enabled"
                ));
            }
            if matches!(block_attention_residual.block_history_window, Some(0)) {
                return Err(anyhow!(
                    "model.block_attention_residual.block_history_window must be > 0 when set"
                ));
            }
            if matches!(block_attention_residual.intra_block_history_window, Some(0)) {
                return Err(anyhow!(
                    "model.block_attention_residual.intra_block_history_window must be > 0 when set"
                ));
            }
        }
        if let Some(mhc) = self.model.mhc.as_ref()
            && mhc.enabled
            && self.model.residual_connector != Some(ResidualConnectorKind::Mhc)
        {
            return Err(anyhow!(
                "model.residual_connector = \"mhc\" is required when model.mhc.enabled = true"
            ));
        }
        if let Some(attention_residual) = self.model.attention_residual.as_ref()
            && attention_residual.enabled
            && self.model.residual_connector != Some(ResidualConnectorKind::AttentionResidual)
        {
            return Err(anyhow!(
                "model.residual_connector = \"attention_residual\" is required when model.attention_residual.enabled = true"
            ));
        }
        if let Some(block_attention_residual) = self.model.block_attention_residual.as_ref()
            && block_attention_residual.enabled
            && self.model.residual_connector != Some(ResidualConnectorKind::BlockAttentionResidual)
        {
            return Err(anyhow!(
                "model.residual_connector = \"block_attention_residual\" is required when model.block_attention_residual.enabled = true"
            ));
        }
        if let Some(residual_connector) = self.model.residual_connector {
            match residual_connector {
                ResidualConnectorKind::Vanilla => {}
                ResidualConnectorKind::Mhc => {
                    let mhc = self.model.mhc.as_ref().ok_or_else(|| {
                        anyhow!("model.mhc must be set when model.residual_connector = \"mhc\"")
                    })?;
                    if !mhc.enabled {
                        return Err(anyhow!(
                            "model.mhc.enabled must be true when model.residual_connector = \"mhc\""
                        ));
                    }
                }
                ResidualConnectorKind::AttentionResidual => {
                    let attention_residual = self
                        .model
                        .attention_residual
                        .as_ref()
                        .ok_or_else(|| anyhow!("model.attention_residual must be set when model.residual_connector = \"attention_residual\""))?;
                    if !attention_residual.enabled {
                        return Err(anyhow!(
                            "model.attention_residual.enabled must be true when model.residual_connector = \"attention_residual\""
                        ));
                    }
                }
                ResidualConnectorKind::BlockAttentionResidual => {
                    let block_attention_residual = self
                        .model
                        .block_attention_residual
                        .as_ref()
                        .ok_or_else(|| anyhow!("model.block_attention_residual must be set when model.residual_connector = \"block_attention_residual\""))?;
                    if !block_attention_residual.enabled {
                        return Err(anyhow!(
                            "model.block_attention_residual.enabled must be true when model.residual_connector = \"block_attention_residual\""
                        ));
                    }
                }
            }
        }

        if let Some(schedule) = &self.optimizer.lr_schedule {
            match schedule {
                LearningRateScheduleConfig::Constant { initial_lr }
                | LearningRateScheduleConfig::Cosine { initial_lr, .. }
                | LearningRateScheduleConfig::Linear { initial_lr, .. }
                | LearningRateScheduleConfig::Exponential { initial_lr, .. }
                | LearningRateScheduleConfig::Step { initial_lr, .. }
                | LearningRateScheduleConfig::Noam { initial_lr, .. } => {
                    if matches!(initial_lr.as_ref(), Some(value) if *value <= 0.0) {
                        return Err(anyhow!("optimizer.lr_schedule.initial_lr must be > 0"));
                    }
                }
            }

            match schedule {
                LearningRateScheduleConfig::Cosine {
                    min_lr,
                    warmup_steps,
                    num_iters,
                    ..
                } => {
                    if matches!(min_lr.as_ref(), Some(value) if *value < 0.0) {
                        return Err(anyhow!("optimizer.lr_schedule.min_lr must be >= 0"));
                    }
                    if matches!(warmup_steps, Some(0)) {
                        return Err(anyhow!("optimizer.lr_schedule.warmup_steps must be > 0"));
                    }
                    if matches!(num_iters, Some(0)) {
                        return Err(anyhow!("optimizer.lr_schedule.num_iters must be > 0"));
                    }
                }
                LearningRateScheduleConfig::Linear {
                    final_lr,
                    num_iters,
                    ..
                } => {
                    if *final_lr < 0.0 {
                        return Err(anyhow!("optimizer.lr_schedule.final_lr must be >= 0"));
                    }
                    if matches!(num_iters, Some(0)) {
                        return Err(anyhow!("optimizer.lr_schedule.num_iters must be > 0"));
                    }
                }
                LearningRateScheduleConfig::Exponential { gamma, .. } => {
                    if *gamma <= 0.0 {
                        return Err(anyhow!("optimizer.lr_schedule.gamma must be > 0"));
                    }
                }
                LearningRateScheduleConfig::Step {
                    gamma, step_size, ..
                } => {
                    if *gamma <= 0.0 {
                        return Err(anyhow!("optimizer.lr_schedule.gamma must be > 0"));
                    }
                    if matches!(step_size, Some(0)) {
                        return Err(anyhow!("optimizer.lr_schedule.step_size must be > 0"));
                    }
                }
                LearningRateScheduleConfig::Noam {
                    warmup_steps,
                    model_size,
                    ..
                } => {
                    if matches!(warmup_steps, Some(0)) {
                        return Err(anyhow!("optimizer.lr_schedule.warmup_steps must be > 0"));
                    }
                    if matches!(model_size, Some(0)) {
                        return Err(anyhow!("optimizer.lr_schedule.model_size must be > 0"));
                    }
                }
                LearningRateScheduleConfig::Constant { .. } => {}
            }
        }

        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    use crate::config::TrainingObjectiveConfig;
    use crate::config::load_training_config;
    use crate::config::train::RuliadSupervisionMode;
    use crate::inference::build_model_config;
    use burn_dragon_core::{HierarchicalDragonSharing, RotaryEmbedding, SequenceMemorySystem};
    use burn_dragon_train::OptimizerKind;
    use std::path::{Path, PathBuf};

    fn parse_config(extra_training: &str) -> TrainingConfig {
        let toml = format!(
            r#"
[dataset]
cache_dir = "target/test-cache"
type = "nemotron_climb_mix"
max_records = 4

[training]
block_size = 8
batch_size = 2
max_iters = 1
log_frequency = 1
{extra_training}

[optimizer]
learning_rate = 0.001
weight_decay = 0.0

[generation]
prompt = ""
"#
        );
        toml::from_str(&toml).expect("training config should parse")
    }

    #[test]
    fn default_objective_is_next_token() {
        let config = parse_config("");
        assert!(config.training.objective.is_next_token());
        config.validate().expect("default objective validates");
    }

    #[test]
    fn explicit_streaming_batching_is_independent_of_state_persistence() {
        let config = parse_config("sequence_batching = \"streaming\"");
        config
            .validate()
            .expect("ordered streaming batches should support a reset-state control");
        assert!(
            config
                .training
                .sequence_batching
                .uses_streaming_loader(config.training.tbptt_persist_across_steps)
        );
    }

    #[test]
    fn persistent_state_rejects_random_batch_order() {
        let config = parse_config(
            "tbptt_chunk_size = 4\ntbptt_persist_across_steps = true\nsequence_batching = \"random\"",
        );
        let error = config
            .validate()
            .expect_err("persistent state cannot follow unrelated random windows");
        assert!(error.to_string().contains("sequence_batching=random"));
    }

    #[test]
    fn sequence_state_probe_supports_matched_stateless_and_persistent_arms() {
        let config = parse_config(
            "sequence_batching = \"streaming\"\n\n[training.sequence_state_probe]\nenabled = true\npaired_batches = 2\nmax_rho_slots = 8",
        );
        config
            .validate()
            .expect("stateless training should still support carried-state evaluation");
        let config = parse_config(
            "tbptt_chunk_size = 4\ntbptt_persist_across_steps = true\nsequence_batching = \"streaming\"\n\n[training.sequence_state_probe]\nenabled = true\npaired_batches = 2\nmax_rho_slots = 8",
        );
        config
            .validate()
            .expect("persistent stream carry diagnostics should validate");
    }

    fn external_evaluator_config() -> TrainingConfig {
        let mut config = parse_config("");
        config.training.validation.execution =
            crate::config::TrainingValidationExecution::ExternalEvaluator;
        config.training.gates.enabled = false;
        config.training.dynamics.enabled = false;
        config.training.neuron_scaling.enabled = false;
        config.training.events.ruliad_correctness_probe_items = 0;
        config.training.events.source_weighted_validation_batches = 0;
        config.training.ruliad_policy_probe.enabled = false;
        config
    }

    #[test]
    fn external_evaluator_contract_validates_when_local_consumers_are_disabled() {
        external_evaluator_config()
            .validate()
            .expect("external evaluator contract should validate");
    }

    #[test]
    fn external_evaluator_contract_rejects_local_validation_consumers() {
        let cases = [
            (
                "gates",
                Box::new(|config: &mut TrainingConfig| config.training.gates.enabled = true)
                    as Box<dyn Fn(&mut TrainingConfig)>,
            ),
            (
                "dynamics",
                Box::new(|config: &mut TrainingConfig| config.training.dynamics.enabled = true),
            ),
            (
                "source_weighted_validation_batches",
                Box::new(|config: &mut TrainingConfig| {
                    config.training.events.source_weighted_validation_batches = 1;
                }),
            ),
            (
                "ruliad_correctness_probe_items",
                Box::new(|config: &mut TrainingConfig| {
                    config.training.events.ruliad_correctness_probe_items = 1;
                }),
            ),
        ];

        for (expected, mutate) in cases {
            let mut config = external_evaluator_config();
            mutate(&mut config);
            let error = config
                .validate()
                .expect_err("local validation consumer should be rejected");
            assert!(
                error.to_string().contains(expected),
                "unexpected error for {expected}: {error}"
            );
        }
    }

    #[test]
    fn latent_reasoning_jepa_training_requires_model_modules() {
        let mut config = parse_config("");
        config.training.latent_reasoning.enabled = true;
        config.training.latent_reasoning.jepa_future_offsets = vec![1];

        let err = config
            .validate()
            .expect_err("latent JEPA training should require model latent reasoning");
        assert!(
            err.to_string()
                .contains("training.latent_reasoning JEPA offsets"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn latent_sigreg_only_training_validates_without_model_modules() {
        let mut config = parse_config("");
        config.training.latent_reasoning.enabled = true;
        config.training.latent_reasoning.jepa_future_offsets = vec![usize::MAX];
        config.training.latent_reasoning.sigreg.enabled = true;
        config.training.latent_reasoning.sigreg.target =
            crate::config::LatentReasoningSigRegTarget::RhoMemorySlots;

        config
            .validate()
            .expect("SIGReg-only latent regularization should not require model latent modules");
    }

    #[test]
    fn hierarchical_dragon_training_config_validates() {
        let config = parse_config(
            r#"
[model.hierarchical_dragon]
enabled = true
last_layers = 1
fast_cycles = 2
slow_cycles = 1
rho_sharing = "split"
weight_sharing = "shared"
slow_to_fast_scale = 0.1
fast_to_slow_scale = 0.1
"#,
        );

        config
            .validate()
            .expect("hierarchical Dragon profile should validate");
    }

    #[test]
    fn hierarchical_dragon_rejects_zero_cycles() {
        let config = parse_config(
            r#"
[model.hierarchical_dragon]
enabled = true
fast_cycles = 0
"#,
        );

        let err = config
            .validate()
            .expect_err("zero fast cycles should be rejected");
        assert!(
            err.to_string()
                .contains("model.hierarchical_dragon.fast_cycles"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn hierarchical_dragon_rejects_pipeline_parallelism() {
        let config = parse_config(
            r#"
[parallel.pipeline]
enabled = true
stage_count = 2
microbatches = 2

[model.hierarchical_dragon]
enabled = true
"#,
        );

        let err = config
            .validate()
            .expect_err("pipeline hierarchy should be rejected");
        assert!(
            err.to_string().contains("parallel.pipeline.enabled"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn next_latent_training_requires_transition_head() {
        let mut config = parse_config("");
        config.training.latent_reasoning.enabled = true;
        config.training.latent_reasoning.jepa_future_offsets = vec![usize::MAX];
        config.training.latent_reasoning.sigreg.enabled = false;
        config.training.latent_reasoning.next_latent.enabled = true;

        let err = config
            .validate()
            .expect_err("NextLat training should require a transition head");
        assert!(
            err.to_string()
                .contains("training.latent_reasoning.next_latent.enabled"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn next_latent_training_does_not_require_inference_latent_reasoning() {
        let mut config = parse_config("");
        config.model.next_latent_transition = Some(Default::default());
        config
            .model
            .next_latent_transition
            .as_mut()
            .expect("next latent transition config")
            .enabled = true;
        config.training.latent_reasoning.enabled = true;
        config.training.latent_reasoning.jepa_future_offsets = vec![usize::MAX];
        config.training.latent_reasoning.sigreg.enabled = false;
        config.training.latent_reasoning.next_latent.enabled = true;

        config
            .validate()
            .expect("NextLat transition training should not require model.latent_reasoning");
    }

    #[test]
    fn dragon_state_training_does_not_require_inference_latent_reasoning_or_transition_head() {
        let mut config = parse_config("");
        config.training.latent_reasoning.enabled = true;
        config.training.latent_reasoning.jepa_future_offsets = vec![usize::MAX];
        config.training.latent_reasoning.sigreg.enabled = false;
        config.training.latent_reasoning.dragon_state.enabled = true;

        config
            .validate()
            .expect("Dragon state consistency should only require recurrent Dragon state");
    }

    #[test]
    fn step_contract_training_requires_inference_latent_reasoning() {
        let mut config = parse_config("");
        config.training.latent_reasoning.enabled = true;
        config.training.latent_reasoning.jepa_future_offsets = vec![usize::MAX];
        config.training.latent_reasoning.sigreg.enabled = false;
        config.training.latent_reasoning.step_contract.enabled = true;

        let err = config
            .validate()
            .expect_err("step contract training should require latent reasoning architecture");
        assert!(
            err.to_string()
                .contains("training.latent_reasoning.step_contract.enabled"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn latent_reasoning_training_validates_with_model_modules() {
        let mut config = parse_config("");
        config.model.latent_reasoning = Some(Default::default());
        config
            .model
            .latent_reasoning
            .as_mut()
            .expect("latent config")
            .enabled = true;
        config.training.latent_reasoning.enabled = true;
        config.training.latent_reasoning.jepa_future_offsets = vec![1, 2];

        config
            .validate()
            .expect("latent reasoning training should validate with model modules enabled");
    }

    #[test]
    fn latent_energy_model_training_requires_model_energy_head() {
        let mut config = parse_config("");
        config.model.latent_reasoning = Some(Default::default());
        config
            .model
            .latent_reasoning
            .as_mut()
            .expect("latent config")
            .enabled = true;
        config.training.latent_reasoning.enabled = true;
        config.training.latent_reasoning.jepa_future_offsets = vec![usize::MAX];
        config.training.latent_reasoning.sigreg.enabled = false;
        config.training.latent_reasoning.energy_model.enabled = true;

        let err = config
            .validate()
            .expect_err("latent EBM training should require model energy head");
        assert!(
            err.to_string()
                .contains("model.latent_reasoning.energy_head"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn latent_energy_model_training_validates_with_model_energy_head() {
        let mut config = parse_config("");
        config.model.latent_reasoning = Some(Default::default());
        let latent = config
            .model
            .latent_reasoning
            .as_mut()
            .expect("latent config");
        latent.enabled = true;
        latent.energy_head = true;
        config.training.latent_reasoning.enabled = true;
        config.training.latent_reasoning.jepa_future_offsets = vec![usize::MAX];
        config.training.latent_reasoning.sigreg.enabled = false;
        config.training.latent_reasoning.energy_model.enabled = true;

        config
            .validate()
            .expect("latent EBM training should validate with model energy head");
    }

    #[test]
    fn latent_reasoning_eval_step_sweep_requires_model_modules() {
        let mut config = parse_config("");
        config.training.latent_reasoning.eval_step_sweep = vec![1, 2, 4];

        let err = config
            .validate()
            .expect_err("eval step sweep should require model latent reasoning");
        assert!(
            err.to_string()
                .contains("training.latent_reasoning.eval_step_sweep"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn latent_reasoning_eval_step_sweep_validates_with_model_modules() {
        let mut config = parse_config("");
        config.model.latent_reasoning = Some(Default::default());
        config
            .model
            .latent_reasoning
            .as_mut()
            .expect("latent config")
            .enabled = true;
        config.training.latent_reasoning.eval_step_sweep = vec![1, 2, 4];

        config
            .validate()
            .expect("eval step sweep should validate with model latent reasoning");
    }

    #[test]
    fn latent_reasoning_eval_step_sweep_rejects_zero() {
        let mut config = parse_config("");
        config.model.latent_reasoning = Some(Default::default());
        config
            .model
            .latent_reasoning
            .as_mut()
            .expect("latent config")
            .enabled = true;
        config.training.latent_reasoning.eval_step_sweep = vec![1, 0, 4];

        let err = config
            .validate()
            .expect_err("zero eval step should fail validation");
        assert!(
            err.to_string()
                .contains("eval_step_sweep must contain only positive"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn latent_reasoning_start_policy_toml_values_parse() {
        let config = parse_config(
            r#"
[training.latent_reasoning]
enabled = true
jepa_start_policy = "fixed_step_and_capability_gate"

[training.latent_reasoning.next_latent]
enabled = true
start_policy = "capability_gate"
"#,
        );

        assert_eq!(
            config.training.latent_reasoning.jepa_start_policy,
            Some(crate::config::LatentReasoningAuxiliaryStartPolicy::FixedStepAndCapabilityGate)
        );
        assert_eq!(
            config.training.latent_reasoning.next_latent.start_policy,
            Some(crate::config::LatentReasoningAuxiliaryStartPolicy::CapabilityGate)
        );
    }

    #[test]
    fn latent_reasoning_training_rejects_zero_every_steps() {
        let mut config = parse_config("");
        config.model.latent_reasoning = Some(Default::default());
        config
            .model
            .latent_reasoning
            .as_mut()
            .expect("latent config")
            .enabled = true;
        config.training.latent_reasoning.enabled = true;
        config.training.latent_reasoning.every_steps = 0;

        let err = config
            .validate()
            .expect_err("zero latent every_steps should fail validation");
        assert!(
            err.to_string()
                .contains("training.latent_reasoning.every_steps"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn latent_reasoning_training_rejects_zero_per_objective_every_steps() {
        let mut config = parse_config("");
        config.model.latent_reasoning = Some(Default::default());
        config
            .model
            .latent_reasoning
            .as_mut()
            .expect("latent config")
            .enabled = true;
        config.model.next_latent_transition = Some(Default::default());
        config
            .model
            .next_latent_transition
            .as_mut()
            .expect("next latent config")
            .enabled = true;
        config.training.latent_reasoning.enabled = true;
        config.training.latent_reasoning.jepa_future_offsets = vec![1];
        config.training.latent_reasoning.next_latent.enabled = true;
        config.training.latent_reasoning.next_latent.every_steps = Some(0);

        let err = config
            .validate()
            .expect_err("zero NextLat every_steps should fail validation");
        assert!(
            err.to_string()
                .contains("training.latent_reasoning.next_latent.every_steps"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn predictive_coding_validates_for_single_tbptt_next_token_training() {
        let mut config = parse_config("");
        config.training.tbptt_chunk_size = Some(4);
        config.training.predictive_coding.enabled = true;

        config
            .validate()
            .expect("predictive coding should validate for local TBPTT next-token training");
    }

    #[test]
    fn predictive_coding_rejects_invalid_amortization_contract() {
        let mut config = parse_config("");
        config.training.tbptt_chunk_size = Some(4);
        config.training.predictive_coding.enabled = true;
        config.training.predictive_coding.amortization_tolerance = f32::NAN;

        let err = config
            .validate()
            .expect_err("non-finite amortization tolerance should fail validation");
        assert!(err.to_string().contains("amortization_tolerance"));

        config.training.predictive_coding.amortization_tolerance = 0.05;
        config
            .training
            .predictive_coding
            .amortization_max_state_slots = 0;
        let err = config
            .validate()
            .expect_err("empty amortization sample should fail validation");
        assert!(err.to_string().contains("amortization_max_state_slots"));
    }

    #[test]
    fn predictive_coding_rejects_unacknowledged_oracle_target_control() {
        let mut config = parse_config("");
        config.training.tbptt_chunk_size = Some(4);
        config.training.predictive_coding.enabled = true;
        config.training.predictive_coding.observation_contract =
            PredictiveCodingObservationContract::OracleNextTokenNegativeControl;

        let err = config
            .validate()
            .expect_err("oracle target leakage must require explicit acknowledgement");
        assert!(
            err.to_string().contains("allow_oracle_target_leak=true"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn predictive_coding_oracle_target_control_is_explicitly_available_for_ablations() {
        let mut config = parse_config("");
        config.training.tbptt_chunk_size = Some(4);
        config.training.predictive_coding.enabled = true;
        config.training.predictive_coding.observation_contract =
            PredictiveCodingObservationContract::OracleNextTokenNegativeControl;
        config.training.predictive_coding.allow_oracle_target_leak = true;

        config
            .validate()
            .expect("acknowledged oracle negative control should remain reproducible");
    }

    #[test]
    fn observed_prefix_predictive_coding_rejects_block_backward() {
        let mut config = parse_config("");
        config.training.tbptt_chunk_size = Some(4);
        config.training.predictive_coding.enabled = true;
        config.training.predictive_coding.backward_mode = PredictiveCodingBackwardMode::Block;

        let err = config
            .validate()
            .expect_err("causal correction must follow each completed chunk");
        assert!(
            err.to_string().contains("requires backward_mode=chunked"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn predictive_coding_optimizer_validates_for_local_chunked_pc_training() {
        let mut config = parse_config("");
        config.optimizer.name = OptimizerKind::PredictiveCoding;
        config.training.tbptt_chunk_size = Some(4);
        config.training.predictive_coding.enabled = true;

        config
            .validate()
            .expect("predictive coding optimizer should validate for local chunked PC training");
    }

    #[test]
    fn predictive_coding_optimizer_requires_enabled_pc_inference() {
        let mut config = parse_config("");
        config.optimizer.name = OptimizerKind::PredictiveCoding;
        config.training.tbptt_chunk_size = Some(4);

        let err = config
            .validate()
            .expect_err("predictive coding optimizer without PC should be rejected");
        assert!(
            err.to_string()
                .contains("requires training.predictive_coding.enabled"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn predictive_coding_optimizer_rejects_state_only_control() {
        let mut config = parse_config("");
        config.optimizer.name = OptimizerKind::PredictiveCoding;
        config.training.tbptt_chunk_size = Some(4);
        config.training.predictive_coding.enabled = true;
        config.training.predictive_coding.parameter_update =
            PredictiveCodingParameterUpdate::StateOnlyControl;

        let err = config
            .validate()
            .expect_err("predictive coding optimizer with state-only control should be rejected");
        assert!(
            err.to_string().contains("parameter_update=optimizer"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn predictive_coding_requires_tbptt() {
        let mut config = parse_config("");
        config.training.tbptt_chunk_size = None;
        config.training.predictive_coding.enabled = true;

        let err = config
            .validate()
            .expect_err("predictive coding without TBPTT should be rejected");
        assert!(
            err.to_string()
                .contains("predictive_coding.enabled requires training.tbptt_chunk_size"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn predictive_coding_rejects_pipeline_training() {
        let mut config = parse_config("");
        config.training.tbptt_chunk_size = Some(4);
        config.training.predictive_coding.enabled = true;
        config.parallel.pipeline.enabled = true;
        config.parallel.pipeline.stage_count = 1;
        config.parallel.pipeline.microbatches = 1;

        let err = config
            .validate()
            .expect_err("predictive coding should not run in pipeline mode");
        assert!(
            err.to_string()
                .contains("predictive_coding.enabled does not support parallel.pipeline.enabled"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn predictive_coding_high_latent_rejects_unsafe_fixed_large_batch() {
        let mut config = parse_config("");
        config.training.tbptt_chunk_size = Some(4);
        config.training.predictive_coding.enabled = true;
        config.training.batch_size = 2;
        config.training.auto_batch_size.enabled = false;
        config.model.latent_total = Some(16_384);

        let err = config
            .validate()
            .expect_err("high-latent PC should require batch one or auto batch sizing");
        assert!(
            err.to_string()
                .contains("predictive_coding with fixed training.batch_size > 1"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn ruliad_policy_batch_is_required_only_by_active_auxiliary_consumers() {
        let mut supervision = crate::RuliadSupervisionConfig::default();
        assert!(!supervision.needs_ruliad_policy_batch());

        supervision.verifier_reward.enabled = true;
        supervision.verifier_reward.weight = 0.0;
        supervision.verifier_reward.structured_contrast_weight = 0.0;
        supervision.verifier_reward.field_binding_contrast_weight = 0.0;
        supervision.verifier_reward.rollout_imitation_weight = 0.0;
        assert!(
            !supervision.needs_ruliad_policy_batch(),
            "enabling verifier config alone should not change loader shape"
        );

        supervision.verifier_reward.field_binding_contrast_weight = 0.01;
        assert!(supervision.needs_ruliad_policy_batch());

        supervision.verifier_reward.field_binding_contrast_weight = 0.0;
        supervision.verifier_reward.rollout_imitation_weight = 0.01;
        assert!(supervision.needs_ruliad_policy_batch());

        supervision.verifier_reward.rollout_imitation_weight = 0.0;
        supervision.answer_denoising.enabled = true;
        supervision.answer_denoising.structured_recovery_weight = 0.25;
        assert!(supervision.needs_ruliad_policy_batch());

        supervision.answer_denoising.enabled = false;
        supervision.answer_denoising.structured_recovery_weight = 0.0;
        supervision.answer_contract.enabled = true;
        supervision.answer_contract.weight = 0.25;
        assert!(supervision.needs_ruliad_policy_batch());
    }

    #[test]
    fn ruliad_policy_batch_schedule_matches_active_auxiliary_cadence() {
        let mut supervision = crate::RuliadSupervisionConfig::default();
        supervision.proof_policy.enabled = true;
        supervision.proof_policy.weight = 0.25;
        supervision.proof_policy.start_after_steps = 4;
        supervision.proof_policy.every_steps = 3;

        for step in 0..10 {
            assert_eq!(
                supervision.needs_ruliad_policy_batch_at_step(step),
                matches!(step, 6 | 9),
                "step={step}"
            );
        }

        supervision.verifier_reward.enabled = true;
        supervision.verifier_reward.weight = 0.05;
        supervision.verifier_reward.start_after_steps = 2;
        supervision.verifier_reward.every_steps = 4;
        assert!(!supervision.needs_ruliad_policy_batch_at_step(2));
        assert!(!supervision.needs_ruliad_policy_batch_at_step(3));
        assert!(supervision.needs_ruliad_policy_batch_at_step(4));
        assert!(supervision.needs_ruliad_policy_batch_at_step(8));
    }

    #[test]
    fn ruliad_1m_baseline_profile_validates_and_stays_small() {
        let config = load_profile("ruliad-1m.training.toml");
        config.validate().expect("ruliad-1m profile validates");
        assert!(matches!(
            &config.dataset.source,
            DatasetSourceConfig::UniversalityRuliad { .. }
        ));
        assert!(matches!(config.optimizer.name, OptimizerKind::Adamw));
        assert_eq!(
            config.training.ruliad_supervision.mode,
            RuliadSupervisionMode::AnswerCompletion
        );
        let estimated = estimate_profile_parameter_budget(&config);
        assert!(
            (750_000..=2_000_000).contains(&estimated),
            "ruliad-1m profile should stay in the fast diagnostic range, estimated params={estimated}"
        );
    }

    #[test]
    fn ruliad_corpus_profiles_warm_start_without_hard_frontier_cap() {
        for profile in ["ruliad-1m.corpus.toml", "ruliad-r1.corpus.toml"] {
            let config = burn_dragon_universality::load_ruliad_config(&profile_path(profile))
                .unwrap_or_else(|err| panic!("load {profile}: {err}"));
            assert!(config.source_selection.enabled, "{profile}");
            assert!(
                config.source_selection.frontier_extension.enabled,
                "{profile}"
            );
            assert_eq!(
                config
                    .source_selection
                    .frontier_extension
                    .max_materialized_levels,
                0,
                "{profile} should keep the live frontier unbounded"
            );
            assert!(
                config.source_selection.cold_start.enabled,
                "{profile} should warm-start cold models on easy buckets"
            );
            assert!(
                config.source_selection.cold_start.max_difficulty_level
                    < config.source_selection.difficulty_levels.max,
                "{profile} cold-start cap should be below the initial materialized frontier"
            );
            assert_eq!(
                config.source_selection.cold_start.max_difficulty_level,
                config.source_selection.difficulty_levels.min,
                "{profile} should bootstrap from the easiest difficulty bucket"
            );
            assert!(
                config.source_selection.cold_start.release_requires_mastery,
                "{profile} should release cold-start difficulty by capability, not time alone"
            );
        }
    }

    #[test]
    fn ruliad_1m_la16k_verifier_proxy_profiles_validate() {
        for (profile, ranking, denoising, structured_recovery) in [
            (
                "ruliad-1m-la-16k.answer-completion.self-recovery.training.toml",
                false,
                false,
                false,
            ),
            (
                "ruliad-1m-la-16k.answer-completion-ranking.self-recovery.training.toml",
                true,
                false,
                false,
            ),
            (
                "ruliad-1m-la-16k.answer-completion-recovery-denoising.self-recovery.training.toml",
                false,
                true,
                true,
            ),
            (
                "ruliad-1m-la-16k.answer-completion-denoising.self-recovery.training.toml",
                false,
                true,
                false,
            ),
            (
                "ruliad-1m-la-16k.answer-completion-ranking-denoising.self-recovery.training.toml",
                true,
                true,
                false,
            ),
            (
                "ruliad-1m-la-16k.field-binding-recovery.training.toml",
                false,
                true,
                true,
            ),
        ] {
            let config = load_profile(profile);
            config
                .validate()
                .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
            assert_eq!(
                config.training.ruliad_supervision.mode,
                RuliadSupervisionMode::AnswerCompletion,
                "{profile}"
            );
            assert!(
                config.training.ruliad_supervision.uses_target_loss_mask(),
                "{profile} should expose answer masks for verifier-proxy objectives"
            );
            assert_eq!(
                config.training.ruliad_supervision.answer_ranking.enabled, ranking,
                "{profile}"
            );
            assert_eq!(
                config.training.ruliad_supervision.answer_denoising.enabled, denoising,
                "{profile}"
            );
            assert_eq!(
                config
                    .training
                    .ruliad_supervision
                    .answer_denoising
                    .structured_recovery_weight
                    > 0.0,
                structured_recovery,
                "{profile}"
            );
            if structured_recovery {
                if !profile.contains("field-binding") {
                    assert!(
                        config.training.tbptt_chunk_size.is_none(),
                        "{profile} structured recovery must run in the non-TBPTT train path"
                    );
                    assert!(
                        !config.training.tbptt_persist_across_steps,
                        "{profile} structured recovery must run in the non-TBPTT train path"
                    );
                }
                assert!(
                    config
                        .training
                        .ruliad_supervision
                        .answer_denoising
                        .structured_recovery_schema_negative_count
                        > 0,
                    "{profile} should include schema-collapse recovery negatives"
                );
            }
            assert_eq!(
                config
                    .training
                    .ruliad_supervision
                    .needs_ruliad_policy_batch(),
                structured_recovery,
                "{profile}"
            );
        }
    }

    #[test]
    fn ruliad_1m_la16k_verifier_reward_profile_validates() {
        let config = load_profile("ruliad-1m-la-16k.verifier-reward.training.toml");
        config
            .validate()
            .expect("ruliad verifier-reward profile should validate");
        assert!(config.training.ruliad_supervision.verifier_reward.enabled);
        assert_eq!(
            config.training.ruliad_supervision.mode,
            RuliadSupervisionMode::AnswerCompletion
        );
        assert_eq!(
            config.training.ruliad_supervision.verifier_reward.mode,
            RuliadVerifierRewardMode::Scalar
        );
        assert!(config.training.tbptt_chunk_size.is_none());
        assert!(!config.training.tbptt_persist_across_steps);
        assert!(config.training.objective.is_next_token());
    }

    #[test]
    fn ruliad_1m_la16k_verifier_vpo_profile_validates() {
        for (
            profile,
            include_oracle_candidate,
            include_structured_negatives,
            structured_contrast,
        ) in [
            (
                "ruliad-1m-la-16k.verifier-vpo.training.toml",
                false,
                false,
                false,
            ),
            (
                "ruliad-1m-la-16k.verifier-vpo-oracle.training.toml",
                true,
                false,
                false,
            ),
            (
                "ruliad-1m-la-16k.verifier-vpo-oracle-structured.training.toml",
                true,
                true,
                false,
            ),
            (
                "ruliad-1m-la-16k.verifier-vpo-oracle-structured-contrast.training.toml",
                true,
                true,
                true,
            ),
        ] {
            let config = load_profile(profile);
            config
                .validate()
                .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
            assert!(
                config.training.ruliad_supervision.verifier_reward.enabled,
                "{profile}"
            );
            assert_eq!(
                config.training.ruliad_supervision.verifier_reward.mode,
                RuliadVerifierRewardMode::VpoIndependent,
                "{profile}"
            );
            assert_eq!(
                config.training.ruliad_supervision.mode,
                RuliadSupervisionMode::AnswerCompletion,
                "{profile}"
            );
            assert!(
                config
                    .training
                    .ruliad_supervision
                    .verifier_reward
                    .vpo_scalarizations
                    > 0,
                "{profile}"
            );
            assert!(
                config
                    .training
                    .ruliad_supervision
                    .verifier_reward
                    .vpo_correctness_mass_floor
                    >= 0.70,
                "{profile}"
            );
            assert!(
                config
                    .training
                    .ruliad_supervision
                    .verifier_reward
                    .vpo_schema_quality_mass_floor
                    >= 0.10,
                "{profile}"
            );
            assert!(
                config
                    .training
                    .ruliad_supervision
                    .verifier_reward
                    .vpo_compactness_max_weight
                    <= 0.05,
                "{profile}"
            );
            assert!(
                config
                    .training
                    .ruliad_supervision
                    .verifier_reward
                    .positive_advantage_requires_correctness,
                "{profile}"
            );
            assert!(
                config
                    .training
                    .ruliad_supervision
                    .verifier_reward
                    .positive_advantage_min_partial_progress_ppm
                    >= 500_000,
                "{profile}"
            );
            assert!(
                config
                    .training
                    .ruliad_supervision
                    .verifier_reward
                    .positive_advantage_min_completion_quality_ppm
                    >= 750_000,
                "{profile}"
            );
            assert_eq!(
                config
                    .training
                    .ruliad_supervision
                    .verifier_reward
                    .start_after_steps,
                512,
                "{profile}"
            );
            assert_eq!(
                config
                    .training
                    .ruliad_supervision
                    .verifier_reward
                    .max_advantage_clip_fraction,
                Some(0.95),
                "{profile}"
            );
            assert!(
                config
                    .training
                    .ruliad_supervision
                    .verifier_reward
                    .clip_range
                    >= 1.0,
                "{profile}"
            );
            assert_eq!(
                config
                    .training
                    .ruliad_supervision
                    .verifier_reward
                    .include_oracle_candidate,
                include_oracle_candidate,
                "{profile}"
            );
            assert_eq!(
                config
                    .training
                    .ruliad_supervision
                    .verifier_reward
                    .include_structured_negative_candidates,
                include_structured_negatives,
                "{profile}"
            );
            if include_structured_negatives {
                assert!(
                    config
                        .training
                        .ruliad_supervision
                        .verifier_reward
                        .structured_negative_count
                        > 0,
                    "{profile}"
                );
                assert!(
                    config
                        .training
                        .ruliad_supervision
                        .verifier_reward
                        .structured_template_negative_count
                        > 0,
                    "{profile}"
                );
                assert!(
                    config
                        .training
                        .ruliad_supervision
                        .verifier_reward
                        .structured_schema_negative_count
                        > 0,
                    "{profile}"
                );
            }
            assert_eq!(
                config
                    .training
                    .ruliad_supervision
                    .verifier_reward
                    .structured_contrast_weight
                    > 0.0,
                structured_contrast,
                "{profile}"
            );
            assert!(config.training.tbptt_chunk_size.is_none(), "{profile}");
            assert!(!config.training.tbptt_persist_across_steps, "{profile}");
            assert!(config.training.objective.is_next_token(), "{profile}");
        }
    }

    #[test]
    fn ruliad_1m_la16k_structured_contrast_profile_validates_without_sampled_policy() {
        let profile = "ruliad-1m-la-16k.structured-contrast.training.toml";
        let config = load_profile(profile);
        config
            .validate()
            .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
        let verifier_reward = config.training.ruliad_supervision.verifier_reward;

        assert!(verifier_reward.enabled);
        assert_eq!(verifier_reward.weight, 0.0);
        assert!(verifier_reward.structured_negative_count > 0);
        assert_eq!(verifier_reward.structured_template_negative_count, 0);
        assert!(verifier_reward.structured_schema_negative_count > 0);
        assert!(verifier_reward.structured_contrast_weight > 0.0);
        assert_eq!(verifier_reward.structured_contrast_start_after_steps, 0);
        assert_eq!(
            config.training.ruliad_supervision.mode,
            RuliadSupervisionMode::AnswerCompletion
        );
        assert!(config.training.tbptt_chunk_size.is_none());
        assert!(!config.training.tbptt_persist_across_steps);
        assert!(
            config
                .training
                .ruliad_supervision
                .needs_ruliad_policy_batch()
        );
    }

    #[test]
    fn ruliad_1m_la16k_field_binding_profile_validates_without_tbptt() {
        let profile =
            "ruliad-1m-la-16k.verifier-vpo-oracle-structured-contrast-field-binding.training.toml";
        let config = load_profile(profile);
        config
            .validate()
            .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
        let verifier_reward = config.training.ruliad_supervision.verifier_reward;

        assert!(verifier_reward.enabled);
        assert!(verifier_reward.field_binding_contrast_weight > 0.0);
        assert!(verifier_reward.structured_schema_negative_count > 0);
        assert!(verifier_reward.structured_contrast_weight > 0.0);
        assert_eq!(verifier_reward.field_binding_contrast_start_after_steps, 0);
        assert_eq!(verifier_reward.field_binding_contrast_every_steps, 8);
        assert_eq!(verifier_reward.field_binding_contrast_pair_weight, 0.5);
        assert_eq!(verifier_reward.field_binding_contrast_max_pairs, 24);
        assert_eq!(verifier_reward.field_binding_contrast_replay_capacity, 64);
        assert_eq!(verifier_reward.generated_attractor_replay_capacity, 128);
        assert_eq!(verifier_reward.generated_attractor_replay_min_count, 2);
        assert_eq!(verifier_reward.generated_attractor_replay_max_candidates, 4);
        assert_eq!(
            verifier_reward.generated_attractor_replay_min_distinct_answers,
            2
        );
        assert_eq!(
            verifier_reward.generated_attractor_replay_max_dominant_fraction,
            0.5
        );
        assert_eq!(
            config.training.ruliad_supervision.mode,
            RuliadSupervisionMode::AnswerCompletion
        );
        assert!(config.training.tbptt_chunk_size.is_none());
        assert!(!config.training.tbptt_persist_across_steps);
        assert!(
            config
                .training
                .ruliad_supervision
                .needs_ruliad_policy_batch()
        );
    }

    #[test]
    fn ruliad_1m_la16k_field_binding_only_profile_validates_without_policy_reward() {
        let profile = "ruliad-1m-la-16k.field-binding-contrast.training.toml";
        let config = load_profile(profile);
        config
            .validate()
            .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
        let verifier_reward = config.training.ruliad_supervision.verifier_reward;

        assert!(verifier_reward.enabled);
        assert_eq!(verifier_reward.weight, 0.0);
        assert!(verifier_reward.field_binding_contrast_weight > 0.0);
        assert_eq!(verifier_reward.field_binding_contrast_every_steps, 4);
        assert_eq!(verifier_reward.field_binding_contrast_pair_weight, 1.0);
        assert_eq!(verifier_reward.field_binding_contrast_max_pairs, 16);
        assert_eq!(verifier_reward.field_binding_contrast_replay_capacity, 128);
        assert!(config.training.tbptt_chunk_size.is_none());
        assert!(!config.training.tbptt_persist_across_steps);
        assert!(
            config
                .training
                .ruliad_supervision
                .needs_ruliad_policy_batch()
        );
    }

    #[test]
    fn ruliad_1m_la64k_field_binding_profile_validates_with_tbptt() {
        let profile = "ruliad-1m-la-64k.field-binding-contrast.training.toml";
        let config = load_profile(profile);
        config
            .validate()
            .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
        let verifier_reward = config.training.ruliad_supervision.verifier_reward;

        assert!(verifier_reward.enabled);
        assert_eq!(verifier_reward.weight, 0.0);
        assert_eq!(verifier_reward.field_binding_contrast_weight, 0.05);
        assert_eq!(verifier_reward.field_binding_contrast_every_steps, 8);
        assert_eq!(verifier_reward.field_binding_contrast_pair_weight, 0.5);
        assert_eq!(verifier_reward.field_binding_contrast_max_pairs, 8);
        assert_eq!(verifier_reward.field_binding_contrast_replay_capacity, 64);
        assert_eq!(config.model.latent_total, Some(65_536));
        assert_eq!(config.training.tbptt_chunk_size, Some(128));
        assert!(config.training.tbptt_persist_across_steps);
        assert!(
            config
                .training
                .ruliad_supervision
                .needs_ruliad_policy_batch()
        );
    }

    #[test]
    fn ruliad_1m_la64k_structured_recovery_profile_validates_with_tbptt() {
        let profile = "ruliad-1m-la-64k.answer-completion-recovery.training.toml";
        let config = load_profile(profile);
        config
            .validate()
            .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
        let denoising = config.training.ruliad_supervision.answer_denoising;

        assert!(denoising.enabled);
        assert_eq!(denoising.weight, 0.0);
        assert_eq!(denoising.structured_recovery_weight, 0.25);
        assert_eq!(denoising.structured_recovery_every_steps, 4);
        assert_eq!(denoising.structured_recovery_schema_negative_count, 4);
        assert_eq!(
            config.training.ruliad_supervision.mode,
            RuliadSupervisionMode::AnswerCompletion
        );
        assert_eq!(config.model.latent_total, Some(65_536));
        assert_eq!(config.training.tbptt_chunk_size, Some(128));
        assert!(config.training.tbptt_persist_across_steps);
        assert!(
            config
                .training
                .ruliad_supervision
                .needs_ruliad_policy_batch()
        );
    }

    #[test]
    fn ruliad_1m_la64k_answer_contract_profile_validates_with_tbptt() {
        let profile = "ruliad-1m-la-64k.answer-contract.training.toml";
        let config = load_profile(profile);
        config
            .validate()
            .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
        let contract = config.training.ruliad_supervision.answer_contract;

        assert!(contract.enabled);
        assert_eq!(contract.weight, 0.25);
        assert_eq!(contract.premature_close_unlikelihood_weight, 0.5);
        assert_eq!(contract.every_steps, 1);
        assert_eq!(contract.max_completion_tokens, 64);
        assert_eq!(contract.max_rows_per_step, 8);
        assert_eq!(
            config.training.ruliad_supervision.mode,
            RuliadSupervisionMode::AnswerCompletion
        );
        assert_eq!(config.model.latent_total, Some(65_536));
        assert_eq!(
            config
                .model
                .latent_reasoning
                .as_ref()
                .expect("answer-contract profile should configure latent reasoning")
                .max_steps,
            2
        );
        assert_eq!(config.training.tbptt_chunk_size, Some(128));
        assert!(config.training.tbptt_persist_across_steps);
        assert!(
            config
                .training
                .ruliad_supervision
                .needs_ruliad_policy_batch()
        );
    }

    #[test]
    fn ruliad_1m_la64k_answer_contract_schema_profile_validates_with_tbptt() {
        let profile = "ruliad-1m-la-64k.answer-contract-schema.training.toml";
        let config = load_profile(profile);
        config
            .validate()
            .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
        let supervision = config.training.ruliad_supervision;
        let contract = supervision.answer_contract;

        assert!(contract.enabled);
        assert_eq!(contract.weight, 0.25);
        assert_eq!(contract.premature_close_unlikelihood_weight, 1.0);
        assert_eq!(contract.schema_token_weight, 4.0);
        assert_eq!(contract.schema_start_token_weight, 0.0);
        assert_eq!(contract.value_token_weight, 1.0);
        assert_eq!(contract.other_token_weight, 0.25);
        assert_eq!(supervision.answer_close_marker_stride, 4);
        assert_eq!(supervision.answer_schema_token_weight, 4);
        assert_eq!(supervision.answer_schema_start_token_weight, 1);
        assert_eq!(supervision.answer_value_token_weight, 1);
        assert_eq!(config.model.latent_total, Some(65_536));
        assert_eq!(config.training.tbptt_chunk_size, Some(128));
        assert!(config.training.tbptt_persist_across_steps);
        assert!(supervision.needs_ruliad_policy_batch());
    }

    #[test]
    fn ruliad_1m_la64k_answer_contract_schema_start_profile_validates_with_tbptt() {
        let profile = "ruliad-1m-la-64k.answer-contract-schema-start.training.toml";
        let config = load_profile(profile);
        config
            .validate()
            .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
        let supervision = config.training.ruliad_supervision;
        let contract = supervision.answer_contract;

        assert!(contract.enabled);
        assert_eq!(contract.weight, 0.25);
        assert_eq!(contract.schema_token_weight, 4.0);
        assert_eq!(contract.schema_start_token_weight, 16.0);
        assert_eq!(contract.value_token_weight, 1.0);
        assert_eq!(supervision.answer_close_marker_stride, 4);
        assert_eq!(supervision.answer_schema_token_weight, 4);
        assert_eq!(supervision.answer_schema_start_token_weight, 12);
        assert_eq!(supervision.answer_value_token_weight, 1);
        assert_eq!(config.model.latent_total, Some(65_536));
        assert_eq!(config.training.tbptt_chunk_size, Some(128));
        assert!(config.training.tbptt_persist_across_steps);
        assert!(supervision.needs_ruliad_policy_batch());
    }

    #[test]
    fn ruliad_1m_la64k_answer_contract_schema_trace_answer_profile_validates_with_tbptt() {
        let profile = "ruliad-1m-la-64k.answer-contract-schema-trace-answer.training.toml";
        let config = load_profile(profile);
        config
            .validate()
            .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
        let supervision = config.training.ruliad_supervision;
        let contract = supervision.answer_contract;

        assert_eq!(supervision.mode, RuliadSupervisionMode::TraceAndAnswer);
        assert!(supervision.mask_high_entropy_spans);
        assert!(supervision.uses_answer_target_mask());
        assert!(supervision.uses_trace_answer_target_mask());
        assert!(contract.enabled);
        assert_eq!(contract.weight, 0.25);
        assert_eq!(contract.schema_token_weight, 4.0);
        assert_eq!(contract.schema_start_token_weight, 0.0);
        assert_eq!(contract.value_token_weight, 1.0);
        assert_eq!(supervision.answer_close_marker_stride, 4);
        assert_eq!(supervision.answer_schema_token_weight, 4);
        assert_eq!(supervision.answer_schema_start_token_weight, 1);
        assert_eq!(supervision.answer_value_token_weight, 1);
        assert_eq!(config.model.latent_total, Some(65_536));
        assert_eq!(config.training.tbptt_chunk_size, Some(128));
        assert!(config.training.tbptt_persist_across_steps);
        assert!(supervision.uses_target_loss_mask());
        assert!(supervision.needs_ruliad_policy_batch());
    }

    #[test]
    fn ruliad_1m_la64k_answer_contract_schema_mixed_trace_profile_validates_with_tbptt() {
        let profile = "ruliad-1m-la-64k.answer-contract-schema-mixed-trace.training.toml";
        let config = load_profile(profile);
        config
            .validate()
            .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
        let supervision = config.training.ruliad_supervision;
        let contract = supervision.answer_contract;

        assert_eq!(supervision.mode, RuliadSupervisionMode::Mixed);
        assert!(supervision.mask_high_entropy_spans);
        assert!(contract.enabled);
        assert_eq!(contract.weight, 0.25);
        assert_eq!(contract.schema_token_weight, 4.0);
        assert_eq!(contract.schema_start_token_weight, 0.0);
        assert_eq!(contract.value_token_weight, 1.0);
        assert_eq!(supervision.answer_close_marker_stride, 4);
        assert_eq!(supervision.answer_schema_token_weight, 4);
        assert_eq!(supervision.answer_schema_start_token_weight, 1);
        assert_eq!(supervision.answer_value_token_weight, 1);
        assert_eq!(config.model.latent_total, Some(65_536));
        assert_eq!(config.training.tbptt_chunk_size, Some(128));
        assert!(config.training.tbptt_persist_across_steps);
        assert!(supervision.uses_target_loss_mask());
        assert!(supervision.needs_ruliad_policy_batch());
    }

    #[test]
    fn ruliad_1m_la64k_answer_contract_schema_field_binding_profile_validates_with_tbptt() {
        let profile = "ruliad-1m-la-64k.answer-contract-schema-field-binding.training.toml";
        let config = load_profile(profile);
        config
            .validate()
            .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
        let supervision = config.training.ruliad_supervision;
        let contract = supervision.answer_contract;
        let verifier_reward = supervision.verifier_reward;

        assert!(contract.enabled);
        assert_eq!(contract.weight, 0.25);
        assert_eq!(contract.premature_close_unlikelihood_weight, 1.0);
        assert_eq!(contract.schema_token_weight, 4.0);
        assert_eq!(contract.schema_start_token_weight, 0.0);
        assert_eq!(contract.value_token_weight, 1.0);
        assert_eq!(supervision.answer_close_marker_stride, 4);
        assert_eq!(supervision.answer_schema_token_weight, 4);
        assert_eq!(supervision.answer_schema_start_token_weight, 1);
        assert_eq!(supervision.answer_value_token_weight, 1);
        assert!(verifier_reward.enabled);
        assert_eq!(verifier_reward.weight, 0.0);
        assert_eq!(verifier_reward.field_binding_contrast_weight, 0.05);
        assert_eq!(verifier_reward.field_binding_contrast_every_steps, 8);
        assert_eq!(verifier_reward.field_binding_contrast_pair_weight, 0.5);
        assert_eq!(verifier_reward.field_binding_contrast_max_pairs, 8);
        assert_eq!(verifier_reward.field_binding_contrast_replay_capacity, 64);
        assert_eq!(config.model.latent_total, Some(65_536));
        assert_eq!(config.training.tbptt_chunk_size, Some(128));
        assert!(config.training.tbptt_persist_across_steps);
        assert!(supervision.needs_ruliad_policy_batch());
    }

    #[test]
    fn ruliad_1m_la64k_answer_contract_value_binding_profile_validates_with_tbptt() {
        let profile = "ruliad-1m-la-64k.answer-contract-value-binding.training.toml";
        let config = load_profile(profile);
        config
            .validate()
            .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
        let supervision = config.training.ruliad_supervision;
        let contract = supervision.answer_contract;
        let verifier_reward = supervision.verifier_reward;

        assert!(contract.enabled);
        assert_eq!(contract.weight, 0.25);
        assert_eq!(contract.schema_token_weight, 4.0);
        assert_eq!(contract.value_token_weight, 1.0);
        assert_eq!(contract.prompt_schema_value_weight, 4.0);
        assert_eq!(contract.prompt_schema_max_rows_per_step, 4);
        assert!(verifier_reward.enabled);
        assert_eq!(verifier_reward.field_binding_contrast_weight, 0.05);
        assert_eq!(
            verifier_reward.field_binding_contrast_rank_metric_every_steps,
            8
        );
        assert_eq!(config.model.latent_total, Some(65_536));
        assert_eq!(config.training.tbptt_chunk_size, Some(128));
        assert!(config.training.tbptt_persist_across_steps);
        assert!(supervision.needs_ruliad_policy_batch());
    }

    #[test]
    fn ruliad_1m_la64k_answer_contract_values_profile_validates_with_tbptt() {
        let profile = "ruliad-1m-la-64k.answer-contract-values.training.toml";
        let config = load_profile(profile);
        config
            .validate()
            .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
        let supervision = config.training.ruliad_supervision;
        let contract = supervision.answer_contract;

        assert!(contract.enabled);
        assert_eq!(contract.weight, 0.50);
        assert_eq!(contract.premature_close_unlikelihood_weight, 0.75);
        assert_eq!(contract.schema_token_weight, 1.0);
        assert_eq!(contract.value_token_weight, 8.0);
        assert_eq!(contract.other_token_weight, 0.25);
        assert_eq!(supervision.answer_close_marker_stride, 1);
        assert_eq!(supervision.answer_close_marker_weight, 2);
        assert_eq!(supervision.answer_schema_token_weight, 1);
        assert_eq!(supervision.answer_value_token_weight, 6);
        assert_eq!(config.model.latent_total, Some(65_536));
        assert_eq!(config.training.tbptt_chunk_size, Some(128));
        assert!(config.training.tbptt_persist_across_steps);
        assert!(supervision.needs_ruliad_policy_batch());
    }

    #[test]
    fn ruliad_1m_la16k_verifier_rollout_imitation_profile_validates_without_tbptt() {
        let profile = "ruliad-1m-la-16k.verifier-rollout-imitation.training.toml";
        let config = load_profile(profile);
        config
            .validate()
            .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
        let verifier_reward = config.training.ruliad_supervision.verifier_reward;

        assert!(verifier_reward.enabled);
        assert_eq!(verifier_reward.weight, 0.0);
        assert_eq!(verifier_reward.structured_contrast_weight, 0.0);
        assert!(verifier_reward.rollout_imitation_weight > 0.0);
        assert_eq!(verifier_reward.rollout_imitation_start_after_steps, 128);
        assert_eq!(
            verifier_reward.rollout_imitation_min_verifier_rate_ppm,
            100_000
        );
        assert_eq!(
            verifier_reward.rollout_imitation_max_schema_wrong_rate_ppm,
            250_000
        );
        assert_eq!(
            config.training.ruliad_supervision.mode,
            RuliadSupervisionMode::AnswerCompletion
        );
        assert!(config.training.tbptt_chunk_size.is_none());
        assert!(!config.training.tbptt_persist_across_steps);
        assert!(
            config
                .training
                .ruliad_supervision
                .needs_ruliad_policy_batch()
        );
    }

    #[test]
    fn ruliad_1m_jepa_default_profiles_validate() {
        for profile in [
            "ruliad-1m.jepa.training.toml",
            "ruliad-1m-la-16k.jepa.training.toml",
            "ruliad-1m-la-32k.jepa.training.toml",
            "ruliad-1m-la-64k.jepa.training.toml",
        ] {
            let config = load_profile(profile);
            config
                .validate()
                .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
            assert!(
                config
                    .model
                    .latent_reasoning
                    .as_ref()
                    .is_some_and(|latent| latent.enabled),
                "{profile} should enable the JEPA latent reasoning model module"
            );
            assert!(
                config.training.latent_reasoning.enabled,
                "{profile} should enable JEPA latent training"
            );
            assert_eq!(
                config.training.latent_reasoning.jepa_future_offsets,
                vec![1],
                "{profile} should use JEPA-only future hidden prediction"
            );
            assert!(
                !config.training.latent_reasoning.next_latent.enabled,
                "{profile} should not enable NextLat by default"
            );
        }
    }

    #[test]
    fn ruliad_10m_screening_profiles_validate_capability_gates() {
        for profile in [
            "ruliad-r1.jepa-10m-screening.toml",
            "ruliad-r1.jepa-nextlat-10m-screening.toml",
        ] {
            let config = load_profile(profile);
            config
                .validate()
                .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
            assert!(config.training.latent_reasoning.enabled, "{profile}");
            assert_eq!(
                config.training.events.ruliad_correctness_probe_items, 128,
                "{profile}"
            );
            assert!(
                config.training.events.source_selection_capability_feedback,
                "{profile} should feed capability probes back into live source selection by default"
            );
            assert_eq!(
                config
                    .training
                    .gates
                    .capability_zero_verifier_patience_epochs,
                8,
                "{profile}"
            );
            assert_eq!(
                config.training.gates.capability_grace_epochs, 3,
                "{profile}"
            );
            assert_eq!(
                config.training.gates.capability_regression_patience_epochs, 2,
                "{profile}"
            );
            assert!(
                config.training.gates.capability_required_after_first_pass,
                "{profile}"
            );
            assert_eq!(
                config.training.gates.capability_schema_wrong_max_rate, 0.50,
                "{profile}"
            );
            assert_eq!(
                config.training.gates.capability_malformed_max_rate, 0.02,
                "{profile}"
            );
            assert_eq!(
                config.training.gates.capability_missing_max_rate, 0.02,
                "{profile}"
            );
            assert_eq!(
                config.training.gates.capability_completion_health_min_rate, 0.40,
                "{profile}"
            );
            assert_eq!(
                config.training.gates.capability_output_entropy_min_bits, 1.25,
                "{profile}"
            );
            assert_eq!(
                config.training.gates.capability_distinct_2_min_fraction, 0.30,
                "{profile}"
            );
            assert_eq!(
                config
                    .training
                    .gates
                    .capability_answer_distinct_min_fraction,
                0.20,
                "{profile}"
            );
            assert_eq!(
                config
                    .training
                    .gates
                    .capability_field_value_distinct_ratio_min,
                0.35,
                "{profile}"
            );
            assert_eq!(
                config.training.gates.capability_field_value_dominance_max, 0.85,
                "{profile}"
            );
        }
    }

    #[test]
    fn ruliad_latent_energy_ablation_profile_validates() {
        for profile in [
            "ruliad-r1.jepa-nextlat-energy-probe128-fixed-ablation.toml",
            "ruliad-r1.jepa-nextlat-energy-contrastive-probe128-fixed-ablation.toml",
            "ruliad-r1.jepa-nextlat-energy-stability-probe128-fixed-ablation.toml",
        ] {
            let config = load_profile(profile);
            config
                .validate()
                .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
            let latent = config
                .model
                .latent_reasoning
                .as_ref()
                .unwrap_or_else(|| panic!("{profile} should configure latent reasoning"));
            assert!(latent.enabled, "{profile}");
            assert!(latent.energy_head, "{profile}");
            assert!(config.training.latent_reasoning.enabled, "{profile}");
            assert!(
                config.training.latent_reasoning.energy_model.enabled,
                "{profile} should enable latent EBM training"
            );
            assert_eq!(
                config.training.latent_reasoning.eval_step_sweep,
                vec![1, 2, 4, 8],
                "{profile}"
            );
        }
    }

    #[test]
    fn ruliad_step_contract_ablation_profile_validates() {
        let profile = "ruliad-r1.jepa-nextlat-step-contract-probe128-fixed-ablation.toml";
        let config = load_profile(profile);
        config
            .validate()
            .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
        let latent = config
            .model
            .latent_reasoning
            .as_ref()
            .unwrap_or_else(|| panic!("{profile} should configure latent reasoning"));
        assert!(latent.enabled, "{profile}");
        assert!(!latent.energy_head, "{profile}");
        assert!(config.training.latent_reasoning.enabled, "{profile}");
        assert!(
            config.training.latent_reasoning.step_contract.enabled,
            "{profile} should enable latent step contract training"
        );
        assert_eq!(
            config.training.latent_reasoning.eval_step_sweep,
            vec![1, 2, 4, 8],
            "{profile}"
        );
    }

    #[test]
    fn ruliad_hierarchical_dragon_ablation_profiles_validate() {
        for (profile, rho_sharing, weight_sharing) in [
            (
                "ruliad-r1.hdragon-shared-rho-shared-weights-probe128-fixed-ablation.toml",
                HierarchicalDragonSharing::Shared,
                HierarchicalDragonSharing::Shared,
            ),
            (
                "ruliad-r1.hdragon-split-rho-shared-weights-probe128-fixed-ablation.toml",
                HierarchicalDragonSharing::Split,
                HierarchicalDragonSharing::Shared,
            ),
            (
                "ruliad-r1.hdragon-split-rho-split-weights-probe128-fixed-ablation.toml",
                HierarchicalDragonSharing::Split,
                HierarchicalDragonSharing::Split,
            ),
        ] {
            let config = load_profile(profile);
            config
                .validate()
                .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
            let hierarchy = config
                .model
                .hierarchical_dragon
                .as_ref()
                .unwrap_or_else(|| panic!("{profile} should configure hierarchical Dragon"));
            assert!(hierarchy.enabled, "{profile}");
            assert_eq!(hierarchy.rho_sharing, rho_sharing, "{profile}");
            assert_eq!(hierarchy.weight_sharing, weight_sharing, "{profile}");
            assert_eq!(hierarchy.last_layers, Some(1), "{profile}");
            assert!(config.training.latent_reasoning.enabled, "{profile}");
            assert!(
                config.training.latent_reasoning.next_latent.enabled,
                "{profile} should inherit NextLat training"
            );
        }
    }

    #[test]
    fn ruliad_1m_high_neuron_sweep_profiles_resolve_expected_long_context_shape() {
        for (profile, latent_total, batch_size) in [
            ("ruliad-1m-la-16k.training.toml", 16_384, 1),
            ("ruliad-1m-la-32k.training.toml", 32_768, 1),
            ("ruliad-1m-la-64k.training.toml", 65_536, 1),
        ] {
            let config = load_profile(profile);
            config.validate().unwrap_or_else(|err| {
                panic!("{profile} should validate as a safe high-neuron sweep profile: {err}")
            });
            assert!(matches!(
                &config.dataset.source,
                DatasetSourceConfig::UniversalityRuliad { .. }
            ));
            assert!(matches!(config.optimizer.name, OptimizerKind::Adamw));
            assert_eq!(config.optimizer.learning_rate, 3.0e-4, "{profile}");
            assert_eq!(config.model.n_layer, Some(4), "{profile}");
            assert_eq!(config.model.n_embd, Some(256), "{profile}");
            assert_eq!(config.model.n_head, Some(4), "{profile}");
            assert_eq!(config.model.latent_total, Some(latent_total), "{profile}");
            assert_eq!(config.training.block_size, 256, "{profile}");
            assert_eq!(config.training.tbptt_chunk_size, Some(128), "{profile}");
            assert!(config.training.tbptt_persist_across_steps, "{profile}");
            assert_eq!(
                config.training.min_logical_block_size,
                Some(512),
                "{profile}"
            );
            assert_eq!(
                config.training.ruliad_supervision.mode,
                RuliadSupervisionMode::AnswerWindow,
                "{profile}"
            );
            assert!(
                config.training.input_corruption.enabled,
                "{profile} should use input corruption as cheap continual-learning regularization"
            );
            assert!(
                config.training.logit_entropy_floor.enabled,
                "{profile} should keep a minimum token-distribution entropy floor"
            );
            assert!(
                config.training.repeat_unlikelihood.enabled,
                "{profile} should penalize short-period repetition"
            );
            assert!(
                config.training.greedy_rollout_unlikelihood.enabled
                    && config.training.greedy_rollout_unlikelihood.recovery_only,
                "{profile} should keep expensive rollout anti-collapse pressure recovery-only"
            );
            assert!(
                config.training.dynamics_anchor.enabled
                    && config.training.dynamics_anchor.weight > 0.0,
                "{profile} should constrain next-token distribution drift with an EMA dynamics anchor"
            );
            assert!(
                !config.training.auto_batch_size.enabled,
                "{profile} should rely on the guarded sweep wrapper, not startup auto-batch probing"
            );
            assert_eq!(config.training.batch_size, batch_size, "{profile}");
            assert_eq!(
                config.training.gates.degeneracy_eos_max_fraction, 0.20,
                "{profile}"
            );
            assert_eq!(
                config.training.gates.degeneracy_period_3_max_fraction, 0.25,
                "{profile}"
            );
            assert_eq!(
                config.training.gates.degeneracy_period_2_to_64_max_fraction, 0.25,
                "{profile}"
            );

            let model = build_model_config(&config.model, config.training.block_size);
            assert_eq!(model.latent_total(), latent_total, "{profile}");
            assert_eq!(
                model.sequence_kernel.memory_system,
                SequenceMemorySystem::LinearAttention,
                "{profile}"
            );
            assert_eq!(
                model.fused_kernels.rotary_embedding,
                RotaryEmbedding::Alibi,
                "{profile}"
            );
        }
    }

    fn load_profile(file_name: &str) -> TrainingConfig {
        let profile_path = profile_path(file_name);
        load_training_config(std::slice::from_ref(&profile_path))
            .unwrap_or_else(|err| panic!("load {}: {err}", profile_path.display()))
    }

    #[test]
    fn ruliad_r3_profile_streams_the_full_formal_proof_contract() {
        let config = load_profile("ruliad-r3.training.toml");
        config.validate().expect("R3 profile should validate");

        assert_eq!(config.training.tbptt_chunk_size, Some(512));
        assert!(config.training.tbptt_persist_across_steps);
        assert_eq!(
            config.training.ruliad_supervision.mode,
            RuliadSupervisionMode::TraceAndAnswer
        );
        assert!(config.training.ruliad_supervision.mask_high_entropy_spans);
    }

    #[test]
    fn ruliad_r3_stateful_tbptt_profiles_form_a_matched_factorial_ablation() {
        use crate::config::SequenceBatchingMode;

        let arms = [
            ("ruliad-r3.stateful-tbptt-block512-reset.toml", 512, false),
            ("ruliad-r3.stateful-tbptt-block512-carry.toml", 512, true),
            ("ruliad-r3.stateful-tbptt-chunk128-reset.toml", 128, false),
            ("ruliad-r3.stateful-tbptt-chunk128-carry.toml", 128, true),
            ("ruliad-r3.stateful-tbptt-chunk64-carry.toml", 64, true),
        ];
        let mut shared_contract = None;
        for (profile, chunk_size, persist) in arms {
            let config = load_profile(profile);
            config
                .validate()
                .unwrap_or_else(|error| panic!("{profile} should validate: {error}"));
            assert_eq!(config.training.block_size, 512, "{profile}");
            assert_eq!(
                config.training.tbptt_chunk_size,
                Some(chunk_size),
                "{profile}"
            );
            assert_eq!(
                config.training.tbptt_persist_across_steps, persist,
                "{profile}"
            );
            assert_eq!(
                config.training.sequence_batching,
                SequenceBatchingMode::Streaming,
                "{profile}"
            );
            assert!(config.training.sequence_state_probe.enabled, "{profile}");
            assert!(
                config.training.ruliad_supervision.balance_trace_answer_mass,
                "{profile}"
            );
            assert!(!config.training.auto_batch_size.enabled, "{profile}");
            assert!(!config.training.continual_backprop.enabled, "{profile}");
            assert!(!config.training.neuron_scaling.enabled, "{profile}");
            assert!(!config.training.gates.enabled, "{profile}");
            assert!(!config.training.dynamics.enabled, "{profile}");
            let contract = (
                config.dataset.clone(),
                config.model.clone(),
                config.optimizer.clone(),
                config.training.ruliad_supervision,
                config.training.ruliad_probe_generation,
                config.training.objective.clone(),
                config.training.batch_size,
                config.training.seed,
            );
            if let Some(expected) = shared_contract.as_ref() {
                assert_eq!(
                    &contract, expected,
                    "{profile} changed a controlled variable"
                );
            } else {
                shared_contract = Some(contract);
            }
        }

        let corpus_path = profile_path("ruliad-r3.stateful-tbptt.corpus.toml");
        let corpus = burn_dragon_universality::load_ruliad_config(&corpus_path)
            .unwrap_or_else(|error| panic!("load {}: {error}", corpus_path.display()));
        assert_eq!(corpus.serialization.document_mode.label(), "single_sample");
        assert_eq!(corpus.serialization.document_chunks.min, 1);
        assert_eq!(corpus.serialization.document_chunks.max, 1);
        assert_eq!(corpus.serialization.document_tokens, 6145);
        assert!(corpus.source_selection.enabled);
        assert!(!corpus.source_selection.feedback_updates_enabled);
    }

    #[test]
    fn ruliad_r3_typed_policy_profile_has_a_long_run_semantic_action_contract() {
        use crate::config::{RuliadProofPolicyCandidateSymmetry, RuliadProofPolicyTrainingMode};

        let config = load_profile("ruliad-r3.typed-policy.training.toml");
        config
            .validate()
            .expect("R3 typed-policy profile should validate");

        assert_eq!(config.training.max_iters, 1_000_000);
        assert!(config.training.auto_batch_size.enabled);
        assert_eq!(config.training.tbptt_chunk_size, Some(512));
        assert!(!config.training.tbptt_persist_across_steps);
        assert_eq!(
            config.training.ruliad_supervision.mode,
            RuliadSupervisionMode::AnswerCompletion
        );
        let policy = config.training.ruliad_supervision.proof_policy;
        assert!(policy.enabled);
        assert_eq!(policy.mode, RuliadProofPolicyTrainingMode::StaticExpert);
        assert_eq!(policy.every_steps, 2);
        assert_eq!(policy.start_after_steps, 128);
        assert_eq!(policy.max_rows_per_update, 8);
        assert_eq!(
            policy.candidate_symmetry,
            RuliadProofPolicyCandidateSymmetry::BalancedRotation
        );
        assert_eq!(
            config.training.ruliad_policy_probe.candidate_symmetry,
            RuliadProofPolicyCandidateSymmetry::CyclicOrbitAverage
        );
        assert_eq!(
            config
                .training
                .ruliad_policy_probe
                .effective_closed_loop_every_epochs(),
            16
        );
        assert!(config.training.ruliad_policy_probe.promotion_gate.enabled);
    }

    #[test]
    fn ruliad_r3_semantic_energy_profile_decouples_policy_from_language_serialization() {
        use crate::config::RuliadProofPolicyScoring;

        let config = load_profile("ruliad-r3.action-policy-semantic-energy-fixed-ablation.toml");
        config
            .validate()
            .expect("R3 semantic-energy profile should validate");
        assert!(
            config
                .model
                .sequence_score_head
                .is_some_and(|head| head.enabled)
        );
        assert_eq!(
            config.training.ruliad_supervision.proof_policy.scoring,
            RuliadProofPolicyScoring::SemanticEnergy
        );
        assert_eq!(
            config.training.ruliad_policy_probe.scoring,
            RuliadProofPolicyScoring::SemanticEnergy
        );
        let proof_policy = config.training.ruliad_supervision.proof_policy;
        assert_eq!(proof_policy.counterfactual_targets_per_state, 1);
        assert_eq!(proof_policy.target_variants_per_state(), 2);
        assert_eq!(proof_policy.semantic_rows_per_update(), 8);
        assert_eq!(proof_policy.base_semantic_rows_per_update(), 4);
        assert!(
            build_model_config(&config.model, config.training.block_size)
                .sequence_score_head
                .enabled
        );
        assert_eq!(
            build_model_config(&config.model, config.training.block_size)
                .sequence_score_head
                .projection_dim,
            64
        );
    }

    #[test]
    fn ruliad_r3_semantic_energy_head_only_profile_is_explicit_and_valid() {
        use crate::config::{RuliadProofPolicyGradientScope, RuliadProofPolicyScoring};

        let config =
            load_profile("ruliad-r3.action-policy-semantic-energy-head-only-fixed-ablation.toml");
        config
            .validate()
            .expect("R3 head-only semantic-energy profile should validate");
        let policy = config.training.ruliad_supervision.proof_policy;
        assert_eq!(policy.scoring, RuliadProofPolicyScoring::SemanticEnergy);
        assert_eq!(
            policy.gradient_scope,
            RuliadProofPolicyGradientScope::ScoreHeadOnly
        );

        let fullrate = load_profile(
            "ruliad-r3.action-policy-semantic-energy-head-only-fullrate-ablation.toml",
        );
        fullrate
            .validate()
            .expect("R3 full-rate head-only semantic-energy profile should validate");
        let policy = fullrate.training.ruliad_supervision.proof_policy;
        assert_eq!(policy.scoring, RuliadProofPolicyScoring::SemanticEnergy);
        assert_eq!(
            policy.gradient_scope,
            RuliadProofPolicyGradientScope::ScoreHeadOnly
        );
        assert_eq!(policy.weight, 1.0);
        assert_eq!(policy.every_steps, 1);
        assert_eq!(policy.start_after_steps, 0);
        assert_eq!(policy.max_rows_per_update, 32);
        assert_eq!(policy.max_presentation_rows_per_update, 32);
    }

    #[test]
    fn score_head_only_gradient_scope_requires_semantic_energy() {
        use crate::config::{RuliadProofPolicyGradientScope, RuliadProofPolicyScoring};

        let mut config =
            load_profile("ruliad-r3.action-policy-semantic-energy-fixed-ablation.toml");
        let policy = &mut config.training.ruliad_supervision.proof_policy;
        policy.scoring = RuliadProofPolicyScoring::CompletionLikelihood;
        policy.gradient_scope = RuliadProofPolicyGradientScope::ScoreHeadOnly;
        let error = config
            .validate()
            .expect_err("head-only scope must not silently target the language head");
        assert!(
            error
                .to_string()
                .contains("gradient_scope=score_head_only requires scoring=semantic_energy"),
            "{error}"
        );
    }

    #[test]
    fn language_head_only_profile_is_explicit_untied_and_valid() {
        use crate::config::{RuliadProofPolicyGradientScope, RuliadProofPolicyScoring};

        let config =
            load_profile("ruliad-r3.semantic-action-language-head-only-fixed-ablation.toml");
        config
            .validate()
            .expect("R3 language-head-only completion profile should validate");
        let policy = config.training.ruliad_supervision.proof_policy;
        assert_eq!(
            policy.scoring,
            RuliadProofPolicyScoring::CompletionLikelihood
        );
        assert_eq!(
            policy.gradient_scope,
            RuliadProofPolicyGradientScope::LanguageHeadOnly
        );
        assert_eq!(policy.counterfactual_targets_per_state, 1);
        assert_eq!(policy.target_variants_per_state(), 2);
        assert_eq!(policy.base_semantic_rows_per_update(), 4);
        assert!(!config.model.tie_input_output_embeddings.unwrap_or(false));
    }

    #[test]
    fn language_head_only_gradient_scope_rejects_tied_embeddings_and_energy_scoring() {
        use crate::config::{RuliadProofPolicyGradientScope, RuliadProofPolicyScoring};

        let mut tied =
            load_profile("ruliad-r3.semantic-action-language-head-only-fixed-ablation.toml");
        tied.model.tie_input_output_embeddings = Some(true);
        let error = tied
            .validate()
            .expect_err("language-head-only scope must not update tied input embeddings");
        assert!(
            error
                .to_string()
                .contains("language_head_only requires model.tie_input_output_embeddings=false"),
            "{error}"
        );

        let mut energy =
            load_profile("ruliad-r3.semantic-action-language-head-only-fixed-ablation.toml");
        energy.model.sequence_score_head = Some(burn_dragon_core::SequenceScoreHeadConfig {
            enabled: true,
            ..Default::default()
        });
        energy.training.ruliad_supervision.proof_policy.scoring =
            RuliadProofPolicyScoring::SemanticEnergy;
        energy
            .training
            .ruliad_supervision
            .proof_policy
            .gradient_scope = RuliadProofPolicyGradientScope::LanguageHeadOnly;
        let error = energy
            .validate()
            .expect_err("language-head-only scope must target completion likelihood");
        assert!(
            error
                .to_string()
                .contains("language_head_only requires scoring=completion_likelihood"),
            "{error}"
        );

        let mut factorized =
            load_profile("ruliad-r3.semantic-action-language-head-only-fixed-ablation.toml");
        factorized.model.language_head =
            Some(burn_dragon_core::LanguageHeadConfig::NcaFactorizedPatch {
                state_count: 2,
                patch_size: 2,
                frame_special_tokens: false,
                eos_id: None,
            });
        let error = factorized
            .validate()
            .expect_err("language-head-only scope requires a flat token projection");
        assert!(
            error.to_string().contains(
                "language_head_only requires model.language_head.type=standard_token_classification"
            ),
            "{error}"
        );

        let mut conditioned =
            load_profile("ruliad-r3.semantic-action-language-head-only-fixed-ablation.toml");
        let mut latent = burn_dragon_core::LatentReasoningConfig::default();
        latent.step_conditioned_decoder = true;
        conditioned.model.latent_reasoning = Some(latent);
        let error = conditioned
            .validate()
            .expect_err("language-head-only scope must not update latent-step conditioning");
        assert!(
            error.to_string().contains(
                "language_head_only requires model.latent_reasoning.step_conditioned_decoder=false"
            ),
            "{error}"
        );
    }

    #[test]
    fn paired_dagger_validation_preserves_causal_and_model_visited_rows() {
        use crate::config::{
            RuliadProofPolicyEffectiveMode, RuliadProofPolicyScoring, RuliadProofPolicyTrainingMode,
        };

        let mut config =
            load_profile("ruliad-r3.action-policy-semantic-energy-fixed-ablation.toml");
        let proof_policy = &mut config.training.ruliad_supervision.proof_policy;
        proof_policy.mode = RuliadProofPolicyTrainingMode::StaticThenPairedDagger;
        proof_policy.dagger_start_after_steps = 512;
        proof_policy.stratified_difficulty_levels = 1;
        proof_policy.rollout_steps = 2;
        config
            .validate()
            .expect("bounded semantic-energy paired DAgger should validate");
        let policy = config.training.ruliad_supervision.proof_policy;
        assert_eq!(policy.scoring, RuliadProofPolicyScoring::SemanticEnergy);
        assert_eq!(
            policy.mode,
            RuliadProofPolicyTrainingMode::StaticThenPairedDagger
        );
        assert_eq!(
            policy.effective_mode(511),
            RuliadProofPolicyEffectiveMode::StaticExpert
        );
        assert_eq!(
            policy.effective_mode(512),
            RuliadProofPolicyEffectiveMode::PairedDagger
        );
        assert_eq!(policy.stratified_difficulty_levels, 1);
        assert_eq!(policy.rollout_steps, 2);
        assert_eq!(policy.counterfactual_targets_per_state, 1);
        assert_eq!(policy.semantic_rows_per_update(), 8);
        assert_eq!(policy.base_semantic_rows_per_update(), 4);

        let mut invalid = config;
        invalid
            .training
            .ruliad_supervision
            .proof_policy
            .stratified_difficulty_levels = 2;
        let error = invalid
            .validate()
            .expect_err("paired DAgger must reserve one visited state per trajectory");
        assert!(
            error
                .to_string()
                .contains("exceeds the paired DAgger trajectory budget"),
            "{error}"
        );
    }

    #[test]
    fn semantic_energy_rejects_an_empty_compatibility_projection() {
        let mut config =
            load_profile("ruliad-r3.action-policy-semantic-energy-fixed-ablation.toml");
        config
            .model
            .sequence_score_head
            .as_mut()
            .expect("semantic-energy head")
            .projection_dim = 0;
        let error = config
            .validate()
            .expect_err("zero-rank compatibility head must be rejected");
        assert!(
            error
                .to_string()
                .contains("sequence_score_head.projection_dim must be > 0"),
            "{error}"
        );
    }

    #[test]
    fn ruliad_counterfactual_policy_requires_energy_candidates_and_complete_groups() {
        use crate::config::RuliadProofPolicyScoring;

        let mut config =
            load_profile("ruliad-r3.action-policy-semantic-energy-fixed-ablation.toml");
        config.training.ruliad_supervision.proof_policy.scoring =
            RuliadProofPolicyScoring::CompletionLikelihood;
        let error = config
            .validate()
            .expect_err("full-model completion counterfactuals must be rejected");
        assert!(
            error
                .to_string()
                .contains("requires scoring=semantic_energy or completion_likelihood with gradient_scope=language_head_only")
        );

        let mut config =
            load_profile("ruliad-r3.action-policy-semantic-energy-fixed-ablation.toml");
        config
            .training
            .ruliad_supervision
            .proof_policy
            .counterfactual_targets_per_state = 4;
        let error = config
            .validate()
            .expect_err("counterfactual targets must leave an original candidate class");
        assert!(error.to_string().contains("must be less than candidates"));

        let mut config =
            load_profile("ruliad-r3.action-policy-semantic-energy-fixed-ablation.toml");
        config
            .training
            .ruliad_supervision
            .proof_policy
            .max_rows_per_update = 1;
        let error = config
            .validate()
            .expect_err("row budget must fit a complete target pair");
        assert!(
            error
                .to_string()
                .contains("target-variant presentation group")
        );
    }

    #[test]
    fn ruliad_action_policy_profiles_load_with_explicit_search_contracts() {
        use crate::config::RuliadProofPolicyCandidateSymmetry::{
            BalancedRotation, CyclicOrbitAverage,
        };
        for (profile, beam_width, dagger, symmetry) in [
            (
                "ruliad-r3.action-policy-fixed-ablation.toml",
                1,
                false,
                BalancedRotation,
            ),
            (
                "ruliad-r3.semantic-action-fixed-ablation.toml",
                1,
                false,
                BalancedRotation,
            ),
            (
                "ruliad-r3.semantic-action-static-fixed-ablation.toml",
                1,
                true,
                BalancedRotation,
            ),
            (
                "ruliad-r3.semantic-action-static-every-step-fixed-ablation.toml",
                1,
                true,
                BalancedRotation,
            ),
            (
                "ruliad-r3.semantic-action-static-every-two-steps-fixed-ablation.toml",
                1,
                true,
                BalancedRotation,
            ),
            (
                "ruliad-r3.semantic-action-static-prefix-fixed-ablation.toml",
                1,
                true,
                BalancedRotation,
            ),
            (
                "ruliad-r3.semantic-action-static-marginal-fixed-ablation.toml",
                1,
                true,
                BalancedRotation,
            ),
            (
                "ruliad-r3.action-policy-beam4-fixed-ablation.toml",
                4,
                false,
                BalancedRotation,
            ),
            (
                "ruliad-r3.action-policy-dagger-fixed-ablation.toml",
                1,
                true,
                BalancedRotation,
            ),
            (
                "ruliad-r3.action-policy-dagger-marginal-fixed-ablation.toml",
                1,
                true,
                BalancedRotation,
            ),
            (
                "ruliad-r3.action-policy-static-marginal-fixed-ablation.toml",
                1,
                true,
                BalancedRotation,
            ),
            (
                "ruliad-r3.action-policy-static-orbit-marginal-fixed-ablation.toml",
                1,
                true,
                CyclicOrbitAverage,
            ),
            (
                "ruliad-r3.action-policy-static-orbit-worst-marginal-fixed-ablation.toml",
                1,
                true,
                CyclicOrbitAverage,
            ),
            (
                "ruliad-r3.action-policy-bc-paired-dagger-marginal-fixed-ablation.toml",
                1,
                true,
                BalancedRotation,
            ),
            (
                "ruliad-r3.action-policy-bc-paired-dagger-orbit-marginal-fixed-ablation.toml",
                1,
                true,
                CyclicOrbitAverage,
            ),
            (
                "ruliad-r3.action-policy-dagger-beam4-fixed-ablation.toml",
                4,
                true,
                BalancedRotation,
            ),
            (
                "ruliad-r3.action-policy-promotion-audit.toml",
                1,
                false,
                BalancedRotation,
            ),
            (
                "ruliad-r3.action-policy-beam4-promotion-audit.toml",
                4,
                false,
                BalancedRotation,
            ),
        ] {
            let config = load_profile(profile);
            config
                .validate()
                .unwrap_or_else(|error| panic!("validate {profile}: {error}"));
            assert!(config.training.ruliad_probe_generation.enabled, "{profile}");
            assert_eq!(
                config.training.ruliad_probe_generation.max_batch_rows, 64,
                "{profile}"
            );
            assert_eq!(
                config.training.ruliad_probe_generation.minimum_batch_rows, 2,
                "{profile}"
            );
            assert_eq!(
                config
                    .training
                    .ruliad_probe_generation
                    .maximum_prompt_position_span,
                32,
                "{profile}"
            );
            assert_eq!(
                config.training.ruliad_probe_generation.device_buffer_tokens, 4,
                "{profile}"
            );
            assert_eq!(config.training.ruliad_policy_probe.beam_width, beam_width);
            assert_eq!(config.training.ruliad_policy_probe.scoring_batch_rows, 32);
            assert_eq!(
                config.training.ruliad_policy_probe.scoring_token_budget,
                32_768
            );
            assert_eq!(
                config.training.ruliad_policy_probe.scoring_pipeline_depth,
                2
            );
            assert_eq!(
                config.training.ruliad_policy_probe.candidate_symmetry, symmetry,
                "{profile}"
            );
            assert_eq!(
                config.training.ruliad_supervision.proof_policy.enabled,
                dagger
            );
            if dagger {
                assert_eq!(
                    config.training.ruliad_supervision.proof_policy.weight, 0.25,
                    "{profile}"
                );
            }
            let promotion_audit = profile.contains("promotion-audit");
            assert_eq!(
                config.training.ruliad_policy_probe.every_epochs,
                if promotion_audit { 1 } else { 4 },
                "{profile}"
            );
            assert_eq!(
                config
                    .training
                    .ruliad_policy_probe
                    .effective_closed_loop_every_epochs(),
                if promotion_audit { 1 } else { 4 },
                "ablation profiles must retain matched closed-loop cadence: {profile}"
            );
            assert_eq!(
                config.training.ruliad_policy_probe.items,
                if promotion_audit { 64 } else { 16 },
                "{profile}"
            );
            assert_eq!(
                config.training.ruliad_policy_probe.max_steps,
                if promotion_audit { 256 } else { 64 },
                "{profile}"
            );
        }

        let marginal = load_profile("ruliad-r3.action-policy-dagger-marginal-fixed-ablation.toml");
        assert_eq!(
            marginal
                .training
                .ruliad_supervision
                .proof_policy
                .normalization,
            crate::config::RuliadProofPolicyNormalization::VocabularyMarginal
        );
        let semantic_marginal =
            load_profile("ruliad-r3.semantic-action-static-marginal-fixed-ablation.toml");
        assert_eq!(
            semantic_marginal
                .training
                .ruliad_supervision
                .proof_policy
                .normalization,
            crate::config::RuliadProofPolicyNormalization::VocabularyMarginal
        );
        assert_eq!(
            semantic_marginal
                .training
                .ruliad_supervision
                .proof_policy
                .max_rows_per_update,
            8
        );
        assert_eq!(
            semantic_marginal
                .training
                .ruliad_supervision
                .proof_policy
                .max_completion_tokens,
            64
        );
        let semantic_prefix =
            load_profile("ruliad-r3.semantic-action-static-prefix-fixed-ablation.toml");
        assert_eq!(
            semantic_prefix
                .training
                .ruliad_supervision
                .proof_policy
                .normalization,
            crate::config::RuliadProofPolicyNormalization::PrefixConditional
        );
        let semantic_every_step =
            load_profile("ruliad-r3.semantic-action-static-every-step-fixed-ablation.toml");
        assert_eq!(
            semantic_every_step
                .training
                .ruliad_supervision
                .proof_policy
                .every_steps,
            1
        );
        assert_eq!(
            semantic_every_step
                .training
                .ruliad_supervision
                .proof_policy
                .start_after_steps,
            0
        );
        let semantic_every_two_steps =
            load_profile("ruliad-r3.semantic-action-static-every-two-steps-fixed-ablation.toml");
        assert_eq!(
            semantic_every_two_steps
                .training
                .ruliad_supervision
                .proof_policy
                .every_steps,
            2
        );
        assert_eq!(
            semantic_every_two_steps
                .training
                .ruliad_supervision
                .proof_policy
                .start_after_steps,
            0
        );
        assert_eq!(
            marginal
                .training
                .ruliad_supervision
                .proof_policy
                .candidate_symmetry,
            crate::config::RuliadProofPolicyCandidateSymmetry::BalancedRotation
        );
        let static_marginal =
            load_profile("ruliad-r3.action-policy-static-marginal-fixed-ablation.toml");
        assert_eq!(
            static_marginal
                .training
                .ruliad_supervision
                .proof_policy
                .mode,
            crate::config::RuliadProofPolicyTrainingMode::StaticExpert
        );
        assert_eq!(
            static_marginal
                .training
                .ruliad_supervision
                .proof_policy
                .every_steps,
            4
        );
        let orbit =
            load_profile("ruliad-r3.action-policy-static-orbit-marginal-fixed-ablation.toml");
        assert_eq!(
            orbit
                .training
                .ruliad_supervision
                .proof_policy
                .candidate_symmetry,
            CyclicOrbitAverage
        );
        assert_eq!(
            orbit
                .training
                .ruliad_supervision
                .proof_policy
                .max_presentation_rows_per_update,
            32
        );
        assert_eq!(
            orbit
                .training
                .ruliad_supervision
                .proof_policy
                .semantic_rows_per_update(),
            8
        );
        let worst_orbit =
            load_profile("ruliad-r3.action-policy-static-orbit-worst-marginal-fixed-ablation.toml");
        assert_eq!(
            worst_orbit
                .training
                .ruliad_supervision
                .proof_policy
                .presentation_risk,
            crate::config::RuliadProofPolicyPresentationRisk::Worst
        );
        let scheduled =
            load_profile("ruliad-r3.action-policy-bc-paired-dagger-marginal-fixed-ablation.toml");
        assert_eq!(
            scheduled.training.ruliad_supervision.proof_policy.mode,
            crate::config::RuliadProofPolicyTrainingMode::StaticThenPairedDagger
        );
        assert_eq!(
            scheduled
                .training
                .ruliad_supervision
                .proof_policy
                .dagger_start_after_steps,
            768
        );
        assert_eq!(
            scheduled
                .training
                .ruliad_supervision
                .proof_policy
                .effective_mode(767),
            crate::config::RuliadProofPolicyEffectiveMode::StaticExpert
        );
        assert_eq!(
            scheduled
                .training
                .ruliad_supervision
                .proof_policy
                .effective_mode(768),
            crate::config::RuliadProofPolicyEffectiveMode::PairedDagger
        );
    }

    #[test]
    fn ruliad_action_policy_probe_rejects_zero_cadence() {
        let mut config = load_profile("ruliad-r3.action-policy-fixed-ablation.toml");
        config.training.ruliad_policy_probe.every_epochs = 0;

        let error = config.validate().expect_err("zero cadence must fail");
        assert!(
            error
                .to_string()
                .contains("ruliad_policy_probe.every_epochs must be > 0"),
            "{error}"
        );
    }

    #[test]
    fn ruliad_action_policy_probe_rejects_zero_closed_loop_cadence() {
        let mut config = load_profile("ruliad-r3.action-policy-fixed-ablation.toml");
        config.training.ruliad_policy_probe.closed_loop_every_epochs = Some(0);

        let error = config
            .validate()
            .expect_err("zero closed-loop cadence must fail");
        assert!(
            error
                .to_string()
                .contains("ruliad_policy_probe.closed_loop_every_epochs must be > 0"),
            "{error}"
        );
    }

    #[test]
    fn ruliad_probe_generation_rejects_unbounded_or_empty_batches() {
        let mut config = load_profile("ruliad-r3.action-policy-fixed-ablation.toml");
        config.training.ruliad_probe_generation.minimum_batch_rows = config
            .training
            .ruliad_probe_generation
            .max_batch_rows
            .saturating_add(1);
        let error = config
            .validate()
            .expect_err("minimum rows above maximum must fail");
        assert!(
            error
                .to_string()
                .contains("minimum_batch_rows must be in 1..=max_batch_rows"),
            "{error}"
        );

        config.training.ruliad_probe_generation.minimum_batch_rows = 2;
        config
            .training
            .ruliad_probe_generation
            .maximum_prompt_position_span = 0;
        let error = config.validate().expect_err("zero prompt span must fail");
        assert!(
            error
                .to_string()
                .contains("maximum_prompt_position_span must be > 0"),
            "{error}"
        );

        config
            .training
            .ruliad_probe_generation
            .maximum_prompt_position_span = 32;
        config.training.ruliad_probe_generation.device_buffer_tokens = 0;
        let error = config.validate().expect_err("zero device buffer must fail");
        assert!(
            error
                .to_string()
                .contains("device_buffer_tokens must be > 0"),
            "{error}"
        );

        config.training.ruliad_probe_generation.device_buffer_tokens = 4;
        config.training.ruliad_probe_generation.max_in_flight_rows = 0;
        let error = config
            .validate()
            .expect_err("zero in-flight row bound must fail");
        assert!(
            error.to_string().contains("max_in_flight_rows must be > 0"),
            "{error}"
        );
    }

    #[test]
    fn ruliad_static_then_paired_dagger_requires_an_aligned_later_transition() {
        let mut config =
            load_profile("ruliad-r3.action-policy-bc-paired-dagger-marginal-fixed-ablation.toml");
        config
            .training
            .ruliad_supervision
            .proof_policy
            .dagger_start_after_steps = 128;
        let error = config
            .validate()
            .expect_err("DAgger transition must follow static warmup");
        assert!(error.to_string().contains("must exceed start_after_steps"));

        config
            .training
            .ruliad_supervision
            .proof_policy
            .dagger_start_after_steps = 769;
        let error = config
            .validate()
            .expect_err("DAgger transition must align to policy cadence");
        assert!(error.to_string().contains("must align with every_steps"));

        config
            .training
            .ruliad_supervision
            .proof_policy
            .dagger_start_after_steps = 768;
        config
            .training
            .ruliad_supervision
            .proof_policy
            .max_rows_per_update = 1;
        let error = config
            .validate()
            .expect_err("paired DAgger needs both row populations");
        assert!(error.to_string().contains("must be at least 2"));
    }

    #[test]
    fn ruliad_orbit_policy_requires_a_complete_bounded_presentation_set() {
        let mut config =
            load_profile("ruliad-r3.action-policy-static-orbit-marginal-fixed-ablation.toml");
        let proof_policy = &mut config.training.ruliad_supervision.proof_policy;
        proof_policy.max_presentation_rows_per_update = proof_policy.candidates - 1;

        let error = config
            .validate()
            .expect_err("an incomplete orbit must not be materialized");
        assert!(error.to_string().contains("fit one complete"), "{error}");
    }

    #[test]
    fn ruliad_worst_presentation_risk_requires_an_exact_orbit() {
        let mut config =
            load_profile("ruliad-r3.action-policy-static-orbit-worst-marginal-fixed-ablation.toml");
        assert!(config.validate().is_ok());
        config
            .training
            .ruliad_supervision
            .proof_policy
            .candidate_symmetry =
            crate::config::RuliadProofPolicyCandidateSymmetry::BalancedRotation;

        let error = config
            .validate()
            .expect_err("worst presentation risk needs a complete orbit");
        assert!(
            error
                .to_string()
                .contains("presentation_risk=worst requires"),
            "{error}"
        );
    }

    #[test]
    fn ruliad_paired_orbit_policy_requires_both_semantic_populations() {
        let mut config = load_profile(
            "ruliad-r3.action-policy-bc-paired-dagger-orbit-marginal-fixed-ablation.toml",
        );
        let proof_policy = &mut config.training.ruliad_supervision.proof_policy;
        proof_policy.rollout_steps = 1;
        proof_policy.max_presentation_rows_per_update = proof_policy.candidates;

        let error = config
            .validate()
            .expect_err("paired DAgger needs two complete semantic orbits");
        assert!(
            error.to_string().contains("at least 2 base semantic rows"),
            "{error}"
        );
    }

    #[test]
    fn random_scaffold_ruliad_matrix_profiles_load_and_validate() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../config/language/experiments/random_scaffold");
        for profile in [
            "ruliad-screen.dense.toml",
            "ruliad-screen.rank1.toml",
            "ruliad-screen.rank4.toml",
            "ruliad-screen.rank8.toml",
            "ruliad-screen.rank16.toml",
            "ruliad-screen.rs-rank8.toml",
            "ruliad-screen.rs-rank16.toml",
            "ruliad-screen.rs-rank32.toml",
            "ruliad-screen.rs-rank64.toml",
            "ruliad-screen.rank8-fixed-gain.toml",
            "ruliad-screen.rademacher-rank8.toml",
            "ruliad-parity.dense.toml",
            "ruliad-parity.rank8.toml",
            "ruliad-parity.rs-rank16.toml",
            "ruliad-parity.rs-rank32.toml",
        ] {
            let path = root.join(profile);
            let config = load_training_config(std::slice::from_ref(&path))
                .unwrap_or_else(|error| panic!("load {}: {error}", path.display()));
            config
                .validate()
                .unwrap_or_else(|error| panic!("validate {}: {error}", path.display()));
        }
    }

    #[test]
    fn ruliad_64k_dynamics_anchor_rejects_unsafe_fixed_large_batch() {
        let mut config = load_profile("ruliad-1m-la-64k.training.toml");
        config.training.batch_size = 32;
        config.training.auto_batch_size.enabled = false;
        config.training.dynamics_anchor.enabled = true;

        let err = config
            .validate()
            .expect_err("64k anchored profile should reject fixed batch sizes above one");
        assert!(
            err.to_string()
                .contains("dynamics_anchor with fixed training.batch_size > 1"),
            "unexpected error: {err}"
        );
    }

    fn profile_path(file_name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../burn_dragon_p2p/deploy/profiles")
            .join(file_name)
    }

    #[test]
    fn eggroll_optimizer_config_validates_for_single_next_token_training() {
        let mut config = parse_config("");
        config.optimizer.name = OptimizerKind::Eggroll;
        config.optimizer.eggroll.population.population_size = 2;
        config.optimizer.eggroll.population.population_chunk_size = 2;
        config
            .validate()
            .expect("minimal single-device EGGROLL config should validate");
    }

    fn estimate_profile_parameter_budget(config: &TrainingConfig) -> usize {
        let layers = config.model.n_layer.unwrap_or(8);
        let width = config.model.n_embd.unwrap_or(512);
        let latent = config.model.latent_total.unwrap_or(width * 2);
        let vocab = config.dataset.tokenizer.vocab_size();
        let embeddings = vocab.saturating_mul(width).saturating_mul(2);
        let per_layer = width
            .saturating_mul(width)
            .saturating_mul(4)
            .saturating_add(width.saturating_mul(latent).saturating_mul(4));
        embeddings.saturating_add(layers.saturating_mul(per_layer))
    }

    #[test]
    fn eggroll_optimizer_rejects_gradient_accumulation() {
        let mut config = parse_config("");
        config.optimizer.name = OptimizerKind::Eggroll;
        config.training.gradient_accumulation_steps = 2;
        let err = config
            .validate()
            .expect_err("EGGROLL gradient accumulation should fail");
        assert!(
            err.to_string().contains("gradient_accumulation_steps = 1"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn eggroll_optimizer_rejects_gradient_correction() {
        let mut config = parse_config("");
        config.optimizer.name = OptimizerKind::Eggroll;
        config.optimizer.eggroll.gradient_learning_rate = Some(1.0e-3);
        let err = config
            .validate()
            .expect_err("EGGROLL gradient correction should fail");
        assert!(
            err.to_string().contains("gradient_learning_rate"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn eggroll_optimizer_rejects_continual_backprop() {
        let mut config = parse_config("");
        config.optimizer.name = OptimizerKind::Eggroll;
        config.training.continual_backprop.enabled = true;
        let err = config
            .validate()
            .expect_err("EGGROLL continual backprop should fail");
        assert!(
            err.to_string().contains("continual_backprop.enabled"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn eggroll_optimizer_rejects_neuron_scaling() {
        let mut config = parse_config("");
        config.optimizer.name = OptimizerKind::Eggroll;
        config.training.neuron_scaling.enabled = true;
        let err = config
            .validate()
            .expect_err("EGGROLL neuron scaling should fail");
        assert!(
            err.to_string().contains("neuron_scaling.enabled"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn ruliad_answer_completion_requires_ruliad_dataset() {
        let mut config = parse_config("");
        config.training.ruliad_supervision.mode = RuliadSupervisionMode::AnswerCompletion;
        let err = config
            .validate()
            .expect_err("answer-completion supervision should reject non-ruliad datasets");
        assert!(
            err.to_string().contains("universality_ruliad"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn ruliad_answer_completion_validates_for_ruliad_adamw() {
        let mut config = parse_config("");
        config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
            config: "target/test-ruliad.toml".into(),
        };
        config.training.ruliad_supervision.mode = RuliadSupervisionMode::AnswerCompletion;
        config
            .validate()
            .expect("ruliad answer-completion AdamW config should validate");
    }

    #[test]
    fn ruliad_answer_ranking_requires_answer_target_mask_mode() {
        let mut config = parse_config("");
        config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
            config: "target/test-ruliad.toml".into(),
        };
        config.training.ruliad_supervision.mode = RuliadSupervisionMode::FullDocument;
        config.training.ruliad_supervision.answer_ranking.enabled = true;
        let err = config
            .validate()
            .expect_err("answer ranking should require an answer target mask mode");
        assert!(
            err.to_string().contains("answer target masks"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn ruliad_answer_ranking_validates_for_ruliad_answer_completion() {
        let mut config = parse_config("");
        config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
            config: "target/test-ruliad.toml".into(),
        };
        config.training.ruliad_supervision.mode = RuliadSupervisionMode::AnswerCompletion;
        config.training.ruliad_supervision.answer_ranking.enabled = true;
        config
            .validate()
            .expect("ruliad answer ranking should validate with answer-completion masks");
    }

    #[test]
    fn ruliad_answer_ranking_rejects_invalid_parameters() {
        let mut config = parse_config("");
        config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
            config: "target/test-ruliad.toml".into(),
        };
        config.training.ruliad_supervision.mode = RuliadSupervisionMode::AnswerCompletion;
        config.training.ruliad_supervision.answer_ranking.enabled = true;
        config
            .training
            .ruliad_supervision
            .answer_ranking
            .corrupt_offset = 0;
        let err = config
            .validate()
            .expect_err("zero corrupt offset should fail");
        assert!(
            err.to_string().contains("corrupt_offset"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn ruliad_answer_denoising_requires_answer_target_mask_mode() {
        let mut config = parse_config("");
        config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
            config: "target/test-ruliad.toml".into(),
        };
        config.training.ruliad_supervision.mode = RuliadSupervisionMode::FullDocument;
        config.training.ruliad_supervision.answer_denoising.enabled = true;
        let err = config
            .validate()
            .expect_err("answer denoising should require an answer target mask mode");
        assert!(
            err.to_string().contains("answer target masks"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn ruliad_answer_denoising_validates_for_ruliad_answer_completion() {
        let mut config = parse_config("");
        config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
            config: "target/test-ruliad.toml".into(),
        };
        config.training.ruliad_supervision.mode = RuliadSupervisionMode::AnswerCompletion;
        config.training.ruliad_supervision.answer_denoising.enabled = true;
        config
            .validate()
            .expect("ruliad answer denoising should validate with answer-completion masks");
    }

    #[test]
    fn ruliad_answer_denoising_rejects_invalid_parameters() {
        let mut config = parse_config("");
        config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
            config: "target/test-ruliad.toml".into(),
        };
        config.training.ruliad_supervision.mode = RuliadSupervisionMode::AnswerCompletion;
        config.training.ruliad_supervision.answer_denoising.enabled = true;
        config
            .training
            .ruliad_supervision
            .answer_denoising
            .probability = 1.5;
        let err = config
            .validate()
            .expect_err("invalid denoising probability should fail");
        assert!(
            err.to_string().contains("answer_denoising.probability"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn ruliad_verifier_reward_requires_ruliad_dataset() {
        let mut config = parse_config("");
        config.training.ruliad_supervision.verifier_reward.enabled = true;
        let err = config
            .validate()
            .expect_err("verifier reward should require ruliad data");
        assert!(
            err.to_string().contains("universality_ruliad"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn ruliad_verifier_reward_rejects_invalid_parameters() {
        let mut config = parse_config("");
        config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
            config: "target/test-ruliad.toml".into(),
        };
        config.training.ruliad_supervision.verifier_reward.enabled = true;
        config
            .training
            .ruliad_supervision
            .verifier_reward
            .group_size = 1;
        let err = config
            .validate()
            .expect_err("single-sample verifier reward group should fail");
        assert!(
            err.to_string().contains("verifier_reward.group_size"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn ruliad_generated_attractor_replay_rejects_invalid_diversity_guard() {
        let mut config = parse_config("");
        config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
            config: "target/test-ruliad.toml".into(),
        };
        let verifier_reward = &mut config.training.ruliad_supervision.verifier_reward;
        verifier_reward.enabled = true;
        verifier_reward.generated_attractor_replay_capacity = 8;
        verifier_reward.field_binding_contrast_weight = 0.01;
        verifier_reward.field_binding_contrast_pair_weight = 0.5;
        verifier_reward.generated_attractor_replay_min_distinct_answers = 0;
        let err = config
            .validate()
            .expect_err("zero generated-attractor distinct-answer guard should fail");
        assert!(
            err.to_string()
                .contains("generated_attractor_replay_min_distinct_answers"),
            "unexpected error: {err}"
        );

        let mut config = parse_config("");
        config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
            config: "target/test-ruliad.toml".into(),
        };
        let verifier_reward = &mut config.training.ruliad_supervision.verifier_reward;
        verifier_reward.enabled = true;
        verifier_reward.generated_attractor_replay_capacity = 8;
        verifier_reward.field_binding_contrast_weight = 0.01;
        verifier_reward.field_binding_contrast_pair_weight = 0.5;
        verifier_reward.generated_attractor_replay_max_dominant_fraction = 1.25;
        let err = config
            .validate()
            .expect_err("dominant generated-attractor fraction above one should fail");
        assert!(
            err.to_string()
                .contains("generated_attractor_replay_max_dominant_fraction"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn ruliad_verifier_reward_validates_for_local_ruliad_next_token() {
        let mut config = parse_config("");
        config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
            config: "target/test-ruliad.toml".into(),
        };
        config.training.ruliad_supervision.verifier_reward.enabled = true;
        config
            .validate()
            .expect("verifier reward should validate for local ruliad next-token training");
    }

    #[test]
    fn ruliad_verifier_reward_vpo_rejects_zero_scalarizations() {
        let mut config = parse_config("");
        config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
            config: "target/test-ruliad.toml".into(),
        };
        config.training.ruliad_supervision.verifier_reward.enabled = true;
        config.training.ruliad_supervision.verifier_reward.mode =
            RuliadVerifierRewardMode::VpoIndependent;
        config
            .training
            .ruliad_supervision
            .verifier_reward
            .vpo_scalarizations = 0;
        let err = config
            .validate()
            .expect_err("zero VPO scalarization count should fail");
        assert!(
            err.to_string().contains("vpo_scalarizations"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn ruliad_supervision_rejects_invalid_answer_value_weight() {
        let mut config = parse_config("");
        config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
            config: "target/test-ruliad.toml".into(),
        };
        config.training.ruliad_supervision.mode = RuliadSupervisionMode::AnswerCompletion;
        config.training.ruliad_supervision.answer_value_token_weight = 0;
        let err = config
            .validate()
            .expect_err("zero answer value token weight should fail");
        assert!(
            err.to_string().contains("answer_value_token_weight"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn ruliad_supervision_rejects_invalid_answer_schema_weight() {
        let mut config = parse_config("");
        config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
            config: "target/test-ruliad.toml".into(),
        };
        config.training.ruliad_supervision.mode = RuliadSupervisionMode::AnswerCompletion;
        config
            .training
            .ruliad_supervision
            .answer_schema_token_weight = 0;
        let err = config
            .validate()
            .expect_err("zero answer schema token weight should fail");
        assert!(
            err.to_string().contains("answer_schema_token_weight"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn ruliad_supervision_rejects_invalid_answer_schema_start_weight() {
        let mut config = parse_config("");
        config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
            config: "target/test-ruliad.toml".into(),
        };
        config.training.ruliad_supervision.mode = RuliadSupervisionMode::AnswerCompletion;
        config
            .training
            .ruliad_supervision
            .answer_schema_start_token_weight = 0;
        let err = config
            .validate()
            .expect_err("zero answer schema start token weight should fail");
        assert!(
            err.to_string().contains("answer_schema_start_token_weight"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn ruliad_answer_contract_rejects_invalid_prompt_schema_value_weight() {
        let mut config = parse_config("");
        config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
            config: "target/test-ruliad.toml".into(),
        };
        config.training.ruliad_supervision.mode = RuliadSupervisionMode::AnswerCompletion;
        config.training.ruliad_supervision.answer_contract.enabled = true;
        config.training.ruliad_supervision.answer_contract.weight = 0.25;
        config
            .training
            .ruliad_supervision
            .answer_contract
            .prompt_schema_value_weight = -1.0;
        let err = config
            .validate()
            .expect_err("negative prompt-schema value weight should fail");
        assert!(
            err.to_string().contains("prompt_schema_value_weight"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn ruliad_supervision_rejects_invalid_answer_close_marker_weight() {
        let mut config = parse_config("");
        config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
            config: "target/test-ruliad.toml".into(),
        };
        config.training.ruliad_supervision.mode = RuliadSupervisionMode::AnswerCompletion;
        config
            .training
            .ruliad_supervision
            .answer_close_marker_weight = 0;
        let err = config
            .validate()
            .expect_err("zero answer close marker weight should fail");
        assert!(
            err.to_string().contains("answer_close_marker_weight"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn ruliad_verifier_reward_vpo_rejects_invalid_mass_floors() {
        let mut config = parse_config("");
        config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
            config: "target/test-ruliad.toml".into(),
        };
        config.training.ruliad_supervision.verifier_reward.enabled = true;
        config.training.ruliad_supervision.verifier_reward.mode =
            RuliadVerifierRewardMode::VpoIndependent;
        config
            .training
            .ruliad_supervision
            .verifier_reward
            .vpo_correctness_mass_floor = 0.8;
        config
            .training
            .ruliad_supervision
            .verifier_reward
            .vpo_completion_health_mass_floor = 0.3;
        let err = config
            .validate()
            .expect_err("VPO mass floors above one should fail");
        assert!(
            err.to_string().contains("mass floors"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn ruliad_verifier_reward_rejects_invalid_advantage_clip_gate() {
        let mut config = parse_config("");
        config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
            config: "target/test-ruliad.toml".into(),
        };
        config.training.ruliad_supervision.verifier_reward.enabled = true;
        config
            .training
            .ruliad_supervision
            .verifier_reward
            .max_advantage_clip_fraction = Some(1.25);
        let err = config
            .validate()
            .expect_err("invalid advantage clip fraction should fail");
        assert!(
            err.to_string().contains("max_advantage_clip_fraction"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn ruliad_verifier_reward_rejects_streaming_tbptt() {
        let mut config = parse_config("");
        config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
            config: "target/test-ruliad.toml".into(),
        };
        config.training.ruliad_supervision.verifier_reward.enabled = true;
        config.training.tbptt_persist_across_steps = true;
        let err = config
            .validate()
            .expect_err("verifier reward should reject persistent TBPTT");
        assert!(
            err.to_string().contains("tbptt_persist_across_steps"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn ruliad_verifier_reward_rejects_tbptt_chunking() {
        let mut config = parse_config("");
        config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
            config: "target/test-ruliad.toml".into(),
        };
        config.training.ruliad_supervision.verifier_reward.enabled = true;
        config.training.tbptt_chunk_size = Some(4);
        let err = config
            .validate()
            .expect_err("verifier reward should reject TBPTT chunking");
        assert!(
            err.to_string().contains("tbptt_chunk_size"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn ruliad_structured_contrast_rejects_tbptt_chunking() {
        let mut config = parse_config("");
        config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
            config: "target/test-ruliad.toml".into(),
        };
        config.training.ruliad_supervision.verifier_reward.enabled = true;
        config.training.ruliad_supervision.verifier_reward.weight = 0.0;
        config
            .training
            .ruliad_supervision
            .verifier_reward
            .structured_contrast_weight = 0.01;
        config
            .training
            .ruliad_supervision
            .verifier_reward
            .structured_negative_count = 1;
        config.training.tbptt_chunk_size = Some(4);
        let err = config
            .validate()
            .expect_err("structured contrast should reject TBPTT chunking");
        assert!(
            err.to_string().contains("structured_contrast_weight > 0")
                && err.to_string().contains("tbptt_chunk_size"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn ruliad_rollout_imitation_rejects_tbptt_chunking() {
        let mut config = parse_config("");
        config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
            config: "target/test-ruliad.toml".into(),
        };
        config.training.ruliad_supervision.verifier_reward.enabled = true;
        config.training.ruliad_supervision.verifier_reward.weight = 0.0;
        config
            .training
            .ruliad_supervision
            .verifier_reward
            .rollout_imitation_weight = 0.01;
        config.training.tbptt_chunk_size = Some(4);
        let err = config
            .validate()
            .expect_err("rollout imitation should reject TBPTT chunking");
        assert!(
            err.to_string().contains("rollout_imitation_weight > 0")
                && err.to_string().contains("tbptt_chunk_size"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn ruliad_field_binding_contrast_accepts_tbptt_chunking() {
        let mut config = parse_config("");
        config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
            config: "target/test-ruliad.toml".into(),
        };
        config.training.ruliad_supervision.verifier_reward.enabled = true;
        config.training.ruliad_supervision.verifier_reward.weight = 0.0;
        config
            .training
            .ruliad_supervision
            .verifier_reward
            .field_binding_contrast_weight = 0.01;
        config.training.tbptt_chunk_size = Some(4);
        config.training.tbptt_persist_across_steps = true;
        config
            .validate()
            .expect("field-binding contrast runs as an auxiliary policy-batch forward and should allow TBPTT");
    }

    #[test]
    fn ruliad_structured_recovery_accepts_tbptt_chunking() {
        let mut config = parse_config("");
        config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
            config: "target/test-ruliad.toml".into(),
        };
        config.training.ruliad_supervision.mode = RuliadSupervisionMode::AnswerCompletion;
        config.training.ruliad_supervision.answer_denoising.enabled = true;
        config
            .training
            .ruliad_supervision
            .answer_denoising
            .structured_recovery_weight = 0.01;
        config
            .training
            .ruliad_supervision
            .answer_denoising
            .structured_recovery_negative_count = 1;
        config.training.tbptt_chunk_size = Some(4);
        config.training.tbptt_persist_across_steps = true;
        config
            .validate()
            .expect("structured recovery should run as an auxiliary forward with TBPTT");
    }

    #[test]
    fn ruliad_structured_recovery_accepts_schema_negatives_without_field_mutations() {
        let mut config = parse_config("");
        config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
            config: "target/test-ruliad.toml".into(),
        };
        config.training.ruliad_supervision.mode = RuliadSupervisionMode::AnswerCompletion;
        config.training.ruliad_supervision.answer_denoising.enabled = true;
        config
            .training
            .ruliad_supervision
            .answer_denoising
            .structured_recovery_weight = 0.01;
        config
            .training
            .ruliad_supervision
            .answer_denoising
            .structured_recovery_schema_negative_count = 1;
        config
            .validate()
            .expect("schema-only structured recovery should validate");
    }

    #[test]
    fn ruliad_structured_contrast_accepts_schema_negatives_without_field_mutations() {
        let mut config = parse_config("");
        config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
            config: "target/test-ruliad.toml".into(),
        };
        config.training.ruliad_supervision.mode = RuliadSupervisionMode::AnswerCompletion;
        config.training.ruliad_supervision.verifier_reward.enabled = true;
        config.training.ruliad_supervision.verifier_reward.weight = 0.0;
        config
            .training
            .ruliad_supervision
            .verifier_reward
            .structured_contrast_weight = 0.01;
        config
            .training
            .ruliad_supervision
            .verifier_reward
            .structured_negative_count = 0;
        config
            .training
            .ruliad_supervision
            .verifier_reward
            .structured_template_negative_count = 0;
        config
            .training
            .ruliad_supervision
            .verifier_reward
            .structured_schema_negative_count = 1;
        config
            .validate()
            .expect("schema-only structured contrast should validate");
    }

    #[test]
    fn source_selection_state_path_requires_ruliad_dataset() {
        let mut config = parse_config("");
        config.training.source_selection_state_path = Some("target/source-state.json".into());
        let err = config
            .validate()
            .expect_err("source-selection handoff should reject non-ruliad datasets");
        assert!(
            err.to_string().contains("universality_ruliad"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn source_selection_state_path_validates_for_ruliad_dataset() {
        let mut config = parse_config("");
        config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
            config: "target/test-ruliad.toml".into(),
        };
        config.training.source_selection_state_path = Some("target/source-state.json".into());
        config
            .validate()
            .expect("ruliad source-selection state should validate");
    }

    #[test]
    fn ruliad_answer_completion_rejects_pure_eggroll_dense_ce_path() {
        let mut config = parse_config("");
        config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
            config: "target/test-ruliad.toml".into(),
        };
        config.training.ruliad_supervision.mode = RuliadSupervisionMode::AnswerCompletion;
        config.optimizer.name = OptimizerKind::Eggroll;
        config.optimizer.eggroll.population.population_size = 2;
        config.optimizer.eggroll.population.population_chunk_size = 2;
        let err = config
            .validate()
            .expect_err("answer-completion supervision should reject current pure EGGROLL path");
        assert!(
            err.to_string().contains("ruliad_supervision"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn auto_batch_size_config_validates() {
        parse_config(
            r#"
[training.auto_batch_size]
enabled = true
min_batch_size = 1
max_batch_size = 32
target_device_memory_mb = 90000
probe_steps = 1
recompute_on_neuron_scale = true
"#,
        )
        .validate()
        .expect("auto batch config should validate");
    }

    #[test]
    fn auto_batch_size_rejects_inverted_bounds() {
        let config = parse_config(
            r#"
[training.auto_batch_size]
enabled = true
min_batch_size = 8
max_batch_size = 4
"#,
        );
        let err = config
            .validate()
            .expect_err("inverted auto batch bounds should fail");
        assert!(
            err.to_string()
                .contains("auto_batch_size.max_batch_size must be >= min_batch_size"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn auto_batch_size_rejects_probe_cap_below_min_batch() {
        let config = parse_config(
            r#"
[training.auto_batch_size]
enabled = true
min_batch_size = 8
max_probe_batch_size = 4
"#,
        );
        let err = config
            .validate()
            .expect_err("probe cap below min batch should fail");
        assert!(
            err.to_string()
                .contains("auto_batch_size.max_probe_batch_size must be >= min_batch_size"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn auto_batch_size_rejects_host_fraction_above_ninety_percent() {
        let config = parse_config(
            r#"
[training.auto_batch_size]
enabled = true
max_system_memory_fraction = 0.95
"#,
        );
        let err = config
            .validate()
            .expect_err("host memory fraction above 90% should fail");
        assert!(
            err.to_string()
                .contains("auto_batch_size.max_system_memory_fraction"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn greedy_rollout_unlikelihood_rejects_zero_history() {
        let config = parse_config(
            r#"
[training.greedy_rollout_unlikelihood]
enabled = true
weight = 0.5
history_tokens = 0
"#,
        );
        let err = config
            .validate()
            .expect_err("zero rollout history should fail");
        assert!(
            err.to_string()
                .contains("greedy_rollout_unlikelihood.history_tokens must be > 0"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn greedy_rollout_unlikelihood_rejects_negative_recovery_weight() {
        let config = parse_config(
            r#"
[training.greedy_rollout_unlikelihood]
enabled = true
weight = 0.5
recovery_weight = -1.0
"#,
        );
        let err = config
            .validate()
            .expect_err("negative rollout recovery weight should fail");
        assert!(
            err.to_string()
                .contains("greedy_rollout_unlikelihood.recovery_weight must be finite and >= 0"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn greedy_rollout_unlikelihood_rejects_negative_sequence_recovery_weight() {
        let config = parse_config(
            r#"
[training.greedy_rollout_unlikelihood]
enabled = true
weight = 0.5
sequence_recovery_weight = -1.0
"#,
        );
        let err = config
            .validate()
            .expect_err("negative rollout sequence recovery weight should fail");
        assert!(
            err.to_string().contains(
                "greedy_rollout_unlikelihood.sequence_recovery_weight must be finite and >= 0"
            ),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn greedy_rollout_unlikelihood_rejects_invalid_cycle_lag_range() {
        let config = parse_config(
            r#"
[training.greedy_rollout_unlikelihood]
enabled = true
cycle_weight = 0.5
cycle_min_lag = 32
cycle_max_lag = 16
"#,
        );
        let err = config
            .validate()
            .expect_err("invalid rollout cycle lag range should fail");
        assert!(
            err.to_string()
                .contains("greedy_rollout_unlikelihood.cycle_max_lag"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn greedy_rollout_unlikelihood_rejects_invalid_margin() {
        let config = parse_config(
            r#"
[training.greedy_rollout_unlikelihood]
enabled = true
weight = 0.5
margin_weight = -1.0
"#,
        );
        let err = config
            .validate()
            .expect_err("negative rollout margin weight should fail");
        assert!(
            err.to_string()
                .contains("greedy_rollout_unlikelihood.margin_weight must be finite and >= 0"),
            "unexpected error: {err}"
        );

        let config = parse_config(
            r#"
[training.greedy_rollout_unlikelihood]
enabled = true
weight = 0.5
margin = -0.25
"#,
        );
        let err = config
            .validate()
            .expect_err("negative rollout margin should fail");
        assert!(
            err.to_string()
                .contains("greedy_rollout_unlikelihood.margin must be finite and >= 0"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn degeneracy_gates_reject_invalid_period_thresholds() {
        let config = parse_config(
            r#"
[training.gates]
degeneracy_distinct_2_min_fraction = 1.1
"#,
        );
        let err = config
            .validate()
            .expect_err("invalid distinct-2 threshold should fail");
        assert!(
            err.to_string()
                .contains("degeneracy_distinct_2_min_fraction"),
            "unexpected error: {err}"
        );

        let config = parse_config(
            r#"
[training.gates]
degeneracy_period_2_max_fraction = -0.1
"#,
        );
        let err = config
            .validate()
            .expect_err("invalid period threshold should fail");
        assert!(
            err.to_string().contains("degeneracy_period_2_max_fraction"),
            "unexpected error: {err}"
        );

        let config = parse_config(
            r#"
[training.gates]
degeneracy_period_2_to_16_max_fraction = 1.1
"#,
        );
        let err = config
            .validate()
            .expect_err("invalid long-cycle period threshold should fail");
        assert!(
            err.to_string()
                .contains("degeneracy_period_2_to_16_max_fraction"),
            "unexpected error: {err}"
        );

        let config = parse_config(
            r#"
[training.gates]
degeneracy_period_2_to_64_max_fraction = 1.1
"#,
        );
        let err = config
            .validate()
            .expect_err("invalid extended long-cycle period threshold should fail");
        assert!(
            err.to_string()
                .contains("degeneracy_period_2_to_64_max_fraction"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn capability_gates_reject_invalid_thresholds() {
        let config = parse_config(
            r#"
[training.gates]
capability_zero_verifier_patience_epochs = 0
"#,
        );
        let err = config
            .validate()
            .expect_err("zero capability patience should fail");
        assert!(
            err.to_string()
                .contains("capability_zero_verifier_patience_epochs"),
            "unexpected error: {err}"
        );

        let config = parse_config(
            r#"
[training.gates]
capability_regression_patience_epochs = 0
"#,
        );
        let err = config
            .validate()
            .expect_err("zero capability regression patience should fail");
        assert!(
            err.to_string()
                .contains("capability_regression_patience_epochs"),
            "unexpected error: {err}"
        );

        let config = parse_config(
            r#"
[training.gates]
capability_schema_wrong_max_rate = 1.1
"#,
        );
        let err = config
            .validate()
            .expect_err("invalid capability schema threshold should fail");
        assert!(
            err.to_string().contains("capability_schema_wrong_max_rate"),
            "unexpected error: {err}"
        );

        let config = parse_config(
            r#"
[training.gates]
capability_malformed_max_rate = -0.1
"#,
        );
        let err = config
            .validate()
            .expect_err("invalid capability malformed threshold should fail");
        assert!(
            err.to_string().contains("capability_malformed_max_rate"),
            "unexpected error: {err}"
        );

        let config = parse_config(
            r#"
[training.gates]
capability_missing_max_rate = 1.1
"#,
        );
        let err = config
            .validate()
            .expect_err("invalid capability missing threshold should fail");
        assert!(
            err.to_string().contains("capability_missing_max_rate"),
            "unexpected error: {err}"
        );

        let config = parse_config(
            r#"
[training.gates]
capability_completion_health_min_rate = 1.1
"#,
        );
        let err = config
            .validate()
            .expect_err("invalid capability completion-health threshold should fail");
        assert!(
            err.to_string()
                .contains("capability_completion_health_min_rate"),
            "unexpected error: {err}"
        );

        let config = parse_config(
            r#"
[training.gates]
capability_distinct_2_min_fraction = -0.1
"#,
        );
        let err = config
            .validate()
            .expect_err("invalid capability distinct-2 threshold should fail");
        assert!(
            err.to_string()
                .contains("capability_distinct_2_min_fraction"),
            "unexpected error: {err}"
        );

        let config = parse_config(
            r#"
[training.gates]
capability_answer_distinct_min_fraction = -0.1
"#,
        );
        let err = config
            .validate()
            .expect_err("invalid capability answer distinct threshold should fail");
        assert!(
            err.to_string()
                .contains("capability_answer_distinct_min_fraction"),
            "unexpected error: {err}"
        );

        let config = parse_config(
            r#"
[training.gates]
capability_field_value_distinct_ratio_min = 1.1
"#,
        );
        let err = config
            .validate()
            .expect_err("invalid capability field-value distinct threshold should fail");
        assert!(
            err.to_string()
                .contains("capability_field_value_distinct_ratio_min"),
            "unexpected error: {err}"
        );

        let config = parse_config(
            r#"
[training.gates]
capability_field_value_dominance_max = -0.1
"#,
        );
        let err = config
            .validate()
            .expect_err("invalid capability field-value dominance threshold should fail");
        assert!(
            err.to_string()
                .contains("capability_field_value_dominance_max"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn repeat_unlikelihood_rejects_zero_history_lag() {
        let config = parse_config(
            r#"
[training.repeat_unlikelihood]
enabled = true
weight = 0.1
history_lags = [1, 0, 8]
"#,
        );
        let err = config.validate().expect_err("zero history lag should fail");
        assert!(
            err.to_string().contains("repeat_unlikelihood.history_lags"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn repeat_unlikelihood_rejects_invalid_cycle_lag_range() {
        let config = parse_config(
            r#"
[training.repeat_unlikelihood]
enabled = true
cycle_weight = 0.5
cycle_min_lag = 32
cycle_max_lag = 16
"#,
        );
        let err = config
            .validate()
            .expect_err("invalid repeat cycle lag range should fail");
        assert!(
            err.to_string()
                .contains("repeat_unlikelihood.cycle_max_lag"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn repeat_unlikelihood_rejects_zero_cycle_lags_per_step_when_enabled() {
        let config = parse_config(
            r#"
[training.repeat_unlikelihood]
enabled = true
cycle_weight = 0.5
cycle_lags_per_step = 0
"#,
        );
        let err = config
            .validate()
            .expect_err("zero repeat cycle lags per step should fail when cycle loss is enabled");
        assert!(
            err.to_string()
                .contains("repeat_unlikelihood.cycle_lags_per_step"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn repeat_unlikelihood_rejects_zero_every_steps() {
        let config = parse_config(
            r#"
[training.repeat_unlikelihood]
enabled = true
weight = 0.5
every_steps = 0
"#,
        );
        let err = config
            .validate()
            .expect_err("zero repeat cadence should fail");
        assert!(
            err.to_string().contains("repeat_unlikelihood.every_steps"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn logit_entropy_floor_rejects_negative_target() {
        let config = parse_config(
            r#"
[training.logit_entropy_floor]
enabled = true
weight = 0.1
target_entropy_bits = -1.0
"#,
        );
        let err = config
            .validate()
            .expect_err("negative entropy floor target should fail");
        assert!(
            err.to_string()
                .contains("logit_entropy_floor.target_entropy_bits"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn logit_entropy_floor_rejects_invalid_marginal_fields() {
        let config = parse_config(
            r#"
[training.logit_entropy_floor]
enabled = true
marginal_weight = -0.1
"#,
        );
        let err = config
            .validate()
            .expect_err("negative marginal weight should fail");
        assert!(
            err.to_string()
                .contains("logit_entropy_floor.marginal_weight"),
            "unexpected error: {err}"
        );

        let config = parse_config(
            r#"
[training.logit_entropy_floor]
enabled = true
target_marginal_entropy_bits = -1.0
"#,
        );
        let err = config
            .validate()
            .expect_err("negative marginal entropy target should fail");
        assert!(
            err.to_string()
                .contains("logit_entropy_floor.target_marginal_entropy_bits"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn logit_entropy_floor_rejects_zero_every_steps() {
        let config = parse_config(
            r#"
[training.logit_entropy_floor]
enabled = true
weight = 0.1
target_entropy_bits = 2.0
every_steps = 0
"#,
        );
        let err = config
            .validate()
            .expect_err("zero entropy cadence should fail");
        assert!(
            err.to_string().contains("logit_entropy_floor.every_steps"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn logit_entropy_floor_rejects_invalid_target_coverage_fields() {
        let config = parse_config(
            r#"
[training.logit_entropy_floor]
enabled = true
target_coverage_weight = -0.1
"#,
        );
        let err = config
            .validate()
            .expect_err("negative target coverage weight should fail");
        assert!(
            err.to_string()
                .contains("logit_entropy_floor.target_coverage_weight"),
            "unexpected error: {err}"
        );

        let config = parse_config(
            r#"
[training.logit_entropy_floor]
enabled = true
target_coverage_epsilon = 1.0
"#,
        );
        let err = config
            .validate()
            .expect_err("invalid target coverage epsilon should fail");
        assert!(
            err.to_string()
                .contains("logit_entropy_floor.target_coverage_epsilon"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn tied_input_output_embeddings_rejects_factorized_head() {
        let config = parse_config(
            r#"
[model]
tie_input_output_embeddings = true

[model.language_head]
type = "nca_factorized_patch"
state_count = 2
patch_size = 2
"#,
        );
        let err = config
            .validate()
            .expect_err("tied embeddings require flat token head");
        assert!(
            err.to_string()
                .contains("model.tie_input_output_embeddings requires"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn neuron_scaling_config_validates_across_memory_kernels() {
        let cases = [
            r#"
[training.neuron_scaling]
enabled = true
max_latent_total = 64

[model]
n_layer = 1
n_embd = 16
n_head = 2
latent_total = 32
"#,
            r#"
[training.neuron_scaling]
enabled = true
max_latent_total = 64

[model]
n_layer = 1
n_embd = 16
n_head = 2
latent_total = 32
sequence_kernel = { memory_system = "linear_attention", executor = "dense_score_short_context" }
"#,
            r#"
[training.neuron_scaling]
enabled = true
max_latent_total = 64

[model]
n_layer = 1
n_embd = 16
n_head = 2
latent_total = 32
sequence_kernel = "mamba3_state_space_duality"

[model.mamba]
headdim = 8
chunk_size = 4
"#,
            r#"
[training.neuron_scaling]
enabled = true
max_latent_total = 64

[model]
n_layer = 1
n_embd = 16
n_head = 2
latent_total = 32
sequence_kernel = "gated_deltanet2"
"#,
            r#"
[training.neuron_scaling]
enabled = true
max_latent_total = 64

[model]
n_layer = 1
n_embd = 16
n_head = 2
latent_total = 32
sequence_kernel = { memory_system = "gated_deltanet2", executor = "gated_delta_chunk_wy" }

[model.gated_deltanet2]
implementation = "upstream_full"
chunk_size = 4
"#,
        ];

        for case in cases {
            parse_config(case)
                .validate()
                .unwrap_or_else(|err| panic!("neuron scaling config should validate: {err}"));
        }
    }

    #[test]
    fn neuron_scaling_rejects_max_below_current_latent_total() {
        let config = parse_config(
            r#"
[training.neuron_scaling]
enabled = true
max_latent_total = 16

[model]
n_layer = 1
n_embd = 16
n_head = 2
latent_total = 32
"#,
        );

        let err = config
            .validate()
            .expect_err("max below current should fail");
        assert!(
            err.to_string()
                .contains("max_latent_total must be >= resolved model.latent_total"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn neuron_scaling_rejects_max_not_divisible_by_head_count() {
        let config = parse_config(
            r#"
[training.neuron_scaling]
enabled = true
max_latent_total = 64

[model]
n_layer = 1
n_embd = 16
n_head = 3
latent_total = 48
"#,
        );

        let err = config
            .validate()
            .expect_err("head-incompatible max should fail");
        assert!(
            err.to_string()
                .contains("max_latent_total must be divisible by model.n_head"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn neuron_scaling_rejects_non_single_parallel_mode() {
        let config = parse_config(
            r#"
[training.neuron_scaling]
enabled = true
max_latent_total = 80

[model]
n_layer = 1
n_embd = 10
n_head = 2
latent_total = 40

[parallel]
mode = "tensor_parallel_neuron"
world_size = 4

[parallel.data]
size = 1

[parallel.tensor]
size = 4
"#,
        );

        let err = config
            .validate()
            .expect_err("tensor-parallel neuron scaling should fail");
        assert!(
            err.to_string()
                .contains("neuron_scaling.enabled currently requires parallel.mode=single"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn sdft_objective_config_validates() {
        let config = parse_config(
            r#"
[training.objective]
type = "sdft"
max_completion_tokens = 4
teacher_update_rate = 0.25
"#,
        );
        assert!(matches!(
            config.training.objective,
            TrainingObjectiveConfig::Sdft(_)
        ));
        config.validate().expect("sdft objective validates");
    }

    #[test]
    fn sdpo_rejects_invalid_alpha() {
        let config = parse_config(
            r#"
[training.objective]
type = "sdpo"
alpha = 1.25
"#,
        );
        let err = config
            .validate()
            .expect_err("invalid sdpo alpha should fail");
        assert!(
            err.to_string().contains("training.objective.alpha"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn sdft_rejects_unwired_top_entropy_quantile() {
        let config = parse_config(
            r#"
[training.objective]
type = "sdft"
top_entropy_quantile = 0.25
"#,
        );
        let err = config
            .validate()
            .expect_err("unwired SDFT entropy mask should fail");
        assert!(
            err.to_string().contains("top_entropy_quantile"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn sdpo_rejects_unwired_reward_feedback_fields() {
        let config = parse_config(
            r#"
[training.objective]
type = "sdpo"
success_reward_threshold = 1.0
include_environment_feedback = true
"#,
        );
        let err = config
            .validate()
            .expect_err("unwired SDPO reward/feedback fields should fail");
        assert!(
            err.to_string().contains("success_reward_threshold"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn sdpo_rejects_unwired_topk_fields() {
        let config = parse_config(
            r#"
[training.objective]
type = "sdpo"
distillation_topk = 100
"#,
        );
        let err = config
            .validate()
            .expect_err("unwired SDPO top-k distillation should fail");
        assert!(
            err.to_string().contains("distillation_topk"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn sdft_sdpo_composite_objective_config_validates() {
        let config = parse_config(
            r#"
[training.objective]
type = "sdft_sdpo"
sdft_weight = 0.25
sdpo_weight = 0.75

[training.objective.sdft]
max_completion_tokens = 2
generate_from_teacher = true

[training.objective.sdpo]
group_size = 2
max_completion_tokens = 2
alpha = 0.25
"#,
        );
        assert!(matches!(
            config.training.objective,
            TrainingObjectiveConfig::SdftSdpo(_)
        ));
        config
            .validate()
            .expect("composite SDFT/SDPO objective validates");
    }

    #[test]
    fn reservoir_model_initialization_config_validates() {
        let config = parse_config(
            r#"
[model]
n_layer = 1
n_embd = 32
n_head = 4
latent_total = 64

[model.initialization]
kind = "reservoir"

[model.initialization.reservoir]
seed = 1337
density = 0.08
encoder_value_scale = 0.70
decoder_scale = 1.00

[model.initialization.topology_prior]
kind = "modular_bridges"
community_count = 4
bridge_fraction = 0.03
intra_community_gain = 1.5
inter_community_gain = 0.5
bridge_gain = 1.0
"#,
        );
        config
            .validate()
            .expect("reservoir model initialization validates");
    }

    #[test]
    fn legacy_gdpo_flag_is_mutually_exclusive_with_objective_switch() {
        let config = parse_config(
            r#"
[training.gdpo]
enabled = true
"#,
        );
        let err = config
            .validate()
            .expect_err("legacy gdpo objective flag should fail");
        assert!(
            err.to_string().contains("training.gdpo.enabled"),
            "unexpected error: {err}"
        );
    }
}

fn validate_dataset_source(
    source: &DatasetSourceConfig,
    tokenizer_kind: &TokenizerKind,
    _allow_validation_only_hf: bool,
    label: &str,
) -> Result<()> {
    match source {
        DatasetSourceConfig::NemotronClimbMix { max_records, .. } => {
            if matches!(max_records, Some(0)) {
                return Err(anyhow!("{label}.max_records must be > 0 when set"));
            }
            if !matches!(tokenizer_kind, TokenizerKind::Pretokenized(_)) {
                return Err(anyhow!(
                    "{label}.tokenizer.type must be `pretokenized` for climbmix datasets"
                ));
            }
        }
        DatasetSourceConfig::UniversalityManifest { manifest } => {
            if manifest.as_os_str().is_empty() {
                return Err(anyhow!("{label}.manifest must not be empty"));
            }
            if !matches!(tokenizer_kind, TokenizerKind::Pretokenized(_)) {
                return Err(anyhow!(
                    "{label}.tokenizer.type must be `pretokenized` for universality manifests"
                ));
            }
        }
        DatasetSourceConfig::UniversalityNca { config } => {
            if config.as_os_str().is_empty() {
                return Err(anyhow!("{label}.config must not be empty"));
            }
            if !matches!(tokenizer_kind, TokenizerKind::Pretokenized(_)) {
                return Err(anyhow!(
                    "{label}.tokenizer.type must be `pretokenized` for on-the-fly universality NCA datasets"
                ));
            }
        }
        DatasetSourceConfig::UniversalityRuliad { config } => {
            if config.as_os_str().is_empty() {
                return Err(anyhow!("{label}.config must not be empty"));
            }
            if !matches!(tokenizer_kind, TokenizerKind::Pretokenized(_)) {
                return Err(anyhow!(
                    "{label}.tokenizer.type must be `pretokenized` for on-the-fly universality ruliad datasets"
                ));
            }
        }
    }
    Ok(())
}
