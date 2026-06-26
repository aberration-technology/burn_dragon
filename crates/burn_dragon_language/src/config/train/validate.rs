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
    PredictiveCodingParameterUpdate, RuliadVerifierRewardMode, TrainingConfig,
};
use crate::tokenizer::TokenizerKind;

impl TrainingConfig {
    pub fn validate(&self) -> Result<()> {
        if self.training.block_size == 0 {
            return Err(anyhow!("training.block_size must be > 0"));
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
        if self.training.batch_size == 0 {
            return Err(anyhow!("training.batch_size must be > 0"));
        }
        if self.training.gradient_accumulation_steps == 0 {
            return Err(anyhow!("training.gradient_accumulation_steps must be > 0"));
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
            if self
                .training
                .repeat_unlikelihood
                .history_lags
                .iter()
                .any(|lag| *lag == 0)
            {
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
            if pc.steps == 0 {
                return Err(anyhow!("training.predictive_coding.steps must be > 0"));
            }
            if pc.step_size <= 0.0 || !pc.step_size.is_finite() {
                return Err(anyhow!(
                    "training.predictive_coding.step_size must be finite and > 0"
                ));
            }
            if pc.latent_decay < 0.0 || !pc.latent_decay.is_finite() {
                return Err(anyhow!(
                    "training.predictive_coding.latent_decay must be finite and >= 0"
                ));
            }
            if let Some(max_grad_norm) = pc.max_grad_norm
                && (max_grad_norm <= 0.0 || !max_grad_norm.is_finite())
            {
                return Err(anyhow!(
                    "training.predictive_coding.max_grad_norm must be finite and > 0"
                ));
            }
            if pc.eps <= 0.0 || !pc.eps.is_finite() {
                return Err(anyhow!(
                    "training.predictive_coding.eps must be finite and > 0"
                ));
            }
            if pc.apply_every_chunks == 0 {
                return Err(anyhow!(
                    "training.predictive_coding.apply_every_chunks must be > 0"
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
        if latent.eval_step_sweep.iter().any(|steps| *steps == 0) {
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
            if latent.jepa_future_offsets.iter().any(|offset| *offset == 0) {
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
        if self.training.ruliad_supervision.verifier_reward.enabled {
            let verifier_reward = self.training.ruliad_supervision.verifier_reward;
            if !verifier_reward.weight.is_finite() || verifier_reward.weight <= 0.0 {
                return Err(anyhow!(
                    "training.ruliad_supervision.verifier_reward.weight must be finite and positive"
                ));
            }
            if verifier_reward.group_size < 2 {
                return Err(anyhow!(
                    "training.ruliad_supervision.verifier_reward.group_size must be at least 2"
                ));
            }
            if verifier_reward.max_completion_tokens == 0 {
                return Err(anyhow!(
                    "training.ruliad_supervision.verifier_reward.max_completion_tokens must be positive"
                ));
            }
            if verifier_reward.every_steps == 0 {
                return Err(anyhow!(
                    "training.ruliad_supervision.verifier_reward.every_steps must be positive"
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
                if verifier_reward.vpo_correctness_mass_floor
                    + verifier_reward.vpo_completion_health_mass_floor
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
            if self.training.tbptt_chunk_size.is_some() {
                return Err(anyhow!(
                    "training.ruliad_supervision.verifier_reward.enabled does not yet support training.tbptt_chunk_size"
                ));
            }
            if self.training.tbptt_persist_across_steps {
                return Err(anyhow!(
                    "training.ruliad_supervision.verifier_reward.enabled does not yet support training.tbptt_persist_across_steps"
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
            let max_latent_total = self.training.neuron_scaling.max_latent_total;
            if max_latent_total < resolved_model.latent_total() {
                return Err(anyhow!(
                    "training.neuron_scaling.max_latent_total must be >= resolved model.latent_total (got max={} current={})",
                    max_latent_total,
                    resolved_model.latent_total()
                ));
            }
            if max_latent_total % resolved_model.n_embd != 0 {
                return Err(anyhow!(
                    "training.neuron_scaling.max_latent_total must be divisible by model.n_embd (got max={} n_embd={})",
                    max_latent_total,
                    resolved_model.n_embd
                ));
            }
            if max_latent_total % resolved_model.n_head != 0 {
                return Err(anyhow!(
                    "training.neuron_scaling.max_latent_total must be divisible by model.n_head (got max={} n_head={})",
                    max_latent_total,
                    resolved_model.n_head
                ));
            }
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
    fn ruliad_1m_la16k_verifier_proxy_profiles_validate() {
        for (profile, ranking, denoising) in [
            (
                "ruliad-1m-la-16k.answer-completion.self-recovery.training.toml",
                false,
                false,
            ),
            (
                "ruliad-1m-la-16k.answer-completion-ranking.self-recovery.training.toml",
                true,
                false,
            ),
            (
                "ruliad-1m-la-16k.answer-completion-denoising.self-recovery.training.toml",
                false,
                true,
            ),
            (
                "ruliad-1m-la-16k.answer-completion-ranking-denoising.self-recovery.training.toml",
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
            config.training.ruliad_supervision.verifier_reward.mode,
            RuliadVerifierRewardMode::Scalar
        );
        assert!(config.training.tbptt_chunk_size.is_none());
        assert!(!config.training.tbptt_persist_across_steps);
        assert!(config.training.objective.is_next_token());
    }

    #[test]
    fn ruliad_1m_la16k_verifier_vpo_profile_validates() {
        let config = load_profile("ruliad-1m-la-16k.verifier-vpo.training.toml");
        config
            .validate()
            .expect("ruliad verifier VPO profile should validate");
        assert!(config.training.ruliad_supervision.verifier_reward.enabled);
        assert_eq!(
            config.training.ruliad_supervision.verifier_reward.mode,
            RuliadVerifierRewardMode::VpoIndependent
        );
        assert!(
            config
                .training
                .ruliad_supervision
                .verifier_reward
                .vpo_scalarizations
                > 0
        );
        assert!(
            config
                .training
                .ruliad_supervision
                .verifier_reward
                .vpo_correctness_mass_floor
                >= 0.70
        );
        assert!(
            config
                .training
                .ruliad_supervision
                .verifier_reward
                .vpo_compactness_max_weight
                <= 0.05
        );
        assert!(config.training.tbptt_chunk_size.is_none());
        assert!(!config.training.tbptt_persist_across_steps);
        assert!(config.training.objective.is_next_token());
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
        load_training_config(&[profile_path.clone()])
            .unwrap_or_else(|err| panic!("load {}: {err}", profile_path.display()))
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
        config.training.tbptt_chunk_size = Some(128);
        let err = config
            .validate()
            .expect_err("verifier reward should reject TBPTT chunking");
        assert!(
            err.to_string().contains("tbptt_chunk_size"),
            "unexpected error: {err}"
        );
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
