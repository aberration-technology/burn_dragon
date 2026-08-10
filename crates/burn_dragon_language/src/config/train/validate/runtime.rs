//! Core training, monitoring, dynamics, and latent-runtime contracts.

use super::*;

impl TrainingConfig {
    pub(super) fn validate_runtime_contracts(&self) -> Result<()> {
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
        if self.training.tbptt_credit_window_chunks == 0 {
            return Err(anyhow!("training.tbptt_credit_window_chunks must be > 0"));
        }
        if self.training.tbptt_credit_window_chunks > 1 && self.training.tbptt_chunk_size.is_none()
        {
            return Err(anyhow!(
                "training.tbptt_credit_window_chunks > 1 requires training.tbptt_chunk_size"
            ));
        }
        if self.training.tbptt_credit_window_chunks > 1 && self.parallel.pipeline.enabled {
            return Err(anyhow!(
                "training.tbptt_credit_window_chunks > 1 does not yet support parallel.pipeline.enabled"
            ));
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
        match self.training.validation.objective {
            crate::config::TrainingValidationObjective::FixedHoldout => {
                if self
                    .training
                    .validation
                    .sampling
                    .uses_live_source_selection()
                {
                    return Err(anyhow!(
                        "training.validation.objective=fixed_holdout requires training.validation.sampling=fixed_holdout"
                    ));
                }
            }
            crate::config::TrainingValidationObjective::SourceWeighted => {
                if self.training.events.source_weighted_validation_batches == 0 {
                    return Err(anyhow!(
                        "training.validation.objective=source_weighted requires training.events.source_weighted_validation_batches > 0"
                    ));
                }
                if !matches!(
                    self.dataset.source,
                    crate::config::DatasetSourceConfig::UniversalityRuliad { .. }
                ) {
                    return Err(anyhow!(
                        "training.validation.objective=source_weighted requires dataset.type=universality_ruliad"
                    ));
                }
            }
            crate::config::TrainingValidationObjective::StreamWarm => {
                if !self.training.tbptt_persist_across_steps
                    && !self.training.sequence_state_probe.enabled
                {
                    return Err(anyhow!(
                        "training.validation.objective=stream_warm requires training.tbptt_persist_across_steps=true or training.sequence_state_probe.enabled=true"
                    ));
                }
                if self.resolved_training_algorithm() == TrainingAlgorithm::Eggroll {
                    return Err(anyhow!(
                        "training.validation.objective=stream_warm is not available for forward-only eggroll validation"
                    ));
                }
            }
        }
        match (
            self.training.validation.ruliad_panel.mode,
            self.training.validation.ruliad_panel.path.as_ref(),
        ) {
            (crate::config::RuliadValidationPanelMode::Dynamic, None)
            | (
                crate::config::RuliadValidationPanelMode::CreateOrReuse
                | crate::config::RuliadValidationPanelMode::RequireExisting,
                Some(_),
            ) => {}
            (crate::config::RuliadValidationPanelMode::Dynamic, Some(_)) => {
                return Err(anyhow!(
                    "training.validation.ruliad_panel.path requires mode=create_or_reuse or require_existing"
                ));
            }
            (
                crate::config::RuliadValidationPanelMode::CreateOrReuse
                | crate::config::RuliadValidationPanelMode::RequireExisting,
                None,
            ) => {
                return Err(anyhow!(
                    "training.validation.ruliad_panel.path is required for a persisted panel"
                ));
            }
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
            if self
                .training
                .ruliad_policy_probe
                .scoring
                .uses_sequence_score_head()
                && !self
                    .model
                    .sequence_score_head
                    .is_some_and(|head| head.enabled)
            {
                return Err(anyhow!(
                    "semantic/residual-energy Ruliad policy probing requires model.sequence_score_head.enabled=true"
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
                if !gate.regression_confidence_z.is_finite() || gate.regression_confidence_z <= 0.0
                {
                    return Err(anyhow!(
                        "training.ruliad_policy_probe.promotion_gate.regression_confidence_z must be finite and > 0"
                    ));
                }
            }
        }
        if self
            .training
            .ruliad_policy_probe
            .checkpoint_capability_contract
            .requires_closed_loop_policy()
        {
            if !self.training.ruliad_policy_probe.enabled {
                return Err(anyhow!(
                    "training.ruliad_policy_probe.checkpoint_capability_contract requires training.ruliad_policy_probe.enabled=true for closed-loop policy capability"
                ));
            }
            if !self.training.ruliad_policy_probe.promotion_gate.enabled {
                return Err(anyhow!(
                    "training.ruliad_policy_probe.checkpoint_capability_contract requires training.ruliad_policy_probe.promotion_gate.enabled=true for closed-loop policy capability"
                ));
            }
            let validation_every = self
                .training
                .events
                .ruliad_correctness_probe_every_epochs
                .max(1);
            let closed_loop_every = self
                .training
                .ruliad_policy_probe
                .effective_closed_loop_every_epochs();
            if !validation_every.is_multiple_of(closed_loop_every) {
                return Err(anyhow!(
                    "training.ruliad_policy_probe closed-loop cadence must divide training.events.ruliad_correctness_probe_every_epochs when checkpoint capability requires policy results"
                ));
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
        Ok(())
    }
}
