//! Ruliad supervision, policy, verifier, and dataset-binding contracts.

use super::*;

impl TrainingConfig {
    pub(super) fn validate_ruliad_contracts(&self) -> Result<()> {
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
        let prompt_value_binding = self.training.ruliad_supervision.prompt_value_binding;
        if prompt_value_binding.enabled {
            if prompt_value_binding.every_steps == 0 {
                return Err(anyhow!(
                    "training.ruliad_supervision.prompt_value_binding.every_steps must be positive when enabled"
                ));
            }
            if prompt_value_binding.phase_steps >= prompt_value_binding.every_steps {
                return Err(anyhow!(
                    "training.ruliad_supervision.prompt_value_binding.phase_steps must be less than every_steps"
                ));
            }
            if prompt_value_binding.max_completion_tokens == 0 {
                return Err(anyhow!(
                    "training.ruliad_supervision.prompt_value_binding.max_completion_tokens must be positive when enabled"
                ));
            }
            if prompt_value_binding.max_rows_per_step == 0 {
                return Err(anyhow!(
                    "training.ruliad_supervision.prompt_value_binding.max_rows_per_step must be positive when enabled"
                ));
            }
            if !self.training.ruliad_supervision.uses_answer_target_mask() {
                return Err(anyhow!(
                    "training.ruliad_supervision.prompt_value_binding.enabled requires an answer-target supervision mode"
                ));
            }
            if self.parallel.pipeline.enabled {
                return Err(anyhow!(
                    "training.ruliad_supervision.prompt_value_binding.enabled does not yet support parallel.pipeline.enabled"
                ));
            }
            if self.training.ruliad_supervision.answer_contract.enabled
                && self
                    .training
                    .ruliad_supervision
                    .answer_contract
                    .prompt_schema_value_weight
                    > f32::EPSILON
            {
                return Err(anyhow!(
                    "prompt_value_binding and answer_contract.prompt_schema_value_weight are mutually exclusive primary and auxiliary value-binding contracts"
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
        let proof_policy = self.training.ruliad_supervision.proof_policy;
        let semantic_refresh = self
            .training
            .ruliad_supervision
            .proof_policy_semantic_refresh;
        if semantic_refresh.enabled {
            if !proof_policy.enabled
                || proof_policy.scoring
                    != crate::config::RuliadProofPolicyScoring::CompletionLikelihood
                || proof_policy.normalization
                    != crate::config::RuliadProofPolicyNormalization::PrefixConditional
                || proof_policy.gradient_scope
                    != crate::config::RuliadProofPolicyGradientScope::FullModel
            {
                return Err(anyhow!(
                    "proof_policy_semantic_refresh requires a full-model prefix-conditional completion-likelihood primary proof policy"
                ));
            }
            if semantic_refresh.every_steps == 0
                || proof_policy.every_steps == 0
                || !semantic_refresh
                    .every_steps
                    .is_multiple_of(proof_policy.every_steps)
            {
                return Err(anyhow!(
                    "proof_policy_semantic_refresh.every_steps must be a positive multiple of proof_policy.every_steps"
                ));
            }
            if semantic_refresh.counterfactual_targets_per_state == 0
                || semantic_refresh.counterfactual_targets_per_state >= proof_policy.candidates
            {
                return Err(anyhow!(
                    "proof_policy_semantic_refresh.counterfactual_targets_per_state must be positive and less than proof_policy.candidates"
                ));
            }
            if !self
                .model
                .sequence_score_head
                .is_some_and(|head| head.enabled)
            {
                return Err(anyhow!(
                    "proof_policy_semantic_refresh requires model.sequence_score_head.enabled=true"
                ));
            }
        }
        if proof_policy.enabled {
            if proof_policy.scoring.uses_sequence_score_head() {
                if !self
                    .model
                    .sequence_score_head
                    .is_some_and(|head| head.enabled)
                {
                    return Err(anyhow!(
                        "semantic/residual-energy proof-policy scoring requires model.sequence_score_head.enabled=true"
                    ));
                }
                if proof_policy.normalization
                    != crate::config::RuliadProofPolicyNormalization::CandidateConditional
                {
                    return Err(anyhow!(
                        "semantic/residual-energy proof-policy scoring requires normalization=candidate_conditional"
                    ));
                }
            }
            match proof_policy.gradient_scope {
                crate::config::RuliadProofPolicyGradientScope::ScoreHeadOnly
                    if !proof_policy.scoring.uses_sequence_score_head() =>
                {
                    return Err(anyhow!(
                        "training.ruliad_supervision.proof_policy.gradient_scope=score_head_only requires scoring=semantic_energy or residual_energy"
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
                let semantic_energy = proof_policy.scoring.uses_sequence_score_head();
                let isolated_completion = proof_policy.scoring
                    == crate::config::RuliadProofPolicyScoring::CompletionLikelihood
                    && proof_policy.gradient_scope
                        == crate::config::RuliadProofPolicyGradientScope::LanguageHeadOnly
                    && proof_policy.normalization
                        == crate::config::RuliadProofPolicyNormalization::CandidateConditional;
                let deployed_decoder_completion = proof_policy.scoring
                    == crate::config::RuliadProofPolicyScoring::CompletionLikelihood
                    && proof_policy.gradient_scope
                        == crate::config::RuliadProofPolicyGradientScope::FullModel
                    && proof_policy.normalization
                        == crate::config::RuliadProofPolicyNormalization::PrefixConditional;
                if !semantic_energy && !isolated_completion && !deployed_decoder_completion {
                    return Err(anyhow!(
                        "training.ruliad_supervision.proof_policy.counterfactual_targets_per_state requires semantic/residual energy, isolated candidate-conditional language-head completion, or full-model prefix-conditional deployed-decoder completion"
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
        if self
            .dataset
            .ruliad_source_selection_feedback_updates_enabled
            .is_some()
            && !matches!(
                self.dataset.source,
                DatasetSourceConfig::UniversalityRuliad { .. }
            )
        {
            return Err(anyhow!(
                "dataset.ruliad_source_selection_feedback_updates_enabled requires dataset.type=\"universality_ruliad\""
            ));
        }
        if self
            .dataset
            .ruliad_source_selection_cold_start_enabled
            .is_some()
            && !matches!(
                self.dataset.source,
                DatasetSourceConfig::UniversalityRuliad { .. }
            )
        {
            return Err(anyhow!(
                "dataset.ruliad_source_selection_cold_start_enabled requires dataset.type=\"universality_ruliad\""
            ));
        }
        Ok(())
    }
}
