//! Coverage proof for self-contained objectives that replace the streamed language update.

use super::{LocalPredictiveCodingTerminalCriterion, TrainingAlgorithm, TrainingHyperparameters};

impl TrainingHyperparameters {
    pub(crate) fn has_required_self_contained_primary_schedule(&self) -> bool {
        let training = self;
        let pc = &training.local_predictive_coding;
        if !training.objective.is_next_token()
            || !matches!(
                self.algorithm,
                TrainingAlgorithm::Backpropagation | TrainingAlgorithm::PredictiveCoding
            )
            || pc.terminal_criterion != LocalPredictiveCodingTerminalCriterion::RuliadVerifierSet
            || pc.learning_schedule != burn_pc::PcLearningSchedule::Equilibrium
        {
            return false;
        }
        let policy = training.ruliad_supervision.proof_policy;
        let binding = training.ruliad_supervision.prompt_value_binding;
        if !policy.enabled
            || !policy.require_scheduled_update
            || policy.decoder_calibration_steps > 0
        {
            return false;
        }
        if policy.every_steps == 1 && policy.start_after_steps == 0 {
            return true;
        }
        if !binding.require_scheduled_update {
            return false;
        }
        if binding.every_steps == 1 && binding.active_at_step(0) {
            return true;
        }
        // Two single-residue periodic schedules with periods >= 2 cover all steps
        // only when both periods are 2 and their residues are complementary.
        policy.every_steps == 2
            && policy.start_after_steps == 0
            && binding.every_steps == 2
            && binding.active_at_step(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TrainingConfig;
    use std::path::Path;

    fn pilot() -> TrainingConfig {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut config = crate::load_training_config(&[root
            .join("config/language/experiments/predictive_coding/evaluation-contract-pilot.toml")])
        .unwrap();
        if let crate::config::DatasetSourceConfig::UniversalityRuliad { config: path } =
            &mut config.dataset.source
        {
            *path = root.join(&*path);
        }
        config
    }

    #[test]
    fn required_binding_is_opt_in_and_requires_enabled_objective() {
        let old = serde_json::json!({"enabled": false});
        let binding: crate::config::RuliadPromptValueBindingConfig =
            serde_json::from_value(old).unwrap();
        assert!(!binding.require_scheduled_update);
        assert!(
            serde_json::to_value(binding)
                .unwrap()
                .get("require_scheduled_update")
                .is_none()
        );
        let mut config = pilot();
        config
            .training
            .ruliad_supervision
            .prompt_value_binding
            .enabled = false;
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("requires prompt_value_binding.enabled")
        );
    }

    #[test]
    fn required_structured_schedule_permits_stateless_streaming_for_both_algorithms() {
        for algorithm in [
            TrainingAlgorithm::Backpropagation,
            TrainingAlgorithm::PredictiveCoding,
        ] {
            let mut config = pilot();
            config.training.algorithm = algorithm;
            config.training.tbptt_persist_across_steps = false;
            assert!(
                config
                    .training
                    .has_required_self_contained_primary_schedule()
            );
            config.validate().unwrap();
            crate::dataset::build_dataset(&config.dataset, &config.training)
                .expect("the real dataset startup audit must accept the same primary contract");
        }
    }

    #[test]
    fn stateless_dataset_audit_still_rejects_optional_primary_schedules() {
        let mut config = pilot();
        config.training.tbptt_persist_across_steps = false;
        config
            .training
            .ruliad_supervision
            .prompt_value_binding
            .require_scheduled_update = false;
        let error = crate::dataset::build_dataset(&config.dataset, &config.training)
            .err()
            .expect("document context remains required for optional primary schedules");
        assert!(error.to_string().contains("not causally visible"));
    }

    #[test]
    fn required_structured_schedule_has_no_periodic_or_startup_gaps() {
        let base = pilot();
        for policy_period in 1..=8 {
            for binding_period in 1..=8 {
                for phase in 0..binding_period {
                    let mut config = base.clone();
                    let supervision = &mut config.training.ruliad_supervision;
                    supervision.proof_policy.every_steps = policy_period;
                    supervision.prompt_value_binding.every_steps = binding_period;
                    supervision.prompt_value_binding.phase_steps = phase;
                    let covered = (0..=256).all(|step| {
                        step % policy_period == 0
                            || supervision.prompt_value_binding.active_at_step(step)
                    });
                    assert_eq!(
                        config
                            .training
                            .has_required_self_contained_primary_schedule(),
                        covered
                    );
                }
            }
        }
        for delay in 1..=3 {
            let mut config = pilot();
            config
                .training
                .ruliad_supervision
                .proof_policy
                .start_after_steps = delay;
            assert!(
                !config
                    .training
                    .has_required_self_contained_primary_schedule()
            );
        }
        let mut config = pilot();
        config
            .training
            .ruliad_supervision
            .prompt_value_binding
            .start_after_steps = 2;
        assert!(
            !config
                .training
                .has_required_self_contained_primary_schedule()
        );
    }

    #[test]
    fn required_structured_schedule_rejects_optional_or_joint_programs() {
        let mut variants = Vec::new();
        let mut config = pilot();
        config
            .training
            .ruliad_supervision
            .proof_policy
            .require_scheduled_update = false;
        variants.push(config);
        let mut config = pilot();
        config
            .training
            .ruliad_supervision
            .prompt_value_binding
            .require_scheduled_update = false;
        variants.push(config);
        let mut config = pilot();
        config
            .training
            .ruliad_supervision
            .proof_policy
            .decoder_calibration_steps = 10;
        variants.push(config);
        let mut config = pilot();
        config.training.local_predictive_coding.terminal_criterion =
            LocalPredictiveCodingTerminalCriterion::RuliadVerifierSetJoint;
        variants.push(config);
        let mut config = pilot();
        config.training.local_predictive_coding.learning_schedule =
            burn_pc::PcLearningSchedule::Incremental;
        variants.push(config);
        for mut config in variants {
            config.training.tbptt_persist_across_steps = false;
            assert!(
                !config
                    .training
                    .has_required_self_contained_primary_schedule()
            );
            assert!(
                config
                    .validate()
                    .unwrap_err()
                    .to_string()
                    .contains("zero-gradient")
            );
        }
    }
}
