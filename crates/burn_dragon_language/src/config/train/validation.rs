//! Validation work budgets, independent of hot-path metric synchronization.

use super::TrainingValidationConfig;
use burn_dragon_train::train::pipeline::resolve_valid_steps_per_epoch;

impl TrainingValidationConfig {
    pub(crate) fn resolve_steps_per_epoch(
        &self,
        train_steps: usize,
        log_frequency: usize,
        available_batches: usize,
    ) -> usize {
        match self.batches {
            Some(batches) => {
                assert!(batches > 0, "validation batch budget must be positive");
                batches.min(available_batches.max(1))
            }
            None => resolve_valid_steps_per_epoch(train_steps, log_frequency, available_batches),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_validation_budget_is_independent_of_logging_and_checkpoint_cadence() {
        let config = TrainingValidationConfig {
            batches: Some(128),
            ..Default::default()
        };
        for steps in [64, 512, 8192] {
            for logging in [1, 32, 256] {
                assert_eq!(config.resolve_steps_per_epoch(steps, logging, 256), 128);
            }
        }
        assert_eq!(config.resolve_steps_per_epoch(512, 32, 64), 64);
        assert_eq!(config.resolve_steps_per_epoch(512, 32, 0), 1);
    }

    #[test]
    fn omitted_validation_budget_preserves_behavior_and_serialized_contract() {
        let config = TrainingValidationConfig::default();
        assert_eq!(config.resolve_steps_per_epoch(512, 32, 256), 16);
        assert_eq!(config.resolve_steps_per_epoch(512, 32, 8), 8);
        let serialized = serde_json::to_value(&config).expect("validation config");
        assert!(serialized.get("batches").is_none());
        let decoded: TrainingValidationConfig =
            serde_json::from_value(serialized).expect("historical config");
        assert_eq!(decoded, config);
    }

    #[test]
    fn explicit_validation_budget_round_trips() {
        let config: TrainingValidationConfig =
            toml::from_str("batches = 128").expect("explicit validation budget");
        let json = serde_json::to_value(&config).expect("validation config");
        assert_eq!(json["batches"], 128);
        assert_eq!(
            serde_json::from_value::<TrainingValidationConfig>(json).unwrap(),
            config
        );
    }
}
