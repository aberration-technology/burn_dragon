//! Run identity controls; never consulted on the training hot path.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct TrainingProvenanceConfig {
    /// Hash realized floating-point parameters once at fresh/transfer startup.
    /// Confirmation experiments require this, not just equality of RNG seeds.
    pub initial_model_fingerprint: bool,
}

impl Default for TrainingProvenanceConfig {
    fn default() -> Self {
        Self {
            initial_model_fingerprint: true,
        }
    }
}

impl TrainingProvenanceConfig {
    pub(crate) fn is_default(&self) -> bool {
        self == &Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provenance_defaults_on_and_rejects_unknown_settings() {
        assert!(
            serde_json::from_str::<TrainingProvenanceConfig>("{}")
                .unwrap()
                .initial_model_fingerprint
        );
        assert!(
            !serde_json::from_str::<TrainingProvenanceConfig>(
                r#"{"initial_model_fingerprint":false}"#
            )
            .unwrap()
            .initial_model_fingerprint
        );
        assert!(
            serde_json::from_str::<TrainingProvenanceConfig>(
                r#"{"initial_model_fingerpint":true}"#
            )
            .is_err()
        );
    }
}
