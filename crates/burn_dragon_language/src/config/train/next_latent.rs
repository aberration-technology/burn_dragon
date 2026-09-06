//! Token-conditioned latent prediction, independent of inference-time refinement.

use super::LatentReasoningAuxiliaryStartPolicy;
use serde::{Deserialize, Serialize};

pub(crate) const NEXT_LATENT_OBJECTIVE_CONTRACT_VERSION: u32 = 2;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct NextLatentPredictionConfig {
    pub enabled: bool,
    pub every_steps: Option<usize>,
    pub start_after_steps: Option<usize>,
    pub start_policy: Option<LatentReasoningAuxiliaryStartPolicy>,
    pub horizon: usize,
    /// Coefficient after an independent valid-token and horizon mean.
    pub regression_weight: f32,
    /// Does not change the regression coefficient or train decoder parameters.
    pub token_kl_weight: f32,
    pub smooth_l1_beta: f32,
    pub detach_action_embedding: bool,
}

impl Default for NextLatentPredictionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            every_steps: None,
            start_after_steps: None,
            start_policy: None,
            horizon: 1,
            regression_weight: 1.0,
            token_kl_weight: 0.0,
            smooth_l1_beta: 1.0,
            detach_action_embedding: true,
        }
    }
}

impl NextLatentPredictionConfig {
    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        if !self.enabled {
            return Ok(());
        }
        anyhow::ensure!(
            self.every_steps != Some(0),
            "training.latent_reasoning.next_latent.every_steps must be > 0 when set"
        );
        anyhow::ensure!(
            self.horizon > 0,
            "training.latent_reasoning.next_latent.horizon must be > 0 when enabled"
        );
        for (name, weight) in [
            ("regression_weight", self.regression_weight),
            ("token_kl_weight", self.token_kl_weight),
        ] {
            anyhow::ensure!(
                weight.is_finite() && weight >= 0.0,
                "training.latent_reasoning.next_latent.{name} must be finite and >= 0"
            );
        }
        anyhow::ensure!(
            self.regression_weight > f32::EPSILON || self.token_kl_weight > f32::EPSILON,
            "training.latent_reasoning.next_latent requires at least one positive loss weight"
        );
        anyhow::ensure!(
            self.smooth_l1_beta.is_finite() && self.smooth_l1_beta > 0.0,
            "training.latent_reasoning.next_latent.smooth_l1_beta must be finite and > 0"
        );
        Ok(())
    }
}
