use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RuliadSampleTelemetry {
    pub oracle_hash: String,
    pub family: String,
    pub task_kind: String,
    pub loss: f32,
    #[serde(default)]
    pub previous_loss: Option<f32>,
    #[serde(default)]
    pub gradient_alignment: Option<f32>,
    #[serde(default = "default_cost")]
    pub verification_cost: f32,
    #[serde(default)]
    pub accepted: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RuliadMetricSnapshot {
    pub sample_count: usize,
    pub verifier_failures: usize,
    pub sampler_entropy_bits: f32,
    pub hash_noise_probability: f32,
    pub mean_loss: f32,
    pub mean_learning_progress: f32,
}

fn default_cost() -> f32 {
    1.0
}
