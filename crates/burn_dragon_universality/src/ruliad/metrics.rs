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
pub struct RuliadCapabilityFeedback {
    pub group_label: String,
    pub item_count: usize,
    pub verifier_rate: f32,
    pub partial_credit_rate: f32,
    pub schema_valid_wrong_rate: f32,
    pub malformed_rate: f32,
    pub missing_rate: f32,
    pub completion_health_rate: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RuliadMetricSnapshot {
    pub sample_count: usize,
    pub verifier_failures: usize,
    pub sampler_entropy_bits: f32,
    #[serde(default)]
    pub active_candidate_count: usize,
    #[serde(default)]
    pub active_max_entropy_bits: f32,
    #[serde(default)]
    pub normalized_sampler_entropy: f32,
    pub hash_noise_probability: f32,
    pub mean_loss: f32,
    pub mean_learning_progress: f32,
    #[serde(default)]
    pub frontier_loss: f32,
    #[serde(default)]
    pub target_loss: f32,
    #[serde(default)]
    pub target_difficulty_score: f32,
    #[serde(default)]
    pub max_difficulty_level: usize,
    #[serde(default)]
    pub mean_difficulty_level: f32,
    #[serde(default)]
    pub normalized_difficulty_score: f32,
    #[serde(default)]
    pub max_difficulty_probability: f32,
    #[serde(default)]
    pub mastered_probability: f32,
    #[serde(default)]
    pub capability_feedback_probability: f32,
    #[serde(default)]
    pub capability_verifier_ema: f32,
    #[serde(default)]
    pub capability_completion_health_ema: f32,
    #[serde(default)]
    pub capability_schema_wrong_ema: f32,
    #[serde(default)]
    pub capability_malformed_ema: f32,
    #[serde(default)]
    pub capability_missing_ema: f32,
    #[serde(default)]
    pub capability_lagging_probability: f32,
    #[serde(default)]
    pub frontier_extension_count: usize,
    #[serde(default)]
    pub frontier_saturated: bool,
    #[serde(default)]
    pub frontier_unbounded: bool,
    #[serde(default)]
    pub top_buckets: Vec<RuliadBucketMetric>,
    #[serde(default)]
    pub difficulty_buckets: Vec<RuliadGroupMetric>,
    #[serde(default)]
    pub family_buckets: Vec<RuliadGroupMetric>,
    #[serde(default)]
    pub task_buckets: Vec<RuliadGroupMetric>,
    #[serde(default)]
    pub contract_buckets: Vec<RuliadGroupMetric>,
}

fn default_cost() -> f32 {
    1.0
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RuliadBucketMetric {
    pub label: String,
    pub family: String,
    pub task_kind: String,
    pub difficulty_level: usize,
    pub probability: f32,
    pub loss_ema: f32,
    pub previous_loss_ema: f32,
    pub learning_progress: f32,
    pub mastered: bool,
    #[serde(default)]
    pub capability_feedback_count: usize,
    #[serde(default)]
    pub capability_verifier_ema: f32,
    #[serde(default)]
    pub capability_completion_health_ema: f32,
    #[serde(default)]
    pub capability_schema_wrong_ema: f32,
    #[serde(default)]
    pub capability_malformed_ema: f32,
    #[serde(default)]
    pub capability_missing_ema: f32,
    #[serde(default)]
    pub capability_lagging: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RuliadGroupMetric {
    pub label: String,
    pub candidate_count: usize,
    pub probability: f32,
    pub mean_loss: f32,
    pub learning_progress: f32,
    pub mastered_probability: f32,
    pub mean_difficulty_level: f32,
    #[serde(default)]
    pub capability_feedback_probability: f32,
    #[serde(default)]
    pub capability_verifier_ema: f32,
    #[serde(default)]
    pub capability_completion_health_ema: f32,
    #[serde(default)]
    pub capability_schema_wrong_ema: f32,
    #[serde(default)]
    pub capability_malformed_ema: f32,
    #[serde(default)]
    pub capability_missing_ema: f32,
    #[serde(default)]
    pub capability_lagging_probability: f32,
}
