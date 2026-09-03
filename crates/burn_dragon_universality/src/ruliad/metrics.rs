use serde::{Deserialize, Serialize};

pub const RULIAD_SOURCE_CAPABILITY_LABEL_PREFIX: &str = "source:";

/// Stable capability-feedback key for one semantic source at one difficulty.
pub fn ruliad_source_capability_label(
    family: &str,
    task_kind: &str,
    difficulty_level: usize,
    answer_contract: &str,
) -> String {
    format!(
        "{RULIAD_SOURCE_CAPABILITY_LABEL_PREFIX}{family}:{task_kind}@d{difficulty_level}#{answer_contract}"
    )
}

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
    /// Highest difficulty with non-negligible probability in this effective
    /// policy snapshot. This can be below the materialized frontier during a
    /// cold start, hold, or capability gate.
    #[serde(default)]
    pub active_max_difficulty_level: usize,
    /// Highest difficulty permanently released by the mastery-gated
    /// curriculum. Sampling can move below this edge without revoking it.
    #[serde(default)]
    pub curriculum_released_max_difficulty_level: usize,
    #[serde(default)]
    pub mean_difficulty_level: f32,
    #[serde(default)]
    pub normalized_difficulty_score: f32,
    #[serde(default)]
    pub max_difficulty_probability: f32,
    #[serde(default)]
    pub active_max_difficulty_probability: f32,
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
    pub capability_frontier_allowed_max_difficulty: usize,
    #[serde(default)]
    pub capability_frontier_coverage: Vec<RuliadCapabilityCoverageMetric>,
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

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RuliadCapabilityCoverageMetric {
    pub difficulty_level: usize,
    pub candidate_coverage: f32,
    pub family_coverage: f32,
    pub task_coverage: f32,
    pub contract_coverage: f32,
    pub observed_items: usize,
    pub mastered: bool,
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
