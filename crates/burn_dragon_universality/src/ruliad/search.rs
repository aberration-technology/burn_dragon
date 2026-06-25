use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::ruliad::metrics::{
    RuliadBucketMetric, RuliadCapabilityFeedback, RuliadGroupMetric, RuliadMetricSnapshot,
    RuliadSampleTelemetry,
};

const SNAPSHOT_TOP_BUCKET_COUNT: usize = 12;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
pub struct RuliadSamplerConfig {
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_exploration_floor")]
    pub exploration_floor: f32,
    #[serde(default = "default_target_loss")]
    pub target_loss: f32,
    #[serde(default = "default_hash_noise_penalty")]
    pub hash_noise_penalty: f32,
    #[serde(default = "default_mastery_escape_weight")]
    pub mastery_escape_weight: f32,
    #[serde(default = "default_mastery_escape_threshold")]
    pub mastery_escape_threshold: f32,
    #[serde(default = "default_mastery_min_normalized_difficulty")]
    pub mastery_min_normalized_difficulty: f32,
    #[serde(default = "default_mastery_min_max_difficulty_probability")]
    pub mastery_min_max_difficulty_probability: f32,
    #[serde(default = "default_mastery_hash_noise_max_probability")]
    pub mastery_hash_noise_max_probability: f32,
}

impl Default for RuliadSamplerConfig {
    fn default() -> Self {
        Self {
            temperature: default_temperature(),
            exploration_floor: default_exploration_floor(),
            target_loss: default_target_loss(),
            hash_noise_penalty: default_hash_noise_penalty(),
            mastery_escape_weight: default_mastery_escape_weight(),
            mastery_escape_threshold: default_mastery_escape_threshold(),
            mastery_min_normalized_difficulty: default_mastery_min_normalized_difficulty(),
            mastery_min_max_difficulty_probability: default_mastery_min_max_difficulty_probability(
            ),
            mastery_hash_noise_max_probability: default_mastery_hash_noise_max_probability(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RuliadSamplerCandidate {
    pub oracle_hash: String,
    pub family: String,
    pub task_kind: String,
    #[serde(default)]
    pub difficulty_level: usize,
    #[serde(default)]
    pub params_hash: String,
    #[serde(default = "default_prior")]
    pub prior: f32,
    #[serde(default = "default_cost")]
    pub cost: f32,
    #[serde(default)]
    pub loss_ema: f32,
    #[serde(default)]
    pub previous_loss_ema: f32,
    #[serde(default)]
    pub gradient_alignment: f32,
    #[serde(default)]
    pub is_hash_noise: bool,
}

impl RuliadSamplerCandidate {
    pub fn utility(&self, config: RuliadSamplerConfig) -> f32 {
        let learning_progress = (self.previous_loss_ema - self.loss_ema).max(0.0);
        let difficulty_gate = difficulty_gate(self.loss_ema, config.target_loss);
        let gradient = self.gradient_alignment.max(0.0);
        let hash_penalty = if self.is_hash_noise {
            config.hash_noise_penalty
        } else {
            0.0
        };
        (learning_progress + difficulty_gate + gradient - hash_penalty) / self.cost.max(1e-6)
    }
}

#[derive(Debug, Clone)]
pub struct RuliadFrontierSampler {
    config: RuliadSamplerConfig,
    candidates: Vec<RuliadSamplerCandidate>,
    verifier_failures: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RuliadFrontierSamplerState {
    pub candidates: Vec<RuliadSamplerCandidate>,
    #[serde(default)]
    pub verifier_failures: usize,
}

impl RuliadFrontierSampler {
    pub fn new(config: RuliadSamplerConfig, candidates: Vec<RuliadSamplerCandidate>) -> Self {
        Self {
            config,
            candidates,
            verifier_failures: 0,
        }
    }

    pub fn candidates(&self) -> &[RuliadSamplerCandidate] {
        &self.candidates
    }

    pub fn export_state(&self) -> RuliadFrontierSamplerState {
        RuliadFrontierSamplerState {
            candidates: self.candidates.clone(),
            verifier_failures: self.verifier_failures,
        }
    }

    pub fn from_state(config: RuliadSamplerConfig, state: RuliadFrontierSamplerState) -> Self {
        Self {
            config,
            candidates: state.candidates,
            verifier_failures: state.verifier_failures,
        }
    }

    pub fn max_difficulty_level(&self) -> usize {
        self.candidates
            .iter()
            .map(|candidate| candidate.difficulty_level)
            .max()
            .unwrap_or(0)
    }

    pub fn add_candidates(&mut self, candidates: impl IntoIterator<Item = RuliadSamplerCandidate>) {
        for candidate in candidates {
            if self
                .candidates
                .iter()
                .any(|existing| existing.oracle_hash == candidate.oracle_hash)
            {
                continue;
            }
            self.candidates.push(candidate);
        }
    }

    pub fn probabilities(&self) -> Vec<f32> {
        if self.candidates.is_empty() {
            return Vec::new();
        }
        let temperature = self.config.temperature.max(1e-6);
        let max_difficulty = self.max_difficulty_level().max(1);
        let mastered_fraction = self
            .candidates
            .iter()
            .filter(|candidate| candidate.loss_ema <= self.config.target_loss)
            .count() as f32
            / self.candidates.len() as f32;
        let mastery_pressure = ((mastered_fraction - self.config.mastery_escape_threshold)
            / (1.0 - self.config.mastery_escape_threshold).max(1e-6))
        .clamp(0.0, 1.0);
        let logits = self
            .candidates
            .iter()
            .map(|candidate| {
                let normalized_difficulty =
                    candidate.difficulty_level as f32 / max_difficulty as f32;
                let mastery_escape = mastery_pressure
                    * self.config.mastery_escape_weight.max(0.0)
                    * normalized_difficulty;
                candidate.prior.max(1e-9).ln()
                    + (candidate.utility(self.config) + mastery_escape) / temperature
            })
            .collect::<Vec<_>>();
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probs = logits
            .iter()
            .map(|logit| (*logit - max_logit).exp())
            .collect::<Vec<_>>();
        let sum = probs.iter().sum::<f32>().max(1e-12);
        for prob in &mut probs {
            *prob /= sum;
        }
        let floor = self.config.exploration_floor.clamp(0.0, 1.0);
        let uniform = 1.0 / probs.len() as f32;
        for prob in &mut probs {
            *prob = *prob * (1.0 - floor) + uniform * floor;
        }
        self.enforce_mastery_frontier(&mut probs);
        probs
    }

    fn enforce_mastery_frontier(&self, probs: &mut [f32]) {
        let max_difficulty = self.max_difficulty_level();
        if max_difficulty == 0 || probs.is_empty() {
            return;
        }
        let mastered_probability = probs
            .iter()
            .zip(&self.candidates)
            .filter_map(|(prob, candidate)| {
                (candidate.loss_ema <= self.config.target_loss).then_some(*prob)
            })
            .sum::<f32>();
        if mastered_probability < self.config.mastery_escape_threshold {
            return;
        }

        let target_normalized = self
            .config
            .mastery_min_normalized_difficulty
            .clamp(0.0, 1.0);
        let target_max_probability = self
            .config
            .mastery_min_max_difficulty_probability
            .clamp(0.0, 1.0);
        let target_hash_noise = self
            .config
            .mastery_hash_noise_max_probability
            .clamp(0.0, 1.0);
        if target_normalized <= 0.0 && target_max_probability <= 0.0 {
            return;
        }

        let high_distribution = self.high_difficulty_distribution(max_difficulty);
        if high_distribution.is_empty() {
            return;
        }

        let satisfies = |candidate_probs: &[f32]| {
            let normalized_difficulty =
                weighted_normalized_difficulty(candidate_probs, &self.candidates, max_difficulty);
            let max_difficulty_probability = weighted_max_difficulty_probability(
                candidate_probs,
                &self.candidates,
                max_difficulty,
            );
            let hash_noise_probability =
                weighted_hash_noise_probability(candidate_probs, &self.candidates);
            normalized_difficulty >= target_normalized
                && max_difficulty_probability >= target_max_probability
                && hash_noise_probability <= target_hash_noise
        };
        if satisfies(probs) {
            return;
        }

        let original = probs.to_vec();
        let mut best = original.clone();
        let mut lo = 0.0f32;
        let mut hi = 1.0f32;
        for _ in 0..24 {
            let alpha = (lo + hi) * 0.5;
            let mixed = mix_probabilities(&original, &high_distribution, alpha);
            if satisfies(&mixed) {
                best = mixed;
                hi = alpha;
            } else {
                lo = alpha;
            }
        }
        if !satisfies(&best) {
            best = high_distribution;
        }
        probs.copy_from_slice(&best);
    }

    fn high_difficulty_distribution(&self, max_difficulty: usize) -> Vec<f32> {
        let mut weights = self
            .candidates
            .iter()
            .map(|candidate| {
                if candidate.difficulty_level == max_difficulty && !candidate.is_hash_noise {
                    candidate.prior.max(1e-9) / candidate.cost.max(1e-6)
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        normalize_distribution(&mut weights);
        weights
    }

    pub fn record_telemetry(&mut self, telemetry: &RuliadSampleTelemetry) {
        if !telemetry.accepted {
            self.verifier_failures += 1;
            return;
        }
        if let Some(candidate) = self
            .candidates
            .iter_mut()
            .find(|candidate| candidate.oracle_hash == telemetry.oracle_hash)
        {
            candidate.previous_loss_ema = telemetry
                .previous_loss
                .unwrap_or(candidate.loss_ema.max(telemetry.loss));
            candidate.loss_ema = if candidate.loss_ema <= f32::EPSILON {
                telemetry.loss
            } else {
                candidate.loss_ema * 0.9 + telemetry.loss * 0.1
            };
            if let Some(gradient_alignment) = telemetry.gradient_alignment {
                candidate.gradient_alignment =
                    candidate.gradient_alignment * 0.9 + gradient_alignment * 0.1;
            }
            candidate.cost = telemetry.verification_cost.max(1e-6);
        }
    }

    pub fn record_capability_feedback(&mut self, feedback: &RuliadCapabilityFeedback) {
        if feedback.item_count == 0 {
            return;
        }
        if !capability_feedback_can_update_sampler(feedback) {
            return;
        }
        let target_loss = self.config.target_loss.max(1e-6);
        let loss = capability_feedback_loss(feedback, target_loss);
        let gradient_alignment = capability_feedback_alignment(feedback);
        let promotion_weight = capability_feedback_promotion_weight(feedback);
        if promotion_weight <= f32::EPSILON {
            return;
        }
        for candidate in &mut self.candidates {
            if !candidate_matches_capability_feedback(candidate, &feedback.group_label) {
                continue;
            }
            let lowers_loss = loss < candidate.loss_ema;
            if !lowers_loss {
                continue;
            }
            let update_alpha = 0.15 * promotion_weight;
            let previous_loss = candidate.loss_ema;
            let next_loss = candidate.loss_ema * (1.0 - update_alpha) + loss * update_alpha;
            candidate.previous_loss_ema = previous_loss;
            candidate.loss_ema = next_loss;
            candidate.gradient_alignment =
                candidate.gradient_alignment * 0.85 + gradient_alignment * 0.15 * promotion_weight;
        }
    }

    pub fn snapshot(&self) -> RuliadMetricSnapshot {
        let probs = self.probabilities();
        self.snapshot_with_probabilities(&probs)
    }

    pub fn snapshot_with_probabilities(&self, probabilities: &[f32]) -> RuliadMetricSnapshot {
        let fallback_probs;
        let probs = if probabilities.len() == self.candidates.len() {
            probabilities
        } else {
            fallback_probs = self.probabilities();
            &fallback_probs
        };
        let sampler_entropy_bits = probs
            .iter()
            .filter(|prob| **prob > 0.0)
            .map(|prob| -prob * prob.log2())
            .sum::<f32>();
        let hash_noise_probability = probs
            .iter()
            .zip(&self.candidates)
            .filter_map(|(prob, candidate)| candidate.is_hash_noise.then_some(*prob))
            .sum::<f32>();
        let mean_loss = mean(self.candidates.iter().map(|candidate| candidate.loss_ema));
        let mean_learning_progress = mean(
            self.candidates
                .iter()
                .map(|candidate| (candidate.previous_loss_ema - candidate.loss_ema).max(0.0)),
        );
        let frontier_loss = probs
            .iter()
            .zip(&self.candidates)
            .map(|(prob, candidate)| prob * candidate.loss_ema)
            .sum::<f32>();
        let target_difficulty_score = probs
            .iter()
            .zip(&self.candidates)
            .map(|(prob, candidate)| {
                prob * difficulty_gate(candidate.loss_ema, self.config.target_loss)
            })
            .sum::<f32>();
        let max_difficulty_level = self.max_difficulty_level();
        let mean_difficulty_level = probs
            .iter()
            .zip(&self.candidates)
            .map(|(prob, candidate)| prob * candidate.difficulty_level as f32)
            .sum::<f32>();
        let normalized_difficulty_score = if max_difficulty_level == 0 {
            0.0
        } else {
            mean_difficulty_level / max_difficulty_level as f32
        };
        let max_difficulty_probability = probs
            .iter()
            .zip(&self.candidates)
            .filter_map(|(prob, candidate)| {
                (candidate.difficulty_level == max_difficulty_level).then_some(*prob)
            })
            .sum::<f32>();
        let mastered_probability = probs
            .iter()
            .zip(&self.candidates)
            .filter_map(|(prob, candidate)| {
                (candidate.loss_ema <= self.config.target_loss).then_some(*prob)
            })
            .sum::<f32>();
        let top_buckets = top_bucket_metrics(
            probs,
            &self.candidates,
            self.config.target_loss,
            SNAPSHOT_TOP_BUCKET_COUNT,
        );
        let mut difficulty_buckets = group_metrics_by(
            probs,
            &self.candidates,
            self.config.target_loss,
            |candidate| format!("d{}", candidate.difficulty_level),
        );
        difficulty_buckets.sort_by(|left, right| {
            difficulty_label_key(&left.label)
                .cmp(&difficulty_label_key(&right.label))
                .then_with(|| left.label.cmp(&right.label))
        });
        let family_buckets = group_metrics_by(
            probs,
            &self.candidates,
            self.config.target_loss,
            |candidate| candidate.family.clone(),
        );
        let task_buckets = group_metrics_by(
            probs,
            &self.candidates,
            self.config.target_loss,
            |candidate| format!("{}:{}", candidate.family, candidate.task_kind),
        );
        RuliadMetricSnapshot {
            sample_count: self.candidates.len(),
            verifier_failures: self.verifier_failures,
            sampler_entropy_bits,
            hash_noise_probability,
            mean_loss,
            mean_learning_progress,
            frontier_loss,
            target_loss: self.config.target_loss,
            target_difficulty_score,
            max_difficulty_level,
            mean_difficulty_level,
            normalized_difficulty_score,
            max_difficulty_probability,
            mastered_probability,
            frontier_extension_count: 0,
            frontier_saturated: false,
            frontier_unbounded: false,
            top_buckets,
            difficulty_buckets,
            family_buckets,
            task_buckets,
        }
    }
}

fn difficulty_gate(loss_ema: f32, target_loss: f32) -> f32 {
    1.0 / (1.0 + (loss_ema - target_loss).abs())
}

fn capability_feedback_structural_error(feedback: &RuliadCapabilityFeedback) -> f32 {
    feedback.schema_valid_wrong_rate.clamp(0.0, 1.0)
        + feedback.malformed_rate.clamp(0.0, 1.0) * 2.0
        + feedback.missing_rate.clamp(0.0, 1.0) * 2.0
        + (0.50 - feedback.completion_health_rate.clamp(0.0, 1.0)).max(0.0) * 2.0
}

fn capability_feedback_is_difficulty_group(feedback: &RuliadCapabilityFeedback) -> bool {
    feedback.group_label.starts_with("difficulty:")
}

fn capability_feedback_can_update_sampler(feedback: &RuliadCapabilityFeedback) -> bool {
    capability_feedback_is_difficulty_group(feedback)
        && feedback.verifier_rate >= 0.05
        && capability_feedback_structural_error(feedback) <= 0.35
}

fn capability_feedback_promotion_weight(feedback: &RuliadCapabilityFeedback) -> f32 {
    let verifier = feedback.verifier_rate.clamp(0.0, 1.0);
    let partial = feedback.partial_credit_rate.clamp(0.0, 1.0);
    let schema_wrong = feedback.schema_valid_wrong_rate.clamp(0.0, 1.0);
    let malformed = feedback.malformed_rate.clamp(0.0, 1.0);
    let missing = feedback.missing_rate.clamp(0.0, 1.0);
    let completion = feedback.completion_health_rate.clamp(0.0, 1.0);
    let structurally_healthy =
        completion >= 0.60 && malformed <= 0.05 && missing <= 0.05 && schema_wrong <= 0.25;
    if !structurally_healthy {
        return 0.0;
    }
    if verifier < 0.05 {
        return 0.0;
    }
    (verifier * 2.0 + partial * 0.5).clamp(0.05, 1.0)
}

fn capability_feedback_loss(feedback: &RuliadCapabilityFeedback, target_loss: f32) -> f32 {
    let verifier_gap = 1.0 - feedback.verifier_rate.clamp(0.0, 1.0);
    let partial_credit = feedback.partial_credit_rate.clamp(0.0, 1.0);
    target_loss * (0.75 + verifier_gap * 0.50 - partial_credit * 0.15).max(0.35)
}

fn capability_feedback_alignment(feedback: &RuliadCapabilityFeedback) -> f32 {
    let structure_health = feedback.completion_health_rate.clamp(0.0, 1.0)
        * (1.0 - feedback.malformed_rate.clamp(0.0, 1.0))
        * (1.0 - feedback.missing_rate.clamp(0.0, 1.0))
        * (1.0 - feedback.schema_valid_wrong_rate.clamp(0.0, 1.0));
    let verifier = feedback.verifier_rate.clamp(0.0, 1.0);
    let partial = if verifier >= 0.05 {
        feedback.partial_credit_rate.clamp(0.0, 1.0)
    } else {
        0.0
    };
    (structure_health * (verifier + partial * 0.25)).clamp(0.0, 1.0)
}

fn candidate_matches_capability_feedback(
    candidate: &RuliadSamplerCandidate,
    group_label: &str,
) -> bool {
    if let Some(difficulty) = group_label
        .strip_prefix("difficulty:")
        .and_then(|label| label.strip_prefix('d'))
        .and_then(|value| value.parse::<usize>().ok())
    {
        return candidate.difficulty_level == difficulty;
    }
    if let Some(family) = group_label.strip_prefix("family:") {
        return candidate.family == family;
    }
    if let Some(task) = group_label.strip_prefix("task:") {
        return candidate.task_kind == task
            || format!("{}:{}", candidate.family, candidate.task_kind) == task;
    }
    if let Some(label) = group_label.strip_prefix("bucket:") {
        return candidate.oracle_hash == label;
    }
    candidate.oracle_hash == group_label
}

fn top_bucket_metrics(
    probs: &[f32],
    candidates: &[RuliadSamplerCandidate],
    target_loss: f32,
    limit: usize,
) -> Vec<RuliadBucketMetric> {
    let mut buckets = probs
        .iter()
        .zip(candidates)
        .map(|(probability, candidate)| RuliadBucketMetric {
            label: candidate.oracle_hash.clone(),
            family: candidate.family.clone(),
            task_kind: candidate.task_kind.clone(),
            difficulty_level: candidate.difficulty_level,
            probability: *probability,
            loss_ema: candidate.loss_ema,
            previous_loss_ema: candidate.previous_loss_ema,
            learning_progress: (candidate.previous_loss_ema - candidate.loss_ema).max(0.0),
            mastered: candidate.loss_ema <= target_loss,
        })
        .collect::<Vec<_>>();
    buckets.sort_by(|left, right| {
        right
            .probability
            .partial_cmp(&left.probability)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.label.cmp(&right.label))
    });
    buckets.truncate(limit);
    buckets
}

fn group_metrics_by(
    probs: &[f32],
    candidates: &[RuliadSamplerCandidate],
    target_loss: f32,
    label: impl Fn(&RuliadSamplerCandidate) -> String,
) -> Vec<RuliadGroupMetric> {
    #[derive(Default)]
    struct Accumulator {
        candidate_count: usize,
        probability: f32,
        weighted_loss: f32,
        learning_progress: f32,
        mastered_probability: f32,
        weighted_difficulty: f32,
    }

    let mut groups = BTreeMap::<String, Accumulator>::new();
    for (probability, candidate) in probs.iter().zip(candidates) {
        let group = groups.entry(label(candidate)).or_default();
        group.candidate_count += 1;
        group.probability += *probability;
        group.weighted_loss += *probability * candidate.loss_ema;
        group.learning_progress +=
            *probability * (candidate.previous_loss_ema - candidate.loss_ema).max(0.0);
        if candidate.loss_ema <= target_loss {
            group.mastered_probability += *probability;
        }
        group.weighted_difficulty += *probability * candidate.difficulty_level as f32;
    }

    let mut metrics = groups
        .into_iter()
        .map(|(label, group)| {
            let probability = group.probability.max(0.0);
            let denominator = probability.max(1e-12);
            RuliadGroupMetric {
                label,
                candidate_count: group.candidate_count,
                probability,
                mean_loss: group.weighted_loss / denominator,
                learning_progress: group.learning_progress / denominator,
                mastered_probability: group.mastered_probability,
                mean_difficulty_level: group.weighted_difficulty / denominator,
            }
        })
        .collect::<Vec<_>>();
    metrics.sort_by(|left, right| {
        right
            .probability
            .partial_cmp(&left.probability)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.label.cmp(&right.label))
    });
    metrics
}

fn difficulty_label_key(label: &str) -> usize {
    label
        .strip_prefix('d')
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(usize::MAX)
}

fn mean(values: impl Iterator<Item = f32>) -> f32 {
    let mut count = 0usize;
    let mut sum = 0.0;
    for value in values {
        count += 1;
        sum += value;
    }
    if count == 0 { 0.0 } else { sum / count as f32 }
}

fn mix_probabilities(left: &[f32], right: &[f32], alpha: f32) -> Vec<f32> {
    let alpha = alpha.clamp(0.0, 1.0);
    left.iter()
        .zip(right)
        .map(|(left, right)| left * (1.0 - alpha) + right * alpha)
        .collect()
}

fn normalize_distribution(values: &mut [f32]) {
    let sum = values
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .sum::<f32>();
    if sum <= f32::EPSILON {
        let uniform = if values.is_empty() {
            0.0
        } else {
            1.0 / values.len() as f32
        };
        for value in values {
            *value = uniform;
        }
        return;
    }
    for value in values {
        *value = if value.is_finite() && *value > 0.0 {
            *value / sum
        } else {
            0.0
        };
    }
}

fn weighted_normalized_difficulty(
    probs: &[f32],
    candidates: &[RuliadSamplerCandidate],
    max_difficulty: usize,
) -> f32 {
    if max_difficulty == 0 {
        return 0.0;
    }
    probs
        .iter()
        .zip(candidates)
        .map(|(prob, candidate)| prob * candidate.difficulty_level as f32)
        .sum::<f32>()
        / max_difficulty as f32
}

fn weighted_max_difficulty_probability(
    probs: &[f32],
    candidates: &[RuliadSamplerCandidate],
    max_difficulty: usize,
) -> f32 {
    probs
        .iter()
        .zip(candidates)
        .filter_map(|(prob, candidate)| {
            (candidate.difficulty_level == max_difficulty).then_some(*prob)
        })
        .sum()
}

fn weighted_hash_noise_probability(probs: &[f32], candidates: &[RuliadSamplerCandidate]) -> f32 {
    probs
        .iter()
        .zip(candidates)
        .filter_map(|(prob, candidate)| candidate.is_hash_noise.then_some(*prob))
        .sum()
}

fn default_temperature() -> f32 {
    1.0
}

fn default_exploration_floor() -> f32 {
    0.05
}

fn default_target_loss() -> f32 {
    2.0
}

fn default_hash_noise_penalty() -> f32 {
    4.0
}

fn default_mastery_escape_weight() -> f32 {
    4.0
}

fn default_mastery_escape_threshold() -> f32 {
    0.70
}

fn default_mastery_min_normalized_difficulty() -> f32 {
    0.90
}

fn default_mastery_min_max_difficulty_probability() -> f32 {
    0.35
}

fn default_mastery_hash_noise_max_probability() -> f32 {
    0.01
}

fn default_prior() -> f32 {
    1.0
}

fn default_cost() -> f32 {
    1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampler_penalizes_hash_noise_canary() {
        let sampler = RuliadFrontierSampler::new(
            RuliadSamplerConfig::default(),
            vec![
                RuliadSamplerCandidate {
                    oracle_hash: "structured".to_string(),
                    family: "eca".to_string(),
                    task_kind: "multi_step_state".to_string(),
                    difficulty_level: 0,
                    params_hash: String::new(),
                    prior: 1.0,
                    cost: 1.0,
                    loss_ema: 2.0,
                    previous_loss_ema: 3.0,
                    gradient_alignment: 0.0,
                    is_hash_noise: false,
                },
                RuliadSamplerCandidate {
                    oracle_hash: "noise".to_string(),
                    family: "hash_noise".to_string(),
                    task_kind: "hash_canary".to_string(),
                    difficulty_level: 0,
                    params_hash: String::new(),
                    prior: 1.0,
                    cost: 1.0,
                    loss_ema: 8.0,
                    previous_loss_ema: 8.0,
                    gradient_alignment: 0.0,
                    is_hash_noise: true,
                },
            ],
        );
        let probs = sampler.probabilities();
        assert!(probs[0] > probs[1]);
        assert!(sampler.snapshot().hash_noise_probability < 0.5);
    }

    #[test]
    fn capability_feedback_does_not_promote_structured_wrong_without_verifier() {
        let mut sampler = RuliadFrontierSampler::new(
            RuliadSamplerConfig {
                exploration_floor: 0.0,
                target_loss: 2.0,
                ..RuliadSamplerConfig::default()
            },
            vec![
                RuliadSamplerCandidate {
                    oracle_hash: "category:verify_category_law@d2#00000001".to_string(),
                    family: "category".to_string(),
                    task_kind: "verify_category_law".to_string(),
                    difficulty_level: 2,
                    params_hash: String::new(),
                    prior: 1.0,
                    cost: 1.0,
                    loss_ema: 8.0,
                    previous_loss_ema: 8.0,
                    gradient_alignment: 0.0,
                    is_hash_noise: false,
                },
                RuliadSamplerCandidate {
                    oracle_hash: "proof_tree:prove_theorem@d2#00000002".to_string(),
                    family: "proof_tree".to_string(),
                    task_kind: "prove_theorem".to_string(),
                    difficulty_level: 2,
                    params_hash: String::new(),
                    prior: 1.0,
                    cost: 1.0,
                    loss_ema: 8.0,
                    previous_loss_ema: 8.0,
                    gradient_alignment: 0.0,
                    is_hash_noise: false,
                },
            ],
        );
        let before = sampler.snapshot();

        sampler.record_capability_feedback(&RuliadCapabilityFeedback {
            group_label: "family:category".to_string(),
            item_count: 16,
            verifier_rate: 0.0,
            partial_credit_rate: 0.25,
            schema_valid_wrong_rate: 0.05,
            malformed_rate: 0.0,
            missing_rate: 0.0,
            completion_health_rate: 0.95,
        });
        let snapshot = sampler.snapshot();
        let category = snapshot
            .top_buckets
            .iter()
            .find(|bucket| bucket.family == "category")
            .expect("category bucket");
        let proof = snapshot
            .top_buckets
            .iter()
            .find(|bucket| bucket.family == "proof_tree")
            .expect("proof bucket");

        assert!(
            category.probability <= proof.probability,
            "partial-only structured feedback should not promote a bucket before verifier signal"
        );
        assert_eq!(category.learning_progress, 0.0);
        assert_eq!(snapshot.mean_difficulty_level, before.mean_difficulty_level);
    }

    #[test]
    fn capability_feedback_ignores_malformed_non_difficulty_groups() {
        let mut sampler = RuliadFrontierSampler::new(
            RuliadSamplerConfig {
                exploration_floor: 0.0,
                target_loss: 2.0,
                ..RuliadSamplerConfig::default()
            },
            vec![RuliadSamplerCandidate {
                oracle_hash: "category:verify_category_law@d1#00000001".to_string(),
                family: "category".to_string(),
                task_kind: "verify_category_law".to_string(),
                difficulty_level: 1,
                params_hash: String::new(),
                prior: 1.0,
                cost: 1.0,
                loss_ema: 2.0,
                previous_loss_ema: 2.0,
                gradient_alignment: 0.0,
                is_hash_noise: false,
            }],
        );

        sampler.record_capability_feedback(&RuliadCapabilityFeedback {
            group_label: "task:verify_category_law".to_string(),
            item_count: 16,
            verifier_rate: 0.0,
            partial_credit_rate: 0.0,
            schema_valid_wrong_rate: 0.20,
            malformed_rate: 0.40,
            missing_rate: 0.10,
            completion_health_rate: 0.30,
        });
        let candidate = sampler.candidates().first().expect("candidate");

        assert_eq!(candidate.cost, 1.0);
        assert_eq!(candidate.loss_ema, 2.0);
        assert_eq!(candidate.previous_loss_ema, 2.0);
    }

    #[test]
    fn zero_verifier_difficulty_feedback_does_not_advance_frontier() {
        let mut sampler = RuliadFrontierSampler::new(
            RuliadSamplerConfig {
                exploration_floor: 0.0,
                target_loss: 2.0,
                mastery_escape_weight: 0.0,
                ..RuliadSamplerConfig::default()
            },
            vec![
                RuliadSamplerCandidate {
                    oracle_hash: "category:verify_category_law@d0#00000001".to_string(),
                    family: "category".to_string(),
                    task_kind: "verify_category_law".to_string(),
                    difficulty_level: 0,
                    params_hash: String::new(),
                    prior: 1.0,
                    cost: 1.0,
                    loss_ema: 8.0,
                    previous_loss_ema: 8.0,
                    gradient_alignment: 0.0,
                    is_hash_noise: false,
                },
                RuliadSamplerCandidate {
                    oracle_hash: "category:verify_category_law@d12#00000002".to_string(),
                    family: "category".to_string(),
                    task_kind: "verify_category_law".to_string(),
                    difficulty_level: 12,
                    params_hash: String::new(),
                    prior: 1.0,
                    cost: 1.0,
                    loss_ema: 8.0,
                    previous_loss_ema: 8.0,
                    gradient_alignment: 0.0,
                    is_hash_noise: false,
                },
            ],
        );
        let before = sampler.snapshot().mean_difficulty_level;

        sampler.record_capability_feedback(&RuliadCapabilityFeedback {
            group_label: "difficulty:d12".to_string(),
            item_count: 32,
            verifier_rate: 0.0,
            partial_credit_rate: 0.0,
            schema_valid_wrong_rate: 0.0,
            malformed_rate: 0.0,
            missing_rate: 0.0,
            completion_health_rate: 1.0,
        });
        let snapshot = sampler.snapshot();
        let hard = sampler
            .candidates()
            .iter()
            .find(|candidate| candidate.difficulty_level == 12)
            .expect("hard candidate");

        assert!(
            snapshot.mean_difficulty_level <= before,
            "zero-verifier difficulty feedback should not raise mean difficulty: before={before} after={}",
            snapshot.mean_difficulty_level
        );
        assert_eq!(hard.loss_ema, 8.0);
        assert_eq!(hard.previous_loss_ema, 8.0);
        assert_eq!(hard.cost, 1.0);
    }

    #[test]
    fn verified_difficulty_feedback_can_advance_frontier() {
        let mut sampler = RuliadFrontierSampler::new(
            RuliadSamplerConfig {
                exploration_floor: 0.0,
                target_loss: 2.0,
                mastery_escape_weight: 0.0,
                ..RuliadSamplerConfig::default()
            },
            vec![
                RuliadSamplerCandidate {
                    oracle_hash: "category:verify_category_law@d0#00000001".to_string(),
                    family: "category".to_string(),
                    task_kind: "verify_category_law".to_string(),
                    difficulty_level: 0,
                    params_hash: String::new(),
                    prior: 1.0,
                    cost: 1.0,
                    loss_ema: 8.0,
                    previous_loss_ema: 8.0,
                    gradient_alignment: 0.0,
                    is_hash_noise: false,
                },
                RuliadSamplerCandidate {
                    oracle_hash: "category:verify_category_law@d12#00000002".to_string(),
                    family: "category".to_string(),
                    task_kind: "verify_category_law".to_string(),
                    difficulty_level: 12,
                    params_hash: String::new(),
                    prior: 1.0,
                    cost: 1.0,
                    loss_ema: 8.0,
                    previous_loss_ema: 8.0,
                    gradient_alignment: 0.0,
                    is_hash_noise: false,
                },
            ],
        );
        let before = sampler.snapshot().mean_difficulty_level;

        sampler.record_capability_feedback(&RuliadCapabilityFeedback {
            group_label: "difficulty:d12".to_string(),
            item_count: 32,
            verifier_rate: 0.25,
            partial_credit_rate: 0.25,
            schema_valid_wrong_rate: 0.0,
            malformed_rate: 0.0,
            missing_rate: 0.0,
            completion_health_rate: 1.0,
        });
        let snapshot = sampler.snapshot();

        assert!(
            snapshot.mean_difficulty_level > before,
            "verified difficulty feedback should be able to advance mean difficulty: before={before} after={}",
            snapshot.mean_difficulty_level
        );
    }

    #[test]
    fn snapshot_reports_weighted_difficulty_frontier() {
        let sampler = RuliadFrontierSampler::new(
            RuliadSamplerConfig {
                temperature: 100.0,
                exploration_floor: 0.0,
                target_loss: 2.0,
                hash_noise_penalty: 4.0,
                mastery_escape_weight: 0.0,
                mastery_escape_threshold: 0.70,
                ..RuliadSamplerConfig::default()
            },
            vec![
                RuliadSamplerCandidate {
                    oracle_hash: "easy".to_string(),
                    family: "category".to_string(),
                    task_kind: "trace".to_string(),
                    difficulty_level: 0,
                    params_hash: String::new(),
                    prior: 1.0,
                    cost: 1.0,
                    loss_ema: 1.0,
                    previous_loss_ema: 1.5,
                    gradient_alignment: 0.0,
                    is_hash_noise: false,
                },
                RuliadSamplerCandidate {
                    oracle_hash: "hard".to_string(),
                    family: "category".to_string(),
                    task_kind: "proof".to_string(),
                    difficulty_level: 1,
                    params_hash: String::new(),
                    prior: 1.0,
                    cost: 1.0,
                    loss_ema: 3.0,
                    previous_loss_ema: 3.5,
                    gradient_alignment: 0.0,
                    is_hash_noise: false,
                },
            ],
        );

        let snapshot = sampler.snapshot();

        assert!((snapshot.frontier_loss - 2.0).abs() < 0.05);
        assert_eq!(snapshot.target_loss, 2.0);
        assert!((snapshot.target_difficulty_score - 0.5).abs() < 0.02);
        assert!((snapshot.mastered_probability - 0.5).abs() < 0.05);
        assert!((snapshot.mean_difficulty_level - 0.5).abs() < 0.05);
        assert!((snapshot.normalized_difficulty_score - 0.5).abs() < 0.05);
        assert!((snapshot.max_difficulty_probability - 0.5).abs() < 0.05);
        assert_eq!(snapshot.top_buckets.len(), 2);
        assert_eq!(snapshot.top_buckets[0].label, "easy");
        assert!(snapshot.top_buckets[0].probability > 0.45);
        assert!(snapshot.top_buckets[0].mastered);
        assert_eq!(
            snapshot
                .difficulty_buckets
                .iter()
                .map(|bucket| bucket.label.as_str())
                .collect::<Vec<_>>(),
            vec!["d0", "d1"]
        );
        assert_eq!(snapshot.family_buckets.len(), 1);
        assert_eq!(snapshot.family_buckets[0].label, "category");
        assert!((snapshot.family_buckets[0].probability - 1.0).abs() < 0.01);
        assert_eq!(snapshot.task_buckets.len(), 2);
    }

    #[test]
    fn sampler_escapes_mastered_easy_buckets_toward_higher_difficulty() {
        let candidates = (0..=6)
            .map(|difficulty_level| RuliadSamplerCandidate {
                oracle_hash: format!("d{difficulty_level}"),
                family: "category".to_string(),
                task_kind: "proof".to_string(),
                difficulty_level,
                params_hash: String::new(),
                prior: 1.0,
                cost: 1.0,
                loss_ema: 0.25,
                previous_loss_ema: 0.30,
                gradient_alignment: 0.0,
                is_hash_noise: false,
            })
            .collect::<Vec<_>>();
        let sampler = RuliadFrontierSampler::new(
            RuliadSamplerConfig {
                temperature: 1.0,
                exploration_floor: 0.0,
                target_loss: 2.0,
                hash_noise_penalty: 4.0,
                mastery_escape_weight: 4.0,
                mastery_escape_threshold: 0.70,
                mastery_min_normalized_difficulty: 0.90,
                mastery_min_max_difficulty_probability: 0.35,
                mastery_hash_noise_max_probability: 0.01,
            },
            candidates,
        );

        let snapshot = sampler.snapshot();

        assert!(
            snapshot.normalized_difficulty_score >= 0.90,
            "expected mastery escape to bias toward high difficulty, got {}",
            snapshot.normalized_difficulty_score
        );
        assert!(
            snapshot.max_difficulty_probability >= 0.35,
            "expected max-difficulty bucket to receive substantial probability, got {}",
            snapshot.max_difficulty_probability
        );
    }
}
