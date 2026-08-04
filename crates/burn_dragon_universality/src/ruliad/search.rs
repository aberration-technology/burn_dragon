use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::ruliad::config::{
    RuliadFamilyKind, RuliadProofActionAnswerContract, RuliadTaskKind, ruliad_source_semantics,
};
use crate::ruliad::metrics::{
    RULIAD_SOURCE_CAPABILITY_LABEL_PREFIX, RuliadBucketMetric, RuliadCapabilityCoverageMetric,
    RuliadCapabilityFeedback, RuliadGroupMetric, RuliadMetricSnapshot, RuliadSampleTelemetry,
    ruliad_source_capability_label,
};
use crate::ruliad::world::{
    RuliadCapabilityCoverage, RuliadCapabilityMasteryThresholds, RuliadCapabilityPosterior,
};

const SNAPSHOT_TOP_BUCKET_COUNT: usize = 12;
const ACTIVE_PROBABILITY_EPSILON: f32 = 1.0e-6;

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
    #[serde(default = "default_max_answer_contract_probability")]
    pub max_answer_contract_probability: f32,
    #[serde(default = "default_min_answer_contract_probability")]
    pub min_answer_contract_probability: f32,
    #[serde(default = "default_capability_frontier_max_ahead")]
    pub capability_frontier_max_ahead: usize,
    #[serde(default = "default_capability_frontier_max_unverified_probability")]
    pub capability_frontier_max_unverified_probability: f32,
    #[serde(default = "default_capability_remediation_weight")]
    pub capability_remediation_weight: f32,
    #[serde(default = "default_capability_frontier_min_coverage")]
    pub capability_frontier_min_coverage: f32,
    #[serde(default)]
    pub capability_mastery: RuliadCapabilityMasteryThresholds,
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
            max_answer_contract_probability: default_max_answer_contract_probability(),
            min_answer_contract_probability: default_min_answer_contract_probability(),
            capability_frontier_max_ahead: default_capability_frontier_max_ahead(),
            capability_frontier_max_unverified_probability:
                default_capability_frontier_max_unverified_probability(),
            capability_remediation_weight: default_capability_remediation_weight(),
            capability_frontier_min_coverage: default_capability_frontier_min_coverage(),
            capability_mastery: RuliadCapabilityMasteryThresholds::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RuliadSamplerCandidate {
    pub oracle_hash: String,
    pub family: String,
    pub task_kind: String,
    /// Stable semantic contract for the supervised answer emitted by this source bucket.
    #[serde(default)]
    pub answer_contract: String,
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
    #[serde(default)]
    pub capability_feedback_count: usize,
    #[serde(default)]
    pub capability_verifier_ema: f32,
    #[serde(default)]
    pub capability_partial_ema: f32,
    #[serde(default)]
    pub capability_completion_health_ema: f32,
    #[serde(default)]
    pub capability_schema_wrong_ema: f32,
    #[serde(default)]
    pub capability_malformed_ema: f32,
    #[serde(default)]
    pub capability_missing_ema: f32,
}

impl RuliadSamplerCandidate {
    pub fn utility(&self, config: RuliadSamplerConfig) -> f32 {
        let learning_progress = (self.previous_loss_ema - self.loss_ema).max(0.0);
        let difficulty_gate = difficulty_gate(self.loss_ema, config.target_loss);
        let gradient = self.gradient_alignment.max(0.0);
        let capability_remediation = self.capability_remediation_utility(config);
        let hash_penalty = if self.is_hash_noise {
            config.hash_noise_penalty
        } else {
            0.0
        };
        (learning_progress + difficulty_gate + gradient + capability_remediation - hash_penalty)
            / self.cost.max(1e-6)
    }

    fn capability_remediation_utility(&self, config: RuliadSamplerConfig) -> f32 {
        if self.capability_feedback_count == 0 {
            return 0.0;
        }
        let verifier = self.capability_verifier_ema.clamp(0.0, 1.0);
        let completion = self.capability_completion_health_ema.clamp(0.0, 1.0);
        let schema_wrong = self.capability_schema_wrong_ema.clamp(0.0, 1.0);
        let malformed = self.capability_malformed_ema.clamp(0.0, 1.0);
        let missing = self.capability_missing_ema.clamp(0.0, 1.0);
        let parse_health = (1.0 - malformed).powi(2) * (1.0 - missing).powi(2);
        let answerable = completion * parse_health;
        if answerable <= 0.05 {
            return 0.0;
        }
        let verifier_gap = ((0.50 - verifier) / 0.50).clamp(0.0, 1.0);
        let completion_gap = ((0.75 - completion) / 0.75).clamp(0.0, 1.0);
        let remediation_need =
            (schema_wrong * 0.70 + verifier_gap * 0.25 + completion_gap * 0.05).clamp(0.0, 1.0);
        config.capability_remediation_weight.max(0.0) * answerable * remediation_need
    }
}

#[derive(Debug, Clone)]
pub struct RuliadFrontierSampler {
    config: RuliadSamplerConfig,
    candidates: Vec<RuliadSamplerCandidate>,
    capability_posteriors: BTreeMap<String, RuliadCapabilityPosterior>,
    verifier_failures: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RuliadFrontierSamplerState {
    pub candidates: Vec<RuliadSamplerCandidate>,
    #[serde(default)]
    pub capability_posteriors: BTreeMap<String, RuliadCapabilityPosterior>,
    #[serde(default)]
    pub verifier_failures: usize,
}

impl RuliadFrontierSampler {
    pub fn new(config: RuliadSamplerConfig, candidates: Vec<RuliadSamplerCandidate>) -> Self {
        Self {
            config,
            candidates,
            capability_posteriors: BTreeMap::new(),
            verifier_failures: 0,
        }
    }

    pub fn candidates(&self) -> &[RuliadSamplerCandidate] {
        &self.candidates
    }

    pub fn export_state(&self) -> RuliadFrontierSamplerState {
        RuliadFrontierSamplerState {
            candidates: self.candidates.clone(),
            capability_posteriors: self.capability_posteriors.clone(),
            verifier_failures: self.verifier_failures,
        }
    }

    pub fn from_state(config: RuliadSamplerConfig, state: RuliadFrontierSamplerState) -> Self {
        Self {
            config,
            candidates: state.candidates,
            capability_posteriors: state.capability_posteriors,
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
            if let Some(existing) = self
                .candidates
                .iter_mut()
                .find(|existing| existing.oracle_hash == candidate.oracle_hash)
            {
                // Runtime metadata evolves independently from learned sampler state. Refresh
                // additive semantic fields when restoring an older checkpoint while preserving
                // its loss and capability EMAs.
                if !candidate.answer_contract.is_empty() {
                    existing.answer_contract = candidate.answer_contract;
                }
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
            .filter(|candidate| candidate_mastered(candidate, self.config.target_loss))
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
        self.enforce_capability_frontier(&mut probs);
        self.enforce_answer_contract_balance(&mut probs);
        probs
    }

    pub fn apply_probability_constraints(&self, probs: &mut [f32]) {
        if probs.len() != self.candidates.len() {
            return;
        }
        normalize_distribution(probs);
        self.enforce_answer_contract_balance(probs);
    }

    fn enforce_answer_contract_balance(&self, probs: &mut [f32]) {
        if probs.is_empty() || probs.len() != self.candidates.len() {
            return;
        }
        let mut groups = BTreeMap::<String, Vec<usize>>::new();
        for (index, candidate) in self.candidates.iter().enumerate() {
            let Some(contract) = candidate_answer_contract(candidate) else {
                continue;
            };
            groups.entry(contract).or_default().push(index);
        }
        if groups.len() <= 1 {
            return;
        }
        let group_uniform = 1.0 / groups.len() as f32;
        let cap = self
            .config
            .max_answer_contract_probability
            .is_finite()
            .then_some(self.config.max_answer_contract_probability)
            .filter(|configured_cap| *configured_cap < 1.0)
            .map(|configured_cap| configured_cap.clamp(0.0, 1.0).max(group_uniform));
        let floor = self
            .config
            .min_answer_contract_probability
            .is_finite()
            .then_some(self.config.min_answer_contract_probability)
            .filter(|configured_floor| *configured_floor > 0.0)
            .map(|configured_floor| {
                configured_floor
                    .clamp(0.0, group_uniform)
                    .min(cap.unwrap_or(1.0))
            });
        if cap.is_none() && floor.is_none() {
            return;
        }
        normalize_distribution(probs);

        let mut group_masses = BTreeMap::<String, f32>::new();
        for (label, indices) in &groups {
            group_masses.insert(
                label.clone(),
                indices.iter().map(|index| probs[*index].max(0.0)).sum(),
            );
        }

        if let Some(cap) = cap {
            let mut excess = 0.0f32;
            for (label, indices) in &groups {
                let mass = group_masses.get(label).copied().unwrap_or_default();
                if mass <= cap || mass <= f32::EPSILON {
                    continue;
                }
                let scale = cap / mass;
                for index in indices {
                    probs[*index] *= scale;
                }
                group_masses.insert(label.clone(), cap);
                excess += mass - cap;
            }
            if excess > f32::EPSILON {
                let headroom = group_masses
                    .values()
                    .map(|mass| (cap - *mass).max(0.0))
                    .sum::<f32>();
                if headroom > f32::EPSILON {
                    for (label, indices) in &groups {
                        let mass = group_masses.get(label).copied().unwrap_or_default();
                        let room = (cap - mass).max(0.0);
                        if room <= f32::EPSILON {
                            continue;
                        }
                        let add = excess * room / headroom;
                        distribute_probability_addition(probs, &self.candidates, indices, add);
                    }
                }
            }
            normalize_distribution(probs);
        }
        if let Some(floor) = floor {
            enforce_answer_contract_floor(probs, &self.candidates, &groups, floor);
        }
        normalize_distribution(probs);
    }

    fn enforce_capability_frontier(&self, probs: &mut [f32]) {
        if probs.is_empty() || self.capability_posteriors.is_empty() {
            return;
        }
        let max_difficulty = self.max_difficulty_level();
        let Some(allowed_max) = self.capability_frontier_allowed_max_difficulty() else {
            return;
        };
        if allowed_max >= max_difficulty {
            return;
        }
        let max_unverified_probability = self
            .config
            .capability_frontier_max_unverified_probability
            .clamp(0.0, 1.0);
        if max_unverified_probability >= 1.0 {
            return;
        }
        let blocked_mass = probs
            .iter()
            .zip(&self.candidates)
            .filter_map(|(prob, candidate)| {
                (candidate.difficulty_level > allowed_max).then_some(*prob)
            })
            .sum::<f32>();
        if blocked_mass <= max_unverified_probability {
            return;
        }
        let allowed_distribution = self.allowed_capability_frontier_distribution(allowed_max);
        if allowed_distribution.is_empty() {
            return;
        }
        let excess = blocked_mass - max_unverified_probability;
        let blocked_scale = if blocked_mass <= f32::EPSILON {
            0.0
        } else {
            max_unverified_probability / blocked_mass
        };
        for (index, prob) in probs.iter_mut().enumerate() {
            if self.candidates[index].difficulty_level > allowed_max {
                *prob *= blocked_scale;
            } else {
                *prob += excess * allowed_distribution[index];
            }
        }
        normalize_distribution(probs);
    }

    fn capability_frontier_allowed_max_difficulty(&self) -> Option<usize> {
        let mut iter = self
            .capability_coverage_by_difficulty()
            .into_iter()
            .map(|coverage| {
                (
                    coverage.difficulty_level,
                    coverage.mastered(self.config.capability_frontier_min_coverage),
                )
            });
        let (min_difficulty, mut backed_mastered) = iter.next()?;
        let mut base = min_difficulty;
        if backed_mastered {
            base = min_difficulty;
            for (difficulty, mastered) in iter {
                backed_mastered = mastered;
                if !backed_mastered {
                    break;
                }
                base = difficulty;
            }
        }
        Some(base.saturating_add(self.config.capability_frontier_max_ahead))
    }

    fn capability_coverage_by_difficulty(&self) -> Vec<RuliadCapabilityCoverage> {
        let mut levels = BTreeMap::<usize, Vec<&RuliadSamplerCandidate>>::new();
        for candidate in &self.candidates {
            if !candidate.is_hash_noise {
                levels
                    .entry(candidate.difficulty_level)
                    .or_default()
                    .push(candidate);
            }
        }
        levels
            .into_iter()
            .map(|(difficulty_level, candidates)| {
                let mastered = candidates
                    .iter()
                    .filter(|candidate| {
                        self.capability_posteriors
                            .get(&candidate.oracle_hash)
                            .is_some_and(|posterior| {
                                posterior.mastered(self.config.capability_mastery)
                            })
                    })
                    .map(|candidate| candidate.oracle_hash.as_str())
                    .collect::<BTreeSet<_>>();
                let candidate_coverage = coverage_ratio(mastered.len(), candidates.len());
                let family_coverage =
                    grouped_mastery_coverage(&candidates, &mastered, |candidate| {
                        candidate.family.as_str()
                    });
                let task_coverage = grouped_mastery_coverage(&candidates, &mastered, |candidate| {
                    candidate.task_kind.as_str()
                });
                let contract_coverage =
                    grouped_mastery_coverage(&candidates, &mastered, |candidate| {
                        if candidate.answer_contract.is_empty() {
                            "untyped"
                        } else {
                            candidate.answer_contract.as_str()
                        }
                    });
                let observed_items = candidates
                    .iter()
                    .filter_map(|candidate| self.capability_posteriors.get(&candidate.oracle_hash))
                    .map(|posterior| posterior.observation_count())
                    .sum();
                RuliadCapabilityCoverage {
                    difficulty_level,
                    candidate_coverage,
                    family_coverage,
                    task_coverage,
                    contract_coverage,
                    observed_items,
                }
            })
            .collect()
    }

    fn allowed_capability_frontier_distribution(&self, allowed_max: usize) -> Vec<f32> {
        let mut weights = self
            .candidates
            .iter()
            .map(|candidate| {
                if candidate.difficulty_level <= allowed_max {
                    candidate.prior.max(1e-9) / candidate.cost.max(1e-6)
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        normalize_distribution(&mut weights);
        weights
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
                candidate_mastered(candidate, self.config.target_loss).then_some(*prob)
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
        let mut matched_any = false;
        for candidate in &mut self.candidates {
            if candidate_matches_capability_feedback(candidate, &feedback.group_label) {
                matched_any = true;
                record_capability_posterior(
                    self.capability_posteriors
                        .entry(candidate.oracle_hash.clone())
                        .or_default(),
                    feedback,
                );
                record_candidate_capability_feedback(candidate, feedback);
            }
        }
        if !matched_any {
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
        let active_candidate_count = probs
            .iter()
            .filter(|prob| **prob > ACTIVE_PROBABILITY_EPSILON)
            .count();
        let active_max_entropy_bits = if active_candidate_count > 1 {
            (active_candidate_count as f32).log2()
        } else {
            0.0
        };
        let normalized_sampler_entropy = if active_candidate_count == 0 {
            0.0
        } else if active_max_entropy_bits <= f32::EPSILON {
            1.0
        } else {
            (sampler_entropy_bits / active_max_entropy_bits).clamp(0.0, 1.0)
        };
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
                candidate_mastered(candidate, self.config.target_loss).then_some(*prob)
            })
            .sum::<f32>();
        let capability_summary = capability_probability_summary(probs, &self.candidates);
        let capability_frontier_coverage = self
            .capability_coverage_by_difficulty()
            .into_iter()
            .map(|coverage| RuliadCapabilityCoverageMetric {
                difficulty_level: coverage.difficulty_level,
                candidate_coverage: coverage.candidate_coverage,
                family_coverage: coverage.family_coverage,
                task_coverage: coverage.task_coverage,
                contract_coverage: coverage.contract_coverage,
                observed_items: coverage.observed_items,
                mastered: coverage.mastered(self.config.capability_frontier_min_coverage),
            })
            .collect::<Vec<_>>();
        let capability_frontier_allowed_max_difficulty = self
            .capability_frontier_allowed_max_difficulty()
            .unwrap_or_else(|| {
                self.candidates
                    .iter()
                    .map(|candidate| candidate.difficulty_level)
                    .min()
                    .unwrap_or(0)
            });
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
        let contract_buckets = group_metrics_by(
            probs,
            &self.candidates,
            self.config.target_loss,
            |candidate| {
                candidate_answer_contract(candidate).unwrap_or_else(|| "unknown".to_string())
            },
        );
        RuliadMetricSnapshot {
            sample_count: self.candidates.len(),
            verifier_failures: self.verifier_failures,
            sampler_entropy_bits,
            active_candidate_count,
            active_max_entropy_bits,
            normalized_sampler_entropy,
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
            capability_feedback_probability: capability_summary.feedback_probability,
            capability_verifier_ema: capability_summary.verifier,
            capability_completion_health_ema: capability_summary.completion_health,
            capability_schema_wrong_ema: capability_summary.schema_wrong,
            capability_malformed_ema: capability_summary.malformed,
            capability_missing_ema: capability_summary.missing,
            capability_lagging_probability: capability_summary.lagging_probability,
            capability_frontier_allowed_max_difficulty,
            capability_frontier_coverage,
            frontier_extension_count: 0,
            frontier_saturated: false,
            frontier_unbounded: false,
            top_buckets,
            difficulty_buckets,
            family_buckets,
            task_buckets,
            contract_buckets,
        }
    }
}

fn difficulty_gate(loss_ema: f32, target_loss: f32) -> f32 {
    1.0 / (1.0 + (loss_ema - target_loss).abs())
}

fn candidate_mastered(candidate: &RuliadSamplerCandidate, target_loss: f32) -> bool {
    candidate.loss_ema <= target_loss && candidate_capability_mastered(candidate)
}

fn candidate_capability_mastered(candidate: &RuliadSamplerCandidate) -> bool {
    if candidate.capability_feedback_count == 0 {
        return false;
    }
    candidate.capability_verifier_ema >= 0.50
        && candidate.capability_completion_health_ema >= 0.75
        && candidate.capability_schema_wrong_ema <= 0.25
        && candidate.capability_malformed_ema <= 0.05
        && candidate.capability_missing_ema <= 0.05
}

fn candidate_capability_lagging(candidate: &RuliadSamplerCandidate) -> bool {
    candidate.capability_feedback_count > 0
        && !candidate_capability_mastered(candidate)
        && (candidate.capability_verifier_ema < 0.05
            || candidate.capability_completion_health_ema < 0.60
            || candidate.capability_schema_wrong_ema > 0.25
            || candidate.capability_malformed_ema > 0.05
            || candidate.capability_missing_ema > 0.05)
}

fn record_candidate_capability_feedback(
    candidate: &mut RuliadSamplerCandidate,
    feedback: &RuliadCapabilityFeedback,
) {
    let alpha = if candidate.capability_feedback_count == 0 {
        1.0
    } else {
        0.25
    };
    candidate.capability_feedback_count = candidate.capability_feedback_count.saturating_add(1);
    candidate.capability_verifier_ema = ema_update(
        candidate.capability_verifier_ema,
        feedback.verifier_rate.clamp(0.0, 1.0),
        alpha,
    );
    candidate.capability_partial_ema = ema_update(
        candidate.capability_partial_ema,
        feedback.partial_credit_rate.clamp(0.0, 1.0),
        alpha,
    );
    candidate.capability_completion_health_ema = ema_update(
        candidate.capability_completion_health_ema,
        feedback.completion_health_rate.clamp(0.0, 1.0),
        alpha,
    );
    candidate.capability_schema_wrong_ema = ema_update(
        candidate.capability_schema_wrong_ema,
        feedback.schema_valid_wrong_rate.clamp(0.0, 1.0),
        alpha,
    );
    candidate.capability_malformed_ema = ema_update(
        candidate.capability_malformed_ema,
        feedback.malformed_rate.clamp(0.0, 1.0),
        alpha,
    );
    candidate.capability_missing_ema = ema_update(
        candidate.capability_missing_ema,
        feedback.missing_rate.clamp(0.0, 1.0),
        alpha,
    );
}

fn record_capability_posterior(
    posterior: &mut RuliadCapabilityPosterior,
    feedback: &RuliadCapabilityFeedback,
) {
    posterior
        .verifier
        .observe_rate(feedback.verifier_rate, feedback.item_count);
    posterior
        .partial_credit
        .observe_rate(feedback.partial_credit_rate, feedback.item_count);
    posterior
        .completion_health
        .observe_rate(feedback.completion_health_rate, feedback.item_count);
    posterior
        .schema_wrong
        .observe_rate(feedback.schema_valid_wrong_rate, feedback.item_count);
    posterior
        .malformed
        .observe_rate(feedback.malformed_rate, feedback.item_count);
    posterior
        .missing
        .observe_rate(feedback.missing_rate, feedback.item_count);
}

fn coverage_ratio(mastered: usize, total: usize) -> f32 {
    if total == 0 {
        1.0
    } else {
        mastered as f32 / total as f32
    }
}

fn grouped_mastery_coverage<'a, F>(
    candidates: &[&'a RuliadSamplerCandidate],
    mastered: &BTreeSet<&str>,
    key: F,
) -> f32
where
    F: Fn(&'a RuliadSamplerCandidate) -> &'a str,
{
    let all_groups = candidates
        .iter()
        .map(|candidate| key(candidate))
        .collect::<BTreeSet<_>>();
    let mastered_groups = candidates
        .iter()
        .filter(|candidate| mastered.contains(candidate.oracle_hash.as_str()))
        .map(|candidate| key(candidate))
        .collect::<BTreeSet<_>>();
    coverage_ratio(mastered_groups.len(), all_groups.len())
}

fn ema_update(previous: f32, value: f32, alpha: f32) -> f32 {
    previous * (1.0 - alpha) + value * alpha
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
    if feedback.verifier_rate < 0.05 {
        return false;
    }
    let structural_error = capability_feedback_structural_error(feedback);
    if capability_feedback_is_difficulty_group(feedback) {
        return structural_error <= 0.35;
    }
    capability_feedback_is_remediation_group(feedback) && structural_error <= 0.50
}

fn capability_feedback_is_remediation_group(feedback: &RuliadCapabilityFeedback) -> bool {
    feedback.group_label.starts_with("family:")
        || feedback.group_label.starts_with("task:")
        || feedback.group_label.starts_with("contract:")
        || feedback.group_label.starts_with("domain:")
        || feedback.group_label.starts_with("math_domain:")
        || feedback.group_label.starts_with("mode:")
        || feedback.group_label.starts_with("reasoning_mode:")
        || feedback.group_label.starts_with("bucket:")
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
    if group_label.starts_with(RULIAD_SOURCE_CAPABILITY_LABEL_PREFIX) {
        let Some(answer_contract) = candidate_answer_contract(candidate) else {
            return false;
        };
        return ruliad_source_capability_label(
            &candidate.family,
            &candidate.task_kind,
            candidate.difficulty_level,
            &answer_contract,
        ) == group_label;
    }
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
    if let Some(contract) = group_label.strip_prefix("contract:") {
        return candidate_answer_contract(candidate).as_deref() == Some(contract);
    }
    if let Some(domain) = group_label
        .strip_prefix("domain:")
        .or_else(|| group_label.strip_prefix("math_domain:"))
    {
        return candidate_matches_math_domain(candidate, domain);
    }
    if let Some(mode) = group_label
        .strip_prefix("mode:")
        .or_else(|| group_label.strip_prefix("reasoning_mode:"))
    {
        return candidate_matches_reasoning_mode(candidate, mode);
    }
    candidate.oracle_hash == group_label
}

fn candidate_matches_math_domain(candidate: &RuliadSamplerCandidate, domain: &str) -> bool {
    let Some((family, task_kind)) = candidate_family_and_task(candidate) else {
        return false;
    };
    ruliad_source_semantics(family, task_kind)
        .math_domains
        .iter()
        .any(|candidate_domain| candidate_domain.label() == domain)
}

fn candidate_matches_reasoning_mode(candidate: &RuliadSamplerCandidate, mode: &str) -> bool {
    let Some((family, task_kind)) = candidate_family_and_task(candidate) else {
        return false;
    };
    ruliad_source_semantics(family, task_kind)
        .reasoning_modes
        .iter()
        .any(|candidate_mode| candidate_mode.label() == mode)
}

#[derive(Debug, Clone, Copy, Default)]
struct CapabilityProbabilitySummary {
    feedback_probability: f32,
    verifier: f32,
    completion_health: f32,
    schema_wrong: f32,
    malformed: f32,
    missing: f32,
    lagging_probability: f32,
}

fn capability_probability_summary(
    probs: &[f32],
    candidates: &[RuliadSamplerCandidate],
) -> CapabilityProbabilitySummary {
    let mut summary = CapabilityProbabilitySummary::default();
    for (probability, candidate) in probs.iter().zip(candidates) {
        if candidate.capability_feedback_count == 0 {
            continue;
        }
        let probability = probability.max(0.0);
        summary.feedback_probability += probability;
        summary.verifier += probability * candidate.capability_verifier_ema.clamp(0.0, 1.0);
        summary.completion_health +=
            probability * candidate.capability_completion_health_ema.clamp(0.0, 1.0);
        summary.schema_wrong += probability * candidate.capability_schema_wrong_ema.clamp(0.0, 1.0);
        summary.malformed += probability * candidate.capability_malformed_ema.clamp(0.0, 1.0);
        summary.missing += probability * candidate.capability_missing_ema.clamp(0.0, 1.0);
        if candidate_capability_lagging(candidate) {
            summary.lagging_probability += probability;
        }
    }
    if summary.feedback_probability > f32::EPSILON {
        let denominator = summary.feedback_probability;
        summary.verifier /= denominator;
        summary.completion_health /= denominator;
        summary.schema_wrong /= denominator;
        summary.malformed /= denominator;
        summary.missing /= denominator;
    }
    summary
}

fn candidate_answer_contract(candidate: &RuliadSamplerCandidate) -> Option<String> {
    if !candidate.answer_contract.is_empty() {
        return Some(candidate.answer_contract.clone());
    }
    let (family, task_kind) = candidate_family_and_task(candidate)?;
    source_answer_contract(
        family,
        task_kind,
        RuliadProofActionAnswerContract::PresentationIndex,
    )
    .map(str::to_string)
}

pub(crate) fn source_answer_contract(
    family: RuliadFamilyKind,
    task_kind: RuliadTaskKind,
    proof_action_contract: RuliadProofActionAnswerContract,
) -> Option<&'static str> {
    Some(match (family, task_kind) {
        (RuliadFamilyKind::Eca, _) => "xlen,xalpha,xcounts,xedge",
        (RuliadFamilyKind::Simulation, RuliadTaskKind::VerifySimulation) => "ok",
        (RuliadFamilyKind::Automaton, RuliadTaskKind::EvaluateAutomaton) => "acc",
        (RuliadFamilyKind::Rewrite, RuliadTaskKind::RewriteNormalForm) => {
            "nflen,nfalpha,nfcounts,nfedge"
        }
        (RuliadFamilyKind::Algebra, RuliadTaskKind::CheckAlgebraLaw) => "ok",
        (RuliadFamilyKind::Category, _) | (RuliadFamilyKind::ProofTree, _) => "ok,l,r",
        (RuliadFamilyKind::FormalProof, RuliadTaskKind::AdvanceProof) => "proof_step",
        (RuliadFamilyKind::FormalProof, RuliadTaskKind::SelectProofAction) => {
            match proof_action_contract {
                RuliadProofActionAnswerContract::PresentationIndex => "action_index",
                RuliadProofActionAnswerContract::SemanticStep => "proof_action_step",
            }
        }
        (RuliadFamilyKind::FormalProof, RuliadTaskKind::ConstructProof) => "certificate",
        (RuliadFamilyKind::FormalProof, RuliadTaskKind::CheckProof) => "ok,vg,vs,g,s,k",
        (RuliadFamilyKind::LeanTask, RuliadTaskKind::CompleteProof)
        | (RuliadFamilyKind::HashNoise, RuliadTaskKind::HashCanary) => "sha",
        _ => return None,
    })
}

fn candidate_family_and_task(
    candidate: &RuliadSamplerCandidate,
) -> Option<(RuliadFamilyKind, RuliadTaskKind)> {
    Some((
        RuliadFamilyKind::from_label(&candidate.family)?,
        RuliadTaskKind::from_label(&candidate.task_kind)?,
    ))
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
            mastered: candidate_mastered(candidate, target_loss),
            capability_feedback_count: candidate.capability_feedback_count,
            capability_verifier_ema: candidate.capability_verifier_ema,
            capability_completion_health_ema: candidate.capability_completion_health_ema,
            capability_schema_wrong_ema: candidate.capability_schema_wrong_ema,
            capability_malformed_ema: candidate.capability_malformed_ema,
            capability_missing_ema: candidate.capability_missing_ema,
            capability_lagging: candidate_capability_lagging(candidate),
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
        capability_feedback_probability: f32,
        capability_verifier: f32,
        capability_completion_health: f32,
        capability_schema_wrong: f32,
        capability_malformed: f32,
        capability_missing: f32,
        capability_lagging_probability: f32,
    }

    let mut groups = BTreeMap::<String, Accumulator>::new();
    for (probability, candidate) in probs.iter().zip(candidates) {
        let group = groups.entry(label(candidate)).or_default();
        group.candidate_count += 1;
        group.probability += *probability;
        group.weighted_loss += *probability * candidate.loss_ema;
        group.learning_progress +=
            *probability * (candidate.previous_loss_ema - candidate.loss_ema).max(0.0);
        if candidate_mastered(candidate, target_loss) {
            group.mastered_probability += *probability;
        }
        group.weighted_difficulty += *probability * candidate.difficulty_level as f32;
        if candidate.capability_feedback_count > 0 {
            group.capability_feedback_probability += *probability;
            group.capability_verifier +=
                *probability * candidate.capability_verifier_ema.clamp(0.0, 1.0);
            group.capability_completion_health +=
                *probability * candidate.capability_completion_health_ema.clamp(0.0, 1.0);
            group.capability_schema_wrong +=
                *probability * candidate.capability_schema_wrong_ema.clamp(0.0, 1.0);
            group.capability_malformed +=
                *probability * candidate.capability_malformed_ema.clamp(0.0, 1.0);
            group.capability_missing +=
                *probability * candidate.capability_missing_ema.clamp(0.0, 1.0);
            if candidate_capability_lagging(candidate) {
                group.capability_lagging_probability += *probability;
            }
        }
    }

    let mut metrics = groups
        .into_iter()
        .map(|(label, group)| {
            let probability = group.probability.max(0.0);
            let denominator = probability.max(1e-12);
            let capability_denominator = group.capability_feedback_probability.max(1e-12);
            RuliadGroupMetric {
                label,
                candidate_count: group.candidate_count,
                probability,
                mean_loss: group.weighted_loss / denominator,
                learning_progress: group.learning_progress / denominator,
                mastered_probability: group.mastered_probability,
                mean_difficulty_level: group.weighted_difficulty / denominator,
                capability_feedback_probability: group.capability_feedback_probability,
                capability_verifier_ema: group.capability_verifier / capability_denominator,
                capability_completion_health_ema: group.capability_completion_health
                    / capability_denominator,
                capability_schema_wrong_ema: group.capability_schema_wrong / capability_denominator,
                capability_malformed_ema: group.capability_malformed / capability_denominator,
                capability_missing_ema: group.capability_missing / capability_denominator,
                capability_lagging_probability: group.capability_lagging_probability,
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

fn distribute_probability_addition(
    probs: &mut [f32],
    candidates: &[RuliadSamplerCandidate],
    indices: &[usize],
    addition: f32,
) {
    if indices.is_empty() || addition <= f32::EPSILON {
        return;
    }
    let existing_mass = indices
        .iter()
        .map(|index| probs[*index].max(0.0))
        .sum::<f32>();
    if existing_mass > f32::EPSILON {
        for index in indices {
            probs[*index] += addition * probs[*index].max(0.0) / existing_mass;
        }
        return;
    }
    let prior_mass = indices
        .iter()
        .map(|index| {
            let candidate = &candidates[*index];
            candidate.prior.max(1e-9) / candidate.cost.max(1e-6)
        })
        .sum::<f32>()
        .max(1e-12);
    for index in indices {
        let candidate = &candidates[*index];
        let weight = candidate.prior.max(1e-9) / candidate.cost.max(1e-6);
        probs[*index] += addition * weight / prior_mass;
    }
}

fn enforce_answer_contract_floor(
    probs: &mut [f32],
    candidates: &[RuliadSamplerCandidate],
    groups: &BTreeMap<String, Vec<usize>>,
    floor: f32,
) {
    if floor <= f32::EPSILON || groups.is_empty() {
        return;
    }
    let mut group_masses = BTreeMap::<String, f32>::new();
    for (label, indices) in groups {
        group_masses.insert(
            label.clone(),
            indices.iter().map(|index| probs[*index].max(0.0)).sum(),
        );
    }
    let deficits = group_masses
        .iter()
        .filter_map(|(label, mass)| {
            let deficit = (floor - *mass).max(0.0);
            (deficit > f32::EPSILON).then_some((label.clone(), deficit))
        })
        .collect::<Vec<_>>();
    if deficits.is_empty() {
        return;
    }
    let deficit_total = deficits
        .iter()
        .map(|(_label, deficit)| *deficit)
        .sum::<f32>();
    let surplus_total = group_masses
        .values()
        .map(|mass| (*mass - floor).max(0.0))
        .sum::<f32>();
    let transfer = deficit_total.min(surplus_total);
    if transfer <= f32::EPSILON {
        return;
    }
    for (label, indices) in groups {
        let mass = group_masses.get(label).copied().unwrap_or_default();
        let surplus = (mass - floor).max(0.0);
        if surplus <= f32::EPSILON || mass <= f32::EPSILON {
            continue;
        }
        let removal = transfer * surplus / surplus_total.max(1.0e-12);
        let scale = ((mass - removal) / mass).clamp(0.0, 1.0);
        for index in indices {
            probs[*index] *= scale;
        }
    }
    for (label, deficit) in deficits {
        let Some(indices) = groups.get(&label) else {
            continue;
        };
        let addition = transfer * deficit / deficit_total.max(1.0e-12);
        distribute_probability_addition(probs, candidates, indices, addition);
    }
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

fn default_max_answer_contract_probability() -> f32 {
    1.0
}

fn default_min_answer_contract_probability() -> f32 {
    0.0
}

fn default_capability_frontier_max_ahead() -> usize {
    1
}

fn default_capability_frontier_max_unverified_probability() -> f32 {
    0.08
}

fn default_capability_remediation_weight() -> f32 {
    0.75
}

fn default_capability_frontier_min_coverage() -> f32 {
    1.0
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

    fn candidate_for_contract_test(
        oracle_hash: &str,
        family: &str,
        task_kind: &str,
    ) -> RuliadSamplerCandidate {
        RuliadSamplerCandidate {
            oracle_hash: oracle_hash.to_string(),
            family: family.to_string(),
            task_kind: task_kind.to_string(),
            answer_contract: String::new(),
            difficulty_level: 1,
            params_hash: String::new(),
            prior: 1.0,
            cost: 1.0,
            loss_ema: 4.0,
            previous_loss_ema: 4.0,
            gradient_alignment: 0.0,
            is_hash_noise: false,
            capability_feedback_count: 0,
            capability_verifier_ema: 0.0,
            capability_partial_ema: 0.0,
            capability_completion_health_ema: 0.0,
            capability_schema_wrong_ema: 0.0,
            capability_malformed_ema: 0.0,
            capability_missing_ema: 0.0,
        }
    }

    #[test]
    fn adding_existing_candidate_refreshes_semantic_metadata_only() {
        let mut restored = candidate_for_contract_test(
            "formal_proof:select_proof_action@d1#00000001",
            "formal_proof",
            "select_proof_action",
        );
        restored.loss_ema = 1.25;
        restored.capability_feedback_count = 7;
        let mut current = restored.clone();
        current.answer_contract = "proof_action_step".to_string();
        current.loss_ema = 9.0;
        current.capability_feedback_count = 0;
        let mut sampler =
            RuliadFrontierSampler::new(RuliadSamplerConfig::default(), vec![restored]);

        sampler.add_candidates([current]);

        assert_eq!(sampler.candidates().len(), 1);
        assert_eq!(sampler.candidates()[0].answer_contract, "proof_action_step");
        assert_eq!(sampler.candidates()[0].loss_ema, 1.25);
        assert_eq!(sampler.candidates()[0].capability_feedback_count, 7);
    }

    #[test]
    fn source_feedback_matches_joint_contract_and_difficulty_once() {
        let mut d1 = candidate_for_contract_test(
            "formal_proof:select_proof_action@d1#00000001",
            "formal_proof",
            "select_proof_action",
        );
        d1.answer_contract = "proof_action_step".to_string();
        let mut d2 = d1.clone();
        d2.oracle_hash = "formal_proof:select_proof_action@d2#00000002".to_string();
        d2.difficulty_level = 2;
        let mut legacy_contract = d1.clone();
        legacy_contract.oracle_hash = "formal_proof:select_proof_action@d1#00000003".to_string();
        legacy_contract.answer_contract = "action_index".to_string();
        let mut sampler = RuliadFrontierSampler::new(
            RuliadSamplerConfig::default(),
            vec![d1, d2, legacy_contract],
        );

        sampler.record_capability_feedback(&RuliadCapabilityFeedback {
            group_label: ruliad_source_capability_label(
                "formal_proof",
                "select_proof_action",
                1,
                "proof_action_step",
            ),
            item_count: 8,
            verifier_rate: 0.75,
            partial_credit_rate: 0.8,
            schema_valid_wrong_rate: 0.25,
            malformed_rate: 0.0,
            missing_rate: 0.0,
            completion_health_rate: 1.0,
        });

        assert_eq!(sampler.candidates()[0].capability_feedback_count, 1);
        assert_eq!(sampler.candidates()[1].capability_feedback_count, 0);
        assert_eq!(sampler.candidates()[2].capability_feedback_count, 0);
    }

    #[test]
    fn sampler_penalizes_hash_noise_canary() {
        let sampler = RuliadFrontierSampler::new(
            RuliadSamplerConfig::default(),
            vec![
                RuliadSamplerCandidate {
                    oracle_hash: "structured".to_string(),
                    family: "eca".to_string(),
                    task_kind: "multi_step_state".to_string(),
                    answer_contract: String::new(),
                    difficulty_level: 0,
                    params_hash: String::new(),
                    prior: 1.0,
                    cost: 1.0,
                    loss_ema: 2.0,
                    previous_loss_ema: 3.0,
                    gradient_alignment: 0.0,
                    is_hash_noise: false,
                    capability_feedback_count: 0,
                    capability_verifier_ema: 0.0,
                    capability_partial_ema: 0.0,
                    capability_completion_health_ema: 0.0,
                    capability_schema_wrong_ema: 0.0,
                    capability_malformed_ema: 0.0,
                    capability_missing_ema: 0.0,
                },
                RuliadSamplerCandidate {
                    oracle_hash: "noise".to_string(),
                    family: "hash_noise".to_string(),
                    task_kind: "hash_canary".to_string(),
                    answer_contract: String::new(),
                    difficulty_level: 0,
                    params_hash: String::new(),
                    prior: 1.0,
                    cost: 1.0,
                    loss_ema: 8.0,
                    previous_loss_ema: 8.0,
                    gradient_alignment: 0.0,
                    is_hash_noise: true,
                    capability_feedback_count: 0,
                    capability_verifier_ema: 0.0,
                    capability_partial_ema: 0.0,
                    capability_completion_health_ema: 0.0,
                    capability_schema_wrong_ema: 0.0,
                    capability_malformed_ema: 0.0,
                    capability_missing_ema: 0.0,
                },
            ],
        );
        let probs = sampler.probabilities();
        assert!(probs[0] > probs[1]);
        assert!(sampler.snapshot().hash_noise_probability < 0.5);
    }

    #[test]
    fn capability_feedback_targets_structured_wrong_without_loss_promotion() {
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
                    answer_contract: String::new(),
                    difficulty_level: 2,
                    params_hash: String::new(),
                    prior: 1.0,
                    cost: 1.0,
                    loss_ema: 8.0,
                    previous_loss_ema: 8.0,
                    gradient_alignment: 0.0,
                    is_hash_noise: false,
                    capability_feedback_count: 0,
                    capability_verifier_ema: 0.0,
                    capability_partial_ema: 0.0,
                    capability_completion_health_ema: 0.0,
                    capability_schema_wrong_ema: 0.0,
                    capability_malformed_ema: 0.0,
                    capability_missing_ema: 0.0,
                },
                RuliadSamplerCandidate {
                    oracle_hash: "proof_tree:prove_theorem@d2#00000002".to_string(),
                    family: "proof_tree".to_string(),
                    task_kind: "prove_theorem".to_string(),
                    answer_contract: String::new(),
                    difficulty_level: 2,
                    params_hash: String::new(),
                    prior: 1.0,
                    cost: 1.0,
                    loss_ema: 8.0,
                    previous_loss_ema: 8.0,
                    gradient_alignment: 0.0,
                    is_hash_noise: false,
                    capability_feedback_count: 0,
                    capability_verifier_ema: 0.0,
                    capability_partial_ema: 0.0,
                    capability_completion_health_ema: 0.0,
                    capability_schema_wrong_ema: 0.0,
                    capability_malformed_ema: 0.0,
                    capability_missing_ema: 0.0,
                },
            ],
        );
        let before = sampler.snapshot();

        sampler.record_capability_feedback(&RuliadCapabilityFeedback {
            group_label: "family:category".to_string(),
            item_count: 16,
            verifier_rate: 0.0,
            partial_credit_rate: 0.25,
            schema_valid_wrong_rate: 0.80,
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
            category.probability > proof.probability,
            "schema-valid wrong feedback should target the lagging family without requiring verifier signal"
        );
        assert_eq!(category.learning_progress, 0.0);
        assert_eq!(category.loss_ema, 8.0);
        assert!(!category.mastered);
        assert_eq!(category.capability_feedback_count, 1);
        assert_eq!(category.capability_verifier_ema, 0.0);
        assert!(category.capability_completion_health_ema >= 0.95);
        assert!(category.capability_schema_wrong_ema >= 0.80);
        assert!(category.capability_lagging);
        assert!(snapshot.capability_feedback_probability > 0.0);
        assert!(snapshot.capability_lagging_probability > 0.0);
        assert_eq!(snapshot.capability_verifier_ema, 0.0);
        assert!(snapshot.capability_completion_health_ema >= 0.95);
        assert!(snapshot.capability_schema_wrong_ema >= 0.80);
        let category_group = snapshot
            .family_buckets
            .iter()
            .find(|group| group.label == "category")
            .expect("category family group");
        assert!(category_group.capability_feedback_probability > 0.0);
        assert!(category_group.capability_lagging_probability > 0.0);
        assert!(category_group.capability_schema_wrong_ema >= 0.80);
        assert_eq!(category_group.capability_verifier_ema, 0.0);
        assert_eq!(snapshot.mean_difficulty_level, before.mean_difficulty_level);
    }

    #[test]
    fn malformed_capability_feedback_does_not_create_remediation_pressure() {
        let mut sampler = RuliadFrontierSampler::new(
            RuliadSamplerConfig {
                exploration_floor: 0.0,
                target_loss: 2.0,
                ..RuliadSamplerConfig::default()
            },
            vec![
                RuliadSamplerCandidate {
                    oracle_hash: "category:verify_category_law@d1#00000001".to_string(),
                    family: "category".to_string(),
                    task_kind: "verify_category_law".to_string(),
                    answer_contract: String::new(),
                    difficulty_level: 1,
                    params_hash: String::new(),
                    prior: 1.0,
                    cost: 1.0,
                    loss_ema: 2.0,
                    previous_loss_ema: 2.0,
                    gradient_alignment: 0.0,
                    is_hash_noise: false,
                    capability_feedback_count: 0,
                    capability_verifier_ema: 0.0,
                    capability_partial_ema: 0.0,
                    capability_completion_health_ema: 0.0,
                    capability_schema_wrong_ema: 0.0,
                    capability_malformed_ema: 0.0,
                    capability_missing_ema: 0.0,
                },
                RuliadSamplerCandidate {
                    oracle_hash: "proof_tree:prove_theorem@d1#00000002".to_string(),
                    family: "proof_tree".to_string(),
                    task_kind: "prove_theorem".to_string(),
                    answer_contract: String::new(),
                    difficulty_level: 1,
                    params_hash: String::new(),
                    prior: 1.0,
                    cost: 1.0,
                    loss_ema: 2.0,
                    previous_loss_ema: 2.0,
                    gradient_alignment: 0.0,
                    is_hash_noise: false,
                    capability_feedback_count: 0,
                    capability_verifier_ema: 0.0,
                    capability_partial_ema: 0.0,
                    capability_completion_health_ema: 0.0,
                    capability_schema_wrong_ema: 0.0,
                    capability_malformed_ema: 0.0,
                    capability_missing_ema: 0.0,
                },
            ],
        );

        sampler.record_capability_feedback(&RuliadCapabilityFeedback {
            group_label: "task:verify_category_law".to_string(),
            item_count: 16,
            verifier_rate: 0.0,
            partial_credit_rate: 0.0,
            schema_valid_wrong_rate: 0.20,
            malformed_rate: 0.80,
            missing_rate: 0.40,
            completion_health_rate: 0.10,
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
        let candidate = sampler.candidates().first().expect("candidate");

        assert_eq!(candidate.cost, 1.0);
        assert_eq!(candidate.loss_ema, 2.0);
        assert_eq!(candidate.previous_loss_ema, 2.0);
        assert!(candidate.capability_malformed_ema >= 0.80);
        assert_eq!(category.capability_feedback_count, 1);
        assert_eq!(category.capability_verifier_ema, 0.0);
        assert!(category.capability_malformed_ema >= 0.80);
        assert!(category.capability_missing_ema >= 0.40);
        assert!(category.capability_lagging);
        assert!(snapshot.capability_feedback_probability > 0.0);
        assert!(snapshot.capability_lagging_probability > 0.0);
        assert_eq!(snapshot.capability_verifier_ema, 0.0);
        assert!(snapshot.capability_malformed_ema >= 0.80);
        assert!(snapshot.capability_missing_ema >= 0.40);
        assert!(
            category.probability <= proof.probability + 0.001,
            "malformed feedback should be tracked but not receive sampling pressure"
        );
    }

    #[test]
    fn healthy_family_feedback_can_update_matching_candidate_loss() {
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
                answer_contract: String::new(),
                difficulty_level: 1,
                params_hash: String::new(),
                prior: 1.0,
                cost: 1.0,
                loss_ema: 8.0,
                previous_loss_ema: 8.0,
                gradient_alignment: 0.0,
                is_hash_noise: false,
                capability_feedback_count: 0,
                capability_verifier_ema: 0.0,
                capability_partial_ema: 0.0,
                capability_completion_health_ema: 0.0,
                capability_schema_wrong_ema: 0.0,
                capability_malformed_ema: 0.0,
                capability_missing_ema: 0.0,
            }],
        );

        sampler.record_capability_feedback(&RuliadCapabilityFeedback {
            group_label: "family:category".to_string(),
            item_count: 16,
            verifier_rate: 1.0,
            partial_credit_rate: 1.0,
            schema_valid_wrong_rate: 0.0,
            malformed_rate: 0.0,
            missing_rate: 0.0,
            completion_health_rate: 1.0,
        });
        let candidate = sampler.candidates().first().expect("candidate");

        assert!(
            candidate.loss_ema < 8.0,
            "healthy family feedback should update the matching curriculum candidate"
        );
        assert!(candidate.capability_verifier_ema >= 1.0);
    }

    #[test]
    fn domain_and_mode_feedback_match_source_semantics() {
        let mut sampler = RuliadFrontierSampler::new(
            RuliadSamplerConfig {
                exploration_floor: 0.0,
                target_loss: 2.0,
                ..RuliadSamplerConfig::default()
            },
            vec![
                RuliadSamplerCandidate {
                    oracle_hash: "category:verify_category_law@d1#00000001".to_string(),
                    family: "category".to_string(),
                    task_kind: "verify_category_law".to_string(),
                    answer_contract: String::new(),
                    difficulty_level: 1,
                    params_hash: String::new(),
                    prior: 1.0,
                    cost: 1.0,
                    loss_ema: 2.0,
                    previous_loss_ema: 2.0,
                    gradient_alignment: 0.0,
                    is_hash_noise: false,
                    capability_feedback_count: 0,
                    capability_verifier_ema: 0.0,
                    capability_partial_ema: 0.0,
                    capability_completion_health_ema: 0.0,
                    capability_schema_wrong_ema: 0.0,
                    capability_malformed_ema: 0.0,
                    capability_missing_ema: 0.0,
                },
                RuliadSamplerCandidate {
                    oracle_hash: "rewrite:rewrite_normal_form@d1#00000002".to_string(),
                    family: "rewrite".to_string(),
                    task_kind: "rewrite_normal_form".to_string(),
                    answer_contract: String::new(),
                    difficulty_level: 1,
                    params_hash: String::new(),
                    prior: 1.0,
                    cost: 1.0,
                    loss_ema: 2.0,
                    previous_loss_ema: 2.0,
                    gradient_alignment: 0.0,
                    is_hash_noise: false,
                    capability_feedback_count: 0,
                    capability_verifier_ema: 0.0,
                    capability_partial_ema: 0.0,
                    capability_completion_health_ema: 0.0,
                    capability_schema_wrong_ema: 0.0,
                    capability_malformed_ema: 0.0,
                    capability_missing_ema: 0.0,
                },
            ],
        );

        sampler.record_capability_feedback(&RuliadCapabilityFeedback {
            group_label: "domain:category_theory".to_string(),
            item_count: 16,
            verifier_rate: 0.0,
            partial_credit_rate: 0.0,
            schema_valid_wrong_rate: 0.60,
            malformed_rate: 0.0,
            missing_rate: 0.0,
            completion_health_rate: 0.90,
        });
        assert_eq!(sampler.candidates()[0].capability_feedback_count, 1);
        assert_eq!(sampler.candidates()[1].capability_feedback_count, 0);

        sampler.record_capability_feedback(&RuliadCapabilityFeedback {
            group_label: "mode:normalization".to_string(),
            item_count: 16,
            verifier_rate: 0.0,
            partial_credit_rate: 0.0,
            schema_valid_wrong_rate: 0.60,
            malformed_rate: 0.0,
            missing_rate: 0.0,
            completion_health_rate: 0.90,
        });
        assert_eq!(sampler.candidates()[0].capability_feedback_count, 1);
        assert_eq!(sampler.candidates()[1].capability_feedback_count, 1);
    }

    #[test]
    fn contract_feedback_matches_answer_schema_not_family_only() {
        let mut sampler = RuliadFrontierSampler::new(
            RuliadSamplerConfig {
                exploration_floor: 0.0,
                target_loss: 2.0,
                capability_remediation_weight: 1.0,
                ..RuliadSamplerConfig::default()
            },
            vec![
                candidate_for_contract_test(
                    "category:verify_category_law@d1#00000001",
                    "category",
                    "verify_category_law",
                ),
                candidate_for_contract_test(
                    "proof_tree:prove_theorem@d1#00000002",
                    "proof_tree",
                    "prove_theorem",
                ),
                candidate_for_contract_test(
                    "automaton:evaluate_automaton@d1#00000003",
                    "automaton",
                    "evaluate_automaton",
                ),
            ],
        );

        sampler.record_capability_feedback(&RuliadCapabilityFeedback {
            group_label: "contract:ok,l,r".to_string(),
            item_count: 16,
            verifier_rate: 0.75,
            partial_credit_rate: 0.85,
            schema_valid_wrong_rate: 0.10,
            malformed_rate: 0.0,
            missing_rate: 0.0,
            completion_health_rate: 0.95,
        });

        assert_eq!(
            candidate_answer_contract(&sampler.candidates()[0]).as_deref(),
            Some("ok,l,r")
        );
        assert_eq!(
            candidate_answer_contract(&sampler.candidates()[1]).as_deref(),
            Some("ok,l,r")
        );
        assert_eq!(
            candidate_answer_contract(&sampler.candidates()[2]).as_deref(),
            Some("acc")
        );
        assert_eq!(sampler.candidates()[0].capability_feedback_count, 1);
        assert_eq!(sampler.candidates()[1].capability_feedback_count, 1);
        assert_eq!(sampler.candidates()[2].capability_feedback_count, 0);
        assert!(
            sampler.candidates()[0].loss_ema < 4.0 && sampler.candidates()[1].loss_ema < 4.0,
            "contract feedback should promote matching ok/l/r candidates"
        );
        assert_eq!(sampler.candidates()[2].loss_ema, 4.0);
    }

    #[test]
    fn formal_proof_tasks_have_distinct_answer_contracts() {
        let advance = candidate_for_contract_test(
            "formal_proof:advance_proof@d0#00000000",
            "formal_proof",
            "advance_proof",
        );
        let construct = candidate_for_contract_test(
            "formal_proof:construct_proof@d0#00000001",
            "formal_proof",
            "construct_proof",
        );
        assert_eq!(
            candidate_answer_contract(&advance).as_deref(),
            Some("proof_step")
        );
        let check = candidate_for_contract_test(
            "formal_proof:check_proof@d0#00000002",
            "formal_proof",
            "check_proof",
        );
        assert_eq!(
            candidate_answer_contract(&construct).as_deref(),
            Some("certificate")
        );
        assert_eq!(
            candidate_answer_contract(&check).as_deref(),
            Some("ok,vg,vs,g,s,k")
        );

        let sampler = RuliadFrontierSampler::new(
            RuliadSamplerConfig {
                exploration_floor: 0.0,
                max_answer_contract_probability: 0.70,
                min_answer_contract_probability: 0.30,
                ..RuliadSamplerConfig::default()
            },
            vec![advance, construct, check],
        );
        let probabilities = sampler.probabilities();
        assert!(probabilities.iter().all(|probability| *probability >= 0.30));
    }

    #[test]
    fn sampler_caps_dominant_answer_contract_probability() {
        let mut category = candidate_for_contract_test(
            "category:verify_category_law@d1#00000001",
            "category",
            "verify_category_law",
        );
        category.prior = 20.0;
        let mut proof = candidate_for_contract_test(
            "proof_tree:prove_theorem@d1#00000002",
            "proof_tree",
            "prove_theorem",
        );
        proof.prior = 20.0;
        let sampler = RuliadFrontierSampler::new(
            RuliadSamplerConfig {
                exploration_floor: 0.0,
                max_answer_contract_probability: 0.40,
                ..RuliadSamplerConfig::default()
            },
            vec![
                category,
                proof,
                candidate_for_contract_test(
                    "automaton:evaluate_automaton@d1#00000003",
                    "automaton",
                    "evaluate_automaton",
                ),
                candidate_for_contract_test(
                    "eca:multi_step_state@d1#00000004",
                    "eca",
                    "multi_step_state",
                ),
                candidate_for_contract_test(
                    "rewrite:rewrite_normal_form@d1#00000005",
                    "rewrite",
                    "rewrite_normal_form",
                ),
            ],
        );

        let snapshot = sampler.snapshot();
        let ok_lr_probability = snapshot
            .contract_buckets
            .iter()
            .find(|bucket| bucket.label == "ok,l,r")
            .map(|bucket| bucket.probability)
            .unwrap_or_default();
        assert!(
            ok_lr_probability <= 0.4001,
            "dominant ok/l/r contract should be capped: {ok_lr_probability}"
        );
        assert!(
            snapshot.contract_buckets.len() >= 4,
            "expected multiple answer contracts in snapshot: {:?}",
            snapshot.contract_buckets
        );
    }

    #[test]
    fn sampler_floors_starved_answer_contract_probability() {
        let mut category = candidate_for_contract_test(
            "category:verify_category_law@d1#00000001",
            "category",
            "verify_category_law",
        );
        category.prior = 50.0;
        let sampler = RuliadFrontierSampler::new(
            RuliadSamplerConfig {
                exploration_floor: 0.0,
                max_answer_contract_probability: 0.50,
                min_answer_contract_probability: 0.12,
                ..RuliadSamplerConfig::default()
            },
            vec![
                category,
                candidate_for_contract_test(
                    "automaton:evaluate_automaton@d1#00000003",
                    "automaton",
                    "evaluate_automaton",
                ),
                candidate_for_contract_test(
                    "eca:multi_step_state@d1#00000004",
                    "eca",
                    "multi_step_state",
                ),
                candidate_for_contract_test(
                    "rewrite:rewrite_normal_form@d1#00000005",
                    "rewrite",
                    "rewrite_normal_form",
                ),
            ],
        );

        let snapshot = sampler.snapshot();
        for contract in [
            "acc",
            "xlen,xalpha,xcounts,xedge",
            "nflen,nfalpha,nfcounts,nfedge",
        ] {
            let probability = snapshot
                .contract_buckets
                .iter()
                .find(|bucket| bucket.label == contract)
                .map(|bucket| bucket.probability)
                .unwrap_or_default();
            assert!(
                probability >= 0.119,
                "contract {contract} should receive the configured floor: {probability}"
            );
        }
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
                    answer_contract: String::new(),
                    difficulty_level: 0,
                    params_hash: String::new(),
                    prior: 1.0,
                    cost: 1.0,
                    loss_ema: 8.0,
                    previous_loss_ema: 8.0,
                    gradient_alignment: 0.0,
                    is_hash_noise: false,
                    capability_feedback_count: 0,
                    capability_verifier_ema: 0.0,
                    capability_partial_ema: 0.0,
                    capability_completion_health_ema: 0.0,
                    capability_schema_wrong_ema: 0.0,
                    capability_malformed_ema: 0.0,
                    capability_missing_ema: 0.0,
                },
                RuliadSamplerCandidate {
                    oracle_hash: "category:verify_category_law@d12#00000002".to_string(),
                    family: "category".to_string(),
                    task_kind: "verify_category_law".to_string(),
                    answer_contract: String::new(),
                    difficulty_level: 12,
                    params_hash: String::new(),
                    prior: 1.0,
                    cost: 1.0,
                    loss_ema: 8.0,
                    previous_loss_ema: 8.0,
                    gradient_alignment: 0.0,
                    is_hash_noise: false,
                    capability_feedback_count: 0,
                    capability_verifier_ema: 0.0,
                    capability_partial_ema: 0.0,
                    capability_completion_health_ema: 0.0,
                    capability_schema_wrong_ema: 0.0,
                    capability_malformed_ema: 0.0,
                    capability_missing_ema: 0.0,
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
    fn zero_verifier_feedback_blocks_loss_only_mastery() {
        let mut sampler = RuliadFrontierSampler::new(
            RuliadSamplerConfig {
                temperature: 100.0,
                exploration_floor: 0.0,
                target_loss: 2.0,
                mastery_escape_weight: 0.0,
                ..RuliadSamplerConfig::default()
            },
            vec![RuliadSamplerCandidate {
                oracle_hash: "eca:multi_step_state@d0#00000001".to_string(),
                family: "eca".to_string(),
                task_kind: "multi_step_state".to_string(),
                answer_contract: String::new(),
                difficulty_level: 0,
                params_hash: String::new(),
                prior: 1.0,
                cost: 1.0,
                loss_ema: 0.25,
                previous_loss_ema: 0.30,
                gradient_alignment: 0.0,
                is_hash_noise: false,
                capability_feedback_count: 0,
                capability_verifier_ema: 0.0,
                capability_partial_ema: 0.0,
                capability_completion_health_ema: 0.0,
                capability_schema_wrong_ema: 0.0,
                capability_malformed_ema: 0.0,
                capability_missing_ema: 0.0,
            }],
        );
        assert!(
            !sampler.snapshot().top_buckets[0].mastered,
            "loss-only cold-start state must not report verifier-backed mastery"
        );

        sampler.record_capability_feedback(&RuliadCapabilityFeedback {
            group_label: "difficulty:d0".to_string(),
            item_count: 32,
            verifier_rate: 0.0,
            partial_credit_rate: 1.0,
            schema_valid_wrong_rate: 0.0,
            malformed_rate: 0.0,
            missing_rate: 0.0,
            completion_health_rate: 1.0,
        });
        let snapshot = sampler.snapshot();

        assert_eq!(snapshot.mastered_probability, 0.0);
        assert!(
            !snapshot.top_buckets[0].mastered,
            "zero-verifier capability feedback must block CE-only mastery"
        );
    }

    #[test]
    fn capability_frontier_limits_unverified_hard_probability() {
        let mut sampler = RuliadFrontierSampler::new(
            RuliadSamplerConfig {
                exploration_floor: 0.0,
                target_loss: 2.0,
                mastery_escape_weight: 0.0,
                capability_frontier_max_ahead: 1,
                capability_frontier_max_unverified_probability: 0.05,
                ..RuliadSamplerConfig::default()
            },
            vec![
                RuliadSamplerCandidate {
                    oracle_hash: "category:verify_category_law@d0#00000001".to_string(),
                    family: "category".to_string(),
                    task_kind: "verify_category_law".to_string(),
                    answer_contract: String::new(),
                    difficulty_level: 0,
                    params_hash: String::new(),
                    prior: 1.0,
                    cost: 1.0,
                    loss_ema: 8.0,
                    previous_loss_ema: 8.0,
                    gradient_alignment: 0.0,
                    is_hash_noise: false,
                    capability_feedback_count: 1,
                    capability_verifier_ema: 1.0,
                    capability_partial_ema: 0.0,
                    capability_completion_health_ema: 1.0,
                    capability_schema_wrong_ema: 0.0,
                    capability_malformed_ema: 0.0,
                    capability_missing_ema: 0.0,
                },
                RuliadSamplerCandidate {
                    oracle_hash: "category:verify_category_law@d1#00000002".to_string(),
                    family: "category".to_string(),
                    task_kind: "verify_category_law".to_string(),
                    answer_contract: String::new(),
                    difficulty_level: 1,
                    params_hash: String::new(),
                    prior: 1.0,
                    cost: 1.0,
                    loss_ema: 8.0,
                    previous_loss_ema: 8.0,
                    gradient_alignment: 0.0,
                    is_hash_noise: false,
                    capability_feedback_count: 0,
                    capability_verifier_ema: 0.0,
                    capability_partial_ema: 0.0,
                    capability_completion_health_ema: 0.0,
                    capability_schema_wrong_ema: 0.0,
                    capability_malformed_ema: 0.0,
                    capability_missing_ema: 0.0,
                },
                RuliadSamplerCandidate {
                    oracle_hash: "category:verify_category_law@d12#00000003".to_string(),
                    family: "category".to_string(),
                    task_kind: "verify_category_law".to_string(),
                    answer_contract: String::new(),
                    difficulty_level: 12,
                    params_hash: String::new(),
                    prior: 1.0,
                    cost: 1.0,
                    loss_ema: 8.0,
                    previous_loss_ema: 8.0,
                    gradient_alignment: 0.0,
                    is_hash_noise: false,
                    capability_feedback_count: 0,
                    capability_verifier_ema: 0.0,
                    capability_partial_ema: 0.0,
                    capability_completion_health_ema: 0.0,
                    capability_schema_wrong_ema: 0.0,
                    capability_malformed_ema: 0.0,
                    capability_missing_ema: 0.0,
                },
            ],
        );

        sampler.record_capability_feedback(&RuliadCapabilityFeedback {
            group_label: "difficulty:d12".to_string(),
            item_count: 32,
            verifier_rate: 0.0,
            partial_credit_rate: 0.25,
            schema_valid_wrong_rate: 0.0,
            malformed_rate: 0.0,
            missing_rate: 0.0,
            completion_health_rate: 1.0,
        });
        let snapshot = sampler.snapshot();

        assert!(
            snapshot.max_difficulty_probability <= 0.051,
            "unverified hard frontier mass should be bounded, got {}",
            snapshot.max_difficulty_probability
        );
    }

    #[test]
    fn one_mastered_bucket_does_not_unlock_a_multitask_level() {
        let mut category = candidate_for_contract_test(
            "category:verify_category_law@d0#00000001",
            "category",
            "verify_category_law",
        );
        category.difficulty_level = 0;
        let mut proof = candidate_for_contract_test(
            "formal_proof:construct_proof@d0#00000002",
            "formal_proof",
            "construct_proof",
        );
        proof.difficulty_level = 0;
        let mut next = candidate_for_contract_test(
            "category:verify_category_law@d1#00000003",
            "category",
            "verify_category_law",
        );
        next.difficulty_level = 1;
        let mut far = candidate_for_contract_test(
            "category:verify_category_law@d2#00000004",
            "category",
            "verify_category_law",
        );
        far.difficulty_level = 2;
        let category_label = category.oracle_hash.clone();
        let mut sampler = RuliadFrontierSampler::new(
            RuliadSamplerConfig {
                temperature: 100.0,
                exploration_floor: 0.0,
                mastery_escape_weight: 0.0,
                capability_frontier_max_ahead: 1,
                capability_frontier_max_unverified_probability: 0.05,
                ..RuliadSamplerConfig::default()
            },
            vec![category, proof, next, far],
        );

        sampler.record_capability_feedback(&RuliadCapabilityFeedback {
            group_label: format!("bucket:{category_label}"),
            item_count: 64,
            verifier_rate: 1.0,
            partial_credit_rate: 1.0,
            schema_valid_wrong_rate: 0.0,
            malformed_rate: 0.0,
            missing_rate: 0.0,
            completion_health_rate: 1.0,
        });
        let snapshot = sampler.snapshot();
        let d0 = snapshot
            .capability_frontier_coverage
            .iter()
            .find(|coverage| coverage.difficulty_level == 0)
            .expect("d0 coverage");
        let d2_probability = snapshot
            .difficulty_buckets
            .iter()
            .find(|bucket| bucket.label == "d2")
            .map(|bucket| bucket.probability)
            .unwrap_or_default();

        assert!((d0.candidate_coverage - 0.5).abs() < 1.0e-6, "{d0:?}");
        assert!(!d0.mastered, "{d0:?}");
        assert_eq!(snapshot.capability_frontier_allowed_max_difficulty, 1);
        assert!(d2_probability <= 0.051, "d2_probability={d2_probability}");
    }

    #[test]
    fn contiguous_capability_mastery_unlocks_next_frontier_band() {
        let candidates = (0..=4)
            .map(|difficulty_level| RuliadSamplerCandidate {
                oracle_hash: format!("category:verify_category_law@d{difficulty_level}"),
                family: "category".to_string(),
                task_kind: "verify_category_law".to_string(),
                answer_contract: String::new(),
                difficulty_level,
                params_hash: String::new(),
                prior: 1.0,
                cost: 1.0,
                loss_ema: 0.25,
                previous_loss_ema: 0.30,
                gradient_alignment: 0.0,
                is_hash_noise: false,
                capability_feedback_count: 0,
                capability_verifier_ema: 0.0,
                capability_partial_ema: 0.0,
                capability_completion_health_ema: 0.0,
                capability_schema_wrong_ema: 0.0,
                capability_malformed_ema: 0.0,
                capability_missing_ema: 0.0,
            })
            .collect::<Vec<_>>();
        let mut sampler = RuliadFrontierSampler::new(
            RuliadSamplerConfig {
                temperature: 100.0,
                exploration_floor: 0.0,
                target_loss: 2.0,
                mastery_escape_weight: 0.0,
                capability_frontier_max_ahead: 1,
                capability_frontier_max_unverified_probability: 0.05,
                ..RuliadSamplerConfig::default()
            },
            candidates,
        );

        for difficulty in [0, 1] {
            sampler.record_capability_feedback(&RuliadCapabilityFeedback {
                group_label: format!("difficulty:d{difficulty}"),
                item_count: 32,
                verifier_rate: 1.0,
                partial_credit_rate: 1.0,
                schema_valid_wrong_rate: 0.0,
                malformed_rate: 0.0,
                missing_rate: 0.0,
                completion_health_rate: 1.0,
            });
        }
        let unlocked_to_d2 = sampler.snapshot();
        let d2_probability = unlocked_to_d2
            .difficulty_buckets
            .iter()
            .find(|bucket| bucket.label == "d2")
            .map(|bucket| bucket.probability)
            .unwrap_or(0.0);
        let blocked_probability = unlocked_to_d2
            .difficulty_buckets
            .iter()
            .filter(|bucket| bucket.label == "d3" || bucket.label == "d4")
            .map(|bucket| bucket.probability)
            .sum::<f32>();

        assert!(
            d2_probability > blocked_probability,
            "next frontier band should be favored over farther unverified buckets: d2={d2_probability} blocked={blocked_probability}"
        );
        assert!(
            blocked_probability <= 0.051,
            "farther unverified frontier mass should stay bounded, got {blocked_probability}"
        );

        sampler.record_capability_feedback(&RuliadCapabilityFeedback {
            group_label: "difficulty:d2".to_string(),
            item_count: 32,
            verifier_rate: 1.0,
            partial_credit_rate: 1.0,
            schema_valid_wrong_rate: 0.0,
            malformed_rate: 0.0,
            missing_rate: 0.0,
            completion_health_rate: 1.0,
        });
        let unlocked_to_d3 = sampler.snapshot();
        let d3_probability = unlocked_to_d3
            .difficulty_buckets
            .iter()
            .find(|bucket| bucket.label == "d3")
            .map(|bucket| bucket.probability)
            .unwrap_or(0.0);

        assert!(
            d3_probability > blocked_probability,
            "mastering d2 should unlock meaningful d3 probability: before_blocked={blocked_probability} d3={d3_probability}"
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
                    answer_contract: String::new(),
                    difficulty_level: 0,
                    params_hash: String::new(),
                    prior: 1.0,
                    cost: 1.0,
                    loss_ema: 1.0,
                    previous_loss_ema: 1.5,
                    gradient_alignment: 0.0,
                    is_hash_noise: false,
                    capability_feedback_count: 1,
                    capability_verifier_ema: 1.0,
                    capability_partial_ema: 0.0,
                    capability_completion_health_ema: 1.0,
                    capability_schema_wrong_ema: 0.0,
                    capability_malformed_ema: 0.0,
                    capability_missing_ema: 0.0,
                },
                RuliadSamplerCandidate {
                    oracle_hash: "hard".to_string(),
                    family: "category".to_string(),
                    task_kind: "proof".to_string(),
                    answer_contract: String::new(),
                    difficulty_level: 1,
                    params_hash: String::new(),
                    prior: 1.0,
                    cost: 1.0,
                    loss_ema: 3.0,
                    previous_loss_ema: 3.5,
                    gradient_alignment: 0.0,
                    is_hash_noise: false,
                    capability_feedback_count: 0,
                    capability_verifier_ema: 0.0,
                    capability_partial_ema: 0.0,
                    capability_completion_health_ema: 0.0,
                    capability_schema_wrong_ema: 0.0,
                    capability_malformed_ema: 0.0,
                    capability_missing_ema: 0.0,
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
    fn snapshot_reports_active_support_entropy() {
        let sampler = RuliadFrontierSampler::new(
            RuliadSamplerConfig::default(),
            vec![
                candidate_for_contract_test("a", "category", "trace"),
                candidate_for_contract_test("b", "category", "proof"),
                candidate_for_contract_test("c", "rewrite", "trace"),
                candidate_for_contract_test("d", "rewrite", "proof"),
            ],
        );

        let snapshot = sampler.snapshot_with_probabilities(&[0.5, 0.5, 0.0, 0.0]);

        assert_eq!(snapshot.sample_count, 4);
        assert_eq!(snapshot.active_candidate_count, 2);
        assert!((snapshot.sampler_entropy_bits - 1.0).abs() < 1e-6);
        assert!((snapshot.active_max_entropy_bits - 1.0).abs() < 1e-6);
        assert!((snapshot.normalized_sampler_entropy - 1.0).abs() < 1e-6);
    }

    #[test]
    fn sampler_escapes_mastered_easy_buckets_toward_higher_difficulty() {
        let candidates = (0..=6)
            .map(|difficulty_level| RuliadSamplerCandidate {
                oracle_hash: format!("d{difficulty_level}"),
                family: "category".to_string(),
                task_kind: "proof".to_string(),
                answer_contract: String::new(),
                difficulty_level,
                params_hash: String::new(),
                prior: 1.0,
                cost: 1.0,
                loss_ema: 0.25,
                previous_loss_ema: 0.30,
                gradient_alignment: 0.0,
                is_hash_noise: false,
                capability_feedback_count: 1,
                capability_verifier_ema: 1.0,
                capability_partial_ema: 0.0,
                capability_completion_health_ema: 1.0,
                capability_schema_wrong_ema: 0.0,
                capability_malformed_ema: 0.0,
                capability_missing_ema: 0.0,
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
                max_answer_contract_probability: 1.0,
                min_answer_contract_probability: 0.0,
                capability_frontier_max_ahead: 1,
                capability_frontier_max_unverified_probability: 0.08,
                capability_remediation_weight: default_capability_remediation_weight(),
                capability_frontier_min_coverage: default_capability_frontier_min_coverage(),
                capability_mastery: RuliadCapabilityMasteryThresholds::default(),
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
