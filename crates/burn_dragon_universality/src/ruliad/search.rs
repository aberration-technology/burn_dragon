use serde::{Deserialize, Serialize};

use crate::ruliad::metrics::{RuliadMetricSnapshot, RuliadSampleTelemetry};

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
}

impl Default for RuliadSamplerConfig {
    fn default() -> Self {
        Self {
            temperature: default_temperature(),
            exploration_floor: default_exploration_floor(),
            target_loss: default_target_loss(),
            hash_noise_penalty: default_hash_noise_penalty(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RuliadSamplerCandidate {
    pub oracle_hash: String,
    pub family: String,
    pub task_kind: String,
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

    pub fn probabilities(&self) -> Vec<f32> {
        if self.candidates.is_empty() {
            return Vec::new();
        }
        let temperature = self.config.temperature.max(1e-6);
        let logits = self
            .candidates
            .iter()
            .map(|candidate| {
                candidate.prior.max(1e-9).ln() + candidate.utility(self.config) / temperature
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
        probs
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

    pub fn snapshot(&self) -> RuliadMetricSnapshot {
        let probs = self.probabilities();
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
        let mastered_probability = probs
            .iter()
            .zip(&self.candidates)
            .filter_map(|(prob, candidate)| {
                (candidate.loss_ema <= self.config.target_loss).then_some(*prob)
            })
            .sum::<f32>();
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
            mastered_probability,
        }
    }
}

fn difficulty_gate(loss_ema: f32, target_loss: f32) -> f32 {
    1.0 / (1.0 + (loss_ema - target_loss).abs())
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
    fn snapshot_reports_weighted_difficulty_frontier() {
        let sampler = RuliadFrontierSampler::new(
            RuliadSamplerConfig {
                temperature: 100.0,
                exploration_floor: 0.0,
                target_loss: 2.0,
                hash_noise_penalty: 4.0,
            },
            vec![
                RuliadSamplerCandidate {
                    oracle_hash: "easy".to_string(),
                    family: "category".to_string(),
                    task_kind: "trace".to_string(),
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
    }
}
