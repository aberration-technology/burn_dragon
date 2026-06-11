use serde::{Deserialize, Serialize};

use crate::config::UsizeRangeConfig;
use crate::ruliad::config::{
    RuliadCorpusConfig, RuliadFamilyConfig, RuliadFamilyKind, RuliadSourceSemantics,
    RuliadTaskKind, ruliad_source_semantics,
};
use crate::ruliad::rng::{SplitMix64, mix_seed};
use crate::ruliad::search::{RuliadSamplerCandidate, RuliadSamplerConfig};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, Hash)]
pub struct RuliadSourceBucketId {
    pub family: RuliadFamilyKind,
    pub task_kind: RuliadTaskKind,
}

impl RuliadSourceBucketId {
    pub fn label(&self) -> String {
        format!("{}:{}", self.family.label(), self.task_kind.label())
    }

    pub fn seed_tag(&self) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for byte in self.label().bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
        }
        hash
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RuliadSourceBucket {
    pub id: RuliadSourceBucketId,
    pub family_config: RuliadFamilyConfig,
    pub prior: f32,
}

impl RuliadSourceBucket {
    pub fn label(&self) -> String {
        self.id.label()
    }

    pub fn is_hash_noise(&self) -> bool {
        self.id.family == RuliadFamilyKind::HashNoise
    }

    pub fn semantics(&self) -> RuliadSourceSemantics {
        ruliad_source_semantics(self.id.family, self.id.task_kind)
    }

    pub fn to_sampler_candidate(&self, config: RuliadSamplerConfig) -> RuliadSamplerCandidate {
        RuliadSamplerCandidate {
            oracle_hash: self.label(),
            family: self.id.family.label().to_string(),
            task_kind: self.id.task_kind.label().to_string(),
            prior: self.prior.max(1e-9),
            cost: 1.0,
            loss_ema: config.target_loss,
            previous_loss_ema: config.target_loss,
            gradient_alignment: 0.0,
            is_hash_noise: self.is_hash_noise(),
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RuliadEpochSourcePlan {
    pub bucket_ids: Vec<String>,
}

impl RuliadEpochSourcePlan {
    pub fn bucket_for_sample(&self, sample_index: usize) -> Option<&str> {
        self.bucket_ids.get(sample_index).map(String::as_str)
    }
}

pub fn ruliad_source_buckets(config: &RuliadCorpusConfig) -> Vec<RuliadSourceBucket> {
    let mut buckets = Vec::new();
    for family in &config.families {
        match family.kind {
            RuliadFamilyKind::Eca => add_eca_buckets(&mut buckets, family),
            RuliadFamilyKind::Simulation => buckets.push(single_bucket(
                family,
                RuliadTaskKind::VerifySimulation,
                family.weight as f32,
            )),
            RuliadFamilyKind::Automaton => buckets.push(single_bucket(
                family,
                RuliadTaskKind::EvaluateAutomaton,
                family.weight as f32,
            )),
            RuliadFamilyKind::Rewrite => buckets.push(single_bucket(
                family,
                RuliadTaskKind::RewriteNormalForm,
                family.weight as f32,
            )),
            RuliadFamilyKind::Algebra => buckets.push(single_bucket(
                family,
                RuliadTaskKind::CheckAlgebraLaw,
                family.weight as f32,
            )),
            RuliadFamilyKind::Category => buckets.push(single_bucket(
                family,
                RuliadTaskKind::ComposeCategoryPath,
                family.weight as f32,
            )),
            RuliadFamilyKind::LeanTask => buckets.push(single_bucket(
                family,
                RuliadTaskKind::CompleteProof,
                family.weight as f32,
            )),
            RuliadFamilyKind::HashNoise => buckets.push(single_bucket(
                family,
                RuliadTaskKind::HashCanary,
                family.weight as f32,
            )),
        }
    }
    buckets
}

pub fn ruliad_sampler_candidates(config: &RuliadCorpusConfig) -> Vec<RuliadSamplerCandidate> {
    ruliad_source_buckets(config)
        .into_iter()
        .map(|bucket| bucket.to_sampler_candidate(config.source_selection.sampler))
        .collect()
}

pub fn plan_epoch_source_buckets(
    buckets: &[RuliadSourceBucket],
    probabilities: &[f32],
    sample_count: usize,
    seed: u64,
    split_tag: u64,
    epoch_index: usize,
) -> RuliadEpochSourcePlan {
    if buckets.is_empty() || sample_count == 0 {
        return RuliadEpochSourcePlan {
            bucket_ids: Vec::new(),
        };
    }

    let mut weights = buckets
        .iter()
        .enumerate()
        .map(|(index, bucket)| {
            probabilities
                .get(index)
                .copied()
                .filter(|value| value.is_finite() && *value > 0.0)
                .unwrap_or(bucket.prior.max(1e-9))
        })
        .collect::<Vec<_>>();
    normalize_weights(&mut weights);

    let mut rng = SplitMix64::new(mix_seed(
        seed,
        [
            split_tag,
            epoch_index as u64,
            sample_count as u64,
            buckets.len() as u64,
        ],
    ));
    let mut selected = Vec::with_capacity(sample_count);
    if sample_count >= buckets.len() {
        selected.extend(buckets.iter().map(RuliadSourceBucket::label));
    }
    while selected.len() < sample_count {
        let index = sample_weighted_index(&weights, &mut rng);
        selected.push(buckets[index].label());
    }
    shuffle(&mut selected, &mut rng);
    RuliadEpochSourcePlan {
        bucket_ids: selected,
    }
}

fn add_eca_buckets(buckets: &mut Vec<RuliadSourceBucket>, family: &RuliadFamilyConfig) {
    let steps = family.steps.unwrap_or(UsizeRangeConfig { min: 4, max: 10 });
    let total = steps.max.saturating_sub(steps.min).saturating_add(1).max(1) as f32;
    if steps.min <= 1 && steps.max >= 1 {
        let mut family_config = family.clone();
        family_config.steps = Some(UsizeRangeConfig { min: 1, max: 1 });
        buckets.push(RuliadSourceBucket {
            id: RuliadSourceBucketId {
                family: RuliadFamilyKind::Eca,
                task_kind: RuliadTaskKind::NextState,
            },
            family_config,
            prior: family.weight as f32 / total,
        });
    }
    if steps.max >= 2 {
        let multi_min = steps.min.max(2);
        let multi_count = steps.max.saturating_sub(multi_min).saturating_add(1).max(1) as f32;
        let mut family_config = family.clone();
        family_config.steps = Some(UsizeRangeConfig {
            min: multi_min,
            max: steps.max,
        });
        buckets.push(RuliadSourceBucket {
            id: RuliadSourceBucketId {
                family: RuliadFamilyKind::Eca,
                task_kind: RuliadTaskKind::MultiStepState,
            },
            family_config,
            prior: family.weight as f32 * multi_count / total,
        });
    }
}

fn single_bucket(
    family: &RuliadFamilyConfig,
    task_kind: RuliadTaskKind,
    prior: f32,
) -> RuliadSourceBucket {
    RuliadSourceBucket {
        id: RuliadSourceBucketId {
            family: family.kind,
            task_kind,
        },
        family_config: family.clone(),
        prior,
    }
}

fn normalize_weights(weights: &mut [f32]) {
    let sum = weights
        .iter()
        .filter(|value| value.is_finite() && **value > 0.0)
        .sum::<f32>();
    if sum <= 0.0 {
        let uniform = 1.0 / weights.len().max(1) as f32;
        for weight in weights {
            *weight = uniform;
        }
        return;
    }
    for weight in weights {
        *weight = weight.max(0.0) / sum;
    }
}

fn sample_weighted_index(weights: &[f32], rng: &mut SplitMix64) -> usize {
    let ticket = rng.next_f32();
    let mut cumulative = 0.0;
    for (index, weight) in weights.iter().enumerate() {
        cumulative += *weight;
        if ticket <= cumulative {
            return index;
        }
    }
    weights.len().saturating_sub(1)
}

fn shuffle(values: &mut [String], rng: &mut SplitMix64) {
    for index in (1..values.len()).rev() {
        let swap_index = rng.next_usize(index + 1);
        values.swap(index, swap_index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ruliad::config::{
        RULIAD_REQUIRED_MATH_DOMAINS, RULIAD_REQUIRED_REASONING_MODES, RuliadSerializationConfig,
        RuliadSourceSelectionConfig, RuliadTokenizationConfig, default_ruliad_families,
    };

    fn config_with_eca_steps(min: usize, max: usize) -> RuliadCorpusConfig {
        RuliadCorpusConfig {
            output_dir: "ignored".into(),
            seed: 1,
            name: "source-selection".to_string(),
            train_samples: 16,
            validation_samples: 4,
            chunk_token_capacity: 1024,
            serialization: RuliadSerializationConfig::default(),
            tokenization: RuliadTokenizationConfig::default(),
            source_selection: RuliadSourceSelectionConfig::default(),
            families: vec![RuliadFamilyConfig {
                kind: RuliadFamilyKind::Eca,
                weight: 4,
                width: Some(UsizeRangeConfig { min: 8, max: 8 }),
                steps: Some(UsizeRangeConfig { min, max }),
            }],
            proof_tasks: None,
            lean_task_limit: None,
        }
    }

    #[test]
    fn eca_range_crossing_one_splits_into_task_buckets() {
        let buckets = ruliad_source_buckets(&config_with_eca_steps(1, 3));
        assert_eq!(buckets.len(), 2);
        assert_eq!(buckets[0].id.task_kind, RuliadTaskKind::NextState);
        assert_eq!(buckets[1].id.task_kind, RuliadTaskKind::MultiStepState);
        assert!((buckets[0].prior - 4.0 / 3.0).abs() < 1e-6);
        assert!((buckets[1].prior - 8.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn source_plan_is_deterministic_and_covers_active_buckets() {
        let config = config_with_eca_steps(1, 3);
        let buckets = ruliad_source_buckets(&config);
        let first = plan_epoch_source_buckets(&buckets, &[0.9, 0.1], 8, 42, 7, 2);
        let second = plan_epoch_source_buckets(&buckets, &[0.9, 0.1], 8, 42, 7, 2);
        assert_eq!(first, second);
        assert!(first.bucket_ids.iter().any(|id| id == "eca:next_state"));
        assert!(
            first
                .bucket_ids
                .iter()
                .any(|id| id == "eca:multi_step_state")
        );
    }

    #[test]
    fn source_plan_mixes_default_buckets_without_long_stripes() {
        let config = RuliadCorpusConfig {
            output_dir: "ignored".into(),
            seed: 17,
            name: "source-selection".to_string(),
            train_samples: 1024,
            validation_samples: 4,
            chunk_token_capacity: 1024,
            serialization: RuliadSerializationConfig::default(),
            tokenization: RuliadTokenizationConfig::default(),
            source_selection: RuliadSourceSelectionConfig::default(),
            families: default_ruliad_families(),
            proof_tasks: None,
            lean_task_limit: None,
        };
        let buckets = ruliad_source_buckets(&config);
        let probabilities = buckets
            .iter()
            .map(|bucket| bucket.prior)
            .collect::<Vec<_>>();
        let plan = plan_epoch_source_buckets(
            &buckets,
            &probabilities,
            config.train_samples,
            config.seed,
            11,
            3,
        );
        let counts =
            plan.bucket_ids
                .iter()
                .fold(std::collections::BTreeMap::new(), |mut counts, id| {
                    *counts.entry(id.as_str()).or_insert(0usize) += 1;
                    counts
                });

        for bucket in &buckets {
            assert!(
                counts
                    .get(bucket.label().as_str())
                    .copied()
                    .unwrap_or_default()
                    > 0,
                "missing bucket {}",
                bucket.label()
            );
        }

        let adjacent_changes = plan
            .bucket_ids
            .windows(2)
            .filter(|pair| pair[0] != pair[1])
            .count();
        let max_run = plan
            .bucket_ids
            .iter()
            .fold((0usize, "", 0usize), |(max_run, current, run), id| {
                let next_run = if id == current { run + 1 } else { 1 };
                (max_run.max(next_run), id.as_str(), next_run)
            })
            .0;

        assert!(
            adjacent_changes > config.train_samples / 2,
            "source plan has too few adjacent changes: {}",
            adjacent_changes
        );
        assert!(
            max_run < 32,
            "source plan has suspiciously long same-source run: {}",
            max_run
        );
    }

    #[test]
    fn default_source_buckets_cover_required_semantics() {
        let config = RuliadCorpusConfig {
            output_dir: "ignored".into(),
            seed: 17,
            name: "source-selection".to_string(),
            train_samples: 1024,
            validation_samples: 4,
            chunk_token_capacity: 1024,
            serialization: RuliadSerializationConfig::default(),
            tokenization: RuliadTokenizationConfig::default(),
            source_selection: RuliadSourceSelectionConfig::default(),
            families: default_ruliad_families(),
            proof_tasks: None,
            lean_task_limit: None,
        };
        let buckets = ruliad_source_buckets(&config);
        let mut domains = std::collections::BTreeSet::new();
        let mut modes = std::collections::BTreeSet::new();
        for bucket in &buckets {
            let semantics = bucket.semantics();
            domains.extend(semantics.math_domains.iter().copied());
            modes.extend(semantics.reasoning_modes.iter().copied());
        }

        for domain in RULIAD_REQUIRED_MATH_DOMAINS {
            assert!(
                domains.contains(domain),
                "missing ruliad math domain {}",
                domain.label()
            );
        }
        for mode in RULIAD_REQUIRED_REASONING_MODES {
            assert!(
                modes.contains(mode),
                "missing ruliad reasoning mode {}",
                mode.label()
            );
        }
    }
}
