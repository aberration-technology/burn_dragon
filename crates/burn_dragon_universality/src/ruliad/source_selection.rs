use serde::{Deserialize, Serialize};

use crate::config::UsizeRangeConfig;
use crate::ruliad::config::{
    RuliadCorpusConfig, RuliadFamilyConfig, RuliadFamilyKind, RuliadProofActionAnswerContract,
    RuliadSourceSemantics, RuliadTaskKind, ruliad_source_semantics,
};
use crate::ruliad::ir::RuliadComplexityVector;
use crate::ruliad::oracles::scale_family_for_difficulty;
use crate::ruliad::rng::{SplitMix64, mix_seed};
use crate::ruliad::search::source_answer_contract;
use crate::ruliad::search::{RuliadSamplerCandidate, RuliadSamplerConfig};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, Hash)]
pub struct RuliadSourceBucketId {
    pub family: RuliadFamilyKind,
    pub task_kind: RuliadTaskKind,
    pub difficulty_level: usize,
    pub params_hash: u64,
}

impl RuliadSourceBucketId {
    pub fn label(&self) -> String {
        format!(
            "{}:{}@d{}#{:08x}",
            self.family.label(),
            self.task_kind.label(),
            self.difficulty_level,
            (self.params_hash & 0xffff_ffff) as u32
        )
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

    pub fn estimated_complexity(&self) -> RuliadComplexityVector {
        let width = self
            .family_config
            .width
            .map(|range| range.max)
            .unwrap_or(1)
            .max(1);
        let steps = self
            .family_config
            .steps
            .map(|range| range.max)
            .unwrap_or(1)
            .max(1);
        if self.id.family != RuliadFamilyKind::FormalProof {
            return RuliadComplexityVector {
                syntax_nodes: width.saturating_mul(steps),
                proof_step_count: steps,
                dependency_width: width,
                maximum_term_depth: steps,
                memory_horizon: steps,
                search_branching: width,
                verifier_work: width.saturating_mul(steps),
                ..RuliadComplexityVector::default()
            };
        }

        let dependency_levels = width
            .saturating_sub(1)
            .checked_next_power_of_two()
            .unwrap_or(usize::MAX)
            .ilog2() as usize;
        let context_depth = 1usize.saturating_add(steps.ilog2() as usize);
        let proof_goal_count = width.saturating_mul(2).saturating_sub(1);
        let proof_step_count = width
            .saturating_mul(steps)
            .saturating_add(width.saturating_sub(1).saturating_mul(2));
        let expanded_leaf_nodes = steps
            .saturating_mul(3)
            .saturating_add(context_depth.saturating_mul(3))
            .saturating_add(2);
        let syntax_nodes = expanded_leaf_nodes
            .saturating_mul(width)
            .saturating_mul(dependency_levels.saturating_add(1))
            .saturating_mul(2);
        RuliadComplexityVector {
            syntax_nodes,
            axiom_count: 3usize.saturating_add(steps.div_ceil(2)),
            proof_goal_count,
            proof_step_count,
            dependency_depth: dependency_levels.saturating_add(1),
            dependency_width: usize::from(width > 1).saturating_add(1).min(width),
            variable_count: 3,
            maximum_term_depth: steps.saturating_mul(2).saturating_add(context_depth),
            distractor_axiom_count: steps.div_ceil(2),
            branch_entropy_millibits: usize::from(width > 1).saturating_mul(1000),
            abstraction_depth: dependency_levels.saturating_add(1),
            memory_horizon: proof_step_count,
            solution_multiplicity: 1,
            search_branching: 3usize.saturating_add(steps.div_ceil(2)).saturating_mul(2),
            verifier_work: syntax_nodes.saturating_add(proof_step_count),
            ..Default::default()
        }
    }

    pub fn to_sampler_candidate(&self, config: RuliadSamplerConfig) -> RuliadSamplerCandidate {
        let complexity = self.estimated_complexity();
        let work = complexity
            .syntax_nodes
            .saturating_add(complexity.proof_step_count)
            .max(1);
        let cost = 1.0 + (work as f32).log2() / 16.0;
        RuliadSamplerCandidate {
            oracle_hash: self.label(),
            family: self.id.family.label().to_string(),
            task_kind: self.id.task_kind.label().to_string(),
            answer_contract: source_answer_contract(
                self.id.family,
                self.id.task_kind,
                RuliadProofActionAnswerContract::PresentationIndex,
            )
            .unwrap_or_default()
            .to_string(),
            difficulty_level: self.id.difficulty_level,
            params_hash: format!("{:016x}", self.id.params_hash),
            prior: self.prior.max(1e-9),
            cost,
            loss_ema: config.target_loss + (cost - 1.0).clamp(0.0, 1.0),
            previous_loss_ema: config.target_loss + (cost - 1.0).clamp(0.0, 1.0),
            gradient_alignment: 0.0,
            is_hash_noise: self.is_hash_noise(),
            capability_feedback_count: 0,
            capability_verifier_ema: 0.0,
            capability_partial_ema: 0.0,
            capability_completion_health_ema: 0.0,
            capability_schema_wrong_ema: 0.0,
            capability_malformed_ema: 0.0,
            capability_missing_ema: 0.0,
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
    for difficulty_level in config.source_selection.difficulty_levels.min
        ..=config.source_selection.difficulty_levels.max
    {
        buckets.extend(ruliad_source_buckets_for_difficulty(
            config,
            difficulty_level,
        ));
    }
    buckets
}

pub fn ruliad_source_buckets_for_difficulty(
    config: &RuliadCorpusConfig,
    difficulty_level: usize,
) -> Vec<RuliadSourceBucket> {
    let mut buckets = Vec::new();
    for family in &config.families {
        let family_config = scale_family_for_difficulty(family, difficulty_level);
        match family.kind {
            RuliadFamilyKind::Eca => {
                add_eca_buckets(&mut buckets, &family_config, difficulty_level)
            }
            RuliadFamilyKind::Simulation => buckets.push(single_bucket(
                &family_config,
                RuliadTaskKind::VerifySimulation,
                family.weight as f32,
                difficulty_level,
            )),
            RuliadFamilyKind::Automaton => buckets.push(single_bucket(
                &family_config,
                RuliadTaskKind::EvaluateAutomaton,
                family.weight as f32,
                difficulty_level,
            )),
            RuliadFamilyKind::Rewrite => buckets.push(single_bucket(
                &family_config,
                RuliadTaskKind::RewriteNormalForm,
                family.weight as f32,
                difficulty_level,
            )),
            RuliadFamilyKind::Algebra => buckets.push(single_bucket(
                &family_config,
                RuliadTaskKind::CheckAlgebraLaw,
                family.weight as f32,
                difficulty_level,
            )),
            RuliadFamilyKind::Category => {
                add_category_buckets(&mut buckets, &family_config, difficulty_level)
            }
            RuliadFamilyKind::ProofTree => buckets.push(single_bucket(
                &family_config,
                RuliadTaskKind::ProveTheorem,
                family.weight as f32,
                difficulty_level,
            )),
            RuliadFamilyKind::FormalProof => {
                let mix = &config.source_selection.formal_task_mix;
                let task_weights = [
                    (RuliadTaskKind::AdvanceProof, mix.advance_proof_weight),
                    (
                        RuliadTaskKind::SelectProofAction,
                        mix.select_proof_action_weight,
                    ),
                    (RuliadTaskKind::ConstructProof, mix.construct_proof_weight),
                    (RuliadTaskKind::CheckProof, mix.check_proof_weight),
                ];
                let total = task_weights
                    .iter()
                    .map(|(_, weight)| *weight)
                    .sum::<usize>()
                    .max(1) as f32;
                for (task, weight) in task_weights {
                    if weight == 0 {
                        continue;
                    }
                    buckets.push(single_bucket(
                        &family_config,
                        task,
                        family.weight as f32 * weight as f32 / total,
                        difficulty_level,
                    ));
                }
            }
            RuliadFamilyKind::LeanTask => buckets.push(single_bucket(
                &family_config,
                RuliadTaskKind::CompleteProof,
                family.weight as f32,
                difficulty_level,
            )),
            RuliadFamilyKind::HashNoise => buckets.push(single_bucket(
                &family_config,
                RuliadTaskKind::HashCanary,
                family.weight as f32,
                difficulty_level,
            )),
        }
    }
    buckets
}

pub fn ruliad_sampler_candidates(config: &RuliadCorpusConfig) -> Vec<RuliadSamplerCandidate> {
    ruliad_source_buckets(config)
        .into_iter()
        .map(|bucket| configured_sampler_candidate(config, &bucket))
        .collect()
}

pub fn ruliad_sampler_candidates_for_difficulty(
    config: &RuliadCorpusConfig,
    difficulty_level: usize,
) -> Vec<RuliadSamplerCandidate> {
    ruliad_source_buckets_for_difficulty(config, difficulty_level)
        .into_iter()
        .map(|bucket| configured_sampler_candidate(config, &bucket))
        .collect()
}

fn configured_sampler_candidate(
    config: &RuliadCorpusConfig,
    bucket: &RuliadSourceBucket,
) -> RuliadSamplerCandidate {
    let mut candidate = bucket.to_sampler_candidate(config.source_selection.sampler);
    candidate.answer_contract = source_answer_contract(
        bucket.id.family,
        bucket.id.task_kind,
        config
            .source_selection
            .formal_task_mix
            .proof_action_answer_contract,
    )
    .unwrap_or_default()
    .to_string();
    candidate
}

pub fn ruliad_source_bucket_by_label(
    config: &RuliadCorpusConfig,
    bucket_label: &str,
) -> Option<RuliadSourceBucket> {
    let difficulty_level = bucket_label_difficulty_level(bucket_label)?;
    ruliad_source_buckets_for_difficulty(config, difficulty_level)
        .into_iter()
        .find(|bucket| bucket.label() == bucket_label)
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
                .filter(|value| value.is_finite() && *value >= 0.0)
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
    while selected.len() < sample_count {
        let index = sample_weighted_index(&weights, &mut rng);
        selected.push(buckets[index].label());
    }
    shuffle(&mut selected, &mut rng);
    RuliadEpochSourcePlan {
        bucket_ids: selected,
    }
}

fn add_eca_buckets(
    buckets: &mut Vec<RuliadSourceBucket>,
    family: &RuliadFamilyConfig,
    difficulty_level: usize,
) {
    let steps = family.steps.unwrap_or(UsizeRangeConfig { min: 4, max: 10 });
    let total = steps.max.saturating_sub(steps.min).saturating_add(1).max(1) as f32;
    if steps.min <= 1 && steps.max >= 1 {
        let mut family_config = family.clone();
        family_config.steps = Some(UsizeRangeConfig { min: 1, max: 1 });
        buckets.push(RuliadSourceBucket {
            id: RuliadSourceBucketId {
                family: RuliadFamilyKind::Eca,
                task_kind: RuliadTaskKind::NextState,
                difficulty_level,
                params_hash: family_config_hash(&family_config, RuliadTaskKind::NextState),
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
                difficulty_level,
                params_hash: family_config_hash(&family_config, RuliadTaskKind::MultiStepState),
            },
            family_config,
            prior: family.weight as f32 * multi_count / total,
        });
    }
}

fn add_category_buckets(
    buckets: &mut Vec<RuliadSourceBucket>,
    family: &RuliadFamilyConfig,
    difficulty_level: usize,
) {
    let prior = family.weight as f32 / 4.0;
    for task_kind in [
        RuliadTaskKind::ComposeCategoryPath,
        RuliadTaskKind::VerifyCategoryLaw,
        RuliadTaskKind::VerifyFunctorPreservation,
        RuliadTaskKind::VerifyNaturalitySquare,
    ] {
        buckets.push(single_bucket(family, task_kind, prior, difficulty_level));
    }
}

fn single_bucket(
    family: &RuliadFamilyConfig,
    task_kind: RuliadTaskKind,
    prior: f32,
    difficulty_level: usize,
) -> RuliadSourceBucket {
    RuliadSourceBucket {
        id: RuliadSourceBucketId {
            family: family.kind,
            task_kind,
            difficulty_level,
            params_hash: family_config_hash(family, task_kind),
        },
        family_config: family.clone(),
        prior,
    }
}

fn family_config_hash(family: &RuliadFamilyConfig, task_kind: RuliadTaskKind) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let label = format!(
        "{}:{}:{:?}:{:?}",
        family.kind.label(),
        task_kind.label(),
        family.width,
        family.steps
    );
    for byte in label.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

fn bucket_label_difficulty_level(bucket_label: &str) -> Option<usize> {
    let (_, suffix) = bucket_label.split_once("@d")?;
    let (level, _) = suffix.split_once('#')?;
    level.parse().ok()
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
        if *weight > 0.0 && ticket <= cumulative {
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
        RULIAD_REQUIRED_MATH_DOMAINS, RULIAD_REQUIRED_REASONING_MODES, RuliadFormalTaskMixConfig,
        RuliadSerializationConfig, RuliadSourceSelectionConfig, RuliadTokenizationConfig,
        default_ruliad_families, formal_ruliad_families,
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
            formal_generalization: Default::default(),
            source_selection: RuliadSourceSelectionConfig {
                difficulty_levels: UsizeRangeConfig { min: 0, max: 0 },
                ..RuliadSourceSelectionConfig::default()
            },
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
    fn formal_task_mix_materializes_normalized_transition_prior() {
        let mut config = config_with_eca_steps(2, 4);
        config.families = formal_ruliad_families();
        config.source_selection.formal_task_mix = RuliadFormalTaskMixConfig {
            advance_proof_weight: 2,
            select_proof_action_weight: 0,
            construct_proof_weight: 1,
            check_proof_weight: 1,
            proof_action_answer_contract: Default::default(),
        };
        let buckets = ruliad_source_buckets(&config);
        assert_eq!(buckets.len(), 3);
        let prior = |task| {
            buckets
                .iter()
                .find(|bucket| bucket.id.task_kind == task)
                .expect("task bucket")
                .prior
        };
        assert!((prior(RuliadTaskKind::AdvanceProof) - 0.5).abs() < 1e-6);
        assert!((prior(RuliadTaskKind::ConstructProof) - 0.25).abs() < 1e-6);
        assert!((prior(RuliadTaskKind::CheckProof) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn semantic_proof_action_contract_reaches_sampler_metrics() {
        let mut config = config_with_eca_steps(2, 4);
        config.families = formal_ruliad_families();
        config.source_selection.formal_task_mix = RuliadFormalTaskMixConfig {
            advance_proof_weight: 0,
            select_proof_action_weight: 1,
            construct_proof_weight: 0,
            check_proof_weight: 0,
            proof_action_answer_contract: RuliadProofActionAnswerContract::SemanticStep,
        };

        let candidates = ruliad_sampler_candidates(&config);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].answer_contract, "proof_action_step");
        let snapshot = crate::ruliad::search::RuliadFrontierSampler::new(
            config.source_selection.sampler,
            candidates,
        )
        .snapshot();
        assert_eq!(snapshot.contract_buckets.len(), 1);
        assert_eq!(snapshot.contract_buckets[0].label, "proof_action_step");
    }

    #[test]
    fn source_plan_is_deterministic_and_samples_weighted_active_buckets() {
        let config = config_with_eca_steps(1, 3);
        let buckets = ruliad_source_buckets(&config);
        let first = plan_epoch_source_buckets(&buckets, &[0.9, 0.1], 128, 42, 7, 2);
        let second = plan_epoch_source_buckets(&buckets, &[0.9, 0.1], 128, 42, 7, 2);
        assert_eq!(first, second);
        assert!(
            first
                .bucket_ids
                .iter()
                .any(|id| id.starts_with("eca:next_state@d0#"))
        );
        assert!(
            first
                .bucket_ids
                .iter()
                .any(|id| id.starts_with("eca:multi_step_state@d0#"))
        );
    }

    #[test]
    fn source_plan_does_not_force_zero_probability_buckets() {
        let config = config_with_eca_steps(1, 3);
        let buckets = ruliad_source_buckets(&config);
        let plan = plan_epoch_source_buckets(&buckets, &[1.0, 0.0], 32, 42, 7, 2);
        assert!(
            plan.bucket_ids
                .iter()
                .all(|id| id.starts_with("eca:next_state@d0#")),
            "zero-probability bucket was forced into plan: {:?}",
            plan.bucket_ids
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
            formal_generalization: Default::default(),
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
    fn source_buckets_materialize_difficulty_frontier_levels() {
        let mut config = config_with_eca_steps(2, 3);
        config.source_selection.difficulty_levels = UsizeRangeConfig { min: 0, max: 2 };
        let buckets = ruliad_source_buckets(&config);
        assert_eq!(buckets.len(), 3);
        assert!(
            buckets
                .iter()
                .any(|bucket| bucket.label().starts_with("eca:multi_step_state@d0#"))
        );
        assert!(
            buckets
                .iter()
                .any(|bucket| bucket.label().starts_with("eca:multi_step_state@d2#"))
        );
        let easy = buckets
            .iter()
            .find(|bucket| bucket.id.difficulty_level == 0)
            .expect("easy bucket");
        let hard = buckets
            .iter()
            .find(|bucket| bucket.id.difficulty_level == 2)
            .expect("hard bucket");
        assert!(
            hard.family_config.steps.as_ref().expect("hard steps").max
                > easy.family_config.steps.as_ref().expect("easy steps").max
        );
        assert!(
            hard.to_sampler_candidate(RuliadSamplerConfig::default())
                .cost
                > easy
                    .to_sampler_candidate(RuliadSamplerConfig::default())
                    .cost
        );
    }

    #[test]
    fn source_bucket_label_resolver_supports_dynamic_difficulty_levels() {
        let config = config_with_eca_steps(2, 3);
        let dynamic_label = ruliad_source_buckets_for_difficulty(&config, 7)
            .into_iter()
            .next()
            .expect("dynamic bucket")
            .label();
        assert!(
            ruliad_source_buckets(&config)
                .iter()
                .all(|bucket| bucket.label() != dynamic_label)
        );
        let resolved =
            ruliad_source_bucket_by_label(&config, &dynamic_label).expect("resolved bucket");
        assert_eq!(resolved.label(), dynamic_label);
        assert_eq!(resolved.id.difficulty_level, 7);
    }

    #[test]
    fn formal_bucket_cost_tracks_vector_work_without_linear_frontier_penalty() {
        let mut config = config_with_eca_steps(2, 3);
        config.families = formal_ruliad_families();
        let easy = ruliad_source_buckets_for_difficulty(&config, 0)
            .into_iter()
            .next()
            .expect("easy formal bucket");
        let far = ruliad_source_buckets_for_difficulty(&config, 1_024)
            .into_iter()
            .next()
            .expect("far formal bucket");
        let easy_complexity = easy.estimated_complexity();
        let far_complexity = far.estimated_complexity();
        assert!(far_complexity.dominates(&easy_complexity));
        let easy_cost = easy
            .to_sampler_candidate(RuliadSamplerConfig::default())
            .cost;
        let far_cost = far
            .to_sampler_candidate(RuliadSamplerConfig::default())
            .cost;
        assert!(far_cost > easy_cost);
        assert!(
            far_cost < 4.0,
            "log-work cost must not make a distant frontier unreachable: {far_cost}"
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
            formal_generalization: Default::default(),
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
