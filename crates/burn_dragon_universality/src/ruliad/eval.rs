use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::str::FromStr;

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::manifest::{
    CorpusKind, SampleSplit, UniversalityCorpusManifest, UniversalitySampleRecord, load_manifest,
};
use crate::ruliad::config::{
    RULIAD_REQUIRED_MATH_DOMAINS, RULIAD_REQUIRED_REASONING_MODES, RuliadCorpusConfig,
    RuliadMathDomain, RuliadReasoningMode,
};
use crate::ruliad::oracles::{
    RuliadSampleSpec, is_degenerate_spec, ruliad_categorical_presentation, ruliad_expected_answer,
    ruliad_prompt_prefix, sample_text, verify_spec,
};
use crate::ruliad::runtime::{OnlineRuliadCorpus, ruliad_serialized_node_count};
use crate::ruliad::source_selection::{RuliadSourceBucket, ruliad_source_buckets};
use crate::stats::SampleStats;

pub const RULIAD_DIAGNOSTIC_REPORT_VERSION: u32 = 1;
pub const RULIAD_EVAL_REPORT_VERSION: u32 = 3;
pub const RULIAD_REASONING_SCORE_VERSION: u32 = 2;

const MAX_REPORTED_EVAL_FAILURES: usize = 64;
const SCORE_PPM_DENOMINATOR: usize = 1_000_000;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RuliadCountShare {
    pub label: String,
    pub count: usize,
    pub share: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RuliadSourceBucketDiagnostic {
    pub bucket_id: String,
    pub family: String,
    pub task_kind: String,
    pub prior: f32,
    pub math_domains: Vec<String>,
    pub reasoning_modes: Vec<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
pub struct RuliadDiagnosticThresholds {
    #[serde(default)]
    pub min_task_share: f32,
    #[serde(default)]
    pub max_duplicate_oracle_hash_rate: f32,
    #[serde(default = "default_require_all_semantics")]
    pub require_all_semantics: bool,
}

impl Default for RuliadDiagnosticThresholds {
    fn default() -> Self {
        Self {
            min_task_share: 0.0,
            max_duplicate_oracle_hash_rate: 0.0,
            require_all_semantics: default_require_all_semantics(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RuliadDiagnosticReport {
    pub version: u32,
    pub dataset_name: String,
    pub sample_count: usize,
    pub token_count: usize,
    pub document_token_count: usize,
    pub payload_token_capacity: usize,
    pub split_counts: Vec<RuliadCountShare>,
    pub family_counts: Vec<RuliadCountShare>,
    pub task_counts: Vec<RuliadCountShare>,
    pub math_domain_counts: Vec<RuliadCountShare>,
    pub reasoning_mode_counts: Vec<RuliadCountShare>,
    pub source_bucket_priors: Vec<RuliadSourceBucketDiagnostic>,
    pub oracle_hash_count: usize,
    pub duplicate_oracle_hash_count: usize,
    pub duplicate_oracle_hash_rate: f32,
    pub missing_ruliad_spec_count: usize,
    pub missing_oracle_hash_count: usize,
    pub verifier_failure_count: usize,
    pub answer_slot_count: usize,
    pub answer_slot_coverage: f32,
    pub proof_trace_count: usize,
    pub proof_trace_coverage: f32,
    pub degenerate_sample_count: usize,
    pub multi_chunk_document_count: usize,
    pub multi_chunk_document_coverage: f32,
    pub categorical_core_count: usize,
    pub hash_canary_count: usize,
    pub token_count_drift_count: usize,
    pub payload_overflow_count: usize,
    pub max_serialized_char_count: usize,
    pub mean_gzip_complexity_ratio: f32,
    pub mean_complexity_score: f32,
    pub gate_failures: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct RuliadEvalConfig {
    #[serde(default = "default_eval_split")]
    pub split: Option<SampleSplit>,
    #[serde(default)]
    pub max_items: Option<usize>,
    #[serde(default = "default_include_hash_canaries")]
    pub include_hash_canaries: bool,
}

impl Default for RuliadEvalConfig {
    fn default() -> Self {
        Self {
            split: default_eval_split(),
            max_items: None,
            include_hash_canaries: default_include_hash_canaries(),
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RuliadEvalBaseline {
    Oracle,
    Corrupt,
}

impl FromStr for RuliadEvalBaseline {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "oracle" => Ok(Self::Oracle),
            "corrupt" | "corrupted" => Ok(Self::Corrupt),
            other => Err(anyhow!(
                "invalid ruliad eval baseline `{other}`; expected oracle or corrupt"
            )),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RuliadEvalItem {
    pub oracle_hash: String,
    pub sample_index: usize,
    pub split: SampleSplit,
    pub family: String,
    pub task_kind: String,
    pub math_domains: Vec<String>,
    pub reasoning_modes: Vec<String>,
    pub prompt: String,
    pub expected_answer: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub difficulty_level: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spec: Option<RuliadSampleSpec>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct RuliadCompletionRecord {
    pub oracle_hash: String,
    #[serde(alias = "answer", alias = "output", alias = "text")]
    pub completion: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RuliadEvalGroupScore {
    pub label: String,
    pub count: usize,
    pub exact_match_count: usize,
    pub semantic_match_count: usize,
    pub verifier_match_count: usize,
    pub partial_credit_count: usize,
    pub schema_valid_wrong_count: usize,
    pub malformed_completion_count: usize,
    pub missing_completion_count: usize,
    pub exact_accuracy: f32,
    pub semantic_accuracy: f32,
    pub verifier_accuracy: f32,
    pub partial_credit_rate: f32,
    pub mean_partial_progress: f32,
    #[serde(default)]
    pub answer_field_correct_count: usize,
    #[serde(default)]
    pub answer_field_expected_count: usize,
    #[serde(default)]
    pub answer_field_accuracy: f32,
    #[serde(default)]
    pub answer_terminated_count: usize,
    #[serde(default)]
    pub answer_termination_rate: f32,
    #[serde(default)]
    pub mean_completion_quality: f32,
    #[serde(default)]
    pub actual_answer_distinct_fraction: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct RuliadEvalFailure {
    pub oracle_hash: String,
    pub family: String,
    pub task_kind: String,
    pub expected_answer: String,
    pub actual_answer: Option<String>,
    pub reason: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RuliadEvalReport {
    pub version: u32,
    pub reasoning_score_version: u32,
    pub dataset_name: String,
    pub item_count: usize,
    pub scored_count: usize,
    pub exact_match_count: usize,
    pub semantic_match_count: usize,
    pub verifier_match_count: usize,
    pub partial_credit_count: usize,
    pub schema_valid_wrong_count: usize,
    pub malformed_completion_count: usize,
    pub missing_completion_count: usize,
    pub unexpected_completion_count: usize,
    pub exact_accuracy: f32,
    pub semantic_accuracy: f32,
    pub verifier_accuracy: f32,
    pub partial_credit_rate: f32,
    pub mean_partial_progress: f32,
    #[serde(default)]
    pub answer_field_correct_count: usize,
    #[serde(default)]
    pub answer_field_expected_count: usize,
    #[serde(default)]
    pub answer_field_accuracy: f32,
    #[serde(default)]
    pub answer_terminated_count: usize,
    #[serde(default)]
    pub answer_termination_rate: f32,
    #[serde(default)]
    pub mean_completion_quality: f32,
    #[serde(default)]
    pub actual_answer_distinct_fraction: f32,
    pub mean_certificate_prefix_coverage: f32,
    pub mean_completion_tokens: f32,
    pub canary_count: usize,
    pub canary_semantic_match_count: usize,
    pub family_scores: Vec<RuliadEvalGroupScore>,
    pub task_scores: Vec<RuliadEvalGroupScore>,
    #[serde(default)]
    pub difficulty_scores: Vec<RuliadEvalGroupScore>,
    pub math_domain_scores: Vec<RuliadEvalGroupScore>,
    pub reasoning_mode_scores: Vec<RuliadEvalGroupScore>,
    pub failures: Vec<RuliadEvalFailure>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum RuliadAnswerStatus {
    Missing,
    Malformed,
    SchemaValidWrong,
    Partial,
    SemanticMatch,
    VerifierMatch,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct RuliadReasoningScoreKey {
    pub status: RuliadAnswerStatus,
    pub partial_progress_ppm: usize,
    pub correct_field_count: usize,
    pub certificate_prefix_ppm: usize,
    pub certificate_valid_prefix_steps: usize,
    pub compactness_rank: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct RuliadReasoningScore {
    pub version: u32,
    pub status: RuliadAnswerStatus,
    pub correct_field_count: usize,
    pub expected_field_count: usize,
    pub partial_progress_ppm: usize,
    pub certificate_valid_prefix_steps: usize,
    pub certificate_expected_steps: usize,
    pub certificate_prefix_ppm: usize,
    pub generated_token_count: usize,
    pub hash_canary: bool,
    #[serde(default)]
    pub answer_terminated: bool,
    #[serde(default = "default_completion_quality_ppm")]
    pub completion_quality_ppm: usize,
}

impl RuliadReasoningScore {
    pub fn ordinal_key(&self) -> RuliadReasoningScoreKey {
        RuliadReasoningScoreKey {
            status: self.status,
            partial_progress_ppm: self.partial_progress_ppm,
            correct_field_count: self.correct_field_count,
            certificate_prefix_ppm: self.certificate_prefix_ppm,
            certificate_valid_prefix_steps: self.certificate_valid_prefix_steps,
            compactness_rank: usize::MAX.saturating_sub(self.generated_token_count),
        }
    }

    pub fn cmp_ordinal(&self, other: &Self) -> std::cmp::Ordering {
        self.ordinal_key().cmp(&other.ordinal_key())
    }

    pub fn verifier_match(&self) -> bool {
        self.status == RuliadAnswerStatus::VerifierMatch
    }

    pub fn partial_credit(&self) -> bool {
        matches!(
            self.status,
            RuliadAnswerStatus::Partial
                | RuliadAnswerStatus::SemanticMatch
                | RuliadAnswerStatus::VerifierMatch
        )
    }
}

pub fn ruliad_reasoning_rank_order(scores: &[RuliadReasoningScore]) -> Vec<usize> {
    let mut indices = (0..scores.len()).collect::<Vec<_>>();
    indices.sort_by(|left, right| {
        scores[*right]
            .cmp_ordinal(&scores[*left])
            .then_with(|| left.cmp(right))
    });
    indices
}

pub fn ruliad_reasoning_rank_fitness(scores: &[RuliadReasoningScore]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }
    if scores.len() == 1 {
        return vec![0.0];
    }
    let order = ruliad_reasoning_rank_order(scores);
    let midpoint = (scores.len() - 1) as f32 / 2.0;
    let scale = midpoint.max(1.0);
    let mut fitness = vec![0.0f32; scores.len()];
    for (rank, index) in order.into_iter().enumerate() {
        fitness[index] = (midpoint - rank as f32) / scale;
    }
    fitness
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct RuliadVerifierRewardWeights {
    pub verifier_match: f32,
    pub semantic_match: f32,
    pub partial_progress: f32,
    pub field_accuracy: f32,
    pub certificate_prefix: f32,
    pub compactness: f32,
    pub malformed_penalty: f32,
    pub missing_penalty: f32,
    pub schema_wrong_penalty: f32,
    pub hash_canary_wrong_penalty: f32,
}

impl Default for RuliadVerifierRewardWeights {
    fn default() -> Self {
        Self {
            verifier_match: 1.0,
            semantic_match: 0.85,
            partial_progress: 0.35,
            field_accuracy: 0.35,
            certificate_prefix: 0.15,
            compactness: 0.0,
            malformed_penalty: -0.35,
            missing_penalty: -0.5,
            schema_wrong_penalty: -0.15,
            hash_canary_wrong_penalty: -0.35,
        }
    }
}

pub fn ruliad_verifier_reward(
    score: &RuliadReasoningScore,
    weights: RuliadVerifierRewardWeights,
) -> f32 {
    let mut reward = match score.status {
        RuliadAnswerStatus::VerifierMatch => weights.verifier_match,
        RuliadAnswerStatus::SemanticMatch => weights.semantic_match,
        RuliadAnswerStatus::Partial => 0.0,
        RuliadAnswerStatus::SchemaValidWrong => weights.schema_wrong_penalty,
        RuliadAnswerStatus::Malformed => weights.malformed_penalty,
        RuliadAnswerStatus::Missing => weights.missing_penalty,
    };
    let partial = score.partial_progress_ppm as f32 / SCORE_PPM_DENOMINATOR as f32;
    let field_accuracy = if score.expected_field_count == 0 {
        0.0
    } else {
        score.correct_field_count as f32 / score.expected_field_count as f32
    };
    let certificate = score.certificate_prefix_ppm as f32 / SCORE_PPM_DENOMINATOR as f32;
    reward += weights.partial_progress * partial;
    reward += weights.field_accuracy * field_accuracy;
    reward += weights.certificate_prefix * certificate;
    if weights.compactness != 0.0 && score.generated_token_count > 0 {
        reward += weights.compactness / score.generated_token_count as f32;
    }
    if score.hash_canary && !score.verifier_match() {
        reward += weights.hash_canary_wrong_penalty;
    }
    reward
}

pub const RULIAD_VERIFIER_REWARD_VECTOR_DIM: usize = 10;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
pub struct RuliadVerifierRewardVector {
    pub verifier_match: f32,
    pub semantic_match: f32,
    pub partial_progress: f32,
    pub field_accuracy: f32,
    pub certificate_prefix: f32,
    pub compactness: f32,
    pub schema_quality: f32,
    pub hash_safety: f32,
    pub answer_termination: f32,
    pub completion_health: f32,
}

impl RuliadVerifierRewardVector {
    pub fn components(self) -> [f32; RULIAD_VERIFIER_REWARD_VECTOR_DIM] {
        [
            self.verifier_match,
            self.semantic_match,
            self.partial_progress,
            self.field_accuracy,
            self.certificate_prefix,
            self.compactness,
            self.schema_quality,
            self.hash_safety,
            self.answer_termination,
            self.completion_health,
        ]
    }

    pub fn scalarize(self, weights: &[f32; RULIAD_VERIFIER_REWARD_VECTOR_DIM]) -> f32 {
        self.components()
            .into_iter()
            .zip(weights.iter().copied())
            .map(|(component, weight)| component * weight)
            .sum()
    }
}

pub fn ruliad_verifier_reward_vector(score: &RuliadReasoningScore) -> RuliadVerifierRewardVector {
    let partial_progress = score.partial_progress_ppm as f32 / SCORE_PPM_DENOMINATOR as f32;
    let field_accuracy = if score.expected_field_count == 0 {
        0.0
    } else {
        score.correct_field_count as f32 / score.expected_field_count as f32
    };
    let certificate_prefix = score.certificate_prefix_ppm as f32 / SCORE_PPM_DENOMINATOR as f32;
    let schema_quality = match score.status {
        RuliadAnswerStatus::VerifierMatch
        | RuliadAnswerStatus::SemanticMatch
        | RuliadAnswerStatus::Partial => 1.0,
        RuliadAnswerStatus::SchemaValidWrong => 0.25,
        RuliadAnswerStatus::Malformed | RuliadAnswerStatus::Missing => 0.0,
    };
    let raw_completion_health = match score.status {
        RuliadAnswerStatus::VerifierMatch
        | RuliadAnswerStatus::SemanticMatch
        | RuliadAnswerStatus::Partial => 1.0,
        RuliadAnswerStatus::SchemaValidWrong
        | RuliadAnswerStatus::Malformed
        | RuliadAnswerStatus::Missing => 0.0,
    };
    let completion_quality = score.completion_quality_ppm.min(SCORE_PPM_DENOMINATOR) as f32
        / SCORE_PPM_DENOMINATOR as f32;
    let completion_health = raw_completion_health * completion_quality;
    let has_correctness_signal = matches!(
        score.status,
        RuliadAnswerStatus::VerifierMatch
            | RuliadAnswerStatus::SemanticMatch
            | RuliadAnswerStatus::Partial
    ) || partial_progress > 0.0
        || field_accuracy > 0.0
        || certificate_prefix > 0.0;
    let compactness = if score.generated_token_count == 0
        || !has_correctness_signal
        || completion_health <= 0.0
        || !score.answer_terminated
    {
        0.0
    } else {
        completion_quality / (score.generated_token_count as f32).sqrt()
    };
    RuliadVerifierRewardVector {
        verifier_match: if score.status == RuliadAnswerStatus::VerifierMatch {
            1.0
        } else {
            0.0
        },
        semantic_match: if matches!(
            score.status,
            RuliadAnswerStatus::VerifierMatch | RuliadAnswerStatus::SemanticMatch
        ) {
            1.0
        } else {
            0.0
        },
        partial_progress,
        field_accuracy,
        certificate_prefix,
        compactness,
        schema_quality,
        hash_safety: if !score.hash_canary || score.verifier_match() {
            1.0
        } else {
            0.0
        },
        answer_termination: if score.answer_terminated { 1.0 } else { 0.0 },
        completion_health,
    }
}

pub fn ruliad_vpo_independent_utilities(
    scores: &[RuliadReasoningScore],
    scalarizations: &[[f32; RULIAD_VERIFIER_REWARD_VECTOR_DIM]],
) -> Vec<f32> {
    let mut utilities = vec![0.0f32; scores.len()];
    if scores.is_empty() || scalarizations.is_empty() {
        return utilities;
    }
    let vectors = scores
        .iter()
        .map(ruliad_verifier_reward_vector)
        .collect::<Vec<_>>();
    for weights in scalarizations {
        let mut best_index = 0usize;
        let mut best_value = f32::NEG_INFINITY;
        for (index, vector) in vectors.iter().copied().enumerate() {
            let value = vector.scalarize(weights);
            if value > best_value {
                best_index = index;
                best_value = value;
            }
        }
        if best_value.is_finite() {
            utilities[best_index] += best_value;
        }
    }
    let scale = scalarizations.len() as f32;
    for utility in utilities.iter_mut() {
        *utility /= scale;
    }
    utilities
}

pub fn centered_advantages(rewards: &[f32]) -> Vec<f32> {
    if rewards.is_empty() {
        return Vec::new();
    }
    let mean = rewards.iter().copied().sum::<f32>() / rewards.len() as f32;
    rewards.iter().map(|reward| reward - mean).collect()
}

pub fn normalized_advantages(rewards: &[f32], epsilon: f32) -> Vec<f32> {
    let centered = centered_advantages(rewards);
    if centered.len() < 2 {
        return centered;
    }
    let variance = centered.iter().map(|value| value * value).sum::<f32>() / centered.len() as f32;
    let scale = variance.sqrt().max(epsilon.max(0.0));
    centered.into_iter().map(|value| value / scale).collect()
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct RuliadExtractedCompletion {
    pub answer: Option<String>,
    pub certificate_lines: Vec<String>,
    pub generated_token_count: usize,
    #[serde(default)]
    pub answer_terminated: bool,
    #[serde(default = "default_completion_quality_ppm")]
    pub completion_quality_ppm: usize,
}

fn default_completion_quality_ppm() -> usize {
    SCORE_PPM_DENOMINATOR
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub struct RuliadAnswerKeyAlignment {
    pub expected_key_count: usize,
    pub actual_key_count: usize,
    pub matching_key_count: usize,
    pub exact_key_match: bool,
    pub overlap_ppm: usize,
}

#[derive(Debug, Clone)]
struct DiagnosticSample {
    split: SampleSplit,
    family: String,
    task_kind: String,
    token_count: usize,
    serialized_char_count: usize,
    stats: SampleStats,
    spec: Option<RuliadSampleSpec>,
    oracle_hash: Option<String>,
    math_domains: Vec<String>,
    reasoning_modes: Vec<String>,
    serialized_preview: Option<String>,
    multi_chunk_document: bool,
}

#[derive(Debug, Clone, Default)]
struct EvalAccumulator {
    count: usize,
    exact_match_count: usize,
    semantic_match_count: usize,
    verifier_match_count: usize,
    partial_credit_count: usize,
    schema_valid_wrong_count: usize,
    malformed_completion_count: usize,
    missing_completion_count: usize,
    partial_progress_ppm_sum: usize,
    answer_field_correct_count: usize,
    answer_field_expected_count: usize,
    answer_terminated_count: usize,
    completion_quality_ppm_sum: usize,
    actual_answer_count: usize,
    actual_answers: BTreeSet<String>,
}

#[derive(Debug, Clone)]
struct EvalOutcome {
    exact_match: bool,
    semantic_match: bool,
    malformed: bool,
    missing: bool,
    answer_terminated: bool,
    actual_answer: Option<String>,
    reasoning_score: RuliadReasoningScore,
}

pub fn diagnose_manifest(
    manifest_path: &Path,
    thresholds: RuliadDiagnosticThresholds,
) -> Result<RuliadDiagnosticReport> {
    let manifest = load_ruliad_manifest(manifest_path)?;
    let records = read_manifest_records(manifest_path, &manifest)?;
    let document_token_count = infer_document_token_count(&manifest, &records);
    let payload_token_capacity =
        document_token_count.saturating_sub(usize::from(manifest.tokenizer.eos_id.is_some()));
    let samples = records
        .into_iter()
        .map(diagnostic_sample_from_record)
        .collect::<Result<Vec<_>>>()?;
    Ok(diagnose_samples(
        manifest.dataset_name,
        manifest.token_count,
        document_token_count,
        payload_token_capacity,
        samples,
        Vec::new(),
        thresholds,
    ))
}

pub fn diagnose_config(
    config: &RuliadCorpusConfig,
    sample_limit_per_split: usize,
    thresholds: RuliadDiagnosticThresholds,
) -> Result<RuliadDiagnosticReport> {
    let corpus = OnlineRuliadCorpus::new(config.clone())?;
    let sample_limit_per_split = sample_limit_per_split.max(1);
    let mut samples = Vec::new();
    for split in [SampleSplit::Train, SampleSplit::Validation] {
        let sample_count = corpus.sample_count(split).min(sample_limit_per_split);
        for sample_index in 0..sample_count {
            let document = corpus.generate_document(split, sample_index)?;
            samples.push(DiagnosticSample {
                split,
                family: document.family,
                task_kind: document.task_kind,
                token_count: document.token_count,
                serialized_char_count: document.serialized_preview.len(),
                stats: document.stats,
                spec: Some(document.spec),
                oracle_hash: Some(document.oracle_hash),
                math_domains: document.math_domains,
                reasoning_modes: document.reasoning_modes,
                multi_chunk_document: is_multi_chunk_document(&document.serialized_preview),
                serialized_preview: Some(document.serialized_preview),
            });
        }
    }
    let document_token_count = corpus.document_token_count();
    let payload_token_capacity = document_token_count
        .saturating_sub(usize::from(corpus.tokenizer_manifest().eos_id.is_some()));
    let token_count = samples
        .iter()
        .map(|sample| sample.token_count)
        .sum::<usize>();
    Ok(diagnose_samples(
        corpus.dataset_name().to_string(),
        token_count,
        document_token_count,
        payload_token_capacity,
        samples,
        source_bucket_diagnostics(&ruliad_source_buckets(config)),
        thresholds,
    ))
}

pub fn build_eval_items_from_manifest(
    manifest_path: &Path,
    config: &RuliadEvalConfig,
) -> Result<Vec<RuliadEvalItem>> {
    let manifest = load_ruliad_manifest(manifest_path)?;
    let records = read_manifest_records(manifest_path, &manifest)?;
    let mut items = Vec::new();
    for record in records {
        if config.split.is_some_and(|split| split != record.split) {
            continue;
        }
        if !config.include_hash_canaries && record.family == "hash_noise" {
            continue;
        }
        let Some(spec_value) = &record.ruliad_spec else {
            continue;
        };
        let Some(oracle_hash) = &record.oracle_hash else {
            continue;
        };
        let spec: RuliadSampleSpec = serde_json::from_value(spec_value.clone())
            .with_context(|| format!("parse sample {} ruliad spec", record.sample_index))?;
        let report = verify_spec(&spec)?;
        if report.oracle_hash != *oracle_hash {
            return Err(anyhow!(
                "sample {} oracle hash mismatch expected={} actual={}",
                record.sample_index,
                oracle_hash,
                report.oracle_hash
            ));
        }
        let task_kind = record
            .task_kind
            .clone()
            .unwrap_or_else(|| report.task_kind.label().to_string());
        items.push(RuliadEvalItem {
            oracle_hash: oracle_hash.clone(),
            sample_index: record.sample_index,
            split: record.split,
            family: record.family,
            task_kind,
            math_domains: record.math_domains,
            reasoning_modes: record.reasoning_modes,
            prompt: ruliad_prompt_prefix(&spec, oracle_hash),
            expected_answer: ruliad_expected_answer(&spec),
            difficulty_level: None,
            spec: Some(spec),
        });
        if config
            .max_items
            .is_some_and(|max_items| items.len() >= max_items)
        {
            break;
        }
    }
    Ok(items)
}

pub fn read_completion_records(path: &Path) -> Result<Vec<RuliadCompletionRecord>> {
    let contents =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    contents
        .lines()
        .enumerate()
        .filter_map(|(index, line)| {
            let line = line.trim();
            (!line.is_empty()).then_some((index, line))
        })
        .map(|(index, line)| {
            serde_json::from_str::<RuliadCompletionRecord>(line)
                .with_context(|| format!("failed to parse completion line {}", index + 1))
        })
        .collect()
}

pub fn write_eval_items_jsonl(path: &Path, items: &[RuliadEvalItem]) -> Result<()> {
    write_jsonl(path, items)
}

pub fn write_completion_records_jsonl(
    path: &Path,
    completions: &[RuliadCompletionRecord],
) -> Result<()> {
    write_jsonl(path, completions)
}

pub fn baseline_completions(
    items: &[RuliadEvalItem],
    baseline: RuliadEvalBaseline,
) -> Vec<RuliadCompletionRecord> {
    items
        .iter()
        .map(|item| {
            let answer = match baseline {
                RuliadEvalBaseline::Oracle => item.expected_answer.clone(),
                RuliadEvalBaseline::Corrupt => corrupt_answer(&item.expected_answer),
            };
            RuliadCompletionRecord {
                oracle_hash: item.oracle_hash.clone(),
                completion: format!("!:{answer}\n[/R2]"),
            }
        })
        .collect()
}

pub fn evaluate_completions(
    dataset_name: impl Into<String>,
    items: &[RuliadEvalItem],
    completions: &[RuliadCompletionRecord],
) -> RuliadEvalReport {
    let dataset_name = dataset_name.into();
    let mut completion_by_hash = BTreeMap::new();
    for completion in completions {
        completion_by_hash.insert(
            completion.oracle_hash.clone(),
            completion.completion.clone(),
        );
    }
    let item_hashes = items
        .iter()
        .map(|item| item.oracle_hash.as_str())
        .collect::<BTreeSet<_>>();
    let unexpected_completion_count = completions
        .iter()
        .filter(|completion| !item_hashes.contains(completion.oracle_hash.as_str()))
        .count();

    let mut family_scores = BTreeMap::<String, EvalAccumulator>::new();
    let mut task_scores = BTreeMap::<String, EvalAccumulator>::new();
    let mut difficulty_scores = BTreeMap::<String, EvalAccumulator>::new();
    let mut math_domain_scores = BTreeMap::<String, EvalAccumulator>::new();
    let mut reasoning_mode_scores = BTreeMap::<String, EvalAccumulator>::new();
    let mut exact_match_count = 0usize;
    let mut semantic_match_count = 0usize;
    let mut verifier_match_count = 0usize;
    let mut partial_credit_count = 0usize;
    let mut schema_valid_wrong_count = 0usize;
    let mut malformed_completion_count = 0usize;
    let mut missing_completion_count = 0usize;
    let mut scored_count = 0usize;
    let mut canary_count = 0usize;
    let mut canary_semantic_match_count = 0usize;
    let mut partial_progress_ppm_sum = 0usize;
    let mut answer_field_correct_count = 0usize;
    let mut answer_field_expected_count = 0usize;
    let mut answer_terminated_count = 0usize;
    let mut completion_quality_ppm_sum = 0usize;
    let mut actual_answer_count = 0usize;
    let mut actual_answers = BTreeSet::new();
    let mut certificate_prefix_ppm_sum = 0usize;
    let mut completion_token_sum = 0usize;
    let mut failures = Vec::new();

    for item in items {
        let completion = completion_by_hash
            .get(&item.oracle_hash)
            .map(String::as_str);
        let outcome = score_item(item, completion);
        scored_count += usize::from(completion.is_some());
        exact_match_count += usize::from(outcome.exact_match);
        semantic_match_count += usize::from(outcome.semantic_match);
        verifier_match_count += usize::from(outcome.reasoning_score.verifier_match());
        partial_credit_count += usize::from(outcome.reasoning_score.partial_credit());
        schema_valid_wrong_count +=
            usize::from(outcome.reasoning_score.status == RuliadAnswerStatus::SchemaValidWrong);
        malformed_completion_count += usize::from(outcome.malformed);
        missing_completion_count += usize::from(outcome.missing);
        partial_progress_ppm_sum =
            partial_progress_ppm_sum.saturating_add(outcome.reasoning_score.partial_progress_ppm);
        answer_field_correct_count =
            answer_field_correct_count.saturating_add(outcome.reasoning_score.correct_field_count);
        answer_field_expected_count = answer_field_expected_count
            .saturating_add(outcome.reasoning_score.expected_field_count);
        answer_terminated_count += usize::from(outcome.answer_terminated);
        completion_quality_ppm_sum = completion_quality_ppm_sum
            .saturating_add(outcome.reasoning_score.completion_quality_ppm);
        if let Some(actual_answer) = outcome.actual_answer.as_deref() {
            actual_answer_count = actual_answer_count.saturating_add(1);
            actual_answers.insert(actual_answer.to_string());
        }
        certificate_prefix_ppm_sum = certificate_prefix_ppm_sum
            .saturating_add(outcome.reasoning_score.certificate_prefix_ppm);
        completion_token_sum =
            completion_token_sum.saturating_add(outcome.reasoning_score.generated_token_count);
        if item.family == "hash_noise" || item.task_kind == "hash_canary" {
            canary_count += 1;
            canary_semantic_match_count += usize::from(outcome.semantic_match);
        }
        add_group_score(&mut family_scores, &item.family, &outcome);
        add_group_score(&mut task_scores, &item.task_kind, &outcome);
        if let Some(difficulty_level) = item.difficulty_level {
            add_group_score(
                &mut difficulty_scores,
                &format!("d{difficulty_level}"),
                &outcome,
            );
        }
        for domain in &item.math_domains {
            add_group_score(&mut math_domain_scores, domain, &outcome);
        }
        for mode in &item.reasoning_modes {
            add_group_score(&mut reasoning_mode_scores, mode, &outcome);
        }
        if !outcome.semantic_match && failures.len() < MAX_REPORTED_EVAL_FAILURES {
            failures.push(RuliadEvalFailure {
                oracle_hash: item.oracle_hash.clone(),
                family: item.family.clone(),
                task_kind: item.task_kind.clone(),
                expected_answer: item.expected_answer.clone(),
                actual_answer: outcome.actual_answer,
                reason: if outcome.missing {
                    "missing_completion".to_string()
                } else if outcome.malformed {
                    "malformed_completion".to_string()
                } else {
                    "answer_mismatch".to_string()
                },
            });
        }
    }

    RuliadEvalReport {
        version: RULIAD_EVAL_REPORT_VERSION,
        reasoning_score_version: RULIAD_REASONING_SCORE_VERSION,
        dataset_name,
        item_count: items.len(),
        scored_count,
        exact_match_count,
        semantic_match_count,
        verifier_match_count,
        partial_credit_count,
        schema_valid_wrong_count,
        malformed_completion_count,
        missing_completion_count,
        unexpected_completion_count,
        exact_accuracy: ratio(exact_match_count, items.len()),
        semantic_accuracy: ratio(semantic_match_count, items.len()),
        verifier_accuracy: ratio(verifier_match_count, items.len()),
        partial_credit_rate: ratio(partial_credit_count, items.len()),
        mean_partial_progress: ratio_ppm(partial_progress_ppm_sum, items.len()),
        answer_field_correct_count,
        answer_field_expected_count,
        answer_field_accuracy: ratio(answer_field_correct_count, answer_field_expected_count),
        answer_terminated_count,
        answer_termination_rate: ratio(answer_terminated_count, items.len()),
        mean_completion_quality: ratio_ppm(completion_quality_ppm_sum, items.len()),
        actual_answer_distinct_fraction: ratio(actual_answers.len(), actual_answer_count),
        mean_certificate_prefix_coverage: ratio_ppm(certificate_prefix_ppm_sum, items.len()),
        mean_completion_tokens: ratio_f32(completion_token_sum as f32, items.len()),
        canary_count,
        canary_semantic_match_count,
        family_scores: finalize_group_scores(family_scores),
        task_scores: finalize_group_scores(task_scores),
        difficulty_scores: finalize_group_scores(difficulty_scores),
        math_domain_scores: finalize_group_scores(math_domain_scores),
        reasoning_mode_scores: finalize_group_scores(reasoning_mode_scores),
        failures,
    }
}

pub fn extract_ruliad_answer(completion: &str) -> Option<String> {
    extract_ruliad_completion(completion).answer
}

pub fn extract_ruliad_completion(completion: &str) -> RuliadExtractedCompletion {
    let answer_start = completion.find("!:").map(|offset| offset + 2).unwrap_or(0);
    let completion_body = &completion[answer_start..];
    let answer_terminated =
        completion_body.contains("[/R2]") || completion_body.contains("[/RTREE]");
    let mut answer = None;
    let mut certificate_lines = Vec::new();
    for line in completion_body.lines() {
        let candidate = line
            .split("[/R2]")
            .next()
            .unwrap_or_default()
            .split("[/RTREE]")
            .next()
            .unwrap_or_default()
            .trim();
        if candidate.is_empty() {
            continue;
        }
        if answer.is_none() && !candidate.starts_with('>') {
            answer = Some(candidate.to_string());
        } else if let Some(step) = candidate.strip_prefix('>') {
            let step = step.trim();
            if !step.is_empty() {
                certificate_lines.push(step.to_string());
            }
        }
    }
    RuliadExtractedCompletion {
        answer,
        certificate_lines,
        generated_token_count: completion_body.split_whitespace().count(),
        answer_terminated,
        completion_quality_ppm: ruliad_completion_quality_ppm(completion_body),
    }
}

fn ruliad_completion_quality_ppm(completion_body: &str) -> usize {
    let body = completion_body
        .split("[/R2]")
        .next()
        .unwrap_or_default()
        .split("[/RTREE]")
        .next()
        .unwrap_or_default();
    let symbols = body
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<Vec<_>>();
    if symbols.len() < 16 {
        return SCORE_PPM_DENOMINATOR;
    }
    let period_penalty = max_char_period_fraction(&symbols, 2..=64);
    let line_penalty = repeated_line_fraction(body);
    let penalty = period_penalty.max(line_penalty);
    if penalty <= 0.45 {
        return SCORE_PPM_DENOMINATOR;
    }
    let normalized_penalty = ((penalty - 0.45) / 0.55).clamp(0.0, 1.0);
    ((1.0 - normalized_penalty) * SCORE_PPM_DENOMINATOR as f64).round() as usize
}

fn max_char_period_fraction(symbols: &[char], periods: impl IntoIterator<Item = usize>) -> f64 {
    periods
        .into_iter()
        .filter(|period| symbols.len() >= period.saturating_mul(2))
        .map(|period| char_period_fraction(symbols, period))
        .fold(0.0, f64::max)
}

fn char_period_fraction(symbols: &[char], period: usize) -> f64 {
    if period == 0 || symbols.len() < period.saturating_mul(2) {
        return 0.0;
    }
    let comparisons = symbols.len() - period;
    let matches = (period..symbols.len())
        .filter(|idx| symbols[*idx] == symbols[*idx - period])
        .count();
    f64::from(ratio(matches, comparisons))
}

fn repeated_line_fraction(body: &str) -> f64 {
    let lines = body
        .lines()
        .map(str::trim)
        .filter(|line| line.len() >= 4)
        .collect::<Vec<_>>();
    if lines.len() < 4 {
        return 0.0;
    }
    let unique = lines.iter().copied().collect::<BTreeSet<_>>().len();
    f64::from(ratio(lines.len().saturating_sub(unique), lines.len()))
}

fn extracted_expected_completion(
    spec: Option<&RuliadSampleSpec>,
    oracle_hash: &str,
) -> RuliadExtractedCompletion {
    spec.map(|spec| {
        let text = sample_text(spec, oracle_hash);
        let mut extracted = extract_ruliad_completion(&text);
        extracted.certificate_lines = extract_ruliad_proof_lines(&text);
        extracted
    })
    .unwrap_or_else(|| RuliadExtractedCompletion {
        answer: None,
        certificate_lines: Vec::new(),
        generated_token_count: 0,
        answer_terminated: false,
        completion_quality_ppm: 0,
    })
}

pub fn score_ruliad_item_completion(
    item: &RuliadEvalItem,
    completion: Option<&str>,
) -> RuliadReasoningScore {
    let expected = extracted_expected_completion(item.spec.as_ref(), &item.oracle_hash);
    score_ruliad_completion_parts(
        item.spec.as_ref(),
        &item.expected_answer,
        expected.certificate_lines.as_slice(),
        completion.map(extract_ruliad_completion),
    )
}

pub fn score_ruliad_completion(
    spec: Option<&RuliadSampleSpec>,
    expected_answer: &str,
    completion: Option<&str>,
) -> RuliadReasoningScore {
    score_ruliad_completion_parts(
        spec,
        expected_answer,
        &[],
        completion.map(extract_ruliad_completion),
    )
}

pub fn score_ruliad_answer(
    spec: Option<&RuliadSampleSpec>,
    expected_answer: &str,
    actual_answer: Option<&str>,
) -> RuliadReasoningScore {
    let completion = actual_answer.map(|answer| RuliadExtractedCompletion {
        answer: Some(answer.to_string()),
        certificate_lines: Vec::new(),
        generated_token_count: answer.split_whitespace().count(),
        answer_terminated: false,
        completion_quality_ppm: SCORE_PPM_DENOMINATOR,
    });
    score_ruliad_completion_parts(spec, expected_answer, &[], completion)
}

fn score_ruliad_completion_parts(
    spec: Option<&RuliadSampleSpec>,
    expected_answer: &str,
    expected_certificate: &[String],
    completion: Option<RuliadExtractedCompletion>,
) -> RuliadReasoningScore {
    let hash_canary = is_hash_canary_answer(expected_answer, spec);
    let Some(completion) = completion else {
        return reasoning_score(
            RuliadAnswerStatus::Missing,
            0,
            expected_answer_field_count(expected_answer),
            0,
            0,
            expected_certificate.len(),
            0,
            hash_canary,
            false,
            0,
        );
    };
    let generated_token_count = completion.generated_token_count;
    let answer_terminated = completion.answer_terminated;
    let completion_quality_ppm = completion.completion_quality_ppm;
    let Some(actual_answer) = completion.answer.as_deref() else {
        return reasoning_score(
            RuliadAnswerStatus::Malformed,
            0,
            expected_answer_field_count(expected_answer),
            0,
            0,
            expected_certificate.len(),
            generated_token_count,
            hash_canary,
            answer_terminated,
            completion_quality_ppm,
        );
    };

    let answer_score = score_answer_fields(expected_answer, actual_answer, hash_canary, spec);
    let (certificate_valid_prefix_steps, certificate_prefix_ppm) =
        score_certificate_prefix(expected_certificate, &completion.certificate_lines);
    reasoning_score(
        answer_score.status,
        answer_score.correct_field_count,
        answer_score.expected_field_count,
        answer_score.partial_progress_ppm,
        certificate_valid_prefix_steps,
        expected_certificate.len(),
        generated_token_count,
        hash_canary,
        answer_terminated,
        completion_quality_ppm,
    )
    .with_certificate_prefix_ppm(certificate_prefix_ppm)
}

#[derive(Debug, Clone, Copy)]
struct AnswerFieldScore {
    status: RuliadAnswerStatus,
    correct_field_count: usize,
    expected_field_count: usize,
    partial_progress_ppm: usize,
}

fn score_answer_fields(
    expected: &str,
    actual: &str,
    hash_canary: bool,
    spec: Option<&RuliadSampleSpec>,
) -> AnswerFieldScore {
    if ruliad_answers_semantic_match(expected, actual) {
        let status = if spec.is_some_and(|spec| verify_spec(spec).is_ok_and(|report| report.ok)) {
            RuliadAnswerStatus::VerifierMatch
        } else {
            RuliadAnswerStatus::SemanticMatch
        };
        let expected_field_count = expected_answer_field_count(expected).max(1);
        return AnswerFieldScore {
            status,
            correct_field_count: expected_field_count,
            expected_field_count,
            partial_progress_ppm: SCORE_PPM_DENOMINATOR,
        };
    }

    match (
        parse_answer_pairs(expected),
        parse_answer_pairs_or_contract_values(expected, actual),
    ) {
        (Some(expected_pairs), Some(actual_pairs)) => {
            let expected_field_count = expected_pairs.len().max(1);
            let correct_field_count = expected_pairs
                .iter()
                .filter(|(key, value)| {
                    actual_pairs
                        .get(*key)
                        .is_some_and(|actual| actual == *value)
                })
                .count();
            let partial_progress_ppm = if hash_canary {
                0
            } else {
                correct_field_count.saturating_mul(SCORE_PPM_DENOMINATOR) / expected_field_count
            };
            let has_expected_schema = actual_pairs
                .keys()
                .any(|key| expected_pairs.contains_key(key));
            let status = if !hash_canary && correct_field_count > 0 {
                RuliadAnswerStatus::Partial
            } else if has_expected_schema || !actual_pairs.is_empty() {
                RuliadAnswerStatus::SchemaValidWrong
            } else {
                RuliadAnswerStatus::Malformed
            };
            AnswerFieldScore {
                status,
                correct_field_count,
                expected_field_count,
                partial_progress_ppm,
            }
        }
        (Some(expected_pairs), None) => AnswerFieldScore {
            status: RuliadAnswerStatus::Malformed,
            correct_field_count: 0,
            expected_field_count: expected_pairs.len().max(1),
            partial_progress_ppm: 0,
        },
        _ => {
            let expected_normalized = normalize_answer(expected);
            let actual_normalized = normalize_answer(actual);
            let partial_progress_ppm = if hash_canary {
                0
            } else {
                common_prefix_chars(&expected_normalized, &actual_normalized)
                    .saturating_mul(SCORE_PPM_DENOMINATOR)
                    / expected_normalized.chars().count().max(1)
            };
            AnswerFieldScore {
                status: if partial_progress_ppm > 0 {
                    RuliadAnswerStatus::Partial
                } else {
                    RuliadAnswerStatus::SchemaValidWrong
                },
                correct_field_count: usize::from(partial_progress_ppm > 0),
                expected_field_count: 1,
                partial_progress_ppm,
            }
        }
    }
}

fn score_certificate_prefix(expected: &[String], actual: &[String]) -> (usize, usize) {
    if expected.is_empty() {
        return (0, 0);
    }
    let valid_prefix_steps = expected
        .iter()
        .zip(actual.iter())
        .take_while(|(expected, actual)| normalize_answer(expected) == normalize_answer(actual))
        .count();
    (
        valid_prefix_steps,
        valid_prefix_steps.saturating_mul(SCORE_PPM_DENOMINATOR) / expected.len().max(1),
    )
}

fn extract_ruliad_proof_lines(document: &str) -> Vec<String> {
    document
        .lines()
        .take_while(|line| !line.trim_start().starts_with("!:"))
        .filter_map(|line| {
            let step = line.trim().strip_prefix('>')?.trim();
            (!step.is_empty()).then_some(step.to_string())
        })
        .collect()
}

fn reasoning_score(
    status: RuliadAnswerStatus,
    correct_field_count: usize,
    expected_field_count: usize,
    partial_progress_ppm: usize,
    certificate_valid_prefix_steps: usize,
    certificate_expected_steps: usize,
    generated_token_count: usize,
    hash_canary: bool,
    answer_terminated: bool,
    completion_quality_ppm: usize,
) -> RuliadReasoningScore {
    let certificate_prefix_ppm = if certificate_expected_steps == 0 {
        0
    } else {
        certificate_valid_prefix_steps.saturating_mul(SCORE_PPM_DENOMINATOR)
            / certificate_expected_steps
    };
    RuliadReasoningScore {
        version: RULIAD_REASONING_SCORE_VERSION,
        status,
        correct_field_count,
        expected_field_count,
        partial_progress_ppm,
        certificate_valid_prefix_steps,
        certificate_expected_steps,
        certificate_prefix_ppm,
        generated_token_count,
        hash_canary,
        answer_terminated,
        completion_quality_ppm,
    }
}

impl RuliadReasoningScore {
    fn with_certificate_prefix_ppm(mut self, certificate_prefix_ppm: usize) -> Self {
        self.certificate_prefix_ppm = certificate_prefix_ppm;
        self
    }
}

fn expected_answer_field_count(expected: &str) -> usize {
    parse_answer_pairs(expected)
        .map(|pairs| pairs.len())
        .unwrap_or(1)
}

fn is_hash_canary_answer(expected: &str, spec: Option<&RuliadSampleSpec>) -> bool {
    matches!(spec, Some(RuliadSampleSpec::HashNoise { .. }))
        || matches!(spec, Some(RuliadSampleSpec::LeanTask { .. }))
        || parse_answer_pairs(expected)
            .is_some_and(|pairs| pairs.len() == 1 && pairs.contains_key("sha"))
}

fn common_prefix_chars(left: &str, right: &str) -> usize {
    left.chars()
        .zip(right.chars())
        .take_while(|(left, right)| left == right)
        .count()
}

/* old implementation retained above through extract_ruliad_completion */
#[allow(dead_code)]
fn _legacy_extract_ruliad_answer(completion: &str) -> Option<String> {
    let answer_start = completion.find("!:").map(|offset| offset + 2).unwrap_or(0);
    completion[answer_start..]
        .lines()
        .filter_map(|line| {
            let candidate = line
                .split("[/R2]")
                .next()
                .unwrap_or_default()
                .split("[/RTREE]")
                .next()
                .unwrap_or_default()
                .trim();
            (!candidate.is_empty()).then_some(candidate.to_string())
        })
        .next()
}

pub fn ruliad_answers_exact_match(expected: &str, actual: &str) -> bool {
    normalize_answer(expected) == normalize_answer(actual)
}

pub fn ruliad_answers_semantic_match(expected: &str, actual: &str) -> bool {
    if ruliad_answers_exact_match(expected, actual) {
        return true;
    }
    match (
        parse_answer_pairs(expected),
        parse_answer_pairs_or_contract_values(expected, actual),
    ) {
        (Some(left), Some(right)) => left == right,
        _ => false,
    }
}

pub fn ruliad_answer_key_alignment(
    expected: &str,
    actual: Option<&str>,
) -> RuliadAnswerKeyAlignment {
    let expected_keys = parse_answer_pairs(expected)
        .map(|pairs| pairs.into_keys().collect::<BTreeSet<_>>())
        .unwrap_or_else(|| BTreeSet::from(["value".to_string()]));
    let actual_keys = actual
        .and_then(|actual| parse_answer_pairs_or_contract_values(expected, actual))
        .map(|pairs| pairs.into_keys().collect::<BTreeSet<_>>())
        .unwrap_or_else(|| {
            actual
                .filter(|value| !normalize_answer(value).is_empty())
                .map(|_| BTreeSet::from(["value".to_string()]))
                .unwrap_or_default()
        });
    let matching_key_count = expected_keys.intersection(&actual_keys).count();
    let denominator = expected_keys.len().max(actual_keys.len()).max(1);
    RuliadAnswerKeyAlignment {
        expected_key_count: expected_keys.len(),
        actual_key_count: actual_keys.len(),
        matching_key_count,
        exact_key_match: expected_keys == actual_keys,
        overlap_ppm: matching_key_count.saturating_mul(SCORE_PPM_DENOMINATOR) / denominator,
    }
}

fn diagnose_samples(
    dataset_name: String,
    token_count: usize,
    document_token_count: usize,
    payload_token_capacity: usize,
    samples: Vec<DiagnosticSample>,
    source_bucket_priors: Vec<RuliadSourceBucketDiagnostic>,
    thresholds: RuliadDiagnosticThresholds,
) -> RuliadDiagnosticReport {
    let mut split_counts = BTreeMap::<String, usize>::new();
    let mut family_counts = BTreeMap::<String, usize>::new();
    let mut task_counts = BTreeMap::<String, usize>::new();
    let mut math_domain_counts = BTreeMap::<String, usize>::new();
    let mut reasoning_mode_counts = BTreeMap::<String, usize>::new();
    let mut oracle_hash_counts = BTreeMap::<String, usize>::new();
    let mut missing_ruliad_spec_count = 0usize;
    let mut missing_oracle_hash_count = 0usize;
    let mut verifier_failure_count = 0usize;
    let mut answer_slot_count = 0usize;
    let mut proof_trace_count = 0usize;
    let mut degenerate_sample_count = 0usize;
    let mut multi_chunk_document_count = 0usize;
    let mut categorical_core_count = 0usize;
    let mut hash_canary_count = 0usize;
    let mut token_count_drift_count = 0usize;
    let mut payload_overflow_count = 0usize;
    let mut max_serialized_char_count = 0usize;
    let mut gzip_sum = 0.0f32;
    let mut complexity_sum = 0.0f32;

    for sample in &samples {
        *split_counts
            .entry(split_label(sample.split).to_string())
            .or_insert(0) += 1;
        *family_counts.entry(sample.family.clone()).or_insert(0) += 1;
        *task_counts.entry(sample.task_kind.clone()).or_insert(0) += 1;
        for domain in &sample.math_domains {
            *math_domain_counts.entry(domain.clone()).or_insert(0) += 1;
        }
        for mode in &sample.reasoning_modes {
            *reasoning_mode_counts.entry(mode.clone()).or_insert(0) += 1;
        }
        if sample.family == "hash_noise" || sample.task_kind == "hash_canary" {
            hash_canary_count += 1;
        }
        if sample.token_count != document_token_count {
            token_count_drift_count += 1;
        }
        max_serialized_char_count = max_serialized_char_count.max(sample.serialized_char_count);
        gzip_sum += sample.stats.gzip_complexity_ratio;
        complexity_sum += sample.stats.complexity_score;

        let Some(spec) = &sample.spec else {
            missing_ruliad_spec_count += 1;
            continue;
        };
        let Some(oracle_hash) = &sample.oracle_hash else {
            missing_oracle_hash_count += 1;
            continue;
        };
        *oracle_hash_counts.entry(oracle_hash.clone()).or_insert(0) += 1;
        if let Ok(report) = verify_spec(spec) {
            if !report.ok || report.oracle_hash != *oracle_hash {
                verifier_failure_count += 1;
            }
        } else {
            verifier_failure_count += 1;
        }
        degenerate_sample_count += usize::from(is_degenerate_spec(spec));
        let expected_answer = ruliad_expected_answer(spec);
        answer_slot_count += usize::from(!expected_answer.trim().is_empty());
        let text = sample
            .serialized_preview
            .clone()
            .unwrap_or_else(|| sample_text(spec, oracle_hash));
        if text.len() > payload_token_capacity {
            payload_overflow_count += 1;
        }
        if text.lines().any(|line| line.starts_with('>')) {
            proof_trace_count += 1;
        }
        multi_chunk_document_count +=
            usize::from(sample.multi_chunk_document || is_multi_chunk_document(&text));
        let view = ruliad_categorical_presentation(spec);
        categorical_core_count += usize::from(view.categorical_core);
    }

    let oracle_hash_count = oracle_hash_counts.len();
    let duplicate_oracle_hash_count = oracle_hash_counts
        .values()
        .map(|count| count.saturating_sub(1))
        .sum::<usize>();
    let duplicate_oracle_hash_rate = ratio(
        duplicate_oracle_hash_count,
        oracle_hash_counts.values().sum::<usize>(),
    );

    let mut gate_failures = Vec::new();
    if missing_ruliad_spec_count > 0 {
        gate_failures.push(format!(
            "missing_ruliad_spec_count={missing_ruliad_spec_count}"
        ));
    }
    if missing_oracle_hash_count > 0 {
        gate_failures.push(format!(
            "missing_oracle_hash_count={missing_oracle_hash_count}"
        ));
    }
    if verifier_failure_count > 0 {
        gate_failures.push(format!("verifier_failure_count={verifier_failure_count}"));
    }
    if token_count_drift_count > 0 {
        gate_failures.push(format!("token_count_drift_count={token_count_drift_count}"));
    }
    if degenerate_sample_count > 0 {
        gate_failures.push(format!("degenerate_sample_count={degenerate_sample_count}"));
    }
    if payload_overflow_count > 0 {
        gate_failures.push(format!("payload_overflow_count={payload_overflow_count}"));
    }
    if duplicate_oracle_hash_rate > thresholds.max_duplicate_oracle_hash_rate {
        gate_failures.push(format!(
            "duplicate_oracle_hash_rate={duplicate_oracle_hash_rate:.6}"
        ));
    }
    if thresholds.require_all_semantics {
        record_missing_required_domains(
            &math_domain_counts,
            &source_bucket_priors,
            &mut gate_failures,
        );
        record_missing_required_modes(
            &reasoning_mode_counts,
            &source_bucket_priors,
            &mut gate_failures,
        );
    }
    if thresholds.min_task_share > 0.0 {
        for task in count_shares(&task_counts, samples.len()) {
            if task.share < thresholds.min_task_share {
                gate_failures.push(format!(
                    "task_share_below_min {}={:.6}",
                    task.label, task.share
                ));
            }
        }
    }

    RuliadDiagnosticReport {
        version: RULIAD_DIAGNOSTIC_REPORT_VERSION,
        dataset_name,
        sample_count: samples.len(),
        token_count,
        document_token_count,
        payload_token_capacity,
        split_counts: count_shares(&split_counts, samples.len()),
        family_counts: count_shares(&family_counts, samples.len()),
        task_counts: count_shares(&task_counts, samples.len()),
        math_domain_counts: count_shares(&math_domain_counts, samples.len()),
        reasoning_mode_counts: count_shares(&reasoning_mode_counts, samples.len()),
        source_bucket_priors,
        oracle_hash_count,
        duplicate_oracle_hash_count,
        duplicate_oracle_hash_rate,
        missing_ruliad_spec_count,
        missing_oracle_hash_count,
        verifier_failure_count,
        answer_slot_count,
        answer_slot_coverage: ratio(answer_slot_count, samples.len()),
        proof_trace_count,
        proof_trace_coverage: ratio(proof_trace_count, samples.len()),
        degenerate_sample_count,
        multi_chunk_document_count,
        multi_chunk_document_coverage: ratio(multi_chunk_document_count, samples.len()),
        categorical_core_count,
        hash_canary_count,
        token_count_drift_count,
        payload_overflow_count,
        max_serialized_char_count,
        mean_gzip_complexity_ratio: ratio_f32(gzip_sum, samples.len()),
        mean_complexity_score: ratio_f32(complexity_sum, samples.len()),
        gate_failures,
    }
}

fn diagnostic_sample_from_record(record: UniversalitySampleRecord) -> Result<DiagnosticSample> {
    let spec = record
        .ruliad_spec
        .map(serde_json::from_value)
        .transpose()
        .with_context(|| format!("parse sample {} ruliad spec", record.sample_index))?;
    Ok(DiagnosticSample {
        split: record.split,
        family: record.family,
        task_kind: record.task_kind.unwrap_or(record.complexity_band),
        token_count: record.token_count,
        serialized_char_count: record.serialized_char_count,
        stats: record.stats,
        spec,
        oracle_hash: record.oracle_hash,
        math_domains: record.math_domains,
        reasoning_modes: record.reasoning_modes,
        multi_chunk_document: record
            .ruliad_document_mode
            .as_deref()
            .is_some_and(|mode| mode == "multi_chunk_proof_tree")
            || record.ruliad_node_count.is_some_and(|count| count > 1),
        serialized_preview: None,
    })
}

fn score_item(item: &RuliadEvalItem, completion: Option<&str>) -> EvalOutcome {
    let reasoning_score = score_ruliad_item_completion(item, completion);
    let Some(completion) = completion else {
        return EvalOutcome {
            exact_match: false,
            semantic_match: false,
            malformed: false,
            missing: true,
            answer_terminated: false,
            actual_answer: None,
            reasoning_score,
        };
    };
    let extracted = extract_ruliad_completion(completion);
    let answer_terminated = extracted.answer_terminated;
    let actual_answer = extracted.answer;
    let Some(actual) = actual_answer.as_deref() else {
        return EvalOutcome {
            exact_match: false,
            semantic_match: false,
            malformed: true,
            missing: false,
            answer_terminated,
            actual_answer,
            reasoning_score,
        };
    };
    let exact_match = ruliad_answers_exact_match(&item.expected_answer, actual);
    let semantic_match = ruliad_answers_semantic_match(&item.expected_answer, actual);
    EvalOutcome {
        exact_match,
        semantic_match,
        malformed: false,
        missing: false,
        answer_terminated,
        actual_answer,
        reasoning_score,
    }
}

fn add_group_score(
    scores: &mut BTreeMap<String, EvalAccumulator>,
    label: &str,
    outcome: &EvalOutcome,
) {
    let score = scores.entry(label.to_string()).or_default();
    score.count += 1;
    score.exact_match_count += usize::from(outcome.exact_match);
    score.semantic_match_count += usize::from(outcome.semantic_match);
    score.verifier_match_count += usize::from(outcome.reasoning_score.verifier_match());
    score.partial_credit_count += usize::from(outcome.reasoning_score.partial_credit());
    score.schema_valid_wrong_count +=
        usize::from(outcome.reasoning_score.status == RuliadAnswerStatus::SchemaValidWrong);
    score.malformed_completion_count += usize::from(outcome.malformed);
    score.missing_completion_count += usize::from(outcome.missing);
    score.partial_progress_ppm_sum = score
        .partial_progress_ppm_sum
        .saturating_add(outcome.reasoning_score.partial_progress_ppm);
    score.answer_field_correct_count = score
        .answer_field_correct_count
        .saturating_add(outcome.reasoning_score.correct_field_count);
    score.answer_field_expected_count = score
        .answer_field_expected_count
        .saturating_add(outcome.reasoning_score.expected_field_count);
    score.answer_terminated_count += usize::from(outcome.answer_terminated);
    score.completion_quality_ppm_sum = score
        .completion_quality_ppm_sum
        .saturating_add(outcome.reasoning_score.completion_quality_ppm);
    if let Some(actual_answer) = outcome.actual_answer.as_deref() {
        score.actual_answer_count = score.actual_answer_count.saturating_add(1);
        score.actual_answers.insert(actual_answer.to_string());
    }
}

fn finalize_group_scores(scores: BTreeMap<String, EvalAccumulator>) -> Vec<RuliadEvalGroupScore> {
    scores
        .into_iter()
        .map(|(label, score)| RuliadEvalGroupScore {
            label,
            count: score.count,
            exact_match_count: score.exact_match_count,
            semantic_match_count: score.semantic_match_count,
            verifier_match_count: score.verifier_match_count,
            partial_credit_count: score.partial_credit_count,
            schema_valid_wrong_count: score.schema_valid_wrong_count,
            malformed_completion_count: score.malformed_completion_count,
            missing_completion_count: score.missing_completion_count,
            exact_accuracy: ratio(score.exact_match_count, score.count),
            semantic_accuracy: ratio(score.semantic_match_count, score.count),
            verifier_accuracy: ratio(score.verifier_match_count, score.count),
            partial_credit_rate: ratio(score.partial_credit_count, score.count),
            mean_partial_progress: ratio_ppm(score.partial_progress_ppm_sum, score.count),
            answer_field_correct_count: score.answer_field_correct_count,
            answer_field_expected_count: score.answer_field_expected_count,
            answer_field_accuracy: ratio(
                score.answer_field_correct_count,
                score.answer_field_expected_count,
            ),
            answer_terminated_count: score.answer_terminated_count,
            answer_termination_rate: ratio(score.answer_terminated_count, score.count),
            mean_completion_quality: ratio_ppm(score.completion_quality_ppm_sum, score.count),
            actual_answer_distinct_fraction: ratio(
                score.actual_answers.len(),
                score.actual_answer_count,
            ),
        })
        .collect()
}

fn load_ruliad_manifest(path: &Path) -> Result<UniversalityCorpusManifest> {
    let manifest = load_manifest(path)?;
    if manifest.corpus_kind != CorpusKind::Ruliad {
        return Err(anyhow!(
            "manifest {} is {:?}, not ruliad",
            path.display(),
            manifest.corpus_kind
        ));
    }
    Ok(manifest)
}

fn read_manifest_records(
    manifest_path: &Path,
    manifest: &UniversalityCorpusManifest,
) -> Result<Vec<UniversalitySampleRecord>> {
    let manifest_dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let records_path = manifest_dir.join(&manifest.sample_records_path);
    let contents = fs::read_to_string(&records_path)
        .with_context(|| format!("failed to read {}", records_path.display()))?;
    contents
        .lines()
        .enumerate()
        .filter_map(|(index, line)| {
            let line = line.trim();
            (!line.is_empty()).then_some((index, line))
        })
        .map(|(index, line)| {
            serde_json::from_str::<UniversalitySampleRecord>(line)
                .with_context(|| format!("failed to parse sample record line {}", index + 1))
        })
        .collect()
}

fn infer_document_token_count(
    manifest: &UniversalityCorpusManifest,
    records: &[UniversalitySampleRecord],
) -> usize {
    if let Some(document_tokens) = manifest
        .train_token_count
        .checked_div(manifest.stats.train_samples)
        && document_tokens > 0
    {
        return document_tokens;
    }
    if let Some(document_tokens) = manifest
        .val_token_count
        .checked_div(manifest.stats.validation_samples)
        && document_tokens > 0
    {
        return document_tokens;
    }
    records
        .first()
        .map(|record| record.token_count)
        .unwrap_or_default()
}

fn source_bucket_diagnostics(buckets: &[RuliadSourceBucket]) -> Vec<RuliadSourceBucketDiagnostic> {
    buckets
        .iter()
        .map(|bucket| {
            let semantics = bucket.semantics();
            RuliadSourceBucketDiagnostic {
                bucket_id: bucket.label(),
                family: bucket.id.family.label().to_string(),
                task_kind: bucket.id.task_kind.label().to_string(),
                prior: bucket.prior,
                math_domains: semantics
                    .math_domains
                    .iter()
                    .map(|domain| domain.label().to_string())
                    .collect(),
                reasoning_modes: semantics
                    .reasoning_modes
                    .iter()
                    .map(|mode| mode.label().to_string())
                    .collect(),
            }
        })
        .collect()
}

fn write_jsonl<T: Serialize>(path: &Path, values: &[T]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let mut out = String::new();
    for value in values {
        out.push_str(&serde_json::to_string(value).context("serialize jsonl value")?);
        out.push('\n');
    }
    fs::write(path, out).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

fn count_shares(counts: &BTreeMap<String, usize>, total: usize) -> Vec<RuliadCountShare> {
    counts
        .iter()
        .map(|(label, count)| RuliadCountShare {
            label: label.clone(),
            count: *count,
            share: ratio(*count, total),
        })
        .collect()
}

fn record_missing_required_domains(
    counts: &BTreeMap<String, usize>,
    source_bucket_priors: &[RuliadSourceBucketDiagnostic],
    gate_failures: &mut Vec<String>,
) {
    for domain in required_math_domain_labels(source_bucket_priors) {
        if !counts.contains_key(domain.as_str()) {
            gate_failures.push(format!("missing_math_domain={domain}"));
        }
    }
}

fn record_missing_required_modes(
    counts: &BTreeMap<String, usize>,
    source_bucket_priors: &[RuliadSourceBucketDiagnostic],
    gate_failures: &mut Vec<String>,
) {
    for mode in required_reasoning_mode_labels(source_bucket_priors) {
        if !counts.contains_key(mode.as_str()) {
            gate_failures.push(format!("missing_reasoning_mode={mode}"));
        }
    }
}

fn required_math_domain_labels(
    source_bucket_priors: &[RuliadSourceBucketDiagnostic],
) -> BTreeSet<String> {
    let labels = source_bucket_priors
        .iter()
        .flat_map(|bucket| bucket.math_domains.iter().cloned())
        .collect::<BTreeSet<_>>();
    if labels.is_empty() {
        RULIAD_REQUIRED_MATH_DOMAINS
            .iter()
            .map(|domain| domain.label().to_string())
            .collect()
    } else {
        labels
    }
}

fn required_reasoning_mode_labels(
    source_bucket_priors: &[RuliadSourceBucketDiagnostic],
) -> BTreeSet<String> {
    let labels = source_bucket_priors
        .iter()
        .flat_map(|bucket| bucket.reasoning_modes.iter().cloned())
        .collect::<BTreeSet<_>>();
    if labels.is_empty() {
        RULIAD_REQUIRED_REASONING_MODES
            .iter()
            .map(|mode| mode.label().to_string())
            .collect()
    } else {
        labels
    }
}

fn corrupt_answer(answer: &str) -> String {
    if answer.contains("true") {
        answer.replacen("true", "false", 1)
    } else if answer.contains("false") {
        answer.replacen("false", "true", 1)
    } else if answer.is_empty() {
        "corrupt".to_string()
    } else {
        format!("{answer}_corrupt")
    }
}

fn normalize_answer(value: &str) -> String {
    value
        .trim()
        .trim_end_matches("[/R2]")
        .trim_end_matches("[/T]")
        .trim_end_matches("[/RTREE]")
        .trim()
        .to_string()
}

fn is_multi_chunk_document(text: &str) -> bool {
    text.contains("[T ") || text.contains("[RTREE") || ruliad_serialized_node_count(text) > 1
}

fn parse_answer_pairs(value: &str) -> Option<BTreeMap<String, String>> {
    let pairs = parse_answer_pair_sequence(value)?;
    Some(pairs.into_iter().collect())
}

fn parse_answer_pairs_or_contract_values(
    expected: &str,
    actual: &str,
) -> Option<BTreeMap<String, String>> {
    if let Some(pairs) = parse_answer_pairs(actual) {
        return Some(pairs);
    }
    let expected_fields = parse_answer_pair_sequence(expected)?;
    let normalized = normalize_answer(actual);
    if normalized.is_empty() || normalized.contains('=') {
        return None;
    }
    let values = normalized.split(';').map(str::trim).collect::<Vec<_>>();
    if values.len() != expected_fields.len()
        || values
            .iter()
            .any(|value| value.is_empty() || value.chars().any(char::is_whitespace))
    {
        return None;
    }
    Some(
        expected_fields
            .into_iter()
            .zip(values)
            .map(|((key, _expected_value), actual_value)| (key, normalize_pair_value(actual_value)))
            .collect(),
    )
}

fn parse_answer_pair_sequence(value: &str) -> Option<Vec<(String, String)>> {
    let normalized = normalize_answer(value);
    let mut pairs = Vec::new();
    for part in normalized.split(';') {
        let (key, value) = part.split_once('=')?;
        let key = key.trim();
        if key.is_empty() {
            return None;
        }
        pairs.push((
            normalize_pair_key(key).to_string(),
            normalize_pair_value(value),
        ));
    }
    (!pairs.is_empty()).then_some(pairs)
}

fn normalize_pair_key(value: &str) -> &str {
    match value.trim() {
        "accepted" => "acc",
        "commutes" | "holds" => "ok",
        "lhs" => "l",
        "normal_form" => "nf",
        "payload_hash" => "sha",
        "rhs" => "r",
        "target" => "x",
        other => other,
    }
}

fn normalize_pair_value(value: &str) -> String {
    match value.trim() {
        "1" | "true" | "True" | "TRUE" => "1".to_string(),
        "0" | "false" | "False" | "FALSE" => "0".to_string(),
        other => other.to_string(),
    }
}

fn split_label(split: SampleSplit) -> &'static str {
    match split {
        SampleSplit::Train => "train",
        SampleSplit::Validation => "validation",
    }
}

fn ratio(numerator: usize, denominator: usize) -> f32 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f32 / denominator as f32
    }
}

fn ratio_f32(numerator: f32, denominator: usize) -> f32 {
    if denominator == 0 {
        0.0
    } else {
        numerator / denominator as f32
    }
}

fn ratio_ppm(numerator_ppm: usize, denominator: usize) -> f32 {
    if denominator == 0 {
        0.0
    } else {
        numerator_ppm as f32 / denominator as f32 / SCORE_PPM_DENOMINATOR as f32
    }
}

fn default_require_all_semantics() -> bool {
    true
}

fn default_eval_split() -> Option<SampleSplit> {
    Some(SampleSplit::Validation)
}

fn default_include_hash_canaries() -> bool {
    true
}

#[allow(dead_code)]
fn _assert_required_label_types(_: RuliadMathDomain, _: RuliadReasoningMode) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::UsizeRangeConfig;
    use crate::ruliad::config::{
        RuliadDocumentMode, RuliadFamilyConfig, RuliadFamilyKind, RuliadSerializationConfig,
        RuliadSourceSelectionConfig, RuliadTokenizationConfig, default_ruliad_families,
    };
    use crate::ruliad::generate::generate_ruliad_corpus;
    use tempfile::tempdir;

    fn test_config() -> RuliadCorpusConfig {
        RuliadCorpusConfig {
            output_dir: "target/ruliad-eval-test".into(),
            seed: 77,
            name: "ruliad-eval-test".to_string(),
            train_samples: 96,
            validation_samples: 32,
            chunk_token_capacity: 8192,
            serialization: RuliadSerializationConfig {
                document_tokens: 513,
                preview_samples: 2,
                ..RuliadSerializationConfig::default()
            },
            tokenization: RuliadTokenizationConfig::default(),
            source_selection: RuliadSourceSelectionConfig::default(),
            families: default_ruliad_families(),
            proof_tasks: None,
            lean_task_limit: None,
        }
    }

    #[test]
    fn answer_extraction_handles_full_document_and_answer_only() {
        assert_eq!(
            extract_ruliad_answer("!:holds=true;rhs=1;lhs=1\n[/R2]"),
            Some("holds=true;rhs=1;lhs=1".to_string())
        );
        assert_eq!(
            extract_ruliad_answer("holds=true;rhs=1;lhs=1"),
            Some("holds=true;rhs=1;lhs=1".to_string())
        );
        assert_eq!(
            extract_ruliad_answer("!:\nholds=true;rhs=1;lhs=1"),
            Some("holds=true;rhs=1;lhs=1".to_string())
        );
        assert_eq!(
            extract_ruliad_answer("[R2 h=x]\n!:holds=true;rhs=1;lhs=1\n[/R2]"),
            Some("holds=true;rhs=1;lhs=1".to_string())
        );
        assert!(ruliad_answers_semantic_match(
            "holds=true;lhs=1;rhs=1",
            "rhs=1;holds=TRUE;lhs=1"
        ));
        assert!(ruliad_answers_semantic_match(
            "ok=1;l=1;r=1",
            "rhs=1;holds=TRUE;lhs=1"
        ));
        assert!(ruliad_answers_semantic_match("ok=1;l=1;r=1", "1;1;1"));
        assert!(ruliad_answers_semantic_match("ok=0", "0"));
        assert!(ruliad_answers_semantic_match("acc=0", "accepted=false"));
    }

    #[test]
    fn answer_key_alignment_detects_wrong_family_schema() {
        let exact = ruliad_answer_key_alignment("ok=1;l=3;r=3", Some("holds=false;rhs=7;lhs=2"));
        assert!(exact.exact_key_match);
        assert_eq!(exact.matching_key_count, 3);
        assert_eq!(exact.overlap_ppm, SCORE_PPM_DENOMINATOR);

        let wrong_family =
            ruliad_answer_key_alignment("x=b128:h923eef785cae9cd9:w63", Some("ok=1;l=0;r=1"));
        assert!(!wrong_family.exact_key_match);
        assert_eq!(wrong_family.matching_key_count, 0);
        assert_eq!(wrong_family.overlap_ppm, 0);

        let partial = ruliad_answer_key_alignment("ok=1;l=3;r=3", Some("ok=0"));
        assert!(!partial.exact_key_match);
        assert_eq!(partial.matching_key_count, 1);
        assert_eq!(partial.overlap_ppm, SCORE_PPM_DENOMINATOR / 3);

        let contract_values = ruliad_answer_key_alignment("ok=1;l=3;r=3", Some("1;3;3"));
        assert!(contract_values.exact_key_match);
        assert_eq!(contract_values.matching_key_count, 3);
    }

    #[test]
    fn ordinal_reasoning_score_tracks_partial_structured_answers() {
        let exact = score_ruliad_answer(None, "ok=1;l=3;r=3", Some("holds=true;rhs=3;lhs=3"));
        let contract_exact = score_ruliad_answer(None, "ok=1;l=3;r=3", Some("1;3;3"));
        let partial = score_ruliad_answer(None, "ok=1;l=3;r=3", Some("ok=1;l=2;r=7"));
        let wrong_schema = score_ruliad_answer(None, "ok=1;l=3;r=3", Some("ok=0;l=2;r=7"));
        let malformed = score_ruliad_answer(None, "ok=1;l=3;r=3", Some("not an answer"));
        let missing = score_ruliad_answer(None, "ok=1;l=3;r=3", None);

        assert_eq!(exact.status, RuliadAnswerStatus::SemanticMatch);
        assert_eq!(contract_exact.status, RuliadAnswerStatus::SemanticMatch);
        assert_eq!(partial.status, RuliadAnswerStatus::Partial);
        assert_eq!(partial.correct_field_count, 1);
        assert_eq!(partial.partial_progress_ppm, SCORE_PPM_DENOMINATOR / 3);
        assert_eq!(wrong_schema.status, RuliadAnswerStatus::SchemaValidWrong);
        assert_eq!(malformed.status, RuliadAnswerStatus::Malformed);
        assert_eq!(missing.status, RuliadAnswerStatus::Missing);
        assert!(exact.cmp_ordinal(&partial).is_gt());
        assert!(partial.cmp_ordinal(&wrong_schema).is_gt());
        assert!(wrong_schema.cmp_ordinal(&malformed).is_gt());
        assert!(malformed.cmp_ordinal(&missing).is_gt());
    }

    #[test]
    fn hash_canary_answers_do_not_receive_prefix_partial_credit() {
        let score = score_ruliad_answer(None, "sha=abcdef0123456789", Some("sha=abcdef9999999999"));
        assert_eq!(score.status, RuliadAnswerStatus::SchemaValidWrong);
        assert_eq!(score.partial_progress_ppm, 0);
        assert!(score.hash_canary);
    }

    #[test]
    fn certificate_prefix_is_scored_after_answer_without_imitation_loss_contract() {
        let dir = tempdir().expect("tempdir");
        let mut config = test_config();
        config.output_dir = dir.path().join("out");
        let report = generate_ruliad_corpus(&config).expect("generate");
        let items = build_eval_items_from_manifest(
            &report.manifest_path,
            &RuliadEvalConfig {
                max_items: Some(1),
                ..RuliadEvalConfig::default()
            },
        )
        .expect("items");
        let item = items.first().expect("item");
        let expected = extracted_expected_completion(item.spec.as_ref(), &item.oracle_hash);
        assert!(!expected.certificate_lines.is_empty());

        let completion = format!(
            "!:{}\n>{}\n>bad-step",
            item.expected_answer, expected.certificate_lines[0]
        );
        let score = score_ruliad_item_completion(item, Some(&completion));
        assert_eq!(score.status, RuliadAnswerStatus::VerifierMatch);
        assert_eq!(score.certificate_valid_prefix_steps, 1);
        assert!(score.certificate_prefix_ppm > 0);
    }

    #[test]
    fn reasoning_rank_fitness_is_ordinal_and_centered() {
        let best = score_ruliad_answer(None, "ok=1;l=3;r=3", Some("ok=1;l=3;r=3"));
        let partial = score_ruliad_answer(None, "ok=1;l=3;r=3", Some("ok=1;l=2;r=7"));
        let wrong = score_ruliad_answer(None, "ok=1;l=3;r=3", Some("ok=0;l=2;r=7"));
        let scores = vec![partial.clone(), wrong.clone(), best.clone()];
        let order = ruliad_reasoning_rank_order(&scores);
        assert_eq!(order, vec![2, 0, 1]);
        let fitness = ruliad_reasoning_rank_fitness(&scores);
        assert!(fitness[2] > fitness[0]);
        assert!(fitness[0] > fitness[1]);
        assert!((fitness.iter().sum::<f32>()).abs() < 1.0e-6);
    }

    #[test]
    fn verifier_reward_orders_reasoning_quality() {
        let weights = RuliadVerifierRewardWeights::default();
        let exact = score_ruliad_answer(None, "ok=1;l=3;r=3", Some("ok=1;l=3;r=3"));
        let partial = score_ruliad_answer(None, "ok=1;l=3;r=3", Some("ok=1;l=2;r=7"));
        let wrong = score_ruliad_answer(None, "ok=1;l=3;r=3", Some("bad=0"));
        let malformed = score_ruliad_completion(None, "ok=1;l=3;r=3", Some("bad completion"));
        let missing = score_ruliad_completion(None, "ok=1;l=3;r=3", None);

        let exact_reward = ruliad_verifier_reward(&exact, weights);
        let partial_reward = ruliad_verifier_reward(&partial, weights);
        let wrong_reward = ruliad_verifier_reward(&wrong, weights);
        let malformed_reward = ruliad_verifier_reward(&malformed, weights);
        let missing_reward = ruliad_verifier_reward(&missing, weights);

        assert!(exact_reward > partial_reward);
        assert!(partial_reward > wrong_reward);
        assert!(wrong_reward > malformed_reward);
        assert!(malformed_reward > missing_reward);
    }

    #[test]
    fn verifier_reward_can_prefer_compact_correct_answers() {
        let weights = RuliadVerifierRewardWeights {
            compactness: 0.5,
            ..RuliadVerifierRewardWeights::default()
        };
        let mut short = score_ruliad_answer(None, "ok=1;l=3;r=3", Some("ok=1;l=3;r=3"));
        short.generated_token_count = 2;
        let mut long = short.clone();
        long.generated_token_count = 20;
        assert!(ruliad_verifier_reward(&short, weights) > ruliad_verifier_reward(&long, weights));
    }

    #[test]
    fn verifier_reward_vector_exposes_independent_quality_axes() {
        let score = RuliadReasoningScore {
            version: RULIAD_REASONING_SCORE_VERSION,
            status: RuliadAnswerStatus::VerifierMatch,
            correct_field_count: 2,
            expected_field_count: 4,
            partial_progress_ppm: SCORE_PPM_DENOMINATOR / 2,
            certificate_valid_prefix_steps: 1,
            certificate_expected_steps: 4,
            certificate_prefix_ppm: SCORE_PPM_DENOMINATOR / 4,
            generated_token_count: 16,
            hash_canary: false,
            answer_terminated: true,
            completion_quality_ppm: SCORE_PPM_DENOMINATOR,
        };
        let vector = ruliad_verifier_reward_vector(&score);
        assert_eq!(vector.verifier_match, 1.0);
        assert_eq!(vector.semantic_match, 1.0);
        assert_eq!(vector.partial_progress, 0.5);
        assert_eq!(vector.field_accuracy, 0.5);
        assert_eq!(vector.certificate_prefix, 0.25);
        assert_eq!(vector.compactness, 0.25);
        assert_eq!(vector.schema_quality, 1.0);
        assert_eq!(vector.hash_safety, 1.0);
        assert_eq!(vector.answer_termination, 1.0);
        assert_eq!(vector.completion_health, 1.0);
    }

    #[test]
    fn verifier_reward_vector_does_not_reward_compact_bad_outputs() {
        let score = RuliadReasoningScore {
            version: RULIAD_REASONING_SCORE_VERSION,
            status: RuliadAnswerStatus::SchemaValidWrong,
            correct_field_count: 0,
            expected_field_count: 4,
            partial_progress_ppm: 0,
            certificate_valid_prefix_steps: 0,
            certificate_expected_steps: 4,
            certificate_prefix_ppm: 0,
            generated_token_count: 1,
            hash_canary: false,
            answer_terminated: true,
            completion_quality_ppm: SCORE_PPM_DENOMINATOR,
        };
        let vector = ruliad_verifier_reward_vector(&score);
        assert_eq!(vector.compactness, 0.0);
        assert_eq!(vector.completion_health, 0.0);
        assert_eq!(vector.schema_quality, 0.25);
        assert_eq!(vector.answer_termination, 1.0);
    }

    #[test]
    fn completion_quality_penalizes_periodic_tails() {
        let healthy = extract_ruliad_completion("!:ok=1;l=3;r=3\n[/R2]");
        let cyclic_tail = "?:ca^20:x\n>x0=b35:h17\n".repeat(12);
        let cyclic = extract_ruliad_completion(&format!("!:ok=1;l=3;r=3\n{cyclic_tail}[/R2]"));
        assert_eq!(healthy.completion_quality_ppm, SCORE_PPM_DENOMINATOR);
        assert!(
            cyclic.completion_quality_ppm < SCORE_PPM_DENOMINATOR / 2,
            "cyclic completion should have low quality, got {}",
            cyclic.completion_quality_ppm
        );
    }

    #[test]
    fn completion_quality_penalizes_short_answer_loops() {
        let healthy = extract_ruliad_completion("!:ok=1;l=3;r=3\n[/R2]");
        let looped = extract_ruliad_completion("!:11:h11:h11:h11:h11:h11:h11:h11:h[/R2]");
        assert_eq!(healthy.completion_quality_ppm, SCORE_PPM_DENOMINATOR);
        assert!(
            looped.completion_quality_ppm < SCORE_PPM_DENOMINATOR / 2,
            "short periodic answer loops should have low quality, got {}",
            looped.completion_quality_ppm
        );
    }

    #[test]
    fn eval_report_tracks_completion_quality_and_answer_collapse() {
        let items = vec![
            RuliadEvalItem {
                oracle_hash: "a".to_string(),
                sample_index: 0,
                split: SampleSplit::Validation,
                family: "proof_tree".to_string(),
                task_kind: "prove_theorem".to_string(),
                math_domains: vec!["category_theory".to_string()],
                reasoning_modes: vec!["equational_reasoning".to_string()],
                prompt: "!:".to_string(),
                expected_answer: "ok=1;l=2;r=2".to_string(),
                difficulty_level: Some(1),
                spec: None,
            },
            RuliadEvalItem {
                oracle_hash: "b".to_string(),
                sample_index: 1,
                split: SampleSplit::Validation,
                family: "proof_tree".to_string(),
                task_kind: "prove_theorem".to_string(),
                math_domains: vec!["category_theory".to_string()],
                reasoning_modes: vec!["equational_reasoning".to_string()],
                prompt: "!:".to_string(),
                expected_answer: "ok=1;l=3;r=3".to_string(),
                difficulty_level: Some(1),
                spec: None,
            },
        ];
        let completions = vec![
            RuliadCompletionRecord {
                oracle_hash: "a".to_string(),
                completion: "!:11:h11:h11:h11:h11:h11:h11:h11:h[/R2]".to_string(),
            },
            RuliadCompletionRecord {
                oracle_hash: "b".to_string(),
                completion: "!:11:h11:h11:h11:h11:h11:h11:h11:h[/R2]".to_string(),
            },
        ];

        let report = evaluate_completions("collapse", &items, &completions);

        assert_eq!(report.item_count, 2);
        assert_eq!(report.scored_count, 2);
        assert_eq!(report.actual_answer_distinct_fraction, 0.5);
        assert!(
            report.mean_completion_quality < 0.5,
            "{}",
            report.mean_completion_quality
        );
        assert_eq!(report.family_scores.len(), 1);
        assert_eq!(report.family_scores[0].actual_answer_distinct_fraction, 0.5);
        assert!(
            report.family_scores[0].mean_completion_quality < 0.5,
            "{}",
            report.family_scores[0].mean_completion_quality
        );
    }

    #[test]
    fn verifier_reward_vector_completion_health_reflects_completion_quality() {
        let healthy = score_ruliad_completion(None, "ok=1;l=3;r=3", Some("!:ok=1;l=3;r=3\n[/R2]"));
        let cyclic_tail = "?:ca^20:x\n>x0=b35:h17\n".repeat(12);
        let cyclic = score_ruliad_completion(
            None,
            "ok=1;l=3;r=3",
            Some(&format!("!:ok=1;l=3;r=3\n{cyclic_tail}[/R2]")),
        );
        let healthy_vector = ruliad_verifier_reward_vector(&healthy);
        let cyclic_vector = ruliad_verifier_reward_vector(&cyclic);
        assert_eq!(healthy_vector.completion_health, 1.0);
        assert!(
            cyclic_vector.completion_health < healthy_vector.completion_health * 0.5,
            "cyclic completion health should be reduced: healthy={} cyclic={}",
            healthy_vector.completion_health,
            cyclic_vector.completion_health
        );
        assert!(
            cyclic_vector.compactness < healthy_vector.compactness,
            "cyclic completion should not win compactness"
        );
    }

    #[test]
    fn vpo_independent_utilities_select_pareto_useful_completions() {
        let exact_long = RuliadReasoningScore {
            version: RULIAD_REASONING_SCORE_VERSION,
            status: RuliadAnswerStatus::VerifierMatch,
            correct_field_count: 4,
            expected_field_count: 4,
            partial_progress_ppm: SCORE_PPM_DENOMINATOR,
            certificate_valid_prefix_steps: 4,
            certificate_expected_steps: 4,
            certificate_prefix_ppm: SCORE_PPM_DENOMINATOR,
            generated_token_count: 100,
            hash_canary: false,
            answer_terminated: true,
            completion_quality_ppm: SCORE_PPM_DENOMINATOR,
        };
        let compact_partial = RuliadReasoningScore {
            version: RULIAD_REASONING_SCORE_VERSION,
            status: RuliadAnswerStatus::Partial,
            correct_field_count: 2,
            expected_field_count: 4,
            partial_progress_ppm: SCORE_PPM_DENOMINATOR / 2,
            certificate_valid_prefix_steps: 1,
            certificate_expected_steps: 4,
            certificate_prefix_ppm: SCORE_PPM_DENOMINATOR / 4,
            generated_token_count: 1,
            hash_canary: false,
            answer_terminated: true,
            completion_quality_ppm: SCORE_PPM_DENOMINATOR,
        };
        let verifier_axis = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let compactness_axis = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let utilities = ruliad_vpo_independent_utilities(
            &[exact_long, compact_partial],
            &[verifier_axis, compactness_axis],
        );
        assert!(
            utilities[0] > 0.0,
            "verifier scalarization should select exact answer"
        );
        assert!(
            utilities[1] > 0.0,
            "compactness scalarization should select compact answer"
        );
    }

    #[test]
    fn verifier_reward_advantages_are_centered_and_normalized() {
        let rewards = [1.0, 0.0, -0.5, 0.5];
        let centered = centered_advantages(&rewards);
        assert!((centered.iter().sum::<f32>()).abs() < 1.0e-6);
        let normalized = normalized_advantages(&rewards, 1.0e-6);
        assert!((normalized.iter().sum::<f32>()).abs() < 1.0e-6);
        let variance =
            normalized.iter().map(|value| value * value).sum::<f32>() / normalized.len() as f32;
        assert!((variance - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn oracle_baseline_scores_all_eval_items() {
        let dir = tempdir().expect("tempdir");
        let mut config = test_config();
        config.output_dir = dir.path().join("out");
        let report = generate_ruliad_corpus(&config).expect("generate");
        let items = build_eval_items_from_manifest(
            &report.manifest_path,
            &RuliadEvalConfig {
                max_items: Some(16),
                ..RuliadEvalConfig::default()
            },
        )
        .expect("items");
        assert_eq!(items.len(), 16);
        let completions = baseline_completions(&items, RuliadEvalBaseline::Oracle);
        let eval = evaluate_completions("ruliad-eval-test", &items, &completions);
        assert_eq!(eval.item_count, 16);
        assert_eq!(eval.semantic_match_count, 16);
        assert_eq!(eval.verifier_match_count, 16);
        assert_eq!(eval.partial_credit_count, 16);
        assert_eq!(eval.mean_partial_progress, 1.0);
        assert_eq!(eval.answer_termination_rate, 1.0);
        assert_eq!(eval.failures.len(), 0);
    }

    #[test]
    fn eval_report_tracks_answer_field_and_termination_health() {
        let items = vec![
            RuliadEvalItem {
                oracle_hash: "h0".to_string(),
                sample_index: 0,
                split: SampleSplit::Validation,
                family: "law".to_string(),
                task_kind: "category_law".to_string(),
                math_domains: vec!["category".to_string()],
                reasoning_modes: vec!["equational".to_string()],
                prompt: "?:q\nA:ok,l,r\n!:".to_string(),
                expected_answer: "ok=1;l=3;r=3".to_string(),
                difficulty_level: Some(1),
                spec: None,
            },
            RuliadEvalItem {
                oracle_hash: "h1".to_string(),
                sample_index: 1,
                split: SampleSplit::Validation,
                family: "law".to_string(),
                task_kind: "category_law".to_string(),
                math_domains: vec!["category".to_string()],
                reasoning_modes: vec!["equational".to_string()],
                prompt: "?:q\nA:ok,l,r\n!:".to_string(),
                expected_answer: "ok=1;l=2;r=2".to_string(),
                difficulty_level: Some(1),
                spec: None,
            },
        ];
        let completions = vec![
            RuliadCompletionRecord {
                oracle_hash: "h0".to_string(),
                completion: "!:1;3;3\n[/R2]".to_string(),
            },
            RuliadCompletionRecord {
                oracle_hash: "h1".to_string(),
                completion: "!:ok=1;l=0;r=2".to_string(),
            },
        ];

        let eval = evaluate_completions("ruliad-eval-test", &items, &completions);
        assert_eq!(eval.answer_field_correct_count, 5);
        assert_eq!(eval.answer_field_expected_count, 6);
        assert!((eval.answer_field_accuracy - (5.0 / 6.0)).abs() < 1.0e-6);
        assert_eq!(eval.answer_terminated_count, 1);
        assert_eq!(eval.answer_termination_rate, 0.5);
        let difficulty = eval.difficulty_scores.first().expect("difficulty group");
        assert_eq!(difficulty.answer_field_correct_count, 5);
        assert_eq!(difficulty.answer_terminated_count, 1);
    }

    #[test]
    fn corrupted_baseline_fails_eval_items() {
        let dir = tempdir().expect("tempdir");
        let mut config = test_config();
        config.output_dir = dir.path().join("out");
        let report = generate_ruliad_corpus(&config).expect("generate");
        let items = build_eval_items_from_manifest(
            &report.manifest_path,
            &RuliadEvalConfig {
                max_items: Some(16),
                ..RuliadEvalConfig::default()
            },
        )
        .expect("items");
        let completions = baseline_completions(&items, RuliadEvalBaseline::Corrupt);
        let eval = evaluate_completions("ruliad-eval-test", &items, &completions);
        assert!(eval.semantic_match_count < eval.item_count);
        assert!(!eval.failures.is_empty());
    }

    #[test]
    fn validation_eval_items_are_disjoint_from_train_hashes() {
        let dir = tempdir().expect("tempdir");
        let mut config = test_config();
        config.output_dir = dir.path().join("out");
        let report = generate_ruliad_corpus(&config).expect("generate");
        let manifest = load_ruliad_manifest(&report.manifest_path).expect("manifest");
        let records = read_manifest_records(&report.manifest_path, &manifest).expect("records");
        let train_hashes = records
            .iter()
            .filter(|record| record.split == SampleSplit::Train)
            .filter_map(|record| record.oracle_hash.as_deref())
            .collect::<BTreeSet<_>>();
        let items = build_eval_items_from_manifest(
            &report.manifest_path,
            &RuliadEvalConfig {
                max_items: None,
                ..RuliadEvalConfig::default()
            },
        )
        .expect("items");
        assert!(!items.is_empty());
        assert!(
            items
                .iter()
                .all(|item| !train_hashes.contains(item.oracle_hash.as_str()))
        );
    }

    #[test]
    fn diagnostics_report_manifest_quality_and_config_buckets() {
        let dir = tempdir().expect("tempdir");
        let mut config = test_config();
        config.output_dir = dir.path().join("out");
        config.serialization.document_mode = RuliadDocumentMode::MultiChunkProofTree;
        config.serialization.document_chunks = UsizeRangeConfig { min: 2, max: 2 };
        let report = generate_ruliad_corpus(&config).expect("generate");
        let diagnostic = diagnose_manifest(
            &report.manifest_path,
            RuliadDiagnosticThresholds {
                require_all_semantics: false,
                ..RuliadDiagnosticThresholds::default()
            },
        )
        .expect("diagnose manifest");
        assert_eq!(
            diagnostic.sample_count,
            config.train_samples + config.validation_samples
        );
        assert_eq!(diagnostic.missing_ruliad_spec_count, 0);
        assert_eq!(diagnostic.answer_slot_coverage, 1.0);
        assert_eq!(diagnostic.payload_overflow_count, 0);
        assert_eq!(diagnostic.multi_chunk_document_coverage, 1.0);

        let config_diagnostic = diagnose_config(
            &config,
            8,
            RuliadDiagnosticThresholds {
                require_all_semantics: false,
                ..RuliadDiagnosticThresholds::default()
            },
        )
        .expect("diagnose config");
        assert!(!config_diagnostic.source_bucket_priors.is_empty());
    }

    #[test]
    fn diagnostics_require_configured_source_semantics_only() {
        let mut config = test_config();
        config.source_selection.enabled = true;
        config.source_selection.difficulty_levels = UsizeRangeConfig { min: 0, max: 0 };
        config.families = vec![RuliadFamilyConfig {
            kind: RuliadFamilyKind::Eca,
            weight: 1,
            width: Some(UsizeRangeConfig { min: 8, max: 8 }),
            steps: Some(UsizeRangeConfig { min: 2, max: 4 }),
        }];
        let diagnostic = diagnose_config(
            &config,
            8,
            RuliadDiagnosticThresholds {
                require_all_semantics: true,
                ..RuliadDiagnosticThresholds::default()
            },
        )
        .expect("diagnose config");
        assert!(
            !diagnostic
                .gate_failures
                .iter()
                .any(|failure| failure.contains("information_theory")
                    || failure.contains("entropy_canary")),
            "focused corpus should not fail on semantics from excluded families: {:?}",
            diagnostic.gate_failures
        );
        assert!(
            diagnostic.gate_failures.is_empty(),
            "ECA-only corpus should cover its advertised semantics: {:?}",
            diagnostic.gate_failures
        );
    }

    #[test]
    fn diagnostics_detect_duplicate_hashes() {
        let mut config = test_config();
        config.families = vec![RuliadFamilyConfig {
            kind: RuliadFamilyKind::Eca,
            weight: 1,
            width: Some(UsizeRangeConfig { min: 8, max: 8 }),
            steps: Some(UsizeRangeConfig { min: 2, max: 2 }),
        }];
        let corpus = OnlineRuliadCorpus::new(config).expect("corpus");
        let document = corpus
            .generate_document(SampleSplit::Train, 0)
            .expect("document");
        let sample = DiagnosticSample {
            split: SampleSplit::Train,
            family: document.family,
            task_kind: document.task_kind,
            token_count: document.token_count,
            serialized_char_count: document.serialized_preview.len(),
            stats: document.stats,
            spec: Some(document.spec),
            oracle_hash: Some(document.oracle_hash),
            math_domains: document.math_domains,
            reasoning_modes: document.reasoning_modes,
            multi_chunk_document: is_multi_chunk_document(&document.serialized_preview),
            serialized_preview: Some(document.serialized_preview),
        };
        let diagnostic = diagnose_samples(
            "duplicates".to_string(),
            1026,
            513,
            512,
            vec![sample.clone(), sample],
            Vec::new(),
            RuliadDiagnosticThresholds {
                require_all_semantics: false,
                ..RuliadDiagnosticThresholds::default()
            },
        );
        assert_eq!(diagnostic.duplicate_oracle_hash_count, 1);
        assert!(!diagnostic.gate_failures.is_empty());
    }
}
