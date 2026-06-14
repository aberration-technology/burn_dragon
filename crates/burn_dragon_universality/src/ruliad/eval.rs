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
use crate::ruliad::runtime::OnlineRuliadCorpus;
use crate::ruliad::source_selection::{RuliadSourceBucket, ruliad_source_buckets};
use crate::stats::SampleStats;

pub const RULIAD_DIAGNOSTIC_REPORT_VERSION: u32 = 1;
pub const RULIAD_EVAL_REPORT_VERSION: u32 = 1;

const MAX_REPORTED_EVAL_FAILURES: usize = 64;

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

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
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
    pub malformed_completion_count: usize,
    pub missing_completion_count: usize,
    pub exact_accuracy: f32,
    pub semantic_accuracy: f32,
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
    pub dataset_name: String,
    pub item_count: usize,
    pub scored_count: usize,
    pub exact_match_count: usize,
    pub semantic_match_count: usize,
    pub malformed_completion_count: usize,
    pub missing_completion_count: usize,
    pub unexpected_completion_count: usize,
    pub exact_accuracy: f32,
    pub semantic_accuracy: f32,
    pub canary_count: usize,
    pub canary_semantic_match_count: usize,
    pub family_scores: Vec<RuliadEvalGroupScore>,
    pub task_scores: Vec<RuliadEvalGroupScore>,
    pub math_domain_scores: Vec<RuliadEvalGroupScore>,
    pub reasoning_mode_scores: Vec<RuliadEvalGroupScore>,
    pub failures: Vec<RuliadEvalFailure>,
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
    malformed_completion_count: usize,
    missing_completion_count: usize,
}

#[derive(Debug, Clone)]
struct EvalOutcome {
    exact_match: bool,
    semantic_match: bool,
    malformed: bool,
    missing: bool,
    actual_answer: Option<String>,
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
                multi_chunk_document: document.serialized_preview.contains("[RTREE"),
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
                completion: format!("!:{answer}"),
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
    let mut math_domain_scores = BTreeMap::<String, EvalAccumulator>::new();
    let mut reasoning_mode_scores = BTreeMap::<String, EvalAccumulator>::new();
    let mut exact_match_count = 0usize;
    let mut semantic_match_count = 0usize;
    let mut malformed_completion_count = 0usize;
    let mut missing_completion_count = 0usize;
    let mut scored_count = 0usize;
    let mut canary_count = 0usize;
    let mut canary_semantic_match_count = 0usize;
    let mut failures = Vec::new();

    for item in items {
        let completion = completion_by_hash
            .get(&item.oracle_hash)
            .map(String::as_str);
        let outcome = score_item(item, completion);
        scored_count += usize::from(completion.is_some());
        exact_match_count += usize::from(outcome.exact_match);
        semantic_match_count += usize::from(outcome.semantic_match);
        malformed_completion_count += usize::from(outcome.malformed);
        missing_completion_count += usize::from(outcome.missing);
        if item.family == "hash_noise" || item.task_kind == "hash_canary" {
            canary_count += 1;
            canary_semantic_match_count += usize::from(outcome.semantic_match);
        }
        add_group_score(&mut family_scores, &item.family, &outcome);
        add_group_score(&mut task_scores, &item.task_kind, &outcome);
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
        dataset_name,
        item_count: items.len(),
        scored_count,
        exact_match_count,
        semantic_match_count,
        malformed_completion_count,
        missing_completion_count,
        unexpected_completion_count,
        exact_accuracy: ratio(exact_match_count, items.len()),
        semantic_accuracy: ratio(semantic_match_count, items.len()),
        canary_count,
        canary_semantic_match_count,
        family_scores: finalize_group_scores(family_scores),
        task_scores: finalize_group_scores(task_scores),
        math_domain_scores: finalize_group_scores(math_domain_scores),
        reasoning_mode_scores: finalize_group_scores(reasoning_mode_scores),
        failures,
    }
}

pub fn extract_ruliad_answer(completion: &str) -> Option<String> {
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
    match (parse_answer_pairs(expected), parse_answer_pairs(actual)) {
        (Some(left), Some(right)) => left == right,
        _ => false,
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
            usize::from(sample.multi_chunk_document || text.contains("[RTREE"));
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
        record_missing_required_domains(&math_domain_counts, &mut gate_failures);
        record_missing_required_modes(&reasoning_mode_counts, &mut gate_failures);
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
    let Some(completion) = completion else {
        return EvalOutcome {
            exact_match: false,
            semantic_match: false,
            malformed: false,
            missing: true,
            actual_answer: None,
        };
    };
    let actual_answer = extract_ruliad_answer(completion);
    let Some(actual) = actual_answer.as_deref() else {
        return EvalOutcome {
            exact_match: false,
            semantic_match: false,
            malformed: true,
            missing: false,
            actual_answer,
        };
    };
    let exact_match = ruliad_answers_exact_match(&item.expected_answer, actual);
    let semantic_match = ruliad_answers_semantic_match(&item.expected_answer, actual);
    EvalOutcome {
        exact_match,
        semantic_match,
        malformed: false,
        missing: false,
        actual_answer,
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
    score.malformed_completion_count += usize::from(outcome.malformed);
    score.missing_completion_count += usize::from(outcome.missing);
}

fn finalize_group_scores(scores: BTreeMap<String, EvalAccumulator>) -> Vec<RuliadEvalGroupScore> {
    scores
        .into_iter()
        .map(|(label, score)| RuliadEvalGroupScore {
            label,
            count: score.count,
            exact_match_count: score.exact_match_count,
            semantic_match_count: score.semantic_match_count,
            malformed_completion_count: score.malformed_completion_count,
            missing_completion_count: score.missing_completion_count,
            exact_accuracy: ratio(score.exact_match_count, score.count),
            semantic_accuracy: ratio(score.semantic_match_count, score.count),
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
    gate_failures: &mut Vec<String>,
) {
    for domain in RULIAD_REQUIRED_MATH_DOMAINS {
        if !counts.contains_key(domain.label()) {
            gate_failures.push(format!("missing_math_domain={}", domain.label()));
        }
    }
}

fn record_missing_required_modes(
    counts: &BTreeMap<String, usize>,
    gate_failures: &mut Vec<String>,
) {
    for mode in RULIAD_REQUIRED_REASONING_MODES {
        if !counts.contains_key(mode.label()) {
            gate_failures.push(format!("missing_reasoning_mode={}", mode.label()));
        }
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
        .trim_end_matches("[/RTREE]")
        .trim()
        .to_string()
}

fn parse_answer_pairs(value: &str) -> Option<BTreeMap<String, String>> {
    let normalized = normalize_answer(value);
    let mut pairs = BTreeMap::new();
    for part in normalized.split(';') {
        let (key, value) = part.split_once('=')?;
        let key = key.trim();
        if key.is_empty() {
            return None;
        }
        pairs.insert(key.to_string(), normalize_pair_value(value));
    }
    (!pairs.is_empty()).then_some(pairs)
}

fn normalize_pair_value(value: &str) -> String {
    match value.trim() {
        "True" | "TRUE" => "true".to_string(),
        "False" | "FALSE" => "false".to_string(),
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
        assert_eq!(eval.failures.len(), 0);
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
            multi_chunk_document: document.serialized_preview.contains("[RTREE"),
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
