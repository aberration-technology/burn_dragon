use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use serde::Serialize;

use crate::manifest::{SampleSplit, UniversalityTokenizerManifest};
use crate::ruliad::config::{RuliadCorpusConfig, RuliadDocumentMode, load_ruliad_config};
use crate::ruliad::eval::RuliadEvalItem;
use crate::ruliad::oracles::{
    GeneratedRuliadSample, LeanProofTask, RuliadCategoricalPresentation, RuliadSampleSpec,
    compact_ruliad_label, default_proof_tasks, generate_sample, generate_sample_for_source_bucket,
    load_proof_tasks, ruliad_answer_contract, ruliad_expected_answer, ruliad_prompt_prefix,
    ruliad_sample_math_domains, ruliad_sample_reasoning_modes,
};
use crate::ruliad::rng::{SplitMix64, mix_seed};
use crate::ruliad::search::RuliadSamplerCandidate;
use crate::ruliad::source_selection::{
    RuliadSourceBucket, ruliad_sampler_candidates, ruliad_source_bucket_by_label,
    ruliad_source_buckets, ruliad_source_buckets_for_difficulty,
};
use crate::ruliad::supervision::{
    RuliadTokenSupervisionConfig, RuliadTokenSupervisionMode, ruliad_token_loss_mask,
};
use crate::ruliad::tokenize::RuliadByteTokenizer;
use crate::stats::{ComplexityHistogramBin, SampleStats, build_complexity_histogram};

const DEFAULT_PROBE_SAMPLES: usize = 32;
const SOURCE_BUCKET_DOCUMENT_RETRY_LIMIT: usize = 32;
pub const RULIAD_SUPERVISION_AUDIT_VERSION: u32 = 2;

#[derive(Debug, Clone)]
pub struct RuliadRuntimeSampleDocument {
    pub split: SampleSplit,
    pub sample_index: usize,
    pub spec: RuliadSampleSpec,
    pub categorical_presentation: RuliadCategoricalPresentation,
    pub family: String,
    pub task_kind: String,
    pub oracle_hash: String,
    pub verifier_version: u32,
    pub math_domains: Vec<String>,
    pub reasoning_modes: Vec<String>,
    pub source_difficulty_level: Option<usize>,
    /// Number of capacity-driven seed substitutions used to materialize this
    /// document. Formal proof buckets never substitute a different seed.
    pub generation_retry_count: usize,
    pub token_count: usize,
    pub tokens: Vec<u32>,
    pub serialized_preview: String,
    pub stats: SampleStats,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RuliadFrontierFeasibilitySample {
    pub bucket_label: String,
    pub difficulty_level: usize,
    pub sample_index: usize,
    pub payload_tokens: usize,
    pub payload_capacity: usize,
    pub fits: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub formal_complexity: Option<crate::ruliad::ir::RuliadComplexityVector>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RuliadFrontierFeasibilityReport {
    pub difficulty_level: usize,
    pub payload_capacity: usize,
    pub sample_count: usize,
    pub fit_count: usize,
    pub fit_fraction: f32,
    pub mean_payload_tokens: f32,
    pub max_payload_tokens: usize,
    pub samples: Vec<RuliadFrontierFeasibilitySample>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RuliadSupervisionAuditBucket {
    pub bucket_label: String,
    pub family: String,
    pub task_kind: String,
    pub answer_contract: String,
    pub difficulty_level: usize,
    pub sample_count: usize,
    pub mean_document_tokens: f64,
    pub max_document_tokens: usize,
    pub mean_stream_chunks: f64,
    pub stream_chunk_share: f64,
    pub mean_answer_target_tokens: f64,
    pub answer_target_share: f64,
    pub mean_trace_answer_target_tokens: f64,
    pub trace_answer_target_share: f64,
    pub mean_answer_weight_units: f64,
    pub mean_trace_answer_weight_units: f64,
    pub query_conditioning_samples: usize,
    pub mean_query_to_answer_tokens: f64,
    pub max_query_to_answer_tokens: usize,
    pub query_visible_within_block_fraction: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RuliadSupervisionAuditReport {
    pub version: u32,
    pub block_size: usize,
    pub samples_per_bucket: usize,
    pub sample_count: usize,
    pub total_document_tokens: usize,
    pub total_stream_chunks: usize,
    pub total_answer_target_tokens: usize,
    pub total_trace_answer_target_tokens: usize,
    pub total_query_conditioning_samples: usize,
    pub query_visible_within_block_fraction: f64,
    pub max_to_min_mean_stream_chunks_ratio: f64,
    pub buckets: Vec<RuliadSupervisionAuditBucket>,
}

#[derive(Default)]
struct RuliadSupervisionAuditAccumulator {
    bucket_label: String,
    family: String,
    task_kind: String,
    answer_contract: String,
    difficulty_level: usize,
    sample_count: usize,
    document_tokens: usize,
    max_document_tokens: usize,
    stream_chunks: usize,
    answer_target_tokens: usize,
    trace_answer_target_tokens: usize,
    answer_weight_units: usize,
    trace_answer_weight_units: usize,
    query_conditioning_samples: usize,
    query_to_answer_tokens: usize,
    max_query_to_answer_tokens: usize,
    query_visible_within_block_samples: usize,
}

#[derive(Clone)]
pub struct OnlineRuliadCorpus {
    config: RuliadCorpusConfig,
    proof_tasks: Arc<Vec<LeanProofTask>>,
    tokenizer: Arc<RuliadByteTokenizer>,
    tokenizer_manifest: UniversalityTokenizerManifest,
    source_buckets: Arc<Vec<RuliadSourceBucket>>,
    sampler_candidates: Arc<Vec<RuliadSamplerCandidate>>,
    document_token_count: usize,
}

impl OnlineRuliadCorpus {
    pub fn new(config: RuliadCorpusConfig) -> Result<Self> {
        config.validate()?;
        let proof_tasks = load_configured_proof_tasks(&config)?;
        let tokenizer = Arc::new(RuliadByteTokenizer::from_config(&config.tokenization)?);
        let tokenizer_manifest = tokenizer.manifest();
        let source_buckets = ruliad_source_buckets(&config);
        let sampler_candidates = ruliad_sampler_candidates(&config);
        let document_token_count = fixed_ruliad_document_token_count(&config)?;
        Ok(Self {
            config,
            proof_tasks: Arc::new(proof_tasks),
            tokenizer,
            tokenizer_manifest,
            source_buckets: Arc::new(source_buckets),
            sampler_candidates: Arc::new(sampler_candidates),
            document_token_count,
        })
    }

    pub fn load(path: &Path) -> Result<Self> {
        let config = load_ruliad_config(path)?;
        Self::new(config)
    }

    pub fn config(&self) -> &RuliadCorpusConfig {
        &self.config
    }

    pub fn dataset_name(&self) -> &str {
        &self.config.name
    }

    pub fn tokenizer_manifest(&self) -> &UniversalityTokenizerManifest {
        &self.tokenizer_manifest
    }

    pub fn train_samples(&self) -> usize {
        self.config.train_samples
    }

    pub fn validation_samples(&self) -> usize {
        self.config.validation_samples
    }

    pub fn sample_count(&self, split: SampleSplit) -> usize {
        match split {
            SampleSplit::Train => self.train_samples(),
            SampleSplit::Validation => self.validation_samples(),
        }
    }

    pub fn document_token_count(&self) -> usize {
        self.document_token_count
    }

    pub fn train_token_count(&self) -> usize {
        self.train_samples()
            .saturating_mul(self.document_token_count())
    }

    pub fn val_token_count(&self) -> usize {
        self.validation_samples()
            .saturating_mul(self.document_token_count())
    }

    pub fn total_token_count(&self) -> usize {
        self.train_token_count()
            .saturating_add(self.val_token_count())
    }

    pub fn source_selection_enabled(&self) -> bool {
        self.config.source_selection.enabled
    }

    pub fn source_buckets(&self) -> &[RuliadSourceBucket] {
        self.source_buckets.as_slice()
    }

    pub fn encode_payload_tokens(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode_payload(text)
    }

    pub fn decode_payload_tokens(&self, tokens: &[u32], stop_at_eos: bool) -> String {
        self.tokenizer.decode_payload(tokens, stop_at_eos)
    }

    pub fn sampler_candidates(&self) -> Vec<RuliadSamplerCandidate> {
        self.sampler_candidates.as_ref().clone()
    }

    /// Measure a frontier level without replacing an over-capacity sample with
    /// a different seed. This exposes resource-induced curriculum bias before
    /// a long-running trainer reaches the level.
    pub fn probe_frontier_feasibility(
        &self,
        difficulty_level: usize,
        samples_per_bucket: usize,
    ) -> Result<RuliadFrontierFeasibilityReport> {
        let payload_capacity = self
            .tokenizer
            .payload_token_capacity(self.document_token_count);
        let buckets = ruliad_source_buckets_for_difficulty(&self.config, difficulty_level);
        let samples_per_bucket = samples_per_bucket.max(1);
        let mut samples = Vec::with_capacity(buckets.len().saturating_mul(samples_per_bucket));
        for bucket in &buckets {
            for sample_index in 0..samples_per_bucket {
                let sample = generate_sample_for_source_bucket(
                    &self.config,
                    self.proof_tasks.as_slice(),
                    SampleSplit::Train,
                    0,
                    sample_index,
                    bucket,
                )?;
                let payload_tokens = self.tokenizer.payload_token_count(&sample.text);
                let formal_complexity = match &sample.spec {
                    RuliadSampleSpec::FormalProof {
                        problem,
                        certificate,
                        ..
                    } => Some(crate::ruliad::kernel::complexity_vector(
                        problem,
                        Some(certificate),
                    )),
                    _ => None,
                };
                samples.push(RuliadFrontierFeasibilitySample {
                    bucket_label: bucket.label(),
                    difficulty_level,
                    sample_index,
                    payload_tokens,
                    payload_capacity,
                    fits: payload_tokens <= payload_capacity,
                    formal_complexity,
                });
            }
        }
        let sample_count = samples.len();
        let fit_count = samples.iter().filter(|sample| sample.fits).count();
        let payload_token_sum = samples
            .iter()
            .map(|sample| sample.payload_tokens)
            .fold(0usize, usize::saturating_add);
        let max_payload_tokens = samples
            .iter()
            .map(|sample| sample.payload_tokens)
            .max()
            .unwrap_or_default();
        Ok(RuliadFrontierFeasibilityReport {
            difficulty_level,
            payload_capacity,
            sample_count,
            fit_count,
            fit_fraction: if sample_count == 0 {
                0.0
            } else {
                fit_count as f32 / sample_count as f32
            },
            mean_payload_tokens: if sample_count == 0 {
                0.0
            } else {
                payload_token_sum as f32 / sample_count as f32
            },
            max_payload_tokens,
            samples,
        })
    }

    /// Audit realized training-token exposure for source buckets. Source
    /// probabilities are per document, while persistent TBPTT consumes one
    /// optimizer update per chunk, so stream-chunk share is the relevant
    /// compute balance rather than sample share alone.
    pub fn audit_frontier_supervision(
        &self,
        difficulty_levels: &[usize],
        samples_per_bucket: usize,
        block_size: usize,
        supervision: RuliadTokenSupervisionConfig,
    ) -> Result<RuliadSupervisionAuditReport> {
        if difficulty_levels.is_empty() {
            return Err(anyhow!(
                "ruliad supervision audit requires at least one difficulty level"
            ));
        }
        let samples_per_bucket = samples_per_bucket.max(1);
        let block_size = block_size.max(1);
        let mut accumulators = BTreeMap::<String, RuliadSupervisionAuditAccumulator>::new();

        for difficulty_level in difficulty_levels.iter().copied() {
            for bucket in ruliad_source_buckets_for_difficulty(&self.config, difficulty_level) {
                let bucket_label = bucket.label();
                for sample_index in 0..samples_per_bucket {
                    let document = self.generate_document_for_source_bucket_with_padding(
                        SampleSplit::Train,
                        0,
                        sample_index,
                        &bucket_label,
                        false,
                    )?;
                    let target_count = document.tokens.len().saturating_sub(1);
                    let stream_chunks = target_count.div_ceil(block_size).max(1);
                    let answer = supervision_mask_summary(
                        &document.tokens,
                        RuliadTokenSupervisionMode::AnswerCompletion,
                        supervision,
                    );
                    let trace_answer = supervision_mask_summary(
                        &document.tokens,
                        RuliadTokenSupervisionMode::TraceAndAnswer,
                        supervision,
                    );
                    let accumulator =
                        accumulators.entry(bucket_label.clone()).or_insert_with(|| {
                            RuliadSupervisionAuditAccumulator {
                                bucket_label: bucket_label.clone(),
                                family: document.family.clone(),
                                task_kind: document.task_kind.clone(),
                                answer_contract: ruliad_answer_contract(&document.spec),
                                difficulty_level,
                                ..RuliadSupervisionAuditAccumulator::default()
                            }
                        });
                    accumulator.sample_count = accumulator.sample_count.saturating_add(1);
                    accumulator.document_tokens = accumulator
                        .document_tokens
                        .saturating_add(document.tokens.len());
                    accumulator.max_document_tokens =
                        accumulator.max_document_tokens.max(document.tokens.len());
                    accumulator.stream_chunks =
                        accumulator.stream_chunks.saturating_add(stream_chunks);
                    accumulator.answer_target_tokens = accumulator
                        .answer_target_tokens
                        .saturating_add(answer.nonzero_targets);
                    accumulator.trace_answer_target_tokens = accumulator
                        .trace_answer_target_tokens
                        .saturating_add(trace_answer.nonzero_targets);
                    accumulator.answer_weight_units = accumulator
                        .answer_weight_units
                        .saturating_add(answer.weight_units);
                    accumulator.trace_answer_weight_units = accumulator
                        .trace_answer_weight_units
                        .saturating_add(trace_answer.weight_units);
                    if let Some(span) =
                        query_to_answer_token_span(&self.tokenizer, &document.serialized_preview)
                    {
                        accumulator.query_conditioning_samples =
                            accumulator.query_conditioning_samples.saturating_add(1);
                        accumulator.query_to_answer_tokens =
                            accumulator.query_to_answer_tokens.saturating_add(span);
                        accumulator.max_query_to_answer_tokens =
                            accumulator.max_query_to_answer_tokens.max(span);
                        if span <= block_size {
                            accumulator.query_visible_within_block_samples = accumulator
                                .query_visible_within_block_samples
                                .saturating_add(1);
                        }
                    }
                }
            }
        }

        let sample_count = accumulators
            .values()
            .map(|bucket| bucket.sample_count)
            .sum::<usize>();
        let total_document_tokens = accumulators
            .values()
            .map(|bucket| bucket.document_tokens)
            .sum::<usize>();
        let total_stream_chunks = accumulators
            .values()
            .map(|bucket| bucket.stream_chunks)
            .sum::<usize>();
        let total_answer_target_tokens = accumulators
            .values()
            .map(|bucket| bucket.answer_target_tokens)
            .sum::<usize>();
        let total_trace_answer_target_tokens = accumulators
            .values()
            .map(|bucket| bucket.trace_answer_target_tokens)
            .sum::<usize>();
        let total_query_conditioning_samples = accumulators
            .values()
            .map(|bucket| bucket.query_conditioning_samples)
            .sum::<usize>();
        let total_query_visible_within_block_samples = accumulators
            .values()
            .map(|bucket| bucket.query_visible_within_block_samples)
            .sum::<usize>();
        let mut min_mean_chunks = f64::INFINITY;
        let mut max_mean_chunks = 0.0f64;
        let buckets = accumulators
            .into_values()
            .map(|bucket| {
                let denominator = bucket.sample_count.max(1) as f64;
                let mean_stream_chunks = bucket.stream_chunks as f64 / denominator;
                min_mean_chunks = min_mean_chunks.min(mean_stream_chunks);
                max_mean_chunks = max_mean_chunks.max(mean_stream_chunks);
                RuliadSupervisionAuditBucket {
                    bucket_label: bucket.bucket_label,
                    family: bucket.family,
                    task_kind: bucket.task_kind,
                    answer_contract: bucket.answer_contract,
                    difficulty_level: bucket.difficulty_level,
                    sample_count: bucket.sample_count,
                    mean_document_tokens: bucket.document_tokens as f64 / denominator,
                    max_document_tokens: bucket.max_document_tokens,
                    mean_stream_chunks,
                    stream_chunk_share: ratio(bucket.stream_chunks, total_stream_chunks),
                    mean_answer_target_tokens: bucket.answer_target_tokens as f64 / denominator,
                    answer_target_share: ratio(
                        bucket.answer_target_tokens,
                        total_answer_target_tokens,
                    ),
                    mean_trace_answer_target_tokens: bucket.trace_answer_target_tokens as f64
                        / denominator,
                    trace_answer_target_share: ratio(
                        bucket.trace_answer_target_tokens,
                        total_trace_answer_target_tokens,
                    ),
                    mean_answer_weight_units: bucket.answer_weight_units as f64 / denominator,
                    mean_trace_answer_weight_units: bucket.trace_answer_weight_units as f64
                        / denominator,
                    query_conditioning_samples: bucket.query_conditioning_samples,
                    mean_query_to_answer_tokens: ratio_mean(
                        bucket.query_to_answer_tokens,
                        bucket.query_conditioning_samples,
                    ),
                    max_query_to_answer_tokens: bucket.max_query_to_answer_tokens,
                    query_visible_within_block_fraction: ratio(
                        bucket.query_visible_within_block_samples,
                        bucket.query_conditioning_samples,
                    ),
                }
            })
            .collect::<Vec<_>>();
        let max_to_min_mean_stream_chunks_ratio =
            if min_mean_chunks.is_finite() && min_mean_chunks > 0.0 {
                max_mean_chunks / min_mean_chunks
            } else {
                0.0
            };

        Ok(RuliadSupervisionAuditReport {
            version: RULIAD_SUPERVISION_AUDIT_VERSION,
            block_size,
            samples_per_bucket,
            sample_count,
            total_document_tokens,
            total_stream_chunks,
            total_answer_target_tokens,
            total_trace_answer_target_tokens,
            total_query_conditioning_samples,
            query_visible_within_block_fraction: ratio(
                total_query_visible_within_block_samples,
                total_query_conditioning_samples,
            ),
            max_to_min_mean_stream_chunks_ratio,
            buckets,
        })
    }

    pub fn generate_document(
        &self,
        split: SampleSplit,
        sample_index: usize,
    ) -> Result<RuliadRuntimeSampleDocument> {
        self.generate_document_for_epoch(split, 0, sample_index)
    }

    pub fn generate_document_for_epoch(
        &self,
        split: SampleSplit,
        epoch_index: usize,
        sample_index: usize,
    ) -> Result<RuliadRuntimeSampleDocument> {
        self.generate_document_for_epoch_with_padding(split, epoch_index, sample_index, true)
    }

    fn generate_document_for_epoch_with_padding(
        &self,
        split: SampleSplit,
        epoch_index: usize,
        sample_index: usize,
        pad_to_envelope: bool,
    ) -> Result<RuliadRuntimeSampleDocument> {
        if self.config.serialization.document_mode == RuliadDocumentMode::MultiChunkProofTree {
            return self.generate_multi_chunk_document_for_epoch(
                split,
                epoch_index,
                sample_index,
                pad_to_envelope,
            );
        }
        let sample = self.generate_raw_sample(split, epoch_index, sample_index)?;
        let text = sample.text.clone();
        self.document_from_sample_text(split, sample_index, sample, text, pad_to_envelope)
    }

    pub fn generate_document_for_source_bucket(
        &self,
        split: SampleSplit,
        epoch_index: usize,
        sample_index: usize,
        bucket_label: &str,
    ) -> Result<RuliadRuntimeSampleDocument> {
        self.generate_document_for_source_bucket_with_padding(
            split,
            epoch_index,
            sample_index,
            bucket_label,
            true,
        )
    }

    fn generate_document_for_source_bucket_with_padding(
        &self,
        split: SampleSplit,
        epoch_index: usize,
        sample_index: usize,
        bucket_label: &str,
        pad_to_envelope: bool,
    ) -> Result<RuliadRuntimeSampleDocument> {
        let bucket = self
            .source_buckets
            .iter()
            .find(|bucket| bucket.label() == bucket_label)
            .or_else(|| {
                self.source_buckets.iter().find(|bucket| {
                    bucket
                        .label()
                        .split('@')
                        .next()
                        .is_some_and(|prefix| prefix == bucket_label)
                })
            })
            .cloned()
            .or_else(|| ruliad_source_bucket_by_label(&self.config, bucket_label))
            .ok_or_else(|| anyhow!("unknown ruliad source bucket `{bucket_label}`"))?;
        let retry_limit =
            if bucket.id.family == crate::ruliad::config::RuliadFamilyKind::FormalProof {
                1
            } else {
                SOURCE_BUCKET_DOCUMENT_RETRY_LIMIT
            };
        let mut last_error = None;
        for retry in 0..retry_limit {
            let candidate_sample_index = source_bucket_retry_sample_index(sample_index, retry);
            let result = if self.config.serialization.document_mode
                == RuliadDocumentMode::MultiChunkProofTree
            {
                self.generate_multi_chunk_document_for_source_bucket(
                    split,
                    epoch_index,
                    candidate_sample_index,
                    &bucket,
                    pad_to_envelope,
                )
            } else {
                let sample = generate_sample_for_source_bucket(
                    &self.config,
                    self.proof_tasks.as_slice(),
                    split,
                    epoch_index,
                    candidate_sample_index,
                    &bucket,
                )?;
                let text = sample.text.clone();
                self.document_from_sample_text(
                    split,
                    candidate_sample_index,
                    sample,
                    text,
                    pad_to_envelope,
                )
            };
            match result {
                Ok(mut document) => {
                    document.sample_index = sample_index;
                    document.source_difficulty_level = Some(bucket.id.difficulty_level);
                    document.generation_retry_count = retry;
                    return Ok(document);
                }
                Err(error) if is_document_payload_capacity_error(&error) => {
                    if bucket.id.family == crate::ruliad::config::RuliadFamilyKind::FormalProof {
                        return Err(anyhow!(
                            "formal source bucket `{bucket_label}` exceeds the configured resource envelope; seed substitution is disabled: {error}"
                        ));
                    }
                    last_error = Some(error);
                }
                Err(error) => return Err(error),
            }
        }
        Err(last_error.unwrap_or_else(|| {
            anyhow!(
                "failed to generate source bucket `{bucket_label}` within document payload capacity"
            )
        }))
    }

    pub fn generate_document_tokens(
        &self,
        split: SampleSplit,
        sample_index: usize,
    ) -> Result<Vec<u32>> {
        self.generate_document_tokens_for_epoch(split, 0, sample_index)
    }

    pub fn generate_document_tokens_for_epoch(
        &self,
        split: SampleSplit,
        epoch_index: usize,
        sample_index: usize,
    ) -> Result<Vec<u32>> {
        Ok(self
            .generate_document_for_epoch(split, epoch_index, sample_index)?
            .tokens)
    }

    pub fn generate_compact_document_tokens_for_epoch(
        &self,
        split: SampleSplit,
        epoch_index: usize,
        sample_index: usize,
    ) -> Result<Vec<u32>> {
        Ok(self
            .generate_document_for_epoch_with_padding(split, epoch_index, sample_index, false)?
            .tokens)
    }

    pub fn generate_compact_document_tokens_for_source_bucket(
        &self,
        split: SampleSplit,
        epoch_index: usize,
        sample_index: usize,
        bucket_label: &str,
    ) -> Result<Vec<u32>> {
        Ok(self
            .generate_document_for_source_bucket_with_padding(
                split,
                epoch_index,
                sample_index,
                bucket_label,
                false,
            )?
            .tokens)
    }

    pub fn generate_eval_item_for_epoch(
        &self,
        split: SampleSplit,
        epoch_index: usize,
        sample_index: usize,
    ) -> Result<RuliadEvalItem> {
        let document =
            self.generate_document_for_epoch_with_padding(split, epoch_index, sample_index, false)?;
        Ok(eval_item_from_document(document))
    }

    pub fn generate_training_serialization_eval_item_for_epoch(
        &self,
        split: SampleSplit,
        epoch_index: usize,
        sample_index: usize,
    ) -> Result<RuliadEvalItem> {
        let document =
            self.generate_document_for_epoch_with_padding(split, epoch_index, sample_index, false)?;
        training_serialization_eval_item_from_document(document)
    }

    pub fn generate_eval_item_for_source_bucket(
        &self,
        split: SampleSplit,
        epoch_index: usize,
        sample_index: usize,
        bucket_label: &str,
    ) -> Result<RuliadEvalItem> {
        let document = self.generate_document_for_source_bucket_with_padding(
            split,
            epoch_index,
            sample_index,
            bucket_label,
            false,
        )?;
        Ok(eval_item_from_document(document))
    }

    pub fn generate_training_serialization_eval_item_for_source_bucket(
        &self,
        split: SampleSplit,
        epoch_index: usize,
        sample_index: usize,
        bucket_label: &str,
    ) -> Result<RuliadEvalItem> {
        let document = self.generate_document_for_source_bucket_with_padding(
            split,
            epoch_index,
            sample_index,
            bucket_label,
            false,
        )?;
        training_serialization_eval_item_from_document(document)
    }

    pub fn generate_raw_sample(
        &self,
        split: SampleSplit,
        epoch_index: usize,
        sample_index: usize,
    ) -> Result<GeneratedRuliadSample> {
        let sample_count = self.sample_count(split);
        if sample_index >= sample_count {
            return Err(anyhow!(
                "sample_index {} out of range for {:?} split with {} samples",
                sample_index,
                split,
                sample_count
            ));
        }
        generate_sample(
            &self.config,
            self.proof_tasks.as_slice(),
            split,
            epoch_index,
            sample_index,
        )
    }

    fn generate_multi_chunk_document_for_epoch(
        &self,
        split: SampleSplit,
        epoch_index: usize,
        sample_index: usize,
        pad_to_envelope: bool,
    ) -> Result<RuliadRuntimeSampleDocument> {
        let chunk_count =
            multi_chunk_count_for_document(&self.config, split, epoch_index, sample_index, 0);
        let mut samples = Vec::with_capacity(chunk_count);
        for node_index in 0..chunk_count {
            samples.push(generate_sample(
                &self.config,
                self.proof_tasks.as_slice(),
                split,
                epoch_index,
                sample_index
                    .saturating_mul(chunk_count)
                    .saturating_add(node_index),
            )?);
        }
        let root = samples
            .last()
            .cloned()
            .ok_or_else(|| anyhow!("multi-chunk ruliad document has no root sample"))?;
        let text = multi_chunk_proof_tree_text(&samples);
        self.document_from_sample_text(split, sample_index, root, text, pad_to_envelope)
    }

    fn generate_multi_chunk_document_for_source_bucket(
        &self,
        split: SampleSplit,
        epoch_index: usize,
        sample_index: usize,
        bucket: &RuliadSourceBucket,
        pad_to_envelope: bool,
    ) -> Result<RuliadRuntimeSampleDocument> {
        let chunk_count = multi_chunk_count_for_document(
            &self.config,
            split,
            epoch_index,
            sample_index,
            bucket.id.seed_tag(),
        );
        let mut samples = Vec::with_capacity(chunk_count);
        for node_index in 0..chunk_count {
            samples.push(generate_sample_for_source_bucket(
                &self.config,
                self.proof_tasks.as_slice(),
                split,
                epoch_index,
                sample_index
                    .saturating_mul(chunk_count)
                    .saturating_add(node_index),
                bucket,
            )?);
        }
        let root = samples
            .last()
            .cloned()
            .ok_or_else(|| anyhow!("multi-chunk ruliad source document has no root sample"))?;
        let text = multi_chunk_proof_tree_text(&samples);
        let mut document =
            self.document_from_sample_text(split, sample_index, root, text, pad_to_envelope)?;
        document.source_difficulty_level = Some(bucket.id.difficulty_level);
        Ok(document)
    }

    fn document_from_sample_text(
        &self,
        split: SampleSplit,
        sample_index: usize,
        sample: GeneratedRuliadSample,
        text: String,
        pad_to_envelope: bool,
    ) -> Result<RuliadRuntimeSampleDocument> {
        let payload_capacity = self
            .tokenizer
            .payload_token_capacity(self.document_token_count);
        let payload_token_count = self.tokenizer.payload_token_count(&text);
        if payload_token_count > payload_capacity {
            return Err(anyhow!(
                "ruliad sample text exceeds document payload capacity (family={} task={} text_tokens={} payload_tokens={} text_bytes={})",
                sample.family.label(),
                sample.task_kind.label(),
                payload_token_count,
                payload_capacity,
                text.len()
            ));
        }
        let tokens = if pad_to_envelope {
            self.tokenizer
                .encode_document(&text, self.document_token_count)
        } else {
            self.tokenizer.encode_compact_document(&text)
        };
        let token_length_is_valid = if pad_to_envelope {
            tokens.len() == self.document_token_count
        } else {
            tokens.len() <= self.document_token_count
        };
        if !token_length_is_valid {
            return Err(anyhow!(
                "ruliad document token length drifted (envelope={} actual={} padded={})",
                self.document_token_count,
                tokens.len(),
                pad_to_envelope
            ));
        }
        let math_domains = ruliad_sample_math_domains(&sample.spec)
            .into_iter()
            .map(|domain| domain.label().to_string())
            .collect::<Vec<_>>();
        let mut reasoning_modes = ruliad_sample_reasoning_modes(&sample.spec)
            .into_iter()
            .map(|mode| mode.label().to_string())
            .collect::<Vec<_>>();
        if matches!(&sample.spec, RuliadSampleSpec::FormalProof { .. }) {
            use crate::ruliad::config::RuliadFormalGeneralizationContract;

            let partition = match (self.config.formal_generalization, split) {
                (RuliadFormalGeneralizationContract::SeedDisjointV1, _) => "seed_disjoint_v1",
                (RuliadFormalGeneralizationContract::StructuralHoldoutV1, SampleSplit::Train) => {
                    "structural_train_v1"
                }
                (
                    RuliadFormalGeneralizationContract::StructuralHoldoutV1,
                    SampleSplit::Validation,
                ) => "structural_validation_v1",
            };
            reasoning_modes.push(partition.to_string());
        }
        let token_count = tokens.len();
        Ok(RuliadRuntimeSampleDocument {
            split,
            sample_index,
            spec: sample.spec,
            categorical_presentation: sample.categorical_presentation,
            family: sample.family.label().to_string(),
            task_kind: sample.task_kind.label().to_string(),
            oracle_hash: sample.oracle_hash,
            verifier_version: sample.verifier_version,
            math_domains,
            reasoning_modes,
            source_difficulty_level: None,
            generation_retry_count: 0,
            token_count,
            tokens,
            serialized_preview: text,
            stats: sample.stats,
        })
    }

    pub fn probe_summary(
        &self,
        split: SampleSplit,
        max_samples: usize,
    ) -> Result<crate::runtime::RuntimeCorpusSummary> {
        let sample_count = self.sample_count(split);
        let probe_count = sample_count.min(max_samples.max(1));
        let mut gzip_ratios = Vec::with_capacity(probe_count);
        let mut complexity_scores = Vec::with_capacity(probe_count);
        for sample_index in 0..probe_count {
            let sample =
                self.generate_document_for_epoch_with_padding(split, 0, sample_index, false)?;
            gzip_ratios.push(sample.stats.gzip_complexity_ratio);
            complexity_scores.push(sample.stats.complexity_score);
        }
        let complexity_histogram = build_complexity_histogram(&complexity_scores);
        Ok(crate::runtime::RuntimeCorpusSummary {
            sample_count,
            token_count: sample_count.saturating_mul(self.document_token_count()),
            document_token_count: self.document_token_count(),
            mean_gzip_complexity_ratio: mean(gzip_ratios.iter().copied()),
            min_gzip_complexity_ratio: min(gzip_ratios.iter().copied()),
            max_gzip_complexity_ratio: max(gzip_ratios.iter().copied()),
            mean_complexity_score: mean(complexity_scores.iter().copied()),
            complexity_histogram,
        })
    }

    pub fn default_probe_summary(
        &self,
        split: SampleSplit,
    ) -> Result<crate::runtime::RuntimeCorpusSummary> {
        self.probe_summary(split, DEFAULT_PROBE_SAMPLES)
    }
}

fn source_bucket_retry_sample_index(sample_index: usize, retry: usize) -> usize {
    sample_index.saturating_add(retry.saturating_mul(1_000_003))
}

fn is_document_payload_capacity_error(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause
            .to_string()
            .contains("exceeds document payload capacity")
    })
}

pub fn fixed_ruliad_document_token_count(config: &RuliadCorpusConfig) -> Result<usize> {
    if config.serialization.document_tokens <= 1 {
        return Err(anyhow!("ruliad document token count must be > 1"));
    }
    Ok(config
        .serialization
        .document_tokens
        .saturating_mul(config.serialization.document_chunks.max.max(1)))
}

fn multi_chunk_count_for_document(
    config: &RuliadCorpusConfig,
    split: SampleSplit,
    epoch_index: usize,
    sample_index: usize,
    source_salt: u64,
) -> usize {
    let min = config.serialization.document_chunks.min.max(1);
    let max = config.serialization.document_chunks.max.max(min);
    if min == max {
        return min;
    }
    let epoch = match split {
        SampleSplit::Train => epoch_index as u64,
        SampleSplit::Validation => 0,
    };
    let split_tag = match split {
        SampleSplit::Train => 0xA11C_E5ED_D15C_A11A,
        SampleSplit::Validation => 0xBADC_0FFE_E5E1_7A1D,
    };
    let mut rng = SplitMix64::new(mix_seed(
        config.seed,
        [
            0xD0C5_7E11_9A7E_C0DE,
            split_tag,
            epoch,
            sample_index as u64,
            source_salt,
        ],
    ));
    rng.range_usize(min, max)
}

pub fn ruliad_serialized_node_count(text: &str) -> usize {
    text.lines()
        .filter(|line| line.trim_start().starts_with(">N"))
        .count()
}

fn multi_chunk_proof_tree_text(samples: &[GeneratedRuliadSample]) -> String {
    let Some(root) = samples.last() else {
        return "[R2 root v0 tree/empty]\nS:\nG:n=0\n?:empty\n!:empty\n[/R2]\n".to_string();
    };
    let (domains, modes) = multi_chunk_semantic_labels(samples);
    let root_view = &root.categorical_presentation;
    let mut text = format!(
        "[R2 root v{} tree/{}/{}/{}]\nS:{}|{}\nG:n={};root={}\n",
        root.verifier_version,
        compact_ruliad_label(root_view.source_family.as_str()),
        compact_ruliad_label(root_view.task_kind.as_str()),
        compact_ruliad_label(root_view.presentation.as_str()),
        compact_label_set(&domains),
        compact_label_set(&modes),
        samples.len(),
        compact_runtime_text(root_view.presentation.as_str(), 48)
    );
    for (node_index, sample) in samples.iter().enumerate() {
        let dependency = if node_index == 0 {
            "-".to_string()
        } else {
            format!("N{}", node_index - 1)
        };
        text.push_str(&multi_chunk_node_line(
            node_index,
            samples.len(),
            &dependency,
            sample,
        ));
        text.push('\n');
    }
    text.push_str(&format!(
        "?:root {}\nA:{}\n!:{}\n[/R2]\n",
        compact_runtime_text(root_view.query.as_str(), 96),
        ruliad_answer_contract(&root.spec),
        ruliad_expected_answer(&root.spec)
    ));
    text
}

fn multi_chunk_semantic_labels(
    samples: &[GeneratedRuliadSample],
) -> (BTreeSet<String>, BTreeSet<String>) {
    let mut domains = BTreeSet::new();
    let mut modes = BTreeSet::new();
    for sample in samples {
        domains.extend(
            ruliad_sample_math_domains(&sample.spec)
                .into_iter()
                .map(|domain| domain.label().to_string()),
        );
        modes.extend(
            ruliad_sample_reasoning_modes(&sample.spec)
                .into_iter()
                .map(|mode| mode.label().to_string()),
        );
    }
    (domains, modes)
}

fn compact_label_set(labels: &BTreeSet<String>) -> String {
    labels
        .iter()
        .map(|label| compact_ruliad_label(label.as_str()))
        .collect::<Vec<_>>()
        .join(",")
}

fn multi_chunk_node_line(
    node_index: usize,
    node_count: usize,
    dependency: &str,
    sample: &GeneratedRuliadSample,
) -> String {
    let view = &sample.categorical_presentation;
    let data = line_payload(sample.text.as_str(), "G:")
        .map(|value| compact_runtime_text(value, 80))
        .unwrap_or_else(|| compact_runtime_text(view.presentation.as_str(), 80));
    let proof = compact_node_proof(sample.text.as_str());
    let remaining = node_count.saturating_sub(node_index.saturating_add(1));
    let phase = if node_index == 0 {
        "first"
    } else if remaining == 0 {
        "last"
    } else {
        "mid"
    };
    format!(
        ">N{node_index}<{dependency} k={node_index};rem={remaining};phase={phase} {}/{}/{} d={} q={} p={} a={}",
        compact_ruliad_label(view.source_family.as_str()),
        compact_ruliad_label(view.task_kind.as_str()),
        compact_ruliad_label(view.presentation.as_str()),
        data,
        compact_runtime_text(view.query.as_str(), 72),
        proof,
        compact_runtime_text(view.answer.as_str(), 72)
    )
}

fn compact_node_proof(text: &str) -> String {
    let proof = text
        .lines()
        .filter_map(|line| line.strip_prefix('>'))
        .take(3)
        .map(|line| compact_runtime_text(line, 48))
        .collect::<Vec<_>>();
    if proof.is_empty() {
        "-".to_string()
    } else {
        proof.join(",")
    }
}

fn line_payload<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
    text.lines()
        .find_map(|line| line.trim_start().strip_prefix(prefix))
}

fn compact_runtime_text(value: &str, max_len: usize) -> String {
    let value = value.replace(['\n', '\r', '\t'], " ");
    let value = bound_runtime_repeated_chars(value.trim(), 6);
    let char_count = value.chars().count();
    if char_count <= max_len {
        return value;
    }
    if max_len <= 2 {
        return value.chars().take(max_len).collect();
    }
    format!(
        "{}..",
        value
            .chars()
            .take(max_len.saturating_sub(2))
            .collect::<String>()
    )
}

fn bound_runtime_repeated_chars(value: &str, max_run: usize) -> String {
    if max_run == 0 {
        return String::new();
    }
    let mut out = String::with_capacity(value.len());
    let mut chars = value.chars().peekable();
    while let Some(ch) = chars.next() {
        let mut run_len = 1usize;
        while chars.peek().is_some_and(|next| *next == ch) {
            chars.next();
            run_len = run_len.saturating_add(1);
        }
        let keep = run_len.min(max_run);
        for _ in 0..keep {
            out.push(ch);
        }
        if run_len > max_run {
            out.push('^');
            out.push_str(&run_len.to_string());
        }
    }
    out
}

fn load_configured_proof_tasks(config: &RuliadCorpusConfig) -> Result<Vec<LeanProofTask>> {
    if let Some(path) = &config.proof_tasks {
        let tasks = load_proof_tasks(path, config.lean_task_limit)?;
        if !tasks.is_empty() {
            return Ok(tasks);
        }
    }
    Ok(default_proof_tasks())
}

#[derive(Debug, Clone, Copy, Default)]
struct SupervisionMaskSummary {
    nonzero_targets: usize,
    weight_units: usize,
}

fn supervision_mask_summary(
    document_tokens: &[u32],
    mode: RuliadTokenSupervisionMode,
    supervision: RuliadTokenSupervisionConfig,
) -> SupervisionMaskSummary {
    let mut mask = vec![0; document_tokens.len().saturating_sub(1)];
    let supervision = RuliadTokenSupervisionConfig {
        mode,
        ..supervision
    };
    if !ruliad_token_loss_mask(document_tokens, &mut mask, supervision) {
        return SupervisionMaskSummary::default();
    }
    SupervisionMaskSummary {
        nonzero_targets: mask.iter().filter(|weight| **weight > 0).count(),
        weight_units: mask
            .iter()
            .filter_map(|weight| usize::try_from((*weight).max(0)).ok())
            .fold(0usize, usize::saturating_add),
    }
}

fn query_to_answer_token_span(tokenizer: &RuliadByteTokenizer, text: &str) -> Option<usize> {
    let query_line_break = text.find("\n?:")?;
    let query_start = query_line_break.saturating_add(1);
    let answer_line_break = text[query_start..]
        .find("\n!:")?
        .saturating_add(query_start);
    let answer_marker_end = answer_line_break.saturating_add(3);
    Some(tokenizer.payload_token_count(text.get(query_start..answer_marker_end)?))
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn ratio_mean(total: usize, count: usize) -> f64 {
    ratio(total, count)
}

fn mean(values: impl Iterator<Item = f32>) -> f32 {
    let values = values.collect::<Vec<_>>();
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

fn min(values: impl Iterator<Item = f32>) -> f32 {
    values.reduce(f32::min).unwrap_or_default()
}

fn max(values: impl Iterator<Item = f32>) -> f32 {
    values.reduce(f32::max).unwrap_or_default()
}

fn eval_item_from_document(document: RuliadRuntimeSampleDocument) -> RuliadEvalItem {
    let prompt = ruliad_prompt_prefix(&document.spec, &document.oracle_hash);
    let expected_answer = ruliad_expected_answer(&document.spec);
    RuliadEvalItem {
        oracle_hash: document.oracle_hash,
        sample_index: document.sample_index,
        split: document.split,
        family: document.family,
        task_kind: document.task_kind,
        math_domains: document.math_domains,
        reasoning_modes: document.reasoning_modes,
        prompt,
        expected_answer,
        difficulty_level: document.source_difficulty_level,
        spec: Some(document.spec),
    }
}

fn training_serialization_eval_item_from_document(
    document: RuliadRuntimeSampleDocument,
) -> Result<RuliadEvalItem> {
    let answer_marker = document
        .serialized_preview
        .rfind("\n!:")
        .ok_or_else(|| anyhow!("ruliad training document is missing its root answer slot"))?;
    let answer_start = answer_marker.saturating_add(3);
    let answer_tail = &document.serialized_preview[answer_start..];
    let answer_end = answer_tail.find('\n').unwrap_or(answer_tail.len());
    let expected_answer = answer_tail[..answer_end].trim().to_string();
    if expected_answer.is_empty() {
        return Err(anyhow!("ruliad training document has an empty root answer"));
    }

    let prompt = document.serialized_preview[..answer_start].to_string();
    let mut item = eval_item_from_document(document);
    item.prompt = prompt;
    item.expected_answer = expected_answer;
    Ok(item)
}

#[allow(dead_code)]
fn _assert_histogram_type(_: Vec<ComplexityHistogramBin>) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::UsizeRangeConfig;
    use crate::ruliad::config::{
        RuliadDocumentMode, RuliadFamilyConfig, RuliadFamilyKind, RuliadFormalTaskMixConfig,
        RuliadSerializationConfig, RuliadSourceSelectionConfig, RuliadTokenizationConfig,
        default_ruliad_families,
    };

    fn config() -> RuliadCorpusConfig {
        RuliadCorpusConfig {
            output_dir: "target/ruliad-runtime-test".into(),
            seed: 11,
            name: "runtime-test".to_string(),
            train_samples: 8,
            validation_samples: 2,
            chunk_token_capacity: 1024,
            serialization: RuliadSerializationConfig {
                document_tokens: 513,
                preview_samples: 2,
                ..RuliadSerializationConfig::default()
            },
            tokenization: RuliadTokenizationConfig::default(),
            formal_generalization: Default::default(),
            source_selection: crate::ruliad::config::RuliadSourceSelectionConfig::default(),
            families: default_ruliad_families(),
            proof_tasks: None,
            lean_task_limit: None,
        }
    }

    #[test]
    fn online_ruliad_is_deterministic_across_instances() {
        let left = OnlineRuliadCorpus::new(config()).expect("left");
        let right = OnlineRuliadCorpus::new(config()).expect("right");
        assert_eq!(
            left.generate_document_tokens_for_epoch(SampleSplit::Train, 3, 1)
                .expect("left doc"),
            right
                .generate_document_tokens_for_epoch(SampleSplit::Train, 3, 1)
                .expect("right doc")
        );
    }

    #[test]
    fn validation_ignores_epoch_index() {
        let corpus = OnlineRuliadCorpus::new(config()).expect("corpus");
        assert_eq!(
            corpus
                .generate_document_tokens_for_epoch(SampleSplit::Validation, 0, 1)
                .expect("doc"),
            corpus
                .generate_document_tokens_for_epoch(SampleSplit::Validation, 9, 1)
                .expect("doc")
        );
    }

    #[test]
    fn structural_holdout_partition_is_visible_to_eval_telemetry_not_model_symbols() {
        let mut config = config();
        config.serialization.document_tokens = 8192;
        config.formal_generalization =
            crate::ruliad::config::RuliadFormalGeneralizationContract::StructuralHoldoutV1;
        config.source_selection = RuliadSourceSelectionConfig {
            enabled: true,
            formal_task_mix: RuliadFormalTaskMixConfig {
                advance_proof_weight: 0,
                select_proof_action_weight: 1,
                construct_proof_weight: 0,
                check_proof_weight: 0,
                proof_action_answer_contract: Default::default(),
            },
            ..RuliadSourceSelectionConfig::default()
        };
        config.families = vec![RuliadFamilyConfig {
            kind: RuliadFamilyKind::FormalProof,
            weight: 1,
            width: Some(UsizeRangeConfig { min: 4, max: 4 }),
            steps: Some(UsizeRangeConfig { min: 3, max: 3 }),
        }];
        let corpus = OnlineRuliadCorpus::new(config).expect("structural corpus");
        let bucket = corpus.source_buckets()[0].label();
        let train = corpus
            .generate_document_for_source_bucket(SampleSplit::Train, 0, 0, &bucket)
            .expect("train document");
        let validation = corpus
            .generate_document_for_source_bucket(SampleSplit::Validation, 0, 0, &bucket)
            .expect("validation document");

        assert!(
            train
                .reasoning_modes
                .iter()
                .any(|mode| mode == "structural_train_v1")
        );
        assert!(
            validation
                .reasoning_modes
                .iter()
                .any(|mode| mode == "structural_validation_v1")
        );
        for document in [&train, &validation] {
            assert!(!document.serialized_preview.contains("identity_left"));
            assert!(!document.serialized_preview.contains("identity_right"));
            assert!(!document.serialized_preview.contains("compose"));
        }
    }

    #[test]
    fn seed_disjoint_control_is_explicit_in_eval_telemetry() {
        let mut config = config();
        config.serialization.document_tokens = 8192;
        config.families = vec![RuliadFamilyConfig {
            kind: RuliadFamilyKind::FormalProof,
            weight: 1,
            width: Some(UsizeRangeConfig { min: 2, max: 2 }),
            steps: Some(UsizeRangeConfig { min: 2, max: 2 }),
        }];
        let corpus = OnlineRuliadCorpus::new(config).expect("seed-disjoint corpus");
        let bucket = corpus.source_buckets()[0].label();
        let validation = corpus
            .generate_document_for_source_bucket(SampleSplit::Validation, 0, 0, &bucket)
            .expect("validation document");

        assert!(
            validation
                .reasoning_modes
                .iter()
                .any(|mode| mode == "seed_disjoint_v1")
        );
        assert!(
            validation
                .reasoning_modes
                .iter()
                .all(|mode| mode != "structural_validation_v1")
        );
    }

    #[test]
    fn structural_action_labels_are_balanced_in_both_partitions() {
        let mut config = config();
        config.serialization.document_tokens = 8192;
        config.formal_generalization =
            crate::ruliad::config::RuliadFormalGeneralizationContract::StructuralHoldoutV1;
        config.source_selection = RuliadSourceSelectionConfig {
            enabled: true,
            formal_task_mix: RuliadFormalTaskMixConfig {
                advance_proof_weight: 0,
                select_proof_action_weight: 1,
                construct_proof_weight: 0,
                check_proof_weight: 0,
                proof_action_answer_contract: Default::default(),
            },
            ..RuliadSourceSelectionConfig::default()
        };
        config.families = vec![RuliadFamilyConfig {
            kind: RuliadFamilyKind::FormalProof,
            weight: 1,
            width: Some(UsizeRangeConfig { min: 2, max: 4 }),
            steps: Some(UsizeRangeConfig { min: 2, max: 4 }),
        }];
        let corpus = OnlineRuliadCorpus::new(config).expect("structural corpus");
        let bucket = corpus.source_buckets()[0].label();

        for split in [SampleSplit::Train, SampleSplit::Validation] {
            let mut counts = [0usize; 4];
            for sample_index in 0..256 {
                let item = corpus
                    .generate_eval_item_for_source_bucket(split, 0, sample_index, &bucket)
                    .expect("action item");
                let index = crate::ruliad::policy::parse_proof_action_index(&item.expected_answer)
                    .expect("action index");
                counts[index] += 1;
            }
            let entropy = counts.iter().fold(0.0f64, |entropy, count| {
                let probability = *count as f64 / 256.0;
                entropy - probability * probability.log2()
            });
            assert!(
                counts.iter().all(|count| (38..=90).contains(count)),
                "{split:?} action labels are position-biased: {counts:?}"
            );
            assert!(
                entropy >= 1.9,
                "{split:?} action-label entropy is too low: entropy={entropy} counts={counts:?}"
            );
        }
    }

    #[test]
    fn ruliad_documents_have_fixed_length() {
        let corpus = OnlineRuliadCorpus::new(config()).expect("corpus");
        let doc = corpus
            .generate_document(SampleSplit::Train, 0)
            .expect("document");
        assert_eq!(doc.tokens.len(), 513);
    }

    #[test]
    fn compact_ruliad_documents_end_at_eos_without_envelope_padding() {
        let corpus = OnlineRuliadCorpus::new(config()).expect("corpus");
        let compact = corpus
            .generate_compact_document_tokens_for_epoch(SampleSplit::Train, 0, 0)
            .expect("compact document");
        assert!(compact.len() < corpus.document_token_count());
        assert_eq!(compact.last().copied(), corpus.tokenizer_manifest().eos_id);
        let padded = corpus
            .generate_document_tokens_for_epoch(SampleSplit::Train, 0, 0)
            .expect("padded document");
        assert_eq!(padded.len(), corpus.document_token_count());
        assert_eq!(&padded[..compact.len()], compact.as_slice());
    }

    #[test]
    fn multi_chunk_ruliad_documents_span_tbptt_chunks() {
        let mut config = config();
        config.serialization.document_mode = RuliadDocumentMode::MultiChunkProofTree;
        config.serialization.document_chunks = UsizeRangeConfig { min: 3, max: 3 };
        config.families = vec![RuliadFamilyConfig {
            kind: RuliadFamilyKind::ProofTree,
            weight: 1,
            width: Some(UsizeRangeConfig { min: 5, max: 7 }),
            steps: Some(UsizeRangeConfig { min: 4, max: 6 }),
        }];
        let corpus = OnlineRuliadCorpus::new(config).expect("corpus");
        assert_eq!(corpus.document_token_count(), 1539);
        let doc = corpus
            .generate_document(SampleSplit::Train, 0)
            .expect("document");
        assert_eq!(doc.tokens.len(), 1539);
        assert_eq!(doc.serialized_preview.matches("[R2 ").count(), 1);
        assert_eq!(ruliad_serialized_node_count(&doc.serialized_preview), 3);
        assert!(doc.serialized_preview.contains("\n>N0<-"));
        assert!(doc.serialized_preview.contains("k=0;rem=2;phase=first"));
        assert!(doc.serialized_preview.contains("k=1;rem=1;phase=mid"));
        assert!(doc.serialized_preview.contains("k=2;rem=0;phase=last"));
        assert!(
            !doc.serialized_preview.contains(";nodes="),
            "multi-chunk root answers must use the canonical compact answer dialect"
        );
        assert!(
            doc.serialized_preview.contains("\nA:"),
            "multi-chunk root prompts must expose the canonical answer-key contract"
        );
        assert!(
            doc.serialized_preview.find("\nA:") < doc.serialized_preview.find("\n!:"),
            "multi-chunk root answer-key contract must precede the answer slot"
        );
        let answer_line = doc
            .serialized_preview
            .lines()
            .find(|line| line.starts_with("!:"))
            .expect("multi-chunk document answer line");
        assert_eq!(
            answer_line,
            format!("!:{}", ruliad_expected_answer(&doc.spec)),
            "multi-chunk root answer slot must train the full keyed expected answer"
        );
        assert!(doc.serialized_preview.contains("[/R2]"));
    }

    #[test]
    fn training_serialization_eval_preserves_multi_chunk_prompt_contract() {
        let mut config = config();
        config.serialization.document_tokens = 1539;
        config.serialization.document_mode = RuliadDocumentMode::MultiChunkProofTree;
        config.serialization.document_chunks = UsizeRangeConfig { min: 3, max: 3 };
        config.families = vec![RuliadFamilyConfig {
            kind: RuliadFamilyKind::FormalProof,
            weight: 1,
            width: Some(UsizeRangeConfig { min: 2, max: 2 }),
            steps: Some(UsizeRangeConfig { min: 2, max: 2 }),
        }];
        let corpus = OnlineRuliadCorpus::new(config).expect("corpus");

        let canonical = corpus
            .generate_eval_item_for_epoch(SampleSplit::Validation, 0, 0)
            .expect("canonical item");
        let matched = corpus
            .generate_training_serialization_eval_item_for_epoch(SampleSplit::Validation, 0, 0)
            .expect("training-serialization item");

        assert!(canonical.prompt.trim_start().starts_with("[R3"));
        assert!(matched.prompt.trim_start().starts_with("[R2"));
        assert!(matched.prompt.ends_with("\n!:"));
        assert_eq!(matched.document_close_marker(), "[/R2]");
        assert_eq!(matched.expected_answer, canonical.expected_answer);
        assert_eq!(matched.oracle_hash, canonical.oracle_hash);
    }

    #[test]
    fn multi_chunk_ruliad_documents_sample_configured_chunk_range() {
        let mut config = config();
        config.serialization.document_mode = RuliadDocumentMode::MultiChunkProofTree;
        config.serialization.document_chunks = UsizeRangeConfig { min: 4, max: 8 };
        config.families = vec![RuliadFamilyConfig {
            kind: RuliadFamilyKind::ProofTree,
            weight: 1,
            width: Some(UsizeRangeConfig { min: 5, max: 7 }),
            steps: Some(UsizeRangeConfig { min: 4, max: 6 }),
        }];
        let corpus = OnlineRuliadCorpus::new(config).expect("corpus");
        let counts = (0..32)
            .map(|sample_index| {
                let document = corpus
                    .generate_document(SampleSplit::Train, sample_index)
                    .expect("document")
                    .serialized_preview;
                ruliad_serialized_node_count(&document)
            })
            .collect::<Vec<_>>();
        assert!(counts.iter().all(|count| (4..=8).contains(count)));
        assert!(
            counts.windows(2).any(|pair| pair[0] != pair[1]),
            "configured multi-chunk range should vary across documents: {counts:?}"
        );
    }

    #[test]
    fn far_out_mixed_difficulty_stays_within_fixed_payload() {
        let mut config = config();
        config.serialization.document_mode = RuliadDocumentMode::MultiChunkProofTree;
        config.serialization.document_chunks = UsizeRangeConfig { min: 4, max: 8 };
        config.source_selection.difficulty_levels = UsizeRangeConfig { min: 24, max: 24 };
        let corpus = OnlineRuliadCorpus::new(config).expect("corpus");
        assert_eq!(corpus.document_token_count(), 4104);
        let payload_capacity = corpus
            .tokenizer_manifest()
            .eos_id
            .map_or(corpus.document_token_count(), |_| {
                corpus.document_token_count().saturating_sub(1)
            });
        for sample_index in 0..8 {
            let doc = corpus
                .generate_document(SampleSplit::Train, sample_index)
                .expect("far-out mixed document");
            assert_eq!(doc.tokens.len(), 4104);
            assert!(doc.serialized_preview.len() <= payload_capacity);
            assert_eq!(doc.serialized_preview.matches("[R2 ").count(), 1);
            assert!(ruliad_serialized_node_count(&doc.serialized_preview) >= 4);
        }
    }

    #[test]
    fn source_bucket_generation_reports_bounded_capacity_retries() {
        let mut config = config();
        config.seed = 1337;
        config.train_samples = 256;
        config.serialization.document_mode = RuliadDocumentMode::MultiChunkProofTree;
        config.serialization.document_chunks = UsizeRangeConfig { min: 4, max: 8 };
        config.source_selection = RuliadSourceSelectionConfig {
            enabled: true,
            difficulty_levels: UsizeRangeConfig { min: 18, max: 18 },
            ..RuliadSourceSelectionConfig::default()
        };
        config.families = vec![RuliadFamilyConfig {
            kind: RuliadFamilyKind::Automaton,
            weight: 1,
            width: Some(UsizeRangeConfig { min: 3, max: 8 }),
            steps: Some(UsizeRangeConfig { min: 6, max: 20 }),
        }];
        let corpus = OnlineRuliadCorpus::new(config).expect("corpus");
        let bucket = corpus
            .source_buckets()
            .iter()
            .find(|bucket| bucket.id.difficulty_level == 18)
            .expect("automaton bucket")
            .label();
        let doc = corpus
            .generate_document_for_source_bucket(SampleSplit::Train, 0, 27, &bucket)
            .expect("source-selected document retries overlong draw");
        assert_eq!(doc.tokens.len(), 4104);
        assert_eq!(doc.sample_index, 27);
        assert_eq!(doc.family, "automaton");
        assert_eq!(doc.task_kind, "evaluate_automaton");
        assert!(doc.generation_retry_count < SOURCE_BUCKET_DOCUMENT_RETRY_LIMIT);
    }

    #[test]
    fn formal_source_overflow_is_explicit_and_never_changes_seed() {
        let mut config = config();
        config.serialization.document_tokens = 64;
        config.serialization.document_mode = RuliadDocumentMode::MultiChunkProofTree;
        config.serialization.document_chunks = UsizeRangeConfig { min: 1, max: 1 };
        config.source_selection = RuliadSourceSelectionConfig {
            enabled: true,
            difficulty_levels: UsizeRangeConfig { min: 8, max: 8 },
            ..RuliadSourceSelectionConfig::default()
        };
        config.families = vec![RuliadFamilyConfig {
            kind: RuliadFamilyKind::FormalProof,
            weight: 1,
            width: Some(UsizeRangeConfig { min: 4, max: 4 }),
            steps: Some(UsizeRangeConfig { min: 4, max: 4 }),
        }];
        let corpus = OnlineRuliadCorpus::new(config).expect("corpus");
        let bucket = corpus.source_buckets()[0].label();
        let error = corpus
            .generate_document_for_source_bucket(SampleSplit::Train, 0, 0, &bucket)
            .expect_err("formal sample exceeds intentionally tiny envelope");
        let message = error.to_string();
        assert!(
            message.contains("seed substitution is disabled"),
            "{message}"
        );
        assert!(
            message.contains("exceeds document payload capacity"),
            "{message}"
        );
    }

    #[test]
    fn formal_supervision_audit_reports_realized_stream_compute_imbalance() {
        let mut config = config();
        config.chunk_token_capacity = 32_768;
        config.serialization.document_tokens = 16_384;
        config.source_selection = RuliadSourceSelectionConfig {
            enabled: true,
            difficulty_levels: UsizeRangeConfig { min: 0, max: 0 },
            formal_task_mix: RuliadFormalTaskMixConfig {
                advance_proof_weight: 2,
                select_proof_action_weight: 0,
                construct_proof_weight: 1,
                check_proof_weight: 1,
                proof_action_answer_contract: Default::default(),
            },
            ..RuliadSourceSelectionConfig::default()
        };
        config.families = vec![RuliadFamilyConfig {
            kind: RuliadFamilyKind::FormalProof,
            weight: 1,
            width: Some(UsizeRangeConfig { min: 2, max: 2 }),
            steps: Some(UsizeRangeConfig { min: 2, max: 2 }),
        }];
        let corpus = OnlineRuliadCorpus::new(config).expect("corpus");

        let report = corpus
            .audit_frontier_supervision(
                &[0],
                4,
                128,
                RuliadTokenSupervisionConfig {
                    mask_high_entropy_spans: true,
                    ..RuliadTokenSupervisionConfig::default()
                },
            )
            .expect("audit");

        assert_eq!(report.version, RULIAD_SUPERVISION_AUDIT_VERSION);
        assert_eq!(report.sample_count, 12);
        assert_eq!(report.buckets.len(), 3);
        assert!(report.total_stream_chunks > 0);
        assert!(report.total_answer_target_tokens > 0);
        assert!(report.total_trace_answer_target_tokens > report.total_answer_target_tokens);
        assert!(report.max_to_min_mean_stream_chunks_ratio >= 1.0);
        assert_eq!(report.total_query_conditioning_samples, report.sample_count);
        assert_eq!(report.query_visible_within_block_fraction, 1.0);
        let stream_share_sum = report
            .buckets
            .iter()
            .map(|bucket| bucket.stream_chunk_share)
            .sum::<f64>();
        assert!((stream_share_sum - 1.0).abs() < 1.0e-9);
        assert!(
            report
                .buckets
                .iter()
                .any(|bucket| bucket.answer_contract == "certificate")
        );
        assert!(
            report
                .buckets
                .iter()
                .any(|bucket| bucket.answer_contract == "ok,vg,vs,g,s,k")
        );
        let transition = report
            .buckets
            .iter()
            .find(|bucket| bucket.answer_contract == "proof_step")
            .expect("transition bucket");
        assert_eq!(transition.query_conditioning_samples, 4);
        assert_eq!(transition.query_visible_within_block_fraction, 1.0);
        assert!(transition.max_query_to_answer_tokens <= 128);
    }

    #[test]
    fn source_bucket_generation_resolves_dynamic_frontier_label() {
        let mut config = config();
        config.source_selection = RuliadSourceSelectionConfig {
            enabled: true,
            difficulty_levels: UsizeRangeConfig { min: 0, max: 0 },
            ..RuliadSourceSelectionConfig::default()
        };
        config.families = vec![RuliadFamilyConfig {
            kind: RuliadFamilyKind::Eca,
            weight: 1,
            width: Some(UsizeRangeConfig { min: 8, max: 8 }),
            steps: Some(UsizeRangeConfig { min: 2, max: 2 }),
        }];
        let dynamic_bucket =
            crate::ruliad::source_selection::ruliad_source_buckets_for_difficulty(&config, 3)
                .into_iter()
                .next()
                .expect("dynamic bucket")
                .label();

        let corpus = OnlineRuliadCorpus::new(config).expect("corpus");
        assert!(
            corpus
                .source_buckets()
                .iter()
                .all(|bucket| bucket.label() != dynamic_bucket)
        );
        let doc = corpus
            .generate_document_for_source_bucket(SampleSplit::Train, 0, 0, &dynamic_bucket)
            .expect("dynamic frontier document");
        assert_eq!(doc.family, "eca");
        assert_eq!(doc.task_kind, "multi_step_state");
        assert!(!doc.serialized_preview.is_empty());
    }

    #[test]
    fn forced_source_bucket_generation_matches_bucket_task() {
        let mut config = config();
        config.families = vec![crate::ruliad::config::RuliadFamilyConfig {
            kind: crate::ruliad::config::RuliadFamilyKind::Eca,
            weight: 1,
            width: Some(crate::config::UsizeRangeConfig { min: 8, max: 8 }),
            steps: Some(crate::config::UsizeRangeConfig { min: 1, max: 2 }),
        }];
        let corpus = OnlineRuliadCorpus::new(config).expect("corpus");
        let next_state = corpus
            .generate_document_for_source_bucket(SampleSplit::Train, 0, 0, "eca:next_state")
            .expect("next state");
        assert_eq!(next_state.family, "eca");
        assert_eq!(next_state.task_kind, "next_state");
        let multi_step = corpus
            .generate_document_for_source_bucket(SampleSplit::Train, 0, 1, "eca:multi_step_state")
            .expect("multi step");
        assert_eq!(multi_step.task_kind, "multi_step_state");
    }
}
