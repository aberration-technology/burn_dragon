use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Result, anyhow};

use crate::manifest::{SampleSplit, UniversalityTokenizerManifest};
use crate::ruliad::config::{
    RuliadCorpusConfig, RuliadDocumentMode, load_ruliad_config, ruliad_source_semantics,
};
use crate::ruliad::oracles::{
    GeneratedRuliadSample, LeanProofTask, RuliadCategoricalPresentation, RuliadSampleSpec,
    compact_ruliad_label, default_proof_tasks, generate_sample, generate_sample_for_source_bucket,
    load_proof_tasks,
};
use crate::ruliad::rng::{SplitMix64, mix_seed};
use crate::ruliad::search::RuliadSamplerCandidate;
use crate::ruliad::source_selection::{
    RuliadSourceBucket, ruliad_sampler_candidates, ruliad_source_bucket_by_label,
    ruliad_source_buckets,
};
use crate::ruliad::tokenize::RuliadByteTokenizer;
use crate::stats::{ComplexityHistogramBin, SampleStats, build_complexity_histogram};

const DEFAULT_PROBE_SAMPLES: usize = 32;
const SOURCE_BUCKET_DOCUMENT_RETRY_LIMIT: usize = 32;

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
    pub token_count: usize,
    pub tokens: Vec<u32>,
    pub serialized_preview: String,
    pub stats: SampleStats,
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

    pub fn sampler_candidates(&self) -> Vec<RuliadSamplerCandidate> {
        self.sampler_candidates.as_ref().clone()
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
        if self.config.serialization.document_mode == RuliadDocumentMode::MultiChunkProofTree {
            return self.generate_multi_chunk_document_for_epoch(split, epoch_index, sample_index);
        }
        let sample = self.generate_raw_sample(split, epoch_index, sample_index)?;
        let text = sample.text.clone();
        self.document_from_sample_text(split, sample_index, sample, text)
    }

    pub fn generate_document_for_source_bucket(
        &self,
        split: SampleSplit,
        epoch_index: usize,
        sample_index: usize,
        bucket_label: &str,
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
        let mut last_error = None;
        for retry in 0..SOURCE_BUCKET_DOCUMENT_RETRY_LIMIT {
            let candidate_sample_index = source_bucket_retry_sample_index(sample_index, retry);
            let result = if self.config.serialization.document_mode
                == RuliadDocumentMode::MultiChunkProofTree
            {
                self.generate_multi_chunk_document_for_source_bucket(
                    split,
                    epoch_index,
                    candidate_sample_index,
                    &bucket,
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
                self.document_from_sample_text(split, candidate_sample_index, sample, text)
            };
            match result {
                Ok(mut document) => {
                    document.sample_index = sample_index;
                    return Ok(document);
                }
                Err(error) if is_document_payload_capacity_error(&error) => {
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
        self.document_from_sample_text(split, sample_index, root, text)
    }

    fn generate_multi_chunk_document_for_source_bucket(
        &self,
        split: SampleSplit,
        epoch_index: usize,
        sample_index: usize,
        bucket: &RuliadSourceBucket,
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
        self.document_from_sample_text(split, sample_index, root, text)
    }

    fn document_from_sample_text(
        &self,
        split: SampleSplit,
        sample_index: usize,
        sample: GeneratedRuliadSample,
        text: String,
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
        let tokens = self
            .tokenizer
            .encode_document(&text, self.document_token_count);
        if tokens.len() != self.document_token_count {
            return Err(anyhow!(
                "ruliad document token length drifted (expected={} actual={})",
                self.document_token_count,
                tokens.len()
            ));
        }
        let semantics = ruliad_source_semantics(sample.family, sample.task_kind);
        let math_domains = semantics
            .math_domains
            .iter()
            .map(|domain| domain.label().to_string())
            .collect::<Vec<_>>();
        let reasoning_modes = semantics
            .reasoning_modes
            .iter()
            .map(|mode| mode.label().to_string())
            .collect::<Vec<_>>();
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
            let sample = self.generate_document(split, sample_index)?;
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
        "?:root {}\n!:{};nodes={}\n[/R2]\n",
        compact_runtime_text(root_view.query.as_str(), 96),
        compact_runtime_text(root_view.answer.as_str(), 96),
        samples.len()
    ));
    text
}

fn multi_chunk_semantic_labels(
    samples: &[GeneratedRuliadSample],
) -> (BTreeSet<String>, BTreeSet<String>) {
    let mut domains = BTreeSet::new();
    let mut modes = BTreeSet::new();
    for sample in samples {
        let semantics = ruliad_source_semantics(sample.family, sample.task_kind);
        domains.extend(
            semantics
                .math_domains
                .iter()
                .map(|domain| domain.label().to_string()),
        );
        modes.extend(
            semantics
                .reasoning_modes
                .iter()
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

#[allow(dead_code)]
fn _assert_histogram_type(_: Vec<ComplexityHistogramBin>) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::UsizeRangeConfig;
    use crate::ruliad::config::{
        RuliadDocumentMode, RuliadFamilyConfig, RuliadFamilyKind, RuliadSerializationConfig,
        RuliadSourceSelectionConfig, RuliadTokenizationConfig, default_ruliad_families,
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
    fn ruliad_documents_have_fixed_length() {
        let corpus = OnlineRuliadCorpus::new(config()).expect("corpus");
        let doc = corpus
            .generate_document(SampleSplit::Train, 0)
            .expect("document");
        assert_eq!(doc.tokens.len(), 513);
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
        assert!(doc.serialized_preview.contains("[/R2]"));
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
    fn source_bucket_generation_retries_overlong_far_out_draws() {
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
