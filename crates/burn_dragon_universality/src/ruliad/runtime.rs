use std::path::Path;
use std::sync::Arc;

use anyhow::{Result, anyhow};

use crate::manifest::{SampleSplit, UniversalityTokenizerManifest};
use crate::ruliad::config::{RuliadCorpusConfig, load_ruliad_config, ruliad_source_semantics};
use crate::ruliad::oracles::{
    GeneratedRuliadSample, LeanProofTask, RuliadCategoricalPresentation, RuliadSampleSpec,
    default_proof_tasks, generate_sample, generate_sample_for_source_bucket, load_proof_tasks,
};
use crate::ruliad::search::RuliadSamplerCandidate;
use crate::ruliad::source_selection::{
    RuliadSourceBucket, ruliad_sampler_candidates, ruliad_source_buckets,
};
use crate::ruliad::tokenize::RuliadByteTokenizer;
use crate::stats::{ComplexityHistogramBin, SampleStats, build_complexity_histogram};

const DEFAULT_PROBE_SAMPLES: usize = 32;

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
        let sample = self.generate_raw_sample(split, epoch_index, sample_index)?;
        self.document_from_sample(split, sample_index, sample)
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
            .ok_or_else(|| anyhow!("unknown ruliad source bucket `{bucket_label}`"))?;
        let sample = generate_sample_for_source_bucket(
            &self.config,
            self.proof_tasks.as_slice(),
            split,
            epoch_index,
            sample_index,
            bucket,
        )?;
        self.document_from_sample(split, sample_index, sample)
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

    fn document_from_sample(
        &self,
        split: SampleSplit,
        sample_index: usize,
        sample: GeneratedRuliadSample,
    ) -> Result<RuliadRuntimeSampleDocument> {
        let tokens = self
            .tokenizer
            .encode_document(&sample.text, self.document_token_count);
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
            serialized_preview: sample.text,
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

pub fn fixed_ruliad_document_token_count(config: &RuliadCorpusConfig) -> Result<usize> {
    if config.serialization.document_tokens <= 1 {
        return Err(anyhow!("ruliad document token count must be > 1"));
    }
    Ok(config.serialization.document_tokens)
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
    use crate::ruliad::config::{
        RuliadSerializationConfig, RuliadTokenizationConfig, default_ruliad_families,
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
                document_tokens: 96,
                preview_samples: 2,
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
        assert_eq!(doc.tokens.len(), 96);
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
