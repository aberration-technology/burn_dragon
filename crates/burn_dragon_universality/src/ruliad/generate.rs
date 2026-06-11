use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::manifest::{
    CorpusKind, SampleSplit, UniversalityChunkManifest, UniversalityCorpusManifest,
    UniversalitySampleRecord, complexity_histogram, write_manifest, write_sample_records,
};
use crate::ruliad::config::RuliadCorpusConfig;
use crate::ruliad::runtime::OnlineRuliadCorpus;
use crate::stats::CorpusStats;

#[derive(Debug, Clone)]
pub struct GeneratedRuliadCorpusReport {
    pub output_dir: PathBuf,
    pub manifest_path: PathBuf,
    pub sample_records_path: PathBuf,
    pub train_token_count: usize,
    pub val_token_count: usize,
    pub token_count: usize,
    pub sample_count: usize,
}

pub fn generate_ruliad_corpus(config: &RuliadCorpusConfig) -> Result<GeneratedRuliadCorpusReport> {
    let corpus = OnlineRuliadCorpus::new(config.clone())?;
    fs::create_dir_all(&config.output_dir)
        .with_context(|| format!("failed to create {}", config.output_dir.display()))?;
    let chunk_dir = PathBuf::from("chunks");
    let preview_dir = PathBuf::from("previews");
    let sample_records_path = PathBuf::from("sample_records.jsonl");
    fs::create_dir_all(config.output_dir.join(&chunk_dir))?;
    fs::create_dir_all(config.output_dir.join(&preview_dir))?;

    let mut chunks = Vec::new();
    let mut records = Vec::new();
    let mut chunk_tokens = Vec::<u32>::new();
    let mut chunk_split = SampleSplit::Train;
    let mut chunk_start = 0usize;
    let mut chunk_sample_count = 0usize;
    let mut token_offset = 0usize;
    let mut chunk_index = 0usize;

    for split in [SampleSplit::Train, SampleSplit::Validation] {
        let sample_count = corpus.sample_count(split);
        for sample_index in 0..sample_count {
            let document = corpus.generate_document(split, sample_index)?;
            if !chunk_tokens.is_empty()
                && (chunk_split != split
                    || chunk_tokens.len().saturating_add(document.tokens.len())
                        > config.chunk_token_capacity)
            {
                flush_chunk(
                    &config.output_dir,
                    &chunk_dir,
                    &mut chunks,
                    &mut chunk_tokens,
                    chunk_split,
                    chunk_start,
                    chunk_sample_count,
                    &mut chunk_index,
                )?;
                chunk_start = token_offset;
                chunk_sample_count = 0;
            }
            if chunk_tokens.is_empty() {
                chunk_split = split;
                chunk_start = token_offset;
            }

            let preview_path = if records.len() < config.serialization.preview_samples {
                let file = format!(
                    "{}-{:06}.txt",
                    match split {
                        SampleSplit::Train => "train",
                        SampleSplit::Validation => "validation",
                    },
                    sample_index
                );
                let relative = preview_dir.join(file);
                let absolute = config.output_dir.join(&relative);
                fs::write(&absolute, &document.serialized_preview)
                    .with_context(|| format!("failed to write {}", absolute.display()))?;
                Some(relative)
            } else {
                None
            };

            records.push(UniversalitySampleRecord {
                sample_index,
                split,
                family: document.family,
                complexity_band: document.task_kind.clone(),
                rule_seed: None,
                complexity_filter_matched: true,
                identity_bias: 0.0,
                temperature: 0.0,
                step_stride: 1,
                start_step: 0,
                token_offset,
                token_count: document.tokens.len(),
                preview_path,
                serialized_char_count: document.serialized_preview.len(),
                stats: document.stats,
                ruliad_spec: Some(serde_json::to_value(document.spec)?),
                oracle_hash: Some(document.oracle_hash),
                task_kind: Some(document.task_kind),
                verifier_version: Some(document.verifier_version),
                math_domains: document.math_domains,
                reasoning_modes: document.reasoning_modes,
            });
            token_offset = token_offset.saturating_add(document.tokens.len());
            chunk_sample_count += 1;
            chunk_tokens.extend(document.tokens);
        }
    }
    if !chunk_tokens.is_empty() {
        flush_chunk(
            &config.output_dir,
            &chunk_dir,
            &mut chunks,
            &mut chunk_tokens,
            chunk_split,
            chunk_start,
            chunk_sample_count,
            &mut chunk_index,
        )?;
    }

    let train_token_count = corpus.train_token_count();
    let val_token_count = corpus.val_token_count();
    let stats = corpus_stats(&records, token_offset);
    let manifest = UniversalityCorpusManifest {
        version: crate::manifest::UNIVERSALITY_MANIFEST_VERSION,
        corpus_kind: CorpusKind::Ruliad,
        dataset_name: config.name.clone(),
        seed: config.seed,
        train_token_count,
        val_token_count,
        token_count: token_offset,
        chunk_token_capacity: config.chunk_token_capacity,
        tokenizer: corpus.tokenizer_manifest().clone(),
        chunk_dir,
        preview_dir,
        sample_records_path: sample_records_path.clone(),
        chunks,
        stats,
    };
    let manifest_path = config.output_dir.join("manifest.json");
    write_manifest(&manifest_path, &manifest)?;
    write_sample_records(&config.output_dir.join(&sample_records_path), &records)?;

    Ok(GeneratedRuliadCorpusReport {
        output_dir: config.output_dir.clone(),
        manifest_path,
        sample_records_path: config.output_dir.join(sample_records_path),
        train_token_count,
        val_token_count,
        token_count: token_offset,
        sample_count: records.len(),
    })
}

fn flush_chunk(
    output_dir: &std::path::Path,
    chunk_dir: &std::path::Path,
    chunks: &mut Vec<UniversalityChunkManifest>,
    chunk_tokens: &mut Vec<u32>,
    split: SampleSplit,
    token_offset: usize,
    sample_count: usize,
    chunk_index: &mut usize,
) -> Result<()> {
    let file_name = format!("chunk-{chunk_index:05}.bin");
    let path = output_dir.join(chunk_dir).join(&file_name);
    let mut bytes = Vec::with_capacity(chunk_tokens.len().saturating_mul(4));
    for token in chunk_tokens.iter().copied() {
        bytes.write_all(&token.to_le_bytes())?;
    }
    fs::write(&path, bytes).with_context(|| format!("failed to write {}", path.display()))?;
    chunks.push(UniversalityChunkManifest {
        file_name,
        split,
        token_offset,
        token_count: chunk_tokens.len(),
        sample_count,
    });
    chunk_tokens.clear();
    *chunk_index += 1;
    Ok(())
}

fn corpus_stats(samples: &[UniversalitySampleRecord], total_token_count: usize) -> CorpusStats {
    let train_samples = samples
        .iter()
        .filter(|sample| sample.split == SampleSplit::Train)
        .count();
    let validation_samples = samples.len().saturating_sub(train_samples);
    let mean_token_count = if samples.is_empty() {
        0.0
    } else {
        total_token_count as f32 / samples.len() as f32
    };
    let mean_entropy_bits = mean(samples.iter().map(|sample| sample.stats.mean_entropy_bits));
    let mean_transition_rate = mean(
        samples
            .iter()
            .map(|sample| sample.stats.mean_transition_rate),
    );
    let mean_active_ratio = mean(samples.iter().map(|sample| sample.stats.active_ratio_mean));
    let gzip_values = samples
        .iter()
        .map(|sample| sample.stats.gzip_complexity_ratio)
        .collect::<Vec<_>>();
    let complexity_values = samples
        .iter()
        .map(|sample| sample.stats.complexity_score)
        .collect::<Vec<_>>();
    CorpusStats {
        total_samples: samples.len(),
        train_samples,
        validation_samples,
        total_token_count,
        mean_token_count,
        mean_entropy_bits,
        mean_transition_rate,
        mean_active_ratio,
        mean_gzip_complexity_ratio: mean(gzip_values.iter().copied()),
        min_gzip_complexity_ratio: min(gzip_values.iter().copied()),
        max_gzip_complexity_ratio: max(gzip_values.iter().copied()),
        mean_complexity_score: mean(complexity_values.iter().copied()),
        min_complexity_score: min(complexity_values.iter().copied()),
        max_complexity_score: max(complexity_values.iter().copied()),
        family_counts: family_counts(samples),
        complexity_histogram: complexity_histogram(samples),
    }
}

fn family_counts(samples: &[UniversalitySampleRecord]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for sample in samples {
        *counts.entry(sample.family.clone()).or_insert(0) += 1;
    }
    counts
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ruliad::config::{
        RuliadSerializationConfig, RuliadTokenizationConfig, default_ruliad_families,
    };
    use tempfile::tempdir;

    #[test]
    fn generated_corpus_writes_manifest() {
        let dir = tempdir().expect("tempdir");
        let config = RuliadCorpusConfig {
            output_dir: dir.path().join("out"),
            seed: 3,
            name: "ruliad-test".to_string(),
            train_samples: 4,
            validation_samples: 2,
            chunk_token_capacity: 512,
            serialization: RuliadSerializationConfig {
                document_tokens: 96,
                preview_samples: 1,
            },
            tokenization: RuliadTokenizationConfig::default(),
            source_selection: crate::ruliad::config::RuliadSourceSelectionConfig::default(),
            families: default_ruliad_families(),
            proof_tasks: None,
            lean_task_limit: None,
        };
        let report = generate_ruliad_corpus(&config).expect("generate");
        assert!(report.manifest_path.is_file());
        assert!(report.sample_records_path.is_file());
        let manifest = crate::manifest::load_manifest(&report.manifest_path).expect("manifest");
        assert_eq!(manifest.corpus_kind, CorpusKind::Ruliad);
        assert_eq!(manifest.stats.total_samples, 6);
    }
}
