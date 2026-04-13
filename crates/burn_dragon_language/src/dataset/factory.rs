use anyhow::{Context, Result};

use crate::config::{
    DatasetConfig, DatasetSourceConfig, HuggingFaceDatasetConfig, HuggingFaceRecordFormat,
    TrainingHyperparameters,
};

use super::{Dataset, HuggingFaceDataset, UniversalityDataset};

pub fn build_dataset(
    cfg: &DatasetConfig,
    training: &TrainingHyperparameters,
) -> Result<(Dataset, String)> {
    let dataset = match &cfg.source {
        DatasetSourceConfig::NemotronClimbMix {
            revision,
            max_records,
        } => {
            let config = nemotron_climbmix_config(revision, *max_records);
            Dataset::from_huggingface(
                HuggingFaceDataset::new(
                    &cfg.cache_dir,
                    training.block_size,
                    training.batch_size,
                    cfg.train_split_ratio,
                    &cfg.tokenizer,
                    &config,
                )
                .with_context(|| "failed to prepare Nemotron-ClimbMix dataset")?,
            )
        }
        DatasetSourceConfig::UniversalityManifest { manifest } => Dataset::from_universality(
            UniversalityDataset::new(
                manifest,
                training.block_size,
                training.batch_size,
                cfg.train_split_ratio,
                &cfg.tokenizer,
            )
            .with_context(|| {
                format!(
                    "failed to prepare universality manifest {}",
                    manifest.display()
                )
            })?,
        ),
        DatasetSourceConfig::UniversalityNca { config } => Dataset::from_universality(
            UniversalityDataset::new_on_the_fly(
                config,
                training.block_size,
                training.batch_size,
                training
                    .min_logical_block_size
                    .map(|value| value.max(training.block_size)),
                &cfg.tokenizer,
            )
            .with_context(|| {
                format!(
                    "failed to prepare on-the-fly universality NCA dataset {}",
                    config.display()
                )
            })?,
        ),
    };

    let description = match &dataset {
        Dataset::HuggingFace(ds) => format!(
            "Prepared Hugging Face dataset {} (rev: {}) with batch_size={}, block_size={}, split_ratio={}",
            ds.repo_id(),
            ds.revision().unwrap_or("main"),
            ds.batch_size(),
            ds.block_size(),
            ds.train_split_ratio()
        ),
        Dataset::Universality(ds) => format!(
            "Prepared {} {} from {} with batch_size={}, block_size={}, split_ratio={}{}",
            ds.source_kind_label(),
            ds.dataset_name(),
            ds.source_path().display(),
            ds.batch_size(),
            ds.block_size(),
            ds.train_split_ratio(),
            ds.train_probe_summary().map(|summary| format!(
                ", train_docs={}, val_docs={}, doc_tokens={}, probe_mean_gzip={:.4}, probe_complexity={:.2}, runtime_doc_cache_limit={}",
                summary.sample_count,
                ds.validation_probe_summary()
                    .map(|probe| probe.sample_count)
                    .unwrap_or_default(),
                summary.document_token_count,
                summary.mean_gzip_complexity_ratio,
                summary.mean_complexity_score,
                ds.runtime_document_cache_limit().unwrap_or_default()
            )).unwrap_or_default()
        ),
    };

    Ok((dataset, description))
}

fn nemotron_climbmix_config(
    revision: &Option<String>,
    max_records: Option<usize>,
) -> HuggingFaceDatasetConfig {
    HuggingFaceDatasetConfig {
        repo_id: "nvidia/Nemotron-CC/data/climb_mix".to_string(),
        token: None,
        revision: revision.clone(),
        format: HuggingFaceRecordFormat::Parquet,
        train_files: Vec::new(),
        auto_discover_train_files: true,
        validation_files: Vec::new(),
        text_fields: Vec::new(),
        sequence_field: Some("input_ids".to_string()),
        field_separator: " ".to_string(),
        template: None,
        max_records,
    }
}
