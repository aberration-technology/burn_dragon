use anyhow::{Context, Result};

use crate::config::{
    DatasetConfig, DatasetSourceConfig, HuggingFaceDatasetConfig, HuggingFaceRecordFormat,
    TrainingHyperparameters,
};

use super::{Dataset, HuggingFaceDataset, UniversalityDataset};
use crate::dataset::universality::{RuliadSourceSelectionOverrides, RuliadSourceSelectionRestore};

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
        DatasetSourceConfig::UniversalityRuliad { config } => Dataset::from_universality(
            UniversalityDataset::new_ruliad_on_the_fly_with_overrides(
                config,
                training.block_size,
                training.batch_size,
                &cfg.tokenizer,
                RuliadSourceSelectionOverrides {
                    cold_start_enabled: cfg.ruliad_source_selection_cold_start_enabled,
                    documents_per_step: cfg.ruliad_source_selection_documents_per_step,
                },
            )
            .and_then(|dataset| {
                dataset.with_source_selection_state_path(
                    training.source_selection_state_path.as_deref(),
                    if matches!(
                        training.launch_mode,
                        burn_dragon_train::train::pipeline::TrainingLaunchMode::ResumeExactRun
                            | burn_dragon_train::train::pipeline::TrainingLaunchMode::ResumeLatestCheckpointIfPresent
                    ) || training.resume_checkpoint_epoch.is_some()
                    {
                        RuliadSourceSelectionRestore::ResumeRun
                    } else {
                        RuliadSourceSelectionRestore::StartNewRun
                    },
                )
            })
            .map(|dataset| {
                dataset.with_source_selection_feedback_updates_enabled(
                    cfg.ruliad_source_selection_feedback_updates_enabled,
                )
            })
            .map(|dataset| dataset.with_ruliad_supervision(training.ruliad_supervision))
            .and_then(|dataset| dataset.with_ruliad_supervision_audit(4))
            .with_context(|| {
                format!(
                    "failed to prepare on-the-fly universality ruliad dataset {}",
                    config.display()
                )
            })?,
        ),
    };

    if let Dataset::Universality(dataset) = &dataset
        && let Some(audit) = dataset.ruliad_supervision_audit()
        && training.ruliad_supervision.uses_answer_target_mask()
        && audit.total_query_conditioning_samples > 0
    {
        let carries_ordered_context = training.tbptt_persist_across_steps
            && training
                .sequence_batching
                .uses_streaming_loader(training.tbptt_persist_across_steps);
        if !carries_ordered_context
            && !training.has_required_self_contained_primary_schedule()
            && audit.query_visible_within_block_fraction < 1.0
        {
            let max_query_to_answer_tokens = audit
                .buckets
                .iter()
                .map(|bucket| bucket.max_query_to_answer_tokens)
                .max()
                .unwrap_or_default();
            return Err(anyhow::anyhow!(
                "answer-masked Ruliad supervision is not causally visible in stateless training: block_size={} query_visible_within_block_fraction={:.4} measured_query_samples={} max_query_to_answer_tokens={}; increase training.block_size, retain ordered stream state, or require self-contained primary objectives covering every step",
                training.block_size,
                audit.query_visible_within_block_fraction,
                audit.total_query_conditioning_samples,
                max_query_to_answer_tokens,
            ));
        }
    }

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
            "Prepared {} {} from {} with batch_size={}, block_size={}, split_ratio={}{}{}{}{}{}",
            ds.source_kind_label(),
            ds.dataset_name(),
            ds.source_path().display(),
            ds.batch_size(),
            ds.block_size(),
            ds.train_split_ratio(),
            ds.source_selection_feedback_updates_enabled()
                .map(|enabled| format!(", source_selection_feedback_updates={enabled}"))
                .unwrap_or_default(),
            ds.source_selection_cold_start_enabled()
                .map(|enabled| format!(", source_selection_cold_start={enabled}"))
                .unwrap_or_default(),
            ds.source_selection_documents_per_step()
                .map(|documents| format!(", source_selection_documents_per_step={documents}"))
                .unwrap_or_default(),
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
            )).unwrap_or_default(),
            ds.ruliad_supervision_audit().map(|audit| {
                let max_query_to_answer_tokens = audit
                    .buckets
                    .iter()
                    .map(|bucket| bucket.max_query_to_answer_tokens)
                    .max()
                    .unwrap_or_default();
                format!(
                    ", supervision_audit_samples={}, query_visible_within_block={:.4}, max_query_to_answer_tokens={}, stream_chunks={}",
                    audit.sample_count,
                    audit.query_visible_within_block_fraction,
                    max_query_to_answer_tokens,
                    audit.total_stream_chunks,
                )
            }).unwrap_or_default()
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
