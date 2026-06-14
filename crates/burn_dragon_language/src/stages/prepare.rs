use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use burn_dragon_train::train::pipeline::resolve_resume_run_dir;
use burn_dragon_universality::{
    GeneratedCorpusReport, GeneratedRuliadCorpusReport, NcaCorpusConfig, RuliadCorpusConfig,
    generate_nca_corpus, generate_ruliad_corpus, load_nca_config, load_ruliad_config,
};

use crate::config::{DatasetSourceConfig, TrainingConfig, load_training_config};
use crate::tokenizer::TokenizerKind;

use super::{ExperimentStageArtifact, ExperimentStageConfig, ExperimentStageKind};

#[derive(Debug, Clone, PartialEq)]
pub enum PreparedUniversalityCorpusConfig {
    Nca(NcaCorpusConfig),
    Ruliad(RuliadCorpusConfig),
}

#[derive(Debug, Clone)]
pub enum GeneratedUniversalityCorpusReport {
    Nca(GeneratedCorpusReport),
    Ruliad(GeneratedRuliadCorpusReport),
}

impl PreparedUniversalityCorpusConfig {
    pub fn output_dir(&self) -> &Path {
        match self {
            Self::Nca(config) => &config.output_dir,
            Self::Ruliad(config) => &config.output_dir,
        }
    }
}

impl GeneratedUniversalityCorpusReport {
    pub fn manifest_path(&self) -> &Path {
        match self {
            Self::Nca(report) => &report.manifest_path,
            Self::Ruliad(report) => &report.manifest_path,
        }
    }

    pub fn sample_records_path(&self) -> &Path {
        match self {
            Self::Nca(report) => &report.sample_records_path,
            Self::Ruliad(report) => &report.sample_records_path,
        }
    }

    pub fn output_dir(&self) -> &Path {
        match self {
            Self::Nca(report) => report
                .manifest_path
                .parent()
                .unwrap_or_else(|| Path::new(".")),
            Self::Ruliad(report) => &report.output_dir,
        }
    }
}

pub fn prepare_universality_stage_config(
    bundle_config_path: &Path,
    stage_dir: &Path,
    source_config_path: &Path,
) -> Result<PreparedUniversalityCorpusConfig> {
    let source_path = resolve_relative_to(bundle_config_path, source_config_path);
    let output_dir = stage_dir.join("output");
    match load_universality_corpus_config(&source_path)? {
        PreparedUniversalityCorpusConfig::Nca(mut config) => {
            config.output_dir = output_dir;
            Ok(PreparedUniversalityCorpusConfig::Nca(config))
        }
        PreparedUniversalityCorpusConfig::Ruliad(mut config) => {
            config.output_dir = output_dir;
            Ok(PreparedUniversalityCorpusConfig::Ruliad(config))
        }
    }
}

pub fn generate_prepared_universality_stage_corpus(
    config: &PreparedUniversalityCorpusConfig,
) -> Result<GeneratedUniversalityCorpusReport> {
    match config {
        PreparedUniversalityCorpusConfig::Nca(config) => {
            generate_nca_corpus(config).map(GeneratedUniversalityCorpusReport::Nca)
        }
        PreparedUniversalityCorpusConfig::Ruliad(config) => {
            generate_ruliad_corpus(config).map(GeneratedUniversalityCorpusReport::Ruliad)
        }
    }
}

pub fn prepare_language_stage_config(
    bundle_config_path: &Path,
    source_config_path: &Path,
    stage_dir: &Path,
    stage: &ExperimentStageConfig,
    dependency_artifacts: &BTreeMap<String, ExperimentStageArtifact>,
) -> Result<TrainingConfig> {
    let source_path = resolve_relative_to(bundle_config_path, source_config_path);
    let mut config = load_training_config(&[source_path])?;
    config.run_layout.base_dir = Some(stage_dir.join("runs"));
    config.run_layout.category = None;
    config.run_layout.mirror_config_path = false;
    let stage_launch_mode = match &stage.kind {
        ExperimentStageKind::LanguageTrain { launch_mode, .. } => *launch_mode,
        ExperimentStageKind::UniversalityGenerate { .. } => unreachable!(),
    };
    config.training.launch_mode = stage_launch_mode;

    if let ExperimentStageKind::LanguageTrain {
        dataset_manifest_from_stage,
        init_checkpoint_from_stage,
        ..
    } = &stage.kind
    {
        if let Some(stage_name) = dataset_manifest_from_stage {
            let artifact = dependency_artifacts.get(stage_name).ok_or_else(|| {
                anyhow!("stage `{}` has no completed artifact available", stage_name)
            })?;
            let manifest_path = artifact
                .manifest_path
                .clone()
                .ok_or_else(|| anyhow!("stage `{stage_name}` did not produce a manifest_path"))?;
            config.dataset.cache_dir = artifact.corpus_output_dir.clone().unwrap_or_else(|| {
                manifest_path
                    .parent()
                    .unwrap_or(Path::new("."))
                    .to_path_buf()
            });
            config.dataset.source = DatasetSourceConfig::UniversalityManifest {
                manifest: manifest_path,
            };
            if !matches!(
                config.dataset.tokenizer.kind,
                TokenizerKind::Pretokenized(_)
            ) {
                return Err(anyhow!(
                    "language stage `{}` requires tokenizer.type = `pretokenized` when sourcing a universality manifest",
                    stage.name
                ));
            }
        }
        if let Some(stage_name) = init_checkpoint_from_stage {
            let artifact = dependency_artifacts.get(stage_name).ok_or_else(|| {
                anyhow!("stage `{}` has no completed artifact available", stage_name)
            })?;
            let checkpoint_dir = artifact.latest_checkpoint_dir.clone().ok_or_else(|| {
                anyhow!("stage `{stage_name}` did not produce a latest_checkpoint_dir")
            })?;
            config.training.init_checkpoint_path = Some(checkpoint_dir);
            config.training.init_checkpoint_epoch = artifact.latest_checkpoint_epoch;
            config.training.resume_run_dir = None;
            config.training.resume_checkpoint_epoch = None;
        }
    }

    let stage_run_root = config
        .run_layout
        .base_dir
        .clone()
        .unwrap_or_else(|| stage_dir.join("runs"));
    config.training.resume_run_dir = resolve_resume_run_dir(
        &stage_run_root,
        config.training.resume_run_dir.as_deref(),
        stage_launch_mode,
    )?;
    if config.training.resume_run_dir.is_some() {
        config.training.resume_checkpoint_epoch = None;
        config.training.init_checkpoint_path = None;
        config.training.init_checkpoint_epoch = None;
    }

    Ok(config)
}

fn resolve_relative_to(bundle_config_path: &Path, relative_or_absolute: &Path) -> PathBuf {
    if relative_or_absolute.is_absolute() {
        relative_or_absolute.to_path_buf()
    } else {
        bundle_config_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(relative_or_absolute)
    }
}

fn load_universality_corpus_config(path: &Path) -> Result<PreparedUniversalityCorpusConfig> {
    if looks_like_ruliad_config(path)? {
        return load_ruliad_config(path).map(PreparedUniversalityCorpusConfig::Ruliad);
    }
    match load_nca_config(path) {
        Ok(config) => Ok(PreparedUniversalityCorpusConfig::Nca(config)),
        Err(nca_error) => load_ruliad_config(path)
            .map(PreparedUniversalityCorpusConfig::Ruliad)
            .with_context(|| {
                format!(
                    "failed to parse {} as NCA or ruliad config; NCA parse error: {nca_error:#}",
                    path.display()
                )
            }),
    }
}

fn looks_like_ruliad_config(path: &Path) -> Result<bool> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("failed to read universality config {}", path.display()))?;
    let value: toml::Value =
        toml::from_str(&contents).with_context(|| format!("failed to parse {}", path.display()))?;
    if value.get("proof_tasks").is_some() || value.get("lean_task_limit").is_some() {
        return Ok(true);
    }
    if value
        .get("serialization")
        .and_then(toml::Value::as_table)
        .is_some_and(|table| table.contains_key("document_tokens"))
    {
        return Ok(true);
    }
    let ruliad_family = value
        .get("families")
        .and_then(toml::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(toml::Value::as_table)
        .filter_map(|family| family.get("kind"))
        .filter_map(toml::Value::as_str)
        .any(|kind| {
            matches!(
                kind,
                "eca"
                    | "simulation"
                    | "automaton"
                    | "rewrite"
                    | "algebra"
                    | "category"
                    | "lean_task"
                    | "hash_noise"
            )
        });
    Ok(ruliad_family)
}
