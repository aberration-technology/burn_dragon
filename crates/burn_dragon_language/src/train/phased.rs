use crate::checkpoint::{RUN_DIR_ENV, RUN_NAME_ENV};
use crate::config::{TrainingConfig, load_training_config};
use anyhow::{Context, Result, anyhow};
use burn_dragon_train::OptimizerKind;
use burn_dragon_train::train::pipeline::TrainingLaunchMode;
use serde::{Deserialize, Serialize};
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const ADAMW_PHASE_KEY: &str = "adamw";
const EGGROLL_PHASE_KEY: &str = "eggroll";

fn default_run_prefix() -> String {
    "adamw-eggroll".to_owned()
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default)]
#[derive(Default)]
pub struct OptimizerPipelinePhaseConfig {
    pub config_paths: Vec<PathBuf>,
    pub run_name: Option<String>,
    pub max_iters: Option<usize>,
    pub checkpoint_interval_iters: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default)]
pub struct AdamwEggrollPipelineConfig {
    pub run_root: Option<PathBuf>,
    pub run_prefix: String,
    pub adamw: OptimizerPipelinePhaseConfig,
    pub eggroll: OptimizerPipelinePhaseConfig,
}

impl Default for AdamwEggrollPipelineConfig {
    fn default() -> Self {
        Self {
            run_root: None,
            run_prefix: default_run_prefix(),
            adamw: OptimizerPipelinePhaseConfig::default(),
            eggroll: OptimizerPipelinePhaseConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AdamwEggrollPipelineRunPlan {
    pub run_root: PathBuf,
    pub run_prefix: String,
    pub adamw_run_name: String,
    pub adamw_run_dir: PathBuf,
    pub eggroll_run_name: String,
    pub eggroll_run_dir: PathBuf,
}

impl AdamwEggrollPipelineRunPlan {
    pub fn adamw_checkpoint_dir(&self) -> PathBuf {
        self.adamw_run_dir.join("checkpoint")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AdamwEggrollPipelineReport {
    pub plan: AdamwEggrollPipelineRunPlan,
    pub adamw_checkpoint_dir: PathBuf,
    pub adamw_checkpoint_epoch: usize,
    pub source_selection_state_path: Option<PathBuf>,
}

pub fn load_adamw_eggroll_pipeline_config(path: &Path) -> Result<AdamwEggrollPipelineConfig> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("read pipeline config {}", path.display()))?;
    let mut config: AdamwEggrollPipelineConfig = toml::from_str(&contents)
        .with_context(|| format!("parse pipeline config {}", path.display()))?;
    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
    resolve_pipeline_config_paths(&mut config, base_dir);
    validate_pipeline_config(&config)?;
    Ok(config)
}

fn resolve_pipeline_config_paths(config: &mut AdamwEggrollPipelineConfig, base_dir: &Path) {
    if let Some(run_root) = &mut config.run_root {
        *run_root = resolve_relative_path(base_dir, run_root);
    }
    resolve_phase_config_paths(&mut config.adamw, base_dir);
    resolve_phase_config_paths(&mut config.eggroll, base_dir);
}

fn resolve_phase_config_paths(phase: &mut OptimizerPipelinePhaseConfig, base_dir: &Path) {
    for path in &mut phase.config_paths {
        *path = resolve_relative_path(base_dir, path);
    }
}

fn resolve_relative_path(base_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base_dir.join(path)
    }
}

fn validate_pipeline_config(config: &AdamwEggrollPipelineConfig) -> Result<()> {
    if config.adamw.config_paths.is_empty() {
        return Err(anyhow!("pipeline adamw.config_paths must not be empty"));
    }
    if config.eggroll.config_paths.is_empty() {
        return Err(anyhow!("pipeline eggroll.config_paths must not be empty"));
    }
    if config.run_prefix.trim().is_empty() {
        return Err(anyhow!("pipeline run_prefix must not be empty"));
    }
    Ok(())
}

pub fn load_phase_training_config(phase: &OptimizerPipelinePhaseConfig) -> Result<TrainingConfig> {
    let mut config = load_training_config(&phase.config_paths)?;
    apply_phase_overrides(&mut config, phase)?;
    Ok(config)
}

pub fn apply_phase_overrides(
    config: &mut TrainingConfig,
    phase: &OptimizerPipelinePhaseConfig,
) -> Result<()> {
    if let Some(max_iters) = phase.max_iters {
        config.training.max_iters = max_iters;
    }
    if let Some(checkpoint_interval_iters) = phase.checkpoint_interval_iters {
        config.training.checkpoint_interval_iters = checkpoint_interval_iters;
    }
    config.validate()?;
    Ok(())
}

pub fn validate_adamw_warmup_config(config: &TrainingConfig) -> Result<()> {
    if !matches!(config.optimizer.name, OptimizerKind::Adamw) {
        return Err(anyhow!(
            "adamw warmup phase requires optimizer.name = \"adamw\""
        ));
    }
    Ok(())
}

pub fn prepare_eggroll_continuation_config(
    mut config: TrainingConfig,
    checkpoint_dir: &Path,
    checkpoint_epoch: Option<usize>,
    source_selection_state_path: Option<&Path>,
) -> Result<TrainingConfig> {
    if !matches!(config.optimizer.name, OptimizerKind::Eggroll) {
        return Err(anyhow!(
            "eggroll continuation phase requires optimizer.name = \"eggroll\""
        ));
    }
    config.training.launch_mode = TrainingLaunchMode::InitFromCheckpoint;
    config.training.resume_run_dir = None;
    config.training.resume_checkpoint_epoch = None;
    config.training.init_checkpoint_path = Some(checkpoint_dir.to_path_buf());
    config.training.init_checkpoint_epoch = checkpoint_epoch;
    config.training.source_selection_state_path =
        source_selection_state_path.map(Path::to_path_buf);
    config.validate()?;
    Ok(config)
}

pub fn plan_adamw_eggroll_runs(config: &AdamwEggrollPipelineConfig) -> AdamwEggrollPipelineRunPlan {
    let run_root = config
        .run_root
        .clone()
        .unwrap_or_else(crate::checkpoint::resolve_run_root);
    let suffix = unix_timestamp_suffix();
    let run_prefix = format!("{}-{}", sanitize_run_name(&config.run_prefix), suffix);
    let adamw_run_name = phase_run_name(&config.adamw, &run_prefix, ADAMW_PHASE_KEY);
    let eggroll_run_name = phase_run_name(&config.eggroll, &run_prefix, EGGROLL_PHASE_KEY);
    AdamwEggrollPipelineRunPlan {
        adamw_run_dir: run_root.join(&adamw_run_name),
        eggroll_run_dir: run_root.join(&eggroll_run_name),
        run_root,
        run_prefix,
        adamw_run_name,
        eggroll_run_name,
    }
}

fn phase_run_name(
    phase: &OptimizerPipelinePhaseConfig,
    run_prefix: &str,
    phase_key: &str,
) -> String {
    phase
        .run_name
        .as_deref()
        .map(sanitize_run_name)
        .unwrap_or_else(|| format!("{run_prefix}-{phase_key}"))
}

fn sanitize_run_name(value: &str) -> String {
    let mut sanitized = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
            sanitized.push(ch);
        } else {
            sanitized.push('-');
        }
    }
    let sanitized = sanitized.trim_matches('-');
    if sanitized.is_empty() {
        "run".to_owned()
    } else {
        sanitized.to_owned()
    }
}

fn unix_timestamp_suffix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

pub fn latest_model_checkpoint_epoch(checkpoint_dir: &Path) -> Result<usize> {
    let entries = fs::read_dir(checkpoint_dir)
        .with_context(|| format!("read checkpoint directory {}", checkpoint_dir.display()))?;
    let mut latest = None;
    for entry in entries {
        let entry =
            entry.with_context(|| format!("read entry from {}", checkpoint_dir.display()))?;
        let Some(name) = entry.file_name().to_str().map(ToOwned::to_owned) else {
            continue;
        };
        let Some(epoch) = parse_model_checkpoint_epoch(&name) else {
            continue;
        };
        latest = Some(latest.map_or(epoch, |current: usize| current.max(epoch)));
    }
    latest.ok_or_else(|| {
        anyhow!(
            "no model-<epoch>.bin checkpoint files found in {}",
            checkpoint_dir.display()
        )
    })
}

fn parse_model_checkpoint_epoch(file_name: &str) -> Option<usize> {
    file_name
        .strip_prefix("model-")?
        .strip_suffix(".bin")?
        .parse::<usize>()
        .ok()
}

pub fn write_pipeline_report(path: &Path, report: &AdamwEggrollPipelineReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create report directory {}", parent.display()))?;
    }
    let payload =
        serde_json::to_string_pretty(report).context("serialize adamw->eggroll report")?;
    fs::write(path, payload).with_context(|| format!("write pipeline report {}", path.display()))
}

pub fn pipeline_source_selection_state_path(plan: &AdamwEggrollPipelineRunPlan) -> PathBuf {
    plan.run_root
        .join(format!("{}-source-selection-state.json", plan.run_prefix))
}

pub struct ScopedRunEnv {
    previous_run_dir: Option<OsString>,
    previous_run_name: Option<OsString>,
}

impl ScopedRunEnv {
    pub fn set(run_dir: &Path, run_name: &str) -> Result<Self> {
        fs::create_dir_all(run_dir)
            .with_context(|| format!("create preassigned run directory {}", run_dir.display()))?;
        let previous_run_dir = std::env::var_os(RUN_DIR_ENV);
        let previous_run_name = std::env::var_os(RUN_NAME_ENV);
        // SAFETY: The staged training CLI mutates these process environment keys before entering
        // a single-threaded training phase. It does not concurrently read or mutate them from
        // other threads while the scoped guard is being installed or restored.
        unsafe {
            std::env::set_var(RUN_DIR_ENV, run_dir);
            std::env::set_var(RUN_NAME_ENV, run_name);
        }
        Ok(Self {
            previous_run_dir,
            previous_run_name,
        })
    }
}

impl Drop for ScopedRunEnv {
    fn drop(&mut self) {
        // SAFETY: See `ScopedRunEnv::set`; restoration is performed by the same single-threaded
        // launcher path after a phase exits.
        unsafe {
            match &self.previous_run_dir {
                Some(value) => std::env::set_var(RUN_DIR_ENV, value),
                None => std::env::remove_var(RUN_DIR_ENV),
            }
            match &self.previous_run_name {
                Some(value) => std::env::set_var(RUN_NAME_ENV, value),
                None => std::env::remove_var(RUN_NAME_ENV),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_eggroll_ruliad_config() -> TrainingConfig {
        toml::from_str(
            r#"
[dataset]
cache_dir = "target/test-cache"
type = "universality_ruliad"
config = "ruliad.toml"

[training]
block_size = 8
batch_size = 2
max_iters = 1
log_frequency = 1

[optimizer]
name = "eggroll"
learning_rate = 0.001
weight_decay = 0.0

[optimizer.eggroll.population]
population_size = 2
population_chunk_size = 2

[generation]
prompt = ""
"#,
        )
        .expect("eggroll ruliad training config should parse")
    }

    #[test]
    fn latest_checkpoint_ignores_non_model_files() -> Result<()> {
        let dir = tempfile::tempdir()?;
        fs::write(dir.path().join("model-1.bin"), [])?;
        fs::write(dir.path().join("model-12.bin"), [])?;
        fs::write(dir.path().join("model-latest.bin"), [])?;
        fs::write(dir.path().join("optimizer-99.bin"), [])?;

        assert_eq!(latest_model_checkpoint_epoch(dir.path())?, 12);
        Ok(())
    }

    #[test]
    fn pipeline_config_paths_are_relative_to_pipeline_file() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let config_path = dir.path().join("pipeline.toml");
        fs::write(
            &config_path,
            r#"
run_root = "runs"
run_prefix = "continual"

[adamw]
config_paths = ["warmup.toml"]
max_iters = 8

[eggroll]
config_paths = ["continuation.toml"]
checkpoint_interval_iters = 16
"#,
        )?;

        let config = load_adamw_eggroll_pipeline_config(&config_path)?;
        assert_eq!(config.run_root, Some(dir.path().join("runs")));
        assert_eq!(
            config.adamw.config_paths,
            vec![dir.path().join("warmup.toml")]
        );
        assert_eq!(
            config.eggroll.config_paths,
            vec![dir.path().join("continuation.toml")]
        );
        assert_eq!(config.adamw.max_iters, Some(8));
        assert_eq!(config.eggroll.checkpoint_interval_iters, Some(16));
        Ok(())
    }

    #[test]
    fn eggroll_continuation_config_carries_source_selection_state_path() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let checkpoint_dir = dir.path().join("checkpoint");
        let source_state_path = dir.path().join("source-selection-state.json");
        let prepared = prepare_eggroll_continuation_config(
            parse_eggroll_ruliad_config(),
            &checkpoint_dir,
            Some(12),
            Some(&source_state_path),
        )?;

        assert_eq!(
            prepared.training.launch_mode,
            TrainingLaunchMode::InitFromCheckpoint
        );
        assert_eq!(prepared.training.init_checkpoint_path, Some(checkpoint_dir));
        assert_eq!(prepared.training.init_checkpoint_epoch, Some(12));
        assert_eq!(
            prepared.training.source_selection_state_path,
            Some(source_state_path)
        );
        Ok(())
    }

    #[test]
    fn run_plan_sanitizes_phase_names() {
        let config = AdamwEggrollPipelineConfig {
            run_root: Some(PathBuf::from("runs")),
            run_prefix: "best practical".to_owned(),
            adamw: OptimizerPipelinePhaseConfig {
                run_name: Some("warmup / adamw".to_owned()),
                ..OptimizerPipelinePhaseConfig::default()
            },
            eggroll: OptimizerPipelinePhaseConfig {
                run_name: Some("continue:eggroll".to_owned()),
                ..OptimizerPipelinePhaseConfig::default()
            },
        };

        let plan = plan_adamw_eggroll_runs(&config);
        assert_eq!(plan.adamw_run_name, "warmup---adamw");
        assert_eq!(plan.eggroll_run_name, "continue-eggroll");
    }
}
