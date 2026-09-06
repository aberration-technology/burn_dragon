use std::fs;
use std::path::{Path, PathBuf};
#[cfg(not(target_arch = "wasm32"))]
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use burn_dragon_core::DragonConfig;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::TrainingConfig;

pub const EXPERIMENT_MANIFEST_FILE_NAME: &str = "experiment_manifest.json";
const EXPERIMENT_MANIFEST_SCHEMA_VERSION: u32 = 2;
const CHECKPOINT_PROGRESS_PREFIX: &str = "training-progress";

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ExperimentGitRevision {
    pub sha: Option<String>,
    pub branch: Option<String>,
    pub dirty: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ExperimentHost {
    pub os: String,
    pub arch: String,
    pub logical_cpus: Option<usize>,
    pub total_memory_bytes: Option<u64>,
    pub hardware_label: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ExperimentLaunch {
    pub unix_time_ms: u128,
    pub command: Vec<String>,
    pub effective_config_sha256: String,
    pub training_contract_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub immutable_training_contract_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_latent_objective_contract_version: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ruliad_supervision_audit_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ruliad_supervision_audit: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_model_tensor_fingerprint_schema: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_model_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub planned_max_iters: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub horizon_extension: Option<ExperimentHorizonExtension>,
    pub launch_mode: burn_dragon_train::train::pipeline::TrainingLaunchMode,
    pub resume_checkpoint_epoch: Option<usize>,
    pub checkpoint_artifacts: Vec<ExperimentCheckpointArtifact>,
    pub config_snapshot: PathBuf,
    pub git: ExperimentGitRevision,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ExperimentHorizonExtension {
    pub previous_max_iters: usize,
    pub requested_max_iters: usize,
    pub resume_completed_steps: usize,
    pub checkpoint_epoch: usize,
    pub schedule_contract: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ExperimentCheckpointProgress {
    pub epoch: usize,
    pub completed_steps: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ExperimentCheckpointArtifact {
    pub file_name: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ExperimentManifest {
    pub schema_version: u32,
    pub package_version: String,
    pub run_name: String,
    pub backend: String,
    pub host: ExperimentHost,
    pub model_spec: burn_dragon_train::ModelSpec,
    pub training_snapshot: PathBuf,
    pub run_config: PathBuf,
    pub launches: Vec<ExperimentLaunch>,
}

#[derive(Serialize)]
struct ExperimentLaunchConfigSnapshot<'a> {
    training: &'a TrainingConfig,
    model: &'a DragonConfig,
}

#[derive(Deserialize)]
struct OwnedExperimentLaunchConfigSnapshot {
    training: TrainingConfig,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum SourceSelectionCheckpointProgress {
    Clocked {
        clock: crate::dataset::RuliadSourceSelectionClock,
    },
    Legacy {
        absolute_step_offset: usize,
    },
}

pub fn write_experiment_manifest(
    config: &TrainingConfig,
    model_config: &DragonConfig,
    run_dir: &Path,
    run_name: &str,
    backend: &str,
) -> Result<()> {
    write_experiment_manifest_with_supervision_audit(
        config,
        model_config,
        run_dir,
        run_name,
        backend,
        None,
    )
}

pub fn write_experiment_manifest_with_supervision_audit(
    config: &TrainingConfig,
    model_config: &DragonConfig,
    run_dir: &Path,
    run_name: &str,
    backend: &str,
    ruliad_supervision_audit_sha256: Option<&str>,
) -> Result<()> {
    write_experiment_manifest_with_identities(
        config,
        model_config,
        run_dir,
        run_name,
        backend,
        ruliad_supervision_audit_sha256,
        None,
    )
}

pub fn write_experiment_manifest_with_identities(
    config: &TrainingConfig,
    model_config: &DragonConfig,
    run_dir: &Path,
    run_name: &str,
    backend: &str,
    ruliad_supervision_audit_sha256: Option<&str>,
    initial_model_sha256: Option<&str>,
) -> Result<()> {
    fs::create_dir_all(run_dir)
        .with_context(|| format!("create experiment run directory {}", run_dir.display()))?;
    let path = run_dir.join(EXPERIMENT_MANIFEST_FILE_NAME);
    let mut launch = ExperimentLaunch {
        unix_time_ms: unix_time_ms(),
        command: std::env::args().collect(),
        effective_config_sha256: effective_config_sha256(config)?,
        training_contract_sha256: training_contract_sha256(config)?,
        immutable_training_contract_sha256: Some(immutable_training_contract_sha256(config)?),
        next_latent_objective_contract_version: (config.training.latent_reasoning.enabled
            && config.training.latent_reasoning.next_latent.enabled)
            .then_some(crate::config::train::NEXT_LATENT_OBJECTIVE_CONTRACT_VERSION),
        ruliad_supervision_audit_sha256: ruliad_supervision_audit_sha256.map(str::to_string),
        ruliad_supervision_audit: ruliad_supervision_audit_sha256
            .map(|_| PathBuf::from(super::utils::RULIAD_SUPERVISION_AUDIT_FILE_NAME)),
        initial_model_tensor_fingerprint_schema: initial_model_sha256
            .map(|_| super::model_identity::MODEL_TENSOR_FINGERPRINT_SCHEMA.to_string()),
        initial_model_sha256: initial_model_sha256.map(str::to_string),
        planned_max_iters: Some(config.training.max_iters),
        horizon_extension: None,
        launch_mode: config.training.launch_mode,
        resume_checkpoint_epoch: resolved_resume_checkpoint_epoch(config, run_dir),
        checkpoint_artifacts: resolved_resume_checkpoint_epoch(config, run_dir)
            .map(|epoch| checkpoint_artifacts(run_dir, epoch))
            .unwrap_or_default(),
        config_snapshot: PathBuf::new(),
        git: git_revision(),
    };
    let mut manifest = if path.is_file() {
        let bytes = fs::read(&path)
            .with_context(|| format!("read experiment manifest {}", path.display()))?;
        let existing: ExperimentManifest = serde_json::from_slice(&bytes)
            .with_context(|| format!("parse experiment manifest {}", path.display()))?;
        if existing.schema_version != EXPERIMENT_MANIFEST_SCHEMA_VERSION {
            return Err(anyhow!(
                "experiment manifest {} has schema {}, expected {}",
                path.display(),
                existing.schema_version,
                EXPERIMENT_MANIFEST_SCHEMA_VERSION
            ));
        }
        if existing.run_name != run_name || existing.backend != backend {
            return Err(anyhow!(
                "experiment manifest identity mismatch in {}: existing run/backend={}/{}, requested={}/{}",
                path.display(),
                existing.run_name,
                existing.backend,
                run_name,
                backend
            ));
        }
        let requested_model_spec = super::utils::build_model_spec(model_config);
        if existing.model_spec != requested_model_spec {
            return Err(anyhow!(
                "experiment model contract mismatch in {}",
                path.display()
            ));
        }
        if let Some(previous) = existing.launches.last()
            && previous.next_latent_objective_contract_version
                != launch.next_latent_objective_contract_version
        {
            return Err(anyhow!(
                "NextLat objective contract changed in {}: existing={:?}, requested={:?}; start an explicit weights-only transfer run instead of an exact resume",
                path.display(),
                previous.next_latent_objective_contract_version,
                launch.next_latent_objective_contract_version,
            ));
        }
        if let Some(previous) = existing.launches.last()
            && previous.training_contract_sha256 != launch.training_contract_sha256
        {
            launch.horizon_extension = Some(validate_horizon_extension(
                config, run_dir, &existing, previous, &launch,
            )?);
        }
        if let Some(previous) = existing.launches.last()
            && previous.ruliad_supervision_audit_sha256 != launch.ruliad_supervision_audit_sha256
        {
            return Err(anyhow!(
                "experiment Ruliad supervision audit mismatch in {}: existing={:?}, requested={:?}",
                path.display(),
                previous.ruliad_supervision_audit_sha256,
                launch.ruliad_supervision_audit_sha256,
            ));
        }
        if let Some(requested) = launch.initial_model_sha256.as_deref()
            && let Some(recorded) = existing
                .launches
                .iter()
                .find_map(|previous| previous.initial_model_sha256.as_deref())
            && recorded != requested
        {
            return Err(anyhow!(
                "experiment initial model identity mismatch in {}: existing={}, requested={}",
                path.display(),
                recorded,
                requested,
            ));
        }
        existing
    } else {
        ExperimentManifest {
            schema_version: EXPERIMENT_MANIFEST_SCHEMA_VERSION,
            package_version: env!("CARGO_PKG_VERSION").to_string(),
            run_name: run_name.to_string(),
            backend: backend.to_string(),
            host: experiment_host(),
            model_spec: super::utils::build_model_spec(model_config),
            training_snapshot: PathBuf::from("training_config.json"),
            run_config: PathBuf::from("config.json"),
            launches: Vec::new(),
        }
    };
    launch.config_snapshot = PathBuf::from("launches")
        .join(format!("launch-{:04}-config.json", manifest.launches.len()));
    let launch_snapshot_path = run_dir.join(&launch.config_snapshot);
    if let Some(parent) = launch_snapshot_path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!("create experiment launch snapshot dir {}", parent.display())
        })?;
    }
    write_json_atomically(
        &launch_snapshot_path,
        &ExperimentLaunchConfigSnapshot {
            training: config,
            model: model_config,
        },
    )?;
    manifest.launches.push(launch);
    write_json_atomically(&path, &manifest)
}

fn effective_config_sha256(config: &TrainingConfig) -> Result<String> {
    config_sha256(config, "effective training config")
}

fn training_contract_sha256(config: &TrainingConfig) -> Result<String> {
    let mut contract = config.clone();
    contract.training.provenance = Default::default();
    contract.training.launch_mode = Default::default();
    contract.training.resume_run_dir = None;
    contract.training.resume_checkpoint_epoch = None;
    contract.training.resume_horizon_extension = Default::default();
    contract.training.source_selection_state_path = None;
    config_sha256(&contract, "normalized training contract")
}

fn immutable_training_contract_sha256(config: &TrainingConfig) -> Result<String> {
    let mut contract = config.clone();
    contract.training.max_iters = 1;
    training_contract_sha256(&contract)
}

fn validate_horizon_extension(
    config: &TrainingConfig,
    run_dir: &Path,
    manifest: &ExperimentManifest,
    previous: &ExperimentLaunch,
    requested: &ExperimentLaunch,
) -> Result<ExperimentHorizonExtension> {
    if !config.training.resume_horizon_extension.enabled {
        return Err(anyhow!(
            "experiment training contract mismatch in {}: existing={}, requested={}; increasing training.max_iters requires training.resume_horizon_extension.enabled=true",
            run_dir.join(EXPERIMENT_MANIFEST_FILE_NAME).display(),
            previous.training_contract_sha256,
            requested.training_contract_sha256
        ));
    }
    if !matches!(
        config.training.launch_mode,
        burn_dragon_train::train::pipeline::TrainingLaunchMode::ResumeExactRun
    ) {
        return Err(anyhow!(
            "experiment horizon extension requires launch_mode=resume_exact_run"
        ));
    }
    if config.training.epochs.is_some() {
        return Err(anyhow!(
            "experiment horizon extension only supports max_iters-based runs"
        ));
    }
    ensure_horizon_independent_learning_schedule(config)?;

    let previous_snapshot = load_launch_snapshot(run_dir, previous)?;
    let previous_config = previous_snapshot.training;
    let previous_immutable = immutable_training_contract_sha256(&previous_config)?;
    let requested_immutable = immutable_training_contract_sha256(config)?;
    if previous_immutable != requested_immutable {
        return Err(anyhow!(
            "experiment horizon extension changed immutable training semantics in {}: existing={}, requested={}",
            run_dir.join(EXPERIMENT_MANIFEST_FILE_NAME).display(),
            previous_immutable,
            requested_immutable
        ));
    }
    if let Some(first) = manifest.launches.first() {
        let first_snapshot = load_launch_snapshot(run_dir, first)?;
        let first_immutable = immutable_training_contract_sha256(&first_snapshot.training)?;
        if first_immutable != requested_immutable {
            return Err(anyhow!(
                "experiment horizon extension no longer matches the original immutable training contract"
            ));
        }
    }

    let previous_max_iters = previous_config.training.max_iters;
    let requested_max_iters = config.training.max_iters;
    if requested_max_iters <= previous_max_iters {
        return Err(anyhow!(
            "experiment horizon extension must grow monotonically: previous max_iters={previous_max_iters}, requested={requested_max_iters}"
        ));
    }
    let checkpoint_epoch = requested.resume_checkpoint_epoch.ok_or_else(|| {
        anyhow!("experiment horizon extension requires a resolved resume checkpoint epoch")
    })?;
    let resume_completed_steps =
        checkpoint_completed_steps(run_dir, checkpoint_epoch, &previous_config)?;
    if requested_max_iters <= resume_completed_steps {
        return Err(anyhow!(
            "experiment horizon extension must exceed checkpoint progress: completed_steps={resume_completed_steps}, requested max_iters={requested_max_iters}"
        ));
    }

    Ok(ExperimentHorizonExtension {
        previous_max_iters,
        requested_max_iters,
        resume_completed_steps,
        checkpoint_epoch,
        schedule_contract: "horizon_independent_v1".to_string(),
    })
}

fn ensure_horizon_independent_learning_schedule(config: &TrainingConfig) -> Result<()> {
    if config
        .training
        .module_lr_scales
        .iter()
        .any(|entry| entry.schedule.is_some())
    {
        return Err(anyhow!(
            "experiment horizon extension does not support fraction-of-total module LR schedules"
        ));
    }
    let independent = match &config.optimizer.lr_schedule {
        None
        | Some(burn_dragon_train::LearningRateScheduleConfig::Constant { .. })
        | Some(burn_dragon_train::LearningRateScheduleConfig::Exponential { .. }) => true,
        Some(burn_dragon_train::LearningRateScheduleConfig::Cosine { num_iters, .. })
        | Some(burn_dragon_train::LearningRateScheduleConfig::Linear { num_iters, .. }) => {
            num_iters.is_some()
        }
        Some(burn_dragon_train::LearningRateScheduleConfig::Step { step_size, .. }) => {
            step_size.is_some()
        }
        Some(burn_dragon_train::LearningRateScheduleConfig::Noam { warmup_steps, .. }) => {
            warmup_steps.is_some()
        }
    };
    if !independent {
        return Err(anyhow!(
            "experiment horizon extension requires an LR schedule independent of max_iters"
        ));
    }
    Ok(())
}

fn load_launch_snapshot(
    run_dir: &Path,
    launch: &ExperimentLaunch,
) -> Result<OwnedExperimentLaunchConfigSnapshot> {
    let path = run_dir.join(&launch.config_snapshot);
    let bytes = fs::read(&path)
        .with_context(|| format!("read experiment launch snapshot {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("parse experiment launch snapshot {}", path.display()))
}

fn checkpoint_completed_steps(
    run_dir: &Path,
    epoch: usize,
    previous_config: &TrainingConfig,
) -> Result<usize> {
    let progress_path = experiment_checkpoint_progress_path(run_dir, epoch);
    if progress_path.is_file() {
        let progress: ExperimentCheckpointProgress =
            serde_json::from_slice(&fs::read(&progress_path).with_context(|| {
                format!("read checkpoint progress {}", progress_path.display())
            })?)
            .with_context(|| format!("parse checkpoint progress {}", progress_path.display()))?;
        if progress.epoch != epoch || progress.completed_steps == 0 {
            return Err(anyhow!(
                "invalid checkpoint progress {}: epoch={}, completed_steps={}",
                progress_path.display(),
                progress.epoch,
                progress.completed_steps
            ));
        }
        return Ok(progress.completed_steps);
    }

    let source_path = run_dir
        .join("checkpoint")
        .join(format!("source-selection-state-{epoch}.json"));
    if source_path.is_file() {
        let progress: SourceSelectionCheckpointProgress =
            serde_json::from_slice(&fs::read(&source_path).with_context(|| {
                format!("read legacy source progress {}", source_path.display())
            })?)
            .with_context(|| format!("parse legacy source progress {}", source_path.display()))?;
        return Ok(match progress {
            SourceSelectionCheckpointProgress::Clocked { clock } => clock.completed_run_steps,
            SourceSelectionCheckpointProgress::Legacy {
                absolute_step_offset,
            } => absolute_step_offset.saturating_add(1),
        });
    }

    Ok(epoch
        .saturating_mul(previous_config.training.checkpoint_interval_iters.max(1))
        .min(previous_config.training.max_iters.max(1)))
}

pub(crate) fn experiment_checkpoint_progress_path(run_dir: &Path, epoch: usize) -> PathBuf {
    run_dir
        .join("checkpoint")
        .join(format!("{CHECKPOINT_PROGRESS_PREFIX}-{epoch}.json"))
}

pub(crate) fn save_experiment_checkpoint_progress(
    run_dir: &Path,
    epoch: usize,
    completed_steps: usize,
) -> Result<()> {
    let path = experiment_checkpoint_progress_path(run_dir, epoch);
    write_json_atomically(
        &path,
        &ExperimentCheckpointProgress {
            epoch,
            completed_steps,
        },
    )
}

fn config_sha256(config: &TrainingConfig, label: &str) -> Result<String> {
    let encoded = serde_json::to_vec(config).with_context(|| format!("serialize {label}"))?;
    let digest = Sha256::digest(encoded);
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

fn resolved_resume_checkpoint_epoch(config: &TrainingConfig, run_dir: &Path) -> Option<usize> {
    config.training.resume_run_dir.as_ref()?;
    crate::checkpoint::resolve_checkpoint_base(
        &run_dir.join("checkpoint"),
        config.training.resume_checkpoint_epoch,
    )
    .ok()
    .map(|(_, epoch)| epoch)
}

fn checkpoint_artifacts(run_dir: &Path, epoch: usize) -> Vec<ExperimentCheckpointArtifact> {
    let checkpoint_dir = run_dir.join("checkpoint");
    let suffixes = [
        format!("-{epoch}.bin"),
        format!("-{epoch}.json"),
        format!("-{epoch}.bin.gz"),
    ];
    let mut artifacts = fs::read_dir(checkpoint_dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let file_name = entry.file_name().to_string_lossy().into_owned();
            suffixes
                .iter()
                .any(|suffix| file_name.ends_with(suffix))
                .then(|| ExperimentCheckpointArtifact {
                    file_name,
                    bytes: entry.metadata().map(|metadata| metadata.len()).unwrap_or(0),
                })
        })
        .collect::<Vec<_>>();
    artifacts.sort_by(|left, right| left.file_name.cmp(&right.file_name));
    artifacts
}

fn unix_time_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn experiment_host() -> ExperimentHost {
    ExperimentHost {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        logical_cpus: std::thread::available_parallelism().ok().map(usize::from),
        total_memory_bytes: linux_total_memory_bytes(),
        hardware_label: std::env::var("BURN_DRAGON_HARDWARE_LABEL").ok(),
    }
}

fn linux_total_memory_bytes() -> Option<u64> {
    let contents = fs::read_to_string("/proc/meminfo").ok()?;
    let kib = contents.lines().find_map(|line| {
        let value = line.strip_prefix("MemTotal:")?.trim();
        value.split_whitespace().next()?.parse::<u64>().ok()
    })?;
    kib.checked_mul(1024)
}

#[cfg(not(target_arch = "wasm32"))]
fn git_revision() -> ExperimentGitRevision {
    fn output(args: &[&str]) -> Option<String> {
        let output = Command::new("git").args(args).output().ok()?;
        output
            .status
            .success()
            .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
    }
    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| !output.stdout.is_empty());
    ExperimentGitRevision {
        sha: output(&["rev-parse", "HEAD"]),
        branch: output(&["rev-parse", "--abbrev-ref", "HEAD"]),
        dirty,
    }
}

#[cfg(target_arch = "wasm32")]
fn git_revision() -> ExperimentGitRevision {
    ExperimentGitRevision {
        sha: option_env!("BURN_DRAGON_GIT_SHA").map(str::to_string),
        branch: option_env!("BURN_DRAGON_GIT_BRANCH").map(str::to_string),
        dirty: None,
    }
}

fn write_json_atomically(path: &Path, value: &impl Serialize) -> Result<()> {
    let temporary = path.with_extension("json.tmp");
    fs::write(
        &temporary,
        serde_json::to_vec_pretty(value).context("serialize experiment manifest")?,
    )
    .with_context(|| {
        format!(
            "write temporary experiment manifest {}",
            temporary.display()
        )
    })?;
    fs::rename(&temporary, path).with_context(|| {
        format!(
            "replace experiment manifest {} from {}",
            path.display(),
            temporary.display()
        )
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::load_training_config;

    #[test]
    fn alibi_schedule_change_is_not_an_exact_resume_or_horizon_extension() {
        let directory = tempfile::tempdir().unwrap();
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut config = load_training_config(&[
            workspace.join("config/language/experiments/next_latent/capacity-base.toml")
        ])
        .unwrap();
        let write = |config: &TrainingConfig| {
            let model = crate::build_model_config(&config.model, config.training.block_size);
            write_experiment_manifest(
                config,
                &model,
                directory.path(),
                "alibi-contract",
                "ndarray",
            )
        };
        write(&config).unwrap();
        write(&config).unwrap();
        config.model.alibi_slopes = Some(vec![0.25, 0.0625, 0.015625, 0.00390625]);
        assert!(write(&config).unwrap_err().to_string().contains("contract"));
        config.training.resume_horizon_extension.enabled = true;
        config.training.max_iters *= 2;
        assert!(write(&config).is_err());
    }

    #[test]
    fn next_latent_objective_revision_cannot_silently_resume_an_old_checkpoint() {
        let directory = tempfile::tempdir().unwrap();
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut config = load_training_config(&[
            workspace.join("config/language/experiments/predictive_coding/local-pc-smoke.toml")
        ])
        .unwrap();
        config.training.latent_reasoning.enabled = true;
        config.training.latent_reasoning.next_latent.enabled = true;
        let tokenizer = config.dataset.tokenizer.load(&workspace).unwrap();
        let model_config = crate::build_model_config_with_tokenizer(
            &config.model,
            config.training.block_size,
            tokenizer.as_ref(),
        )
        .unwrap();
        let write = |config: &TrainingConfig| {
            write_experiment_manifest(
                config,
                &model_config,
                directory.path(),
                "nextlat-revision",
                "ndarray",
            )
        };
        write(&config).unwrap();
        write(&config).unwrap();
        let path = directory.path().join(EXPERIMENT_MANIFEST_FILE_NAME);
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(
            manifest["launches"][1]["next_latent_objective_contract_version"],
            2
        );
        for launch in manifest["launches"].as_array_mut().unwrap() {
            launch
                .as_object_mut()
                .unwrap()
                .remove("next_latent_objective_contract_version");
        }
        fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        config.training.resume_horizon_extension.enabled = true;
        config.training.max_iters *= 2;
        let error = write(&config).unwrap_err().to_string();
        assert!(
            error.contains("NextLat objective contract changed"),
            "{error}"
        );
        assert!(error.contains("weights-only transfer"), "{error}");
    }

    #[test]
    fn provenance_capture_does_not_change_the_resume_contract() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut config = load_training_config(&[
            workspace.join("config/language/experiments/predictive_coding/local-pc-smoke.toml")
        ])
        .unwrap();
        let original = training_contract_sha256(&config).unwrap();
        let effective = effective_config_sha256(&config).unwrap();
        let serialized = serde_json::to_value(&config).unwrap();
        assert!(serialized["training"].get("provenance").is_none());
        config.training.provenance.initial_model_fingerprint = false;
        assert_eq!(original, training_contract_sha256(&config).unwrap());
        assert_ne!(effective, effective_config_sha256(&config).unwrap());
    }

    #[test]
    fn experiment_manifest_rejects_a_changed_supervision_audit() {
        let directory = tempfile::tempdir().expect("temporary run directory");
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let config = load_training_config(&[
            workspace.join("config/language/experiments/predictive_coding/local-pc-smoke.toml"),
            workspace.join(
                "config/language/experiments/predictive_coding/pc-fixed-prediction.overlay.toml",
            ),
        ])
        .expect("training config");
        let tokenizer = config
            .dataset
            .tokenizer
            .load(&workspace)
            .expect("tokenizer");
        let model_config = crate::build_model_config_with_tokenizer(
            &config.model,
            config.training.block_size,
            tokenizer.as_ref(),
        )
        .expect("model config");

        write_experiment_manifest_with_supervision_audit(
            &config,
            &model_config,
            directory.path(),
            "audit-manifest-test",
            "ndarray",
            Some("audit-a"),
        )
        .expect("first audited launch");
        let error = write_experiment_manifest_with_supervision_audit(
            &config,
            &model_config,
            directory.path(),
            "audit-manifest-test",
            "ndarray",
            Some("audit-b"),
        )
        .expect_err("a checkpoint run must retain its startup supervision identity");
        assert!(
            error
                .to_string()
                .contains("Ruliad supervision audit mismatch"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn experiment_manifest_records_and_preserves_initial_model_identity() {
        let directory = tempfile::tempdir().expect("temporary run directory");
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let config = load_training_config(&[
            workspace.join("config/language/experiments/predictive_coding/local-pc-smoke.toml"),
            workspace.join(
                "config/language/experiments/predictive_coding/pc-fixed-prediction.overlay.toml",
            ),
        ])
        .expect("training config");
        let tokenizer = config
            .dataset
            .tokenizer
            .load(&workspace)
            .expect("tokenizer");
        let model_config = crate::build_model_config_with_tokenizer(
            &config.model,
            config.training.block_size,
            tokenizer.as_ref(),
        )
        .expect("model config");

        write_experiment_manifest_with_identities(
            &config,
            &model_config,
            directory.path(),
            "model-identity-manifest-test",
            "ndarray",
            None,
            Some("initial-a"),
        )
        .expect("first launch with model identity");
        let manifest: ExperimentManifest = serde_json::from_slice(
            &fs::read(directory.path().join(EXPERIMENT_MANIFEST_FILE_NAME))
                .expect("read experiment manifest"),
        )
        .expect("parse experiment manifest");
        assert_eq!(
            manifest.launches[0]
                .initial_model_tensor_fingerprint_schema
                .as_deref(),
            Some(crate::train::model_identity::MODEL_TENSOR_FINGERPRINT_SCHEMA)
        );
        assert_eq!(
            manifest.launches[0].initial_model_sha256.as_deref(),
            Some("initial-a")
        );

        let error = write_experiment_manifest_with_identities(
            &config,
            &model_config,
            directory.path(),
            "model-identity-manifest-test",
            "ndarray",
            None,
            Some("initial-b"),
        )
        .expect_err("a run cannot change its recorded initial model identity");
        assert!(
            error
                .to_string()
                .contains("initial model identity mismatch"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn experiment_manifest_appends_launches_without_replacing_identity() {
        let directory = tempfile::tempdir().expect("temporary run directory");
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let config = load_training_config(&[
            workspace.join("config/language/experiments/predictive_coding/local-pc-smoke.toml"),
            workspace.join(
                "config/language/experiments/predictive_coding/pc-fixed-prediction.overlay.toml",
            ),
        ])
        .expect("training config");
        let tokenizer = config
            .dataset
            .tokenizer
            .load(&workspace)
            .expect("tokenizer");
        let model_config = crate::build_model_config_with_tokenizer(
            &config.model,
            config.training.block_size,
            tokenizer.as_ref(),
        )
        .expect("model config");

        let mut resume_config = config.clone();
        resume_config.training.launch_mode =
            burn_dragon_train::train::pipeline::TrainingLaunchMode::ResumeExactRun;
        resume_config.training.resume_run_dir = Some(directory.path().to_path_buf());
        write_experiment_manifest(
            &resume_config,
            &model_config,
            directory.path(),
            "manifest-test",
            "ndarray",
        )
        .expect("first manifest launch");
        write_experiment_manifest(
            &config,
            &model_config,
            directory.path(),
            "manifest-test",
            "ndarray",
        )
        .expect("second manifest launch");

        let bytes =
            fs::read(directory.path().join(EXPERIMENT_MANIFEST_FILE_NAME)).expect("read manifest");
        let manifest: ExperimentManifest = serde_json::from_slice(&bytes).expect("parse manifest");
        assert_eq!(manifest.schema_version, EXPERIMENT_MANIFEST_SCHEMA_VERSION);
        assert_eq!(manifest.run_name, "manifest-test");
        assert_eq!(manifest.launches.len(), 2);
        assert_eq!(
            manifest.launches[0].training_contract_sha256,
            manifest.launches[1].training_contract_sha256
        );
        assert_ne!(
            manifest.launches[0].effective_config_sha256,
            manifest.launches[1].effective_config_sha256
        );

        let mut changed_contract = resume_config;
        changed_contract.training.seed = changed_contract.training.seed.saturating_add(1);
        let error = write_experiment_manifest(
            &changed_contract,
            &model_config,
            directory.path(),
            "manifest-test",
            "ndarray",
        )
        .expect_err("changed training contract must be rejected");
        assert!(error.to_string().contains("training contract mismatch"));
    }

    #[test]
    fn exact_resume_horizon_extension_is_monotonic_and_semantics_preserving() {
        let directory = tempfile::tempdir().expect("temporary run directory");
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let config = load_training_config(&[
            workspace.join("config/language/experiments/predictive_coding/local-pc-smoke.toml")
        ])
        .expect("training config");
        let tokenizer = config
            .dataset
            .tokenizer
            .load(&workspace)
            .expect("tokenizer");
        let model_config = crate::build_model_config_with_tokenizer(
            &config.model,
            config.training.block_size,
            tokenizer.as_ref(),
        )
        .expect("model config");

        write_experiment_manifest(
            &config,
            &model_config,
            directory.path(),
            "horizon-extension-test",
            "ndarray",
        )
        .expect("fresh manifest");
        fs::create_dir_all(directory.path().join("checkpoint")).expect("checkpoint directory");
        fs::write(directory.path().join("checkpoint/model-1.bin"), b"model")
            .expect("checkpoint sentinel");
        save_experiment_checkpoint_progress(directory.path(), 1, 64).expect("checkpoint progress");

        let mut extension = config.clone();
        extension.training.launch_mode =
            burn_dragon_train::train::pipeline::TrainingLaunchMode::ResumeExactRun;
        extension.training.resume_run_dir = Some(directory.path().to_path_buf());
        extension.training.resume_checkpoint_epoch = Some(1);
        extension.training.resume_horizon_extension.enabled = true;
        extension.training.max_iters = 128;
        write_experiment_manifest(
            &extension,
            &model_config,
            directory.path(),
            "horizon-extension-test",
            "ndarray",
        )
        .expect("safe horizon extension");

        let manifest: ExperimentManifest = serde_json::from_slice(
            &fs::read(directory.path().join(EXPERIMENT_MANIFEST_FILE_NAME)).expect("read manifest"),
        )
        .expect("parse manifest");
        let horizon = manifest.launches[1]
            .horizon_extension
            .as_ref()
            .expect("extension audit record");
        assert_eq!(horizon.previous_max_iters, 64);
        assert_eq!(horizon.requested_max_iters, 128);
        assert_eq!(horizon.resume_completed_steps, 64);
        assert_eq!(horizon.checkpoint_epoch, 1);

        write_experiment_manifest(
            &extension,
            &model_config,
            directory.path(),
            "horizon-extension-test",
            "ndarray",
        )
        .expect("same extended horizon remains resumable");

        let mut shrink = extension.clone();
        shrink.training.max_iters = 96;
        let error = write_experiment_manifest(
            &shrink,
            &model_config,
            directory.path(),
            "horizon-extension-test",
            "ndarray",
        )
        .expect_err("horizon shrink must fail closed");
        assert!(error.to_string().contains("grow monotonically"));

        let mut semantic_drift = extension;
        semantic_drift.training.max_iters = 192;
        semantic_drift.training.seed = semantic_drift.training.seed.saturating_add(1);
        let error = write_experiment_manifest(
            &semantic_drift,
            &model_config,
            directory.path(),
            "horizon-extension-test",
            "ndarray",
        )
        .expect_err("immutable semantic drift must fail closed");
        assert!(error.to_string().contains("immutable training semantics"));
    }

    #[test]
    fn exact_resume_horizon_extension_rejects_implicit_finite_schedule() {
        let directory = tempfile::tempdir().expect("temporary run directory");
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut config = load_training_config(&[
            workspace.join("config/language/experiments/predictive_coding/local-pc-smoke.toml")
        ])
        .expect("training config");
        config.optimizer.lr_schedule =
            Some(burn_dragon_train::LearningRateScheduleConfig::Cosine {
                initial_lr: None,
                min_lr: Some(1.0e-5),
                warmup_steps: Some(8),
                num_iters: None,
            });
        let tokenizer = config
            .dataset
            .tokenizer
            .load(&workspace)
            .expect("tokenizer");
        let model_config = crate::build_model_config_with_tokenizer(
            &config.model,
            config.training.block_size,
            tokenizer.as_ref(),
        )
        .expect("model config");
        write_experiment_manifest(
            &config,
            &model_config,
            directory.path(),
            "implicit-schedule-test",
            "ndarray",
        )
        .expect("fresh manifest");
        fs::create_dir_all(directory.path().join("checkpoint")).expect("checkpoint directory");
        fs::write(directory.path().join("checkpoint/model-1.bin"), b"model")
            .expect("checkpoint sentinel");

        config.training.launch_mode =
            burn_dragon_train::train::pipeline::TrainingLaunchMode::ResumeExactRun;
        config.training.resume_run_dir = Some(directory.path().to_path_buf());
        config.training.resume_checkpoint_epoch = Some(1);
        config.training.resume_horizon_extension.enabled = true;
        config.training.max_iters = 128;
        let error = write_experiment_manifest(
            &config,
            &model_config,
            directory.path(),
            "implicit-schedule-test",
            "ndarray",
        )
        .expect_err("implicit finite schedule must fail closed");
        assert!(error.to_string().contains("independent of max_iters"));
    }
}
