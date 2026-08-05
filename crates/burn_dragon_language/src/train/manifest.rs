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
    pub launch_mode: burn_dragon_train::train::pipeline::TrainingLaunchMode,
    pub resume_checkpoint_epoch: Option<usize>,
    pub checkpoint_artifacts: Vec<ExperimentCheckpointArtifact>,
    pub config_snapshot: PathBuf,
    pub git: ExperimentGitRevision,
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

pub fn write_experiment_manifest(
    config: &TrainingConfig,
    model_config: &DragonConfig,
    run_dir: &Path,
    run_name: &str,
    backend: &str,
) -> Result<()> {
    fs::create_dir_all(run_dir)
        .with_context(|| format!("create experiment run directory {}", run_dir.display()))?;
    let path = run_dir.join(EXPERIMENT_MANIFEST_FILE_NAME);
    let launch = ExperimentLaunch {
        unix_time_ms: unix_time_ms(),
        command: std::env::args().collect(),
        effective_config_sha256: effective_config_sha256(config)?,
        training_contract_sha256: training_contract_sha256(config)?,
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
        if let Some(first) = existing.launches.first()
            && first.training_contract_sha256 != launch.training_contract_sha256
        {
            return Err(anyhow!(
                "experiment training contract mismatch in {}: existing={}, requested={}",
                path.display(),
                first.training_contract_sha256,
                launch.training_contract_sha256
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
    let mut launch = launch;
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
    contract.training.launch_mode = Default::default();
    contract.training.resume_run_dir = None;
    contract.training.resume_checkpoint_epoch = None;
    contract.training.source_selection_state_path = None;
    config_sha256(&contract, "normalized training contract")
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
}
