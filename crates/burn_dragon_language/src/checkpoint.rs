#![cfg(feature = "train")]

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use burn::module::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::Backend as BackendTrait;
use burn_dragon_checkpoint::{
    BurnpackBundleExportOptions, BurnpackBundleExportReport, export_model_to_burnpack_bundle,
    format_checkpoint_load_error, load_json_snapshot,
    resolve_checkpoint_base as resolve_checkpoint_base_shared,
    resolve_checkpoint_run_dir as resolve_checkpoint_run_dir_shared, run_snapshot_path,
    write_json_snapshot,
};
use burn_dragon_train::train::metrics::MetricsSinkSpec;
use burn_dragon_train::train::pipeline::resolve_latest_run_dir_in as resolve_latest_run_dir_shared;
use burn_dragon_train::{KernelSpec, ModelSpec, OptimizerSpec, ParallelSpec, StateLayout};
use burn_ndarray::NdArray;
use serde::{Deserialize, Serialize};

use crate::config::load_training_config;
use crate::tokenizer::{SharedTokenizer, Tokenizer};
use crate::{DragonModel, ModelOverrides, TrainingConfig, build_model_config_with_tokenizer};

const RUN_CONFIG_FILE_NAME: &str = "config.json";
const TRAINING_SNAPSHOT_FILE_NAME: &str = "training_config.json";
const TOKENIZER_SNAPSHOT_FILE_NAME: &str = "tokenizer.json";
pub const RUN_ROOT_ENV: &str = "BURN_DRAGON_RUN_ROOT";
pub const RUN_DIR_ENV: &str = "BURN_DRAGON_RUN_DIR";
pub const RUN_NAME_ENV: &str = "BURN_DRAGON_RUN_NAME";

type ExportBackend = NdArray<f32>;

#[derive(Debug, Clone, Deserialize, Serialize, Default, PartialEq)]
pub struct LanguageRunConfigSnapshot {
    #[serde(default)]
    pub block_size: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub training_execution_form: Option<String>,
    #[serde(default)]
    pub training_launch_mode_requested:
        Option<burn_dragon_train::train::pipeline::TrainingLaunchMode>,
    #[serde(default)]
    pub training_sequence_kernel_override: Option<burn_dragon_core::SequenceKernelConfig>,
    #[serde(default)]
    pub arch_version: Option<String>,
    #[serde(default)]
    pub shard_layout_version: Option<u32>,
    #[serde(default)]
    pub overrides: ModelOverrides,
    #[serde(default)]
    pub model_spec: Option<ModelSpec>,
    #[serde(default)]
    pub optimizer_spec: Option<OptimizerSpec>,
    #[serde(default)]
    pub parallel_spec: Option<ParallelSpec>,
    #[serde(default)]
    pub kernel_spec: Option<KernelSpec>,
    #[serde(default)]
    pub state_layout: Option<StateLayout>,
    #[serde(default)]
    pub metrics_sink: Option<MetricsSinkSpec>,
}

#[derive(Debug, Clone)]
pub struct LanguageBurnpackExportReport {
    pub checkpoint_base: PathBuf,
    pub epoch: usize,
    pub vocab_size: usize,
    pub run_dir: Option<PathBuf>,
    pub bundle: BurnpackBundleExportReport,
}

pub fn write_training_snapshot(
    config: &TrainingConfig,
    run_dir: &Path,
    tokenizer: &dyn Tokenizer,
) -> Result<()> {
    fs::create_dir_all(run_dir)
        .with_context(|| format!("failed to create run directory {}", run_dir.display()))?;

    let mut snapshot = config.clone();
    if snapshot
        .dataset
        .tokenizer
        .storage_path(Path::new("."))
        .is_some()
    {
        let tokenizer_path = tokenizer_snapshot_path(run_dir);
        let source_tokenizer_path = snapshot
            .dataset
            .tokenizer
            .storage_path(&snapshot.dataset.cache_dir);
        if let Err(error) = snapshot.dataset.tokenizer.save(tokenizer, &tokenizer_path) {
            let copied = source_tokenizer_path
                .as_ref()
                .filter(|path| path.is_file())
                .and_then(|source_path| {
                    if fs::copy(source_path, &tokenizer_path).is_ok() {
                        Some(())
                    } else {
                        None
                    }
                })
                .is_some();
            if !copied {
                return Err(error).with_context(|| {
                    format!(
                        "failed to save tokenizer snapshot {}",
                        tokenizer_path.display()
                    )
                });
            }
        }
        snapshot.dataset.cache_dir = PathBuf::from(".");
        snapshot.dataset.tokenizer.vocab_path = Some(PathBuf::from(TOKENIZER_SNAPSHOT_FILE_NAME));
    }

    write_json_snapshot(run_dir, TRAINING_SNAPSHOT_FILE_NAME, &snapshot)
}

pub fn load_training_snapshot_from_run_dir(run_dir: &Path) -> Result<TrainingConfig> {
    let mut config: TrainingConfig = load_json_snapshot(run_dir, TRAINING_SNAPSHOT_FILE_NAME)?;
    apply_run_dir_tokenizer_snapshot(&mut config, run_dir);
    absolutize_snapshot_cache_dir(&mut config, run_dir);
    Ok(config)
}

pub fn load_training_config_for_checkpoint(
    config_paths: &[PathBuf],
    checkpoint: Option<&PathBuf>,
    backend_name: &str,
) -> Result<TrainingConfig> {
    let run_dir = resolve_checkpoint_run_dir(checkpoint, backend_name);

    if !config_paths.is_empty() {
        let mut config = load_training_config(config_paths)?;
        if let Some(run_dir) = run_dir.as_deref() {
            apply_run_dir_tokenizer_snapshot(&mut config, run_dir);
        }
        return Ok(config);
    }

    if let Some(run_dir) = run_dir.as_deref() {
        let snapshot_path = training_snapshot_path(run_dir);
        if snapshot_path.is_file() {
            return load_training_snapshot_from_run_dir(run_dir);
        }
    }

    let mut config = load_training_config(&[PathBuf::from("config/language/base.toml")])?;
    if let Some(path) = resolve_run_config_path(checkpoint, backend_name) {
        let contents = fs::read_to_string(&path)
            .with_context(|| format!("failed to read run config {}", path.display()))?;
        let run_config: LanguageRunConfigSnapshot = serde_json::from_str(&contents)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        apply_run_config(&mut config, &run_config);
    }
    if let Some(run_dir) = run_dir.as_deref() {
        apply_run_dir_tokenizer_snapshot(&mut config, run_dir);
    }
    Ok(config)
}

pub fn export_language_checkpoint_to_burnpack(
    checkpoint: &Path,
    epoch: Option<usize>,
    config_paths: &[PathBuf],
    backend_name: &str,
    output_base: &Path,
    options: &BurnpackBundleExportOptions,
) -> Result<LanguageBurnpackExportReport> {
    let (checkpoint_base, epoch) = resolve_checkpoint_base(checkpoint, epoch)?;
    let checkpoint_path = checkpoint.to_path_buf();
    let config =
        load_training_config_for_checkpoint(config_paths, Some(&checkpoint_path), backend_name)?;

    let tokenizer_path = config
        .dataset
        .tokenizer
        .storage_path(&config.dataset.cache_dir);
    let tokenizer = if let Some(path) = tokenizer_path {
        config
            .dataset
            .tokenizer
            .load(&path)
            .with_context(|| format!("failed to load tokenizer {}", path.display()))?
    } else {
        config
            .dataset
            .tokenizer
            .fit(std::iter::empty::<&str>())
            .context("failed to initialize tokenizer")?
    };

    let model_config = build_model_config_with_tokenizer(
        &config.model,
        config.training.block_size,
        tokenizer.as_ref(),
    )?;

    let device = burn::tensor::Device::<ExportBackend>::default();
    ExportBackend::seed(&device, 1337);
    let mut model = DragonModel::<ExportBackend>::new(model_config, &device);
    let record = BinFileRecorder::<FullPrecisionSettings>::new()
        .load::<<DragonModel<ExportBackend> as Module<ExportBackend>>::Record>(
            checkpoint_base.clone(),
            &device,
        )
        .map_err(|err| anyhow!(format_checkpoint_load_error(&checkpoint_base, err)))?;
    model = model.load_record(record);

    let bundle = export_model_to_burnpack_bundle(&model, output_base, options)
        .map_err(|err| anyhow!(err))?;

    Ok(LanguageBurnpackExportReport {
        checkpoint_base,
        epoch,
        vocab_size: tokenizer.len(),
        run_dir: resolve_checkpoint_run_dir(Some(&checkpoint_path), backend_name),
        bundle,
    })
}

pub fn load_tokenizer_for_checkpoint(
    config_paths: &[PathBuf],
    checkpoint: Option<&PathBuf>,
    backend_name: &str,
) -> Result<SharedTokenizer> {
    let config = load_training_config_for_checkpoint(config_paths, checkpoint, backend_name)?;
    let tokenizer_path = config
        .dataset
        .tokenizer
        .storage_path(&config.dataset.cache_dir);
    match tokenizer_path {
        Some(path) => config
            .dataset
            .tokenizer
            .load(&path)
            .with_context(|| format!("failed to load tokenizer {}", path.display())),
        None => config
            .dataset
            .tokenizer
            .fit(std::iter::empty::<&str>())
            .context("failed to initialize tokenizer"),
    }
}

pub fn load_language_core_from_checkpoint<B: BackendTrait>(
    checkpoint: &Path,
    epoch: Option<usize>,
    config_paths: &[PathBuf],
    backend_name: &str,
    device: &B::Device,
) -> Result<DragonModel<B>> {
    let (checkpoint_base, _epoch) = resolve_checkpoint_base(checkpoint, epoch)?;
    let checkpoint_path = checkpoint.to_path_buf();
    let config =
        load_training_config_for_checkpoint(config_paths, Some(&checkpoint_path), backend_name)?;
    let tokenizer =
        load_tokenizer_for_checkpoint(config_paths, Some(&checkpoint_path), backend_name)?;
    let model_config = build_model_config_with_tokenizer(
        &config.model,
        config.training.block_size,
        tokenizer.as_ref(),
    )?;
    let mut model = DragonModel::<B>::new(model_config, device);
    let record = BinFileRecorder::<FullPrecisionSettings>::new()
        .load::<<DragonModel<B> as Module<B>>::Record>(checkpoint_base.clone(), device)
        .map_err(|err| anyhow!(format_checkpoint_load_error(&checkpoint_base, err)))?;
    model = model.load_record(record);
    Ok(model)
}

pub fn apply_init_checkpoint_to_language_core<B: BackendTrait>(
    target_model: &DragonModel<B>,
    target_config: &TrainingConfig,
    init_checkpoint_path: &Path,
    init_checkpoint_epoch: Option<usize>,
    backend_name: &str,
    device: &B::Device,
) -> Result<DragonModel<B>> {
    let checkpoint_path = init_checkpoint_path.to_path_buf();
    let (checkpoint_base, epoch) = resolve_checkpoint_base(&checkpoint_path, init_checkpoint_epoch)
        .with_context(|| {
            format!(
                "failed to resolve init checkpoint from {}",
                checkpoint_path.display()
            )
        })?;
    let record = BinFileRecorder::<FullPrecisionSettings>::new()
        .load::<<DragonModel<B> as Module<B>>::Record>(checkpoint_base.clone(), device)
        .map_err(|err| anyhow!(format_checkpoint_load_error(&checkpoint_base, err)))?;
    let source_config =
        load_training_config_for_checkpoint(&[], Some(&checkpoint_path), backend_name)
            .with_context(|| {
                format!(
                    "failed to load source training config for init checkpoint {}",
                    checkpoint_path.display()
                )
            })?;
    let current_language_head = target_config
        .model
        .language_head
        .clone()
        .unwrap_or_default();
    let source_language_head = source_config
        .model
        .language_head
        .clone()
        .unwrap_or_default();
    let preserve_input_embedding =
        source_config.dataset.tokenizer.kind != target_config.dataset.tokenizer.kind;
    let preserve_output_head =
        preserve_input_embedding || source_language_head != current_language_head;
    let loaded = if preserve_input_embedding || preserve_output_head {
        target_model.load_record_preserving_tokenizer_surfaces(
            record,
            preserve_input_embedding,
            preserve_output_head,
        )
    } else {
        target_model.clone().load_record(record)
    };
    let interface_reference = if let Some(interface_checkpoint_path) = target_config
        .training
        .init_transfer
        .interface_checkpoint_path
        .as_ref()
    {
        let (interface_base, interface_epoch) = resolve_checkpoint_base(
            interface_checkpoint_path,
            target_config
                .training
                .init_transfer
                .interface_checkpoint_epoch,
        )
        .with_context(|| {
            format!(
                "failed to resolve init transfer interface checkpoint from {}",
                interface_checkpoint_path.display()
            )
        })?;
        let interface_record = BinFileRecorder::<FullPrecisionSettings>::new()
            .load::<<DragonModel<B> as Module<B>>::Record>(interface_base.clone(), device)
            .map_err(|err| anyhow!(format_checkpoint_load_error(&interface_base, err)))?;
        let interface_config =
            load_training_config_for_checkpoint(&[], Some(interface_checkpoint_path), backend_name)
                .with_context(|| {
                    format!(
                        "failed to load interface training config for checkpoint {}",
                        interface_checkpoint_path.display()
                    )
                })?;
        let interface_language_head = interface_config
            .model
            .language_head
            .clone()
            .unwrap_or_default();
        let preserve_interface_embedding =
            interface_config.dataset.tokenizer.kind != target_config.dataset.tokenizer.kind;
        let preserve_interface_head =
            preserve_interface_embedding || interface_language_head != current_language_head;
        let interface_model = if preserve_interface_embedding || preserve_interface_head {
            target_model.load_record_preserving_tokenizer_surfaces(
                interface_record,
                preserve_interface_embedding,
                preserve_interface_head,
            )
        } else {
            target_model.clone().load_record(interface_record)
        };
        Some((interface_model, interface_base, interface_epoch))
    } else {
        None
    };
    let reference_model = interface_reference
        .as_ref()
        .map(|(model, _, _)| model)
        .unwrap_or(target_model);
    let loaded = if let Some((interface_model, _, _)) = interface_reference.as_ref() {
        let interface_checkpoint_config = load_training_config_for_checkpoint(
            &[],
            target_config
                .training
                .init_transfer
                .interface_checkpoint_path
                .as_ref(),
            backend_name,
        )?;
        if target_config
            .training
            .init_transfer
            .preserve_interface_input_embedding
            || target_config
                .training
                .init_transfer
                .preserve_interface_output_head
            || target_config
                .training
                .init_transfer
                .interface_output_head_blend_alpha
                .is_some()
        {
            anyhow::ensure!(
                interface_checkpoint_config.dataset.tokenizer.kind
                    == target_config.dataset.tokenizer.kind,
                "training.init_transfer.preserve_interface_input_embedding/output_head requires interface tokenizer kind to match target tokenizer kind"
            );
            anyhow::ensure!(
                interface_checkpoint_config
                    .model
                    .language_head
                    .clone()
                    .unwrap_or_default()
                    == current_language_head,
                "training.init_transfer.preserve_interface_output_head requires interface language head to match target language head"
            );
        }
        loaded
            .with_tokenizer_surfaces_from(
                interface_model,
                target_config
                    .training
                    .init_transfer
                    .preserve_interface_input_embedding,
                target_config
                    .training
                    .init_transfer
                    .preserve_interface_output_head,
            )
            .with_output_head_blended_from(
                interface_model,
                target_config
                    .training
                    .init_transfer
                    .interface_output_head_blend_alpha
                    .unwrap_or(0.0),
            )
    } else {
        loaded
    };
    let loaded = loaded.adapted_transferred_backbone(
        reference_model,
        target_config.training.init_transfer.backbone_blend_alpha,
        target_config.training.init_transfer.decoder_blend_alpha,
        target_config.training.init_transfer.norm_blend_alpha,
        target_config.training.init_transfer.fresh_top_layers,
        target_config.training.init_transfer.preserve_fresh_decoder,
        target_config.training.init_transfer.preserve_fresh_norm,
        target_config.training.init_transfer.match_fresh_rms,
    );
    tracing::info!(
        "initialized model weights from checkpoint epoch {epoch} at {} (interface_checkpoint={}, interface_epoch={:?}, interface_embed={}, interface_head={}, interface_head_blend_alpha={:?}, blend_alpha={:?}, decoder_blend_alpha={:?}, norm_blend_alpha={:?}, fresh_top_layers={:?}, preserve_fresh_decoder={}, preserve_fresh_norm={}, match_fresh_rms={})",
        checkpoint_base.display(),
        interface_reference
            .as_ref()
            .map(|(_, base, _)| base.display().to_string())
            .unwrap_or_else(|| "none".to_string()),
        interface_reference.as_ref().map(|(_, _, epoch)| *epoch),
        target_config
            .training
            .init_transfer
            .preserve_interface_input_embedding,
        target_config
            .training
            .init_transfer
            .preserve_interface_output_head,
        target_config
            .training
            .init_transfer
            .interface_output_head_blend_alpha,
        target_config.training.init_transfer.backbone_blend_alpha,
        target_config.training.init_transfer.decoder_blend_alpha,
        target_config.training.init_transfer.norm_blend_alpha,
        target_config.training.init_transfer.fresh_top_layers,
        target_config.training.init_transfer.preserve_fresh_decoder,
        target_config.training.init_transfer.preserve_fresh_norm,
        target_config.training.init_transfer.match_fresh_rms
    );
    Ok(loaded)
}

pub fn apply_run_config(config: &mut TrainingConfig, run_config: &LanguageRunConfigSnapshot) {
    let block_override = run_config
        .block_size
        .or(run_config.overrides.block_size)
        .map(|value| value.max(1));
    if let Some(block_size) = block_override {
        config.training.block_size = block_size;
    }
    if let Some(sequence_kernel_override) = run_config.training_sequence_kernel_override {
        config.training.sequence_kernel_override = Some(sequence_kernel_override);
    }
    if let Some(launch_mode) = run_config.training_launch_mode_requested {
        config.training.launch_mode = launch_mode;
    }
    merge_model_overrides(&mut config.model, &run_config.overrides);
}

pub fn merge_model_overrides(base: &mut ModelOverrides, incoming: &ModelOverrides) {
    if let Some(value) = incoming.n_layer {
        base.n_layer = Some(value);
    }
    if let Some(value) = incoming.n_embd {
        base.n_embd = Some(value);
    }
    if let Some(value) = incoming.n_head {
        base.n_head = Some(value);
    }
    if let Some(value) = incoming.mlp_internal_dim_multiplier {
        base.mlp_internal_dim_multiplier = Some(value);
    }
    if let Some(value) = incoming.latent_total {
        base.latent_total = Some(value);
    }
    if let Some(value) = &incoming.initialization {
        base.initialization = Some(value.clone());
    }
    if let Some(value) = incoming.sequence_kernel {
        base.sequence_kernel = Some(value);
    }
    if let Some(value) = &incoming.mamba {
        base.mamba = Some(value.clone());
    }
    if let Some(value) = &incoming.gated_deltanet2 {
        base.gated_deltanet2 = Some(value.clone());
    }
    if let Some(value) = incoming.residual_connector {
        base.residual_connector = Some(value);
    }
    if let Some(value) = &incoming.attention_residual {
        base.attention_residual = Some(value.clone());
    }
    if let Some(value) = &incoming.block_attention_residual {
        base.block_attention_residual = Some(value.clone());
    }
    if let Some(value) = &incoming.latent_fanout_schedule {
        base.latent_fanout_schedule = Some(value.clone());
    }
    if let Some(value) = incoming.relu_threshold {
        base.relu_threshold = Some(value);
    }
    if let Some(value) = incoming.dropout {
        base.dropout = Some(value);
    }
    if let Some(value) = &incoming.normalization {
        base.normalization = Some(value.clone());
    }
    if let Some(value) = incoming.fused_kernels {
        base.fused_kernels = Some(value);
    }
    if let Some(value) = incoming.block_size {
        base.block_size = Some(value);
    }
    if let Some(value) = incoming.rollout_fast_steps_per_slow_step {
        base.rollout_fast_steps_per_slow_step = Some(value);
    }
    if let Some(value) = incoming.rotary_embedding {
        base.rotary_embedding = Some(value);
    }
    if let Some(value) = &incoming.y_neuron_recurrence {
        base.y_neuron_recurrence = Some(value.clone());
    }
    if let Some(value) = &incoming.clocked_slow_memory {
        base.clocked_slow_memory = Some(value.clone());
    }
    if let Some(value) = &incoming.summary_memory {
        base.summary_memory = Some(value.clone());
    }
    if let Some(value) = &incoming.mhc {
        base.mhc = Some(value.clone());
    }
}

pub fn resolve_run_config_path(
    checkpoint: Option<&PathBuf>,
    backend_name: &str,
) -> Option<PathBuf> {
    resolve_checkpoint_run_dir(checkpoint, backend_name).and_then(|run_dir| {
        let path = run_dir.join(RUN_CONFIG_FILE_NAME);
        path.is_file().then_some(path)
    })
}

pub(crate) fn resolve_checkpoint_run_dir(
    checkpoint: Option<&PathBuf>,
    backend_name: &str,
) -> Option<PathBuf> {
    let checkpoint_path = checkpoint
        .cloned()
        .unwrap_or_else(|| default_checkpoint_dir(backend_name));
    resolve_checkpoint_run_dir_shared(&checkpoint_path)
}

pub fn default_checkpoint_dir(backend_name: &str) -> PathBuf {
    resolve_latest_run_dir(backend_name)
        .map(|dir| dir.join("checkpoint"))
        .unwrap_or_else(|| resolve_run_root().join("checkpoint"))
}

pub fn resolve_latest_run_dir(backend_name: &str) -> Option<PathBuf> {
    let run_root = resolve_run_root();
    resolve_latest_run_dir_shared(&run_root).or_else(|| {
        let device_root = run_root.join(backend_name);
        resolve_latest_run_dir_shared(&device_root)
    })
}

pub fn resolve_latest_run_dir_in(run_root: &Path) -> Option<PathBuf> {
    resolve_latest_run_dir_shared(run_root)
}

pub fn resolve_run_root() -> PathBuf {
    std::env::var_os(RUN_ROOT_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("runs"))
}

pub fn training_snapshot_path(run_dir: &Path) -> PathBuf {
    run_snapshot_path(run_dir, TRAINING_SNAPSHOT_FILE_NAME)
}

pub fn tokenizer_snapshot_path(run_dir: &Path) -> PathBuf {
    run_dir.join(TOKENIZER_SNAPSHOT_FILE_NAME)
}

pub(crate) fn resolve_checkpoint_base(
    path: &Path,
    epoch: Option<usize>,
) -> Result<(PathBuf, usize)> {
    resolve_checkpoint_base_shared(path, epoch)
}

fn apply_run_dir_tokenizer_snapshot(config: &mut TrainingConfig, run_dir: &Path) {
    let tokenizer_path = tokenizer_snapshot_path(run_dir);
    if tokenizer_path.is_file() {
        config.dataset.cache_dir = run_dir.to_path_buf();
        config.dataset.tokenizer.vocab_path = Some(PathBuf::from(TOKENIZER_SNAPSHOT_FILE_NAME));
    }
}

fn absolutize_snapshot_cache_dir(config: &mut TrainingConfig, run_dir: &Path) {
    if !config.dataset.cache_dir.is_absolute() {
        let cwd_relative = std::env::current_dir()
            .ok()
            .map(|cwd| cwd.join(&config.dataset.cache_dir));
        config.dataset.cache_dir = match cwd_relative {
            Some(path) if path.exists() => path,
            _ => run_dir.join(&config.dataset.cache_dir),
        };
    }
    if let Some(validation) = &mut config.dataset.validation
        && let Some(cache_dir) = &mut validation.cache_dir
        && !cache_dir.is_absolute()
    {
        let cwd_relative = std::env::current_dir()
            .ok()
            .map(|cwd| cwd.join(&*cache_dir));
        *cache_dir = match cwd_relative {
            Some(path) if path.exists() => path,
            _ => run_dir.join(&*cache_dir),
        };
    }
}
