use std::path::PathBuf;

pub use burn_dragon_core::objective::{
    RepromptTruncation, SdftObjectiveConfig, SdftSdpoObjectiveConfig, SdpoObjectiveConfig,
    SelfDistillationKlKind, TeacherRegularization, TrainingObjectiveConfig, TrainingObjectiveKind,
};
use burn_dragon_core::{LanguageModuleLrScaleTarget, SequenceKernelConfig};
use burn_dragon_train::ContinualBackpropConfig;

use super::*;

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
pub struct DatasetConfig {
    pub cache_dir: PathBuf,
    #[serde(default = "default_train_split_ratio")]
    pub train_split_ratio: f32,
    #[serde(default)]
    pub validation: Option<ValidationDatasetConfig>,
    #[serde(flatten)]
    pub source: DatasetSourceConfig,
    #[serde(default)]
    pub tokenizer: TokenizerConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ValidationDatasetConfig {
    #[serde(default)]
    pub cache_dir: Option<PathBuf>,
    #[serde(default)]
    pub train_split_ratio: Option<f32>,
    #[serde(flatten)]
    pub source: DatasetSourceConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DatasetSourceConfig {
    NemotronClimbMix {
        #[serde(default)]
        revision: Option<String>,
        #[serde(default)]
        max_records: Option<usize>,
    },
    UniversalityManifest {
        manifest: PathBuf,
    },
    UniversalityNca {
        config: PathBuf,
    },
    UniversalityRuliad {
        config: PathBuf,
    },
}

impl Default for DatasetSourceConfig {
    fn default() -> Self {
        Self::NemotronClimbMix {
            revision: None,
            max_records: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct HuggingFaceDatasetConfig {
    pub repo_id: String,
    #[serde(default)]
    pub token: Option<String>,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub format: HuggingFaceRecordFormat,
    #[serde(default = "default_hf_train_files")]
    pub train_files: Vec<String>,
    #[serde(default)]
    pub auto_discover_train_files: bool,
    #[serde(default)]
    pub validation_files: Vec<String>,
    #[serde(default = "default_hf_text_fields")]
    pub text_fields: Vec<String>,
    #[serde(default)]
    pub sequence_field: Option<String>,
    #[serde(default = "default_hf_field_separator")]
    pub field_separator: String,
    #[serde(default)]
    pub template: Option<String>,
    #[serde(default)]
    pub max_records: Option<usize>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum HuggingFaceRecordFormat {
    #[default]
    Jsonl,
    Text,
    Parquet,
    Csv,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct InitTransferConfig {
    #[serde(default)]
    pub interface_checkpoint_path: Option<PathBuf>,
    #[serde(default)]
    pub interface_checkpoint_epoch: Option<usize>,
    #[serde(default)]
    pub preserve_interface_input_embedding: bool,
    #[serde(default)]
    pub preserve_interface_output_head: bool,
    #[serde(default)]
    pub interface_output_head_blend_alpha: Option<f32>,
    #[serde(default)]
    pub backbone_blend_alpha: Option<f32>,
    #[serde(default)]
    pub decoder_blend_alpha: Option<f32>,
    #[serde(default)]
    pub norm_blend_alpha: Option<f32>,
    #[serde(default)]
    pub backbone_grad_scale: Option<f32>,
    #[serde(default)]
    pub backbone_grad_scale_steps: Option<usize>,
    #[serde(default)]
    pub fresh_top_layers: Option<usize>,
    #[serde(default)]
    pub preserve_fresh_decoder: bool,
    #[serde(default)]
    pub preserve_fresh_norm: bool,
    #[serde(default)]
    pub match_fresh_rms: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ModuleLrScaleScheduleConfig {
    pub final_scale: f32,
    #[serde(default)]
    pub start_fraction: f32,
    #[serde(default = "default_module_lr_scale_schedule_end_fraction")]
    pub end_fraction: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ModuleLrScaleEntry {
    pub target: LanguageModuleLrScaleTarget,
    pub scale: f32,
    #[serde(default)]
    pub schedule: Option<ModuleLrScaleScheduleConfig>,
}

fn default_neuron_scaling_max_latent_total() -> usize {
    8192
}

fn default_neuron_scaling_min_steps_between_scales() -> usize {
    2_000
}

fn default_neuron_scaling_max_scale_events() -> usize {
    4
}

fn default_neuron_scaling_capacity_patience_epochs() -> usize {
    2
}

fn default_neuron_scaling_freeze_base_steps() -> usize {
    256
}

fn default_neuron_scaling_unfreeze_ramp_steps() -> usize {
    256
}

fn default_neuron_scaling_lr_scale() -> f32 {
    1.0
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum NeuronScalingGrowth {
    #[default]
    Double,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct NeuronScalingStabilizationConfig {
    #[serde(default = "default_neuron_scaling_freeze_base_steps")]
    pub freeze_base_steps: usize,
    #[serde(default = "default_neuron_scaling_unfreeze_ramp_steps")]
    pub unfreeze_ramp_steps: usize,
    #[serde(default = "default_neuron_scaling_lr_scale")]
    pub new_slice_lr_scale: f32,
    #[serde(default = "default_neuron_scaling_lr_scale")]
    pub base_lr_scale_after_ramp: f32,
}

impl Default for NeuronScalingStabilizationConfig {
    fn default() -> Self {
        Self {
            freeze_base_steps: default_neuron_scaling_freeze_base_steps(),
            unfreeze_ramp_steps: default_neuron_scaling_unfreeze_ramp_steps(),
            new_slice_lr_scale: default_neuron_scaling_lr_scale(),
            base_lr_scale_after_ramp: default_neuron_scaling_lr_scale(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct NeuronScalingConfig {
    pub enabled: bool,
    #[serde(default = "default_neuron_scaling_max_latent_total")]
    pub max_latent_total: usize,
    pub growth: NeuronScalingGrowth,
    #[serde(default = "default_neuron_scaling_min_steps_between_scales")]
    pub min_steps_between_scales: usize,
    #[serde(default = "default_neuron_scaling_max_scale_events")]
    pub max_scale_events: usize,
    #[serde(default = "default_neuron_scaling_capacity_patience_epochs")]
    pub capacity_patience_epochs: usize,
    pub require_live_source_selection: bool,
    pub stabilization: NeuronScalingStabilizationConfig,
}

impl Default for NeuronScalingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_latent_total: default_neuron_scaling_max_latent_total(),
            growth: NeuronScalingGrowth::default(),
            min_steps_between_scales: default_neuron_scaling_min_steps_between_scales(),
            max_scale_events: default_neuron_scaling_max_scale_events(),
            capacity_patience_epochs: default_neuron_scaling_capacity_patience_epochs(),
            require_live_source_selection: true,
            stabilization: NeuronScalingStabilizationConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct TrainingHyperparameters {
    pub block_size: usize,
    #[serde(default)]
    pub tbptt_chunk_size: Option<usize>,
    #[serde(default)]
    pub tbptt_persist_across_steps: bool,
    #[serde(default)]
    pub min_logical_block_size: Option<usize>,
    pub batch_size: usize,
    #[serde(default = "default_training_seed")]
    pub seed: u64,
    #[serde(default = "default_gradient_accumulation_steps")]
    pub gradient_accumulation_steps: usize,
    #[serde(default)]
    pub target_effective_batch_size: Option<usize>,
    #[serde(default)]
    pub epochs: Option<usize>,
    pub max_iters: usize,
    #[serde(default = "default_checkpoint_interval_iters")]
    pub checkpoint_interval_iters: usize,
    pub log_frequency: usize,
    #[serde(default)]
    pub launch_mode: burn_dragon_train::train::pipeline::TrainingLaunchMode,
    #[serde(default)]
    pub resume_run_dir: Option<PathBuf>,
    #[serde(default)]
    pub resume_checkpoint_epoch: Option<usize>,
    #[serde(default)]
    pub init_checkpoint_path: Option<PathBuf>,
    #[serde(default)]
    pub init_checkpoint_epoch: Option<usize>,
    #[serde(default)]
    pub init_transfer: InitTransferConfig,
    #[serde(default)]
    pub continual_backprop: ContinualBackpropConfig,
    #[serde(default)]
    pub neuron_scaling: NeuronScalingConfig,
    #[serde(default)]
    pub module_lr_scales: Vec<ModuleLrScaleEntry>,
    #[serde(default = "default_context_strategy")]
    pub context_strategy: ContextStrategyConfig,
    #[serde(default)]
    pub sequence_kernel_override: Option<SequenceKernelConfig>,
    #[serde(default)]
    pub objective: TrainingObjectiveConfig,
    #[serde(default)]
    pub gdpo: Option<burn_dragon_train::GdpoConfig>,
    #[serde(default)]
    pub events: burn_dragon_train::TrainingEventsConfig,
    #[serde(default)]
    pub gates: burn_dragon_train::TrainingGatesConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct TrainingConfig {
    pub dataset: DatasetConfig,
    pub training: TrainingHyperparameters,
    pub optimizer: burn_dragon_train::OptimizerConfig,
    #[serde(default)]
    pub parallel: burn_dragon_train::ParallelConfig,
    pub generation: GenerationConfig,
    #[serde(default)]
    pub wgpu: burn_dragon_train::WgpuRuntimeConfig,
    #[serde(default)]
    pub run_layout: burn_dragon_train::RunLayoutConfig,
    #[serde(default)]
    pub model: ModelOverrides,
}

fn default_train_split_ratio() -> f32 {
    0.9
}

fn default_hf_train_files() -> Vec<String> {
    vec!["train.jsonl".to_string()]
}

fn default_hf_text_fields() -> Vec<String> {
    vec!["text".to_string()]
}

fn default_hf_field_separator() -> String {
    "\n".to_string()
}

fn default_context_strategy() -> ContextStrategyConfig {
    ContextStrategyConfig::Infinite
}

fn default_module_lr_scale_schedule_end_fraction() -> f32 {
    1.0
}

fn default_training_seed() -> u64 {
    1337
}

fn default_gradient_accumulation_steps() -> usize {
    1
}

fn default_checkpoint_interval_iters() -> usize {
    2_000
}
