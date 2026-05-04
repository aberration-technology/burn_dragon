use std::path::PathBuf;

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

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TrainingObjectiveConfig {
    #[default]
    NextToken,
    Sdft(SdftObjectiveConfig),
    Sdpo(SdpoObjectiveConfig),
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrainingObjectiveKind {
    NextToken,
    Sdft,
    Sdpo,
}

impl TrainingObjectiveConfig {
    pub fn kind(&self) -> TrainingObjectiveKind {
        match self {
            Self::NextToken => TrainingObjectiveKind::NextToken,
            Self::Sdft(_) => TrainingObjectiveKind::Sdft,
            Self::Sdpo(_) => TrainingObjectiveKind::Sdpo,
        }
    }

    pub fn is_next_token(&self) -> bool {
        matches!(self, Self::NextToken)
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SelfDistillationKlKind {
    #[default]
    Forward,
    Reverse,
    JensenShannon,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum TeacherRegularization {
    #[default]
    Ema,
    TrustRegion,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum RepromptTruncation {
    Left,
    #[default]
    Right,
    Error,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct SdftObjectiveConfig {
    pub max_completion_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub kl: SelfDistillationKlKind,
    pub generate_from_teacher: bool,
    pub teacher_update_rate: f32,
    pub top_entropy_quantile: Option<f32>,
    pub num_loss_tokens_to_skip: usize,
}

impl Default for SdftObjectiveConfig {
    fn default() -> Self {
        Self {
            max_completion_tokens: 32,
            temperature: 1.0,
            top_k: None,
            kl: SelfDistillationKlKind::Forward,
            generate_from_teacher: false,
            teacher_update_rate: 0.01,
            top_entropy_quantile: None,
            num_loss_tokens_to_skip: 0,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct SdpoObjectiveConfig {
    pub group_size: usize,
    pub max_completion_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub full_logit_distillation: bool,
    pub alpha: f32,
    pub success_reward_threshold: f32,
    pub teacher_regularization: TeacherRegularization,
    pub teacher_update_rate: f32,
    pub distillation_topk: Option<usize>,
    pub distillation_add_tail: bool,
    pub is_clip: Option<f32>,
    pub max_reprompt_len: usize,
    pub reprompt_truncation: RepromptTruncation,
    pub dont_reprompt_on_self_success: bool,
    pub remove_thinking_from_demonstration: bool,
    pub reprompt_template: Option<String>,
    pub solution_template: Option<String>,
    pub feedback_template: Option<String>,
    pub include_environment_feedback: bool,
    pub environment_feedback_only_without_solution: bool,
}

impl Default for SdpoObjectiveConfig {
    fn default() -> Self {
        Self {
            group_size: 2,
            max_completion_tokens: 32,
            temperature: 1.0,
            top_k: None,
            full_logit_distillation: true,
            alpha: 0.5,
            success_reward_threshold: 1.0,
            teacher_regularization: TeacherRegularization::Ema,
            teacher_update_rate: 0.05,
            distillation_topk: Some(100),
            distillation_add_tail: true,
            is_clip: Some(2.0),
            max_reprompt_len: 10_240,
            reprompt_truncation: RepromptTruncation::Right,
            dont_reprompt_on_self_success: true,
            remove_thinking_from_demonstration: true,
            reprompt_template: None,
            solution_template: None,
            feedback_template: None,
            include_environment_feedback: true,
            environment_feedback_only_without_solution: true,
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
    pub module_lr_scales: Vec<ModuleLrScaleEntry>,
    #[serde(default = "default_context_strategy")]
    pub context_strategy: ContextStrategyConfig,
    #[serde(default)]
    pub sequence_kernel_override: Option<SequenceKernelConfig>,
    #[serde(default)]
    pub objective: TrainingObjectiveConfig,
    #[serde(default)]
    pub gdpo: Option<burn_dragon_train::GdpoConfig>,
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
