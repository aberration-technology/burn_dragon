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

fn default_auto_batch_min_batch_size() -> usize {
    1
}

fn default_auto_batch_probe_steps() -> usize {
    1
}

fn default_auto_batch_binary_search() -> bool {
    true
}

fn default_auto_batch_recompute_on_neuron_scale() -> bool {
    true
}

fn default_auto_batch_scale_memory_exponent() -> f32 {
    1.0
}

fn default_auto_batch_max_system_memory_fraction() -> f32 {
    0.9
}

fn default_auto_batch_probe_safety_margin() -> f32 {
    1.15
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct AutoBatchSizeConfig {
    pub enabled: bool,
    /// Preferred upper bound for automatic selection. Set to 32 for min(32, fit_in_memory).
    pub max_batch_size: Option<usize>,
    /// Optional cap on actual startup probe batch size. Larger candidates use conservative
    /// prediction from lower-batch probes to avoid probe-only memory spikes.
    #[serde(default)]
    pub max_probe_batch_size: Option<usize>,
    #[serde(default = "default_auto_batch_min_batch_size")]
    pub min_batch_size: usize,
    /// Hard memory target in MiB. A value of 0 disables the target and only rejects failed probes.
    pub target_device_memory_mb: usize,
    #[serde(default = "default_auto_batch_probe_steps")]
    pub probe_steps: usize,
    #[serde(default = "default_auto_batch_binary_search")]
    pub binary_search: bool,
    #[serde(default = "default_auto_batch_recompute_on_neuron_scale")]
    pub recompute_on_neuron_scale: bool,
    /// Conservative post-scale batch estimate: batch scales by (old_capacity / new_capacity)^x.
    #[serde(default = "default_auto_batch_scale_memory_exponent")]
    pub scale_memory_exponent: f32,
    /// Hard host-memory cap for unified-memory systems, expressed as a fraction of MemTotal.
    #[serde(default = "default_auto_batch_max_system_memory_fraction")]
    pub max_system_memory_fraction: f32,
    /// Prediction margin applied before probing larger candidates.
    #[serde(default = "default_auto_batch_probe_safety_margin")]
    pub probe_safety_margin: f32,
}

impl Default for AutoBatchSizeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_batch_size: None,
            max_probe_batch_size: None,
            min_batch_size: default_auto_batch_min_batch_size(),
            target_device_memory_mb: 0,
            probe_steps: default_auto_batch_probe_steps(),
            binary_search: default_auto_batch_binary_search(),
            recompute_on_neuron_scale: default_auto_batch_recompute_on_neuron_scale(),
            scale_memory_exponent: default_auto_batch_scale_memory_exponent(),
            max_system_memory_fraction: default_auto_batch_max_system_memory_fraction(),
            probe_safety_margin: default_auto_batch_probe_safety_margin(),
        }
    }
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
#[serde(default)]
pub struct CausalInputCorruptionConfig {
    pub enabled: bool,
    pub probability: f32,
    pub warmup_steps: usize,
    pub ramp_steps: usize,
    pub replacement_token_id: Option<u32>,
}

impl Default for CausalInputCorruptionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            probability: 0.0,
            warmup_steps: 0,
            ramp_steps: 0,
            replacement_token_id: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct LogitEntropyFloorConfig {
    pub enabled: bool,
    pub weight: f32,
    pub target_entropy_bits: f32,
    pub marginal_weight: f32,
    pub target_marginal_entropy_bits: f32,
    pub target_coverage_weight: f32,
    pub target_coverage_epsilon: f32,
    pub warmup_steps: usize,
    pub ramp_steps: usize,
    pub every_steps: usize,
}

impl Default for LogitEntropyFloorConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            weight: 0.0,
            target_entropy_bits: 0.0,
            marginal_weight: 0.0,
            target_marginal_entropy_bits: 0.0,
            target_coverage_weight: 0.0,
            target_coverage_epsilon: 1.0e-8,
            warmup_steps: 0,
            ramp_steps: 0,
            every_steps: 1,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct RepeatUnlikelihoodConfig {
    pub enabled: bool,
    pub weight: f32,
    pub cycle_weight: f32,
    pub cycle_margin_weight: f32,
    pub cycle_margin: f32,
    pub cycle_min_lag: usize,
    pub cycle_max_lag: usize,
    pub cycle_lags_per_step: usize,
    pub warmup_steps: usize,
    pub ramp_steps: usize,
    pub every_steps: usize,
    #[serde(default)]
    pub history_lags: Vec<usize>,
    pub epsilon: f32,
}

impl Default for RepeatUnlikelihoodConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            weight: 0.0,
            cycle_weight: 0.0,
            cycle_margin_weight: 0.0,
            cycle_margin: 0.0,
            cycle_min_lag: 2,
            cycle_max_lag: 64,
            cycle_lags_per_step: 8,
            warmup_steps: 0,
            ramp_steps: 0,
            every_steps: 1,
            history_lags: Vec::new(),
            epsilon: 1.0e-4,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct GreedyRolloutUnlikelihoodConfig {
    pub enabled: bool,
    /// Run the expensive autoregressive rollout auxiliary only while dynamics is in a recovery
    /// mode. This keeps stable training on the vectorized hot path while preserving stronger
    /// anti-collapse pressure when the monitor detects output degeneracy.
    pub recovery_only: bool,
    pub weight: f32,
    pub margin_weight: f32,
    pub margin: f32,
    pub recovery_weight: f32,
    pub sequence_recovery_weight: f32,
    pub entropy_floor_weight: f32,
    pub target_entropy_bits: f32,
    pub cycle_weight: f32,
    pub cycle_margin_weight: f32,
    pub cycle_min_lag: usize,
    pub cycle_max_lag: usize,
    pub warmup_steps: usize,
    pub ramp_steps: usize,
    pub every_steps: usize,
    pub prompt_tokens: usize,
    pub rollout_tokens: usize,
    pub history_tokens: usize,
    pub batch_prompts: usize,
    pub epsilon: f32,
}

impl Default for GreedyRolloutUnlikelihoodConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            recovery_only: false,
            weight: 0.0,
            margin_weight: 0.0,
            margin: 0.0,
            recovery_weight: 0.0,
            sequence_recovery_weight: 0.0,
            entropy_floor_weight: 0.0,
            target_entropy_bits: 0.0,
            cycle_weight: 0.0,
            cycle_margin_weight: 0.0,
            cycle_min_lag: 2,
            cycle_max_lag: 64,
            warmup_steps: 0,
            ramp_steps: 0,
            every_steps: 128,
            prompt_tokens: 32,
            rollout_tokens: 8,
            history_tokens: 8,
            batch_prompts: 1,
            epsilon: 1.0e-4,
        }
    }
}

fn default_dynamics_anchor_teacher_update_rate() -> f32 {
    0.01
}

fn default_dynamics_anchor_every_steps() -> usize {
    1
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum DynamicsAnchorMask {
    #[default]
    AllTokens,
    ContextTokens,
    TargetTokens,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct DynamicsAnchorConfig {
    pub enabled: bool,
    pub weight: f32,
    #[serde(default = "default_dynamics_anchor_teacher_update_rate")]
    pub teacher_update_rate: f32,
    pub kl: SelfDistillationKlKind,
    pub mask: DynamicsAnchorMask,
    pub warmup_steps: usize,
    pub ramp_steps: usize,
    #[serde(default = "default_dynamics_anchor_every_steps")]
    pub every_steps: usize,
}

impl Default for DynamicsAnchorConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            weight: 0.0,
            teacher_update_rate: default_dynamics_anchor_teacher_update_rate(),
            kl: SelfDistillationKlKind::JensenShannon,
            mask: DynamicsAnchorMask::default(),
            warmup_steps: 0,
            ramp_steps: 0,
            every_steps: default_dynamics_anchor_every_steps(),
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum PredictiveCodingMode {
    #[default]
    RecurrentState,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum PredictiveCodingStateScope {
    #[default]
    Core,
    All,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum PredictiveCodingBackwardMode {
    #[default]
    Chunked,
    Block,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum PredictiveCodingParameterUpdate {
    #[default]
    Optimizer,
    #[serde(alias = "frozen")]
    StateOnlyControl,
}

fn default_predictive_coding_step_size() -> f32 {
    0.03
}

fn default_predictive_coding_max_grad_norm() -> Option<f32> {
    Some(1.0)
}

fn default_predictive_coding_eps() -> f32 {
    1.0e-8
}

fn default_predictive_coding_apply_every_chunks() -> usize {
    1
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct PredictiveCodingConfig {
    pub enabled: bool,
    pub mode: PredictiveCodingMode,
    pub state_scope: PredictiveCodingStateScope,
    pub backward_mode: PredictiveCodingBackwardMode,
    pub parameter_update: PredictiveCodingParameterUpdate,
    pub steps: usize,
    #[serde(default = "default_predictive_coding_step_size")]
    pub step_size: f32,
    pub latent_decay: f32,
    #[serde(default = "default_predictive_coding_max_grad_norm")]
    pub max_grad_norm: Option<f32>,
    #[serde(default = "default_predictive_coding_eps")]
    pub eps: f32,
    #[serde(default = "default_predictive_coding_apply_every_chunks")]
    pub apply_every_chunks: usize,
    pub warmup_steps: usize,
    pub sync_diagnostics: bool,
}

impl Default for PredictiveCodingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: PredictiveCodingMode::default(),
            state_scope: PredictiveCodingStateScope::default(),
            backward_mode: PredictiveCodingBackwardMode::default(),
            parameter_update: PredictiveCodingParameterUpdate::default(),
            steps: 1,
            step_size: default_predictive_coding_step_size(),
            latent_decay: 0.0,
            max_grad_norm: default_predictive_coding_max_grad_norm(),
            eps: default_predictive_coding_eps(),
            apply_every_chunks: default_predictive_coding_apply_every_chunks(),
            warmup_steps: 0,
            sync_diagnostics: false,
        }
    }
}

fn default_latent_reasoning_future_offsets() -> Vec<usize> {
    vec![1, 2, 4, 8]
}

fn default_latent_reasoning_teacher_update_rate() -> f32 {
    0.01
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum LatentReasoningTargetEncoder {
    EmaTeacher,
    #[default]
    DetachedStudent,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum LatentReasoningAuxiliaryStartPolicy {
    #[default]
    FixedStep,
    CapabilityGate,
    FixedStepAndCapabilityGate,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum LatentReasoningNegativeSource {
    #[default]
    InBatchAndCorruptAnswer,
    TemporalShift,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct LatentReasoningSigRegConfig {
    pub enabled: bool,
    #[serde(default)]
    pub every_steps: Option<usize>,
    #[serde(default)]
    pub start_after_steps: Option<usize>,
    #[serde(default)]
    pub start_policy: Option<LatentReasoningAuxiliaryStartPolicy>,
    pub mode: LatentReasoningSigRegMode,
    pub target: LatentReasoningSigRegTarget,
    pub target_variance: f32,
    pub min_variance: f32,
    pub mean_tolerance: f32,
    pub max_rho_slots: usize,
}

impl Default for LatentReasoningSigRegConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            every_steps: None,
            start_after_steps: None,
            start_policy: None,
            mode: LatentReasoningSigRegMode::default(),
            target: LatentReasoningSigRegTarget::default(),
            target_variance: 1.0,
            min_variance: 0.2,
            mean_tolerance: 0.05,
            max_rho_slots: 128,
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum LatentReasoningSigRegMode {
    #[default]
    WeakCovariance,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum LatentReasoningSigRegTarget {
    #[default]
    Hidden,
    RhoMemorySlots,
    HiddenAndRhoMemorySlots,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct LatentReasoningConstraintBalancerConfig {
    pub enabled: bool,
    pub normalized_aux_scale: f32,
    pub start_after_steps: usize,
    pub warmup_steps: usize,
    pub stop_target_mean_steps: f32,
    pub stop_tolerance_steps: f32,
}

impl Default for LatentReasoningConstraintBalancerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            normalized_aux_scale: 1.0,
            start_after_steps: 0,
            warmup_steps: 0,
            stop_target_mean_steps: 2.0,
            stop_tolerance_steps: 0.5,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct NextLatentPredictionConfig {
    pub enabled: bool,
    #[serde(default)]
    pub every_steps: Option<usize>,
    #[serde(default)]
    pub start_after_steps: Option<usize>,
    #[serde(default)]
    pub start_policy: Option<LatentReasoningAuxiliaryStartPolicy>,
    pub horizon: usize,
    pub regression_weight: f32,
    pub token_kl_weight: f32,
    pub smooth_l1_beta: f32,
    pub detach_action_embedding: bool,
}

impl Default for NextLatentPredictionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            every_steps: None,
            start_after_steps: None,
            start_policy: None,
            horizon: 1,
            regression_weight: 1.0,
            token_kl_weight: 0.0,
            smooth_l1_beta: 1.0,
            detach_action_embedding: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct DragonStateConsistencyConfig {
    pub enabled: bool,
    #[serde(default)]
    pub every_steps: Option<usize>,
    #[serde(default)]
    pub start_after_steps: Option<usize>,
    #[serde(default)]
    pub start_policy: Option<LatentReasoningAuxiliaryStartPolicy>,
    pub rho_weight: f32,
    pub rho_energy_weight: f32,
    pub smooth_l1_beta: f32,
    pub max_rho_slots: usize,
}

impl Default for DragonStateConsistencyConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            every_steps: None,
            start_after_steps: None,
            start_policy: None,
            rho_weight: 1.0,
            rho_energy_weight: 0.25,
            smooth_l1_beta: 1.0,
            max_rho_slots: 64,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct LatentEnergyModelConfig {
    pub enabled: bool,
    #[serde(default)]
    pub every_steps: Option<usize>,
    #[serde(default)]
    pub start_after_steps: Option<usize>,
    #[serde(default)]
    pub start_policy: Option<LatentReasoningAuxiliaryStartPolicy>,
    pub contrastive_weight: f32,
    pub monotonic_weight: f32,
    pub contractive_weight: f32,
    pub margin: f32,
    pub monotonic_tolerance: f32,
    pub trust_radius: f32,
    pub max_rollout_steps_for_loss: usize,
}

impl Default for LatentEnergyModelConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            every_steps: None,
            start_after_steps: None,
            start_policy: None,
            contrastive_weight: 1.0,
            monotonic_weight: 0.25,
            contractive_weight: 0.05,
            margin: 1.0,
            monotonic_tolerance: 0.0,
            trust_radius: 1.0,
            max_rollout_steps_for_loss: 4,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct LatentStepContractConfig {
    pub enabled: bool,
    #[serde(default)]
    pub every_steps: Option<usize>,
    #[serde(default)]
    pub start_after_steps: Option<usize>,
    #[serde(default)]
    pub start_policy: Option<LatentReasoningAuxiliaryStartPolicy>,
    pub max_rollout_steps_for_loss: usize,
    pub ce_weight: f32,
    pub token_kl_weight: f32,
    pub monotonic_ce_weight: f32,
    pub contractive_weight: f32,
    pub ce_tolerance: f32,
    pub trust_radius: f32,
}

impl Default for LatentStepContractConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            every_steps: None,
            start_after_steps: None,
            start_policy: None,
            max_rollout_steps_for_loss: 4,
            ce_weight: 0.0,
            token_kl_weight: 0.0,
            monotonic_ce_weight: 1.0,
            contractive_weight: 0.05,
            ce_tolerance: 0.0,
            trust_radius: 1.0,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct LatentReasoningTrainingConfig {
    pub enabled: bool,
    pub every_steps: usize,
    #[serde(default)]
    pub start_after_capability_gate_passed: bool,
    #[serde(default)]
    pub eval_step_sweep: Vec<usize>,
    #[serde(default)]
    pub jepa_every_steps: Option<usize>,
    #[serde(default)]
    pub jepa_start_after_steps: Option<usize>,
    #[serde(default)]
    pub jepa_start_policy: Option<LatentReasoningAuxiliaryStartPolicy>,
    #[serde(default = "default_latent_reasoning_future_offsets")]
    pub jepa_future_offsets: Vec<usize>,
    pub target_encoder: LatentReasoningTargetEncoder,
    #[serde(default = "default_latent_reasoning_teacher_update_rate")]
    pub teacher_update_rate: f32,
    pub negative_source: LatentReasoningNegativeSource,
    pub next_latent: NextLatentPredictionConfig,
    pub dragon_state: DragonStateConsistencyConfig,
    pub energy_model: LatentEnergyModelConfig,
    pub step_contract: LatentStepContractConfig,
    pub sigreg: LatentReasoningSigRegConfig,
    pub constraint_balancer: LatentReasoningConstraintBalancerConfig,
}

impl Default for LatentReasoningTrainingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            every_steps: 1,
            start_after_capability_gate_passed: false,
            eval_step_sweep: Vec::new(),
            jepa_every_steps: None,
            jepa_start_after_steps: None,
            jepa_start_policy: None,
            jepa_future_offsets: default_latent_reasoning_future_offsets(),
            target_encoder: LatentReasoningTargetEncoder::default(),
            teacher_update_rate: default_latent_reasoning_teacher_update_rate(),
            negative_source: LatentReasoningNegativeSource::default(),
            next_latent: NextLatentPredictionConfig::default(),
            dragon_state: DragonStateConsistencyConfig::default(),
            energy_model: LatentEnergyModelConfig::default(),
            step_contract: LatentStepContractConfig::default(),
            sigreg: LatentReasoningSigRegConfig::default(),
            constraint_balancer: LatentReasoningConstraintBalancerConfig::default(),
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
    pub source_selection_state_path: Option<PathBuf>,
    #[serde(default)]
    pub init_transfer: InitTransferConfig,
    #[serde(default)]
    pub continual_backprop: ContinualBackpropConfig,
    #[serde(default)]
    pub neuron_scaling: NeuronScalingConfig,
    #[serde(default)]
    pub auto_batch_size: AutoBatchSizeConfig,
    #[serde(default)]
    pub input_corruption: CausalInputCorruptionConfig,
    #[serde(default)]
    pub logit_entropy_floor: LogitEntropyFloorConfig,
    #[serde(default)]
    pub repeat_unlikelihood: RepeatUnlikelihoodConfig,
    #[serde(default)]
    pub greedy_rollout_unlikelihood: GreedyRolloutUnlikelihoodConfig,
    #[serde(default)]
    pub dynamics_anchor: DynamicsAnchorConfig,
    #[serde(default)]
    pub predictive_coding: PredictiveCodingConfig,
    #[serde(default)]
    pub latent_reasoning: LatentReasoningTrainingConfig,
    #[serde(default)]
    pub ruliad_supervision: RuliadSupervisionConfig,
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
    #[serde(default)]
    pub dynamics: burn_dragon_train::train::events::DynamicsEquilibriumPolicy,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum RuliadSupervisionMode {
    #[default]
    FullDocument,
    AnswerWindow,
    AnswerCompletion,
    Mixed,
}

impl RuliadSupervisionMode {
    pub fn uses_answer_target_mask(self) -> bool {
        matches!(self, Self::AnswerCompletion | Self::Mixed)
    }

    pub fn prefer_answer_window(
        self,
        validation: bool,
        epoch_index: usize,
        absolute_step: usize,
    ) -> bool {
        match self {
            Self::FullDocument => false,
            Self::AnswerWindow => true,
            Self::AnswerCompletion => true,
            Self::Mixed => validation || (epoch_index.wrapping_add(absolute_step) & 1) == 0,
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct RuliadSupervisionConfig {
    pub mode: RuliadSupervisionMode,
    pub mask_high_entropy_spans: bool,
    pub answer_close_marker_stride: usize,
    pub answer_ranking: RuliadAnswerRankingConfig,
    pub answer_denoising: RuliadAnswerDenoisingConfig,
    pub verifier_reward: RuliadVerifierRewardConfig,
}

impl Default for RuliadSupervisionConfig {
    fn default() -> Self {
        Self {
            mode: RuliadSupervisionMode::default(),
            mask_high_entropy_spans: false,
            answer_close_marker_stride: 1,
            answer_ranking: RuliadAnswerRankingConfig::default(),
            answer_denoising: RuliadAnswerDenoisingConfig::default(),
            verifier_reward: RuliadVerifierRewardConfig::default(),
        }
    }
}

impl RuliadSupervisionConfig {
    pub fn uses_answer_target_mask(self) -> bool {
        self.mode.uses_answer_target_mask()
    }

    pub fn uses_target_loss_mask(self) -> bool {
        self.uses_answer_target_mask() || self.mask_high_entropy_spans
    }

    pub fn prefer_answer_window(
        self,
        validation: bool,
        epoch_index: usize,
        absolute_step: usize,
    ) -> bool {
        self.mode
            .prefer_answer_window(validation, epoch_index, absolute_step)
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct RuliadAnswerRankingConfig {
    pub enabled: bool,
    pub weight: f32,
    pub margin: f32,
    pub corrupt_offset: i64,
}

impl Default for RuliadAnswerRankingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            weight: 0.25,
            margin: 0.5,
            corrupt_offset: 1,
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct RuliadAnswerDenoisingConfig {
    pub enabled: bool,
    pub weight: f32,
    pub probability: f32,
    pub corrupt_offset: i64,
}

impl Default for RuliadAnswerDenoisingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            weight: 0.5,
            probability: 1.0,
            corrupt_offset: 1,
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RuliadVerifierRewardMode {
    Scalar,
    VpoIndependent,
}

impl Default for RuliadVerifierRewardMode {
    fn default() -> Self {
        Self::Scalar
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct RuliadVerifierRewardConfig {
    pub enabled: bool,
    pub mode: RuliadVerifierRewardMode,
    pub weight: f32,
    pub group_size: usize,
    pub max_completion_tokens: usize,
    pub every_steps: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub kl_weight: f32,
    pub clip_range: f32,
    pub advantage_epsilon: f32,
    pub vpo_scalarizations: usize,
    pub vpo_correctness_mass_floor: f32,
    pub vpo_completion_health_mass_floor: f32,
    pub vpo_compactness_max_weight: f32,
    pub reward: burn_dragon_universality::RuliadVerifierRewardWeights,
}

impl Default for RuliadVerifierRewardConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: RuliadVerifierRewardMode::Scalar,
            weight: 0.05,
            group_size: 4,
            max_completion_tokens: 64,
            every_steps: 16,
            temperature: 0.8,
            top_k: 64,
            kl_weight: 0.01,
            clip_range: 0.2,
            advantage_epsilon: 1.0e-6,
            vpo_scalarizations: 16,
            vpo_correctness_mass_floor: 0.70,
            vpo_completion_health_mass_floor: 0.10,
            vpo_compactness_max_weight: 0.05,
            reward: burn_dragon_universality::RuliadVerifierRewardWeights::default(),
        }
    }
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
