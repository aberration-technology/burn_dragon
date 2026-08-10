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
    /// Optional run-level override for live Ruliad curriculum feedback.
    /// `false` produces an open-loop, seed-deterministic source stream for
    /// controlled optimizer and training-algorithm comparisons.
    #[serde(default)]
    pub ruliad_source_selection_feedback_updates_enabled: Option<bool>,
    /// Optional run-level override for the Ruliad cold-start curriculum gate.
    /// Disabling it exposes every currently materialized difficulty bucket and
    /// is useful for open-loop train/holdout distribution controls.
    #[serde(default)]
    pub ruliad_source_selection_cold_start_enabled: Option<bool>,
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

/// Observation contract for recurrent-state inference.
///
/// `ObservedPrefix` uses tokens already present in the causal stream to infer a
/// detached recurrent-state teacher. Training amortizes that inference into the
/// ordinary Dragon transition, which remains the state used by later chunks and
/// by deployment. The oracle variant is retained solely to reproduce historical
/// negative controls.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum PredictiveCodingObservationContract {
    #[default]
    ObservedPrefix,
    OracleNextTokenNegativeControl,
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

/// Algorithm responsible for producing parameter derivatives or updates.
///
/// This is deliberately separate from `optimizer`: AdamW can transform either
/// globally backpropagated gradients or local predictive-coding derivatives.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum TrainingAlgorithm {
    /// Preserve existing profiles: EGGROLL selects its forward-only executor;
    /// every gradient optimizer selects global backpropagation.
    #[default]
    Auto,
    Backpropagation,
    PredictiveCoding,
    Eggroll,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum PredictiveCodingFactorReduction {
    #[default]
    Sum,
    Mean,
}

/// Terminal factor scheduled by the local predictive-coding program.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum LocalPredictiveCodingTerminalCriterion {
    #[default]
    NextToken,
    /// Replace the next-token terminal factor at proof-policy cadence with a
    /// verifier-enumerated conditional action-set factor. This is an
    /// alternating factor schedule, not an arbitrarily weighted auxiliary.
    RuliadVerifierSet,
}

/// Activity/error solver used by canonical layer-local predictive coding.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum LocalPredictiveCodingSolver {
    /// Parallel block-Jacobi activity relaxation. Credit advances through the
    /// depth graph over repeated inference rounds.
    #[default]
    SynchronousEquilibrium,
    /// Reverse block Gauss-Seidel activity relaxation. Each sweep updates
    /// activities from the terminal factor toward the clamped input, so an
    /// already-updated child error can influence every shallower activity in
    /// the same sweep while parameter learning remains factor-local.
    ReverseGaussSeidel,
    /// Augmented-Lagrangian predictive coding (PC-ALM). Activity descent is
    /// interleaved with per-factor dual ascent, yielding the local composite
    /// signal `lambda + rho * residual`. Shared Dragon weights receive the
    /// sum of all logical depth-factor derivatives after finite inference.
    AugmentedLagrangian,
    /// Error-coordinate predictive coding (ePC). Hidden activities are
    /// reconstructed as local predictions plus inferred error variables. The
    /// inference wave may transport terminal derivatives through activities,
    /// but model-parameter derivatives remain factor-local and use detached
    /// reconstructed activities.
    ErrorEquilibrium,
    /// Solve the fixed-prediction triangular error system with one reverse
    /// local-VJP wave. This is a backprop-equivalent PC control, but it never
    /// creates a global autodiff graph or calls global backward.
    FixedPrediction,
    /// Attach a supervised next-token prediction factor to every shared
    /// Dragon layer use. Activities between factors are detached, while all
    /// local readout and shared-parameter VJPs are batched over layer depth.
    /// This removes the reverse depth chain at the cost of optimizing a
    /// layer-local semi-gradient rather than the terminal-loss derivative.
    LayerLocalPrediction,
    /// Direct Kolen-Pollack predictive coding. A terminal hidden residual is
    /// projected to every depth use in one factor batch, a preliminary shared
    /// body update is projected onto Dragon's tied-parameter manifold, and the
    /// feedback bank is updated with local Kolen-Pollack correlations.
    DirectKolenPollack,
    /// Use a backend-resident feedback bank as an amortized approximation to
    /// every layer-output adjoint. Periodic exact factor-local VJP waves both
    /// anchor the parameter update and calibrate the intervening direct
    /// signals. One outer optimizer update is applied per training step.
    AmortizedAdjoint,
    /// Probe every residual factor with the terminal error in one batched
    /// local VJP, then compose the first-order residual-Jacobian corrections
    /// with an exclusive suffix sum. This removes the serial reverse-depth
    /// wave without introducing a learned feedback bank or teacher schedule.
    FirstOrderAdjoint,
}

impl LocalPredictiveCodingSolver {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SynchronousEquilibrium => "synchronous_equilibrium",
            Self::ReverseGaussSeidel => "reverse_gauss_seidel",
            Self::AugmentedLagrangian => "augmented_lagrangian",
            Self::ErrorEquilibrium => "error_equilibrium",
            Self::FixedPrediction => "fixed_prediction",
            Self::LayerLocalPrediction => "layer_local_prediction",
            Self::DirectKolenPollack => "direct_kolen_pollack",
            Self::AmortizedAdjoint => "amortized_adjoint",
            Self::FirstOrderAdjoint => "first_order_adjoint",
        }
    }
}

/// Dragon trace feature supplied to a residual-conditioned adjoint predictor.
///
/// This remains downstream-owned because the useful state summary depends on
/// the model factorization; `burn_pc` only defines how an arbitrary condition
/// is normalized and consumed.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum LocalPredictiveCodingAdjointConditioning {
    /// Residual emitted by the factor whose output adjoint is predicted.
    #[default]
    LocalResidual,
    /// Difference between this factor's output and the terminal hidden state.
    /// This summarizes the downstream residual computation responsible for
    /// transforming terminal credit before it reaches the factor.
    TerminalDisplacement,
}

/// Canonical layer-local predictive-coding learning configuration.
///
/// This is distinct from [`PredictiveCodingConfig`], which is the historical
/// recurrent-state replay auxiliary used inside global backpropagation.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct LocalPredictiveCodingConfig {
    pub solver: LocalPredictiveCodingSolver,
    pub terminal_criterion: LocalPredictiveCodingTerminalCriterion,
    /// Penalty-PC activity settings used by synchronous and Gauss-Seidel
    /// solvers. PC-ALM uses `augmented_lagrangian` instead.
    pub inference: burn_pc::PcInferenceConfig,
    /// Primal-dual finite-inference settings used only by the PC-ALM solver.
    pub augmented_lagrangian: burn_pc::PcAlmConfig,
    /// Credit carried through recurrent rho states across TBPTT chunks.
    pub temporal_credit: burn_pc::PcTemporalCreditConfig,
    pub learning_schedule: burn_pc::PcLearningSchedule,
    /// Width/depth scaling contract used by the PC factor program.
    pub parameterization: burn_pc::PcParameterizationKind,
    /// Reduction applied when one shared Dragon tensor receives derivatives
    /// from multiple recurrent depth uses under the muPC research profile.
    pub shared_reuse_reduction: burn_pc::PcSharedReuseReduction,
    /// Direct-feedback geometry shared by the DKP and amortized-adjoint
    /// solvers. The preliminary step is used only by two-phase DKP.
    pub direct_feedback: burn_pc::PcDirectFeedbackConfig,
    /// Periodic exact-local-VJP teacher for an amortized feedback bank.
    pub amortized_adjoint: burn_pc::PcAmortizedAdjointConfig,
    /// Dragon trace feature used by residual-conditioned adjoints.
    pub adjoint_conditioning: LocalPredictiveCodingAdjointConditioning,
    /// Shared-manifold projection used for DKP preliminary body updates.
    pub tied_consensus: burn_pc::PcTiedConsensusConfig,
    /// Multiplier applied to the outer learning rate for every interleaved
    /// parameter update in the incremental schedule. It is deliberately
    /// explicit: one batch performs `inference.steps` optimizer updates.
    pub incremental_parameter_step_scale: f64,
    pub prediction_precision: f32,
    pub factor_reduction: PredictiveCodingFactorReduction,
    pub sync_diagnostics: bool,
}

impl Default for LocalPredictiveCodingConfig {
    fn default() -> Self {
        Self {
            solver: LocalPredictiveCodingSolver::SynchronousEquilibrium,
            terminal_criterion: LocalPredictiveCodingTerminalCriterion::NextToken,
            inference: burn_pc::PcInferenceConfig {
                steps: 4,
                step_size: 0.05,
                latent_decay: 0.0,
                max_grad_norm: Some(1.0),
                gradient_norm_scope: burn_pc::PcGradientNormScope::PerRow,
                eps: 1.0e-8,
            },
            augmented_lagrangian: burn_pc::PcAlmConfig {
                steps: 8,
                primal_step_size: 0.03,
                dual_step_size: 0.1,
                penalty: 1.0,
                max_primal_grad_norm: Some(1.0),
                gradient_norm_scope: burn_pc::PcGradientNormScope::PerRow,
                eps: 1.0e-8,
            },
            temporal_credit: burn_pc::PcTemporalCreditConfig::default(),
            learning_schedule: burn_pc::PcLearningSchedule::Equilibrium,
            parameterization: burn_pc::PcParameterizationKind::Standard,
            shared_reuse_reduction: burn_pc::PcSharedReuseReduction::RootMeanSquare,
            direct_feedback: burn_pc::PcDirectFeedbackConfig::default(),
            amortized_adjoint: burn_pc::PcAmortizedAdjointConfig::default(),
            adjoint_conditioning: LocalPredictiveCodingAdjointConditioning::default(),
            tied_consensus: burn_pc::PcTiedConsensusConfig::default(),
            incremental_parameter_step_scale: 1.0,
            prediction_precision: 1.0,
            factor_reduction: PredictiveCodingFactorReduction::Sum,
            sync_diagnostics: false,
        }
    }
}

impl LocalPredictiveCodingConfig {
    /// Derivative boundary enforced by every production local-PC solver.
    /// Dragon's ePC implementation uses analytic activity VJPs; it does not
    /// retain an autodiff graph over either errors or model parameters.
    pub const fn execution_contract(&self) -> burn_pc::PcExecutionContract {
        burn_pc::PcExecutionContract::strict_local()
    }
}

fn default_predictive_context_probe_every_steps() -> usize {
    8
}

fn default_predictive_context_probe_tokens() -> usize {
    16
}

fn default_predictive_context_novelty_confirmations() -> u64 {
    3
}

fn default_predictive_context_active_fraction() -> f32 {
    0.25
}

/// Run-scoped causal context discovery and sparse subnetwork routing.
///
/// Contexts own optimizer moments and recurrent stream state. Model weights
/// remain one shared Dragon parameter set; deterministic sparse masks isolate
/// the selected low-rank and residual channels.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct PredictiveContextRoutingConfig {
    pub enabled: bool,
    #[serde(default = "default_predictive_context_probe_every_steps")]
    pub probe_every_steps: usize,
    #[serde(default = "default_predictive_context_probe_tokens")]
    pub probe_tokens: usize,
    #[serde(default = "default_predictive_context_novelty_confirmations")]
    pub novelty_confirmations: u64,
    #[serde(default = "default_predictive_context_active_fraction")]
    pub active_fraction: f32,
    pub bank: burn_pc::PredictiveContextBankConfig,
}

impl Default for PredictiveContextRoutingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            probe_every_steps: default_predictive_context_probe_every_steps(),
            probe_tokens: default_predictive_context_probe_tokens(),
            novelty_confirmations: default_predictive_context_novelty_confirmations(),
            active_fraction: default_predictive_context_active_fraction(),
            bank: burn_pc::PredictiveContextBankConfig {
                max_contexts: 8,
                calibration_update_rate: 0.5,
                novelty_standard_deviations: 3.0,
                ..burn_pc::PredictiveContextBankConfig::default()
            },
        }
    }
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

fn default_predictive_coding_amortization_tolerance() -> f32 {
    0.05
}

fn default_predictive_coding_amortization_max_state_slots() -> usize {
    128
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct PredictiveCodingConfig {
    pub enabled: bool,
    pub mode: PredictiveCodingMode,
    pub state_scope: PredictiveCodingStateScope,
    pub backward_mode: PredictiveCodingBackwardMode,
    pub parameter_update: PredictiveCodingParameterUpdate,
    pub observation_contract: PredictiveCodingObservationContract,
    /// Required acknowledgement for the non-causal historical ablation.
    pub allow_oracle_target_leak: bool,
    pub steps: usize,
    #[serde(default = "default_predictive_coding_step_size")]
    pub step_size: f32,
    pub latent_decay: f32,
    #[serde(default = "default_predictive_coding_max_grad_norm")]
    pub max_grad_norm: Option<f32>,
    /// Clipping geometry for independent recurrent-state corrections.
    pub gradient_norm_scope: burn_pc::PcGradientNormScope,
    #[serde(default = "default_predictive_coding_eps")]
    pub eps: f32,
    #[serde(default = "default_predictive_coding_apply_every_chunks")]
    pub apply_every_chunks: usize,
    /// Relative RMS state error tolerated before the amortization constraint activates.
    #[serde(default = "default_predictive_coding_amortization_tolerance")]
    pub amortization_tolerance: f32,
    /// Uniformly sampled recurrent slots used by the amortization constraint.
    #[serde(default = "default_predictive_coding_amortization_max_state_slots")]
    pub amortization_max_state_slots: usize,
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
            observation_contract: PredictiveCodingObservationContract::default(),
            allow_oracle_target_leak: false,
            steps: 1,
            step_size: default_predictive_coding_step_size(),
            latent_decay: 0.0,
            max_grad_norm: default_predictive_coding_max_grad_norm(),
            gradient_norm_scope: burn_pc::PcGradientNormScope::PerSample,
            eps: default_predictive_coding_eps(),
            apply_every_chunks: default_predictive_coding_apply_every_chunks(),
            amortization_tolerance: default_predictive_coding_amortization_tolerance(),
            amortization_max_state_slots: default_predictive_coding_amortization_max_state_slots(),
            warmup_steps: 0,
            sync_diagnostics: false,
        }
    }
}

impl PredictiveCodingConfig {
    pub fn inference_config(&self) -> burn_pc::PcInferenceConfig {
        burn_pc::PcInferenceConfig {
            steps: self.steps,
            step_size: self.step_size,
            latent_decay: self.latent_decay,
            max_grad_norm: self.max_grad_norm,
            gradient_norm_scope: self.gradient_norm_scope,
            eps: self.eps,
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

/// Selects where validation and checkpoint promotion are executed.
///
/// External evaluator mode keeps the trainer's epoch boundary limited to checkpoint and
/// telemetry persistence. A separate evaluator must consume those checkpoints and own all
/// validation, gating, and promotion decisions.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrainingValidationExecution {
    #[default]
    Local,
    ExternalEvaluator,
}

impl TrainingValidationExecution {
    pub fn is_local(self) -> bool {
        matches!(self, Self::Local)
    }
}

/// Selects the distribution used by the primary teacher-forced validation loss.
///
/// Fixed holdout validation is independent of the live curriculum policy, which makes validation
/// losses comparable across epochs and repeated runs. Live source selection is retained as an
/// explicit compatibility/diagnostic mode; prefer `training.events.source_weighted_validation_batches`
/// when both a stable promotion metric and current-policy telemetry are needed.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrainingValidationSampling {
    #[default]
    FixedHoldout,
    LiveSourceSelection,
}

impl TrainingValidationSampling {
    pub fn uses_live_source_selection(self) -> bool {
        matches!(self, Self::LiveSourceSelection)
    }
}

/// Selects the validation loss consumed by gates, checkpoint promotion, and
/// continual-learning dynamics.
///
/// All available validation views remain observable. This setting only names
/// the one distribution that is allowed to drive control decisions, avoiding
/// silent fallback between a fixed holdout, the live curriculum, and carried
/// recurrent state.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrainingValidationObjective {
    /// Seed-stable teacher-forced holdout selected by `validation.sampling`.
    #[default]
    FixedHoldout,
    /// Teacher-forced batches sampled from the effective live source policy.
    SourceWeighted,
    /// Ordered validation stream evaluated with recurrent state carried.
    StreamWarm,
}

impl TrainingValidationObjective {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FixedHoldout => "fixed_holdout",
            Self::SourceWeighted => "source_weighted",
            Self::StreamWarm => "stream_warm",
        }
    }
}

fn default_validation_seed() -> u64 {
    0xD12A_60A5
}

/// Persistence contract for verifier-backed Ruliad validation items.
///
/// `CreateOrReuse` publishes one immutable panel and reuses it across optimizer
/// arms. `RequireExisting` is the fail-closed promotion mode used after the
/// panel has been reviewed or generated by a matrix coordinator.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RuliadValidationPanelMode {
    #[default]
    Dynamic,
    CreateOrReuse,
    RequireExisting,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default)]
pub struct RuliadValidationPanelConfig {
    pub mode: RuliadValidationPanelMode,
    pub path: Option<PathBuf>,
    /// Number of lowest materialized difficulty strata represented in the
    /// immutable base correctness panel. `0` retains legacy unstratified
    /// sampling; positive values balance items across difficulty before
    /// cycling across source family/task contracts within each stratum.
    pub base_difficulty_levels: usize,
}

impl Default for RuliadValidationPanelConfig {
    fn default() -> Self {
        Self {
            mode: RuliadValidationPanelMode::default(),
            path: None,
            base_difficulty_levels: 4,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default)]
pub struct TrainingValidationConfig {
    pub execution: TrainingValidationExecution,
    pub sampling: TrainingValidationSampling,
    /// Single named loss contract consumed by validation-driven control logic.
    pub objective: TrainingValidationObjective,
    /// Sampling seed for the fixed teacher-forced holdout. This is deliberately independent of
    /// the training seed so training-seed ablations evaluate identical examples.
    pub seed: u64,
    /// Optional immutable verifier panel shared by paired training arms.
    pub ruliad_panel: RuliadValidationPanelConfig,
}

impl Default for TrainingValidationConfig {
    fn default() -> Self {
        Self {
            execution: TrainingValidationExecution::default(),
            sampling: TrainingValidationSampling::default(),
            objective: TrainingValidationObjective::default(),
            seed: default_validation_seed(),
            ruliad_panel: RuliadValidationPanelConfig::default(),
        }
    }
}

/// Selects how token blocks are presented to the training step independently of whether recurrent
/// state is retained between steps.
///
/// `Auto` preserves the historical behavior: persistent TBPTT uses a streaming loader and all
/// other modes use random windows. Explicit `Streaming` is useful for matched carry ablations where
/// every arm must consume the same ordered blocks even when one arm deliberately resets rho.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SequenceBatchingMode {
    #[default]
    Auto,
    Random,
    Streaming,
}

impl SequenceBatchingMode {
    pub fn uses_streaming_loader(self, persist_across_steps: bool) -> bool {
        match self {
            Self::Auto => persist_across_steps,
            Self::Random => false,
            Self::Streaming => true,
        }
    }
}

fn default_sequence_state_probe_paired_batches() -> usize {
    4
}

fn default_sequence_state_probe_max_rho_slots() -> usize {
    64
}

/// Validation-only diagnostics for the recurrent sequence state.
///
/// The paired loss probe evaluates the same continuation block with the live stream state and a
/// reset state. It never contributes gradients or changes checkpoint promotion semantics.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default)]
pub struct SequenceStateProbeConfig {
    pub enabled: bool,
    #[serde(default = "default_sequence_state_probe_paired_batches")]
    pub paired_batches: usize,
    #[serde(default = "default_sequence_state_probe_max_rho_slots")]
    pub max_rho_slots: usize,
}

/// Explicit opt-in for extending the stopping horizon of an exact resumed run.
///
/// The experiment manifest still requires every learning-semantic field to
/// match the checkpointed run. Only `training.max_iters` may increase, and the
/// validator restricts this mode to schedules that do not derive their shape
/// from that stopping horizon.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(default)]
pub struct ResumeHorizonExtensionConfig {
    pub enabled: bool,
}

impl ResumeHorizonExtensionConfig {
    pub(crate) fn is_disabled(&self) -> bool {
        !self.enabled
    }
}

impl Default for SequenceStateProbeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            paired_batches: default_sequence_state_probe_paired_batches(),
            max_rho_slots: default_sequence_state_probe_max_rho_slots(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct TrainingHyperparameters {
    #[serde(default)]
    pub algorithm: TrainingAlgorithm,
    pub block_size: usize,
    #[serde(default)]
    pub tbptt_chunk_size: Option<usize>,
    /// Number of adjacent TBPTT chunks retained in one exact autodiff window.
    ///
    /// `1` preserves conventional detached TBPTT. Larger values retain the
    /// recurrent-state graph only within each bounded window and detach at the
    /// window boundary, providing a matched control for bounded local-PC
    /// temporal credit without unbounded activation retention.
    #[serde(default = "default_tbptt_credit_window_chunks")]
    pub tbptt_credit_window_chunks: usize,
    #[serde(default)]
    pub tbptt_persist_across_steps: bool,
    #[serde(default)]
    pub sequence_batching: SequenceBatchingMode,
    /// Retain a terminal recurrent state even when the training step cannot otherwise consume it.
    /// This is primarily a compatibility and performance-ablation override.
    #[serde(default)]
    pub retain_ephemeral_terminal_sequence_state: bool,
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
    #[serde(
        default,
        skip_serializing_if = "ResumeHorizonExtensionConfig::is_disabled"
    )]
    pub resume_horizon_extension: ResumeHorizonExtensionConfig,
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
    pub local_predictive_coding: LocalPredictiveCodingConfig,
    #[serde(default)]
    pub predictive_context_routing: PredictiveContextRoutingConfig,
    #[serde(default)]
    pub latent_reasoning: LatentReasoningTrainingConfig,
    #[serde(default)]
    pub ruliad_supervision: RuliadSupervisionConfig,
    #[serde(default)]
    pub ruliad_probe_generation: RuliadProbeGenerationConfig,
    #[serde(default)]
    pub ruliad_policy_probe: RuliadPolicyProbeConfig,
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
    pub validation: TrainingValidationConfig,
    #[serde(default)]
    pub sequence_state_probe: SequenceStateProbeConfig,
    #[serde(default)]
    pub gates: burn_dragon_train::TrainingGatesConfig,
    #[serde(default)]
    pub dynamics: burn_dragon_train::train::events::DynamicsEquilibriumPolicy,
}

fn default_ruliad_probe_generation_max_batch_rows() -> usize {
    32
}

fn default_ruliad_probe_generation_minimum_batch_rows() -> usize {
    2
}

fn default_ruliad_probe_generation_maximum_prompt_position_span() -> usize {
    32
}

fn default_ruliad_probe_generation_device_buffer_tokens() -> usize {
    4
}

fn default_ruliad_probe_generation_max_in_flight_rows() -> usize {
    16
}

/// Exact free-run verifier generation policy.
///
/// Ragged rows share one absolute recurrent position: rows still in prefill consume prompt tokens
/// while completed prompts consume model argmax tokens. Small tails use the independent decoder.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default)]
pub struct RuliadProbeGenerationConfig {
    pub enabled: bool,
    #[serde(default = "default_ruliad_probe_generation_max_batch_rows")]
    pub max_batch_rows: usize,
    #[serde(default = "default_ruliad_probe_generation_minimum_batch_rows")]
    pub minimum_batch_rows: usize,
    /// Maximum prompt-length difference inside one ragged cohort. This bounds the token-at-a-time
    /// tail after the cohort's common multi-token prefill.
    #[serde(default = "default_ruliad_probe_generation_maximum_prompt_position_span")]
    pub maximum_prompt_position_span: usize,
    /// Greedy steps retained on the accelerator before resolving stop tokens on the host.
    #[serde(default = "default_ruliad_probe_generation_device_buffer_tokens")]
    pub device_buffer_tokens: usize,
    /// Maximum evaluator rows resident at once. The runtime additionally caps this at the
    /// training batch size, so validation cannot silently select a larger row batch than the
    /// already-qualified training configuration.
    #[serde(default = "default_ruliad_probe_generation_max_in_flight_rows")]
    pub max_in_flight_rows: usize,
}

impl Default for RuliadProbeGenerationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_batch_rows: default_ruliad_probe_generation_max_batch_rows(),
            minimum_batch_rows: default_ruliad_probe_generation_minimum_batch_rows(),
            maximum_prompt_position_span:
                default_ruliad_probe_generation_maximum_prompt_position_span(),
            device_buffer_tokens: default_ruliad_probe_generation_device_buffer_tokens(),
            max_in_flight_rows: default_ruliad_probe_generation_max_in_flight_rows(),
        }
    }
}

fn default_ruliad_policy_probe_items() -> usize {
    4
}

fn default_ruliad_policy_probe_max_steps() -> usize {
    256
}

fn default_ruliad_policy_probe_candidates() -> usize {
    4
}

fn default_ruliad_policy_probe_beam_width() -> usize {
    1
}

fn default_ruliad_policy_probe_scoring_batch_rows() -> usize {
    4
}

fn default_ruliad_policy_probe_scoring_token_budget() -> usize {
    32_768
}

fn default_ruliad_policy_probe_scoring_pipeline_depth() -> usize {
    2
}

fn default_ruliad_policy_probe_stratified_difficulty_levels() -> usize {
    0
}

fn default_ruliad_policy_probe_every_epochs() -> usize {
    1
}

fn default_ruliad_policy_probe_candidate_symmetry() -> RuliadProofPolicyCandidateSymmetry {
    RuliadProofPolicyCandidateSymmetry::BalancedRotation
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct RuliadPolicyProbeConfig {
    pub enabled: bool,
    /// Model contract used to rank verifier-enumerated proof actions.
    #[serde(default)]
    pub scoring: RuliadProofPolicyScoring,
    /// Probability contract used to rank autoregressive semantic actions.
    ///
    /// This should match proof-policy training. `PrefixConditional` scores only legal branching
    /// tokens in the semantic action trie; deterministic serialization is not part of the policy.
    #[serde(default)]
    pub normalization: RuliadProofPolicyNormalization,
    /// Run the same-item and counterfactual constrained-action scorers every N validation epochs.
    /// Teacher-forced and free-generation validation retain their own cadence.
    #[serde(default = "default_ruliad_policy_probe_every_epochs")]
    pub every_epochs: usize,
    /// Run the substantially more expensive verifier-backed closed-loop rollout every N
    /// validation epochs. `None` preserves the legacy behavior by using `every_epochs`.
    #[serde(default)]
    pub closed_loop_every_epochs: Option<usize>,
    #[serde(default = "default_ruliad_policy_probe_items")]
    pub items: usize,
    #[serde(default = "default_ruliad_policy_probe_max_steps")]
    pub max_steps: usize,
    #[serde(default = "default_ruliad_policy_probe_candidates")]
    pub candidates: usize,
    #[serde(default = "default_ruliad_policy_probe_beam_width")]
    pub beam_width: usize,
    /// Maximum active proof states scored in one model forward. This is an inference-only
    /// evaluator batch and does not change the training batch size.
    #[serde(default = "default_ruliad_policy_probe_scoring_batch_rows")]
    pub scoring_batch_rows: usize,
    /// Maximum padded prompt tokens in one proof-policy scoring forward. The row and token
    /// limits jointly bound evaluator memory across native and browser backends.
    #[serde(default = "default_ruliad_policy_probe_scoring_token_budget")]
    pub scoring_token_budget: usize,
    /// Maximum queued scoring forwards before resolving CUDA results. In-flight padded tokens are
    /// bounded by `scoring_pipeline_depth * scoring_token_budget`.
    #[serde(default = "default_ruliad_policy_probe_scoring_pipeline_depth")]
    pub scoring_pipeline_depth: usize,
    /// Evaluate evenly over materialized difficulty levels `[0, n)` instead of
    /// inheriting the live sampler's current distribution. Zero disables it.
    #[serde(default = "default_ruliad_policy_probe_stratified_difficulty_levels")]
    pub stratified_difficulty_levels: usize,
    /// Candidate indices are presentation details. Balanced rotation prevents a deterministic
    /// proof-menu order from becoming an evaluator shortcut while preserving action semantics.
    #[serde(default = "default_ruliad_policy_probe_candidate_symmetry")]
    pub candidate_symmetry: RuliadProofPolicyCandidateSymmetry,
    /// Capability contract used by checkpoint promotion and continual-learning recovery.
    ///
    /// This is deliberately independent from the validation loss objective. Semantic-action
    /// models can be deployed through verifier-constrained proof search even when unrestricted
    /// text generation is not their serving contract.
    #[serde(default)]
    pub checkpoint_capability_contract: RuliadCheckpointCapabilityContract,
    #[serde(default)]
    pub promotion_gate: RuliadPolicyPromotionGateConfig,
}

impl Default for RuliadPolicyProbeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            scoring: RuliadProofPolicyScoring::default(),
            normalization: RuliadProofPolicyNormalization::default(),
            every_epochs: default_ruliad_policy_probe_every_epochs(),
            closed_loop_every_epochs: None,
            items: default_ruliad_policy_probe_items(),
            max_steps: default_ruliad_policy_probe_max_steps(),
            candidates: default_ruliad_policy_probe_candidates(),
            beam_width: default_ruliad_policy_probe_beam_width(),
            scoring_batch_rows: default_ruliad_policy_probe_scoring_batch_rows(),
            scoring_token_budget: default_ruliad_policy_probe_scoring_token_budget(),
            scoring_pipeline_depth: default_ruliad_policy_probe_scoring_pipeline_depth(),
            stratified_difficulty_levels: default_ruliad_policy_probe_stratified_difficulty_levels(
            ),
            candidate_symmetry: RuliadProofPolicyCandidateSymmetry::BalancedRotation,
            checkpoint_capability_contract: RuliadCheckpointCapabilityContract::default(),
            promotion_gate: RuliadPolicyPromotionGateConfig::default(),
        }
    }
}

impl RuliadPolicyProbeConfig {
    pub fn effective_closed_loop_every_epochs(&self) -> usize {
        self.closed_loop_every_epochs.unwrap_or(self.every_epochs)
    }
}

/// Selects the deployed Ruliad capability that controls checkpoint promotion and recovery.
///
/// `FreeRunText` preserves the historical autoregressive contract. `ClosedLoopPolicy` treats
/// verifier-constrained semantic-action search as the serving contract and keeps free-run text as
/// diagnostic telemetry. `Joint` requires both contracts to remain healthy and non-regressing.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum RuliadCheckpointCapabilityContract {
    #[default]
    FreeRunText,
    ClosedLoopPolicy,
    Joint,
}

impl RuliadCheckpointCapabilityContract {
    pub const fn requires_free_run(self) -> bool {
        matches!(self, Self::FreeRunText | Self::Joint)
    }

    pub const fn requires_closed_loop_policy(self) -> bool {
        matches!(self, Self::ClosedLoopPolicy | Self::Joint)
    }
}

fn default_ruliad_policy_gate_minimum_items() -> usize {
    16
}

fn default_ruliad_policy_gate_minimum_solve_rate() -> f64 {
    0.50
}

fn default_ruliad_policy_gate_minimum_goal_completion_rate() -> f64 {
    0.80
}

fn default_ruliad_policy_gate_minimum_valid_action_rate() -> f64 {
    0.95
}

fn default_ruliad_policy_gate_maximum_invalid_action_rate() -> f64 {
    0.05
}

fn default_ruliad_policy_gate_maximum_repeated_state_rate() -> f64 {
    0.35
}

fn default_ruliad_policy_gate_maximum_backtrack_rate() -> f64 {
    0.25
}

fn default_ruliad_policy_regression_confidence_z() -> f64 {
    1.959_963_984_540_054
}

/// Closed-loop acceptance criteria for promoting a proof-action objective.
///
/// These are deliberately separate from token-level capability gates: a model
/// must keep making verifier-valid progress after its own earlier decisions.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct RuliadPolicyPromotionGateConfig {
    pub enabled: bool,
    #[serde(default = "default_ruliad_policy_gate_minimum_items")]
    pub minimum_items: usize,
    #[serde(default = "default_ruliad_policy_gate_minimum_solve_rate")]
    pub minimum_solve_rate: f64,
    #[serde(default = "default_ruliad_policy_gate_minimum_goal_completion_rate")]
    pub minimum_goal_completion_rate: f64,
    #[serde(default = "default_ruliad_policy_gate_minimum_valid_action_rate")]
    pub minimum_valid_action_rate: f64,
    #[serde(default = "default_ruliad_policy_gate_maximum_invalid_action_rate")]
    pub maximum_invalid_action_rate: f64,
    #[serde(default = "default_ruliad_policy_gate_maximum_repeated_state_rate")]
    pub maximum_repeated_state_rate: f64,
    #[serde(default = "default_ruliad_policy_gate_maximum_backtrack_rate")]
    pub maximum_backtrack_rate: f64,
    /// Normal quantile used by Wilson intervals for continual-regression evidence.
    /// The default is the conventional two-sided 95% interval. Promotion thresholds remain
    /// point-estimate gates; this value only prevents noisy finite panels from causing rollback.
    #[serde(default = "default_ruliad_policy_regression_confidence_z")]
    pub regression_confidence_z: f64,
}

impl Default for RuliadPolicyPromotionGateConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            minimum_items: default_ruliad_policy_gate_minimum_items(),
            minimum_solve_rate: default_ruliad_policy_gate_minimum_solve_rate(),
            minimum_goal_completion_rate: default_ruliad_policy_gate_minimum_goal_completion_rate(),
            minimum_valid_action_rate: default_ruliad_policy_gate_minimum_valid_action_rate(),
            maximum_invalid_action_rate: default_ruliad_policy_gate_maximum_invalid_action_rate(),
            maximum_repeated_state_rate: default_ruliad_policy_gate_maximum_repeated_state_rate(),
            maximum_backtrack_rate: default_ruliad_policy_gate_maximum_backtrack_rate(),
            regression_confidence_z: default_ruliad_policy_regression_confidence_z(),
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum RuliadSupervisionMode {
    #[default]
    FullDocument,
    AnswerWindow,
    AnswerCompletion,
    AnswerValues,
    TraceAndAnswer,
    Mixed,
}

impl RuliadSupervisionMode {
    pub fn uses_answer_target_mask(self) -> bool {
        matches!(
            self,
            Self::AnswerCompletion | Self::AnswerValues | Self::TraceAndAnswer | Self::Mixed
        )
    }

    pub fn uses_trace_answer_target_mask(self) -> bool {
        matches!(self, Self::TraceAndAnswer)
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
            Self::AnswerValues => true,
            Self::TraceAndAnswer => false,
            Self::Mixed => validation || (epoch_index.wrapping_add(absolute_step) & 1) == 0,
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct RuliadSupervisionConfig {
    pub mode: RuliadSupervisionMode,
    pub mask_high_entropy_spans: bool,
    /// Equalize aggregate trace and answer target mass from observed targets in each mixed window.
    pub balance_trace_answer_mass: bool,
    pub answer_close_marker_stride: usize,
    pub answer_close_marker_weight: i64,
    pub answer_schema_token_weight: i64,
    pub answer_schema_start_token_weight: i64,
    pub answer_value_token_weight: i64,
    pub answer_ranking: RuliadAnswerRankingConfig,
    pub answer_denoising: RuliadAnswerDenoisingConfig,
    pub answer_contract: RuliadAnswerContractConfig,
    pub verifier_reward: RuliadVerifierRewardConfig,
    pub proof_policy: RuliadProofPolicyTrainingConfig,
    /// Sparse semantic-energy replacements for primary proof-policy terminals.
    ///
    /// A refresh slot replaces, rather than adds to, the primary objective so
    /// each optimizer update retains one unit-weight training contract.
    pub proof_policy_semantic_refresh: RuliadProofPolicySemanticRefreshConfig,
}

impl Default for RuliadSupervisionConfig {
    fn default() -> Self {
        Self {
            mode: RuliadSupervisionMode::default(),
            mask_high_entropy_spans: false,
            balance_trace_answer_mass: false,
            answer_close_marker_stride: 1,
            answer_close_marker_weight: 1,
            answer_schema_token_weight: 1,
            answer_schema_start_token_weight: 1,
            answer_value_token_weight: 1,
            answer_ranking: RuliadAnswerRankingConfig::default(),
            answer_denoising: RuliadAnswerDenoisingConfig::default(),
            answer_contract: RuliadAnswerContractConfig::default(),
            verifier_reward: RuliadVerifierRewardConfig::default(),
            proof_policy: RuliadProofPolicyTrainingConfig::default(),
            proof_policy_semantic_refresh: RuliadProofPolicySemanticRefreshConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default)]
pub struct RuliadProofPolicySemanticRefreshConfig {
    pub enabled: bool,
    pub every_steps: usize,
    pub start_after_steps: usize,
    pub counterfactual_targets_per_state: usize,
}

impl Default for RuliadProofPolicySemanticRefreshConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            every_steps: 64,
            start_after_steps: 64,
            counterfactual_targets_per_state: 1,
        }
    }
}

impl RuliadProofPolicySemanticRefreshConfig {
    pub fn active_at_step(self, absolute_step: usize) -> bool {
        self.enabled
            && self.every_steps > 0
            && absolute_step >= self.start_after_steps
            && absolute_step.is_multiple_of(self.every_steps)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct RuliadPolicyBatchCadence {
    every_steps: usize,
    start_after_steps: usize,
}

impl RuliadPolicyBatchCadence {
    fn new(enabled: bool, weight: f32, every_steps: usize, start_after_steps: usize) -> Self {
        if enabled && weight > 0.0 && every_steps > 0 {
            Self {
                every_steps,
                start_after_steps,
            }
        } else {
            Self::default()
        }
    }

    fn includes(self, absolute_step: usize) -> bool {
        self.every_steps > 0
            && absolute_step >= self.start_after_steps
            && absolute_step.is_multiple_of(self.every_steps)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct RuliadPolicyBatchCadences {
    values: [RuliadPolicyBatchCadence; 7],
}

impl RuliadPolicyBatchCadences {
    pub(crate) fn enabled(self) -> bool {
        self.values.iter().any(|cadence| cadence.every_steps > 0)
    }

    pub(crate) fn includes(self, absolute_step: usize) -> bool {
        self.values
            .iter()
            .any(|cadence| cadence.includes(absolute_step))
    }
}

impl RuliadSupervisionConfig {
    pub fn proof_policy_for_step(self, absolute_step: usize) -> RuliadProofPolicyTrainingConfig {
        let mut policy = self.proof_policy;
        if self
            .proof_policy_semantic_refresh
            .active_at_step(absolute_step)
        {
            policy.scoring = RuliadProofPolicyScoring::SemanticEnergy;
            policy.normalization = RuliadProofPolicyNormalization::CandidateConditional;
            policy.counterfactual_targets_per_state = self
                .proof_policy_semantic_refresh
                .counterfactual_targets_per_state;
        }
        policy
    }

    pub(crate) fn policy_batch_cadences(self) -> RuliadPolicyBatchCadences {
        let verifier = self.verifier_reward;
        let denoising = self.answer_denoising;
        let contract = self.answer_contract;
        let proof_policy = self.proof_policy;
        RuliadPolicyBatchCadences {
            values: [
                RuliadPolicyBatchCadence::new(
                    verifier.enabled,
                    verifier.weight,
                    verifier.every_steps,
                    verifier.start_after_steps,
                ),
                RuliadPolicyBatchCadence::new(
                    verifier.enabled,
                    verifier.structured_contrast_weight,
                    verifier.structured_contrast_every_steps,
                    verifier.structured_contrast_start_after_steps,
                ),
                RuliadPolicyBatchCadence::new(
                    verifier.enabled,
                    verifier.field_binding_contrast_weight,
                    verifier.field_binding_contrast_every_steps,
                    verifier.field_binding_contrast_start_after_steps,
                ),
                RuliadPolicyBatchCadence::new(
                    verifier.enabled,
                    verifier
                        .rollout_imitation_weight
                        .max(verifier.rollout_recovery_weight),
                    verifier.rollout_imitation_every_steps,
                    verifier.rollout_imitation_start_after_steps,
                ),
                RuliadPolicyBatchCadence::new(
                    denoising.enabled,
                    denoising.structured_recovery_weight,
                    denoising.structured_recovery_every_steps,
                    denoising.structured_recovery_start_after_steps,
                ),
                RuliadPolicyBatchCadence::new(
                    contract.enabled,
                    contract.weight,
                    contract.every_steps,
                    contract.start_after_steps,
                ),
                RuliadPolicyBatchCadence::new(
                    proof_policy.enabled,
                    proof_policy.weight,
                    proof_policy.every_steps,
                    proof_policy.start_after_steps,
                ),
            ],
        }
    }

    pub fn token_supervision(
        self,
    ) -> burn_dragon_universality::ruliad::RuliadTokenSupervisionConfig {
        use burn_dragon_universality::ruliad::RuliadTokenSupervisionMode as PortableMode;

        let mode = match self.mode {
            RuliadSupervisionMode::FullDocument => PortableMode::FullDocument,
            RuliadSupervisionMode::AnswerWindow => PortableMode::AnswerWindow,
            RuliadSupervisionMode::AnswerCompletion => PortableMode::AnswerCompletion,
            RuliadSupervisionMode::AnswerValues => PortableMode::AnswerValues,
            RuliadSupervisionMode::TraceAndAnswer => PortableMode::TraceAndAnswer,
            RuliadSupervisionMode::Mixed => PortableMode::Mixed,
        };
        burn_dragon_universality::ruliad::RuliadTokenSupervisionConfig {
            mode,
            mask_high_entropy_spans: self.mask_high_entropy_spans,
            balance_trace_answer_mass: self.balance_trace_answer_mass,
            answer_close_marker_stride: self.answer_close_marker_stride,
            answer_close_marker_weight: self.answer_close_marker_weight,
            answer_schema_token_weight: self.answer_schema_token_weight,
            answer_schema_start_token_weight: self.answer_schema_start_token_weight,
            answer_value_token_weight: self.answer_value_token_weight,
        }
    }

    pub fn uses_answer_target_mask(self) -> bool {
        self.mode.uses_answer_target_mask()
    }

    pub fn uses_trace_answer_target_mask(self) -> bool {
        self.mode.uses_trace_answer_target_mask()
    }

    pub fn uses_target_loss_mask(self) -> bool {
        self.uses_answer_target_mask() || self.mask_high_entropy_spans
    }

    pub fn needs_ruliad_policy_batch(self) -> bool {
        self.policy_batch_cadences().enabled()
    }

    /// Returns whether a training batch at `absolute_step` needs the expensive formal-policy
    /// metadata sidecar. Validation never consumes this sidecar and should leave it disabled.
    pub fn needs_ruliad_policy_batch_at_step(self, absolute_step: usize) -> bool {
        self.policy_batch_cadences().includes(absolute_step)
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

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RuliadProofPolicyTrainingMode {
    /// Supervise the source-selected certificate state without a model rollout.
    StaticExpert,
    /// Roll the current model through visited states and relabel each state with the verifier.
    #[default]
    Dagger,
    /// Begin with expert states, then pair every model-visited row with a fresh expert row.
    StaticThenPairedDagger,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuliadProofPolicyEffectiveMode {
    StaticExpert,
    Dagger,
    PairedDagger,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RuliadProofPolicyScoring {
    /// Rank candidates by normalized autoregressive completion likelihood.
    #[default]
    CompletionLikelihood,
    /// Rank complete semantic actions with Dragon's task-neutral scalar sequence head.
    ///
    /// This keeps proof choice off the vocabulary projection and leaves ordinary language-model
    /// cross entropy responsible for serialization.
    SemanticEnergy,
    /// Correct the autoregressive candidate prior with a learned semantic residual energy.
    ///
    /// Candidate logits are `mean_log_p_lm + residual_energy`. Candidate-conditional cross
    /// entropy therefore fits the residual as a conditional density-ratio correction without a
    /// hand-tuned interpolation coefficient. Zero residual recovers ordinary completion scoring.
    ResidualEnergy,
}

impl RuliadProofPolicyScoring {
    pub fn uses_sequence_score_head(self) -> bool {
        matches!(self, Self::SemanticEnergy | Self::ResidualEnergy)
    }
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RuliadProofPolicyGradientScope {
    /// Allow the proof-policy objective to update both Dragon and its sequence-score head.
    #[default]
    FullModel,
    /// Stop the proof-policy gradient at Dragon's hidden representations and autoregressive
    /// candidate scores.
    ///
    /// Language-model cross entropy still updates the complete model in ordinary steps; only the
    /// semantic or residual-energy policy objective is restricted to the score head.
    ScoreHeadOnly,
    /// Stop the completion-policy gradient at Dragon's hidden representations.
    ///
    /// The verifier-equivalent completion objective updates only the untied language projection.
    /// Ordinary language-model cross entropy still updates the complete model in the same
    /// optimizer step. Tied input/output embeddings are rejected because they would make this
    /// scope modify the recurrent model's input interface.
    LanguageHeadOnly,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RuliadProofPolicyNormalization {
    /// Optimize the verifier-equivalent probability conditioned on the finite candidate set.
    #[default]
    CandidateConditional,
    /// Optimize every verifier-relevant branch in the semantic candidate trie.
    ///
    /// Each proof state has unit weight regardless of action length or trie depth, preventing
    /// late suffix likelihood from hiding an incorrect early goal/source decision.
    PrefixConditional,
    /// Optimize the full-vocabulary marginal probability of every verifier-equivalent action.
    VocabularyMarginal,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RuliadProofPolicyCandidateSymmetry {
    /// Preserve the deterministic action-menu order emitted by the proof kernel.
    #[default]
    Canonical,
    /// Apply a label-preserving rotation so each auxiliary batch presents balanced target slots.
    BalancedRotation,
    /// Average the exact cyclic presentation orbit for each proof state.
    ///
    /// Training minimizes the mean risk over every candidate rotation. Evaluation maps every
    /// rotated prediction back to the semantic action set before averaging probabilities, making
    /// the resulting decision invariant to cyclic menu position.
    CyclicOrbitAverage,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RuliadProofPolicyPresentationRisk {
    /// Minimize the expected verifier-equivalent negative log-likelihood across presentations.
    #[default]
    Mean,
    /// Minimize the maximum verifier-equivalent negative log-likelihood in each exact orbit.
    ///
    /// This finite-group distributionally robust objective prevents a strong orbit average from
    /// hiding a presentation on which the policy fails.
    Worst,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct RuliadProofPolicyTrainingConfig {
    pub enabled: bool,
    pub mode: RuliadProofPolicyTrainingMode,
    pub scoring: RuliadProofPolicyScoring,
    pub gradient_scope: RuliadProofPolicyGradientScope,
    pub normalization: RuliadProofPolicyNormalization,
    pub candidate_symmetry: RuliadProofPolicyCandidateSymmetry,
    pub presentation_risk: RuliadProofPolicyPresentationRisk,
    pub weight: f32,
    pub every_steps: usize,
    pub start_after_steps: usize,
    /// Absolute optimizer step where `static_then_paired_dagger` adds model rollouts.
    pub dagger_start_after_steps: usize,
    /// Number of materialized proof-policy difficulty levels represented in each auxiliary
    /// metadata batch. Zero preserves the live source-selected bucket only.
    pub stratified_difficulty_levels: usize,
    pub rollout_steps: usize,
    /// Semantic proof states retained across one optimizer update. DAgger distributes this budget
    /// across rollout depth so later states are drawn from model-visited trajectories.
    #[serde(alias = "max_rows_per_step")]
    pub max_rows_per_update: usize,
    /// Hard cap on tensorized presentation rows. Orbit averaging may present one semantic state
    /// several times, but must remain within this explicit compute and memory budget.
    pub max_presentation_rows_per_update: usize,
    /// Verifier-valid alternate goals paired with each sampled proof state.
    ///
    /// Counterfactual rows preserve the formal laws, current term, and candidate actions while
    /// changing only the target and its verifier-equivalent label. They prevent semantic-energy
    /// training from solving candidate ranking with a target-independent completion shortcut.
    pub counterfactual_targets_per_state: usize,
    pub candidates: usize,
    pub max_completion_tokens: usize,
}

impl Default for RuliadProofPolicyTrainingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: RuliadProofPolicyTrainingMode::Dagger,
            scoring: RuliadProofPolicyScoring::CompletionLikelihood,
            gradient_scope: RuliadProofPolicyGradientScope::FullModel,
            normalization: RuliadProofPolicyNormalization::CandidateConditional,
            candidate_symmetry: RuliadProofPolicyCandidateSymmetry::Canonical,
            presentation_risk: RuliadProofPolicyPresentationRisk::Mean,
            weight: 1.0,
            every_steps: 16,
            start_after_steps: 256,
            dagger_start_after_steps: 512,
            stratified_difficulty_levels: 0,
            rollout_steps: 8,
            max_rows_per_update: 32,
            max_presentation_rows_per_update: 32,
            counterfactual_targets_per_state: 0,
            candidates: burn_dragon_universality::ruliad::DEFAULT_PROOF_ACTION_CANDIDATES,
            max_completion_tokens: 16,
        }
    }
}

impl RuliadProofPolicyTrainingConfig {
    pub fn presentations_per_state(self) -> usize {
        match self.candidate_symmetry {
            RuliadProofPolicyCandidateSymmetry::CyclicOrbitAverage => self.candidates.max(1),
            RuliadProofPolicyCandidateSymmetry::Canonical
            | RuliadProofPolicyCandidateSymmetry::BalancedRotation => 1,
        }
    }

    pub fn target_variants_per_state(self) -> usize {
        self.counterfactual_targets_per_state.saturating_add(1)
    }

    pub fn semantic_rows_per_update(self) -> usize {
        let available = self.max_rows_per_update.min(
            self.max_presentation_rows_per_update
                .checked_div(self.presentations_per_state())
                .unwrap_or_default(),
        );
        let variants = self.target_variants_per_state();
        available.checked_div(variants).unwrap_or_default() * variants
    }

    pub fn base_semantic_rows_per_update(self) -> usize {
        self.semantic_rows_per_update()
            .checked_div(self.target_variants_per_state())
            .unwrap_or_default()
    }

    pub fn effective_mode(self, absolute_step: usize) -> RuliadProofPolicyEffectiveMode {
        match self.mode {
            RuliadProofPolicyTrainingMode::StaticThenPairedDagger
                if absolute_step < self.dagger_start_after_steps =>
            {
                RuliadProofPolicyEffectiveMode::StaticExpert
            }
            RuliadProofPolicyTrainingMode::StaticThenPairedDagger => {
                RuliadProofPolicyEffectiveMode::PairedDagger
            }
            RuliadProofPolicyTrainingMode::StaticExpert => {
                RuliadProofPolicyEffectiveMode::StaticExpert
            }
            RuliadProofPolicyTrainingMode::Dagger => RuliadProofPolicyEffectiveMode::Dagger,
        }
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
pub struct RuliadAnswerContractConfig {
    pub enabled: bool,
    pub weight: f32,
    pub premature_close_unlikelihood_weight: f32,
    pub every_steps: usize,
    pub start_after_steps: usize,
    pub max_completion_tokens: usize,
    pub max_rows_per_step: usize,
    /// Maximum schema-forced value rows per step. A value of 0 reuses
    /// `max_rows_per_step` for backwards-compatible configs.
    pub prompt_schema_max_rows_per_step: usize,
    pub schema_token_weight: f32,
    pub schema_start_token_weight: f32,
    pub value_token_weight: f32,
    pub other_token_weight: f32,
    pub prompt_schema_value_weight: f32,
}

impl Default for RuliadAnswerContractConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            weight: 0.25,
            premature_close_unlikelihood_weight: 0.0,
            every_steps: 1,
            start_after_steps: 0,
            max_completion_tokens: 64,
            max_rows_per_step: 16,
            prompt_schema_max_rows_per_step: 0,
            schema_token_weight: 2.0,
            schema_start_token_weight: 0.0,
            value_token_weight: 1.0,
            other_token_weight: 1.0,
            prompt_schema_value_weight: 0.0,
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
    pub structured_recovery_weight: f32,
    pub structured_recovery_every_steps: usize,
    pub structured_recovery_start_after_steps: usize,
    pub structured_recovery_max_completion_tokens: usize,
    pub structured_recovery_negative_count: usize,
    pub structured_recovery_template_negative_count: usize,
    pub structured_recovery_schema_negative_count: usize,
}

impl Default for RuliadAnswerDenoisingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            weight: 0.5,
            probability: 1.0,
            corrupt_offset: 1,
            structured_recovery_weight: 0.0,
            structured_recovery_every_steps: 8,
            structured_recovery_start_after_steps: 0,
            structured_recovery_max_completion_tokens: 64,
            structured_recovery_negative_count: 0,
            structured_recovery_template_negative_count: 0,
            structured_recovery_schema_negative_count: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum RuliadVerifierRewardMode {
    #[default]
    Scalar,
    VpoIndependent,
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
    pub start_after_steps: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub kl_weight: f32,
    pub clip_range: f32,
    pub max_advantage_clip_fraction: Option<f32>,
    pub positive_advantage_requires_correctness: bool,
    pub positive_advantage_min_partial_progress_ppm: usize,
    pub positive_advantage_min_completion_quality_ppm: usize,
    pub advantage_epsilon: f32,
    pub vpo_scalarizations: usize,
    pub vpo_correctness_mass_floor: f32,
    pub vpo_schema_quality_mass_floor: f32,
    pub vpo_completion_health_mass_floor: f32,
    pub vpo_compactness_max_weight: f32,
    pub include_oracle_candidate: bool,
    pub include_structured_negative_candidates: bool,
    pub structured_negative_count: usize,
    pub structured_template_negative_count: usize,
    pub structured_schema_negative_count: usize,
    pub structured_contrast_weight: f32,
    pub structured_contrast_every_steps: usize,
    pub structured_contrast_start_after_steps: usize,
    pub structured_contrast_margin: f32,
    pub field_binding_contrast_weight: f32,
    pub field_binding_contrast_every_steps: usize,
    pub field_binding_contrast_start_after_steps: usize,
    pub field_binding_contrast_margin: f32,
    pub field_binding_contrast_pair_weight: f32,
    pub field_binding_contrast_max_pairs: usize,
    pub field_binding_contrast_rank_metric_every_steps: usize,
    pub field_binding_contrast_replay_capacity: usize,
    pub generated_attractor_replay_capacity: usize,
    pub generated_attractor_replay_min_count: usize,
    pub generated_attractor_replay_max_candidates: usize,
    pub generated_attractor_replay_min_distinct_answers: usize,
    pub generated_attractor_replay_max_dominant_fraction: f32,
    pub rollout_imitation_weight: f32,
    pub rollout_imitation_every_steps: usize,
    pub rollout_imitation_start_after_steps: usize,
    pub rollout_imitation_min_partial_progress_ppm: usize,
    pub rollout_imitation_min_completion_quality_ppm: usize,
    pub rollout_imitation_min_verifier_rate_ppm: usize,
    pub rollout_imitation_max_schema_wrong_rate_ppm: usize,
    pub rollout_imitation_max_malformed_rate_ppm: usize,
    pub rollout_imitation_max_rows_per_step: usize,
    pub rollout_recovery_weight: f32,
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
            start_after_steps: 0,
            temperature: 0.8,
            top_k: 64,
            kl_weight: 0.01,
            clip_range: 0.2,
            max_advantage_clip_fraction: None,
            positive_advantage_requires_correctness: false,
            positive_advantage_min_partial_progress_ppm: 0,
            positive_advantage_min_completion_quality_ppm: 0,
            advantage_epsilon: 1.0e-6,
            vpo_scalarizations: 16,
            vpo_correctness_mass_floor: 0.70,
            vpo_schema_quality_mass_floor: 0.10,
            vpo_completion_health_mass_floor: 0.10,
            vpo_compactness_max_weight: 0.05,
            include_oracle_candidate: false,
            include_structured_negative_candidates: false,
            structured_negative_count: 2,
            structured_template_negative_count: 0,
            structured_schema_negative_count: 0,
            structured_contrast_weight: 0.0,
            structured_contrast_every_steps: 8,
            structured_contrast_start_after_steps: 0,
            structured_contrast_margin: 0.25,
            field_binding_contrast_weight: 0.0,
            field_binding_contrast_every_steps: 8,
            field_binding_contrast_start_after_steps: 0,
            field_binding_contrast_margin: 0.25,
            field_binding_contrast_pair_weight: 0.0,
            field_binding_contrast_max_pairs: 16,
            field_binding_contrast_rank_metric_every_steps: 8,
            field_binding_contrast_replay_capacity: 0,
            generated_attractor_replay_capacity: 0,
            generated_attractor_replay_min_count: 2,
            generated_attractor_replay_max_candidates: 4,
            generated_attractor_replay_min_distinct_answers: 2,
            generated_attractor_replay_max_dominant_fraction: 0.5,
            rollout_imitation_weight: 0.0,
            rollout_imitation_every_steps: 16,
            rollout_imitation_start_after_steps: 0,
            rollout_imitation_min_partial_progress_ppm: 500_000,
            rollout_imitation_min_completion_quality_ppm: 750_000,
            rollout_imitation_min_verifier_rate_ppm: 100_000,
            rollout_imitation_max_schema_wrong_rate_ppm: 250_000,
            rollout_imitation_max_malformed_rate_ppm: 250_000,
            rollout_imitation_max_rows_per_step: 16,
            rollout_recovery_weight: 0.0,
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

fn default_tbptt_credit_window_chunks() -> usize {
    1
}

fn default_gradient_accumulation_steps() -> usize {
    1
}

fn default_checkpoint_interval_iters() -> usize {
    2_000
}
