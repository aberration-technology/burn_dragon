mod attention;
mod attention_residual;
mod config;
mod dragon;
mod dragon_support;
mod halt;
mod init;
mod mhc;
mod micro_transformer;
mod norm;
mod residual_stream;
mod scaffold;
mod sequence;
mod state;
mod widen;

pub use attention_residual::{
    AttentionResidual, AttentionResidualConfig, BlockAttentionResidual,
    BlockAttentionResidualConfig, BlockAttentionResidualSummaryMode, ResidualConnectorKind,
};
pub use burn_dragon_kernel::api::projection::LowrankGradInputExecutor;
pub use config::{
    ClockedSlowMemoryConfig, DragonConfig, DragonRandomScaffoldConfig, FusedAttentionExecutor,
    FusedKernelConfig, FusedProjectionExecutor, HierarchicalDragonConfig,
    HierarchicalDragonSharing, LanguageHeadConfig, LatentFanoutScheduleConfig,
    LatentReasoningConfig, NextLatentTransitionConfig, SummaryMemoryConfig,
    YNeuronRecurrenceConfig,
};
pub use dragon::{
    DragonModel, LanguageModuleLrScaleTarget, LatentReasoningOutput,
    SharedLowrankActivationBatchStats, SharedLowrankContinualBackpropRuntime,
    SharedLowrankFeatureMetrics, SharedLowrankParamIds, SharedLowrankPopulationFactors,
    SharedLowrankPopulationWeights, SharedLowrankWeights,
};
#[cfg(any(feature = "probe", test))]
pub use dragon::{
    HeadTensorComparisonDiagnostics, HeadTensorGeometryDiagnostics,
    LanguageLayerStateDeltaDiagnostics, LanguageLayerStateSummaryDiagnostics,
    LanguageLowRankLayerComparisonDiagnostics, LanguageLowRankLayerGeometryDiagnostics,
    TensorComparisonDiagnostics, TensorDistributionDiagnostics, TensorStateDeltaDiagnostics,
    TensorStateSummaryDiagnostics, compare_model_states, summarize_model_state,
};
#[cfg(any(feature = "probe", test))]
pub use dragon_support::LanguageDragonInitLayerDiagnostics;
pub use dragon_support::{
    LanguageMhcLayerDiagnostics, LanguagePipelineState, LogitsProjectionProfileSnapshot,
    logits_projection_profile_reset, logits_projection_profile_snapshot,
};
pub use halt::HaltHead;
pub use init::{
    DragonActivationThresholds, DragonFiringTargetConfig, DragonFiringTargetKind,
    DragonInitializationConfig, DragonInitializationKind, DragonInitializer,
    DragonNeuronGainConfig, DragonNeuronGainKind, DragonProjectionRole,
    DragonReservoirInitializationConfig, DragonResidualScalingConfig, DragonResidualScalingKind,
    DragonTopologyPriorConfig, DragonTopologyPriorKind, near_critical_embedding_initializer,
    near_critical_embedding_std, near_critical_projection_std, near_critical_residual_output_std,
};
pub use mhc::{
    ManifoldHyperConnectionCoefficientPolicy, ManifoldHyperConnectionCoefficients,
    ManifoldHyperConnectionStreamCoefficients, ManifoldHyperConnectionStreamOutput,
    ManifoldHyperConnectionWidthOutput, ManifoldHyperConnections, ManifoldHyperConnectionsConfig,
    mhc_merge, mhc_merge_with_coefficients, mhc_passthrough, mhc_passthrough_with_coefficients,
    mhc_split, mhc_split_with_coefficients,
};
pub use micro_transformer::MicroTransformerBlock;
pub use norm::{DragonNorm, DragonNormConfig, DragonNormKind};
pub use residual_stream::{
    LowRankResidualMemoryProfileSnapshot, LowRankResidualMemoryStageSnapshot,
    LowRankResidualOutput, LowRankResidualProfileSnapshot, lowrank_residual_memory_profile_reset,
    lowrank_residual_memory_profile_snapshot, lowrank_residual_profile_reset,
    lowrank_residual_profile_snapshot, lowrank_residual_step, lowrank_residual_step_next,
};
pub use scaffold::{DragonRandomScaffoldReport, build_dragon_random_scaffold_manifest};
pub use sequence::{
    GatedDeltaNet2Config, GatedDeltaNet2GateMode, GatedDeltaNet2Implementation,
    GatedDeltaNet2StatePrecision, MambaSequenceConfig, SequenceKernelConfig, SequenceMemorySystem,
    SequenceTrainingExecutor, gated_deltanet2_reference,
};
#[cfg(any(feature = "viz", feature = "probe"))]
pub use state::LayerVizState;
pub use state::{LayerState, ModelState};
