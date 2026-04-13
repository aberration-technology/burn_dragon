#![recursion_limit = "256"]

//! Shared Dragon core math and state contracts for the focused language-training repo.

pub mod constants;
pub mod experimental;
pub mod kernel;
pub mod model;
pub mod positional;

pub use model::LanguageModuleLrScaleTarget;

pub mod api {
    pub mod config {
        pub use crate::model::LanguageModuleLrScaleTarget;
        pub use crate::{
            AttentionResidualConfig, BlockAttentionResidualConfig,
            BlockAttentionResidualSummaryMode, ClockedSlowMemoryConfig, DragonConfig,
            DragonFiringTargetConfig, DragonFiringTargetKind, DragonInitializationConfig,
            DragonInitializationKind, DragonNeuronGainConfig, DragonNeuronGainKind,
            DragonNormConfig, DragonNormKind, DragonResidualScalingConfig,
            DragonResidualScalingKind, DragonTopologyPriorConfig, DragonTopologyPriorKind,
            FusedAttentionExecutor, FusedKernelConfig, FusedProjectionExecutor, LanguageHeadConfig,
            LatentFanoutScheduleConfig, MambaSequenceConfig,
            ManifoldHyperConnectionCoefficientPolicy, ManifoldHyperConnectionsConfig,
            ResidualConnectorKind, SequenceKernelConfig, SequenceMemorySystem,
            SequenceTrainingExecutor, SummaryMemoryConfig, YNeuronRecurrenceConfig,
        };
        pub use burn_dragon_kernel::api::projection::LowrankGradInputExecutor;
    }

    pub mod state {
        pub use crate::ModelState;
    }

    pub mod recurrent {
        pub use crate::{
            DragonModel, DragonNorm, HaltHead, LanguageMhcLayerDiagnostics, LanguagePipelineState,
            LogitsProjectionProfileSnapshot, LowRankResidualMemoryProfileSnapshot,
            LowRankResidualMemoryStageSnapshot, LowRankResidualOutput,
            LowRankResidualProfileSnapshot, logits_projection_profile_reset,
            logits_projection_profile_snapshot, lowrank_residual_memory_profile_reset,
            lowrank_residual_memory_profile_snapshot, lowrank_residual_profile_reset,
            lowrank_residual_profile_snapshot, lowrank_residual_step, lowrank_residual_step_next,
        };
        #[cfg(any(feature = "probe", test))]
        pub use crate::{
            HeadTensorComparisonDiagnostics, HeadTensorGeometryDiagnostics,
            LanguageDragonInitLayerDiagnostics, LanguageLayerStateDeltaDiagnostics,
            LanguageLayerStateSummaryDiagnostics, LanguageLowRankLayerComparisonDiagnostics,
            LanguageLowRankLayerGeometryDiagnostics, TensorComparisonDiagnostics,
            TensorDistributionDiagnostics, TensorStateDeltaDiagnostics,
            TensorStateSummaryDiagnostics, compare_model_states, summarize_model_state,
        };
    }

    pub mod mhc {
        pub use crate::{
            ManifoldHyperConnectionCoefficients, ManifoldHyperConnectionStreamCoefficients,
            ManifoldHyperConnectionStreamOutput, ManifoldHyperConnectionWidthOutput,
            ManifoldHyperConnections, mhc_merge, mhc_merge_with_coefficients, mhc_passthrough,
            mhc_passthrough_with_coefficients, mhc_split, mhc_split_with_coefficients,
        };
    }

    pub mod experimental {
        pub mod connectors {
            pub use crate::{
                AttentionResidual, AttentionResidualConfig, BlockAttentionResidual,
                BlockAttentionResidualConfig, BlockAttentionResidualSummaryMode,
                ResidualConnectorKind,
            };
        }

        pub mod sequence {
            pub use crate::{
                MambaSequenceConfig, SequenceKernelConfig, SequenceMemorySystem,
                SequenceTrainingExecutor,
            };
        }
    }

    pub mod expert {
        pub use crate::constants;
        pub use crate::kernel;
        pub use crate::model;
        pub use crate::positional;
    }
}

pub use burn_dragon_kernel::api::projection::LowrankGradInputExecutor;
pub use kernel::{BlockPattern1d, BlockPattern2d, BlockSparseConfig};
#[cfg(any(feature = "probe", test))]
pub use model::LanguageDragonInitLayerDiagnostics;
#[cfg(any(feature = "viz", feature = "probe"))]
pub use model::LayerVizState;
pub use model::{
    AttentionResidual, AttentionResidualConfig, BlockAttentionResidual,
    BlockAttentionResidualConfig, BlockAttentionResidualSummaryMode, ClockedSlowMemoryConfig,
    DragonActivationThresholds, DragonConfig, DragonFiringTargetConfig, DragonFiringTargetKind,
    DragonInitializationConfig, DragonInitializationKind, DragonInitializer, DragonModel,
    DragonNeuronGainConfig, DragonNeuronGainKind, DragonNorm, DragonNormConfig, DragonNormKind,
    DragonProjectionRole, DragonResidualScalingConfig, DragonResidualScalingKind,
    DragonTopologyPriorConfig, DragonTopologyPriorKind, FusedAttentionExecutor, FusedKernelConfig,
    FusedProjectionExecutor, HaltHead, LanguageHeadConfig, LanguageMhcLayerDiagnostics,
    LanguagePipelineState, LatentFanoutScheduleConfig, LogitsProjectionProfileSnapshot,
    LowRankResidualMemoryProfileSnapshot, LowRankResidualMemoryStageSnapshot,
    LowRankResidualOutput, LowRankResidualProfileSnapshot, MambaSequenceConfig,
    ManifoldHyperConnectionCoefficientPolicy, ManifoldHyperConnectionCoefficients,
    ManifoldHyperConnectionStreamCoefficients, ManifoldHyperConnectionStreamOutput,
    ManifoldHyperConnectionWidthOutput, ManifoldHyperConnections, ManifoldHyperConnectionsConfig,
    MicroTransformerBlock, ModelState, ResidualConnectorKind, SequenceKernelConfig,
    SequenceMemorySystem, SequenceTrainingExecutor, SharedLowrankActivationBatchStats,
    SharedLowrankContinualBackpropRuntime, SharedLowrankFeatureMetrics, SharedLowrankParamIds,
    SummaryMemoryConfig, YNeuronRecurrenceConfig, logits_projection_profile_reset,
    logits_projection_profile_snapshot, lowrank_residual_memory_profile_reset,
    lowrank_residual_memory_profile_snapshot, lowrank_residual_profile_reset,
    lowrank_residual_profile_snapshot, lowrank_residual_step, lowrank_residual_step_next,
    mhc_merge, mhc_merge_with_coefficients, mhc_passthrough, mhc_passthrough_with_coefficients,
    mhc_split, mhc_split_with_coefficients, near_critical_embedding_initializer,
    near_critical_embedding_std, near_critical_projection_std, near_critical_residual_output_std,
};
#[cfg(any(feature = "probe", test))]
pub use model::{
    HeadTensorComparisonDiagnostics, HeadTensorGeometryDiagnostics,
    LanguageLayerStateDeltaDiagnostics, LanguageLayerStateSummaryDiagnostics,
    LanguageLowRankLayerComparisonDiagnostics, LanguageLowRankLayerGeometryDiagnostics,
    TensorComparisonDiagnostics, TensorDistributionDiagnostics, TensorStateDeltaDiagnostics,
    TensorStateSummaryDiagnostics, compare_model_states, summarize_model_state,
};
pub use positional::RotaryEmbedding;
