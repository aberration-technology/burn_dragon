pub mod core;
#[cfg(feature = "train")]
pub(crate) mod merge;
#[cfg(feature = "train")]
pub mod train;

pub use core::{
    ContextStrategyConfig, GenerationConfig, GenerationOutputFormat,
    GenerationTokenizerSourceConfig, ModelOverrides,
};
#[cfg(feature = "train")]
pub use train::{
    AutoBatchSizeConfig, CausalInputCorruptionConfig, DatasetConfig, DatasetSourceConfig,
    DragonStateConsistencyConfig, DynamicsAnchorConfig, DynamicsAnchorMask,
    GreedyRolloutUnlikelihoodConfig, HuggingFaceDatasetConfig, HuggingFaceRecordFormat,
    LatentEnergyModelConfig, LatentReasoningAuxiliaryStartPolicy,
    LatentReasoningConstraintBalancerConfig, LatentReasoningNegativeSource,
    LatentReasoningSigRegConfig, LatentReasoningSigRegMode, LatentReasoningSigRegTarget,
    LatentReasoningTargetEncoder, LatentReasoningTrainingConfig, LatentStepContractConfig,
    LogitEntropyFloorConfig, NextLatentPredictionConfig, PredictiveCodingBackwardMode,
    PredictiveCodingConfig, PredictiveCodingMode, PredictiveCodingParameterUpdate,
    PredictiveCodingStateScope, RepeatUnlikelihoodConfig, RepromptTruncation,
    RuliadAnswerDenoisingConfig, RuliadAnswerRankingConfig, RuliadSupervisionConfig,
    RuliadSupervisionMode, SdftObjectiveConfig, SdftSdpoObjectiveConfig, SdpoObjectiveConfig,
    SelfDistillationKlKind, TeacherRegularization, TrainingConfig, TrainingHyperparameters,
    TrainingObjectiveConfig, TrainingObjectiveKind, ValidationDatasetConfig, load_training_config,
};
