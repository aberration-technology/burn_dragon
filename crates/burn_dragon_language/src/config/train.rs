use serde::{Deserialize, Serialize};

mod load;
mod schema;
mod validate;

pub use burn_dragon_train::{
    ContinualBackpropConfig, ContinualBackpropLrCoupling, ContinualBackpropTarget,
};
pub use load::load_training_config;
pub use schema::{
    AutoBatchSizeConfig, CausalInputCorruptionConfig, DatasetConfig, DatasetSourceConfig,
    DragonStateConsistencyConfig, DynamicsAnchorConfig, DynamicsAnchorMask,
    GreedyRolloutUnlikelihoodConfig, HuggingFaceDatasetConfig, HuggingFaceRecordFormat,
    LatentEnergyModelConfig, LatentReasoningAuxiliaryStartPolicy,
    LatentReasoningConstraintBalancerConfig, LatentReasoningNegativeSource,
    LatentReasoningSigRegConfig, LatentReasoningSigRegMode, LatentReasoningSigRegTarget,
    LatentReasoningTargetEncoder, LatentReasoningTrainingConfig, LatentStepContractConfig,
    LogitEntropyFloorConfig, ModuleLrScaleEntry, ModuleLrScaleScheduleConfig, NeuronScalingConfig,
    NeuronScalingGrowth, NeuronScalingStabilizationConfig, NextLatentPredictionConfig,
    PredictiveCodingBackwardMode, PredictiveCodingConfig, PredictiveCodingMode,
    PredictiveCodingParameterUpdate, PredictiveCodingStateScope, RepeatUnlikelihoodConfig,
    RepromptTruncation, RuliadAnswerDenoisingConfig, RuliadAnswerRankingConfig,
    RuliadSupervisionConfig, RuliadSupervisionMode, SdftObjectiveConfig, SdftSdpoObjectiveConfig,
    SdpoObjectiveConfig, SelfDistillationKlKind, TeacherRegularization, TrainingConfig,
    TrainingHyperparameters, TrainingObjectiveConfig, TrainingObjectiveKind,
    ValidationDatasetConfig,
};

use crate::tokenizer::TokenizerConfig;

use super::{ContextStrategyConfig, GenerationConfig, ModelOverrides};
