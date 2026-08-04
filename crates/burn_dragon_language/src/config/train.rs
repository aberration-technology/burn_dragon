use serde::{Deserialize, Serialize};

mod load;
mod schema;
mod validate;

pub use burn_dragon_train::{
    ContinualBackpropConfig, ContinualBackpropLrCoupling, ContinualBackpropTarget,
};
pub use load::load_training_config;
pub(crate) use schema::RuliadPolicyBatchCadences;
pub use schema::{
    AutoBatchSizeConfig, CausalInputCorruptionConfig, DatasetConfig, DatasetSourceConfig,
    DragonStateConsistencyConfig, DynamicsAnchorConfig, DynamicsAnchorMask,
    GreedyRolloutUnlikelihoodConfig, HuggingFaceDatasetConfig, HuggingFaceRecordFormat,
    LatentEnergyModelConfig, LatentReasoningAuxiliaryStartPolicy,
    LatentReasoningConstraintBalancerConfig, LatentReasoningNegativeSource,
    LatentReasoningSigRegConfig, LatentReasoningSigRegMode, LatentReasoningSigRegTarget,
    LatentReasoningTargetEncoder, LatentReasoningTrainingConfig, LatentStepContractConfig,
    LocalPredictiveCodingConfig, LocalPredictiveCodingSolver, LogitEntropyFloorConfig,
    ModuleLrScaleEntry, ModuleLrScaleScheduleConfig, NeuronScalingConfig, NeuronScalingGrowth,
    NeuronScalingStabilizationConfig, NextLatentPredictionConfig, PredictiveCodingBackwardMode,
    PredictiveCodingConfig, PredictiveCodingFactorReduction, PredictiveCodingMode,
    PredictiveCodingObservationContract, PredictiveCodingParameterUpdate,
    PredictiveCodingStateScope, RepeatUnlikelihoodConfig, RepromptTruncation,
    RuliadAnswerContractConfig, RuliadAnswerDenoisingConfig, RuliadAnswerRankingConfig,
    RuliadPolicyProbeConfig, RuliadPolicyPromotionGateConfig, RuliadProbeGenerationConfig,
    RuliadProofPolicyCandidateSymmetry, RuliadProofPolicyEffectiveMode,
    RuliadProofPolicyGradientScope, RuliadProofPolicyNormalization,
    RuliadProofPolicyPresentationRisk, RuliadProofPolicyScoring, RuliadProofPolicyTrainingConfig,
    RuliadProofPolicyTrainingMode, RuliadSupervisionConfig, RuliadSupervisionMode,
    RuliadVerifierRewardConfig, RuliadVerifierRewardMode, SdftObjectiveConfig,
    SdftSdpoObjectiveConfig, SdpoObjectiveConfig, SelfDistillationKlKind, SequenceBatchingMode,
    SequenceStateProbeConfig, TeacherRegularization, TrainingAlgorithm, TrainingConfig,
    TrainingHyperparameters, TrainingObjectiveConfig, TrainingObjectiveKind,
    TrainingValidationConfig, TrainingValidationExecution, TrainingValidationSampling,
    ValidationDatasetConfig,
};

use crate::tokenizer::TokenizerConfig;

use super::{ContextStrategyConfig, GenerationConfig, ModelOverrides};
