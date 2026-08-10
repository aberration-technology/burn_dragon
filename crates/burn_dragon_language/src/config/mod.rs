pub mod core;
#[cfg(feature = "train")]
pub(crate) mod merge;
#[cfg(feature = "train")]
pub mod train;

#[cfg(feature = "train")]
pub use burn_pc::PcGradientNormScope;

pub use core::{
    ContextStrategyConfig, GenerationConfig, GenerationOutputFormat,
    GenerationTokenizerSourceConfig, ModelOverrides,
};
#[cfg(feature = "train")]
pub(crate) use train::RuliadPolicyBatchCadences;
#[cfg(feature = "train")]
pub use train::{
    AutoBatchSizeConfig, CausalInputCorruptionConfig, DatasetConfig, DatasetSourceConfig,
    DragonStateConsistencyConfig, DynamicsAnchorConfig, DynamicsAnchorMask,
    GreedyRolloutUnlikelihoodConfig, HuggingFaceDatasetConfig, HuggingFaceRecordFormat,
    LatentEnergyModelConfig, LatentReasoningAuxiliaryStartPolicy,
    LatentReasoningConstraintBalancerConfig, LatentReasoningNegativeSource,
    LatentReasoningSigRegConfig, LatentReasoningSigRegMode, LatentReasoningSigRegTarget,
    LatentReasoningTargetEncoder, LatentReasoningTrainingConfig, LatentStepContractConfig,
    LocalPredictiveCodingAdjointConditioning, LocalPredictiveCodingConfig,
    LocalPredictiveCodingSolver, LocalPredictiveCodingTerminalCriterion, LogitEntropyFloorConfig,
    NextLatentPredictionConfig, PredictiveCodingBackwardMode, PredictiveCodingConfig,
    PredictiveCodingFactorReduction, PredictiveCodingMode, PredictiveCodingObservationContract,
    PredictiveCodingParameterUpdate, PredictiveCodingStateScope, PredictiveContextRoutingConfig,
    RepeatUnlikelihoodConfig, RepromptTruncation, ResumeHorizonExtensionConfig,
    RuliadAnswerDenoisingConfig, RuliadAnswerRankingConfig, RuliadCheckpointCapabilityContract,
    RuliadPolicyProbeConfig, RuliadPolicyPromotionGateConfig, RuliadProbeGenerationConfig,
    RuliadProofPolicyCandidateSymmetry, RuliadProofPolicyEffectiveMode,
    RuliadProofPolicyGradientScope, RuliadProofPolicyNormalization,
    RuliadProofPolicyPresentationRisk, RuliadProofPolicyScoring,
    RuliadProofPolicySemanticRefreshConfig, RuliadProofPolicyTrainingConfig,
    RuliadProofPolicyTrainingMode, RuliadSupervisionConfig, RuliadSupervisionMode,
    RuliadValidationPanelConfig, RuliadValidationPanelMode, RuliadVerifierRewardConfig,
    RuliadVerifierRewardMode, SdftObjectiveConfig, SdftSdpoObjectiveConfig, SdpoObjectiveConfig,
    SelfDistillationKlKind, SequenceBatchingMode, SequenceStateProbeConfig, TeacherRegularization,
    TrainingAlgorithm, TrainingConfig, TrainingHyperparameters, TrainingObjectiveConfig,
    TrainingObjectiveKind, TrainingValidationConfig, TrainingValidationExecution,
    TrainingValidationObjective, TrainingValidationSampling, ValidationDatasetConfig,
    load_training_config,
};
