use serde::{Deserialize, Serialize};

mod load;
mod next_latent;
mod provenance;
mod schema;
mod structured_schedule;
mod validate;
mod validation;

pub use burn_dragon_train::{
    ContinualBackpropConfig, ContinualBackpropLrCoupling, ContinualBackpropTarget,
};
pub use load::load_training_config;
pub(crate) use next_latent::NEXT_LATENT_OBJECTIVE_CONTRACT_VERSION;
pub use next_latent::NextLatentPredictionConfig;
pub use provenance::TrainingProvenanceConfig;
pub(crate) use schema::RuliadPolicyBatchCadences;
pub use schema::{
    AutoBatchSizeConfig, CausalInputCorruptionConfig, DatasetConfig, DatasetSourceConfig,
    DragonStateConsistencyConfig, DynamicsAnchorConfig, DynamicsAnchorMask,
    GreedyRolloutUnlikelihoodConfig, HuggingFaceDatasetConfig, HuggingFaceRecordFormat,
    LatentEnergyModelConfig, LatentReasoningAuxiliaryStartPolicy,
    LatentReasoningConstraintBalancerConfig, LatentReasoningNegativeSource,
    LatentReasoningSigRegConfig, LatentReasoningSigRegMode, LatentReasoningSigRegTarget,
    LatentReasoningTargetEncoder, LatentReasoningTrainingConfig, LatentStepContractConfig,
    LocalPredictiveCodingAdjointConditioning, LocalPredictiveCodingConfig,
    LocalPredictiveCodingObjectiveRoutingConfig, LocalPredictiveCodingSolver,
    LocalPredictiveCodingTerminalCriterion, LogitEntropyFloorConfig, ModuleLrScaleEntry,
    ModuleLrScaleScheduleConfig, NeuronScalingConfig, NeuronScalingGrowth,
    NeuronScalingStabilizationConfig, PredictiveCodingBackwardMode, PredictiveCodingConfig,
    PredictiveCodingFactorReduction, PredictiveCodingMode, PredictiveCodingObservationContract,
    PredictiveCodingParameterUpdate, PredictiveCodingStateScope, PredictiveContextRoutingConfig,
    RepeatUnlikelihoodConfig, RepromptTruncation, ResumeHorizonExtensionConfig,
    RuliadAnswerContractConfig, RuliadAnswerDenoisingConfig, RuliadAnswerRankingConfig,
    RuliadCheckpointCapabilityContract, RuliadConsolidationConfig, RuliadConsolidationCoordinate,
    RuliadPolicyProbeConfig, RuliadPolicyPromotionGateConfig, RuliadProbeGenerationConfig,
    RuliadPromptValueBindingConfig, RuliadPromptValueBindingContext,
    RuliadPromptValueBindingObjective, RuliadProofPolicyCandidateSymmetry,
    RuliadProofPolicyCounterfactualObjective, RuliadProofPolicyEffectiveMode,
    RuliadProofPolicyGradientScope, RuliadProofPolicyNormalization,
    RuliadProofPolicyPresentationRisk, RuliadProofPolicyPromptContext, RuliadProofPolicyScoring,
    RuliadProofPolicySemanticRefreshConfig, RuliadProofPolicyTarget,
    RuliadProofPolicyTrainingConfig, RuliadProofPolicyTrainingMode, RuliadSupervisionConfig,
    RuliadSupervisionMode, RuliadValidationPanelConfig, RuliadValidationPanelMode,
    RuliadVerifierRewardConfig, RuliadVerifierRewardMode, SdftObjectiveConfig,
    SdftSdpoObjectiveConfig, SdpoObjectiveConfig, SelfDistillationKlKind, SequenceBatchingMode,
    SequenceStateProbeConfig, TeacherRegularization, TrainingAlgorithm, TrainingConfig,
    TrainingHyperparameters, TrainingObjectiveConfig, TrainingObjectiveKind,
    TrainingValidationConfig, TrainingValidationExecution, TrainingValidationObjective,
    TrainingValidationSampling, ValidationDatasetConfig,
};

use crate::tokenizer::TokenizerConfig;

use super::{ContextStrategyConfig, GenerationConfig, ModelOverrides};
