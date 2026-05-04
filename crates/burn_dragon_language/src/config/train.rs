use serde::{Deserialize, Serialize};

mod load;
mod schema;
mod validate;

pub use burn_dragon_train::{
    ContinualBackpropConfig, ContinualBackpropLrCoupling, ContinualBackpropTarget,
};
pub use load::load_training_config;
pub use schema::{
    DatasetConfig, DatasetSourceConfig, HuggingFaceDatasetConfig, HuggingFaceRecordFormat,
    ModuleLrScaleEntry, ModuleLrScaleScheduleConfig, RepromptTruncation, SdftObjectiveConfig,
    SdpoObjectiveConfig, SelfDistillationKlKind, TeacherRegularization, TrainingConfig,
    TrainingHyperparameters, TrainingObjectiveConfig, TrainingObjectiveKind,
    ValidationDatasetConfig,
};

use crate::tokenizer::TokenizerConfig;

use super::{ContextStrategyConfig, GenerationConfig, ModelOverrides};
