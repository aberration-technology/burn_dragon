#![allow(unused_imports)]

pub(crate) use burn_dragon_time::{SystemTime, UNIX_EPOCH};
pub(crate) use std::fs;
pub(crate) use std::path::{Path, PathBuf};
pub(crate) use std::sync::Arc;
pub(crate) use std::sync::atomic::{AtomicBool, Ordering};

pub(crate) use anyhow::{Context, Result, anyhow};
pub(crate) use rand::{Rng, SeedableRng, rngs::StdRng, thread_rng};

pub(crate) use burn::data::dataloader::DataLoader;
pub(crate) use burn::lr_scheduler::{
    LrScheduler,
    cosine::{CosineAnnealingLrScheduler, CosineAnnealingLrSchedulerConfig},
    exponential::{ExponentialLrScheduler, ExponentialLrSchedulerConfig},
    linear::{LinearLrScheduler, LinearLrSchedulerConfig},
    noam::{NoamLrScheduler, NoamLrSchedulerConfig},
    step::{StepLrScheduler, StepLrSchedulerConfig},
};
pub(crate) use burn::module::{
    AutodiffModule, Content, Module, ModuleDisplay, ModuleDisplayDefault, Param,
};
pub(crate) use burn::nn::loss::CrossEntropyLossConfig;
pub(crate) use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
pub(crate) use burn::optim::adaptor::OptimizerAdaptor;
pub(crate) use burn::optim::grad_clipping::GradientClippingConfig;
pub(crate) use burn::optim::{
    AdamW, AdamWConfig, GradientsAccumulator, GradientsParams, LearningRate, Optimizer,
};
pub(crate) use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
pub(crate) use burn::tensor::Distribution as TensorDistribution;
pub(crate) use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
pub(crate) use burn::tensor::{Int, Tensor, TensorData};
#[cfg(feature = "ddp")]
pub(crate) use burn_collective::{
    PeerId, ReduceOperation, all_reduce, broadcast, finish_collective, register,
};
#[cfg(feature = "ddp")]
pub(crate) use burn_train::checkpoint::{Checkpointer, FileCheckpointer};
pub(crate) use burn_train::metric::{Adaptor, ItemLazy, LearningRateMetric, LossMetric};
pub(crate) use burn_train::{
    ExecutionStrategy, InferenceStep as ValidStep, LearningResult as TrainingResult,
    MultiDeviceOptim, SupervisedTraining, TrainOutput, TrainStep,
    TrainingStrategy as LearningStrategy,
};
pub(crate) use tracing::info;

#[cfg(all(feature = "cuda", test))]
pub(crate) use burn_cuda::Cuda;

pub(crate) use serde::Serialize;

pub(crate) use crate::config::{
    DatasetConfig, DatasetSourceConfig, HuggingFaceDatasetConfig, HuggingFaceRecordFormat,
    RepromptTruncation, SdftObjectiveConfig, SdftSdpoObjectiveConfig, SdpoObjectiveConfig,
    SelfDistillationKlKind, TeacherRegularization, TrainingConfig, TrainingHyperparameters,
    TrainingObjectiveConfig, TrainingObjectiveKind, ValidationDatasetConfig,
};
pub(crate) use crate::dataset::{
    Dataset, DatasetSplit, RandomDataLoader, SequenceBatch, StreamingDataLoader, build_dataset,
    sample_batch_with_shape,
};
pub(crate) use crate::inference::{
    WgpuFusedCoreOverride, apply_wgpu_fused_core_override, build_model_config,
    build_model_config_with_tokenizer,
};
pub(crate) use crate::tokenizer::TokenizerConfig;
pub(crate) use crate::{ContextStrategyConfig, GenerationConfig, ModelOverrides};

pub(crate) use crate::loss::language_model_loss;
pub(crate) use crate::train::objective::{
    ObjectiveTrainerKind, RolloutObjectiveRuntimeConstraints, SelectedTokenDistillationHiddenBatch,
    SelectedTokenSdpoLossConfig, assert_flat_logits_for_rollout_objective, clipped_policy_loss,
    ensure_objective_supported, ensure_rollout_objective_runtime, log_probs_from_logits,
    masked_token_mean, sdpo_token_advantage, selected_token_distillation_loss_from_hidden,
    selected_token_log_prob_mse_loss, selected_token_log_probs,
    selected_token_log_probs_from_hidden, selected_token_sdpo_loss_from_hidden,
    self_distillation_loss_from_logits, self_distillation_per_token_from_log_probs,
};
pub(crate) use crate::train::steps::LanguageTrainModel;
pub(crate) use burn_dragon_core::{
    DragonConfig, DragonInitializationKind, DragonModel, LanguagePipelineState, ModelState,
};
pub(crate) use burn_dragon_train::train::constants::ValidBackend;
pub(crate) use burn_dragon_train::train::metrics::{
    DeviceMetric, LanguageModelOutput, LanguageModelTrainItem, LossValue, MetricSinkEntry,
    MetricSinkSplit, MetricSinkValueKind, MetricsSinkSpec, ScalarMetric, ScalarValue,
};
pub(crate) use burn_dragon_train::train::pipeline::{
    PipelinePlan, PipelineRankWorkload, ResolvedLrScheduler, ResolvedOptimizer, ScheduleSource,
    TrainSchedule, adamw_config_from_optimizer, build_pipeline_plan, build_pipeline_rank_workload,
    create_run_dir, resolve_optimizer, resolve_valid_steps_per_epoch,
    simulate_pipeline_communication, split_microbatch_ranges, write_latest_run,
};
#[cfg(feature = "ddp")]
pub(crate) use burn_dragon_train::train::runtime::resolve_collective_config;
pub(crate) use burn_dragon_train::train::runtime::{
    DeviceMemoryUsage, ParallelRuntime, PipelineParallelLayout, PipelineRankAssignment,
    cleanup_device_memory, device_memory_usage_safe, resolve_parallel_runtime,
    resolve_pipeline_parallel_layout, resolve_training_devices,
};
pub(crate) use burn_dragon_train::{
    GatedDeltaNet2Spec, GdpoConfig, GdpoHardGate, KernelSpec, LayerStateSpec,
    LearningRateScheduleConfig, ModelSpec, OptimizerConfig, OptimizerKind, OptimizerScheduleMode,
    OptimizerSpec, ParallelConfig, ParallelSpec, ParallelismKind, ReservoirInitializationSpec,
    SequenceKernelConfig, StateAxisSpec, StateLayout, StateTensorSpec, WgpuRuntimeConfig,
};
