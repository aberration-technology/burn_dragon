use crate::config::train::ContinualBackpropConfig;
use crate::train::prelude::*;
use burn_dragon_core::{
    DragonModel, SharedLowrankActivationBatchStats, SharedLowrankContinualBackpropRuntime,
};
use burn_dragon_train::ContinualBackpropTarget;
use burn_dragon_train::train::continual_backprop::{
    ContinualBackpropAdapter, ContinualBackpropFeatureMetrics, ContinualBackpropOptimizer,
    ContinualBackpropOptimizerRecord, ContinualBackpropParamResetTargets,
    attach_continual_backprop_runtime, resolve_optimizer_with_continual_backprop,
    validate_continual_backprop_world_size,
};
use std::marker::PhantomData;
use std::sync::atomic::Ordering;

#[derive(Clone, Default)]
pub struct DragonLanguageContinualBackpropAdapter<B>(PhantomData<B>);

impl<B> ContinualBackpropAdapter<B, LanguageTrainModel<B>>
    for DragonLanguageContinualBackpropAdapter<B>
where
    B: AutodiffBackend,
{
    type FreshModel = DragonModel<B>;
    type BatchStats = SharedLowrankActivationBatchStats;

    fn validate_config(
        config: &ContinualBackpropConfig,
        fresh_model: &Self::FreshModel,
    ) -> Result<()> {
        if !config.enabled {
            return Ok(());
        }
        anyhow::ensure!(
            matches!(config.target, ContinualBackpropTarget::SharedLowrankLatents),
            "training.continual_backprop.target must be \"shared_lowrank_latents\""
        );
        anyhow::ensure!(
            fresh_model.supports_shared_lowrank_continual_backprop(),
            "training.continual_backprop currently requires rollout_fast_steps_per_slow_step = 1 and y_neuron_recurrence disabled"
        );
        Ok(())
    }

    fn attach_runtime(
        mut module: LanguageTrainModel<B>,
        config: &ContinualBackpropConfig,
    ) -> LanguageTrainModel<B> {
        if !config.enabled {
            return module;
        }
        let runtime = SharedLowrankContinualBackpropRuntime::new(config.sample_interval_steps);
        module.model = module
            .model
            .with_shared_lowrank_continual_backprop_runtime(Some(runtime));
        module
    }

    fn take_batch_stats(module: &LanguageTrainModel<B>) -> Option<Self::BatchStats> {
        module
            .model
            .take_shared_lowrank_continual_backprop_batch_stats()
    }

    fn batch_stats_mean(batch_stats: &Self::BatchStats) -> Vec<f32> {
        batch_stats.mean()
    }

    fn batch_stats_mean_abs(batch_stats: &Self::BatchStats) -> Vec<f32> {
        batch_stats.mean_abs()
    }

    fn feature_count(module: &LanguageTrainModel<B>) -> usize {
        module.model.shared_lowrank_feature_count()
    }

    fn device(module: &LanguageTrainModel<B>) -> B::Device {
        module.model.shared_lowrank_device()
    }

    fn target_lr_scale(module: &LanguageTrainModel<B>) -> f32 {
        module.continual_backprop_target_lr_scale()
    }

    fn feature_metrics(module: &LanguageTrainModel<B>) -> ContinualBackpropFeatureMetrics {
        let metrics = module.model.shared_lowrank_feature_metrics();
        ContinualBackpropFeatureMetrics {
            incoming_l1: metrics.incoming_l1,
            outgoing_l1: metrics.outgoing_l1,
        }
    }

    fn reinitialize_features(
        mut module: LanguageTrainModel<B>,
        fresh_model: &Self::FreshModel,
        selected: &[usize],
    ) -> LanguageTrainModel<B> {
        module.model = module
            .model
            .with_reinitialized_shared_lowrank_features(fresh_model, selected);
        module
    }

    fn optimizer_reset_targets(
        module: &LanguageTrainModel<B>,
    ) -> ContinualBackpropParamResetTargets {
        let param_ids = module.model.shared_lowrank_param_ids();
        let latent_per_head = module.model.shared_lowrank_feature_count();
        ContinualBackpropParamResetTargets {
            feature_tensors_3d: vec![param_ids.encoder, param_ids.encoder_v],
            row_feature_tensors_2d: vec![(param_ids.decoder, latent_per_head)],
            feature_tensors_2d: vec![param_ids.rwkv_time_decay],
        }
    }

    fn complete_optimizer_step(module: &LanguageTrainModel<B>) {
        if let Some(runtime) = module.model.shared_lowrank_continual_backprop_runtime() {
            runtime.optimizer_step().fetch_add(1, Ordering::Relaxed);
        }
    }
}

pub type LanguageOptimizer<B> =
    ContinualBackpropOptimizer<B, LanguageTrainModel<B>, DragonLanguageContinualBackpropAdapter<B>>;

pub type LanguageOptimizerRecord<B> = ContinualBackpropOptimizerRecord<LanguageTrainModel<B>, B>;

pub fn validate_dragon_continual_backprop<B>(
    training: &TrainingHyperparameters,
    model: &DragonModel<B>,
    world_size: usize,
) -> Result<()>
where
    B: AutodiffBackend,
{
    validate_continual_backprop_world_size(&training.continual_backprop, world_size)?;
    DragonLanguageContinualBackpropAdapter::<B>::validate_config(
        &training.continual_backprop,
        model,
    )
}

pub fn resolve_dragon_language_optimizer<B>(
    training: &TrainingHyperparameters,
    optimizer_cfg: &OptimizerConfig,
    total_steps: usize,
    fresh_model: DragonModel<B>,
) -> Result<LanguageOptimizer<B>>
where
    B: AutodiffBackend,
{
    resolve_optimizer_with_continual_backprop::<
        B,
        LanguageTrainModel<B>,
        DragonLanguageContinualBackpropAdapter<B>,
    >(
        optimizer_cfg,
        total_steps,
        &training.continual_backprop,
        fresh_model,
    )
}

impl<B> LanguageTrainModel<B>
where
    B: AutodiffBackend,
{
    pub fn with_continual_backprop(self, config: &ContinualBackpropConfig) -> Self {
        attach_continual_backprop_runtime::<B, _, DragonLanguageContinualBackpropAdapter<B>>(
            self, config,
        )
    }
}
