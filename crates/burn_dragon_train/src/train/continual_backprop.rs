use crate::train::prelude::*;
use crate::{ContinualBackpropConfig, ContinualBackpropLrCoupling, OptimizerKind};
use burn::module::{ModuleMapper, ModuleVisitor, ParamId};
use burn::optim::AdaptiveMomentumState;
use burn::optim::MultiGradientsParams;
use burn::optim::SimpleOptimizer;
use burn::optim::grad_clipping::GradientClipping;
use burn::optim::record::AdaptorRecord;
use hashbrown::{HashMap, HashSet};
use std::marker::PhantomData;

use crate::train::pipeline::{ResolvedOptimizer, ResolvedOptimizerRecord, resolve_optimizer};

#[derive(Clone, Debug, Default)]
pub struct ContinualBackpropFeatureMetrics {
    pub incoming_l1: Vec<f32>,
    pub outgoing_l1: Vec<f32>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ContinualBackpropTelemetry {
    pub optimizer_step: usize,
    pub feature_count: usize,
    pub eligible_count: usize,
    pub replacement_count: usize,
    pub replacement_budget: f32,
    pub lr_multiplier: f32,
    pub paused: bool,
    pub pause_reason: Option<String>,
    pub utility_min: f32,
    pub utility_mean: f32,
    pub utility_max: f32,
    pub age_mean: f32,
    pub age_max: f32,
}

#[derive(Clone, Debug, Default)]
pub struct ContinualBackpropParamResetTargets {
    pub feature_tensors_3d: Vec<ParamId>,
    pub row_feature_tensors_2d: Vec<(ParamId, usize)>,
    pub feature_tensors_2d: Vec<ParamId>,
}

pub trait ContinualBackpropAdapter<B, M>: Clone + Send
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + Clone + Send,
{
    type FreshModel: Clone + Send;
    type BatchStats;

    fn validate_config(
        config: &ContinualBackpropConfig,
        fresh_model: &Self::FreshModel,
    ) -> Result<()>;
    fn attach_runtime(module: M, config: &ContinualBackpropConfig) -> M;
    fn take_batch_stats(module: &M) -> Option<Self::BatchStats>;
    fn batch_stats_mean(batch_stats: &Self::BatchStats) -> Vec<f32>;
    fn batch_stats_mean_abs(batch_stats: &Self::BatchStats) -> Vec<f32>;
    fn feature_count(module: &M) -> usize;
    fn device(module: &M) -> B::Device;
    fn target_lr_scale(module: &M) -> f32;
    fn feature_metrics(module: &M) -> ContinualBackpropFeatureMetrics;
    fn reinitialize_features(module: M, fresh_model: &Self::FreshModel, selected: &[usize]) -> M;
    fn optimizer_reset_targets(module: &M) -> ContinualBackpropParamResetTargets;
    fn complete_optimizer_step(_module: &M) {}
}

pub fn validate_continual_backprop_world_size(
    config: &ContinualBackpropConfig,
    world_size: usize,
) -> Result<()> {
    anyhow::ensure!(
        !config.enabled || world_size == 1,
        "training.continual_backprop currently requires single-process training"
    );
    Ok(())
}

pub fn attach_continual_backprop_runtime<B, M, A>(module: M, config: &ContinualBackpropConfig) -> M
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + Clone + Send,
    A: ContinualBackpropAdapter<B, M>,
{
    A::attach_runtime(module, config)
}

#[derive(Clone)]
struct DragonAdamW {
    beta_1: f32,
    beta_2: f32,
    epsilon: f32,
    weight_decay: f32,
}

#[derive(burn::record::Record, Clone)]
struct DragonAdamWState<B: BackendTrait, const D: usize> {
    momentum: AdaptiveMomentumState<B, D>,
}

impl<B: BackendTrait> SimpleOptimizer<B> for DragonAdamW {
    type State<const D: usize> = DragonAdamWState<B, D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<B, D>,
        grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        let factor_1 = 1.0 - self.beta_1;
        let factor_2 = 1.0 - self.beta_2;
        let state = if let Some(mut state) = state {
            state.momentum.moment_1 = state
                .momentum
                .moment_1
                .mul_scalar(self.beta_1)
                .add(grad.clone().mul_scalar(factor_1));
            state.momentum.moment_2 = state
                .momentum
                .moment_2
                .mul_scalar(self.beta_2)
                .add(grad.square().mul_scalar(factor_2));
            state.momentum.max_moment_2 = None;
            state.momentum.time += 1;
            state
        } else {
            DragonAdamWState {
                momentum: AdaptiveMomentumState {
                    time: 1,
                    moment_1: grad.clone().mul_scalar(factor_1),
                    moment_2: grad.square().mul_scalar(factor_2),
                    max_moment_2: None,
                },
            }
        };

        let time = state.momentum.time as i32;
        let moment_1_corrected = state
            .momentum
            .moment_1
            .clone()
            .div_scalar(1.0 - self.beta_1.powi(time));
        let moment_2_corrected = state
            .momentum
            .moment_2
            .clone()
            .div_scalar(1.0 - self.beta_2.powi(time));
        let update_delta =
            moment_1_corrected.div(moment_2_corrected.sqrt().add_scalar(self.epsilon));
        let decay_rate = lr * self.weight_decay as f64;
        let decayed_tensor = if decay_rate == 0.0 {
            tensor.clone()
        } else {
            tensor.clone().mul_scalar(1.0 - decay_rate)
        };
        let updated = decayed_tensor - update_delta.mul_scalar(lr);
        (updated, Some(state))
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &B::Device) -> Self::State<D> {
        state.momentum = state.momentum.to_device(device);
        state
    }
}

enum GradAdaptor {
    Single(GradientsParams),
    Multi(MultiGradientsParams),
}

impl GradAdaptor {
    fn remove<B: BackendTrait, const D: usize>(
        &mut self,
        id: ParamId,
    ) -> Option<(Tensor<B, D>, B::Device)> {
        match self {
            GradAdaptor::Single(grads) => grads.remove(id).map(|tensor| {
                let device = tensor.device();
                (tensor, device)
            }),
            GradAdaptor::Multi(grads) => grads.remove(id),
        }
    }
}

#[derive(burn::record::Record, Clone)]
struct ContinualBackpropState<B: BackendTrait> {
    step: usize,
    replacement_budget: f32,
    age: Tensor<B, 1>,
    avg_activation: Tensor<B, 1>,
    avg_abs_activation: Tensor<B, 1>,
}

#[derive(burn::record::Record, Clone)]
pub struct ContinualBackpropAdamWRecord<B: AutodiffBackend> {
    records: HashMap<ParamId, AdaptorRecord<DragonAdamW, B>>,
    state: Option<ContinualBackpropState<B>>,
}

#[derive(Clone)]
struct ContinualBackpropAdamWOptimizer<B, M, A>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + Clone + Send,
    A: ContinualBackpropAdapter<B, M>,
{
    optimizer: DragonAdamW,
    records: HashMap<ParamId, AdaptorRecord<DragonAdamW, B>>,
    grad_clipping: Option<GradientClipping>,
    state: Option<ContinualBackpropState<B>>,
    config: ContinualBackpropConfig,
    base_learning_rate: LearningRate,
    fresh_model: A::FreshModel,
    pause_until_step: usize,
    pause_reason: Option<String>,
    cooldown_until_step: usize,
    last_telemetry: Option<ContinualBackpropTelemetry>,
    _adapter: PhantomData<A>,
    module: PhantomData<M>,
}

#[derive(burn::record::Record, Clone)]
pub struct ContinualBackpropOptimizerRecord<M, B>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    kind: u8,
    standard: Option<ResolvedOptimizerRecord<M, B>>,
    continual_backprop: Option<ContinualBackpropAdamWRecord<B>>,
}

#[derive(Clone)]
pub struct ContinualBackpropOptimizer<B, M, A>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + Clone + Send,
    A: ContinualBackpropAdapter<B, M>,
{
    kind: ContinualBackpropOptimizerKind<B, M, A>,
}

#[derive(Clone)]
enum ContinualBackpropOptimizerKind<B, M, A>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + Clone + Send,
    A: ContinualBackpropAdapter<B, M>,
{
    Standard(ResolvedOptimizer<B, M>),
    ContinualBackprop(ContinualBackpropAdamWOptimizer<B, M, A>),
}

pub fn resolve_optimizer_with_continual_backprop<B, M, A>(
    optimizer_cfg: &OptimizerConfig,
    total_steps: usize,
    config: &ContinualBackpropConfig,
    fresh_model: A::FreshModel,
) -> Result<ContinualBackpropOptimizer<B, M, A>>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + Clone + Send,
    A: ContinualBackpropAdapter<B, M>,
{
    let kind = if config.enabled {
        ContinualBackpropOptimizerKind::ContinualBackprop(ContinualBackpropAdamWOptimizer::new(
            optimizer_cfg,
            config.clone(),
            fresh_model,
        )?)
    } else {
        ContinualBackpropOptimizerKind::Standard(resolve_optimizer::<B, M>(
            optimizer_cfg,
            total_steps,
        )?)
    };
    Ok(ContinualBackpropOptimizer { kind })
}

impl<B, M, A> ContinualBackpropAdamWOptimizer<B, M, A>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + Clone + Send,
    A: ContinualBackpropAdapter<B, M>,
{
    fn new(
        optimizer_cfg: &OptimizerConfig,
        config: ContinualBackpropConfig,
        fresh_model: A::FreshModel,
    ) -> Result<Self> {
        anyhow::ensure!(
            matches!(optimizer_cfg.name, OptimizerKind::Adamw),
            "training.continual_backprop currently supports optimizer.name = \"adamw\" only"
        );
        A::validate_config(&config, &fresh_model)?;
        let grad_clipping = if let Some(clip) = optimizer_cfg.grad_clip_norm {
            Some(GradientClippingConfig::Norm(clip).init())
        } else {
            optimizer_cfg
                .grad_clip_value
                .map(|clip| GradientClippingConfig::Value(clip).init())
        };
        Ok(Self {
            optimizer: DragonAdamW {
                beta_1: 0.9,
                beta_2: 0.999,
                epsilon: 1.0e-5,
                weight_decay: optimizer_cfg.weight_decay,
            },
            records: HashMap::new(),
            grad_clipping,
            state: None,
            config,
            base_learning_rate: optimizer_cfg.learning_rate,
            fresh_model,
            pause_until_step: 0,
            pause_reason: None,
            cooldown_until_step: 0,
            last_telemetry: None,
            _adapter: PhantomData,
            module: PhantomData,
        })
    }

    fn step_impl(&mut self, lr: LearningRate, module: M, grads: GradAdaptor) -> M {
        let mut grads = grads;
        let mut mapper = ContinualBackpropAdamWMapper::<B>::new(
            &self.optimizer,
            &mut self.records,
            &mut grads,
            lr,
            self.grad_clipping.as_ref(),
        );
        let mut updated = module.map(&mut mapper);
        updated = self.apply_continual_backprop(updated, lr);
        updated
    }

    fn apply_continual_backprop(&mut self, module: M, lr: LearningRate) -> M {
        let feature_count = A::feature_count(&module);
        let device = A::device(&module);
        let mut state = self.state.take().unwrap_or_else(|| ContinualBackpropState {
            step: 0,
            replacement_budget: 0.0,
            age: Tensor::<B, 1>::zeros([feature_count], &device),
            avg_activation: Tensor::<B, 1>::zeros([feature_count], &device),
            avg_abs_activation: Tensor::<B, 1>::zeros([feature_count], &device),
        });
        state.step = state.step.saturating_add(1);
        state.age = state.age.add_scalar(1.0);

        if let Some(batch_stats) = A::take_batch_stats(&module) {
            state = self.update_state_from_batch_stats(state, batch_stats, &device, feature_count);
        }

        let mut updated_module = module;
        if state
            .step
            .is_multiple_of(self.config.replace_interval_steps)
        {
            let target_lr_scale = A::target_lr_scale(&updated_module);
            let (selected, telemetry) =
                self.select_features_to_replace(&updated_module, &mut state, lr, target_lr_scale);
            self.last_telemetry = Some(telemetry);
            if !selected.is_empty() {
                updated_module =
                    A::reinitialize_features(updated_module, &self.fresh_model, &selected);
                let reset_targets = A::optimizer_reset_targets(&updated_module);
                self.reset_optimizer_state_for_features(reset_targets, &selected);
                state = self.reset_state_for_features(state, &selected, &device, feature_count);
                self.cooldown_until_step = state.step.saturating_add(self.config.cooldown_steps);
                info!(
                    "continual backprop replaced {} features at optimizer_step={}",
                    selected.len(),
                    state.step
                );
            }
        }

        A::complete_optimizer_step(&updated_module);
        self.state = Some(state);
        updated_module
    }

    fn continual_backprop_lr_multiplier(&self, lr: LearningRate, target_lr_scale: f32) -> f32 {
        let base_ratio = if self.base_learning_rate > 0.0 {
            (lr / self.base_learning_rate).max(0.0) as f32
        } else {
            1.0
        };
        let multiplier = match self.config.lr_coupling {
            ContinualBackpropLrCoupling::None => 1.0,
            ContinualBackpropLrCoupling::GlobalRatio => base_ratio,
            ContinualBackpropLrCoupling::TargetGroupRatio => base_ratio * target_lr_scale.max(0.0),
        };
        multiplier.powf(self.config.lr_coupling_power.max(0.0))
    }

    fn update_state_from_batch_stats(
        &self,
        mut state: ContinualBackpropState<B>,
        batch_stats: A::BatchStats,
        device: &B::Device,
        feature_count: usize,
    ) -> ContinualBackpropState<B> {
        let mean = A::batch_stats_mean(&batch_stats);
        let mean_abs = A::batch_stats_mean_abs(&batch_stats);
        if mean.len() != feature_count || mean_abs.len() != feature_count {
            return state;
        }
        let keep = self.config.utility_decay;
        let update = 1.0 - keep;
        let mean_tensor = Tensor::<B, 1>::from_data(TensorData::new(mean, [feature_count]), device);
        let mean_abs_tensor =
            Tensor::<B, 1>::from_data(TensorData::new(mean_abs, [feature_count]), device);
        state.avg_activation = state
            .avg_activation
            .mul_scalar(keep)
            .add(mean_tensor.mul_scalar(update));
        state.avg_abs_activation = state
            .avg_abs_activation
            .mul_scalar(keep)
            .add(mean_abs_tensor.mul_scalar(update));
        state
    }

    fn select_features_to_replace(
        &mut self,
        module: &M,
        state: &mut ContinualBackpropState<B>,
        lr: LearningRate,
        target_lr_scale: f32,
    ) -> (Vec<usize>, ContinualBackpropTelemetry) {
        let metrics = A::feature_metrics(module);
        let age = state
            .age
            .clone()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("cbp age vec");
        let avg = state
            .avg_activation
            .clone()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("cbp avg activation vec");
        let avg_abs = state
            .avg_abs_activation
            .clone()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("cbp avg abs activation vec");
        let lr_multiplier = self.continual_backprop_lr_multiplier(lr, target_lr_scale);
        let eligible = age
            .iter()
            .enumerate()
            .filter_map(|(idx, age)| (*age >= self.config.maturity_steps as f32).then_some(idx))
            .collect::<Vec<_>>();
        let mut telemetry = ContinualBackpropTelemetry {
            optimizer_step: state.step,
            feature_count: metrics.incoming_l1.len(),
            eligible_count: eligible.len(),
            replacement_count: 0,
            replacement_budget: state.replacement_budget,
            lr_multiplier,
            paused: false,
            pause_reason: None,
            utility_min: 0.0,
            utility_mean: 0.0,
            utility_max: 0.0,
            age_mean: mean_f32(&age),
            age_max: max_f32(&age),
        };

        let paused_reason = if state.step <= self.config.warmup_steps {
            Some("warmup".to_string())
        } else if state.step < self.pause_until_step {
            self.pause_reason
                .clone()
                .or_else(|| Some("paused".to_string()))
        } else if state.step < self.cooldown_until_step {
            Some("cooldown".to_string())
        } else {
            None
        };
        if let Some(reason) = paused_reason {
            telemetry.paused = true;
            telemetry.pause_reason = Some(reason);
            return (Vec::new(), telemetry);
        }
        if state.step >= self.pause_until_step {
            self.pause_reason = None;
        }
        if eligible.is_empty() {
            return (Vec::new(), telemetry);
        }
        state.replacement_budget += self.config.replacement_rate
            * lr_multiplier
            * eligible.len() as f32
            * self.config.replace_interval_steps as f32;
        telemetry.replacement_budget = state.replacement_budget;
        let epsilon = self.config.utility_epsilon;
        let mut ranked = eligible
            .iter()
            .copied()
            .filter(|idx| {
                *idx < avg.len()
                    && *idx < avg_abs.len()
                    && *idx < metrics.incoming_l1.len()
                    && *idx < metrics.outgoing_l1.len()
            })
            .map(|idx| {
                let centered = (avg_abs[idx] - avg[idx].abs()).max(0.0);
                let score =
                    centered * metrics.outgoing_l1[idx] / metrics.incoming_l1[idx].max(epsilon);
                (idx, score)
            })
            .collect::<Vec<_>>();
        let utility_values = ranked.iter().map(|(_, score)| *score).collect::<Vec<_>>();
        telemetry.utility_min = min_f32(&utility_values);
        telemetry.utility_mean = mean_f32(&utility_values);
        telemetry.utility_max = max_f32(&utility_values);
        let n_replace = (state.replacement_budget.floor() as usize)
            .min(self.config.max_replacements_per_interval);
        if n_replace == 0 {
            return (Vec::new(), telemetry);
        }
        ranked.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let selected = ranked
            .into_iter()
            .take(n_replace.min(metrics.incoming_l1.len()))
            .map(|(idx, _)| idx)
            .collect::<Vec<_>>();
        if !selected.is_empty() {
            state.replacement_budget -= selected.len() as f32;
        }
        telemetry.replacement_count = selected.len();
        telemetry.replacement_budget = state.replacement_budget;
        (selected, telemetry)
    }

    fn reset_state_for_features(
        &self,
        mut state: ContinualBackpropState<B>,
        selected: &[usize],
        device: &B::Device,
        feature_count: usize,
    ) -> ContinualBackpropState<B> {
        let mut age = state
            .age
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("cbp age vec");
        let mut avg = state
            .avg_activation
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("cbp avg activation vec");
        let mut avg_abs = state
            .avg_abs_activation
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("cbp avg abs activation vec");
        for idx in selected.iter().copied().filter(|idx| *idx < feature_count) {
            age[idx] = 0.0;
            avg[idx] = 0.0;
            avg_abs[idx] = 0.0;
        }
        state.age = Tensor::<B, 1>::from_data(TensorData::new(age, [feature_count]), device);
        state.avg_activation =
            Tensor::<B, 1>::from_data(TensorData::new(avg, [feature_count]), device);
        state.avg_abs_activation =
            Tensor::<B, 1>::from_data(TensorData::new(avg_abs, [feature_count]), device);
        state
    }

    fn reset_optimizer_state_for_features(
        &mut self,
        reset_targets: ContinualBackpropParamResetTargets,
        selected: &[usize],
    ) {
        for param_id in reset_targets.feature_tensors_3d {
            reset_adamw_state_3d::<B>(&mut self.records, param_id, selected);
        }
        for (param_id, latent_per_head) in reset_targets.row_feature_tensors_2d {
            reset_adamw_state_2d_rows::<B>(&mut self.records, param_id, selected, latent_per_head);
        }
        for param_id in reset_targets.feature_tensors_2d {
            reset_adamw_state_2d_features::<B>(&mut self.records, param_id, selected);
        }
    }
}

struct ContinualBackpropAdamWMapper<'a, B>
where
    B: AutodiffBackend,
{
    optimizer: &'a DragonAdamW,
    records: &'a mut HashMap<ParamId, AdaptorRecord<DragonAdamW, B>>,
    grads: &'a mut GradAdaptor,
    lr: LearningRate,
    grad_clipping: Option<&'a GradientClipping>,
}

impl<'a, B> ContinualBackpropAdamWMapper<'a, B>
where
    B: AutodiffBackend,
{
    fn new(
        optimizer: &'a DragonAdamW,
        records: &'a mut HashMap<ParamId, AdaptorRecord<DragonAdamW, B>>,
        grads: &'a mut GradAdaptor,
        lr: LearningRate,
        grad_clipping: Option<&'a GradientClipping>,
    ) -> Self {
        Self {
            optimizer,
            records,
            grads,
            lr,
            grad_clipping,
        }
    }
}

impl<B> ModuleMapper<B> for ContinualBackpropAdamWMapper<'_, B>
where
    B: AutodiffBackend,
{
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        let grad = self.grads.remove(id);
        let tensor = if let Some((grad, device)) = grad {
            let is_require_grad = tensor.is_require_grad();
            let (key, record) = self.records.remove_entry(&id).unzip();
            let tensor = if tensor.device() != device {
                tensor.to_device(&device)
            } else {
                tensor
            };
            let grad = if let Some(grad_clipping) = self.grad_clipping {
                grad_clipping.clip_gradient(grad)
            } else {
                grad
            };
            let (updated, state) = self.optimizer.step(
                self.lr,
                tensor.inner(),
                grad,
                record.map(|record| DragonAdamW::to_device(record.into_state(), &device)),
            );
            if let Some(state) = state {
                self.records
                    .insert(key.unwrap_or(id), AdaptorRecord::from_state(state));
            }
            let mut updated = Tensor::from_inner(updated);
            if is_require_grad {
                updated = updated.require_grad();
            }
            updated
        } else {
            tensor
        };
        Param::from_mapped_value(id, tensor, mapper)
    }
}

impl<B, M, A> Optimizer<M, B> for ContinualBackpropAdamWOptimizer<B, M, A>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + Clone + Send,
    A: ContinualBackpropAdapter<B, M>,
{
    type Record = ContinualBackpropAdamWRecord<B>;

    fn step(&mut self, lr: LearningRate, module: M, grads: GradientsParams) -> M {
        self.step_impl(lr, module, GradAdaptor::Single(grads))
    }

    fn step_multi(&mut self, lr: LearningRate, module: M, grads: MultiGradientsParams) -> M {
        self.step_impl(lr, module, GradAdaptor::Multi(grads))
    }

    fn to_record(&self) -> Self::Record {
        ContinualBackpropAdamWRecord {
            records: self.records.clone(),
            state: self.state.clone(),
        }
    }

    fn load_record(mut self, record: Self::Record) -> Self {
        self.records = record.records;
        self.state = record.state;
        self
    }
}

impl<B, M, A> ContinualBackpropAdamWOptimizer<B, M, A>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + Clone + Send,
    A: ContinualBackpropAdapter<B, M>,
{
    fn prepare_after_module_scale(&mut self, module: &M) {
        let ids = collect_param_ids::<B, M>(module);
        self.records.retain(|id, _| ids.contains(id));
        let feature_count = A::feature_count(module);
        let device = A::device(module);
        if let Some(state) = self.state.take() {
            self.state = Some(expand_continual_backprop_state(
                state,
                feature_count,
                &device,
            ));
        }
        if let Some(state) = &mut self.state {
            state.replacement_budget = 0.0;
        }
    }

    fn pause_for_steps(&mut self, steps: usize, reason: impl Into<String>) {
        if steps == 0 {
            return;
        }
        let current_step = self.state.as_ref().map(|state| state.step).unwrap_or(0);
        self.pause_until_step = self
            .pause_until_step
            .max(current_step.saturating_add(steps));
        self.pause_reason = Some(reason.into());
    }

    fn clear_pause(&mut self) {
        self.pause_until_step = 0;
        self.pause_reason = None;
    }

    fn latest_telemetry(&self) -> Option<ContinualBackpropTelemetry> {
        self.last_telemetry.clone()
    }
}

impl<B, M, A> Optimizer<M, B> for ContinualBackpropOptimizer<B, M, A>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + Clone + Send,
    A: ContinualBackpropAdapter<B, M>,
{
    type Record = ContinualBackpropOptimizerRecord<M, B>;

    fn step(&mut self, lr: LearningRate, module: M, grads: GradientsParams) -> M {
        match &mut self.kind {
            ContinualBackpropOptimizerKind::Standard(optimizer) => {
                optimizer.step(lr, module, grads)
            }
            ContinualBackpropOptimizerKind::ContinualBackprop(optimizer) => {
                optimizer.step(lr, module, grads)
            }
        }
    }

    fn step_multi(&mut self, lr: LearningRate, module: M, grads: MultiGradientsParams) -> M {
        match &mut self.kind {
            ContinualBackpropOptimizerKind::Standard(optimizer) => {
                optimizer.step_multi(lr, module, grads)
            }
            ContinualBackpropOptimizerKind::ContinualBackprop(optimizer) => {
                optimizer.step_multi(lr, module, grads)
            }
        }
    }

    fn to_record(&self) -> Self::Record {
        match &self.kind {
            ContinualBackpropOptimizerKind::Standard(optimizer) => {
                ContinualBackpropOptimizerRecord {
                    kind: 0,
                    standard: Some(optimizer.to_record()),
                    continual_backprop: None,
                }
            }
            ContinualBackpropOptimizerKind::ContinualBackprop(optimizer) => {
                ContinualBackpropOptimizerRecord {
                    kind: 1,
                    standard: None,
                    continual_backprop: Some(optimizer.to_record()),
                }
            }
        }
    }

    fn load_record(self, record: Self::Record) -> Self {
        let kind = match (self.kind, record.kind) {
            (ContinualBackpropOptimizerKind::Standard(optimizer), 0) => {
                ContinualBackpropOptimizerKind::Standard(
                    optimizer.load_record(
                        record
                            .standard
                            .expect("continual backprop optimizer record"),
                    ),
                )
            }
            (ContinualBackpropOptimizerKind::ContinualBackprop(optimizer), 1) => {
                ContinualBackpropOptimizerKind::ContinualBackprop(
                    optimizer.load_record(
                        record
                            .continual_backprop
                            .expect("continual backprop adamw record"),
                    ),
                )
            }
            (variant, kind) => panic!(
                "continual backprop optimizer record kind {kind} does not match optimizer variant {}",
                match variant {
                    ContinualBackpropOptimizerKind::Standard(_) => "standard",
                    ContinualBackpropOptimizerKind::ContinualBackprop(_) => "continual_backprop",
                }
            ),
        };
        Self { kind }
    }
}

impl<B, M, A> ContinualBackpropOptimizer<B, M, A>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + Clone + Send,
    A: ContinualBackpropAdapter<B, M>,
{
    pub fn prepare_after_neuron_scale(&mut self, module: &M) {
        match &mut self.kind {
            ContinualBackpropOptimizerKind::Standard(_) => {}
            ContinualBackpropOptimizerKind::ContinualBackprop(optimizer) => {
                optimizer.prepare_after_module_scale(module);
            }
        }
    }

    pub fn pause_continual_backprop_for_steps(&mut self, steps: usize, reason: impl Into<String>) {
        if let ContinualBackpropOptimizerKind::ContinualBackprop(optimizer) = &mut self.kind {
            optimizer.pause_for_steps(steps, reason);
        }
    }

    pub fn clear_continual_backprop_pause(&mut self) {
        if let ContinualBackpropOptimizerKind::ContinualBackprop(optimizer) = &mut self.kind {
            optimizer.clear_pause();
        }
    }

    pub fn continual_backprop_telemetry(&self) -> Option<ContinualBackpropTelemetry> {
        match &self.kind {
            ContinualBackpropOptimizerKind::Standard(_) => None,
            ContinualBackpropOptimizerKind::ContinualBackprop(optimizer) => {
                optimizer.latest_telemetry()
            }
        }
    }
}

#[derive(Default)]
struct ParamIdCollector {
    ids: HashSet<ParamId>,
}

impl<B: BackendTrait> ModuleVisitor<B> for ParamIdCollector {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        self.ids.insert(param.id);
    }
}

fn collect_param_ids<B, M>(module: &M) -> HashSet<ParamId>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    let mut collector = ParamIdCollector::default();
    module.visit(&mut collector);
    collector.ids
}

fn mean_f32(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

fn min_f32(values: &[f32]) -> f32 {
    values.iter().copied().reduce(f32::min).unwrap_or(0.0)
}

fn max_f32(values: &[f32]) -> f32 {
    values.iter().copied().reduce(f32::max).unwrap_or(0.0)
}

fn expand_continual_backprop_state<B: AutodiffBackend>(
    mut state: ContinualBackpropState<B>,
    feature_count: usize,
    device: &B::Device,
) -> ContinualBackpropState<B> {
    state.age = expand_state_vector(state.age, feature_count, device);
    state.avg_activation = expand_state_vector(state.avg_activation, feature_count, device);
    state.avg_abs_activation = expand_state_vector(state.avg_abs_activation, feature_count, device);
    state
}

fn expand_state_vector<B: BackendTrait>(
    tensor: Tensor<B, 1>,
    feature_count: usize,
    device: &B::Device,
) -> Tensor<B, 1> {
    let mut values = tensor
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("continual backprop state vec");
    values.resize(feature_count, 0.0);
    Tensor::<B, 1>::from_data(TensorData::new(values, [feature_count]), device)
}

fn reset_adamw_state_3d<B: AutodiffBackend>(
    records: &mut HashMap<ParamId, AdaptorRecord<DragonAdamW, B>>,
    param_id: ParamId,
    selected: &[usize],
) {
    let Some(record) = records.remove(&param_id) else {
        return;
    };
    let mut state: DragonAdamWState<B::InnerBackend, 3> = record.into_state();
    state.momentum.moment_1 = zero_selected_3d_feature_tensor(state.momentum.moment_1, selected);
    state.momentum.moment_2 = zero_selected_3d_feature_tensor(state.momentum.moment_2, selected);
    state.momentum.max_moment_2 = state
        .momentum
        .max_moment_2
        .take()
        .map(|tensor| zero_selected_3d_feature_tensor(tensor, selected));
    records.insert(param_id, AdaptorRecord::from_state(state));
}

fn reset_adamw_state_2d_rows<B: AutodiffBackend>(
    records: &mut HashMap<ParamId, AdaptorRecord<DragonAdamW, B>>,
    param_id: ParamId,
    selected: &[usize],
    latent_per_head: usize,
) {
    let Some(record) = records.remove(&param_id) else {
        return;
    };
    let mut state: DragonAdamWState<B::InnerBackend, 2> = record.into_state();
    state.momentum.moment_1 =
        zero_selected_2d_rows_tensor(state.momentum.moment_1, selected, latent_per_head);
    state.momentum.moment_2 =
        zero_selected_2d_rows_tensor(state.momentum.moment_2, selected, latent_per_head);
    state.momentum.max_moment_2 = state
        .momentum
        .max_moment_2
        .take()
        .map(|tensor| zero_selected_2d_rows_tensor(tensor, selected, latent_per_head));
    records.insert(param_id, AdaptorRecord::from_state(state));
}

fn reset_adamw_state_2d_features<B: AutodiffBackend>(
    records: &mut HashMap<ParamId, AdaptorRecord<DragonAdamW, B>>,
    param_id: ParamId,
    selected: &[usize],
) {
    let Some(record) = records.remove(&param_id) else {
        return;
    };
    let mut state: DragonAdamWState<B::InnerBackend, 2> = record.into_state();
    state.momentum.moment_1 = zero_selected_2d_feature_tensor(state.momentum.moment_1, selected);
    state.momentum.moment_2 = zero_selected_2d_feature_tensor(state.momentum.moment_2, selected);
    state.momentum.max_moment_2 = state
        .momentum
        .max_moment_2
        .take()
        .map(|tensor| zero_selected_2d_feature_tensor(tensor, selected));
    records.insert(param_id, AdaptorRecord::from_state(state));
}

fn zero_selected_3d_feature_tensor<B: BackendTrait>(
    tensor: Tensor<B, 3>,
    selected: &[usize],
) -> Tensor<B, 3> {
    let device = tensor.device();
    let [dim0, dim1, dim2] = tensor.shape().dims::<3>();
    let mut values = tensor
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("3d feature tensor vec");
    for idx in selected.iter().copied().filter(|idx| *idx < dim2) {
        for outer in 0..dim0 {
            for inner in 0..dim1 {
                let offset = (outer * dim1 + inner) * dim2 + idx;
                values[offset] = 0.0;
            }
        }
    }
    Tensor::<B, 3>::from_data(TensorData::new(values, [dim0, dim1, dim2]), &device)
}

fn zero_selected_2d_rows_tensor<B: BackendTrait>(
    tensor: Tensor<B, 2>,
    selected: &[usize],
    latent_per_head: usize,
) -> Tensor<B, 2> {
    let device = tensor.device();
    let [rows, cols] = tensor.shape().dims::<2>();
    let mut values = tensor
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("2d rows tensor vec");
    if latent_per_head == 0 {
        return Tensor::<B, 2>::from_data(TensorData::new(values, [rows, cols]), &device);
    }
    let head_count = rows / latent_per_head;
    for idx in selected
        .iter()
        .copied()
        .filter(|idx| *idx < latent_per_head)
    {
        for head in 0..head_count {
            let row = head * latent_per_head + idx;
            let row_start = row * cols;
            let row_end = row_start + cols;
            values[row_start..row_end].fill(0.0);
        }
    }
    Tensor::<B, 2>::from_data(TensorData::new(values, [rows, cols]), &device)
}

fn zero_selected_2d_feature_tensor<B: BackendTrait>(
    tensor: Tensor<B, 2>,
    selected: &[usize],
) -> Tensor<B, 2> {
    let device = tensor.device();
    let [rows, cols] = tensor.shape().dims::<2>();
    let mut values = tensor
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("2d feature tensor vec");
    for idx in selected.iter().copied().filter(|idx| *idx < cols) {
        for row in 0..rows {
            values[row * cols + idx] = 0.0;
        }
    }
    Tensor::<B, 2>::from_data(TensorData::new(values, [rows, cols]), &device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_autodiff::Autodiff;
    use burn_ndarray::NdArray;

    type TestBackend = Autodiff<NdArray<f32>>;

    #[derive(Clone, Debug)]
    struct TestBatchStats {
        mean: Vec<f32>,
        mean_abs: Vec<f32>,
    }

    #[derive(Module, Debug)]
    struct TestModule<B: BackendTrait> {
        anchor: Param<Tensor<B, 1>>,
        #[module(skip)]
        stats: Option<TestBatchStats>,
    }

    #[derive(Clone)]
    struct TestAdapter;

    impl ContinualBackpropAdapter<TestBackend, TestModule<TestBackend>> for TestAdapter {
        type FreshModel = TestModule<TestBackend>;
        type BatchStats = TestBatchStats;

        fn validate_config(
            _config: &ContinualBackpropConfig,
            _fresh_model: &Self::FreshModel,
        ) -> Result<()> {
            Ok(())
        }

        fn attach_runtime(
            module: TestModule<TestBackend>,
            _config: &ContinualBackpropConfig,
        ) -> TestModule<TestBackend> {
            module
        }

        fn take_batch_stats(module: &TestModule<TestBackend>) -> Option<Self::BatchStats> {
            module.stats.clone()
        }

        fn batch_stats_mean(batch_stats: &Self::BatchStats) -> Vec<f32> {
            batch_stats.mean.clone()
        }

        fn batch_stats_mean_abs(batch_stats: &Self::BatchStats) -> Vec<f32> {
            batch_stats.mean_abs.clone()
        }

        fn feature_count(_module: &TestModule<TestBackend>) -> usize {
            2
        }

        fn device(module: &TestModule<TestBackend>) -> burn::tensor::Device<TestBackend> {
            module.anchor.val().device()
        }

        fn target_lr_scale(_module: &TestModule<TestBackend>) -> f32 {
            1.0
        }

        fn feature_metrics(_module: &TestModule<TestBackend>) -> ContinualBackpropFeatureMetrics {
            ContinualBackpropFeatureMetrics {
                incoming_l1: vec![1.0, 1.0],
                outgoing_l1: vec![1.0, 1.0],
            }
        }

        fn reinitialize_features(
            module: TestModule<TestBackend>,
            _fresh_model: &Self::FreshModel,
            _selected: &[usize],
        ) -> TestModule<TestBackend> {
            module
        }

        fn optimizer_reset_targets(
            _module: &TestModule<TestBackend>,
        ) -> ContinualBackpropParamResetTargets {
            ContinualBackpropParamResetTargets::default()
        }
    }

    fn test_module() -> TestModule<TestBackend> {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestModule {
            anchor: Param::from_tensor(Tensor::<TestBackend, 1>::zeros([1], &device)),
            stats: None,
        }
    }

    fn test_state(step: usize, replacement_budget: f32) -> ContinualBackpropState<TestBackend> {
        let device = burn::tensor::Device::<TestBackend>::default();
        ContinualBackpropState {
            step,
            replacement_budget,
            age: Tensor::<TestBackend, 1>::from_data(
                TensorData::new(vec![2048.0, 2048.0], [2]),
                &device,
            ),
            avg_activation: Tensor::<TestBackend, 1>::from_data(
                TensorData::new(vec![0.0, 0.0], [2]),
                &device,
            ),
            avg_abs_activation: Tensor::<TestBackend, 1>::from_data(
                TensorData::new(vec![1.0, 2.0], [2]),
                &device,
            ),
        }
    }

    fn test_optimizer(
        config: ContinualBackpropConfig,
    ) -> ContinualBackpropAdamWOptimizer<TestBackend, TestModule<TestBackend>, TestAdapter> {
        ContinualBackpropAdamWOptimizer {
            optimizer: DragonAdamW {
                beta_1: 0.9,
                beta_2: 0.999,
                epsilon: 1.0e-5,
                weight_decay: 0.0,
            },
            records: HashMap::new(),
            grad_clipping: None,
            state: None,
            config,
            base_learning_rate: 1.0e-3,
            fresh_model: test_module(),
            pause_until_step: 0,
            pause_reason: None,
            cooldown_until_step: 0,
            last_telemetry: None,
            _adapter: PhantomData,
            module: PhantomData,
        }
    }

    #[test]
    fn pause_preserves_replacement_budget() {
        let mut optimizer = test_optimizer(ContinualBackpropConfig::default());
        optimizer.state = Some(test_state(2048, 0.75));

        optimizer.pause_for_steps(512, "structural_stabilization");

        assert_eq!(
            optimizer
                .state
                .as_ref()
                .expect("continual backprop state")
                .replacement_budget,
            0.75
        );
    }

    #[test]
    fn utility_telemetry_updates_before_replacement_threshold() {
        let mut config = ContinualBackpropConfig {
            enabled: true,
            warmup_steps: 0,
            maturity_steps: 1,
            cooldown_steps: 0,
            replacement_rate: 1.0e-5,
            replace_interval_steps: 256,
            ..ContinualBackpropConfig::default()
        };
        config.max_replacements_per_interval = 1;
        let mut optimizer = test_optimizer(config);
        let mut state = test_state(2048, 0.0);
        let module = test_module();

        let (selected, telemetry) =
            optimizer.select_features_to_replace(&module, &mut state, 1.0e-3, 1.0);

        assert!(selected.is_empty());
        assert_eq!(telemetry.replacement_count, 0);
        assert!(telemetry.replacement_budget > 0.0);
        assert_eq!(telemetry.utility_min, 1.0);
        assert_eq!(telemetry.utility_mean, 1.5);
        assert_eq!(telemetry.utility_max, 2.0);
    }
}
