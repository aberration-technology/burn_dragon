use crate::config::train::NeuronScalingStabilizationConfig;
use crate::train::prelude::*;
use burn::tensor::activation;
use burn_dragon_core::ModelState;
use burn_dragon_time::Instant;
use std::any::{Any, TypeId};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

type StreamingStateStore = HashMap<(usize, TypeId), Box<dyn Any + Send>>;
type TeacherModelStore = HashMap<(usize, TypeId), Box<dyn Any + Send>>;

fn streaming_state_store() -> &'static Mutex<StreamingStateStore> {
    static STORE: OnceLock<Mutex<StreamingStateStore>> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn lock_streaming_state_store() -> std::sync::MutexGuard<'static, StreamingStateStore> {
    streaming_state_store()
        .lock()
        .expect("streaming tbptt runtime lock poisoned")
}

fn teacher_model_store() -> &'static Mutex<TeacherModelStore> {
    static STORE: OnceLock<Mutex<TeacherModelStore>> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn lock_teacher_model_store() -> std::sync::MutexGuard<'static, TeacherModelStore> {
    teacher_model_store()
        .lock()
        .expect("teacher model runtime lock poisoned")
}

fn next_streaming_runtime_key() -> usize {
    static NEXT_KEY: AtomicUsize = AtomicUsize::new(1);
    NEXT_KEY.fetch_add(1, Ordering::Relaxed)
}

#[derive(Clone, Debug, Default)]
struct GradientScaleSchedule {
    param_scale_rules: Arc<HashMap<burn::module::ParamId, ParamScaleScheduleRule>>,
    shared_lowrank_param_ids: Arc<Vec<burn::module::ParamId>>,
    backbone_grad_scale: Option<f32>,
    backbone_grad_scale_steps: usize,
    backbone_param_ids: Arc<HashSet<burn::module::ParamId>>,
    neuron_scale_stabilization: Option<NeuronScaleGradientStabilization>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ParamScaleScheduleRule {
    initial_scale: f32,
    final_scale: f32,
    start_step_index: usize,
    end_step_index: usize,
}

impl ParamScaleScheduleRule {
    fn constant(scale: f32) -> Self {
        Self {
            initial_scale: scale,
            final_scale: scale,
            start_step_index: 0,
            end_step_index: 0,
        }
    }

    fn for_total_steps(
        initial_scale: f32,
        schedule: Option<&crate::config::train::ModuleLrScaleScheduleConfig>,
        total_steps: usize,
    ) -> Self {
        let Some(schedule) = schedule else {
            return Self::constant(initial_scale);
        };
        let total_steps = total_steps.max(1);
        let last_step_index = total_steps.saturating_sub(1);
        let start_step_index =
            ((last_step_index as f32) * schedule.start_fraction.clamp(0.0, 1.0)).round() as usize;
        let end_step_index =
            ((last_step_index as f32) * schedule.end_fraction.clamp(0.0, 1.0)).round() as usize;
        Self {
            initial_scale,
            final_scale: schedule.final_scale,
            start_step_index,
            end_step_index,
        }
    }

    fn scale_for_step_index(self, step_index: usize) -> f32 {
        if step_index <= self.start_step_index {
            return self.initial_scale;
        }
        if step_index >= self.end_step_index {
            return self.final_scale;
        }
        if self.end_step_index <= self.start_step_index {
            return self.final_scale;
        }
        let progress = (step_index - self.start_step_index) as f32
            / (self.end_step_index - self.start_step_index) as f32;
        self.initial_scale + (self.final_scale - self.initial_scale) * progress
    }
}

#[derive(Clone, Debug)]
struct NeuronScaleGradientStabilization {
    start_step_index: usize,
    old_latent_per_head: usize,
    new_latent_per_head: usize,
    freeze_base_steps: usize,
    unfreeze_ramp_steps: usize,
    new_slice_lr_scale: f32,
    base_lr_scale_after_ramp: f32,
    base_param_ids: Arc<HashSet<burn::module::ParamId>>,
    shared_encoder: burn::module::ParamId,
    shared_encoder_v: burn::module::ParamId,
    shared_decoder: burn::module::ParamId,
}

impl NeuronScaleGradientStabilization {
    fn base_scale_for_step_index(&self, step_index: usize) -> f32 {
        let elapsed = step_index.saturating_sub(self.start_step_index);
        if elapsed < self.freeze_base_steps {
            return 0.0;
        }
        if self.unfreeze_ramp_steps == 0 {
            return self.base_lr_scale_after_ramp;
        }
        let ramp_elapsed = elapsed.saturating_sub(self.freeze_base_steps);
        if ramp_elapsed >= self.unfreeze_ramp_steps {
            return self.base_lr_scale_after_ramp;
        }
        self.base_lr_scale_after_ramp * (ramp_elapsed as f32 / self.unfreeze_ramp_steps as f32)
    }

    fn is_active_for_step_index(&self, step_index: usize) -> bool {
        let elapsed = step_index.saturating_sub(self.start_step_index);
        elapsed
            < self
                .freeze_base_steps
                .saturating_add(self.unfreeze_ramp_steps)
            || (self.base_lr_scale_after_ramp - 1.0).abs() > f32::EPSILON
            || (self.new_slice_lr_scale - 1.0).abs() > f32::EPSILON
    }
}

impl GradientScaleSchedule {
    fn from_training<B: BackendTrait>(
        model: &DragonModel<B>,
        training: &TrainingHyperparameters,
        total_steps: usize,
    ) -> Self {
        let param_scale_rules =
            Self::build_param_scale_rules(model, &training.module_lr_scales, total_steps);
        let shared_lowrank_param_ids = vec![
            model.shared_lowrank_param_ids().encoder,
            model.shared_lowrank_param_ids().encoder_v,
            model.shared_lowrank_param_ids().decoder,
        ];
        let Some(backbone_grad_scale) = training.init_transfer.backbone_grad_scale else {
            return Self {
                param_scale_rules: Arc::new(param_scale_rules),
                shared_lowrank_param_ids: Arc::new(shared_lowrank_param_ids),
                ..Self::default()
            };
        };
        let Some(backbone_grad_scale_steps) = training.init_transfer.backbone_grad_scale_steps
        else {
            return Self {
                param_scale_rules: Arc::new(param_scale_rules),
                shared_lowrank_param_ids: Arc::new(shared_lowrank_param_ids),
                ..Self::default()
            };
        };
        if backbone_grad_scale_steps == 0 || (backbone_grad_scale - 1.0).abs() <= f32::EPSILON {
            return Self {
                param_scale_rules: Arc::new(param_scale_rules),
                shared_lowrank_param_ids: Arc::new(shared_lowrank_param_ids),
                ..Self::default()
            };
        }
        let backbone_param_ids = model
            .transferred_backbone_param_ids(
                training.init_transfer.preserve_fresh_decoder,
                training.init_transfer.preserve_fresh_norm,
            )
            .into_iter()
            .collect::<HashSet<_>>();
        Self {
            param_scale_rules: Arc::new(param_scale_rules),
            shared_lowrank_param_ids: Arc::new(shared_lowrank_param_ids),
            backbone_grad_scale: Some(backbone_grad_scale),
            backbone_grad_scale_steps,
            backbone_param_ids: Arc::new(backbone_param_ids),
            neuron_scale_stabilization: None,
        }
    }

    fn build_param_scale_rules<B: BackendTrait>(
        model: &DragonModel<B>,
        entries: &[crate::config::train::ModuleLrScaleEntry],
        total_steps: usize,
    ) -> HashMap<burn::module::ParamId, ParamScaleScheduleRule> {
        let mut scales = HashMap::new();
        for entry in entries {
            for param_id in model.language_module_lr_scale_param_ids(entry.target) {
                scales.insert(
                    param_id,
                    ParamScaleScheduleRule::for_total_steps(
                        entry.scale,
                        entry.schedule.as_ref(),
                        total_steps,
                    ),
                );
            }
        }
        scales
    }

    fn mean_scale_for_param_ids(
        rules: &HashMap<burn::module::ParamId, ParamScaleScheduleRule>,
        param_ids: &[burn::module::ParamId],
        step_index: usize,
    ) -> f32 {
        if param_ids.is_empty() {
            return 1.0;
        }
        let sum = param_ids
            .iter()
            .map(|param_id| {
                rules
                    .get(param_id)
                    .copied()
                    .unwrap_or_else(|| ParamScaleScheduleRule::constant(1.0))
                    .scale_for_step_index(step_index)
            })
            .sum::<f32>();
        sum / param_ids.len() as f32
    }

    fn shared_lowrank_target_lr_scale_for_step_index(&self, step_index: usize) -> f32 {
        Self::mean_scale_for_param_ids(
            self.param_scale_rules.as_ref(),
            self.shared_lowrank_param_ids.as_ref(),
            step_index,
        )
    }

    fn with_neuron_scale_stabilization<B: BackendTrait>(
        mut self,
        model: &DragonModel<B>,
        old_latent_total: usize,
        new_latent_total: usize,
        start_step_index: usize,
        config: &NeuronScalingStabilizationConfig,
    ) -> Self {
        if new_latent_total <= old_latent_total {
            return self;
        }
        let new_latent_per_head = model.latent_per_head_capacity();
        if new_latent_per_head == 0 {
            return self;
        }
        let n_head = new_latent_total / new_latent_per_head;
        if n_head == 0 {
            return self;
        }
        let old_latent_per_head = old_latent_total / n_head;
        if old_latent_per_head >= new_latent_per_head {
            return self;
        }
        let shared = model.shared_lowrank_param_ids();
        let shared_ids = HashSet::from([shared.encoder, shared.encoder_v, shared.decoder]);
        let mut base_param_ids = collect_model_param_ids(model);
        for param_id in &shared_ids {
            base_param_ids.remove(param_id);
        }
        self.neuron_scale_stabilization = Some(NeuronScaleGradientStabilization {
            start_step_index,
            old_latent_per_head,
            new_latent_per_head,
            freeze_base_steps: config.freeze_base_steps,
            unfreeze_ramp_steps: config.unfreeze_ramp_steps,
            new_slice_lr_scale: config.new_slice_lr_scale,
            base_lr_scale_after_ramp: config.base_lr_scale_after_ramp,
            base_param_ids: Arc::new(base_param_ids),
            shared_encoder: shared.encoder,
            shared_encoder_v: shared.encoder_v,
            shared_decoder: shared.decoder,
        });
        self
    }
}

fn scale_gradients_by_schedule<B, M>(
    module: &M,
    grads: &mut GradientsParams,
    param_scale_rules: &HashMap<burn::module::ParamId, ParamScaleScheduleRule>,
    step_index: usize,
    extra_param_ids: &HashSet<burn::module::ParamId>,
    extra_scale: Option<f32>,
    neuron_scale_stabilization: Option<&NeuronScaleGradientStabilization>,
) where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    let has_static_scales = param_scale_rules
        .values()
        .any(|rule| (rule.scale_for_step_index(step_index) - 1.0).abs() > f32::EPSILON);
    let has_extra_scale = extra_scale
        .is_some_and(|scale| (scale - 1.0).abs() > f32::EPSILON && !extra_param_ids.is_empty());
    let has_neuron_scale_stabilization = neuron_scale_stabilization
        .is_some_and(|schedule| schedule.is_active_for_step_index(step_index));
    if !has_static_scales && !has_extra_scale && !has_neuron_scale_stabilization {
        return;
    }

    struct GradientScaleVisitor<'a, B: AutodiffBackend> {
        grads: &'a mut GradientsParams,
        param_scale_rules: &'a HashMap<burn::module::ParamId, ParamScaleScheduleRule>,
        step_index: usize,
        extra_param_ids: &'a HashSet<burn::module::ParamId>,
        extra_scale: Option<f32>,
        neuron_scale_stabilization: Option<&'a NeuronScaleGradientStabilization>,
        _marker: std::marker::PhantomData<B>,
    }

    impl<B: AutodiffBackend> burn::module::ModuleVisitor<B> for GradientScaleVisitor<'_, B> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            let mut scale = self
                .param_scale_rules
                .get(&param.id)
                .copied()
                .unwrap_or_else(|| ParamScaleScheduleRule::constant(1.0))
                .scale_for_step_index(self.step_index);
            if let Some(extra_scale) = self.extra_scale
                && self.extra_param_ids.contains(&param.id)
            {
                scale *= extra_scale;
            }
            if let Some(schedule) = self.neuron_scale_stabilization
                && schedule.base_param_ids.contains(&param.id)
            {
                scale *= schedule.base_scale_for_step_index(self.step_index);
            }
            if (scale - 1.0).abs() <= f32::EPSILON {
                return;
            }
            if let Some(grad) = self.grads.remove::<B::InnerBackend, D>(param.id) {
                self.grads.register(param.id, grad.mul_scalar(scale));
            }
        }
    }

    let mut visitor = GradientScaleVisitor::<B> {
        grads,
        param_scale_rules,
        step_index,
        extra_param_ids,
        extra_scale,
        neuron_scale_stabilization,
        _marker: std::marker::PhantomData,
    };
    module.visit(&mut visitor);
    if let Some(schedule) = neuron_scale_stabilization {
        scale_shared_lowrank_gradients::<B, M>(module, grads, schedule, step_index);
    }
}

#[derive(Default)]
struct ParamIdCollector {
    ids: HashSet<burn::module::ParamId>,
}

impl<B: BackendTrait> burn::module::ModuleVisitor<B> for ParamIdCollector {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        self.ids.insert(param.id);
    }
}

fn collect_model_param_ids<B: BackendTrait>(
    model: &DragonModel<B>,
) -> HashSet<burn::module::ParamId> {
    let mut collector = ParamIdCollector::default();
    model.visit(&mut collector);
    collector.ids
}

fn scale_shared_lowrank_gradients<B, M>(
    module: &M,
    grads: &mut GradientsParams,
    schedule: &NeuronScaleGradientStabilization,
    step_index: usize,
) where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    let base_scale = schedule.base_scale_for_step_index(step_index);
    let tail_scale = schedule.new_slice_lr_scale;
    if (base_scale - 1.0).abs() <= f32::EPSILON && (tail_scale - 1.0).abs() <= f32::EPSILON {
        return;
    }

    struct SharedLowrankGradientVisitor<'a, B: AutodiffBackend> {
        grads: &'a mut GradientsParams,
        schedule: &'a NeuronScaleGradientStabilization,
        base_scale: f32,
        tail_scale: f32,
        _marker: std::marker::PhantomData<B>,
    }

    impl<B: AutodiffBackend> burn::module::ModuleVisitor<B> for SharedLowrankGradientVisitor<'_, B> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            if param.id == self.schedule.shared_encoder
                || param.id == self.schedule.shared_encoder_v
            {
                if let Some(grad) = self.grads.remove::<B::InnerBackend, D>(param.id) {
                    let scaled = scale_3d_latent_tail(
                        grad,
                        self.schedule.old_latent_per_head,
                        self.schedule.new_latent_per_head,
                        self.base_scale,
                        self.tail_scale,
                    );
                    self.grads.register(param.id, scaled);
                }
            } else if param.id == self.schedule.shared_decoder
                && let Some(grad) = self.grads.remove::<B::InnerBackend, D>(param.id)
            {
                let scaled = scale_2d_headed_latent_rows(
                    grad,
                    self.schedule.old_latent_per_head,
                    self.schedule.new_latent_per_head,
                    self.base_scale,
                    self.tail_scale,
                );
                self.grads.register(param.id, scaled);
            }
        }
    }

    let mut visitor = SharedLowrankGradientVisitor::<B> {
        grads,
        schedule,
        base_scale,
        tail_scale,
        _marker: std::marker::PhantomData,
    };
    module.visit(&mut visitor);
}

fn scale_3d_latent_tail<B: BackendTrait, const D: usize>(
    tensor: Tensor<B, D>,
    old_latent_per_head: usize,
    new_latent_per_head: usize,
    base_scale: f32,
    tail_scale: f32,
) -> Tensor<B, D> {
    if D != 3 || old_latent_per_head >= new_latent_per_head {
        return tensor.mul_scalar(base_scale);
    }
    let dims: [usize; D] = tensor.shape().dims();
    let device = tensor.device();
    let heads = dims[0];
    let embd = dims[1];
    let latent = dims[2];
    if latent != new_latent_per_head {
        return tensor;
    }
    let scale =
        latent_tail_scale_vector::<B>(latent, old_latent_per_head, base_scale, tail_scale, &device)
            .reshape([1, 1, latent])
            .repeat_dim(0, heads)
            .repeat_dim(1, embd);
    (tensor.reshape([heads, embd, latent]) * scale).reshape(dims)
}

fn scale_2d_headed_latent_rows<B: BackendTrait, const D: usize>(
    tensor: Tensor<B, D>,
    old_latent_per_head: usize,
    new_latent_per_head: usize,
    base_scale: f32,
    tail_scale: f32,
) -> Tensor<B, D> {
    if D != 2 || old_latent_per_head >= new_latent_per_head {
        return tensor.mul_scalar(base_scale);
    }
    let dims: [usize; D] = tensor.shape().dims();
    let device = tensor.device();
    let rows = dims[0];
    let cols = dims[1];
    if rows % new_latent_per_head != 0 {
        return tensor;
    }
    let heads = rows / new_latent_per_head;
    let scale = latent_tail_scale_vector::<B>(
        new_latent_per_head,
        old_latent_per_head,
        base_scale,
        tail_scale,
        &device,
    )
    .reshape([1, new_latent_per_head, 1])
    .repeat_dim(0, heads)
    .repeat_dim(2, cols);
    (tensor.reshape([heads, new_latent_per_head, cols]) * scale).reshape(dims)
}

fn latent_tail_scale_vector<B: BackendTrait>(
    latent: usize,
    old_latent_per_head: usize,
    base_scale: f32,
    tail_scale: f32,
    device: &B::Device,
) -> Tensor<B, 1> {
    let base_mask = Tensor::<B, 1, Int>::arange(0..latent as i64, device)
        .float()
        .lower_elem(old_latent_per_head.min(latent) as f32)
        .float();
    base_mask.clone().mul_scalar(base_scale)
        + base_mask
            .mul_scalar(-1.0)
            .add_scalar(1.0)
            .mul_scalar(tail_scale)
}

#[derive(Debug)]
struct TeacherModelRuntime<B: BackendTrait> {
    model: DragonModel<B>,
    update_count: usize,
}

impl<B: BackendTrait> TeacherModelRuntime<B> {
    fn new(model: DragonModel<B>) -> Self {
        Self {
            model: detach_teacher_model(&model),
            update_count: 0,
        }
    }
}

fn detach_teacher_model<B: BackendTrait>(model: &DragonModel<B>) -> DragonModel<B> {
    struct DetachParamMapper<B: BackendTrait> {
        _marker: std::marker::PhantomData<B>,
    }

    impl<B: BackendTrait> burn::module::ModuleMapper<B> for DetachParamMapper<B> {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
            let (id, tensor, mapper) = param.consume();
            Param::from_mapped_value(id, tensor.detach().set_require_grad(false), mapper)
        }
    }

    model.clone().map(&mut DetachParamMapper::<B> {
        _marker: std::marker::PhantomData,
    })
}

fn ema_blend_model<B: BackendTrait>(
    teacher: &DragonModel<B>,
    online: &DragonModel<B>,
    rate: f32,
) -> DragonModel<B> {
    let rate = rate.clamp(0.0, 1.0);
    if rate <= f32::EPSILON {
        return teacher.clone();
    }
    if (rate - 1.0).abs() <= f32::EPSILON {
        return detach_teacher_model(online);
    }

    struct OnlineParamCollector<B: BackendTrait> {
        params: VecDeque<Box<dyn Any>>,
        _marker: std::marker::PhantomData<B>,
    }

    impl<B: BackendTrait> burn::module::ModuleVisitor<B> for OnlineParamCollector<B> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            self.params.push_back(Box::new(param.val().detach()));
        }
    }

    struct EmaParamMapper<B: BackendTrait> {
        params: VecDeque<Box<dyn Any>>,
        rate: f32,
        _marker: std::marker::PhantomData<B>,
    }

    impl<B: BackendTrait> burn::module::ModuleMapper<B> for EmaParamMapper<B> {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
            let online = self
                .params
                .pop_front()
                .expect("teacher EMA source parameter missing")
                .downcast::<Tensor<B, D>>()
                .unwrap_or_else(|_| panic!("teacher EMA source parameter shape mismatch"));
            let (id, tensor, mapper) = param.consume();
            let blended = (tensor.detach().mul_scalar(1.0 - self.rate)
                + online.detach().mul_scalar(self.rate))
            .detach()
            .set_require_grad(false);
            Param::from_mapped_value(id, blended, mapper)
        }
    }

    let mut collector = OnlineParamCollector::<B> {
        params: VecDeque::new(),
        _marker: std::marker::PhantomData,
    };
    online.visit(&mut collector);
    let mut mapper = EmaParamMapper::<B> {
        params: collector.params,
        rate,
        _marker: std::marker::PhantomData,
    };
    let blended = teacher.clone().map(&mut mapper);
    assert!(
        mapper.params.is_empty(),
        "teacher EMA source parameter count exceeded teacher parameter count"
    );
    blended
}

fn attach_predictive_coding_tensor<B: BackendTrait, const D: usize>(
    slot: &mut Option<Tensor<B, D>>,
) -> bool {
    let Some(tensor) = slot.take() else {
        return false;
    };
    *slot = Some(tensor.detach().require_grad());
    true
}

fn update_predictive_coding_tensor<B: AutodiffBackend, const D: usize>(
    slot: &mut Option<Tensor<B, D>>,
    grads: &B::Gradients,
    config: &burn_pc::PcInferenceConfig,
    sync_diagnostics: bool,
    stats: &mut PredictiveCodingTensorUpdateStats,
) {
    let Some(tensor) = slot.take() else {
        return;
    };
    let grad = tensor.grad(grads);
    let base = tensor.detach().inner();
    let Some(grad) = grad else {
        *slot = Some(Tensor::from_inner(base).detach());
        return;
    };
    if sync_diagnostics {
        let update = burn_pc::pc_sgd_update_with_metrics(base, grad, config);
        stats.record_synced(update.grad_norm, update.delta_rms);
        *slot = Some(Tensor::from_inner(update.tensor).detach());
    } else {
        let updated = burn_pc::pc_sgd_update(base, grad, config);
        stats.record_unsynced();
        *slot = Some(Tensor::from_inner(updated).detach());
    }
}

fn scalar_tensor_to_f64<B: BackendTrait>(tensor: Tensor<B, 1>) -> f64 {
    let values = tensor
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("scalar tensor");
    values.first().copied().unwrap_or(0.0) as f64
}

fn latent_energy_contrastive_margin_loss<B: BackendTrait>(
    positive_energy: Tensor<B, 3>,
    negative_energy: Tensor<B, 3>,
    margin: f32,
) -> Tensor<B, 1> {
    activation::softplus(positive_energy - negative_energy + margin.max(0.0), 1.0)
        .mean()
        .reshape([1])
}

fn latent_energy_monotonic_penalty<B: BackendTrait>(
    previous_energy: Tensor<B, 3>,
    current_energy: Tensor<B, 3>,
    tolerance: f32,
) -> Tensor<B, 1> {
    (current_energy - previous_energy.detach())
        .add_scalar(-tolerance.max(0.0))
        .clamp_min(0.0)
        .mean()
        .reshape([1])
}

fn latent_energy_contractivity_penalty<B: BackendTrait>(
    state: Tensor<B, 3>,
    target: Tensor<B, 3>,
    trust_radius: f32,
) -> Tensor<B, 1> {
    let target_scale = target
        .clone()
        .powf_scalar(2.0)
        .mean()
        .sqrt()
        .reshape([1])
        .detach()
        .clamp_min(1.0e-6);
    let delta_rms = (state - target).powf_scalar(2.0).mean().sqrt().reshape([1]);
    (delta_rms / target_scale)
        .add_scalar(-trust_radius.max(0.0))
        .clamp_min(0.0)
}

#[derive(Module, Debug)]
pub struct LanguageTrainModel<B: BackendTrait> {
    pub model: DragonModel<B>,
    pub tbptt_chunk_size: Option<usize>,
    #[module(skip)]
    pub pipeline_plan: Option<PipelinePlan>,
    #[module(skip)]
    pub tbptt_persist_across_steps: bool,
    #[module(skip)]
    pub objective: TrainingObjectiveConfig,
    #[module(skip)]
    input_corruption: CausalInputCorruptionConfig,
    #[module(skip)]
    logit_entropy_floor: LogitEntropyFloorConfig,
    #[module(skip)]
    repeat_unlikelihood: RepeatUnlikelihoodConfig,
    #[module(skip)]
    greedy_rollout_unlikelihood: GreedyRolloutUnlikelihoodConfig,
    #[module(skip)]
    dynamics_anchor: DynamicsAnchorConfig,
    #[module(skip)]
    predictive_coding: PredictiveCodingConfig,
    #[module(skip)]
    latent_reasoning: LatentReasoningTrainingConfig,
    #[module(skip)]
    ruliad_supervision: RuliadSupervisionConfig,
    #[module(skip)]
    latent_reasoning_capability_gate_open: Arc<AtomicBool>,
    #[module(skip)]
    greedy_rollout_recovery_active: Arc<AtomicBool>,
    #[module(skip)]
    input_vocab_size: usize,
    #[module(skip)]
    teacher_model: Option<DragonModel<B>>,
    #[module(skip)]
    streaming_runtime_key: usize,
    #[module(skip)]
    gradient_scale_schedule: GradientScaleSchedule,
    #[module(skip)]
    gradient_scale_step: Arc<AtomicUsize>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct OutputDegeneracyStats {
    pub token_count: usize,
    pub entropy_bits: f64,
    pub mean_max_probability: f64,
    pub argmax_unique_fraction: f64,
    pub eos_fraction: f64,
    pub repetition_fraction: f64,
    pub distinct_1_fraction: f64,
    pub distinct_2_fraction: f64,
    pub period_2_fraction: f64,
    pub period_3_fraction: f64,
    pub max_period_2_to_16_fraction: f64,
    pub max_period_2_to_64_fraction: f64,
    pub dominant_period_2_to_64: usize,
    pub prompt_max_period_2_to_64_fraction: f64,
    pub prompt_dominant_period_2_to_64: usize,
    pub prompt_tokens: Vec<i64>,
    pub generated_tokens: Vec<i64>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct LatentReasoningStepDiagnostics {
    pub raw_loss: f64,
    pub final_loss: f64,
    pub raw_entropy_bits: f64,
    pub final_entropy_bits: f64,
    pub final_delta_rms: f64,
    pub final_raw_cosine: f64,
    pub step_loss: Vec<f64>,
    pub step_ce_delta: Vec<f64>,
    pub step_ce_monotonic_violation_rate: Vec<f64>,
    pub step_entropy_bits: Vec<f64>,
    pub step_delta_rms: Vec<f64>,
    pub step_raw_cosine: Vec<f64>,
    pub step_energy_mean: Vec<f64>,
    pub step_energy_delta: Vec<f64>,
    pub step_energy_monotonic_violation_rate: Vec<f64>,
    pub best_energy_step: Option<usize>,
}

#[derive(Clone, Copy, Debug, Default)]
struct PredictiveCodingChunkReport {
    chunks_seen: usize,
    chunks_corrected: usize,
    inference_steps: usize,
    skipped_empty_state: usize,
    energy_before: Option<f64>,
    energy_after: Option<f64>,
    grad_norm_mean: Option<f64>,
    grad_norm_max: Option<f64>,
    delta_rms_mean: Option<f64>,
    elapsed_ns: u128,
}

impl PredictiveCodingChunkReport {
    fn has_activity(self) -> bool {
        self.chunks_seen > 0 || self.inference_steps > 0 || self.elapsed_ns > 0
    }

    fn accumulate_unsynced(&mut self, report: Self) {
        self.chunks_seen = self.chunks_seen.saturating_add(report.chunks_seen);
        self.chunks_corrected = self
            .chunks_corrected
            .saturating_add(report.chunks_corrected);
        self.inference_steps = self.inference_steps.saturating_add(report.inference_steps);
        self.skipped_empty_state = self
            .skipped_empty_state
            .saturating_add(report.skipped_empty_state);
        self.elapsed_ns = self.elapsed_ns.saturating_add(report.elapsed_ns);
    }

    fn record(self) {
        crate::train::profile::record_predictive_coding(
            self.chunks_seen,
            self.chunks_corrected,
            self.inference_steps,
            self.skipped_empty_state,
            self.energy_before,
            self.energy_after,
            self.grad_norm_mean,
            self.grad_norm_max,
            self.delta_rms_mean,
            self.elapsed_ns,
        );
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct PredictiveCodingTensorUpdateStats {
    tensor_count: usize,
    diagnostic_count: usize,
    grad_norm_sum: f64,
    grad_norm_max: f64,
    delta_rms_sum: f64,
}

impl PredictiveCodingTensorUpdateStats {
    fn record_unsynced(&mut self) {
        self.tensor_count = self.tensor_count.saturating_add(1);
    }

    fn record_synced<B: BackendTrait>(&mut self, grad_norm: Tensor<B, 1>, delta_rms: Tensor<B, 1>) {
        let grad_norm = scalar_tensor_to_f64(grad_norm);
        let delta_rms = scalar_tensor_to_f64(delta_rms);
        if grad_norm.is_finite() && delta_rms.is_finite() {
            self.tensor_count = self.tensor_count.saturating_add(1);
            self.diagnostic_count = self.diagnostic_count.saturating_add(1);
            self.grad_norm_sum += grad_norm;
            self.grad_norm_max = self.grad_norm_max.max(grad_norm);
            self.delta_rms_sum += delta_rms;
        }
    }

    fn grad_norm_mean(self) -> Option<f64> {
        (self.diagnostic_count > 0).then(|| self.grad_norm_sum / self.diagnostic_count as f64)
    }

    fn grad_norm_max(self) -> Option<f64> {
        (self.diagnostic_count > 0).then_some(self.grad_norm_max)
    }

    fn delta_rms_mean(self) -> Option<f64> {
        (self.diagnostic_count > 0).then(|| self.delta_rms_sum / self.diagnostic_count as f64)
    }
}

struct ObjectiveScoreBatch<B: BackendTrait> {
    student_inputs: Tensor<B, 2, Int>,
    student_targets: Tensor<B, 2, Int>,
    teacher_inputs: Tensor<B, 2, Int>,
    teacher_targets: Tensor<B, 2, Int>,
    mask: Tensor<B, 2, Int>,
}

#[derive(Clone, Copy)]
struct RolloutScoreConfig {
    max_completion_tokens: usize,
    group_size: usize,
    temperature: f32,
    top_k: Option<usize>,
    num_loss_tokens_to_skip: usize,
    max_reprompt_len: usize,
    reprompt_truncation: RepromptTruncation,
}

impl<B: BackendTrait> LanguageTrainModel<B> {
    pub fn new(model: DragonModel<B>) -> Self {
        Self {
            input_vocab_size: model.vocab_size(),
            model,
            tbptt_chunk_size: None,
            pipeline_plan: None,
            tbptt_persist_across_steps: false,
            objective: TrainingObjectiveConfig::NextToken,
            input_corruption: CausalInputCorruptionConfig::default(),
            logit_entropy_floor: LogitEntropyFloorConfig::default(),
            repeat_unlikelihood: RepeatUnlikelihoodConfig::default(),
            greedy_rollout_unlikelihood: GreedyRolloutUnlikelihoodConfig::default(),
            dynamics_anchor: DynamicsAnchorConfig::default(),
            predictive_coding: PredictiveCodingConfig::default(),
            latent_reasoning: LatentReasoningTrainingConfig::default(),
            ruliad_supervision: RuliadSupervisionConfig::default(),
            latent_reasoning_capability_gate_open: Arc::new(AtomicBool::new(false)),
            greedy_rollout_recovery_active: Arc::new(AtomicBool::new(false)),
            teacher_model: None,
            streaming_runtime_key: next_streaming_runtime_key(),
            gradient_scale_schedule: GradientScaleSchedule::default(),
            gradient_scale_step: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn with_tbptt_chunk_size(mut self, tbptt_chunk_size: Option<usize>) -> Self {
        self.tbptt_chunk_size = tbptt_chunk_size;
        self
    }

    pub(crate) fn map_model(mut self, f: impl FnOnce(DragonModel<B>) -> DragonModel<B>) -> Self {
        self.model = f(self.model);
        self
    }

    pub fn with_pipeline_plan(mut self, pipeline_plan: Option<PipelinePlan>) -> Self {
        self.pipeline_plan = pipeline_plan;
        self
    }

    pub fn with_tbptt_persist_across_steps(mut self, enabled: bool) -> Self {
        self.tbptt_persist_across_steps = enabled;
        self
    }

    pub fn with_training_objective(mut self, objective: TrainingObjectiveConfig) -> Self {
        self.teacher_model =
            (!objective.is_next_token()).then(|| detach_teacher_model(&self.model));
        let key = (self.streaming_runtime_key, TypeId::of::<B>());
        let mut teachers = lock_teacher_model_store();
        teachers.remove(&key);
        if let Some(teacher_model) = self.teacher_model.clone() {
            teachers.insert(key, Box::new(TeacherModelRuntime::new(teacher_model)));
        }
        self.objective = objective;
        self
    }

    pub fn with_input_corruption(mut self, config: CausalInputCorruptionConfig) -> Self {
        self.input_corruption = config;
        self
    }

    pub fn with_logit_entropy_floor(mut self, config: LogitEntropyFloorConfig) -> Self {
        self.logit_entropy_floor = config;
        self
    }

    pub fn with_repeat_unlikelihood(mut self, config: RepeatUnlikelihoodConfig) -> Self {
        self.repeat_unlikelihood = config;
        self
    }

    pub fn with_greedy_rollout_unlikelihood(
        mut self,
        config: GreedyRolloutUnlikelihoodConfig,
    ) -> Self {
        self.greedy_rollout_unlikelihood = config;
        self
    }

    pub fn with_dynamics_anchor(mut self, config: DynamicsAnchorConfig) -> Self {
        self.dynamics_anchor = config;
        if self.dynamics_anchor.enabled && self.dynamics_anchor.weight > f32::EPSILON {
            let key = (self.streaming_runtime_key, TypeId::of::<B>());
            let teacher_model = self
                .teacher_model
                .clone()
                .unwrap_or_else(|| detach_teacher_model(&self.model));
            let teacher_model = detach_teacher_model(&teacher_model);
            self.teacher_model = Some(teacher_model.clone());
            let mut teachers = lock_teacher_model_store();
            teachers
                .entry(key)
                .or_insert_with(|| Box::new(TeacherModelRuntime::new(teacher_model)));
        }
        self
    }

    pub fn with_predictive_coding(mut self, config: PredictiveCodingConfig) -> Self {
        self.predictive_coding = config;
        self
    }

    pub fn with_latent_reasoning(mut self, config: LatentReasoningTrainingConfig) -> Self {
        self.latent_reasoning = config;
        if self.latent_reasoning.enabled
            && (matches!(
                self.latent_reasoning.target_encoder,
                crate::config::LatentReasoningTargetEncoder::EmaTeacher
            ) || self.latent_reasoning.dragon_state.enabled)
        {
            let key = (self.streaming_runtime_key, TypeId::of::<B>());
            let teacher_model = self
                .teacher_model
                .clone()
                .unwrap_or_else(|| detach_teacher_model(&self.model));
            let teacher_model = detach_teacher_model(&teacher_model);
            self.teacher_model = Some(teacher_model.clone());
            let mut teachers = lock_teacher_model_store();
            teachers
                .entry(key)
                .or_insert_with(|| Box::new(TeacherModelRuntime::new(teacher_model)));
        }
        self
    }

    pub fn with_ruliad_supervision(mut self, config: RuliadSupervisionConfig) -> Self {
        self.ruliad_supervision = config;
        self
    }

    pub fn set_recovery_auxiliary_active(&self, active: bool) {
        self.greedy_rollout_recovery_active
            .store(active, Ordering::Relaxed);
    }

    pub fn set_latent_reasoning_capability_gate_open(&self, open: bool) {
        self.latent_reasoning_capability_gate_open
            .store(open, Ordering::Relaxed);
    }

    pub fn with_gradient_scale_schedule(
        mut self,
        training: &TrainingHyperparameters,
        total_steps: usize,
    ) -> Self {
        self.gradient_scale_schedule =
            GradientScaleSchedule::from_training(&self.model, training, total_steps);
        self
    }

    pub fn gradient_scale_step_index(&self) -> usize {
        self.gradient_scale_step
            .load(Ordering::Relaxed)
            .saturating_sub(1)
    }

    pub fn with_neuron_scale_stabilization(
        mut self,
        old_latent_total: usize,
        new_latent_total: usize,
        config: &NeuronScalingStabilizationConfig,
    ) -> Self {
        let start_step_index = self.gradient_scale_step_index().saturating_add(1);
        self.gradient_scale_schedule = self
            .gradient_scale_schedule
            .with_neuron_scale_stabilization(
                &self.model,
                old_latent_total,
                new_latent_total,
                start_step_index,
                config,
            );
        self
    }

    pub fn continual_backprop_target_lr_scale(&self) -> f32 {
        let step_index = self
            .gradient_scale_step
            .load(Ordering::Relaxed)
            .saturating_sub(1);
        self.gradient_scale_schedule
            .shared_lowrank_target_lr_scale_for_step_index(step_index)
    }

    fn apply_gradient_scale_schedule(&self, mut grads: GradientsParams) -> GradientsParams
    where
        B: AutodiffBackend,
    {
        let step = self.gradient_scale_step.fetch_add(1, Ordering::Relaxed) + 1;
        let step_index = step.saturating_sub(1);
        let extra_scale = self
            .gradient_scale_schedule
            .backbone_grad_scale
            .filter(|_| step <= self.gradient_scale_schedule.backbone_grad_scale_steps);
        scale_gradients_by_schedule::<B, _>(
            self,
            &mut grads,
            self.gradient_scale_schedule.param_scale_rules.as_ref(),
            step_index,
            self.gradient_scale_schedule.backbone_param_ids.as_ref(),
            extra_scale,
            self.gradient_scale_schedule
                .neuron_scale_stabilization
                .as_ref(),
        );
        grads
    }

    fn effective_tbptt_chunk_size(&self, block_size: usize) -> Option<usize> {
        self.tbptt_chunk_size
            .filter(|chunk_size| *chunk_size > 0 && *chunk_size < block_size)
    }

    fn load_step_state(&self, reset_stream_state: bool) -> ModelState<B> {
        if !self.tbptt_persist_across_steps {
            return self.model.init_state_ephemeral();
        }
        let key = (self.streaming_runtime_key, TypeId::of::<B>());
        let mut runtime = lock_streaming_state_store();
        if reset_stream_state {
            runtime.remove(&key);
        }
        runtime
            .remove(&key)
            .and_then(|state| state.downcast::<ModelState<B>>().ok().map(|state| *state))
            .unwrap_or_else(|| self.model.init_state())
    }

    fn store_step_state(&self, mut state: ModelState<B>) {
        if !self.tbptt_persist_across_steps {
            return;
        }
        state.detach_in_place();
        let key = (self.streaming_runtime_key, TypeId::of::<B>());
        let mut runtime = lock_streaming_state_store();
        runtime.insert(key, Box::new(state));
    }

    #[cfg(test)]
    fn peek_step_state_for_test(&self) -> Option<ModelState<B>> {
        lock_streaming_state_store()
            .get(&(self.streaming_runtime_key, TypeId::of::<B>()))
            .and_then(|state| state.downcast_ref::<ModelState<B>>().cloned())
    }

    fn slice_tokens(
        tensor: Tensor<B, 2, Int>,
        batch_size: usize,
        start: usize,
        end: usize,
    ) -> Tensor<B, 2, Int> {
        tensor.slice([0..batch_size, start..end])
    }

    fn slice_batch(
        tensor: Tensor<B, 2, Int>,
        batch_start: usize,
        batch_end: usize,
    ) -> Tensor<B, 2, Int> {
        let [_batch_size, block_size] = tensor.shape().dims();
        tensor.slice([batch_start..batch_end, 0..block_size])
    }

    fn pipeline_enabled(&self) -> bool {
        self.pipeline_plan.is_some()
    }

    fn language_loss_from_hidden(
        &self,
        hidden: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
        loss_mask: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 1> {
        self.language_loss_from_hidden_for_latent_step(
            hidden,
            targets,
            loss_mask,
            self.model.latent_reasoning_config().max_steps,
        )
    }

    fn language_loss_from_hidden_for_latent_step(
        &self,
        hidden: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
        loss_mask: Option<Tensor<B, 2, Int>>,
        step: usize,
    ) -> Tensor<B, 1> {
        if let Some(mask) = loss_mask {
            return masked_token_mean(
                self.model
                    .language_token_losses_from_hidden_for_latent_step(hidden, targets, step),
                Some(mask),
            );
        }
        self.model
            .language_loss_from_hidden_for_latent_step(hidden, targets, step)
    }

    fn language_loss_from_logits(
        &self,
        logits: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
        loss_mask: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 1> {
        if let Some(mask) = loss_mask {
            return masked_token_mean(
                self.model
                    .language_token_losses_from_logits(logits, targets),
                Some(mask),
            );
        }
        self.model.language_loss_from_logits(logits, targets)
    }

    fn forward_hidden_with_pipeline_for_objective(
        &self,
        inputs: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let plan = self
            .pipeline_plan
            .as_ref()
            .expect("pipeline objective forward requires a pipeline plan");
        let [batch_size, _block_size] = inputs.shape().dims();
        let ranges = split_microbatch_ranges(batch_size, plan.microbatches)
            .expect("pipeline objective execution requires batch_size >= microbatches");
        let chunk_inputs = ranges
            .iter()
            .map(|range| Self::slice_batch(inputs.clone(), range.start, range.end))
            .collect::<Vec<_>>();

        let mut chunk_states = (0..plan.microbatches)
            .map(|_| self.model.init_state_ephemeral())
            .collect::<Vec<_>>();
        let mut pipeline_states = vec![None; plan.microbatches];

        for event in plan.events.iter().filter(|event| {
            matches!(
                event.kind,
                burn_dragon_train::train::pipeline::PipelineEventKind::Forward
            )
        }) {
            let microbatch_id = event.microbatch_id;
            if pipeline_states[microbatch_id].is_none() {
                pipeline_states[microbatch_id] = Some(
                    self.model
                        .begin_language_pipeline(chunk_inputs[microbatch_id].clone()),
                );
            }
            let assignment = plan.assignment(event.virtual_stage_id).clone();
            let state = &mut chunk_states[microbatch_id];
            let stage_state = pipeline_states[microbatch_id]
                .take()
                .expect("microbatch stage state");
            pipeline_states[microbatch_id] =
                Some(self.model.forward_language_pipeline_stage_with_state(
                    stage_state,
                    state,
                    assignment.layer_range.clone(),
                    None,
                ));
        }

        let mut hidden_chunks = Vec::with_capacity(plan.microbatches);
        for microbatch_id in 0..plan.microbatches {
            hidden_chunks.push(
                self.model.finish_language_pipeline_hidden_with_state(
                    pipeline_states[microbatch_id]
                        .take()
                        .expect("pipeline state after scheduled forward"),
                    &mut chunk_states[microbatch_id],
                ),
            );
        }
        Tensor::cat(hidden_chunks, 0)
    }

    fn forward_hidden_for_objective(&self, inputs: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        if self.pipeline_enabled() {
            self.forward_hidden_with_pipeline_for_objective(inputs)
        } else {
            self.model.forward_hidden(inputs)
        }
    }

    fn current_teacher_model(&self) -> DragonModel<B> {
        let key = (self.streaming_runtime_key, TypeId::of::<B>());
        let teachers = lock_teacher_model_store();
        if let Some(runtime) = teachers
            .get(&key)
            .and_then(|runtime| runtime.downcast_ref::<TeacherModelRuntime<B>>())
        {
            return runtime.model.clone();
        }
        self.teacher_model
            .clone()
            .unwrap_or_else(|| self.model.clone())
    }

    fn objective_teacher_update_rate(&self) -> f32 {
        let objective_rate = match &self.objective {
            TrainingObjectiveConfig::NextToken => 0.0,
            TrainingObjectiveConfig::Sdft(config) => config.teacher_update_rate,
            TrainingObjectiveConfig::Sdpo(config) => config.teacher_update_rate,
            TrainingObjectiveConfig::SdftSdpo(config) => {
                let sdft_weight = config.sdft_weight.max(0.0);
                let sdpo_weight = config.sdpo_weight.max(0.0);
                let weight_sum = sdft_weight + sdpo_weight;
                if weight_sum <= f32::EPSILON {
                    0.0
                } else {
                    (config.sdft.teacher_update_rate * sdft_weight
                        + config.sdpo.teacher_update_rate * sdpo_weight)
                        / weight_sum
                }
            }
        };
        let anchor_rate =
            if self.dynamics_anchor.enabled && self.dynamics_anchor.weight > f32::EPSILON {
                self.dynamics_anchor.teacher_update_rate.clamp(0.0, 1.0)
            } else {
                0.0
            };
        let latent_rate = if self.latent_reasoning.enabled
            && (matches!(
                self.latent_reasoning.target_encoder,
                crate::config::LatentReasoningTargetEncoder::EmaTeacher
            ) || self.latent_reasoning.dragon_state.enabled)
        {
            self.latent_reasoning.teacher_update_rate.clamp(0.0, 1.0)
        } else {
            0.0
        };
        objective_rate.max(anchor_rate).max(latent_rate)
    }

    fn update_teacher_runtime(&self) {
        let rate = self.objective_teacher_update_rate().clamp(0.0, 1.0);
        if rate <= f32::EPSILON {
            return;
        };
        let key = (self.streaming_runtime_key, TypeId::of::<B>());
        let mut teachers = lock_teacher_model_store();
        let runtime = teachers.entry(key).or_insert_with(|| {
            Box::new(TeacherModelRuntime::new(
                self.teacher_model
                    .clone()
                    .unwrap_or_else(|| self.model.clone()),
            ))
        });
        let Some(runtime) = runtime.downcast_mut::<TeacherModelRuntime<B>>() else {
            return;
        };
        runtime.model = ema_blend_model(&runtime.model, &self.model, rate);
        runtime.update_count = runtime.update_count.saturating_add(1);
    }

    pub(crate) fn validation_loss_and_output_degeneracy(
        &self,
        batch: SequenceBatch<B>,
        probe_tokens: usize,
        eos_id: Option<i64>,
    ) -> (Tensor<B, 1>, Option<OutputDegeneracyStats>) {
        let output = <Self as ValidStep>::step(self, batch.clone());
        let loss_value: LossValue<B> = output.adapt();
        let stats = self.output_degeneracy_for_batch(batch, probe_tokens, eos_id);
        (loss_value.value(), stats)
    }

    pub(crate) fn latent_reasoning_step_diagnostics(
        &self,
        batch: SequenceBatch<B>,
    ) -> Option<LatentReasoningStepDiagnostics> {
        if !self.model.latent_reasoning_enabled()
            || self.pipeline_enabled()
            || self.model.uses_factorized_language_head()
        {
            return None;
        }
        let raw = self.model.forward_hidden_raw(batch.inputs);
        let output = self.model.reason_hidden(raw.clone());
        if output.step_hiddens.is_empty() {
            return None;
        }
        let raw_loss = scalar_tensor_to_f64(self.language_loss_from_hidden_for_latent_step(
            raw.clone(),
            batch.targets.clone(),
            batch.loss_mask.clone(),
            0,
        ));
        let raw_entropy_bits = self.hidden_entropy_bits_for_latent_step(raw.clone(), 0);
        let final_loss = scalar_tensor_to_f64(self.language_loss_from_hidden_for_latent_step(
            output.final_hidden.clone(),
            batch.targets.clone(),
            batch.loss_mask.clone(),
            output.steps_used,
        ));
        let final_entropy_bits = self
            .hidden_entropy_bits_for_latent_step(output.final_hidden.clone(), output.steps_used);
        let final_delta_rms = Self::tensor_delta_rms(raw.clone(), output.final_hidden.clone());
        let final_raw_cosine = Self::tensor_cosine(raw.clone(), output.final_hidden.clone());
        let mut previous = raw.clone();
        let mut previous_ce = raw_loss;
        let mut step_loss = Vec::with_capacity(output.step_hiddens.len());
        let mut step_ce_delta = Vec::with_capacity(output.step_hiddens.len());
        let mut step_ce_monotonic_violation_rate = Vec::with_capacity(output.step_hiddens.len());
        let mut step_entropy_bits = Vec::with_capacity(output.step_hiddens.len());
        let mut step_delta_rms = Vec::with_capacity(output.step_hiddens.len());
        let mut step_raw_cosine = Vec::with_capacity(output.step_hiddens.len());
        let mut step_energy_mean = Vec::with_capacity(output.energies.len());
        let mut step_energy_delta = Vec::with_capacity(output.energies.len());
        let mut step_energy_monotonic_violation_rate = Vec::with_capacity(output.energies.len());
        let mut previous_energy = self.model.latent_energy_from_hidden(raw.clone());
        for (index, hidden) in output.step_hiddens.into_iter().enumerate() {
            let step = index.saturating_add(1);
            let loss = scalar_tensor_to_f64(self.language_loss_from_hidden_for_latent_step(
                hidden.clone(),
                batch.targets.clone(),
                batch.loss_mask.clone(),
                step,
            ));
            let ce_delta = loss - previous_ce;
            step_loss.push(loss);
            step_ce_delta.push(ce_delta);
            step_ce_monotonic_violation_rate.push(f64::from(ce_delta > 1.0e-6));
            previous_ce = loss;
            step_entropy_bits.push(self.hidden_entropy_bits_for_latent_step(hidden.clone(), step));
            step_delta_rms.push(Self::tensor_delta_rms(previous.clone(), hidden.clone()));
            step_raw_cosine.push(Self::tensor_cosine(raw.clone(), hidden.clone()));
            previous = hidden;
            if let Some(energy) = output.energies.get(index) {
                step_energy_mean.push(scalar_tensor_to_f64(energy.clone().mean().reshape([1])));
                if let Some(prev_energy) = previous_energy.as_ref() {
                    let energy_delta = energy.clone() - prev_energy.clone();
                    step_energy_delta.push(scalar_tensor_to_f64(
                        energy_delta.clone().mean().reshape([1]),
                    ));
                    let violations = energy_delta.greater_elem(0.0).float().mean().reshape([1]);
                    step_energy_monotonic_violation_rate.push(scalar_tensor_to_f64(violations));
                }
                previous_energy = Some(energy.clone());
            }
        }
        let best_energy_step = step_energy_mean
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, value)| value.is_finite())
            .min_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
            .map(|(index, _)| index.saturating_add(1));
        Some(LatentReasoningStepDiagnostics {
            raw_loss,
            final_loss,
            raw_entropy_bits,
            final_entropy_bits,
            final_delta_rms,
            final_raw_cosine,
            step_loss,
            step_ce_delta,
            step_ce_monotonic_violation_rate,
            step_entropy_bits,
            step_delta_rms,
            step_raw_cosine,
            step_energy_mean,
            step_energy_delta,
            step_energy_monotonic_violation_rate,
            best_energy_step,
        })
    }

    fn hidden_entropy_bits(&self, hidden: Tensor<B, 3>) -> f64 {
        self.hidden_entropy_bits_for_latent_step(
            hidden,
            self.model.latent_reasoning_config().max_steps,
        )
    }

    fn hidden_entropy_bits_for_latent_step(&self, hidden: Tensor<B, 3>, step: usize) -> f64 {
        let logits = self.model.logits_from_hidden_for_latent_step(hidden, step);
        let [batch, time, vocab] = logits.shape().dims::<3>();
        if batch == 0 || time == 0 || vocab == 0 {
            return 0.0;
        }
        let log_probs = activation::log_softmax(logits.reshape([batch * time, vocab]), 1);
        let entropy = (log_probs.clone().exp() * log_probs)
            .sum_dim(1)
            .mean()
            .mul_scalar(-1.0 / std::f32::consts::LN_2);
        scalar_tensor_to_f64(entropy.reshape([1]))
    }

    fn tensor_delta_rms(lhs: Tensor<B, 3>, rhs: Tensor<B, 3>) -> f64 {
        scalar_tensor_to_f64((rhs - lhs).powf_scalar(2.0).mean().sqrt().reshape([1]))
    }

    fn tensor_cosine(lhs: Tensor<B, 3>, rhs: Tensor<B, 3>) -> f64 {
        let dot = scalar_tensor_to_f64((lhs.clone() * rhs.clone()).mean().reshape([1]));
        let lhs_rms = scalar_tensor_to_f64(lhs.powf_scalar(2.0).mean().sqrt().reshape([1]));
        let rhs_rms = scalar_tensor_to_f64(rhs.powf_scalar(2.0).mean().sqrt().reshape([1]));
        let denom = (lhs_rms * rhs_rms).max(1.0e-12);
        dot / denom
    }

    fn output_degeneracy_for_batch(
        &self,
        batch: SequenceBatch<B>,
        probe_tokens: usize,
        eos_id: Option<i64>,
    ) -> Option<OutputDegeneracyStats> {
        if probe_tokens == 0
            || self.pipeline_enabled()
            || self.model.uses_factorized_language_head()
        {
            return None;
        }
        let [batch_size, block_size] = batch.inputs.shape().dims::<2>();
        if batch_size == 0 || block_size == 0 {
            return None;
        }
        let probe_batch = batch_size.min(4);
        let generated_tokens = probe_tokens.max(1);
        let prompt_time = block_size.min(probe_tokens.clamp(1, 32));
        let prompt_available = block_size.saturating_sub(prompt_time);
        let device = batch.inputs.device();
        let mut accumulator = OutputDegeneracyAccumulator::new(eos_id);
        for prompt_index in 0..probe_batch {
            let prompt_start =
                validation_degeneracy_prompt_start(prompt_index, probe_batch, prompt_available);
            let inputs = batch.inputs.clone().slice([
                prompt_index..prompt_index + 1,
                prompt_start..(prompt_start + prompt_time),
            ]);
            let prompt_tokens = inputs
                .clone()
                .to_data()
                .convert::<i64>()
                .into_vec::<i64>()
                .expect("validation degeneracy prompt tokens");
            accumulator.record_prompt_tokens(prompt_tokens);
            let summary_event_mask = batch.summary_event_mask.clone().map(|mask| {
                mask.slice([
                    prompt_index..prompt_index + 1,
                    prompt_start..(prompt_start + prompt_time),
                ])
            });
            let mut state = self.model.init_state();
            let logits = if let Some(mask) = summary_event_mask {
                self.model
                    .forward_with_state_and_summary_event_mask(inputs, mask, &mut state)
            } else {
                self.model.forward_with_state(inputs, &mut state)
            };
            let [_, time, vocab] = logits.shape().dims::<3>();
            if time == 0 || vocab == 0 {
                continue;
            }
            let mut last_logits = logits.slice_dim(1, (time - 1)..time).reshape([vocab]);
            for _ in 0..generated_tokens {
                let Some(step) = output_degeneracy_step_from_logits(last_logits.clone()) else {
                    continue;
                };
                accumulator.record(step);
                let next = step.argmax as i64;
                accumulator.record_generated_token(next);
                let next_tensor =
                    Tensor::<B, 2, Int>::from_data(TensorData::new(vec![next], [1, 1]), &device);
                let logits = self.model.forward_with_state(next_tensor, &mut state);
                let [_, time, vocab] = logits.shape().dims::<3>();
                if time == 0 || vocab == 0 {
                    break;
                }
                last_logits = logits.slice_dim(1, (time - 1)..time).reshape([vocab]);
            }
        }
        Some(accumulator.finish()).filter(|stats| stats.token_count > 0)
    }

    #[cfg(test)]
    fn teacher_update_count_for_test(&self) -> Option<usize> {
        let key = (self.streaming_runtime_key, TypeId::of::<B>());
        lock_teacher_model_store()
            .get(&key)
            .and_then(|runtime| runtime.downcast_ref::<TeacherModelRuntime<B>>())
            .map(|runtime| runtime.update_count)
    }

    fn assert_flat_logits_for_rollout_objective(&self) {
        assert_flat_logits_for_rollout_objective(
            &self.objective,
            self.model.uses_factorized_language_head(),
        );
    }

    fn causal_input_corruption_probability(&self) -> f32 {
        if !self.input_corruption.enabled {
            return 0.0;
        }
        let probability = self.input_corruption.probability.clamp(0.0, 1.0);
        if probability <= f32::EPSILON {
            return 0.0;
        }
        let step_index = self.gradient_scale_step.load(Ordering::Relaxed);
        if step_index < self.input_corruption.warmup_steps {
            return 0.0;
        }
        if self.input_corruption.ramp_steps == 0 {
            return probability;
        }
        let ramp_step = step_index
            .saturating_sub(self.input_corruption.warmup_steps)
            .saturating_add(1);
        let ramp = (ramp_step as f32 / self.input_corruption.ramp_steps as f32).clamp(0.0, 1.0);
        probability * ramp
    }

    fn scheduled_weight(
        enabled: bool,
        weight: f32,
        warmup_steps: usize,
        ramp_steps: usize,
        step_index: usize,
    ) -> f32 {
        if !enabled || weight <= f32::EPSILON {
            return 0.0;
        }
        if step_index < warmup_steps {
            return 0.0;
        }
        if ramp_steps == 0 {
            return weight;
        }
        let ramp_step = step_index.saturating_sub(warmup_steps).saturating_add(1);
        let ramp = (ramp_step as f32 / ramp_steps as f32).clamp(0.0, 1.0);
        weight * ramp
    }

    fn scheduled_weight_on_cadence(
        enabled: bool,
        weight: f32,
        warmup_steps: usize,
        ramp_steps: usize,
        every_steps: usize,
        step_index: usize,
    ) -> f32 {
        if every_steps > 1 && !step_index.is_multiple_of(every_steps) {
            return 0.0;
        }
        Self::scheduled_weight(enabled, weight, warmup_steps, ramp_steps, step_index)
    }

    fn repeat_unlikelihood_weight(&self) -> f32 {
        Self::scheduled_weight_on_cadence(
            self.repeat_unlikelihood.enabled,
            self.repeat_unlikelihood.weight,
            self.repeat_unlikelihood.warmup_steps,
            self.repeat_unlikelihood.ramp_steps,
            self.repeat_unlikelihood.every_steps,
            self.gradient_scale_step.load(Ordering::Relaxed),
        )
    }

    fn repeat_cycle_weight(&self) -> f32 {
        Self::scheduled_weight_on_cadence(
            self.repeat_unlikelihood.enabled,
            self.repeat_unlikelihood.cycle_weight,
            self.repeat_unlikelihood.warmup_steps,
            self.repeat_unlikelihood.ramp_steps,
            self.repeat_unlikelihood.every_steps,
            self.gradient_scale_step.load(Ordering::Relaxed),
        )
    }

    fn repeat_cycle_margin_weight(&self) -> f32 {
        Self::scheduled_weight_on_cadence(
            self.repeat_unlikelihood.enabled,
            self.repeat_unlikelihood.cycle_margin_weight,
            self.repeat_unlikelihood.warmup_steps,
            self.repeat_unlikelihood.ramp_steps,
            self.repeat_unlikelihood.every_steps,
            self.gradient_scale_step.load(Ordering::Relaxed),
        )
    }

    fn logit_entropy_floor_weight(&self) -> f32 {
        Self::scheduled_weight_on_cadence(
            self.logit_entropy_floor.enabled,
            self.logit_entropy_floor.weight,
            self.logit_entropy_floor.warmup_steps,
            self.logit_entropy_floor.ramp_steps,
            self.logit_entropy_floor.every_steps,
            self.gradient_scale_step.load(Ordering::Relaxed),
        )
    }

    fn logit_marginal_entropy_floor_weight(&self) -> f32 {
        Self::scheduled_weight_on_cadence(
            self.logit_entropy_floor.enabled,
            self.logit_entropy_floor.marginal_weight,
            self.logit_entropy_floor.warmup_steps,
            self.logit_entropy_floor.ramp_steps,
            self.logit_entropy_floor.every_steps,
            self.gradient_scale_step.load(Ordering::Relaxed),
        )
    }

    fn logit_target_coverage_weight(&self) -> f32 {
        Self::scheduled_weight_on_cadence(
            self.logit_entropy_floor.enabled,
            self.logit_entropy_floor.target_coverage_weight,
            self.logit_entropy_floor.warmup_steps,
            self.logit_entropy_floor.ramp_steps,
            self.logit_entropy_floor.every_steps,
            self.gradient_scale_step.load(Ordering::Relaxed),
        )
    }

    fn dynamics_anchor_weight(&self) -> f32 {
        Self::scheduled_weight_on_cadence(
            self.dynamics_anchor.enabled,
            self.dynamics_anchor.weight,
            self.dynamics_anchor.warmup_steps,
            self.dynamics_anchor.ramp_steps,
            self.dynamics_anchor.every_steps,
            self.gradient_scale_step.load(Ordering::Relaxed),
        )
    }

    fn dynamics_anchor_teacher_model(&self) -> Option<DragonModel<B>> {
        if self.dynamics_anchor_weight() <= f32::EPSILON
            || self.pipeline_enabled()
            || self.model.uses_factorized_language_head()
        {
            return None;
        }
        Some(self.current_teacher_model())
    }

    fn latent_dragon_state_consistency_active(&self) -> bool {
        self.latent_reasoning.enabled
            && self.latent_reasoning.dragon_state.enabled
            && !self.pipeline_enabled()
    }

    fn recurrent_teacher_model(&self) -> Option<(DragonModel<B>, bool)> {
        if let Some(teacher) = self.dynamics_anchor_teacher_model() {
            return Some((teacher, true));
        }
        self.latent_dragon_state_consistency_active()
            .then(|| (self.current_teacher_model(), false))
    }

    fn dynamics_anchor_mask(
        &self,
        loss_mask: Option<Tensor<B, 2, Int>>,
    ) -> Option<Tensor<B, 2, Int>> {
        match self.dynamics_anchor.mask {
            DynamicsAnchorMask::AllTokens => None,
            DynamicsAnchorMask::TargetTokens => loss_mask,
            DynamicsAnchorMask::ContextTokens => loss_mask.map(|mask| mask.equal_elem(0).int()),
        }
    }

    fn dynamics_anchor_loss_from_log_probs(
        &self,
        student_log_probs: Tensor<B, 3>,
        teacher_logits: Tensor<B, 3>,
        loss_mask: Option<Tensor<B, 2, Int>>,
    ) -> Option<Tensor<B, 1>> {
        let weight = self.dynamics_anchor_weight();
        if weight <= f32::EPSILON {
            return None;
        }
        let teacher_log_probs = log_probs_from_logits(teacher_logits.detach());
        let per_token = self_distillation_per_token_from_log_probs(
            student_log_probs,
            teacher_log_probs,
            self.dynamics_anchor.kl,
        );
        Some(masked_token_mean(per_token, self.dynamics_anchor_mask(loss_mask)).mul_scalar(weight))
    }

    fn teacher_logits_with_state(
        teacher: &DragonModel<B>,
        inputs: Tensor<B, 2, Int>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
        state: &mut ModelState<B>,
    ) -> Tensor<B, 3> {
        if let Some(mask) = summary_event_mask {
            teacher.forward_with_state_and_summary_event_mask(inputs, mask, state)
        } else {
            teacher.forward_with_state(inputs, state)
        }
        .detach()
    }

    fn teacher_forward_with_state(
        teacher: &DragonModel<B>,
        emit_logits: bool,
        inputs: Tensor<B, 2, Int>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
        state: &mut ModelState<B>,
    ) -> Option<Tensor<B, 3>> {
        if emit_logits {
            return Some(Self::teacher_logits_with_state(
                teacher,
                inputs,
                summary_event_mask,
                state,
            ));
        }
        if let Some(mask) = summary_event_mask {
            teacher.forward_hidden_with_state_and_summary_event_mask(inputs, mask, state);
        } else {
            teacher.forward_hidden_with_state(inputs, state);
        }
        None
    }

    fn predictive_coding_active_for_chunk(&self, step_index: usize, chunk_index: usize) -> bool {
        self.predictive_coding.enabled
            && step_index >= self.predictive_coding.warmup_steps
            && chunk_index.is_multiple_of(self.predictive_coding.apply_every_chunks.max(1))
    }

    fn predictive_coding_inference_config(&self) -> burn_pc::PcInferenceConfig {
        burn_pc::PcInferenceConfig {
            steps: self.predictive_coding.steps,
            step_size: self.predictive_coding.step_size,
            latent_decay: self.predictive_coding.latent_decay,
            max_grad_norm: self.predictive_coding.max_grad_norm,
            eps: self.predictive_coding.eps,
        }
    }

    fn predictive_coding_state_has_latents(
        state: &ModelState<B>,
        scope: PredictiveCodingStateScope,
    ) -> bool {
        state.layers.iter().any(|layer| {
            let core = layer.rho.is_some() || layer.y_neuron_state.is_some();
            core || (matches!(scope, PredictiveCodingStateScope::All)
                && (layer.sequence_aux.is_some()
                    || layer.mamba_angle_state.is_some()
                    || layer.mamba_k_state.is_some()
                    || layer.mamba_v_state.is_some()
                    || layer.clocked_slow_hidden.is_some()
                    || layer.summary_memory_hidden.is_some()))
        })
    }

    fn attach_predictive_coding_state_latents(
        state: &mut ModelState<B>,
        scope: PredictiveCodingStateScope,
    ) -> bool {
        let mut attached = false;
        for layer in &mut state.layers {
            layer.rho_norm = None;
            attached |= attach_predictive_coding_tensor(&mut layer.rho);
            attached |= attach_predictive_coding_tensor(&mut layer.y_neuron_state);
            if matches!(scope, PredictiveCodingStateScope::All) {
                attached |= attach_predictive_coding_tensor(&mut layer.sequence_aux);
                attached |= attach_predictive_coding_tensor(&mut layer.mamba_angle_state);
                attached |= attach_predictive_coding_tensor(&mut layer.mamba_k_state);
                attached |= attach_predictive_coding_tensor(&mut layer.mamba_v_state);
                attached |= attach_predictive_coding_tensor(&mut layer.clocked_slow_hidden);
                attached |= attach_predictive_coding_tensor(&mut layer.summary_memory_hidden);
            }
        }
        attached
    }

    fn update_predictive_coding_state_latents(
        state: &mut ModelState<B>,
        grads: &B::Gradients,
        config: &burn_pc::PcInferenceConfig,
        sync_diagnostics: bool,
        scope: PredictiveCodingStateScope,
    ) -> PredictiveCodingTensorUpdateStats
    where
        B: AutodiffBackend,
    {
        let mut stats = PredictiveCodingTensorUpdateStats::default();
        for layer in &mut state.layers {
            layer.rho_norm = None;
            update_predictive_coding_tensor(
                &mut layer.rho,
                grads,
                config,
                sync_diagnostics,
                &mut stats,
            );
            update_predictive_coding_tensor(
                &mut layer.y_neuron_state,
                grads,
                config,
                sync_diagnostics,
                &mut stats,
            );
            if matches!(scope, PredictiveCodingStateScope::All) {
                update_predictive_coding_tensor(
                    &mut layer.sequence_aux,
                    grads,
                    config,
                    sync_diagnostics,
                    &mut stats,
                );
                update_predictive_coding_tensor(
                    &mut layer.mamba_angle_state,
                    grads,
                    config,
                    sync_diagnostics,
                    &mut stats,
                );
                update_predictive_coding_tensor(
                    &mut layer.mamba_k_state,
                    grads,
                    config,
                    sync_diagnostics,
                    &mut stats,
                );
                update_predictive_coding_tensor(
                    &mut layer.mamba_v_state,
                    grads,
                    config,
                    sync_diagnostics,
                    &mut stats,
                );
                update_predictive_coding_tensor(
                    &mut layer.clocked_slow_hidden,
                    grads,
                    config,
                    sync_diagnostics,
                    &mut stats,
                );
                update_predictive_coding_tensor(
                    &mut layer.summary_memory_hidden,
                    grads,
                    config,
                    sync_diagnostics,
                    &mut stats,
                );
            }
        }
        stats
    }

    fn predictive_coding_energy_with_state(
        &self,
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
        loss_mask: Option<Tensor<B, 2, Int>>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
        state: &mut ModelState<B>,
    ) -> Tensor<B, 1> {
        let hidden = if let Some(mask) = summary_event_mask {
            self.model
                .forward_hidden_with_state_and_summary_event_mask(inputs, mask, state)
        } else {
            self.model.forward_hidden_with_state(inputs, state)
        };
        self.language_loss_from_hidden(hidden, targets, loss_mask)
    }

    fn correct_state_with_predictive_coding(
        &self,
        state: ModelState<B>,
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
        loss_mask: Option<Tensor<B, 2, Int>>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> (ModelState<B>, PredictiveCodingChunkReport)
    where
        B: AutodiffBackend,
    {
        let start = Instant::now();
        let mut report = PredictiveCodingChunkReport {
            chunks_seen: 1,
            ..PredictiveCodingChunkReport::default()
        };
        let state_scope = self.predictive_coding.state_scope;
        if !Self::predictive_coding_state_has_latents(&state, state_scope) {
            report.skipped_empty_state = 1;
            report.elapsed_ns = start.elapsed().as_nanos();
            return (state.detached_clone(), report);
        }

        let config = self.predictive_coding_inference_config();
        let sync_diagnostics = self.predictive_coding.sync_diagnostics;
        let mut corrected = state.detached_clone();
        let mut update_stats = PredictiveCodingTensorUpdateStats::default();
        for step in 0..config.steps {
            if !Self::attach_predictive_coding_state_latents(&mut corrected, state_scope) {
                report.skipped_empty_state = report.skipped_empty_state.saturating_add(1);
                break;
            }
            let mut inference_state = corrected.clone();
            let energy = self.predictive_coding_energy_with_state(
                inputs.clone(),
                targets.clone(),
                loss_mask.clone(),
                summary_event_mask.clone(),
                &mut inference_state,
            );
            if sync_diagnostics && step == 0 {
                report.energy_before = Some(scalar_tensor_to_f64(energy.clone().detach().inner()));
            }
            let grads = energy.backward();
            let step_stats = Self::update_predictive_coding_state_latents(
                &mut corrected,
                &grads,
                &config,
                sync_diagnostics,
                state_scope,
            );
            if step_stats.tensor_count == 0 {
                report.skipped_empty_state = report.skipped_empty_state.saturating_add(1);
                corrected.detach_in_place();
                break;
            }
            update_stats.tensor_count = update_stats
                .tensor_count
                .saturating_add(step_stats.tensor_count);
            update_stats.diagnostic_count = update_stats
                .diagnostic_count
                .saturating_add(step_stats.diagnostic_count);
            update_stats.grad_norm_sum += step_stats.grad_norm_sum;
            update_stats.grad_norm_max = update_stats.grad_norm_max.max(step_stats.grad_norm_max);
            update_stats.delta_rms_sum += step_stats.delta_rms_sum;
            report.inference_steps = report.inference_steps.saturating_add(1);
            corrected.detach_in_place();
        }

        if report.inference_steps > 0 {
            if sync_diagnostics {
                let mut post_state = corrected.clone();
                let post_energy = self.predictive_coding_energy_with_state(
                    inputs,
                    targets,
                    loss_mask,
                    summary_event_mask,
                    &mut post_state,
                );
                report.energy_after = Some(scalar_tensor_to_f64(post_energy.detach().inner()));
            }
            report.chunks_corrected = 1;
            report.grad_norm_mean = update_stats.grad_norm_mean();
            report.grad_norm_max = update_stats.grad_norm_max();
            report.delta_rms_mean = update_stats.delta_rms_mean();
        }
        report.elapsed_ns = start.elapsed().as_nanos();
        (corrected, report)
    }

    fn greedy_rollout_unlikelihood_weight(&self) -> f32 {
        Self::scheduled_weight(
            self.greedy_rollout_unlikelihood.enabled,
            self.greedy_rollout_unlikelihood.weight,
            self.greedy_rollout_unlikelihood.warmup_steps,
            self.greedy_rollout_unlikelihood.ramp_steps,
            self.gradient_scale_step.load(Ordering::Relaxed),
        )
    }

    fn greedy_rollout_unlikelihood_margin_weight(&self) -> f32 {
        Self::scheduled_weight(
            self.greedy_rollout_unlikelihood.enabled,
            self.greedy_rollout_unlikelihood.margin_weight,
            self.greedy_rollout_unlikelihood.warmup_steps,
            self.greedy_rollout_unlikelihood.ramp_steps,
            self.gradient_scale_step.load(Ordering::Relaxed),
        )
    }

    fn greedy_rollout_cycle_weight(&self) -> f32 {
        Self::scheduled_weight(
            self.greedy_rollout_unlikelihood.enabled,
            self.greedy_rollout_unlikelihood.cycle_weight,
            self.greedy_rollout_unlikelihood.warmup_steps,
            self.greedy_rollout_unlikelihood.ramp_steps,
            self.gradient_scale_step.load(Ordering::Relaxed),
        )
    }

    fn greedy_rollout_cycle_margin_weight(&self) -> f32 {
        Self::scheduled_weight(
            self.greedy_rollout_unlikelihood.enabled,
            self.greedy_rollout_unlikelihood.cycle_margin_weight,
            self.greedy_rollout_unlikelihood.warmup_steps,
            self.greedy_rollout_unlikelihood.ramp_steps,
            self.gradient_scale_step.load(Ordering::Relaxed),
        )
    }

    fn next_token_loss_from_logits(
        &self,
        logits: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
        clean_inputs: Tensor<B, 2, Int>,
        loss_mask: Option<Tensor<B, 2, Int>>,
        dynamics_teacher_logits: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 1> {
        let [batch_size, time, vocab] = logits.shape().dims();
        let log_probs = log_probs_from_logits(logits.clone());
        let mut loss =
            next_token_loss_from_log_probs(log_probs.clone(), targets.clone(), loss_mask.clone());
        if let Some(answer_ranking_loss) = self.ruliad_answer_ranking_loss_from_logits(
            logits.clone(),
            targets.clone(),
            loss_mask.clone(),
        ) {
            loss = loss + answer_ranking_loss;
        }
        let weight = self.repeat_unlikelihood_weight();
        let cycle_weight = self.repeat_cycle_weight();
        let cycle_margin_weight = self.repeat_cycle_margin_weight();
        let needs_lagged_aux = weight > f32::EPSILON
            || cycle_weight > f32::EPSILON
            || cycle_margin_weight > f32::EPSILON;
        if needs_lagged_aux {
            if weight > f32::EPSILON {
                let mut total_loss: Option<Tensor<B, 1>> = None;
                let mut total_hits: Option<Tensor<B, 1>> = None;
                for lag in self.repeat_unlikelihood_lags(time) {
                    let Some((lag_log_probs, lag_targets, history_tokens)) =
                        lagged_prediction_tensors(
                            log_probs.clone(),
                            targets.clone(),
                            clean_inputs.clone(),
                            lag,
                            batch_size,
                            time,
                            vocab,
                        )
                    else {
                        continue;
                    };
                    let repeat_weight = history_tokens.clone().not_equal(lag_targets).int().float();
                    let unlikelihood = unlikelihood_from_log_probs(
                        lag_log_probs,
                        history_tokens,
                        self.repeat_unlikelihood.epsilon,
                    );
                    let lag_loss = (unlikelihood * repeat_weight.clone()).sum().reshape([1]);
                    let lag_hits = repeat_weight.sum().reshape([1]);
                    total_loss = Some(match total_loss {
                        Some(accumulated) => accumulated + lag_loss,
                        None => lag_loss,
                    });
                    total_hits = Some(match total_hits {
                        Some(accumulated) => accumulated + lag_hits,
                        None => lag_hits,
                    });
                }
                if let Some(total_loss) = total_loss {
                    loss = loss
                        + total_loss
                            .div(
                                total_hits
                                    .expect("repeat unlikelihood hit accumulator")
                                    .clamp_min(1.0),
                            )
                            .mul_scalar(weight);
                }
            }
            if cycle_weight > f32::EPSILON || cycle_margin_weight > f32::EPSILON {
                let mut total_cycle: Option<Tensor<B, 1>> = None;
                let mut total_cycle_hits: Option<Tensor<B, 1>> = None;
                let mut total_cycle_margin: Option<Tensor<B, 1>> = None;
                let mut total_cycle_margin_hits: Option<Tensor<B, 1>> = None;
                let mean_logits_by_position = (cycle_margin_weight > f32::EPSILON)
                    .then(|| logits.clone().mean_dim(2).reshape([batch_size, time]));
                for lag in self.repeat_cycle_lags(time) {
                    let Some((lag_log_probs, lag_targets, history_tokens)) =
                        lagged_prediction_tensors(
                            log_probs.clone(),
                            targets.clone(),
                            clean_inputs.clone(),
                            lag,
                            batch_size,
                            time,
                            vocab,
                        )
                    else {
                        continue;
                    };
                    let cycle_weight_tensor =
                        history_tokens.clone().not_equal(lag_targets).int().float();
                    if cycle_weight > f32::EPSILON {
                        let unlikelihood = unlikelihood_from_log_probs(
                            lag_log_probs,
                            history_tokens.clone(),
                            self.repeat_unlikelihood.epsilon,
                        );
                        let lag_loss = (unlikelihood * cycle_weight_tensor.clone())
                            .sum()
                            .reshape([1]);
                        let lag_hits = cycle_weight_tensor.clone().sum().reshape([1]);
                        total_cycle = Some(match total_cycle {
                            Some(accumulated) => accumulated + lag_loss,
                            None => lag_loss,
                        });
                        total_cycle_hits = Some(match total_cycle_hits {
                            Some(accumulated) => accumulated + lag_hits,
                            None => lag_hits,
                        });
                    }
                    if cycle_margin_weight > f32::EPSILON {
                        let start = lag.saturating_sub(1);
                        let lag_logits =
                            logits.clone().slice([0..batch_size, start..time, 0..vocab]);
                        let history_logits =
                            selected_token_logits(lag_logits.clone(), history_tokens);
                        let mean_logits = mean_logits_by_position
                            .as_ref()
                            .expect("cycle margin mean logits")
                            .clone()
                            .slice([0..batch_size, start..time]);
                        let margin_penalty = activation::softplus(
                            history_logits - mean_logits + self.repeat_unlikelihood.cycle_margin,
                            1.0,
                        );
                        let lag_margin = (margin_penalty * cycle_weight_tensor.clone())
                            .sum()
                            .reshape([1]);
                        let lag_hits = cycle_weight_tensor.sum().reshape([1]);
                        total_cycle_margin = Some(match total_cycle_margin {
                            Some(accumulated) => accumulated + lag_margin,
                            None => lag_margin,
                        });
                        total_cycle_margin_hits = Some(match total_cycle_margin_hits {
                            Some(accumulated) => accumulated + lag_hits,
                            None => lag_hits,
                        });
                    }
                }
                if let Some(total_cycle) = total_cycle {
                    loss = loss
                        + total_cycle
                            .div(
                                total_cycle_hits
                                    .expect("repeat cycle hit accumulator")
                                    .clamp_min(1.0),
                            )
                            .mul_scalar(cycle_weight);
                }
                if let Some(total_cycle_margin) = total_cycle_margin {
                    loss = loss
                        + total_cycle_margin
                            .div(
                                total_cycle_margin_hits
                                    .expect("repeat cycle margin hit accumulator")
                                    .clamp_min(1.0),
                            )
                            .mul_scalar(cycle_margin_weight);
                }
            }
        }
        if let Some(entropy_floor_loss) =
            self.logit_entropy_floor_loss(log_probs.clone(), targets.clone())
        {
            loss = loss + entropy_floor_loss;
        }
        if let Some(teacher_logits) = dynamics_teacher_logits
            && let Some(anchor_loss) =
                self.dynamics_anchor_loss_from_log_probs(log_probs, teacher_logits, loss_mask)
        {
            loss = loss + anchor_loss;
        }
        loss
    }

    fn repeat_unlikelihood_lags(&self, time: usize) -> Vec<usize> {
        if time == 0 {
            return Vec::new();
        }
        let mut lags = Vec::with_capacity(self.repeat_unlikelihood.history_lags.len() + 1);
        lags.push(1);
        lags.extend(self.repeat_unlikelihood.history_lags.iter().copied());
        lags.retain(|lag| (1..=time).contains(lag));
        lags.sort_unstable();
        lags.dedup();
        lags
    }

    fn repeat_cycle_lags(&self, time: usize) -> Vec<usize> {
        if time == 0
            || self.repeat_unlikelihood.cycle_min_lag == 0
            || self.repeat_unlikelihood.cycle_max_lag < self.repeat_unlikelihood.cycle_min_lag
        {
            return Vec::new();
        }
        let min_lag = self.repeat_unlikelihood.cycle_min_lag.min(time);
        let max_lag = self.repeat_unlikelihood.cycle_max_lag.min(time);
        if max_lag < min_lag {
            return Vec::new();
        }
        let available = max_lag - min_lag + 1;
        let budget = self
            .repeat_unlikelihood
            .cycle_lags_per_step
            .max(1)
            .min(available);
        if budget == available {
            return (min_lag..=max_lag).collect();
        }
        let step_index = self.gradient_scale_step.load(Ordering::Relaxed);
        let mut lags = Vec::with_capacity(budget);
        let stride = (available / budget).max(1);
        let offset = step_index % available;
        for index in 0..budget {
            let relative = (offset + index * stride) % available;
            lags.push(min_lag + relative);
        }
        lags.sort_unstable();
        lags.dedup();
        lags
    }

    fn next_token_loss_from_hidden(
        &self,
        hidden: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
        clean_inputs: Tensor<B, 2, Int>,
        loss_mask: Option<Tensor<B, 2, Int>>,
        dynamics_teacher_logits: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 1> {
        if (self.repeat_unlikelihood_weight() <= f32::EPSILON
            && self.repeat_cycle_weight() <= f32::EPSILON
            && self.repeat_cycle_margin_weight() <= f32::EPSILON
            && self.logit_entropy_floor_weight() <= f32::EPSILON
            && self.logit_marginal_entropy_floor_weight() <= f32::EPSILON
            && self.logit_target_coverage_weight() <= f32::EPSILON
            && self.ruliad_answer_ranking_weight() <= f32::EPSILON
            && self.ruliad_answer_denoising_weight() <= f32::EPSILON
            && dynamics_teacher_logits.is_none())
            || self.model.uses_factorized_language_head()
        {
            let mut loss =
                self.language_loss_from_hidden(hidden.clone(), targets.clone(), loss_mask.clone());
            if let Some(aux) = self.latent_reasoning_auxiliary_loss(
                hidden,
                clean_inputs.clone(),
                Some(targets.clone()),
                loss_mask.clone(),
            ) {
                loss = loss + aux;
            }
            if let Some(denoising) =
                self.ruliad_answer_denoising_loss(clean_inputs, targets, loss_mask)
            {
                loss = loss + denoising;
            }
            return loss;
        }
        let logits = self.model.logits_from_hidden(hidden.clone());
        let mut loss = self.next_token_loss_from_logits(
            logits,
            targets.clone(),
            clean_inputs.clone(),
            loss_mask.clone(),
            dynamics_teacher_logits,
        );
        if let Some(aux) = self.latent_reasoning_auxiliary_loss(
            hidden,
            clean_inputs.clone(),
            Some(targets.clone()),
            loss_mask.clone(),
        ) {
            loss = loss + aux;
        }
        if let Some(denoising) = self.ruliad_answer_denoising_loss(clean_inputs, targets, loss_mask)
        {
            loss = loss + denoising;
        }
        loss
    }

    fn ruliad_answer_ranking_weight(&self) -> f32 {
        let config = self.ruliad_supervision.answer_ranking;
        if config.enabled {
            config.weight.max(0.0)
        } else {
            0.0
        }
    }

    fn ruliad_answer_ranking_loss_from_logits(
        &self,
        logits: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
        loss_mask: Option<Tensor<B, 2, Int>>,
    ) -> Option<Tensor<B, 1>> {
        let config = self.ruliad_supervision.answer_ranking;
        let weight = self.ruliad_answer_ranking_weight();
        if weight <= f32::EPSILON {
            return None;
        }
        let mask = loss_mask?;
        let [batch, time, vocab] = logits.shape().dims();
        if batch == 0 || time == 0 || vocab <= 1 {
            return None;
        }
        let offset = (config.corrupt_offset % vocab as i64).max(1);
        let corrupt_targets = targets
            .clone()
            .add_scalar(offset)
            .remainder_scalar(vocab as i64);
        let oracle_logits = selected_token_logits(logits.clone(), targets);
        let corrupt_logits = selected_token_logits(logits, corrupt_targets);
        let penalty =
            activation::softplus(corrupt_logits - oracle_logits + config.margin.max(0.0), 1.0);
        Some(masked_token_mean(penalty, Some(mask)).mul_scalar(weight))
    }

    fn ruliad_answer_denoising_weight(&self) -> f32 {
        let config = self.ruliad_supervision.answer_denoising;
        if config.enabled {
            config.weight.max(0.0)
        } else {
            0.0
        }
    }

    fn ruliad_answer_denoising_loss(
        &self,
        clean_inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
        loss_mask: Option<Tensor<B, 2, Int>>,
    ) -> Option<Tensor<B, 1>> {
        let config = self.ruliad_supervision.answer_denoising;
        let weight = self.ruliad_answer_denoising_weight();
        if weight <= f32::EPSILON || self.pipeline_enabled() {
            return None;
        }
        let mask = loss_mask?;
        let prefix_mask = answer_prefix_input_mask(mask.clone());
        let corrupted_inputs =
            self.corrupt_ruliad_answer_prefix_inputs(clean_inputs, prefix_mask, config);
        let hidden = self.model.forward_hidden(corrupted_inputs);
        Some(
            self.language_loss_from_hidden(hidden, targets, Some(mask))
                .mul_scalar(weight),
        )
    }

    fn corrupt_ruliad_answer_prefix_inputs(
        &self,
        inputs: Tensor<B, 2, Int>,
        prefix_mask: Tensor<B, 2, Int>,
        config: RuliadAnswerDenoisingConfig,
    ) -> Tensor<B, 2, Int> {
        let probability = config.probability.clamp(0.0, 1.0);
        if probability <= f32::EPSILON {
            return inputs;
        }
        let [batch, time] = inputs.shape().dims();
        if batch == 0 || time == 0 || self.input_vocab_size <= 1 {
            return inputs;
        }
        let vocab = self.input_vocab_size as i64;
        let offset = (config.corrupt_offset % vocab).max(1);
        let mut mask = prefix_mask.equal_elem(1);
        if probability < 1.0 {
            let device = inputs.device();
            let keep = Tensor::<B, 2>::random(
                [batch, time],
                TensorDistribution::Uniform(0.0, 1.0),
                &device,
            )
            .lower_elem(probability);
            mask = mask.bool_and(keep);
        }
        let replacements = inputs.clone().add_scalar(offset).remainder_scalar(vocab);
        inputs.mask_where(mask, replacements)
    }

    fn latent_reasoning_target_hidden(
        &self,
        hidden: Tensor<B, 3>,
        clean_inputs: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        if !self.latent_reasoning.enabled
            || self.pipeline_enabled()
            || !matches!(
                self.latent_reasoning.target_encoder,
                crate::config::LatentReasoningTargetEncoder::EmaTeacher
            )
        {
            return hidden.detach();
        }
        self.current_teacher_model()
            .forward_hidden(clean_inputs)
            .detach()
    }

    fn shifted_latent_negative(target: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, time, dim] = target.shape().dims();
        if batch > 1 {
            let head = target.clone().slice([0..1, 0..time, 0..dim]);
            let tail = target.slice([1..batch, 0..time, 0..dim]);
            Tensor::cat(vec![tail, head], 0)
        } else if time > 1 {
            let head = target.clone().slice([0..batch, 0..1, 0..dim]);
            let tail = target.slice([0..batch, 1..time, 0..dim]);
            Tensor::cat(vec![tail, head], 1)
        } else {
            target
        }
    }

    fn sigreg_loss_from_hidden(&self, hidden: Tensor<B, 3>) -> Option<Tensor<B, 1>> {
        if !self.latent_reasoning.sigreg.enabled
            || !matches!(
                self.latent_reasoning.sigreg.target,
                crate::config::LatentReasoningSigRegTarget::Hidden
                    | crate::config::LatentReasoningSigRegTarget::HiddenAndRhoMemorySlots
            )
        {
            return None;
        }
        let [batch, time, dim] = hidden.shape().dims();
        if batch == 0 || time == 0 || dim == 0 {
            return None;
        }
        let mean = hidden.clone().mean_dim(0).mean_dim(1);
        let centered = hidden - mean.clone().repeat_dim(0, batch).repeat_dim(1, time);
        let variance = centered.powf_scalar(2.0).mean_dim(0).mean_dim(1);
        let variance_floor = self.latent_reasoning.sigreg.min_variance.max(0.0);
        let variance_loss = variance
            .mul_scalar(-1.0)
            .add_scalar(variance_floor)
            .clamp_min(0.0)
            .powf_scalar(2.0)
            .mean();
        let mean_tolerance = self.latent_reasoning.sigreg.mean_tolerance.max(0.0);
        let mean_loss = mean
            .abs()
            .add_scalar(-mean_tolerance)
            .clamp_min(0.0)
            .powf_scalar(2.0)
            .mean();
        Some((variance_loss + mean_loss).reshape([1]))
    }

    fn sigreg_loss_from_rho_memory_state(&self, state: &ModelState<B>) -> Option<Tensor<B, 1>> {
        if !self.latent_reasoning.sigreg.enabled
            || !matches!(
                self.latent_reasoning.sigreg.target,
                crate::config::LatentReasoningSigRegTarget::RhoMemorySlots
                    | crate::config::LatentReasoningSigRegTarget::HiddenAndRhoMemorySlots
            )
        {
            return None;
        }
        let mut total: Option<Tensor<B, 1>> = None;
        let mut components = 0usize;
        for rho in state.layers.iter().filter_map(|layer| layer.rho.as_ref()) {
            let [batch, heads, original_slots, dim] = rho.shape().dims::<4>();
            if batch == 0 || heads == 0 || original_slots < 2 || dim == 0 {
                continue;
            }
            let rho = self.sigreg_sample_rho_slots(rho.clone(), original_slots);
            let [batch, heads, slots, dim] = rho.shape().dims::<4>();
            let groups = batch * heads;
            let rows = rho.reshape([groups, slots, dim]);
            let row_mean = rows.clone().mean_dim(2);
            let centered = rows - row_mean.repeat_dim(2, dim);
            let row_energy = centered
                .clone()
                .powf_scalar(2.0)
                .sum_dim(2)
                .clamp_min(1.0e-8);
            let normalized = centered / row_energy.clone().sqrt().repeat_dim(2, dim);
            let gram = normalized
                .clone()
                .matmul(normalized.clone().swap_dims(1, 2));
            let total_sq = gram.powf_scalar(2.0).sum().reshape([1]);
            let diag_sq = normalized
                .powf_scalar(2.0)
                .sum_dim(2)
                .powf_scalar(2.0)
                .sum()
                .reshape([1]);
            let denom = (groups * slots * slots.saturating_sub(1)).max(1) as f32;
            let loss = (total_sq - diag_sq).clamp_min(0.0).div_scalar(denom);
            total = Some(match total {
                Some(accumulated) => accumulated + loss,
                None => loss,
            });
            components = components.saturating_add(1);
        }
        total.map(|loss| loss.div_scalar(components.max(1) as f32))
    }

    fn sigreg_sample_rho_slots(&self, rho: Tensor<B, 4>, slots: usize) -> Tensor<B, 4> {
        Self::sample_rho_slots_with_limit(rho, slots, self.latent_reasoning.sigreg.max_rho_slots)
    }

    fn sample_rho_slots_with_limit(
        rho: Tensor<B, 4>,
        slots: usize,
        max_slots: usize,
    ) -> Tensor<B, 4> {
        let max_slots = max_slots.max(2);
        if slots <= max_slots {
            return rho;
        }
        let sample_slots = max_slots.min(slots);
        let denominator = sample_slots.saturating_sub(1).max(1);
        let source_span = slots.saturating_sub(1);
        let indices = (0..sample_slots)
            .map(|idx| ((idx * source_span + denominator / 2) / denominator) as i64)
            .collect::<Vec<_>>();
        let device = rho.device();
        let indices =
            Tensor::<B, 1, Int>::from_data(TensorData::new(indices, [sample_slots]), &device);
        rho.select(2, indices)
    }

    fn normalized_rho_rows(rho: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [_batch, _heads, _slots, dim] = rho.shape().dims::<4>();
        let energy = rho
            .clone()
            .powf_scalar(2.0)
            .mean_dim(3)
            .clamp_min(1.0e-8)
            .sqrt();
        let normalized = rho / energy.clone().repeat_dim(3, dim);
        (normalized, energy)
    }

    fn dragon_state_consistency_loss(
        &self,
        student_state: &ModelState<B>,
        teacher_state: &ModelState<B>,
    ) -> (Option<Tensor<B, 1>>, usize) {
        let config = &self.latent_reasoning.dragon_state;
        if !config.enabled {
            return (None, 0);
        }
        let rho_weight = config.rho_weight.max(0.0);
        let rho_energy_weight = config.rho_energy_weight.max(0.0);
        if rho_weight <= f32::EPSILON && rho_energy_weight <= f32::EPSILON {
            return (None, 0);
        }
        let mut total: Option<Tensor<B, 1>> = None;
        let mut components = 0usize;
        for (student_layer, teacher_layer) in student_state.layers.iter().zip(&teacher_state.layers)
        {
            let (Some(student_rho), Some(teacher_rho)) =
                (student_layer.rho.as_ref(), teacher_layer.rho.as_ref())
            else {
                continue;
            };
            let student_dims = student_rho.shape().dims::<4>();
            if student_dims != teacher_rho.shape().dims::<4>() {
                continue;
            }
            let [_batch, _heads, slots, _dim] = student_dims;
            if slots < 2 {
                continue;
            }
            let student_rho =
                Self::sample_rho_slots_with_limit(student_rho.clone(), slots, config.max_rho_slots);
            let teacher_rho =
                Self::sample_rho_slots_with_limit(teacher_rho.clone(), slots, config.max_rho_slots)
                    .detach();
            let (student_rows, student_energy) = Self::normalized_rho_rows(student_rho);
            let (teacher_rows, teacher_energy) = Self::normalized_rho_rows(teacher_rho);
            if rho_weight > f32::EPSILON {
                let row_loss = crate::train::next_latent::smooth_l1_mean(
                    student_rows,
                    teacher_rows.detach(),
                    config.smooth_l1_beta,
                )
                .mul_scalar(rho_weight);
                total = Some(match total {
                    Some(accumulated) => accumulated + row_loss,
                    None => row_loss,
                });
                components = components.saturating_add(1);
            }
            if rho_energy_weight > f32::EPSILON {
                let energy_loss = crate::train::next_latent::smooth_l1_mean(
                    student_energy,
                    teacher_energy.detach(),
                    config.smooth_l1_beta,
                )
                .mul_scalar(rho_energy_weight);
                total = Some(match total {
                    Some(accumulated) => accumulated + energy_loss,
                    None => energy_loss,
                });
                components = components.saturating_add(1);
            }
        }
        (
            total.map(|loss| loss.div_scalar(components.max(1) as f32)),
            components,
        )
    }

    fn next_latent_auxiliary_loss(
        &self,
        hidden: Tensor<B, 3>,
        target_hidden: Tensor<B, 3>,
        clean_inputs: Tensor<B, 2, Int>,
    ) -> (Option<Tensor<B, 1>>, usize) {
        let config = &self.latent_reasoning.next_latent;
        if !config.enabled || !self.model.next_latent_transition_enabled() {
            return (None, 0);
        }
        let regression_weight = config.regression_weight.max(0.0);
        let token_kl_weight = config.token_kl_weight.max(0.0);
        if regression_weight <= f32::EPSILON && token_kl_weight <= f32::EPSILON {
            return (None, 0);
        }
        let [batch, time, dim] = hidden.shape().dims();
        if batch == 0 || time < 2 || dim == 0 {
            return (None, 0);
        }
        let max_horizon = config.horizon.min(time.saturating_sub(1));
        let mut rollout_state = hidden;
        let mut total: Option<Tensor<B, 1>> = None;
        let mut loss_components = 0usize;
        let mut transition_components = 0usize;
        for horizon_index in 0..max_horizon {
            let rollout_time = time.saturating_sub(horizon_index + 1);
            if rollout_time == 0 {
                break;
            }
            let current = rollout_state.slice([0..batch, 0..rollout_time, 0..dim]);
            let action_tokens = clean_inputs
                .clone()
                .slice([0..batch, horizon_index + 1..time]);
            let mut action_embedding = self.model.embed_tokens(action_tokens);
            if config.detach_action_embedding {
                action_embedding = action_embedding.detach();
            }
            let Some(prediction) = self
                .model
                .next_latent_prediction_from_hidden_action(current, action_embedding)
            else {
                break;
            };
            let target = target_hidden
                .clone()
                .slice([0..batch, horizon_index + 1..time, 0..dim])
                .detach();
            if regression_weight > f32::EPSILON {
                let regression = crate::train::next_latent::smooth_l1_mean(
                    prediction.clone(),
                    target.clone(),
                    config.smooth_l1_beta,
                )
                .mul_scalar(regression_weight);
                total = Some(match total {
                    Some(accumulated) => accumulated + regression,
                    None => regression,
                });
                loss_components = loss_components.saturating_add(1);
            }
            if token_kl_weight > f32::EPSILON && !self.model.uses_factorized_language_head() {
                let student_logits = self.model.logits_from_hidden(prediction.clone());
                let teacher_logits = self.model.logits_from_hidden(target).detach();
                let token_kl = crate::train::next_latent::token_kl_mean_from_logits(
                    student_logits,
                    teacher_logits,
                )
                .mul_scalar(token_kl_weight);
                total = Some(match total {
                    Some(accumulated) => accumulated + token_kl,
                    None => token_kl,
                });
                loss_components = loss_components.saturating_add(1);
            }
            rollout_state = prediction;
            transition_components = transition_components.saturating_add(1);
        }
        (
            total.map(|loss| loss.div_scalar(loss_components.max(1) as f32)),
            transition_components,
        )
    }

    fn latent_energy_model_auxiliary_loss(
        &self,
        hidden: Tensor<B, 3>,
        target_hidden: Tensor<B, 3>,
    ) -> (Option<Tensor<B, 1>>, usize) {
        let config = &self.latent_reasoning.energy_model;
        if !config.enabled || !self.model.latent_reasoning_enabled() {
            return (None, 0);
        }
        let contrastive_weight = config.contrastive_weight.max(0.0);
        let monotonic_weight = config.monotonic_weight.max(0.0);
        let contractive_weight = config.contractive_weight.max(0.0);
        if contrastive_weight <= f32::EPSILON
            && monotonic_weight <= f32::EPSILON
            && contractive_weight <= f32::EPSILON
        {
            return (None, 0);
        }
        let Some(mut previous_energy) = self.model.latent_energy_from_hidden(hidden.clone()) else {
            return (None, 0);
        };
        let output = self.model.reason_hidden(hidden);
        if output.step_hiddens.is_empty() || output.energies.is_empty() {
            return (None, 0);
        }
        let target = target_hidden.detach();
        let negative = match self.latent_reasoning.negative_source {
            crate::config::LatentReasoningNegativeSource::InBatchAndCorruptAnswer
            | crate::config::LatentReasoningNegativeSource::TemporalShift => {
                Self::shifted_latent_negative(target.clone()).detach()
            }
        };
        let negative_energy = self.model.latent_energy_from_hidden(negative);
        let step_limit = config
            .max_rollout_steps_for_loss
            .min(output.step_hiddens.len())
            .min(output.energies.len());
        let mut total: Option<Tensor<B, 1>> = None;
        let mut components = 0usize;
        for step_index in 0..step_limit {
            let state = output
                .step_hiddens
                .get(step_index)
                .expect("step hidden")
                .clone();
            let energy = output
                .energies
                .get(step_index)
                .expect("step energy")
                .clone();
            if contrastive_weight > f32::EPSILON
                && let Some(negative_energy) = negative_energy.as_ref()
            {
                let contrastive = latent_energy_contrastive_margin_loss(
                    energy.clone(),
                    negative_energy.clone(),
                    config.margin,
                )
                .mul_scalar(contrastive_weight);
                total = Some(match total {
                    Some(accumulated) => accumulated + contrastive,
                    None => contrastive,
                });
                components = components.saturating_add(1);
            }
            if monotonic_weight > f32::EPSILON {
                let monotonic = latent_energy_monotonic_penalty(
                    previous_energy.clone(),
                    energy.clone(),
                    config.monotonic_tolerance,
                )
                .mul_scalar(monotonic_weight);
                total = Some(match total {
                    Some(accumulated) => accumulated + monotonic,
                    None => monotonic,
                });
                components = components.saturating_add(1);
            }
            if contractive_weight > f32::EPSILON {
                let contractive =
                    latent_energy_contractivity_penalty(state, target.clone(), config.trust_radius)
                        .mul_scalar(contractive_weight);
                total = Some(match total {
                    Some(accumulated) => accumulated + contractive,
                    None => contractive,
                });
                components = components.saturating_add(1);
            }
            previous_energy = energy;
        }
        (
            total.map(|loss| loss.div_scalar(components.max(1) as f32)),
            components,
        )
    }

    fn latent_step_contract_auxiliary_loss(
        &self,
        hidden: Tensor<B, 3>,
        targets: Option<Tensor<B, 2, Int>>,
        loss_mask: Option<Tensor<B, 2, Int>>,
    ) -> (Option<Tensor<B, 1>>, usize) {
        let config = &self.latent_reasoning.step_contract;
        if !config.enabled || !self.model.latent_reasoning_enabled() {
            return (None, 0);
        }
        let ce_weight = config.ce_weight.max(0.0);
        let token_kl_weight = config.token_kl_weight.max(0.0);
        let monotonic_ce_weight = config.monotonic_ce_weight.max(0.0);
        let contractive_weight = config.contractive_weight.max(0.0);
        if ce_weight <= f32::EPSILON
            && token_kl_weight <= f32::EPSILON
            && monotonic_ce_weight <= f32::EPSILON
            && contractive_weight <= f32::EPSILON
        {
            return (None, 0);
        }

        let output = self.model.reason_hidden(hidden.clone());
        if output.step_hiddens.is_empty() {
            return (None, 0);
        }

        let step_limit = config
            .max_rollout_steps_for_loss
            .max(1)
            .min(output.step_hiddens.len());
        let mut total: Option<Tensor<B, 1>> = None;
        let mut components = 0usize;
        let mut previous_hidden = hidden.clone().detach();
        let mut previous_ce = targets.as_ref().map(|targets| {
            self.language_loss_from_hidden_for_latent_step(
                hidden.clone(),
                targets.clone(),
                loss_mask.clone(),
                0,
            )
            .detach()
        });
        let reference_logits = (token_kl_weight > f32::EPSILON
            && !self.model.uses_factorized_language_head())
        .then(|| {
            self.model
                .logits_from_hidden_for_latent_step(output.final_hidden, output.steps_used)
                .detach()
        });

        for (index, state) in output.step_hiddens.into_iter().take(step_limit).enumerate() {
            let step = index.saturating_add(1);
            let step_ce = targets.as_ref().map(|targets| {
                self.language_loss_from_hidden_for_latent_step(
                    state.clone(),
                    targets.clone(),
                    loss_mask.clone(),
                    step,
                )
            });
            if ce_weight > f32::EPSILON
                && let Some(step_ce) = step_ce.as_ref()
            {
                let component = step_ce.clone().mul_scalar(ce_weight);
                total = Some(match total {
                    Some(accumulated) => accumulated + component,
                    None => component,
                });
                components = components.saturating_add(1);
            }
            if monotonic_ce_weight > f32::EPSILON
                && let (Some(step_ce), Some(previous_ce_value)) =
                    (step_ce.as_ref(), previous_ce.as_ref())
            {
                let penalty = (step_ce.clone()
                    - previous_ce_value
                        .clone()
                        .add_scalar(config.ce_tolerance.max(0.0)))
                .clamp_min(0.0)
                .mul_scalar(monotonic_ce_weight);
                total = Some(match total {
                    Some(accumulated) => accumulated + penalty,
                    None => penalty,
                });
                components = components.saturating_add(1);
            }
            if token_kl_weight > f32::EPSILON
                && let Some(reference_logits) = reference_logits.as_ref()
            {
                let step_logits = self
                    .model
                    .logits_from_hidden_for_latent_step(state.clone(), step);
                let token_kl = crate::train::next_latent::token_kl_mean_from_logits(
                    step_logits,
                    reference_logits.clone(),
                )
                .mul_scalar(token_kl_weight);
                total = Some(match total {
                    Some(accumulated) => accumulated + token_kl,
                    None => token_kl,
                });
                components = components.saturating_add(1);
            }
            if contractive_weight > f32::EPSILON {
                let contractive = latent_energy_contractivity_penalty(
                    state.clone(),
                    previous_hidden.clone(),
                    config.trust_radius,
                )
                .mul_scalar(contractive_weight);
                total = Some(match total {
                    Some(accumulated) => accumulated + contractive,
                    None => contractive,
                });
                components = components.saturating_add(1);
            }
            previous_hidden = state.detach();
            if let Some(step_ce) = step_ce {
                previous_ce = Some(step_ce.detach());
            }
        }

        (
            total.map(|loss| loss.div_scalar(components.max(1) as f32)),
            components,
        )
    }

    fn latent_reasoning_fallback_every_steps(&self) -> usize {
        self.latent_reasoning.every_steps.max(1)
    }

    fn latent_reasoning_fallback_start_after_steps(&self) -> usize {
        self.latent_reasoning.constraint_balancer.start_after_steps
    }

    fn latent_reasoning_jepa_every_steps(&self) -> usize {
        self.latent_reasoning
            .jepa_every_steps
            .unwrap_or_else(|| self.latent_reasoning_fallback_every_steps())
            .max(1)
    }

    fn latent_reasoning_jepa_start_after_steps(&self) -> usize {
        self.latent_reasoning
            .jepa_start_after_steps
            .unwrap_or_else(|| self.latent_reasoning_fallback_start_after_steps())
    }

    fn latent_reasoning_default_start_policy(&self) -> LatentReasoningAuxiliaryStartPolicy {
        if self.latent_reasoning.start_after_capability_gate_passed {
            LatentReasoningAuxiliaryStartPolicy::FixedStepAndCapabilityGate
        } else {
            LatentReasoningAuxiliaryStartPolicy::FixedStep
        }
    }

    fn latent_reasoning_jepa_start_policy(&self) -> LatentReasoningAuxiliaryStartPolicy {
        self.latent_reasoning
            .jepa_start_policy
            .unwrap_or_else(|| self.latent_reasoning_default_start_policy())
    }

    fn latent_reasoning_next_latent_every_steps(&self) -> usize {
        self.latent_reasoning
            .next_latent
            .every_steps
            .unwrap_or_else(|| self.latent_reasoning_fallback_every_steps())
            .max(1)
    }

    fn latent_reasoning_next_latent_start_after_steps(&self) -> usize {
        self.latent_reasoning
            .next_latent
            .start_after_steps
            .unwrap_or_else(|| self.latent_reasoning_fallback_start_after_steps())
    }

    fn latent_reasoning_next_latent_start_policy(&self) -> LatentReasoningAuxiliaryStartPolicy {
        self.latent_reasoning
            .next_latent
            .start_policy
            .unwrap_or_else(|| self.latent_reasoning_default_start_policy())
    }

    fn latent_reasoning_dragon_state_every_steps(&self) -> usize {
        self.latent_reasoning
            .dragon_state
            .every_steps
            .unwrap_or_else(|| self.latent_reasoning_fallback_every_steps())
            .max(1)
    }

    fn latent_reasoning_dragon_state_start_after_steps(&self) -> usize {
        self.latent_reasoning
            .dragon_state
            .start_after_steps
            .unwrap_or_else(|| self.latent_reasoning_fallback_start_after_steps())
    }

    fn latent_reasoning_dragon_state_start_policy(&self) -> LatentReasoningAuxiliaryStartPolicy {
        self.latent_reasoning
            .dragon_state
            .start_policy
            .unwrap_or_else(|| self.latent_reasoning_default_start_policy())
    }

    fn latent_reasoning_energy_model_every_steps(&self) -> usize {
        self.latent_reasoning
            .energy_model
            .every_steps
            .unwrap_or_else(|| self.latent_reasoning_fallback_every_steps())
            .max(1)
    }

    fn latent_reasoning_energy_model_start_after_steps(&self) -> usize {
        self.latent_reasoning
            .energy_model
            .start_after_steps
            .unwrap_or_else(|| self.latent_reasoning_fallback_start_after_steps())
    }

    fn latent_reasoning_energy_model_start_policy(&self) -> LatentReasoningAuxiliaryStartPolicy {
        self.latent_reasoning
            .energy_model
            .start_policy
            .unwrap_or_else(|| self.latent_reasoning_default_start_policy())
    }

    fn latent_reasoning_step_contract_every_steps(&self) -> usize {
        self.latent_reasoning
            .step_contract
            .every_steps
            .unwrap_or_else(|| self.latent_reasoning_fallback_every_steps())
            .max(1)
    }

    fn latent_reasoning_step_contract_start_after_steps(&self) -> usize {
        self.latent_reasoning
            .step_contract
            .start_after_steps
            .unwrap_or_else(|| self.latent_reasoning_fallback_start_after_steps())
    }

    fn latent_reasoning_step_contract_start_policy(&self) -> LatentReasoningAuxiliaryStartPolicy {
        self.latent_reasoning
            .step_contract
            .start_policy
            .unwrap_or_else(|| self.latent_reasoning_default_start_policy())
    }

    fn latent_reasoning_sigreg_every_steps(&self) -> usize {
        self.latent_reasoning
            .sigreg
            .every_steps
            .unwrap_or_else(|| self.latent_reasoning_fallback_every_steps())
            .max(1)
    }

    fn latent_reasoning_sigreg_start_after_steps(&self) -> usize {
        self.latent_reasoning
            .sigreg
            .start_after_steps
            .unwrap_or_else(|| self.latent_reasoning_fallback_start_after_steps())
    }

    fn latent_reasoning_sigreg_start_policy(&self) -> LatentReasoningAuxiliaryStartPolicy {
        self.latent_reasoning
            .sigreg
            .start_policy
            .unwrap_or_else(|| self.latent_reasoning_default_start_policy())
    }

    fn latent_reasoning_auxiliary_scale(&self) -> Option<f32> {
        self.latent_reasoning_auxiliary_scale_for_schedule(
            self.latent_reasoning_fallback_every_steps(),
            self.latent_reasoning_fallback_start_after_steps(),
            self.latent_reasoning_default_start_policy(),
        )
    }

    fn latent_reasoning_auxiliary_scale_for_every_steps(&self, every_steps: usize) -> Option<f32> {
        self.latent_reasoning_auxiliary_scale_for_schedule(
            every_steps,
            self.latent_reasoning_fallback_start_after_steps(),
            self.latent_reasoning_default_start_policy(),
        )
    }

    fn latent_reasoning_auxiliary_scale_for_schedule(
        &self,
        every_steps: usize,
        start_after_steps: usize,
        start_policy: LatentReasoningAuxiliaryStartPolicy,
    ) -> Option<f32> {
        if !self.latent_reasoning.enabled {
            return None;
        }
        let requires_capability = matches!(
            start_policy,
            LatentReasoningAuxiliaryStartPolicy::CapabilityGate
                | LatentReasoningAuxiliaryStartPolicy::FixedStepAndCapabilityGate
        );
        let requires_fixed_step = matches!(
            start_policy,
            LatentReasoningAuxiliaryStartPolicy::FixedStep
                | LatentReasoningAuxiliaryStartPolicy::FixedStepAndCapabilityGate
        );
        if requires_capability
            && !self
                .latent_reasoning_capability_gate_open
                .load(Ordering::Relaxed)
        {
            return None;
        }
        let step = self.gradient_scale_step.load(Ordering::Relaxed);
        let current_step = step.saturating_add(1);
        if requires_fixed_step && start_after_steps > 0 && current_step <= start_after_steps {
            return None;
        }
        let every_steps = every_steps.max(1);
        if every_steps > 1 && !current_step.is_multiple_of(every_steps) {
            return None;
        }
        let mut aux_scale = self
            .latent_reasoning
            .constraint_balancer
            .normalized_aux_scale
            .max(0.0);
        let warmup_steps = self.latent_reasoning.constraint_balancer.warmup_steps;
        if warmup_steps > 0 {
            let warmup_start = if requires_fixed_step {
                start_after_steps
            } else {
                0
            };
            let active_step = current_step.saturating_sub(warmup_start).max(1);
            let progress = (active_step as f32 / warmup_steps as f32).min(1.0);
            aux_scale *= progress;
        }
        (aux_scale > f32::EPSILON).then_some(aux_scale)
    }

    fn latent_rho_memory_auxiliary_loss(&self, state: &ModelState<B>) -> Option<Tensor<B, 1>> {
        let aux_scale = self.latent_reasoning_auxiliary_scale_for_schedule(
            self.latent_reasoning_sigreg_every_steps(),
            self.latent_reasoning_sigreg_start_after_steps(),
            self.latent_reasoning_sigreg_start_policy(),
        )?;
        let loss = self.sigreg_loss_from_rho_memory_state(state)?;
        crate::train::profile::record_latent_reasoning(
            0,
            0,
            0,
            0,
            0,
            1,
            self.model.latent_reasoning_config().max_steps,
        );
        Some(loss.mul_scalar(aux_scale))
    }

    fn add_latent_rho_memory_auxiliary_loss(
        &self,
        loss: Tensor<B, 1>,
        state: &ModelState<B>,
    ) -> Tensor<B, 1> {
        self.latent_rho_memory_auxiliary_loss(state)
            .map(|aux| loss.clone() + aux)
            .unwrap_or(loss)
    }

    fn latent_dragon_state_auxiliary_loss(
        &self,
        student_state: &ModelState<B>,
        teacher_state: Option<&ModelState<B>>,
    ) -> Option<Tensor<B, 1>> {
        let aux_scale = self.latent_reasoning_auxiliary_scale_for_schedule(
            self.latent_reasoning_dragon_state_every_steps(),
            self.latent_reasoning_dragon_state_start_after_steps(),
            self.latent_reasoning_dragon_state_start_policy(),
        )?;
        let teacher_state = teacher_state?;
        let (loss, components) = self.dragon_state_consistency_loss(student_state, teacher_state);
        if components > 0 {
            crate::train::profile::record_latent_reasoning(
                0,
                components,
                0,
                0,
                0,
                0,
                self.model.latent_reasoning_config().max_steps,
            );
        }
        loss.map(|loss| loss.mul_scalar(aux_scale))
    }

    fn add_latent_dragon_state_auxiliary_loss(
        &self,
        loss: Tensor<B, 1>,
        student_state: &ModelState<B>,
        teacher_state: Option<&ModelState<B>>,
    ) -> Tensor<B, 1> {
        self.latent_dragon_state_auxiliary_loss(student_state, teacher_state)
            .map(|aux| loss.clone() + aux)
            .unwrap_or(loss)
    }

    fn latent_reasoning_auxiliary_loss(
        &self,
        hidden: Tensor<B, 3>,
        clean_inputs: Tensor<B, 2, Int>,
        targets: Option<Tensor<B, 2, Int>>,
        loss_mask: Option<Tensor<B, 2, Int>>,
    ) -> Option<Tensor<B, 1>> {
        let next_latent_aux_scale = self.latent_reasoning_auxiliary_scale_for_schedule(
            self.latent_reasoning_next_latent_every_steps(),
            self.latent_reasoning_next_latent_start_after_steps(),
            self.latent_reasoning_next_latent_start_policy(),
        );
        let jepa_aux_scale = self.latent_reasoning_auxiliary_scale_for_schedule(
            self.latent_reasoning_jepa_every_steps(),
            self.latent_reasoning_jepa_start_after_steps(),
            self.latent_reasoning_jepa_start_policy(),
        );
        let energy_model_aux_scale = self.latent_reasoning_auxiliary_scale_for_schedule(
            self.latent_reasoning_energy_model_every_steps(),
            self.latent_reasoning_energy_model_start_after_steps(),
            self.latent_reasoning_energy_model_start_policy(),
        );
        let step_contract_aux_scale = self.latent_reasoning_auxiliary_scale_for_schedule(
            self.latent_reasoning_step_contract_every_steps(),
            self.latent_reasoning_step_contract_start_after_steps(),
            self.latent_reasoning_step_contract_start_policy(),
        );
        let sigreg_aux_scale = self.latent_reasoning_auxiliary_scale_for_schedule(
            self.latent_reasoning_sigreg_every_steps(),
            self.latent_reasoning_sigreg_start_after_steps(),
            self.latent_reasoning_sigreg_start_policy(),
        );
        if next_latent_aux_scale.is_none()
            && jepa_aux_scale.is_none()
            && energy_model_aux_scale.is_none()
            && step_contract_aux_scale.is_none()
            && sigreg_aux_scale.is_none()
        {
            return None;
        }
        let [batch, time, dim] = hidden.shape().dims();
        if batch == 0 || time == 0 || dim == 0 {
            let aux_scale = sigreg_aux_scale?;
            let loss = self.sigreg_loss_from_hidden(hidden);
            if loss.is_some() {
                crate::train::profile::record_latent_reasoning(
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    self.model.latent_reasoning_config().max_steps,
                );
            }
            return loss.map(|loss| loss.mul_scalar(aux_scale));
        }

        let target_hidden =
            self.latent_reasoning_target_hidden(hidden.clone(), clean_inputs.clone());
        let mut total: Option<Tensor<B, 1>> = None;
        let mut components = 0usize;
        let mut next_latent_components = 0usize;
        if let Some(next_latent_aux_scale) = next_latent_aux_scale {
            let (next_latent_loss, active_components) = self.next_latent_auxiliary_loss(
                hidden.clone(),
                target_hidden.clone(),
                clean_inputs.clone(),
            );
            next_latent_components = active_components;
            if let Some(next_latent_loss) = next_latent_loss {
                let next_latent_loss = next_latent_loss.mul_scalar(next_latent_aux_scale);
                total = Some(match total {
                    Some(accumulated) => accumulated + next_latent_loss,
                    None => next_latent_loss,
                });
                components = components.saturating_add(1);
            }
        }
        let mut jepa_components = 0usize;
        if let Some(jepa_aux_scale) = jepa_aux_scale {
            for offset in self
                .latent_reasoning
                .jepa_future_offsets
                .iter()
                .copied()
                .filter(|offset| *offset > 0 && *offset < time)
            {
                let context = hidden.clone().slice([0..batch, 0..time - offset, 0..dim]);
                let target = target_hidden
                    .clone()
                    .slice([0..batch, offset..time, 0..dim])
                    .detach();
                let prediction = self.model.latent_jepa_prediction_from_hidden(context);
                let positive_energy = (prediction.clone() - target.clone())
                    .powf_scalar(2.0)
                    .mean()
                    .reshape([1]);
                let negative = Self::shifted_latent_negative(target).detach();
                let negative_energy = (prediction - negative).powf_scalar(2.0).mean().reshape([1]);
                let margin = self.model.latent_reasoning_config().energy_margin;
                let margin_loss =
                    activation::softplus(positive_energy.clone() - negative_energy + margin, 1.0);
                let component = (positive_energy + margin_loss).mul_scalar(jepa_aux_scale);
                total = Some(match total {
                    Some(accumulated) => accumulated + component,
                    None => component,
                });
                components = components.saturating_add(1);
                jepa_components = jepa_components.saturating_add(1);
            }
        }
        let mut energy_model_components = 0usize;
        if let Some(energy_model_aux_scale) = energy_model_aux_scale {
            let (energy_model_loss, active_components) =
                self.latent_energy_model_auxiliary_loss(hidden.clone(), target_hidden.clone());
            energy_model_components = active_components;
            if let Some(energy_model_loss) = energy_model_loss {
                let energy_model_loss = energy_model_loss.mul_scalar(energy_model_aux_scale);
                total = Some(match total {
                    Some(accumulated) => accumulated + energy_model_loss,
                    None => energy_model_loss,
                });
                components = components.saturating_add(1);
            }
        }
        let mut step_contract_components = 0usize;
        if let Some(step_contract_aux_scale) = step_contract_aux_scale {
            let (step_contract_loss, active_components) =
                self.latent_step_contract_auxiliary_loss(hidden.clone(), targets, loss_mask);
            step_contract_components = active_components;
            if let Some(step_contract_loss) = step_contract_loss {
                let step_contract_loss = step_contract_loss.mul_scalar(step_contract_aux_scale);
                total = Some(match total {
                    Some(accumulated) => accumulated + step_contract_loss,
                    None => step_contract_loss,
                });
                components = components.saturating_add(1);
            }
        }
        let mut sigreg_components = 0usize;
        if let Some(sigreg_aux_scale) = sigreg_aux_scale
            && let Some(sigreg) = self.sigreg_loss_from_hidden(hidden)
        {
            let sigreg = sigreg.mul_scalar(sigreg_aux_scale);
            total = Some(match total {
                Some(accumulated) => accumulated + sigreg,
                None => sigreg,
            });
            components = components.saturating_add(1);
            sigreg_components = sigreg_components.saturating_add(1);
        }
        if components > 0 {
            crate::train::profile::record_latent_reasoning(
                next_latent_components,
                0,
                jepa_components,
                energy_model_components,
                step_contract_components,
                sigreg_components,
                self.model.latent_reasoning_config().max_steps,
            );
        }
        total.map(|loss| loss.div_scalar(components.max(1) as f32))
    }

    fn logit_entropy_floor_loss(
        &self,
        log_probs: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
    ) -> Option<Tensor<B, 1>> {
        let [batch, time, vocab] = log_probs.shape().dims();
        if batch == 0 || time == 0 || vocab == 0 {
            return None;
        }
        let token_count = batch * time;
        let flat_log_probs = log_probs.reshape([token_count, vocab]);
        let flat_probs = flat_log_probs.clone().exp();
        let weight = self.logit_entropy_floor_weight();
        let target_entropy_bits = self.logit_entropy_floor.target_entropy_bits;
        let marginal_weight = self.logit_marginal_entropy_floor_weight();
        let target_marginal_entropy_bits = self.logit_entropy_floor.target_marginal_entropy_bits;
        let target_coverage_weight = self.logit_target_coverage_weight();
        let mut total = if weight > f32::EPSILON && target_entropy_bits > f32::EPSILON {
            entropy_floor_loss_from_flat_log_probs(
                flat_log_probs.clone(),
                flat_probs.clone(),
                target_entropy_bits,
            )
            .map(|loss| loss.mul_scalar(weight))
        } else {
            None
        };
        let marginal_probs = (marginal_weight > f32::EPSILON
            || target_coverage_weight > f32::EPSILON)
            .then(|| flat_probs.mean_dim(0));
        if marginal_weight > f32::EPSILON && target_marginal_entropy_bits > f32::EPSILON {
            if let Some(loss) = marginal_entropy_floor_loss_from_marginal(
                marginal_probs
                    .as_ref()
                    .expect("marginal probabilities")
                    .clone(),
                target_marginal_entropy_bits,
            )
            .map(|loss| loss.mul_scalar(marginal_weight))
            {
                total = Some(match total {
                    Some(accumulated) => accumulated + loss,
                    None => loss,
                });
            }
        }
        if target_coverage_weight > f32::EPSILON
            && let Some(loss) = target_marginal_coverage_loss_from_marginal(
                marginal_probs.expect("marginal probabilities"),
                targets,
                self.logit_entropy_floor.target_coverage_epsilon,
            )
            .map(|loss| loss.mul_scalar(target_coverage_weight))
        {
            total = Some(match total {
                Some(accumulated) => accumulated + loss,
                None => loss,
            });
        }
        total
    }

    fn greedy_rollout_entropy_floor_weight(&self) -> f32 {
        Self::scheduled_weight(
            self.greedy_rollout_unlikelihood.enabled,
            self.greedy_rollout_unlikelihood.entropy_floor_weight,
            self.greedy_rollout_unlikelihood.warmup_steps,
            self.greedy_rollout_unlikelihood.ramp_steps,
            self.gradient_scale_step.load(Ordering::Relaxed),
        )
    }

    fn greedy_rollout_entropy_floor_loss(&self, log_probs: Tensor<B, 3>) -> Option<Tensor<B, 1>> {
        let weight = self.greedy_rollout_entropy_floor_weight();
        let target_entropy_bits = self.greedy_rollout_unlikelihood.target_entropy_bits;
        if weight <= f32::EPSILON || target_entropy_bits <= f32::EPSILON {
            return None;
        }
        entropy_floor_loss_from_log_probs(log_probs, target_entropy_bits)
            .map(|loss| loss.mul_scalar(weight))
    }

    fn greedy_rollout_unlikelihood_loss(
        &self,
        clean_inputs: Tensor<B, 2, Int>,
    ) -> Option<Tensor<B, 1>> {
        let step_index = self.gradient_scale_step.load(Ordering::Relaxed);
        let config = &self.greedy_rollout_unlikelihood;
        if config.recovery_only && !self.greedy_rollout_recovery_active.load(Ordering::Relaxed) {
            return None;
        }
        let weight = self.greedy_rollout_unlikelihood_weight();
        let margin_weight = self.greedy_rollout_unlikelihood_margin_weight();
        let cycle_weight = self.greedy_rollout_cycle_weight();
        let cycle_margin_weight = self.greedy_rollout_cycle_margin_weight();
        let entropy_floor_weight = self.greedy_rollout_entropy_floor_weight();
        let recovery_weight = Self::scheduled_weight(
            config.enabled,
            config.recovery_weight,
            config.warmup_steps,
            config.ramp_steps,
            step_index,
        );
        let sequence_recovery_weight = Self::scheduled_weight(
            config.enabled,
            config.sequence_recovery_weight,
            config.warmup_steps,
            config.ramp_steps,
            step_index,
        );
        if (weight <= f32::EPSILON
            && margin_weight <= f32::EPSILON
            && cycle_weight <= f32::EPSILON
            && cycle_margin_weight <= f32::EPSILON
            && recovery_weight <= f32::EPSILON
            && sequence_recovery_weight <= f32::EPSILON
            && entropy_floor_weight <= f32::EPSILON)
            || self.pipeline_enabled()
            || self.model.uses_factorized_language_head()
            || !step_index.is_multiple_of(config.every_steps)
        {
            return None;
        }
        let [batch_size, block_size] = clean_inputs.shape().dims();
        let prompt_batch = batch_size.min(config.batch_prompts.max(1));
        let prompt_tokens = block_size.min(config.prompt_tokens.max(1));
        if prompt_batch == 0 || prompt_tokens == 0 {
            return None;
        }
        let prompt_start =
            rollout_prompt_start(step_index, config.every_steps, block_size, prompt_tokens);
        let prompt = clean_inputs.clone().slice([
            0..prompt_batch,
            prompt_start..(prompt_start + prompt_tokens),
        ]);
        let mut state = self.model.init_state();
        let logits = self.model.forward_with_state(prompt.clone(), &mut state);
        let [_, time, vocab] = logits.shape().dims::<3>();
        if time == 0 || vocab == 0 {
            return None;
        }
        let needs_step_log_probs = weight > f32::EPSILON
            || cycle_weight > f32::EPSILON
            || recovery_weight > f32::EPSILON
            || entropy_floor_weight > f32::EPSILON;
        let needs_step_logits = needs_step_log_probs
            || margin_weight > f32::EPSILON
            || cycle_margin_weight > f32::EPSILON;
        let mut last_logits = logits
            .slice_dim(1, (time - 1)..time)
            .reshape([prompt_batch, vocab]);
        if !needs_step_logits {
            last_logits = last_logits.detach();
            state.detach_in_place();
        }
        let history_tokens = config.history_tokens.max(1);
        let mut history = Vec::with_capacity(history_tokens);
        for offset in 0..prompt_tokens.min(history_tokens) {
            let start = prompt_tokens - 1 - offset;
            history.push(prompt.clone().slice([0..prompt_batch, start..(start + 1)]));
        }
        let mut total_loss: Option<Tensor<B, 1>> = None;
        let mut total_hits: Option<Tensor<B, 1>> = None;
        let mut total_margin: Option<Tensor<B, 1>> = None;
        let mut total_margin_hits: Option<Tensor<B, 1>> = None;
        let mut total_cycle: Option<Tensor<B, 1>> = None;
        let mut total_cycle_hits: Option<Tensor<B, 1>> = None;
        let mut total_cycle_margin: Option<Tensor<B, 1>> = None;
        let mut total_cycle_margin_hits: Option<Tensor<B, 1>> = None;
        let mut total_recovery: Option<Tensor<B, 1>> = None;
        let mut recovery_steps = 0usize;
        let mut generated_tokens = Vec::with_capacity(config.rollout_tokens);
        let mut total_entropy_floor: Option<Tensor<B, 1>> = None;
        let mut entropy_floor_steps = 0usize;
        for rollout_index in 0..config.rollout_tokens {
            let step_logits =
                needs_step_logits.then(|| last_logits.clone().reshape([prompt_batch, 1, vocab]));
            let step_log_probs = needs_step_log_probs.then(|| {
                log_probs_from_logits(
                    step_logits
                        .as_ref()
                        .expect("step logits are required for rollout log-probs")
                        .clone(),
                )
            });
            if let Some(entropy_loss) = step_log_probs
                .as_ref()
                .and_then(|log_probs| self.greedy_rollout_entropy_floor_loss(log_probs.clone()))
            {
                total_entropy_floor = Some(match total_entropy_floor {
                    Some(accumulated) => accumulated + entropy_loss,
                    None => entropy_loss,
                });
                entropy_floor_steps = entropy_floor_steps.saturating_add(1);
            }
            let next = last_logits.clone().argmax(1).reshape([prompt_batch, 1]);
            let mut repeat_mask = next.clone().equal(
                history
                    .first()
                    .expect("greedy rollout history should not be empty")
                    .clone(),
            );
            for previous in history.iter().skip(1) {
                repeat_mask = repeat_mask.bool_or(next.clone().equal(previous.clone()));
            }
            let repeat_mask = repeat_mask.int();
            let cycle_mask =
                cycle_repeat_mask(&next, &history, config.cycle_min_lag, config.cycle_max_lag);
            if weight > f32::EPSILON {
                let next_log_probs = selected_token_log_probs(
                    step_log_probs
                        .as_ref()
                        .expect("step log-probs are required for rollout unlikelihood")
                        .clone(),
                    next.clone(),
                );
                let next_prob = next_log_probs
                    .exp()
                    .clamp_min(0.0)
                    .clamp_max(1.0 - config.epsilon);
                let unlikelihood = next_prob
                    .mul_scalar(-1.0)
                    .add_scalar(1.0)
                    .clamp_min(config.epsilon)
                    .log()
                    .mul_scalar(-1.0);
                let repeat_weight = repeat_mask.clone().float();
                let step_loss = (unlikelihood * repeat_weight.clone()).sum().reshape([1]);
                let step_hits = repeat_weight.sum().reshape([1]);
                total_loss = Some(match total_loss {
                    Some(accumulated) => accumulated + step_loss,
                    None => step_loss,
                });
                total_hits = Some(match total_hits {
                    Some(accumulated) => accumulated + step_hits,
                    None => step_hits,
                });
            }
            if cycle_weight > f32::EPSILON
                && let Some(cycle_mask) = cycle_mask.clone()
            {
                let next_log_probs = selected_token_log_probs(
                    step_log_probs
                        .as_ref()
                        .expect("step log-probs are required for rollout cycle unlikelihood")
                        .clone(),
                    next.clone(),
                );
                let next_prob = next_log_probs
                    .exp()
                    .clamp_min(0.0)
                    .clamp_max(1.0 - config.epsilon);
                let unlikelihood = next_prob
                    .mul_scalar(-1.0)
                    .add_scalar(1.0)
                    .clamp_min(config.epsilon)
                    .log()
                    .mul_scalar(-1.0);
                let cycle_weight_tensor = cycle_mask.float();
                let step_cycle = (unlikelihood * cycle_weight_tensor.clone())
                    .sum()
                    .reshape([1]);
                let step_hits = cycle_weight_tensor.sum().reshape([1]);
                total_cycle = Some(match total_cycle {
                    Some(accumulated) => accumulated + step_cycle,
                    None => step_cycle,
                });
                total_cycle_hits = Some(match total_cycle_hits {
                    Some(accumulated) => accumulated + step_hits,
                    None => step_hits,
                });
            }
            if margin_weight > f32::EPSILON {
                let repeat_weight = repeat_mask.float();
                let step_logits = step_logits
                    .as_ref()
                    .expect("step logits are required for rollout margin");
                let next_logits = selected_token_logits(step_logits.clone(), next.clone());
                let mean_logits = step_logits.clone().mean_dim(2).reshape([prompt_batch, 1]);
                let margin_penalty =
                    activation::softplus(next_logits - mean_logits + config.margin, 1.0);
                let step_margin = (margin_penalty * repeat_weight.clone()).sum().reshape([1]);
                let step_hits = repeat_weight.sum().reshape([1]);
                total_margin = Some(match total_margin {
                    Some(accumulated) => accumulated + step_margin,
                    None => step_margin,
                });
                total_margin_hits = Some(match total_margin_hits {
                    Some(accumulated) => accumulated + step_hits,
                    None => step_hits,
                });
            }
            if cycle_margin_weight > f32::EPSILON
                && let Some(cycle_mask) = cycle_mask
            {
                let cycle_weight_tensor = cycle_mask.float();
                let step_logits = step_logits
                    .as_ref()
                    .expect("step logits are required for rollout cycle margin");
                let next_logits = selected_token_logits(step_logits.clone(), next.clone());
                let mean_logits = step_logits.clone().mean_dim(2).reshape([prompt_batch, 1]);
                let margin_penalty =
                    activation::softplus(next_logits - mean_logits + config.margin, 1.0);
                let step_margin = (margin_penalty * cycle_weight_tensor.clone())
                    .sum()
                    .reshape([1]);
                let step_hits = cycle_weight_tensor.sum().reshape([1]);
                total_cycle_margin = Some(match total_cycle_margin {
                    Some(accumulated) => accumulated + step_margin,
                    None => step_margin,
                });
                total_cycle_margin_hits = Some(match total_cycle_margin_hits {
                    Some(accumulated) => accumulated + step_hits,
                    None => step_hits,
                });
            }
            let target_pos = prompt_start + prompt_tokens + rollout_index;
            if recovery_weight > f32::EPSILON && target_pos < block_size {
                let recovery_target = clean_inputs
                    .clone()
                    .slice([0..prompt_batch, target_pos..(target_pos + 1)]);
                let recovery_loss = selected_token_log_probs(
                    step_log_probs
                        .as_ref()
                        .expect("step log-probs are required for rollout recovery")
                        .clone(),
                    recovery_target,
                )
                .mul_scalar(-1.0)
                .mean()
                .reshape([1]);
                total_recovery = Some(match total_recovery {
                    Some(accumulated) => accumulated + recovery_loss,
                    None => recovery_loss,
                });
                recovery_steps = recovery_steps.saturating_add(1);
            }
            generated_tokens.push(next.clone());
            let logits = self.model.forward_with_state(next.clone(), &mut state);
            let [_, time, vocab] = logits.shape().dims::<3>();
            if time == 0 || vocab == 0 {
                break;
            }
            last_logits = logits
                .slice_dim(1, (time - 1)..time)
                .reshape([prompt_batch, vocab]);
            if !needs_step_logits {
                last_logits = last_logits.detach();
                state.detach_in_place();
            }
            history.insert(0, next);
            if history.len() > history_tokens {
                history.pop();
            }
        }
        let mut loss = total_loss.map(|loss| {
            loss.div(
                total_hits
                    .expect("greedy rollout hit accumulator should exist")
                    .clamp_min(1.0),
            )
            .mul_scalar(weight)
        });
        if let Some(margin) = total_margin {
            let margin = margin
                .div(
                    total_margin_hits
                        .expect("greedy rollout margin hit accumulator should exist")
                        .clamp_min(1.0),
                )
                .mul_scalar(margin_weight);
            loss = Some(match loss {
                Some(accumulated) => accumulated + margin,
                None => margin,
            });
        }
        if let Some(cycle) = total_cycle {
            let cycle = cycle
                .div(
                    total_cycle_hits
                        .expect("greedy rollout cycle hit accumulator should exist")
                        .clamp_min(1.0),
                )
                .mul_scalar(cycle_weight);
            loss = Some(match loss {
                Some(accumulated) => accumulated + cycle,
                None => cycle,
            });
        }
        if let Some(cycle_margin) = total_cycle_margin {
            let cycle_margin = cycle_margin
                .div(
                    total_cycle_margin_hits
                        .expect("greedy rollout cycle margin hit accumulator should exist")
                        .clamp_min(1.0),
                )
                .mul_scalar(cycle_margin_weight);
            loss = Some(match loss {
                Some(accumulated) => accumulated + cycle_margin,
                None => cycle_margin,
            });
        }
        if recovery_steps > 0
            && let Some(recovery) = total_recovery
        {
            let recovery = recovery.mul_scalar(recovery_weight / recovery_steps as f32);
            loss = Some(match loss {
                Some(accumulated) => accumulated + recovery,
                None => recovery,
            });
        }
        if sequence_recovery_weight > f32::EPSILON
            && !generated_tokens.is_empty()
            && prompt_start + prompt_tokens < block_size
        {
            let available_targets = generated_tokens
                .len()
                .min(block_size - prompt_start - prompt_tokens);
            if available_targets > 0 {
                let generated = Tensor::cat(
                    generated_tokens
                        .into_iter()
                        .take(available_targets)
                        .collect(),
                    1,
                );
                let recovery_inputs = Tensor::cat(vec![prompt.clone(), generated], 1);
                let recovery_logits = self.model.forward(recovery_inputs);
                let [_, recovery_time, recovery_vocab] = recovery_logits.shape().dims::<3>();
                let logit_start = prompt_tokens.saturating_sub(1);
                let logit_end = (logit_start + available_targets).min(recovery_time);
                let used_targets = logit_end.saturating_sub(logit_start);
                if used_targets > 0 && recovery_vocab > 0 {
                    let recovery_targets = clean_inputs.clone().slice([
                        0..prompt_batch,
                        (prompt_start + prompt_tokens)
                            ..(prompt_start + prompt_tokens + used_targets),
                    ]);
                    let recovery_log_probs = log_probs_from_logits(recovery_logits.slice([
                        0..prompt_batch,
                        logit_start..logit_end,
                        0..recovery_vocab,
                    ]));
                    let sequence_recovery =
                        selected_token_log_probs(recovery_log_probs, recovery_targets)
                            .mul_scalar(-1.0)
                            .mean()
                            .reshape([1])
                            .mul_scalar(sequence_recovery_weight);
                    loss = Some(match loss {
                        Some(accumulated) => accumulated + sequence_recovery,
                        None => sequence_recovery,
                    });
                }
            }
        }
        if entropy_floor_steps > 0
            && let Some(entropy_floor) = total_entropy_floor
        {
            let entropy_floor = entropy_floor.mul_scalar(1.0 / entropy_floor_steps as f32);
            loss = Some(match loss {
                Some(accumulated) => accumulated + entropy_floor,
                None => entropy_floor,
            });
        }
        loss
    }

    fn corrupt_causal_inputs(&self, inputs: Tensor<B, 2, Int>) -> Tensor<B, 2, Int> {
        let probability = self.causal_input_corruption_probability();
        if probability <= f32::EPSILON {
            return inputs;
        }
        let shape = inputs.shape();
        let device = inputs.device();
        let mask = Tensor::<B, 2>::random(
            shape.clone(),
            TensorDistribution::Uniform(0.0, 1.0),
            &device,
        )
        .lower_elem(probability);
        let replacements = if let Some(token_id) = self.input_corruption.replacement_token_id {
            Tensor::<B, 2, Int>::full(shape, i64::from(token_id), &device)
        } else {
            let vocab_size = self.input_vocab_size.max(1);
            Tensor::<B, 2>::random(
                shape,
                TensorDistribution::Uniform(0.0, vocab_size as f64),
                &device,
            )
            .clamp_min(0.0)
            .clamp_max(vocab_size.saturating_sub(1) as f32)
            .int()
        };
        inputs.mask_where(mask, replacements)
    }

    fn truncate_reprompt_tokens(
        mut tokens: Vec<i64>,
        max_len: usize,
        truncation: RepromptTruncation,
    ) -> Vec<i64> {
        if tokens.len() <= max_len {
            return tokens;
        }
        match truncation {
            RepromptTruncation::Right => tokens.split_off(tokens.len() - max_len),
            RepromptTruncation::Left => {
                tokens.truncate(max_len);
                tokens
            }
            RepromptTruncation::Error => {
                panic!(
                    "teacher-conditioned reprompt length {} exceeds max_reprompt_len {}",
                    tokens.len(),
                    max_len
                )
            }
        }
    }

    fn rollout_score_batch(
        &self,
        generator_model: &DragonModel<B>,
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
        config: RolloutScoreConfig,
    ) -> ObjectiveScoreBatch<B> {
        let [batch_size, block_size] = inputs.shape().dims();
        let device = inputs.device();
        let completion_len = config
            .max_completion_tokens
            .max(1)
            .min(block_size.saturating_sub(1).max(1));
        let prompt_len = block_size.saturating_sub(completion_len).max(1);
        let score_len = prompt_len + completion_len - 1;
        let group_size = config.group_size.max(1);

        let input_tokens = inputs
            .to_data()
            .convert::<i64>()
            .into_vec::<i64>()
            .expect("objective rollout inputs to host tokens");
        let target_tokens = targets
            .to_data()
            .convert::<i64>()
            .into_vec::<i64>()
            .expect("objective rollout targets to host tokens");

        let total_rows = batch_size * group_size;
        let mut student_inputs = Vec::with_capacity(total_rows * score_len);
        let mut student_targets = Vec::with_capacity(total_rows * score_len);
        let mut teacher_inputs = Vec::with_capacity(total_rows * score_len);
        let mut teacher_targets = Vec::with_capacity(total_rows * score_len);
        let mut mask = Vec::with_capacity(total_rows * score_len);

        for batch_idx in 0..batch_size {
            let row_start = batch_idx * block_size;
            let prompt = input_tokens[row_start..row_start + prompt_len].to_vec();
            let completion_start = prompt_len.saturating_sub(1);
            let golden_completion = target_tokens
                [row_start + completion_start..row_start + completion_start + completion_len]
                .to_vec();
            for _ in 0..group_size {
                let generated = crate::generation::generate_tokens(
                    generator_model,
                    prompt.clone(),
                    &device,
                    crate::generation::GenerationSettings {
                        max_new_tokens: Some(completion_len),
                        temperature: config.temperature,
                        top_k: config.top_k,
                        strategy: crate::generation::ContextStrategy::Infinite,
                    },
                    None,
                )
                .expect("objective rollout generation should succeed");
                let completion = generated[prompt_len..prompt_len + completion_len].to_vec();
                let mut teacher_sequence = prompt.clone();
                teacher_sequence.extend_from_slice(&golden_completion);
                teacher_sequence.extend_from_slice(&completion);
                let teacher_sequence = Self::truncate_reprompt_tokens(
                    teacher_sequence,
                    config.max_reprompt_len.max(score_len + 1),
                    config.reprompt_truncation,
                );

                student_inputs.extend_from_slice(&generated[..score_len]);
                student_targets.extend_from_slice(&generated[1..score_len + 1]);
                teacher_inputs.extend_from_slice(
                    &teacher_sequence
                        [teacher_sequence.len() - (score_len + 1)..teacher_sequence.len() - 1],
                );
                teacher_targets.extend_from_slice(
                    &teacher_sequence[teacher_sequence.len() - score_len..teacher_sequence.len()],
                );
                let loss_start = prompt_len.saturating_sub(1)
                    + config.num_loss_tokens_to_skip.min(completion_len);
                for position in 0..score_len {
                    mask.push((position >= loss_start) as i64);
                }
            }
        }

        ObjectiveScoreBatch {
            student_inputs: Tensor::<B, 2, Int>::from_data(
                TensorData::new(student_inputs, [total_rows, score_len]),
                &device,
            ),
            student_targets: Tensor::<B, 2, Int>::from_data(
                TensorData::new(student_targets, [total_rows, score_len]),
                &device,
            ),
            teacher_inputs: Tensor::<B, 2, Int>::from_data(
                TensorData::new(teacher_inputs, [total_rows, score_len]),
                &device,
            ),
            teacher_targets: Tensor::<B, 2, Int>::from_data(
                TensorData::new(teacher_targets, [total_rows, score_len]),
                &device,
            ),
            mask: Tensor::<B, 2, Int>::from_data(
                TensorData::new(mask, [total_rows, score_len]),
                &device,
            ),
        }
    }

    fn objective_loss(&self, inputs: Tensor<B, 2, Int>, targets: Tensor<B, 2, Int>) -> Tensor<B, 1>
    where
        B: AutodiffBackend,
    {
        assert!(
            !(self.pipeline_enabled() && self.tbptt_persist_across_steps),
            "pipeline objective execution does not support persistent stream state"
        );
        self.assert_flat_logits_for_rollout_objective();
        match &self.objective {
            TrainingObjectiveConfig::NextToken => unreachable!("next_token uses the CE fast path"),
            TrainingObjectiveConfig::Sdft(config) => self.sdft_loss(inputs, targets, config),
            TrainingObjectiveConfig::Sdpo(config) => self.sdpo_loss(inputs, targets, config),
            TrainingObjectiveConfig::SdftSdpo(config) => {
                self.composite_sdft_sdpo_loss(inputs, targets, config)
            }
        }
    }

    fn sdft_loss(
        &self,
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
        config: &SdftObjectiveConfig,
    ) -> Tensor<B, 1>
    where
        B: AutodiffBackend,
    {
        let teacher = self.current_teacher_model();
        let generator_model = if config.generate_from_teacher {
            &teacher
        } else {
            &self.model
        };
        let rollout = self.rollout_score_batch(
            generator_model,
            inputs,
            targets,
            RolloutScoreConfig {
                max_completion_tokens: config.max_completion_tokens,
                group_size: 1,
                temperature: config.temperature,
                top_k: config.top_k,
                num_loss_tokens_to_skip: config.num_loss_tokens_to_skip,
                max_reprompt_len: usize::MAX,
                reprompt_truncation: RepromptTruncation::Right,
            },
        );
        let student_hidden = self.forward_hidden_for_objective(rollout.student_inputs);
        let teacher_hidden = teacher.forward_hidden(rollout.teacher_inputs);
        self_distillation_loss_from_logits(
            self.model.logits_from_hidden(student_hidden),
            teacher.logits_from_hidden(teacher_hidden).detach(),
            Some(rollout.mask),
            config.kl,
        )
    }

    fn sdpo_loss(
        &self,
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
        config: &SdpoObjectiveConfig,
    ) -> Tensor<B, 1>
    where
        B: AutodiffBackend,
    {
        let teacher = self.current_teacher_model();
        let rollout = self.rollout_score_batch(
            &self.model,
            inputs,
            targets,
            RolloutScoreConfig {
                max_completion_tokens: config.max_completion_tokens,
                group_size: config.group_size,
                temperature: config.temperature,
                top_k: config.top_k,
                num_loss_tokens_to_skip: 0,
                max_reprompt_len: config.max_reprompt_len,
                reprompt_truncation: config.reprompt_truncation,
            },
        );
        let mask = rollout.mask;
        let student_hidden = self.forward_hidden_for_objective(rollout.student_inputs);
        let teacher_hidden = teacher.forward_hidden(rollout.teacher_inputs);
        let student_logits = self.model.logits_from_hidden(student_hidden);
        let teacher_logits = teacher.logits_from_hidden(teacher_hidden).detach();
        let student_log_probs = log_probs_from_logits(student_logits);
        let teacher_log_probs = log_probs_from_logits(teacher_logits);
        let new_token_log_probs =
            selected_token_log_probs(student_log_probs.clone(), rollout.student_targets);
        let old_token_log_probs = new_token_log_probs.clone().detach();
        let mut per_token_loss = self_distillation_per_token_from_log_probs(
            student_log_probs,
            teacher_log_probs,
            SelfDistillationKlKind::from_sdpo_alpha(config.alpha),
        );
        if let Some(max_ratio) = config.is_clip.filter(|value| *value > 0.0) {
            let log_ratio = (new_token_log_probs - old_token_log_probs)
                .clamp_min(-20.0)
                .clamp_max(20.0);
            let ratio = log_ratio.exp().clamp_max(max_ratio);
            per_token_loss = per_token_loss * ratio;
        }
        masked_token_mean(per_token_loss, Some(mask))
    }

    fn composite_sdft_sdpo_loss(
        &self,
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
        config: &SdftSdpoObjectiveConfig,
    ) -> Tensor<B, 1>
    where
        B: AutodiffBackend,
    {
        let sdft_weight = config.sdft_weight.max(0.0);
        let sdpo_weight = config.sdpo_weight.max(0.0);
        let weight_sum = (sdft_weight + sdpo_weight).max(1.0e-6);
        self.sdft_loss(inputs.clone(), targets.clone(), &config.sdft)
            .mul_scalar(sdft_weight / weight_sum)
            + self
                .sdpo_loss(inputs, targets, &config.sdpo)
                .mul_scalar(sdpo_weight / weight_sum)
    }

    fn forward_loss_with_pipeline(
        &self,
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
        loss_mask: Option<Tensor<B, 2, Int>>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> (Tensor<B, 1>, Tensor<B, 3>, Tensor<B, 3>) {
        let plan = self
            .pipeline_plan
            .as_ref()
            .expect("forward_loss_with_pipeline requires a pipeline plan");
        assert!(
            !self.tbptt_persist_across_steps,
            "pipeline execution does not support tbptt_persist_across_steps"
        );
        assert!(
            self.tbptt_chunk_size.is_none(),
            "pipeline execution does not support tbptt chunking"
        );

        let [batch_size, _block_size] = inputs.shape().dims();
        let ranges = split_microbatch_ranges(batch_size, plan.microbatches)
            .expect("pipeline execution requires batch_size >= microbatches");
        let chunk_inputs = ranges
            .iter()
            .map(|range| Self::slice_batch(inputs.clone(), range.start, range.end))
            .collect::<Vec<_>>();
        let chunk_targets = ranges
            .iter()
            .map(|range| Self::slice_batch(targets.clone(), range.start, range.end))
            .collect::<Vec<_>>();
        let chunk_loss_masks = ranges
            .iter()
            .map(|range| {
                loss_mask
                    .clone()
                    .map(|mask| Self::slice_batch(mask, range.start, range.end))
            })
            .collect::<Vec<_>>();
        let chunk_masks = ranges
            .iter()
            .map(|range| {
                summary_event_mask
                    .clone()
                    .map(|mask| Self::slice_batch(mask, range.start, range.end))
            })
            .collect::<Vec<_>>();
        let factorized_head = self.model.uses_factorized_language_head();

        let mut chunk_states = (0..plan.microbatches)
            .map(|_| self.model.init_state_ephemeral())
            .collect::<Vec<_>>();
        let mut pipeline_states = vec![None; plan.microbatches];

        for event in plan.events.iter().filter(|event| {
            matches!(
                event.kind,
                burn_dragon_train::train::pipeline::PipelineEventKind::Forward
            )
        }) {
            let microbatch_id = event.microbatch_id;
            if pipeline_states[microbatch_id].is_none() {
                pipeline_states[microbatch_id] = Some(
                    self.model
                        .begin_language_pipeline(chunk_inputs[microbatch_id].clone()),
                );
            }
            let assignment = plan.assignment(event.virtual_stage_id).clone();
            let state = &mut chunk_states[microbatch_id];
            let stage_state = pipeline_states[microbatch_id]
                .take()
                .expect("microbatch stage state");
            pipeline_states[microbatch_id] =
                Some(self.model.forward_language_pipeline_stage_with_state(
                    stage_state,
                    state,
                    assignment.layer_range.clone(),
                    chunk_masks[microbatch_id].clone(),
                ));
        }

        let mut total_loss: Option<Tensor<B, 1>> = None;
        let mut hidden_chunks = Vec::with_capacity(plan.microbatches);
        let mut logits_chunks = Vec::with_capacity(plan.microbatches);
        for microbatch_id in 0..plan.microbatches {
            let hidden = self.model.finish_language_pipeline_hidden_with_state(
                pipeline_states[microbatch_id]
                    .take()
                    .expect("pipeline state after scheduled forward"),
                &mut chunk_states[microbatch_id],
            );
            let weight = ranges[microbatch_id].len() as f32 / batch_size as f32;
            let chunk_loss = self
                .language_loss_from_hidden(
                    hidden.clone(),
                    chunk_targets[microbatch_id].clone(),
                    chunk_loss_masks[microbatch_id].clone(),
                )
                .mul_scalar(weight);
            total_loss = Some(match total_loss {
                Some(accumulated) => accumulated + chunk_loss,
                None => chunk_loss,
            });
            if !factorized_head {
                logits_chunks.push(self.model.logits_from_hidden(hidden.clone()));
            }
            hidden_chunks.push(hidden);
        }

        (
            total_loss.expect("pipeline forward should produce at least one microbatch loss"),
            Tensor::cat(hidden_chunks, 0),
            if logits_chunks.is_empty() {
                let device = inputs.device();
                Tensor::<B, 3>::zeros([batch_size, 0, 1], &device)
            } else {
                Tensor::cat(logits_chunks, 0)
            },
        )
    }

    fn forward_loss_with_tbptt(
        &self,
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
        chunk_size: usize,
        state: &mut ModelState<B>,
    ) -> (Tensor<B, 1>, u128) {
        let [batch_size, block_size] = inputs.shape().dims();
        debug_assert!(chunk_size > 0 && chunk_size < block_size);

        let mut total_loss: Option<Tensor<B, 1>> = None;
        let mut total_forward_ns = 0u128;

        for start in (0..block_size).step_by(chunk_size) {
            let end = (start + chunk_size).min(block_size);
            let chunk_inputs = Self::slice_tokens(inputs.clone(), batch_size, start, end);
            let chunk_targets = Self::slice_tokens(targets.clone(), batch_size, start, end);
            let chunk_summary_event_mask = summary_event_mask
                .clone()
                .map(|mask| Self::slice_tokens(mask, batch_size, start, end));

            let chunk_forward_start = Instant::now();
            let logits = if let Some(mask) = chunk_summary_event_mask {
                self.model
                    .forward_with_state_and_summary_event_mask(chunk_inputs, mask, state)
            } else {
                self.model.forward_with_state(chunk_inputs, state)
            };
            total_forward_ns += chunk_forward_start.elapsed().as_nanos();

            let chunk_weight = (end - start) as f32 / block_size as f32;
            let chunk_loss =
                language_model_loss::<B>(logits, chunk_targets).mul_scalar(chunk_weight);
            total_loss = Some(match total_loss {
                Some(accumulated) => accumulated + chunk_loss,
                None => chunk_loss,
            });

            if end < block_size {
                state.detach_in_place();
            }
        }

        (
            total_loss.expect("tbptt forward should produce at least one loss chunk"),
            total_forward_ns,
        )
    }
}

impl<B: AutodiffBackend> TrainStep for LanguageTrainModel<B> {
    type Input = SequenceBatch<B>;
    type Output = LanguageModelTrainItem<B>;

    fn step(&self, batch: SequenceBatch<B>) -> TrainOutput<LanguageModelTrainItem<B>> {
        let prof_enabled = crate::train::profile::enabled();
        let step_index = self.gradient_scale_step.load(Ordering::Relaxed);
        let detail_prof_enabled = prof_enabled && crate::train::profile::detail_due(step_index);
        let memory_prof_enabled = prof_enabled && crate::train::profile::memory_enabled();
        let forward_start = prof_enabled.then(Instant::now);
        let clean_inputs = batch.inputs;
        let targets = batch.targets;
        let loss_mask = batch.loss_mask;
        if !self.objective.is_next_token() {
            self.update_teacher_runtime();
            let loss = self.objective_loss(clean_inputs, targets);
            let grads = loss.backward();
            return TrainOutput {
                grads: self.apply_gradient_scale_schedule(GradientsParams::from_grads(grads, self)),
                item: LanguageModelTrainItem::new(loss),
            };
        }
        if self.latent_reasoning.enabled {
            self.update_teacher_runtime();
        }
        let inputs = self.corrupt_causal_inputs(clean_inputs.clone());
        let clean_inputs_for_aux = clean_inputs.clone();
        let summary_event_mask = batch.summary_event_mask;
        let reset_stream_state = batch.reset_stream_state;
        let step_device = memory_prof_enabled.then(|| inputs.device());
        let step_memory_before = step_device
            .as_ref()
            .and_then(|device| device_memory_usage_safe::<B>(device));
        let [_batch_size, block_size] = inputs.shape().dims();
        let tbptt_chunk_size = self.effective_tbptt_chunk_size(block_size);
        let factorized_head = self.model.uses_factorized_language_head();
        let recurrent_teacher = self.recurrent_teacher_model();
        let (recurrent_teacher, recurrent_teacher_emits_logits) = match recurrent_teacher {
            Some((teacher, emit_logits)) => (Some(teacher), emit_logits),
            None => (None, false),
        };
        let mut recurrent_teacher_state = recurrent_teacher
            .as_ref()
            .map(|teacher| teacher.init_state());
        let probe_inputs = detail_prof_enabled.then(|| inputs.clone());
        let probe_summary_event_mask = detail_prof_enabled
            .then(|| summary_event_mask.clone())
            .flatten();
        let mut step_state = self.load_step_state(reset_stream_state);
        let (loss, probe_hidden, probe_logits, forward_ns) = if self.pipeline_enabled() {
            let forward_start = Instant::now();
            let (loss, hidden, logits) = self.forward_loss_with_pipeline(
                inputs,
                targets.clone(),
                loss_mask.clone(),
                summary_event_mask,
            );
            step_state = self.model.init_state();
            (
                loss,
                Some(hidden),
                (!factorized_head).then_some(logits),
                forward_start.elapsed().as_nanos(),
            )
        } else if let Some(chunk_size) = tbptt_chunk_size {
            let use_tbptt_block_backward = if self.predictive_coding.enabled {
                matches!(
                    self.predictive_coding.backward_mode,
                    PredictiveCodingBackwardMode::Block
                )
            } else {
                detail_prof_enabled
            };
            if use_tbptt_block_backward {
                let [batch_size, block_size] = inputs.shape().dims();
                let mut hidden_chunks = Vec::new();
                let mut logits_chunks = Vec::new();
                let mut teacher_logits_chunks = Vec::new();
                let mut total_forward_ns = 0u128;
                let mut predictive_coding_step_report = PredictiveCodingChunkReport::default();
                for (chunk_index, start) in (0..block_size).step_by(chunk_size).enumerate() {
                    let end = (start + chunk_size).min(block_size);
                    let chunk_inputs = Self::slice_tokens(inputs.clone(), batch_size, start, end);
                    let chunk_summary_event_mask = summary_event_mask
                        .clone()
                        .map(|mask| Self::slice_tokens(mask, batch_size, start, end));
                    if self.predictive_coding_active_for_chunk(step_index, chunk_index) {
                        let chunk_targets =
                            Self::slice_tokens(targets.clone(), batch_size, start, end);
                        let chunk_loss_mask = loss_mask
                            .clone()
                            .map(|mask| Self::slice_tokens(mask, batch_size, start, end));
                        let (corrected_state, report) = self.correct_state_with_predictive_coding(
                            step_state,
                            chunk_inputs.clone(),
                            chunk_targets,
                            chunk_loss_mask,
                            chunk_summary_event_mask.clone(),
                        );
                        step_state = corrected_state;
                        if self.predictive_coding.sync_diagnostics {
                            report.record();
                        } else {
                            predictive_coding_step_report.accumulate_unsynced(report);
                        }
                    }
                    let chunk_teacher_logits = if let (Some(teacher), Some(teacher_state)) =
                        (recurrent_teacher.as_ref(), recurrent_teacher_state.as_mut())
                    {
                        Self::teacher_forward_with_state(
                            teacher,
                            recurrent_teacher_emits_logits,
                            chunk_inputs.clone(),
                            chunk_summary_event_mask.clone(),
                            teacher_state,
                        )
                    } else {
                        None
                    };
                    let chunk_forward_start = Instant::now();
                    let hidden = if let Some(mask) = chunk_summary_event_mask.clone() {
                        self.model.forward_hidden_with_state_and_summary_event_mask(
                            chunk_inputs,
                            mask,
                            &mut step_state,
                        )
                    } else {
                        self.model
                            .forward_hidden_with_state(chunk_inputs, &mut step_state)
                    };
                    total_forward_ns += chunk_forward_start.elapsed().as_nanos();
                    hidden_chunks.push(hidden);
                    if detail_prof_enabled && !factorized_head {
                        logits_chunks.push(
                            self.model
                                .logits_from_hidden(hidden_chunks.last().expect("hidden").clone()),
                        );
                    }
                    if let Some(logits) = chunk_teacher_logits {
                        teacher_logits_chunks.push(logits);
                    }
                    if end < block_size {
                        step_state.detach_in_place();
                        if let Some(teacher_state) = recurrent_teacher_state.as_mut() {
                            teacher_state.detach_in_place();
                        }
                    }
                }
                if predictive_coding_step_report.has_activity() {
                    predictive_coding_step_report.record();
                }
                let hidden = Tensor::cat(hidden_chunks, 1);
                let teacher_logits = (!teacher_logits_chunks.is_empty())
                    .then(|| Tensor::cat(teacher_logits_chunks, 1));
                let loss = self.next_token_loss_from_hidden(
                    hidden.clone(),
                    targets.clone(),
                    clean_inputs.clone(),
                    loss_mask.clone(),
                    teacher_logits,
                );
                let loss = self.add_latent_rho_memory_auxiliary_loss(loss, &step_state);
                let loss = self.add_latent_dragon_state_auxiliary_loss(
                    loss,
                    &step_state,
                    recurrent_teacher_state.as_ref(),
                );
                let logits = (!factorized_head && !logits_chunks.is_empty())
                    .then(|| Tensor::cat(logits_chunks, 1));
                (
                    loss,
                    detail_prof_enabled.then_some(hidden),
                    logits,
                    total_forward_ns,
                )
            } else {
                let [batch_size, block_size] = inputs.shape().dims();
                let mut total_forward_ns = 0u128;
                let mut total_backward_ns = 0u128;
                let mut total_loss: Option<Tensor<B, 1>> = None;
                let mut accumulator = GradientsAccumulator::new();
                let mut predictive_coding_step_report = PredictiveCodingChunkReport::default();

                for (chunk_index, start) in (0..block_size).step_by(chunk_size).enumerate() {
                    let end = (start + chunk_size).min(block_size);
                    let chunk_inputs = Self::slice_tokens(inputs.clone(), batch_size, start, end);
                    let chunk_clean_inputs =
                        Self::slice_tokens(clean_inputs.clone(), batch_size, start, end);
                    let chunk_targets = Self::slice_tokens(targets.clone(), batch_size, start, end);
                    let chunk_loss_mask = loss_mask
                        .clone()
                        .map(|mask| Self::slice_tokens(mask, batch_size, start, end));
                    let chunk_summary_event_mask = summary_event_mask
                        .clone()
                        .map(|mask| Self::slice_tokens(mask, batch_size, start, end));
                    if self.predictive_coding_active_for_chunk(step_index, chunk_index) {
                        let (corrected_state, report) = self.correct_state_with_predictive_coding(
                            step_state,
                            chunk_inputs.clone(),
                            chunk_targets.clone(),
                            chunk_loss_mask.clone(),
                            chunk_summary_event_mask.clone(),
                        );
                        step_state = corrected_state;
                        if self.predictive_coding.sync_diagnostics {
                            report.record();
                        } else {
                            predictive_coding_step_report.accumulate_unsynced(report);
                        }
                    }
                    let chunk_teacher_logits = if let (Some(teacher), Some(teacher_state)) =
                        (recurrent_teacher.as_ref(), recurrent_teacher_state.as_mut())
                    {
                        Self::teacher_forward_with_state(
                            teacher,
                            recurrent_teacher_emits_logits,
                            chunk_inputs.clone(),
                            chunk_summary_event_mask.clone(),
                            teacher_state,
                        )
                    } else {
                        None
                    };

                    let chunk_forward_start = Instant::now();
                    let chunk_loss = if let Some(mask) = chunk_summary_event_mask {
                        let hidden = self.model.forward_hidden_with_state_and_summary_event_mask(
                            chunk_inputs,
                            mask,
                            &mut step_state,
                        );
                        self.next_token_loss_from_hidden(
                            hidden,
                            chunk_targets.clone(),
                            chunk_clean_inputs.clone(),
                            chunk_loss_mask.clone(),
                            chunk_teacher_logits,
                        )
                    } else {
                        let hidden = self
                            .model
                            .forward_hidden_with_state(chunk_inputs, &mut step_state);
                        self.next_token_loss_from_hidden(
                            hidden,
                            chunk_targets.clone(),
                            chunk_clean_inputs.clone(),
                            chunk_loss_mask.clone(),
                            chunk_teacher_logits,
                        )
                    };
                    let chunk_loss =
                        self.add_latent_rho_memory_auxiliary_loss(chunk_loss, &step_state);
                    let chunk_loss = self.add_latent_dragon_state_auxiliary_loss(
                        chunk_loss,
                        &step_state,
                        recurrent_teacher_state.as_ref(),
                    );
                    total_forward_ns += chunk_forward_start.elapsed().as_nanos();

                    let chunk_weight = (end - start) as f32 / block_size as f32;
                    let chunk_loss = chunk_loss.mul_scalar(chunk_weight);
                    total_loss = Some(match total_loss {
                        Some(accumulated) => accumulated + chunk_loss.clone().detach(),
                        None => chunk_loss.clone().detach(),
                    });

                    let chunk_backward_start = Instant::now();
                    let chunk_grads = chunk_loss.backward();
                    total_backward_ns += chunk_backward_start.elapsed().as_nanos();
                    accumulator.accumulate(self, GradientsParams::from_grads(chunk_grads, self));

                    if end < block_size {
                        step_state.detach_in_place();
                        if let Some(teacher_state) = recurrent_teacher_state.as_mut() {
                            teacher_state.detach_in_place();
                        }
                    }
                }
                if predictive_coding_step_report.has_activity() {
                    predictive_coding_step_report.record();
                }

                self.store_step_state(step_state);

                let step_memory_after_forward = step_device
                    .as_ref()
                    .and_then(|device| device_memory_usage_safe::<B>(device));
                if prof_enabled {
                    crate::train::profile::record_train_step(total_forward_ns, total_backward_ns);
                    if let (Some(before), Some(after_forward), Some(device)) = (
                        step_memory_before,
                        step_memory_after_forward,
                        step_device.as_ref(),
                    ) {
                        let after_backward =
                            device_memory_usage_safe::<B>(device).unwrap_or(after_forward);
                        crate::train::profile::record_train_step_memory(
                            before.reserved_bytes,
                            before.in_use_bytes,
                            after_forward.reserved_bytes,
                            after_forward.in_use_bytes,
                            after_backward.reserved_bytes,
                            after_backward.in_use_bytes,
                        );
                    }
                }

                return TrainOutput {
                    grads: self.apply_gradient_scale_schedule(accumulator.grads()),
                    item: LanguageModelTrainItem::new(
                        total_loss
                            .expect("tbptt train step should produce at least one loss chunk"),
                    ),
                };
            }
        } else if detail_prof_enabled {
            if let Some(summary_event_mask) = summary_event_mask {
                let teacher_logits = if let (Some(teacher), Some(teacher_state)) =
                    (recurrent_teacher.as_ref(), recurrent_teacher_state.as_mut())
                {
                    Self::teacher_forward_with_state(
                        teacher,
                        recurrent_teacher_emits_logits,
                        inputs.clone(),
                        Some(summary_event_mask.clone()),
                        teacher_state,
                    )
                } else {
                    None
                };
                let hidden = self.model.forward_hidden_with_state_and_summary_event_mask(
                    inputs,
                    summary_event_mask,
                    &mut step_state,
                );
                let forward_ns = forward_start
                    .map(|start| start.elapsed().as_nanos())
                    .unwrap_or_default();
                let loss = self.next_token_loss_from_hidden(
                    hidden.clone(),
                    targets.clone(),
                    clean_inputs.clone(),
                    loss_mask.clone(),
                    teacher_logits,
                );
                let loss = self.add_latent_rho_memory_auxiliary_loss(loss, &step_state);
                let loss = self.add_latent_dragon_state_auxiliary_loss(
                    loss,
                    &step_state,
                    recurrent_teacher_state.as_ref(),
                );
                let logits =
                    (!factorized_head).then(|| self.model.logits_from_hidden(hidden.clone()));
                (loss, Some(hidden), logits, forward_ns)
            } else {
                let teacher_logits = if let (Some(teacher), Some(teacher_state)) =
                    (recurrent_teacher.as_ref(), recurrent_teacher_state.as_mut())
                {
                    Self::teacher_forward_with_state(
                        teacher,
                        recurrent_teacher_emits_logits,
                        inputs.clone(),
                        None,
                        teacher_state,
                    )
                } else {
                    None
                };
                let hidden = self
                    .model
                    .forward_hidden_with_state(inputs, &mut step_state);
                let forward_ns = forward_start
                    .map(|start| start.elapsed().as_nanos())
                    .unwrap_or_default();
                let loss = self.next_token_loss_from_hidden(
                    hidden.clone(),
                    targets.clone(),
                    clean_inputs.clone(),
                    loss_mask.clone(),
                    teacher_logits,
                );
                let loss = self.add_latent_rho_memory_auxiliary_loss(loss, &step_state);
                let loss = self.add_latent_dragon_state_auxiliary_loss(
                    loss,
                    &step_state,
                    recurrent_teacher_state.as_ref(),
                );
                let logits =
                    (!factorized_head).then(|| self.model.logits_from_hidden(hidden.clone()));
                (loss, Some(hidden), logits, forward_ns)
            }
        } else {
            let teacher_logits = if let (Some(teacher), Some(teacher_state)) =
                (recurrent_teacher.as_ref(), recurrent_teacher_state.as_mut())
            {
                Self::teacher_forward_with_state(
                    teacher,
                    recurrent_teacher_emits_logits,
                    inputs.clone(),
                    summary_event_mask.clone(),
                    teacher_state,
                )
            } else {
                None
            };
            let hidden = if let Some(summary_event_mask) = summary_event_mask {
                self.model.forward_hidden_with_state_and_summary_event_mask(
                    inputs,
                    summary_event_mask,
                    &mut step_state,
                )
            } else {
                self.model
                    .forward_hidden_with_state(inputs, &mut step_state)
            };
            let forward_ns = forward_start
                .map(|start| start.elapsed().as_nanos())
                .unwrap_or_default();
            let loss = self.next_token_loss_from_hidden(
                hidden,
                targets.clone(),
                clean_inputs.clone(),
                loss_mask.clone(),
                teacher_logits,
            );
            let loss = self.add_latent_rho_memory_auxiliary_loss(loss, &step_state);
            let loss = self.add_latent_dragon_state_auxiliary_loss(
                loss,
                &step_state,
                recurrent_teacher_state.as_ref(),
            );
            (loss, None, None, forward_ns)
        };
        let loss = if let Some(rollout_loss) =
            self.greedy_rollout_unlikelihood_loss(clean_inputs_for_aux)
        {
            loss + rollout_loss
        } else {
            loss
        };
        self.store_step_state(step_state);
        let step_memory_after_forward = step_device
            .as_ref()
            .and_then(|device| device_memory_usage_safe::<B>(device));

        let probe_targets = (prof_enabled && detail_prof_enabled).then(|| targets.clone());
        let probe_logits = if prof_enabled && detail_prof_enabled {
            probe_logits.clone().map(|logits| logits.detach())
        } else {
            None
        };
        let probe_hidden = probe_hidden.map(|hidden| hidden.detach());

        let loss_backward_start = prof_enabled.then(Instant::now);
        let grads = loss.backward();
        let loss_backward_ns = loss_backward_start
            .map(|start| start.elapsed().as_nanos())
            .unwrap_or_default();

        if prof_enabled {
            crate::train::profile::record_train_step(forward_ns, loss_backward_ns);
            if let (Some(before), Some(after_forward), Some(device)) = (
                step_memory_before,
                step_memory_after_forward,
                step_device.as_ref(),
            ) {
                let after_backward = device_memory_usage_safe::<B>(device).unwrap_or(after_forward);
                crate::train::profile::record_train_step_memory(
                    before.reserved_bytes,
                    before.in_use_bytes,
                    after_forward.reserved_bytes,
                    after_forward.in_use_bytes,
                    after_backward.reserved_bytes,
                    after_backward.in_use_bytes,
                );
            }
            if detail_prof_enabled {
                let mut embed_probe_ns = 0;
                let mut first_layer_forward_probe_ns = 0;
                let mut first_layer_probe_ns = 0;
                let mut logits_loss_probe_ns = 0;
                let mut hidden_logits_loss_probe_ns = 0;
                let mut hidden_model_forward_probe_ns = 0;
                let mut hidden_model_probe_ns = 0;
                if let Some(probe_inputs) = probe_inputs.clone() {
                    let embed_start = Instant::now();
                    let probe_embedded = self.model.embed_tokens(probe_inputs);
                    let embed_loss = probe_embedded.clone().tanh().powf_scalar(2.0).mean();
                    let _embed_grads = embed_loss.backward();
                    let _ = B::sync(&probe_embedded.device());
                    embed_probe_ns = embed_start.elapsed().as_nanos();

                    let first_layer_forward_start = Instant::now();
                    let first_layer_forward_hidden = self
                        .model
                        .forward_hidden_prefix_layers_from_embedded_for_profile(
                            probe_embedded.clone().detach(),
                            1,
                            probe_summary_event_mask.clone(),
                        );
                    let _ = B::sync(&first_layer_forward_hidden.device());
                    first_layer_forward_probe_ns = first_layer_forward_start.elapsed().as_nanos();

                    let first_layer_start = Instant::now();
                    let probe_embedded_leaf = probe_embedded.detach().require_grad();
                    let first_layer_hidden = self
                        .model
                        .forward_hidden_prefix_layers_from_embedded_for_profile(
                            probe_embedded_leaf.clone(),
                            1,
                            probe_summary_event_mask.clone(),
                        );
                    let first_layer_loss =
                        first_layer_hidden.clone().tanh().powf_scalar(2.0).mean();
                    let _first_layer_grads = first_layer_loss.backward();
                    let _ = B::sync(&probe_embedded_leaf.device());
                    first_layer_probe_ns = first_layer_start.elapsed().as_nanos();
                }
                if let (Some(probe_targets), Some(probe_logits), Some(probe_hidden)) =
                    (probe_targets, probe_logits, probe_hidden)
                {
                    let hidden_model_forward_start = Instant::now();
                    let probe_hidden_forward = if let Some(mask) = probe_summary_event_mask.clone()
                    {
                        let mut probe_hidden_forward_state = self.model.init_state();
                        self.model
                            .forward_with_hidden_and_state_and_summary_event_mask(
                                probe_inputs
                                    .clone()
                                    .expect("probe inputs for hidden forward probe"),
                                mask,
                                &mut probe_hidden_forward_state,
                            )
                            .0
                    } else {
                        self.model
                            .forward_with_hidden(
                                probe_inputs
                                    .clone()
                                    .expect("probe inputs for hidden forward probe"),
                            )
                            .0
                    };
                    let _ = B::sync(&probe_hidden_forward.device());
                    hidden_model_forward_probe_ns = hidden_model_forward_start.elapsed().as_nanos();

                    let logits_only_start = Instant::now();
                    let probe_logits_leaf = probe_logits.require_grad();
                    let logits_only_loss =
                        language_model_loss::<B>(probe_logits_leaf.clone(), probe_targets.clone());
                    let logits_only_grads = logits_only_loss.backward();
                    let _ = probe_logits_leaf
                        .grad(&logits_only_grads)
                        .expect("probe logits grad")
                        .sum()
                        .into_data();
                    logits_loss_probe_ns = logits_only_start.elapsed().as_nanos();

                    let hidden_logits_start = Instant::now();
                    let probe_hidden_leaf = probe_hidden.require_grad();
                    let hidden_logits_loss = language_model_loss::<B>(
                        self.model.logits_from_hidden(probe_hidden_leaf.clone()),
                        probe_targets,
                    );
                    let hidden_logits_grads = hidden_logits_loss.backward();
                    let _ = probe_hidden_leaf
                        .grad(&hidden_logits_grads)
                        .expect("probe hidden grad")
                        .sum()
                        .into_data();
                    hidden_logits_loss_probe_ns = hidden_logits_start.elapsed().as_nanos();
                }
                if let Some(probe_inputs) = probe_inputs {
                    let hidden_model_start = Instant::now();
                    let probe_hidden_model =
                        if let Some(summary_event_mask) = probe_summary_event_mask {
                            let mut probe_state = self.model.init_state();
                            self.model
                                .forward_with_hidden_and_state_and_summary_event_mask(
                                    probe_inputs,
                                    summary_event_mask,
                                    &mut probe_state,
                                )
                                .0
                        } else {
                            self.model.forward_with_hidden(probe_inputs).0
                        };
                    let hidden_model_loss =
                        probe_hidden_model.clone().tanh().powf_scalar(2.0).mean();
                    let _hidden_model_grads = hidden_model_loss.backward();
                    let _ = B::sync(&probe_hidden_model.device());
                    hidden_model_probe_ns = hidden_model_start.elapsed().as_nanos();
                }
                crate::train::profile::record_detail_probe(
                    embed_probe_ns,
                    first_layer_forward_probe_ns,
                    first_layer_probe_ns,
                    logits_loss_probe_ns,
                    hidden_logits_loss_probe_ns,
                    hidden_model_forward_probe_ns,
                    hidden_model_probe_ns,
                );
            }
        }

        TrainOutput {
            grads: self.apply_gradient_scale_schedule(GradientsParams::from_grads(grads, self)),
            item: LanguageModelTrainItem::new(loss),
        }
    }
}

impl<B: BackendTrait> ValidStep for LanguageTrainModel<B> {
    type Input = SequenceBatch<B>;
    type Output = LanguageModelOutput<B>;

    fn step(&self, batch: SequenceBatch<B>) -> LanguageModelOutput<B> {
        let loss_mask = batch.loss_mask;
        if self.pipeline_enabled() {
            let (loss, _hidden, _logits) = self.forward_loss_with_pipeline(
                batch.inputs,
                batch.targets,
                loss_mask,
                batch.summary_event_mask,
            );
            return LanguageModelOutput::new(loss);
        }
        if let Some(summary_event_mask) = batch.summary_event_mask {
            if let Some(chunk_size) =
                self.effective_tbptt_chunk_size(batch.inputs.shape().dims::<2>()[1])
            {
                let [batch_size, block_size] = batch.inputs.shape().dims();
                let mut state = self.model.init_state();
                let mut loss: Option<Tensor<B, 1>> = None;
                for start in (0..block_size).step_by(chunk_size) {
                    let end = (start + chunk_size).min(block_size);
                    let chunk_inputs =
                        Self::slice_tokens(batch.inputs.clone(), batch_size, start, end);
                    let chunk_targets =
                        Self::slice_tokens(batch.targets.clone(), batch_size, start, end);
                    let chunk_loss_mask = loss_mask
                        .clone()
                        .map(|mask| Self::slice_tokens(mask, batch_size, start, end));
                    let chunk_mask =
                        Self::slice_tokens(summary_event_mask.clone(), batch_size, start, end);
                    let hidden = self.model.forward_hidden_with_state_and_summary_event_mask(
                        chunk_inputs,
                        chunk_mask,
                        &mut state,
                    );
                    let chunk_weight = (end - start) as f32 / block_size as f32;
                    let chunk_loss = self
                        .language_loss_from_hidden(hidden, chunk_targets, chunk_loss_mask)
                        .mul_scalar(chunk_weight);
                    loss = Some(match loss {
                        Some(accumulated) => accumulated + chunk_loss,
                        None => chunk_loss,
                    });
                }
                LanguageModelOutput::new(
                    loss.expect("tbptt valid step should produce at least one loss chunk"),
                )
            } else {
                let mut state = self.model.init_state();
                let hidden = self.model.forward_hidden_with_state_and_summary_event_mask(
                    batch.inputs,
                    summary_event_mask,
                    &mut state,
                );
                let loss = self.language_loss_from_hidden(hidden, batch.targets, loss_mask);
                LanguageModelOutput::new(loss)
            }
        } else if let Some(chunk_size) =
            self.effective_tbptt_chunk_size(batch.inputs.shape().dims::<2>()[1])
        {
            let [batch_size, block_size] = batch.inputs.shape().dims();
            let mut state = self.model.init_state();
            let mut loss: Option<Tensor<B, 1>> = None;
            for start in (0..block_size).step_by(chunk_size) {
                let end = (start + chunk_size).min(block_size);
                let chunk_inputs = Self::slice_tokens(batch.inputs.clone(), batch_size, start, end);
                let chunk_targets =
                    Self::slice_tokens(batch.targets.clone(), batch_size, start, end);
                let chunk_loss_mask = loss_mask
                    .clone()
                    .map(|mask| Self::slice_tokens(mask, batch_size, start, end));
                let hidden = self
                    .model
                    .forward_hidden_with_state(chunk_inputs, &mut state);
                let chunk_weight = (end - start) as f32 / block_size as f32;
                let chunk_loss = self
                    .language_loss_from_hidden(hidden, chunk_targets, chunk_loss_mask)
                    .mul_scalar(chunk_weight);
                loss = Some(match loss {
                    Some(accumulated) => accumulated + chunk_loss,
                    None => chunk_loss,
                });
            }
            LanguageModelOutput::new(
                loss.expect("tbptt valid step should produce at least one loss chunk"),
            )
        } else {
            let hidden = self.model.forward_hidden(batch.inputs);
            let loss = self.language_loss_from_hidden(hidden, batch.targets, loss_mask);
            LanguageModelOutput::new(loss)
        }
    }
}

impl<B: BackendTrait> LanguageTrainModel<B> {
    pub(crate) fn step_with_stream_state(
        &self,
        batch: SequenceBatch<B>,
        state: &mut ModelState<B>,
    ) -> LanguageModelOutput<B> {
        if batch.reset_stream_state {
            *state = self.model.init_state();
        }
        if self.pipeline_enabled() {
            return <Self as ValidStep>::step(self, batch);
        }
        let loss_mask = batch.loss_mask;
        if let Some(summary_event_mask) = batch.summary_event_mask {
            if let Some(chunk_size) =
                self.effective_tbptt_chunk_size(batch.inputs.shape().dims::<2>()[1])
            {
                let [batch_size, block_size] = batch.inputs.shape().dims();
                let mut loss: Option<Tensor<B, 1>> = None;
                for start in (0..block_size).step_by(chunk_size) {
                    let end = (start + chunk_size).min(block_size);
                    let chunk_inputs =
                        Self::slice_tokens(batch.inputs.clone(), batch_size, start, end);
                    let chunk_targets =
                        Self::slice_tokens(batch.targets.clone(), batch_size, start, end);
                    let chunk_loss_mask = loss_mask
                        .clone()
                        .map(|mask| Self::slice_tokens(mask, batch_size, start, end));
                    let chunk_mask =
                        Self::slice_tokens(summary_event_mask.clone(), batch_size, start, end);
                    let hidden = self.model.forward_hidden_with_state_and_summary_event_mask(
                        chunk_inputs,
                        chunk_mask,
                        state,
                    );
                    let chunk_weight = (end - start) as f32 / block_size as f32;
                    let chunk_loss = self
                        .language_loss_from_hidden(hidden, chunk_targets, chunk_loss_mask)
                        .mul_scalar(chunk_weight);
                    loss = Some(match loss {
                        Some(accumulated) => accumulated + chunk_loss,
                        None => chunk_loss,
                    });
                }
                return LanguageModelOutput::new(
                    loss.expect("streaming valid step should produce at least one loss chunk"),
                );
            }
            let hidden = self.model.forward_hidden_with_state_and_summary_event_mask(
                batch.inputs,
                summary_event_mask,
                state,
            );
            let loss = self.language_loss_from_hidden(hidden, batch.targets, loss_mask);
            return LanguageModelOutput::new(loss);
        }
        if let Some(chunk_size) =
            self.effective_tbptt_chunk_size(batch.inputs.shape().dims::<2>()[1])
        {
            let [batch_size, block_size] = batch.inputs.shape().dims();
            let mut loss: Option<Tensor<B, 1>> = None;
            for start in (0..block_size).step_by(chunk_size) {
                let end = (start + chunk_size).min(block_size);
                let chunk_inputs = Self::slice_tokens(batch.inputs.clone(), batch_size, start, end);
                let chunk_targets =
                    Self::slice_tokens(batch.targets.clone(), batch_size, start, end);
                let chunk_loss_mask = loss_mask
                    .clone()
                    .map(|mask| Self::slice_tokens(mask, batch_size, start, end));
                let hidden = self.model.forward_hidden_with_state(chunk_inputs, state);
                let chunk_weight = (end - start) as f32 / block_size as f32;
                let chunk_loss = self
                    .language_loss_from_hidden(hidden, chunk_targets, chunk_loss_mask)
                    .mul_scalar(chunk_weight);
                loss = Some(match loss {
                    Some(accumulated) => accumulated + chunk_loss,
                    None => chunk_loss,
                });
            }
            return LanguageModelOutput::new(
                loss.expect("streaming valid step should produce at least one loss chunk"),
            );
        }
        let hidden = self.model.forward_hidden_with_state(batch.inputs, state);
        let loss = self.language_loss_from_hidden(hidden, batch.targets, loss_mask);
        LanguageModelOutput::new(loss)
    }
}

fn output_degeneracy_from_logits<B: BackendTrait>(
    logits: Tensor<B, 3>,
    eos_id: Option<i64>,
) -> OutputDegeneracyStats {
    let [batch, time, vocab] = logits.shape().dims::<3>();
    if batch == 0 || time == 0 || vocab == 0 {
        return OutputDegeneracyStats::default();
    }
    let values = logits
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("validation degeneracy logits vec");
    let mut accumulator = OutputDegeneracyAccumulator::new(eos_id);

    for b in 0..batch {
        for t in 0..time {
            let start = (b * time + t) * vocab;
            if let Some(step) = output_degeneracy_step_from_row(&values[start..start + vocab]) {
                accumulator.record(step);
            }
        }
    }

    accumulator.finish()
}

fn validation_degeneracy_prompt_start(
    prompt_index: usize,
    prompt_count: usize,
    available: usize,
) -> usize {
    if available == 0 || prompt_index == 0 || prompt_count <= 1 {
        return 0;
    }
    let min_start = available.min(64);
    let interior = available.saturating_sub(min_start);
    let interior_index = prompt_index.saturating_sub(1);
    let interior_count = prompt_count.saturating_sub(1).max(1);
    min_start + (interior_index.saturating_mul(interior + 1) / interior_count).min(interior)
}

fn rollout_prompt_start(
    step_index: usize,
    every_steps: usize,
    block_size: usize,
    prompt_tokens: usize,
) -> usize {
    let available = block_size.saturating_sub(prompt_tokens);
    if available == 0 {
        return 0;
    }
    let min_start = available.min(prompt_tokens.max(1));
    let span = available.saturating_sub(min_start);
    if span == 0 {
        return min_start;
    }
    let rollout_index = step_index / every_steps.max(1);
    min_start + (rollout_index.saturating_mul(prompt_tokens.max(1)) % (span + 1))
}

fn lagged_prediction_tensors<B: BackendTrait>(
    log_probs: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
    clean_inputs: Tensor<B, 2, Int>,
    lag: usize,
    batch_size: usize,
    time: usize,
    vocab: usize,
) -> Option<(Tensor<B, 3>, Tensor<B, 2, Int>, Tensor<B, 2, Int>)> {
    if lag == 0 || time == 0 || lag > time {
        return None;
    }
    let start = lag.saturating_sub(1);
    let valid_time = time.saturating_sub(start);
    if valid_time == 0 {
        return None;
    }
    Some((
        log_probs.slice([0..batch_size, start..time, 0..vocab]),
        targets.slice([0..batch_size, start..time]),
        clean_inputs.slice([0..batch_size, 0..valid_time]),
    ))
}

fn unlikelihood_from_log_probs<B: BackendTrait>(
    log_probs: Tensor<B, 3>,
    tokens: Tensor<B, 2, Int>,
    epsilon: f32,
) -> Tensor<B, 2> {
    selected_token_log_probs(log_probs, tokens)
        .exp()
        .clamp_min(0.0)
        .clamp_max(1.0 - epsilon)
        .mul_scalar(-1.0)
        .add_scalar(1.0)
        .clamp_min(epsilon)
        .log()
        .mul_scalar(-1.0)
}

fn cycle_repeat_mask<B: BackendTrait>(
    next: &Tensor<B, 2, Int>,
    history: &[Tensor<B, 2, Int>],
    min_lag: usize,
    max_lag: usize,
) -> Option<Tensor<B, 2, burn::tensor::Bool>> {
    if history.is_empty() || min_lag == 0 || max_lag < min_lag {
        return None;
    }
    let mut mask: Option<Tensor<B, 2, burn::tensor::Bool>> = None;
    for lag in min_lag..=max_lag {
        let Some(previous) = history.get(lag.saturating_sub(1)) else {
            continue;
        };
        let lag_mask = next.clone().equal(previous.clone());
        mask = Some(match mask {
            Some(accumulated) => accumulated.bool_or(lag_mask),
            None => lag_mask,
        });
    }
    mask
}

#[derive(Clone, Copy, Debug)]
struct OutputDegeneracyStep {
    argmax: usize,
    entropy_bits: f64,
    max_probability: f64,
}

#[derive(Debug)]
struct OutputDegeneracyAccumulator {
    eos_id: Option<i64>,
    token_count: usize,
    entropy_sum: f64,
    max_probability_sum: f64,
    eos_count: usize,
    repetition_count: usize,
    repetition_denominator: usize,
    previous: Option<usize>,
    unique: HashSet<usize>,
    steps: Vec<OutputDegeneracyStep>,
    prompt_tokens: Vec<i64>,
    generated_tokens: Vec<i64>,
}

struct OutputDegeneracySummary {
    entropy_bits: f64,
    mean_max_probability: f64,
    argmax_unique_fraction: f64,
    repetition_fraction: f64,
}

impl OutputDegeneracyAccumulator {
    const MIN_PAYLOAD_TOKENS_BEFORE_EOS_PADDING: usize = 16;

    fn new(eos_id: Option<i64>) -> Self {
        Self {
            eos_id,
            token_count: 0,
            entropy_sum: 0.0,
            max_probability_sum: 0.0,
            eos_count: 0,
            repetition_count: 0,
            repetition_denominator: 0,
            previous: None,
            unique: HashSet::new(),
            steps: Vec::new(),
            prompt_tokens: Vec::new(),
            generated_tokens: Vec::new(),
        }
    }

    fn record(&mut self, step: OutputDegeneracyStep) {
        self.entropy_sum += step.entropy_bits;
        self.max_probability_sum += step.max_probability;
        if self
            .eos_id
            .is_some_and(|id| id >= 0 && step.argmax == id as usize)
        {
            self.eos_count = self.eos_count.saturating_add(1);
        }
        if let Some(previous) = self.previous {
            self.repetition_denominator = self.repetition_denominator.saturating_add(1);
            if previous == step.argmax {
                self.repetition_count = self.repetition_count.saturating_add(1);
            }
        }
        self.previous = Some(step.argmax);
        self.unique.insert(step.argmax);
        self.steps.push(step);
        self.token_count = self.token_count.saturating_add(1);
    }

    fn record_generated_token(&mut self, token: i64) {
        self.generated_tokens.push(token);
    }

    fn record_prompt_tokens(&mut self, tokens: impl IntoIterator<Item = i64>) {
        self.prompt_tokens.extend(tokens);
    }

    fn finish(self) -> OutputDegeneracyStats {
        if self.token_count == 0 {
            return OutputDegeneracyStats::default();
        }
        let first_eos_index = self.eos_id.and_then(|eos_id| {
            self.generated_tokens
                .iter()
                .position(|token| *token == eos_id)
        });
        let scored_len = first_eos_index
            .filter(|index| *index >= Self::MIN_PAYLOAD_TOKENS_BEFORE_EOS_PADDING)
            .unwrap_or(self.generated_tokens.len())
            .min(self.steps.len());
        let scored_steps = &self.steps[..scored_len];
        let scored_generated_tokens = &self.generated_tokens[..scored_len];
        let scored = Self::summarize_steps(scored_steps).unwrap_or_else(|| {
            Self::summarize_steps(&self.steps).expect("non-empty output degeneracy accumulator")
        });
        let eos_fraction = if scored_len < self.generated_tokens.len() {
            0.0
        } else {
            self.eos_count as f64 / self.token_count as f64
        };
        let distinct_1_fraction = distinct_n_fraction(scored_generated_tokens, 1);
        let distinct_2_fraction = distinct_n_fraction(scored_generated_tokens, 2);
        let period_2_fraction = period_fraction(scored_generated_tokens, 2);
        let period_3_fraction = period_fraction(scored_generated_tokens, 3);
        let max_period_2_to_16_fraction = max_period_fraction(scored_generated_tokens, 2..=16);
        let (dominant_period_2_to_64, max_period_2_to_64_fraction) =
            dominant_period_fraction(scored_generated_tokens, 2..=64);
        let (prompt_dominant_period_2_to_64, prompt_max_period_2_to_64_fraction) =
            dominant_period_fraction(&self.prompt_tokens, 2..=64);
        OutputDegeneracyStats {
            token_count: self.token_count,
            entropy_bits: scored.entropy_bits,
            mean_max_probability: scored.mean_max_probability,
            argmax_unique_fraction: scored.argmax_unique_fraction,
            eos_fraction,
            repetition_fraction: scored.repetition_fraction,
            distinct_1_fraction,
            distinct_2_fraction,
            period_2_fraction,
            period_3_fraction,
            max_period_2_to_16_fraction,
            max_period_2_to_64_fraction,
            dominant_period_2_to_64,
            prompt_max_period_2_to_64_fraction,
            prompt_dominant_period_2_to_64,
            prompt_tokens: self.prompt_tokens,
            generated_tokens: self.generated_tokens,
        }
    }

    fn summarize_steps(steps: &[OutputDegeneracyStep]) -> Option<OutputDegeneracySummary> {
        if steps.is_empty() {
            return None;
        }
        let mut entropy_sum = 0.0;
        let mut max_probability_sum = 0.0;
        let mut unique = HashSet::new();
        let mut previous = None;
        let mut repetition_count = 0usize;
        let mut repetition_denominator = 0usize;
        for step in steps {
            entropy_sum += step.entropy_bits;
            max_probability_sum += step.max_probability;
            unique.insert(step.argmax);
            if let Some(previous) = previous {
                repetition_denominator = repetition_denominator.saturating_add(1);
                if previous == step.argmax {
                    repetition_count = repetition_count.saturating_add(1);
                }
            }
            previous = Some(step.argmax);
        }
        Some(OutputDegeneracySummary {
            entropy_bits: entropy_sum / steps.len() as f64,
            mean_max_probability: max_probability_sum / steps.len() as f64,
            argmax_unique_fraction: unique.len() as f64 / steps.len() as f64,
            repetition_fraction: if repetition_denominator == 0 {
                0.0
            } else {
                repetition_count as f64 / repetition_denominator as f64
            },
        })
    }
}

fn distinct_n_fraction(tokens: &[i64], n: usize) -> f64 {
    if n == 0 || tokens.len() < n {
        return 0.0;
    }
    let total = tokens.len() + 1 - n;
    let distinct = tokens
        .windows(n)
        .map(|window| window.to_vec())
        .collect::<HashSet<_>>()
        .len();
    distinct as f64 / total as f64
}

fn period_fraction(tokens: &[i64], period: usize) -> f64 {
    if period == 0 || tokens.len() < period.saturating_mul(2) {
        return 0.0;
    }
    let matches = (period..tokens.len())
        .filter(|idx| tokens[*idx] == tokens[*idx - period])
        .count();
    matches as f64 / (tokens.len() - period) as f64
}

fn max_period_fraction(tokens: &[i64], periods: impl IntoIterator<Item = usize>) -> f64 {
    dominant_period_fraction(tokens, periods).1
}

fn dominant_period_fraction(
    tokens: &[i64],
    periods: impl IntoIterator<Item = usize>,
) -> (usize, f64) {
    periods
        .into_iter()
        .map(|period| (period, period_fraction(tokens, period)))
        .max_by(|(_, left), (_, right)| {
            left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or((0, 0.0))
}

fn selected_token_logits<B: BackendTrait>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
) -> Tensor<B, 2> {
    let [batch, time, _vocab] = logits.shape().dims();
    logits
        .gather(2, targets.reshape([batch, time, 1]))
        .reshape([batch, time])
}

fn answer_prefix_input_mask<B: BackendTrait>(loss_mask: Tensor<B, 2, Int>) -> Tensor<B, 2, Int> {
    let [batch, time] = loss_mask.shape().dims();
    let device = loss_mask.device();
    if time == 0 {
        return Tensor::<B, 2, Int>::zeros([batch, 0], &device);
    }
    let head = Tensor::<B, 2, Int>::zeros([batch, 1], &device);
    if time == 1 {
        return head;
    }
    let previous_targets = loss_mask.slice([0..batch, 0..(time - 1)]);
    Tensor::cat(vec![head, previous_targets], 1)
}

fn next_token_loss_from_log_probs<B: BackendTrait>(
    log_probs: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
    loss_mask: Option<Tensor<B, 2, Int>>,
) -> Tensor<B, 1> {
    masked_token_mean(
        selected_token_log_probs(log_probs, targets).mul_scalar(-1.0),
        loss_mask,
    )
}

fn entropy_floor_loss_from_logits<B: BackendTrait>(
    logits: Tensor<B, 3>,
    target_entropy_bits: f32,
) -> Option<Tensor<B, 1>> {
    entropy_floor_loss_from_log_probs(log_probs_from_logits(logits), target_entropy_bits)
}

fn entropy_floor_loss_from_log_probs<B: BackendTrait>(
    log_probs: Tensor<B, 3>,
    target_entropy_bits: f32,
) -> Option<Tensor<B, 1>> {
    let [batch, time, vocab] = log_probs.shape().dims();
    if batch == 0 || time == 0 || vocab == 0 || target_entropy_bits <= f32::EPSILON {
        return None;
    }
    let flat_log_probs = log_probs.reshape([batch * time, vocab]);
    let flat_probs = flat_log_probs.clone().exp();
    entropy_floor_loss_from_flat_log_probs(flat_log_probs, flat_probs, target_entropy_bits)
}

fn entropy_floor_loss_from_flat_log_probs<B: BackendTrait>(
    flat_log_probs: Tensor<B, 2>,
    flat_probs: Tensor<B, 2>,
    target_entropy_bits: f32,
) -> Option<Tensor<B, 1>> {
    let [token_count, vocab] = flat_log_probs.shape().dims();
    if token_count == 0 || vocab == 0 || target_entropy_bits <= f32::EPSILON {
        return None;
    }
    let entropy = (flat_probs * flat_log_probs)
        .sum_dim(1)
        .mul_scalar(-1.0)
        .mean()
        .reshape([1]);
    let target_nats = target_entropy_bits * std::f32::consts::LN_2;
    Some(
        entropy
            .mul_scalar(-1.0)
            .add_scalar(target_nats)
            .clamp_min(0.0),
    )
}

fn predicted_marginal_from_logits<B: BackendTrait>(logits: Tensor<B, 3>) -> Option<Tensor<B, 2>> {
    predicted_marginal_from_log_probs(log_probs_from_logits(logits))
}

fn predicted_marginal_from_log_probs<B: BackendTrait>(
    log_probs: Tensor<B, 3>,
) -> Option<Tensor<B, 2>> {
    let [batch, time, vocab] = log_probs.shape().dims();
    if batch == 0 || time == 0 || vocab == 0 {
        return None;
    }
    Some(log_probs.reshape([batch * time, vocab]).exp().mean_dim(0))
}

fn marginal_entropy_floor_loss_from_logits<B: BackendTrait>(
    logits: Tensor<B, 3>,
    target_entropy_bits: f32,
) -> Option<Tensor<B, 1>> {
    marginal_entropy_floor_loss_from_marginal(
        predicted_marginal_from_logits(logits)?,
        target_entropy_bits,
    )
}

fn marginal_entropy_floor_loss_from_marginal<B: BackendTrait>(
    marginal: Tensor<B, 2>,
    target_entropy_bits: f32,
) -> Option<Tensor<B, 1>> {
    if target_entropy_bits <= f32::EPSILON {
        return None;
    }
    let entropy = (marginal.clone() * marginal.clamp_min(1.0e-12).log())
        .sum_dim(1)
        .mul_scalar(-1.0)
        .reshape([1]);
    let target_nats = target_entropy_bits * std::f32::consts::LN_2;
    Some(
        entropy
            .mul_scalar(-1.0)
            .add_scalar(target_nats)
            .clamp_min(0.0),
    )
}

fn target_marginal_coverage_loss_from_logits<B: BackendTrait>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
    epsilon: f32,
) -> Option<Tensor<B, 1>> {
    target_marginal_coverage_loss_from_marginal(
        predicted_marginal_from_logits(logits)?,
        targets,
        epsilon,
    )
}

fn target_marginal_coverage_loss_from_marginal<B: BackendTrait>(
    marginal: Tensor<B, 2>,
    targets: Tensor<B, 2, Int>,
    epsilon: f32,
) -> Option<Tensor<B, 1>> {
    let [_marginal_batch, vocab] = marginal.shape().dims();
    if vocab == 0 || epsilon <= 0.0 || epsilon >= 1.0 {
        return None;
    }
    let [batch, time] = targets.shape().dims();
    let token_count = batch * time;
    if token_count == 0 {
        return None;
    }
    let log_marginal = marginal.clamp_min(epsilon).log().repeat_dim(0, token_count);
    Some(
        log_marginal
            .gather(1, targets.reshape([token_count, 1]))
            .mean()
            .reshape([1])
            .mul_scalar(-1.0),
    )
}

fn output_degeneracy_step_from_logits<B: BackendTrait>(
    logits: Tensor<B, 1>,
) -> Option<OutputDegeneracyStep> {
    let values = logits
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("validation free-running degeneracy logits vec");
    output_degeneracy_step_from_row(&values)
}

fn output_degeneracy_step_from_row(row: &[f32]) -> Option<OutputDegeneracyStep> {
    let (argmax, max_logit) = row
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|(_, left), (_, right)| {
            left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
        })?;
    let mut exp_sum = 0.0f64;
    let mut weighted_logit_sum = 0.0f64;
    for value in row.iter().copied().filter(|value| value.is_finite()) {
        let weight = (value as f64 - max_logit as f64).exp();
        exp_sum += weight;
        weighted_logit_sum += weight * value as f64;
    }
    if exp_sum <= 0.0 || !exp_sum.is_finite() {
        return None;
    }
    let logsumexp = max_logit as f64 + exp_sum.ln();
    let entropy_nats = logsumexp - weighted_logit_sum / exp_sum;
    Some(OutputDegeneracyStep {
        argmax,
        entropy_bits: entropy_nats.max(0.0) / std::f64::consts::LN_2,
        max_probability: 1.0 / exp_sum,
    })
}

#[cfg(test)]
mod objective_step_tests {
    use super::*;
    use burn_autodiff::Autodiff;
    use burn_ndarray::NdArray;

    type TestBackend = Autodiff<NdArray<f32>>;
    type TestInnerBackend = NdArray<f32>;

    fn tensor_scalar(tensor: Tensor<TestBackend, 1>) -> f32 {
        tensor
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("scalar tensor")[0]
    }

    fn tiny_model_config() -> DragonConfig {
        DragonConfig {
            n_layer: 1,
            n_embd: 8,
            n_head: 1,
            mlp_internal_dim_multiplier: 1,
            dropout: 0.0,
            vocab_size: 16,
            ..Default::default()
        }
    }

    fn tiny_factorized_model_config() -> DragonConfig {
        let mut config = tiny_model_config();
        config.vocab_size = 32;
        config.language_head = burn_dragon_core::LanguageHeadConfig::NcaFactorizedPatch {
            state_count: 2,
            patch_size: 2,
            frame_special_tokens: true,
            eos_id: Some(31),
        };
        config
    }

    fn tiny_pipeline_plan() -> PipelinePlan {
        build_pipeline_plan(
            2,
            &burn_dragon_train::ParallelPipelineConfig {
                enabled: true,
                stage_count: 2,
                virtual_stages_per_rank: 1,
                schedule: burn_dragon_train::PipelineScheduleKind::Interleaved1f1b,
                microbatches: 2,
                ..Default::default()
            },
        )
        .expect("pipeline plan")
    }

    fn batch(device: &burn::tensor::Device<TestBackend>) -> SequenceBatch<TestBackend> {
        SequenceBatch::new(
            Tensor::<TestBackend, 2, Int>::from_data(
                TensorData::new(vec![0, 1, 2, 3, 4, 5, 6, 7], [2, 4]),
                device,
            ),
            Tensor::<TestBackend, 2, Int>::from_data(
                TensorData::new(vec![1, 2, 3, 4, 5, 6, 7, 8], [2, 4]),
                device,
            ),
            None,
        )
    }

    fn scalar_loss(output: TrainOutput<LanguageModelTrainItem<TestBackend>>) -> f32 {
        let synced = output.item.sync();
        let loss: LossValue<TestInnerBackend> = synced.adapt();
        loss.value()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("loss vec")[0]
    }

    #[test]
    fn predictive_coding_corrects_tbptt_state_directly() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 11);
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_tbptt_chunk_size(Some(2))
        .with_predictive_coding(PredictiveCodingConfig {
            enabled: true,
            steps: 1,
            step_size: 0.01,
            sync_diagnostics: true,
            ..Default::default()
        });
        let batch = batch(&device);
        let [batch_size, _block_size] = batch.inputs.shape().dims();
        let mut state = model.model.init_state_ephemeral();
        let first_inputs =
            LanguageTrainModel::<TestBackend>::slice_tokens(batch.inputs.clone(), batch_size, 0, 2);
        let _ = model
            .model
            .forward_hidden_with_state(first_inputs, &mut state);
        state.detach_in_place();

        let second_inputs =
            LanguageTrainModel::<TestBackend>::slice_tokens(batch.inputs, batch_size, 2, 4);
        let second_targets =
            LanguageTrainModel::<TestBackend>::slice_tokens(batch.targets, batch_size, 2, 4);
        let (_corrected_state, report) = model.correct_state_with_predictive_coding(
            state,
            second_inputs,
            second_targets,
            None,
            None,
        );

        assert!(
            report.chunks_seen > 0,
            "PC should observe at least one TBPTT state handoff, report={report:?}"
        );
        assert!(
            report.chunks_corrected > 0,
            "PC should correct at least one recurrent state, report={report:?}"
        );
        assert!(
            report.chunks_corrected <= report.chunks_seen,
            "corrected chunks should be bounded by observed chunks, report={report:?}"
        );
        assert!(
            report
                .energy_before
                .zip(report.energy_after)
                .is_some_and(|(before, after)| before.is_finite() && after.is_finite()),
            "PC should record finite before/after energy, report={report:?}"
        );
    }

    fn require_grad_param_count<B: BackendTrait>(model: &DragonModel<B>) -> usize {
        #[derive(Default)]
        struct RequireGradCounter {
            count: usize,
        }

        impl<B: BackendTrait> burn::module::ModuleVisitor<B> for RequireGradCounter {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
                self.count += usize::from(param.val().is_require_grad());
            }
        }

        let mut counter = RequireGradCounter::default();
        model.visit(&mut counter);
        counter.count
    }

    #[test]
    fn teacher_runtime_detaches_trainable_parameters() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let model = DragonModel::<TestBackend>::new(tiny_model_config(), &device);
        assert!(
            require_grad_param_count(&model) > 0,
            "training model should own trainable autodiff parameters"
        );

        let teacher = TeacherModelRuntime::new(model);
        assert_eq!(
            require_grad_param_count(&teacher.model),
            0,
            "teacher snapshots must not build parameter-gradient graphs"
        );
    }

    #[test]
    fn neuron_scale_3d_gradient_scaling_preserves_headed_tail_semantics() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let tensor = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 1, 4]),
            &device,
        );
        let scaled = scale_3d_latent_tail(tensor, 2, 4, 0.5, 2.0)
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("scaled 3d gradient");
        assert_eq!(scaled, vec![0.5, 1.0, 6.0, 8.0, 2.5, 3.0, 14.0, 16.0]);
    }

    #[test]
    fn neuron_scale_2d_gradient_scaling_preserves_headed_tail_semantics() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let tensor = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(
                (1..=16).map(|value| value as f32).collect::<Vec<_>>(),
                [8, 2],
            ),
            &device,
        );
        let scaled = scale_2d_headed_latent_rows(tensor, 2, 4, 0.5, 2.0)
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("scaled 2d gradient");
        assert_eq!(
            scaled,
            vec![
                0.5, 1.0, 1.5, 2.0, 10.0, 12.0, 14.0, 16.0, 4.5, 5.0, 5.5, 6.0, 26.0, 28.0, 30.0,
                32.0,
            ]
        );
    }

    #[test]
    fn output_degeneracy_step_reports_overconfident_argmax() {
        let step =
            output_degeneracy_step_from_row(&[12.0, -8.0, -9.0, -10.0]).expect("finite step");
        assert_eq!(step.argmax, 0);
        assert!(
            step.entropy_bits < 0.001,
            "unexpected entropy: {}",
            step.entropy_bits
        );
        assert!(
            step.max_probability > 0.999,
            "unexpected max probability: {}",
            step.max_probability
        );
    }

    #[test]
    fn output_degeneracy_accumulator_tracks_repetition_and_eos() {
        let mut accumulator = OutputDegeneracyAccumulator::new(Some(2));
        for argmax in [2, 2, 3, 3] {
            accumulator.record(OutputDegeneracyStep {
                argmax,
                entropy_bits: 0.25,
                max_probability: 0.9,
            });
            accumulator.record_generated_token(argmax as i64);
        }
        let stats = accumulator.finish();
        assert_eq!(stats.token_count, 4);
        assert_eq!(stats.argmax_unique_fraction, 0.5);
        assert_eq!(stats.eos_fraction, 0.5);
        assert!((stats.repetition_fraction - (2.0 / 3.0)).abs() < 1e-12);
        assert_eq!(stats.distinct_1_fraction, 0.5);
        assert_eq!(stats.distinct_2_fraction, 1.0);
        assert_eq!(stats.period_2_fraction, 0.0);
    }

    #[test]
    fn output_degeneracy_accumulator_ignores_eos_padding_after_payload() {
        let eos_id = 99usize;
        let mut accumulator = OutputDegeneracyAccumulator::new(Some(eos_id as i64));
        for argmax in (0usize..24).chain(std::iter::repeat(eos_id).take(40)) {
            accumulator.record(OutputDegeneracyStep {
                argmax,
                entropy_bits: if argmax == eos_id { 0.01 } else { 2.0 },
                max_probability: if argmax == eos_id { 0.99 } else { 0.3 },
            });
            accumulator.record_generated_token(argmax as i64);
        }
        let stats = accumulator.finish();
        assert_eq!(stats.token_count, 64);
        assert_eq!(
            stats.eos_fraction, 0.0,
            "EOS padding after a payload should not trip EOS collapse"
        );
        assert!(
            stats.repetition_fraction < 0.01,
            "payload repetition should be scored before EOS padding: {}",
            stats.repetition_fraction
        );
        assert!(
            stats.entropy_bits > 1.9,
            "payload entropy should be scored before EOS padding: {}",
            stats.entropy_bits
        );
        assert_eq!(stats.distinct_1_fraction, 1.0);
    }

    #[test]
    fn output_degeneracy_accumulator_tracks_long_period_cycles() {
        let mut accumulator = OutputDegeneracyAccumulator::new(None);
        for index in 0..128 {
            let argmax = index % 37;
            accumulator.record(OutputDegeneracyStep {
                argmax,
                entropy_bits: 4.0,
                max_probability: 0.25,
            });
            accumulator.record_generated_token(argmax as i64);
        }
        let stats = accumulator.finish();
        assert!(
            stats.max_period_2_to_16_fraction < 0.05,
            "period-2..16 should not catch a period-37 loop: {}",
            stats.max_period_2_to_16_fraction
        );
        assert_eq!(stats.dominant_period_2_to_64, 37);
        assert!(
            stats.max_period_2_to_64_fraction > 0.95,
            "expected high extended long-cycle fraction, got {}",
            stats.max_period_2_to_64_fraction
        );
        assert!(
            stats.period_2_fraction < 0.01 && stats.period_3_fraction < 0.01,
            "period-2/3 should not catch a period-37 loop"
        );
    }

    #[test]
    fn output_degeneracy_ignores_single_comparison_long_period_aliases() {
        let mut tokens: Vec<i64> = (0..32).collect();
        tokens[31] = tokens[0];

        assert_eq!(period_fraction(&tokens, 31), 0.0);
        let (dominant_period, max_fraction) = dominant_period_fraction(&tokens, 2..=64);

        assert_ne!(
            dominant_period, 31,
            "period-31 only has one comparison over a 32-token probe"
        );
        assert!(
            max_fraction < 1.0,
            "single-comparison alias should not produce a perfect period score"
        );
    }

    #[test]
    fn validation_degeneracy_probe_rolls_out_generated_tokens() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ));
        let (_loss, stats) = model.validation_loss_and_output_degeneracy(batch(&device), 3, None);
        let stats = stats.expect("free-running degeneracy stats");
        assert_eq!(stats.token_count, 6);
        assert!(stats.entropy_bits.is_finite());
        assert!(stats.mean_max_probability.is_finite());
        assert!((0.0..=1.0).contains(&stats.argmax_unique_fraction));
        assert!((0.0..=1.0).contains(&stats.repetition_fraction));
        assert_eq!(stats.generated_tokens.len(), 6);
        assert!((0.0..=1.0).contains(&stats.distinct_1_fraction));
        assert!((0.0..=1.0).contains(&stats.distinct_2_fraction));
        assert!((0.0..=1.0).contains(&stats.period_2_fraction));
        assert!((0.0..=1.0).contains(&stats.period_3_fraction));
    }

    #[test]
    fn validation_degeneracy_prompts_cover_header_and_interior_windows() {
        let starts = (0..4)
            .map(|index| validation_degeneracy_prompt_start(index, 4, 224))
            .collect::<Vec<_>>();
        assert_eq!(starts[0], 0);
        assert!(starts[1] >= 64, "{starts:?}");
        assert!(starts[3] <= 224, "{starts:?}");
        assert!(starts.windows(2).all(|window| window[0] <= window[1]));
    }

    #[test]
    fn rollout_unlikelihood_prompt_rotates_away_from_header() {
        let first = rollout_prompt_start(0, 1, 256, 32);
        let later = rollout_prompt_start(1, 1, 256, 32);
        assert_eq!(first, 32);
        assert_ne!(first, later);
        assert!(later > 0);
    }

    #[test]
    fn selected_token_logits_gathers_raw_logits_not_log_probs() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let logits = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![1.0, 2.0, 9.0, -1.0, 4.0, 3.0, 7.0, 8.0], [1, 2, 4]),
            &device,
        );
        let targets =
            Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![2, 0], [1, 2]), &device);
        let selected = selected_token_logits(logits, targets)
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("selected logits");
        assert_eq!(selected, vec![9.0, 4.0]);
    }

    #[test]
    fn causal_input_corruption_replaces_inputs_with_fixed_token() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_input_corruption(CausalInputCorruptionConfig {
            enabled: true,
            probability: 1.0,
            replacement_token_id: Some(3),
            ..Default::default()
        });
        let inputs = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![0, 1, 2, 4, 5, 6], [2, 3]),
            &device,
        );
        let corrupted = model.corrupt_causal_inputs(inputs);
        let values = corrupted
            .to_data()
            .convert::<i64>()
            .into_vec::<i64>()
            .expect("corrupted inputs");
        assert_eq!(values, vec![3; 6]);
    }

    #[test]
    fn causal_input_corruption_respects_warmup() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_input_corruption(CausalInputCorruptionConfig {
            enabled: true,
            probability: 1.0,
            warmup_steps: 10,
            replacement_token_id: Some(3),
            ..Default::default()
        });
        let inputs = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![0, 1, 2, 4, 5, 6], [2, 3]),
            &device,
        );
        let corrupted = model.corrupt_causal_inputs(inputs);
        let values = corrupted
            .to_data()
            .convert::<i64>()
            .into_vec::<i64>()
            .expect("corrupted inputs");
        assert_eq!(values, vec![0, 1, 2, 4, 5, 6]);
    }

    #[test]
    fn next_token_loss_honors_optional_target_mask() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ));
        let logits = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![8.0, 0.0, 0.0, 8.0], [1, 2, 2]),
            &device,
        );
        let clean_inputs =
            Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![0, 1], [1, 2]), &device);
        let targets =
            Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![0, 0], [1, 2]), &device);
        let first_only_mask =
            Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![1, 0], [1, 2]), &device);

        let unmasked = tensor_scalar(model.next_token_loss_from_logits(
            logits.clone(),
            targets.clone(),
            clean_inputs.clone(),
            None,
            None,
        ));
        let masked = tensor_scalar(model.next_token_loss_from_logits(
            logits,
            targets,
            clean_inputs,
            Some(first_only_mask),
            None,
        ));

        assert!(unmasked > masked + 3.0);
        assert!(
            masked < 1.0e-3,
            "masked loss should keep only the confident first token"
        );
    }

    #[test]
    fn ruliad_answer_ranking_penalizes_corrupt_answer_logits() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_ruliad_supervision(RuliadSupervisionConfig {
            mode: RuliadSupervisionMode::AnswerCompletion,
            answer_ranking: RuliadAnswerRankingConfig {
                enabled: true,
                weight: 1.0,
                margin: 0.5,
                corrupt_offset: 1,
            },
            ..Default::default()
        });
        let targets =
            Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![1, 2], [1, 2]), &device);
        let mask =
            Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![1, 1], [1, 2]), &device);
        let preferred = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                vec![
                    0.0, 5.0, -2.0, 0.0, //
                    0.0, 0.0, 5.0, -2.0,
                ],
                [1, 2, 4],
            ),
            &device,
        );
        let inverted = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                vec![
                    0.0, -2.0, 5.0, 0.0, //
                    0.0, 0.0, -2.0, 5.0,
                ],
                [1, 2, 4],
            ),
            &device,
        );

        let preferred_loss = tensor_scalar(
            model
                .ruliad_answer_ranking_loss_from_logits(
                    preferred,
                    targets.clone(),
                    Some(mask.clone()),
                )
                .expect("preferred ranking loss"),
        );
        let inverted_loss = tensor_scalar(
            model
                .ruliad_answer_ranking_loss_from_logits(inverted, targets, Some(mask))
                .expect("inverted ranking loss"),
        );

        assert!(
            inverted_loss > preferred_loss + 5.0,
            "ranking loss should reward oracle answer logits over corrupt answer logits: preferred={preferred_loss} inverted={inverted_loss}"
        );
    }

    #[test]
    fn answer_prefix_input_mask_shifts_target_answer_mask_right() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let mask = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![0, 1, 1, 0, 1], [1, 5]),
            &device,
        );
        let shifted = answer_prefix_input_mask(mask)
            .to_data()
            .convert::<i64>()
            .into_vec::<i64>()
            .expect("shifted mask");
        assert_eq!(shifted, vec![0, 0, 1, 1, 0]);
    }

    #[test]
    fn ruliad_answer_denoising_corrupts_only_answer_prefix_inputs() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ));
        let inputs = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![10, 11, 12, 13, 14], [1, 5]),
            &device,
        );
        let target_mask = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![0, 1, 1, 0, 1], [1, 5]),
            &device,
        );
        let prefix_mask = answer_prefix_input_mask(target_mask);
        let corrupted = model
            .corrupt_ruliad_answer_prefix_inputs(
                inputs,
                prefix_mask,
                RuliadAnswerDenoisingConfig {
                    enabled: true,
                    weight: 1.0,
                    probability: 1.0,
                    corrupt_offset: 1,
                },
            )
            .to_data()
            .convert::<i64>()
            .into_vec::<i64>()
            .expect("corrupted inputs");
        assert_eq!(corrupted, vec![10, 11, 13, 14, 14]);
    }

    #[test]
    fn ruliad_answer_denoising_loss_is_finite_for_masked_answer_batch() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_ruliad_supervision(RuliadSupervisionConfig {
            mode: RuliadSupervisionMode::AnswerCompletion,
            answer_denoising: RuliadAnswerDenoisingConfig {
                enabled: true,
                weight: 0.5,
                probability: 1.0,
                corrupt_offset: 1,
            },
            ..Default::default()
        });
        let inputs = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![0, 1, 2, 3, 4, 5], [1, 6]),
            &device,
        );
        let targets = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1, 2, 3, 4, 5, 6], [1, 6]),
            &device,
        );
        let mask = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![0, 1, 1, 1, 0, 0], [1, 6]),
            &device,
        );
        let loss = tensor_scalar(
            model
                .ruliad_answer_denoising_loss(inputs, targets, Some(mask))
                .expect("denoising loss"),
        );
        assert!(loss.is_finite(), "denoising loss should be finite: {loss}");
        assert!(loss > 0.0, "denoising loss should be non-zero: {loss}");
    }

    #[test]
    fn dynamics_anchor_penalizes_teacher_distribution_drift() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let plain = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ));
        let anchored = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_dynamics_anchor(DynamicsAnchorConfig {
            enabled: true,
            weight: 1.0,
            teacher_update_rate: 0.0,
            kl: SelfDistillationKlKind::Forward,
            ..Default::default()
        });
        let student_logits = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![8.0, 0.0, 8.0, 0.0], [1, 2, 2]),
            &device,
        );
        let teacher_logits = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![0.0, 8.0, 0.0, 8.0], [1, 2, 2]),
            &device,
        );
        let clean_inputs =
            Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![0, 1], [1, 2]), &device);
        let targets =
            Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![0, 0], [1, 2]), &device);

        let ce = tensor_scalar(plain.next_token_loss_from_logits(
            student_logits.clone(),
            targets.clone(),
            clean_inputs.clone(),
            None,
            None,
        ));
        let anchored_loss = tensor_scalar(anchored.next_token_loss_from_logits(
            student_logits,
            targets,
            clean_inputs,
            None,
            Some(teacher_logits),
        ));

        assert!(
            anchored_loss > ce + 6.0,
            "anchor should add KL pressure when student diverges from teacher: ce={ce} anchored={anchored_loss}"
        );
    }

    #[test]
    fn dynamics_anchor_context_mask_uses_unsupervised_tokens() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_dynamics_anchor(DynamicsAnchorConfig {
            enabled: true,
            weight: 1.0,
            mask: DynamicsAnchorMask::ContextTokens,
            ..Default::default()
        });
        let target_mask = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1, 0, 1], [1, 3]),
            &device,
        );
        let context_mask = model
            .dynamics_anchor_mask(Some(target_mask))
            .expect("context mask")
            .to_data()
            .convert::<i64>()
            .into_vec::<i64>()
            .expect("mask values");

        assert_eq!(context_mask, vec![0, 1, 0]);
    }

    #[test]
    fn repeat_unlikelihood_penalizes_wrong_copy_predictions() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let plain = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ));
        let repeat_penalized = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_repeat_unlikelihood(RepeatUnlikelihoodConfig {
            enabled: true,
            weight: 0.5,
            ..Default::default()
        });
        let logits = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![5.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0], [1, 2, 4]),
            &device,
        );
        let clean_inputs =
            Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![0, 1], [1, 2]), &device);
        let targets =
            Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![1, 2], [1, 2]), &device);
        let ce = plain.next_token_loss_from_logits(
            logits.clone(),
            targets.clone(),
            clean_inputs.clone(),
            None,
            None,
        );
        let penalized =
            repeat_penalized.next_token_loss_from_logits(logits, targets, clean_inputs, None, None);
        let ce_value = ce.to_data().convert::<f32>().into_vec::<f32>().expect("ce")[0];
        let penalized_value = penalized
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("penalized")[0];
        assert!(
            penalized_value > ce_value,
            "repeat unlikelihood should increase loss for wrong-copy logits: ce={ce_value} penalized={penalized_value}"
        );
    }

    #[test]
    fn logit_entropy_floor_penalizes_overconfident_logits() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let plain = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ));
        let entropy_penalized = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_logit_entropy_floor(LogitEntropyFloorConfig {
            enabled: true,
            weight: 0.5,
            target_entropy_bits: 2.0,
            ..Default::default()
        });
        let logits = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![8.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0], [1, 2, 4]),
            &device,
        );
        let clean_inputs =
            Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![0, 1], [1, 2]), &device);
        let targets =
            Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![0, 1], [1, 2]), &device);
        let ce = plain.next_token_loss_from_logits(
            logits.clone(),
            targets.clone(),
            clean_inputs.clone(),
            None,
            None,
        );
        let penalized = entropy_penalized.next_token_loss_from_logits(
            logits,
            targets,
            clean_inputs,
            None,
            None,
        );
        let ce_value = ce.to_data().convert::<f32>().into_vec::<f32>().expect("ce")[0];
        let penalized_value = penalized
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("penalized")[0];
        assert!(
            penalized_value > ce_value,
            "entropy floor should increase loss for overconfident logits: ce={ce_value} penalized={penalized_value}"
        );
    }

    #[test]
    fn logit_entropy_floor_respects_every_steps() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let plain = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ));
        let throttled = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_logit_entropy_floor(LogitEntropyFloorConfig {
            enabled: true,
            weight: 0.5,
            target_entropy_bits: 2.0,
            every_steps: 4,
            ..Default::default()
        });
        let logits = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![8.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0], [1, 2, 4]),
            &device,
        );
        let clean_inputs =
            Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![0, 1], [1, 2]), &device);
        let targets =
            Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![0, 1], [1, 2]), &device);
        let ce = tensor_scalar(plain.next_token_loss_from_logits(
            logits.clone(),
            targets.clone(),
            clean_inputs.clone(),
            None,
            None,
        ));
        throttled
            .gradient_scale_step
            .store(2, std::sync::atomic::Ordering::Relaxed);
        let off_cadence = tensor_scalar(throttled.next_token_loss_from_logits(
            logits.clone(),
            targets.clone(),
            clean_inputs.clone(),
            None,
            None,
        ));
        throttled
            .gradient_scale_step
            .store(4, std::sync::atomic::Ordering::Relaxed);
        let on_cadence = tensor_scalar(throttled.next_token_loss_from_logits(
            logits,
            targets,
            clean_inputs,
            None,
            None,
        ));
        assert!(
            (off_cadence - ce).abs() < 1.0e-5,
            "off-cadence entropy loss should match CE: ce={ce} off={off_cadence}"
        );
        assert!(
            on_cadence > ce,
            "on-cadence entropy loss should add penalty: ce={ce} on={on_cadence}"
        );
    }

    #[test]
    fn logit_entropy_floor_does_not_penalize_logits_above_floor() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let plain = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ));
        let entropy_penalized = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_logit_entropy_floor(LogitEntropyFloorConfig {
            enabled: true,
            weight: 0.5,
            target_entropy_bits: 1.0,
            ..Default::default()
        });
        let logits = Tensor::<TestBackend, 3>::zeros([1, 2, 4], &device);
        let clean_inputs =
            Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![0, 1], [1, 2]), &device);
        let targets =
            Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![0, 1], [1, 2]), &device);
        let ce = plain.next_token_loss_from_logits(
            logits.clone(),
            targets.clone(),
            clean_inputs.clone(),
            None,
            None,
        );
        let penalized = entropy_penalized.next_token_loss_from_logits(
            logits,
            targets,
            clean_inputs,
            None,
            None,
        );
        let ce_value = ce.to_data().convert::<f32>().into_vec::<f32>().expect("ce")[0];
        let penalized_value = penalized
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("penalized")[0];
        assert!(
            (penalized_value - ce_value).abs() < 1.0e-5,
            "entropy floor should not penalize logits already above the floor: ce={ce_value} penalized={penalized_value}"
        );
    }

    #[test]
    fn marginal_entropy_floor_penalizes_collapsed_batch_distribution() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let collapsed = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                vec![
                    8.0, 0.0, 0.0, 0.0, //
                    8.0, 0.0, 0.0, 0.0, //
                    8.0, 0.0, 0.0, 0.0, //
                    8.0, 0.0, 0.0, 0.0,
                ],
                [1, 4, 4],
            ),
            &device,
        );
        let diverse = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                vec![
                    8.0, 0.0, 0.0, 0.0, //
                    0.0, 8.0, 0.0, 0.0, //
                    0.0, 0.0, 8.0, 0.0, //
                    0.0, 0.0, 0.0, 8.0,
                ],
                [1, 4, 4],
            ),
            &device,
        );
        let collapsed_loss =
            tensor_scalar(marginal_entropy_floor_loss_from_logits(collapsed, 2.0).expect("loss"));
        let diverse_loss =
            tensor_scalar(marginal_entropy_floor_loss_from_logits(diverse, 2.0).expect("loss"));
        assert!(
            collapsed_loss > diverse_loss + 1.0,
            "marginal entropy should penalize collapsed predicted support: collapsed={collapsed_loss} diverse={diverse_loss}"
        );
    }

    #[test]
    fn target_marginal_coverage_penalizes_missing_batch_targets() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let targets = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![0, 1, 2, 3], [1, 4]),
            &device,
        );
        let collapsed = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                vec![
                    6.0, 0.0, 0.0, 0.0, //
                    6.0, 0.0, 0.0, 0.0, //
                    6.0, 0.0, 0.0, 0.0, //
                    6.0, 0.0, 0.0, 0.0,
                ],
                [1, 4, 4],
            ),
            &device,
        );
        let covered = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                vec![
                    6.0, 0.0, 0.0, 0.0, //
                    0.0, 6.0, 0.0, 0.0, //
                    0.0, 0.0, 6.0, 0.0, //
                    0.0, 0.0, 0.0, 6.0,
                ],
                [1, 4, 4],
            ),
            &device,
        );
        let collapsed_loss = tensor_scalar(
            target_marginal_coverage_loss_from_logits(collapsed, targets.clone(), 1.0e-8)
                .expect("collapsed loss"),
        );
        let covered_loss = tensor_scalar(
            target_marginal_coverage_loss_from_logits(covered, targets, 1.0e-8)
                .expect("covered loss"),
        );
        assert!(
            collapsed_loss > covered_loss + 2.0,
            "target marginal coverage should penalize missing target support: collapsed={collapsed_loss} covered={covered_loss}"
        );
    }

    #[test]
    fn logit_entropy_floor_target_coverage_increases_training_loss() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let plain = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ));
        let coverage_penalized = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_logit_entropy_floor(LogitEntropyFloorConfig {
            enabled: true,
            target_coverage_weight: 0.5,
            ..Default::default()
        });
        let logits = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                vec![
                    6.0, 0.0, 0.0, 0.0, //
                    6.0, 0.0, 0.0, 0.0, //
                    6.0, 0.0, 0.0, 0.0, //
                    6.0, 0.0, 0.0, 0.0,
                ],
                [1, 4, 4],
            ),
            &device,
        );
        let clean_inputs = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![0, 1, 2, 3], [1, 4]),
            &device,
        );
        let targets = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![0, 1, 2, 3], [1, 4]),
            &device,
        );
        let ce = plain.next_token_loss_from_logits(
            logits.clone(),
            targets.clone(),
            clean_inputs.clone(),
            None,
            None,
        );
        let penalized = coverage_penalized.next_token_loss_from_logits(
            logits,
            targets,
            clean_inputs,
            None,
            None,
        );
        let ce_value = tensor_scalar(ce);
        let penalized_value = tensor_scalar(penalized);
        assert!(
            penalized_value > ce_value,
            "target coverage should increase loss for collapsed marginal support: ce={ce_value} penalized={penalized_value}"
        );
    }

    #[test]
    fn repeat_unlikelihood_penalizes_configured_history_lags() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let immediate_only = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_repeat_unlikelihood(RepeatUnlikelihoodConfig {
            enabled: true,
            weight: 0.5,
            ..Default::default()
        });
        let lagged = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_repeat_unlikelihood(RepeatUnlikelihoodConfig {
            enabled: true,
            weight: 0.5,
            history_lags: vec![2],
            ..Default::default()
        });
        let logits = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                vec![
                    0.0, 0.0, 0.0, 0.0, //
                    5.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0,
                ],
                [1, 3, 4],
            ),
            &device,
        );
        let clean_inputs = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![0, 1, 2], [1, 3]),
            &device,
        );
        let targets = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1, 2, 3], [1, 3]),
            &device,
        );
        let immediate = immediate_only.next_token_loss_from_logits(
            logits.clone(),
            targets.clone(),
            clean_inputs.clone(),
            None,
            None,
        );
        let lagged = lagged.next_token_loss_from_logits(logits, targets, clean_inputs, None, None);
        let immediate_value = immediate
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("immediate")[0];
        let lagged_value = lagged
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("lagged")[0];
        assert!(
            lagged_value > immediate_value,
            "configured history lag should add unlikelihood loss: immediate={immediate_value} lagged={lagged_value}"
        );
    }

    #[test]
    fn repeat_cycle_lags_respect_budget_and_rotate() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_repeat_unlikelihood(RepeatUnlikelihoodConfig {
            enabled: true,
            cycle_weight: 0.5,
            cycle_min_lag: 2,
            cycle_max_lag: 16,
            cycle_lags_per_step: 4,
            ..Default::default()
        });
        let first = model.repeat_cycle_lags(16);
        assert_eq!(first.len(), 4);
        assert!(first.iter().all(|lag| (2..=16).contains(lag)));
        model
            .gradient_scale_step
            .store(1, std::sync::atomic::Ordering::Relaxed);
        let second = model.repeat_cycle_lags(16);
        assert_eq!(second.len(), 4);
        assert_ne!(first, second);
    }

    #[test]
    fn repeat_unlikelihood_respects_every_steps() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let plain = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ));
        let throttled = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_repeat_unlikelihood(RepeatUnlikelihoodConfig {
            enabled: true,
            weight: 0.5,
            every_steps: 4,
            ..Default::default()
        });
        let logits = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![5.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0], [1, 2, 4]),
            &device,
        );
        let clean_inputs =
            Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![0, 1], [1, 2]), &device);
        let targets =
            Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![1, 2], [1, 2]), &device);
        let ce = tensor_scalar(plain.next_token_loss_from_logits(
            logits.clone(),
            targets.clone(),
            clean_inputs.clone(),
            None,
            None,
        ));
        throttled
            .gradient_scale_step
            .store(2, std::sync::atomic::Ordering::Relaxed);
        let off_cadence = tensor_scalar(throttled.next_token_loss_from_logits(
            logits.clone(),
            targets.clone(),
            clean_inputs.clone(),
            None,
            None,
        ));
        throttled
            .gradient_scale_step
            .store(4, std::sync::atomic::Ordering::Relaxed);
        let on_cadence = tensor_scalar(throttled.next_token_loss_from_logits(
            logits,
            targets,
            clean_inputs,
            None,
            None,
        ));
        assert!(
            (off_cadence - ce).abs() < 1.0e-5,
            "off-cadence repeat loss should match CE: ce={ce} off={off_cadence}"
        );
        assert!(
            on_cadence > ce,
            "on-cadence repeat loss should add penalty: ce={ce} on={on_cadence}"
        );
    }

    #[test]
    fn repeat_cycle_unlikelihood_penalizes_wrong_cycle_predictions() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let plain = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ));
        let cycle_penalized = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_repeat_unlikelihood(RepeatUnlikelihoodConfig {
            enabled: true,
            cycle_weight: 0.5,
            cycle_margin_weight: 0.5,
            cycle_margin: 0.05,
            cycle_min_lag: 2,
            cycle_max_lag: 2,
            cycle_lags_per_step: 1,
            ..Default::default()
        });
        let logits = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                vec![
                    0.0, 0.0, 0.0, 0.0, //
                    5.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0,
                ],
                [1, 3, 4],
            ),
            &device,
        );
        let clean_inputs = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![0, 1, 2], [1, 3]),
            &device,
        );
        let targets = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1, 2, 3], [1, 3]),
            &device,
        );
        let ce = plain.next_token_loss_from_logits(
            logits.clone(),
            targets.clone(),
            clean_inputs.clone(),
            None,
            None,
        );
        let penalized =
            cycle_penalized.next_token_loss_from_logits(logits, targets, clean_inputs, None, None);
        let ce_value = ce.to_data().convert::<f32>().into_vec::<f32>().expect("ce")[0];
        let penalized_value = penalized
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("penalized")[0];
        assert!(
            penalized_value > ce_value,
            "cycle unlikelihood should increase loss for wrong-cycle logits: ce={ce_value} penalized={penalized_value}"
        );
    }

    #[test]
    fn greedy_rollout_recovery_only_skips_stable_hot_path() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_greedy_rollout_unlikelihood(GreedyRolloutUnlikelihoodConfig {
            enabled: true,
            recovery_only: true,
            weight: 0.5,
            prompt_tokens: 1,
            rollout_tokens: 1,
            history_tokens: 1,
            batch_prompts: 1,
            every_steps: 1,
            ..Default::default()
        });
        let clean_inputs = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![0, 1, 2, 3], [1, 4]),
            &device,
        );

        assert!(
            model
                .greedy_rollout_unlikelihood_loss(clean_inputs.clone())
                .is_none(),
            "recovery-only rollout must not run during stable training"
        );
        model.set_recovery_auxiliary_active(true);
        assert!(
            model
                .greedy_rollout_unlikelihood_loss(clean_inputs)
                .is_some(),
            "recovery-only rollout should run when dynamics enters recovery"
        );
    }

    #[test]
    fn greedy_rollout_sequence_recovery_runs_without_step_penalties() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_greedy_rollout_unlikelihood(GreedyRolloutUnlikelihoodConfig {
            enabled: true,
            sequence_recovery_weight: 0.5,
            prompt_tokens: 2,
            rollout_tokens: 2,
            history_tokens: 2,
            batch_prompts: 1,
            every_steps: 1,
            ..Default::default()
        });
        let clean_inputs = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![0, 1, 2, 3, 4], [1, 5]),
            &device,
        );

        let loss = model
            .greedy_rollout_unlikelihood_loss(clean_inputs)
            .expect("sequence recovery should produce a rollout loss");
        let loss = scalar_tensor_to_f64(loss.inner());
        assert!(
            loss.is_finite(),
            "unexpected sequence recovery loss: {loss}"
        );
    }

    #[test]
    fn sdft_train_step_runs_rollout_objective() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_training_objective(TrainingObjectiveConfig::Sdft(SdftObjectiveConfig {
            max_completion_tokens: 2,
            top_k: Some(1),
            ..Default::default()
        }));
        let loss = scalar_loss(TrainStep::step(&model, batch(&device)));
        assert!(loss.is_finite(), "unexpected SDFT loss: {loss}");
    }

    #[test]
    fn latent_reasoning_train_step_runs_next_token_objective() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let mut config = tiny_model_config();
        config.latent_reasoning.enabled = true;
        config.latent_reasoning.max_steps = 2;
        config.latent_reasoning.min_steps = 1;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_latent_reasoning(LatentReasoningTrainingConfig {
                enabled: true,
                jepa_future_offsets: vec![1],
                ..Default::default()
            });
        let loss = scalar_loss(TrainStep::step(&model, batch(&device)));
        assert!(loss.is_finite(), "unexpected latent reasoning loss: {loss}");
    }

    #[test]
    fn latent_energy_margin_loss_prefers_lower_positive_energy() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let low_positive = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![0.0, 0.1], [1, 2, 1]),
            &device,
        );
        let high_negative = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![3.0, 2.5], [1, 2, 1]),
            &device,
        );
        let high_positive = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![3.0, 2.5], [1, 2, 1]),
            &device,
        );
        let low_negative = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![0.0, 0.1], [1, 2, 1]),
            &device,
        );

        let preferred = tensor_scalar(latent_energy_contrastive_margin_loss(
            low_positive,
            high_negative,
            1.0,
        ));
        let inverted = tensor_scalar(latent_energy_contrastive_margin_loss(
            high_positive,
            low_negative,
            1.0,
        ));
        assert!(
            inverted > preferred + 2.0,
            "contrastive energy should prefer low positives: preferred={preferred} inverted={inverted}"
        );
    }

    #[test]
    fn latent_energy_monotonic_penalty_catches_ascending_energy() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let previous = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![1.0, 1.0], [1, 2, 1]),
            &device,
        );
        let descending = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![0.75, 0.5], [1, 2, 1]),
            &device,
        );
        let ascending = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![1.25, 1.5], [1, 2, 1]),
            &device,
        );

        let descending = tensor_scalar(latent_energy_monotonic_penalty(
            previous.clone(),
            descending,
            0.0,
        ));
        let ascending = tensor_scalar(latent_energy_monotonic_penalty(previous, ascending, 0.0));
        assert!(
            descending <= 1.0e-6,
            "descending energy should have no monotonic penalty: {descending}"
        );
        assert!(
            ascending > 0.25,
            "ascending energy should be penalized: {ascending}"
        );
    }

    #[test]
    fn latent_energy_contractivity_penalty_catches_large_hidden_drift() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let target = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![1.0, -1.0], [1, 1, 2]),
            &device,
        );
        let close = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![1.05, -0.95], [1, 1, 2]),
            &device,
        );
        let far = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![3.0, -3.0], [1, 1, 2]),
            &device,
        );

        let close = tensor_scalar(latent_energy_contractivity_penalty(
            close,
            target.clone(),
            0.5,
        ));
        let far = tensor_scalar(latent_energy_contractivity_penalty(far, target, 0.5));
        assert!(
            close <= 1.0e-6,
            "nearby hidden states should fit within the trust radius: {close}"
        );
        assert!(
            far > close + 1.0,
            "large hidden drift should be penalized: close={close} far={far}"
        );
    }

    #[test]
    fn latent_energy_model_train_step_runs() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let mut config = tiny_model_config();
        config.latent_reasoning.enabled = true;
        config.latent_reasoning.max_steps = 2;
        config.latent_reasoning.min_steps = 2;
        config.latent_reasoning.energy_head = true;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_latent_reasoning(LatentReasoningTrainingConfig {
                enabled: true,
                jepa_future_offsets: vec![usize::MAX],
                energy_model: crate::config::LatentEnergyModelConfig {
                    enabled: true,
                    max_rollout_steps_for_loss: 2,
                    ..Default::default()
                },
                sigreg: LatentReasoningSigRegConfig {
                    enabled: false,
                    ..Default::default()
                },
                constraint_balancer: LatentReasoningConstraintBalancerConfig {
                    normalized_aux_scale: 0.01,
                    ..Default::default()
                },
                ..Default::default()
            });
        let loss = scalar_loss(TrainStep::step(&model, batch(&device)));
        assert!(loss.is_finite(), "unexpected latent EBM loss: {loss}");
    }

    #[test]
    fn latent_step_contract_train_step_runs_and_records_components() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        crate::train::profile::reset();
        let mut config = tiny_model_config();
        config.latent_reasoning.enabled = true;
        config.latent_reasoning.max_steps = 2;
        config.latent_reasoning.min_steps = 2;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_latent_reasoning(LatentReasoningTrainingConfig {
                enabled: true,
                jepa_future_offsets: vec![usize::MAX],
                step_contract: LatentStepContractConfig {
                    enabled: true,
                    max_rollout_steps_for_loss: 2,
                    ce_weight: 0.1,
                    monotonic_ce_weight: 0.5,
                    contractive_weight: 0.05,
                    ..Default::default()
                },
                sigreg: LatentReasoningSigRegConfig {
                    enabled: false,
                    ..Default::default()
                },
                constraint_balancer: LatentReasoningConstraintBalancerConfig {
                    normalized_aux_scale: 0.01,
                    ..Default::default()
                },
                ..Default::default()
            });
        let loss = scalar_loss(TrainStep::step(&model, batch(&device)));
        assert!(
            loss.is_finite(),
            "unexpected latent step contract loss: {loss}"
        );
        let snapshot = crate::train::profile::take_latent_reasoning();
        assert!(
            snapshot.step_contract_components > 0,
            "step contract should record active components: {snapshot:?}"
        );
    }

    #[test]
    fn latent_reasoning_step_diagnostics_are_finite() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let mut config = tiny_model_config();
        config.latent_reasoning.enabled = true;
        config.latent_reasoning.max_steps = 3;
        config.latent_reasoning.min_steps = 3;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device));

        let diagnostics = model
            .latent_reasoning_step_diagnostics(batch(&device))
            .expect("latent diagnostics");
        assert_eq!(diagnostics.step_loss.len(), 3);
        assert_eq!(diagnostics.step_ce_delta.len(), 3);
        assert_eq!(diagnostics.step_ce_monotonic_violation_rate.len(), 3);
        assert_eq!(diagnostics.step_entropy_bits.len(), 3);
        assert_eq!(diagnostics.step_delta_rms.len(), 3);
        assert_eq!(diagnostics.step_raw_cosine.len(), 3);
        for value in [
            diagnostics.raw_loss,
            diagnostics.final_loss,
            diagnostics.raw_entropy_bits,
            diagnostics.final_entropy_bits,
            diagnostics.final_delta_rms,
            diagnostics.final_raw_cosine,
        ]
        .into_iter()
        .chain(diagnostics.step_loss)
        .chain(diagnostics.step_ce_delta)
        .chain(diagnostics.step_ce_monotonic_violation_rate)
        .chain(diagnostics.step_entropy_bits)
        .chain(diagnostics.step_delta_rms)
        .chain(diagnostics.step_raw_cosine)
        {
            assert!(
                value.is_finite(),
                "diagnostic value was not finite: {value}"
            );
        }
    }

    #[test]
    fn latent_reasoning_step_diagnostics_include_energy_when_head_enabled() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let mut config = tiny_model_config();
        config.latent_reasoning.enabled = true;
        config.latent_reasoning.max_steps = 3;
        config.latent_reasoning.min_steps = 3;
        config.latent_reasoning.energy_head = true;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device));

        let diagnostics = model
            .latent_reasoning_step_diagnostics(batch(&device))
            .expect("latent diagnostics");
        assert_eq!(diagnostics.step_energy_mean.len(), 3);
        assert_eq!(diagnostics.step_energy_delta.len(), 3);
        assert_eq!(diagnostics.step_energy_monotonic_violation_rate.len(), 3);
        assert!(diagnostics.best_energy_step.is_some());
        for value in diagnostics
            .step_energy_mean
            .into_iter()
            .chain(diagnostics.step_energy_delta)
            .chain(diagnostics.step_energy_monotonic_violation_rate)
        {
            assert!(
                value.is_finite(),
                "energy diagnostic value was not finite: {value}"
            );
        }
    }

    #[test]
    fn latent_reasoning_auxiliary_scale_respects_start_after_steps() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let mut config = tiny_model_config();
        config.latent_reasoning.enabled = true;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_latent_reasoning(LatentReasoningTrainingConfig {
                enabled: true,
                every_steps: 1,
                jepa_future_offsets: vec![1],
                constraint_balancer: LatentReasoningConstraintBalancerConfig {
                    normalized_aux_scale: 0.25,
                    start_after_steps: 2,
                    ..Default::default()
                },
                ..Default::default()
            });

        model.gradient_scale_step.store(0, Ordering::Relaxed);
        assert_eq!(model.latent_reasoning_auxiliary_scale(), None);
        model.gradient_scale_step.store(1, Ordering::Relaxed);
        assert_eq!(model.latent_reasoning_auxiliary_scale(), None);
        model.gradient_scale_step.store(2, Ordering::Relaxed);
        assert_eq!(model.latent_reasoning_auxiliary_scale(), Some(0.25));
    }

    #[test]
    fn latent_reasoning_auxiliary_scale_can_wait_for_capability_gate() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let mut config = tiny_model_config();
        config.latent_reasoning.enabled = true;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_latent_reasoning(LatentReasoningTrainingConfig {
                enabled: true,
                every_steps: 1,
                start_after_capability_gate_passed: true,
                jepa_future_offsets: vec![1],
                constraint_balancer: LatentReasoningConstraintBalancerConfig {
                    normalized_aux_scale: 0.25,
                    ..Default::default()
                },
                ..Default::default()
            });

        model.gradient_scale_step.store(32, Ordering::Relaxed);
        assert_eq!(model.latent_reasoning_auxiliary_scale(), None);
        model.set_latent_reasoning_capability_gate_open(true);
        assert_eq!(model.latent_reasoning_auxiliary_scale(), Some(0.25));
    }

    #[test]
    fn latent_reasoning_auxiliary_scale_respects_per_objective_every_steps() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let mut config = tiny_model_config();
        config.latent_reasoning.enabled = true;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_latent_reasoning(LatentReasoningTrainingConfig {
                enabled: true,
                every_steps: 8,
                jepa_every_steps: Some(8),
                jepa_future_offsets: vec![1],
                next_latent: NextLatentPredictionConfig {
                    enabled: true,
                    every_steps: Some(16),
                    start_after_steps: Some(8),
                    ..Default::default()
                },
                constraint_balancer: LatentReasoningConstraintBalancerConfig {
                    normalized_aux_scale: 0.25,
                    ..Default::default()
                },
                ..Default::default()
            });

        model.gradient_scale_step.store(7, Ordering::Relaxed);
        assert_eq!(
            model.latent_reasoning_auxiliary_scale_for_every_steps(
                model.latent_reasoning_jepa_every_steps()
            ),
            Some(0.25)
        );
        assert_eq!(
            model.latent_reasoning_auxiliary_scale_for_schedule(
                model.latent_reasoning_next_latent_every_steps(),
                model.latent_reasoning_next_latent_start_after_steps(),
                model.latent_reasoning_next_latent_start_policy()
            ),
            None
        );
        model.gradient_scale_step.store(15, Ordering::Relaxed);
        assert_eq!(
            model.latent_reasoning_auxiliary_scale_for_schedule(
                model.latent_reasoning_next_latent_every_steps(),
                model.latent_reasoning_next_latent_start_after_steps(),
                model.latent_reasoning_next_latent_start_policy()
            ),
            Some(0.25)
        );
    }

    #[test]
    fn latent_reasoning_start_policy_can_gate_specific_objectives() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let mut config = tiny_model_config();
        config.latent_reasoning.enabled = true;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_latent_reasoning(LatentReasoningTrainingConfig {
                enabled: true,
                every_steps: 1,
                jepa_start_policy: Some(LatentReasoningAuxiliaryStartPolicy::FixedStep),
                jepa_future_offsets: vec![1],
                next_latent: NextLatentPredictionConfig {
                    enabled: true,
                    start_policy: Some(
                        LatentReasoningAuxiliaryStartPolicy::FixedStepAndCapabilityGate,
                    ),
                    ..Default::default()
                },
                constraint_balancer: LatentReasoningConstraintBalancerConfig {
                    normalized_aux_scale: 0.25,
                    start_after_steps: 4,
                    ..Default::default()
                },
                ..Default::default()
            });

        model.gradient_scale_step.store(4, Ordering::Relaxed);
        assert_eq!(
            model.latent_reasoning_auxiliary_scale_for_schedule(
                model.latent_reasoning_jepa_every_steps(),
                model.latent_reasoning_jepa_start_after_steps(),
                model.latent_reasoning_jepa_start_policy()
            ),
            Some(0.25)
        );
        assert_eq!(
            model.latent_reasoning_auxiliary_scale_for_schedule(
                model.latent_reasoning_next_latent_every_steps(),
                model.latent_reasoning_next_latent_start_after_steps(),
                model.latent_reasoning_next_latent_start_policy()
            ),
            None
        );
        model.set_latent_reasoning_capability_gate_open(true);
        assert_eq!(
            model.latent_reasoning_auxiliary_scale_for_schedule(
                model.latent_reasoning_next_latent_every_steps(),
                model.latent_reasoning_next_latent_start_after_steps(),
                model.latent_reasoning_next_latent_start_policy()
            ),
            Some(0.25)
        );
    }

    #[test]
    fn latent_reasoning_capability_gate_policy_can_ignore_fixed_step_start() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let mut config = tiny_model_config();
        config.latent_reasoning.enabled = true;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_latent_reasoning(LatentReasoningTrainingConfig {
                enabled: true,
                every_steps: 1,
                next_latent: NextLatentPredictionConfig {
                    enabled: true,
                    start_after_steps: Some(512),
                    start_policy: Some(LatentReasoningAuxiliaryStartPolicy::CapabilityGate),
                    ..Default::default()
                },
                constraint_balancer: LatentReasoningConstraintBalancerConfig {
                    normalized_aux_scale: 0.25,
                    ..Default::default()
                },
                ..Default::default()
            });

        model.gradient_scale_step.store(0, Ordering::Relaxed);
        assert_eq!(
            model.latent_reasoning_auxiliary_scale_for_schedule(
                model.latent_reasoning_next_latent_every_steps(),
                model.latent_reasoning_next_latent_start_after_steps(),
                model.latent_reasoning_next_latent_start_policy()
            ),
            None
        );
        model.set_latent_reasoning_capability_gate_open(true);
        assert_eq!(
            model.latent_reasoning_auxiliary_scale_for_schedule(
                model.latent_reasoning_next_latent_every_steps(),
                model.latent_reasoning_next_latent_start_after_steps(),
                model.latent_reasoning_next_latent_start_policy()
            ),
            Some(0.25)
        );
    }

    #[test]
    fn latent_reasoning_global_capability_gate_remains_compatibility_default() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let mut config = tiny_model_config();
        config.latent_reasoning.enabled = true;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_latent_reasoning(LatentReasoningTrainingConfig {
                enabled: true,
                every_steps: 1,
                start_after_capability_gate_passed: true,
                jepa_future_offsets: vec![1],
                constraint_balancer: LatentReasoningConstraintBalancerConfig {
                    normalized_aux_scale: 0.25,
                    start_after_steps: 2,
                    ..Default::default()
                },
                ..Default::default()
            });

        model.gradient_scale_step.store(2, Ordering::Relaxed);
        assert_eq!(
            model.latent_reasoning_jepa_start_policy(),
            LatentReasoningAuxiliaryStartPolicy::FixedStepAndCapabilityGate
        );
        assert_eq!(model.latent_reasoning_auxiliary_scale(), None);
        model.set_latent_reasoning_capability_gate_open(true);
        assert_eq!(model.latent_reasoning_auxiliary_scale(), Some(0.25));
    }

    #[test]
    fn next_latent_train_step_runs_without_inference_latent_reasoning() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let mut config = tiny_model_config();
        config.next_latent_transition.enabled = true;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_latent_reasoning(LatentReasoningTrainingConfig {
                enabled: true,
                jepa_future_offsets: vec![usize::MAX],
                next_latent: NextLatentPredictionConfig {
                    enabled: true,
                    horizon: 2,
                    regression_weight: 1.0,
                    token_kl_weight: 0.01,
                    smooth_l1_beta: 1.0,
                    detach_action_embedding: true,
                    ..Default::default()
                },
                sigreg: LatentReasoningSigRegConfig {
                    enabled: false,
                    ..Default::default()
                },
                constraint_balancer: LatentReasoningConstraintBalancerConfig {
                    normalized_aux_scale: 0.01,
                    ..Default::default()
                },
                ..Default::default()
            });
        assert!(!model.model.latent_reasoning_enabled());
        assert!(model.model.next_latent_transition_enabled());
        let loss = scalar_loss(TrainStep::step(&model, batch(&device)));
        assert!(loss.is_finite(), "unexpected NextLat loss: {loss}");
    }

    #[test]
    fn dragon_state_consistency_train_step_runs_without_latent_reasoning_architecture() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let mut config = tiny_model_config();
        config.latent_reasoning.enabled = false;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_latent_reasoning(LatentReasoningTrainingConfig {
                enabled: true,
                jepa_future_offsets: vec![usize::MAX],
                dragon_state: DragonStateConsistencyConfig {
                    enabled: true,
                    rho_weight: 1.0,
                    rho_energy_weight: 0.25,
                    smooth_l1_beta: 1.0,
                    max_rho_slots: 4,
                    ..Default::default()
                },
                constraint_balancer: LatentReasoningConstraintBalancerConfig {
                    normalized_aux_scale: 0.01,
                    ..Default::default()
                },
                ..Default::default()
            });
        assert!(!model.model.latent_reasoning_enabled());
        let loss = scalar_loss(TrainStep::step(&model, batch(&device)));
        assert!(
            loss.is_finite(),
            "unexpected Dragon state consistency loss: {loss}"
        );
    }

    #[test]
    fn latent_reasoning_rho_memory_sigreg_train_step_runs() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let mut config = tiny_model_config();
        config.latent_reasoning.enabled = true;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_latent_reasoning(LatentReasoningTrainingConfig {
                enabled: true,
                jepa_future_offsets: vec![usize::MAX],
                sigreg: LatentReasoningSigRegConfig {
                    enabled: true,
                    target: crate::config::LatentReasoningSigRegTarget::RhoMemorySlots,
                    ..Default::default()
                },
                constraint_balancer: LatentReasoningConstraintBalancerConfig {
                    normalized_aux_scale: 0.01,
                    ..Default::default()
                },
                ..Default::default()
            });
        let loss = scalar_loss(TrainStep::step(&model, batch(&device)));
        assert!(
            loss.is_finite(),
            "unexpected rho-memory latent reasoning loss: {loss}"
        );
    }

    #[test]
    fn rho_memory_sigreg_train_step_runs_without_latent_reasoning_architecture() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let mut config = tiny_model_config();
        config.latent_reasoning.enabled = false;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_latent_reasoning(LatentReasoningTrainingConfig {
                enabled: true,
                jepa_future_offsets: vec![usize::MAX],
                sigreg: LatentReasoningSigRegConfig {
                    enabled: true,
                    target: LatentReasoningSigRegTarget::RhoMemorySlots,
                    ..Default::default()
                },
                constraint_balancer: LatentReasoningConstraintBalancerConfig {
                    normalized_aux_scale: 0.01,
                    ..Default::default()
                },
                ..Default::default()
            });
        let loss = scalar_loss(TrainStep::step(&model, batch(&device)));
        assert!(
            loss.is_finite(),
            "unexpected rho-memory regularized base Dragon loss: {loss}"
        );
    }

    #[test]
    fn rho_memory_sigreg_penalizes_redundant_slots() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_latent_reasoning(LatentReasoningTrainingConfig {
            enabled: true,
            sigreg: LatentReasoningSigRegConfig {
                enabled: true,
                target: LatentReasoningSigRegTarget::RhoMemorySlots,
                ..Default::default()
            },
            ..Default::default()
        });
        let mut duplicate_state = model.model.init_state_ephemeral();
        duplicate_state.layers[0].rho = Some(Tensor::<TestBackend, 4>::from_data(
            TensorData::new(vec![1.0, -1.0, 0.0, 1.0, -1.0, 0.0], [1, 1, 2, 3]),
            &device,
        ));
        let mut orthogonal_state = model.model.init_state_ephemeral();
        orthogonal_state.layers[0].rho = Some(Tensor::<TestBackend, 4>::from_data(
            TensorData::new(vec![1.0, -1.0, 0.0, 1.0, 1.0, -2.0], [1, 1, 2, 3]),
            &device,
        ));

        let duplicate = tensor_scalar(
            model
                .sigreg_loss_from_rho_memory_state(&duplicate_state)
                .expect("duplicate rho loss"),
        );
        let orthogonal = tensor_scalar(
            model
                .sigreg_loss_from_rho_memory_state(&orthogonal_state)
                .expect("orthogonal rho loss"),
        );

        assert!(
            duplicate > orthogonal + 0.5,
            "duplicate slots should be penalized more strongly: duplicate={duplicate} orthogonal={orthogonal}"
        );
        assert!(
            orthogonal < 1.0e-5,
            "centered orthogonal slots should have near-zero redundancy penalty: {orthogonal}"
        );
    }

    #[test]
    fn rho_memory_sigreg_samples_slots_deterministically() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_latent_reasoning(LatentReasoningTrainingConfig {
            enabled: true,
            sigreg: LatentReasoningSigRegConfig {
                enabled: true,
                target: LatentReasoningSigRegTarget::RhoMemorySlots,
                max_rho_slots: 3,
                ..Default::default()
            },
            ..Default::default()
        });
        let rho = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(vec![0.0, 1.0, 2.0, 3.0, 4.0], [1, 1, 5, 1]),
            &device,
        );

        let sampled = model.sigreg_sample_rho_slots(rho, 5);
        assert_eq!(sampled.shape().dims::<4>(), [1, 1, 3, 1]);
        let values = sampled
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("sampled rho");
        assert_eq!(values, vec![0.0, 2.0, 4.0]);
    }

    #[test]
    fn dragon_state_consistency_is_zero_for_matching_rho_and_positive_for_drift() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_latent_reasoning(LatentReasoningTrainingConfig {
            enabled: true,
            dragon_state: DragonStateConsistencyConfig {
                enabled: true,
                rho_weight: 1.0,
                rho_energy_weight: 1.0,
                smooth_l1_beta: 1.0,
                max_rho_slots: 2,
                ..Default::default()
            },
            ..Default::default()
        });
        let mut student_state = model.model.init_state_ephemeral();
        student_state.layers[0].rho = Some(Tensor::<TestBackend, 4>::from_data(
            TensorData::new(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [1, 1, 2, 3]),
            &device,
        ));
        let teacher_state = student_state.clone();

        let (matching_loss, matching_components) =
            model.dragon_state_consistency_loss(&student_state, &teacher_state);
        assert_eq!(matching_components, 2);
        let matching_loss = tensor_scalar(matching_loss.expect("matching rho loss"));
        assert!(
            matching_loss.abs() < 1.0e-6,
            "matching rho state should have zero consistency loss: {matching_loss}"
        );

        let mut drifted_teacher_state = model.model.init_state_ephemeral();
        drifted_teacher_state.layers[0].rho = Some(Tensor::<TestBackend, 4>::from_data(
            TensorData::new(vec![1.0, 0.0, 0.0, 0.0, -1.0, 0.0], [1, 1, 2, 3]),
            &device,
        ));
        let (drift_loss, drift_components) =
            model.dragon_state_consistency_loss(&student_state, &drifted_teacher_state);
        assert_eq!(drift_components, 2);
        let drift_loss = tensor_scalar(drift_loss.expect("drift rho loss"));
        assert!(
            drift_loss > matching_loss + 0.1,
            "drifted rho rows should be penalized: matching={matching_loss} drift={drift_loss}"
        );
    }

    #[test]
    fn sigreg_combined_target_enables_hidden_and_rho_losses() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_latent_reasoning(LatentReasoningTrainingConfig {
            enabled: true,
            sigreg: LatentReasoningSigRegConfig {
                enabled: true,
                target: LatentReasoningSigRegTarget::HiddenAndRhoMemorySlots,
                ..Default::default()
            },
            ..Default::default()
        });
        let hidden = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![0.0, 0.1, 0.2, 0.3], [1, 2, 2]),
            &device,
        );
        let mut state = model.model.init_state_ephemeral();
        state.layers[0].rho = Some(Tensor::<TestBackend, 4>::from_data(
            TensorData::new(vec![1.0, -1.0, 0.0, 1.0, -1.0, 0.0], [1, 1, 2, 3]),
            &device,
        ));

        assert!(model.sigreg_loss_from_hidden(hidden).is_some());
        assert!(model.sigreg_loss_from_rho_memory_state(&state).is_some());
    }

    #[test]
    fn sdpo_train_step_runs_rollout_objective() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_training_objective(TrainingObjectiveConfig::Sdpo(SdpoObjectiveConfig {
            group_size: 2,
            max_completion_tokens: 2,
            top_k: Some(1),
            ..Default::default()
        }));
        let loss = scalar_loss(TrainStep::step(&model, batch(&device)));
        assert!(loss.is_finite(), "unexpected SDPO loss: {loss}");
    }

    #[test]
    #[should_panic(
        expected = "paper-aligned SDFT/SDPO rollout objectives require flat token logits"
    )]
    fn sdft_train_step_guards_factorized_language_head() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_factorized_model_config(),
            &device,
        ))
        .with_training_objective(TrainingObjectiveConfig::Sdft(SdftObjectiveConfig {
            max_completion_tokens: 2,
            top_k: Some(1),
            ..Default::default()
        }));
        let _ = TrainStep::step(&model, batch(&device));
    }

    #[test]
    fn sdft_train_step_updates_teacher_runtime() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_training_objective(TrainingObjectiveConfig::Sdft(SdftObjectiveConfig {
            max_completion_tokens: 2,
            top_k: Some(1),
            teacher_update_rate: 0.5,
            ..Default::default()
        }));
        let _ = scalar_loss(TrainStep::step(&model, batch(&device)));
        let update_count = model
            .teacher_update_count_for_test()
            .expect("teacher update count");
        assert_eq!(update_count, 1);
    }

    #[test]
    fn rollout_teacher_context_contains_gold_demonstration() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ));
        let inputs = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![0, 1, 2, 3], [1, 4]),
            &device,
        );
        let targets = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1, 2, 9, 10], [1, 4]),
            &device,
        );
        let rollout = model.rollout_score_batch(
            &model.model,
            inputs,
            targets,
            RolloutScoreConfig {
                max_completion_tokens: 2,
                group_size: 1,
                temperature: 1.0,
                top_k: Some(1),
                num_loss_tokens_to_skip: 0,
                max_reprompt_len: usize::MAX,
                reprompt_truncation: RepromptTruncation::Right,
            },
        );
        let teacher_inputs = rollout
            .teacher_inputs
            .to_data()
            .convert::<i64>()
            .into_vec::<i64>()
            .expect("teacher input vec");
        assert_eq!(teacher_inputs[0], 2);
        assert_eq!(teacher_inputs[1], 9);
    }

    #[test]
    fn sdft_sdpo_composite_train_step_runs() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_training_objective(TrainingObjectiveConfig::SdftSdpo(
            SdftSdpoObjectiveConfig {
                sdft: SdftObjectiveConfig {
                    max_completion_tokens: 2,
                    top_k: Some(1),
                    ..Default::default()
                },
                sdpo: SdpoObjectiveConfig {
                    group_size: 2,
                    max_completion_tokens: 2,
                    top_k: Some(1),
                    ..Default::default()
                },
                ..Default::default()
            },
        ));
        let loss = scalar_loss(TrainStep::step(&model, batch(&device)));
        assert!(loss.is_finite(), "unexpected composite loss: {loss}");
    }

    #[test]
    fn sdpo_train_step_runs_with_single_process_pipeline_plan() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let mut config = tiny_model_config();
        config.n_layer = 2;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_pipeline_plan(Some(tiny_pipeline_plan()))
            .with_training_objective(TrainingObjectiveConfig::Sdpo(SdpoObjectiveConfig {
                group_size: 2,
                max_completion_tokens: 2,
                top_k: Some(1),
                ..Default::default()
            }));
        let loss = scalar_loss(TrainStep::step(&model, batch(&device)));
        assert!(loss.is_finite(), "unexpected pipeline SDPO loss: {loss}");
    }
}
