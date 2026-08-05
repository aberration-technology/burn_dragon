use crate::config::train::NeuronScalingStabilizationConfig;
use crate::train::prelude::*;
use burn::tensor::activation;
use burn::tensor::backend::Backend;
use burn_dragon_core::ModelState;
use burn_dragon_time::Instant;
use std::any::Any;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

const RULIAD_FIELD_BINDING_OBJECTIVE: &str =
    "first_divergence_plus_paired_sequence_log_probability_v2";

const STOCHASTIC_STREAM_MAIN: u64 = 0x6d61_696e_5f73_7465;
const STOCHASTIC_STREAM_PROOF_POLICY: u64 = 0x7072_6f6f_665f_706f;
const STOCHASTIC_STREAM_VERIFIER_POLICY: u64 = 0x7665_7269_6669_6572;
const STOCHASTIC_STREAM_PC_AMORTIZATION: u64 = 0x7063_5f61_6d6f_7274;

fn stochastic_step_seed(base_seed: u64, step_index: usize, stream: u64) -> u64 {
    let mut value = base_seed
        ^ (step_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ stream.rotate_left(23);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[derive(Clone, Default)]
struct PipelineRuntimeCell {
    inner: Arc<Mutex<Option<Box<dyn Any + Send>>>>,
}

impl std::fmt::Debug for PipelineRuntimeCell {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PipelineRuntimeCell")
            .finish_non_exhaustive()
    }
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

fn rescale_gradients_by_device_scalar<B, M>(
    module: &M,
    grads: &mut GradientsParams,
    scalar: Tensor<B::InnerBackend, 1>,
    divide: bool,
) where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    struct DeviceScalarGradientVisitor<'a, B: AutodiffBackend> {
        grads: &'a mut GradientsParams,
        scalar: Tensor<B::InnerBackend, 1>,
        divide: bool,
    }
    impl<B: AutodiffBackend> burn::module::ModuleVisitor<B> for DeviceScalarGradientVisitor<'_, B> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            if let Some(grad) = self.grads.remove::<B::InnerBackend, D>(param.id) {
                let scalar = self.scalar.clone().reshape([1; D]);
                let grad = if self.divide {
                    grad / scalar
                } else {
                    grad * scalar
                };
                self.grads.register(param.id, grad);
            }
        }
    }
    module.visit(&mut DeviceScalarGradientVisitor::<B> {
        grads,
        scalar,
        divide,
    });
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
    if !rows.is_multiple_of(new_latent_per_head) {
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

trait PredictiveCodingStateMapper<B: BackendTrait> {
    fn map_rank3(
        &mut self,
        name: &'static str,
        tensor: Option<Tensor<B, 3>>,
    ) -> Option<Tensor<B, 3>>;

    fn map_rank4(
        &mut self,
        name: &'static str,
        tensor: Option<Tensor<B, 4>>,
    ) -> Option<Tensor<B, 4>>;
}

fn map_predictive_coding_state<B: BackendTrait>(
    state: &mut ModelState<B>,
    scope: PredictiveCodingStateScope,
    mapper: &mut impl PredictiveCodingStateMapper<B>,
) {
    for layer in &mut state.layers {
        layer.rho_norm = None;
        layer.rho = mapper.map_rank4("rho", layer.rho.take());
        layer.y_neuron_state = mapper.map_rank3("y_neuron_state", layer.y_neuron_state.take());
        if !matches!(scope, PredictiveCodingStateScope::All) {
            continue;
        }

        layer.slow_rho_norm = None;
        layer.sequence_aux = mapper.map_rank4("sequence_aux", layer.sequence_aux.take());
        layer.mamba_angle_state =
            mapper.map_rank3("mamba_angle_state", layer.mamba_angle_state.take());
        layer.mamba_k_state = mapper.map_rank3("mamba_k_state", layer.mamba_k_state.take());
        layer.mamba_v_state = mapper.map_rank3("mamba_v_state", layer.mamba_v_state.take());
        layer.slow_rho = mapper.map_rank4("slow_rho", layer.slow_rho.take());
        layer.slow_sequence_aux =
            mapper.map_rank4("slow_sequence_aux", layer.slow_sequence_aux.take());
        layer.slow_mamba_angle_state = mapper.map_rank3(
            "slow_mamba_angle_state",
            layer.slow_mamba_angle_state.take(),
        );
        layer.slow_mamba_k_state =
            mapper.map_rank3("slow_mamba_k_state", layer.slow_mamba_k_state.take());
        layer.slow_mamba_v_state =
            mapper.map_rank3("slow_mamba_v_state", layer.slow_mamba_v_state.take());
        layer.hierarchical_slow_hidden = mapper.map_rank4(
            "hierarchical_slow_hidden",
            layer.hierarchical_slow_hidden.take(),
        );
        layer.clocked_slow_hidden =
            mapper.map_rank4("clocked_slow_hidden", layer.clocked_slow_hidden.take());
        layer.summary_memory_hidden =
            mapper.map_rank4("summary_memory_hidden", layer.summary_memory_hidden.take());
    }
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

type PredictiveCodingSampleIndexCache<B> = HashMap<(usize, usize, usize), Tensor<B, 1, Int>>;

fn rotating_sample_state_axis_pair<B: BackendTrait, const D: usize>(
    student: Tensor<B, D>,
    teacher: Tensor<B, D>,
    axis: usize,
    max_slots: usize,
    sample_offset: usize,
    cache: &mut PredictiveCodingSampleIndexCache<B>,
) -> (Tensor<B, D>, Tensor<B, D>) {
    let dims = student.shape().dims::<D>();
    let slots = dims[axis];
    if slots <= max_slots.max(1) {
        return (student, teacher);
    }
    let sample_slots = max_slots.max(1).min(slots);
    let sample_offset = sample_offset % slots;
    let key = (slots, sample_slots, sample_offset);
    let indices = cache
        .entry(key)
        .or_insert_with(|| {
            let indices = (0..sample_slots)
                .map(|index| {
                    (((index * slots + sample_slots / 2) / sample_slots + sample_offset) % slots)
                        as i64
                })
                .collect::<Vec<_>>();
            Tensor::<B, 1, Int>::from_data(
                TensorData::new(indices, [sample_slots]),
                &student.device(),
            )
        })
        .clone();
    (
        student.select(axis, indices.clone()),
        teacher.select(axis, indices),
    )
}

#[derive(Clone, Copy)]
struct PredictiveCodingAmortizationConstraint {
    sample_axis: usize,
    max_slots: usize,
    sample_offset: usize,
    tolerance: f32,
    eps: f32,
}

fn predictive_coding_chunk_due(
    observation_contract: PredictiveCodingObservationContract,
    step_index: usize,
    chunk_index: usize,
    chunks_per_step: usize,
    apply_every_chunks: usize,
) -> bool {
    let ordinal = step_index
        .saturating_mul(chunks_per_step.max(1))
        .saturating_add(chunk_index);
    let cadence = apply_every_chunks.max(1);
    match observation_contract {
        // A causal correction needs a state produced by an earlier observed chunk.
        PredictiveCodingObservationContract::ObservedPrefix => {
            ordinal.saturating_add(1).is_multiple_of(cadence)
        }
        // Preserve historical phase alignment for the explicitly non-causal control.
        PredictiveCodingObservationContract::OracleNextTokenNegativeControl => {
            ordinal.is_multiple_of(cadence)
        }
    }
}

fn accumulate_predictive_coding_amortization_constraint<B: BackendTrait, const D: usize>(
    total: &mut Option<Tensor<B, 1>>,
    components: &mut usize,
    student: &Option<Tensor<B, D>>,
    teacher: &Option<Tensor<B, D>>,
    constraint: PredictiveCodingAmortizationConstraint,
    sample_indices: &mut PredictiveCodingSampleIndexCache<B>,
) {
    let (Some(student), Some(teacher)) = (student.as_ref(), teacher.as_ref()) else {
        return;
    };
    if student.shape().dims::<D>() != teacher.shape().dims::<D>() {
        return;
    }
    let (student, teacher) = rotating_sample_state_axis_pair(
        student.clone(),
        teacher.clone().detach(),
        constraint.sample_axis,
        constraint.max_slots,
        constraint.sample_offset,
        sample_indices,
    );
    let student_scale = student
        .clone()
        .detach()
        .powf_scalar(2.0)
        .mean()
        .reshape([1]);
    let teacher_scale = teacher.clone().powf_scalar(2.0).mean().reshape([1]);
    let scale = (student_scale + teacher_scale)
        .clamp_min(constraint.eps.max(1.0e-12))
        .detach();
    let relative_mse = (student - teacher).powf_scalar(2.0).mean().reshape([1]) / scale;
    let violation = relative_mse
        .add_scalar(-constraint.tolerance.max(0.0).powi(2))
        .clamp_min(0.0);
    *total = Some(match total.take() {
        Some(accumulated) => accumulated + violation,
        None => violation,
    });
    *components = components.saturating_add(1);
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
        stats.record_synced(
            update.grad_norm,
            update.grad_norm_max,
            update.delta_rms,
            update.clip_fraction,
        );
        *slot = Some(Tensor::from_inner(update.tensor).detach());
    } else {
        let updated = burn_pc::pc_sgd_update(base, grad, config);
        stats.record_unsynced();
        *slot = Some(Tensor::from_inner(updated).detach());
    }
}

#[derive(Default)]
struct PredictiveCodingPresenceMapper {
    present: bool,
}

impl<B: BackendTrait> PredictiveCodingStateMapper<B> for PredictiveCodingPresenceMapper {
    fn map_rank3(
        &mut self,
        _name: &'static str,
        tensor: Option<Tensor<B, 3>>,
    ) -> Option<Tensor<B, 3>> {
        self.present |= tensor.is_some();
        tensor
    }

    fn map_rank4(
        &mut self,
        _name: &'static str,
        tensor: Option<Tensor<B, 4>>,
    ) -> Option<Tensor<B, 4>> {
        self.present |= tensor.is_some();
        tensor
    }
}

#[derive(Default)]
struct PredictiveCodingAttachMapper {
    attached: bool,
}

impl<B: BackendTrait> PredictiveCodingStateMapper<B> for PredictiveCodingAttachMapper {
    fn map_rank3(
        &mut self,
        _name: &'static str,
        mut tensor: Option<Tensor<B, 3>>,
    ) -> Option<Tensor<B, 3>> {
        self.attached |= attach_predictive_coding_tensor(&mut tensor);
        tensor
    }

    fn map_rank4(
        &mut self,
        _name: &'static str,
        mut tensor: Option<Tensor<B, 4>>,
    ) -> Option<Tensor<B, 4>> {
        self.attached |= attach_predictive_coding_tensor(&mut tensor);
        tensor
    }
}

struct PredictiveCodingUpdateMapper<'a, B: AutodiffBackend> {
    grads: &'a B::Gradients,
    config: &'a burn_pc::PcInferenceConfig,
    sync_diagnostics: bool,
    stats: PredictiveCodingTensorUpdateStats,
}

impl<B: AutodiffBackend> PredictiveCodingStateMapper<B> for PredictiveCodingUpdateMapper<'_, B> {
    fn map_rank3(
        &mut self,
        _name: &'static str,
        mut tensor: Option<Tensor<B, 3>>,
    ) -> Option<Tensor<B, 3>> {
        update_predictive_coding_tensor(
            &mut tensor,
            self.grads,
            self.config,
            self.sync_diagnostics,
            &mut self.stats,
        );
        tensor
    }

    fn map_rank4(
        &mut self,
        _name: &'static str,
        mut tensor: Option<Tensor<B, 4>>,
    ) -> Option<Tensor<B, 4>> {
        update_predictive_coding_tensor(
            &mut tensor,
            self.grads,
            self.config,
            self.sync_diagnostics,
            &mut self.stats,
        );
        tensor
    }
}

struct PredictiveCodingStateSnapshot<B: BackendTrait> {
    rank3: Vec<(&'static str, Option<Tensor<B, 3>>)>,
    rank4: Vec<(&'static str, Option<Tensor<B, 4>>)>,
}

impl<B: BackendTrait> Default for PredictiveCodingStateSnapshot<B> {
    fn default() -> Self {
        Self {
            rank3: Vec::new(),
            rank4: Vec::new(),
        }
    }
}

impl<B: BackendTrait> PredictiveCodingStateMapper<B> for PredictiveCodingStateSnapshot<B> {
    fn map_rank3(
        &mut self,
        name: &'static str,
        tensor: Option<Tensor<B, 3>>,
    ) -> Option<Tensor<B, 3>> {
        self.rank3.push((name, tensor.clone()));
        tensor
    }

    fn map_rank4(
        &mut self,
        name: &'static str,
        tensor: Option<Tensor<B, 4>>,
    ) -> Option<Tensor<B, 4>> {
        self.rank4.push((name, tensor.clone()));
        tensor
    }
}

fn predictive_coding_state_snapshot<B: BackendTrait>(
    state: &ModelState<B>,
    scope: PredictiveCodingStateScope,
) -> PredictiveCodingStateSnapshot<B> {
    let mut state = state.clone();
    let mut snapshot = PredictiveCodingStateSnapshot::default();
    map_predictive_coding_state(&mut state, scope, &mut snapshot);
    snapshot
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
    retain_ephemeral_terminal_sequence_state: bool,
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
    training_algorithm: TrainingAlgorithm,
    #[module(skip)]
    local_predictive_coding: LocalPredictiveCodingConfig,
    #[module(skip)]
    local_predictive_coding_profile: super::local_predictive_coding::LocalPredictiveCodingProfile,
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
    teacher_runtime: PipelineRuntimeCell,
    #[module(skip)]
    streaming_state: PipelineRuntimeCell,
    #[module(skip)]
    gradient_scale_schedule: GradientScaleSchedule,
    #[module(skip)]
    gradient_scale_step: Arc<AtomicUsize>,
    #[module(skip)]
    stochastic_seed: u64,
    #[module(skip)]
    ruliad_policy_telemetry_path: Option<Arc<PathBuf>>,
    #[module(skip)]
    ruliad_structured_recovery_telemetry_path: Option<Arc<PathBuf>>,
    #[module(skip)]
    ruliad_answer_contract_telemetry_path: Option<Arc<PathBuf>>,
    #[module(skip)]
    ruliad_structured_contrast_telemetry_path: Option<Arc<PathBuf>>,
    #[module(skip)]
    ruliad_field_binding_contrast_telemetry_path: Option<Arc<PathBuf>>,
    #[module(skip)]
    ruliad_field_binding_replay: Arc<Mutex<VecDeque<RuliadFieldBindingReplaySample>>>,
    #[module(skip)]
    ruliad_generated_attractor_replay: Arc<Mutex<RuliadGeneratedAttractorReplay>>,
    #[module(skip)]
    ruliad_generated_attractor_telemetry_path: Option<Arc<PathBuf>>,
    #[module(skip)]
    ruliad_verifier_rollout_telemetry_path: Option<Arc<PathBuf>>,
    #[module(skip)]
    ruliad_proof_policy_telemetry_path: Option<Arc<PathBuf>>,
}

#[derive(Clone, Debug, Serialize)]
struct RuliadPolicyRewardTelemetry {
    version: u32,
    step_index: usize,
    mode: String,
    sample_groups: usize,
    completion_rows: usize,
    oracle_sample_groups: usize,
    oracle_completion_rows: usize,
    oracle_truncated_completion_rows: usize,
    structured_negative_completion_rows: usize,
    generated_attractor_completion_rows: usize,
    gated_sample_groups: usize,
    gated_completion_rows: usize,
    scalarization_count: usize,
    reward_mean: f64,
    reward_std: f64,
    reward_min: f64,
    reward_max: f64,
    advantage_mean: f64,
    advantage_std: f64,
    advantage_clip_fraction: f64,
    policy_update_applied: bool,
    policy_skip_reason: Option<String>,
    vector_sample_count: usize,
    vector_verifier_match_mean: f64,
    vector_semantic_match_mean: f64,
    vector_partial_progress_mean: f64,
    vector_field_accuracy_mean: f64,
    vector_certificate_prefix_mean: f64,
    vector_compactness_mean: f64,
    vector_schema_quality_mean: f64,
    vector_hash_safety_mean: f64,
    vector_answer_termination_mean: f64,
    vector_completion_health_mean: f64,
    vpo_scalarization_dominant_verifier_match: usize,
    vpo_scalarization_dominant_semantic_match: usize,
    vpo_scalarization_dominant_partial_progress: usize,
    vpo_scalarization_dominant_field_accuracy: usize,
    vpo_scalarization_dominant_certificate_prefix: usize,
    vpo_scalarization_dominant_compactness: usize,
    vpo_scalarization_dominant_schema_quality: usize,
    vpo_scalarization_dominant_hash_safety: usize,
    vpo_scalarization_dominant_answer_termination: usize,
    vpo_scalarization_dominant_completion_health: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RuliadStructuredNegativeKind {
    FieldMutation,
    TemplateCollapse,
    SchemaCollapse,
}

#[derive(Clone, Debug, Serialize)]
struct RuliadStructuredContrastTelemetry {
    version: u32,
    step_index: usize,
    skip_reason: Option<String>,
    sample_groups: usize,
    oracle_completion_rows: usize,
    field_negative_completion_rows: usize,
    template_negative_completion_rows: usize,
    schema_negative_completion_rows: usize,
    generated_attractor_negative_completion_rows: usize,
    contrast_pairs: usize,
    contrast_discriminative_tokens: usize,
    structured_contrast_weight: f32,
    structured_contrast_margin: f32,
}

#[derive(Clone, Debug, Serialize)]
struct RuliadAnswerContractTelemetry {
    version: u32,
    step_index: usize,
    policy_batch_present: bool,
    skip_reason: Option<String>,
    sample_groups: usize,
    prompt_schema_sample_groups: usize,
    oracle_rows: usize,
    prompt_schema_rows: usize,
    contract_tokens: usize,
    prompt_schema_value_tokens: usize,
    schema_tokens: usize,
    schema_start_tokens: usize,
    value_tokens: usize,
    other_tokens: usize,
    premature_close_tokens: usize,
    answer_contract_weight: f32,
    premature_close_unlikelihood_weight: f32,
    max_completion_tokens: usize,
    max_rows_per_step: usize,
    prompt_schema_max_rows_per_step: usize,
}

#[derive(Clone, Debug, Serialize)]
struct RuliadFieldBindingContrastTelemetry {
    version: u32,
    objective: &'static str,
    step_index: usize,
    skip_reason: Option<String>,
    sample_groups: usize,
    oracle_prompt_count: usize,
    prompt_pairs: usize,
    contrast_pairs: usize,
    candidate_pairs: usize,
    filtered_presented_action_candidates: usize,
    contrast_discriminative_tokens: usize,
    negative_pool_size: usize,
    replay_pool_size: usize,
    replay_contrast_pairs: usize,
    generated_attractor_pool_size: usize,
    generated_attractor_negative_pool_size: usize,
    generated_attractor_contrast_pairs: usize,
    rank_metric_pairs: usize,
    rank_metric_tokens: usize,
    logit_margin_mean: Option<f64>,
    positive_token_fraction: Option<f64>,
    margin_satisfied_token_fraction: Option<f64>,
    exact_pair_rank_fraction: Option<f64>,
    exact_pair_margin_fraction: Option<f64>,
    sequence_rank_metric_pairs: usize,
    sequence_log_probability_margin_mean: Option<f64>,
    positive_sequence_fraction: Option<f64>,
    sequence_margin_satisfied_fraction: Option<f64>,
    field_binding_contrast_weight: f32,
    field_binding_contrast_margin: f32,
    field_binding_contrast_pair_weight: f32,
}

#[derive(Clone, Debug)]
struct RuliadFieldBindingReplaySample {
    answer: String,
    family: String,
    task_kind: String,
    contract: String,
    oracle_completion: Vec<i64>,
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
struct RuliadGeneratedAttractorKey {
    family: String,
    task_kind: String,
    contract: String,
    answer: String,
}

#[derive(Clone, Debug)]
struct RuliadGeneratedAttractorEntry {
    key: RuliadGeneratedAttractorKey,
    count: usize,
    last_step_index: usize,
    status: burn_dragon_universality::ruliad::RuliadAnswerStatus,
}

#[derive(Clone, Debug, Default)]
struct RuliadGeneratedAttractorReplay {
    entries: HashMap<RuliadGeneratedAttractorKey, RuliadGeneratedAttractorEntry>,
    order: VecDeque<RuliadGeneratedAttractorKey>,
}

#[derive(Clone, Debug, Default)]
struct RuliadGeneratedAttractorReplaySummary {
    pool_size: usize,
    active_count: usize,
    active_observation_count: usize,
    dominant_count: usize,
    distinct_answers: usize,
}

#[derive(Clone, Debug, Serialize)]
struct RuliadGeneratedAttractorReplayTelemetry {
    version: u32,
    step_index: usize,
    source: String,
    skip_reason: Option<String>,
    observed_completion_rows: usize,
    recorded_attractor_rows: usize,
    selected_candidate_rows: usize,
    selected_field_binding_pairs: usize,
    replay_pool_size: usize,
    active_attractor_count: usize,
    active_observation_count: usize,
    distinct_answer_count: usize,
    dominant_answer_count: usize,
    dominant_answer_fraction: f64,
    min_count: usize,
    max_candidates: usize,
    min_distinct_answers: usize,
    max_dominant_fraction: f32,
}

struct RuliadGeneratedAttractorQuery<'a> {
    family: &'a str,
    task_kind: &'a str,
    expected_contract: &'a str,
    expected_answer: &'a str,
    min_count: usize,
    max_candidates: usize,
    min_distinct_answers: usize,
    max_dominant_fraction: f32,
}

type RuliadPromptSchemaValueRow = (Vec<i64>, Vec<i64>, Vec<f32>, usize);

fn take_rows_round_robin<T: Clone>(groups: &[Vec<T>], limit: usize) -> Vec<(usize, T)> {
    let mut selected = Vec::with_capacity(limit);
    let mut rank = 0usize;
    while selected.len() < limit {
        let mut selected_this_round = 0usize;
        for (group_index, group) in groups.iter().enumerate() {
            if let Some(row) = group.get(rank) {
                selected.push((group_index, row.clone()));
                selected_this_round = selected_this_round.saturating_add(1);
                if selected.len() == limit {
                    break;
                }
            }
        }
        if selected_this_round == 0 {
            break;
        }
        rank = rank.saturating_add(1);
    }
    selected
}
type LaggedPredictionTensors<B> = (Tensor<B, 3>, Tensor<B, 2, Int>, Tensor<B, 2, Int>);

impl RuliadGeneratedAttractorReplay {
    fn record(
        &mut self,
        key: RuliadGeneratedAttractorKey,
        status: burn_dragon_universality::ruliad::RuliadAnswerStatus,
        step_index: usize,
        capacity: usize,
    ) -> bool {
        if capacity == 0 {
            return false;
        }
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.count = entry.count.saturating_add(1);
            entry.last_step_index = step_index;
            entry.status = status;
            return true;
        }
        self.order.push_back(key.clone());
        self.entries.insert(
            key.clone(),
            RuliadGeneratedAttractorEntry {
                key,
                count: 1,
                last_step_index: step_index,
                status,
            },
        );
        while self.entries.len() > capacity {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            if self
                .entries
                .get(&oldest)
                .is_some_and(|entry| entry.key == oldest)
            {
                self.entries.remove(&oldest);
            }
        }
        true
    }

    fn candidates_for(
        &self,
        query: RuliadGeneratedAttractorQuery<'_>,
    ) -> Vec<RuliadGeneratedAttractorEntry> {
        let RuliadGeneratedAttractorQuery {
            family,
            task_kind,
            expected_contract,
            expected_answer,
            min_count,
            max_candidates,
            min_distinct_answers,
            max_dominant_fraction,
        } = query;
        if max_candidates == 0 {
            return Vec::new();
        }
        if self
            .summary(min_count)
            .diversity_skip_reason(min_distinct_answers, max_dominant_fraction)
            .is_some()
        {
            return Vec::new();
        }
        let mut candidates = self
            .entries
            .values()
            .filter(|entry| {
                entry.count >= min_count
                    && entry.key.family == family
                    && entry.key.task_kind == task_kind
                    && entry.key.answer != expected_answer
            })
            .cloned()
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| {
            let left_same_contract = left.key.contract == expected_contract;
            let right_same_contract = right.key.contract == expected_contract;
            right_same_contract
                .cmp(&left_same_contract)
                .then_with(|| right.count.cmp(&left.count))
                .then_with(|| right.last_step_index.cmp(&left.last_step_index))
                .then_with(|| left.key.answer.cmp(&right.key.answer))
        });
        candidates.truncate(max_candidates);
        candidates
    }

    fn summary(&self, min_count: usize) -> RuliadGeneratedAttractorReplaySummary {
        let mut counts_by_answer = HashMap::<&str, usize>::new();
        let mut active_count = 0usize;
        let mut active_observation_count = 0usize;
        for entry in self.entries.values() {
            if entry.count < min_count {
                continue;
            }
            active_count = active_count.saturating_add(1);
            active_observation_count = active_observation_count.saturating_add(entry.count);
            *counts_by_answer
                .entry(entry.key.answer.as_str())
                .or_default() += entry.count;
        }
        let dominant_count = counts_by_answer.values().copied().max().unwrap_or(0);
        RuliadGeneratedAttractorReplaySummary {
            pool_size: self.entries.len(),
            active_count,
            active_observation_count,
            dominant_count,
            distinct_answers: counts_by_answer.len(),
        }
    }
}

impl RuliadGeneratedAttractorReplaySummary {
    fn dominant_fraction(&self) -> f64 {
        if self.active_observation_count == 0 {
            0.0
        } else {
            self.dominant_count as f64 / self.active_observation_count as f64
        }
    }

    fn diversity_skip_reason(
        &self,
        min_distinct_answers: usize,
        max_dominant_fraction: f32,
    ) -> Option<&'static str> {
        if self.active_count == 0 {
            return Some("generated_attractor_no_active_entries");
        }
        if self.distinct_answers < min_distinct_answers.max(1) {
            return Some("generated_attractor_low_answer_diversity");
        }
        if self.dominant_fraction() > f64::from(max_dominant_fraction) {
            return Some("generated_attractor_dominant_answer");
        }
        None
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct RuliadFieldBindingRankStats {
    pairs: usize,
    tokens: usize,
    logit_margin_mean: Option<f64>,
    positive_token_fraction: Option<f64>,
    margin_satisfied_token_fraction: Option<f64>,
    exact_pair_rank_fraction: Option<f64>,
    exact_pair_margin_fraction: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default)]
struct RuliadFieldBindingSequenceRankStats {
    pairs: usize,
    log_probability_margin_mean: Option<f64>,
    positive_sequence_fraction: Option<f64>,
    margin_satisfied_sequence_fraction: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
struct RuliadStructuredRecoveryTelemetry {
    version: u32,
    step_index: usize,
    policy_batch_present: bool,
    skip_reason: Option<String>,
    sample_groups: usize,
    recovery_rows: usize,
    field_negative_recovery_rows: usize,
    template_negative_recovery_rows: usize,
    schema_negative_recovery_rows: usize,
    structured_recovery_weight: f32,
    structured_recovery_max_completion_tokens: usize,
}

#[derive(Clone, Debug, Serialize)]
struct RuliadVerifierRolloutImitationTelemetry {
    version: u32,
    step_index: usize,
    skip_reason: Option<String>,
    sample_groups: usize,
    generated_completion_rows: usize,
    candidate_completion_rows: usize,
    accepted_completion_rows: usize,
    accepted_imitation_rows: usize,
    accepted_recovery_rows: usize,
    health_gate_passed: bool,
    verifier_rate_ppm: usize,
    schema_wrong_rate_ppm: usize,
    malformed_rate_ppm: usize,
    verifier_match_rows: usize,
    semantic_match_rows: usize,
    partial_rows: usize,
    schema_wrong_rows: usize,
    malformed_rows: usize,
    missing_rows: usize,
    recovery_partial_rows: usize,
    recovery_schema_wrong_rows: usize,
    recovery_malformed_rows: usize,
    recovery_missing_rows: usize,
    field_accuracy_mean: f64,
    partial_progress_mean: f64,
    completion_quality_mean: f64,
    rollout_imitation_weight: f32,
    rollout_recovery_weight: f32,
    max_completion_tokens: usize,
}

#[derive(Clone, Debug, Serialize)]
struct RuliadProofPolicyDaggerTelemetry {
    version: u32,
    answer_contract: &'static str,
    objective: &'static str,
    gradient_scope: &'static str,
    presentation_risk: &'static str,
    configured_mode: &'static str,
    mode: &'static str,
    candidate_symmetry: &'static str,
    step_index: usize,
    skip_reason: Option<String>,
    available_sample_groups: usize,
    sample_groups: usize,
    nonzero_start_trajectories: usize,
    mean_start_step: f64,
    visited_states: usize,
    semantic_state_rows: usize,
    base_semantic_state_rows: usize,
    counterfactual_semantic_state_rows: usize,
    counterfactual_target_shortfall: usize,
    expert_rows: usize,
    static_expert_rows: usize,
    dagger_expert_rows: usize,
    model_visited_expert_rows: usize,
    supervised_action_tokens: usize,
    supervised_presentation_rows: usize,
    mean_presentations_per_state: f64,
    model_valid_actions: usize,
    model_invalid_actions: usize,
    model_expert_equivalent_actions: usize,
    model_off_expert_actions: usize,
    repeated_states: usize,
    model_backtracks: usize,
    solved_proofs: usize,
    model_scoring_batches: usize,
    maximum_model_scoring_batch_rows: usize,
    model_scoring_padded_tokens: usize,
    sampling_model_materialize_ms: f64,
    state_prepare_ms: f64,
    rollout_cpu_prepare_ms: f64,
    model_scoring_ms: f64,
    difficulty_sample_groups: BTreeMap<usize, usize>,
    difficulty_visited_states: BTreeMap<usize, usize>,
    difficulty_expert_rows: BTreeMap<usize, usize>,
    expert_selected_index_histogram: BTreeMap<usize, usize>,
    expert_equivalent_index_histogram: BTreeMap<usize, usize>,
    model_selected_index_histogram: BTreeMap<usize, usize>,
    candidate_target_tokens: usize,
    equivalent_target_tokens: usize,
    mean_candidate_targets_per_row: f64,
    mean_equivalent_targets_per_row: f64,
    prefix_branch_rows: usize,
    prefix_candidate_tokens: usize,
    prefix_equivalent_tokens: usize,
    weight: f32,
    rollout_steps: usize,
    rollout_depth_reached: usize,
    configured_rollout_steps: usize,
    trajectory_budget: usize,
    semantic_row_budget: usize,
    base_semantic_row_budget: usize,
    configured_counterfactual_targets_per_state: usize,
    target_variants_per_state: usize,
    max_rows_per_update: usize,
    max_presentation_rows_per_update: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RuliadProofPolicyBatchPlan {
    static_row_budget: usize,
    dagger_row_budget: usize,
    dagger_trajectory_budget: usize,
    dagger_base_depth: usize,
    dagger_depth_remainder: usize,
    rollout_steps: usize,
}

impl RuliadProofPolicyBatchPlan {
    fn new(
        mode: crate::config::RuliadProofPolicyEffectiveMode,
        max_rows_per_update: usize,
        configured_rollout_steps: usize,
        stratified_difficulty_levels: usize,
    ) -> Self {
        let maximum_rows = max_rows_per_update.max(1);
        let (static_row_budget, dagger_row_budget) = match mode {
            crate::config::RuliadProofPolicyEffectiveMode::StaticExpert => (maximum_rows, 0),
            crate::config::RuliadProofPolicyEffectiveMode::Dagger => (0, maximum_rows),
            crate::config::RuliadProofPolicyEffectiveMode::PairedDagger => {
                let dagger_rows = maximum_rows / 2;
                (maximum_rows - dagger_rows, dagger_rows)
            }
        };
        let configured_rollout_steps = configured_rollout_steps.max(1);
        let dagger_trajectory_budget = if dagger_row_budget == 0 {
            0
        } else {
            dagger_row_budget
                .div_ceil(configured_rollout_steps)
                .max(stratified_difficulty_levels.max(1).min(dagger_row_budget))
        };
        let dagger_base_depth = dagger_row_budget
            .checked_div(dagger_trajectory_budget)
            .unwrap_or(0);
        let dagger_depth_remainder = dagger_row_budget
            .checked_rem(dagger_trajectory_budget)
            .unwrap_or(0);
        let rollout_steps = if dagger_trajectory_budget == 0 {
            1
        } else {
            dagger_base_depth + usize::from(dagger_depth_remainder > 0)
        };
        debug_assert!(rollout_steps <= configured_rollout_steps);
        Self {
            static_row_budget,
            dagger_row_budget,
            dagger_trajectory_budget,
            dagger_base_depth,
            dagger_depth_remainder,
            rollout_steps,
        }
    }

    fn trajectory_budget(self) -> usize {
        self.static_row_budget
            .saturating_add(self.dagger_trajectory_budget)
    }

    fn dagger_depth(self, trajectory_index: usize) -> usize {
        self.dagger_base_depth + usize::from(trajectory_index < self.dagger_depth_remainder)
    }
}

fn verifier_equivalent_action_loss<B: Backend>(
    branch_logits: Tensor<B, 3>,
    candidate_mask: Tensor<B, 3>,
    equivalent_mask: Tensor<B, 3>,
    normalization: crate::config::RuliadProofPolicyNormalization,
    weight: f32,
) -> Tensor<B, 1> {
    let row_count = branch_logits.shape().dims::<3>()[0];
    verifier_equivalent_action_log_probabilities(
        branch_logits,
        candidate_mask,
        equivalent_mask,
        normalization,
    )
    .sum()
    .reshape([1])
    .div_scalar(row_count.max(1) as f32)
    .mul_scalar(-weight)
}

fn verifier_equivalent_action_log_probabilities<B: Backend>(
    branch_logits: Tensor<B, 3>,
    candidate_mask: Tensor<B, 3>,
    equivalent_mask: Tensor<B, 3>,
    normalization: crate::config::RuliadProofPolicyNormalization,
) -> Tensor<B, 1> {
    let [row_count, branch_count, _] = branch_logits.shape().dims::<3>();
    debug_assert_eq!(branch_count, 1);
    let probabilities = log_probs_from_logits(branch_logits).exp();
    let candidate_probability = (probabilities.clone() * candidate_mask)
        .sum_dim(2)
        .reshape([row_count])
        .clamp_min(1.0e-12);
    let equivalent_probability = (probabilities * equivalent_mask)
        .sum_dim(2)
        .reshape([row_count])
        .clamp_min(1.0e-12);
    let objective_probability = match normalization {
        crate::config::RuliadProofPolicyNormalization::CandidateConditional => {
            equivalent_probability
                .div(candidate_probability)
                .clamp_max(1.0)
        }
        crate::config::RuliadProofPolicyNormalization::PrefixConditional => equivalent_probability
            .div(candidate_probability)
            .clamp_max(1.0),
        crate::config::RuliadProofPolicyNormalization::VocabularyMarginal => equivalent_probability,
    };
    objective_probability.clamp_min(1.0e-12).log()
}

#[allow(clippy::too_many_arguments)]
fn grouped_verifier_equivalent_action_loss<B: Backend>(
    branch_logits: Tensor<B, 3>,
    candidate_mask: Tensor<B, 3>,
    equivalent_mask: Tensor<B, 3>,
    row_weights: Tensor<B, 1>,
    normalization: crate::config::RuliadProofPolicyNormalization,
    presentation_risk: crate::config::RuliadProofPolicyPresentationRisk,
    presentation_group_size: usize,
    weight: f32,
) -> Tensor<B, 1> {
    let row_log_probabilities = verifier_equivalent_action_log_probabilities(
        branch_logits,
        candidate_mask,
        equivalent_mask,
        normalization,
    );
    grouped_action_log_probability_loss(
        row_log_probabilities,
        row_weights,
        presentation_risk,
        presentation_group_size,
        weight,
    )
}

fn sequence_logsumexp<B: Backend>(scores: Tensor<B, 2>) -> Tensor<B, 1> {
    let row_count = scores.shape().dims::<2>()[0];
    let maximum = scores.clone().max_dim(1);
    ((scores - maximum.clone())
        .exp()
        .sum_dim(1)
        .clamp_min(1.0e-12)
        .log()
        + maximum)
        .reshape([row_count])
}

fn verifier_equivalent_sequence_log_probabilities<B: Backend>(
    mean_log_scores: Tensor<B, 2>,
    sum_log_scores: Tensor<B, 2>,
    equivalent_mask: Tensor<B, 2>,
    normalization: crate::config::RuliadProofPolicyNormalization,
) -> Tensor<B, 1> {
    match normalization {
        crate::config::RuliadProofPolicyNormalization::CandidateConditional => {
            let equivalent_scores =
                mean_log_scores.clone() + equivalent_mask.sub_scalar(1.0).mul_scalar(1.0e9);
            sequence_logsumexp(equivalent_scores) - sequence_logsumexp(mean_log_scores)
        }
        crate::config::RuliadProofPolicyNormalization::PrefixConditional => {
            let equivalent_scores =
                mean_log_scores.clone() + equivalent_mask.sub_scalar(1.0).mul_scalar(1.0e9);
            sequence_logsumexp(equivalent_scores) - sequence_logsumexp(mean_log_scores)
        }
        crate::config::RuliadProofPolicyNormalization::VocabularyMarginal => {
            let equivalent_scores =
                sum_log_scores + equivalent_mask.sub_scalar(1.0).mul_scalar(1.0e9);
            sequence_logsumexp(equivalent_scores)
        }
    }
}

#[derive(Clone, Copy)]
struct GroupedVerifierSequenceLossConfig {
    normalization: crate::config::RuliadProofPolicyNormalization,
    presentation_risk: crate::config::RuliadProofPolicyPresentationRisk,
    presentation_group_size: usize,
    weight: f32,
}

fn grouped_verifier_equivalent_sequence_loss<B: Backend>(
    mean_log_scores: Tensor<B, 2>,
    sum_log_scores: Tensor<B, 2>,
    equivalent_mask: Tensor<B, 2>,
    row_weights: Tensor<B, 1>,
    config: GroupedVerifierSequenceLossConfig,
) -> Tensor<B, 1> {
    let row_log_probabilities = verifier_equivalent_sequence_log_probabilities(
        mean_log_scores,
        sum_log_scores,
        equivalent_mask,
        config.normalization,
    );
    grouped_action_log_probability_loss(
        row_log_probabilities,
        row_weights,
        config.presentation_risk,
        config.presentation_group_size,
        config.weight,
    )
}

fn grouped_action_log_probability_loss<B: Backend>(
    row_log_probabilities: Tensor<B, 1>,
    row_weights: Tensor<B, 1>,
    presentation_risk: crate::config::RuliadProofPolicyPresentationRisk,
    presentation_group_size: usize,
    weight: f32,
) -> Tensor<B, 1> {
    if presentation_risk == crate::config::RuliadProofPolicyPresentationRisk::Mean {
        let normalizer = row_weights.clone().sum().reshape([1]).clamp_min(1.0e-12);
        return (row_log_probabilities * row_weights)
            .sum()
            .reshape([1])
            .div(normalizer)
            .mul_scalar(-weight);
    }
    let row_count = row_log_probabilities.shape().dims::<1>()[0];
    let group_size = presentation_group_size.max(1);
    debug_assert!(row_count.is_multiple_of(group_size));
    let group_count = row_count.checked_div(group_size).unwrap_or_default().max(1);
    row_log_probabilities
        .reshape([group_count, group_size])
        .min_dim(1)
        .sum()
        .reshape([1])
        .div_scalar(group_count as f32)
        .mul_scalar(-weight)
}

#[derive(Clone, Debug)]
struct RuliadPolicyRewardTelemetryAccumulator {
    mode: String,
    step_index: usize,
    sample_groups: usize,
    completion_rows: usize,
    oracle_sample_groups: usize,
    oracle_completion_rows: usize,
    oracle_truncated_completion_rows: usize,
    structured_negative_completion_rows: usize,
    generated_attractor_completion_rows: usize,
    gated_sample_groups: usize,
    gated_completion_rows: usize,
    scalarization_count: usize,
    rewards: Vec<f32>,
    advantages: Vec<f32>,
    clipped_advantage_count: usize,
    policy_update_applied: bool,
    policy_skip_reason: Option<String>,
    vector_sums: [f64; burn_dragon_universality::ruliad::RULIAD_VERIFIER_REWARD_VECTOR_DIM],
    vector_sample_count: usize,
    vpo_scalarization_dominant_axis_counts:
        [usize; burn_dragon_universality::ruliad::RULIAD_VERIFIER_REWARD_VECTOR_DIM],
}

impl RuliadPolicyRewardTelemetryAccumulator {
    fn new(mode: crate::config::train::RuliadVerifierRewardMode, step_index: usize) -> Self {
        Self {
            mode: match mode {
                crate::config::train::RuliadVerifierRewardMode::Scalar => "scalar".to_string(),
                crate::config::train::RuliadVerifierRewardMode::VpoIndependent => {
                    "vpo_independent".to_string()
                }
            },
            step_index,
            sample_groups: 0,
            completion_rows: 0,
            oracle_sample_groups: 0,
            oracle_completion_rows: 0,
            oracle_truncated_completion_rows: 0,
            structured_negative_completion_rows: 0,
            generated_attractor_completion_rows: 0,
            gated_sample_groups: 0,
            gated_completion_rows: 0,
            scalarization_count: 0,
            rewards: Vec::new(),
            advantages: Vec::new(),
            clipped_advantage_count: 0,
            policy_update_applied: true,
            policy_skip_reason: None,
            vector_sums: [0.0; burn_dragon_universality::ruliad::RULIAD_VERIFIER_REWARD_VECTOR_DIM],
            vector_sample_count: 0,
            vpo_scalarization_dominant_axis_counts: [0;
                burn_dragon_universality::ruliad::RULIAD_VERIFIER_REWARD_VECTOR_DIM],
        }
    }

    fn record_vectors(
        &mut self,
        scores: &[burn_dragon_universality::ruliad::RuliadReasoningScore],
    ) {
        for score in scores {
            let vector = burn_dragon_universality::ruliad::ruliad_verifier_reward_vector(score);
            for (index, component) in vector.components().into_iter().enumerate() {
                self.vector_sums[index] += f64::from(component);
            }
            self.vector_sample_count = self.vector_sample_count.saturating_add(1);
        }
    }

    fn record_rewards_and_advantages(
        &mut self,
        rewards: &[f32],
        advantages: &[f32],
        clip_range: f32,
    ) {
        self.sample_groups = self.sample_groups.saturating_add(1);
        self.completion_rows = self.completion_rows.saturating_add(rewards.len());
        self.rewards.extend_from_slice(rewards);
        self.advantages.extend_from_slice(advantages);
        self.clipped_advantage_count = self.clipped_advantage_count.saturating_add(
            advantages
                .iter()
                .filter(|advantage| advantage.abs() > clip_range)
                .count(),
        );
    }

    fn record_gated_group(&mut self, completion_rows: usize) {
        self.gated_sample_groups = self.gated_sample_groups.saturating_add(1);
        self.gated_completion_rows = self.gated_completion_rows.saturating_add(completion_rows);
    }

    fn record_oracle_candidate(&mut self, truncated: bool) {
        self.oracle_sample_groups = self.oracle_sample_groups.saturating_add(1);
        self.oracle_completion_rows = self.oracle_completion_rows.saturating_add(1);
        if truncated {
            self.oracle_truncated_completion_rows =
                self.oracle_truncated_completion_rows.saturating_add(1);
        }
    }

    fn record_structured_negative_candidate(&mut self) {
        self.structured_negative_completion_rows =
            self.structured_negative_completion_rows.saturating_add(1);
    }

    fn record_generated_attractor_candidate(&mut self) {
        self.generated_attractor_completion_rows =
            self.generated_attractor_completion_rows.saturating_add(1);
    }

    fn has_observations(&self) -> bool {
        self.completion_rows > 0 || self.gated_completion_rows > 0 || self.vector_sample_count > 0
    }

    fn record_vpo_scalarization(
        &mut self,
        weights: &[f32; burn_dragon_universality::ruliad::RULIAD_VERIFIER_REWARD_VECTOR_DIM],
    ) {
        self.scalarization_count = self.scalarization_count.saturating_add(1);
        let axis = weights
            .iter()
            .copied()
            .enumerate()
            .max_by(|(_, left), (_, right)| {
                left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(index, _)| index)
            .unwrap_or(0);
        if let Some(count) = self.vpo_scalarization_dominant_axis_counts.get_mut(axis) {
            *count = count.saturating_add(1);
        }
    }

    fn advantage_clip_fraction(&self) -> f64 {
        if self.advantages.is_empty() {
            0.0
        } else {
            self.clipped_advantage_count as f64 / self.advantages.len() as f64
        }
    }

    fn mark_skipped(&mut self, reason: impl Into<String>) {
        self.policy_update_applied = false;
        self.policy_skip_reason = Some(reason.into());
    }

    fn finish(self) -> Option<RuliadPolicyRewardTelemetry> {
        if !self.has_observations() {
            return None;
        }
        let (reward_mean, reward_std, reward_min, reward_max) = stats_f64(&self.rewards);
        let (advantage_mean, advantage_std, _, _) = stats_f64(&self.advantages);
        let advantage_clip_fraction = self.advantage_clip_fraction();
        let vector_mean = |index: usize| {
            if self.vector_sample_count == 0 {
                0.0
            } else {
                self.vector_sums[index] / self.vector_sample_count as f64
            }
        };
        Some(RuliadPolicyRewardTelemetry {
            version: 2,
            step_index: self.step_index,
            mode: self.mode,
            sample_groups: self.sample_groups,
            completion_rows: self.completion_rows,
            oracle_sample_groups: self.oracle_sample_groups,
            oracle_completion_rows: self.oracle_completion_rows,
            oracle_truncated_completion_rows: self.oracle_truncated_completion_rows,
            structured_negative_completion_rows: self.structured_negative_completion_rows,
            generated_attractor_completion_rows: self.generated_attractor_completion_rows,
            gated_sample_groups: self.gated_sample_groups,
            gated_completion_rows: self.gated_completion_rows,
            scalarization_count: self.scalarization_count,
            reward_mean,
            reward_std,
            reward_min,
            reward_max,
            advantage_mean,
            advantage_std,
            advantage_clip_fraction,
            policy_update_applied: self.policy_update_applied,
            policy_skip_reason: self.policy_skip_reason,
            vector_sample_count: self.vector_sample_count,
            vector_verifier_match_mean: vector_mean(0),
            vector_semantic_match_mean: vector_mean(1),
            vector_partial_progress_mean: vector_mean(2),
            vector_field_accuracy_mean: vector_mean(3),
            vector_certificate_prefix_mean: vector_mean(4),
            vector_compactness_mean: vector_mean(5),
            vector_schema_quality_mean: vector_mean(6),
            vector_hash_safety_mean: vector_mean(7),
            vector_answer_termination_mean: vector_mean(8),
            vector_completion_health_mean: vector_mean(9),
            vpo_scalarization_dominant_verifier_match: self.vpo_scalarization_dominant_axis_counts
                [0],
            vpo_scalarization_dominant_semantic_match: self.vpo_scalarization_dominant_axis_counts
                [1],
            vpo_scalarization_dominant_partial_progress: self
                .vpo_scalarization_dominant_axis_counts[2],
            vpo_scalarization_dominant_field_accuracy: self.vpo_scalarization_dominant_axis_counts
                [3],
            vpo_scalarization_dominant_certificate_prefix: self
                .vpo_scalarization_dominant_axis_counts[4],
            vpo_scalarization_dominant_compactness: self.vpo_scalarization_dominant_axis_counts[5],
            vpo_scalarization_dominant_schema_quality: self.vpo_scalarization_dominant_axis_counts
                [6],
            vpo_scalarization_dominant_hash_safety: self.vpo_scalarization_dominant_axis_counts[7],
            vpo_scalarization_dominant_answer_termination: self
                .vpo_scalarization_dominant_axis_counts[8],
            vpo_scalarization_dominant_completion_health: self
                .vpo_scalarization_dominant_axis_counts[9],
        })
    }
}

fn stats_f64(values: &[f32]) -> (f64, f64, f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut sum = 0.0;
    for value in values.iter().copied().map(f64::from) {
        min = min.min(value);
        max = max.max(value);
        sum += value;
    }
    let mean = sum / values.len() as f64;
    let variance = values
        .iter()
        .copied()
        .map(f64::from)
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / values.len() as f64;
    (mean, variance.sqrt(), min, max)
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
pub(crate) struct SequenceStateDiagnostics {
    pub rho_layers: usize,
    pub rho_rms: f64,
    /// Fraction of rho energy that varies across sampled latent-memory rows.
    pub rho_slot_variance_ratio: f64,
    /// RMS off-diagonal cosine similarity between sampled rho rows. Lower is less redundant.
    pub rho_slot_redundancy: f64,
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
    clip_fraction_mean: Option<f64>,
    amortization_components: usize,
    amortization_loss: Option<f64>,
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
        self.amortization_components = self
            .amortization_components
            .saturating_add(report.amortization_components);
        self.elapsed_ns = self.elapsed_ns.saturating_add(report.elapsed_ns);
    }

    fn record(self) {
        crate::train::profile::record_predictive_coding(
            crate::train::profile::PredictiveCodingProfileRecord {
                chunks_seen: self.chunks_seen,
                chunks_corrected: self.chunks_corrected,
                inference_steps: self.inference_steps,
                skipped_empty_state: self.skipped_empty_state,
                energy_before: self.energy_before,
                energy_after: self.energy_after,
                grad_norm_mean: self.grad_norm_mean,
                grad_norm_max: self.grad_norm_max,
                delta_rms_mean: self.delta_rms_mean,
                clip_fraction_mean: self.clip_fraction_mean,
                amortization_components: self.amortization_components,
                amortization_loss: self.amortization_loss,
                elapsed_ns: self.elapsed_ns,
            },
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
    clip_fraction_sum: f64,
}

impl PredictiveCodingTensorUpdateStats {
    fn record_unsynced(&mut self) {
        self.tensor_count = self.tensor_count.saturating_add(1);
    }

    fn record_synced<B: BackendTrait>(
        &mut self,
        grad_norm: Tensor<B, 1>,
        grad_norm_max: Tensor<B, 1>,
        delta_rms: Tensor<B, 1>,
        clip_fraction: Tensor<B, 1>,
    ) {
        let values = Tensor::cat(vec![grad_norm, grad_norm_max, delta_rms, clip_fraction], 0)
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("predictive-coding diagnostic tensor");
        let [grad_norm, grad_norm_max, delta_rms, clip_fraction] = values.as_slice() else {
            return;
        };
        let (grad_norm, grad_norm_max, delta_rms, clip_fraction) = (
            *grad_norm as f64,
            *grad_norm_max as f64,
            *delta_rms as f64,
            *clip_fraction as f64,
        );
        if grad_norm.is_finite()
            && grad_norm_max.is_finite()
            && delta_rms.is_finite()
            && clip_fraction.is_finite()
        {
            self.tensor_count = self.tensor_count.saturating_add(1);
            self.diagnostic_count = self.diagnostic_count.saturating_add(1);
            self.grad_norm_sum += grad_norm;
            self.grad_norm_max = self.grad_norm_max.max(grad_norm_max);
            self.delta_rms_sum += delta_rms;
            self.clip_fraction_sum += clip_fraction;
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

    fn clip_fraction_mean(self) -> Option<f64> {
        (self.diagnostic_count > 0).then(|| self.clip_fraction_sum / self.diagnostic_count as f64)
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
            retain_ephemeral_terminal_sequence_state: false,
            objective: TrainingObjectiveConfig::NextToken,
            input_corruption: CausalInputCorruptionConfig::default(),
            logit_entropy_floor: LogitEntropyFloorConfig::default(),
            repeat_unlikelihood: RepeatUnlikelihoodConfig::default(),
            greedy_rollout_unlikelihood: GreedyRolloutUnlikelihoodConfig::default(),
            dynamics_anchor: DynamicsAnchorConfig::default(),
            predictive_coding: PredictiveCodingConfig::default(),
            training_algorithm: TrainingAlgorithm::Auto,
            local_predictive_coding: LocalPredictiveCodingConfig::default(),
            local_predictive_coding_profile:
                super::local_predictive_coding::LocalPredictiveCodingProfile::default(),
            latent_reasoning: LatentReasoningTrainingConfig::default(),
            ruliad_supervision: RuliadSupervisionConfig::default(),
            latent_reasoning_capability_gate_open: Arc::new(AtomicBool::new(false)),
            greedy_rollout_recovery_active: Arc::new(AtomicBool::new(false)),
            teacher_model: None,
            teacher_runtime: PipelineRuntimeCell::default(),
            streaming_state: PipelineRuntimeCell::default(),
            gradient_scale_schedule: GradientScaleSchedule::default(),
            gradient_scale_step: Arc::new(AtomicUsize::new(0)),
            stochastic_seed: 0,
            ruliad_policy_telemetry_path: None,
            ruliad_structured_recovery_telemetry_path: None,
            ruliad_answer_contract_telemetry_path: None,
            ruliad_structured_contrast_telemetry_path: None,
            ruliad_field_binding_contrast_telemetry_path: None,
            ruliad_field_binding_replay: Arc::new(Mutex::new(VecDeque::new())),
            ruliad_generated_attractor_replay: Arc::new(Mutex::new(
                RuliadGeneratedAttractorReplay::default(),
            )),
            ruliad_generated_attractor_telemetry_path: None,
            ruliad_verifier_rollout_telemetry_path: None,
            ruliad_proof_policy_telemetry_path: None,
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

    pub(crate) fn materialize_random_scaffold_for_inference(mut self) -> Self {
        self.model = self.model.materialize_random_scaffold_for_inference();
        self.teacher_model = self
            .teacher_model
            .map(DragonModel::materialize_random_scaffold_for_inference);
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

    pub fn with_ephemeral_terminal_sequence_state_retention(mut self, retain: bool) -> Self {
        self.retain_ephemeral_terminal_sequence_state = retain;
        self
    }

    pub fn with_training_objective(mut self, objective: TrainingObjectiveConfig) -> Self {
        self.teacher_model =
            (!objective.is_next_token()).then(|| detach_teacher_model(&self.model));
        *self
            .teacher_runtime
            .inner
            .lock()
            .expect("teacher model runtime lock poisoned") = self
            .teacher_model
            .clone()
            .map(|model| Box::new(TeacherModelRuntime::new(model)) as Box<dyn Any + Send>);
        self.objective = objective;
        self
    }

    /// Applies the objective and auxiliary-loss portion of a language training contract.
    ///
    /// Keep this path shared by local, distributed, and peer-to-peer executors so that
    /// changing the launch mode cannot silently change what the model is optimizing.
    pub fn with_training_objectives(self, training: &TrainingHyperparameters) -> Self {
        self.with_stochastic_seed(training.seed)
            .with_training_objective(training.objective.clone())
            .with_input_corruption(training.input_corruption.clone())
            .with_logit_entropy_floor(training.logit_entropy_floor.clone())
            .with_repeat_unlikelihood(training.repeat_unlikelihood.clone())
            .with_greedy_rollout_unlikelihood(training.greedy_rollout_unlikelihood.clone())
            .with_dynamics_anchor(training.dynamics_anchor.clone())
            .with_predictive_coding(training.predictive_coding.clone())
            .with_training_algorithm(training.algorithm)
            .with_local_predictive_coding(training.local_predictive_coding.clone())
            .with_latent_reasoning(training.latent_reasoning.clone())
            .with_ruliad_supervision(training.ruliad_supervision)
    }

    pub fn with_stochastic_seed(mut self, seed: u64) -> Self {
        self.stochastic_seed = seed;
        self
    }

    /// Applies the complete launch-independent language training contract.
    pub fn with_training_configuration(
        self,
        training: &TrainingHyperparameters,
        total_steps: usize,
    ) -> Self
    where
        B: AutodiffBackend,
    {
        self.with_training_objectives(training)
            .with_tbptt_chunk_size(training.tbptt_chunk_size)
            .with_tbptt_persist_across_steps(training.tbptt_persist_across_steps)
            .with_ephemeral_terminal_sequence_state_retention(
                training.retain_ephemeral_terminal_sequence_state,
            )
            .with_continual_backprop(&training.continual_backprop)
            .with_gradient_scale_schedule(training, total_steps)
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
            let teacher_model = self
                .teacher_model
                .clone()
                .unwrap_or_else(|| detach_teacher_model(&self.model));
            let teacher_model = detach_teacher_model(&teacher_model);
            self.teacher_model = Some(teacher_model.clone());
            let mut runtime = self
                .teacher_runtime
                .inner
                .lock()
                .expect("teacher model runtime lock poisoned");
            if runtime.is_none() {
                *runtime = Some(Box::new(TeacherModelRuntime::new(teacher_model)));
            }
        }
        self
    }

    pub fn with_predictive_coding(mut self, config: PredictiveCodingConfig) -> Self {
        self.predictive_coding = config;
        self
    }

    pub fn with_training_algorithm(mut self, algorithm: TrainingAlgorithm) -> Self {
        self.training_algorithm = algorithm;
        self
    }

    pub fn with_local_predictive_coding(mut self, config: LocalPredictiveCodingConfig) -> Self {
        self.local_predictive_coding = config;
        self
    }

    pub fn local_predictive_coding_profile(
        &self,
    ) -> super::local_predictive_coding::LocalPredictiveCodingProfile {
        self.local_predictive_coding_profile.clone()
    }

    pub fn with_latent_reasoning(mut self, config: LatentReasoningTrainingConfig) -> Self {
        self.latent_reasoning = config;
        if self.latent_reasoning.enabled
            && (matches!(
                self.latent_reasoning.target_encoder,
                crate::config::LatentReasoningTargetEncoder::EmaTeacher
            ) || self.latent_reasoning.dragon_state.enabled)
        {
            let teacher_model = self
                .teacher_model
                .clone()
                .unwrap_or_else(|| detach_teacher_model(&self.model));
            let teacher_model = detach_teacher_model(&teacher_model);
            self.teacher_model = Some(teacher_model.clone());
            let mut runtime = self
                .teacher_runtime
                .inner
                .lock()
                .expect("teacher model runtime lock poisoned");
            if runtime.is_none() {
                *runtime = Some(Box::new(TeacherModelRuntime::new(teacher_model)));
            }
        }
        self
    }

    pub fn with_ruliad_supervision(mut self, config: RuliadSupervisionConfig) -> Self {
        if config.verifier_reward.enabled && config.verifier_reward.kl_weight > f32::EPSILON {
            let teacher_model = detach_teacher_model(&self.model);
            self.teacher_model = Some(teacher_model.clone());
            let mut runtime = self
                .teacher_runtime
                .inner
                .lock()
                .expect("teacher model runtime lock poisoned");
            if runtime.is_none() {
                *runtime = Some(Box::new(TeacherModelRuntime::new(teacher_model)));
            }
        }
        self.ruliad_supervision = config;
        self
    }

    pub fn with_ruliad_policy_telemetry_path(mut self, path: Option<PathBuf>) -> Self {
        self.ruliad_policy_telemetry_path = path.map(Arc::new);
        self
    }

    pub fn with_ruliad_structured_recovery_telemetry_path(mut self, path: Option<PathBuf>) -> Self {
        self.ruliad_structured_recovery_telemetry_path = path.map(Arc::new);
        self
    }

    pub fn with_ruliad_answer_contract_telemetry_path(mut self, path: Option<PathBuf>) -> Self {
        self.ruliad_answer_contract_telemetry_path = path.map(Arc::new);
        self
    }

    pub fn with_ruliad_structured_contrast_telemetry_path(mut self, path: Option<PathBuf>) -> Self {
        self.ruliad_structured_contrast_telemetry_path = path.map(Arc::new);
        self
    }

    pub fn with_ruliad_field_binding_contrast_telemetry_path(
        mut self,
        path: Option<PathBuf>,
    ) -> Self {
        self.ruliad_field_binding_contrast_telemetry_path = path.map(Arc::new);
        self
    }

    pub fn with_ruliad_generated_attractor_telemetry_path(mut self, path: Option<PathBuf>) -> Self {
        self.ruliad_generated_attractor_telemetry_path = path.map(Arc::new);
        self
    }

    pub fn with_ruliad_verifier_rollout_telemetry_path(mut self, path: Option<PathBuf>) -> Self {
        self.ruliad_verifier_rollout_telemetry_path = path.map(Arc::new);
        self
    }

    pub fn with_ruliad_proof_policy_telemetry_path(mut self, path: Option<PathBuf>) -> Self {
        self.ruliad_proof_policy_telemetry_path = path.map(Arc::new);
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

    fn can_elide_terminal_sequence_state(&self, block_size: usize) -> bool {
        !self.retain_ephemeral_terminal_sequence_state
            && !self.tbptt_persist_across_steps
            && self.effective_tbptt_chunk_size(block_size).is_none()
            && !self.pipeline_enabled()
            && !self.predictive_coding.enabled
            && !(self.latent_reasoning.enabled
                && (self.latent_reasoning.dragon_state.enabled
                    || (self.latent_reasoning.sigreg.enabled
                        && matches!(
                            self.latent_reasoning.sigreg.target,
                            crate::config::LatentReasoningSigRegTarget::RhoMemorySlots
                                | crate::config::LatentReasoningSigRegTarget::HiddenAndRhoMemorySlots
                        ))))
            && self.model.supports_terminal_sequence_state_elision()
    }

    fn load_step_state(&self, reset_stream_state: bool, block_size: usize) -> ModelState<B> {
        if !self.tbptt_persist_across_steps {
            return if self.can_elide_terminal_sequence_state(block_size) {
                self.model.init_state_stateless()
            } else {
                self.model.init_state_ephemeral()
            };
        }
        let mut runtime = self
            .streaming_state
            .inner
            .lock()
            .expect("streaming TBPTT state lock poisoned");
        if reset_stream_state {
            *runtime = None;
        }
        runtime
            .take()
            .and_then(|state| state.downcast::<ModelState<B>>().ok().map(|state| *state))
            .unwrap_or_else(|| self.model.init_state())
    }

    fn store_step_state(&self, mut state: ModelState<B>) {
        if !self.tbptt_persist_across_steps {
            return;
        }
        state.detach_in_place();
        *self
            .streaming_state
            .inner
            .lock()
            .expect("streaming TBPTT state lock poisoned") = Some(Box::new(state));
    }

    pub(crate) fn streaming_state_for_checkpoint(&self) -> Option<ModelState<B>> {
        self.streaming_state
            .inner
            .lock()
            .expect("streaming TBPTT state lock poisoned")
            .as_ref()
            .and_then(|state| state.downcast_ref::<ModelState<B>>().cloned())
            .map(|mut state| {
                state.detach_in_place();
                state
            })
    }

    pub(crate) fn gradient_scale_step_for_checkpoint(&self) -> usize {
        self.gradient_scale_step.load(Ordering::Relaxed)
    }

    pub(crate) fn restore_gradient_scale_step_from_checkpoint(&self, step: usize) {
        self.gradient_scale_step.store(step, Ordering::Relaxed);
    }

    pub(crate) fn teacher_model_for_checkpoint(&self) -> Option<(DragonModel<B>, usize)> {
        self.teacher_model.as_ref()?;
        let runtime = self
            .teacher_runtime
            .inner
            .lock()
            .expect("teacher model runtime lock poisoned");
        let runtime = runtime
            .as_ref()
            .and_then(|runtime| runtime.downcast_ref::<TeacherModelRuntime<B>>());
        Some(runtime.map_or_else(
            || {
                (
                    self.teacher_model
                        .clone()
                        .expect("checked teacher model presence"),
                    0,
                )
            },
            |runtime| (runtime.model.clone(), runtime.update_count),
        ))
    }

    pub(crate) fn restore_teacher_model_from_checkpoint(
        &self,
        model: DragonModel<B>,
        update_count: usize,
    ) {
        *self
            .teacher_runtime
            .inner
            .lock()
            .expect("teacher model runtime lock poisoned") = Some(Box::new(TeacherModelRuntime {
            model,
            update_count,
        }));
    }

    pub(crate) fn restore_streaming_state_from_checkpoint(
        &self,
        mut state: ModelState<B>,
    ) -> Result<(), String> {
        let expected_layers = self.model.init_state().layers.len();
        if state.layers.len() != expected_layers {
            return Err(format!(
                "runtime-state checkpoint has {} layers, expected {expected_layers}",
                state.layers.len()
            ));
        }
        state.detach_in_place();
        *self
            .streaming_state
            .inner
            .lock()
            .expect("streaming TBPTT state lock poisoned") = Some(Box::new(state));
        Ok(())
    }

    #[cfg(test)]
    fn peek_step_state_for_test(&self) -> Option<ModelState<B>> {
        self.streaming_state
            .inner
            .lock()
            .expect("streaming TBPTT state lock poisoned")
            .as_ref()
            .and_then(|state| state.downcast_ref::<ModelState<B>>().cloned())
    }

    pub(crate) fn slice_tokens(
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
        let runtime = self
            .teacher_runtime
            .inner
            .lock()
            .expect("teacher model runtime lock poisoned");
        if let Some(runtime) = runtime
            .as_ref()
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
        let mut runtime = self
            .teacher_runtime
            .inner
            .lock()
            .expect("teacher model runtime lock poisoned");
        if runtime.is_none() {
            *runtime = Some(Box::new(TeacherModelRuntime::new(
                self.teacher_model
                    .clone()
                    .unwrap_or_else(|| self.model.clone()),
            )));
        }
        let runtime = runtime
            .as_mut()
            .and_then(|runtime| runtime.downcast_mut::<TeacherModelRuntime<B>>())
            .expect("teacher runtime backend type must match learner backend");
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

    pub(crate) fn validation_loss_and_output_degeneracy_with_subnetwork_masks(
        &self,
        batch: SequenceBatch<B>,
        neuron_mask: Tensor<B, 4>,
        activity_mask: Tensor<B, 4>,
        probe_tokens: usize,
        eos_id: Option<i64>,
    ) -> (Tensor<B, 1>, Option<OutputDegeneracyStats>)
    where
        B::Device: 'static,
        B::FloatTensorPrimitive: 'static,
    {
        let logits = self
            .model
            .predictive_coding_forward_with_subnetwork_masks(
                batch.inputs.clone(),
                neuron_mask.clone(),
                activity_mask.clone(),
            )
            .expect("validated predictive context masks");
        let loss = masked_token_mean(
            self.model
                .language_token_losses_from_logits(logits, batch.targets.clone()),
            batch.loss_mask.clone(),
        );
        let stats = self.output_degeneracy_for_batch_with_subnetwork_masks(
            batch,
            probe_tokens,
            eos_id,
            neuron_mask,
            activity_mask,
        );
        (loss, stats)
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
        self.output_degeneracy_for_batch_impl(batch, probe_tokens, eos_id, None)
    }

    fn output_degeneracy_for_batch_with_subnetwork_masks(
        &self,
        batch: SequenceBatch<B>,
        probe_tokens: usize,
        eos_id: Option<i64>,
        neuron_mask: Tensor<B, 4>,
        activity_mask: Tensor<B, 4>,
    ) -> Option<OutputDegeneracyStats>
    where
        B::Device: 'static,
        B::FloatTensorPrimitive: 'static,
    {
        self.output_degeneracy_for_batch_impl(
            batch,
            probe_tokens,
            eos_id,
            Some((neuron_mask, activity_mask)),
        )
    }

    fn output_degeneracy_for_batch_impl(
        &self,
        batch: SequenceBatch<B>,
        probe_tokens: usize,
        eos_id: Option<i64>,
        context_masks: Option<(Tensor<B, 4>, Tensor<B, 4>)>,
    ) -> Option<OutputDegeneracyStats>
    where
        B::Device: 'static,
        B::FloatTensorPrimitive: 'static,
    {
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
            let logits = if let Some((neuron_mask, activity_mask)) = context_masks.as_ref() {
                self.model
                    .predictive_coding_forward_with_subnetwork_masks_and_state(
                        inputs,
                        neuron_mask.clone(),
                        activity_mask.clone(),
                        &mut state,
                    )
                    .expect("validated predictive context masks")
            } else if let Some(mask) = summary_event_mask {
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
                let logits = match context_masks.as_ref() {
                    Some((neuron_mask, activity_mask)) => self
                        .model
                        .predictive_coding_forward_with_subnetwork_masks_and_state(
                            next_tensor,
                            neuron_mask.clone(),
                            activity_mask.clone(),
                            &mut state,
                        )
                        .expect("validated predictive context masks"),
                    None => self.model.forward_with_state(next_tensor, &mut state),
                };
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
        self.teacher_runtime
            .inner
            .lock()
            .expect("teacher model runtime lock poisoned")
            .as_ref()
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

    fn predictive_coding_active_for_chunk(
        &self,
        step_index: usize,
        chunk_index: usize,
        chunks_per_step: usize,
    ) -> bool {
        if !self.predictive_coding.enabled || step_index < self.predictive_coding.warmup_steps {
            return false;
        }
        predictive_coding_chunk_due(
            self.predictive_coding.observation_contract,
            step_index,
            chunk_index,
            chunks_per_step,
            self.predictive_coding.apply_every_chunks,
        )
    }

    fn predictive_coding_inference_config(&self) -> burn_pc::PcInferenceConfig {
        self.predictive_coding.inference_config()
    }

    fn predictive_coding_state_has_latents(
        state: &ModelState<B>,
        scope: PredictiveCodingStateScope,
    ) -> bool {
        let mut state = state.clone();
        let mut mapper = PredictiveCodingPresenceMapper::default();
        map_predictive_coding_state(&mut state, scope, &mut mapper);
        mapper.present
    }

    fn attach_predictive_coding_state_latents(
        state: &mut ModelState<B>,
        scope: PredictiveCodingStateScope,
    ) -> bool {
        let mut mapper = PredictiveCodingAttachMapper::default();
        map_predictive_coding_state(state, scope, &mut mapper);
        mapper.attached
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
        let mut mapper = PredictiveCodingUpdateMapper::<B> {
            grads,
            config,
            sync_diagnostics,
            stats: PredictiveCodingTensorUpdateStats::default(),
        };
        map_predictive_coding_state(state, scope, &mut mapper);
        mapper.stats
    }

    fn predictive_coding_oracle_energy_with_state(
        &self,
        inference_model: &DragonModel<B>,
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
        loss_mask: Option<Tensor<B, 2, Int>>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
        state: &mut ModelState<B>,
    ) -> Tensor<B, 1> {
        let hidden = if let Some(mask) = summary_event_mask {
            inference_model.forward_hidden_with_state_and_summary_event_mask(inputs, mask, state)
        } else {
            inference_model.forward_hidden_with_state(inputs, state)
        };
        self.language_loss_from_hidden(hidden, targets, loss_mask)
    }

    fn predictive_coding_amortization_constraint(
        &self,
        student: &ModelState<B>,
        teacher: &ModelState<B>,
    ) -> (Option<Tensor<B, 1>>, usize) {
        let mut total = None;
        let mut components = 0usize;
        let step_index = self.gradient_scale_step.load(Ordering::Relaxed);
        let constraint = PredictiveCodingAmortizationConstraint {
            sample_axis: 2,
            max_slots: self.predictive_coding.amortization_max_state_slots.max(1),
            sample_offset: stochastic_step_seed(
                self.stochastic_seed,
                step_index,
                STOCHASTIC_STREAM_PC_AMORTIZATION,
            ) as usize,
            tolerance: self.predictive_coding.amortization_tolerance.max(0.0),
            eps: self.predictive_coding.eps.max(1.0e-12),
        };
        let scope = self.predictive_coding.state_scope;
        let student = predictive_coding_state_snapshot(student, scope);
        let teacher = predictive_coding_state_snapshot(teacher, scope);
        let mut sample_indices = PredictiveCodingSampleIndexCache::new();
        debug_assert_eq!(student.rank3.len(), teacher.rank3.len());
        debug_assert_eq!(student.rank4.len(), teacher.rank4.len());
        for ((student_name, student), (teacher_name, teacher)) in
            student.rank3.iter().zip(&teacher.rank3)
        {
            debug_assert_eq!(student_name, teacher_name);
            accumulate_predictive_coding_amortization_constraint(
                &mut total,
                &mut components,
                student,
                teacher,
                constraint,
                &mut sample_indices,
            );
        }
        for ((student_name, student), (teacher_name, teacher)) in
            student.rank4.iter().zip(&teacher.rank4)
        {
            debug_assert_eq!(student_name, teacher_name);
            accumulate_predictive_coding_amortization_constraint(
                &mut total,
                &mut components,
                student,
                teacher,
                constraint,
                &mut sample_indices,
            );
        }
        (
            total.map(|loss| loss.div_scalar(components.max(1) as f32)),
            components,
        )
    }

    fn correct_state_with_oracle_predictive_coding_using_model(
        &self,
        inference_model: &DragonModel<B>,
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
            let energy = self.predictive_coding_oracle_energy_with_state(
                inference_model,
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
            update_stats.clip_fraction_sum += step_stats.clip_fraction_sum;
            report.inference_steps = report.inference_steps.saturating_add(1);
            corrected.detach_in_place();
        }

        if report.inference_steps > 0 {
            if sync_diagnostics {
                let mut post_state = corrected.clone();
                let post_energy = self.predictive_coding_oracle_energy_with_state(
                    inference_model,
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
            report.clip_fraction_mean = update_stats.clip_fraction_mean();
        }
        report.elapsed_ns = start.elapsed().as_nanos();
        (corrected, report)
    }

    fn correct_state_with_oracle_predictive_coding(
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
        let inference_model = detach_teacher_model(&self.model);
        self.correct_state_with_oracle_predictive_coding_using_model(
            &inference_model,
            state,
            inputs,
            targets,
            loss_mask,
            summary_event_mask,
        )
    }

    fn replay_observed_prefix(
        &self,
        inference_model: &DragonModel<B>,
        mut state: ModelState<B>,
        observed_inputs: Tensor<B, 2, Int>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> ModelState<B> {
        if let Some(mask) = summary_event_mask {
            inference_model.forward_hidden_with_state_and_summary_event_mask(
                observed_inputs,
                mask,
                &mut state,
            );
        } else {
            inference_model.forward_hidden_with_state(observed_inputs, &mut state);
        }
        state.detach_in_place();
        state
    }

    /// Corrects the state entering an already-observed token span, then replays
    /// that span to produce state for subsequent predictions. No next-token
    /// targets are accepted by this API.
    fn correct_state_from_observed_prefix_using_model(
        &self,
        inference_model: &DragonModel<B>,
        state: ModelState<B>,
        observed_inputs: Tensor<B, 2, Int>,
        observed_loss_mask: Option<Tensor<B, 2, Int>>,
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
            let replayed = self.replay_observed_prefix(
                inference_model,
                state,
                observed_inputs,
                summary_event_mask,
            );
            report.elapsed_ns = start.elapsed().as_nanos();
            return (replayed, report);
        }
        let [batch_size, observed_length] = observed_inputs.shape().dims();
        if observed_length < 2 {
            report.skipped_empty_state = 1;
            let replayed = self.replay_observed_prefix(
                inference_model,
                state,
                observed_inputs,
                summary_event_mask,
            );
            report.elapsed_ns = start.elapsed().as_nanos();
            return (replayed, report);
        }

        let energy_inputs =
            Self::slice_tokens(observed_inputs.clone(), batch_size, 0, observed_length - 1);
        let energy_targets =
            Self::slice_tokens(observed_inputs.clone(), batch_size, 1, observed_length);
        let energy_loss_mask = observed_loss_mask
            .clone()
            .map(|mask| Self::slice_tokens(mask, batch_size, 0, observed_length - 1));
        let energy_summary_mask = summary_event_mask
            .clone()
            .map(|mask| Self::slice_tokens(mask, batch_size, 0, observed_length - 1));

        let config = self.predictive_coding_inference_config();
        let sync_diagnostics = self.predictive_coding.sync_diagnostics;
        let mut corrected_entry = state.detached_clone();
        let mut update_stats = PredictiveCodingTensorUpdateStats::default();
        for step in 0..config.steps {
            if !Self::attach_predictive_coding_state_latents(&mut corrected_entry, state_scope) {
                report.skipped_empty_state = report.skipped_empty_state.saturating_add(1);
                break;
            }
            let mut inference_state = corrected_entry.clone();
            let hidden = if let Some(mask) = energy_summary_mask.clone() {
                inference_model.forward_hidden_with_state_and_summary_event_mask(
                    energy_inputs.clone(),
                    mask,
                    &mut inference_state,
                )
            } else {
                inference_model
                    .forward_hidden_with_state(energy_inputs.clone(), &mut inference_state)
            };
            let energy = self.language_loss_from_hidden(
                hidden,
                energy_targets.clone(),
                energy_loss_mask.clone(),
            );
            if sync_diagnostics && step == 0 {
                report.energy_before = Some(scalar_tensor_to_f64(energy.clone().detach().inner()));
            }
            let grads = energy.backward();
            let step_stats = Self::update_predictive_coding_state_latents(
                &mut corrected_entry,
                &grads,
                &config,
                sync_diagnostics,
                state_scope,
            );
            if step_stats.tensor_count == 0 {
                report.skipped_empty_state = report.skipped_empty_state.saturating_add(1);
                corrected_entry.detach_in_place();
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
            update_stats.clip_fraction_sum += step_stats.clip_fraction_sum;
            report.inference_steps = report.inference_steps.saturating_add(1);
            corrected_entry.detach_in_place();
        }

        if report.inference_steps == 0 {
            let replayed = self.replay_observed_prefix(
                inference_model,
                state,
                observed_inputs,
                summary_event_mask,
            );
            report.elapsed_ns = start.elapsed().as_nanos();
            return (replayed, report);
        }
        if sync_diagnostics {
            let mut post_state = corrected_entry.clone();
            let hidden = if let Some(mask) = energy_summary_mask {
                inference_model.forward_hidden_with_state_and_summary_event_mask(
                    energy_inputs,
                    mask,
                    &mut post_state,
                )
            } else {
                inference_model.forward_hidden_with_state(energy_inputs, &mut post_state)
            };
            let post_energy =
                self.language_loss_from_hidden(hidden, energy_targets, energy_loss_mask);
            report.energy_after = Some(scalar_tensor_to_f64(post_energy.detach().inner()));
        }

        let replayed = self.replay_observed_prefix(
            inference_model,
            corrected_entry,
            observed_inputs,
            summary_event_mask,
        );
        report.chunks_corrected = 1;
        report.grad_norm_mean = update_stats.grad_norm_mean();
        report.grad_norm_max = update_stats.grad_norm_max();
        report.delta_rms_mean = update_stats.delta_rms_mean();
        report.clip_fraction_mean = update_stats.clip_fraction_mean();
        report.elapsed_ns = start.elapsed().as_nanos();
        (replayed, report)
    }

    fn correct_state_from_observed_prefix(
        &self,
        state: ModelState<B>,
        observed_inputs: Tensor<B, 2, Int>,
        observed_loss_mask: Option<Tensor<B, 2, Int>>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> (ModelState<B>, PredictiveCodingChunkReport)
    where
        B: AutodiffBackend,
    {
        let inference_model = detach_teacher_model(&self.model);
        self.correct_state_from_observed_prefix_using_model(
            &inference_model,
            state,
            observed_inputs,
            observed_loss_mask,
            summary_event_mask,
        )
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

    fn ruliad_structured_answer_recovery_weight(&self) -> f32 {
        let config = self.ruliad_supervision.answer_denoising;
        if !config.enabled
            || config.structured_recovery_weight <= f32::EPSILON
            || config.structured_recovery_every_steps == 0
        {
            return 0.0;
        }
        let step_index = self.gradient_scale_step.load(Ordering::Relaxed);
        if step_index < config.structured_recovery_start_after_steps {
            return 0.0;
        }
        if !step_index.is_multiple_of(config.structured_recovery_every_steps) {
            return 0.0;
        }
        config.structured_recovery_weight
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

    fn ruliad_answer_contract_weight(&self) -> f32 {
        let config = self.ruliad_supervision.answer_contract;
        if !config.enabled || config.weight <= f32::EPSILON || config.every_steps == 0 {
            return 0.0;
        }
        let step_index = self.gradient_scale_step.load(Ordering::Relaxed);
        if step_index < config.start_after_steps {
            return 0.0;
        }
        if !step_index.is_multiple_of(config.every_steps) {
            return 0.0;
        }
        config.weight
    }

    fn ruliad_answer_contract_loss(
        &self,
        policy_batch: &crate::dataset::RuliadPolicyBatch,
        device: &B::Device,
        block_size: usize,
    ) -> Option<Tensor<B, 1>> {
        let config = self.ruliad_supervision.answer_contract;
        let weight = self.ruliad_answer_contract_weight();
        if weight <= f32::EPSILON || policy_batch.samples.is_empty() || self.pipeline_enabled() {
            return None;
        }
        let tokenizer =
            burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer::from_config(
                &policy_batch.tokenization,
            )
            .ok()?;
        let completion_budget = config
            .max_completion_tokens
            .max(1)
            .min(block_size.saturating_sub(1).max(1));
        let max_rows = config.max_rows_per_step.max(1);
        let prompt_schema_max_rows = if config.prompt_schema_max_rows_per_step == 0 {
            max_rows
        } else {
            config.prompt_schema_max_rows_per_step
        }
        .max(1);

        #[derive(Clone)]
        struct ContractRow {
            inputs: Vec<i64>,
            targets: Vec<i64>,
            mask: Vec<f32>,
            premature_close_mask: Vec<f32>,
        }

        // A sequence terminator may only be penalized as one event when the
        // tokenizer represents it with one structural token. Penalizing each
        // byte in `[/R*]` independently suppresses common answer characters.
        let close_token_ids = policy_batch.stop_token_id.into_iter().collect::<Vec<_>>();
        let premature_close_weight = config.premature_close_unlikelihood_weight;
        let mut rows = Vec::<ContractRow>::new();
        let mut sample_groups = 0usize;
        let mut prompt_schema_sample_groups = 0usize;
        let mut contract_tokens = 0usize;
        let mut prompt_schema_value_tokens = 0usize;
        let mut prompt_schema_rows = 0usize;
        let mut schema_tokens = 0usize;
        let mut schema_start_tokens = 0usize;
        let mut value_tokens = 0usize;
        let mut other_tokens = 0usize;
        let mut premature_close_tokens = 0usize;
        for sample in policy_batch.samples.iter() {
            if rows.len() >= max_rows {
                break;
            }
            let mut prompt = sample.prompt_tokens.clone();
            if prompt.is_empty() || sample.item.expected_answer.trim().is_empty() {
                continue;
            }
            let Some((oracle_completion, _oracle_text, _truncated)) =
                Self::ruliad_oracle_completion_tokens(&tokenizer, sample, completion_budget)
            else {
                continue;
            };
            prompt = Self::ruliad_trim_prompt_for_completion(
                &prompt,
                oracle_completion.len(),
                block_size,
            );
            let Some((inputs, targets, _default_mask)) =
                Self::ruliad_policy_row_from_completion(&prompt, &oracle_completion)
            else {
                continue;
            };
            let completion_start = prompt.len().saturating_sub(1).min(targets.len());
            let schema_mask = Self::ruliad_answer_schema_completion_mask(
                &tokenizer,
                &sample.item.expected_answer,
                oracle_completion.len(),
            );
            let schema_start_mask = Self::ruliad_answer_schema_start_completion_mask(
                &tokenizer,
                &sample.item.expected_answer,
                oracle_completion.len(),
            );
            let value_mask = Self::ruliad_answer_value_completion_mask(
                &tokenizer,
                &sample.item.expected_answer,
                oracle_completion.len(),
            );
            let mut mask = vec![0.0f32; targets.len()];
            let mut premature_close_mask = vec![0.0f32; targets.len()];
            let mut active_tokens = 0usize;
            for completion_index in 0..oracle_completion.len() {
                let target_index = completion_start.saturating_add(completion_index);
                if target_index >= mask.len() {
                    continue;
                }
                let schema_token = schema_mask.get(completion_index).copied().unwrap_or(false);
                let schema_start_token = schema_start_mask
                    .get(completion_index)
                    .copied()
                    .unwrap_or(false);
                let token_weight = if schema_token {
                    schema_tokens = schema_tokens.saturating_add(1);
                    if schema_start_token {
                        schema_start_tokens = schema_start_tokens.saturating_add(1);
                        config
                            .schema_token_weight
                            .max(config.schema_start_token_weight)
                    } else {
                        config.schema_token_weight
                    }
                } else if value_mask.get(completion_index).copied().unwrap_or(false) {
                    value_tokens = value_tokens.saturating_add(1);
                    config.value_token_weight
                } else {
                    other_tokens = other_tokens.saturating_add(1);
                    config.other_token_weight
                };
                if token_weight > f32::EPSILON {
                    mask[target_index] = token_weight;
                    active_tokens = active_tokens.saturating_add(1);
                }
            }
            if premature_close_weight > f32::EPSILON && !close_token_ids.is_empty() {
                let answer_token_len = tokenizer
                    .encode_payload(sample.item.expected_answer.trim())
                    .len()
                    .min(oracle_completion.len());
                for completion_index in 0..answer_token_len {
                    let target_index = completion_start.saturating_add(completion_index);
                    if let Some(slot) = premature_close_mask.get_mut(target_index)
                        && *slot <= f32::EPSILON
                    {
                        *slot = 1.0;
                        premature_close_tokens = premature_close_tokens.saturating_add(1);
                    }
                }
            }
            if active_tokens == 0 {
                continue;
            }
            contract_tokens = contract_tokens.saturating_add(active_tokens);
            sample_groups = sample_groups.saturating_add(1);
            rows.push(ContractRow {
                inputs,
                targets,
                mask,
                premature_close_mask,
            });
        }
        let oracle_rows = rows.len();
        if config.prompt_schema_value_weight > f32::EPSILON {
            let field_rows_by_sample = policy_batch
                .samples
                .iter()
                .filter_map(|sample| {
                    let prompt = sample.prompt_tokens.clone();
                    if prompt.is_empty() || sample.item.expected_answer.trim().is_empty() {
                        return None;
                    }
                    let field_rows = Self::ruliad_prompt_schema_value_completion_rows(
                        &tokenizer,
                        &prompt,
                        &sample.item.expected_answer,
                        sample.item.document_close_marker(),
                        completion_budget,
                        block_size,
                        prompt_schema_max_rows,
                    );
                    (!field_rows.is_empty()).then_some(field_rows)
                })
                .collect::<Vec<_>>();
            let selected_rows =
                take_rows_round_robin(&field_rows_by_sample, prompt_schema_max_rows);
            prompt_schema_sample_groups = selected_rows
                .iter()
                .map(|(sample_index, _)| *sample_index)
                .collect::<HashSet<_>>()
                .len();
            for (_sample_index, (inputs, targets, mask, active_tokens)) in selected_rows {
                if active_tokens == 0 {
                    continue;
                }
                let mask = mask
                    .into_iter()
                    .map(|value| {
                        if value > f32::EPSILON {
                            config.prompt_schema_value_weight
                        } else {
                            0.0
                        }
                    })
                    .collect::<Vec<_>>();
                let premature_close_mask = vec![0.0f32; targets.len()];
                contract_tokens = contract_tokens.saturating_add(active_tokens);
                prompt_schema_value_tokens =
                    prompt_schema_value_tokens.saturating_add(active_tokens);
                prompt_schema_rows = prompt_schema_rows.saturating_add(1);
                rows.push(ContractRow {
                    inputs,
                    targets,
                    mask,
                    premature_close_mask,
                });
            }
        }
        let skip_reason = rows
            .is_empty()
            .then(|| "no_answer_contract_rows".to_string());
        self.write_ruliad_answer_contract_telemetry(RuliadAnswerContractTelemetry {
            version: 1,
            step_index: self.gradient_scale_step.load(Ordering::Relaxed),
            policy_batch_present: true,
            skip_reason,
            sample_groups,
            prompt_schema_sample_groups,
            oracle_rows,
            prompt_schema_rows,
            contract_tokens,
            prompt_schema_value_tokens,
            schema_tokens,
            schema_start_tokens,
            value_tokens,
            other_tokens,
            premature_close_tokens,
            answer_contract_weight: weight,
            premature_close_unlikelihood_weight: premature_close_weight,
            max_completion_tokens: completion_budget,
            max_rows_per_step: max_rows,
            prompt_schema_max_rows_per_step: prompt_schema_max_rows,
        });
        if rows.is_empty() {
            return None;
        }

        let max_len = rows.iter().map(|row| row.inputs.len()).max()?.max(1);
        let row_count = rows.len();
        let mut input_values = vec![0i64; row_count * max_len];
        let mut target_values = vec![0i64; row_count * max_len];
        let mut mask_values = vec![0.0f32; row_count * max_len];
        let mut premature_close_mask_values = vec![0.0f32; row_count * max_len];
        for (row_index, row) in rows.iter().enumerate() {
            let offset = row_index * max_len;
            let len = row.inputs.len().min(max_len);
            input_values[offset..offset + len].copy_from_slice(&row.inputs[..len]);
            target_values[offset..offset + len].copy_from_slice(&row.targets[..len]);
            mask_values[offset..offset + len].copy_from_slice(&row.mask[..len]);
            premature_close_mask_values[offset..offset + len]
                .copy_from_slice(&row.premature_close_mask[..len]);
        }
        let inputs = Tensor::<B, 2, Int>::from_data(
            TensorData::new(input_values, [row_count, max_len]),
            device,
        );
        let targets = Tensor::<B, 2, Int>::from_data(
            TensorData::new(target_values, [row_count, max_len]),
            device,
        );
        let mask =
            Tensor::<B, 2>::from_data(TensorData::new(mask_values, [row_count, max_len]), device);
        let logits = self.model.forward(inputs);
        let log_probs = log_probs_from_logits(logits);
        let token_log_probs = selected_token_log_probs(log_probs.clone(), targets);
        let active = mask.clone().sum().reshape([1]).clamp_min(1.0);
        let mut loss = (token_log_probs * mask)
            .sum()
            .reshape([1])
            .div(active)
            .mul_scalar(-weight);
        if premature_close_weight > f32::EPSILON
            && premature_close_tokens > 0
            && !close_token_ids.is_empty()
        {
            let close_mask = Tensor::<B, 2>::from_data(
                TensorData::new(premature_close_mask_values, [row_count, max_len]),
                device,
            );
            let close_active = close_mask.clone().sum().reshape([1]).clamp_min(1.0);
            let mut close_loss: Option<Tensor<B, 1>> = None;
            let close_token_count = close_token_ids.len().max(1) as f32;
            for close_token_id in close_token_ids {
                let close_targets = Tensor::<B, 2, Int>::from_data(
                    TensorData::new(
                        vec![close_token_id; row_count * max_len],
                        [row_count, max_len],
                    ),
                    device,
                );
                let token_loss =
                    unlikelihood_from_log_probs(log_probs.clone(), close_targets, 1.0e-6);
                let masked = (token_loss * close_mask.clone())
                    .sum()
                    .reshape([1])
                    .div(close_active.clone());
                close_loss = Some(match close_loss {
                    Some(accumulated) => accumulated + masked,
                    None => masked,
                });
            }
            if let Some(close_loss) = close_loss {
                loss = loss
                    + close_loss
                        .div_scalar(close_token_count)
                        .mul_scalar(premature_close_weight);
            }
        }
        Some(loss)
    }

    fn ruliad_answer_contract_auxiliary_loss(
        &self,
        policy_batch: Option<&crate::dataset::RuliadPolicyBatch>,
        device: &B::Device,
        block_size: usize,
    ) -> Option<Tensor<B, 1>> {
        let contract_weight = self.ruliad_answer_contract_weight();
        if contract_weight <= f32::EPSILON {
            return None;
        }
        if let Some(policy_batch) = policy_batch {
            self.ruliad_answer_contract_loss(policy_batch, device, block_size)
        } else {
            self.write_ruliad_answer_contract_skip("missing_policy_batch", contract_weight);
            None
        }
    }

    fn ruliad_structured_answer_recovery_loss(
        &self,
        policy_batch: &crate::dataset::RuliadPolicyBatch,
        device: &B::Device,
        block_size: usize,
    ) -> Option<Tensor<B, 1>> {
        let config = self.ruliad_supervision.answer_denoising;
        let weight = self.ruliad_structured_answer_recovery_weight();
        if weight <= f32::EPSILON || policy_batch.samples.is_empty() || self.pipeline_enabled() {
            return None;
        }
        let tokenizer =
            burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer::from_config(
                &policy_batch.tokenization,
            )
            .ok()?;
        let completion_budget = config
            .structured_recovery_max_completion_tokens
            .max(1)
            .min(block_size.saturating_sub(1).max(1));
        let prompt_budget = block_size.saturating_sub(completion_budget).max(1);

        #[derive(Clone)]
        struct RecoveryRow {
            inputs: Vec<i64>,
            targets: Vec<i64>,
            mask: Vec<i64>,
        }

        let mut rows = Vec::<RecoveryRow>::new();
        let mut sample_groups = 0usize;
        let mut field_negative_recovery_rows = 0usize;
        let mut template_negative_recovery_rows = 0usize;
        let mut schema_negative_recovery_rows = 0usize;
        for sample in policy_batch.samples.iter() {
            let mut prompt = sample.prompt_tokens.clone();
            if prompt.is_empty() {
                continue;
            }
            if prompt.len() > prompt_budget {
                prompt = prompt[prompt.len() - prompt_budget..].to_vec();
            }
            let Some((oracle_completion, _oracle_text, _truncated)) =
                Self::ruliad_oracle_completion_tokens(&tokenizer, sample, completion_budget)
            else {
                continue;
            };
            let Some((oracle_inputs, oracle_targets, oracle_mask)) =
                Self::ruliad_policy_row_from_completion(&prompt, &oracle_completion)
            else {
                continue;
            };
            let completion_start = prompt.len().saturating_sub(1).min(oracle_inputs.len());
            let mut sample_rows = 0usize;
            for (negative, negative_kind) in Self::ruliad_structured_negative_answers_with_schema(
                &sample.item.expected_answer,
                config.structured_recovery_negative_count,
                config.structured_recovery_template_negative_count,
                config.structured_recovery_schema_negative_count,
            ) {
                let Some((negative_completion, _negative_text)) =
                    Self::ruliad_completion_tokens_from_answer(
                        &tokenizer,
                        &negative,
                        sample.item.document_close_marker(),
                        completion_budget,
                    )
                else {
                    continue;
                };
                let mut corrupted_inputs = oracle_inputs.clone();
                for (index, value) in corrupted_inputs
                    .iter_mut()
                    .enumerate()
                    .skip(completion_start)
                {
                    let negative_index = index - completion_start;
                    if let Some(negative_token) = negative_completion.get(negative_index) {
                        *value = *negative_token;
                    }
                }
                rows.push(RecoveryRow {
                    inputs: corrupted_inputs,
                    targets: oracle_targets.clone(),
                    mask: oracle_mask
                        .iter()
                        .map(|value| if *value > 0.0 { 1 } else { 0 })
                        .collect(),
                });
                sample_rows = sample_rows.saturating_add(1);
                match negative_kind {
                    RuliadStructuredNegativeKind::FieldMutation => {
                        field_negative_recovery_rows =
                            field_negative_recovery_rows.saturating_add(1);
                    }
                    RuliadStructuredNegativeKind::TemplateCollapse => {
                        template_negative_recovery_rows =
                            template_negative_recovery_rows.saturating_add(1);
                    }
                    RuliadStructuredNegativeKind::SchemaCollapse => {
                        schema_negative_recovery_rows =
                            schema_negative_recovery_rows.saturating_add(1);
                    }
                }
            }
            if sample_rows > 0 {
                sample_groups = sample_groups.saturating_add(1);
            }
        }
        self.write_ruliad_structured_recovery_telemetry(RuliadStructuredRecoveryTelemetry {
            version: 1,
            step_index: self.gradient_scale_step.load(Ordering::Relaxed),
            policy_batch_present: true,
            skip_reason: None,
            sample_groups,
            recovery_rows: rows.len(),
            field_negative_recovery_rows,
            template_negative_recovery_rows,
            schema_negative_recovery_rows,
            structured_recovery_weight: weight,
            structured_recovery_max_completion_tokens: completion_budget,
        });
        if rows.is_empty() {
            return None;
        }

        let max_len = rows.iter().map(|row| row.inputs.len()).max()?.max(1);
        let row_count = rows.len();
        let mut input_values = vec![0i64; row_count * max_len];
        let mut target_values = vec![0i64; row_count * max_len];
        let mut mask_values = vec![0i64; row_count * max_len];
        for (row_index, row) in rows.into_iter().enumerate() {
            let offset = row_index * max_len;
            let len = row.inputs.len().min(max_len);
            input_values[offset..offset + len].copy_from_slice(&row.inputs[..len]);
            target_values[offset..offset + len].copy_from_slice(&row.targets[..len]);
            mask_values[offset..offset + len].copy_from_slice(&row.mask[..len]);
        }
        let inputs = Tensor::<B, 2, Int>::from_data(
            TensorData::new(input_values, [row_count, max_len]),
            device,
        );
        let targets = Tensor::<B, 2, Int>::from_data(
            TensorData::new(target_values, [row_count, max_len]),
            device,
        );
        let mask = Tensor::<B, 2, Int>::from_data(
            TensorData::new(mask_values, [row_count, max_len]),
            device,
        );
        let hidden = self.model.forward_hidden(inputs);
        Some(
            self.language_loss_from_hidden(hidden, targets, Some(mask))
                .mul_scalar(weight),
        )
    }

    fn ruliad_structured_answer_recovery_auxiliary_loss(
        &self,
        policy_batch: Option<&crate::dataset::RuliadPolicyBatch>,
        device: &B::Device,
        block_size: usize,
    ) -> Option<Tensor<B, 1>> {
        let recovery_weight = self.ruliad_structured_answer_recovery_weight();
        if recovery_weight <= f32::EPSILON {
            return None;
        }
        if let Some(policy_batch) = policy_batch {
            self.ruliad_structured_answer_recovery_loss(policy_batch, device, block_size)
        } else {
            self.write_ruliad_structured_recovery_skip("missing_policy_batch", recovery_weight);
            None
        }
    }

    fn write_ruliad_structured_recovery_skip(&self, reason: &str, weight: f32) {
        self.write_ruliad_structured_recovery_telemetry(RuliadStructuredRecoveryTelemetry {
            version: 1,
            step_index: self.gradient_scale_step.load(Ordering::Relaxed),
            policy_batch_present: false,
            skip_reason: Some(reason.to_string()),
            sample_groups: 0,
            recovery_rows: 0,
            field_negative_recovery_rows: 0,
            template_negative_recovery_rows: 0,
            schema_negative_recovery_rows: 0,
            structured_recovery_weight: weight,
            structured_recovery_max_completion_tokens: self
                .ruliad_supervision
                .answer_denoising
                .structured_recovery_max_completion_tokens,
        });
    }

    fn ruliad_verifier_reward_weight(&self) -> f32 {
        let config = self.ruliad_supervision.verifier_reward;
        if !config.enabled || config.weight <= f32::EPSILON || config.every_steps == 0 {
            return 0.0;
        }
        let step_index = self.gradient_scale_step.load(Ordering::Relaxed);
        if step_index < config.start_after_steps {
            return 0.0;
        }
        if !step_index.is_multiple_of(config.every_steps) {
            return 0.0;
        }
        config.weight
    }

    fn ruliad_structured_contrast_weight(&self) -> f32 {
        let config = self.ruliad_supervision.verifier_reward;
        if !config.enabled
            || config.structured_contrast_weight <= f32::EPSILON
            || config.structured_contrast_every_steps == 0
        {
            return 0.0;
        }
        let step_index = self.gradient_scale_step.load(Ordering::Relaxed);
        if step_index < config.structured_contrast_start_after_steps {
            return 0.0;
        }
        if !step_index.is_multiple_of(config.structured_contrast_every_steps) {
            return 0.0;
        }
        config.structured_contrast_weight
    }

    fn ruliad_field_binding_contrast_weight(&self) -> f32 {
        let config = self.ruliad_supervision.verifier_reward;
        if !config.enabled
            || config.field_binding_contrast_weight <= f32::EPSILON
            || config.field_binding_contrast_every_steps == 0
        {
            return 0.0;
        }
        let step_index = self.gradient_scale_step.load(Ordering::Relaxed);
        if step_index < config.field_binding_contrast_start_after_steps {
            return 0.0;
        }
        if !step_index.is_multiple_of(config.field_binding_contrast_every_steps) {
            return 0.0;
        }
        config.field_binding_contrast_weight
    }

    fn ruliad_verifier_rollout_feedback_active(&self) -> bool {
        let config = self.ruliad_supervision.verifier_reward;
        if !config.enabled
            || (config.rollout_imitation_weight <= f32::EPSILON
                && config.rollout_recovery_weight <= f32::EPSILON)
            || config.rollout_imitation_every_steps == 0
        {
            return false;
        }
        let step_index = self.gradient_scale_step.load(Ordering::Relaxed);
        if step_index < config.rollout_imitation_start_after_steps {
            return false;
        }
        if !step_index.is_multiple_of(config.rollout_imitation_every_steps) {
            return false;
        }
        true
    }

    fn ruliad_proof_policy_dagger_weight(&self) -> f32 {
        let config = self.ruliad_supervision.proof_policy;
        if !config.enabled || config.weight <= f32::EPSILON || config.every_steps == 0 {
            return 0.0;
        }
        let step_index = self.gradient_scale_step.load(Ordering::Relaxed);
        if step_index < config.start_after_steps || !step_index.is_multiple_of(config.every_steps) {
            return 0.0;
        }
        config.weight
    }

    fn mix_ruliad_policy_seed(mut value: u64) -> u64 {
        value ^= value >> 30;
        value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value ^= value >> 27;
        value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn ruliad_vpo_scalarizations(
        &self,
        sample_index: usize,
        count: usize,
        config: crate::config::train::RuliadVerifierRewardConfig,
    ) -> Vec<[f32; burn_dragon_universality::ruliad::RULIAD_VERIFIER_REWARD_VECTOR_DIM]> {
        let step_index = self.gradient_scale_step.load(Ordering::Relaxed) as u64;
        let seed = Self::mix_ruliad_policy_seed(
            step_index
                ^ (sample_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
                ^ (count as u64).wrapping_mul(0xd1b5_4a32_d192_ed03),
        );
        let mut rng = StdRng::seed_from_u64(seed);
        let mut scalarizations = Vec::with_capacity(count);
        for _ in 0..count {
            let mut weights =
                [0.0f32; burn_dragon_universality::ruliad::RULIAD_VERIFIER_REWARD_VECTOR_DIM];
            let mut sum = 0.0f32;
            for weight in weights.iter_mut() {
                let draw = -rng.gen_range(f32::MIN_POSITIVE..1.0).ln();
                *weight = draw;
                sum += draw;
            }
            if !sum.is_finite() || sum <= f32::EPSILON {
                let uniform = 1.0 / weights.len() as f32;
                weights.fill(uniform);
            } else {
                for weight in weights.iter_mut() {
                    *weight /= sum;
                }
            }
            Self::constrain_ruliad_vpo_scalarization(&mut weights, config);
            scalarizations.push(weights);
        }
        scalarizations
    }

    fn constrain_ruliad_vpo_scalarization(
        weights: &mut [f32; burn_dragon_universality::ruliad::RULIAD_VERIFIER_REWARD_VECTOR_DIM],
        config: crate::config::train::RuliadVerifierRewardConfig,
    ) {
        const CORRECTNESS_AXES: &[usize] = &[0, 1, 2, 3, 4];
        const SCHEMA_QUALITY_AXES: &[usize] = &[6];
        const HEALTH_AXES: &[usize] = &[8, 9];
        const COMPACTNESS_AXIS: usize = 5;
        let original = *weights;
        let correctness_floor = config.vpo_correctness_mass_floor.clamp(0.0, 1.0);
        let schema_floor = config
            .vpo_schema_quality_mass_floor
            .clamp(0.0, 1.0 - correctness_floor);
        let health_floor = config
            .vpo_completion_health_mass_floor
            .clamp(0.0, 1.0 - correctness_floor - schema_floor);
        let residual_mass = (1.0 - correctness_floor - schema_floor - health_floor).max(0.0);
        for (weight, original_weight) in weights.iter_mut().zip(original) {
            *weight = original_weight * residual_mass;
        }
        Self::add_weighted_group_mass(weights, &original, CORRECTNESS_AXES, correctness_floor);
        Self::add_weighted_group_mass(weights, &original, SCHEMA_QUALITY_AXES, schema_floor);
        Self::add_weighted_group_mass(weights, &original, HEALTH_AXES, health_floor);
        let compactness_max = config.vpo_compactness_max_weight.clamp(0.0, 1.0);
        if weights[COMPACTNESS_AXIS] > compactness_max {
            let excess = weights[COMPACTNESS_AXIS] - compactness_max;
            weights[COMPACTNESS_AXIS] = compactness_max;
            Self::add_uniform_mass(weights, CORRECTNESS_AXES, excess * 0.60);
            Self::add_uniform_mass(weights, SCHEMA_QUALITY_AXES, excess * 0.25);
            Self::add_uniform_mass(weights, HEALTH_AXES, excess * 0.15);
        }
        Self::renormalize_scalarization(weights);
    }

    fn add_weighted_group_mass(
        weights: &mut [f32; burn_dragon_universality::ruliad::RULIAD_VERIFIER_REWARD_VECTOR_DIM],
        original: &[f32; burn_dragon_universality::ruliad::RULIAD_VERIFIER_REWARD_VECTOR_DIM],
        axes: &[usize],
        mass: f32,
    ) {
        if axes.is_empty() || mass <= f32::EPSILON {
            return;
        }
        let group_mass = axes.iter().map(|axis| original[*axis]).sum::<f32>();
        if group_mass <= f32::EPSILON {
            Self::add_uniform_mass(weights, axes, mass);
            return;
        }
        for axis in axes {
            weights[*axis] += mass * original[*axis] / group_mass;
        }
    }

    fn add_uniform_mass(
        weights: &mut [f32; burn_dragon_universality::ruliad::RULIAD_VERIFIER_REWARD_VECTOR_DIM],
        axes: &[usize],
        mass: f32,
    ) {
        if axes.is_empty() || mass <= f32::EPSILON {
            return;
        }
        let share = mass / axes.len() as f32;
        for axis in axes {
            weights[*axis] += share;
        }
    }

    fn renormalize_scalarization(
        weights: &mut [f32; burn_dragon_universality::ruliad::RULIAD_VERIFIER_REWARD_VECTOR_DIM],
    ) {
        let sum = weights.iter().copied().sum::<f32>();
        if sum <= f32::EPSILON || !sum.is_finite() {
            let uniform = 1.0 / weights.len() as f32;
            weights.fill(uniform);
            return;
        }
        for weight in weights.iter_mut() {
            *weight = (*weight / sum).max(0.0);
        }
    }

    fn ruliad_vpo_independent_utilities_with_telemetry(
        &self,
        scores: &[burn_dragon_universality::ruliad::RuliadReasoningScore],
        scalarizations: &[
            [f32; burn_dragon_universality::ruliad::RULIAD_VERIFIER_REWARD_VECTOR_DIM]
        ],
        telemetry: &mut RuliadPolicyRewardTelemetryAccumulator,
    ) -> Vec<f32> {
        let mut utilities = vec![0.0f32; scores.len()];
        if scores.is_empty() || scalarizations.is_empty() {
            return utilities;
        }
        let vectors = scores
            .iter()
            .map(burn_dragon_universality::ruliad::ruliad_verifier_reward_vector)
            .collect::<Vec<_>>();
        for weights in scalarizations {
            telemetry.record_vpo_scalarization(weights);
            let mut best_index = 0usize;
            let mut best_value = f32::NEG_INFINITY;
            for (index, vector) in vectors.iter().copied().enumerate() {
                let value = vector.scalarize(weights);
                if value > best_value {
                    best_index = index;
                    best_value = value;
                }
            }
            if best_value.is_finite() {
                utilities[best_index] += best_value;
            }
        }
        let scale = scalarizations.len() as f32;
        for utility in utilities.iter_mut() {
            *utility /= scale;
        }
        utilities
    }

    fn ruliad_score_has_policy_correctness_signal(
        score: &burn_dragon_universality::ruliad::RuliadReasoningScore,
        min_partial_progress_ppm: usize,
        min_completion_quality_ppm: usize,
    ) -> bool {
        if score.completion_quality_ppm < min_completion_quality_ppm {
            return false;
        }
        matches!(
            score.status,
            burn_dragon_universality::ruliad::RuliadAnswerStatus::VerifierMatch
                | burn_dragon_universality::ruliad::RuliadAnswerStatus::SemanticMatch
        ) || (score.status == burn_dragon_universality::ruliad::RuliadAnswerStatus::Partial
            && score.partial_progress_ppm >= min_partial_progress_ppm)
    }

    fn ruliad_score_has_rollout_recovery_signal(
        score: &burn_dragon_universality::ruliad::RuliadReasoningScore,
        min_partial_progress_ppm: usize,
        min_completion_quality_ppm: usize,
    ) -> bool {
        if score.completion_quality_ppm < min_completion_quality_ppm {
            return false;
        }
        match score.status {
            burn_dragon_universality::ruliad::RuliadAnswerStatus::Partial => {
                score.partial_progress_ppm >= min_partial_progress_ppm
            }
            burn_dragon_universality::ruliad::RuliadAnswerStatus::SchemaValidWrong => true,
            burn_dragon_universality::ruliad::RuliadAnswerStatus::Malformed
            | burn_dragon_universality::ruliad::RuliadAnswerStatus::Missing => true,
            burn_dragon_universality::ruliad::RuliadAnswerStatus::VerifierMatch
            | burn_dragon_universality::ruliad::RuliadAnswerStatus::SemanticMatch => false,
        }
    }

    fn ruliad_score_passes_policy_positive_advantage_gate(
        score: &burn_dragon_universality::ruliad::RuliadReasoningScore,
        config: crate::config::train::RuliadVerifierRewardConfig,
    ) -> bool {
        Self::ruliad_score_has_policy_correctness_signal(
            score,
            config.positive_advantage_min_partial_progress_ppm,
            config.positive_advantage_min_completion_quality_ppm,
        )
    }

    fn constrain_ruliad_policy_advantages(
        scores: &[burn_dragon_universality::ruliad::RuliadReasoningScore],
        advantages: &mut [f32],
        config: crate::config::train::RuliadVerifierRewardConfig,
    ) -> bool {
        if !config.positive_advantage_requires_correctness {
            return true;
        }
        let mut has_correctness_candidate = false;
        for score in scores {
            if Self::ruliad_score_passes_policy_positive_advantage_gate(score, config) {
                has_correctness_candidate = true;
                break;
            }
        }
        if !has_correctness_candidate {
            return false;
        }
        for (score, advantage) in scores.iter().zip(advantages.iter_mut()) {
            if *advantage > 0.0
                && !Self::ruliad_score_passes_policy_positive_advantage_gate(score, config)
            {
                *advantage = 0.0;
            }
        }
        advantages
            .iter()
            .any(|advantage| advantage.abs() > f32::EPSILON)
    }

    fn write_ruliad_policy_telemetry(&self, telemetry: RuliadPolicyRewardTelemetry) {
        let Some(path) = self.ruliad_policy_telemetry_path.as_ref() else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let Ok(line) = serde_json::to_string(&telemetry) else {
            return;
        };
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.as_ref())
        {
            let _ = writeln!(file, "{line}");
        }
    }

    fn write_ruliad_answer_contract_telemetry(&self, telemetry: RuliadAnswerContractTelemetry) {
        let Some(path) = self.ruliad_answer_contract_telemetry_path.as_ref() else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let Ok(line) = serde_json::to_string(&telemetry) else {
            return;
        };
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.as_ref())
        {
            let _ = writeln!(file, "{line}");
        }
    }

    fn write_ruliad_answer_contract_skip(&self, reason: &str, weight: f32) {
        self.write_ruliad_answer_contract_telemetry(RuliadAnswerContractTelemetry {
            version: 1,
            step_index: self.gradient_scale_step.load(Ordering::Relaxed),
            policy_batch_present: false,
            skip_reason: Some(reason.to_string()),
            sample_groups: 0,
            prompt_schema_sample_groups: 0,
            oracle_rows: 0,
            prompt_schema_rows: 0,
            contract_tokens: 0,
            prompt_schema_value_tokens: 0,
            schema_tokens: 0,
            schema_start_tokens: 0,
            value_tokens: 0,
            other_tokens: 0,
            premature_close_tokens: 0,
            answer_contract_weight: weight,
            premature_close_unlikelihood_weight: self
                .ruliad_supervision
                .answer_contract
                .premature_close_unlikelihood_weight,
            max_completion_tokens: self
                .ruliad_supervision
                .answer_contract
                .max_completion_tokens,
            max_rows_per_step: self.ruliad_supervision.answer_contract.max_rows_per_step,
            prompt_schema_max_rows_per_step: {
                let contract = self.ruliad_supervision.answer_contract;
                if contract.prompt_schema_max_rows_per_step == 0 {
                    contract.max_rows_per_step
                } else {
                    contract.prompt_schema_max_rows_per_step
                }
            },
        });
    }

    fn write_ruliad_structured_contrast_telemetry(
        &self,
        telemetry: RuliadStructuredContrastTelemetry,
    ) {
        let Some(path) = self.ruliad_structured_contrast_telemetry_path.as_ref() else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let Ok(line) = serde_json::to_string(&telemetry) else {
            return;
        };
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.as_ref())
        {
            let _ = writeln!(file, "{line}");
        }
    }

    fn write_ruliad_structured_contrast_skip(&self, reason: &str, weight: f32) {
        self.write_ruliad_structured_contrast_telemetry(RuliadStructuredContrastTelemetry {
            version: 1,
            step_index: self.gradient_scale_step.load(Ordering::Relaxed),
            skip_reason: Some(reason.to_string()),
            sample_groups: 0,
            oracle_completion_rows: 0,
            field_negative_completion_rows: 0,
            template_negative_completion_rows: 0,
            schema_negative_completion_rows: 0,
            generated_attractor_negative_completion_rows: 0,
            contrast_pairs: 0,
            contrast_discriminative_tokens: 0,
            structured_contrast_weight: weight,
            structured_contrast_margin: self
                .ruliad_supervision
                .verifier_reward
                .structured_contrast_margin,
        });
    }

    fn write_ruliad_field_binding_contrast_telemetry(
        &self,
        telemetry: RuliadFieldBindingContrastTelemetry,
    ) {
        let Some(path) = self.ruliad_field_binding_contrast_telemetry_path.as_ref() else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let Ok(line) = serde_json::to_string(&telemetry) else {
            return;
        };
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.as_ref())
        {
            let _ = writeln!(file, "{line}");
        }
    }

    fn write_ruliad_field_binding_contrast_skip(&self, reason: &str, weight: f32) {
        self.write_ruliad_field_binding_contrast_telemetry(RuliadFieldBindingContrastTelemetry {
            version: 3,
            objective: RULIAD_FIELD_BINDING_OBJECTIVE,
            step_index: self.gradient_scale_step.load(Ordering::Relaxed),
            skip_reason: Some(reason.to_string()),
            sample_groups: 0,
            oracle_prompt_count: 0,
            prompt_pairs: 0,
            contrast_pairs: 0,
            candidate_pairs: 0,
            filtered_presented_action_candidates: 0,
            contrast_discriminative_tokens: 0,
            negative_pool_size: 0,
            replay_pool_size: 0,
            replay_contrast_pairs: 0,
            generated_attractor_pool_size: 0,
            generated_attractor_negative_pool_size: 0,
            generated_attractor_contrast_pairs: 0,
            rank_metric_pairs: 0,
            rank_metric_tokens: 0,
            logit_margin_mean: None,
            positive_token_fraction: None,
            margin_satisfied_token_fraction: None,
            exact_pair_rank_fraction: None,
            exact_pair_margin_fraction: None,
            sequence_rank_metric_pairs: 0,
            sequence_log_probability_margin_mean: None,
            positive_sequence_fraction: None,
            sequence_margin_satisfied_fraction: None,
            field_binding_contrast_weight: weight,
            field_binding_contrast_margin: self
                .ruliad_supervision
                .verifier_reward
                .field_binding_contrast_margin,
            field_binding_contrast_pair_weight: self
                .ruliad_supervision
                .verifier_reward
                .field_binding_contrast_pair_weight,
        });
    }

    fn write_ruliad_generated_attractor_telemetry(
        &self,
        telemetry: RuliadGeneratedAttractorReplayTelemetry,
    ) {
        let Some(path) = self.ruliad_generated_attractor_telemetry_path.as_ref() else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let Ok(line) = serde_json::to_string(&telemetry) else {
            return;
        };
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.as_ref())
        {
            let _ = writeln!(file, "{line}");
        }
    }

    fn ruliad_generated_attractor_summary(&self) -> RuliadGeneratedAttractorReplaySummary {
        let config = self.ruliad_supervision.verifier_reward;
        self.ruliad_generated_attractor_replay
            .lock()
            .map(|replay| replay.summary(config.generated_attractor_replay_min_count.max(1)))
            .unwrap_or_default()
    }

    fn ruliad_generated_attractor_replay_skip_reason(
        &self,
        summary: &RuliadGeneratedAttractorReplaySummary,
        selected_candidate_rows: usize,
    ) -> Option<String> {
        let config = self.ruliad_supervision.verifier_reward;
        if config.generated_attractor_replay_capacity == 0 || selected_candidate_rows > 0 {
            return None;
        }
        summary
            .diversity_skip_reason(
                config
                    .generated_attractor_replay_min_distinct_answers
                    .max(1),
                config.generated_attractor_replay_max_dominant_fraction,
            )
            .map(str::to_string)
    }

    fn write_ruliad_structured_recovery_telemetry(
        &self,
        telemetry: RuliadStructuredRecoveryTelemetry,
    ) {
        let Some(path) = self.ruliad_structured_recovery_telemetry_path.as_ref() else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let Ok(line) = serde_json::to_string(&telemetry) else {
            return;
        };
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.as_ref())
        {
            let _ = writeln!(file, "{line}");
        }
    }

    fn write_ruliad_verifier_rollout_telemetry(
        &self,
        telemetry: RuliadVerifierRolloutImitationTelemetry,
    ) {
        let Some(path) = self.ruliad_verifier_rollout_telemetry_path.as_ref() else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let Ok(line) = serde_json::to_string(&telemetry) else {
            return;
        };
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.as_ref())
        {
            let _ = writeln!(file, "{line}");
        }
    }

    fn write_ruliad_proof_policy_dagger_telemetry(
        &self,
        telemetry: RuliadProofPolicyDaggerTelemetry,
    ) {
        let Some(path) = self.ruliad_proof_policy_telemetry_path.as_ref() else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let Ok(line) = serde_json::to_string(&telemetry) else {
            return;
        };
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.as_ref())
        {
            let _ = writeln!(file, "{line}");
        }
    }

    fn ruliad_policy_row_from_completion(
        prompt: &[i64],
        completion: &[i64],
    ) -> Option<(Vec<i64>, Vec<i64>, Vec<f32>)> {
        if completion.is_empty() {
            return None;
        }
        let mut sequence = prompt.to_vec();
        sequence.extend_from_slice(completion);
        if sequence.len() < 2 {
            return None;
        }
        let input_len = sequence.len() - 1;
        let inputs = sequence[..input_len].to_vec();
        let targets = sequence[1..].to_vec();
        let mut mask = vec![0.0f32; input_len];
        let completion_start = prompt.len().saturating_sub(1).min(input_len);
        for value in mask.iter_mut().skip(completion_start) {
            *value = 1.0;
        }
        Some((inputs, targets, mask))
    }

    fn ruliad_policy_row_from_completion_token(
        prompt: &[i64],
        completion: &[i64],
        completion_token_index: usize,
    ) -> Option<(Vec<i64>, Vec<i64>, Vec<f32>)> {
        if completion_token_index >= completion.len() {
            return None;
        }
        let (inputs, targets, mut mask) =
            Self::ruliad_policy_row_from_completion(prompt, completion)?;
        mask.fill(0.0);
        let target_index = prompt
            .len()
            .saturating_sub(1)
            .saturating_add(completion_token_index);
        *mask.get_mut(target_index)? = 1.0;
        Some((inputs, targets, mask))
    }

    fn ruliad_trim_prompt_for_completion(
        prompt: &[i64],
        completion_len: usize,
        block_size: usize,
    ) -> Vec<i64> {
        if prompt.is_empty() {
            return Vec::new();
        }
        let max_prompt_len = block_size.saturating_sub(completion_len.max(1)).max(1);
        if prompt.len() > max_prompt_len {
            prompt[prompt.len() - max_prompt_len..].to_vec()
        } else {
            prompt.to_vec()
        }
    }

    fn ruliad_oracle_completion_tokens(
        tokenizer: &burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer,
        sample: &crate::dataset::RuliadPolicySample,
        completion_budget: usize,
    ) -> Option<(Vec<i64>, String, bool)> {
        let answer = sample.item.expected_answer.trim();
        if answer.is_empty() || completion_budget == 0 {
            return None;
        }
        let full_completion = format!("{answer}\n{}", sample.item.document_close_marker());
        let mut payload_tokens = tokenizer.encode_payload(&full_completion);
        let truncated = payload_tokens.len() > completion_budget;
        payload_tokens.truncate(completion_budget);
        if payload_tokens.is_empty() {
            return None;
        }
        let completion_text = tokenizer.decode_payload(&payload_tokens, true);
        let completion = payload_tokens
            .into_iter()
            .map(i64::from)
            .collect::<Vec<_>>();
        Some((completion, completion_text, truncated))
    }

    fn record_ruliad_generated_attractor(
        &self,
        sample: &crate::dataset::RuliadPolicySample,
        completion_text: &str,
        score: &burn_dragon_universality::ruliad::RuliadReasoningScore,
        step_index: usize,
    ) -> bool {
        let config = self.ruliad_supervision.verifier_reward;
        if config.generated_attractor_replay_capacity == 0 {
            return false;
        }
        let extracted =
            burn_dragon_universality::ruliad::extract_ruliad_completion(completion_text);
        let Some(answer) = extracted.answer.map(|answer| answer.trim().to_string()) else {
            return false;
        };
        if answer.is_empty() || answer == sample.item.expected_answer.trim() {
            return false;
        }
        let Some(contract) = Self::ruliad_answer_contract(&answer) else {
            return false;
        };
        let key = RuliadGeneratedAttractorKey {
            family: sample.item.family.clone(),
            task_kind: sample.item.task_kind.clone(),
            contract,
            answer,
        };
        self.ruliad_generated_attractor_replay
            .lock()
            .map(|mut replay| {
                replay.record(
                    key,
                    score.status,
                    step_index,
                    config.generated_attractor_replay_capacity,
                )
            })
            .unwrap_or(false)
    }

    fn ruliad_generated_attractor_candidates_for_sample(
        &self,
        sample: &crate::dataset::RuliadPolicySample,
    ) -> Vec<RuliadGeneratedAttractorEntry> {
        let config = self.ruliad_supervision.verifier_reward;
        if config.generated_attractor_replay_capacity == 0
            || config.generated_attractor_replay_max_candidates == 0
        {
            return Vec::new();
        }
        let Some(expected_contract) = Self::ruliad_answer_contract(&sample.item.expected_answer)
        else {
            return Vec::new();
        };
        self.ruliad_generated_attractor_replay
            .lock()
            .map(|replay| {
                replay.candidates_for(RuliadGeneratedAttractorQuery {
                    family: &sample.item.family,
                    task_kind: &sample.item.task_kind,
                    expected_contract: &expected_contract,
                    expected_answer: sample.item.expected_answer.trim(),
                    min_count: config.generated_attractor_replay_min_count.max(1),
                    max_candidates: config.generated_attractor_replay_max_candidates,
                    min_distinct_answers: config
                        .generated_attractor_replay_min_distinct_answers
                        .max(1),
                    max_dominant_fraction: config.generated_attractor_replay_max_dominant_fraction,
                })
            })
            .unwrap_or_default()
    }

    fn ruliad_structured_negative_answers(answer: &str, count: usize) -> Vec<String> {
        Self::ruliad_structured_negative_answers_with_templates(answer, count, 0)
            .into_iter()
            .map(|(answer, _kind)| answer)
            .collect()
    }

    fn ruliad_model_proof_step_negative_answers(
        answer: &str,
        mutation_count: usize,
        template_count: usize,
    ) -> Option<Vec<(String, RuliadStructuredNegativeKind)>> {
        use burn_dragon_universality::ruliad::{
            RuliadProofSource, RuliadProofStep, RuliadRewriteDirection,
        };

        let (goal, step) = burn_dragon_universality::ruliad::wire::decode_model_proof_step(answer)?;

        let mut negatives = Vec::with_capacity(mutation_count.saturating_add(template_count));
        let mut template_rows = 0usize;
        for index in 0..template_count.saturating_add(4) {
            if template_rows >= template_count {
                break;
            }
            let candidate = burn_dragon_universality::ruliad::wire::encode_model_proof_step(
                index,
                &RuliadProofStep {
                    source: RuliadProofSource::Axiom {
                        id: format!("r{index}"),
                    },
                    direction: if index.is_multiple_of(2) {
                        RuliadRewriteDirection::Forward
                    } else {
                        RuliadRewriteDirection::Reverse
                    },
                    path: (!index.is_multiple_of(3))
                        .then_some(vec![0])
                        .unwrap_or_default(),
                },
            );
            let previous_len = negatives.len();
            Self::push_ruliad_negative_answer(
                &mut negatives,
                answer,
                candidate,
                RuliadStructuredNegativeKind::TemplateCollapse,
            );
            template_rows =
                template_rows.saturating_add(usize::from(negatives.len() > previous_len));
        }

        for index in 0..mutation_count {
            let mut candidate_goal = goal;
            let mut candidate_step = step.clone();
            let field_count = 4;
            let delta = index / field_count + 1;
            match index % field_count {
                0 => {
                    candidate_goal = candidate_goal.saturating_add(delta);
                }
                1 => match &mut candidate_step.source {
                    RuliadProofSource::Axiom { id } => {
                        let mutated = Self::mutate_ruliad_answer_value(id, delta);
                        *id = mutated
                            .strip_suffix("_wrong")
                            .map(|prefix| format!("{prefix}x"))
                            .unwrap_or(mutated);
                    }
                    RuliadProofSource::Lemma { goal } => {
                        *goal = goal.saturating_add(delta);
                    }
                },
                2 => {
                    candidate_step.direction = match candidate_step.direction {
                        RuliadRewriteDirection::Forward => RuliadRewriteDirection::Reverse,
                        RuliadRewriteDirection::Reverse => RuliadRewriteDirection::Forward,
                    };
                }
                3 => {
                    if candidate_step.path.is_empty() {
                        candidate_step.path.push(delta.saturating_sub(1));
                    } else {
                        let path_index = (index / field_count) % candidate_step.path.len();
                        let value = candidate_step.path.get_mut(path_index)?;
                        *value = value.saturating_add(delta);
                    }
                }
                _ => unreachable!(),
            }
            Self::push_ruliad_negative_answer(
                &mut negatives,
                answer,
                burn_dragon_universality::ruliad::wire::encode_model_proof_step(
                    candidate_goal,
                    &candidate_step,
                ),
                RuliadStructuredNegativeKind::FieldMutation,
            );
        }
        Some(negatives)
    }

    fn ruliad_structured_negative_answers_with_templates(
        answer: &str,
        mutation_count: usize,
        template_count: usize,
    ) -> Vec<(String, RuliadStructuredNegativeKind)> {
        let answer = answer.trim();
        if answer.is_empty() || (mutation_count == 0 && template_count == 0) {
            return Vec::new();
        }
        if let Some(negatives) =
            Self::ruliad_model_proof_step_negative_answers(answer, mutation_count, template_count)
        {
            return negatives;
        }
        let fields = answer
            .split(';')
            .filter_map(|part| {
                let (key, value) = part.split_once('=')?;
                let key = key.trim();
                if key.is_empty() {
                    return None;
                }
                Some((key.to_string(), value.trim().to_string()))
            })
            .collect::<Vec<_>>();
        if fields.is_empty() {
            return (0..mutation_count.max(1))
                .map(|_| {
                    (
                        format!("{answer}_wrong"),
                        RuliadStructuredNegativeKind::FieldMutation,
                    )
                })
                .take(mutation_count.max(template_count))
                .collect();
        }

        let mut negatives = Vec::with_capacity(mutation_count + template_count);
        for template in Self::ruliad_template_collapse_negative_answers(answer, &fields) {
            if negatives.len() >= template_count {
                break;
            }
            Self::push_ruliad_negative_answer(
                &mut negatives,
                answer,
                template,
                RuliadStructuredNegativeKind::TemplateCollapse,
            );
        }

        for index in 0..mutation_count {
            let mutate_index = index % fields.len();
            let mut candidate = fields.clone();
            let mutated = Self::mutate_ruliad_answer_value(&candidate[mutate_index].1, index + 1);
            candidate[mutate_index].1 = mutated;
            let text = candidate
                .into_iter()
                .map(|(key, value)| format!("{key}={value}"))
                .collect::<Vec<_>>()
                .join(";");
            Self::push_ruliad_negative_answer(
                &mut negatives,
                answer,
                text,
                RuliadStructuredNegativeKind::FieldMutation,
            );
        }
        negatives
    }

    fn ruliad_structured_negative_answers_with_schema(
        answer: &str,
        mutation_count: usize,
        template_count: usize,
        schema_count: usize,
    ) -> Vec<(String, RuliadStructuredNegativeKind)> {
        let mut negatives = Self::ruliad_structured_negative_answers_with_templates(
            answer,
            mutation_count,
            template_count,
        );
        if schema_count == 0 {
            return negatives;
        }
        negatives.reserve(schema_count);
        for schema_negative in Self::ruliad_schema_collapse_negative_answers(answer)
            .into_iter()
            .take(schema_count)
        {
            Self::push_ruliad_negative_answer(
                &mut negatives,
                answer,
                schema_negative,
                RuliadStructuredNegativeKind::SchemaCollapse,
            );
        }
        negatives
    }

    fn push_ruliad_negative_answer(
        negatives: &mut Vec<(String, RuliadStructuredNegativeKind)>,
        answer: &str,
        candidate: String,
        kind: RuliadStructuredNegativeKind,
    ) {
        let candidate = candidate.trim();
        if candidate.is_empty() || candidate == answer {
            return;
        }
        if negatives
            .iter()
            .any(|(existing, _existing_kind)| existing == candidate)
        {
            return;
        }
        negatives.push((candidate.to_string(), kind));
    }

    fn ruliad_template_collapse_negative_answers(
        answer: &str,
        fields: &[(String, String)],
    ) -> Vec<String> {
        let has_key = |key: &str| fields.iter().any(|(candidate, _)| candidate == key);
        let mut templates = Vec::<String>::new();
        let mut push = |candidate: &str| {
            if candidate != answer && !templates.iter().any(|existing| existing == candidate) {
                templates.push(candidate.to_string());
            }
        };

        if has_key("ok") && has_key("l") && has_key("r") {
            push("ok=1;l=5;r=5");
            push("ok=1;l=1;r=1");
            push("ok=0;l=0;r=0");
            push("ok=1;l=0;r=0");
        } else if has_key("ok") {
            push("ok=0");
            push("ok=1");
        }

        if has_key("acc") {
            push("acc=0");
            push("acc=1");
        }

        if has_key("xlen") && has_key("xalpha") && has_key("xcounts") && has_key("xedge") {
            push("xlen=13;xalpha=abc;nfcounts=1,1,0;nfedge=ba");
            push("xlen=1;xalpha=01;xcounts=1,1;xedge=00");
            push("xlen=10;xalpha=01;xcounts=10,10;xedge=00");
            push("xlen=21;xalpha=01;xcounts=10,11;xedge=00");
            push("xlen=64;xalpha=01;xcounts=32,32;xedge=00");
        }

        if has_key("nflen") && has_key("nfalpha") && has_key("nfcounts") && has_key("nfedge") {
            push("nflen=5;nfalpha=abc;nfcounts=1,1,0;nfedge=ba");
            push("nflen=1;nfalpha=01;nfcounts=1,1;nfedge=00");
            push("nflen=10;nfalpha=01;nfcounts=10,10;nfedge=00");
            push("nflen=21;nfalpha=01;nfcounts=10,11;nfedge=00");
            push("nflen=64;nfalpha=01;nfcounts=32,32;nfedge=00");
        }

        templates
    }

    fn ruliad_template_collapse_negative_answers_from_answer(answer: &str) -> Vec<String> {
        let answer = answer.trim();
        if answer.is_empty() {
            return Vec::new();
        }
        let fields = answer
            .split(';')
            .filter_map(|part| {
                let (key, value) = part.split_once('=')?;
                let key = key.trim();
                if key.is_empty() {
                    return None;
                }
                Some((key.to_string(), value.trim().to_string()))
            })
            .collect::<Vec<_>>();
        if fields.is_empty() {
            return Vec::new();
        }
        Self::ruliad_template_collapse_negative_answers(answer, &fields)
    }

    fn ruliad_schema_collapse_negative_answers(answer: &str) -> Vec<String> {
        let answer = answer.trim();
        if answer.is_empty() {
            return Vec::new();
        }
        let fields = answer
            .split(';')
            .filter_map(|part| {
                let (key, value) = part.split_once('=')?;
                let key = key.trim();
                if key.is_empty() {
                    return None;
                }
                Some((key.to_string(), value.trim().to_string()))
            })
            .collect::<Vec<_>>();
        let keys = fields
            .iter()
            .map(|(key, _value)| key.as_str())
            .collect::<Vec<_>>();
        let values = fields
            .iter()
            .map(|(_key, value)| value.as_str())
            .collect::<Vec<_>>();
        let mut negatives = Vec::<String>::new();
        let mut push = |candidate: String| {
            if candidate != answer && !negatives.iter().any(|existing| existing == &candidate) {
                negatives.push(candidate);
            }
        };
        if fields.len() > 1 {
            push(
                fields[..fields.len() - 1]
                    .iter()
                    .map(|(key, value)| format!("{key}={value}"))
                    .collect::<Vec<_>>()
                    .join(";"),
            );
            push(format!("{}={}", fields[0].0, fields[0].1));
        }
        if keys == ["xlen", "xalpha", "xcounts", "xedge"] && values.len() == 4 {
            push(format!(
                "xlen={};nfalpha={};nfcounts={};xedge={}",
                values[0], values[1], values[2], values[3]
            ));
            push(format!(
                "nflen={};nfalpha={};nfcounts={};nfedge={}",
                values[0], values[1], values[2], values[3]
            ));
        } else if keys == ["nflen", "nfalpha", "nfcounts", "nfedge"] && values.len() == 4 {
            push(format!(
                "nflen={};xalpha={};xcounts={};nfedge={}",
                values[0], values[1], values[2], values[3]
            ));
            push(format!(
                "xlen={};xalpha={};xcounts={};xedge={}",
                values[0], values[1], values[2], values[3]
            ));
        } else if keys == ["ok", "l", "r"] && values.len() == 3 {
            push(format!("ok={}", values[0]));
        } else if keys == ["ok"] && values.len() == 1 {
            push(format!("ok={};l=1;r=1", values[0]));
        }
        for prototype in Self::ruliad_cross_contract_prototype_negatives(&keys) {
            push(prototype);
        }
        negatives
    }

    fn ruliad_cross_contract_prototype_negatives(keys: &[&str]) -> Vec<String> {
        let contract = keys.join(",");
        let mut prototypes = match contract.as_str() {
            "xlen,xalpha,xcounts,xedge" | "nflen,nfalpha,nfcounts,nfedge" => {
                vec!["ok=1;l=1;r=1", "ok=0;l=0;r=0", "acc=1", "acc=0"]
            }
            "ok,l,r" | "ok" => vec![
                "xlen=1;xalpha=01;xcounts=1,0;xedge=00",
                "nflen=1;nfalpha=ABC;nfcounts=1,0,0;nfedge=AA",
                "acc=1",
                "acc=0",
            ],
            "acc" => vec![
                "ok=1;l=1;r=1",
                "xlen=1;xalpha=01;xcounts=1,0;xedge=00",
                "nflen=1;nfalpha=ABC;nfcounts=1,0,0;nfedge=AA",
            ],
            _ => Vec::new(),
        };
        prototypes.drain(..).map(str::to_string).collect::<Vec<_>>()
    }

    fn mutate_ruliad_answer_value(value: &str, delta: usize) -> String {
        let delta = delta.max(1) as u64;
        if value == "0" {
            return "1".to_string();
        }
        if value == "1" {
            return "0".to_string();
        }
        if value.len() > 1 && value.bytes().all(|byte| byte == b'0' || byte == b'1') {
            let mut bytes = value.as_bytes().to_vec();
            let index = delta as usize % bytes.len();
            bytes[index] = if bytes[index] == b'0' { b'1' } else { b'0' };
            return String::from_utf8(bytes).unwrap_or_else(|_| value.to_string());
        }

        let mut output = String::with_capacity(value.len() + 4);
        let bytes = value.as_bytes();
        let mut index = 0usize;
        let mut mutated_any = false;
        while index < bytes.len() {
            if bytes[index].is_ascii_digit() {
                let start = index;
                while index < bytes.len() && bytes[index].is_ascii_digit() {
                    index += 1;
                }
                let text = &value[start..index];
                let width = text.len();
                let modulus = 10u64.saturating_pow(width.min(18) as u32).max(2);
                let parsed = text.parse::<u64>().unwrap_or(0);
                let mut next = (parsed + delta) % modulus;
                if next == parsed {
                    next = (next + 1) % modulus;
                }
                output.push_str(&format!("{next:0width$}"));
                mutated_any = true;
            } else {
                output.push(bytes[index] as char);
                index += 1;
            }
        }
        if mutated_any {
            output
        } else {
            format!("{value}_wrong")
        }
    }

    fn ruliad_completion_tokens_from_answer(
        tokenizer: &burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer,
        answer: &str,
        close_marker: &str,
        completion_budget: usize,
    ) -> Option<(Vec<i64>, String)> {
        if answer.trim().is_empty() || completion_budget == 0 {
            return None;
        }
        let full_completion = format!("{}\n{close_marker}", answer.trim());
        let mut payload_tokens = tokenizer.encode_payload(&full_completion);
        payload_tokens.truncate(completion_budget);
        if payload_tokens.is_empty() {
            return None;
        }
        let completion_text = tokenizer.decode_payload(&payload_tokens, true);
        let completion = payload_tokens
            .into_iter()
            .map(i64::from)
            .collect::<Vec<_>>();
        Some((completion, completion_text))
    }

    fn ruliad_answer_value_completion_mask(
        tokenizer: &burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer,
        answer: &str,
        completion_len: usize,
    ) -> Vec<bool> {
        let answer = answer.trim();
        if answer.is_empty() || completion_len == 0 {
            return vec![false; completion_len];
        }
        let full_completion = format!("{answer}\n[/R2]");
        let mut mask = vec![false; completion_len];
        if burn_dragon_universality::ruliad::wire::decode_model_proof_step(answer).is_some() {
            let mut segment_start = 0usize;
            for (segment_index, segment) in answer.split('|').enumerate() {
                let value_offset = match segment_index {
                    0 => 1,
                    1 => 2,
                    2 | 3 => 0,
                    _ => return vec![false; completion_len],
                };
                let value_start = segment_start.saturating_add(value_offset);
                let value_end = segment_start.saturating_add(segment.len());
                if value_start < value_end {
                    let prefix_tokens = tokenizer
                        .encode_payload(&full_completion[..value_start])
                        .len();
                    let value_tokens = tokenizer
                        .encode_payload(&full_completion[value_start..value_end])
                        .len();
                    for index in prefix_tokens..prefix_tokens.saturating_add(value_tokens) {
                        if let Some(slot) = mask.get_mut(index) {
                            *slot = true;
                        }
                    }
                }
                segment_start = value_end.saturating_add(1);
            }
            return mask;
        }
        let bytes = answer.as_bytes();
        let mut field_start = 0usize;
        while field_start < answer.len() {
            let field_end = bytes[field_start..]
                .iter()
                .position(|byte| *byte == b';')
                .map(|offset| field_start + offset)
                .unwrap_or(answer.len());
            let field = &answer[field_start..field_end];
            if let Some(eq_offset) = field.find('=') {
                let mut value_start = field_start + eq_offset + 1;
                while value_start < field_end
                    && answer.as_bytes()[value_start].is_ascii_whitespace()
                {
                    value_start += 1;
                }
                let mut value_end = field_end;
                while value_end > value_start
                    && answer.as_bytes()[value_end - 1].is_ascii_whitespace()
                {
                    value_end -= 1;
                }
                if value_start < value_end {
                    let prefix_tokens = tokenizer
                        .encode_payload(&full_completion[..value_start])
                        .len();
                    let value_tokens = tokenizer
                        .encode_payload(&full_completion[value_start..value_end])
                        .len();
                    for index in prefix_tokens..prefix_tokens.saturating_add(value_tokens) {
                        if let Some(slot) = mask.get_mut(index) {
                            *slot = true;
                        }
                    }
                }
            }
            field_start = field_end.saturating_add(1);
        }
        mask
    }

    fn ruliad_answer_key_completion_mask(
        tokenizer: &burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer,
        answer: &str,
        completion_len: usize,
    ) -> Vec<bool> {
        let answer = answer.trim();
        if answer.is_empty() || completion_len == 0 {
            return vec![false; completion_len];
        }
        let full_completion = format!("{answer}\n[/R2]");
        let mut mask = vec![false; completion_len];
        let bytes = answer.as_bytes();
        let mut field_start = 0usize;
        while field_start < answer.len() {
            let field_end = bytes[field_start..]
                .iter()
                .position(|byte| *byte == b';')
                .map(|offset| field_start + offset)
                .unwrap_or(answer.len());
            let field = &answer[field_start..field_end];
            if let Some(eq_offset) = field.find('=') {
                let mut key_start = field_start;
                while key_start < field_start + eq_offset
                    && answer.as_bytes()[key_start].is_ascii_whitespace()
                {
                    key_start += 1;
                }
                let mut key_end = field_start + eq_offset;
                while key_end > key_start && answer.as_bytes()[key_end - 1].is_ascii_whitespace() {
                    key_end -= 1;
                }
                if key_start < key_end {
                    let prefix_tokens = tokenizer
                        .encode_payload(&full_completion[..key_start])
                        .len();
                    let key_tokens = tokenizer
                        .encode_payload(&full_completion[key_start..key_end])
                        .len();
                    for index in prefix_tokens..prefix_tokens.saturating_add(key_tokens) {
                        if let Some(slot) = mask.get_mut(index) {
                            *slot = true;
                        }
                    }
                }
            }
            field_start = field_end.saturating_add(1);
        }
        mask
    }

    fn ruliad_answer_schema_completion_mask(
        tokenizer: &burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer,
        answer: &str,
        completion_len: usize,
    ) -> Vec<bool> {
        let answer = answer.trim();
        if answer.is_empty() || completion_len == 0 {
            return vec![false; completion_len];
        }
        let full_completion = format!("{answer}\n[/R2]");
        let mut mask = vec![false; completion_len];
        let bytes = answer.as_bytes();
        for (byte_index, byte) in bytes.iter().enumerate() {
            let active = byte.is_ascii_alphabetic() || *byte == b'=' || *byte == b';';
            if !active {
                continue;
            }
            let prefix_tokens = tokenizer
                .encode_payload(&full_completion[..byte_index])
                .len();
            let token_count = tokenizer
                .encode_payload(&full_completion[byte_index..byte_index + 1])
                .len();
            for index in prefix_tokens..prefix_tokens.saturating_add(token_count) {
                if let Some(slot) = mask.get_mut(index) {
                    *slot = true;
                }
            }
        }
        mask
    }

    fn ruliad_answer_schema_start_completion_mask(
        tokenizer: &burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer,
        answer: &str,
        completion_len: usize,
    ) -> Vec<bool> {
        let answer = answer.trim();
        if answer.is_empty() || completion_len == 0 {
            return vec![false; completion_len];
        }
        let full_completion = format!("{answer}\n[/R2]");
        let mut mask = vec![false; completion_len];
        let bytes = answer.as_bytes();
        let mut field_start = 0usize;
        while field_start < answer.len() {
            let field_end = bytes[field_start..]
                .iter()
                .position(|byte| *byte == b';')
                .map(|offset| field_start + offset)
                .unwrap_or(answer.len());
            let field = &answer[field_start..field_end];
            if let Some(eq_offset) = field.find('=') {
                let mut key_start = field_start;
                while key_start < field_start + eq_offset
                    && answer.as_bytes()[key_start].is_ascii_whitespace()
                {
                    key_start += 1;
                }
                if key_start < field_start + eq_offset {
                    let first = answer.as_bytes()[key_start];
                    if first.is_ascii_alphabetic() || first == b'_' {
                        let prefix_tokens = tokenizer
                            .encode_payload(&full_completion[..key_start])
                            .len();
                        let token_count = tokenizer
                            .encode_payload(&full_completion[key_start..key_start + 1])
                            .len();
                        for index in prefix_tokens..prefix_tokens.saturating_add(token_count) {
                            if let Some(slot) = mask.get_mut(index) {
                                *slot = true;
                            }
                        }
                    }
                }
            }
            field_start = field_end.saturating_add(1);
        }
        mask
    }

    fn ruliad_answer_contract(answer: &str) -> Option<String> {
        if burn_dragon_universality::ruliad::wire::decode_model_proof_step(answer).is_some() {
            return Some("proof_action_step".to_string());
        }
        let mut keys = Vec::<String>::new();
        for part in answer.trim().split(';') {
            let (key, _value) = part.split_once('=')?;
            let key = key.trim();
            if key.is_empty() {
                return None;
            }
            keys.push(key.to_string());
        }
        (!keys.is_empty()).then(|| keys.join(";"))
    }

    fn ruliad_answer_fields(answer: &str) -> Option<Vec<(String, String)>> {
        let mut fields = Vec::<(String, String)>::new();
        for part in answer.trim().split(';') {
            let (key, value) = part.split_once('=')?;
            let key = key.trim();
            let value = value.trim();
            if key.is_empty() || value.is_empty() {
                return None;
            }
            fields.push((key.to_string(), value.to_string()));
        }
        (!fields.is_empty()).then_some(fields)
    }

    fn ruliad_prompt_schema_value_completion_rows(
        tokenizer: &burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer,
        base_prompt: &[i64],
        answer: &str,
        close_marker: &str,
        completion_budget: usize,
        block_size: usize,
        max_rows: usize,
    ) -> Vec<RuliadPromptSchemaValueRow> {
        if base_prompt.is_empty() || completion_budget == 0 || block_size < 4 || max_rows == 0 {
            return Vec::new();
        }
        let fields = if burn_dragon_universality::ruliad::wire::decode_model_proof_step(answer)
            .is_some()
        {
            let parts = answer.trim().split('|').collect::<Vec<_>>();
            if parts.len() != 4 {
                return Vec::new();
            }
            let Some(goal) = parts[0].strip_prefix('g') else {
                return Vec::new();
            };
            let (source_schema, source_value) = if let Some(source) = parts[1].strip_prefix("a:") {
                ("a:", source)
            } else if let Some(source) = parts[1].strip_prefix("l:") {
                ("l:", source)
            } else {
                return Vec::new();
            };
            vec![
                ("g".to_string(), goal.to_string(), "|".to_string()),
                (
                    format!("g{goal}|{source_schema}"),
                    source_value.to_string(),
                    "|".to_string(),
                ),
                (
                    format!("g{goal}|{}|", parts[1]),
                    parts[2].to_string(),
                    "|".to_string(),
                ),
                (
                    format!("g{goal}|{}|{}|", parts[1], parts[2]),
                    parts[3].to_string(),
                    format!("\n{close_marker}"),
                ),
            ]
        } else {
            let Some(answer_fields) = Self::ruliad_answer_fields(answer) else {
                return Vec::new();
            };
            let mut fields = Vec::with_capacity(answer_fields.len());
            let mut prior = String::new();
            let field_count = answer_fields.len();
            for (index, (key, value)) in answer_fields.into_iter().enumerate() {
                let close = if index + 1 == field_count {
                    format!("\n{close_marker}")
                } else {
                    ";".to_string()
                };
                fields.push((format!("{prior}{key}="), value.clone(), close));
                prior.push_str(&key);
                prior.push('=');
                prior.push_str(&value);
                prior.push(';');
            }
            fields
        };
        let row_completion_budget = completion_budget.min(block_size.saturating_sub(2).max(1));
        let mut rows = Vec::<RuliadPromptSchemaValueRow>::new();
        for (schema_prefix, value, close) in fields {
            if rows.len() >= max_rows {
                break;
            }
            let mut completion_tokens = tokenizer.encode_payload(&format!("{value}{close}"));
            completion_tokens.truncate(row_completion_budget);
            if completion_tokens.is_empty() {
                continue;
            }
            let mut schema_prefix_tokens = tokenizer.encode_payload(&schema_prefix);
            let prefix_budget = block_size
                .saturating_sub(completion_tokens.len())
                .saturating_sub(1);
            if prefix_budget == 0 {
                continue;
            }
            if schema_prefix_tokens.len() > prefix_budget {
                schema_prefix_tokens =
                    schema_prefix_tokens[schema_prefix_tokens.len() - prefix_budget..].to_vec();
            }
            let prompt_budget = block_size
                .saturating_sub(schema_prefix_tokens.len())
                .saturating_sub(completion_tokens.len())
                .max(1);
            let mut prompt = if base_prompt.len() > prompt_budget {
                base_prompt[base_prompt.len() - prompt_budget..].to_vec()
            } else {
                base_prompt.to_vec()
            };
            prompt.extend(schema_prefix_tokens.into_iter().map(i64::from));
            let completion = completion_tokens
                .into_iter()
                .map(i64::from)
                .collect::<Vec<_>>();
            if let Some((inputs, targets, mask)) =
                Self::ruliad_policy_row_from_completion(&prompt, &completion)
            {
                let active_tokens = mask.iter().filter(|value| **value > f32::EPSILON).count();
                if active_tokens > 0 {
                    rows.push((inputs, targets, mask, active_tokens));
                }
            }
        }
        rows
    }

    fn ruliad_field_binding_rank_stats(
        logit_margin: Tensor<B, 2>,
        mask_values: &[i64],
        row_count: usize,
        max_len: usize,
        required_margin: f64,
    ) -> RuliadFieldBindingRankStats {
        let Ok(margin_values) = logit_margin.to_data().convert::<f32>().into_vec::<f32>() else {
            return RuliadFieldBindingRankStats::default();
        };
        if margin_values.len() != mask_values.len() || mask_values.len() != row_count * max_len {
            return RuliadFieldBindingRankStats::default();
        }

        let mut token_count = 0usize;
        let mut positive_token_count = 0usize;
        let mut margin_token_count = 0usize;
        let mut margin_sum = 0.0f64;
        let mut pair_count = 0usize;
        let mut exact_pair_rank_count = 0usize;
        let mut exact_pair_margin_count = 0usize;

        for row_index in 0..row_count {
            let row_offset = row_index * max_len;
            let mut row_tokens = 0usize;
            let mut row_positive = 0usize;
            let mut row_margin = 0usize;
            for column in 0..max_len {
                let index = row_offset + column;
                if mask_values[index] == 0 {
                    continue;
                }
                let margin = margin_values[index] as f64;
                if !margin.is_finite() {
                    continue;
                }
                row_tokens = row_tokens.saturating_add(1);
                token_count = token_count.saturating_add(1);
                margin_sum += margin;
                if margin > 0.0 {
                    row_positive = row_positive.saturating_add(1);
                    positive_token_count = positive_token_count.saturating_add(1);
                }
                if margin >= required_margin {
                    row_margin = row_margin.saturating_add(1);
                    margin_token_count = margin_token_count.saturating_add(1);
                }
            }
            if row_tokens > 0 {
                pair_count = pair_count.saturating_add(1);
                if row_positive == row_tokens {
                    exact_pair_rank_count = exact_pair_rank_count.saturating_add(1);
                }
                if row_margin == row_tokens {
                    exact_pair_margin_count = exact_pair_margin_count.saturating_add(1);
                }
            }
        }

        let token_denominator = token_count as f64;
        let pair_denominator = pair_count as f64;
        RuliadFieldBindingRankStats {
            pairs: pair_count,
            tokens: token_count,
            logit_margin_mean: (token_count > 0).then_some(margin_sum / token_denominator),
            positive_token_fraction: (token_count > 0)
                .then_some(positive_token_count as f64 / token_denominator),
            margin_satisfied_token_fraction: (token_count > 0)
                .then_some(margin_token_count as f64 / token_denominator),
            exact_pair_rank_fraction: (pair_count > 0)
                .then_some(exact_pair_rank_count as f64 / pair_denominator),
            exact_pair_margin_fraction: (pair_count > 0)
                .then_some(exact_pair_margin_count as f64 / pair_denominator),
        }
    }

    fn ruliad_field_binding_sequence_rank_stats(
        log_probability_margin: Tensor<B, 1>,
        required_margin: f64,
    ) -> RuliadFieldBindingSequenceRankStats {
        let Ok(margins) = log_probability_margin
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
        else {
            return RuliadFieldBindingSequenceRankStats::default();
        };
        let finite = margins
            .into_iter()
            .map(f64::from)
            .filter(|margin| margin.is_finite())
            .collect::<Vec<_>>();
        if finite.is_empty() {
            return RuliadFieldBindingSequenceRankStats::default();
        }
        let denominator = finite.len() as f64;
        RuliadFieldBindingSequenceRankStats {
            pairs: finite.len(),
            log_probability_margin_mean: Some(finite.iter().sum::<f64>() / denominator),
            positive_sequence_fraction: Some(
                finite.iter().filter(|margin| **margin > 0.0).count() as f64 / denominator,
            ),
            margin_satisfied_sequence_fraction: Some(
                finite
                    .iter()
                    .filter(|margin| **margin >= required_margin)
                    .count() as f64
                    / denominator,
            ),
        }
    }

    fn ruliad_field_binding_contrast_loss(
        &self,
        policy_batch: &crate::dataset::RuliadPolicyBatch,
        device: &B::Device,
        block_size: usize,
    ) -> Option<Tensor<B, 1>> {
        let config = self.ruliad_supervision.verifier_reward;
        let weight = self.ruliad_field_binding_contrast_weight();
        let step_index = self.gradient_scale_step.load(Ordering::Relaxed);
        if weight <= f32::EPSILON || policy_batch.samples.is_empty() || self.pipeline_enabled() {
            return None;
        }
        let tokenizer =
            burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer::from_config(
                &policy_batch.tokenization,
            )
            .ok()?;
        let completion_budget = config
            .max_completion_tokens
            .max(1)
            .min(block_size.saturating_sub(1).max(1));

        #[derive(Clone)]
        struct EligibleSample {
            source_index: usize,
            prompt: Vec<i64>,
            answer: String,
            family: String,
            task_kind: String,
            contract: String,
            presented_action_answers: Option<HashSet<String>>,
            oracle_completion: Vec<i64>,
            value_mask: Vec<bool>,
            schema_mask: Vec<bool>,
        }

        #[derive(Clone)]
        struct NegativeSample {
            current_source_index: Option<usize>,
            answer: String,
            family: String,
            task_kind: String,
            contract: String,
            oracle_completion: Vec<i64>,
            from_replay: bool,
            from_generated_attractor: bool,
            schema_negative: bool,
        }

        #[derive(Clone)]
        struct ContrastCandidate {
            oracle_index: usize,
            negative_index: usize,
            negative_source_index: Option<usize>,
            negative_answer: String,
            discriminative_tokens: usize,
            from_replay: bool,
            from_generated_attractor: bool,
            schema_negative: bool,
        }

        #[derive(Clone)]
        struct ContrastRow {
            prompt: Vec<i64>,
            oracle_completion: Vec<i64>,
            negative_completion: Vec<i64>,
            inputs: Vec<i64>,
            oracle_targets: Vec<i64>,
            negative_targets: Vec<i64>,
            mask: Vec<i64>,
            discriminative_tokens: usize,
            source_index: usize,
            negative_source_index: Option<usize>,
            from_replay: bool,
            from_generated_attractor: bool,
        }

        let mut eligible = Vec::<EligibleSample>::new();
        for (source_index, sample) in policy_batch.samples.iter().enumerate() {
            let answer = sample.item.expected_answer.trim();
            if answer.is_empty() {
                continue;
            }
            let Some(contract) = Self::ruliad_answer_contract(answer) else {
                continue;
            };
            let prompt = sample.prompt_tokens.clone();
            if prompt.is_empty() {
                continue;
            }
            let Some((oracle_completion, _oracle_text, _truncated)) =
                Self::ruliad_oracle_completion_tokens(&tokenizer, sample, completion_budget)
            else {
                continue;
            };
            let value_mask = Self::ruliad_answer_value_completion_mask(
                &tokenizer,
                answer,
                oracle_completion.len(),
            );
            let schema_mask = Self::ruliad_answer_schema_completion_mask(
                &tokenizer,
                answer,
                oracle_completion.len(),
            );
            if !value_mask.iter().any(|active| *active) && !schema_mask.iter().any(|active| *active)
            {
                continue;
            }
            eligible.push(EligibleSample {
                source_index,
                prompt,
                answer: answer.to_string(),
                family: sample.item.family.clone(),
                task_kind: sample.item.task_kind.clone(),
                contract,
                presented_action_answers:
                    burn_dragon_universality::ruliad::ruliad_presented_action_answers(&sample.item)
                        .map(|answers| answers.into_iter().collect()),
                oracle_completion,
                value_mask,
                schema_mask,
            });
        }

        let replay_capacity = config.field_binding_contrast_replay_capacity;
        let replay_snapshot = if replay_capacity > 0 {
            self.ruliad_field_binding_replay
                .lock()
                .map(|replay| replay.iter().cloned().collect::<Vec<_>>())
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let replay_pool_size = replay_snapshot.len();
        let mut negative_pool = eligible
            .iter()
            .map(|sample| NegativeSample {
                current_source_index: Some(sample.source_index),
                answer: sample.answer.clone(),
                family: sample.family.clone(),
                task_kind: sample.task_kind.clone(),
                contract: sample.contract.clone(),
                oracle_completion: sample.oracle_completion.clone(),
                from_replay: false,
                from_generated_attractor: false,
                schema_negative: false,
            })
            .collect::<Vec<_>>();
        negative_pool.extend(replay_snapshot.into_iter().map(|sample| NegativeSample {
            current_source_index: None,
            answer: sample.answer,
            family: sample.family,
            task_kind: sample.task_kind,
            contract: sample.contract,
            oracle_completion: sample.oracle_completion,
            from_replay: true,
            from_generated_attractor: false,
            schema_negative: false,
        }));
        let generated_attractor_snapshot = eligible
            .iter()
            .flat_map(|sample| {
                self.ruliad_generated_attractor_candidates_for_sample(
                    &policy_batch.samples[sample.source_index],
                )
            })
            .collect::<Vec<_>>();
        let generated_attractor_pool_size = generated_attractor_snapshot.len();
        let mut seen_generated_attractors = HashSet::<(String, String, String, String)>::new();
        for entry in generated_attractor_snapshot {
            let key = (
                entry.key.answer.clone(),
                entry.key.family.clone(),
                entry.key.task_kind.clone(),
                entry.key.contract.clone(),
            );
            if !seen_generated_attractors.insert(key) {
                continue;
            }
            let Some((oracle_completion, _completion_text)) =
                Self::ruliad_completion_tokens_from_answer(
                    &tokenizer,
                    &entry.key.answer,
                    burn_dragon_universality::ruliad::RULIAD_V2_DOCUMENT_CLOSE_MARKER,
                    completion_budget,
                )
            else {
                continue;
            };
            negative_pool.push(NegativeSample {
                current_source_index: None,
                answer: entry.key.answer,
                family: entry.key.family,
                task_kind: entry.key.task_kind,
                contract: entry.key.contract,
                oracle_completion,
                from_replay: false,
                from_generated_attractor: true,
                schema_negative: false,
            });
        }
        let mut seen_template_negatives = HashSet::<(String, String, String, String)>::new();
        for sample in eligible.iter() {
            for answer in
                Self::ruliad_template_collapse_negative_answers_from_answer(&sample.answer)
            {
                if answer == sample.answer {
                    continue;
                }
                let Some(contract) = Self::ruliad_answer_contract(&answer) else {
                    continue;
                };
                if contract != sample.contract {
                    continue;
                }
                let key = (
                    answer.clone(),
                    sample.family.clone(),
                    sample.task_kind.clone(),
                    contract.clone(),
                );
                if !seen_template_negatives.insert(key) {
                    continue;
                }
                let Some((oracle_completion, _completion_text)) =
                    Self::ruliad_completion_tokens_from_answer(
                        &tokenizer,
                        &answer,
                        burn_dragon_universality::ruliad::RULIAD_V2_DOCUMENT_CLOSE_MARKER,
                        completion_budget,
                    )
                else {
                    continue;
                };
                negative_pool.push(NegativeSample {
                    current_source_index: None,
                    answer,
                    family: sample.family.clone(),
                    task_kind: sample.task_kind.clone(),
                    contract,
                    oracle_completion,
                    from_replay: false,
                    from_generated_attractor: false,
                    schema_negative: false,
                });
            }
            for answer in Self::ruliad_schema_collapse_negative_answers(&sample.answer) {
                if answer == sample.answer {
                    continue;
                }
                let Some(contract) = Self::ruliad_answer_contract(&answer) else {
                    continue;
                };
                let key = (
                    answer.clone(),
                    sample.family.clone(),
                    sample.task_kind.clone(),
                    contract.clone(),
                );
                if !seen_template_negatives.insert(key) {
                    continue;
                }
                let Some((oracle_completion, _completion_text)) =
                    Self::ruliad_completion_tokens_from_answer(
                        &tokenizer,
                        &answer,
                        burn_dragon_universality::ruliad::RULIAD_V2_DOCUMENT_CLOSE_MARKER,
                        completion_budget,
                    )
                else {
                    continue;
                };
                negative_pool.push(NegativeSample {
                    current_source_index: None,
                    answer,
                    family: sample.family.clone(),
                    task_kind: sample.task_kind.clone(),
                    contract,
                    oracle_completion,
                    from_replay: false,
                    from_generated_attractor: false,
                    schema_negative: true,
                });
            }
        }
        let generated_attractor_negative_pool_size = negative_pool
            .iter()
            .filter(|negative| negative.from_generated_attractor)
            .count();
        let negative_pool_size = negative_pool.len();

        let max_pairs = config.field_binding_contrast_max_pairs.max(1);
        let mut candidates_by_oracle = (0..eligible.len())
            .map(|_| Vec::<ContrastCandidate>::new())
            .collect::<Vec<_>>();
        let mut candidate_pairs = 0usize;
        let mut filtered_presented_action_candidates = 0usize;
        for (oracle_index, oracle) in eligible.iter().enumerate() {
            for (negative_index, negative) in negative_pool.iter().enumerate() {
                if negative.current_source_index == Some(oracle.source_index)
                    || oracle.answer == negative.answer
                    || oracle.family != negative.family
                    || oracle.task_kind != negative.task_kind
                    || (!negative.schema_negative && oracle.contract != negative.contract)
                {
                    continue;
                }
                if !negative.schema_negative
                    && oracle
                        .presented_action_answers
                        .as_ref()
                        .is_some_and(|answers| answers.contains(negative.answer.trim()))
                {
                    filtered_presented_action_candidates =
                        filtered_presented_action_candidates.saturating_add(1);
                    continue;
                }
                let diff_len = oracle
                    .oracle_completion
                    .len()
                    .min(negative.oracle_completion.len());
                let mut discriminative_tokens = 0usize;
                for completion_index in 0..diff_len {
                    let active = if negative.schema_negative {
                        oracle
                            .schema_mask
                            .get(completion_index)
                            .copied()
                            .unwrap_or(false)
                    } else {
                        oracle
                            .value_mask
                            .get(completion_index)
                            .copied()
                            .unwrap_or(false)
                    };
                    if active
                        && oracle.oracle_completion[completion_index]
                            != negative.oracle_completion[completion_index]
                    {
                        discriminative_tokens = discriminative_tokens.saturating_add(1);
                    }
                }
                if discriminative_tokens == 0 {
                    continue;
                }
                candidates_by_oracle[oracle_index].push(ContrastCandidate {
                    oracle_index,
                    negative_index,
                    negative_source_index: negative.current_source_index,
                    negative_answer: negative.answer.clone(),
                    discriminative_tokens,
                    from_replay: negative.from_replay,
                    from_generated_attractor: negative.from_generated_attractor,
                    schema_negative: negative.schema_negative,
                });
                candidate_pairs = candidate_pairs.saturating_add(1);
            }
        }
        let candidate_priority = |candidate: &ContrastCandidate| {
            if candidate.from_generated_attractor {
                0usize
            } else if candidate.schema_negative {
                2
            } else {
                1
            }
        };
        for candidates in candidates_by_oracle.iter_mut() {
            candidates.sort_by(|left, right| {
                candidate_priority(left)
                    .cmp(&candidate_priority(right))
                    .then_with(|| right.discriminative_tokens.cmp(&left.discriminative_tokens))
                    .then_with(|| left.from_replay.cmp(&right.from_replay))
                    .then_with(|| left.negative_index.cmp(&right.negative_index))
            });
        }
        // Spend the bounded auxiliary batch across prompts before taking a second negative for any
        // prompt. A global top-k here repeatedly trained only the rows with the longest byte-level
        // answer differences and left most prompts without a binding gradient.
        let mut selected_candidates = Vec::<ContrastCandidate>::new();
        let mut rank = 0usize;
        while selected_candidates.len() < max_pairs {
            let mut selected_this_round = 0usize;
            for candidates in candidates_by_oracle.iter() {
                if let Some(candidate) = candidates.get(rank) {
                    selected_candidates.push(candidate.clone());
                    selected_this_round = selected_this_round.saturating_add(1);
                    if selected_candidates.len() == max_pairs {
                        break;
                    }
                }
            }
            if selected_this_round == 0 {
                break;
            }
            rank = rank.saturating_add(1);
        }
        let mut rows = Vec::<ContrastRow>::new();
        for candidate in selected_candidates {
            let oracle = &eligible[candidate.oracle_index];
            let Some((negative_completion, _negative_text)) =
                Self::ruliad_completion_tokens_from_answer(
                    &tokenizer,
                    &candidate.negative_answer,
                    policy_batch.samples[oracle.source_index]
                        .item
                        .document_close_marker(),
                    completion_budget,
                )
            else {
                continue;
            };
            let prompt = Self::ruliad_trim_prompt_for_completion(
                &oracle.prompt,
                oracle
                    .oracle_completion
                    .len()
                    .max(negative_completion.len()),
                block_size,
            );
            let Some((mut inputs, mut oracle_targets, _oracle_mask)) =
                Self::ruliad_policy_row_from_completion(&prompt, &oracle.oracle_completion)
            else {
                continue;
            };
            let completion_start = prompt.len().saturating_sub(1).min(oracle_targets.len());
            let diff_len = oracle
                .oracle_completion
                .len()
                .min(negative_completion.len());
            let mut negative_targets = oracle_targets.clone();
            let mut mask = vec![0i64; oracle_targets.len()];
            let mut first_discriminative_token = None;
            for (completion_index, (&oracle_token, &negative_token)) in oracle
                .oracle_completion
                .iter()
                .zip(&negative_completion)
                .take(diff_len)
                .enumerate()
            {
                let target_index = completion_start.saturating_add(completion_index);
                let active = if candidate.schema_negative {
                    oracle
                        .schema_mask
                        .get(completion_index)
                        .copied()
                        .unwrap_or(false)
                } else {
                    oracle
                        .value_mask
                        .get(completion_index)
                        .copied()
                        .unwrap_or(false)
                };
                if active && target_index < negative_targets.len() && oracle_token != negative_token
                {
                    negative_targets[target_index] = negative_token;
                    mask[target_index] = 1;
                    first_discriminative_token = Some(completion_index);
                    break;
                }
            }
            let Some(first_discriminative_token) = first_discriminative_token else {
                continue;
            };
            let causal_len = completion_start
                .saturating_add(first_discriminative_token)
                .saturating_add(1);
            inputs.truncate(causal_len);
            oracle_targets.truncate(causal_len);
            negative_targets.truncate(causal_len);
            mask.truncate(causal_len);
            rows.push(ContrastRow {
                prompt,
                oracle_completion: oracle.oracle_completion.clone(),
                negative_completion,
                inputs,
                oracle_targets,
                negative_targets,
                mask,
                discriminative_tokens: 1,
                source_index: oracle.source_index,
                negative_source_index: candidate.negative_source_index,
                from_replay: candidate.from_replay,
                from_generated_attractor: candidate.from_generated_attractor,
            });
        }

        if replay_capacity > 0
            && !eligible.is_empty()
            && let Ok(mut replay) = self.ruliad_field_binding_replay.lock()
        {
            for sample in eligible.iter() {
                replay.push_back(RuliadFieldBindingReplaySample {
                    answer: sample.answer.clone(),
                    family: sample.family.clone(),
                    task_kind: sample.task_kind.clone(),
                    contract: sample.contract.clone(),
                    oracle_completion: sample.oracle_completion.clone(),
                });
            }
            while replay.len() > replay_capacity {
                replay.pop_front();
            }
        }

        if rows.is_empty() {
            let replay_summary = self.ruliad_generated_attractor_summary();
            self.write_ruliad_generated_attractor_telemetry(
                RuliadGeneratedAttractorReplayTelemetry {
                    version: 1,
                    step_index,
                    source: "field_binding".to_string(),
                    skip_reason: self
                        .ruliad_generated_attractor_replay_skip_reason(
                            &replay_summary,
                            generated_attractor_negative_pool_size,
                        )
                        .or_else(|| Some("no_counterfactual_pairs".to_string())),
                    observed_completion_rows: 0,
                    recorded_attractor_rows: 0,
                    selected_candidate_rows: generated_attractor_negative_pool_size,
                    selected_field_binding_pairs: 0,
                    replay_pool_size: replay_summary.pool_size,
                    active_attractor_count: replay_summary.active_count,
                    active_observation_count: replay_summary.active_observation_count,
                    distinct_answer_count: replay_summary.distinct_answers,
                    dominant_answer_count: replay_summary.dominant_count,
                    dominant_answer_fraction: replay_summary.dominant_fraction(),
                    min_count: config.generated_attractor_replay_min_count.max(1),
                    max_candidates: config.generated_attractor_replay_max_candidates,
                    min_distinct_answers: config
                        .generated_attractor_replay_min_distinct_answers
                        .max(1),
                    max_dominant_fraction: config.generated_attractor_replay_max_dominant_fraction,
                },
            );
            self.write_ruliad_field_binding_contrast_telemetry(
                RuliadFieldBindingContrastTelemetry {
                    version: 3,
                    objective: RULIAD_FIELD_BINDING_OBJECTIVE,
                    step_index,
                    skip_reason: Some("no_counterfactual_pairs".to_string()),
                    sample_groups: eligible.len(),
                    oracle_prompt_count: 0,
                    prompt_pairs: 0,
                    contrast_pairs: 0,
                    candidate_pairs,
                    filtered_presented_action_candidates,
                    contrast_discriminative_tokens: 0,
                    negative_pool_size,
                    replay_pool_size,
                    replay_contrast_pairs: 0,
                    generated_attractor_pool_size,
                    generated_attractor_negative_pool_size,
                    generated_attractor_contrast_pairs: 0,
                    rank_metric_pairs: 0,
                    rank_metric_tokens: 0,
                    logit_margin_mean: None,
                    positive_token_fraction: None,
                    margin_satisfied_token_fraction: None,
                    exact_pair_rank_fraction: None,
                    exact_pair_margin_fraction: None,
                    sequence_rank_metric_pairs: 0,
                    sequence_log_probability_margin_mean: None,
                    positive_sequence_fraction: None,
                    sequence_margin_satisfied_fraction: None,
                    field_binding_contrast_weight: weight,
                    field_binding_contrast_margin: config.field_binding_contrast_margin,
                    field_binding_contrast_pair_weight: config.field_binding_contrast_pair_weight,
                },
            );
            return None;
        }

        let mut participating_samples = HashSet::<usize>::new();
        let mut oracle_prompts = HashSet::<usize>::new();
        for row in rows.iter() {
            oracle_prompts.insert(row.source_index);
            participating_samples.insert(row.source_index);
            if let Some(negative_source_index) = row.negative_source_index {
                participating_samples.insert(negative_source_index);
            }
        }
        let replay_contrast_pairs = rows.iter().filter(|row| row.from_replay).count();
        let generated_attractor_contrast_pairs = rows
            .iter()
            .filter(|row| row.from_generated_attractor)
            .count();
        let contrast_discriminative_tokens = rows
            .iter()
            .map(|row| row.discriminative_tokens)
            .sum::<usize>();

        let max_len = rows.iter().map(|row| row.inputs.len()).max()?.max(1);
        let row_count = rows.len();
        let mut input_values = vec![0i64; row_count * max_len];
        let mut oracle_target_values = vec![0i64; row_count * max_len];
        let mut negative_target_values = vec![0i64; row_count * max_len];
        let mut mask_values = vec![0i64; row_count * max_len];
        for (row_index, row) in rows.iter().enumerate() {
            let offset = row_index * max_len;
            let len = row.inputs.len().min(max_len);
            input_values[offset..offset + len].copy_from_slice(&row.inputs[..len]);
            oracle_target_values[offset..offset + len].copy_from_slice(&row.oracle_targets[..len]);
            negative_target_values[offset..offset + len]
                .copy_from_slice(&row.negative_targets[..len]);
            mask_values[offset..offset + len].copy_from_slice(&row.mask[..len]);
        }
        let inputs = Tensor::<B, 2, Int>::from_data(
            TensorData::new(input_values, [row_count, max_len]),
            device,
        );
        let oracle_targets = Tensor::<B, 2, Int>::from_data(
            TensorData::new(oracle_target_values, [row_count, max_len]),
            device,
        );
        let negative_targets = Tensor::<B, 2, Int>::from_data(
            TensorData::new(negative_target_values, [row_count, max_len]),
            device,
        );
        let mask = Tensor::<B, 2, Int>::from_data(
            TensorData::new(mask_values.clone(), [row_count, max_len]),
            device,
        );
        let logits = self.model.forward(inputs);
        let oracle_logits = selected_token_logits(logits.clone(), oracle_targets);
        let negative_logits = selected_token_logits(logits, negative_targets);
        let logit_margin = oracle_logits.clone() - negative_logits.clone();
        let contrast_margin = config.field_binding_contrast_margin.max(0.0);
        let should_collect_rank_metric = config.field_binding_contrast_rank_metric_every_steps > 0
            && step_index.is_multiple_of(config.field_binding_contrast_rank_metric_every_steps);
        let rank_stats = should_collect_rank_metric.then(|| {
            Self::ruliad_field_binding_rank_stats(
                logit_margin.clone(),
                &mask_values,
                row_count,
                max_len,
                contrast_margin as f64,
            )
        });
        let pair_weight = config.field_binding_contrast_pair_weight.max(0.0);
        let sequence_log_probability_margin = if pair_weight > f32::EPSILON {
            let prompts = rows
                .iter()
                .map(|row| row.prompt.clone())
                .collect::<Vec<_>>();
            let candidates = rows
                .iter()
                .map(|row| {
                    vec![
                        row.oracle_completion.clone(),
                        row.negative_completion.clone(),
                    ]
                })
                .collect::<Vec<_>>();
            let scores = crate::train::ruliad_policy::sequence_completion_score_tensor(
                &self.model,
                &prompts,
                &candidates,
                device,
            )
            .ok()?;
            if scores.group_sizes.iter().any(|group_size| *group_size != 2) {
                return None;
            }
            let scores = scores.mean_log_scores.reshape([row_count, 2]);
            let oracle_scores = scores
                .clone()
                .slice([0..row_count, 0..1])
                .reshape([row_count]);
            let negative_scores = scores.slice([0..row_count, 1..2]).reshape([row_count]);
            Some(oracle_scores - negative_scores)
        } else {
            None
        };
        let sequence_rank_stats = should_collect_rank_metric
            .then(|| {
                sequence_log_probability_margin.clone().map(|margin| {
                    Self::ruliad_field_binding_sequence_rank_stats(margin, contrast_margin as f64)
                })
            })
            .flatten();
        self.write_ruliad_field_binding_contrast_telemetry(RuliadFieldBindingContrastTelemetry {
            version: 3,
            objective: RULIAD_FIELD_BINDING_OBJECTIVE,
            step_index,
            skip_reason: None,
            sample_groups: participating_samples.len(),
            oracle_prompt_count: oracle_prompts.len(),
            prompt_pairs: row_count,
            contrast_pairs: row_count,
            candidate_pairs,
            filtered_presented_action_candidates,
            contrast_discriminative_tokens,
            negative_pool_size,
            replay_pool_size,
            replay_contrast_pairs,
            generated_attractor_pool_size,
            generated_attractor_negative_pool_size,
            generated_attractor_contrast_pairs,
            rank_metric_pairs: rank_stats.as_ref().map(|stats| stats.pairs).unwrap_or(0),
            rank_metric_tokens: rank_stats.as_ref().map(|stats| stats.tokens).unwrap_or(0),
            logit_margin_mean: rank_stats
                .as_ref()
                .and_then(|stats| stats.logit_margin_mean),
            positive_token_fraction: rank_stats
                .as_ref()
                .and_then(|stats| stats.positive_token_fraction),
            margin_satisfied_token_fraction: rank_stats
                .as_ref()
                .and_then(|stats| stats.margin_satisfied_token_fraction),
            exact_pair_rank_fraction: rank_stats
                .as_ref()
                .and_then(|stats| stats.exact_pair_rank_fraction),
            exact_pair_margin_fraction: rank_stats
                .as_ref()
                .and_then(|stats| stats.exact_pair_margin_fraction),
            sequence_rank_metric_pairs: sequence_rank_stats
                .as_ref()
                .map(|stats| stats.pairs)
                .unwrap_or(0),
            sequence_log_probability_margin_mean: sequence_rank_stats
                .as_ref()
                .and_then(|stats| stats.log_probability_margin_mean),
            positive_sequence_fraction: sequence_rank_stats
                .as_ref()
                .and_then(|stats| stats.positive_sequence_fraction),
            sequence_margin_satisfied_fraction: sequence_rank_stats
                .as_ref()
                .and_then(|stats| stats.margin_satisfied_sequence_fraction),
            field_binding_contrast_weight: weight,
            field_binding_contrast_margin: config.field_binding_contrast_margin,
            field_binding_contrast_pair_weight: config.field_binding_contrast_pair_weight,
        });
        let replay_summary = self.ruliad_generated_attractor_summary();
        self.write_ruliad_generated_attractor_telemetry(RuliadGeneratedAttractorReplayTelemetry {
            version: 1,
            step_index,
            source: "field_binding".to_string(),
            skip_reason: self.ruliad_generated_attractor_replay_skip_reason(
                &replay_summary,
                generated_attractor_negative_pool_size,
            ),
            observed_completion_rows: 0,
            recorded_attractor_rows: 0,
            selected_candidate_rows: generated_attractor_negative_pool_size,
            selected_field_binding_pairs: generated_attractor_contrast_pairs,
            replay_pool_size: replay_summary.pool_size,
            active_attractor_count: replay_summary.active_count,
            active_observation_count: replay_summary.active_observation_count,
            distinct_answer_count: replay_summary.distinct_answers,
            dominant_answer_count: replay_summary.dominant_count,
            dominant_answer_fraction: replay_summary.dominant_fraction(),
            min_count: config.generated_attractor_replay_min_count.max(1),
            max_candidates: config.generated_attractor_replay_max_candidates,
            min_distinct_answers: config
                .generated_attractor_replay_min_distinct_answers
                .max(1),
            max_dominant_fraction: config.generated_attractor_replay_max_dominant_fraction,
        });
        let token_loss = masked_token_mean(
            activation::softplus(
                negative_logits.clone() - oracle_logits.clone() + contrast_margin,
                1.0,
            ),
            Some(mask.clone()),
        );
        let loss = sequence_log_probability_margin.map_or(token_loss.clone(), |margin| {
            token_loss
                + activation::softplus(margin.mul_scalar(-1.0) + contrast_margin, 1.0)
                    .mean()
                    .reshape([1])
                    .mul_scalar(pair_weight)
        });
        Some(loss.mul_scalar(weight))
    }

    fn ruliad_structured_answer_contrast_loss(
        &self,
        policy_batch: &crate::dataset::RuliadPolicyBatch,
        device: &B::Device,
        block_size: usize,
    ) -> Option<Tensor<B, 1>> {
        let config = self.ruliad_supervision.verifier_reward;
        let weight = self.ruliad_structured_contrast_weight();
        if weight <= f32::EPSILON || policy_batch.samples.is_empty() || self.pipeline_enabled() {
            return None;
        }
        let tokenizer =
            burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer::from_config(
                &policy_batch.tokenization,
            )
            .ok()?;
        let completion_budget = config
            .max_completion_tokens
            .max(1)
            .min(block_size.saturating_sub(1).max(1));

        #[derive(Clone)]
        struct ContrastRow {
            inputs: Vec<i64>,
            oracle_targets: Vec<i64>,
            negative_targets: Vec<i64>,
            mask: Vec<i64>,
            discriminative_tokens: usize,
        }

        let mut rows = Vec::<ContrastRow>::new();
        let mut oracle_completion_rows = 0usize;
        let mut field_negative_completion_rows = 0usize;
        let mut template_negative_completion_rows = 0usize;
        let mut schema_negative_completion_rows = 0usize;
        let mut generated_attractor_negative_completion_rows = 0usize;
        let mut sample_groups = 0usize;
        for sample in policy_batch.samples.iter() {
            let mut prompt = sample.prompt_tokens.clone();
            if prompt.is_empty() {
                continue;
            }
            let Some((oracle_completion, _oracle_text, _truncated)) =
                Self::ruliad_oracle_completion_tokens(&tokenizer, sample, completion_budget)
            else {
                continue;
            };
            prompt = Self::ruliad_trim_prompt_for_completion(
                &prompt,
                oracle_completion.len(),
                block_size,
            );
            let value_mask = Self::ruliad_answer_value_completion_mask(
                &tokenizer,
                &sample.item.expected_answer,
                oracle_completion.len(),
            );
            let schema_mask = Self::ruliad_answer_schema_completion_mask(
                &tokenizer,
                &sample.item.expected_answer,
                oracle_completion.len(),
            );
            if !value_mask.iter().any(|active| *active) && !schema_mask.iter().any(|active| *active)
            {
                continue;
            }
            let Some((inputs, oracle_targets, _oracle_mask)) =
                Self::ruliad_policy_row_from_completion(&prompt, &oracle_completion)
            else {
                continue;
            };
            let completion_start = prompt.len().saturating_sub(1).min(oracle_targets.len());
            oracle_completion_rows = oracle_completion_rows.saturating_add(1);
            let mut sample_pair_count = 0usize;

            for (negative, negative_kind) in Self::ruliad_structured_negative_answers_with_schema(
                &sample.item.expected_answer,
                config.structured_negative_count,
                config.structured_template_negative_count,
                config.structured_schema_negative_count,
            ) {
                let Some((completion, _completion_text)) =
                    Self::ruliad_completion_tokens_from_answer(
                        &tokenizer,
                        &negative,
                        sample.item.document_close_marker(),
                        completion_budget,
                    )
                else {
                    continue;
                };
                let diff_len = oracle_completion.len().min(completion.len());
                let mut negative_targets = oracle_targets.clone();
                let mut mask = vec![0i64; oracle_targets.len()];
                let mut discriminative_tokens = 0usize;
                for completion_index in 0..diff_len {
                    let target_index = completion_start.saturating_add(completion_index);
                    let active = match negative_kind {
                        RuliadStructuredNegativeKind::SchemaCollapse => {
                            schema_mask.get(completion_index).copied().unwrap_or(false)
                        }
                        RuliadStructuredNegativeKind::FieldMutation
                        | RuliadStructuredNegativeKind::TemplateCollapse => {
                            value_mask.get(completion_index).copied().unwrap_or(false)
                        }
                    };
                    if active
                        && target_index < negative_targets.len()
                        && oracle_completion[completion_index] != completion[completion_index]
                    {
                        negative_targets[target_index] = completion[completion_index];
                        mask[target_index] = 1;
                        discriminative_tokens = discriminative_tokens.saturating_add(1);
                    }
                }
                if discriminative_tokens == 0 {
                    continue;
                }
                rows.push(ContrastRow {
                    inputs: inputs.clone(),
                    oracle_targets: oracle_targets.clone(),
                    negative_targets,
                    mask,
                    discriminative_tokens,
                });
                sample_pair_count = sample_pair_count.saturating_add(1);
                match negative_kind {
                    RuliadStructuredNegativeKind::FieldMutation => {
                        field_negative_completion_rows =
                            field_negative_completion_rows.saturating_add(1);
                    }
                    RuliadStructuredNegativeKind::TemplateCollapse => {
                        template_negative_completion_rows =
                            template_negative_completion_rows.saturating_add(1);
                    }
                    RuliadStructuredNegativeKind::SchemaCollapse => {
                        schema_negative_completion_rows =
                            schema_negative_completion_rows.saturating_add(1);
                    }
                }
            }
            let expected_contract = Self::ruliad_answer_contract(&sample.item.expected_answer);
            for entry in self.ruliad_generated_attractor_candidates_for_sample(sample) {
                let Some((completion, _completion_text)) =
                    Self::ruliad_completion_tokens_from_answer(
                        &tokenizer,
                        &entry.key.answer,
                        sample.item.document_close_marker(),
                        completion_budget,
                    )
                else {
                    continue;
                };
                let schema_negative = expected_contract
                    .as_ref()
                    .is_some_and(|contract| contract != &entry.key.contract);
                let diff_len = oracle_completion.len().min(completion.len());
                let mut negative_targets = oracle_targets.clone();
                let mut mask = vec![0i64; oracle_targets.len()];
                let mut discriminative_tokens = 0usize;
                for completion_index in 0..diff_len {
                    let target_index = completion_start.saturating_add(completion_index);
                    let active = if schema_negative {
                        schema_mask.get(completion_index).copied().unwrap_or(false)
                    } else {
                        value_mask.get(completion_index).copied().unwrap_or(false)
                    };
                    if active
                        && target_index < negative_targets.len()
                        && oracle_completion[completion_index] != completion[completion_index]
                    {
                        negative_targets[target_index] = completion[completion_index];
                        mask[target_index] = 1;
                        discriminative_tokens = discriminative_tokens.saturating_add(1);
                    }
                }
                if discriminative_tokens == 0 {
                    continue;
                }
                rows.push(ContrastRow {
                    inputs: inputs.clone(),
                    oracle_targets: oracle_targets.clone(),
                    negative_targets,
                    mask,
                    discriminative_tokens,
                });
                sample_pair_count = sample_pair_count.saturating_add(1);
                generated_attractor_negative_completion_rows =
                    generated_attractor_negative_completion_rows.saturating_add(1);
            }
            sample_groups = sample_groups.saturating_add(usize::from(sample_pair_count > 0));
        }
        if rows.is_empty() {
            self.write_ruliad_structured_contrast_telemetry(RuliadStructuredContrastTelemetry {
                version: 1,
                step_index: self.gradient_scale_step.load(Ordering::Relaxed),
                skip_reason: Some("no_field_value_pairs".to_string()),
                sample_groups,
                oracle_completion_rows,
                field_negative_completion_rows,
                template_negative_completion_rows,
                schema_negative_completion_rows,
                generated_attractor_negative_completion_rows,
                contrast_pairs: 0,
                contrast_discriminative_tokens: 0,
                structured_contrast_weight: weight,
                structured_contrast_margin: config.structured_contrast_margin,
            });
            return None;
        }
        let contrast_discriminative_tokens = rows
            .iter()
            .map(|row| row.discriminative_tokens)
            .sum::<usize>();
        self.write_ruliad_structured_contrast_telemetry(RuliadStructuredContrastTelemetry {
            version: 1,
            step_index: self.gradient_scale_step.load(Ordering::Relaxed),
            skip_reason: None,
            sample_groups,
            oracle_completion_rows,
            field_negative_completion_rows,
            template_negative_completion_rows,
            schema_negative_completion_rows,
            generated_attractor_negative_completion_rows,
            contrast_pairs: rows.len(),
            contrast_discriminative_tokens,
            structured_contrast_weight: weight,
            structured_contrast_margin: config.structured_contrast_margin,
        });

        let max_len = rows.iter().map(|row| row.inputs.len()).max()?.max(1);
        let row_count = rows.len();
        let mut input_values = vec![0i64; row_count * max_len];
        let mut oracle_target_values = vec![0i64; row_count * max_len];
        let mut negative_target_values = vec![0i64; row_count * max_len];
        let mut mask_values = vec![0i64; row_count * max_len];
        for (row_index, row) in rows.into_iter().enumerate() {
            let offset = row_index * max_len;
            let len = row.inputs.len().min(max_len);
            input_values[offset..offset + len].copy_from_slice(&row.inputs[..len]);
            oracle_target_values[offset..offset + len].copy_from_slice(&row.oracle_targets[..len]);
            negative_target_values[offset..offset + len]
                .copy_from_slice(&row.negative_targets[..len]);
            mask_values[offset..offset + len].copy_from_slice(&row.mask[..len]);
        }
        let inputs = Tensor::<B, 2, Int>::from_data(
            TensorData::new(input_values, [row_count, max_len]),
            device,
        );
        let oracle_targets = Tensor::<B, 2, Int>::from_data(
            TensorData::new(oracle_target_values, [row_count, max_len]),
            device,
        );
        let negative_targets = Tensor::<B, 2, Int>::from_data(
            TensorData::new(negative_target_values, [row_count, max_len]),
            device,
        );
        let mask = Tensor::<B, 2, Int>::from_data(
            TensorData::new(mask_values, [row_count, max_len]),
            device,
        );
        let logits = self.model.forward(inputs);
        let oracle_logits = selected_token_logits(logits.clone(), oracle_targets);
        let negative_logits = selected_token_logits(logits, negative_targets);
        Some(
            masked_token_mean(
                activation::softplus(
                    negative_logits - oracle_logits + config.structured_contrast_margin.max(0.0),
                    1.0,
                ),
                Some(mask),
            )
            .mul_scalar(weight),
        )
    }

    fn ruliad_verifier_rollout_imitation_loss(
        &self,
        policy_batch: &crate::dataset::RuliadPolicyBatch,
        device: &B::Device,
        block_size: usize,
    ) -> Option<Tensor<B, 1>> {
        let config = self.ruliad_supervision.verifier_reward;
        if !self.ruliad_verifier_rollout_feedback_active()
            || policy_batch.samples.is_empty()
            || self.pipeline_enabled()
        {
            return None;
        }
        let imitation_weight = config.rollout_imitation_weight.max(0.0);
        let recovery_weight = config.rollout_recovery_weight.max(0.0);
        let tokenizer =
            burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer::from_config(
                &policy_batch.tokenization,
            )
            .ok()?;
        let completion_budget = config
            .max_completion_tokens
            .max(1)
            .min(block_size.saturating_sub(1).max(1));
        let prompt_budget = block_size.saturating_sub(completion_budget).max(1);
        let max_rows = config.rollout_imitation_max_rows_per_step.max(1);
        let step_index = self.gradient_scale_step.load(Ordering::Relaxed);

        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        enum RolloutFeedbackKind {
            Imitation,
            Recovery,
        }

        #[derive(Clone)]
        struct RolloutFeedbackRow {
            inputs: Vec<i64>,
            targets: Vec<i64>,
            mask: Vec<f32>,
            weight: f32,
            kind: RolloutFeedbackKind,
        }

        let mut rows = Vec::<RolloutFeedbackRow>::new();
        let mut sample_groups = 0usize;
        let mut generated_completion_rows = 0usize;
        let mut recorded_attractor_rows = 0usize;
        let mut verifier_match_rows = 0usize;
        let mut semantic_match_rows = 0usize;
        let mut partial_rows = 0usize;
        let mut schema_wrong_rows = 0usize;
        let mut malformed_rows = 0usize;
        let mut missing_rows = 0usize;
        let mut recovery_partial_rows = 0usize;
        let mut recovery_schema_wrong_rows = 0usize;
        let mut recovery_malformed_rows = 0usize;
        let mut recovery_missing_rows = 0usize;
        let mut field_accuracy_sum = 0.0f64;
        let mut partial_progress_sum = 0.0f64;
        let mut completion_quality_sum = 0.0f64;

        'samples: for sample in policy_batch.samples.iter() {
            let mut prompt = sample.prompt_tokens.clone();
            if prompt.is_empty() {
                continue;
            }
            if prompt.len() > prompt_budget {
                prompt = prompt[prompt.len() - prompt_budget..].to_vec();
            }
            let oracle_row = if recovery_weight > f32::EPSILON {
                Self::ruliad_oracle_completion_tokens(&tokenizer, sample, completion_budget)
                    .and_then(|(oracle_completion, _oracle_text, _truncated)| {
                        Self::ruliad_policy_row_from_completion(&prompt, &oracle_completion)
                    })
            } else {
                None
            };
            let mut generated_for_sample = 0usize;
            for group_index in 0..config.group_size.max(1) {
                if rows.len() >= max_rows {
                    break 'samples;
                }
                let rollout_seed = Self::mix_ruliad_policy_seed(
                    (step_index as u64).rotate_left(17)
                        ^ (sample.item.sample_index as u64).rotate_left(7)
                        ^ group_index as u64,
                );
                let generated = crate::generation::generate_tokens_seeded(
                    &self.model,
                    prompt.clone(),
                    device,
                    crate::generation::GenerationSettings {
                        max_new_tokens: Some(completion_budget),
                        temperature: config.temperature,
                        top_k: Some(config.top_k),
                        strategy: crate::generation::ContextStrategy::Infinite,
                        stop_on_token: policy_batch.stop_token_id,
                    },
                    rollout_seed,
                    None,
                )
                .ok()?;
                if generated.len() <= prompt.len() {
                    continue;
                }
                let completion = generated[prompt.len()..].to_vec();
                if completion.is_empty() {
                    continue;
                }
                let completion_tokens = completion
                    .iter()
                    .filter_map(|token| u32::try_from(*token).ok())
                    .collect::<Vec<_>>();
                let completion_text = tokenizer.decode_payload(&completion_tokens, true);
                let score = burn_dragon_universality::ruliad::score_ruliad_item_completion(
                    &sample.item,
                    Some(&completion_text),
                );
                generated_completion_rows = generated_completion_rows.saturating_add(1);
                recorded_attractor_rows = recorded_attractor_rows.saturating_add(usize::from(
                    self.record_ruliad_generated_attractor(
                        sample,
                        &completion_text,
                        &score,
                        step_index,
                    ),
                ));
                generated_for_sample = generated_for_sample.saturating_add(1);
                match score.status {
                    burn_dragon_universality::ruliad::RuliadAnswerStatus::VerifierMatch => {
                        verifier_match_rows = verifier_match_rows.saturating_add(1)
                    }
                    burn_dragon_universality::ruliad::RuliadAnswerStatus::SemanticMatch => {
                        semantic_match_rows = semantic_match_rows.saturating_add(1)
                    }
                    burn_dragon_universality::ruliad::RuliadAnswerStatus::Partial => {
                        partial_rows = partial_rows.saturating_add(1)
                    }
                    burn_dragon_universality::ruliad::RuliadAnswerStatus::SchemaValidWrong => {
                        schema_wrong_rows = schema_wrong_rows.saturating_add(1)
                    }
                    burn_dragon_universality::ruliad::RuliadAnswerStatus::Malformed => {
                        malformed_rows = malformed_rows.saturating_add(1)
                    }
                    burn_dragon_universality::ruliad::RuliadAnswerStatus::Missing => {
                        missing_rows = missing_rows.saturating_add(1)
                    }
                }
                let field_accuracy = if score.expected_field_count == 0 {
                    0.0
                } else {
                    score.correct_field_count as f64 / score.expected_field_count as f64
                };
                field_accuracy_sum += field_accuracy;
                partial_progress_sum += score.partial_progress_ppm as f64 / 1_000_000.0;
                completion_quality_sum += score.completion_quality_ppm as f64 / 1_000_000.0;
                let has_imitation_signal = Self::ruliad_score_has_policy_correctness_signal(
                    &score,
                    config.rollout_imitation_min_partial_progress_ppm,
                    config.rollout_imitation_min_completion_quality_ppm,
                );
                let has_recovery_signal = Self::ruliad_score_has_rollout_recovery_signal(
                    &score,
                    config.rollout_imitation_min_partial_progress_ppm,
                    config.rollout_imitation_min_completion_quality_ppm,
                );
                if !has_imitation_signal && !has_recovery_signal {
                    continue;
                }
                let is_correct = matches!(
                    score.status,
                    burn_dragon_universality::ruliad::RuliadAnswerStatus::VerifierMatch
                        | burn_dragon_universality::ruliad::RuliadAnswerStatus::SemanticMatch
                );
                if imitation_weight > f32::EPSILON
                    && let Some((inputs, targets, mask)) =
                        Self::ruliad_policy_row_from_completion(&prompt, &completion)
                {
                    rows.push(RolloutFeedbackRow {
                        inputs,
                        targets,
                        mask,
                        weight: imitation_weight,
                        kind: RolloutFeedbackKind::Imitation,
                    });
                }
                if recovery_weight > f32::EPSILON
                    && !is_correct
                    && has_recovery_signal
                    && let Some((oracle_inputs, oracle_targets, oracle_mask)) = oracle_row.as_ref()
                {
                    let completion_start = prompt.len().saturating_sub(1).min(oracle_inputs.len());
                    let mut corrupted_inputs = oracle_inputs.clone();
                    for (index, value) in corrupted_inputs
                        .iter_mut()
                        .enumerate()
                        .skip(completion_start)
                    {
                        let completion_index = index - completion_start;
                        if let Some(generated_token) = completion.get(completion_index) {
                            *value = *generated_token;
                        }
                    }
                    rows.push(RolloutFeedbackRow {
                        inputs: corrupted_inputs,
                        targets: oracle_targets.clone(),
                        mask: oracle_mask.clone(),
                        weight: recovery_weight,
                        kind: RolloutFeedbackKind::Recovery,
                    });
                    match score.status {
                        burn_dragon_universality::ruliad::RuliadAnswerStatus::Partial => {
                            recovery_partial_rows = recovery_partial_rows.saturating_add(1)
                        }
                        burn_dragon_universality::ruliad::RuliadAnswerStatus::SchemaValidWrong => {
                            recovery_schema_wrong_rows =
                                recovery_schema_wrong_rows.saturating_add(1)
                        }
                        burn_dragon_universality::ruliad::RuliadAnswerStatus::Malformed => {
                            recovery_malformed_rows = recovery_malformed_rows.saturating_add(1)
                        }
                        burn_dragon_universality::ruliad::RuliadAnswerStatus::Missing => {
                            recovery_missing_rows = recovery_missing_rows.saturating_add(1)
                        }
                        burn_dragon_universality::ruliad::RuliadAnswerStatus::VerifierMatch
                        | burn_dragon_universality::ruliad::RuliadAnswerStatus::SemanticMatch => {}
                    }
                }
            }
            sample_groups += usize::from(generated_for_sample > 0);
        }

        let rate_ppm = |count: usize| -> usize {
            count
                .saturating_mul(1_000_000)
                .checked_div(generated_completion_rows)
                .unwrap_or_default()
        };
        let verifier_rate_ppm = rate_ppm(verifier_match_rows.saturating_add(semantic_match_rows));
        let schema_wrong_rate_ppm = rate_ppm(schema_wrong_rows);
        let malformed_rate_ppm = rate_ppm(malformed_rows);
        let candidate_completion_rows = rows.len();
        let health_gate_passed = generated_completion_rows > 0
            && verifier_rate_ppm >= config.rollout_imitation_min_verifier_rate_ppm
            && schema_wrong_rate_ppm <= config.rollout_imitation_max_schema_wrong_rate_ppm
            && malformed_rate_ppm <= config.rollout_imitation_max_malformed_rate_ppm;
        if !health_gate_passed {
            rows.retain(|row| row.kind == RolloutFeedbackKind::Recovery);
        }
        let accepted_imitation_rows = rows
            .iter()
            .filter(|row| row.kind == RolloutFeedbackKind::Imitation)
            .count();
        let accepted_recovery_rows = rows
            .iter()
            .filter(|row| row.kind == RolloutFeedbackKind::Recovery)
            .count();
        let skip_reason = if generated_completion_rows == 0 {
            Some("no_generated_completion".to_string())
        } else if candidate_completion_rows == 0 {
            Some("no_candidate_completion".to_string())
        } else if rows.is_empty() && !health_gate_passed {
            Some("rollout_health_gate".to_string())
        } else if rows.is_empty() {
            Some("no_accepted_completion".to_string())
        } else {
            None
        };

        let denominator = generated_completion_rows.max(1) as f64;
        self.write_ruliad_verifier_rollout_telemetry(RuliadVerifierRolloutImitationTelemetry {
            version: 1,
            step_index,
            skip_reason,
            sample_groups,
            generated_completion_rows,
            candidate_completion_rows,
            accepted_completion_rows: rows.len(),
            accepted_imitation_rows,
            accepted_recovery_rows,
            health_gate_passed,
            verifier_rate_ppm,
            schema_wrong_rate_ppm,
            malformed_rate_ppm,
            verifier_match_rows,
            semantic_match_rows,
            partial_rows,
            schema_wrong_rows,
            malformed_rows,
            missing_rows,
            recovery_partial_rows,
            recovery_schema_wrong_rows,
            recovery_malformed_rows,
            recovery_missing_rows,
            field_accuracy_mean: field_accuracy_sum / denominator,
            partial_progress_mean: partial_progress_sum / denominator,
            completion_quality_mean: completion_quality_sum / denominator,
            rollout_imitation_weight: imitation_weight,
            rollout_recovery_weight: recovery_weight,
            max_completion_tokens: completion_budget,
        });
        let replay_summary = self.ruliad_generated_attractor_summary();
        self.write_ruliad_generated_attractor_telemetry(RuliadGeneratedAttractorReplayTelemetry {
            version: 1,
            step_index,
            source: "rollout".to_string(),
            skip_reason: (generated_completion_rows == 0).then(|| "no_generated_rows".to_string()),
            observed_completion_rows: generated_completion_rows,
            recorded_attractor_rows,
            selected_candidate_rows: 0,
            selected_field_binding_pairs: 0,
            replay_pool_size: replay_summary.pool_size,
            active_attractor_count: replay_summary.active_count,
            active_observation_count: replay_summary.active_observation_count,
            distinct_answer_count: replay_summary.distinct_answers,
            dominant_answer_count: replay_summary.dominant_count,
            dominant_answer_fraction: replay_summary.dominant_fraction(),
            min_count: config.generated_attractor_replay_min_count.max(1),
            max_candidates: config.generated_attractor_replay_max_candidates,
            min_distinct_answers: config
                .generated_attractor_replay_min_distinct_answers
                .max(1),
            max_dominant_fraction: config.generated_attractor_replay_max_dominant_fraction,
        });

        if rows.is_empty() {
            return None;
        }
        let max_len = rows.iter().map(|row| row.inputs.len()).max()?.max(1);
        let row_count = rows.len();
        let mut input_values = vec![0i64; row_count * max_len];
        let mut target_values = vec![0i64; row_count * max_len];
        let mut active_mask_values = vec![0.0f32; row_count * max_len];
        let mut weighted_mask_values = vec![0.0f32; row_count * max_len];
        for (row_index, row) in rows.into_iter().enumerate() {
            let offset = row_index * max_len;
            let len = row.inputs.len().min(max_len);
            input_values[offset..offset + len].copy_from_slice(&row.inputs[..len]);
            target_values[offset..offset + len].copy_from_slice(&row.targets[..len]);
            for (mask_index, value) in row.mask.iter().copied().take(len).enumerate() {
                active_mask_values[offset + mask_index] = value;
                weighted_mask_values[offset + mask_index] = value * row.weight;
            }
        }
        let inputs = Tensor::<B, 2, Int>::from_data(
            TensorData::new(input_values, [row_count, max_len]),
            device,
        );
        let targets = Tensor::<B, 2, Int>::from_data(
            TensorData::new(target_values, [row_count, max_len]),
            device,
        );
        let active_mask = Tensor::<B, 2>::from_data(
            TensorData::new(active_mask_values, [row_count, max_len]),
            device,
        );
        let weighted_mask = Tensor::<B, 2>::from_data(
            TensorData::new(weighted_mask_values, [row_count, max_len]),
            device,
        );
        let logits = self.model.forward(inputs);
        let log_probs = log_probs_from_logits(logits);
        let token_log_probs = selected_token_log_probs(log_probs, targets);
        let active = active_mask.sum().reshape([1]).clamp_min(1.0);
        Some(
            (token_log_probs * weighted_mask)
                .sum()
                .reshape([1])
                .div(active)
                .mul_scalar(-1.0),
        )
    }

    fn ruliad_proof_policy_dagger_loss(
        &self,
        policy_batch: &crate::dataset::RuliadPolicyBatch,
        device: &B::Device,
        block_size: usize,
    ) -> Option<Tensor<B, 1>>
    where
        B: AutodiffBackend,
    {
        let config = self.ruliad_supervision.proof_policy;
        let weight = self.ruliad_proof_policy_dagger_weight();
        if weight <= f32::EPSILON || policy_batch.samples.is_empty() || self.pipeline_enabled() {
            return None;
        }
        let tokenizer =
            burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer::from_config(
                &policy_batch.tokenization,
            )
            .ok()?;
        let completion_budget = config
            .max_completion_tokens
            .max(1)
            .min(block_size.saturating_sub(1).max(1));
        let step_index = self.gradient_scale_step.load(Ordering::Relaxed);
        let effective_mode = config.effective_mode(step_index);
        let semantic_row_budget = config.semantic_rows_per_update();
        let base_semantic_row_budget = config.base_semantic_rows_per_update();
        let batch_plan = RuliadProofPolicyBatchPlan::new(
            effective_mode,
            base_semantic_row_budget,
            config.rollout_steps,
            config.stratified_difficulty_levels,
        );
        let trajectory_budget = batch_plan.trajectory_budget();
        let sampling_model_started = Instant::now();
        let sampling_model = (batch_plan.dagger_trajectory_budget > 0).then(|| {
            self.model
                .valid()
                .materialize_random_scaffold_for_inference()
        });
        let sampling_model_materialize_ms =
            sampling_model_started.elapsed().as_micros() as f64 / 1_000.0;

        #[derive(Clone)]
        enum ExpertRowObjective {
            PresentationIndex {
                inputs: Vec<i64>,
                branch_position: usize,
                candidate_target_tokens: Vec<i64>,
                equivalent_target_tokens: Vec<i64>,
            },
            SemanticStep {
                prompt: Vec<i64>,
                candidate_completions: Vec<Vec<i64>>,
                equivalent_indices: Vec<usize>,
            },
        }

        #[derive(Clone)]
        struct ExpertRow {
            objective: ExpertRowObjective,
            presentation_weight: f32,
        }

        struct PrefixBranchRow {
            inputs: Vec<i64>,
            branch_position: usize,
            candidate_target_tokens: Vec<i64>,
            equivalent_target_tokens: Vec<i64>,
            weight: f32,
        }

        let mut rows = Vec::<ExpertRow>::new();
        let mut visited_prompts = HashSet::<Vec<i64>>::new();
        let mut available_sample_groups = 0usize;
        let mut sample_groups = 0usize;
        let mut nonzero_start_trajectories = 0usize;
        let mut start_step_sum = 0usize;
        let mut visited_states = 0usize;
        let mut semantic_state_rows = 0usize;
        let mut base_semantic_state_rows = 0usize;
        let mut counterfactual_semantic_state_rows = 0usize;
        let mut counterfactual_target_shortfall = 0usize;
        let mut static_expert_rows = 0usize;
        let mut dagger_expert_rows = 0usize;
        let mut model_visited_expert_rows = 0usize;
        let mut model_valid_actions = 0usize;
        let mut model_invalid_actions = 0usize;
        let mut model_expert_equivalent_actions = 0usize;
        let mut model_off_expert_actions = 0usize;
        let mut repeated_states = 0usize;
        let mut model_backtracks = 0usize;
        let mut model_scoring_batches = 0usize;
        let mut maximum_model_scoring_batch_rows = 0usize;
        let mut model_scoring_padded_tokens = 0usize;
        let mut rollout_cpu_prepare_ms = 0.0f64;
        let mut model_scoring_ms = 0.0f64;
        let mut difficulty_sample_groups = BTreeMap::<usize, usize>::new();
        let mut difficulty_visited_states = BTreeMap::<usize, usize>::new();
        let mut difficulty_expert_rows = BTreeMap::<usize, usize>::new();
        let mut expert_selected_index_histogram = BTreeMap::<usize, usize>::new();
        let mut expert_equivalent_index_histogram = BTreeMap::<usize, usize>::new();
        let mut model_selected_index_histogram = BTreeMap::<usize, usize>::new();
        let mut candidate_target_tokens = 0usize;
        let mut equivalent_target_tokens = 0usize;
        let mut supervised_action_tokens = 0usize;
        let mut rollout_depth_reached = 0usize;
        let mut presentation_budget_exhausted = false;

        struct DaggerTrajectory {
            sample_index: usize,
            difficulty_level: usize,
            is_dagger: bool,
            max_depth: usize,
            answer_contract: burn_dragon_universality::ruliad::RuliadProofActionAnswerContract,
            state: burn_dragon_universality::ruliad::RuliadProofPolicyState,
        }

        struct DaggerExpansion {
            trajectory_index: usize,
            actions: burn_dragon_universality::ruliad::RuliadProofActionSet,
            presentations: Vec<DaggerScoringPresentation>,
        }

        struct DaggerScoringPresentation {
            rotation: usize,
            prompt: Vec<i64>,
            candidate_completions: Vec<Vec<i64>>,
            answer_contract: burn_dragon_universality::ruliad::RuliadProofActionAnswerContract,
        }

        struct PreparedExpertState {
            canonical_prompt: Vec<i64>,
            presentation_rows: Vec<ExpertRow>,
            scoring_presentations: Vec<DaggerScoringPresentation>,
            presentation_selected_indices: Vec<usize>,
            presentation_equivalent_indices: Vec<Vec<usize>>,
        }

        let prepare_expert_state = |
            problem: &burn_dragon_universality::ruliad::RuliadProofProblem,
            actions: &burn_dragon_universality::ruliad::RuliadProofActionSet,
            presentation_index: usize,
            scoring_contract: burn_dragon_universality::ruliad::RuliadProofActionAnswerContract,
            base_rotations: Option<&[usize]>,
        | -> Option<PreparedExpertState> {
            let rotations = crate::train::ruliad_policy::target_group_presentation_rotations(
                config.candidate_symmetry,
                actions.selected_index,
                actions.candidates.len(),
                presentation_index,
                base_rotations,
            )
            .ok()?;
            let canonical_prompt = tokenizer
                .encode_payload(
                    &burn_dragon_universality::ruliad::ruliad_proof_action_prompt(
                        problem, actions,
                    )
                    .ok()?,
                )
                .into_iter()
                .map(i64::from)
                .collect::<Vec<_>>();
            let presentation_weight = 1.0 / rotations.len().max(1) as f32;
            let mut presentation_rows = Vec::<ExpertRow>::with_capacity(rotations.len());
            let mut scoring_presentations =
                Vec::<DaggerScoringPresentation>::with_capacity(rotations.len());
            let mut presentation_selected_indices = Vec::<usize>::with_capacity(rotations.len());
            let mut presentation_equivalent_indices =
                Vec::<Vec<usize>>::with_capacity(rotations.len());
            for rotation in rotations {
                let presented_actions = actions.rotate_left(rotation).ok()?;
                let prompt_text =
                    burn_dragon_universality::ruliad::ruliad_proof_action_prompt(
                        problem,
                        &presented_actions,
                    )
                    .ok()?;
                let candidate_completions = (0..presented_actions.candidates.len())
                    .map(|candidate_index| {
                        let answer = burn_dragon_universality::ruliad::proof_action_answer(
                            &presented_actions,
                            candidate_index,
                            scoring_contract,
                        )
                        .ok()?;
                        Some(
                            tokenizer
                                .encode_payload(&answer)
                                .into_iter()
                                .map(i64::from)
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Option<Vec<_>>>()?;
                if candidate_completions.iter().any(|completion| {
                    completion.is_empty() || completion.len() > completion_budget
                }) {
                    return None;
                }
                let expert_completion = candidate_completions
                    .get(presented_actions.selected_index)
                    .cloned()?;
                if presented_actions.equivalent_indices.is_empty()
                    || presented_actions
                        .equivalent_indices
                        .iter()
                        .any(|index| *index >= candidate_completions.len())
                {
                    return None;
                }
                let prompt = tokenizer
                    .encode_payload(&prompt_text)
                    .into_iter()
                    .map(i64::from)
                    .collect::<Vec<_>>();
                let prompt = Self::ruliad_trim_prompt_for_completion(
                    &prompt,
                    candidate_completions
                        .iter()
                        .map(Vec::len)
                        .max()
                        .unwrap_or(expert_completion.len()),
                    block_size,
                );
                if prompt.is_empty() {
                    return None;
                }
                let objective = match scoring_contract {
                    burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::PresentationIndex => {
                        let branch_token_index = crate::train::ruliad_policy::candidate_branch_index(
                            &candidate_completions,
                        )
                        .ok()?;
                        let equivalent_tokens = presented_actions
                            .equivalent_indices
                            .iter()
                            .filter_map(|candidate_index| candidate_completions.get(*candidate_index))
                            .filter_map(|completion| completion.get(branch_token_index).copied())
                            .collect::<std::collections::BTreeSet<_>>()
                            .into_iter()
                            .collect::<Vec<_>>();
                        let candidate_tokens = candidate_completions
                            .iter()
                            .filter_map(|completion| completion.get(branch_token_index).copied())
                            .collect::<std::collections::BTreeSet<_>>()
                            .into_iter()
                            .collect::<Vec<_>>();
                        if equivalent_tokens.is_empty()
                            || candidate_tokens.len() != candidate_completions.len()
                        {
                            return None;
                        }
                        let (inputs, targets, mask) =
                            Self::ruliad_policy_row_from_completion_token(
                                &prompt,
                                &expert_completion,
                                branch_token_index,
                            )?;
                        let branch_position = mask.iter().position(|value| *value > 0.0)?;
                        debug_assert_eq!(
                            targets[branch_position],
                            expert_completion[branch_token_index]
                        );
                        ExpertRowObjective::PresentationIndex {
                            inputs,
                            branch_position,
                            candidate_target_tokens: candidate_tokens,
                            equivalent_target_tokens: equivalent_tokens,
                        }
                    }
                    burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep => {
                        ExpertRowObjective::SemanticStep {
                            prompt: prompt.clone(),
                            candidate_completions: candidate_completions.clone(),
                            equivalent_indices: presented_actions.equivalent_indices.clone(),
                        }
                    }
                };
                presentation_selected_indices.push(presented_actions.selected_index);
                presentation_equivalent_indices
                    .push(presented_actions.equivalent_indices.clone());
                presentation_rows.push(ExpertRow {
                    objective,
                    presentation_weight,
                });
                scoring_presentations.push(DaggerScoringPresentation {
                    rotation,
                    prompt,
                    candidate_completions,
                    answer_contract: scoring_contract,
                });
            }
            (!presentation_rows.is_empty() && !scoring_presentations.is_empty()).then_some(
                PreparedExpertState {
                    canonical_prompt,
                    presentation_rows,
                    scoring_presentations,
                    presentation_selected_indices,
                    presentation_equivalent_indices,
                },
            )
        };

        let state_prepare_started = Instant::now();
        let mut trajectories = Vec::<DaggerTrajectory>::new();
        let mut answer_contract = None;
        for (sample_index, sample) in policy_batch.samples.iter().enumerate() {
            let Some(burn_dragon_universality::RuliadSampleSpec::FormalProof {
                problem,
                certificate,
                proof_step_index,
                action_answer_contract,
                task: burn_dragon_universality::RuliadTaskKind::SelectProofAction,
                ..
            }) = sample.item.spec.as_ref()
            else {
                continue;
            };
            available_sample_groups = available_sample_groups.saturating_add(1);
            if trajectories.len() >= trajectory_budget {
                continue;
            }
            let scoring_contract = match config.scoring {
                crate::config::RuliadProofPolicyScoring::CompletionLikelihood => {
                    *action_answer_contract
                }
                crate::config::RuliadProofPolicyScoring::SemanticEnergy => {
                    burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep
                }
            };
            if answer_contract.is_some_and(|contract| contract != scoring_contract) {
                return None;
            }
            answer_contract.get_or_insert(scoring_contract);
            let difficulty_level = sample.item.difficulty_level.unwrap_or(0);
            sample_groups = sample_groups.saturating_add(1);
            *difficulty_sample_groups
                .entry(difficulty_level)
                .or_default() += 1;
            let start_step = proof_step_index.unwrap_or_default();
            nonzero_start_trajectories =
                nonzero_start_trajectories.saturating_add(usize::from(start_step > 0));
            start_step_sum = start_step_sum.saturating_add(start_step);
            let state =
                burn_dragon_universality::ruliad::RuliadProofPolicyState::from_certificate_prefix(
                    problem,
                    certificate,
                    start_step,
                )
                .ok()?;
            let trajectory_index = trajectories.len();
            let (is_dagger, max_depth) = if trajectory_index < batch_plan.static_row_budget {
                (false, 1)
            } else {
                let dagger_index = trajectory_index - batch_plan.static_row_budget;
                (true, batch_plan.dagger_depth(dagger_index))
            };
            trajectories.push(DaggerTrajectory {
                sample_index,
                difficulty_level,
                is_dagger,
                max_depth,
                answer_contract: scoring_contract,
                state,
            });
        }
        let state_prepare_ms = state_prepare_started.elapsed().as_micros() as f64 / 1_000.0;

        for rollout_depth in 0..batch_plan.rollout_steps {
            if presentation_budget_exhausted
                || base_semantic_state_rows >= base_semantic_row_budget
                || trajectories
                    .iter()
                    .all(|item| rollout_depth >= item.max_depth || item.state.solved())
            {
                break;
            }
            let wave_prepare_started = Instant::now();
            let states_before_wave = base_semantic_state_rows;
            let mut expansions = Vec::<DaggerExpansion>::new();
            for (trajectory_index, trajectory) in trajectories.iter_mut().enumerate() {
                if rollout_depth >= trajectory.max_depth
                    || trajectory.state.solved()
                    || base_semantic_state_rows >= base_semantic_row_budget
                {
                    continue;
                }
                let sample = &policy_batch.samples[trajectory.sample_index];
                let Some(burn_dragon_universality::RuliadSampleSpec::FormalProof {
                    problem, ..
                }) = sample.item.spec.as_ref()
                else {
                    continue;
                };
                let actions = match trajectory.state.action_set(problem, config.candidates) {
                    Ok(actions) => actions,
                    Err(_) if trajectory.state.backtrack() => {
                        model_backtracks = model_backtracks.saturating_add(1);
                        continue;
                    }
                    Err(_) => {
                        model_invalid_actions = model_invalid_actions.saturating_add(1);
                        continue;
                    }
                };
                let Some(mut original_state) = prepare_expert_state(
                    problem,
                    &actions,
                    semantic_state_rows,
                    trajectory.answer_contract,
                    None,
                ) else {
                    model_invalid_actions = model_invalid_actions.saturating_add(1);
                    continue;
                };
                visited_states = visited_states.saturating_add(1);
                *difficulty_visited_states
                    .entry(trajectory.difficulty_level)
                    .or_default() += 1;

                // Counterfactual targets are supervision only. The model rollout below still
                // scores and applies the original formal transition.
                let target_group_rotations = original_state
                    .scoring_presentations
                    .iter()
                    .map(|presentation| presentation.rotation)
                    .collect::<Vec<_>>();
                let scoring_presentations =
                    std::mem::take(&mut original_state.scoring_presentations);
                let mut prepared_states = vec![original_state];
                let counterfactual_indices =
                    crate::train::ruliad_policy::counterfactual_candidate_indices(
                        &actions,
                        config.counterfactual_targets_per_state,
                        actions
                            .selected_index
                            .saturating_add(base_semantic_state_rows)
                            .saturating_add(1),
                    );
                let mut group_shortfall = config
                    .counterfactual_targets_per_state
                    .saturating_sub(counterfactual_indices.len());
                for candidate_index in counterfactual_indices {
                    let Some((counterfactual_problem, counterfactual_actions)) =
                        burn_dragon_universality::ruliad::counterfactual_proof_action_target(
                            problem,
                            &actions,
                            candidate_index,
                        )
                        .ok()
                    else {
                        group_shortfall = group_shortfall.saturating_add(1);
                        continue;
                    };
                    let Some(counterfactual_state) = prepare_expert_state(
                        &counterfactual_problem,
                        &counterfactual_actions,
                        semantic_state_rows.saturating_add(prepared_states.len()),
                        trajectory.answer_contract,
                        Some(&target_group_rotations),
                    ) else {
                        group_shortfall = group_shortfall.saturating_add(1);
                        continue;
                    };
                    prepared_states.push(counterfactual_state);
                }
                counterfactual_target_shortfall =
                    counterfactual_target_shortfall.saturating_add(group_shortfall);
                let complete_target_group = group_shortfall == 0
                    && prepared_states.len() == config.target_variants_per_state();
                let presentation_rows = prepared_states
                    .iter()
                    .map(|state| state.presentation_rows.len())
                    .sum::<usize>();
                if complete_target_group
                    && rows.len().saturating_add(presentation_rows)
                        > config.max_presentation_rows_per_update
                {
                    presentation_budget_exhausted = true;
                    break;
                }
                let unique_target_group = complete_target_group
                    && prepared_states
                        .iter()
                        .all(|state| !visited_prompts.contains(&state.canonical_prompt));
                if unique_target_group {
                    let variants_added = prepared_states.len();
                    for state in prepared_states {
                        visited_prompts.insert(state.canonical_prompt);
                        for selected_index in state.presentation_selected_indices {
                            *expert_selected_index_histogram
                                .entry(selected_index)
                                .or_default() += 1;
                        }
                        for equivalent_indices in state.presentation_equivalent_indices {
                            for candidate_index in equivalent_indices {
                                *expert_equivalent_index_histogram
                                    .entry(candidate_index)
                                    .or_default() += 1;
                            }
                        }
                        for row in &state.presentation_rows {
                            match &row.objective {
                                ExpertRowObjective::PresentationIndex {
                                    candidate_target_tokens: candidate_tokens,
                                    equivalent_target_tokens: equivalent_tokens,
                                    ..
                                } => {
                                    supervised_action_tokens =
                                        supervised_action_tokens.saturating_add(1);
                                    candidate_target_tokens = candidate_target_tokens
                                        .saturating_add(candidate_tokens.len());
                                    equivalent_target_tokens = equivalent_target_tokens
                                        .saturating_add(equivalent_tokens.len());
                                }
                                ExpertRowObjective::SemanticStep {
                                    candidate_completions,
                                    equivalent_indices,
                                    ..
                                } => {
                                    let candidate_tokens =
                                        candidate_completions.iter().map(Vec::len).sum::<usize>();
                                    let equivalent_tokens = equivalent_indices
                                        .iter()
                                        .filter_map(|index| candidate_completions.get(*index))
                                        .map(Vec::len)
                                        .sum::<usize>();
                                    supervised_action_tokens =
                                        supervised_action_tokens.saturating_add(candidate_tokens);
                                    candidate_target_tokens =
                                        candidate_target_tokens.saturating_add(candidate_tokens);
                                    equivalent_target_tokens =
                                        equivalent_target_tokens.saturating_add(equivalent_tokens);
                                }
                            }
                        }
                        rows.extend(state.presentation_rows);
                    }
                    semantic_state_rows = semantic_state_rows.saturating_add(variants_added);
                    base_semantic_state_rows = base_semantic_state_rows.saturating_add(1);
                    counterfactual_semantic_state_rows = counterfactual_semantic_state_rows
                        .saturating_add(variants_added.saturating_sub(1));
                    static_expert_rows = static_expert_rows.saturating_add(
                        variants_added.saturating_mul(usize::from(!trajectory.is_dagger)),
                    );
                    dagger_expert_rows = dagger_expert_rows.saturating_add(
                        variants_added.saturating_mul(usize::from(trajectory.is_dagger)),
                    );
                    model_visited_expert_rows = model_visited_expert_rows
                        .saturating_add(variants_added.saturating_mul(usize::from(
                            trajectory.is_dagger && rollout_depth > 0,
                        )));
                    *difficulty_expert_rows
                        .entry(trajectory.difficulty_level)
                        .or_default() += variants_added;
                }
                if trajectory.is_dagger && rollout_depth.saturating_add(1) < trajectory.max_depth {
                    expansions.push(DaggerExpansion {
                        trajectory_index,
                        actions,
                        presentations: scoring_presentations,
                    });
                }
            }
            // The last supervised wave is already represented in `rows`. Scoring it cannot
            // produce another training row once the row budget is full, so avoid a synchronized
            // inference forward that only changes diagnostic terminal state.
            if presentation_budget_exhausted || base_semantic_state_rows >= base_semantic_row_budget
            {
                expansions.clear();
            }
            rollout_cpu_prepare_ms += wave_prepare_started.elapsed().as_micros() as f64 / 1_000.0;
            if base_semantic_state_rows > states_before_wave || !expansions.is_empty() {
                rollout_depth_reached = rollout_depth_reached.max(rollout_depth.saturating_add(1));
            }
            if expansions.is_empty() {
                break;
            }
            let scoring_presentations = expansions
                .iter()
                .enumerate()
                .flat_map(|(expansion_index, expansion)| {
                    expansion
                        .presentations
                        .iter()
                        .map(move |presentation| (expansion_index, presentation))
                })
                .collect::<Vec<_>>();
            let prompts = scoring_presentations
                .iter()
                .map(|(_, presentation)| presentation.prompt.clone())
                .collect::<Vec<_>>();
            let candidates = scoring_presentations
                .iter()
                .map(|(_, presentation)| presentation.candidate_completions.clone())
                .collect::<Vec<_>>();
            model_scoring_batches = model_scoring_batches.saturating_add(1);
            maximum_model_scoring_batch_rows =
                maximum_model_scoring_batch_rows.max(scoring_presentations.len());
            let scoring_contract = scoring_presentations
                .first()
                .map(|(_, presentation)| presentation.answer_contract)?;
            if scoring_presentations
                .iter()
                .any(|(_, presentation)| presentation.answer_contract != scoring_contract)
            {
                return None;
            }
            let scoring_max_len = scoring_presentations
                .iter()
                .filter_map(|(_, presentation)| match scoring_contract {
                    burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::PresentationIndex => {
                        crate::train::ruliad_policy::candidate_branch_index(
                            &presentation.candidate_completions,
                        )
                        .ok()
                        .map(|prefix_len| presentation.prompt.len().saturating_add(prefix_len))
                    }
                    burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep => {
                        presentation
                            .candidate_completions
                            .iter()
                            .map(Vec::len)
                            .max()
                            .map(|completion_len| {
                                presentation
                                    .prompt
                                    .len()
                                    .saturating_add(completion_len)
                                    .saturating_sub(1)
                            })
                    }
                })
                .max()
                .unwrap_or_default();
            model_scoring_padded_tokens = model_scoring_padded_tokens
                .saturating_add(scoring_max_len.saturating_mul(scoring_presentations.len()));
            let model_scoring_started = Instant::now();
            let score_rows = crate::train::ruliad_policy::proof_action_scores_batch(
                sampling_model.as_ref()?,
                &prompts,
                &candidates,
                scoring_contract,
                config.scoring,
                device,
            )
            .ok()?;
            model_scoring_ms += model_scoring_started.elapsed().as_micros() as f64 / 1_000.0;
            let mut scores_by_expansion = (0..expansions.len())
                .map(|_| Vec::<(usize, Vec<f32>)>::new())
                .collect::<Vec<_>>();
            for ((expansion_index, presentation), scores) in
                scoring_presentations.iter().zip(score_rows)
            {
                scores_by_expansion[*expansion_index].push((presentation.rotation, scores));
            }
            drop(scoring_presentations);
            for (expansion, presentation_scores) in expansions.into_iter().zip(scores_by_expansion)
            {
                let scores = crate::train::ruliad_policy::semantic_action_log_probs(
                    &presentation_scores,
                    expansion.actions.candidates.len(),
                )
                .ok()?;
                let Some(candidate_index) =
                    crate::train::ruliad_policy::best_candidate_index(&scores)
                else {
                    model_invalid_actions = model_invalid_actions.saturating_add(1);
                    continue;
                };
                *model_selected_index_histogram
                    .entry(candidate_index)
                    .or_default() += 1;
                if expansion.actions.is_equivalent_index(candidate_index) {
                    model_expert_equivalent_actions =
                        model_expert_equivalent_actions.saturating_add(1);
                } else {
                    model_off_expert_actions = model_off_expert_actions.saturating_add(1);
                }
                match trajectories[expansion.trajectory_index]
                    .state
                    .apply(&expansion.actions, candidate_index)
                {
                    Ok(repeated) => {
                        model_valid_actions = model_valid_actions.saturating_add(1);
                        repeated_states = repeated_states.saturating_add(usize::from(repeated));
                    }
                    Err(_) => {
                        model_invalid_actions = model_invalid_actions.saturating_add(1);
                    }
                }
            }
        }
        let solved_proofs = trajectories
            .iter()
            .filter(|trajectory| trajectory.state.solved())
            .count();

        let mut prefix_branch_rows = Vec::<PrefixBranchRow>::new();
        if config.normalization == crate::config::RuliadProofPolicyNormalization::PrefixConditional
            && answer_contract
                == Some(
                    burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep,
                )
        {
            for row in &rows {
                let ExpertRowObjective::SemanticStep {
                    prompt,
                    candidate_completions,
                    equivalent_indices,
                } = &row.objective
                else {
                    return None;
                };
                let branches = crate::train::ruliad_policy::semantic_candidate_trie_branches(
                    candidate_completions,
                    equivalent_indices,
                )
                .ok()?;
                let branch_weight = row.presentation_weight / branches.len().max(1) as f32;
                for branch in branches {
                    let mut inputs = prompt.clone();
                    inputs.extend(branch.prefix);
                    let branch_position = inputs.len().checked_sub(1)?;
                    prefix_branch_rows.push(PrefixBranchRow {
                        inputs,
                        branch_position,
                        candidate_target_tokens: branch.candidate_tokens,
                        equivalent_target_tokens: branch.equivalent_tokens,
                        weight: branch_weight,
                    });
                }
            }
        }
        let prefix_candidate_tokens = prefix_branch_rows
            .iter()
            .map(|row| row.candidate_target_tokens.len())
            .sum::<usize>();
        let prefix_equivalent_tokens = prefix_branch_rows
            .iter()
            .map(|row| row.equivalent_target_tokens.len())
            .sum::<usize>();

        debug_assert!(rows.len() <= config.max_presentation_rows_per_update);
        self.write_ruliad_proof_policy_dagger_telemetry(RuliadProofPolicyDaggerTelemetry {
            version: 19,
            answer_contract: answer_contract.unwrap_or_default().label(),
            objective: if config.scoring == crate::config::RuliadProofPolicyScoring::SemanticEnergy
            {
                if config.counterfactual_targets_per_state > 0 {
                    "semantic_sequence_energy_counterfactual_v1"
                } else {
                    "semantic_sequence_energy_v1"
                }
            } else {
                match config.normalization {
                    crate::config::RuliadProofPolicyNormalization::CandidateConditional => {
                        if config.counterfactual_targets_per_state > 0 {
                            "candidate_normalized_counterfactual_v1"
                        } else {
                            "candidate_normalized_equivalent_v1"
                        }
                    }
                    crate::config::RuliadProofPolicyNormalization::PrefixConditional => {
                        "prefix_conditional_equivalent_v1"
                    }
                    crate::config::RuliadProofPolicyNormalization::VocabularyMarginal => {
                        "vocabulary_marginal_equivalent_v1"
                    }
                }
            },
            gradient_scope: match config.gradient_scope {
                crate::config::RuliadProofPolicyGradientScope::FullModel => "full_model",
                crate::config::RuliadProofPolicyGradientScope::ScoreHeadOnly => "score_head_only",
                crate::config::RuliadProofPolicyGradientScope::LanguageHeadOnly => {
                    "language_head_only"
                }
            },
            presentation_risk: match config.presentation_risk {
                crate::config::RuliadProofPolicyPresentationRisk::Mean => "mean",
                crate::config::RuliadProofPolicyPresentationRisk::Worst => "worst",
            },
            configured_mode: match config.mode {
                crate::config::RuliadProofPolicyTrainingMode::StaticExpert => "static_expert",
                crate::config::RuliadProofPolicyTrainingMode::Dagger => "dagger",
                crate::config::RuliadProofPolicyTrainingMode::StaticThenPairedDagger => {
                    "static_then_paired_dagger"
                }
            },
            mode: match effective_mode {
                crate::config::RuliadProofPolicyEffectiveMode::StaticExpert => "static_expert",
                crate::config::RuliadProofPolicyEffectiveMode::Dagger => "dagger",
                crate::config::RuliadProofPolicyEffectiveMode::PairedDagger => "paired_dagger",
            },
            candidate_symmetry: match config.candidate_symmetry {
                crate::config::RuliadProofPolicyCandidateSymmetry::Canonical => "canonical",
                crate::config::RuliadProofPolicyCandidateSymmetry::BalancedRotation => {
                    "balanced_rotation"
                }
                crate::config::RuliadProofPolicyCandidateSymmetry::CyclicOrbitAverage => {
                    "cyclic_orbit_average"
                }
            },
            step_index,
            skip_reason: rows
                .is_empty()
                .then(|| "no_formal_policy_states".to_string()),
            available_sample_groups,
            sample_groups,
            nonzero_start_trajectories,
            mean_start_step: start_step_sum as f64 / sample_groups.max(1) as f64,
            visited_states,
            semantic_state_rows,
            base_semantic_state_rows,
            counterfactual_semantic_state_rows,
            counterfactual_target_shortfall,
            expert_rows: semantic_state_rows,
            static_expert_rows,
            dagger_expert_rows,
            model_visited_expert_rows,
            supervised_action_tokens,
            supervised_presentation_rows: rows.len(),
            mean_presentations_per_state: rows.len() as f64 / semantic_state_rows.max(1) as f64,
            model_valid_actions,
            model_invalid_actions,
            model_expert_equivalent_actions,
            model_off_expert_actions,
            repeated_states,
            model_backtracks,
            solved_proofs,
            model_scoring_batches,
            maximum_model_scoring_batch_rows,
            model_scoring_padded_tokens,
            sampling_model_materialize_ms,
            state_prepare_ms,
            rollout_cpu_prepare_ms,
            model_scoring_ms,
            difficulty_sample_groups,
            difficulty_visited_states,
            difficulty_expert_rows,
            expert_selected_index_histogram,
            expert_equivalent_index_histogram,
            model_selected_index_histogram,
            candidate_target_tokens,
            equivalent_target_tokens,
            mean_candidate_targets_per_row: candidate_target_tokens as f64
                / rows.len().max(1) as f64,
            mean_equivalent_targets_per_row: equivalent_target_tokens as f64
                / rows.len().max(1) as f64,
            prefix_branch_rows: prefix_branch_rows.len(),
            prefix_candidate_tokens,
            prefix_equivalent_tokens,
            weight,
            rollout_steps: batch_plan.rollout_steps,
            rollout_depth_reached,
            configured_rollout_steps: config.rollout_steps,
            trajectory_budget,
            semantic_row_budget,
            base_semantic_row_budget,
            configured_counterfactual_targets_per_state: config.counterfactual_targets_per_state,
            target_variants_per_state: config.target_variants_per_state(),
            max_rows_per_update: config.max_rows_per_update,
            max_presentation_rows_per_update: config.max_presentation_rows_per_update,
        });
        if rows.is_empty() {
            return None;
        }
        if semantic_state_rows == 0 || !rows.len().is_multiple_of(semantic_state_rows) {
            return None;
        }
        let presentation_group_size = rows.len() / semantic_state_rows;
        let row_count = rows.len();
        let row_weights = Tensor::<B, 1>::from_data(
            TensorData::new(
                rows.iter()
                    .map(|row| row.presentation_weight)
                    .collect::<Vec<_>>(),
                [row_count],
            ),
            device,
        );
        match answer_contract? {
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::PresentationIndex => {
                let max_len = rows
                    .iter()
                    .filter_map(|row| match &row.objective {
                        ExpertRowObjective::PresentationIndex { inputs, .. } => Some(inputs.len()),
                        ExpertRowObjective::SemanticStep { .. } => None,
                    })
                    .max()?
                    .max(1);
                let mut input_values = vec![0i64; row_count * max_len];
                let mut branch_positions = Vec::with_capacity(row_count);
                for (row_index, row) in rows.iter().enumerate() {
                    let ExpertRowObjective::PresentationIndex {
                        inputs,
                        branch_position,
                        ..
                    } = &row.objective
                    else {
                        return None;
                    };
                    let offset = row_index * max_len;
                    let len = inputs.len().min(max_len);
                    input_values[offset..offset + len].copy_from_slice(&inputs[..len]);
                    branch_positions.push(*branch_position);
                }
                let inputs = Tensor::<B, 2, Int>::from_data(
                    TensorData::new(input_values, [row_count, max_len]),
                    device,
                );
                let branch_logits = crate::train::ruliad_policy::logits_at_sequence_positions(
                    &self.model,
                    inputs,
                    &branch_positions,
                    device,
                )
                .ok()?;
                let [_, vocab] = branch_logits.shape().dims::<2>();
                let branch_logits = branch_logits.reshape([row_count, 1, vocab]);
                let mut candidate_mask_values =
                    vec![0.0f32; row_count.saturating_mul(vocab)];
                let mut equivalent_mask_values =
                    vec![0.0f32; row_count.saturating_mul(vocab)];
                for (row_index, row) in rows.iter().enumerate() {
                    let ExpertRowObjective::PresentationIndex {
                        candidate_target_tokens,
                        equivalent_target_tokens,
                        ..
                    } = &row.objective
                    else {
                        return None;
                    };
                    for token in candidate_target_tokens {
                        let token = usize::try_from(*token).ok()?;
                        if token >= vocab {
                            return None;
                        }
                        candidate_mask_values[row_index * vocab + token] = 1.0;
                    }
                    for token in equivalent_target_tokens {
                        let token = usize::try_from(*token).ok()?;
                        if token >= vocab {
                            return None;
                        }
                        equivalent_mask_values[row_index * vocab + token] = 1.0;
                    }
                }
                let candidate_mask = Tensor::<B, 3>::from_data(
                    TensorData::new(candidate_mask_values, [row_count, 1, vocab]),
                    device,
                );
                let equivalent_mask = Tensor::<B, 3>::from_data(
                    TensorData::new(equivalent_mask_values, [row_count, 1, vocab]),
                    device,
                );
                Some(grouped_verifier_equivalent_action_loss(
                    branch_logits,
                    candidate_mask,
                    equivalent_mask,
                    row_weights,
                    config.normalization,
                    config.presentation_risk,
                    presentation_group_size,
                    weight,
                ))
            }
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep => {
                if config.normalization
                    == crate::config::RuliadProofPolicyNormalization::PrefixConditional
                {
                    let branch_row_count = prefix_branch_rows.len();
                    if branch_row_count == 0 {
                        return None;
                    }
                    let max_len = prefix_branch_rows
                        .iter()
                        .map(|row| row.inputs.len())
                        .max()?
                        .max(1);
                    let mut input_values = vec![0i64; branch_row_count * max_len];
                    let mut branch_positions = Vec::with_capacity(branch_row_count);
                    let branch_weights = prefix_branch_rows
                        .iter()
                        .map(|row| row.weight)
                        .collect::<Vec<_>>();
                    for (row_index, row) in prefix_branch_rows.iter().enumerate() {
                        let offset = row_index * max_len;
                        input_values[offset..offset + row.inputs.len()]
                            .copy_from_slice(&row.inputs);
                        branch_positions.push(row.branch_position);
                    }
                    let inputs = Tensor::<B, 2, Int>::from_data(
                        TensorData::new(input_values, [branch_row_count, max_len]),
                        device,
                    );
                    let branch_logits =
                        crate::train::ruliad_policy::logits_at_sequence_positions(
                            &self.model,
                            inputs,
                            &branch_positions,
                            device,
                        )
                        .ok()?;
                    let [_, vocab] = branch_logits.shape().dims::<2>();
                    let mut candidate_mask_values =
                        vec![0.0f32; branch_row_count.saturating_mul(vocab)];
                    let mut equivalent_mask_values =
                        vec![0.0f32; branch_row_count.saturating_mul(vocab)];
                    for (row_index, row) in prefix_branch_rows.iter().enumerate() {
                        for token in &row.candidate_target_tokens {
                            let token = usize::try_from(*token).ok()?;
                            if token >= vocab {
                                return None;
                            }
                            candidate_mask_values[row_index * vocab + token] = 1.0;
                        }
                        for token in &row.equivalent_target_tokens {
                            let token = usize::try_from(*token).ok()?;
                            if token >= vocab {
                                return None;
                            }
                            equivalent_mask_values[row_index * vocab + token] = 1.0;
                        }
                    }
                    let candidate_mask = Tensor::<B, 3>::from_data(
                        TensorData::new(candidate_mask_values, [branch_row_count, 1, vocab]),
                        device,
                    );
                    let equivalent_mask = Tensor::<B, 3>::from_data(
                        TensorData::new(equivalent_mask_values, [branch_row_count, 1, vocab]),
                        device,
                    );
                    let row_weights = Tensor::<B, 1>::from_data(
                        TensorData::new(branch_weights, [branch_row_count]),
                        device,
                    );
                    return Some(grouped_verifier_equivalent_action_loss(
                        branch_logits.reshape([branch_row_count, 1, vocab]),
                        candidate_mask,
                        equivalent_mask,
                        row_weights,
                        crate::config::RuliadProofPolicyNormalization::CandidateConditional,
                        crate::config::RuliadProofPolicyPresentationRisk::Mean,
                        1,
                        weight,
                    ));
                }
                let mut prompts = Vec::with_capacity(row_count);
                let mut candidates = Vec::with_capacity(row_count);
                let mut equivalent_indices = Vec::with_capacity(row_count);
                for row in &rows {
                    let ExpertRowObjective::SemanticStep {
                        prompt,
                        candidate_completions,
                        equivalent_indices: row_equivalent_indices,
                    } = &row.objective
                    else {
                        return None;
                    };
                    prompts.push(prompt.clone());
                    candidates.push(candidate_completions.clone());
                    equivalent_indices.push(row_equivalent_indices.clone());
                }
                let candidate_count = candidates.first()?.len();
                if candidate_count < 2
                    || candidates.iter().any(|group| group.len() != candidate_count)
                {
                    return None;
                }
                let (mean_log_scores, sum_log_scores, group_sizes) = match config.scoring {
                    crate::config::RuliadProofPolicyScoring::CompletionLikelihood => {
                        let scores =
                            crate::train::ruliad_policy::sequence_completion_score_tensor_with_gradient_scope(
                                &self.model,
                                &prompts,
                                &candidates,
                                config.gradient_scope,
                                device,
                            )
                            .ok()?;
                        (
                            scores.mean_log_scores,
                            scores.sum_log_scores,
                            scores.group_sizes,
                        )
                    }
                    crate::config::RuliadProofPolicyScoring::SemanticEnergy => {
                        let (scores, group_sizes) =
                            crate::train::ruliad_policy::sequence_energy_score_tensor_with_gradient_scope(
                                &self.model,
                                &prompts,
                                &candidates,
                                config.gradient_scope,
                                device,
                            )
                            .ok()?;
                        (scores.clone(), scores, group_sizes)
                    }
                };
                if group_sizes
                    .iter()
                    .any(|group_size| *group_size != candidate_count)
                {
                    return None;
                }
                let mut equivalent_mask_values =
                    vec![0.0f32; row_count.saturating_mul(candidate_count)];
                for (row_index, indices) in equivalent_indices.iter().enumerate() {
                    for index in indices {
                        if *index >= candidate_count {
                            return None;
                        }
                        equivalent_mask_values[row_index * candidate_count + *index] = 1.0;
                    }
                }
                let equivalent_mask = Tensor::<B, 2>::from_data(
                    TensorData::new(equivalent_mask_values, [row_count, candidate_count]),
                    device,
                );
                Some(grouped_verifier_equivalent_sequence_loss(
                    mean_log_scores.reshape([row_count, candidate_count]),
                    sum_log_scores.reshape([row_count, candidate_count]),
                    equivalent_mask,
                    row_weights,
                    GroupedVerifierSequenceLossConfig {
                        normalization: config.normalization,
                        presentation_risk: config.presentation_risk,
                        presentation_group_size,
                        weight,
                    },
                ))
            }
        }
    }

    fn ruliad_verifier_policy_loss(
        &self,
        policy_batch: &crate::dataset::RuliadPolicyBatch,
        device: &B::Device,
        block_size: usize,
    ) -> Option<Tensor<B, 1>>
    where
        B: AutodiffBackend,
    {
        let config = self.ruliad_supervision.verifier_reward;
        let weight = self.ruliad_verifier_reward_weight();
        if weight <= f32::EPSILON || policy_batch.samples.is_empty() || self.pipeline_enabled() {
            return None;
        }
        let tokenizer =
            burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer::from_config(
                &policy_batch.tokenization,
            )
            .ok()?;
        let completion_budget = config
            .max_completion_tokens
            .max(1)
            .min(block_size.saturating_sub(1).max(1));
        let prompt_budget = block_size.saturating_sub(completion_budget).max(1);
        let group_size = config.group_size.max(2);

        #[derive(Clone)]
        struct PolicyRow {
            inputs: Vec<i64>,
            targets: Vec<i64>,
            mask: Vec<f32>,
            advantage: f32,
        }

        let mut rows = Vec::new();
        let mut telemetry = RuliadPolicyRewardTelemetryAccumulator::new(
            config.mode,
            self.gradient_scale_step.load(Ordering::Relaxed),
        );
        let mut observed_generated_rows = 0usize;
        let mut recorded_attractor_rows = 0usize;
        let sampling_model = self
            .model
            .valid()
            .materialize_random_scaffold_for_inference();
        for sample in policy_batch.samples.iter() {
            let mut prompt = sample.prompt_tokens.clone();
            if prompt.is_empty() {
                continue;
            }
            if prompt.len() > prompt_budget {
                prompt = prompt[prompt.len() - prompt_budget..].to_vec();
            }
            let configured_structured_negatives = if config.include_structured_negative_candidates {
                config
                    .structured_negative_count
                    .saturating_add(config.structured_template_negative_count)
                    .saturating_add(config.structured_schema_negative_count)
            } else {
                0
            };
            let generated_attractor_candidates =
                self.ruliad_generated_attractor_candidates_for_sample(sample);
            let mut group_rows = Vec::with_capacity(
                group_size
                    + usize::from(config.include_oracle_candidate)
                    + configured_structured_negatives
                    + generated_attractor_candidates.len(),
            );
            let mut scores = Vec::with_capacity(
                group_size
                    + usize::from(config.include_oracle_candidate)
                    + configured_structured_negatives
                    + generated_attractor_candidates.len(),
            );
            if config.include_oracle_candidate
                && let Some((oracle_completion, oracle_completion_text, oracle_truncated)) =
                    Self::ruliad_oracle_completion_tokens(&tokenizer, sample, completion_budget)
                && let Some(row) =
                    Self::ruliad_policy_row_from_completion(&prompt, &oracle_completion)
            {
                let score = burn_dragon_universality::ruliad::score_ruliad_item_completion(
                    &sample.item,
                    Some(&oracle_completion_text),
                );
                telemetry.record_oracle_candidate(oracle_truncated);
                scores.push(score);
                group_rows.push(row);
            }
            if config.include_structured_negative_candidates {
                for (negative, _negative_kind) in
                    Self::ruliad_structured_negative_answers_with_schema(
                        &sample.item.expected_answer,
                        config.structured_negative_count,
                        config.structured_template_negative_count,
                        config.structured_schema_negative_count,
                    )
                {
                    let Some((completion, completion_text)) =
                        Self::ruliad_completion_tokens_from_answer(
                            &tokenizer,
                            &negative,
                            sample.item.document_close_marker(),
                            completion_budget,
                        )
                    else {
                        continue;
                    };
                    let Some(row) = Self::ruliad_policy_row_from_completion(&prompt, &completion)
                    else {
                        continue;
                    };
                    let score = burn_dragon_universality::ruliad::score_ruliad_item_completion(
                        &sample.item,
                        Some(&completion_text),
                    );
                    telemetry.record_structured_negative_candidate();
                    scores.push(score);
                    group_rows.push(row);
                }
            }
            for entry in generated_attractor_candidates {
                let Some((completion, completion_text)) =
                    Self::ruliad_completion_tokens_from_answer(
                        &tokenizer,
                        &entry.key.answer,
                        sample.item.document_close_marker(),
                        completion_budget,
                    )
                else {
                    continue;
                };
                let Some(row) = Self::ruliad_policy_row_from_completion(&prompt, &completion)
                else {
                    continue;
                };
                let score = burn_dragon_universality::ruliad::score_ruliad_item_completion(
                    &sample.item,
                    Some(&completion_text),
                );
                telemetry.record_generated_attractor_candidate();
                scores.push(score);
                group_rows.push(row);
            }
            for _ in 0..group_size {
                let generated = crate::generation::generate_tokens(
                    &sampling_model,
                    prompt.clone(),
                    device,
                    crate::generation::GenerationSettings {
                        max_new_tokens: Some(completion_budget),
                        temperature: config.temperature,
                        top_k: Some(config.top_k),
                        strategy: crate::generation::ContextStrategy::Infinite,
                        stop_on_token: policy_batch.stop_token_id,
                    },
                    None,
                )
                .ok()?;
                if generated.len() <= prompt.len() {
                    continue;
                }
                let completion = generated[prompt.len()..].to_vec();
                if completion.is_empty() {
                    continue;
                }
                let completion_tokens = completion
                    .iter()
                    .filter_map(|token| u32::try_from(*token).ok())
                    .collect::<Vec<_>>();
                let completion_text = tokenizer.decode_payload(&completion_tokens, true);
                let score = burn_dragon_universality::ruliad::score_ruliad_item_completion(
                    &sample.item,
                    Some(&completion_text),
                );
                observed_generated_rows = observed_generated_rows.saturating_add(1);
                recorded_attractor_rows = recorded_attractor_rows.saturating_add(usize::from(
                    self.record_ruliad_generated_attractor(
                        sample,
                        &completion_text,
                        &score,
                        telemetry.step_index,
                    ),
                ));
                scores.push(score);
                if let Some(row) = Self::ruliad_policy_row_from_completion(&prompt, &completion) {
                    group_rows.push(row);
                }
            }
            if group_rows.is_empty() || scores.len() != group_rows.len() {
                continue;
            }
            telemetry.record_vectors(&scores);
            let rewards = match config.mode {
                crate::config::train::RuliadVerifierRewardMode::Scalar => scores
                    .iter()
                    .map(|score| {
                        burn_dragon_universality::ruliad::ruliad_verifier_reward(
                            score,
                            config.reward,
                        )
                    })
                    .collect::<Vec<_>>(),
                crate::config::train::RuliadVerifierRewardMode::VpoIndependent => {
                    let scalarizations = self.ruliad_vpo_scalarizations(
                        sample.item.sample_index,
                        config.vpo_scalarizations.max(1),
                        config,
                    );
                    self.ruliad_vpo_independent_utilities_with_telemetry(
                        &scores,
                        &scalarizations,
                        &mut telemetry,
                    )
                }
            };
            let mut advantages = burn_dragon_universality::ruliad::normalized_advantages(
                &rewards,
                config.advantage_epsilon,
            );
            if !Self::constrain_ruliad_policy_advantages(&scores, &mut advantages, config) {
                telemetry.record_gated_group(rewards.len());
                continue;
            }
            telemetry.record_rewards_and_advantages(&rewards, &advantages, config.clip_range);
            rows.extend(group_rows.into_iter().zip(advantages).map(
                |((inputs, targets, mask), advantage)| PolicyRow {
                    inputs,
                    targets,
                    mask,
                    advantage: advantage.clamp(-config.clip_range, config.clip_range),
                },
            ));
        }
        let replay_summary = self.ruliad_generated_attractor_summary();
        self.write_ruliad_generated_attractor_telemetry(RuliadGeneratedAttractorReplayTelemetry {
            version: 1,
            step_index: telemetry.step_index,
            source: "policy".to_string(),
            skip_reason: (observed_generated_rows == 0)
                .then(|| "no_generated_rows".to_string())
                .or_else(|| {
                    self.ruliad_generated_attractor_replay_skip_reason(
                        &replay_summary,
                        telemetry.generated_attractor_completion_rows,
                    )
                }),
            observed_completion_rows: observed_generated_rows,
            recorded_attractor_rows,
            selected_candidate_rows: telemetry.generated_attractor_completion_rows,
            selected_field_binding_pairs: 0,
            replay_pool_size: replay_summary.pool_size,
            active_attractor_count: replay_summary.active_count,
            active_observation_count: replay_summary.active_observation_count,
            distinct_answer_count: replay_summary.distinct_answers,
            dominant_answer_count: replay_summary.dominant_count,
            dominant_answer_fraction: replay_summary.dominant_fraction(),
            min_count: config.generated_attractor_replay_min_count.max(1),
            max_candidates: config.generated_attractor_replay_max_candidates,
            min_distinct_answers: config
                .generated_attractor_replay_min_distinct_answers
                .max(1),
            max_dominant_fraction: config.generated_attractor_replay_max_dominant_fraction,
        });
        if rows.is_empty() {
            if telemetry.has_observations() {
                telemetry.mark_skipped("positive_advantage_gate");
                if let Some(telemetry) = telemetry.finish() {
                    self.write_ruliad_policy_telemetry(telemetry);
                }
            }
            return None;
        }
        if let Some(max_clip_fraction) = config.max_advantage_clip_fraction {
            let clip_fraction = telemetry.advantage_clip_fraction();
            if clip_fraction > f64::from(max_clip_fraction) {
                telemetry.mark_skipped(format!("advantage_clip_fraction>{max_clip_fraction:.6}"));
                if let Some(telemetry) = telemetry.finish() {
                    self.write_ruliad_policy_telemetry(telemetry);
                }
                return None;
            }
        }
        if let Some(telemetry) = telemetry.finish() {
            self.write_ruliad_policy_telemetry(telemetry);
        }
        let max_len = rows.iter().map(|row| row.inputs.len()).max()?.max(1);
        let row_count = rows.len();
        let mut input_values = vec![0i64; row_count * max_len];
        let mut target_values = vec![0i64; row_count * max_len];
        let mut mask_values = vec![0.0f32; row_count * max_len];
        let mut advantage_values = vec![0.0f32; row_count * max_len];
        for (row_index, row) in rows.into_iter().enumerate() {
            let offset = row_index * max_len;
            let len = row.inputs.len().min(max_len);
            input_values[offset..offset + len].copy_from_slice(&row.inputs[..len]);
            target_values[offset..offset + len].copy_from_slice(&row.targets[..len]);
            mask_values[offset..offset + len].copy_from_slice(&row.mask[..len]);
            for value in advantage_values[offset..offset + len].iter_mut() {
                *value = row.advantage;
            }
        }
        let inputs = Tensor::<B, 2, Int>::from_data(
            TensorData::new(input_values, [row_count, max_len]),
            device,
        );
        let targets = Tensor::<B, 2, Int>::from_data(
            TensorData::new(target_values, [row_count, max_len]),
            device,
        );
        let mask =
            Tensor::<B, 2>::from_data(TensorData::new(mask_values, [row_count, max_len]), device);
        let advantages = Tensor::<B, 2>::from_data(
            TensorData::new(advantage_values, [row_count, max_len]),
            device,
        );
        let logits = self.model.forward(inputs.clone());
        let log_probs = log_probs_from_logits(logits);
        let token_log_probs = selected_token_log_probs(log_probs.clone(), targets);
        let active = mask.clone().sum().reshape([1]).clamp_min(1.0);
        let mut loss = (token_log_probs * advantages * mask.clone())
            .sum()
            .reshape([1])
            .div(active)
            .mul_scalar(-weight);
        if config.kl_weight > f32::EPSILON && self.teacher_model.is_some() {
            let teacher_log_probs =
                log_probs_from_logits(self.current_teacher_model().forward(inputs).detach());
            let [rows, time, _vocab] = log_probs.shape().dims();
            let per_token_kl = (log_probs.clone().exp() * (log_probs - teacher_log_probs))
                .sum_dim(2)
                .reshape([rows, time]);
            let active = mask.clone().sum().reshape([1]).clamp_min(1.0);
            let kl_loss = (per_token_kl * mask)
                .sum()
                .reshape([1])
                .div(active)
                .mul_scalar(config.kl_weight);
            loss = loss + kl_loss;
        }
        Some(loss)
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
        if marginal_weight > f32::EPSILON
            && target_marginal_entropy_bits > f32::EPSILON
            && let Some(loss) = marginal_entropy_floor_loss_from_marginal(
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
                        stop_on_token: None,
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

pub(crate) struct PredictiveContextTrainStep<B: AutodiffBackend> {
    pub output: TrainOutput<LanguageModelTrainItem<B>>,
    pub terminal_state: Option<ModelState<B>>,
}

impl<B: AutodiffBackend> LanguageTrainModel<B> {
    pub(crate) fn predictive_context_probe_loss(
        &self,
        batch: &SequenceBatch<B>,
        neuron_mask: Tensor<B, 4>,
        activity_mask: Tensor<B, 4>,
        probe_tokens: usize,
    ) -> Tensor<B::InnerBackend, 1>
    where
        B::Device: 'static,
        B::FloatTensorPrimitive: 'static,
    {
        let [batch_size, block_size] = batch.inputs.shape().dims();
        let time = probe_tokens.min(block_size).max(1);
        let inputs = Self::slice_tokens(batch.inputs.clone(), batch_size, 0, time).inner();
        let targets = Self::slice_tokens(batch.targets.clone(), batch_size, 0, time).inner();
        let loss_mask = batch
            .loss_mask
            .clone()
            .map(|mask| Self::slice_tokens(mask, batch_size, 0, time).inner());
        let plain = self.model.valid();
        let logits = plain
            .predictive_coding_forward_with_subnetwork_masks(
                inputs,
                neuron_mask.inner(),
                activity_mask.inner(),
            )
            .expect("validated predictive context masks");
        burn_dragon_core::objective::masked_token_mean(
            plain.language_token_losses_from_logits(logits, targets),
            loss_mask,
        )
    }

    pub(crate) fn predictive_context_train_step(
        &self,
        batch: SequenceBatch<B>,
        neuron_mask: Tensor<B, 4>,
        activity_mask: Tensor<B, 4>,
        initial_state: Option<ModelState<B>>,
    ) -> PredictiveContextTrainStep<B>
    where
        B::Device: 'static,
        B::FloatTensorPrimitive: 'static,
    {
        let step_index = self.gradient_scale_step.load(Ordering::Relaxed);
        B::seed(
            &batch.inputs.device(),
            stochastic_step_seed(self.stochastic_seed, step_index, STOCHASTIC_STREAM_MAIN),
        );
        let [batch_size, block_size] = batch.inputs.shape().dims::<2>();
        let chunk_size = self.effective_tbptt_chunk_size(block_size);
        if chunk_size.is_none() {
            let initial_state = if self.tbptt_persist_across_steps {
                Some(initial_state.unwrap_or_else(|| self.model.init_state()))
            } else {
                initial_state
            };
            let step = super::local_predictive_coding::local_predictive_coding_train_step_with_state_and_context_masks(
                &self.model,
                batch.inputs,
                batch.targets,
                batch.loss_mask,
                initial_state,
                super::local_predictive_coding::LocalPredictiveCodingContextMasks {
                    neuron: Some(neuron_mask),
                    activity: Some(activity_mask),
                },
                &self.local_predictive_coding,
                &self.local_predictive_coding_profile,
            );
            debug_assert_eq!(step.report.global_backward_calls, 0);
            if crate::train::profile::enabled() {
                crate::train::profile::record_local_learning_step(step.report.elapsed_ns);
            }
            return PredictiveContextTrainStep {
                output: TrainOutput {
                    grads: self.apply_gradient_scale_schedule(step.grads),
                    item: LanguageModelTrainItem::new(step.loss),
                },
                terminal_state: self
                    .tbptt_persist_across_steps
                    .then_some(step.terminal_state),
            };
        }

        let mut state = initial_state.unwrap_or_else(|| self.model.init_state());
        let mut accumulator = GradientsAccumulator::new();
        let mut total_loss: Option<Tensor<B, 1>> = None;
        let mut total_supervised_tokens: Option<Tensor<B, 1>> = None;
        let mut total_elapsed_ns = 0u128;
        let chunk_size = chunk_size.expect("checked predictive context chunk size");
        for start in (0..block_size).step_by(chunk_size) {
            let end = (start + chunk_size).min(block_size);
            let chunk_inputs = Self::slice_tokens(batch.inputs.clone(), batch_size, start, end);
            let chunk_targets = Self::slice_tokens(batch.targets.clone(), batch_size, start, end);
            let chunk_loss_mask = batch
                .loss_mask
                .clone()
                .map(|mask| Self::slice_tokens(mask, batch_size, start, end));
            let mut step = super::local_predictive_coding::local_predictive_coding_train_step_with_state_and_context_masks(
                &self.model,
                chunk_inputs,
                chunk_targets,
                chunk_loss_mask,
                Some(state),
                super::local_predictive_coding::LocalPredictiveCodingContextMasks {
                    neuron: Some(neuron_mask.clone()),
                    activity: Some(activity_mask.clone()),
                },
                &self.local_predictive_coding,
                &self.local_predictive_coding_profile,
            );
            debug_assert_eq!(step.report.global_backward_calls, 0);
            state = step.terminal_state;
            let supervised_tokens = step.supervised_tokens;
            rescale_gradients_by_device_scalar::<B, _>(
                self,
                &mut step.grads,
                supervised_tokens.clone().inner(),
                false,
            );
            accumulator.accumulate(self, step.grads);
            let weighted_loss = step.loss * supervised_tokens.clone();
            total_loss = Some(match total_loss {
                Some(accumulated) => accumulated + weighted_loss,
                None => weighted_loss,
            });
            total_supervised_tokens = Some(match total_supervised_tokens {
                Some(accumulated) => accumulated + supervised_tokens,
                None => supervised_tokens,
            });
            total_elapsed_ns = total_elapsed_ns.saturating_add(step.report.elapsed_ns);
        }
        if crate::train::profile::enabled() {
            crate::train::profile::record_local_learning_step(total_elapsed_ns);
        }
        let supervised_tokens = total_supervised_tokens
            .expect("predictive context TBPTT requires at least one chunk")
            .clamp_min(1.0);
        let mut grads = accumulator.grads();
        rescale_gradients_by_device_scalar::<B, _>(
            self,
            &mut grads,
            supervised_tokens.clone().inner(),
            true,
        );
        let loss = total_loss.expect("predictive context TBPTT requires at least one chunk")
            / supervised_tokens;
        PredictiveContextTrainStep {
            output: TrainOutput {
                grads: self.apply_gradient_scale_schedule(grads),
                item: LanguageModelTrainItem::new(loss),
            },
            terminal_state: self.tbptt_persist_across_steps.then_some(state),
        }
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
        B::seed(
            &clean_inputs.device(),
            stochastic_step_seed(self.stochastic_seed, step_index, STOCHASTIC_STREAM_MAIN),
        );
        let targets = batch.targets;
        let loss_mask = batch.loss_mask;
        if matches!(self.training_algorithm, TrainingAlgorithm::PredictiveCoding) {
            let [batch_size, block_size] = clean_inputs.shape().dims::<2>();
            let chunk_size = self.effective_tbptt_chunk_size(block_size);
            if chunk_size.is_none() && !self.tbptt_persist_across_steps {
                let step = super::local_predictive_coding::local_predictive_coding_train_step(
                    &self.model,
                    clean_inputs,
                    targets,
                    loss_mask,
                    &self.local_predictive_coding,
                    &self.local_predictive_coding_profile,
                );
                debug_assert_eq!(step.report.global_backward_calls, 0);
                if prof_enabled {
                    crate::train::profile::record_local_learning_step(step.report.elapsed_ns);
                }
                return TrainOutput {
                    grads: self.apply_gradient_scale_schedule(step.grads),
                    item: LanguageModelTrainItem::new(step.loss),
                };
            }

            let mut state = self.load_step_state(batch.reset_stream_state, block_size);
            let mut accumulator = GradientsAccumulator::new();
            let mut total_loss: Option<Tensor<B, 1>> = None;
            let mut total_supervised_tokens: Option<Tensor<B, 1>> = None;
            let mut total_elapsed_ns = 0u128;
            let chunk_size = chunk_size.unwrap_or(block_size);
            for start in (0..block_size).step_by(chunk_size) {
                let end = (start + chunk_size).min(block_size);
                let chunk_inputs = Self::slice_tokens(clean_inputs.clone(), batch_size, start, end);
                let chunk_targets = Self::slice_tokens(targets.clone(), batch_size, start, end);
                let chunk_loss_mask = loss_mask
                    .clone()
                    .map(|mask| Self::slice_tokens(mask, batch_size, start, end));
                let mut step =
                    super::local_predictive_coding::local_predictive_coding_train_step_with_state(
                        &self.model,
                        chunk_inputs,
                        chunk_targets,
                        chunk_loss_mask,
                        state,
                        &self.local_predictive_coding,
                        &self.local_predictive_coding_profile,
                    );
                debug_assert_eq!(step.report.global_backward_calls, 0);
                state = step.terminal_state;
                let supervised_tokens = step.supervised_tokens;
                rescale_gradients_by_device_scalar::<B, _>(
                    self,
                    &mut step.grads,
                    supervised_tokens.clone().inner(),
                    false,
                );
                accumulator.accumulate(self, step.grads);
                let weighted_loss = step.loss * supervised_tokens.clone();
                total_loss = Some(match total_loss {
                    Some(accumulated) => accumulated + weighted_loss,
                    None => weighted_loss,
                });
                total_supervised_tokens = Some(match total_supervised_tokens {
                    Some(accumulated) => accumulated + supervised_tokens,
                    None => supervised_tokens,
                });
                total_elapsed_ns = total_elapsed_ns.saturating_add(step.report.elapsed_ns);
            }
            self.store_step_state(state);
            if prof_enabled {
                crate::train::profile::record_local_learning_step(total_elapsed_ns);
            }
            let supervised_tokens = total_supervised_tokens
                .expect("local PC TBPTT requires at least one chunk")
                .clamp_min(1.0);
            let mut grads = accumulator.grads();
            rescale_gradients_by_device_scalar::<B, _>(
                self,
                &mut grads,
                supervised_tokens.clone().inner(),
                true,
            );
            let loss =
                total_loss.expect("local PC TBPTT requires at least one chunk") / supervised_tokens;
            return TrainOutput {
                grads: self.apply_gradient_scale_schedule(grads),
                item: LanguageModelTrainItem::new(loss),
            };
        }
        let ruliad_policy_batch = batch.ruliad_policy_batch;
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
        // State inference needs gradients only for recurrent-state leaves. Build
        // one current-weight detached parameter view per train step and reuse it
        // across all corrected chunks.
        let predictive_coding_model_needed = tbptt_chunk_size.is_some_and(|chunk_size| {
            let chunks_per_step = block_size.div_ceil(chunk_size.max(1));
            (0..chunks_per_step).any(|chunk_index| {
                self.predictive_coding_active_for_chunk(step_index, chunk_index, chunks_per_step)
            })
        });
        let predictive_coding_model =
            predictive_coding_model_needed.then(|| detach_teacher_model(&self.model));
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
        let mut step_state = self.load_step_state(reset_stream_state, block_size);
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
                let chunks_per_step = block_size.div_ceil(chunk_size);
                for (chunk_index, start) in (0..block_size).step_by(chunk_size).enumerate() {
                    let end = (start + chunk_size).min(block_size);
                    let chunk_inputs = Self::slice_tokens(inputs.clone(), batch_size, start, end);
                    let chunk_summary_event_mask = summary_event_mask
                        .clone()
                        .map(|mask| Self::slice_tokens(mask, batch_size, start, end));
                    if self.predictive_coding_active_for_chunk(
                        step_index,
                        chunk_index,
                        chunks_per_step,
                    ) && matches!(
                        self.predictive_coding.observation_contract,
                        PredictiveCodingObservationContract::OracleNextTokenNegativeControl
                    ) {
                        let chunk_targets =
                            Self::slice_tokens(targets.clone(), batch_size, start, end);
                        let chunk_loss_mask = loss_mask
                            .clone()
                            .map(|mask| Self::slice_tokens(mask, batch_size, start, end));
                        let (corrected_state, report) = self
                            .correct_state_with_oracle_predictive_coding_using_model(
                                predictive_coding_model
                                    .as_ref()
                                    .expect("enabled predictive-coding model"),
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
                let chunks_per_step = block_size.div_ceil(chunk_size);

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
                    let predictive_coding_active = self.predictive_coding_active_for_chunk(
                        step_index,
                        chunk_index,
                        chunks_per_step,
                    );
                    let observed_pc_entry = (predictive_coding_active
                        && matches!(
                            self.predictive_coding.observation_contract,
                            PredictiveCodingObservationContract::ObservedPrefix
                        ))
                    .then(|| step_state.detached_clone())
                    .filter(|state| {
                        Self::predictive_coding_state_has_latents(
                            state,
                            self.predictive_coding.state_scope,
                        )
                    });
                    if predictive_coding_active
                        && matches!(
                            self.predictive_coding.observation_contract,
                            PredictiveCodingObservationContract::OracleNextTokenNegativeControl
                        )
                    {
                        let (corrected_state, report) = self
                            .correct_state_with_oracle_predictive_coding_using_model(
                                predictive_coding_model
                                    .as_ref()
                                    .expect("enabled predictive-coding model"),
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
                    let chunk_loss = if let Some(mask) = chunk_summary_event_mask.clone() {
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
                    let mut chunk_loss = self.add_latent_dragon_state_auxiliary_loss(
                        chunk_loss,
                        &step_state,
                        recurrent_teacher_state.as_ref(),
                    );
                    total_forward_ns += chunk_forward_start.elapsed().as_nanos();

                    if let Some(entry_state) = observed_pc_entry {
                        let (corrected_state, mut report) = self
                            .correct_state_from_observed_prefix_using_model(
                                predictive_coding_model
                                    .as_ref()
                                    .expect("enabled predictive-coding model"),
                                entry_state,
                                chunk_clean_inputs,
                                chunk_loss_mask,
                                chunk_summary_event_mask,
                            );
                        if report.chunks_corrected > 0 {
                            if matches!(
                                self.predictive_coding.parameter_update,
                                PredictiveCodingParameterUpdate::Optimizer
                            ) {
                                let (constraint, components) = self
                                    .predictive_coding_amortization_constraint(
                                        &step_state,
                                        &corrected_state,
                                    );
                                report.amortization_components = components;
                                if let Some(constraint) = constraint {
                                    if self.predictive_coding.sync_diagnostics {
                                        report.amortization_loss = Some(scalar_tensor_to_f64(
                                            constraint.clone().detach().inner(),
                                        ));
                                    }
                                    chunk_loss = chunk_loss + constraint;
                                }
                            } else {
                                // This explicitly non-learning control retains online state
                                // inference so it remains distinct from the AdamW baseline.
                                step_state = corrected_state;
                            }
                        }
                        if self.predictive_coding.sync_diagnostics {
                            report.record();
                        } else {
                            predictive_coding_step_report.accumulate_unsynced(report);
                        }
                    }

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

                if let Some(contract_loss) = self.ruliad_answer_contract_auxiliary_loss(
                    ruliad_policy_batch.as_deref(),
                    &targets.device(),
                    block_size,
                ) {
                    total_loss = Some(match total_loss {
                        Some(accumulated) => accumulated + contract_loss.clone().detach(),
                        None => contract_loss.clone().detach(),
                    });
                    let contract_grads = contract_loss.backward();
                    accumulator.accumulate(self, GradientsParams::from_grads(contract_grads, self));
                }

                if let Some(recovery_loss) = self.ruliad_structured_answer_recovery_auxiliary_loss(
                    ruliad_policy_batch.as_deref(),
                    &targets.device(),
                    block_size,
                ) {
                    total_loss = Some(match total_loss {
                        Some(accumulated) => accumulated + recovery_loss.clone().detach(),
                        None => recovery_loss.clone().detach(),
                    });
                    let recovery_grads = recovery_loss.backward();
                    accumulator.accumulate(self, GradientsParams::from_grads(recovery_grads, self));
                }

                let field_binding_weight = self.ruliad_field_binding_contrast_weight();
                if field_binding_weight > f32::EPSILON {
                    let field_binding_loss =
                        if let Some(policy_batch) = ruliad_policy_batch.as_deref() {
                            self.ruliad_field_binding_contrast_loss(
                                policy_batch,
                                &targets.device(),
                                block_size,
                            )
                        } else {
                            self.write_ruliad_field_binding_contrast_skip(
                                "missing_policy_batch",
                                field_binding_weight,
                            );
                            None
                        };
                    if let Some(field_binding_loss) = field_binding_loss {
                        total_loss = Some(match total_loss {
                            Some(accumulated) => accumulated + field_binding_loss.clone().detach(),
                            None => field_binding_loss.clone().detach(),
                        });
                        let field_binding_grads = field_binding_loss.backward();
                        accumulator.accumulate(
                            self,
                            GradientsParams::from_grads(field_binding_grads, self),
                        );
                    }
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
        let auxiliary_objective_start = prof_enabled.then(Instant::now);
        let loss = if let Some(rollout_loss) =
            self.greedy_rollout_unlikelihood_loss(clean_inputs_for_aux)
        {
            loss + rollout_loss
        } else {
            loss
        };
        let loss = if let Some(contract_loss) = self.ruliad_answer_contract_auxiliary_loss(
            ruliad_policy_batch.as_deref(),
            &targets.device(),
            block_size,
        ) {
            loss + contract_loss
        } else {
            loss
        };
        let loss = if let Some(recovery_loss) = self
            .ruliad_structured_answer_recovery_auxiliary_loss(
                ruliad_policy_batch.as_deref(),
                &targets.device(),
                block_size,
            ) {
            loss + recovery_loss
        } else {
            loss
        };
        let contrast_weight = self.ruliad_structured_contrast_weight();
        let loss = if contrast_weight > f32::EPSILON {
            if let Some(policy_batch) = ruliad_policy_batch.as_deref() {
                if let Some(contrast_loss) = self.ruliad_structured_answer_contrast_loss(
                    policy_batch,
                    &targets.device(),
                    block_size,
                ) {
                    loss + contrast_loss
                } else {
                    loss
                }
            } else {
                self.write_ruliad_structured_contrast_skip("missing_policy_batch", contrast_weight);
                loss
            }
        } else {
            loss
        };
        let field_binding_weight = self.ruliad_field_binding_contrast_weight();
        let loss = if field_binding_weight > f32::EPSILON {
            if let Some(policy_batch) = ruliad_policy_batch.as_deref() {
                if let Some(field_binding_loss) = self.ruliad_field_binding_contrast_loss(
                    policy_batch,
                    &targets.device(),
                    block_size,
                ) {
                    loss + field_binding_loss
                } else {
                    loss
                }
            } else {
                self.write_ruliad_field_binding_contrast_skip(
                    "missing_policy_batch",
                    field_binding_weight,
                );
                loss
            }
        } else {
            loss
        };
        let loss = if let Some(policy_batch) = ruliad_policy_batch.as_deref()
            && let Some(rollout_imitation_loss) = self.ruliad_verifier_rollout_imitation_loss(
                policy_batch,
                &targets.device(),
                block_size,
            ) {
            loss + rollout_imitation_loss
        } else {
            loss
        };
        B::seed(
            &targets.device(),
            stochastic_step_seed(
                self.stochastic_seed,
                step_index,
                STOCHASTIC_STREAM_PROOF_POLICY,
            ),
        );
        let proof_policy_start = prof_enabled.then(Instant::now);
        let loss = if let Some(policy_batch) = ruliad_policy_batch.as_deref()
            && let Some(proof_policy_loss) =
                self.ruliad_proof_policy_dagger_loss(policy_batch, &targets.device(), block_size)
        {
            loss + proof_policy_loss
        } else {
            loss
        };
        let proof_policy_ns = proof_policy_start
            .map(|start| start.elapsed().as_nanos())
            .unwrap_or_default();
        B::seed(
            &targets.device(),
            stochastic_step_seed(
                self.stochastic_seed,
                step_index,
                STOCHASTIC_STREAM_VERIFIER_POLICY,
            ),
        );
        let loss = if let Some(policy_batch) = ruliad_policy_batch.as_deref()
            && let Some(policy_loss) =
                self.ruliad_verifier_policy_loss(policy_batch, &targets.device(), block_size)
        {
            loss + policy_loss
        } else {
            loss
        };
        let auxiliary_objective_ns = auxiliary_objective_start
            .map(|start| start.elapsed().as_nanos())
            .unwrap_or_default();
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
            crate::train::profile::record_auxiliary_objectives(
                auxiliary_objective_ns,
                proof_policy_ns,
            );
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
    pub(crate) fn sequence_state_diagnostics(
        state: &ModelState<B>,
        max_rho_slots: usize,
    ) -> Option<SequenceStateDiagnostics> {
        let mut rho_rms: Option<Tensor<B, 1>> = None;
        let mut slot_variance_ratio: Option<Tensor<B, 1>> = None;
        let mut slot_redundancy: Option<Tensor<B, 1>> = None;
        let mut layers = 0usize;

        for rho in state.layers.iter().filter_map(|layer| layer.rho.as_ref()) {
            let [batch, heads, original_slots, dim] = rho.shape().dims::<4>();
            if batch == 0 || heads == 0 || original_slots < 2 || dim == 0 {
                continue;
            }
            let rho = Self::sample_rho_slots_with_limit(
                rho.clone(),
                original_slots,
                max_rho_slots.max(2),
            );
            let [batch, heads, slots, dim] = rho.shape().dims::<4>();
            let groups = batch.saturating_mul(heads);
            let rows = rho.reshape([groups, slots, dim]);
            let layer_energy = rows.clone().powf_scalar(2.0).mean().reshape([1]);
            let layer_rms = layer_energy.clone().clamp_min(1.0e-12).sqrt();

            let slot_mean = rows.clone().mean_dim(1);
            let slot_variance = (rows.clone() - slot_mean.repeat_dim(1, slots))
                .powf_scalar(2.0)
                .mean()
                .reshape([1]);
            let layer_variance_ratio = slot_variance / layer_energy.clamp_min(1.0e-12);

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
            let off_diagonal = (groups * slots * slots.saturating_sub(1)).max(1) as f32;
            let layer_redundancy = (total_sq - diag_sq)
                .clamp_min(0.0)
                .div_scalar(off_diagonal)
                .sqrt();

            rho_rms = Some(match rho_rms {
                Some(total) => total + layer_rms,
                None => layer_rms,
            });
            slot_variance_ratio = Some(match slot_variance_ratio {
                Some(total) => total + layer_variance_ratio,
                None => layer_variance_ratio,
            });
            slot_redundancy = Some(match slot_redundancy {
                Some(total) => total + layer_redundancy,
                None => layer_redundancy,
            });
            layers = layers.saturating_add(1);
        }

        let scalar = |tensor: Tensor<B, 1>| {
            tensor
                .div_scalar(layers.max(1) as f32)
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("sequence-state diagnostic tensor")[0] as f64
        };
        Some(SequenceStateDiagnostics {
            rho_layers: layers,
            rho_rms: scalar(rho_rms?),
            rho_slot_variance_ratio: scalar(slot_variance_ratio?),
            rho_slot_redundancy: scalar(slot_redundancy?),
        })
    }

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

    pub(crate) fn step_with_predictive_context_stream_state(
        &self,
        batch: SequenceBatch<B>,
        neuron_mask: Tensor<B, 4>,
        activity_mask: Tensor<B, 4>,
        state: &mut ModelState<B>,
    ) -> LanguageModelOutput<B>
    where
        B::Device: 'static,
        B::FloatTensorPrimitive: 'static,
    {
        if batch.reset_stream_state {
            *state = self.model.init_state();
        }
        debug_assert!(
            batch.summary_event_mask.is_none(),
            "analytic predictive coding rejects summary memory"
        );
        let [batch_size, block_size] = batch.inputs.shape().dims::<2>();
        let chunk_size = self
            .effective_tbptt_chunk_size(block_size)
            .unwrap_or(block_size)
            .max(1);
        let mut loss: Option<Tensor<B, 1>> = None;
        for start in (0..block_size).step_by(chunk_size) {
            let end = (start + chunk_size).min(block_size);
            let inputs = Self::slice_tokens(batch.inputs.clone(), batch_size, start, end);
            let targets = Self::slice_tokens(batch.targets.clone(), batch_size, start, end);
            let loss_mask = batch
                .loss_mask
                .clone()
                .map(|mask| Self::slice_tokens(mask, batch_size, start, end));
            let logits = self
                .model
                .predictive_coding_forward_with_subnetwork_masks_and_state(
                    inputs,
                    neuron_mask.clone(),
                    activity_mask.clone(),
                    state,
                )
                .expect("validated predictive context masks");
            let chunk_loss = masked_token_mean(
                self.model
                    .language_token_losses_from_logits(logits, targets),
                loss_mask,
            )
            .mul_scalar((end - start) as f32 / block_size.max(1) as f32);
            loss = Some(match loss {
                Some(total) => total + chunk_loss,
                None => chunk_loss,
            });
        }
        LanguageModelOutput::new(loss.expect("streaming context batch must contain tokens"))
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
) -> Option<LaggedPredictionTensors<B>> {
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

    #[test]
    fn causal_predictive_coding_cadence_uses_post_observation_chunks() {
        let due = |step, chunk| {
            predictive_coding_chunk_due(
                PredictiveCodingObservationContract::ObservedPrefix,
                step,
                chunk,
                4,
                2,
            )
        };

        assert_eq!(
            (0..4).map(|chunk| due(0, chunk)).collect::<Vec<_>>(),
            vec![false, true, false, true]
        );
        assert_eq!(
            (0..4).map(|chunk| due(1, chunk)).collect::<Vec<_>>(),
            vec![false, true, false, true]
        );
    }

    #[test]
    fn causal_predictive_coding_sparse_cadence_crosses_step_boundaries() {
        let due = |step, chunk| {
            predictive_coding_chunk_due(
                PredictiveCodingObservationContract::ObservedPrefix,
                step,
                chunk,
                4,
                8,
            )
        };

        assert!((0..4).all(|chunk| !due(0, chunk)));
        assert_eq!(
            (0..4).map(|chunk| due(1, chunk)).collect::<Vec<_>>(),
            vec![false, false, false, true]
        );
    }

    #[test]
    fn oracle_negative_control_preserves_historical_cadence_phase() {
        let due = |chunk| {
            predictive_coding_chunk_due(
                PredictiveCodingObservationContract::OracleNextTokenNegativeControl,
                0,
                chunk,
                4,
                2,
            )
        };

        assert_eq!(
            (0..4).map(due).collect::<Vec<_>>(),
            vec![true, false, true, false]
        );
    }

    #[test]
    fn stochastic_step_streams_are_reproducible_and_domain_separated() {
        let base = 1_337;
        let main = stochastic_step_seed(base, 19, STOCHASTIC_STREAM_MAIN);
        assert_eq!(main, stochastic_step_seed(base, 19, STOCHASTIC_STREAM_MAIN));
        assert_ne!(main, stochastic_step_seed(base, 20, STOCHASTIC_STREAM_MAIN));
        assert_ne!(
            main,
            stochastic_step_seed(base, 19, STOCHASTIC_STREAM_PROOF_POLICY)
        );
        assert_ne!(
            stochastic_step_seed(base, 19, STOCHASTIC_STREAM_PROOF_POLICY),
            stochastic_step_seed(base, 19, STOCHASTIC_STREAM_VERIFIER_POLICY)
        );
    }

    #[test]
    fn streaming_state_is_pipeline_owned_and_clone_shared() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let model_a = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_tbptt_persist_across_steps(true);
        let model_b = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_tbptt_persist_across_steps(true);
        let mut state = model_a.model.init_state();
        state.position = 17;
        model_a.store_step_state(state);

        assert_eq!(
            model_a
                .peek_step_state_for_test()
                .expect("model a state")
                .position,
            17
        );
        assert!(
            model_b.peek_step_state_for_test().is_none(),
            "independent pipelines must not share recurrent state"
        );
        let cloned = model_a.clone();
        assert_eq!(
            cloned
                .peek_step_state_for_test()
                .expect("cloned learner state")
                .position,
            17,
            "Burn learner clones must retain the same pipeline runtime cell"
        );
        assert_eq!(model_a.load_step_state(true, 4).position, 0);
        assert!(model_a.peek_step_state_for_test().is_none());
    }

    #[test]
    fn local_predictive_coding_tbptt_carries_and_resets_rho_state() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let mut config = tiny_model_config();
        config.n_layer = 2;
        config.sequence_kernel =
            burn_dragon_core::SequenceKernelConfig::dense_score_short_context();
        config.fused_kernels.rotary_embedding = burn_dragon_core::RotaryEmbedding::Alibi;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_training_algorithm(TrainingAlgorithm::PredictiveCoding)
            .with_local_predictive_coding(LocalPredictiveCodingConfig {
                solver: LocalPredictiveCodingSolver::FixedPrediction,
                ..LocalPredictiveCodingConfig::default()
            })
            .with_tbptt_chunk_size(Some(2))
            .with_tbptt_persist_across_steps(true);
        let batch = |reset_stream_state| SequenceBatch {
            inputs: Tensor::from_data(
                TensorData::new(vec![1_i64, 2, 3, 4, 5, 6, 7, 8], [1, 8]),
                &device,
            ),
            targets: Tensor::from_data(
                TensorData::new(vec![2_i64, 3, 4, 5, 6, 7, 8, 9], [1, 8]),
                &device,
            ),
            loss_mask: None,
            summary_event_mask: None,
            ruliad_policy_batch: None,
            reset_stream_state,
        };

        let first = burn_train::TrainStep::step(&model, batch(true));
        assert_eq!(first.grads.len(), 9);
        let first_state = model
            .peek_step_state_for_test()
            .expect("persistent local PC state after first step");
        assert_eq!(first_state.position, 8);
        assert!(first_state.layers.iter().all(|layer| layer.rho.is_some()));

        let second = burn_train::TrainStep::step(&model, batch(false));
        assert_eq!(second.grads.len(), 9);
        assert_eq!(
            model
                .peek_step_state_for_test()
                .expect("persistent local PC state after second step")
                .position,
            16
        );

        let reset = burn_train::TrainStep::step(&model, batch(true));
        assert_eq!(reset.grads.len(), 9);
        assert_eq!(
            model
                .peek_step_state_for_test()
                .expect("reset local PC state")
                .position,
            8
        );
    }

    #[test]
    fn local_predictive_coding_tbptt_uses_supervised_token_loss_weighting() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 71);
        let mut config = tiny_model_config();
        config.n_layer = 2;
        config.sequence_kernel =
            burn_dragon_core::SequenceKernelConfig::dense_score_short_context();
        config.fused_kernels.rotary_embedding = burn_dragon_core::RotaryEmbedding::Alibi;
        let base = DragonModel::<TestBackend>::new(config, &device);
        let make_model = |model| {
            LanguageTrainModel::new(model)
                .with_training_algorithm(TrainingAlgorithm::PredictiveCoding)
                .with_local_predictive_coding(LocalPredictiveCodingConfig {
                    solver: LocalPredictiveCodingSolver::FixedPrediction,
                    ..LocalPredictiveCodingConfig::default()
                })
        };
        let chunked = make_model(base).with_tbptt_chunk_size(Some(2));
        let batch = || SequenceBatch {
            inputs: Tensor::from_data(
                TensorData::new(vec![1_i64, 2, 3, 4, 5, 6, 7, 8], [1, 8]),
                &device,
            ),
            targets: Tensor::from_data(
                TensorData::new(vec![2_i64, 3, 4, 5, 6, 7, 8, 9], [1, 8]),
                &device,
            ),
            loss_mask: Some(Tensor::from_data(
                TensorData::new(vec![1_i64, 0, 0, 0, 1, 1, 1, 1], [1, 8]),
                &device,
            )),
            summary_event_mask: None,
            ruliad_policy_batch: None,
            reset_stream_state: true,
        };

        let source = batch();
        let mut state = chunked.model.init_state_ephemeral();
        let mut weighted_loss = 0.0_f32;
        let mut supervised_tokens = 0.0_f32;
        let config = LocalPredictiveCodingConfig {
            solver: LocalPredictiveCodingSolver::FixedPrediction,
            ..LocalPredictiveCodingConfig::default()
        };
        for start in (0..8).step_by(2) {
            let end = start + 2;
            let step = crate::train::local_predictive_coding_derivatives_with_state(
                &chunked.model,
                LanguageTrainModel::<TestBackend>::slice_tokens(
                    source.inputs.clone(),
                    1,
                    start,
                    end,
                ),
                LanguageTrainModel::<TestBackend>::slice_tokens(
                    source.targets.clone(),
                    1,
                    start,
                    end,
                ),
                source.loss_mask.clone().map(|mask| {
                    LanguageTrainModel::<TestBackend>::slice_tokens(mask, 1, start, end)
                }),
                state,
                &config,
            )
            .expect("manual recurrent local-PC factor");
            let chunk_loss = burn_pc::diagnostic_scalar_f32(step.loss.inner());
            let chunk_tokens = burn_pc::diagnostic_scalar_f32(step.supervised_tokens.inner());
            weighted_loss += chunk_loss * chunk_tokens;
            supervised_tokens += chunk_tokens;
            state = step.terminal_state;
        }
        let expected_loss = weighted_loss / supervised_tokens.max(1.0);
        let chunked_loss = scalar_loss(burn_train::TrainStep::step(&chunked, batch()));
        assert!(
            (expected_loss - chunked_loss).abs() < 1.0e-5,
            "expected={expected_loss} chunked={chunked_loss}"
        );
    }

    #[test]
    fn sequence_state_diagnostics_detect_redundant_rho_slots() {
        let device = burn::tensor::Device::<TestInnerBackend>::default();
        let mut state = ModelState::<TestInnerBackend>::new(1);
        state.layers[0].rho = Some(Tensor::from_data(
            TensorData::new(
                vec![1.0f32, -1.0, 0.5, -0.5, 1.0, -1.0, 0.5, -0.5],
                [1, 1, 2, 4],
            ),
            &device,
        ));

        let diagnostics =
            LanguageTrainModel::<TestInnerBackend>::sequence_state_diagnostics(&state, 2)
                .expect("rho diagnostics");
        assert_eq!(diagnostics.rho_layers, 1);
        assert!((diagnostics.rho_rms - 0.790_569_4).abs() < 1.0e-5);
        assert!(diagnostics.rho_slot_variance_ratio.abs() < 1.0e-6);
        assert!((diagnostics.rho_slot_redundancy - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn sequence_state_diagnostics_detect_distinct_rho_slots() {
        let device = burn::tensor::Device::<TestInnerBackend>::default();
        let mut state = ModelState::<TestInnerBackend>::new(1);
        state.layers[0].rho = Some(Tensor::from_data(
            TensorData::new(
                vec![1.0f32, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
                [1, 1, 2, 4],
            ),
            &device,
        ));

        let diagnostics =
            LanguageTrainModel::<TestInnerBackend>::sequence_state_diagnostics(&state, 2)
                .expect("rho diagnostics");
        assert!(diagnostics.rho_slot_variance_ratio > 0.49);
        assert!(diagnostics.rho_slot_redundancy < 1.0e-5);
    }

    #[test]
    fn terminal_sequence_state_elision_requires_a_stateless_training_contract() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let mut config = tiny_model_config();
        config.sequence_kernel =
            burn_dragon_core::SequenceKernelConfig::dense_score_short_context();

        let baseline =
            LanguageTrainModel::new(DragonModel::<TestBackend>::new(config.clone(), &device));
        assert!(
            !baseline.load_step_state(false, 4).layers[0].retain_terminal_sequence_state,
            "an unchunked nonpersistent dense-score step should elide unused terminal state"
        );

        let retained =
            LanguageTrainModel::new(DragonModel::<TestBackend>::new(config.clone(), &device))
                .with_ephemeral_terminal_sequence_state_retention(true);
        assert!(retained.load_step_state(false, 4).layers[0].retain_terminal_sequence_state);

        let chunked =
            LanguageTrainModel::new(DragonModel::<TestBackend>::new(config.clone(), &device))
                .with_tbptt_chunk_size(Some(2));
        assert!(chunked.load_step_state(false, 4).layers[0].retain_terminal_sequence_state);

        let persistent =
            LanguageTrainModel::new(DragonModel::<TestBackend>::new(config.clone(), &device))
                .with_tbptt_persist_across_steps(true);
        assert!(persistent.load_step_state(false, 4).layers[0].retain_terminal_sequence_state);

        let mut pipeline_config = config.clone();
        pipeline_config.n_layer = 2;
        let pipeline =
            LanguageTrainModel::new(DragonModel::<TestBackend>::new(pipeline_config, &device))
                .with_pipeline_plan(Some(tiny_pipeline_plan()));
        assert!(pipeline.load_step_state(false, 4).layers[0].retain_terminal_sequence_state);

        let predictive_coding = PredictiveCodingConfig {
            enabled: true,
            ..Default::default()
        };
        let predictive =
            LanguageTrainModel::new(DragonModel::<TestBackend>::new(config.clone(), &device))
                .with_predictive_coding(predictive_coding);
        assert!(predictive.load_step_state(false, 4).layers[0].retain_terminal_sequence_state);

        let latent_reasoning = LatentReasoningTrainingConfig {
            enabled: true,
            sigreg: crate::config::LatentReasoningSigRegConfig {
                target: crate::config::LatentReasoningSigRegTarget::RhoMemorySlots,
                ..Default::default()
            },
            ..Default::default()
        };
        let rho_regularized =
            LanguageTrainModel::new(DragonModel::<TestBackend>::new(config.clone(), &device))
                .with_latent_reasoning(latent_reasoning);
        assert!(rho_regularized.load_step_state(false, 4).layers[0].retain_terminal_sequence_state);

        let dragon_state_reasoning = LatentReasoningTrainingConfig {
            enabled: true,
            dragon_state: crate::config::DragonStateConsistencyConfig {
                enabled: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let dragon_state =
            LanguageTrainModel::new(DragonModel::<TestBackend>::new(config.clone(), &device))
                .with_latent_reasoning(dragon_state_reasoning);
        assert!(dragon_state.load_step_state(false, 4).layers[0].retain_terminal_sequence_state);

        let mut summary_memory_config = config;
        summary_memory_config.summary_memory.enabled = true;
        let summary_memory = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            summary_memory_config,
            &device,
        ));
        assert!(summary_memory.load_step_state(false, 4).layers[0].retain_terminal_sequence_state);

        let mut reference_config = tiny_model_config();
        reference_config.sequence_kernel = burn_dragon_core::SequenceKernelConfig::default();
        let reference =
            LanguageTrainModel::new(DragonModel::<TestBackend>::new(reference_config, &device));
        assert!(reference.load_step_state(false, 4).layers[0].retain_terminal_sequence_state);

        let mut multi_step_config = tiny_model_config();
        multi_step_config.sequence_kernel =
            burn_dragon_core::SequenceKernelConfig::dense_score_short_context();
        multi_step_config.rollout_fast_steps_per_slow_step = 2;
        let multi_step =
            LanguageTrainModel::new(DragonModel::<TestBackend>::new(multi_step_config, &device));
        assert!(multi_step.load_step_state(false, 4).layers[0].retain_terminal_sequence_state);

        let mut y_neuron_config = tiny_model_config();
        y_neuron_config.sequence_kernel =
            burn_dragon_core::SequenceKernelConfig::dense_score_short_context();
        y_neuron_config.y_neuron_recurrence.enabled = true;
        let y_neuron =
            LanguageTrainModel::new(DragonModel::<TestBackend>::new(y_neuron_config, &device));
        assert!(y_neuron.load_step_state(false, 4).layers[0].retain_terminal_sequence_state);

        let mut hierarchical_config = tiny_model_config();
        hierarchical_config.sequence_kernel =
            burn_dragon_core::SequenceKernelConfig::dense_score_short_context();
        hierarchical_config.hierarchical_dragon.enabled = true;
        let hierarchical = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            hierarchical_config,
            &device,
        ));
        assert!(hierarchical.load_step_state(false, 4).layers[0].retain_terminal_sequence_state);

        let mut clocked_config = tiny_model_config();
        clocked_config.sequence_kernel =
            burn_dragon_core::SequenceKernelConfig::dense_score_short_context();
        clocked_config.clocked_slow_memory.enabled = true;
        let clocked =
            LanguageTrainModel::new(DragonModel::<TestBackend>::new(clocked_config, &device));
        assert!(clocked.load_step_state(false, 4).layers[0].retain_terminal_sequence_state);
    }

    fn ruliad_test_score(
        status: burn_dragon_universality::ruliad::RuliadAnswerStatus,
        partial_progress_ppm: usize,
        completion_quality_ppm: usize,
    ) -> burn_dragon_universality::ruliad::RuliadReasoningScore {
        burn_dragon_universality::ruliad::RuliadReasoningScore {
            version: 1,
            status,
            correct_field_count: 0,
            expected_field_count: 1,
            observed_field_count: 0,
            partial_progress_ppm,
            certificate_valid_prefix_steps: 0,
            certificate_expected_steps: 0,
            certificate_prefix_ppm: 0,
            generated_token_count: 8,
            hash_canary: false,
            answer_terminated: true,
            completion_quality_ppm,
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

    #[test]
    fn all_active_context_stream_validation_matches_dense_tbptt() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7_311);
        let mut config = tiny_model_config();
        config.sequence_kernel.executor =
            burn_dragon_core::SequenceTrainingExecutor::DenseScoreShortContext;
        config.fused_kernels.rotary_embedding = burn_dragon_core::RotaryEmbedding::Alibi;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_tbptt_chunk_size(Some(2));
        model
            .model
            .predictive_coding_support()
            .expect("PC-compatible test model");
        let mut dense_state = model.model.init_state();
        let mut context_state = model.model.init_state();
        let dense = model.step_with_stream_state(batch(&device), &mut dense_state);
        let routed = model.step_with_predictive_context_stream_state(
            batch(&device),
            Tensor::ones([1, 1, 1, 8], &device),
            Tensor::ones([1, 1, 1, 8], &device),
            &mut context_state,
        );
        let dense_loss: LossValue<TestBackend> = dense.adapt();
        let routed_loss: LossValue<TestBackend> = routed.adapt();
        let loss_diff = (dense_loss.value() - routed_loss.value())
            .abs()
            .max()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("loss difference")[0];
        assert!(
            loss_diff < 1.0e-5,
            "routed stream loss mismatch: {loss_diff}"
        );
        assert_eq!(dense_state.position, context_state.position);
        let rho_diff = (dense_state.layers[0].rho.clone().expect("dense rho")
            - context_state.layers[0].rho.clone().expect("context rho"))
        .abs()
        .max()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("rho difference")[0];
        assert!(rho_diff < 1.0e-5, "routed stream rho mismatch: {rho_diff}");
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
    fn oracle_predictive_coding_negative_control_corrects_state() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 11);
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_tbptt_chunk_size(Some(2))
        .with_predictive_coding(PredictiveCodingConfig {
            enabled: true,
            observation_contract:
                PredictiveCodingObservationContract::OracleNextTokenNegativeControl,
            allow_oracle_target_leak: true,
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
        let (_corrected_state, report) = model.correct_state_with_oracle_predictive_coding(
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

    #[test]
    fn observed_prefix_predictive_coding_uses_no_future_target() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 13);
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_tbptt_chunk_size(Some(2))
        .with_predictive_coding(PredictiveCodingConfig {
            enabled: true,
            observation_contract: PredictiveCodingObservationContract::ObservedPrefix,
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
        let observed_inputs =
            LanguageTrainModel::<TestBackend>::slice_tokens(batch.inputs, batch_size, 2, 4);
        let (corrected_state, report) =
            model.correct_state_from_observed_prefix(state, observed_inputs, None, None);

        assert!(report.chunks_corrected > 0, "report={report:?}");
        assert!(
            report
                .energy_before
                .zip(report.energy_after)
                .is_some_and(|(before, after)| {
                    before.is_finite() && after.is_finite() && after <= before + 1.0e-4
                }),
            "observed-prefix inference should descend its causal energy: {report:?}"
        );
        assert!(
            LanguageTrainModel::<TestBackend>::predictive_coding_state_has_latents(
                &corrected_state,
                PredictiveCodingStateScope::Core,
            )
        );
    }

    #[test]
    fn observed_prefix_empty_entry_replays_instead_of_resetting_state() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 17);
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_predictive_coding(PredictiveCodingConfig {
            enabled: true,
            observation_contract: PredictiveCodingObservationContract::ObservedPrefix,
            ..Default::default()
        });
        let observed_inputs = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![0, 1, 2, 3], [2, 2]),
            &device,
        );

        let (replayed, report) = model.correct_state_from_observed_prefix(
            model.model.init_state_ephemeral(),
            observed_inputs,
            None,
            None,
        );

        assert_eq!(report.skipped_empty_state, 1);
        assert_eq!(report.chunks_corrected, 0);
        assert_eq!(replayed.position, 2);
        assert!(
            LanguageTrainModel::<TestBackend>::predictive_coding_state_has_latents(
                &replayed,
                PredictiveCodingStateScope::Core,
            )
        );
    }

    #[test]
    fn predictive_coding_amortization_constraint_detects_state_drift() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 19);
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_predictive_coding(PredictiveCodingConfig {
            enabled: true,
            amortization_tolerance: 0.0,
            ..Default::default()
        });
        let mut student = model.model.init_state_ephemeral();
        let inputs = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![0, 1, 2, 3], [2, 2]),
            &device,
        );
        model.model.forward_hidden_with_state(inputs, &mut student);
        let teacher = student.detached_clone();
        let (same, components) =
            model.predictive_coding_amortization_constraint(&student, &teacher);
        let same = scalar_tensor_to_f64(same.expect("same-state constraint").detach().inner());
        assert!(components > 0);
        assert!(same <= 1.0e-8, "same-state constraint={same}");

        let mut drifted = teacher;
        for layer in &mut drifted.layers {
            layer.rho = layer.rho.take().map(|rho| rho.add_scalar(1.0).detach());
            layer.y_neuron_state = layer
                .y_neuron_state
                .take()
                .map(|state| state.add_scalar(1.0).detach());
        }
        let (drift, drift_components) =
            model.predictive_coding_amortization_constraint(&student, &drifted);
        let drift = scalar_tensor_to_f64(drift.expect("drift constraint").detach().inner());
        assert_eq!(drift_components, components);
        assert!(drift > 1.0e-4, "drift constraint={drift}");
    }

    #[test]
    fn predictive_coding_amortization_has_finite_zero_error_gradient() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let student = Tensor::<TestBackend, 3>::zeros([2, 2, 4], &device).require_grad();
        let teacher = Tensor::<TestBackend, 3>::zeros([2, 2, 4], &device);
        let mut total = None;
        let mut components = 0;
        let mut sample_indices = PredictiveCodingSampleIndexCache::new();
        accumulate_predictive_coding_amortization_constraint(
            &mut total,
            &mut components,
            &Some(student.clone()),
            &Some(teacher),
            PredictiveCodingAmortizationConstraint {
                sample_axis: 2,
                max_slots: 4,
                sample_offset: 0,
                tolerance: 0.0,
                eps: 1.0e-8,
            },
            &mut sample_indices,
        );

        let grads = total.expect("constraint").backward();
        let grad = student.grad(&grads).expect("student state gradient");
        let values = grad
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("gradient values");

        assert_eq!(components, 1);
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(values.iter().all(|value| value.abs() <= 1.0e-8));
    }

    #[test]
    fn observed_prefix_train_step_amortizes_without_online_state_replacement() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 23);
        crate::train::profile::reset_predictive_coding();
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_tbptt_chunk_size(Some(2))
        .with_predictive_coding(PredictiveCodingConfig {
            enabled: true,
            observation_contract: PredictiveCodingObservationContract::ObservedPrefix,
            parameter_update: PredictiveCodingParameterUpdate::Optimizer,
            steps: 1,
            step_size: 0.01,
            ..Default::default()
        });

        let loss = scalar_loss(TrainStep::step(&model, batch(&device)));
        let profile = crate::train::profile::take_predictive_coding();

        assert!(loss.is_finite());
        assert!(profile.chunks_corrected > 0, "profile={profile:?}");
        assert!(
            profile.amortization_components > 0,
            "causal PC must constrain the ordinary deployment transition: {profile:?}"
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
    fn predictive_coding_all_scope_covers_every_slow_state_family() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let mut state = ModelState::<TestBackend>::new(1);
        let layer = &mut state.layers[0];
        layer.slow_rho = Some(Tensor::zeros([1, 1, 2, 2], &device));
        layer.slow_rho_norm = Some(Tensor::zeros([1, 1, 2], &device));
        layer.slow_sequence_aux = Some(Tensor::zeros([1, 1, 2, 2], &device));
        layer.slow_mamba_angle_state = Some(Tensor::zeros([1, 1, 2], &device));
        layer.slow_mamba_k_state = Some(Tensor::zeros([1, 1, 2], &device));
        layer.slow_mamba_v_state = Some(Tensor::zeros([1, 1, 2], &device));
        layer.hierarchical_slow_hidden = Some(Tensor::zeros([1, 1, 2, 2], &device));

        assert!(
            !LanguageTrainModel::<TestBackend>::predictive_coding_state_has_latents(
                &state,
                PredictiveCodingStateScope::Core,
            )
        );
        assert!(
            LanguageTrainModel::<TestBackend>::predictive_coding_state_has_latents(
                &state,
                PredictiveCodingStateScope::All,
            )
        );

        let snapshot = predictive_coding_state_snapshot(&state, PredictiveCodingStateScope::All);
        let names = snapshot
            .rank3
            .iter()
            .map(|(name, _)| *name)
            .chain(snapshot.rank4.iter().map(|(name, _)| *name))
            .collect::<HashSet<_>>();
        for required in [
            "slow_rho",
            "slow_sequence_aux",
            "slow_mamba_angle_state",
            "slow_mamba_k_state",
            "slow_mamba_v_state",
            "hierarchical_slow_hidden",
        ] {
            assert!(names.contains(required), "missing state field {required}");
        }

        assert!(
            LanguageTrainModel::<TestBackend>::attach_predictive_coding_state_latents(
                &mut state,
                PredictiveCodingStateScope::All,
            )
        );
        let layer = &state.layers[0];
        assert!(layer.slow_rho.as_ref().is_some_and(Tensor::is_require_grad));
        assert!(
            layer
                .slow_mamba_k_state
                .as_ref()
                .is_some_and(Tensor::is_require_grad)
        );
        assert!(
            layer
                .hierarchical_slow_hidden
                .as_ref()
                .is_some_and(Tensor::is_require_grad)
        );
        assert!(layer.slow_rho_norm.is_none());
    }

    #[test]
    fn predictive_coding_rotating_sampler_covers_all_slots() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let tensor = Tensor::<TestBackend, 3>::from_data(
            TensorData::new((0..10).map(|value| value as f32).collect(), [1, 1, 10]),
            &device,
        );
        let mut cache = PredictiveCodingSampleIndexCache::new();
        let mut covered = HashSet::new();

        for offset in 0..10 {
            let (student, teacher) = rotating_sample_state_axis_pair(
                tensor.clone(),
                tensor.clone(),
                2,
                3,
                offset,
                &mut cache,
            );
            let student = student
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("sampled student");
            let teacher = teacher
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("sampled teacher");
            assert_eq!(student, teacher);
            covered.extend(student.into_iter().map(|value| value as usize));
        }

        assert_eq!(covered, (0..10).collect::<HashSet<_>>());
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
        for argmax in (0usize..24).chain(std::iter::repeat_n(eos_id, 40)) {
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
                    ..Default::default()
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
                ..Default::default()
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
    fn ruliad_structured_answer_recovery_loss_trains_oracle_after_wrong_prefix() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 29);
        let mut config = tiny_model_config();
        config.vocab_size = 257;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_ruliad_supervision(RuliadSupervisionConfig {
                mode: RuliadSupervisionMode::AnswerCompletion,
                answer_denoising: RuliadAnswerDenoisingConfig {
                    enabled: true,
                    weight: 0.0,
                    structured_recovery_weight: 0.25,
                    structured_recovery_every_steps: 2,
                    structured_recovery_start_after_steps: 4,
                    structured_recovery_max_completion_tokens: 24,
                    structured_recovery_negative_count: 1,
                    structured_recovery_template_negative_count: 1,
                    ..Default::default()
                },
                ..Default::default()
            });
        let item = burn_dragon_universality::RuliadEvalItem {
            oracle_hash: "h0".to_string(),
            sample_index: 43,
            split: burn_dragon_universality::SampleSplit::Train,
            family: "trajectory_category".to_string(),
            task_kind: "eca_summary".to_string(),
            math_domains: vec!["category".to_string(), "finite_state".to_string()],
            reasoning_modes: vec!["symbolic_execution".to_string()],
            prompt: "?:eca\n!:".to_string(),
            expected_answer: "xlen=44;xalpha=01;xcounts=20,24;xedge=01".to_string(),
            difficulty_level: Some(0),
            spec: None,
        };
        let policy_batch = crate::dataset::RuliadPolicyBatch {
            samples: vec![crate::dataset::RuliadPolicySample {
                item,
                prompt_tokens: vec![1, 2, 3],
            }],
            tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
                vocab_size: 257,
                eos_id: None,
            },
            stop_token_id: None,
        };

        model.gradient_scale_step.store(3, Ordering::Relaxed);
        assert!(
            model
                .ruliad_structured_answer_recovery_loss(&policy_batch, &device, 64)
                .is_none(),
            "structured recovery should respect start_after_steps"
        );
        model.gradient_scale_step.store(5, Ordering::Relaxed);
        assert!(
            model
                .ruliad_structured_answer_recovery_loss(&policy_batch, &device, 64)
                .is_none(),
            "structured recovery should respect every_steps cadence"
        );
        model.gradient_scale_step.store(6, Ordering::Relaxed);
        let loss = model
            .ruliad_structured_answer_recovery_loss(&policy_batch, &device, 64)
            .expect("structured answer recovery loss");
        let loss = tensor_scalar(loss);
        assert!(loss.is_finite(), "recovery loss should be finite: {loss}");
        assert!(loss > 0.0, "recovery loss should be non-zero: {loss}");
    }

    #[test]
    fn ruliad_structured_answer_recovery_loss_writes_activity_telemetry() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 31);
        let dir = tempfile::tempdir().expect("tempdir");
        let telemetry_path = dir
            .path()
            .join("events")
            .join("ruliad_structured_recovery.jsonl");
        let mut config = tiny_model_config();
        config.vocab_size = 257;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_ruliad_supervision(RuliadSupervisionConfig {
                mode: RuliadSupervisionMode::AnswerCompletion,
                answer_denoising: RuliadAnswerDenoisingConfig {
                    enabled: true,
                    weight: 0.0,
                    structured_recovery_weight: 0.25,
                    structured_recovery_every_steps: 1,
                    structured_recovery_start_after_steps: 0,
                    structured_recovery_max_completion_tokens: 24,
                    structured_recovery_negative_count: 2,
                    structured_recovery_template_negative_count: 2,
                    structured_recovery_schema_negative_count: 2,
                    ..Default::default()
                },
                ..Default::default()
            })
            .with_ruliad_structured_recovery_telemetry_path(Some(telemetry_path.clone()));
        let item = burn_dragon_universality::RuliadEvalItem {
            oracle_hash: "h0".to_string(),
            sample_index: 44,
            split: burn_dragon_universality::SampleSplit::Train,
            family: "proof_tree".to_string(),
            task_kind: "prove_theorem".to_string(),
            math_domains: vec!["category".to_string(), "formal_proof".to_string()],
            reasoning_modes: vec!["equational".to_string()],
            prompt: "?:ss\n!:".to_string(),
            expected_answer: "ok=1;l=17;r=17".to_string(),
            difficulty_level: Some(0),
            spec: None,
        };
        let policy_batch = crate::dataset::RuliadPolicyBatch {
            samples: vec![crate::dataset::RuliadPolicySample {
                item,
                prompt_tokens: vec![1, 2, 3],
            }],
            tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
                vocab_size: 257,
                eos_id: None,
            },
            stop_token_id: None,
        };

        let loss = model
            .ruliad_structured_answer_recovery_loss(&policy_batch, &device, 64)
            .expect("structured answer recovery loss");
        let loss = tensor_scalar(loss);
        assert!(loss.is_finite(), "recovery loss should be finite: {loss}");

        let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
        let event: serde_json::Value =
            serde_json::from_str(content.lines().next().expect("telemetry line"))
                .expect("telemetry json");
        assert_eq!(event["sample_groups"].as_u64(), Some(1));
        assert_eq!(event["field_negative_recovery_rows"].as_u64(), Some(2));
        assert_eq!(event["template_negative_recovery_rows"].as_u64(), Some(2));
        let schema_rows = event["schema_negative_recovery_rows"]
            .as_u64()
            .expect("schema recovery rows");
        assert!(
            schema_rows > 0,
            "schema-collapse recovery rows should be present: {event}"
        );
        assert_eq!(
            event["recovery_rows"].as_u64(),
            Some(4 + schema_rows),
            "recovery rows should include field, template, and schema negatives"
        );
    }

    #[test]
    fn ruliad_answer_contract_loss_trains_full_oracle_contract_and_respects_cadence() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 41);
        let mut config = tiny_model_config();
        config.vocab_size = 257;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_ruliad_supervision(RuliadSupervisionConfig {
                mode: RuliadSupervisionMode::AnswerCompletion,
                answer_contract: crate::config::train::RuliadAnswerContractConfig {
                    enabled: true,
                    weight: 0.25,
                    premature_close_unlikelihood_weight: 0.5,
                    every_steps: 2,
                    start_after_steps: 4,
                    max_completion_tokens: 24,
                    max_rows_per_step: 1,
                    prompt_schema_max_rows_per_step: 0,
                    schema_token_weight: 2.0,
                    schema_start_token_weight: 8.0,
                    value_token_weight: 1.0,
                    other_token_weight: 1.0,
                    prompt_schema_value_weight: 0.0,
                },
                ..Default::default()
            });
        let item = burn_dragon_universality::RuliadEvalItem {
            oracle_hash: "h0".to_string(),
            sample_index: 46,
            split: burn_dragon_universality::SampleSplit::Train,
            family: "proof_tree".to_string(),
            task_kind: "prove_theorem".to_string(),
            math_domains: vec!["category".to_string(), "formal_proof".to_string()],
            reasoning_modes: vec!["equational".to_string()],
            prompt: "?:ss\n!:".to_string(),
            expected_answer: "ok=1;l=17;r=17".to_string(),
            difficulty_level: Some(0),
            spec: None,
        };
        let policy_batch = crate::dataset::RuliadPolicyBatch {
            samples: vec![crate::dataset::RuliadPolicySample {
                item,
                prompt_tokens: vec![1, 2, 3],
            }],
            tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
                vocab_size: 257,
                eos_id: None,
            },
            stop_token_id: None,
        };

        model.gradient_scale_step.store(3, Ordering::Relaxed);
        assert!(
            model
                .ruliad_answer_contract_loss(&policy_batch, &device, 64)
                .is_none(),
            "answer contract loss should respect start_after_steps"
        );
        model.gradient_scale_step.store(5, Ordering::Relaxed);
        assert!(
            model
                .ruliad_answer_contract_loss(&policy_batch, &device, 64)
                .is_none(),
            "answer contract loss should respect every_steps cadence"
        );
        model.gradient_scale_step.store(6, Ordering::Relaxed);
        let loss = model
            .ruliad_answer_contract_loss(&policy_batch, &device, 64)
            .expect("answer contract loss");
        let loss = tensor_scalar(loss);
        assert!(loss.is_finite(), "contract loss should be finite: {loss}");
        assert!(loss > 0.0, "contract loss should be non-zero: {loss}");
    }

    #[test]
    fn ruliad_answer_contract_loss_writes_activity_telemetry_and_caps_rows() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 43);
        let dir = tempfile::tempdir().expect("tempdir");
        let telemetry_path = dir
            .path()
            .join("events")
            .join("ruliad_answer_contract.jsonl");
        let mut config = tiny_model_config();
        config.vocab_size = 272;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_ruliad_supervision(RuliadSupervisionConfig {
                mode: RuliadSupervisionMode::AnswerCompletion,
                answer_contract: crate::config::train::RuliadAnswerContractConfig {
                    enabled: true,
                    weight: 0.25,
                    premature_close_unlikelihood_weight: 0.5,
                    every_steps: 1,
                    start_after_steps: 0,
                    max_completion_tokens: 24,
                    max_rows_per_step: 1,
                    prompt_schema_max_rows_per_step: 1,
                    schema_token_weight: 2.0,
                    schema_start_token_weight: 8.0,
                    value_token_weight: 1.0,
                    other_token_weight: 1.0,
                    prompt_schema_value_weight: 2.0,
                },
                ..Default::default()
            })
            .with_ruliad_answer_contract_telemetry_path(Some(telemetry_path.clone()));
        let make_item = |sample_index, answer: &str| burn_dragon_universality::RuliadEvalItem {
            oracle_hash: format!("h{sample_index}"),
            sample_index,
            split: burn_dragon_universality::SampleSplit::Train,
            family: "proof_tree".to_string(),
            task_kind: "prove_theorem".to_string(),
            math_domains: vec!["category".to_string(), "formal_proof".to_string()],
            reasoning_modes: vec!["equational".to_string()],
            prompt: "?:ss\n!:".to_string(),
            expected_answer: answer.to_string(),
            difficulty_level: Some(0),
            spec: None,
        };
        let policy_batch = crate::dataset::RuliadPolicyBatch {
            samples: vec![
                crate::dataset::RuliadPolicySample {
                    item: make_item(47, "ok=1;l=17;r=17"),
                    prompt_tokens: vec![1, 2, 3],
                },
                crate::dataset::RuliadPolicySample {
                    item: make_item(48, "nflen=3;nfalpha=ABC;nfcounts=1,1,1;nfedge=AB"),
                    prompt_tokens: vec![1, 2, 3],
                },
            ],
            tokenization: burn_dragon_universality::RuliadTokenizationConfig::StructuredSymbolic {
                vocab_size: 272,
                eos_id: Some(271),
            },
            stop_token_id: Some(265),
        };

        let loss = model
            .ruliad_answer_contract_loss(&policy_batch, &device, 64)
            .expect("answer contract loss");
        let loss = tensor_scalar(loss);
        assert!(loss.is_finite(), "contract loss should be finite: {loss}");

        let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
        let event: serde_json::Value =
            serde_json::from_str(content.lines().next().expect("telemetry line"))
                .expect("contract telemetry json");
        assert_eq!(event["policy_batch_present"].as_bool(), Some(true));
        assert_eq!(event["oracle_rows"].as_u64(), Some(1));
        assert!(
            event["sample_groups"].as_u64().unwrap_or_default() >= 1,
            "contract objective should report active sample groups: {event}"
        );
        assert!(
            event["prompt_schema_sample_groups"]
                .as_u64()
                .unwrap_or_default()
                >= 1,
            "contract objective should report active prompt-schema sample groups: {event}"
        );
        assert_eq!(event["prompt_schema_rows"].as_u64(), Some(1));
        assert_eq!(event["prompt_schema_max_rows_per_step"].as_u64(), Some(1));
        assert!(
            event["schema_tokens"].as_u64().unwrap_or_default() > 0,
            "contract objective should supervise schema tokens: {event}"
        );
        assert!(
            event["schema_start_tokens"].as_u64().unwrap_or_default() > 0,
            "contract objective should identify schema-start tokens: {event}"
        );
        assert!(
            event["value_tokens"].as_u64().unwrap_or_default() > 0,
            "contract objective should supervise value tokens: {event}"
        );
        assert!(
            event["prompt_schema_value_tokens"]
                .as_u64()
                .unwrap_or_default()
                > 0,
            "contract objective should supervise schema-forced value tokens: {event}"
        );
        assert!(
            event["premature_close_tokens"].as_u64().unwrap_or_default() > 0,
            "contract objective should penalize premature close markers: {event}"
        );
    }

    #[test]
    fn ruliad_verifier_policy_loss_builds_from_policy_metadata() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_ruliad_supervision(RuliadSupervisionConfig {
            verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                enabled: true,
                weight: 0.1,
                group_size: 2,
                max_completion_tokens: 2,
                every_steps: 1,
                top_k: 1,
                ..Default::default()
            },
            ..Default::default()
        });
        let item = burn_dragon_universality::RuliadEvalItem {
            oracle_hash: "h0".to_string(),
            sample_index: 0,
            split: burn_dragon_universality::SampleSplit::Train,
            family: "law".to_string(),
            task_kind: "category_law".to_string(),
            math_domains: vec!["category".to_string()],
            reasoning_modes: vec!["equational".to_string()],
            prompt: "?:q\n!:".to_string(),
            expected_answer: "ok=1".to_string(),
            difficulty_level: Some(0),
            spec: None,
        };
        let policy_batch = crate::dataset::RuliadPolicyBatch {
            samples: vec![crate::dataset::RuliadPolicySample {
                item,
                prompt_tokens: vec![1, 2, 3],
            }],
            tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
                vocab_size: 257,
                eos_id: None,
            },
            stop_token_id: None,
        };
        let loss = model
            .ruliad_verifier_policy_loss(&policy_batch, &device, 8)
            .expect("verifier policy loss");
        let loss = tensor_scalar(loss);
        assert!(
            loss.is_finite(),
            "verifier policy loss should be finite: {loss}"
        );
    }

    #[test]
    fn ruliad_verifier_policy_loss_respects_start_after_steps() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_ruliad_supervision(RuliadSupervisionConfig {
            verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                enabled: true,
                weight: 0.1,
                every_steps: 1,
                start_after_steps: 4,
                ..Default::default()
            },
            ..Default::default()
        });
        model.gradient_scale_step.store(3, Ordering::Relaxed);
        assert_eq!(model.ruliad_verifier_reward_weight(), 0.0);
        model.gradient_scale_step.store(4, Ordering::Relaxed);
        assert_eq!(model.ruliad_verifier_reward_weight(), 0.1);
    }

    #[test]
    fn ruliad_policy_telemetry_marks_saturated_updates_as_skipped() {
        let mut telemetry = RuliadPolicyRewardTelemetryAccumulator::new(
            crate::config::train::RuliadVerifierRewardMode::VpoIndependent,
            64,
        );
        telemetry.record_rewards_and_advantages(&[0.0, 1.0, -1.0], &[0.0, 1.0, -1.0], 0.2);
        assert!(
            telemetry.advantage_clip_fraction() > 0.5,
            "test should exercise a saturated policy update"
        );
        telemetry.mark_skipped("advantage_clip_fraction>0.500000");
        let telemetry = telemetry.finish().expect("telemetry");
        assert!(!telemetry.policy_update_applied);
        assert_eq!(
            telemetry.policy_skip_reason.as_deref(),
            Some("advantage_clip_fraction>0.500000")
        );
    }

    #[test]
    fn ruliad_policy_telemetry_reports_gated_groups_without_rows() {
        let mut telemetry = RuliadPolicyRewardTelemetryAccumulator::new(
            crate::config::train::RuliadVerifierRewardMode::VpoIndependent,
            64,
        );
        telemetry.record_gated_group(4);
        telemetry.mark_skipped("positive_advantage_gate");
        let telemetry = telemetry.finish().expect("telemetry");
        assert_eq!(telemetry.completion_rows, 0);
        assert_eq!(telemetry.gated_sample_groups, 1);
        assert_eq!(telemetry.gated_completion_rows, 4);
        assert!(!telemetry.policy_update_applied);
        assert_eq!(
            telemetry.policy_skip_reason.as_deref(),
            Some("positive_advantage_gate")
        );
    }

    fn strict_policy_advantage_gate_config() -> crate::config::train::RuliadVerifierRewardConfig {
        crate::config::train::RuliadVerifierRewardConfig {
            positive_advantage_requires_correctness: true,
            positive_advantage_min_partial_progress_ppm: 500_000,
            positive_advantage_min_completion_quality_ppm: 750_000,
            ..Default::default()
        }
    }

    fn policy_score(
        status: burn_dragon_universality::ruliad::RuliadAnswerStatus,
        partial_progress_ppm: usize,
        completion_quality_ppm: usize,
    ) -> burn_dragon_universality::ruliad::RuliadReasoningScore {
        let expected_field_count = 4;
        let correct_field_count = partial_progress_ppm
            .saturating_mul(expected_field_count)
            .div_ceil(1_000_000)
            .min(expected_field_count);
        burn_dragon_universality::ruliad::RuliadReasoningScore {
            version: burn_dragon_universality::ruliad::RULIAD_REASONING_SCORE_VERSION,
            status,
            correct_field_count,
            expected_field_count,
            observed_field_count: correct_field_count,
            partial_progress_ppm,
            certificate_valid_prefix_steps: 0,
            certificate_expected_steps: 0,
            certificate_prefix_ppm: 0,
            generated_token_count: if partial_progress_ppm > 0 { 8 } else { 1 },
            hash_canary: false,
            answer_terminated: status
                != burn_dragon_universality::ruliad::RuliadAnswerStatus::Malformed,
            completion_quality_ppm,
        }
    }

    #[test]
    fn ruliad_rollout_recovery_signal_accepts_wrong_and_malformed_corruptions() {
        let partial = policy_score(
            burn_dragon_universality::ruliad::RuliadAnswerStatus::Partial,
            500_000,
            1_000_000,
        );
        assert!(
            LanguageTrainModel::<TestBackend>::ruliad_score_has_rollout_recovery_signal(
                &partial, 500_000, 750_000,
            )
        );

        let mut schema_wrong = policy_score(
            burn_dragon_universality::ruliad::RuliadAnswerStatus::SchemaValidWrong,
            0,
            1_000_000,
        );
        schema_wrong.observed_field_count = 1;
        assert!(
            LanguageTrainModel::<TestBackend>::ruliad_score_has_rollout_recovery_signal(
                &schema_wrong,
                500_000,
                750_000,
            )
        );

        let malformed = policy_score(
            burn_dragon_universality::ruliad::RuliadAnswerStatus::Malformed,
            0,
            1_000_000,
        );
        assert!(
            LanguageTrainModel::<TestBackend>::ruliad_score_has_rollout_recovery_signal(
                &malformed, 0, 0,
            )
        );
        assert!(
            !LanguageTrainModel::<TestBackend>::ruliad_score_has_rollout_recovery_signal(
                &malformed, 0, 1_000_001,
            )
        );
    }

    #[test]
    fn ruliad_policy_advantage_guard_blocks_positive_wrong_schema() {
        let config = strict_policy_advantage_gate_config();
        let scores = vec![
            policy_score(
                burn_dragon_universality::ruliad::RuliadAnswerStatus::Partial,
                500_000,
                1_000_000,
            ),
            policy_score(
                burn_dragon_universality::ruliad::RuliadAnswerStatus::SchemaValidWrong,
                0,
                1_000_000,
            ),
        ];
        let mut advantages = [-0.4, 0.9];
        assert!(
            LanguageTrainModel::<TestBackend>::constrain_ruliad_policy_advantages(
                &scores,
                &mut advantages,
                config,
            )
        );
        assert_eq!(advantages[0], -0.4);
        assert_eq!(advantages[1], 0.0);
    }

    #[test]
    fn ruliad_policy_advantage_guard_skips_all_wrong_groups() {
        let config = strict_policy_advantage_gate_config();
        let scores = vec![
            policy_score(
                burn_dragon_universality::ruliad::RuliadAnswerStatus::SchemaValidWrong,
                0,
                1_000_000,
            ),
            policy_score(
                burn_dragon_universality::ruliad::RuliadAnswerStatus::Malformed,
                0,
                1_000_000,
            ),
        ];
        let mut advantages = [0.9, -0.9];
        assert!(
            !LanguageTrainModel::<TestBackend>::constrain_ruliad_policy_advantages(
                &scores,
                &mut advantages,
                config,
            )
        );
    }

    #[test]
    fn ruliad_policy_advantage_guard_skips_weak_partial_groups() {
        let config = strict_policy_advantage_gate_config();
        let scores = vec![
            policy_score(
                burn_dragon_universality::ruliad::RuliadAnswerStatus::Partial,
                250_000,
                1_000_000,
            ),
            policy_score(
                burn_dragon_universality::ruliad::RuliadAnswerStatus::SchemaValidWrong,
                0,
                1_000_000,
            ),
        ];
        let mut advantages = [0.9, -0.9];
        assert!(
            !LanguageTrainModel::<TestBackend>::constrain_ruliad_policy_advantages(
                &scores,
                &mut advantages,
                config,
            )
        );
    }

    #[test]
    fn ruliad_policy_advantage_guard_skips_low_quality_partials() {
        let config = strict_policy_advantage_gate_config();
        let scores = vec![policy_score(
            burn_dragon_universality::ruliad::RuliadAnswerStatus::Partial,
            500_000,
            250_000,
        )];
        let mut advantages = [0.9];
        assert!(
            !LanguageTrainModel::<TestBackend>::constrain_ruliad_policy_advantages(
                &scores,
                &mut advantages,
                config,
            )
        );
    }

    #[test]
    fn ruliad_verifier_policy_loss_supports_vpo_mode() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 11);
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_ruliad_supervision(RuliadSupervisionConfig {
            verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                enabled: true,
                mode: crate::config::train::RuliadVerifierRewardMode::VpoIndependent,
                weight: 0.1,
                group_size: 2,
                max_completion_tokens: 2,
                every_steps: 1,
                top_k: 1,
                kl_weight: 0.0,
                vpo_scalarizations: 4,
                ..Default::default()
            },
            ..Default::default()
        });
        let vpo_config = crate::config::train::RuliadVerifierRewardConfig {
            enabled: true,
            mode: crate::config::train::RuliadVerifierRewardMode::VpoIndependent,
            vpo_correctness_mass_floor: 0.70,
            vpo_schema_quality_mass_floor: 0.10,
            vpo_completion_health_mass_floor: 0.10,
            vpo_compactness_max_weight: 0.05,
            ..Default::default()
        };
        let scalarizations = model.ruliad_vpo_scalarizations(17, 4, vpo_config);
        assert_eq!(scalarizations.len(), 4);
        for scalarization in scalarizations {
            assert!(
                scalarization.iter().all(|weight| *weight >= 0.0),
                "VPO scalarization weights should be non-negative"
            );
            let sum = scalarization.iter().sum::<f32>();
            assert!(
                (sum - 1.0).abs() < 1.0e-5,
                "VPO scalarization should sum to one, got {sum}"
            );
            let correctness_mass = scalarization[0..=4].iter().sum::<f32>();
            let schema_mass = scalarization[6];
            let health_mass = scalarization[8..=9].iter().sum::<f32>();
            assert!(
                correctness_mass >= 0.70 - 1.0e-5,
                "correctness mass floor should hold, got {correctness_mass}"
            );
            assert!(
                schema_mass >= 0.10 - 1.0e-5,
                "schema-quality mass floor should hold, got {schema_mass}"
            );
            assert!(
                health_mass >= 0.10 - 1.0e-5,
                "health mass floor should hold, got {health_mass}"
            );
            assert!(
                scalarization[5] <= 0.05 + 1.0e-5,
                "compactness weight should be capped"
            );
        }
        let item = burn_dragon_universality::RuliadEvalItem {
            oracle_hash: "h0".to_string(),
            sample_index: 17,
            split: burn_dragon_universality::SampleSplit::Train,
            family: "law".to_string(),
            task_kind: "category_law".to_string(),
            math_domains: vec!["category".to_string()],
            reasoning_modes: vec!["equational".to_string()],
            prompt: "?:q\n!:".to_string(),
            expected_answer: "ok=1".to_string(),
            difficulty_level: Some(0),
            spec: None,
        };
        let policy_batch = crate::dataset::RuliadPolicyBatch {
            samples: vec![crate::dataset::RuliadPolicySample {
                item,
                prompt_tokens: vec![1, 2, 3],
            }],
            tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
                vocab_size: 257,
                eos_id: None,
            },
            stop_token_id: None,
        };
        let loss = model
            .ruliad_verifier_policy_loss(&policy_batch, &device, 8)
            .expect("VPO verifier policy loss");
        let loss = tensor_scalar(loss);
        assert!(
            loss.is_finite(),
            "VPO verifier policy loss should be finite: {loss}"
        );
    }

    #[test]
    fn ruliad_verifier_policy_loss_can_include_oracle_candidate() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 17);
        let dir = tempfile::tempdir().expect("tempdir");
        let telemetry_path = dir
            .path()
            .join("events")
            .join("ruliad_verifier_policy.jsonl");
        let mut config = tiny_model_config();
        config.vocab_size = 257;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_ruliad_supervision(RuliadSupervisionConfig {
                verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                    enabled: true,
                    mode: crate::config::train::RuliadVerifierRewardMode::VpoIndependent,
                    weight: 0.1,
                    group_size: 2,
                    max_completion_tokens: 16,
                    every_steps: 1,
                    top_k: 1,
                    kl_weight: 0.0,
                    vpo_scalarizations: 4,
                    positive_advantage_requires_correctness: true,
                    positive_advantage_min_partial_progress_ppm: 500_000,
                    positive_advantage_min_completion_quality_ppm: 750_000,
                    include_oracle_candidate: true,
                    ..Default::default()
                },
                ..Default::default()
            })
            .with_ruliad_policy_telemetry_path(Some(telemetry_path.clone()));
        let item = burn_dragon_universality::RuliadEvalItem {
            oracle_hash: "h0".to_string(),
            sample_index: 29,
            split: burn_dragon_universality::SampleSplit::Train,
            family: "law".to_string(),
            task_kind: "category_law".to_string(),
            math_domains: vec!["category".to_string()],
            reasoning_modes: vec!["equational".to_string()],
            prompt: "?:q\n!:".to_string(),
            expected_answer: "ok=1".to_string(),
            difficulty_level: Some(0),
            spec: None,
        };
        let policy_batch = crate::dataset::RuliadPolicyBatch {
            samples: vec![crate::dataset::RuliadPolicySample {
                item,
                prompt_tokens: vec![1, 2, 3],
            }],
            tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
                vocab_size: 257,
                eos_id: None,
            },
            stop_token_id: None,
        };
        let loss = model
            .ruliad_verifier_policy_loss(&policy_batch, &device, 32)
            .expect("oracle VPO verifier policy loss");
        assert!(tensor_scalar(loss).is_finite());
        let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
        let value: serde_json::Value =
            serde_json::from_str(content.lines().next().expect("telemetry line"))
                .expect("telemetry json");
        assert_eq!(value["oracle_sample_groups"], 1);
        assert_eq!(value["oracle_completion_rows"], 1);
        assert_eq!(value["oracle_truncated_completion_rows"], 0);
        assert_eq!(value["policy_update_applied"], true);
        assert!(
            value["vector_semantic_match_mean"]
                .as_f64()
                .expect("semantic mean")
                > 0.0,
            "oracle candidate should provide a correctness-positive row"
        );
    }

    #[test]
    fn ruliad_rollout_recovery_accepts_malformed_and_missing_corruptions() {
        use burn_dragon_universality::ruliad::RuliadAnswerStatus;

        let min_partial = 500_000;
        let min_quality = 750_000;
        let accepts = |status, partial, quality| {
            LanguageTrainModel::<TestBackend>::ruliad_score_has_rollout_recovery_signal(
                &ruliad_test_score(status, partial, quality),
                min_partial,
                min_quality,
            )
        };

        assert!(accepts(
            RuliadAnswerStatus::SchemaValidWrong,
            0,
            min_quality
        ));
        assert!(accepts(RuliadAnswerStatus::Malformed, 0, min_quality));
        assert!(accepts(RuliadAnswerStatus::Missing, 0, min_quality));
        assert!(accepts(
            RuliadAnswerStatus::Partial,
            min_partial,
            min_quality
        ));
        assert!(!accepts(
            RuliadAnswerStatus::Partial,
            min_partial.saturating_sub(1),
            min_quality
        ));
        assert!(!accepts(
            RuliadAnswerStatus::Malformed,
            0,
            min_quality.saturating_sub(1)
        ));
        assert!(!accepts(
            RuliadAnswerStatus::VerifierMatch,
            1_000_000,
            min_quality
        ));
        assert!(!accepts(
            RuliadAnswerStatus::SemanticMatch,
            1_000_000,
            min_quality
        ));
    }

    #[test]
    fn ruliad_structured_negative_answers_mutate_prompt_bound_fields() {
        let negatives = LanguageTrainModel::<TestBackend>::ruliad_structured_negative_answers(
            "ok=1;l=17;r=17",
            3,
        );

        assert_eq!(negatives.len(), 3);
        assert!(negatives.iter().all(|answer| answer != "ok=1;l=17;r=17"));
        assert!(
            negatives.iter().any(|answer| answer.starts_with("ok=0")),
            "boolean fields should be mutated into plausible wrong answers: {negatives:?}"
        );
        assert!(
            negatives
                .iter()
                .any(|answer| answer.contains("l=19") || answer.contains("l=18")),
            "numeric fields should be mutated without destroying the answer schema: {negatives:?}"
        );
    }

    #[test]
    fn ruliad_structured_negative_answers_include_template_collapse_hard_negatives() {
        let proof_negatives =
            LanguageTrainModel::<TestBackend>::ruliad_structured_negative_answers_with_templates(
                "ok=1;l=17;r=17",
                1,
                2,
            );
        let proof_texts = proof_negatives
            .iter()
            .map(|(answer, _kind)| answer.as_str())
            .collect::<Vec<_>>();
        assert!(
            proof_texts.contains(&"ok=1;l=5;r=5"),
            "proof hard negatives should target the observed l=5/r=5 attractor: {proof_texts:?}"
        );
        assert!(
            proof_texts.contains(&"ok=1;l=1;r=1"),
            "proof hard negatives should target the observed l/r collapse: {proof_texts:?}"
        );
        assert!(
            proof_negatives
                .iter()
                .any(|(_answer, kind)| *kind == RuliadStructuredNegativeKind::TemplateCollapse),
            "template rows should be tracked separately"
        );

        let automaton_negatives =
            LanguageTrainModel::<TestBackend>::ruliad_structured_negative_answers_with_templates(
                "acc=1", 0, 2,
            )
            .into_iter()
            .map(|(answer, _kind)| answer)
            .collect::<Vec<_>>();
        assert_eq!(automaton_negatives, vec!["acc=0".to_string()]);

        let eca_negatives =
            LanguageTrainModel::<TestBackend>::ruliad_structured_negative_answers_with_templates(
                "xlen=44;xalpha=01;xcounts=20,24;xedge=01",
                0,
                3,
            )
            .into_iter()
            .map(|(answer, _kind)| answer)
            .collect::<Vec<_>>();
        assert!(
            eca_negatives
                .iter()
                .any(|answer| answer == "xlen=13;xalpha=abc;nfcounts=1,1,0;nfedge=ba"),
            "ECA hard negatives should include the observed mixed x/nf lowercase attractor: {eca_negatives:?}"
        );

        let normal_form_negatives =
            LanguageTrainModel::<TestBackend>::ruliad_structured_negative_answers_with_templates(
                "nflen=44;nfalpha=01;nfcounts=20,24;nfedge=01",
                0,
                3,
            )
            .into_iter()
            .map(|(answer, _kind)| answer)
            .collect::<Vec<_>>();
        assert!(
            normal_form_negatives
                .iter()
                .all(|answer| answer.contains("nfedge=") && !answer.contains("xedge=")),
            "normal-form hard negatives should preserve the nfedge schema: {normal_form_negatives:?}"
        );
        assert!(
            normal_form_negatives
                .iter()
                .any(|answer| answer == "nflen=5;nfalpha=abc;nfcounts=1,1,0;nfedge=ba"),
            "normal-form hard negatives should include the observed lowercase normal-form attractor: {normal_form_negatives:?}"
        );
    }

    #[test]
    fn ruliad_structured_negative_answers_with_schema_include_contract_sibling_negatives() {
        let negatives =
            LanguageTrainModel::<TestBackend>::ruliad_structured_negative_answers_with_schema(
                "xlen=44;xalpha=01;xcounts=20,24;xedge=01",
                0,
                0,
                8,
            );
        assert_eq!(negatives.len(), 8);
        assert!(
            negatives
                .iter()
                .all(|(_answer, kind)| *kind == RuliadStructuredNegativeKind::SchemaCollapse),
            "schema rows should be tracked separately: {negatives:?}"
        );
        let texts = negatives
            .iter()
            .map(|(answer, _kind)| answer.as_str())
            .collect::<Vec<_>>();
        assert!(
            texts.iter().any(|answer| answer.contains("nfalpha=")),
            "schema-collapse negatives should expose sibling normal-form keys: {texts:?}"
        );
        assert!(
            texts.contains(&"xlen=44;xalpha=01;xcounts=20,24"),
            "schema-collapse negatives should include missing-tail-field answers: {texts:?}"
        );
        assert!(
            texts.contains(&"xlen=44"),
            "schema-collapse negatives should include first-field-only answer collapse: {texts:?}"
        );
        assert!(
            texts.contains(&"ok=1;l=1;r=1"),
            "schema-collapse negatives should include the observed ok/l/r cross-contract prototype: {texts:?}"
        );
        assert!(
            texts.contains(&"acc=1"),
            "schema-collapse negatives should include compact cross-contract prototypes: {texts:?}"
        );
        assert!(
            texts
                .iter()
                .all(|answer| *answer != "xlen=44;xalpha=01;xcounts=20,24;xedge=01"),
            "schema-collapse negatives must not duplicate the oracle answer: {texts:?}"
        );
    }

    #[test]
    fn ruliad_structured_proof_step_negatives_preserve_the_wire_contract() {
        let answer = "g4|a:r0|f|1.1";
        let negatives =
            LanguageTrainModel::<TestBackend>::ruliad_structured_negative_answers_with_templates(
                answer, 4, 1,
            );

        assert_eq!(negatives.len(), 5, "{negatives:?}");
        assert_eq!(
            negatives
                .iter()
                .filter(|(_, kind)| *kind == RuliadStructuredNegativeKind::TemplateCollapse)
                .count(),
            1
        );
        assert_eq!(
            negatives
                .iter()
                .filter(|(_, kind)| *kind == RuliadStructuredNegativeKind::FieldMutation)
                .count(),
            4
        );
        assert!(negatives.iter().all(|(candidate, _)| {
            candidate != answer
                && burn_dragon_universality::ruliad::wire::decode_model_proof_step(candidate)
                    .is_some()
        }));
        let oracle_fields = answer.split('|').collect::<Vec<_>>();
        for (field_index, oracle_field) in oracle_fields.iter().enumerate() {
            assert!(negatives.iter().any(|(candidate, kind)| {
                *kind == RuliadStructuredNegativeKind::FieldMutation
                    && candidate
                        .split('|')
                        .nth(field_index)
                        .is_some_and(|field| field != *oracle_field)
            }));
        }
    }

    #[test]
    fn ruliad_answer_value_completion_mask_marks_only_answer_values() {
        let tokenizer =
            burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer::from_config(
                &burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
                    vocab_size: 257,
                    eos_id: None,
                },
            )
            .expect("tokenizer");
        let answer = "ok=1;l=17;r=17";
        let completion = tokenizer.encode_payload(&format!("{answer}\n[/R2]"));
        let mask = LanguageTrainModel::<TestBackend>::ruliad_answer_value_completion_mask(
            &tokenizer,
            answer,
            completion.len(),
        );
        let marked = completion
            .iter()
            .zip(mask.iter())
            .filter_map(|(token, active)| active.then_some(*token))
            .filter_map(char::from_u32)
            .collect::<String>();

        assert_eq!(marked, "11717");

        let answer = "g4|a:r0|f|1.1";
        let completion = tokenizer.encode_payload(&format!("{answer}\n[/R3]"));
        let mask = LanguageTrainModel::<TestBackend>::ruliad_answer_value_completion_mask(
            &tokenizer,
            answer,
            completion.len(),
        );
        let marked = completion
            .iter()
            .zip(mask.iter())
            .filter_map(|(token, active)| active.then_some(*token))
            .filter_map(char::from_u32)
            .collect::<String>();

        assert_eq!(marked, "4r0f1.1");
        assert_eq!(
            LanguageTrainModel::<TestBackend>::ruliad_answer_contract(answer).as_deref(),
            Some("proof_action_step")
        );
    }

    #[test]
    fn ruliad_answer_key_completion_mask_marks_only_answer_keys() {
        let tokenizer =
            burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer::from_config(
                &burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
                    vocab_size: 257,
                    eos_id: None,
                },
            )
            .expect("tokenizer");
        let answer = "ok=1;l=17;r=17";
        let completion = tokenizer.encode_payload(&format!("{answer}\n[/R2]"));
        let mask = LanguageTrainModel::<TestBackend>::ruliad_answer_key_completion_mask(
            &tokenizer,
            answer,
            completion.len(),
        );
        let marked = completion
            .iter()
            .zip(mask.iter())
            .filter_map(|(token, active)| active.then_some(*token))
            .filter_map(char::from_u32)
            .collect::<String>();

        assert_eq!(marked, "oklr");
    }

    #[test]
    fn ruliad_answer_schema_completion_mask_marks_keys_and_field_separators() {
        let tokenizer =
            burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer::from_config(
                &burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
                    vocab_size: 257,
                    eos_id: None,
                },
            )
            .expect("tokenizer");
        let answer = "ok=1;l=17;r=17";
        let completion = tokenizer.encode_payload(&format!("{answer}\n[/R2]"));
        let mask = LanguageTrainModel::<TestBackend>::ruliad_answer_schema_completion_mask(
            &tokenizer,
            answer,
            completion.len(),
        );
        let marked = completion
            .iter()
            .zip(mask.iter())
            .filter_map(|(token, active)| active.then_some(*token))
            .filter_map(char::from_u32)
            .collect::<String>();

        assert_eq!(marked, "ok=;l=;r=");
    }

    #[test]
    fn ruliad_answer_schema_start_completion_mask_marks_first_key_bytes() {
        let tokenizer =
            burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer::from_config(
                &burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
                    vocab_size: 257,
                    eos_id: None,
                },
            )
            .expect("tokenizer");
        let answer = "xlen=14;xalpha=01;xcounts=8,6;xedge=01";
        let completion = tokenizer.encode_payload(&format!("{answer}\n[/R2]"));
        let mask = LanguageTrainModel::<TestBackend>::ruliad_answer_schema_start_completion_mask(
            &tokenizer,
            answer,
            completion.len(),
        );
        let marked = completion
            .iter()
            .zip(mask.iter())
            .filter_map(|(token, active)| active.then_some(*token))
            .filter_map(char::from_u32)
            .collect::<String>();

        assert_eq!(marked, "xxxx");
    }

    #[test]
    fn ruliad_prompt_schema_value_rows_train_values_under_supplied_keys() {
        let tokenizer =
            burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer::from_config(
                &burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
                    vocab_size: 257,
                    eos_id: None,
                },
            )
            .expect("tokenizer");
        let prompt = tokenizer
            .encode_payload("?:prove\n!:")
            .into_iter()
            .map(i64::from)
            .collect::<Vec<_>>();

        let rows = LanguageTrainModel::<TestBackend>::ruliad_prompt_schema_value_completion_rows(
            &tokenizer,
            &prompt,
            "ok=1;l=17;r=17",
            burn_dragon_universality::ruliad::RULIAD_V2_DOCUMENT_CLOSE_MARKER,
            32,
            96,
            8,
        );

        assert_eq!(rows.len(), 3);
        let decoded_targets = rows
            .iter()
            .map(|(_inputs, targets, mask, active)| {
                assert_eq!(*active, mask.iter().filter(|value| **value > 0.0).count());
                let tokens = targets
                    .iter()
                    .zip(mask.iter())
                    .filter_map(|(token, active)| (*active > 0.0).then_some(*token as u32))
                    .collect::<Vec<_>>();
                tokenizer.decode_payload(&tokens, true)
            })
            .collect::<Vec<_>>();

        assert_eq!(
            decoded_targets,
            vec!["1;", "17;", "17\n[/R2]"],
            "schema-forced value rows should target field values and close markers"
        );
    }

    #[test]
    fn ruliad_prompt_schema_value_rows_train_semantic_proof_step_fields() {
        let tokenizer =
            burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer::from_config(
                &burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
                    vocab_size: 257,
                    eos_id: None,
                },
            )
            .expect("tokenizer");
        let prompt = tokenizer
            .encode_payload("?:select;g=3;dst=x;at=1.1\n!:")
            .into_iter()
            .map(i64::from)
            .collect::<Vec<_>>();

        let rows = LanguageTrainModel::<TestBackend>::ruliad_prompt_schema_value_completion_rows(
            &tokenizer,
            &prompt,
            "g3|a:r0|f|1.1",
            burn_dragon_universality::ruliad::RULIAD_V2_DOCUMENT_CLOSE_MARKER,
            32,
            96,
            8,
        );

        assert_eq!(rows.len(), 4);
        let decoded_targets = rows
            .iter()
            .map(|(_inputs, targets, mask, active)| {
                assert_eq!(*active, mask.iter().filter(|value| **value > 0.0).count());
                let tokens = targets
                    .iter()
                    .zip(mask.iter())
                    .filter_map(|(token, active)| (*active > 0.0).then_some(*token as u32))
                    .collect::<Vec<_>>();
                tokenizer.decode_payload(&tokens, true)
            })
            .collect::<Vec<_>>();

        assert_eq!(decoded_targets, vec!["3|", "r0|", "f|", "1.1\n[/R2]"]);
    }

    #[test]
    fn prompt_schema_row_budget_is_spread_across_samples_first() {
        let groups = vec![vec!["a0", "a1"], vec!["b0", "b1"], vec!["c0", "c1"]];

        assert_eq!(
            take_rows_round_robin(&groups, 4),
            vec![(0, "a0"), (1, "b0"), (2, "c0"), (0, "a1")]
        );
    }

    #[test]
    fn ruliad_schema_collapse_negative_answers_cover_sibling_contracts() {
        let eca_negatives =
            LanguageTrainModel::<TestBackend>::ruliad_schema_collapse_negative_answers(
                "xlen=14;xalpha=01;xcounts=8,6;xedge=01",
            );
        assert!(
            eca_negatives
                .iter()
                .any(|answer| answer == "xlen=14;xalpha=01;xcounts=8,6"),
            "ECA schema negatives should include tail-field omission: {eca_negatives:?}"
        );
        assert!(
            eca_negatives
                .iter()
                .any(|answer| answer == "xlen=14;nfalpha=01;nfcounts=8,6;xedge=01"),
            "ECA schema negatives should include the observed x/nf mixed-key collapse: {eca_negatives:?}"
        );
        assert!(
            eca_negatives
                .iter()
                .any(|answer| answer == "nflen=14;nfalpha=01;nfcounts=8,6;nfedge=01"),
            "ECA schema negatives should include the full sibling rewrite contract: {eca_negatives:?}"
        );

        let proof_negatives =
            LanguageTrainModel::<TestBackend>::ruliad_schema_collapse_negative_answers(
                "ok=1;l=17;r=17",
            );
        assert_eq!(
            &proof_negatives[..2],
            ["ok=1;l=17".to_string(), "ok=1".to_string()],
            "proof-specific truncation negatives should remain first"
        );
        assert!(
            proof_negatives
                .iter()
                .any(|answer| answer.starts_with("xlen=")),
            "proof negatives should include the ECA sibling contract: {proof_negatives:?}"
        );
        assert!(
            proof_negatives
                .iter()
                .any(|answer| answer.starts_with("nflen=")),
            "proof negatives should include the normal-form sibling contract: {proof_negatives:?}"
        );
        assert!(
            proof_negatives.iter().any(|answer| answer == "acc=0"),
            "proof negatives should include the acceptance sibling contract: {proof_negatives:?}"
        );
    }

    #[test]
    fn ruliad_trim_prompt_for_completion_preserves_maximum_context() {
        let prompt = vec![10, 11, 12, 13, 14, 15, 16, 17];
        let trimmed =
            LanguageTrainModel::<TestBackend>::ruliad_trim_prompt_for_completion(&prompt, 3, 7);
        assert_eq!(trimmed, vec![14, 15, 16, 17]);

        let untrimmed =
            LanguageTrainModel::<TestBackend>::ruliad_trim_prompt_for_completion(&prompt, 2, 16);
        assert_eq!(untrimmed, prompt);

        let overlong_completion =
            LanguageTrainModel::<TestBackend>::ruliad_trim_prompt_for_completion(&prompt, 99, 7);
        assert_eq!(overlong_completion, vec![17]);
    }

    #[test]
    fn ruliad_structured_answer_contrast_loss_supports_structured_symbolic_tokenizer() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 29);
        let mut config = tiny_model_config();
        config.vocab_size = 512;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_ruliad_supervision(RuliadSupervisionConfig {
                verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                    enabled: true,
                    structured_contrast_weight: 0.25,
                    structured_contrast_every_steps: 1,
                    structured_negative_count: 2,
                    structured_template_negative_count: 1,
                    max_completion_tokens: 32,
                    ..Default::default()
                },
                ..Default::default()
            });
        let tokenizer =
            burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer::from_config(
                &burn_dragon_universality::RuliadTokenizationConfig::StructuredSymbolic {
                    vocab_size: 512,
                    eos_id: None,
                },
            )
            .expect("tokenizer");
        let prompt_tokens = tokenizer
            .encode_payload("?:ss\n!:")
            .into_iter()
            .map(i64::from)
            .collect::<Vec<_>>();
        let item = burn_dragon_universality::RuliadEvalItem {
            oracle_hash: "h0".to_string(),
            sample_index: 43,
            split: burn_dragon_universality::SampleSplit::Train,
            family: "formal_proof".to_string(),
            task_kind: "select_proof_action".to_string(),
            math_domains: vec!["category".to_string()],
            reasoning_modes: vec!["equational".to_string()],
            prompt: "?:ss\n!:".to_string(),
            expected_answer: "g4|a:r0|f|1.1".to_string(),
            difficulty_level: Some(0),
            spec: None,
        };
        let policy_batch = crate::dataset::RuliadPolicyBatch {
            samples: vec![crate::dataset::RuliadPolicySample {
                item,
                prompt_tokens,
            }],
            tokenization: burn_dragon_universality::RuliadTokenizationConfig::StructuredSymbolic {
                vocab_size: 512,
                eos_id: None,
            },
            stop_token_id: None,
        };

        let loss = model
            .ruliad_structured_answer_contrast_loss(&policy_batch, &device, 64)
            .expect("structured symbolic contrast loss");

        let loss = tensor_scalar(loss);
        assert!(loss.is_finite(), "contrast loss should be finite: {loss}");
        assert!(loss > 0.0, "contrast loss should be non-zero: {loss}");
    }

    #[test]
    fn ruliad_verifier_policy_loss_can_include_structured_negative_candidates() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 19);
        let dir = tempfile::tempdir().expect("tempdir");
        let telemetry_path = dir
            .path()
            .join("events")
            .join("ruliad_verifier_policy.jsonl");
        let mut config = tiny_model_config();
        config.vocab_size = 257;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_ruliad_supervision(RuliadSupervisionConfig {
                verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                    enabled: true,
                    mode: crate::config::train::RuliadVerifierRewardMode::VpoIndependent,
                    weight: 0.1,
                    group_size: 2,
                    max_completion_tokens: 24,
                    every_steps: 1,
                    top_k: 1,
                    kl_weight: 0.0,
                    vpo_scalarizations: 4,
                    positive_advantage_requires_correctness: true,
                    positive_advantage_min_partial_progress_ppm: 500_000,
                    positive_advantage_min_completion_quality_ppm: 750_000,
                    include_oracle_candidate: true,
                    include_structured_negative_candidates: true,
                    structured_negative_count: 2,
                    ..Default::default()
                },
                ..Default::default()
            })
            .with_ruliad_policy_telemetry_path(Some(telemetry_path.clone()));
        let item = burn_dragon_universality::RuliadEvalItem {
            oracle_hash: "h0".to_string(),
            sample_index: 31,
            split: burn_dragon_universality::SampleSplit::Train,
            family: "formal_proof".to_string(),
            task_kind: "select_proof_action".to_string(),
            math_domains: vec!["category".to_string(), "formal_proof".to_string()],
            reasoning_modes: vec!["equational".to_string()],
            prompt: "?:ss\n!:".to_string(),
            expected_answer: "ok=1;l=17;r=17".to_string(),
            difficulty_level: Some(0),
            spec: None,
        };
        let policy_batch = crate::dataset::RuliadPolicyBatch {
            samples: vec![crate::dataset::RuliadPolicySample {
                item,
                prompt_tokens: vec![1, 2, 3],
            }],
            tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
                vocab_size: 257,
                eos_id: None,
            },
            stop_token_id: None,
        };
        let loss = model
            .ruliad_verifier_policy_loss(&policy_batch, &device, 48)
            .expect("structured-negative VPO verifier policy loss");
        assert!(tensor_scalar(loss).is_finite());
        let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
        let value: serde_json::Value =
            serde_json::from_str(content.lines().next().expect("telemetry line"))
                .expect("telemetry json");

        assert_eq!(value["oracle_completion_rows"], 1);
        assert_eq!(value["structured_negative_completion_rows"], 2);
        assert_eq!(value["policy_update_applied"], true);
        assert!(
            value["completion_rows"].as_u64().expect("completion rows") >= 3,
            "oracle plus structured negatives should contribute trainable policy rows"
        );
        assert!(
            value["vector_schema_quality_mean"]
                .as_f64()
                .expect("schema quality")
                > 0.0,
            "structured negatives should remain parseable enough to teach field binding"
        );
    }

    #[test]
    fn ruliad_verifier_rollout_imitation_writes_skip_telemetry_for_wrong_generations() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 20);
        let dir = tempfile::tempdir().expect("tempdir");
        let telemetry_path = dir
            .path()
            .join("events")
            .join("ruliad_verifier_rollout_imitation.jsonl");
        let mut config = tiny_model_config();
        config.vocab_size = 257;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_ruliad_supervision(RuliadSupervisionConfig {
                verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                    enabled: true,
                    weight: 0.0,
                    group_size: 2,
                    max_completion_tokens: 8,
                    top_k: 1,
                    rollout_imitation_weight: 0.05,
                    rollout_imitation_every_steps: 1,
                    rollout_imitation_min_partial_progress_ppm: 500_000,
                    rollout_imitation_min_completion_quality_ppm: 750_000,
                    ..Default::default()
                },
                ..Default::default()
            })
            .with_ruliad_verifier_rollout_telemetry_path(Some(telemetry_path.clone()));
        let item = burn_dragon_universality::RuliadEvalItem {
            oracle_hash: "h0".to_string(),
            sample_index: 33,
            split: burn_dragon_universality::SampleSplit::Train,
            family: "law".to_string(),
            task_kind: "category_law".to_string(),
            math_domains: vec!["category".to_string()],
            reasoning_modes: vec!["equational".to_string()],
            prompt: "?:q\n!:".to_string(),
            expected_answer: "ok=1".to_string(),
            difficulty_level: Some(0),
            spec: None,
        };
        let policy_batch = crate::dataset::RuliadPolicyBatch {
            samples: vec![crate::dataset::RuliadPolicySample {
                item,
                prompt_tokens: vec![1, 2, 3],
            }],
            tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
                vocab_size: 257,
                eos_id: None,
            },
            stop_token_id: None,
        };

        assert!(
            model
                .ruliad_verifier_rollout_imitation_loss(&policy_batch, &device, 16)
                .is_none(),
            "wrong generated completions should not be reinforced"
        );
        let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
        let value: serde_json::Value =
            serde_json::from_str(content.lines().next().expect("telemetry line"))
                .expect("telemetry json");
        let skip_reason = value["skip_reason"].as_str();
        assert!(
            matches!(
                skip_reason,
                Some("no_candidate_completion") | Some("rollout_health_gate")
            ),
            "unexpected skip reason: {skip_reason:?}"
        );
        assert_eq!(value["accepted_completion_rows"].as_u64(), Some(0));
        assert_eq!(value["accepted_imitation_rows"].as_u64(), Some(0));
        assert_eq!(value["accepted_recovery_rows"].as_u64(), Some(0));
        let candidate_rows = value["candidate_completion_rows"]
            .as_u64()
            .expect("candidate rows");
        if skip_reason == Some("rollout_health_gate") {
            assert!(candidate_rows > 0);
        } else {
            assert_eq!(candidate_rows, 0);
        }
        assert_eq!(value["health_gate_passed"].as_bool(), Some(false));
        assert!(
            value["generated_completion_rows"]
                .as_u64()
                .expect("generated rows")
                > 0
        );
    }

    #[test]
    fn ruliad_verifier_rollout_recovery_accepts_generated_malformed_prefixes() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 23);
        let dir = tempfile::tempdir().expect("tempdir");
        let telemetry_path = dir
            .path()
            .join("events")
            .join("ruliad_verifier_rollout_recovery.jsonl");
        let mut config = tiny_model_config();
        config.vocab_size = 257;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_ruliad_supervision(RuliadSupervisionConfig {
                verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                    enabled: true,
                    weight: 0.0,
                    group_size: 1,
                    max_completion_tokens: 1,
                    top_k: 1,
                    rollout_recovery_weight: 0.05,
                    rollout_imitation_weight: 0.0,
                    rollout_imitation_every_steps: 1,
                    rollout_imitation_min_partial_progress_ppm: 0,
                    rollout_imitation_min_completion_quality_ppm: 0,
                    rollout_imitation_max_rows_per_step: 1,
                    ..Default::default()
                },
                ..Default::default()
            })
            .with_ruliad_verifier_rollout_telemetry_path(Some(telemetry_path.clone()));
        let item = burn_dragon_universality::RuliadEvalItem {
            oracle_hash: "h0".to_string(),
            sample_index: 37,
            split: burn_dragon_universality::SampleSplit::Train,
            family: "rewrite".to_string(),
            task_kind: "rewrite_normal_form".to_string(),
            math_domains: vec!["symbolic_rewriting".to_string()],
            reasoning_modes: vec!["normalization".to_string()],
            prompt: "?:q\n!:".to_string(),
            expected_answer: "nflen=3;nfalpha=ABC;nfcounts=1,1,1;nfedge=AB".to_string(),
            difficulty_level: Some(0),
            spec: None,
        };
        let policy_batch = crate::dataset::RuliadPolicyBatch {
            samples: vec![crate::dataset::RuliadPolicySample {
                item,
                prompt_tokens: vec![1, 2, 3],
            }],
            tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
                vocab_size: 257,
                eos_id: None,
            },
            stop_token_id: None,
        };

        let loss = model
            .ruliad_verifier_rollout_imitation_loss(&policy_batch, &device, 16)
            .expect("malformed rollout should create an oracle recovery row");
        assert!(tensor_scalar(loss).is_finite());
        let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
        let value: serde_json::Value =
            serde_json::from_str(content.lines().next().expect("telemetry line"))
                .expect("telemetry json");
        assert_eq!(value["accepted_imitation_rows"].as_u64(), Some(0));
        assert_eq!(value["accepted_recovery_rows"].as_u64(), Some(1));
        let malformed = value["recovery_malformed_rows"]
            .as_u64()
            .unwrap_or_default();
        let missing = value["recovery_missing_rows"].as_u64().unwrap_or_default();
        let schema_wrong = value["recovery_schema_wrong_rows"]
            .as_u64()
            .unwrap_or_default();
        let partial = value["recovery_partial_rows"].as_u64().unwrap_or_default();
        assert_eq!(malformed + missing + schema_wrong + partial, 1);
    }

    #[test]
    fn ruliad_proof_policy_masks_only_the_action_bearing_token() {
        let prompt = [1, 2, 3];
        let completion = [4, 5, 6, 7];
        let (_, targets, mask) =
            LanguageTrainModel::<TestBackend>::ruliad_policy_row_from_completion_token(
                &prompt,
                &completion,
                2,
            )
            .expect("action-token policy row");
        assert_eq!(mask.iter().filter(|value| **value > 0.0).count(), 1);
        assert_eq!(mask[4], 1.0);
        assert_eq!(targets[4], 6);
    }

    #[test]
    fn verifier_equivalent_action_loss_marginalizes_all_valid_tokens() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let logits = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![0.0, 0.0, 0.0, 0.0], [1, 1, 4]),
            &device,
        );
        let one_valid = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![1.0, 0.0, 0.0, 0.0], [1, 1, 4]),
            &device,
        );
        let two_valid = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![1.0, 1.0, 0.0, 0.0], [1, 1, 4]),
            &device,
        );
        let candidates = Tensor::<TestBackend, 3>::ones([1, 1, 4], &device);

        let one_loss = tensor_scalar(verifier_equivalent_action_loss(
            logits.clone(),
            candidates.clone(),
            one_valid,
            crate::config::RuliadProofPolicyNormalization::CandidateConditional,
            1.0,
        ));
        let two_loss = tensor_scalar(verifier_equivalent_action_loss(
            logits,
            candidates,
            two_valid,
            crate::config::RuliadProofPolicyNormalization::CandidateConditional,
            1.0,
        ));
        assert!((one_loss - 4.0f32.ln()).abs() < 1.0e-5, "{one_loss}");
        assert!((two_loss - 2.0f32.ln()).abs() < 1.0e-5, "{two_loss}");
        assert!(two_loss < one_loss);
    }

    #[test]
    fn verifier_equivalent_action_loss_ignores_non_candidate_logits() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let candidate_mask = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![1.0, 1.0, 0.0], [1, 1, 3]),
            &device,
        );
        let equivalent_mask = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![1.0, 0.0, 0.0], [1, 1, 3]),
            &device,
        );
        let baseline = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![0.0, 0.0, 0.0], [1, 1, 3]),
            &device,
        );
        let dominant_non_candidate = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![0.0, 0.0, 20.0], [1, 1, 3]),
            &device,
        );

        let baseline_loss = tensor_scalar(verifier_equivalent_action_loss(
            baseline,
            candidate_mask.clone(),
            equivalent_mask.clone(),
            crate::config::RuliadProofPolicyNormalization::CandidateConditional,
            1.0,
        ));
        let perturbed_loss = tensor_scalar(verifier_equivalent_action_loss(
            dominant_non_candidate,
            candidate_mask,
            equivalent_mask,
            crate::config::RuliadProofPolicyNormalization::CandidateConditional,
            1.0,
        ));
        assert!((baseline_loss - 2.0f32.ln()).abs() < 1.0e-5);
        assert!((perturbed_loss - baseline_loss).abs() < 1.0e-5);
    }

    #[test]
    fn vocabulary_marginal_action_loss_penalizes_non_candidate_probability() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let candidate_mask = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![1.0, 1.0, 0.0], [1, 1, 3]),
            &device,
        );
        let equivalent_mask = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![1.0, 0.0, 0.0], [1, 1, 3]),
            &device,
        );
        let baseline = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![0.0, 0.0, 0.0], [1, 1, 3]),
            &device,
        );
        let dominant_non_candidate = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![0.0, 0.0, 20.0], [1, 1, 3]),
            &device,
        );

        let baseline_loss = tensor_scalar(verifier_equivalent_action_loss(
            baseline,
            candidate_mask.clone(),
            equivalent_mask.clone(),
            crate::config::RuliadProofPolicyNormalization::VocabularyMarginal,
            1.0,
        ));
        let perturbed_loss = tensor_scalar(verifier_equivalent_action_loss(
            dominant_non_candidate,
            candidate_mask,
            equivalent_mask,
            crate::config::RuliadProofPolicyNormalization::VocabularyMarginal,
            1.0,
        ));
        assert!((baseline_loss - 3.0f32.ln()).abs() < 1.0e-5);
        assert!(perturbed_loss > baseline_loss + 10.0, "{perturbed_loss}");
    }

    #[test]
    fn semantic_sequence_policy_loss_marginalizes_verifier_equivalent_actions() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let scores = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![0.4f32.ln(), 0.1f32.ln(), 0.1f32.ln()], [1, 3]),
            &device,
        );
        let equivalent = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![1.0, 0.0, 1.0], [1, 3]),
            &device,
        );
        let weights = Tensor::<TestBackend, 1>::ones([1], &device);
        let conditional = tensor_scalar(grouped_verifier_equivalent_sequence_loss(
            scores.clone(),
            scores.clone(),
            equivalent.clone(),
            weights.clone(),
            GroupedVerifierSequenceLossConfig {
                normalization: crate::config::RuliadProofPolicyNormalization::CandidateConditional,
                presentation_risk: crate::config::RuliadProofPolicyPresentationRisk::Mean,
                presentation_group_size: 1,
                weight: 1.0,
            },
        ));
        let marginal = tensor_scalar(grouped_verifier_equivalent_sequence_loss(
            scores.clone(),
            scores,
            equivalent,
            weights,
            GroupedVerifierSequenceLossConfig {
                normalization: crate::config::RuliadProofPolicyNormalization::VocabularyMarginal,
                presentation_risk: crate::config::RuliadProofPolicyPresentationRisk::Mean,
                presentation_group_size: 1,
                weight: 1.0,
            },
        ));
        let expected_conditional = -(5.0f32 / 6.0).ln();
        let expected_marginal = -0.5f32.ln();
        assert!(
            (conditional - expected_conditional).abs() < 1.0e-5,
            "conditional={conditional}"
        );
        assert!(
            (marginal - expected_marginal).abs() < 1.0e-5,
            "marginal={marginal}"
        );
        assert!(marginal > conditional);
    }

    #[test]
    fn worst_presentation_risk_targets_each_groups_weakest_orbit_member() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let logits = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                vec![
                    0.9f32.ln(),
                    0.1f32.ln(),
                    0.6f32.ln(),
                    0.4f32.ln(),
                    0.8f32.ln(),
                    0.2f32.ln(),
                    0.2f32.ln(),
                    0.8f32.ln(),
                ],
                [4, 1, 2],
            ),
            &device,
        );
        let candidates = Tensor::<TestBackend, 3>::ones([4, 1, 2], &device);
        let equivalent = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [4, 1, 2]),
            &device,
        );
        let row_weights =
            Tensor::<TestBackend, 1>::from_data(TensorData::new(vec![0.5; 4], [4]), &device);
        let mean = tensor_scalar(grouped_verifier_equivalent_action_loss(
            logits.clone(),
            candidates.clone(),
            equivalent.clone(),
            row_weights.clone(),
            crate::config::RuliadProofPolicyNormalization::VocabularyMarginal,
            crate::config::RuliadProofPolicyPresentationRisk::Mean,
            2,
            1.0,
        ));
        let worst = tensor_scalar(grouped_verifier_equivalent_action_loss(
            logits,
            candidates,
            equivalent,
            row_weights,
            crate::config::RuliadProofPolicyNormalization::VocabularyMarginal,
            crate::config::RuliadProofPolicyPresentationRisk::Worst,
            2,
            1.0,
        ));

        let expected_worst = -(0.6f32.ln() + 0.2f32.ln()) / 2.0;
        assert!((worst - expected_worst).abs() < 1.0e-5, "{worst}");
        assert!(worst > mean, "mean={mean} worst={worst}");
    }

    #[test]
    fn ruliad_proof_policy_dagger_labels_model_visited_state_with_expert_action() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 29);
        let dir = tempfile::tempdir().expect("tempdir");
        let telemetry_path = dir
            .path()
            .join("events")
            .join("ruliad_proof_policy_dagger.jsonl");
        let mut model_config = tiny_model_config();
        model_config.vocab_size = 272;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(model_config, &device))
            .with_ruliad_supervision(RuliadSupervisionConfig {
                proof_policy: crate::config::RuliadProofPolicyTrainingConfig {
                    enabled: true,
                    mode: crate::config::RuliadProofPolicyTrainingMode::Dagger,
                    scoring: crate::config::RuliadProofPolicyScoring::CompletionLikelihood,
                    gradient_scope: crate::config::RuliadProofPolicyGradientScope::FullModel,
                    normalization:
                        crate::config::RuliadProofPolicyNormalization::VocabularyMarginal,
                    candidate_symmetry:
                        crate::config::RuliadProofPolicyCandidateSymmetry::BalancedRotation,
                    presentation_risk: crate::config::RuliadProofPolicyPresentationRisk::Mean,
                    weight: 1.0,
                    every_steps: 1,
                    start_after_steps: 0,
                    dagger_start_after_steps: 1,
                    stratified_difficulty_levels: 0,
                    rollout_steps: 2,
                    max_rows_per_update: 2,
                    max_presentation_rows_per_update: 32,
                    counterfactual_targets_per_state: 0,
                    candidates: 4,
                    max_completion_tokens: 16,
                },
                ..Default::default()
            })
            .with_ruliad_proof_policy_telemetry_path(Some(telemetry_path.clone()));
        let bundle = burn_dragon_universality::ruliad::formal::generate_formal_bundle(
            29,
            burn_dragon_universality::ruliad::formal::RuliadFormalGeneratorConfig {
                rewrite_depth: 2,
                leaf_count: 3,
                context_depth: 1,
                distractor_axioms: 1,
                ..Default::default()
            },
        )
        .expect("formal bundle");
        let proof_step_index = 1.min(bundle.certificate.step_count().saturating_sub(1));
        assert!(proof_step_index > 0, "fixture needs a nonzero proof step");
        let actions = burn_dragon_universality::ruliad::oracle_proof_action_set(
            &bundle.problem,
            &bundle.certificate,
            proof_step_index,
            4,
        )
        .expect("oracle action set");
        let problem_hash = bundle.problem.canonical_hash().expect("problem hash");
        let item = burn_dragon_universality::RuliadEvalItem {
            oracle_hash: problem_hash,
            sample_index: 29,
            split: burn_dragon_universality::SampleSplit::Train,
            family: "formal_proof".to_string(),
            task_kind: burn_dragon_universality::RuliadTaskKind::SelectProofAction
                .label()
                .to_string(),
            math_domains: vec!["formal_proof".to_string()],
            reasoning_modes: vec!["proof_construction".to_string()],
            prompt: burn_dragon_universality::ruliad::ruliad_proof_action_prompt(
                &bundle.problem,
                &actions,
            )
            .expect("policy prompt"),
            expected_answer: format!("c={}", actions.selected_index),
            difficulty_level: Some(0),
            spec: Some(burn_dragon_universality::RuliadSampleSpec::FormalProof {
                problem: bundle.problem,
                certificate: bundle.certificate,
                candidate: None,
                proof_step_index: Some(proof_step_index),
                action_presentation_rotation: Some(0),
                action_answer_contract: Default::default(),
                task: burn_dragon_universality::RuliadTaskKind::SelectProofAction,
            }),
        };
        let mut policy_batch = crate::dataset::RuliadPolicyBatch {
            samples: vec![crate::dataset::RuliadPolicySample {
                item,
                prompt_tokens: vec![1],
            }],
            tokenization: burn_dragon_universality::RuliadTokenizationConfig::StructuredSymbolic {
                vocab_size: 272,
                eos_id: Some(271),
            },
            stop_token_id: Some(271),
        };
        policy_batch.samples.push(policy_batch.samples[0].clone());

        let loss = model
            .ruliad_proof_policy_dagger_loss(&policy_batch, &device, 512)
            .expect("DAgger expert correction loss");
        assert!(tensor_scalar(loss).is_finite());
        let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
        let value: serde_json::Value =
            serde_json::from_str(content.lines().next().expect("telemetry line"))
                .expect("telemetry json");
        assert_eq!(value["version"], 19);
        assert_eq!(value["answer_contract"], "presentation_index");
        assert_eq!(value["objective"], "vocabulary_marginal_equivalent_v1");
        assert_eq!(value["presentation_risk"], "mean");
        assert_eq!(value["configured_mode"], "dagger");
        assert_eq!(value["mode"], "dagger");
        assert_eq!(value["candidate_symmetry"], "balanced_rotation");
        assert_eq!(value["available_sample_groups"], 2);
        assert_eq!(value["sample_groups"], 1);
        assert_eq!(value["nonzero_start_trajectories"], 1);
        assert_eq!(value["mean_start_step"], proof_step_index as f64);
        assert!(value["visited_states"].as_u64().unwrap_or_default() >= 1);
        assert_eq!(value["semantic_state_rows"], value["expert_rows"]);
        assert!(value["expert_rows"].as_u64().unwrap_or_default() >= 1);
        assert_eq!(value["static_expert_rows"], 0);
        assert!(value["dagger_expert_rows"].as_u64().unwrap_or_default() >= 1);
        assert_eq!(value["supervised_action_tokens"], value["expert_rows"]);
        assert_eq!(value["supervised_presentation_rows"], value["expert_rows"]);
        assert_eq!(value["mean_presentations_per_state"], 1.0);
        assert!(value["model_scoring_batches"].as_u64().unwrap_or_default() >= 1);
        assert_eq!(value["maximum_model_scoring_batch_rows"], 1);
        assert!(
            value["model_scoring_padded_tokens"]
                .as_u64()
                .unwrap_or_default()
                > 0
        );
        assert!(value["sampling_model_materialize_ms"].is_number());
        assert!(value["state_prepare_ms"].is_number());
        assert!(value["rollout_cpu_prepare_ms"].is_number());
        assert!(value["model_scoring_ms"].is_number());
        assert_eq!(value["trajectory_budget"], 1);
        assert_eq!(value["semantic_row_budget"], 2);
        assert_eq!(value["max_rows_per_update"], 2);
        assert_eq!(value["max_presentation_rows_per_update"], 32);
        assert!(value["rollout_depth_reached"].as_u64().unwrap_or_default() >= 2);
        assert!(
            value["model_visited_expert_rows"]
                .as_u64()
                .unwrap_or_default()
                >= 1
        );
        assert!(
            value["equivalent_target_tokens"]
                .as_u64()
                .unwrap_or_default()
                >= value["expert_rows"].as_u64().unwrap_or_default()
        );
        assert!(
            value["candidate_target_tokens"]
                .as_u64()
                .unwrap_or_default()
                >= value["equivalent_target_tokens"]
                    .as_u64()
                    .unwrap_or_default()
        );
        assert!(
            value["mean_candidate_targets_per_row"]
                .as_f64()
                .unwrap_or_default()
                >= value["mean_equivalent_targets_per_row"]
                    .as_f64()
                    .unwrap_or_default()
        );
        assert!(
            value["mean_equivalent_targets_per_row"]
                .as_f64()
                .unwrap_or_default()
                >= 1.0
        );
        assert!(value["expert_selected_index_histogram"].is_object());
        assert!(value["expert_equivalent_index_histogram"].is_object());
        assert!(value["model_selected_index_histogram"].is_object());
        assert_eq!(value["difficulty_sample_groups"]["0"], 1);
        assert!(
            value["difficulty_visited_states"]["0"]
                .as_u64()
                .unwrap_or_default()
                >= 1
        );

        let static_telemetry_path = dir
            .path()
            .join("events")
            .join("ruliad_proof_policy_static.jsonl");
        let mut static_model_config = tiny_model_config();
        static_model_config.vocab_size = 272;
        let static_model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            static_model_config,
            &device,
        ))
        .with_ruliad_supervision(RuliadSupervisionConfig {
            proof_policy: crate::config::RuliadProofPolicyTrainingConfig {
                enabled: true,
                mode: crate::config::RuliadProofPolicyTrainingMode::StaticExpert,
                scoring: crate::config::RuliadProofPolicyScoring::CompletionLikelihood,
                gradient_scope: crate::config::RuliadProofPolicyGradientScope::FullModel,
                normalization: crate::config::RuliadProofPolicyNormalization::CandidateConditional,
                candidate_symmetry:
                    crate::config::RuliadProofPolicyCandidateSymmetry::CyclicOrbitAverage,
                presentation_risk: crate::config::RuliadProofPolicyPresentationRisk::Mean,
                weight: 1.0,
                every_steps: 1,
                start_after_steps: 0,
                dagger_start_after_steps: 1,
                stratified_difficulty_levels: 0,
                rollout_steps: 8,
                max_rows_per_update: 2,
                max_presentation_rows_per_update: 8,
                counterfactual_targets_per_state: 0,
                candidates: 4,
                max_completion_tokens: 16,
            },
            ..Default::default()
        })
        .with_ruliad_proof_policy_telemetry_path(Some(static_telemetry_path.clone()));
        let static_loss = static_model
            .ruliad_proof_policy_dagger_loss(&policy_batch, &device, 512)
            .expect("static expert policy loss");
        assert!(tensor_scalar(static_loss).is_finite());
        let static_content =
            std::fs::read_to_string(static_telemetry_path).expect("static telemetry sidecar");
        let static_value: serde_json::Value = serde_json::from_str(
            static_content
                .lines()
                .next()
                .expect("static telemetry line"),
        )
        .expect("static telemetry json");
        assert_eq!(static_value["version"], 19);
        assert_eq!(static_value["answer_contract"], "presentation_index");
        assert_eq!(static_value["presentation_risk"], "mean");
        assert_eq!(static_value["configured_mode"], "static_expert");
        assert_eq!(static_value["mode"], "static_expert");
        assert_eq!(static_value["candidate_symmetry"], "cyclic_orbit_average");
        assert_eq!(static_value["rollout_steps"], 1);
        assert_eq!(static_value["configured_rollout_steps"], 8);
        assert_eq!(static_value["model_scoring_batches"], 0);
        assert_eq!(static_value["semantic_row_budget"], 2);
        assert_eq!(static_value["max_presentation_rows_per_update"], 8);
        assert!(static_value["expert_rows"].as_u64().unwrap_or_default() >= 1);
        assert!(
            static_value["static_expert_rows"]
                .as_u64()
                .unwrap_or_default()
                >= 1
        );
        assert_eq!(static_value["dagger_expert_rows"], 0);
        assert!(
            static_value["supervised_presentation_rows"]
                .as_u64()
                .unwrap_or_default()
                >= static_value["expert_rows"]
                    .as_u64()
                    .unwrap_or_default()
                    .saturating_mul(2)
        );
        assert!(
            static_value["mean_presentations_per_state"]
                .as_f64()
                .unwrap_or_default()
                >= 2.0
        );
        assert!(
            static_value["supervised_presentation_rows"]
                .as_u64()
                .unwrap_or_default()
                <= 8
        );

        let semantic_telemetry_path = dir
            .path()
            .join("events")
            .join("ruliad_proof_policy_semantic.jsonl");
        let mut semantic_batch = policy_batch.clone();
        for sample in &mut semantic_batch.samples {
            let Some(burn_dragon_universality::RuliadSampleSpec::FormalProof {
                action_answer_contract,
                ..
            }) = sample.item.spec.as_mut()
            else {
                panic!("formal proof fixture");
            };
            *action_answer_contract =
                burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep;
        }
        let mut semantic_model_config = tiny_model_config();
        semantic_model_config.vocab_size = 272;
        let semantic_model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            semantic_model_config,
            &device,
        ))
        .with_ruliad_supervision(RuliadSupervisionConfig {
            proof_policy: crate::config::RuliadProofPolicyTrainingConfig {
                enabled: true,
                mode: crate::config::RuliadProofPolicyTrainingMode::StaticExpert,
                scoring: crate::config::RuliadProofPolicyScoring::CompletionLikelihood,
                gradient_scope: crate::config::RuliadProofPolicyGradientScope::FullModel,
                normalization: crate::config::RuliadProofPolicyNormalization::CandidateConditional,
                candidate_symmetry:
                    crate::config::RuliadProofPolicyCandidateSymmetry::CyclicOrbitAverage,
                presentation_risk: crate::config::RuliadProofPolicyPresentationRisk::Worst,
                weight: 1.0,
                every_steps: 1,
                start_after_steps: 0,
                dagger_start_after_steps: 1,
                stratified_difficulty_levels: 0,
                rollout_steps: 1,
                max_rows_per_update: 1,
                max_presentation_rows_per_update: 8,
                counterfactual_targets_per_state: 0,
                candidates: 4,
                max_completion_tokens: 128,
            },
            ..Default::default()
        })
        .with_ruliad_proof_policy_telemetry_path(Some(semantic_telemetry_path.clone()));
        let semantic_loss = semantic_model
            .ruliad_proof_policy_dagger_loss(&semantic_batch, &device, 512)
            .expect("semantic proof-step policy loss");
        assert!(tensor_scalar(semantic_loss.clone()).is_finite());
        let _semantic_gradients = semantic_loss.backward();
        let semantic_content =
            std::fs::read_to_string(semantic_telemetry_path).expect("semantic telemetry sidecar");
        let semantic_value: serde_json::Value = serde_json::from_str(
            semantic_content
                .lines()
                .next()
                .expect("semantic telemetry line"),
        )
        .expect("semantic telemetry json");
        assert_eq!(semantic_value["version"], 19);
        assert_eq!(semantic_value["answer_contract"], "semantic_step");
        assert_eq!(semantic_value["presentation_risk"], "worst");
        assert!(
            semantic_value["supervised_action_tokens"]
                .as_u64()
                .unwrap_or_default()
                > semantic_value["supervised_presentation_rows"]
                    .as_u64()
                    .unwrap_or_default()
        );

        let energy_telemetry_path = dir
            .path()
            .join("events")
            .join("ruliad_proof_policy_semantic_energy.jsonl");
        let mut energy_model_config = tiny_model_config();
        energy_model_config.vocab_size = 272;
        energy_model_config.sequence_score_head.enabled = true;
        let energy_model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            energy_model_config,
            &device,
        ))
        .with_ruliad_supervision(RuliadSupervisionConfig {
            proof_policy: crate::config::RuliadProofPolicyTrainingConfig {
                enabled: true,
                mode: crate::config::RuliadProofPolicyTrainingMode::StaticExpert,
                scoring: crate::config::RuliadProofPolicyScoring::SemanticEnergy,
                gradient_scope: crate::config::RuliadProofPolicyGradientScope::FullModel,
                normalization: crate::config::RuliadProofPolicyNormalization::CandidateConditional,
                candidate_symmetry:
                    crate::config::RuliadProofPolicyCandidateSymmetry::BalancedRotation,
                presentation_risk: crate::config::RuliadProofPolicyPresentationRisk::Mean,
                weight: 1.0,
                every_steps: 1,
                start_after_steps: 0,
                dagger_start_after_steps: 1,
                stratified_difficulty_levels: 0,
                rollout_steps: 1,
                max_rows_per_update: 2,
                max_presentation_rows_per_update: 2,
                counterfactual_targets_per_state: 1,
                candidates: 4,
                max_completion_tokens: 128,
            },
            ..Default::default()
        })
        .with_ruliad_proof_policy_telemetry_path(Some(energy_telemetry_path.clone()));
        let energy_loss = energy_model
            .ruliad_proof_policy_dagger_loss(&policy_batch, &device, 512)
            .expect("semantic-energy proof policy loss");
        assert!(tensor_scalar(energy_loss.clone()).is_finite());
        let _energy_gradients = energy_loss.backward();
        let energy_content =
            std::fs::read_to_string(energy_telemetry_path).expect("energy telemetry sidecar");
        let energy_value: serde_json::Value = serde_json::from_str(
            energy_content
                .lines()
                .next()
                .expect("energy telemetry line"),
        )
        .expect("energy telemetry json");
        assert_eq!(energy_value["version"], 19);
        assert_eq!(energy_value["answer_contract"], "semantic_step");
        assert_eq!(energy_value["gradient_scope"], "full_model");
        assert_eq!(
            energy_value["objective"],
            "semantic_sequence_energy_counterfactual_v1"
        );
        assert_eq!(
            energy_value["configured_counterfactual_targets_per_state"],
            1
        );
        assert_eq!(energy_value["target_variants_per_state"], 2);
        assert_eq!(energy_value["base_semantic_row_budget"], 1);
        assert_eq!(energy_value["base_semantic_state_rows"], 1);
        assert_eq!(energy_value["counterfactual_semantic_state_rows"], 1);
        assert_eq!(energy_value["counterfactual_target_shortfall"], 0);
        assert_eq!(energy_value["semantic_state_rows"], 2);

        let language_head_telemetry_path = dir
            .path()
            .join("events")
            .join("ruliad_proof_policy_language_head.jsonl");
        let mut language_head_model_config = tiny_model_config();
        language_head_model_config.vocab_size = 272;
        language_head_model_config.tie_input_output_embeddings = false;
        let language_head_model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            language_head_model_config,
            &device,
        ))
        .with_ruliad_supervision(RuliadSupervisionConfig {
            proof_policy: crate::config::RuliadProofPolicyTrainingConfig {
                enabled: true,
                mode: crate::config::RuliadProofPolicyTrainingMode::StaticExpert,
                scoring: crate::config::RuliadProofPolicyScoring::CompletionLikelihood,
                gradient_scope: crate::config::RuliadProofPolicyGradientScope::LanguageHeadOnly,
                normalization: crate::config::RuliadProofPolicyNormalization::CandidateConditional,
                candidate_symmetry:
                    crate::config::RuliadProofPolicyCandidateSymmetry::BalancedRotation,
                presentation_risk: crate::config::RuliadProofPolicyPresentationRisk::Mean,
                weight: 1.0,
                every_steps: 1,
                start_after_steps: 0,
                dagger_start_after_steps: 1,
                stratified_difficulty_levels: 0,
                rollout_steps: 1,
                max_rows_per_update: 2,
                max_presentation_rows_per_update: 2,
                counterfactual_targets_per_state: 1,
                candidates: 4,
                max_completion_tokens: 128,
            },
            ..Default::default()
        })
        .with_ruliad_proof_policy_telemetry_path(Some(language_head_telemetry_path.clone()));
        let language_head_loss = language_head_model
            .ruliad_proof_policy_dagger_loss(&semantic_batch, &device, 512)
            .expect("language-head-only counterfactual proof policy loss");
        assert!(tensor_scalar(language_head_loss.clone()).is_finite());
        let _language_head_gradients = language_head_loss.backward();
        let language_head_content = std::fs::read_to_string(language_head_telemetry_path)
            .expect("language-head telemetry sidecar");
        let language_head_value: serde_json::Value = serde_json::from_str(
            language_head_content
                .lines()
                .next()
                .expect("language-head telemetry line"),
        )
        .expect("language-head telemetry json");
        assert_eq!(language_head_value["version"], 19);
        assert_eq!(language_head_value["answer_contract"], "semantic_step");
        assert_eq!(language_head_value["gradient_scope"], "language_head_only");
        assert_eq!(
            language_head_value["objective"],
            "candidate_normalized_counterfactual_v1"
        );
        assert_eq!(
            language_head_value["configured_counterfactual_targets_per_state"],
            1
        );
        assert_eq!(language_head_value["target_variants_per_state"], 2);
        assert_eq!(language_head_value["base_semantic_state_rows"], 1);
        assert_eq!(language_head_value["counterfactual_semantic_state_rows"], 1);
        assert_eq!(language_head_value["counterfactual_target_shortfall"], 0);

        let prefix_telemetry_path = dir
            .path()
            .join("events")
            .join("ruliad_proof_policy_semantic_prefix.jsonl");
        let mut prefix_model_config = tiny_model_config();
        prefix_model_config.vocab_size = 272;
        let prefix_model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            prefix_model_config,
            &device,
        ))
        .with_ruliad_supervision(RuliadSupervisionConfig {
            proof_policy: crate::config::RuliadProofPolicyTrainingConfig {
                enabled: true,
                mode: crate::config::RuliadProofPolicyTrainingMode::StaticExpert,
                scoring: crate::config::RuliadProofPolicyScoring::CompletionLikelihood,
                gradient_scope: crate::config::RuliadProofPolicyGradientScope::FullModel,
                normalization: crate::config::RuliadProofPolicyNormalization::PrefixConditional,
                candidate_symmetry:
                    crate::config::RuliadProofPolicyCandidateSymmetry::BalancedRotation,
                presentation_risk: crate::config::RuliadProofPolicyPresentationRisk::Mean,
                weight: 1.0,
                every_steps: 1,
                start_after_steps: 0,
                dagger_start_after_steps: 1,
                stratified_difficulty_levels: 0,
                rollout_steps: 1,
                max_rows_per_update: 2,
                max_presentation_rows_per_update: 2,
                counterfactual_targets_per_state: 0,
                candidates: 4,
                max_completion_tokens: 128,
            },
            ..Default::default()
        })
        .with_ruliad_proof_policy_telemetry_path(Some(prefix_telemetry_path.clone()));
        let prefix_loss = prefix_model
            .ruliad_proof_policy_dagger_loss(&semantic_batch, &device, 512)
            .expect("semantic prefix policy loss");
        assert!(tensor_scalar(prefix_loss.clone()).is_finite());
        let _gradients = prefix_loss.backward();
        let prefix_content =
            std::fs::read_to_string(prefix_telemetry_path).expect("prefix telemetry sidecar");
        let prefix_value: serde_json::Value = serde_json::from_str(
            prefix_content
                .lines()
                .next()
                .expect("prefix telemetry line"),
        )
        .expect("prefix telemetry json");
        assert_eq!(prefix_value["version"], 19);
        assert_eq!(prefix_value["answer_contract"], "semantic_step");
        assert_eq!(
            prefix_value["objective"],
            "prefix_conditional_equivalent_v1"
        );
        assert!(
            prefix_value["prefix_branch_rows"]
                .as_u64()
                .unwrap_or_default()
                > 0
        );
        assert!(
            prefix_value["prefix_candidate_tokens"]
                .as_u64()
                .unwrap_or_default()
                > prefix_value["prefix_equivalent_tokens"]
                    .as_u64()
                    .unwrap_or_default()
        );
    }

    #[test]
    fn ruliad_proof_policy_batch_plan_pairs_expert_and_model_visited_rows() {
        let plan = RuliadProofPolicyBatchPlan::new(
            crate::config::RuliadProofPolicyEffectiveMode::PairedDagger,
            32,
            4,
            4,
        );
        assert_eq!(plan.static_row_budget, 16);
        assert_eq!(plan.dagger_row_budget, 16);
        assert_eq!(plan.dagger_trajectory_budget, 4);
        assert_eq!(plan.trajectory_budget(), 20);
        assert_eq!(plan.rollout_steps, 4);
        assert_eq!(
            (0..plan.dagger_trajectory_budget)
                .map(|index| plan.dagger_depth(index))
                .sum::<usize>(),
            plan.dagger_row_budget
        );

        let uneven = RuliadProofPolicyBatchPlan::new(
            crate::config::RuliadProofPolicyEffectiveMode::PairedDagger,
            10,
            4,
            2,
        );
        assert_eq!(uneven.static_row_budget, 5);
        assert_eq!(uneven.dagger_row_budget, 5);
        assert_eq!(uneven.dagger_trajectory_budget, 2);
        assert_eq!(uneven.rollout_steps, 3);
        assert_eq!(uneven.dagger_depth(0), 3);
        assert_eq!(uneven.dagger_depth(1), 2);

        let bounded_causal = RuliadProofPolicyBatchPlan::new(
            crate::config::RuliadProofPolicyEffectiveMode::PairedDagger,
            4,
            2,
            1,
        );
        assert_eq!(bounded_causal.static_row_budget, 2);
        assert_eq!(bounded_causal.dagger_row_budget, 2);
        assert_eq!(bounded_causal.dagger_trajectory_budget, 1);
        assert_eq!(bounded_causal.rollout_steps, 2);
        assert_eq!(bounded_causal.dagger_depth(0), 2);
    }

    #[test]
    fn ruliad_structured_answer_contrast_loss_scores_oracle_against_field_negatives() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 21);
        let mut config = tiny_model_config();
        config.vocab_size = 257;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_ruliad_supervision(RuliadSupervisionConfig {
                verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                    enabled: true,
                    structured_contrast_weight: 0.25,
                    structured_contrast_every_steps: 2,
                    structured_contrast_start_after_steps: 4,
                    structured_contrast_margin: 0.25,
                    structured_negative_count: 2,
                    structured_template_negative_count: 2,
                    max_completion_tokens: 24,
                    ..Default::default()
                },
                ..Default::default()
            });
        let item = burn_dragon_universality::RuliadEvalItem {
            oracle_hash: "h0".to_string(),
            sample_index: 37,
            split: burn_dragon_universality::SampleSplit::Train,
            family: "eca".to_string(),
            task_kind: "multi_step_state".to_string(),
            math_domains: vec!["computation".to_string()],
            reasoning_modes: vec!["iterated".to_string()],
            prompt: "?:eca\n!:".to_string(),
            expected_answer: "xlen=44;xalpha=01;xcounts=20,24;xedge=01".to_string(),
            difficulty_level: Some(0),
            spec: None,
        };
        let policy_batch = crate::dataset::RuliadPolicyBatch {
            samples: vec![crate::dataset::RuliadPolicySample {
                item,
                prompt_tokens: vec![1, 2, 3],
            }],
            tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
                vocab_size: 257,
                eos_id: None,
            },
            stop_token_id: None,
        };

        model.gradient_scale_step.store(3, Ordering::Relaxed);
        assert!(
            model
                .ruliad_structured_answer_contrast_loss(&policy_batch, &device, 64)
                .is_none(),
            "contrast loss should respect start_after_steps"
        );
        model.gradient_scale_step.store(5, Ordering::Relaxed);
        assert!(
            model
                .ruliad_structured_answer_contrast_loss(&policy_batch, &device, 64)
                .is_none(),
            "contrast loss should respect every_steps cadence"
        );
        model.gradient_scale_step.store(6, Ordering::Relaxed);
        let loss = model
            .ruliad_structured_answer_contrast_loss(&policy_batch, &device, 64)
            .expect("structured answer contrast loss");

        let loss = tensor_scalar(loss);
        assert!(loss.is_finite(), "contrast loss should be finite: {loss}");
        assert!(loss > 0.0, "contrast loss should be non-zero: {loss}");
    }

    #[test]
    fn ruliad_structured_answer_contrast_loss_scores_schema_negatives() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 35);
        let dir = tempfile::tempdir().expect("tempdir");
        let telemetry_path = dir
            .path()
            .join("events")
            .join("ruliad_structured_contrast.jsonl");
        let mut config = tiny_model_config();
        config.vocab_size = 257;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_ruliad_supervision(RuliadSupervisionConfig {
                verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                    enabled: true,
                    structured_contrast_weight: 0.25,
                    structured_contrast_every_steps: 1,
                    structured_negative_count: 0,
                    structured_template_negative_count: 0,
                    structured_schema_negative_count: 4,
                    max_completion_tokens: 32,
                    ..Default::default()
                },
                ..Default::default()
            })
            .with_ruliad_structured_contrast_telemetry_path(Some(telemetry_path.clone()));
        let item = burn_dragon_universality::RuliadEvalItem {
            oracle_hash: "h0".to_string(),
            sample_index: 56,
            split: burn_dragon_universality::SampleSplit::Train,
            family: "eca".to_string(),
            task_kind: "multi_step_state".to_string(),
            math_domains: vec!["computation".to_string()],
            reasoning_modes: vec!["iterated".to_string()],
            prompt: "?:eca\n!:".to_string(),
            expected_answer: "xlen=14;xalpha=01;xcounts=8,6;xedge=01".to_string(),
            difficulty_level: Some(0),
            spec: None,
        };
        let policy_batch = crate::dataset::RuliadPolicyBatch {
            samples: vec![crate::dataset::RuliadPolicySample {
                item,
                prompt_tokens: vec![1, 2, 3],
            }],
            tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
                vocab_size: 257,
                eos_id: None,
            },
            stop_token_id: None,
        };

        let loss = model
            .ruliad_structured_answer_contrast_loss(&policy_batch, &device, 64)
            .expect("schema-only structured answer contrast loss");
        assert!(tensor_scalar(loss).is_finite());
        let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
        let value: serde_json::Value =
            serde_json::from_str(content.lines().next().expect("telemetry line"))
                .expect("telemetry json");
        assert_eq!(value["field_negative_completion_rows"], 0);
        assert_eq!(value["template_negative_completion_rows"], 0);
        assert!(
            value["schema_negative_completion_rows"]
                .as_u64()
                .expect("schema rows")
                > 0
        );
        assert!(
            value["contrast_discriminative_tokens"]
                .as_u64()
                .expect("schema discriminative tokens")
                > 0
        );
    }

    #[test]
    fn ruliad_field_binding_contrast_loss_scores_prompt_counterfactuals() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 24);
        let mut config = tiny_model_config();
        config.vocab_size = 257;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_ruliad_supervision(RuliadSupervisionConfig {
                verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                    enabled: true,
                    weight: 0.0,
                    field_binding_contrast_weight: 0.25,
                    field_binding_contrast_every_steps: 2,
                    field_binding_contrast_start_after_steps: 4,
                    field_binding_contrast_margin: 0.25,
                    field_binding_contrast_pair_weight: 1.0,
                    field_binding_contrast_max_pairs: 4,
                    max_completion_tokens: 24,
                    ..Default::default()
                },
                ..Default::default()
            });
        let item_a = burn_dragon_universality::RuliadEvalItem {
            oracle_hash: "h0".to_string(),
            sample_index: 43,
            split: burn_dragon_universality::SampleSplit::Train,
            family: "formal_proof".to_string(),
            task_kind: "select_proof_action".to_string(),
            math_domains: vec!["category".to_string()],
            reasoning_modes: vec!["equational".to_string()],
            prompt: "?:a\n!:".to_string(),
            expected_answer: "g4|a:r0|f|1.1".to_string(),
            difficulty_level: Some(0),
            spec: None,
        };
        let item_b = burn_dragon_universality::RuliadEvalItem {
            oracle_hash: "h1".to_string(),
            sample_index: 44,
            split: burn_dragon_universality::SampleSplit::Train,
            family: "formal_proof".to_string(),
            task_kind: "select_proof_action".to_string(),
            math_domains: vec!["category".to_string()],
            reasoning_modes: vec!["equational".to_string()],
            prompt: "?:b\n!:".to_string(),
            expected_answer: "g7|l:3|r|0.2".to_string(),
            difficulty_level: Some(0),
            spec: None,
        };
        let policy_batch = crate::dataset::RuliadPolicyBatch {
            samples: vec![
                crate::dataset::RuliadPolicySample {
                    item: item_a,
                    prompt_tokens: vec![1, 2, 3],
                },
                crate::dataset::RuliadPolicySample {
                    item: item_b,
                    prompt_tokens: vec![1, 2, 4],
                },
            ],
            tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
                vocab_size: 257,
                eos_id: None,
            },
            stop_token_id: None,
        };

        model.gradient_scale_step.store(3, Ordering::Relaxed);
        assert!(
            model
                .ruliad_field_binding_contrast_loss(&policy_batch, &device, 64)
                .is_none(),
            "field-binding contrast should respect start_after_steps"
        );
        model.gradient_scale_step.store(5, Ordering::Relaxed);
        assert!(
            model
                .ruliad_field_binding_contrast_loss(&policy_batch, &device, 64)
                .is_none(),
            "field-binding contrast should respect every_steps cadence"
        );
        model.gradient_scale_step.store(6, Ordering::Relaxed);
        let loss = model
            .ruliad_field_binding_contrast_loss(&policy_batch, &device, 64)
            .expect("field-binding contrast loss");

        let loss = tensor_scalar(loss);
        assert!(
            loss.is_finite(),
            "field-binding contrast loss should be finite: {loss}"
        );
        assert!(
            loss > 0.0,
            "field-binding contrast loss should be non-zero: {loss}"
        );
    }

    #[test]
    fn ruliad_field_binding_contrast_loss_writes_activity_and_skip_telemetry() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 25);
        let dir = tempfile::tempdir().expect("tempdir");
        let telemetry_path = dir
            .path()
            .join("events")
            .join("ruliad_field_binding_contrast.jsonl");
        let mut config = tiny_model_config();
        config.vocab_size = 257;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_ruliad_supervision(RuliadSupervisionConfig {
                verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                    enabled: true,
                    weight: 0.0,
                    field_binding_contrast_weight: 0.25,
                    field_binding_contrast_every_steps: 1,
                    field_binding_contrast_pair_weight: 1.0,
                    field_binding_contrast_max_pairs: 2,
                    max_completion_tokens: 24,
                    ..Default::default()
                },
                ..Default::default()
            })
            .with_ruliad_field_binding_contrast_telemetry_path(Some(telemetry_path.clone()));
        let item_a = burn_dragon_universality::RuliadEvalItem {
            oracle_hash: "h0".to_string(),
            sample_index: 45,
            split: burn_dragon_universality::SampleSplit::Train,
            family: "proof_tree".to_string(),
            task_kind: "prove_theorem".to_string(),
            math_domains: vec!["category".to_string()],
            reasoning_modes: vec!["equational".to_string()],
            prompt: "?:a\n!:".to_string(),
            expected_answer: "ok=1;l=17;r=17".to_string(),
            difficulty_level: Some(0),
            spec: None,
        };
        let item_b = burn_dragon_universality::RuliadEvalItem {
            oracle_hash: "h1".to_string(),
            sample_index: 46,
            split: burn_dragon_universality::SampleSplit::Train,
            family: "proof_tree".to_string(),
            task_kind: "prove_theorem".to_string(),
            math_domains: vec!["category".to_string()],
            reasoning_modes: vec!["equational".to_string()],
            prompt: "?:b\n!:".to_string(),
            expected_answer: "ok=1;l=19;r=19".to_string(),
            difficulty_level: Some(0),
            spec: None,
        };
        let policy_batch = crate::dataset::RuliadPolicyBatch {
            samples: vec![
                crate::dataset::RuliadPolicySample {
                    item: item_a.clone(),
                    prompt_tokens: vec![1, 2, 3],
                },
                crate::dataset::RuliadPolicySample {
                    item: item_b,
                    prompt_tokens: vec![1, 2, 4],
                },
            ],
            tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
                vocab_size: 257,
                eos_id: None,
            },
            stop_token_id: None,
        };
        let loss = model
            .ruliad_field_binding_contrast_loss(&policy_batch, &device, 64)
            .expect("field-binding contrast loss");
        assert!(tensor_scalar(loss).is_finite());

        let item_c = burn_dragon_universality::RuliadEvalItem {
            oracle_hash: "h2".to_string(),
            sample_index: 47,
            split: burn_dragon_universality::SampleSplit::Train,
            family: "custom".to_string(),
            task_kind: "field_binding".to_string(),
            math_domains: vec!["category".to_string()],
            reasoning_modes: vec!["equational".to_string()],
            prompt: "?:c\n!:".to_string(),
            expected_answer: "v=17".to_string(),
            difficulty_level: Some(0),
            spec: None,
        };
        let one_sample_batch = crate::dataset::RuliadPolicyBatch {
            samples: vec![crate::dataset::RuliadPolicySample {
                item: item_c,
                prompt_tokens: vec![1, 2, 3],
            }],
            tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
                vocab_size: 257,
                eos_id: None,
            },
            stop_token_id: None,
        };
        assert!(
            model
                .ruliad_field_binding_contrast_loss(&one_sample_batch, &device, 64)
                .is_none(),
            "single oracle sample without a template schema should not produce a counterfactual pair"
        );

        let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
        let lines = content.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 2);
        let active: serde_json::Value =
            serde_json::from_str(lines[0]).expect("active telemetry json");
        assert_eq!(active["version"], 3);
        assert_eq!(active["objective"], RULIAD_FIELD_BINDING_OBJECTIVE);
        assert_eq!(active["sample_groups"], 2);
        assert_eq!(
            active["oracle_prompt_count"], 2,
            "the bounded contrast batch should cover both prompts before reusing either"
        );
        assert!(
            active["prompt_pairs"].as_u64().expect("prompt pairs") >= 2,
            "template hard negatives may add extra field-binding rows"
        );
        assert!(
            active["contrast_pairs"].as_u64().expect("contrast pairs") >= 2,
            "template hard negatives may add extra contrast pairs"
        );
        assert!(
            active["candidate_pairs"].as_u64().expect("candidate pairs") >= 2,
            "template hard negatives may add extra candidates"
        );
        assert!(
            active["negative_pool_size"]
                .as_u64()
                .expect("negative pool size")
                > 2,
            "template hard negatives should be included in the pool"
        );
        assert_eq!(active["replay_pool_size"], 0);
        assert_eq!(active["replay_contrast_pairs"], 0);
        assert!(
            active["contrast_discriminative_tokens"]
                .as_u64()
                .expect("discriminative tokens")
                > 0
        );
        assert!(
            active["rank_metric_pairs"].as_u64().expect("rank pairs") >= 2,
            "active field-binding telemetry should rank natural and/or template pairs"
        );
        assert!(
            active["rank_metric_tokens"].as_u64().expect("rank tokens") > 0,
            "active field-binding telemetry should include rank-token evidence"
        );
        let positive_fraction = active["positive_token_fraction"]
            .as_f64()
            .expect("positive token fraction");
        assert!(
            (0.0..=1.0).contains(&positive_fraction),
            "positive token fraction should be bounded: {positive_fraction}"
        );
        assert!(
            active["logit_margin_mean"]
                .as_f64()
                .expect("margin mean")
                .is_finite(),
            "rank telemetry should include a finite margin mean"
        );
        assert!(
            active["sequence_rank_metric_pairs"]
                .as_u64()
                .expect("sequence rank pairs")
                >= 2
        );
        let positive_sequence_fraction = active["positive_sequence_fraction"]
            .as_f64()
            .expect("positive sequence fraction");
        assert!((0.0..=1.0).contains(&positive_sequence_fraction));
        assert!(
            active["sequence_log_probability_margin_mean"]
                .as_f64()
                .expect("sequence log-probability margin")
                .is_finite()
        );
        let skipped: serde_json::Value =
            serde_json::from_str(lines[1]).expect("skip telemetry json");
        assert_eq!(
            skipped["skip_reason"].as_str(),
            Some("no_counterfactual_pairs")
        );
        assert_eq!(skipped["contrast_pairs"], 0);
        assert_eq!(skipped["rank_metric_tokens"], 0);
        assert_eq!(skipped["sequence_rank_metric_pairs"], 0);
        assert!(skipped["logit_margin_mean"].is_null());
    }

    #[test]
    fn ruliad_field_binding_contrast_never_uses_presented_actions_as_negatives() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 41);
        let dir = tempfile::tempdir().expect("tempdir");
        let telemetry_path = dir
            .path()
            .join("events")
            .join("ruliad_field_binding_contrast.jsonl");
        let mut model_config = tiny_model_config();
        model_config.vocab_size = 257;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(model_config, &device))
            .with_ruliad_supervision(RuliadSupervisionConfig {
                verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                    enabled: true,
                    weight: 0.0,
                    field_binding_contrast_weight: 0.25,
                    field_binding_contrast_every_steps: 1,
                    field_binding_contrast_max_pairs: 4,
                    max_completion_tokens: 64,
                    ..Default::default()
                },
                ..Default::default()
            })
            .with_ruliad_field_binding_contrast_telemetry_path(Some(telemetry_path.clone()));
        let bundle = burn_dragon_universality::ruliad::formal::generate_formal_bundle(
            41,
            burn_dragon_universality::ruliad::formal::RuliadFormalGeneratorConfig {
                rewrite_depth: 2,
                leaf_count: 3,
                context_depth: 1,
                distractor_axioms: 1,
                ..Default::default()
            },
        )
        .expect("formal bundle");
        let proof_step_index = 0;
        let actions = burn_dragon_universality::ruliad::oracle_proof_action_set(
            &bundle.problem,
            &bundle.certificate,
            proof_step_index,
            4,
        )
        .expect("oracle action set");
        let contract = burn_dragon_universality::RuliadProofActionAnswerContract::SemanticStep;
        let oracle_answer = burn_dragon_universality::ruliad::proof_action_answer(
            &actions,
            actions.selected_index,
            contract,
        )
        .expect("oracle answer");
        let distractor_index = (0..actions.candidates.len())
            .find(|index| *index != actions.selected_index)
            .expect("distractor action");
        let distractor_answer = burn_dragon_universality::ruliad::proof_action_answer(
            &actions,
            distractor_index,
            contract,
        )
        .expect("distractor answer");
        let problem_hash = bundle.problem.canonical_hash().expect("problem hash");
        let item = burn_dragon_universality::RuliadEvalItem {
            oracle_hash: problem_hash,
            sample_index: 41,
            split: burn_dragon_universality::SampleSplit::Train,
            family: "formal_proof".to_string(),
            task_kind: burn_dragon_universality::RuliadTaskKind::SelectProofAction
                .label()
                .to_string(),
            math_domains: vec!["formal_proof".to_string()],
            reasoning_modes: vec!["proof_construction".to_string()],
            prompt: burn_dragon_universality::ruliad::ruliad_proof_action_prompt(
                &bundle.problem,
                &actions,
            )
            .expect("policy prompt"),
            expected_answer: oracle_answer,
            difficulty_level: Some(0),
            spec: Some(burn_dragon_universality::RuliadSampleSpec::FormalProof {
                problem: bundle.problem,
                certificate: bundle.certificate,
                candidate: None,
                proof_step_index: Some(proof_step_index),
                action_presentation_rotation: Some(0),
                action_answer_contract: contract,
                task: burn_dragon_universality::RuliadTaskKind::SelectProofAction,
            }),
        };
        let mut distractor_item = item.clone();
        distractor_item.sample_index = 42;
        distractor_item.expected_answer = distractor_answer;
        let policy_batch = crate::dataset::RuliadPolicyBatch {
            samples: vec![
                crate::dataset::RuliadPolicySample {
                    item,
                    prompt_tokens: vec![1, 2, 3],
                },
                crate::dataset::RuliadPolicySample {
                    item: distractor_item,
                    prompt_tokens: vec![1, 2, 4],
                },
            ],
            tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
                vocab_size: 257,
                eos_id: None,
            },
            stop_token_id: None,
        };

        assert!(
            model
                .ruliad_field_binding_contrast_loss(&policy_batch, &device, 128)
                .is_none(),
            "presented distractors must not produce a negative training pair"
        );
        let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
        let value: serde_json::Value =
            serde_json::from_str(content.lines().next().expect("telemetry row"))
                .expect("telemetry json");
        assert_eq!(value["candidate_pairs"], 0);
        assert!(
            value["filtered_presented_action_candidates"]
                .as_u64()
                .expect("filtered candidates")
                >= 2
        );
    }

    #[test]
    fn ruliad_field_binding_contrast_uses_template_negatives_for_single_sample() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 33);
        let dir = tempfile::tempdir().expect("tempdir");
        let telemetry_path = dir
            .path()
            .join("events")
            .join("ruliad_field_binding_contrast.jsonl");
        let mut config = tiny_model_config();
        config.vocab_size = 257;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_ruliad_supervision(RuliadSupervisionConfig {
                verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                    enabled: true,
                    weight: 0.0,
                    field_binding_contrast_weight: 0.25,
                    field_binding_contrast_every_steps: 1,
                    field_binding_contrast_max_pairs: 4,
                    field_binding_contrast_replay_capacity: 0,
                    max_completion_tokens: 24,
                    ..Default::default()
                },
                ..Default::default()
            })
            .with_ruliad_field_binding_contrast_telemetry_path(Some(telemetry_path.clone()));
        let item = burn_dragon_universality::RuliadEvalItem {
            oracle_hash: "h0".to_string(),
            sample_index: 54,
            split: burn_dragon_universality::SampleSplit::Train,
            family: "proof_tree".to_string(),
            task_kind: "prove_theorem".to_string(),
            math_domains: vec!["category".to_string()],
            reasoning_modes: vec!["equational".to_string()],
            prompt: "?:single\n!:".to_string(),
            expected_answer: "ok=1;l=17;r=17".to_string(),
            difficulty_level: Some(0),
            spec: None,
        };
        let policy_batch = crate::dataset::RuliadPolicyBatch {
            samples: vec![crate::dataset::RuliadPolicySample {
                item,
                prompt_tokens: vec![1, 2, 3],
            }],
            tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
                vocab_size: 257,
                eos_id: None,
            },
            stop_token_id: None,
        };

        let loss = model
            .ruliad_field_binding_contrast_loss(&policy_batch, &device, 64)
            .expect("template hard negatives should provide a single-sample contrast pair");
        assert!(tensor_scalar(loss).is_finite());
        let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
        let active: serde_json::Value =
            serde_json::from_str(content.lines().next().expect("telemetry line"))
                .expect("field-binding telemetry json");
        assert_eq!(active["sample_groups"], 1);
        assert!(
            active["contrast_pairs"].as_u64().expect("contrast pairs") > 0,
            "template hard negatives should create contrast rows"
        );
        assert!(
            active["negative_pool_size"]
                .as_u64()
                .expect("negative pool size")
                > 1,
            "template hard negatives should augment the natural single-answer pool"
        );
        assert_eq!(active["replay_pool_size"], 0);
        assert_eq!(active["replay_contrast_pairs"], 0);
    }

    #[test]
    fn ruliad_field_binding_contrast_uses_schema_negatives_for_single_sample() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 34);
        let dir = tempfile::tempdir().expect("tempdir");
        let telemetry_path = dir
            .path()
            .join("events")
            .join("ruliad_field_binding_contrast.jsonl");
        let mut config = tiny_model_config();
        config.vocab_size = 257;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_ruliad_supervision(RuliadSupervisionConfig {
                verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                    enabled: true,
                    weight: 0.0,
                    field_binding_contrast_weight: 0.25,
                    field_binding_contrast_every_steps: 1,
                    field_binding_contrast_max_pairs: 4,
                    field_binding_contrast_replay_capacity: 0,
                    max_completion_tokens: 32,
                    ..Default::default()
                },
                ..Default::default()
            })
            .with_ruliad_field_binding_contrast_telemetry_path(Some(telemetry_path.clone()));
        let item = burn_dragon_universality::RuliadEvalItem {
            oracle_hash: "h0".to_string(),
            sample_index: 55,
            split: burn_dragon_universality::SampleSplit::Train,
            family: "eca".to_string(),
            task_kind: "multi_step_state".to_string(),
            math_domains: vec!["computation".to_string()],
            reasoning_modes: vec!["iterated".to_string()],
            prompt: "?:eca\n!:".to_string(),
            expected_answer: "xlen=14;xalpha=01;xcounts=8,6;xedge=01".to_string(),
            difficulty_level: Some(0),
            spec: None,
        };
        let policy_batch = crate::dataset::RuliadPolicyBatch {
            samples: vec![crate::dataset::RuliadPolicySample {
                item,
                prompt_tokens: vec![1, 2, 3],
            }],
            tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
                vocab_size: 257,
                eos_id: None,
            },
            stop_token_id: None,
        };

        let loss = model
            .ruliad_field_binding_contrast_loss(&policy_batch, &device, 64)
            .expect("schema hard negatives should provide a single-sample contrast pair");
        assert!(tensor_scalar(loss).is_finite());
        let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
        let active: serde_json::Value =
            serde_json::from_str(content.lines().next().expect("telemetry line"))
                .expect("field-binding telemetry json");
        assert_eq!(active["sample_groups"], 1);
        assert!(
            active["contrast_discriminative_tokens"]
                .as_u64()
                .expect("discriminative key tokens")
                >= 4,
            "schema negatives should activate key-token contrast"
        );
        assert!(
            active["negative_pool_size"]
                .as_u64()
                .expect("negative pool size")
                > 1,
            "schema hard negatives should augment the natural single-answer pool"
        );
    }

    #[test]
    fn ruliad_field_binding_contrast_prioritizes_prompt_coverage_over_global_byte_distance() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 32);
        let dir = tempfile::tempdir().expect("tempdir");
        let telemetry_path = dir
            .path()
            .join("events")
            .join("ruliad_field_binding_contrast.jsonl");
        let mut config = tiny_model_config();
        config.vocab_size = 257;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_ruliad_supervision(RuliadSupervisionConfig {
                verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                    enabled: true,
                    weight: 0.0,
                    field_binding_contrast_weight: 0.25,
                    field_binding_contrast_every_steps: 1,
                    field_binding_contrast_max_pairs: 1,
                    max_completion_tokens: 24,
                    ..Default::default()
                },
                ..Default::default()
            })
            .with_ruliad_field_binding_contrast_telemetry_path(Some(telemetry_path.clone()));
        let make_item = |sample_index, answer: &str| burn_dragon_universality::RuliadEvalItem {
            oracle_hash: format!("h{sample_index}"),
            sample_index,
            split: burn_dragon_universality::SampleSplit::Train,
            family: "proof_tree".to_string(),
            task_kind: "prove_theorem".to_string(),
            math_domains: vec!["category".to_string()],
            reasoning_modes: vec!["equational".to_string()],
            prompt: format!("?:sample{sample_index}\n!:"),
            expected_answer: answer.to_string(),
            difficulty_level: Some(0),
            spec: None,
        };
        let policy_batch = crate::dataset::RuliadPolicyBatch {
            samples: vec![
                crate::dataset::RuliadPolicySample {
                    item: make_item(51, "ok=1;l=17;r=17"),
                    prompt_tokens: vec![1, 2, 3],
                },
                crate::dataset::RuliadPolicySample {
                    item: make_item(52, "ok=1;l=19;r=19"),
                    prompt_tokens: vec![1, 2, 4],
                },
                crate::dataset::RuliadPolicySample {
                    item: make_item(53, "ok=0;l=00;r=00"),
                    prompt_tokens: vec![1, 2, 5],
                },
            ],
            tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
                vocab_size: 257,
                eos_id: None,
            },
            stop_token_id: None,
        };

        let loss = model
            .ruliad_field_binding_contrast_loss(&policy_batch, &device, 64)
            .expect("field-binding contrast loss");
        assert!(tensor_scalar(loss).is_finite());
        let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
        let active: serde_json::Value =
            serde_json::from_str(content.lines().next().expect("telemetry line"))
                .expect("telemetry json");

        assert_eq!(active["contrast_pairs"], 1);
        assert_eq!(
            active["contrast_discriminative_tokens"], 1,
            "the bounded pair should supervise only the causally valid first divergence"
        );
        assert_eq!(active["oracle_prompt_count"], 1);
    }

    #[test]
    fn ruliad_field_binding_contrast_uses_replay_for_single_sample_batches() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 26);
        let dir = tempfile::tempdir().expect("tempdir");
        let telemetry_path = dir
            .path()
            .join("events")
            .join("ruliad_field_binding_contrast.jsonl");
        let mut config = tiny_model_config();
        config.vocab_size = 257;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_ruliad_supervision(RuliadSupervisionConfig {
                verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                    enabled: true,
                    weight: 0.0,
                    field_binding_contrast_weight: 0.25,
                    field_binding_contrast_every_steps: 1,
                    field_binding_contrast_max_pairs: 4,
                    field_binding_contrast_replay_capacity: 4,
                    max_completion_tokens: 24,
                    ..Default::default()
                },
                ..Default::default()
            })
            .with_ruliad_field_binding_contrast_telemetry_path(Some(telemetry_path.clone()));
        let item_a = burn_dragon_universality::RuliadEvalItem {
            oracle_hash: "h0".to_string(),
            sample_index: 47,
            split: burn_dragon_universality::SampleSplit::Train,
            family: "proof_tree".to_string(),
            task_kind: "prove_theorem".to_string(),
            math_domains: vec!["category".to_string()],
            reasoning_modes: vec!["equational".to_string()],
            prompt: "?:a\n!:".to_string(),
            expected_answer: "v=17".to_string(),
            difficulty_level: Some(0),
            spec: None,
        };
        let item_b = burn_dragon_universality::RuliadEvalItem {
            oracle_hash: "h1".to_string(),
            sample_index: 48,
            split: burn_dragon_universality::SampleSplit::Train,
            family: "proof_tree".to_string(),
            task_kind: "prove_theorem".to_string(),
            math_domains: vec!["category".to_string()],
            reasoning_modes: vec!["equational".to_string()],
            prompt: "?:b\n!:".to_string(),
            expected_answer: "v=19".to_string(),
            difficulty_level: Some(0),
            spec: None,
        };
        let tokenization = burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
            vocab_size: 257,
            eos_id: None,
        };
        let first_batch = crate::dataset::RuliadPolicyBatch {
            samples: vec![crate::dataset::RuliadPolicySample {
                item: item_a,
                prompt_tokens: vec![1, 2, 3],
            }],
            tokenization: tokenization.clone(),
            stop_token_id: None,
        };
        assert!(
            model
                .ruliad_field_binding_contrast_loss(&first_batch, &device, 64)
                .is_none(),
            "first single-sample batch should fill replay but have no contrast pair"
        );

        let second_batch = crate::dataset::RuliadPolicyBatch {
            samples: vec![crate::dataset::RuliadPolicySample {
                item: item_b,
                prompt_tokens: vec![1, 2, 4],
            }],
            tokenization,
            stop_token_id: None,
        };
        let loss = model
            .ruliad_field_binding_contrast_loss(&second_batch, &device, 64)
            .expect("replay should provide a counterfactual pair");
        assert!(tensor_scalar(loss).is_finite());

        let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
        let lines = content.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 2);
        let replay_active: serde_json::Value =
            serde_json::from_str(lines[1]).expect("replay telemetry json");
        assert_eq!(replay_active["sample_groups"], 1);
        assert_eq!(replay_active["contrast_pairs"], 1);
        assert_eq!(replay_active["candidate_pairs"], 1);
        assert_eq!(replay_active["replay_pool_size"], 1);
        assert_eq!(replay_active["replay_contrast_pairs"], 1);
        assert!(
            replay_active["rank_metric_tokens"]
                .as_u64()
                .expect("rank tokens")
                > 0
        );
    }

    #[test]
    fn ruliad_generated_attractor_replay_tracks_repeated_wrong_answers() {
        let mut replay = RuliadGeneratedAttractorReplay::default();
        let key = RuliadGeneratedAttractorKey {
            family: "proof_tree".to_string(),
            task_kind: "prove_theorem".to_string(),
            contract: "ok;l;r".to_string(),
            answer: "ok=1;l=5;r=5".to_string(),
        };
        assert!(replay.record(
            key.clone(),
            burn_dragon_universality::ruliad::RuliadAnswerStatus::SchemaValidWrong,
            1,
            8,
        ));
        assert!(
            replay
                .candidates_for(RuliadGeneratedAttractorQuery {
                    family: "proof_tree",
                    task_kind: "prove_theorem",
                    expected_contract: "ok;l;r",
                    expected_answer: "ok=1;l=17;r=17",
                    min_count: 2,
                    max_candidates: 4,
                    min_distinct_answers: 1,
                    max_dominant_fraction: 1.0,
                },)
                .is_empty()
        );
        assert!(replay.record(
            key,
            burn_dragon_universality::ruliad::RuliadAnswerStatus::Partial,
            2,
            8,
        ));
        let candidates = replay.candidates_for(RuliadGeneratedAttractorQuery {
            family: "proof_tree",
            task_kind: "prove_theorem",
            expected_contract: "ok;l;r",
            expected_answer: "ok=1;l=17;r=17",
            min_count: 2,
            max_candidates: 4,
            min_distinct_answers: 1,
            max_dominant_fraction: 1.0,
        });
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].count, 2);
        assert_eq!(candidates[0].key.answer, "ok=1;l=5;r=5");
        assert!(
            replay
                .candidates_for(RuliadGeneratedAttractorQuery {
                    family: "proof_tree",
                    task_kind: "prove_theorem",
                    expected_contract: "ok;l;r",
                    expected_answer: "ok=1;l=5;r=5",
                    min_count: 2,
                    max_candidates: 4,
                    min_distinct_answers: 1,
                    max_dominant_fraction: 1.0,
                },)
                .is_empty()
        );
        let summary = replay.summary(2);
        assert_eq!(summary.pool_size, 1);
        assert_eq!(summary.active_count, 1);
        assert_eq!(summary.active_observation_count, 2);
        assert_eq!(summary.dominant_count, 2);
        assert_eq!(summary.distinct_answers, 1);
    }

    #[test]
    fn ruliad_generated_attractor_replay_requires_diverse_answers() {
        let mut replay = RuliadGeneratedAttractorReplay::default();
        let key_a = RuliadGeneratedAttractorKey {
            family: "proof_tree".to_string(),
            task_kind: "prove_theorem".to_string(),
            contract: "ok;l;r".to_string(),
            answer: "ok=1;l=5;r=5".to_string(),
        };
        let key_b = RuliadGeneratedAttractorKey {
            family: "proof_tree".to_string(),
            task_kind: "prove_theorem".to_string(),
            contract: "ok;l;r".to_string(),
            answer: "ok=1;l=9;r=9".to_string(),
        };
        for step_index in 1..=3 {
            assert!(replay.record(
                key_a.clone(),
                burn_dragon_universality::ruliad::RuliadAnswerStatus::SchemaValidWrong,
                step_index,
                8,
            ));
        }
        assert!(replay.record(
            key_b.clone(),
            burn_dragon_universality::ruliad::RuliadAnswerStatus::SchemaValidWrong,
            4,
            8,
        ));

        let dominated_summary = replay.summary(1);
        assert_eq!(
            dominated_summary.diversity_skip_reason(2, 0.5),
            Some("generated_attractor_dominant_answer")
        );
        assert!(
            replay
                .candidates_for(RuliadGeneratedAttractorQuery {
                    family: "proof_tree",
                    task_kind: "prove_theorem",
                    expected_contract: "ok;l;r",
                    expected_answer: "ok=1;l=17;r=17",
                    min_count: 1,
                    max_candidates: 4,
                    min_distinct_answers: 2,
                    max_dominant_fraction: 0.5,
                },)
                .is_empty()
        );

        for step_index in 5..=6 {
            assert!(replay.record(
                key_b.clone(),
                burn_dragon_universality::ruliad::RuliadAnswerStatus::Partial,
                step_index,
                8,
            ));
        }
        let balanced_summary = replay.summary(1);
        assert_eq!(balanced_summary.dominant_fraction(), 0.5);
        assert_eq!(balanced_summary.diversity_skip_reason(2, 0.5), None);
        let candidates = replay.candidates_for(RuliadGeneratedAttractorQuery {
            family: "proof_tree",
            task_kind: "prove_theorem",
            expected_contract: "ok;l;r",
            expected_answer: "ok=1;l=17;r=17",
            min_count: 1,
            max_candidates: 4,
            min_distinct_answers: 2,
            max_dominant_fraction: 0.5,
        });
        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn ruliad_field_binding_contrast_uses_generated_attractor_replay() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 38);
        let dir = tempfile::tempdir().expect("tempdir");
        let telemetry_path = dir
            .path()
            .join("events")
            .join("ruliad_field_binding_contrast.jsonl");
        let attractor_telemetry_path = dir
            .path()
            .join("events")
            .join("ruliad_generated_attractor_replay.jsonl");
        let mut config = tiny_model_config();
        config.vocab_size = 257;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_ruliad_supervision(RuliadSupervisionConfig {
                verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                    enabled: true,
                    weight: 0.01,
                    field_binding_contrast_weight: 0.25,
                    field_binding_contrast_every_steps: 1,
                    field_binding_contrast_max_pairs: 16,
                    field_binding_contrast_replay_capacity: 0,
                    generated_attractor_replay_capacity: 8,
                    generated_attractor_replay_min_count: 1,
                    generated_attractor_replay_max_candidates: 4,
                    generated_attractor_replay_min_distinct_answers: 1,
                    generated_attractor_replay_max_dominant_fraction: 1.0,
                    max_completion_tokens: 24,
                    ..Default::default()
                },
                ..Default::default()
            })
            .with_ruliad_field_binding_contrast_telemetry_path(Some(telemetry_path.clone()))
            .with_ruliad_generated_attractor_telemetry_path(Some(attractor_telemetry_path.clone()));
        let item = burn_dragon_universality::RuliadEvalItem {
            oracle_hash: "h0".to_string(),
            sample_index: 61,
            split: burn_dragon_universality::SampleSplit::Train,
            family: "proof_tree".to_string(),
            task_kind: "prove_theorem".to_string(),
            math_domains: vec!["category".to_string()],
            reasoning_modes: vec!["equational".to_string()],
            prompt: "?:single\n!:".to_string(),
            expected_answer: "ok=1;l=17;r=17".to_string(),
            difficulty_level: Some(0),
            spec: None,
        };
        let sample = crate::dataset::RuliadPolicySample {
            item,
            prompt_tokens: vec![1, 2, 3],
        };
        let score = burn_dragon_universality::ruliad::score_ruliad_item_completion(
            &sample.item,
            Some("ok=1;l=5;r=5\n[/R2]"),
        );
        assert!(
            model.record_ruliad_generated_attractor(&sample, "ok=1;l=5;r=5\n[/R2]", &score, 3,)
        );
        let policy_batch = crate::dataset::RuliadPolicyBatch {
            samples: vec![sample],
            tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
                vocab_size: 257,
                eos_id: None,
            },
            stop_token_id: None,
        };
        let loss = model
            .ruliad_field_binding_contrast_loss(&policy_batch, &device, 64)
            .expect("generated attractor should provide a contrast pair");
        assert!(tensor_scalar(loss).is_finite());
        let content = std::fs::read_to_string(&telemetry_path).expect("field telemetry");
        let active: serde_json::Value =
            serde_json::from_str(content.lines().next().expect("telemetry line"))
                .expect("field-binding telemetry json");
        assert_eq!(active["sample_groups"], 1);
        assert!(
            active["generated_attractor_negative_pool_size"]
                .as_u64()
                .expect("generated attractor pool")
                >= 1
        );
        assert!(
            active["generated_attractor_contrast_pairs"]
                .as_u64()
                .expect("generated attractor pairs")
                >= 1
        );
        let attractor_content =
            std::fs::read_to_string(&attractor_telemetry_path).expect("attractor telemetry");
        let replay_event: serde_json::Value =
            serde_json::from_str(attractor_content.lines().next().expect("attractor line"))
                .expect("attractor telemetry json");
        assert_eq!(replay_event["source"], "field_binding");
        assert!(
            replay_event["selected_field_binding_pairs"]
                .as_u64()
                .expect("selected field-binding pairs")
                >= 1
        );
    }

    #[test]
    fn ruliad_verifier_policy_loss_uses_generated_attractor_replay_candidates() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 39);
        let dir = tempfile::tempdir().expect("tempdir");
        let telemetry_path = dir
            .path()
            .join("events")
            .join("ruliad_verifier_policy.jsonl");
        let attractor_telemetry_path = dir
            .path()
            .join("events")
            .join("ruliad_generated_attractor_replay.jsonl");
        let mut config = tiny_model_config();
        config.vocab_size = 257;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_ruliad_supervision(RuliadSupervisionConfig {
                verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                    enabled: true,
                    mode: crate::config::train::RuliadVerifierRewardMode::VpoIndependent,
                    weight: 0.01,
                    group_size: 2,
                    every_steps: 1,
                    include_oracle_candidate: true,
                    generated_attractor_replay_capacity: 8,
                    generated_attractor_replay_min_count: 1,
                    generated_attractor_replay_max_candidates: 4,
                    generated_attractor_replay_min_distinct_answers: 1,
                    generated_attractor_replay_max_dominant_fraction: 1.0,
                    max_completion_tokens: 24,
                    ..Default::default()
                },
                ..Default::default()
            })
            .with_ruliad_policy_telemetry_path(Some(telemetry_path.clone()))
            .with_ruliad_generated_attractor_telemetry_path(Some(attractor_telemetry_path.clone()));
        let item = burn_dragon_universality::RuliadEvalItem {
            oracle_hash: "h0".to_string(),
            sample_index: 62,
            split: burn_dragon_universality::SampleSplit::Train,
            family: "proof_tree".to_string(),
            task_kind: "prove_theorem".to_string(),
            math_domains: vec!["category".to_string()],
            reasoning_modes: vec!["equational".to_string()],
            prompt: "?:single\n!:".to_string(),
            expected_answer: "ok=1;l=17;r=17".to_string(),
            difficulty_level: Some(0),
            spec: None,
        };
        let sample = crate::dataset::RuliadPolicySample {
            item,
            prompt_tokens: vec![1, 2, 3],
        };
        let score = burn_dragon_universality::ruliad::score_ruliad_item_completion(
            &sample.item,
            Some("ok=1;l=5;r=5\n[/R2]"),
        );
        assert!(
            model.record_ruliad_generated_attractor(&sample, "ok=1;l=5;r=5\n[/R2]", &score, 4,)
        );
        let policy_batch = crate::dataset::RuliadPolicyBatch {
            samples: vec![sample],
            tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
                vocab_size: 257,
                eos_id: None,
            },
            stop_token_id: None,
        };
        let loss = model
            .ruliad_verifier_policy_loss(&policy_batch, &device, 64)
            .expect("policy loss should include generated-attractor candidate");
        assert!(tensor_scalar(loss).is_finite());
        let content = std::fs::read_to_string(&telemetry_path).expect("policy telemetry");
        let active: serde_json::Value =
            serde_json::from_str(content.lines().next().expect("telemetry line"))
                .expect("policy telemetry json");
        assert!(
            active["generated_attractor_completion_rows"]
                .as_u64()
                .expect("generated attractor candidate rows")
                >= 1
        );
        let attractor_content =
            std::fs::read_to_string(&attractor_telemetry_path).expect("attractor telemetry");
        let replay_event: serde_json::Value =
            serde_json::from_str(attractor_content.lines().next().expect("attractor line"))
                .expect("attractor telemetry json");
        assert_eq!(replay_event["source"], "policy");
        assert!(
            replay_event["selected_candidate_rows"]
                .as_u64()
                .expect("selected candidates")
                >= 1
        );
    }

    #[test]
    fn ruliad_structured_answer_contrast_loss_writes_activity_telemetry() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 23);
        let dir = tempfile::tempdir().expect("tempdir");
        let telemetry_path = dir
            .path()
            .join("events")
            .join("ruliad_structured_contrast.jsonl");
        let mut config = tiny_model_config();
        config.vocab_size = 257;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_ruliad_supervision(RuliadSupervisionConfig {
                verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                    enabled: true,
                    structured_contrast_weight: 0.25,
                    structured_contrast_every_steps: 1,
                    structured_negative_count: 2,
                    structured_template_negative_count: 2,
                    structured_schema_negative_count: 2,
                    max_completion_tokens: 24,
                    ..Default::default()
                },
                ..Default::default()
            })
            .with_ruliad_structured_contrast_telemetry_path(Some(telemetry_path.clone()));
        let item = burn_dragon_universality::RuliadEvalItem {
            oracle_hash: "h0".to_string(),
            sample_index: 41,
            split: burn_dragon_universality::SampleSplit::Train,
            family: "proof_tree".to_string(),
            task_kind: "prove_theorem".to_string(),
            math_domains: vec!["category".to_string(), "formal_proof".to_string()],
            reasoning_modes: vec!["equational".to_string()],
            prompt: "?:ss\n!:".to_string(),
            expected_answer: "ok=1;l=17;r=17".to_string(),
            difficulty_level: Some(0),
            spec: None,
        };
        let policy_batch = crate::dataset::RuliadPolicyBatch {
            samples: vec![crate::dataset::RuliadPolicySample {
                item,
                prompt_tokens: vec![1, 2, 3],
            }],
            tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
                vocab_size: 257,
                eos_id: None,
            },
            stop_token_id: None,
        };

        let loss = model
            .ruliad_structured_answer_contrast_loss(&policy_batch, &device, 64)
            .expect("structured answer contrast loss");
        assert!(tensor_scalar(loss).is_finite());
        let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
        let value: serde_json::Value =
            serde_json::from_str(content.lines().next().expect("telemetry line"))
                .expect("telemetry json");
        assert_eq!(value["sample_groups"], 1);
        assert_eq!(value["oracle_completion_rows"], 1);
        assert_eq!(value["field_negative_completion_rows"], 2);
        assert_eq!(value["template_negative_completion_rows"], 2);
        assert_eq!(value["schema_negative_completion_rows"], 2);
        assert_eq!(value["contrast_pairs"], 6);
        assert!(
            value["contrast_discriminative_tokens"]
                .as_u64()
                .expect("discriminative tokens")
                > 0
        );
    }

    #[test]
    fn ruliad_verifier_policy_loss_writes_reward_telemetry() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 13);
        let dir = tempfile::tempdir().expect("tempdir");
        let telemetry_path = dir
            .path()
            .join("events")
            .join("ruliad_verifier_policy.jsonl");
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_ruliad_supervision(RuliadSupervisionConfig {
            verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                enabled: true,
                mode: crate::config::train::RuliadVerifierRewardMode::VpoIndependent,
                weight: 0.1,
                group_size: 2,
                max_completion_tokens: 2,
                every_steps: 1,
                top_k: 1,
                kl_weight: 0.0,
                vpo_scalarizations: 4,
                ..Default::default()
            },
            ..Default::default()
        })
        .with_ruliad_policy_telemetry_path(Some(telemetry_path.clone()));
        let item = burn_dragon_universality::RuliadEvalItem {
            oracle_hash: "h0".to_string(),
            sample_index: 23,
            split: burn_dragon_universality::SampleSplit::Train,
            family: "law".to_string(),
            task_kind: "category_law".to_string(),
            math_domains: vec!["category".to_string()],
            reasoning_modes: vec!["equational".to_string()],
            prompt: "?:q\n!:".to_string(),
            expected_answer: "ok=1".to_string(),
            difficulty_level: Some(0),
            spec: None,
        };
        let policy_batch = crate::dataset::RuliadPolicyBatch {
            samples: vec![crate::dataset::RuliadPolicySample {
                item,
                prompt_tokens: vec![1, 2, 3],
            }],
            tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
                vocab_size: 257,
                eos_id: None,
            },
            stop_token_id: None,
        };
        let loss = model
            .ruliad_verifier_policy_loss(&policy_batch, &device, 8)
            .expect("VPO verifier policy loss");
        assert!(tensor_scalar(loss).is_finite());
        let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
        let line = content.lines().next().expect("telemetry line");
        let value: serde_json::Value = serde_json::from_str(line).expect("telemetry json");
        assert_eq!(value["mode"], "vpo_independent");
        assert_eq!(value["scalarization_count"], 4);
        assert_eq!(value["completion_rows"], 2);
        assert!(
            value["reward_mean"]
                .as_f64()
                .expect("reward mean")
                .is_finite(),
            "reward mean should be finite"
        );
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
    fn train_step_writes_recovery_skip_telemetry_without_policy_batch() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 17);
        let dir = tempfile::tempdir().expect("tempdir");
        let telemetry_path = dir
            .path()
            .join("events")
            .join("ruliad_structured_recovery.jsonl");
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            tiny_model_config(),
            &device,
        ))
        .with_ruliad_supervision(RuliadSupervisionConfig {
            mode: RuliadSupervisionMode::AnswerCompletion,
            answer_denoising: RuliadAnswerDenoisingConfig {
                enabled: true,
                weight: 0.0,
                structured_recovery_weight: 0.25,
                structured_recovery_every_steps: 1,
                structured_recovery_start_after_steps: 0,
                structured_recovery_max_completion_tokens: 24,
                structured_recovery_negative_count: 1,
                structured_recovery_template_negative_count: 1,
                ..Default::default()
            },
            ..Default::default()
        })
        .with_ruliad_structured_recovery_telemetry_path(Some(telemetry_path.clone()));

        let loss = scalar_loss(TrainStep::step(&model, batch(&device)));
        assert!(loss.is_finite(), "unexpected train loss: {loss}");
        let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
        let event: serde_json::Value =
            serde_json::from_str(content.lines().next().expect("telemetry line"))
                .expect("telemetry json");
        assert_eq!(event["policy_batch_present"].as_bool(), Some(false));
        assert_eq!(event["skip_reason"].as_str(), Some("missing_policy_batch"));
    }

    #[test]
    fn train_step_runs_structured_recovery_with_tbptt_policy_batch() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 19);
        let dir = tempfile::tempdir().expect("tempdir");
        let telemetry_path = dir
            .path()
            .join("events")
            .join("ruliad_structured_recovery.jsonl");
        let mut config = tiny_model_config();
        config.vocab_size = 257;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_tbptt_chunk_size(Some(4))
            .with_ruliad_supervision(RuliadSupervisionConfig {
                mode: RuliadSupervisionMode::AnswerCompletion,
                answer_denoising: RuliadAnswerDenoisingConfig {
                    enabled: true,
                    weight: 0.0,
                    structured_recovery_weight: 0.25,
                    structured_recovery_every_steps: 1,
                    structured_recovery_start_after_steps: 0,
                    structured_recovery_max_completion_tokens: 24,
                    structured_recovery_negative_count: 1,
                    structured_recovery_template_negative_count: 1,
                    ..Default::default()
                },
                ..Default::default()
            })
            .with_ruliad_structured_recovery_telemetry_path(Some(telemetry_path.clone()));
        let item = burn_dragon_universality::RuliadEvalItem {
            oracle_hash: "h0".to_string(),
            sample_index: 45,
            split: burn_dragon_universality::SampleSplit::Train,
            family: "proof_tree".to_string(),
            task_kind: "prove_theorem".to_string(),
            math_domains: vec!["category".to_string(), "formal_proof".to_string()],
            reasoning_modes: vec!["equational".to_string()],
            prompt: "?:ss\n!:".to_string(),
            expected_answer: "ok=1;l=17;r=17".to_string(),
            difficulty_level: Some(0),
            spec: None,
        };
        let policy_batch = Arc::new(crate::dataset::RuliadPolicyBatch {
            samples: vec![crate::dataset::RuliadPolicySample {
                item,
                prompt_tokens: vec![1, 2, 3],
            }],
            tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
                vocab_size: 257,
                eos_id: None,
            },
            stop_token_id: None,
        });
        let train_batch = SequenceBatch::new(
            Tensor::<TestBackend, 2, Int>::from_data(
                TensorData::new(
                    vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                    [2, 8],
                ),
                &device,
            ),
            Tensor::<TestBackend, 2, Int>::from_data(
                TensorData::new(
                    vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                    [2, 8],
                ),
                &device,
            ),
            None,
        )
        .with_ruliad_policy_batch(Some(policy_batch));

        let loss = scalar_loss(TrainStep::step(&model, train_batch));
        assert!(loss.is_finite(), "unexpected train loss: {loss}");
        let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
        let event: serde_json::Value =
            serde_json::from_str(content.lines().next().expect("telemetry line"))
                .expect("telemetry json");
        assert_eq!(event["policy_batch_present"].as_bool(), Some(true));
        assert!(
            event["recovery_rows"].as_u64().unwrap_or_default() > 0,
            "expected active recovery rows, event={event}"
        );
    }

    #[test]
    fn train_step_runs_field_binding_contrast_with_tbptt_policy_batch() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 37);
        let dir = tempfile::tempdir().expect("tempdir");
        let telemetry_path = dir
            .path()
            .join("events")
            .join("ruliad_field_binding_contrast.jsonl");
        let mut config = tiny_model_config();
        config.vocab_size = 257;
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
            .with_tbptt_chunk_size(Some(8))
            .with_tbptt_persist_across_steps(true)
            .with_ruliad_supervision(RuliadSupervisionConfig {
                mode: RuliadSupervisionMode::AnswerCompletion,
                verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                    enabled: true,
                    weight: 0.0,
                    field_binding_contrast_weight: 0.25,
                    field_binding_contrast_every_steps: 1,
                    field_binding_contrast_start_after_steps: 0,
                    field_binding_contrast_max_pairs: 4,
                    field_binding_contrast_replay_capacity: 0,
                    max_completion_tokens: 24,
                    ..Default::default()
                },
                ..Default::default()
            })
            .with_ruliad_field_binding_contrast_telemetry_path(Some(telemetry_path.clone()));
        let item = burn_dragon_universality::RuliadEvalItem {
            oracle_hash: "h0".to_string(),
            sample_index: 56,
            split: burn_dragon_universality::SampleSplit::Train,
            family: "proof_tree".to_string(),
            task_kind: "prove_theorem".to_string(),
            math_domains: vec!["category".to_string(), "formal_proof".to_string()],
            reasoning_modes: vec!["equational".to_string()],
            prompt: "?:fb\n!:".to_string(),
            expected_answer: "ok=1;l=17;r=17".to_string(),
            difficulty_level: Some(0),
            spec: None,
        };
        let policy_batch = Arc::new(crate::dataset::RuliadPolicyBatch {
            samples: vec![crate::dataset::RuliadPolicySample {
                item,
                prompt_tokens: vec![1, 2, 3],
            }],
            tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
                vocab_size: 257,
                eos_id: None,
            },
            stop_token_id: None,
        });
        let inputs = (0..64)
            .map(|value| (value % 128) as i64)
            .collect::<Vec<_>>();
        let targets = (1..65)
            .map(|value| (value % 128) as i64)
            .collect::<Vec<_>>();
        let train_batch = SequenceBatch::new(
            Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(inputs, [2, 32]), &device),
            Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(targets, [2, 32]), &device),
            None,
        )
        .with_ruliad_policy_batch(Some(policy_batch));

        let loss = scalar_loss(TrainStep::step(&model, train_batch));
        assert!(loss.is_finite(), "unexpected train loss: {loss}");
        let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
        let event: serde_json::Value =
            serde_json::from_str(content.lines().next().expect("telemetry line"))
                .expect("field-binding telemetry json");
        assert_eq!(event["sample_groups"], 1);
        assert!(
            event["contrast_pairs"].as_u64().unwrap_or_default() > 0,
            "expected active field-binding contrast rows under TBPTT, event={event}"
        );
        assert!(
            event["rank_metric_tokens"].as_u64().unwrap_or_default() > 0,
            "expected field-binding rank telemetry under TBPTT, event={event}"
        );
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
