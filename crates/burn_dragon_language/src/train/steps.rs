use crate::config::train::NeuronScalingStabilizationConfig;
use crate::train::objective::{
    NextTokenLossParts, masked_token_mean_with_count, supervised_token_count,
};
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

struct IncrementalPredictiveCodingPendingBatch<B: AutodiffBackend> {
    batch: SequenceBatch<B>,
    initial_state: ModelState<B>,
    first_chunk: super::local_predictive_coding::IncrementalPredictiveCodingChunk<B>,
}

struct DkpPredictiveCodingPendingBatch<B: AutodiffBackend> {
    batch: SequenceBatch<B>,
    initial_state: ModelState<B>,
    first_chunk: super::local_predictive_coding::DkpPreparedChunk<B>,
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
    pub tbptt_credit_window_chunks: usize,
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
    incremental_predictive_coding_runtime: PipelineRuntimeCell,
    #[module(skip)]
    dkp_predictive_coding_runtime: PipelineRuntimeCell,
    #[module(skip)]
    dkp_feedback_bank: PipelineRuntimeCell,
    #[module(skip)]
    latent_reasoning: LatentReasoningTrainingConfig,
    #[module(skip)]
    next_latent_token_layout: Option<super::next_latent::NextLatentTokenLayout>,
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
    ruliad_prompt_value_binding_telemetry_path: Option<Arc<PathBuf>>,
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

struct PreparedRuliadPromptValueBindingBatch<B: Backend> {
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
    loss_mask: Tensor<B, 2, Int>,
    sample_groups: usize,
    rows: usize,
    active_tokens: usize,
    padded_tokens: usize,
}

#[derive(Clone, Debug, Serialize)]
struct RuliadPromptValueBindingTelemetry {
    version: u32,
    step_index: usize,
    algorithm: &'static str,
    prompt_context: &'static str,
    objective: &'static str,
    skip_reason: Option<&'static str>,
    sample_groups: usize,
    rows: usize,
    active_tokens: usize,
    padded_tokens: usize,
    global_backward_calls: usize,
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

const RULIAD_PROOF_POLICY_TELEMETRY_VERSION: u32 = 28;

fn ruliad_proof_policy_objective_label(
    config: &crate::config::RuliadProofPolicyTrainingConfig,
) -> &'static str {
    if config.counterfactual_objective.uses_target_group_support() {
        return match config.scoring {
            crate::config::RuliadProofPolicyScoring::CompletionLikelihood => {
                "completion_target_group_conditional_v1"
            }
            crate::config::RuliadProofPolicyScoring::SemanticEnergy => {
                "semantic_energy_target_group_conditional_v1"
            }
            crate::config::RuliadProofPolicyScoring::ResidualEnergy => {
                "residual_energy_target_group_conditional_v1"
            }
        };
    }
    match config.scoring {
        crate::config::RuliadProofPolicyScoring::SemanticEnergy => {
            if config.counterfactual_targets_per_state > 0 {
                "semantic_sequence_energy_counterfactual_v1"
            } else {
                "semantic_sequence_energy_v1"
            }
        }
        crate::config::RuliadProofPolicyScoring::ResidualEnergy => {
            if config.counterfactual_targets_per_state > 0 {
                "autoregressive_residual_energy_counterfactual_v1"
            } else {
                "autoregressive_residual_energy_v1"
            }
        }
        crate::config::RuliadProofPolicyScoring::CompletionLikelihood => {
            match config.normalization {
                crate::config::RuliadProofPolicyNormalization::CandidateConditional => {
                    if config.counterfactual_targets_per_state > 0 {
                        "candidate_normalized_counterfactual_v1"
                    } else {
                        "candidate_normalized_equivalent_v1"
                    }
                }
                crate::config::RuliadProofPolicyNormalization::PrefixConditional => {
                    if config.counterfactual_targets_per_state > 0 {
                        "prefix_conditional_counterfactual_v1"
                    } else {
                        "prefix_conditional_equivalent_v1"
                    }
                }
                crate::config::RuliadProofPolicyNormalization::VocabularyMarginal => {
                    "vocabulary_marginal_equivalent_v1"
                }
            }
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
struct RuliadProofPolicyDaggerTelemetry {
    version: u32,
    answer_contract: &'static str,
    objective: &'static str,
    prompt_context: &'static str,
    target: &'static str,
    gradient_scope: &'static str,
    presentation_risk: &'static str,
    configured_mode: &'static str,
    mode: &'static str,
    candidate_symmetry: &'static str,
    step_index: usize,
    policy_batch_fingerprint: u64,
    objective_panel_fingerprint: u64,
    consolidation_logical_epoch_index: usize,
    consolidation_logical_selection_step: usize,
    consolidation_generation_epoch_index: usize,
    consolidation_enabled: bool,
    consolidation_generation_step: usize,
    consolidation_released_unique_steps: usize,
    consolidation_novel: bool,
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
    target_group_conditional_groups: usize,
    target_group_conditional_rows: usize,
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
    original_prompt_tokens: usize,
    retained_prompt_tokens: usize,
    maximum_original_prompt_tokens: usize,
    maximum_retained_prompt_tokens: usize,
    truncated_presentations: usize,
    prompt_retention_fraction: f64,
    weight: f32,
    pub(crate) rollout_steps: usize,
    rollout_depth_reached: usize,
    configured_rollout_steps: usize,
    trajectory_budget: usize,
    semantic_row_budget: usize,
    base_semantic_row_budget: usize,
    configured_counterfactual_targets_per_state: usize,
    counterfactual_objective: &'static str,
    target_variants_per_state: usize,
    max_rows_per_update: usize,
    max_presentation_rows_per_update: usize,
}

impl RuliadProofPolicyDaggerTelemetry {
    fn with_policy_sampling(
        mut self,
        policy_batch: Option<&crate::dataset::RuliadPolicyBatch>,
    ) -> Self {
        if let Some(metadata) = policy_batch.and_then(|batch| batch.sampling_metadata) {
            self.consolidation_logical_epoch_index = metadata.logical_epoch_index;
            self.consolidation_logical_selection_step = metadata.logical_selection_step;
            self.consolidation_generation_epoch_index = metadata.generation_epoch_index;
            self.consolidation_enabled = metadata.consolidation_enabled;
            self.consolidation_generation_step = metadata.generation_step;
            self.consolidation_released_unique_steps = metadata.released_unique_steps;
            self.consolidation_novel = metadata.novel;
        }
        self
    }

    fn skipped(
        policy_batch: Option<&crate::dataset::RuliadPolicyBatch>,
        config: crate::config::RuliadProofPolicyTrainingConfig,
        step_index: usize,
        reason: impl Into<String>,
    ) -> Self {
        let effective_mode = config.effective_mode(step_index);
        let plan = RuliadProofPolicyBatchPlan::new(
            effective_mode,
            config.base_semantic_rows_per_update(),
            config.rollout_steps,
            config.stratified_difficulty_levels,
        );
        Self {
            version: RULIAD_PROOF_POLICY_TELEMETRY_VERSION,
            objective: ruliad_proof_policy_objective_label(&config),
            prompt_context: config.prompt_context.as_str(),
            target: config.target.as_str(),
            gradient_scope: config.gradient_scope.as_str(),
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
            policy_batch_fingerprint: policy_batch.map_or(0, |batch| batch.fingerprint()),
            skip_reason: Some(reason.into()),
            available_sample_groups: policy_batch.map_or(0, |batch| batch.samples.len()),
            weight: config.weight,
            rollout_steps: plan.rollout_steps_for_dagger_count(
                plan.dagger_trajectories_for_samples(
                    policy_batch.map_or(0, |batch| batch.samples.len()),
                ),
            ),
            configured_rollout_steps: config.rollout_steps,
            trajectory_budget: plan.trajectory_budget(),
            semantic_row_budget: config.semantic_rows_per_update(),
            base_semantic_row_budget: config.base_semantic_rows_per_update(),
            configured_counterfactual_targets_per_state: config.counterfactual_targets_per_state,
            counterfactual_objective: config.counterfactual_objective.as_str(),
            target_variants_per_state: config.target_variants_per_state(),
            max_rows_per_update: config.max_rows_per_update,
            max_presentation_rows_per_update: config.max_presentation_rows_per_update,
            ..Self::default()
        }
        .with_policy_sampling(policy_batch)
    }

    fn from_verifier_panel(
        stats: &crate::train::local_predictive_coding::RuliadVerifierPanelStats,
        config: crate::config::RuliadProofPolicyTrainingConfig,
        step_index: usize,
        decision_rows: usize,
    ) -> Self {
        let plan = RuliadProofPolicyBatchPlan::new(
            config.effective_mode(step_index),
            config.base_semantic_rows_per_update(),
            config.rollout_steps,
            config.stratified_difficulty_levels,
        );
        Self {
            version: RULIAD_PROOF_POLICY_TELEMETRY_VERSION,
            answer_contract: stats.answer_contract,
            objective: ruliad_proof_policy_objective_label(&config),
            prompt_context: config.prompt_context.as_str(),
            target: config.target.as_str(),
            gradient_scope: config.gradient_scope.as_str(),
            presentation_risk: "mean",
            configured_mode: stats.configured_mode,
            mode: stats.effective_mode,
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
            policy_batch_fingerprint: stats.policy_batch_fingerprint,
            objective_panel_fingerprint: stats.objective_panel_fingerprint,
            available_sample_groups: stats.available_sample_groups,
            sample_groups: stats.sample_groups,
            nonzero_start_trajectories: stats.nonzero_start_trajectories,
            mean_start_step: stats.start_step_sum as f64 / stats.sample_groups.max(1) as f64,
            visited_states: stats.semantic_states,
            semantic_state_rows: stats.semantic_states,
            base_semantic_state_rows: stats.base_semantic_states,
            counterfactual_semantic_state_rows: stats.counterfactual_semantic_states,
            counterfactual_target_shortfall: stats.counterfactual_target_shortfall,
            target_group_conditional_groups: stats.target_group_conditional_groups,
            target_group_conditional_rows: stats.target_group_conditional_rows,
            expert_rows: stats.semantic_states,
            static_expert_rows: stats.static_expert_states,
            dagger_expert_rows: stats.dagger_expert_states,
            model_visited_expert_rows: stats.model_visited_states,
            supervised_presentation_rows: decision_rows,
            mean_presentations_per_state: decision_rows as f64
                / stats.semantic_states.max(1) as f64,
            model_valid_actions: stats.model_valid_actions,
            model_invalid_actions: stats.model_invalid_actions,
            model_expert_equivalent_actions: stats.model_expert_equivalent_actions,
            model_off_expert_actions: stats.model_off_expert_actions,
            repeated_states: stats.repeated_states,
            model_backtracks: stats.backtracks,
            solved_proofs: stats.solved_proofs,
            model_scoring_batches: stats.model_scoring_batches,
            supervised_action_tokens: stats.supervised_action_tokens,
            candidate_target_tokens: stats.candidate_target_tokens,
            equivalent_target_tokens: stats.equivalent_target_tokens,
            mean_candidate_targets_per_row: stats.candidate_target_tokens as f64
                / decision_rows.max(1) as f64,
            mean_equivalent_targets_per_row: stats.equivalent_target_tokens as f64
                / decision_rows.max(1) as f64,
            prefix_branch_rows: stats.prefix_branch_rows,
            prefix_candidate_tokens: stats.prefix_candidate_tokens,
            prefix_equivalent_tokens: stats.prefix_equivalent_tokens,
            original_prompt_tokens: stats.original_prompt_tokens,
            retained_prompt_tokens: stats.retained_prompt_tokens,
            maximum_original_prompt_tokens: stats.maximum_original_prompt_tokens,
            maximum_retained_prompt_tokens: stats.maximum_retained_prompt_tokens,
            truncated_presentations: stats.truncated_presentations,
            prompt_retention_fraction: stats.retained_prompt_tokens as f64
                / stats.original_prompt_tokens.max(1) as f64,
            difficulty_sample_groups: stats.difficulty_sample_groups.clone(),
            difficulty_visited_states: stats.difficulty_visited_states.clone(),
            difficulty_expert_rows: stats.difficulty_expert_rows.clone(),
            expert_selected_index_histogram: stats.expert_selected_index_histogram.clone(),
            expert_equivalent_index_histogram: stats.expert_equivalent_index_histogram.clone(),
            model_selected_index_histogram: stats.model_selected_index_histogram.clone(),
            rollout_steps: plan.rollout_steps_for_dagger_count(
                plan.dagger_trajectories_for_samples(stats.available_sample_groups),
            ),
            rollout_depth_reached: stats.rollout_depth_reached,
            configured_rollout_steps: config.rollout_steps,
            trajectory_budget: plan.trajectory_budget(),
            semantic_row_budget: config.semantic_rows_per_update(),
            base_semantic_row_budget: config.base_semantic_rows_per_update(),
            configured_counterfactual_targets_per_state: config.counterfactual_targets_per_state,
            counterfactual_objective: config.counterfactual_objective.as_str(),
            target_variants_per_state: config.target_variants_per_state(),
            max_rows_per_update: config.max_rows_per_update,
            max_presentation_rows_per_update: config.max_presentation_rows_per_update,
            weight: config.weight,
            ..Self::default()
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RuliadProofPolicyBatchPlan {
    pub(crate) static_row_budget: usize,
    pub(crate) dagger_row_budget: usize,
    pub(crate) dagger_trajectory_budget: usize,
    dagger_base_depth: usize,
    dagger_depth_remainder: usize,
    configured_rollout_steps: usize,
    pub(crate) rollout_steps: usize,
}

impl RuliadProofPolicyBatchPlan {
    pub(crate) fn new(
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
            configured_rollout_steps,
            rollout_steps,
        }
    }

    pub(crate) fn trajectory_budget(self) -> usize {
        self.static_row_budget
            .saturating_add(self.dagger_trajectory_budget)
    }

    pub(crate) fn dagger_depth(self, trajectory_index: usize) -> usize {
        self.dagger_base_depth + usize::from(trajectory_index < self.dagger_depth_remainder)
    }

    /// Number of dynamic trajectories executable from the available source rows.
    ///
    /// Paired mode may reuse a source row once: one copy remains an expert row
    /// and the other follows the current policy. This keeps dynamic supervision
    /// executable for batch size one without inventing a second document.
    pub(crate) fn dagger_trajectories_for_samples(self, available_samples: usize) -> usize {
        self.dagger_trajectory_budget.min(available_samples)
    }

    pub(crate) fn dagger_depth_for_count(
        self,
        trajectory_index: usize,
        trajectory_count: usize,
    ) -> usize {
        if trajectory_count == 0 || trajectory_index >= trajectory_count {
            return 0;
        }
        let total_depth = self
            .dagger_row_budget
            .min(trajectory_count.saturating_mul(self.configured_rollout_steps));
        let base = total_depth / trajectory_count;
        let remainder = total_depth % trajectory_count;
        base + usize::from(trajectory_index < remainder)
    }

    pub(crate) fn rollout_steps_for_dagger_count(self, trajectory_count: usize) -> usize {
        if trajectory_count == 0 {
            1
        } else {
            self.dagger_depth_for_count(0, trajectory_count).max(1)
        }
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
    let [row_count, branch_count, vocab] = branch_logits.shape().dims::<3>();
    debug_assert_eq!(branch_count, 1);
    let candidate_mask = match normalization {
        crate::config::RuliadProofPolicyNormalization::CandidateConditional
        | crate::config::RuliadProofPolicyNormalization::PrefixConditional => {
            candidate_mask.reshape([row_count, vocab])
        }
        crate::config::RuliadProofPolicyNormalization::VocabularyMarginal => {
            Tensor::<B, 2>::ones([row_count, vocab], &branch_logits.device())
        }
    };
    burn_pc::categorical_conditional_set_log_probabilities(
        branch_logits.reshape([row_count, vocab]),
        candidate_mask,
        equivalent_mask.reshape([row_count, vocab]),
        1.0e-12,
    )
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
    support_mask: Tensor<B, 2>,
    equivalent_mask: Tensor<B, 2>,
    normalization: crate::config::RuliadProofPolicyNormalization,
) -> Tensor<B, 1> {
    match normalization {
        crate::config::RuliadProofPolicyNormalization::CandidateConditional => {
            let equivalent_scores =
                mean_log_scores.clone() + equivalent_mask.sub_scalar(1.0).mul_scalar(1.0e9);
            let support_scores = mean_log_scores + support_mask.sub_scalar(1.0).mul_scalar(1.0e9);
            sequence_logsumexp(equivalent_scores) - sequence_logsumexp(support_scores)
        }
        crate::config::RuliadProofPolicyNormalization::PrefixConditional => {
            let equivalent_scores =
                mean_log_scores.clone() + equivalent_mask.sub_scalar(1.0).mul_scalar(1.0e9);
            let support_scores = mean_log_scores + support_mask.sub_scalar(1.0).mul_scalar(1.0e9);
            sequence_logsumexp(equivalent_scores) - sequence_logsumexp(support_scores)
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
    support_mask: Tensor<B, 2>,
    equivalent_mask: Tensor<B, 2>,
    row_weights: Tensor<B, 1>,
    config: GroupedVerifierSequenceLossConfig,
) -> Tensor<B, 1> {
    let row_log_probabilities = verifier_equivalent_sequence_log_probabilities(
        mean_log_scores,
        sum_log_scores,
        support_mask,
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

fn grouped_verifier_progress_distribution_loss<B: Backend>(
    mean_log_scores: Tensor<B, 2>,
    support_mask: Tensor<B, 2>,
    target_action_weights: Tensor<B, 2>,
    row_weights: Tensor<B, 1>,
    config: GroupedVerifierSequenceLossConfig,
) -> Tensor<B, 1> {
    let row_cross_entropy = burn_pc::categorical_conditional_distribution_cross_entropy_rows(
        mean_log_scores,
        support_mask,
        target_action_weights,
        1.0e-12,
    );
    grouped_action_log_probability_loss(
        row_cross_entropy.mul_scalar(-1.0),
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

mod degeneracy;
mod latent_objectives;
mod local_pc;
mod loss_objectives;
mod model;
mod prompt_value_binding;
mod ruliad_contract;
mod ruliad_training;
mod train_step;
mod validation;
mod verifier_terminal;

use degeneracy::*;

#[cfg(test)]
mod objective_step_tests;
#[cfg(test)]
mod next_latent_tests;
