use crate::train::prelude::*;
use burn_dragon_core::ModelState;
use burn_dragon_time::Instant;
use std::any::{Any, TypeId};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
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

impl GradientScaleSchedule {
    fn from_training<B: BackendTrait>(
        model: &DragonModel<B>,
        training: &TrainingHyperparameters,
        total_steps: usize,
    ) -> Self {
        let param_scale_rules =
            Self::build_param_scale_rules(model, &training.module_lr_scales, total_steps);
        let shared_lowrank_param_ids = vec![
            model.shared_lowrank_param_ids().rwkv_time_decay,
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
}

fn scale_gradients_by_schedule<B, M>(
    module: &M,
    grads: &mut GradientsParams,
    param_scale_rules: &HashMap<burn::module::ParamId, ParamScaleScheduleRule>,
    step_index: usize,
    extra_param_ids: &HashSet<burn::module::ParamId>,
    extra_scale: Option<f32>,
) where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    let has_static_scales = param_scale_rules
        .values()
        .any(|rule| (rule.scale_for_step_index(step_index) - 1.0).abs() > f32::EPSILON);
    let has_extra_scale = extra_scale
        .is_some_and(|scale| (scale - 1.0).abs() > f32::EPSILON && !extra_param_ids.is_empty());
    if !has_static_scales && !has_extra_scale {
        return;
    }

    struct GradientScaleVisitor<'a, B: AutodiffBackend> {
        grads: &'a mut GradientsParams,
        param_scale_rules: &'a HashMap<burn::module::ParamId, ParamScaleScheduleRule>,
        step_index: usize,
        extra_param_ids: &'a HashSet<burn::module::ParamId>,
        extra_scale: Option<f32>,
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
        _marker: std::marker::PhantomData,
    };
    module.visit(&mut visitor);
}

#[derive(Debug)]
struct TeacherModelRuntime<B: BackendTrait> {
    model: DragonModel<B>,
    update_count: usize,
}

impl<B: BackendTrait> TeacherModelRuntime<B> {
    fn new(model: DragonModel<B>) -> Self {
        Self {
            model,
            update_count: 0,
        }
    }
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
        return online.clone();
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
            let require_grad = tensor.is_require_grad();
            let mut blended = (tensor.detach().mul_scalar(1.0 - self.rate)
                + online.detach().mul_scalar(self.rate))
            .detach();
            if require_grad {
                blended = blended.require_grad();
            }
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
    teacher_model: Option<DragonModel<B>>,
    #[module(skip)]
    streaming_runtime_key: usize,
    #[module(skip)]
    gradient_scale_schedule: GradientScaleSchedule,
    #[module(skip)]
    gradient_scale_step: Arc<AtomicUsize>,
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
            model,
            tbptt_chunk_size: None,
            pipeline_plan: None,
            tbptt_persist_across_steps: false,
            objective: TrainingObjectiveConfig::NextToken,
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

    pub fn with_pipeline_plan(mut self, pipeline_plan: Option<PipelinePlan>) -> Self {
        self.pipeline_plan = pipeline_plan;
        self
    }

    pub fn with_tbptt_persist_across_steps(mut self, enabled: bool) -> Self {
        self.tbptt_persist_across_steps = enabled;
        self
    }

    pub fn with_training_objective(mut self, objective: TrainingObjectiveConfig) -> Self {
        self.teacher_model = (!objective.is_next_token()).then(|| self.model.clone());
        let key = (self.streaming_runtime_key, TypeId::of::<B>());
        let mut teachers = lock_teacher_model_store();
        teachers.remove(&key);
        if let Some(teacher_model) = self.teacher_model.clone() {
            teachers.insert(key, Box::new(TeacherModelRuntime::new(teacher_model)));
        }
        self.objective = objective;
        self
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
            .and_then(|state| {
                state
                    .downcast_ref::<ModelState<B>>()
                    .map(|state| state.clone())
            })
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
    ) -> Tensor<B, 1> {
        self.model.language_loss_from_hidden(hidden, targets)
    }

    fn language_loss_from_logits(
        &self,
        logits: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
    ) -> Tensor<B, 1> {
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
        match &self.objective {
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
        }
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
                .language_loss_from_hidden(hidden.clone(), chunk_targets[microbatch_id].clone())
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
        let detail_prof_enabled = crate::train::profile::detail_enabled();
        let memory_prof_enabled = prof_enabled && crate::train::profile::memory_enabled();
        let forward_start = prof_enabled.then(Instant::now);
        let inputs = batch.inputs;
        let targets = batch.targets;
        if !self.objective.is_next_token() {
            self.update_teacher_runtime();
            let loss = self.objective_loss(inputs, targets);
            let grads = loss.backward();
            return TrainOutput {
                grads: self.apply_gradient_scale_schedule(GradientsParams::from_grads(grads, self)),
                item: LanguageModelTrainItem::new(loss),
            };
        }
        let summary_event_mask = batch.summary_event_mask;
        let reset_stream_state = batch.reset_stream_state;
        let step_device = memory_prof_enabled.then(|| inputs.device());
        let step_memory_before = step_device
            .as_ref()
            .and_then(|device| device_memory_usage_safe::<B>(device));
        let [_batch_size, block_size] = inputs.shape().dims();
        let tbptt_chunk_size = self.effective_tbptt_chunk_size(block_size);
        let factorized_head = self.model.uses_factorized_language_head();
        let probe_inputs = detail_prof_enabled.then(|| inputs.clone());
        let probe_summary_event_mask = detail_prof_enabled
            .then(|| summary_event_mask.clone())
            .flatten();
        let mut step_state = self.load_step_state(reset_stream_state);
        let (loss, probe_hidden, probe_logits, forward_ns) = if self.pipeline_enabled() {
            let forward_start = Instant::now();
            let (loss, hidden, logits) =
                self.forward_loss_with_pipeline(inputs, targets.clone(), summary_event_mask);
            step_state = self.model.init_state();
            (
                loss,
                Some(hidden),
                (!factorized_head).then_some(logits),
                forward_start.elapsed().as_nanos(),
            )
        } else if let Some(chunk_size) = tbptt_chunk_size {
            if detail_prof_enabled {
                let [batch_size, block_size] = inputs.shape().dims();
                let mut hidden_chunks = Vec::new();
                let mut logits_chunks = Vec::new();
                let mut total_forward_ns = 0u128;
                for start in (0..block_size).step_by(chunk_size) {
                    let end = (start + chunk_size).min(block_size);
                    let chunk_inputs = Self::slice_tokens(inputs.clone(), batch_size, start, end);
                    let chunk_summary_event_mask = summary_event_mask
                        .clone()
                        .map(|mask| Self::slice_tokens(mask, batch_size, start, end));
                    let chunk_forward_start = Instant::now();
                    let hidden = if let Some(mask) = chunk_summary_event_mask {
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
                    if !factorized_head {
                        logits_chunks.push(
                            self.model
                                .logits_from_hidden(hidden_chunks.last().expect("hidden").clone()),
                        );
                    }
                    if end < block_size {
                        step_state.detach_in_place();
                    }
                }
                let hidden = Tensor::cat(hidden_chunks, 1);
                let loss = self.language_loss_from_hidden(hidden.clone(), targets.clone());
                let logits = (!factorized_head).then(|| Tensor::cat(logits_chunks, 1));
                (loss, Some(hidden), logits, total_forward_ns)
            } else {
                let [batch_size, block_size] = inputs.shape().dims();
                let mut total_forward_ns = 0u128;
                let mut total_backward_ns = 0u128;
                let mut total_loss: Option<Tensor<B, 1>> = None;
                let mut accumulator = GradientsAccumulator::new();

                for start in (0..block_size).step_by(chunk_size) {
                    let end = (start + chunk_size).min(block_size);
                    let chunk_inputs = Self::slice_tokens(inputs.clone(), batch_size, start, end);
                    let chunk_targets = Self::slice_tokens(targets.clone(), batch_size, start, end);
                    let chunk_summary_event_mask = summary_event_mask
                        .clone()
                        .map(|mask| Self::slice_tokens(mask, batch_size, start, end));

                    let chunk_forward_start = Instant::now();
                    let chunk_loss = if let Some(mask) = chunk_summary_event_mask {
                        let hidden = self.model.forward_hidden_with_state_and_summary_event_mask(
                            chunk_inputs,
                            mask,
                            &mut step_state,
                        );
                        self.language_loss_from_hidden(hidden, chunk_targets.clone())
                    } else {
                        let hidden = self
                            .model
                            .forward_hidden_with_state(chunk_inputs, &mut step_state);
                        self.language_loss_from_hidden(hidden, chunk_targets.clone())
                    };
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
                let hidden = self.model.forward_hidden_with_state_and_summary_event_mask(
                    inputs,
                    summary_event_mask,
                    &mut step_state,
                );
                let forward_ns = forward_start
                    .map(|start| start.elapsed().as_nanos())
                    .unwrap_or_default();
                let loss = self.language_loss_from_hidden(hidden.clone(), targets.clone());
                let logits =
                    (!factorized_head).then(|| self.model.logits_from_hidden(hidden.clone()));
                (loss, Some(hidden), logits, forward_ns)
            } else {
                let hidden = self
                    .model
                    .forward_hidden_with_state(inputs, &mut step_state);
                let forward_ns = forward_start
                    .map(|start| start.elapsed().as_nanos())
                    .unwrap_or_default();
                let loss = self.language_loss_from_hidden(hidden.clone(), targets.clone());
                let logits =
                    (!factorized_head).then(|| self.model.logits_from_hidden(hidden.clone()));
                (loss, Some(hidden), logits, forward_ns)
            }
        } else {
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
            let loss = self.language_loss_from_hidden(hidden, targets.clone());
            (loss, None, None, forward_ns)
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
        if self.pipeline_enabled() {
            let (loss, _hidden, _logits) = self.forward_loss_with_pipeline(
                batch.inputs,
                batch.targets,
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
                    let chunk_mask =
                        Self::slice_tokens(summary_event_mask.clone(), batch_size, start, end);
                    let hidden = self.model.forward_hidden_with_state_and_summary_event_mask(
                        chunk_inputs,
                        chunk_mask,
                        &mut state,
                    );
                    let chunk_weight = (end - start) as f32 / block_size as f32;
                    let chunk_loss = self
                        .language_loss_from_hidden(hidden, chunk_targets)
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
                let loss = self.language_loss_from_hidden(hidden, batch.targets);
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
                let hidden = self
                    .model
                    .forward_hidden_with_state(chunk_inputs, &mut state);
                let chunk_weight = (end - start) as f32 / block_size as f32;
                let chunk_loss = self
                    .language_loss_from_hidden(hidden, chunk_targets)
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
            let loss = self.language_loss_from_hidden(hidden, batch.targets);
            LanguageModelOutput::new(loss)
        }
    }
}

#[cfg(test)]
mod objective_step_tests {
    use super::*;
    use burn_autodiff::Autodiff;
    use burn_ndarray::NdArray;

    type TestBackend = Autodiff<NdArray<f32>>;
    type TestInnerBackend = NdArray<f32>;

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

    fn batch(device: &<TestBackend as BackendTrait>::Device) -> SequenceBatch<TestBackend> {
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
    fn sdft_train_step_runs_rollout_objective() {
        let device = <TestBackend as BackendTrait>::Device::default();
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
    fn sdpo_train_step_runs_rollout_objective() {
        let device = <TestBackend as BackendTrait>::Device::default();
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
        let device = <TestBackend as BackendTrait>::Device::default();
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
        let device = <TestBackend as BackendTrait>::Device::default();
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
        let device = <TestBackend as BackendTrait>::Device::default();
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
        let device = <TestBackend as BackendTrait>::Device::default();
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
        let device = <TestBackend as BackendTrait>::Device::default();
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
