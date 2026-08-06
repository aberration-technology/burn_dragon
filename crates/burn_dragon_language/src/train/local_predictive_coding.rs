use std::sync::{Arc, Mutex};

use burn::module::AutodiffModule;
use burn::optim::GradientsParams;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor};
use burn_dragon_core::{
    DragonModel, DragonPredictiveCodingLayerTrace, DragonPredictiveCodingLayerVjp,
    DragonPredictiveCodingParameterIds, ModelState,
};
use burn_dragon_time::Instant;

use crate::config::{
    LocalPredictiveCodingConfig, LocalPredictiveCodingSolver, PredictiveCodingFactorReduction,
};

mod diagnostics;
pub use diagnostics::*;

#[derive(Debug)]
pub struct LocalPredictiveCodingDerivatives<B: AutodiffBackend> {
    /// Factor-local parameter derivatives on the backend's plain tensor type.
    pub grads: GradientsParams,
    /// Feed-forward masked next-token loss before activity inference.
    pub loss: Tensor<B, 1>,
    /// Raw supervised-token count for exact device-resident aggregation across
    /// truncated factors. It may be zero for an entirely masked chunk.
    pub supervised_tokens: Tensor<B, 1>,
    /// Detached recurrent state after the feed-forward factor initialization.
    /// Local activity inference is transient and never mutates this causal
    /// stream state.
    pub terminal_state: ModelState<B>,
    pub report: LocalPredictiveCodingStepReport,
}

#[derive(Debug, Default)]
pub(super) struct LocalPredictiveCodingContextMasks<B: Backend> {
    pub(super) neuron: Option<Tensor<B, 4>>,
    pub(super) activity: Option<Tensor<B, 4>>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize)]
pub struct LocalPredictiveCodingStepReport {
    pub solver: LocalPredictiveCodingSolver,
    pub inference_steps: usize,
    pub factors: usize,
    pub local_vjp_calls: usize,
    pub global_backward_calls: usize,
    pub gradient_tensors: usize,
    pub energy_before: Option<f64>,
    pub energy_after: Option<f64>,
    pub elapsed_ns: u128,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LocalPredictiveCodingProfileSnapshot {
    pub steps: u64,
    pub inference_steps: u64,
    pub factors: u64,
    pub local_vjp_calls: u64,
    pub global_backward_calls: u64,
    pub gradient_tensors: u64,
    pub elapsed_ns: u128,
    pub last_energy_before: Option<f64>,
    pub last_energy_after: Option<f64>,
}

/// Materialized, backend-independent factor graph for one Dragon block.
/// Activity nodes are inferred per batch; the token target remains clamped.
pub fn dragon_predictive_coding_graph(layers: usize) -> burn_pc::PcGraphSpec {
    assert!(
        layers > 0,
        "Dragon predictive coding requires at least one layer"
    );
    let target_id = burn_pc::PcNodeId((layers + 1).try_into().expect("layer count fits u32"));
    let mut nodes = (0..=layers)
        .map(|layer| burn_pc::PcNodeSpec {
            id: burn_pc::PcNodeId(layer.try_into().expect("layer count fits u32")),
            name: format!("activity_{layer}"),
            clamped: layer == 0,
        })
        .collect::<Vec<_>>();
    nodes.push(burn_pc::PcNodeSpec {
        id: target_id,
        name: "token_target".to_string(),
        clamped: true,
    });
    let mut factors = (0..layers)
        .map(|layer| burn_pc::PcFactorSpec {
            id: burn_pc::PcFactorId(layer.try_into().expect("layer count fits u32")),
            name: format!("dragon_layer_{layer}"),
            parents: vec![burn_pc::PcNodeId(
                layer.try_into().expect("layer count fits u32"),
            )],
            target: burn_pc::PcNodeId((layer + 1).try_into().expect("layer count fits u32")),
        })
        .collect::<Vec<_>>();
    factors.push(burn_pc::PcFactorSpec {
        id: burn_pc::PcFactorId(layers.try_into().expect("layer count fits u32")),
        name: "next_token".to_string(),
        parents: vec![burn_pc::PcNodeId(
            layers.try_into().expect("layer count fits u32"),
        )],
        target: target_id,
    });
    burn_pc::PcGraphSpec::new(nodes, factors)
}

/// Run-scoped local-PC telemetry shared by the train model and its ECS run
/// entity. Multiple pipelines in one process never contend on one global slot.
#[derive(Debug, Clone, Default)]
pub struct LocalPredictiveCodingProfile {
    inner: Arc<Mutex<LocalPredictiveCodingProfileSnapshot>>,
}

impl LocalPredictiveCodingProfile {
    pub fn reset(&self) {
        if let Ok(mut profile) = self.inner.lock() {
            *profile = LocalPredictiveCodingProfileSnapshot::default();
        }
    }

    pub fn snapshot(&self) -> LocalPredictiveCodingProfileSnapshot {
        self.inner
            .lock()
            .map(|profile| *profile)
            .unwrap_or_default()
    }

    pub fn take(&self) -> LocalPredictiveCodingProfileSnapshot {
        self.inner
            .lock()
            .map(|mut profile| std::mem::take(&mut *profile))
            .unwrap_or_default()
    }

    fn record(&self, report: LocalPredictiveCodingStepReport) {
        if let Ok(mut profile) = self.inner.lock() {
            profile.steps = profile.steps.saturating_add(1);
            profile.inference_steps = profile
                .inference_steps
                .saturating_add(report.inference_steps as u64);
            profile.factors = profile.factors.saturating_add(report.factors as u64);
            profile.local_vjp_calls = profile
                .local_vjp_calls
                .saturating_add(report.local_vjp_calls as u64);
            profile.global_backward_calls = profile
                .global_backward_calls
                .saturating_add(report.global_backward_calls as u64);
            profile.gradient_tensors = profile
                .gradient_tensors
                .saturating_add(report.gradient_tensors as u64);
            profile.elapsed_ns = profile.elapsed_ns.saturating_add(report.elapsed_ns);
            profile.last_energy_before = report.energy_before;
            profile.last_energy_after = report.energy_after;
        }
    }
}

fn factor_scale(config: &LocalPredictiveCodingConfig, factors: usize) -> f32 {
    match config.factor_reduction {
        PredictiveCodingFactorReduction::Sum => 1.0,
        PredictiveCodingFactorReduction::Mean => 1.0 / factors.max(1) as f32,
    }
}

fn prediction_error_gradient<B: Backend>(
    prediction: Tensor<B, 4>,
    activity: Tensor<B, 4>,
    precision: f32,
    scale: f32,
    normalization: Tensor<B, 1>,
) -> Tensor<B, 4> {
    (prediction - activity).mul_scalar(precision * scale) / normalization.reshape([1, 1, 1, 1])
}

fn prediction_error<B: Backend>(
    prediction: Tensor<B, 4>,
    activity: Tensor<B, 4>,
    precision: f32,
    scale: f32,
) -> Tensor<B, 4> {
    (prediction - activity).mul_scalar(precision * scale)
}

fn prediction_energy<B: Backend>(
    prediction: Tensor<B, 4>,
    activity: Tensor<B, 4>,
    precision: f32,
    normalization: Tensor<B, 1>,
) -> Tensor<B, 1> {
    (prediction - activity)
        .square()
        .sum()
        .div(normalization)
        .mul_scalar(0.5 * precision)
        .reshape([1])
}

fn apply_activity_mask<B: Backend>(
    activity: Tensor<B, 4>,
    activity_mask: Option<&Tensor<B, 4>>,
) -> Tensor<B, 4> {
    match activity_mask {
        Some(mask) => activity * mask.clone(),
        None => activity,
    }
}

fn forward_trace_batch<B: Backend>(
    model: &DragonModel<B>,
    activities: &[Tensor<B, 4>],
    initial_rhos: &[Option<Tensor<B, 4>>],
    neuron_mask: Option<&Tensor<B, 4>>,
    activity_mask: Option<&Tensor<B, 4>>,
) -> DragonPredictiveCodingLayerTrace<B>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let layers = model.predictive_coding_layer_count();
    assert_eq!(
        initial_rhos.len(),
        layers,
        "one recurrent state is required per local factor"
    );
    let inputs = Tensor::cat(activities.iter().take(layers).cloned().collect(), 0);
    let initial_rho = if initial_rhos.iter().all(Option::is_some) {
        Some(Tensor::cat(
            initial_rhos
                .iter()
                .map(|rho| rho.clone().expect("checked recurrent state presence"))
                .collect(),
            0,
        ))
    } else {
        assert!(
            initial_rhos.iter().all(Option::is_none),
            "batched local factors must agree on recurrent-state presence"
        );
        None
    };
    let mut trace = model
        .predictive_coding_forward_layer_with_recurrent_state(
            inputs,
            0,
            initial_rho,
            neuron_mask.cloned(),
            activity_mask.cloned(),
        )
        .expect("validated batched recurrent local factors");
    trace.next = apply_activity_mask(trace.next, activity_mask);
    trace
}

fn layer_parameter_vjp<B: Backend>(
    model: &DragonModel<B>,
    layer: usize,
    trace: &DragonPredictiveCodingLayerTrace<B>,
    grad_next: Tensor<B, 4>,
    neuron_mask: Option<&Tensor<B, 4>>,
    activity_mask: Option<&Tensor<B, 4>>,
) -> DragonPredictiveCodingLayerVjp<B>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let grad_next = apply_activity_mask(grad_next, activity_mask);
    match (neuron_mask, activity_mask) {
        (Some(neuron_mask), Some(activity_mask)) => model
            .predictive_coding_layer_vjp_with_subnetwork_masks(
                layer,
                trace,
                grad_next,
                neuron_mask.clone(),
                activity_mask.clone(),
            ),
        (Some(neuron_mask), None) => model.predictive_coding_layer_vjp_with_neuron_mask(
            layer,
            trace,
            grad_next,
            neuron_mask.clone(),
        ),
        (None, None) => model.predictive_coding_layer_vjp(layer, trace, grad_next),
        (None, Some(_)) => unreachable!("activity mask requires a neuron mask"),
    }
}

fn layer_activity_vjp<B: Backend>(
    model: &DragonModel<B>,
    layer: usize,
    trace: &DragonPredictiveCodingLayerTrace<B>,
    grad_next: Tensor<B, 4>,
    neuron_mask: Option<&Tensor<B, 4>>,
    activity_mask: Option<&Tensor<B, 4>>,
) -> Tensor<B, 4>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let grad_next = apply_activity_mask(grad_next, activity_mask);
    match (neuron_mask, activity_mask) {
        (Some(neuron_mask), Some(activity_mask)) => model
            .predictive_coding_layer_activity_vjp_with_subnetwork_masks(
                layer,
                trace,
                grad_next,
                neuron_mask.clone(),
                activity_mask.clone(),
            ),
        (Some(neuron_mask), None) => model.predictive_coding_layer_activity_vjp_with_neuron_mask(
            layer,
            trace,
            grad_next,
            neuron_mask.clone(),
        ),
        (None, None) => model.predictive_coding_layer_activity_vjp(layer, trace, grad_next),
        (None, Some(_)) => unreachable!("activity mask requires a neuron mask"),
    }
}

fn slice_batch<B: Backend>(tensor: Tensor<B, 4>, start: usize, end: usize) -> Tensor<B, 4> {
    let [_, axis1, axis2, axis3] = tensor.shape().dims::<4>();
    tensor.slice([start..end, 0..axis1, 0..axis2, 0..axis3])
}

fn slice_trace_batch<B: Backend>(
    trace: &DragonPredictiveCodingLayerTrace<B>,
    start: usize,
    end: usize,
) -> DragonPredictiveCodingLayerTrace<B> {
    DragonPredictiveCodingLayerTrace {
        input: slice_batch(trace.input.clone(), start, end),
        initial_rho: trace
            .initial_rho
            .clone()
            .map(|rho| slice_batch(rho, start, end)),
        attention_pre_norm: slice_batch(trace.attention_pre_norm.clone(), start, end),
        attention_readout: slice_batch(trace.attention_readout.clone(), start, end),
        residual_pre_norm: slice_batch(trace.residual_pre_norm.clone(), start, end),
        residual_delta: slice_batch(trace.residual_delta.clone(), start, end),
        x_neuron: slice_batch(trace.x_neuron.clone(), start, end),
        y_gate: slice_batch(trace.y_gate.clone(), start, end),
        y_neuron: slice_batch(trace.y_neuron.clone(), start, end),
        next: slice_batch(trace.next.clone(), start, end),
    }
}

fn concatenate_traces<B: Backend>(
    traces: &[DragonPredictiveCodingLayerTrace<B>],
) -> DragonPredictiveCodingLayerTrace<B> {
    assert!(!traces.is_empty(), "PC trace list must not be empty");
    let initial_rho = if traces.iter().all(|trace| trace.initial_rho.is_some()) {
        Some(Tensor::cat(
            traces
                .iter()
                .map(|trace| trace.initial_rho.clone().expect("checked rho presence"))
                .collect(),
            0,
        ))
    } else {
        assert!(
            traces.iter().all(|trace| trace.initial_rho.is_none()),
            "batched PC traces must agree on recurrent-state presence"
        );
        None
    };
    DragonPredictiveCodingLayerTrace {
        input: Tensor::cat(traces.iter().map(|trace| trace.input.clone()).collect(), 0),
        initial_rho,
        attention_pre_norm: Tensor::cat(
            traces
                .iter()
                .map(|trace| trace.attention_pre_norm.clone())
                .collect(),
            0,
        ),
        attention_readout: Tensor::cat(
            traces
                .iter()
                .map(|trace| trace.attention_readout.clone())
                .collect(),
            0,
        ),
        residual_pre_norm: Tensor::cat(
            traces
                .iter()
                .map(|trace| trace.residual_pre_norm.clone())
                .collect(),
            0,
        ),
        residual_delta: Tensor::cat(
            traces
                .iter()
                .map(|trace| trace.residual_delta.clone())
                .collect(),
            0,
        ),
        x_neuron: Tensor::cat(
            traces.iter().map(|trace| trace.x_neuron.clone()).collect(),
            0,
        ),
        y_gate: Tensor::cat(traces.iter().map(|trace| trace.y_gate.clone()).collect(), 0),
        y_neuron: Tensor::cat(
            traces.iter().map(|trace| trace.y_neuron.clone()).collect(),
            0,
        ),
        next: Tensor::cat(traces.iter().map(|trace| trace.next.clone()).collect(), 0),
    }
}

struct SharedParameterVjp<B: Backend> {
    grad_encoder: Tensor<B, 3>,
    grad_encoder_v: Tensor<B, 3>,
    grad_decoder: Tensor<B, 2>,
    grad_norm_gamma: Tensor<B, 1>,
    grad_norm_beta: Tensor<B, 1>,
    grad_norm_alpha: Tensor<B, 1>,
    grad_norm_shift: Tensor<B, 1>,
}

impl<B: Backend> SharedParameterVjp<B> {
    fn from_layer(vjp: DragonPredictiveCodingLayerVjp<B>) -> (Tensor<B, 4>, Self) {
        (
            vjp.grad_input,
            Self {
                grad_encoder: vjp.grad_encoder,
                grad_encoder_v: vjp.grad_encoder_v,
                grad_decoder: vjp.grad_decoder,
                grad_norm_gamma: vjp.grad_norm_gamma,
                grad_norm_beta: vjp.grad_norm_beta,
                grad_norm_alpha: vjp.grad_norm_alpha,
                grad_norm_shift: vjp.grad_norm_shift,
            },
        )
    }

    fn accumulate_layer(&mut self, vjp: DragonPredictiveCodingLayerVjp<B>) -> Tensor<B, 4> {
        self.grad_encoder = self.grad_encoder.clone() + vjp.grad_encoder;
        self.grad_encoder_v = self.grad_encoder_v.clone() + vjp.grad_encoder_v;
        self.grad_decoder = self.grad_decoder.clone() + vjp.grad_decoder;
        self.grad_norm_gamma = self.grad_norm_gamma.clone() + vjp.grad_norm_gamma;
        self.grad_norm_beta = self.grad_norm_beta.clone() + vjp.grad_norm_beta;
        self.grad_norm_alpha = self.grad_norm_alpha.clone() + vjp.grad_norm_alpha;
        self.grad_norm_shift = self.grad_norm_shift.clone() + vjp.grad_norm_shift;
        vjp.grad_input
    }
}

struct FixedPredictionContext<B: Backend> {
    parameter_ids: DragonPredictiveCodingParameterIds,
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
    loss_mask: Option<Tensor<B, 2, Int>>,
    activities: Vec<Tensor<B, 4>>,
    traces: Vec<DragonPredictiveCodingLayerTrace<B>>,
    neuron_mask: Option<Tensor<B, 4>>,
    activity_mask: Option<Tensor<B, 4>>,
    terminal_state: ModelState<B>,
    factors: usize,
    scale: f32,
}

fn fixed_prediction_train_step<B: AutodiffBackend>(
    plain: &DragonModel<B::InnerBackend>,
    context: FixedPredictionContext<B::InnerBackend>,
    started: Instant,
    profile: &LocalPredictiveCodingProfile,
) -> LocalPredictiveCodingDerivatives<B>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let FixedPredictionContext {
        parameter_ids,
        inputs,
        targets,
        loss_mask,
        activities,
        traces,
        neuron_mask,
        activity_mask,
        terminal_state,
        factors,
        scale,
    } = context;
    let terminal_activity = activities.last().expect("terminal PC activity");
    let terminal_hidden = plain.predictive_coding_hidden_from_activity(terminal_activity.clone());
    let terminal = plain.predictive_coding_head_vjp(terminal_hidden, targets, loss_mask);
    let mut grad_activity = terminal
        .grad_hidden
        .clone()
        .reshape(terminal_activity.shape())
        .mul_scalar(scale);

    let mut shared: Option<SharedParameterVjp<B::InnerBackend>> = None;
    for (layer, trace) in traces.into_iter().enumerate().rev() {
        let vjp = layer_parameter_vjp(
            plain,
            layer,
            &trace,
            grad_activity,
            neuron_mask.as_ref(),
            activity_mask.as_ref(),
        );
        grad_activity = match shared.as_mut() {
            Some(shared) => shared.accumulate_layer(vjp),
            None => {
                let (grad_input, first) = SharedParameterVjp::from_layer(vjp);
                shared = Some(first);
                grad_input
            }
        };
    }
    let shared = shared.expect("validated PC model has at least one layer");
    let grad_activity = apply_activity_mask(grad_activity, activity_mask.as_ref());
    let initial_vjp = match activity_mask.as_ref() {
        Some(mask) => plain.predictive_coding_initial_vjp_with_activity_mask(
            inputs,
            grad_activity,
            mask.clone(),
        ),
        None => plain.predictive_coding_initial_vjp(inputs, grad_activity),
    };
    let mut grads = GradientsParams::new();
    grads.register(parameter_ids.embedding, initial_vjp.grad_embedding);
    grads.register(parameter_ids.encoder, shared.grad_encoder);
    grads.register(parameter_ids.encoder_v, shared.grad_encoder_v);
    grads.register(parameter_ids.decoder, shared.grad_decoder);
    grads.register(
        parameter_ids.norm_gamma,
        shared.grad_norm_gamma + initial_vjp.grad_norm_gamma,
    );
    grads.register(
        parameter_ids.norm_beta,
        shared.grad_norm_beta + initial_vjp.grad_norm_beta,
    );
    grads.register(
        parameter_ids.norm_alpha,
        shared.grad_norm_alpha + initial_vjp.grad_norm_alpha,
    );
    grads.register(
        parameter_ids.norm_shift,
        shared.grad_norm_shift + initial_vjp.grad_norm_shift,
    );
    grads.register(
        parameter_ids.lm_head,
        terminal.grad_lm_head.mul_scalar(scale),
    );

    let report = LocalPredictiveCodingStepReport {
        solver: LocalPredictiveCodingSolver::FixedPrediction,
        inference_steps: 1,
        factors,
        local_vjp_calls: factors + 1,
        global_backward_calls: 0,
        gradient_tensors: grads.len(),
        energy_before: None,
        energy_after: None,
        elapsed_ns: started.elapsed().as_nanos(),
    };
    profile.record(report);
    LocalPredictiveCodingDerivatives {
        grads,
        loss: Tensor::<B, 1>::from_inner(terminal.loss),
        supervised_tokens: Tensor::<B, 1>::from_inner(terminal.supervised_tokens),
        terminal_state: ModelState::<B>::from_inner_cloned(terminal_state),
        report,
    }
}

struct LayerLocalPredictionContext<B: Backend> {
    parameter_ids: DragonPredictiveCodingParameterIds,
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
    loss_mask: Option<Tensor<B, 2, Int>>,
    activities: Vec<Tensor<B, 4>>,
    traces: Vec<DragonPredictiveCodingLayerTrace<B>>,
    neuron_mask: Option<Tensor<B, 4>>,
    activity_mask: Option<Tensor<B, 4>>,
    terminal_state: ModelState<B>,
}

fn layer_local_prediction_train_step<B: AutodiffBackend>(
    plain: &DragonModel<B::InnerBackend>,
    context: LayerLocalPredictionContext<B::InnerBackend>,
    started: Instant,
    profile: &LocalPredictiveCodingProfile,
) -> LocalPredictiveCodingDerivatives<B>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let LayerLocalPredictionContext {
        parameter_ids,
        inputs,
        targets,
        loss_mask,
        activities,
        traces,
        neuron_mask,
        activity_mask,
        terminal_state,
    } = context;
    let layers = traces.len();
    let [batch, streams, time, dim] = activities[0].shape().dims::<4>();
    assert_eq!(
        streams, 1,
        "layer-local prediction requires the validated single residual stream"
    );

    // Depth is folded into the batch axis. The head and shared Dragon factor
    // therefore each launch one large VJP instead of one serial VJP per layer.
    let hidden = Tensor::cat(
        activities
            .iter()
            .skip(1)
            .map(|activity| plain.predictive_coding_hidden_from_activity(activity.clone()))
            .collect(),
        0,
    );
    let repeated_targets = targets.repeat_dim(0, layers);
    let repeated_mask = loss_mask.map(|mask| mask.repeat_dim(0, layers));
    let local_head = plain.predictive_coding_head_vjp(hidden, repeated_targets, repeated_mask);
    let supervised_tokens = local_head
        .supervised_tokens
        .clone()
        .div_scalar(layers as f32);
    let terminal_loss = local_head
        .masked_token_losses
        .clone()
        .slice([(layers - 1) * batch..layers * batch, 0..time])
        .sum()
        .div(supervised_tokens.clone().clamp_min(1.0))
        .reshape([1]);
    // The batched head normalizes over `layers * supervised_tokens`. Restore
    // one full local error per shared body use, since Dragon's body parameters
    // occur once at every depth. Keep the auxiliary head derivative averaged
    // so its update scale remains invariant to the number of layer readouts.
    let grad_activities = local_head.grad_hidden.mul_scalar(layers as f32).reshape([
        layers * batch,
        streams,
        time,
        dim,
    ]);
    let batched_trace = concatenate_traces(&traces);
    let batched_vjp = layer_parameter_vjp(
        plain,
        0,
        &batched_trace,
        grad_activities,
        neuron_mask.as_ref(),
        activity_mask.as_ref(),
    );
    let initial_grad = apply_activity_mask(
        slice_batch(batched_vjp.grad_input, 0, batch),
        activity_mask.as_ref(),
    );
    let initial_vjp = match activity_mask.as_ref() {
        Some(mask) => plain.predictive_coding_initial_vjp_with_activity_mask(
            inputs,
            initial_grad,
            mask.clone(),
        ),
        None => plain.predictive_coding_initial_vjp(inputs, initial_grad),
    };

    let mut grads = GradientsParams::new();
    grads.register(parameter_ids.embedding, initial_vjp.grad_embedding);
    grads.register(parameter_ids.encoder, batched_vjp.grad_encoder);
    grads.register(parameter_ids.encoder_v, batched_vjp.grad_encoder_v);
    grads.register(parameter_ids.decoder, batched_vjp.grad_decoder);
    grads.register(
        parameter_ids.norm_gamma,
        batched_vjp.grad_norm_gamma + initial_vjp.grad_norm_gamma,
    );
    grads.register(
        parameter_ids.norm_beta,
        batched_vjp.grad_norm_beta + initial_vjp.grad_norm_beta,
    );
    grads.register(
        parameter_ids.norm_alpha,
        batched_vjp.grad_norm_alpha + initial_vjp.grad_norm_alpha,
    );
    grads.register(
        parameter_ids.norm_shift,
        batched_vjp.grad_norm_shift + initial_vjp.grad_norm_shift,
    );
    grads.register(parameter_ids.lm_head, local_head.grad_lm_head);

    let report = LocalPredictiveCodingStepReport {
        solver: LocalPredictiveCodingSolver::LayerLocalPrediction,
        inference_steps: 1,
        factors: layers * 2,
        local_vjp_calls: 3,
        global_backward_calls: 0,
        gradient_tensors: grads.len(),
        energy_before: None,
        energy_after: None,
        elapsed_ns: started.elapsed().as_nanos(),
    };
    profile.record(report);
    LocalPredictiveCodingDerivatives {
        grads,
        loss: Tensor::<B, 1>::from_inner(terminal_loss),
        supervised_tokens: Tensor::<B, 1>::from_inner(supervised_tokens),
        terminal_state: ModelState::<B>::from_inner_cloned(terminal_state),
        report,
    }
}

fn total_energy<B: Backend>(
    model: &DragonModel<B>,
    activities: &[Tensor<B, 4>],
    initial_rhos: &[Option<Tensor<B, 4>>],
    targets: Tensor<B, 2, Int>,
    loss_mask: Option<Tensor<B, 2, Int>>,
    config: &LocalPredictiveCodingConfig,
    context_masks: (Option<&Tensor<B, 4>>, Option<&Tensor<B, 4>>),
) -> Tensor<B, 1>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let (neuron_mask, activity_mask) = context_masks;
    let trace = forward_trace_batch(model, activities, initial_rhos, neuron_mask, activity_mask);
    let predicted = trace.next;
    let inferred = Tensor::cat(activities.iter().skip(1).cloned().collect(), 0);
    let hidden = model.predictive_coding_hidden_from_activity(
        activities.last().expect("terminal PC activity").clone(),
    );
    let terminal = model.predictive_coding_head_activity_vjp(hidden, targets, loss_mask);
    let mut energy = prediction_energy(
        predicted,
        inferred,
        config.prediction_precision,
        terminal.normalization.clone(),
    );
    energy = energy + terminal.loss;
    energy.mul_scalar(factor_scale(
        config,
        model.predictive_coding_layer_count() + 1,
    ))
}

pub(crate) fn local_predictive_coding_train_step<B: AutodiffBackend>(
    model: &DragonModel<B>,
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
    loss_mask: Option<Tensor<B, 2, Int>>,
    config: &LocalPredictiveCodingConfig,
    profile: &LocalPredictiveCodingProfile,
) -> LocalPredictiveCodingDerivatives<B>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    local_predictive_coding_train_step_with_context_masks(
        model,
        inputs,
        targets,
        loss_mask,
        LocalPredictiveCodingContextMasks::default(),
        config,
        profile,
    )
}

pub(crate) fn local_predictive_coding_train_step_with_state<B: AutodiffBackend>(
    model: &DragonModel<B>,
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
    loss_mask: Option<Tensor<B, 2, Int>>,
    initial_state: ModelState<B>,
    config: &LocalPredictiveCodingConfig,
    profile: &LocalPredictiveCodingProfile,
) -> LocalPredictiveCodingDerivatives<B>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    local_predictive_coding_train_step_with_state_and_context_masks(
        model,
        inputs,
        targets,
        loss_mask,
        Some(initial_state),
        LocalPredictiveCodingContextMasks::default(),
        config,
        profile,
    )
}

fn local_predictive_coding_train_step_with_context_masks<B: AutodiffBackend>(
    model: &DragonModel<B>,
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
    loss_mask: Option<Tensor<B, 2, Int>>,
    context_masks: LocalPredictiveCodingContextMasks<B>,
    config: &LocalPredictiveCodingConfig,
    profile: &LocalPredictiveCodingProfile,
) -> LocalPredictiveCodingDerivatives<B>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    local_predictive_coding_train_step_with_state_and_context_masks(
        model,
        inputs,
        targets,
        loss_mask,
        None,
        context_masks,
        config,
        profile,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn local_predictive_coding_train_step_with_state_and_context_masks<B: AutodiffBackend>(
    model: &DragonModel<B>,
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
    loss_mask: Option<Tensor<B, 2, Int>>,
    initial_state: Option<ModelState<B>>,
    context_masks: LocalPredictiveCodingContextMasks<B>,
    config: &LocalPredictiveCodingConfig,
    profile: &LocalPredictiveCodingProfile,
) -> LocalPredictiveCodingDerivatives<B>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let started = Instant::now();
    let parameter_ids = model
        .predictive_coding_parameter_ids()
        .expect("validated local predictive-coding model");
    let plain = model.valid();
    plain
        .predictive_coding_support()
        .expect("validated plain local predictive-coding model");
    let mut terminal_state = initial_state
        .map(|state| state.inner_cloned())
        .unwrap_or_else(|| plain.init_state_ephemeral());
    assert_eq!(
        terminal_state.layers.len(),
        plain.predictive_coding_layer_count(),
        "local PC recurrent-state layer count must match the model"
    );
    let block_time = inputs.shape().dims::<2>()[1];
    let inputs = inputs.inner();
    let targets = targets.inner();
    let loss_mask = loss_mask.map(Tensor::inner);
    let neuron_mask = context_masks.neuron.map(Tensor::inner);
    let activity_mask = context_masks.activity.map(Tensor::inner);
    let initial = match activity_mask.as_ref() {
        Some(mask) => plain
            .predictive_coding_initial_activity_with_activity_mask(inputs.clone(), mask.clone()),
        None => plain.predictive_coding_initial_activity(inputs.clone()),
    };
    let layers = plain.predictive_coding_layer_count();
    let graph = dragon_predictive_coding_graph(layers);
    debug_assert!(graph.validate().is_ok());
    let factors = layers + 1;
    let scale = factor_scale(config, factors);

    let mut activities = Vec::with_capacity(layers + 1);
    let mut feedforward_traces = Vec::with_capacity(layers);
    activities.push(initial);
    for layer in 0..layers {
        let mut trace = plain
            .predictive_coding_forward_layer_with_recurrent_state(
                activities[layer].clone(),
                layer,
                terminal_state.layers[layer].rho.clone(),
                neuron_mask.clone(),
                activity_mask.clone(),
            )
            .expect("validated recurrent local PC layer factor");
        terminal_state.layers[layer].rho = Some(plain.predictive_coding_terminal_rho(&trace));
        terminal_state.layers[layer].rho_norm = None;
        terminal_state.layers[layer].sequence_aux = None;
        trace.next = apply_activity_mask(trace.next, activity_mask.as_ref());
        activities.push(trace.next.clone().detach());
        feedforward_traces.push(trace);
    }
    terminal_state.position = terminal_state.position.saturating_add(block_time);
    terminal_state.detach_in_place();

    if matches!(config.solver, LocalPredictiveCodingSolver::FixedPrediction) {
        return fixed_prediction_train_step::<B>(
            &plain,
            FixedPredictionContext {
                parameter_ids,
                inputs,
                targets,
                loss_mask,
                activities,
                traces: feedforward_traces,
                neuron_mask,
                activity_mask,
                terminal_state,
                factors,
                scale,
            },
            started,
            profile,
        );
    }
    if matches!(
        config.solver,
        LocalPredictiveCodingSolver::LayerLocalPrediction
    ) {
        return layer_local_prediction_train_step::<B>(
            &plain,
            LayerLocalPredictionContext {
                parameter_ids,
                inputs,
                targets,
                loss_mask,
                activities,
                traces: feedforward_traces,
                neuron_mask,
                activity_mask,
                terminal_state,
            },
            started,
            profile,
        );
    }
    let factor_initial_rhos = feedforward_traces
        .iter()
        .map(|trace| trace.initial_rho.clone())
        .collect::<Vec<_>>();
    let energy_before = config.sync_diagnostics.then(|| {
        burn_pc::diagnostic_scalar_f32(total_energy(
            &plain,
            &activities,
            &factor_initial_rhos,
            targets.clone(),
            loss_mask.clone(),
            config,
            (neuron_mask.as_ref(), activity_mask.as_ref()),
        )) as f64
    });

    let mut local_vjp_calls = 0usize;
    let mut feedforward_loss = None;
    let mut feedforward_trace = Some(concatenate_traces(&feedforward_traces));
    drop(feedforward_traces);
    match config.solver {
        LocalPredictiveCodingSolver::SynchronousEquilibrium => {
            for _ in 0..config.inference.steps {
                let trace = feedforward_trace.take().unwrap_or_else(|| {
                    forward_trace_batch(
                        &plain,
                        &activities,
                        &factor_initial_rhos,
                        neuron_mask.as_ref(),
                        activity_mask.as_ref(),
                    )
                });
                let terminal_hidden = plain.predictive_coding_hidden_from_activity(
                    activities.last().expect("terminal PC activity").clone(),
                );
                let terminal = plain.predictive_coding_head_activity_vjp(
                    terminal_hidden,
                    targets.clone(),
                    loss_mask.clone(),
                );
                if feedforward_loss.is_none() {
                    feedforward_loss = Some(terminal.loss.clone());
                }
                local_vjp_calls = local_vjp_calls.saturating_add(1);
                let terminal_grad = (terminal.grad_hidden
                    * terminal.normalization.reshape([1, 1, 1]))
                .reshape(activities.last().expect("terminal PC activity").shape());
                let [batch, streams, time, dim] = activities[0].shape().dims::<4>();
                let inferred = Tensor::cat(activities.iter().skip(1).cloned().collect(), 0);
                let errors = prediction_error(
                    trace.next.clone(),
                    inferred,
                    config.prediction_precision,
                    scale,
                );
                let internal_child_grads = (layers > 1).then(|| {
                    layer_activity_vjp(
                        &plain,
                        0,
                        &slice_trace_batch(&trace, batch, layers * batch),
                        slice_batch(errors.clone(), batch, layers * batch),
                        neuron_mask.as_ref(),
                        activity_mask.as_ref(),
                    )
                });
                local_vjp_calls = local_vjp_calls.saturating_add(layers.saturating_sub(1));

                let mut updates = Vec::with_capacity(layers);
                for (activity_index, activity) in
                    activities.iter().enumerate().take(layers + 1).skip(1)
                {
                    let own_offset = (activity_index - 1) * batch;
                    let own = slice_batch(errors.clone(), own_offset, own_offset + batch)
                        .mul_scalar(-1.0);
                    let child = if activity_index == layers {
                        terminal_grad.clone().mul_scalar(scale)
                    } else {
                        let offset = (activity_index - 1) * batch;
                        internal_child_grads
                            .as_ref()
                            .expect("non-terminal PC activity has a child factor")
                            .clone()
                            .slice([offset..offset + batch, 0..streams, 0..time, 0..dim])
                    };
                    updates.push(burn_pc::pc_sgd_update(
                        activity.clone(),
                        own + child,
                        &config.inference,
                    ));
                }
                for (activity, update) in activities.iter_mut().skip(1).zip(updates) {
                    *activity = apply_activity_mask(update, activity_mask.as_ref()).detach();
                }
            }
        }
        LocalPredictiveCodingSolver::ReverseGaussSeidel => {
            for _ in 0..config.inference.steps {
                // Predictions are held fixed within one reverse sweep. Since
                // each factor's parent has not yet been updated, this is a
                // block Gauss-Seidel solve rather than a stale approximation.
                let trace = feedforward_trace.take().unwrap_or_else(|| {
                    forward_trace_batch(
                        &plain,
                        &activities,
                        &factor_initial_rhos,
                        neuron_mask.as_ref(),
                        activity_mask.as_ref(),
                    )
                });
                let terminal_hidden = plain.predictive_coding_hidden_from_activity(
                    activities.last().expect("terminal PC activity").clone(),
                );
                let terminal = plain.predictive_coding_head_activity_vjp(
                    terminal_hidden,
                    targets.clone(),
                    loss_mask.clone(),
                );
                if feedforward_loss.is_none() {
                    feedforward_loss = Some(terminal.loss.clone());
                }
                local_vjp_calls = local_vjp_calls.saturating_add(1);
                let terminal_grad = (terminal.grad_hidden
                    * terminal.normalization.reshape([1, 1, 1]))
                .reshape(activities.last().expect("terminal PC activity").shape());
                let [batch, _, _, _] = activities[0].shape().dims::<4>();

                for activity_index in (1..=layers).rev() {
                    let own_offset = (activity_index - 1) * batch;
                    let own = prediction_error(
                        slice_batch(trace.next.clone(), own_offset, own_offset + batch),
                        activities[activity_index].clone(),
                        config.prediction_precision,
                        scale,
                    )
                    .mul_scalar(-1.0);
                    let child = if activity_index == layers {
                        terminal_grad.clone().mul_scalar(scale)
                    } else {
                        let child_offset = activity_index * batch;
                        let child_error = prediction_error(
                            slice_batch(trace.next.clone(), child_offset, child_offset + batch),
                            activities[activity_index + 1].clone(),
                            config.prediction_precision,
                            scale,
                        );
                        local_vjp_calls = local_vjp_calls.saturating_add(1);
                        layer_activity_vjp(
                            &plain,
                            activity_index,
                            &slice_trace_batch(&trace, child_offset, child_offset + batch),
                            child_error,
                            neuron_mask.as_ref(),
                            activity_mask.as_ref(),
                        )
                    };
                    activities[activity_index] = apply_activity_mask(
                        burn_pc::pc_sgd_update(
                            activities[activity_index].clone(),
                            own + child,
                            &config.inference,
                        ),
                        activity_mask.as_ref(),
                    )
                    .detach();
                }
            }
        }
        LocalPredictiveCodingSolver::FixedPrediction => {
            unreachable!("fixed-prediction solver returns before activity inference")
        }
        LocalPredictiveCodingSolver::LayerLocalPrediction => {
            unreachable!("layer-local solver returns before activity inference")
        }
    }

    let energy_after = config.sync_diagnostics.then(|| {
        burn_pc::diagnostic_scalar_f32(total_energy(
            &plain,
            &activities,
            &factor_initial_rhos,
            targets.clone(),
            loss_mask.clone(),
            config,
            (neuron_mask.as_ref(), activity_mask.as_ref()),
        )) as f64
    });

    let trace = forward_trace_batch(
        &plain,
        &activities,
        &factor_initial_rhos,
        neuron_mask.as_ref(),
        activity_mask.as_ref(),
    );
    let terminal_hidden = plain.predictive_coding_hidden_from_activity(
        activities.last().expect("terminal PC activity").clone(),
    );
    let terminal = plain.predictive_coding_head_vjp(terminal_hidden, targets, loss_mask.clone());
    let normalization = terminal.supervised_tokens.clone().clamp_min(1.0);
    let errors = prediction_error_gradient(
        trace.next.clone(),
        Tensor::cat(activities.iter().skip(1).cloned().collect(), 0),
        config.prediction_precision,
        scale,
        normalization,
    );
    let batched_vjp = layer_parameter_vjp(
        &plain,
        0,
        &trace,
        errors,
        neuron_mask.as_ref(),
        activity_mask.as_ref(),
    );
    local_vjp_calls = local_vjp_calls.saturating_add(layers);
    let [batch, streams, time, dim] = activities[0].shape().dims::<4>();
    let initial_grad = apply_activity_mask(
        batched_vjp
            .grad_input
            .slice([0..batch, 0..streams, 0..time, 0..dim]),
        activity_mask.as_ref(),
    );
    let initial_vjp = match activity_mask.as_ref() {
        Some(mask) => plain.predictive_coding_initial_vjp_with_activity_mask(
            inputs,
            initial_grad,
            mask.clone(),
        ),
        None => plain.predictive_coding_initial_vjp(inputs, initial_grad),
    };
    local_vjp_calls = local_vjp_calls.saturating_add(1);
    let grad_norm_gamma = batched_vjp.grad_norm_gamma + initial_vjp.grad_norm_gamma;
    let grad_norm_beta = batched_vjp.grad_norm_beta + initial_vjp.grad_norm_beta;
    let grad_norm_alpha = batched_vjp.grad_norm_alpha + initial_vjp.grad_norm_alpha;
    let grad_norm_shift = batched_vjp.grad_norm_shift + initial_vjp.grad_norm_shift;
    // Report the ordinary feed-forward CE from the initialized activities.
    // The settled terminal CE is an inference diagnostic and is not directly
    // comparable with the backpropagation baseline's train metric.
    let loss = Tensor::<B, 1>::from_inner(
        feedforward_loss.expect("validated local PC runs at least one inference step"),
    );
    let mut grads = GradientsParams::new();
    grads.register(parameter_ids.embedding, initial_vjp.grad_embedding);
    grads.register(parameter_ids.encoder, batched_vjp.grad_encoder);
    grads.register(parameter_ids.encoder_v, batched_vjp.grad_encoder_v);
    grads.register(parameter_ids.decoder, batched_vjp.grad_decoder);
    grads.register(parameter_ids.norm_gamma, grad_norm_gamma);
    grads.register(parameter_ids.norm_beta, grad_norm_beta);
    grads.register(parameter_ids.norm_alpha, grad_norm_alpha);
    grads.register(parameter_ids.norm_shift, grad_norm_shift);
    grads.register(
        parameter_ids.lm_head,
        terminal.grad_lm_head.mul_scalar(scale),
    );
    local_vjp_calls = local_vjp_calls.saturating_add(1);

    let report = LocalPredictiveCodingStepReport {
        solver: config.solver,
        inference_steps: config.inference.steps,
        factors,
        local_vjp_calls,
        global_backward_calls: 0,
        gradient_tensors: grads.len(),
        energy_before,
        energy_after,
        elapsed_ns: started.elapsed().as_nanos(),
    };
    profile.record(report);
    LocalPredictiveCodingDerivatives {
        grads,
        loss,
        supervised_tokens: Tensor::<B, 1>::from_inner(terminal.supervised_tokens),
        terminal_state: ModelState::<B>::from_inner_cloned(terminal_state),
        report,
    }
}

/// Produce one canonical Dragon predictive-coding derivative step without a
/// global backward pass or run-scoped telemetry side effects.
///
/// This is the public experiment/integration boundary. Production training
/// uses the same implementation with an entity-scoped profile sink.
pub fn local_predictive_coding_derivatives<B: AutodiffBackend>(
    model: &DragonModel<B>,
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
    loss_mask: Option<Tensor<B, 2, Int>>,
    config: &LocalPredictiveCodingConfig,
) -> Result<LocalPredictiveCodingDerivatives<B>, String>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    validate_local_predictive_coding_derivative_request(
        model,
        &inputs,
        &targets,
        loss_mask.as_ref(),
        config,
    )?;
    Ok(local_predictive_coding_train_step(
        model,
        inputs,
        targets,
        loss_mask,
        config,
        &LocalPredictiveCodingProfile::default(),
    ))
}

/// Produce local derivatives for one truncated recurrent factor graph.
///
/// Incoming rho is treated as a clamped, detached parent. The returned state
/// can be fed into the next chunk; its derivative is intentionally not
/// propagated across this API boundary, matching canonical TBPTT truncation.
pub fn local_predictive_coding_derivatives_with_state<B: AutodiffBackend>(
    model: &DragonModel<B>,
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
    loss_mask: Option<Tensor<B, 2, Int>>,
    initial_state: ModelState<B>,
    config: &LocalPredictiveCodingConfig,
) -> Result<LocalPredictiveCodingDerivatives<B>, String>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    validate_local_predictive_coding_derivative_request(
        model,
        &inputs,
        &targets,
        loss_mask.as_ref(),
        config,
    )?;
    validate_local_predictive_coding_state(model, &inputs, &initial_state)?;
    Ok(
        local_predictive_coding_train_step_with_state_and_context_masks(
            model,
            inputs,
            targets,
            loss_mask,
            Some(initial_state),
            LocalPredictiveCodingContextMasks::default(),
            config,
            &LocalPredictiveCodingProfile::default(),
        ),
    )
}

/// Produce factor-local derivatives under a fixed context-competition neuron mask.
/// This is the explicit oracle-context upper bound used to separate the effect
/// of prospective configuration from sparse neuron-space routing.
pub fn local_predictive_coding_derivatives_with_neuron_mask<B: AutodiffBackend>(
    model: &DragonModel<B>,
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
    loss_mask: Option<Tensor<B, 2, Int>>,
    neuron_mask: Tensor<B, 4>,
    config: &LocalPredictiveCodingConfig,
) -> Result<LocalPredictiveCodingDerivatives<B>, String>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    validate_local_predictive_coding_derivative_request(
        model,
        &inputs,
        &targets,
        loss_mask.as_ref(),
        config,
    )?;
    model.predictive_coding_validate_neuron_mask(&neuron_mask)?;
    Ok(local_predictive_coding_train_step_with_context_masks(
        model,
        inputs,
        targets,
        loss_mask,
        LocalPredictiveCodingContextMasks {
            neuron: Some(neuron_mask),
            activity: None,
        },
        config,
        &LocalPredictiveCodingProfile::default(),
    ))
}

/// Produce factor-local derivatives under a context-selected Dragon
/// subnetwork. The rho mask gates low-rank neuron channels and the activity
/// mask gates residual-state channels at every recurrent layer boundary.
pub fn local_predictive_coding_derivatives_with_subnetwork_masks<B: AutodiffBackend>(
    model: &DragonModel<B>,
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
    loss_mask: Option<Tensor<B, 2, Int>>,
    neuron_mask: Tensor<B, 4>,
    activity_mask: Tensor<B, 4>,
    config: &LocalPredictiveCodingConfig,
) -> Result<LocalPredictiveCodingDerivatives<B>, String>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    validate_local_predictive_coding_derivative_request(
        model,
        &inputs,
        &targets,
        loss_mask.as_ref(),
        config,
    )?;
    model.predictive_coding_validate_neuron_mask(&neuron_mask)?;
    model.predictive_coding_validate_activity_mask(&activity_mask)?;
    Ok(local_predictive_coding_train_step_with_context_masks(
        model,
        inputs,
        targets,
        loss_mask,
        LocalPredictiveCodingContextMasks {
            neuron: Some(neuron_mask),
            activity: Some(activity_mask),
        },
        config,
        &LocalPredictiveCodingProfile::default(),
    ))
}

/// Recurrent counterpart of
/// [`local_predictive_coding_derivatives_with_subnetwork_masks`].
#[allow(clippy::too_many_arguments)]
pub fn local_predictive_coding_derivatives_with_state_and_subnetwork_masks<B: AutodiffBackend>(
    model: &DragonModel<B>,
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
    loss_mask: Option<Tensor<B, 2, Int>>,
    initial_state: ModelState<B>,
    neuron_mask: Tensor<B, 4>,
    activity_mask: Tensor<B, 4>,
    config: &LocalPredictiveCodingConfig,
) -> Result<LocalPredictiveCodingDerivatives<B>, String>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    validate_local_predictive_coding_derivative_request(
        model,
        &inputs,
        &targets,
        loss_mask.as_ref(),
        config,
    )?;
    validate_local_predictive_coding_state(model, &inputs, &initial_state)?;
    model.predictive_coding_validate_neuron_mask(&neuron_mask)?;
    model.predictive_coding_validate_activity_mask(&activity_mask)?;
    Ok(
        local_predictive_coding_train_step_with_state_and_context_masks(
            model,
            inputs,
            targets,
            loss_mask,
            Some(initial_state),
            LocalPredictiveCodingContextMasks {
                neuron: Some(neuron_mask),
                activity: Some(activity_mask),
            },
            config,
            &LocalPredictiveCodingProfile::default(),
        ),
    )
}

fn validate_local_predictive_coding_state<B: AutodiffBackend>(
    model: &DragonModel<B>,
    inputs: &Tensor<B, 2, Int>,
    state: &ModelState<B>,
) -> Result<(), String>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let support = model.predictive_coding_support()?;
    if state.layers.len() != support.layers {
        return Err(format!(
            "local predictive-coding state has {} layers, expected {}",
            state.layers.len(),
            support.layers
        ));
    }
    let batch = inputs.shape().dims::<2>()[0];
    for (layer, layer_state) in state.layers.iter().enumerate() {
        if let Some(rho) = layer_state.rho.as_ref() {
            model
                .predictive_coding_validate_rho_state(rho, batch)
                .map_err(|error| format!("local predictive-coding state layer {layer}: {error}"))?;
        }
    }
    Ok(())
}

fn validate_local_predictive_coding_derivative_request<B: AutodiffBackend>(
    model: &DragonModel<B>,
    inputs: &Tensor<B, 2, Int>,
    targets: &Tensor<B, 2, Int>,
    loss_mask: Option<&Tensor<B, 2, Int>>,
    config: &LocalPredictiveCodingConfig,
) -> Result<(), String>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    model.predictive_coding_support()?;
    config
        .inference
        .validate("local_predictive_coding_derivatives.inference")
        .map_err(|error| error.to_string())?;
    if config.prediction_precision <= 0.0 || !config.prediction_precision.is_finite() {
        return Err(
            "local_predictive_coding_derivatives.prediction_precision must be finite and > 0"
                .to_string(),
        );
    }
    if !matches!(
        config.learning_schedule,
        burn_pc::PcLearningSchedule::Equilibrium
    ) {
        return Err(
            "local_predictive_coding_derivatives supports only learning_schedule=equilibrium"
                .to_string(),
        );
    }
    if matches!(
        config.solver,
        LocalPredictiveCodingSolver::LayerLocalPrediction
    ) && !matches!(
        config.factor_reduction,
        PredictiveCodingFactorReduction::Mean
    ) {
        return Err(
            "local_predictive_coding_derivatives layer_local_prediction requires factor_reduction=mean"
                .to_string(),
        );
    }
    if matches!(
        config.solver,
        LocalPredictiveCodingSolver::LayerLocalPrediction
    ) && config.sync_diagnostics
    {
        return Err(
            "local_predictive_coding_derivatives layer_local_prediction does not define equilibrium-energy diagnostics"
                .to_string(),
        );
    }
    let input_shape = inputs.shape();
    if input_shape.num_elements() == 0 {
        return Err("local predictive coding requires at least one token".to_string());
    }
    if targets.shape() != input_shape {
        return Err(format!(
            "target shape {:?} does not match input shape {input_shape:?}",
            targets.shape()
        ));
    }
    if let Some(mask) = loss_mask
        && mask.shape() != input_shape
    {
        return Err(format!(
            "loss-mask shape {:?} does not match input shape {input_shape:?}",
            mask.shape()
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::Module;
    use burn::optim::{AdamWConfig, Optimizer};
    use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
    use burn::tensor::TensorData;
    use burn_autodiff::Autodiff;
    use burn_dragon_core::{DragonConfig, SequenceTrainingExecutor};
    use burn_ndarray::NdArray;

    type TestBackend = Autodiff<NdArray<f32>>;
    type PlainBackend = NdArray<f32>;

    fn max_abs_diff<const D: usize>(
        left: Tensor<PlainBackend, D>,
        right: Tensor<PlainBackend, D>,
    ) -> f32 {
        (left - right)
            .abs()
            .max()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("maximum difference")[0]
    }

    fn model(device: &burn::tensor::Device<TestBackend>) -> DragonModel<TestBackend> {
        let mut config = DragonConfig {
            n_layer: 2,
            n_embd: 8,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 16,
            dropout: 0.0,
            ..DragonConfig::default()
        };
        config.sequence_kernel.executor = SequenceTrainingExecutor::DenseScoreShortContext;
        config.fused_kernels.rotary_embedding = burn_dragon_core::RotaryEmbedding::Alibi;
        DragonModel::new(config, device)
    }

    fn batch(
        device: &burn::tensor::Device<TestBackend>,
    ) -> (Tensor<TestBackend, 2, Int>, Tensor<TestBackend, 2, Int>) {
        (
            Tensor::from_data(
                TensorData::new(vec![1_i64, 2, 3, 1, 2, 3, 1, 2], [2, 4]),
                device,
            ),
            Tensor::from_data(
                TensorData::new(vec![2_i64, 3, 1, 2, 3, 1, 2, 3], [2, 4]),
                device,
            ),
        )
    }

    fn batch_for_step(
        device: &burn::tensor::Device<TestBackend>,
        step: usize,
    ) -> (Tensor<TestBackend, 2, Int>, Tensor<TestBackend, 2, Int>) {
        let inputs = (0..8)
            .map(|index| ((index * 3 + step * 5) % 15 + 1) as i64)
            .collect::<Vec<_>>();
        let targets = inputs
            .iter()
            .enumerate()
            .map(|(index, value)| ((value + index as i64 + 1) % 15) + 1)
            .collect::<Vec<_>>();
        (
            Tensor::from_data(TensorData::new(inputs, [2, 4]), device),
            Tensor::from_data(TensorData::new(targets, [2, 4]), device),
        )
    }

    fn loss(model: &DragonModel<TestBackend>, device: &burn::tensor::Device<TestBackend>) -> f32 {
        let (inputs, targets) = batch(device);
        let plain = model.valid();
        let hidden = plain.forward_hidden(inputs.inner());
        burn_pc::diagnostic_scalar_f32(
            plain
                .predictive_coding_head_vjp(hidden, targets.inner(), None)
                .loss,
        )
    }

    #[test]
    fn local_pc_step_emits_only_local_gradients_and_descends() {
        let device = Default::default();
        let mut model = model(&device);
        let config = LocalPredictiveCodingConfig {
            inference: burn_pc::PcInferenceConfig {
                steps: 4,
                step_size: 0.1,
                max_grad_norm: None,
                ..burn_pc::PcInferenceConfig::default()
            },
            ..LocalPredictiveCodingConfig::default()
        };
        let profile = LocalPredictiveCodingProfile::default();
        let initial_loss = loss(&model, &device);
        let mut optimizer = AdamWConfig::new().init::<TestBackend, DragonModel<TestBackend>>();
        let mut last_report = LocalPredictiveCodingStepReport::default();
        for _ in 0..24 {
            let (inputs, targets) = batch(&device);
            let step = local_predictive_coding_train_step(
                &model, inputs, targets, None, &config, &profile,
            );
            last_report = step.report;
            model = optimizer.step(2.0e-3, model, step.grads);
        }
        let final_loss = loss(&model, &device);

        assert_eq!(last_report.global_backward_calls, 0);
        assert_eq!(last_report.gradient_tensors, 9);
        assert!(last_report.local_vjp_calls > 0);
        assert_eq!(profile.snapshot().global_backward_calls, 0);
        assert!(
            final_loss < initial_loss,
            "local PC should descend next-token loss: initial={initial_loss} final={final_loss}"
        );
    }

    #[test]
    fn local_pc_reports_feedforward_loss_before_activity_inference() {
        let device = Default::default();
        let model = model(&device);
        let expected = loss(&model, &device);
        let (inputs, targets) = batch(&device);
        let step = local_predictive_coding_train_step(
            &model,
            inputs,
            targets,
            None,
            &LocalPredictiveCodingConfig {
                inference: burn_pc::PcInferenceConfig {
                    steps: 4,
                    step_size: 0.05,
                    max_grad_norm: None,
                    ..burn_pc::PcInferenceConfig::default()
                },
                ..LocalPredictiveCodingConfig::default()
            },
            &LocalPredictiveCodingProfile::default(),
        );
        let reported = burn_pc::diagnostic_scalar_f32(step.loss.inner());
        assert!(
            (reported - expected).abs() < 1.0e-6,
            "reported={reported} expected={expected}"
        );
    }

    #[test]
    fn fixed_prediction_tracks_backpropagation_across_optimizer_steps() {
        let device = Default::default();
        TestBackend::seed(&device, 20260806);
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let bytes = recorder
            .record(model(&device).into_record(), ())
            .expect("serialize shared initial model");
        let mut backprop_model = model(&device).load_record(
            recorder
                .load(bytes.clone(), &device)
                .expect("load backprop control"),
        );
        let mut fixed_model = model(&device).load_record(
            recorder
                .load(bytes, &device)
                .expect("load fixed-prediction control"),
        );
        let initial_backprop_loss = loss(&backprop_model, &device);
        let repeated_backprop_loss = loss(&backprop_model, &device);
        let initial_fixed_loss = loss(&fixed_model, &device);
        assert_eq!(
            initial_backprop_loss, repeated_backprop_loss,
            "the deterministic test model must have repeatable forwards"
        );
        assert_eq!(
            initial_backprop_loss, initial_fixed_loss,
            "serialized controls must start identically"
        );
        let mut backprop_optimizer =
            AdamWConfig::new().init::<TestBackend, DragonModel<TestBackend>>();
        let mut fixed_optimizer =
            AdamWConfig::new().init::<TestBackend, DragonModel<TestBackend>>();
        let config = LocalPredictiveCodingConfig {
            solver: LocalPredictiveCodingSolver::FixedPrediction,
            ..LocalPredictiveCodingConfig::default()
        };
        let profile = LocalPredictiveCodingProfile::default();
        let learning_rate = 1.0e-3;

        for step_index in 0..16 {
            let (inputs, targets) = batch_for_step(&device, step_index);
            let logits = backprop_model.forward(inputs.clone());
            let training_loss = burn_dragon_core::objective::masked_token_mean(
                backprop_model.language_token_losses_from_logits(logits, targets.clone()),
                None,
            );
            let grads = GradientsParams::from_grads(training_loss.backward(), &backprop_model);
            let fixed = local_predictive_coding_train_step(
                &fixed_model,
                inputs,
                targets,
                None,
                &config,
                &profile,
            );
            assert_eq!(fixed.report.global_backward_calls, 0);
            backprop_model = backprop_optimizer.step(learning_rate, backprop_model, grads);
            fixed_model = fixed_optimizer.step(learning_rate, fixed_model, fixed.grads);

            let backprop_loss = loss(&backprop_model, &device);
            let fixed_loss = loss(&fixed_model, &device);
            assert!(
                (backprop_loss - fixed_loss).abs() < 2.0e-4,
                "optimizer trajectories diverged at step {step_index}: backprop={backprop_loss} fixed={fixed_loss}"
            );
        }

        assert_eq!(profile.snapshot().steps, 16);
        assert_eq!(profile.snapshot().global_backward_calls, 0);
    }

    #[test]
    fn layer_local_prediction_batches_explicit_local_factors_exactly() {
        let device = Default::default();
        TestBackend::seed(&device, 20260806);
        let model = model(&device);
        let plain = model.valid();
        let (inputs, targets) = batch(&device);
        let inputs_plain = inputs.clone().inner();
        let targets_plain = targets.clone().inner();
        let mut activity = plain.predictive_coding_initial_activity(inputs_plain.clone());
        let layers = plain.predictive_coding_layer_count();
        let mut expected_shared: Option<SharedParameterVjp<PlainBackend>> = None;
        let mut expected_initial_grad: Option<Tensor<PlainBackend, 4>> = None;
        let mut expected_head: Option<Tensor<PlainBackend, 2>> = None;

        for layer in 0..layers {
            let trace = plain.predictive_coding_forward_layer(activity, layer);
            activity = trace.next.clone().detach();
            let shape = activity.shape();
            let head = plain.predictive_coding_head_vjp(
                plain.predictive_coding_hidden_from_activity(activity.clone()),
                targets_plain.clone(),
                None,
            );
            let vjp = layer_parameter_vjp(
                &plain,
                layer,
                &trace,
                head.grad_hidden.reshape(shape),
                None,
                None,
            );
            match expected_shared.as_mut() {
                Some(shared) => {
                    let _ = shared.accumulate_layer(vjp);
                }
                None => {
                    let (grad_input, shared) = SharedParameterVjp::from_layer(vjp);
                    expected_initial_grad = Some(grad_input);
                    expected_shared = Some(shared);
                }
            }
            let grad_head = head.grad_lm_head.div_scalar(layers as f32);
            expected_head = Some(match expected_head {
                Some(accumulated) => accumulated + grad_head,
                None => grad_head,
            });
        }
        let expected_initial = plain.predictive_coding_initial_vjp(
            inputs_plain,
            expected_initial_grad.expect("first local factor input derivative"),
        );
        let expected_shared = expected_shared.expect("shared local factor derivatives");
        let config = LocalPredictiveCodingConfig {
            solver: LocalPredictiveCodingSolver::LayerLocalPrediction,
            factor_reduction: PredictiveCodingFactorReduction::Mean,
            ..LocalPredictiveCodingConfig::default()
        };
        let step = local_predictive_coding_train_step(
            &model,
            inputs,
            targets,
            None,
            &config,
            &LocalPredictiveCodingProfile::default(),
        );
        let ids = model
            .predictive_coding_parameter_ids()
            .expect("supported PC model");

        let comparisons = [
            (
                "encoder",
                max_abs_diff(
                    step.grads
                        .get::<PlainBackend, 3>(ids.encoder)
                        .expect("batched encoder derivative"),
                    expected_shared.grad_encoder,
                ),
            ),
            (
                "encoder_v",
                max_abs_diff(
                    step.grads
                        .get::<PlainBackend, 3>(ids.encoder_v)
                        .expect("batched encoder-v derivative"),
                    expected_shared.grad_encoder_v,
                ),
            ),
            (
                "decoder",
                max_abs_diff(
                    step.grads
                        .get::<PlainBackend, 2>(ids.decoder)
                        .expect("batched decoder derivative"),
                    expected_shared.grad_decoder,
                ),
            ),
            (
                "embedding",
                max_abs_diff(
                    step.grads
                        .get::<PlainBackend, 2>(ids.embedding)
                        .expect("batched embedding derivative"),
                    expected_initial.grad_embedding,
                ),
            ),
            (
                "lm_head",
                max_abs_diff(
                    step.grads
                        .get::<PlainBackend, 2>(ids.lm_head)
                        .expect("batched head derivative"),
                    expected_head.expect("explicit local head derivative"),
                ),
            ),
        ];
        for (family, error) in comparisons {
            assert!(
                error < 2.0e-5,
                "tensorized layer-local {family} derivative mismatch: {error}"
            );
        }
        assert_eq!(step.report.solver, config.solver);
        assert_eq!(step.report.local_vjp_calls, 3);
        assert_eq!(step.report.global_backward_calls, 0);
    }

    #[test]
    fn layer_local_prediction_preserves_terminal_loss_and_token_accounting() {
        let device = Default::default();
        let model = model(&device);
        let expected_loss = loss(&model, &device);
        let (inputs, targets) = batch(&device);
        let mask = Tensor::from_data(
            TensorData::new(vec![1_i64, 1, 0, 1, 1, 0, 1, 1], [2, 4]),
            &device,
        );
        let expected_masked_loss = burn_pc::diagnostic_scalar_f32(
            model
                .valid()
                .predictive_coding_head_vjp(
                    model.valid().forward_hidden(inputs.clone().inner()),
                    targets.clone().inner(),
                    Some(mask.clone().inner()),
                )
                .loss,
        );
        let step = local_predictive_coding_train_step(
            &model,
            inputs,
            targets,
            Some(mask),
            &LocalPredictiveCodingConfig {
                solver: LocalPredictiveCodingSolver::LayerLocalPrediction,
                factor_reduction: PredictiveCodingFactorReduction::Mean,
                ..LocalPredictiveCodingConfig::default()
            },
            &LocalPredictiveCodingProfile::default(),
        );
        let reported_loss = burn_pc::diagnostic_scalar_f32(step.loss.inner());
        let supervised = burn_pc::diagnostic_scalar_f32(step.supervised_tokens.inner());

        assert!(expected_loss.is_finite());
        assert!(
            (reported_loss - expected_masked_loss).abs() < 1.0e-6,
            "reported={reported_loss} expected={expected_masked_loss}"
        );
        assert!((supervised - 6.0).abs() < 1.0e-6);
        assert_eq!(step.report.factors, 4);
        assert_eq!(step.report.gradient_tensors, 9);
    }

    #[test]
    fn local_pc_inference_descends_joint_factor_energy() {
        let device = Default::default();
        let model = model(&device);
        let config = LocalPredictiveCodingConfig {
            inference: burn_pc::PcInferenceConfig {
                steps: 4,
                step_size: 0.05,
                max_grad_norm: None,
                ..burn_pc::PcInferenceConfig::default()
            },
            sync_diagnostics: true,
            ..LocalPredictiveCodingConfig::default()
        };
        let (inputs, targets) = batch(&device);
        let report = local_predictive_coding_train_step(
            &model,
            inputs,
            targets,
            None,
            &config,
            &LocalPredictiveCodingProfile::default(),
        )
        .report;
        let before = report.energy_before.expect("energy before inference");
        let after = report.energy_after.expect("energy after inference");
        assert!(
            after < before,
            "activity inference must descend joint energy: before={before} after={after}"
        );
    }

    #[test]
    fn local_pc_derivatives_are_invariant_to_batch_duplication() {
        let device = Default::default();
        let model = model(&device);
        let config = LocalPredictiveCodingConfig {
            inference: burn_pc::PcInferenceConfig {
                steps: 4,
                step_size: 0.05,
                max_grad_norm: None,
                ..burn_pc::PcInferenceConfig::default()
            },
            ..LocalPredictiveCodingConfig::default()
        };
        let single_inputs =
            Tensor::from_data(TensorData::new(vec![1_i64, 2, 3, 1], [1, 4]), &device);
        let single_targets =
            Tensor::from_data(TensorData::new(vec![2_i64, 3, 1, 2], [1, 4]), &device);
        let double_inputs = Tensor::from_data(
            TensorData::new(vec![1_i64, 2, 3, 1, 1, 2, 3, 1], [2, 4]),
            &device,
        );
        let double_targets = Tensor::from_data(
            TensorData::new(vec![2_i64, 3, 1, 2, 2, 3, 1, 2], [2, 4]),
            &device,
        );
        let single = local_predictive_coding_train_step(
            &model,
            single_inputs,
            single_targets,
            None,
            &config,
            &LocalPredictiveCodingProfile::default(),
        );
        let doubled = local_predictive_coding_train_step(
            &model,
            double_inputs,
            double_targets,
            None,
            &config,
            &LocalPredictiveCodingProfile::default(),
        );
        let ids = model
            .predictive_coding_parameter_ids()
            .expect("supported PC model");
        let encoder_diff = max_abs_diff(
            single
                .grads
                .get::<PlainBackend, 3>(ids.encoder)
                .expect("single encoder derivative"),
            doubled
                .grads
                .get::<PlainBackend, 3>(ids.encoder)
                .expect("doubled encoder derivative"),
        );
        let head_diff = max_abs_diff(
            single
                .grads
                .get::<PlainBackend, 2>(ids.lm_head)
                .expect("single head derivative"),
            doubled
                .grads
                .get::<PlainBackend, 2>(ids.lm_head)
                .expect("doubled head derivative"),
        );
        assert!(
            encoder_diff < 1.0e-5,
            "encoder derivative changed after duplicating the batch: {encoder_diff}"
        );
        assert!(
            head_diff < 1.0e-5,
            "head derivative changed after duplicating the batch: {head_diff}"
        );
    }

    #[test]
    fn dragon_graph_is_versioned_and_well_formed() {
        let graph = dragon_predictive_coding_graph(3);
        graph.validate().expect("valid Dragon PC graph");
        assert_eq!(graph.version, burn_pc::PcGraphSpec::CURRENT_VERSION);
        assert_eq!(graph.nodes.len(), 5);
        assert_eq!(graph.factors.len(), 4);
        assert!(graph.nodes[0].clamped);
        assert!(graph.nodes[4].clamped);
    }
}
