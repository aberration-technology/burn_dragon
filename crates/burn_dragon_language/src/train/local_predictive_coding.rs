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

mod criterion;
mod diagnostics;
mod graph;
mod telemetry;
mod verifier;
use criterion::LocalPcTerminalCriterion;
pub use diagnostics::*;
pub use graph::{dragon_predictive_coding_checkpoint_manifest, dragon_predictive_coding_graph};
pub(crate) use telemetry::validate_step_execution_contract;
pub use telemetry::{
    LocalPredictiveCodingProfile, LocalPredictiveCodingProfileSnapshot,
    LocalPredictiveCodingStepReport,
};
pub(crate) use verifier::{prepare_ruliad_verifier_terminal, verifier_terminal_due};

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
    /// Updated training-only direct-feedback bank. It is absent for solvers
    /// without amortized local credit and never becomes an inference model
    /// parameter.
    pub dkp_feedback: Option<Tensor<B, 3>>,
    pub report: LocalPredictiveCodingStepReport,
}

#[derive(Debug, Clone)]
pub(super) struct DkpFeedbackState<B: Backend> {
    pub(super) feedback: Tensor<B, 3>,
    pub(super) updates: u64,
}

fn initial_dkp_feedback<B: Backend>(
    layers: usize,
    dim: usize,
    initialization: burn_pc::PcFeedbackInitialization,
    device: &B::Device,
) -> Tensor<B, 3> {
    match initialization {
        burn_pc::PcFeedbackInitialization::Gaussian => Tensor::random(
            [layers, dim, dim],
            burn::tensor::Distribution::Normal(0.0, (dim.max(1) as f64).sqrt().recip()),
            device,
        ),
        burn_pc::PcFeedbackInitialization::Identity => {
            Tensor::<B, 1, Int>::arange(0..dim as i64, device)
                .one_hot::<2>(dim)
                .float()
                .unsqueeze_dim::<3>(0)
                .repeat_dim(0, layers)
        }
    }
}

#[derive(Debug, Default)]
pub(super) struct LocalPredictiveCodingContextMasks<B: Backend> {
    pub(super) neuron: Option<Tensor<B, 4>>,
    pub(super) activity: Option<Tensor<B, 4>>,
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
    criterion: LocalPcTerminalCriterion<B>,
    activities: Vec<Tensor<B, 4>>,
    traces: Vec<DragonPredictiveCodingLayerTrace<B>>,
    neuron_mask: Option<Tensor<B, 4>>,
    activity_mask: Option<Tensor<B, 4>>,
    terminal_state: ModelState<B>,
    factors: usize,
    scale: f32,
}

#[allow(clippy::too_many_arguments)]
fn prepare_fixed_prediction_context<B: AutodiffBackend>(
    model: &DragonModel<B>,
    inputs: Tensor<B::InnerBackend, 2, Int>,
    criterion: LocalPcTerminalCriterion<B::InnerBackend>,
    initial_state: Option<ModelState<B::InnerBackend>>,
    neuron_mask: Option<Tensor<B::InnerBackend, 4>>,
    activity_mask: Option<Tensor<B::InnerBackend, 4>>,
    config: &LocalPredictiveCodingConfig,
) -> (
    DragonModel<B::InnerBackend>,
    FixedPredictionContext<B::InnerBackend>,
)
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let parameter_ids = model
        .predictive_coding_parameter_ids()
        .expect("validated local predictive-coding model");
    let plain = model.valid();
    plain
        .predictive_coding_support()
        .expect("validated plain local predictive-coding model");
    let mut terminal_state = initial_state.unwrap_or_else(|| plain.init_state_ephemeral());
    assert_eq!(
        terminal_state.layers.len(),
        plain.predictive_coding_layer_count(),
        "local PC recurrent-state layer count must match the model"
    );
    let block_time = inputs.shape().dims::<2>()[1];
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
    let mut traces = Vec::with_capacity(layers);
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
        traces.push(trace);
    }
    terminal_state.position = terminal_state.position.saturating_add(block_time);
    terminal_state.detach_in_place();
    let context = FixedPredictionContext {
        parameter_ids,
        inputs,
        criterion,
        activities,
        traces,
        neuron_mask,
        activity_mask,
        terminal_state,
        factors,
        scale,
    };
    (plain, context)
}

pub(super) fn local_predictive_coding_verifier_train_step<B: AutodiffBackend>(
    model: &DragonModel<B>,
    prepared: verifier::PreparedRuliadVerifierTerminal<B::InnerBackend>,
    config: &LocalPredictiveCodingConfig,
    profile: &LocalPredictiveCodingProfile,
) -> LocalPredictiveCodingDerivatives<B>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let started = Instant::now();
    let semantic_states = prepared.semantic_states;
    let decision_rows = prepared.decision_rows;
    let (plain, context) = prepare_fixed_prediction_context::<B>(
        model,
        prepared.inputs,
        prepared.criterion,
        None,
        None,
        None,
        config,
    );
    let result = match config.solver {
        LocalPredictiveCodingSolver::ErrorEquilibrium => {
            error_equilibrium_train_step::<B>(&plain, context, config, started, profile)
        }
        LocalPredictiveCodingSolver::FixedPrediction => {
            fixed_prediction_train_step::<B>(&plain, context, config, started, profile)
        }
        _ => unreachable!("verifier terminal validation restricts the local PC solver"),
    };
    profile.record_structured_terminal(semantic_states, decision_rows);
    result
}

pub(super) struct DkpPreparedChunk<B: AutodiffBackend> {
    parameter_ids: DragonPredictiveCodingParameterIds,
    inputs: Tensor<B::InnerBackend, 2, Int>,
    targets: Tensor<B::InnerBackend, 2, Int>,
    loss_mask: Option<Tensor<B::InnerBackend, 2, Int>>,
    activities: Vec<Tensor<B::InnerBackend, 4>>,
    initial_rhos: Vec<Option<Tensor<B::InnerBackend, 4>>>,
    pub(super) feedback: Tensor<B::InnerBackend, 3>,
    feedback_updates: u64,
    pub(super) preliminary_grads: GradientsParams,
    pub(super) loss: Tensor<B, 1>,
    pub(super) supervised_tokens: Tensor<B, 1>,
    pub(super) terminal_state: ModelState<B>,
    factors: usize,
    scale: f32,
}

pub(super) struct DkpChunkObservation<B: AutodiffBackend> {
    pub(super) loss: Tensor<B, 1>,
    pub(super) supervised_tokens: Tensor<B, 1>,
    pub(super) terminal_state: ModelState<B>,
}

fn tied_preliminary_gradient<B: Backend, const D: usize>(
    gradient_sum: Tensor<B, D>,
    factors: usize,
    config: &burn_pc::PcTiedConsensusConfig,
) -> Tensor<B, D> {
    burn_pc::tied_uniform_consensus_from_sum(gradient_sum.mul_scalar(-1.0), factors, config)
        .expect("validated tied DKP consensus")
        .update
        .mul_scalar(-1.0)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_dkp_predictive_coding_chunk<B: AutodiffBackend>(
    model: &DragonModel<B>,
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
    loss_mask: Option<Tensor<B, 2, Int>>,
    initial_state: ModelState<B>,
    feedback: Option<Tensor<B, 3>>,
    feedback_updates: u64,
    config: &LocalPredictiveCodingConfig,
) -> DkpPreparedChunk<B>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    debug_assert!(matches!(
        config.solver,
        LocalPredictiveCodingSolver::DirectKolenPollack
    ));
    let parameter_ids = model
        .predictive_coding_parameter_ids()
        .expect("validated DKP model parameter ids");
    let plain = model.valid();
    let mut terminal_state = initial_state.inner_cloned();
    let block_time = inputs.shape().dims::<2>()[1];
    let inputs = inputs.inner();
    let targets = targets.inner();
    let loss_mask = loss_mask.map(Tensor::inner);
    let layers = plain.predictive_coding_layer_count();
    let factors = layers + 1;
    let scale = factor_scale(config, factors);
    let mut activities = Vec::with_capacity(layers + 1);
    let mut traces = Vec::with_capacity(layers);
    activities.push(plain.predictive_coding_initial_activity(inputs.clone()));
    for layer in 0..layers {
        let trace = plain
            .predictive_coding_forward_layer_with_recurrent_state(
                activities[layer].clone(),
                layer,
                terminal_state.layers[layer].rho.clone(),
                None,
                None,
            )
            .expect("validated recurrent DKP layer factor");
        terminal_state.layers[layer].rho = Some(plain.predictive_coding_terminal_rho(&trace));
        terminal_state.layers[layer].rho_norm = None;
        terminal_state.layers[layer].sequence_aux = None;
        activities.push(trace.next.clone().detach());
        traces.push(trace);
    }
    terminal_state.position = terminal_state.position.saturating_add(block_time);
    terminal_state.detach_in_place();
    let initial_rhos = traces
        .iter()
        .map(|trace| trace.initial_rho.clone())
        .collect::<Vec<_>>();
    let terminal_activity = activities.last().expect("terminal DKP activity");
    let [batch, streams, time, dim] = terminal_activity.shape().dims::<4>();
    let terminal_hidden = plain.predictive_coding_hidden_from_activity(terminal_activity.clone());
    let terminal = plain.predictive_coding_head_activity_vjp(
        terminal_hidden,
        targets.clone(),
        loss_mask.clone(),
    );
    let feedback = feedback.map(Tensor::inner).unwrap_or_else(|| {
        initial_dkp_feedback::<B::InnerBackend>(
            layers,
            dim,
            config.direct_feedback.initialization,
            &terminal_activity.device(),
        )
    });
    assert_eq!(
        feedback.shape().dims::<3>(),
        [layers, dim, dim],
        "DKP feedback checkpoint geometry must match model depth and embedding"
    );
    let terminal_derivative = terminal.grad_hidden.reshape([batch * time, dim]);
    let direct_derivatives = burn_pc::direct_feedback_signal_batched(
        terminal_derivative
            .reshape([1, batch * time, dim])
            .repeat_dim(0, layers),
        feedback.clone(),
        config.direct_feedback.signal_scale,
    )
    .reshape([layers * batch, streams, time, dim]);
    let batched_trace = concatenate_traces(&traces);
    let preliminary =
        layer_parameter_vjp(&plain, 0, &batched_trace, direct_derivatives, None, None);
    let mut preliminary_grads = GradientsParams::new();
    preliminary_grads.register(
        parameter_ids.encoder,
        tied_preliminary_gradient(preliminary.grad_encoder, layers, &config.tied_consensus),
    );
    preliminary_grads.register(
        parameter_ids.encoder_v,
        tied_preliminary_gradient(preliminary.grad_encoder_v, layers, &config.tied_consensus),
    );
    preliminary_grads.register(
        parameter_ids.decoder,
        tied_preliminary_gradient(preliminary.grad_decoder, layers, &config.tied_consensus),
    );

    DkpPreparedChunk {
        parameter_ids,
        inputs,
        targets,
        loss_mask,
        activities,
        initial_rhos,
        feedback,
        feedback_updates,
        preliminary_grads,
        loss: Tensor::<B, 1>::from_inner(terminal.loss),
        supervised_tokens: Tensor::<B, 1>::from_inner(terminal.normalization.reshape([1])),
        terminal_state: ModelState::<B>::from_inner_cloned(terminal_state),
        factors,
        scale,
    }
}

/// Measure a later TBPTT chunk without constructing the preliminary DKP body
/// derivative that must be recomputed after preceding chunks update weights.
/// The full recurrent forward is retained so the displayed batch loss and rho
/// carry remain exact at the pre-update snapshot.
pub(super) fn observe_dkp_predictive_coding_chunk<B: AutodiffBackend>(
    model: &DragonModel<B>,
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
    loss_mask: Option<Tensor<B, 2, Int>>,
    initial_state: ModelState<B>,
) -> DkpChunkObservation<B>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let plain = model.valid();
    let mut terminal_state = initial_state.inner_cloned();
    let hidden = plain.forward_hidden_with_state(inputs.inner(), &mut terminal_state);
    let terminal = plain.predictive_coding_head_activity_vjp(
        hidden,
        targets.inner(),
        loss_mask.map(Tensor::inner),
    );
    terminal_state.detach_in_place();
    DkpChunkObservation {
        loss: Tensor::<B, 1>::from_inner(terminal.loss),
        supervised_tokens: Tensor::<B, 1>::from_inner(terminal.normalization.reshape([1])),
        terminal_state: ModelState::<B>::from_inner_cloned(terminal_state),
    }
}

/// Exact local-VJP teacher for every shared-depth output activity.
///
/// This is a sequence of factor-local VJPs, not a global autodiff traversal.
/// It runs only on the configured calibration cadence and teaches the batched
/// direct-feedback bank used by intervening updates.
fn exact_layer_output_adjoint_batch<B: Backend>(
    model: &DragonModel<B>,
    batched_trace: &DragonPredictiveCodingLayerTrace<B>,
    terminal_gradient: Tensor<B, 4>,
    layers: usize,
    batch: usize,
) -> Tensor<B, 3>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let [_, streams, time, dim] = terminal_gradient.shape().dims::<4>();
    let observations = batch * streams * time;
    let mut gradient = terminal_gradient;
    let mut reversed = Vec::with_capacity(layers);
    for layer in (0..layers).rev() {
        reversed.push(gradient.clone().reshape([1, observations, dim]));
        if layer > 0 {
            gradient = layer_activity_vjp(
                model,
                layer,
                &slice_trace_batch(batched_trace, layer * batch, (layer + 1) * batch),
                gradient,
                None,
                None,
            );
        }
    }
    reversed.reverse();
    Tensor::cat(reversed, 0)
}

/// One-step local-credit update driven directly by an amortized adjoint bank.
///
/// Exact teacher steps still use only factor-local activity VJPs. Their
/// per-layer adjoints are tensorized into one shared-parameter VJP and also
/// calibrate the feedback bank used on intervening steps. This keeps the
/// optimizer contract to one update per batch and makes the approximation
/// directly responsible for the body derivative it is intended to replace.
#[allow(clippy::too_many_arguments)]
pub(super) fn amortized_adjoint_predictive_coding_train_step<B: AutodiffBackend>(
    model: &DragonModel<B>,
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
    loss_mask: Option<Tensor<B, 2, Int>>,
    initial_state: ModelState<B>,
    feedback: Option<Tensor<B, 3>>,
    feedback_updates: u64,
    config: &LocalPredictiveCodingConfig,
    profile: &LocalPredictiveCodingProfile,
) -> LocalPredictiveCodingDerivatives<B>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    debug_assert!(matches!(
        config.solver,
        LocalPredictiveCodingSolver::AmortizedAdjoint
    ));
    let started = Instant::now();
    let parameter_ids = model
        .predictive_coding_parameter_ids()
        .expect("validated amortized-adjoint model parameter ids");
    let plain = model.valid();
    let inputs = inputs.inner();
    let targets = targets.inner();
    let loss_mask = loss_mask.map(Tensor::inner);
    let mut terminal_state = initial_state.inner_cloned();
    let block_time = inputs.shape().dims::<2>()[1];
    let layers = plain.predictive_coding_layer_count();
    let factors = layers + 1;
    let mut activities = Vec::with_capacity(layers + 1);
    let mut traces = Vec::with_capacity(layers);
    activities.push(plain.predictive_coding_initial_activity(inputs.clone()));
    for layer in 0..layers {
        let trace = plain
            .predictive_coding_forward_layer_with_recurrent_state(
                activities[layer].clone(),
                layer,
                terminal_state.layers[layer].rho.clone(),
                None,
                None,
            )
            .expect("validated recurrent amortized-adjoint layer factor");
        terminal_state.layers[layer].rho = Some(plain.predictive_coding_terminal_rho(&trace));
        terminal_state.layers[layer].rho_norm = None;
        terminal_state.layers[layer].sequence_aux = None;
        activities.push(trace.next.clone().detach());
        traces.push(trace);
    }
    terminal_state.position = terminal_state.position.saturating_add(block_time);
    terminal_state.detach_in_place();

    let terminal_activity = activities
        .last()
        .expect("terminal amortized-adjoint activity");
    let [batch, streams, time, dim] = terminal_activity.shape().dims::<4>();
    let terminal = plain.predictive_coding_head_vjp(
        plain.predictive_coding_hidden_from_activity(terminal_activity.clone()),
        targets,
        loss_mask,
    );
    let terminal_gradient = terminal
        .grad_hidden
        .clone()
        .reshape(terminal_activity.shape());
    let terminal_signal = terminal_gradient
        .clone()
        .reshape([1, batch * streams * time, dim])
        .repeat_dim(0, layers);
    let feedback = feedback.map(Tensor::inner).unwrap_or_else(|| {
        initial_dkp_feedback::<B::InnerBackend>(
            layers,
            dim,
            config.direct_feedback.initialization,
            &terminal_activity.device(),
        )
    });
    assert_eq!(
        feedback.shape().dims::<3>(),
        [layers, dim, dim],
        "amortized-adjoint feedback checkpoint geometry must match model depth and embedding"
    );
    let batched_trace = concatenate_traces(&traces);
    let teacher_due = config.amortized_adjoint.teacher_due(feedback_updates);
    let (layer_signals, updated_feedback, teacher_vjp_calls) = if teacher_due {
        let teacher_signal = exact_layer_output_adjoint_batch(
            &plain,
            &batched_trace,
            terminal_gradient,
            layers,
            batch,
        );
        let calibration = burn_pc::calibrate_adjoint_batched(
            feedback,
            terminal_signal,
            teacher_signal.clone(),
            &config.amortized_adjoint.calibration,
        );
        if config.sync_diagnostics {
            profile.record_adjoint_calibration(
                f64::from(burn_pc::diagnostic_scalar_f32(calibration.loss.clone())),
                f64::from(burn_pc::diagnostic_scalar_f32(
                    calibration.cosine_alignment.clone(),
                )),
                f64::from(burn_pc::diagnostic_scalar_f32(
                    calibration.prediction_teacher_norm_ratio.clone(),
                )),
                f64::from(burn_pc::diagnostic_scalar_f32(
                    calibration.update_rms.clone(),
                )),
            );
        }
        (
            teacher_signal,
            calibration.feedback,
            layers.saturating_sub(1),
        )
    } else {
        (
            burn_pc::direct_feedback_signal_batched(
                terminal_signal,
                feedback.clone(),
                config.direct_feedback.signal_scale,
            ),
            feedback,
            0,
        )
    };
    let layer_vjp = layer_parameter_vjp(
        &plain,
        0,
        &batched_trace,
        layer_signals.reshape([layers * batch, streams, time, dim]),
        None,
        None,
    );
    let initial_vjp = plain.predictive_coding_initial_vjp(
        inputs,
        layer_vjp
            .grad_input
            .slice([0..batch, 0..streams, 0..time, 0..dim]),
    );
    let mut grads = GradientsParams::new();
    grads.register(parameter_ids.embedding, initial_vjp.grad_embedding);
    grads.register(parameter_ids.encoder, layer_vjp.grad_encoder);
    grads.register(parameter_ids.encoder_v, layer_vjp.grad_encoder_v);
    grads.register(parameter_ids.decoder, layer_vjp.grad_decoder);
    grads.register(
        parameter_ids.norm_gamma,
        layer_vjp.grad_norm_gamma + initial_vjp.grad_norm_gamma,
    );
    grads.register(
        parameter_ids.norm_beta,
        layer_vjp.grad_norm_beta + initial_vjp.grad_norm_beta,
    );
    grads.register(
        parameter_ids.norm_alpha,
        layer_vjp.grad_norm_alpha + initial_vjp.grad_norm_alpha,
    );
    grads.register(
        parameter_ids.norm_shift,
        layer_vjp.grad_norm_shift + initial_vjp.grad_norm_shift,
    );
    grads.register(parameter_ids.lm_head, terminal.grad_lm_head);

    let report = LocalPredictiveCodingStepReport {
        solver: LocalPredictiveCodingSolver::AmortizedAdjoint,
        inference_steps: 0,
        factors,
        local_vjp_calls: layers + 2 + teacher_vjp_calls,
        global_backward_calls: 0,
        gradient_tensors: grads.len(),
        direct_forward_updates: layers,
        feedback_parameter_updates: usize::from(teacher_due).saturating_mul(layers),
        adjoint_teacher_updates: usize::from(teacher_due).saturating_mul(layers),
        adjoint_local_updates: usize::from(!teacher_due).saturating_mul(layers),
        parameter_updates: 1,
        energy_before: None,
        energy_after: None,
        elapsed_ns: started.elapsed().as_nanos(),
    };
    validate_step_execution_contract(config, &report);
    profile.record(report);
    LocalPredictiveCodingDerivatives {
        grads,
        loss: Tensor::<B, 1>::from_inner(terminal.loss),
        supervised_tokens: Tensor::<B, 1>::from_inner(terminal.supervised_tokens),
        terminal_state: ModelState::<B>::from_inner_cloned(terminal_state),
        dkp_feedback: Some(Tensor::<B, 3>::from_inner(updated_feedback)),
        report,
    }
}

pub(super) fn finish_dkp_predictive_coding_chunk<B: AutodiffBackend>(
    model: &DragonModel<B>,
    mut chunk: DkpPreparedChunk<B>,
    config: &LocalPredictiveCodingConfig,
    profile: &LocalPredictiveCodingProfile,
) -> LocalPredictiveCodingDerivatives<B>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let started = Instant::now();
    let plain = model.valid();
    let layers = plain.predictive_coding_layer_count();
    let energy_before = config.sync_diagnostics.then(|| {
        burn_pc::diagnostic_scalar_f32(total_energy(
            &plain,
            &chunk.activities,
            &chunk.initial_rhos,
            chunk.targets.clone(),
            chunk.loss_mask.clone(),
            config,
            (None, None),
        )) as f64
    });
    let mut local_vjp_calls = 1usize;

    for _ in 0..config.inference.steps {
        let trace = forward_trace_batch(&plain, &chunk.activities, &chunk.initial_rhos, None, None);
        let terminal_activity = chunk.activities.last().expect("terminal DKP activity");
        let terminal_hidden =
            plain.predictive_coding_hidden_from_activity(terminal_activity.clone());
        let terminal = plain.predictive_coding_head_activity_vjp(
            terminal_hidden,
            chunk.targets.clone(),
            chunk.loss_mask.clone(),
        );
        let terminal_grad = (terminal.grad_hidden * terminal.normalization.reshape([1, 1, 1]))
            .reshape(terminal_activity.shape());
        let [batch, streams, time, dim] = chunk.activities[0].shape().dims::<4>();
        let inferred = Tensor::cat(chunk.activities.iter().skip(1).cloned().collect(), 0);
        let errors = prediction_error(
            trace.next.clone(),
            inferred,
            config.prediction_precision,
            chunk.scale,
        );
        let internal_child_grads = (layers > 1).then(|| {
            layer_activity_vjp(
                &plain,
                0,
                &slice_trace_batch(&trace, batch, layers * batch),
                slice_batch(errors.clone(), batch, layers * batch),
                None,
                None,
            )
        });
        local_vjp_calls = local_vjp_calls.saturating_add(layers.saturating_sub(1) + 1);
        let mut updates = Vec::with_capacity(layers);
        for (activity_index, activity) in
            chunk.activities.iter().enumerate().take(layers + 1).skip(1)
        {
            let own_offset = (activity_index - 1) * batch;
            let own = slice_batch(errors.clone(), own_offset, own_offset + batch).mul_scalar(-1.0);
            let child = if activity_index == layers {
                terminal_grad.clone().mul_scalar(chunk.scale)
            } else {
                let offset = (activity_index - 1) * batch;
                internal_child_grads
                    .as_ref()
                    .expect("non-terminal DKP activity has a child factor")
                    .clone()
                    .slice([offset..offset + batch, 0..streams, 0..time, 0..dim])
            };
            updates.push(
                burn_pc::pc_sgd_update(activity.clone(), own + child, &config.inference).detach(),
            );
        }
        for (activity, update) in chunk.activities.iter_mut().skip(1).zip(updates) {
            *activity = update;
        }
    }

    let energy_after = config.sync_diagnostics.then(|| {
        burn_pc::diagnostic_scalar_f32(total_energy(
            &plain,
            &chunk.activities,
            &chunk.initial_rhos,
            chunk.targets.clone(),
            chunk.loss_mask.clone(),
            config,
            (None, None),
        )) as f64
    });
    let trace = forward_trace_batch(&plain, &chunk.activities, &chunk.initial_rhos, None, None);
    let terminal_activity = chunk.activities.last().expect("terminal DKP activity");
    let [batch, streams, time, dim] = terminal_activity.shape().dims::<4>();
    let terminal_hidden = plain.predictive_coding_hidden_from_activity(terminal_activity.clone());
    let terminal = plain.predictive_coding_head_vjp(
        terminal_hidden,
        chunk.targets.clone(),
        chunk.loss_mask.clone(),
    );
    let normalization = terminal.supervised_tokens.clone().clamp_min(1.0);
    let errors = prediction_error_gradient(
        trace.next.clone(),
        Tensor::cat(chunk.activities.iter().skip(1).cloned().collect(), 0),
        config.prediction_precision,
        chunk.scale,
        normalization.clone(),
    );
    let batched_vjp = layer_parameter_vjp(&plain, 0, &trace, errors, None, None);
    let initial_grad = batched_vjp
        .grad_input
        .slice([0..batch, 0..streams, 0..time, 0..dim]);
    let initial_vjp = plain.predictive_coding_initial_vjp(chunk.inputs, initial_grad);
    let mut grads = GradientsParams::new();
    grads.register(chunk.parameter_ids.embedding, initial_vjp.grad_embedding);
    grads.register(chunk.parameter_ids.encoder, batched_vjp.grad_encoder);
    grads.register(chunk.parameter_ids.encoder_v, batched_vjp.grad_encoder_v);
    grads.register(chunk.parameter_ids.decoder, batched_vjp.grad_decoder);
    grads.register(
        chunk.parameter_ids.norm_gamma,
        batched_vjp.grad_norm_gamma + initial_vjp.grad_norm_gamma,
    );
    grads.register(
        chunk.parameter_ids.norm_beta,
        batched_vjp.grad_norm_beta + initial_vjp.grad_norm_beta,
    );
    grads.register(
        chunk.parameter_ids.norm_alpha,
        batched_vjp.grad_norm_alpha + initial_vjp.grad_norm_alpha,
    );
    grads.register(
        chunk.parameter_ids.norm_shift,
        batched_vjp.grad_norm_shift + initial_vjp.grad_norm_shift,
    );
    grads.register(
        chunk.parameter_ids.lm_head,
        terminal.grad_lm_head.mul_scalar(chunk.scale),
    );
    local_vjp_calls = local_vjp_calls.saturating_add(layers + 2);

    let terminal_signal = terminal
        .grad_hidden
        .clone()
        .reshape([1, batch * streams * time, dim])
        .repeat_dim(0, layers);
    let teacher_due = config.amortized_adjoint.teacher_due(chunk.feedback_updates);
    let updated_feedback = if teacher_due {
        let teacher_signal = exact_layer_output_adjoint_batch(
            &plain,
            &trace,
            terminal.grad_hidden.reshape(terminal_activity.shape()),
            layers,
            batch,
        );
        local_vjp_calls = local_vjp_calls.saturating_add(layers.saturating_sub(1));
        let calibration = burn_pc::calibrate_adjoint_batched(
            chunk.feedback,
            terminal_signal,
            teacher_signal,
            &config.amortized_adjoint.calibration,
        );
        if config.sync_diagnostics {
            profile.record_adjoint_calibration(
                f64::from(burn_pc::diagnostic_scalar_f32(calibration.loss.clone())),
                f64::from(burn_pc::diagnostic_scalar_f32(
                    calibration.cosine_alignment.clone(),
                )),
                f64::from(burn_pc::diagnostic_scalar_f32(
                    calibration.prediction_teacher_norm_ratio.clone(),
                )),
                f64::from(burn_pc::diagnostic_scalar_f32(
                    calibration.update_rms.clone(),
                )),
            );
        }
        calibration.feedback
    } else {
        let factor_activities = Tensor::cat(chunk.activities.iter().skip(1).cloned().collect(), 0)
            .reshape([layers, batch * streams * time, dim]);
        let terminal_activity_error = terminal_signal
            .mul_scalar(-1.0)
            .mul(normalization.reshape([1, 1, 1]));
        burn_pc::kolen_pollack_feedback_update_batched(
            chunk.feedback,
            factor_activities,
            terminal_activity_error,
            &config.direct_feedback,
        )
    };

    let report = LocalPredictiveCodingStepReport {
        solver: LocalPredictiveCodingSolver::DirectKolenPollack,
        inference_steps: config.inference.steps,
        factors: chunk.factors,
        local_vjp_calls,
        global_backward_calls: 0,
        gradient_tensors: grads.len() + 3,
        direct_forward_updates: layers,
        feedback_parameter_updates: layers,
        adjoint_teacher_updates: usize::from(teacher_due).saturating_mul(layers),
        adjoint_local_updates: usize::from(!teacher_due).saturating_mul(layers),
        parameter_updates: 2,
        energy_before,
        energy_after,
        elapsed_ns: started.elapsed().as_nanos(),
    };
    validate_step_execution_contract(config, &report);
    profile.record(report);
    LocalPredictiveCodingDerivatives {
        grads,
        loss: chunk.loss,
        supervised_tokens: chunk.supervised_tokens,
        terminal_state: chunk.terminal_state,
        dkp_feedback: Some(Tensor::<B, 3>::from_inner(updated_feedback)),
        report,
    }
}

fn fixed_prediction_train_step<B: AutodiffBackend>(
    plain: &DragonModel<B::InnerBackend>,
    context: FixedPredictionContext<B::InnerBackend>,
    config: &LocalPredictiveCodingConfig,
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
        criterion,
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
    let terminal = criterion.parameter_factor(plain, terminal_hidden);
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
        direct_forward_updates: 0,
        feedback_parameter_updates: 0,
        adjoint_teacher_updates: 0,
        adjoint_local_updates: 0,
        parameter_updates: 1,
        energy_before: None,
        energy_after: None,
        elapsed_ns: started.elapsed().as_nanos(),
    };
    validate_step_execution_contract(config, &report);
    profile.record(report);
    LocalPredictiveCodingDerivatives {
        grads,
        loss: Tensor::<B, 1>::from_inner(terminal.loss),
        supervised_tokens: Tensor::<B, 1>::from_inner(terminal.supervised_tokens),
        terminal_state: ModelState::<B>::from_inner_cloned(terminal_state),
        dkp_feedback: None,
        report,
    }
}

fn reconstruct_error_activities<B: Backend>(
    model: &DragonModel<B>,
    clamped_activity: Tensor<B, 4>,
    errors: &[Tensor<B, 4>],
    initial_rhos: &[Option<Tensor<B, 4>>],
    neuron_mask: Option<&Tensor<B, 4>>,
    activity_mask: Option<&Tensor<B, 4>>,
) -> (Vec<Tensor<B, 4>>, Vec<DragonPredictiveCodingLayerTrace<B>>)
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    assert_eq!(errors.len(), model.predictive_coding_layer_count());
    assert_eq!(initial_rhos.len(), errors.len());
    let mut activities = Vec::with_capacity(errors.len() + 1);
    let mut traces = Vec::with_capacity(errors.len());
    activities.push(clamped_activity);
    for layer in 0..errors.len() {
        let mut trace = model
            .predictive_coding_forward_layer_with_recurrent_state(
                activities[layer].clone(),
                layer,
                initial_rhos[layer].clone(),
                neuron_mask.cloned(),
                activity_mask.cloned(),
            )
            .expect("validated ePC layer factor");
        trace.next = apply_activity_mask(trace.next, activity_mask);
        let activity =
            apply_activity_mask(trace.next.clone() + errors[layer].clone(), activity_mask).detach();
        traces.push(trace);
        activities.push(activity);
    }
    (activities, traces)
}

fn error_coordinate_energy<B: Backend>(
    model: &DragonModel<B>,
    activities: &[Tensor<B, 4>],
    errors: &[Tensor<B, 4>],
    criterion: &LocalPcTerminalCriterion<B>,
    config: &LocalPredictiveCodingConfig,
) -> Tensor<B, 1>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let hidden = model.predictive_coding_hidden_from_activity(
        activities.last().expect("terminal ePC activity").clone(),
    );
    let terminal = criterion.activity_factor(model, hidden);
    let normalization = terminal.normalization.clamp_min(1.0);
    let error_energy = Tensor::cat(errors.to_vec(), 0)
        .square()
        .sum()
        .div(normalization)
        .mul_scalar(0.5 * config.prediction_precision)
        .reshape([1]);
    (terminal.loss + error_energy).mul_scalar(factor_scale(config, errors.len() + 1))
}

fn error_equilibrium_train_step<B: AutodiffBackend>(
    plain: &DragonModel<B::InnerBackend>,
    context: FixedPredictionContext<B::InnerBackend>,
    config: &LocalPredictiveCodingConfig,
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
        criterion,
        activities: feedforward_activities,
        traces: feedforward_traces,
        neuron_mask,
        activity_mask,
        terminal_state,
        factors,
        scale,
    } = context;
    let layers = feedforward_traces.len();
    let clamped_activity = feedforward_activities[0].clone();
    let initial_rhos = feedforward_traces
        .iter()
        .map(|trace| trace.initial_rho.clone())
        .collect::<Vec<_>>();
    let feedforward_hidden = plain.predictive_coding_hidden_from_activity(
        feedforward_activities
            .last()
            .expect("terminal feedforward activity")
            .clone(),
    );
    let feedforward_terminal = criterion.activity_factor(plain, feedforward_hidden);
    let supervised_tokens = feedforward_terminal.normalization.clone().reshape([1]);
    let feedforward_loss = feedforward_terminal.loss.clone();
    let mut errors = feedforward_activities
        .iter()
        .skip(1)
        .map(Tensor::zeros_like)
        .collect::<Vec<_>>();
    let energy_before = config
        .sync_diagnostics
        .then(|| burn_pc::diagnostic_scalar_f32(feedforward_loss.clone().mul_scalar(scale)) as f64);
    let mut local_vjp_calls = 1usize;

    for inference_step in 0..config.inference.steps {
        let (activities, traces, terminal) = if inference_step == 0 {
            (
                feedforward_activities.clone(),
                feedforward_traces.clone(),
                feedforward_terminal.clone(),
            )
        } else {
            let (activities, traces) = reconstruct_error_activities(
                plain,
                clamped_activity.clone(),
                &errors,
                &initial_rhos,
                neuron_mask.as_ref(),
                activity_mask.as_ref(),
            );
            let terminal_activity = activities.last().expect("terminal ePC activity");
            let hidden = plain.predictive_coding_hidden_from_activity(terminal_activity.clone());
            let terminal = criterion.activity_factor(plain, hidden);
            local_vjp_calls = local_vjp_calls.saturating_add(1);
            (activities, traces, terminal)
        };
        let terminal_activity = activities.last().expect("terminal ePC activity");
        // `head_activity_vjp` returns a mean derivative. Error optimization in
        // ePC uses a sum over independent examples, so restore the token sum;
        // parameter derivatives are normalized again below.
        let mut downstream = (terminal.grad_hidden * terminal.normalization.reshape([1, 1, 1]))
            .reshape(terminal_activity.shape());
        let mut downstream_by_layer = errors.iter().map(Tensor::zeros_like).collect::<Vec<_>>();
        for layer in (0..layers).rev() {
            downstream_by_layer[layer] =
                apply_activity_mask(downstream.clone(), activity_mask.as_ref());
            downstream = layer_activity_vjp(
                plain,
                layer,
                &traces[layer],
                downstream,
                neuron_mask.as_ref(),
                activity_mask.as_ref(),
            );
            local_vjp_calls = local_vjp_calls.saturating_add(1);
        }
        errors = errors
            .into_iter()
            .zip(downstream_by_layer)
            .map(|(error, downstream)| {
                let gradient = burn_pc::epc_error_gradient(
                    error.clone().mul_scalar(config.prediction_precision),
                    downstream,
                )
                .mul_scalar(scale);
                apply_activity_mask(
                    burn_pc::pc_sgd_update(error, gradient, &config.inference),
                    activity_mask.as_ref(),
                )
                .detach()
            })
            .collect();
    }

    let (activities, traces) = reconstruct_error_activities(
        plain,
        clamped_activity,
        &errors,
        &initial_rhos,
        neuron_mask.as_ref(),
        activity_mask.as_ref(),
    );
    let energy_after = config.sync_diagnostics.then(|| {
        burn_pc::diagnostic_scalar_f32(error_coordinate_energy(
            plain,
            &activities,
            &errors,
            &criterion,
            config,
        )) as f64
    });
    let terminal_activity = activities.last().expect("terminal ePC activity");
    let terminal_hidden = plain.predictive_coding_hidden_from_activity(terminal_activity.clone());
    let terminal = criterion.parameter_factor(plain, terminal_hidden);
    let normalization = terminal.supervised_tokens.clone().clamp_min(1.0);
    let local_prediction_grads = Tensor::cat(
        errors
            .iter()
            .map(|error| {
                error
                    .clone()
                    .mul_scalar(-config.prediction_precision * scale)
                    / normalization.clone().reshape([1, 1, 1, 1])
            })
            .collect(),
        0,
    );
    let batched_trace = concatenate_traces(&traces);
    let mut batched_vjp = layer_parameter_vjp(
        plain,
        0,
        &batched_trace,
        local_prediction_grads,
        neuron_mask.as_ref(),
        activity_mask.as_ref(),
    );
    local_vjp_calls = local_vjp_calls.saturating_add(layers);

    let shared_scale = if matches!(
        config.parameterization,
        burn_pc::PcParameterizationKind::MuPc
    ) {
        burn_pc::shared_reuse_scale(layers, config.shared_reuse_reduction)
            .expect("validated shared reuse geometry") as f32
    } else {
        1.0
    };
    batched_vjp.grad_encoder = batched_vjp.grad_encoder.mul_scalar(shared_scale);
    batched_vjp.grad_encoder_v = batched_vjp.grad_encoder_v.mul_scalar(shared_scale);
    batched_vjp.grad_decoder = batched_vjp.grad_decoder.mul_scalar(shared_scale);
    batched_vjp.grad_norm_gamma = batched_vjp.grad_norm_gamma.mul_scalar(shared_scale);
    batched_vjp.grad_norm_beta = batched_vjp.grad_norm_beta.mul_scalar(shared_scale);
    batched_vjp.grad_norm_alpha = batched_vjp.grad_norm_alpha.mul_scalar(shared_scale);
    batched_vjp.grad_norm_shift = batched_vjp.grad_norm_shift.mul_scalar(shared_scale);

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
    grads.register(
        parameter_ids.lm_head,
        terminal.grad_lm_head.mul_scalar(scale),
    );

    let report = LocalPredictiveCodingStepReport {
        solver: LocalPredictiveCodingSolver::ErrorEquilibrium,
        inference_steps: config.inference.steps,
        factors,
        local_vjp_calls,
        global_backward_calls: 0,
        gradient_tensors: grads.len(),
        direct_forward_updates: 0,
        feedback_parameter_updates: 0,
        adjoint_teacher_updates: 0,
        adjoint_local_updates: 0,
        parameter_updates: 1,
        energy_before,
        energy_after,
        elapsed_ns: started.elapsed().as_nanos(),
    };
    validate_step_execution_contract(config, &report);
    profile.record(report);
    LocalPredictiveCodingDerivatives {
        grads,
        loss: Tensor::<B, 1>::from_inner(feedforward_loss),
        supervised_tokens: Tensor::<B, 1>::from_inner(supervised_tokens),
        terminal_state: ModelState::<B>::from_inner_cloned(terminal_state),
        dkp_feedback: None,
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
    config: &LocalPredictiveCodingConfig,
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
        direct_forward_updates: 0,
        feedback_parameter_updates: 0,
        adjoint_teacher_updates: 0,
        adjoint_local_updates: 0,
        parameter_updates: 1,
        energy_before: None,
        energy_after: None,
        elapsed_ns: started.elapsed().as_nanos(),
    };
    validate_step_execution_contract(config, &report);
    profile.record(report);
    LocalPredictiveCodingDerivatives {
        grads,
        loss: Tensor::<B, 1>::from_inner(terminal_loss),
        supervised_tokens: Tensor::<B, 1>::from_inner(supervised_tokens),
        terminal_state: ModelState::<B>::from_inner_cloned(terminal_state),
        dkp_feedback: None,
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

/// Activity state retained across the interleaved inference/parameter phases
/// of incremental predictive coding. All activities are plain-backend tensors,
/// so this schedule never retains a global autodiff graph between updates.
pub(super) struct IncrementalPredictiveCodingChunk<B: AutodiffBackend> {
    inputs: Tensor<B::InnerBackend, 2, Int>,
    targets: Tensor<B::InnerBackend, 2, Int>,
    loss_mask: Option<Tensor<B::InnerBackend, 2, Int>>,
    activities: Vec<Tensor<B::InnerBackend, 4>>,
    initial_rhos: Vec<Option<Tensor<B::InnerBackend, 4>>>,
    pub(super) loss: Tensor<B, 1>,
    pub(super) supervised_tokens: Tensor<B, 1>,
    pub(super) terminal_state: ModelState<B>,
    factors: usize,
    scale: f32,
}

pub(super) struct IncrementalPredictiveCodingParameterDerivatives {
    pub(super) grads: GradientsParams,
    pub(super) local_vjp_calls: usize,
    pub(super) gradient_tensors: usize,
}

pub(super) fn prepare_incremental_predictive_coding_chunk<B: AutodiffBackend>(
    model: &DragonModel<B>,
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
    loss_mask: Option<Tensor<B, 2, Int>>,
    initial_state: ModelState<B>,
    config: &LocalPredictiveCodingConfig,
) -> IncrementalPredictiveCodingChunk<B>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let plain = model.valid();
    plain
        .predictive_coding_support()
        .expect("validated incremental predictive-coding model");
    let mut terminal_state = initial_state.inner_cloned();
    assert_eq!(
        terminal_state.layers.len(),
        plain.predictive_coding_layer_count(),
        "incremental PC recurrent-state layer count must match the model"
    );
    let block_time = inputs.shape().dims::<2>()[1];
    let inputs = inputs.inner();
    let targets = targets.inner();
    let loss_mask = loss_mask.map(Tensor::inner);
    let layers = plain.predictive_coding_layer_count();
    let factors = layers + 1;
    let scale = factor_scale(config, factors);
    let mut activities = Vec::with_capacity(layers + 1);
    let mut initial_rhos = Vec::with_capacity(layers);
    activities.push(plain.predictive_coding_initial_activity(inputs.clone()));
    for layer in 0..layers {
        let trace = plain
            .predictive_coding_forward_layer_with_recurrent_state(
                activities[layer].clone(),
                layer,
                terminal_state.layers[layer].rho.clone(),
                None,
                None,
            )
            .expect("validated incremental local PC layer factor");
        initial_rhos.push(trace.initial_rho.clone());
        terminal_state.layers[layer].rho = Some(plain.predictive_coding_terminal_rho(&trace));
        terminal_state.layers[layer].rho_norm = None;
        terminal_state.layers[layer].sequence_aux = None;
        activities.push(trace.next.detach());
    }
    terminal_state.position = terminal_state.position.saturating_add(block_time);
    terminal_state.detach_in_place();
    let hidden = plain.predictive_coding_hidden_from_activity(
        activities
            .last()
            .expect("incremental PC terminal activity")
            .clone(),
    );
    let [batch, time] = targets.shape().dims::<2>();
    let supervised_tokens = loss_mask.clone().map_or_else(
        || Tensor::<B::InnerBackend, 2>::ones([batch, time], &targets.device()).sum(),
        |mask| mask.float().sum(),
    );
    let terminal =
        plain.predictive_coding_head_activity_vjp(hidden, targets.clone(), loss_mask.clone());

    IncrementalPredictiveCodingChunk {
        inputs,
        targets,
        loss_mask,
        activities,
        initial_rhos,
        loss: Tensor::<B, 1>::from_inner(terminal.loss),
        supervised_tokens: Tensor::<B, 1>::from_inner(supervised_tokens.reshape([1])),
        terminal_state: ModelState::<B>::from_inner_cloned(terminal_state),
        factors,
        scale,
    }
}

/// Perform one activity-inference phase against the current parameter values.
/// Inferred activities persist into the next call; the clamped token embedding
/// is refreshed because its parameter may have changed in the preceding phase.
pub(super) fn incremental_predictive_coding_infer<B: AutodiffBackend>(
    model: &DragonModel<B>,
    state: &mut IncrementalPredictiveCodingChunk<B>,
    config: &LocalPredictiveCodingConfig,
) -> usize
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let plain = model.valid();
    state.activities[0] = plain.predictive_coding_initial_activity(state.inputs.clone());
    let layers = plain.predictive_coding_layer_count();
    let trace = forward_trace_batch(&plain, &state.activities, &state.initial_rhos, None, None);
    let hidden = plain.predictive_coding_hidden_from_activity(
        state
            .activities
            .last()
            .expect("incremental PC terminal activity")
            .clone(),
    );
    let terminal = plain.predictive_coding_head_activity_vjp(
        hidden,
        state.targets.clone(),
        state.loss_mask.clone(),
    );
    let terminal_grad = (terminal.grad_hidden * terminal.normalization.reshape([1, 1, 1])).reshape(
        state
            .activities
            .last()
            .expect("incremental PC terminal activity")
            .shape(),
    );
    let [batch, streams, time, dim] = state.activities[0].shape().dims::<4>();
    let mut local_vjp_calls = 1usize;

    match config.solver {
        LocalPredictiveCodingSolver::SynchronousEquilibrium => {
            let inferred = Tensor::cat(state.activities.iter().skip(1).cloned().collect(), 0);
            let errors = prediction_error(
                trace.next.clone(),
                inferred,
                config.prediction_precision,
                state.scale,
            );
            let internal_child_grads = (layers > 1).then(|| {
                layer_activity_vjp(
                    &plain,
                    0,
                    &slice_trace_batch(&trace, batch, layers * batch),
                    slice_batch(errors.clone(), batch, layers * batch),
                    None,
                    None,
                )
            });
            local_vjp_calls = local_vjp_calls.saturating_add(layers.saturating_sub(1));
            let mut updates = Vec::with_capacity(layers);
            for (activity_index, activity) in
                state.activities.iter().enumerate().take(layers + 1).skip(1)
            {
                let own_offset = (activity_index - 1) * batch;
                let own =
                    slice_batch(errors.clone(), own_offset, own_offset + batch).mul_scalar(-1.0);
                let child = if activity_index == layers {
                    terminal_grad.clone().mul_scalar(state.scale)
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
            for (activity, update) in state.activities.iter_mut().skip(1).zip(updates) {
                *activity = update.detach();
            }
        }
        LocalPredictiveCodingSolver::ReverseGaussSeidel => {
            for activity_index in (1..=layers).rev() {
                let own_offset = (activity_index - 1) * batch;
                let own = prediction_error(
                    slice_batch(trace.next.clone(), own_offset, own_offset + batch),
                    state.activities[activity_index].clone(),
                    config.prediction_precision,
                    state.scale,
                )
                .mul_scalar(-1.0);
                let child = if activity_index == layers {
                    terminal_grad.clone().mul_scalar(state.scale)
                } else {
                    let child_offset = activity_index * batch;
                    let child_error = prediction_error(
                        slice_batch(trace.next.clone(), child_offset, child_offset + batch),
                        state.activities[activity_index + 1].clone(),
                        config.prediction_precision,
                        state.scale,
                    );
                    local_vjp_calls = local_vjp_calls.saturating_add(1);
                    layer_activity_vjp(
                        &plain,
                        activity_index,
                        &slice_trace_batch(&trace, child_offset, child_offset + batch),
                        child_error,
                        None,
                        None,
                    )
                };
                state.activities[activity_index] = burn_pc::pc_sgd_update(
                    state.activities[activity_index].clone(),
                    own + child,
                    &config.inference,
                )
                .detach();
            }
        }
        LocalPredictiveCodingSolver::ErrorEquilibrium
        | LocalPredictiveCodingSolver::FixedPrediction
        | LocalPredictiveCodingSolver::LayerLocalPrediction
        | LocalPredictiveCodingSolver::DirectKolenPollack
        | LocalPredictiveCodingSolver::AmortizedAdjoint => {
            unreachable!("validated incremental PC solver")
        }
    }
    local_vjp_calls
}

pub(super) fn incremental_predictive_coding_parameter_derivatives<B: AutodiffBackend>(
    model: &DragonModel<B>,
    state: &IncrementalPredictiveCodingChunk<B>,
    config: &LocalPredictiveCodingConfig,
) -> IncrementalPredictiveCodingParameterDerivatives
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let parameter_ids = model
        .predictive_coding_parameter_ids()
        .expect("validated incremental predictive-coding model");
    let plain = model.valid();
    let layers = plain.predictive_coding_layer_count();
    let trace = forward_trace_batch(&plain, &state.activities, &state.initial_rhos, None, None);
    let hidden = plain.predictive_coding_hidden_from_activity(
        state
            .activities
            .last()
            .expect("incremental PC terminal activity")
            .clone(),
    );
    let terminal =
        plain.predictive_coding_head_vjp(hidden, state.targets.clone(), state.loss_mask.clone());
    let normalization = terminal.supervised_tokens.clone().clamp_min(1.0);
    let errors = prediction_error_gradient(
        trace.next.clone(),
        Tensor::cat(state.activities.iter().skip(1).cloned().collect(), 0),
        config.prediction_precision,
        state.scale,
        normalization,
    );
    let batched_vjp = layer_parameter_vjp(&plain, 0, &trace, errors, None, None);
    let [batch, streams, time, dim] = state.activities[0].shape().dims::<4>();
    let initial_grad = batched_vjp
        .grad_input
        .slice([0..batch, 0..streams, 0..time, 0..dim]);
    let initial_vjp = plain.predictive_coding_initial_vjp(state.inputs.clone(), initial_grad);
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
    grads.register(
        parameter_ids.lm_head,
        terminal.grad_lm_head.mul_scalar(state.scale),
    );
    let gradient_tensors = grads.len();
    IncrementalPredictiveCodingParameterDerivatives {
        grads,
        local_vjp_calls: layers.saturating_add(2),
        gradient_tensors,
    }
}

pub(super) fn incremental_predictive_coding_energy<B: AutodiffBackend>(
    model: &DragonModel<B>,
    state: &IncrementalPredictiveCodingChunk<B>,
    config: &LocalPredictiveCodingConfig,
) -> f64
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let plain = model.valid();
    burn_pc::diagnostic_scalar_f32(total_energy(
        &plain,
        &state.activities,
        &state.initial_rhos,
        state.targets.clone(),
        state.loss_mask.clone(),
        config,
        (None, None),
    )) as f64
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
    if matches!(
        config.solver,
        LocalPredictiveCodingSolver::ErrorEquilibrium
            | LocalPredictiveCodingSolver::FixedPrediction
    ) {
        let criterion = LocalPcTerminalCriterion::next_token(
            targets.clone().inner(),
            loss_mask.clone().map(Tensor::inner),
        );
        let (plain, context) = prepare_fixed_prediction_context::<B>(
            model,
            inputs.inner(),
            criterion,
            initial_state.map(|state| state.inner_cloned()),
            context_masks.neuron.map(Tensor::inner),
            context_masks.activity.map(Tensor::inner),
            config,
        );
        return match config.solver {
            LocalPredictiveCodingSolver::ErrorEquilibrium => {
                error_equilibrium_train_step::<B>(&plain, context, config, started, profile)
            }
            LocalPredictiveCodingSolver::FixedPrediction => {
                fixed_prediction_train_step::<B>(&plain, context, config, started, profile)
            }
            _ => unreachable!(),
        };
    }
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
            config,
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
        LocalPredictiveCodingSolver::ErrorEquilibrium => {
            unreachable!("error-equilibrium solver returns before state activity inference")
        }
        LocalPredictiveCodingSolver::FixedPrediction => {
            unreachable!("fixed-prediction solver returns before activity inference")
        }
        LocalPredictiveCodingSolver::LayerLocalPrediction => {
            unreachable!("layer-local solver returns before activity inference")
        }
        LocalPredictiveCodingSolver::DirectKolenPollack => {
            unreachable!("DKP uses the optimizer-owned two-phase schedule")
        }
        LocalPredictiveCodingSolver::AmortizedAdjoint => {
            unreachable!("amortized adjoint uses its run-scoped feedback schedule")
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
        direct_forward_updates: 0,
        feedback_parameter_updates: 0,
        adjoint_teacher_updates: 0,
        adjoint_local_updates: 0,
        parameter_updates: 1,
        energy_before,
        energy_after,
        elapsed_ns: started.elapsed().as_nanos(),
    };
    validate_step_execution_contract(config, &report);
    profile.record(report);
    LocalPredictiveCodingDerivatives {
        grads,
        loss,
        supervised_tokens: Tensor::<B, 1>::from_inner(terminal.supervised_tokens),
        terminal_state: ModelState::<B>::from_inner_cloned(terminal_state),
        dkp_feedback: None,
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
    config
        .direct_feedback
        .validate()
        .map_err(|error| error.to_string())?;
    config
        .tied_consensus
        .validate()
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
        config.parameterization,
        burn_pc::PcParameterizationKind::MuPc
    ) && !matches!(config.solver, LocalPredictiveCodingSolver::ErrorEquilibrium)
    {
        return Err(
            "local_predictive_coding_derivatives parameterization=mu_pc requires solver=error_equilibrium"
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
        LocalPredictiveCodingSolver::DirectKolenPollack
            | LocalPredictiveCodingSolver::AmortizedAdjoint
    ) {
        return Err(
            "feedback-bank solvers require LanguageTrainModel rather than the derivative-only API"
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

    #[test]
    fn identity_dkp_feedback_aligns_every_factor_in_the_shared_residual_basis() {
        let device = Default::default();
        let feedback = initial_dkp_feedback::<PlainBackend>(
            3,
            4,
            burn_pc::PcFeedbackInitialization::Identity,
            &device,
        );
        let values = feedback
            .into_data()
            .to_vec::<f32>()
            .expect("identity feedback values");
        for layer in 0..3 {
            for row in 0..4 {
                for column in 0..4 {
                    let expected = f32::from(row == column);
                    assert_eq!(values[layer * 16 + row * 4 + column], expected);
                }
            }
        }
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
    fn fixed_prediction_verifier_gradients_match_global_backpropagation() {
        let device = Default::default();
        TestBackend::seed(&device, 20260808);
        let model = crate::train::test_support::deterministic_matrix_parameters(model(&device));
        let inputs = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 2, 3, 4, 5, 6, 7, 8], [2, 4]),
            &device,
        );
        let positions =
            Tensor::<TestBackend, 1, Int>::from_data(TensorData::new(vec![1_i64, 2], [2]), &device);
        let support = Tensor::<TestBackend, 2>::from_floats(
            [
                [
                    1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                [
                    0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
            ],
            &device,
        );
        let valid = Tensor::<TestBackend, 2>::from_floats(
            [
                [
                    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                [
                    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
            ],
            &device,
        );
        let weights = Tensor::<TestBackend, 1>::from_floats([0.25, 0.75], &device);
        let reference_criterion = LocalPcTerminalCriterion::CategoricalSetAtPositions {
            positions: positions.clone(),
            support_action_mask: support.clone(),
            valid_action_mask: valid.clone(),
            row_weights: weights.clone(),
            eps: 1.0e-12,
        };
        let reference_loss = reference_criterion
            .verifier_autodiff_loss(model.forward(inputs.clone()))
            .expect("verifier autodiff loss");
        let reference_grads =
            GradientsParams::from_grads(reference_loss.clone().backward(), &model);

        let local = local_predictive_coding_verifier_train_step(
            &model,
            verifier::PreparedRuliadVerifierTerminal {
                inputs: inputs.inner(),
                criterion: LocalPcTerminalCriterion::CategoricalSetAtPositions {
                    positions: positions.inner(),
                    support_action_mask: support.inner(),
                    valid_action_mask: valid.inner(),
                    row_weights: weights.inner(),
                    eps: 1.0e-12,
                },
                semantic_states: 2,
                decision_rows: 2,
            },
            &LocalPredictiveCodingConfig {
                solver: LocalPredictiveCodingSolver::FixedPrediction,
                factor_reduction: PredictiveCodingFactorReduction::Sum,
                ..LocalPredictiveCodingConfig::default()
            },
            &LocalPredictiveCodingProfile::default(),
        );
        let reference_loss = burn_pc::diagnostic_scalar_f32(reference_loss.inner());
        let local_loss = burn_pc::diagnostic_scalar_f32(local.loss.inner());
        assert!(
            (reference_loss - local_loss).abs() < 1.0e-6,
            "verifier loss mismatch: reference={reference_loss} local={local_loss}"
        );

        let ids = model
            .predictive_coding_parameter_ids()
            .expect("supported PC model");
        macro_rules! assert_gradient_close {
            ($name:literal, $id:expr, $rank:literal) => {{
                let actual = local.grads.get::<PlainBackend, $rank>($id).expect(concat!(
                    "local ",
                    $name,
                    " gradient"
                ));
                if let Some(expected) = reference_grads.get::<PlainBackend, $rank>($id) {
                    let max_error = max_abs_diff(expected.clone(), actual);
                    let reference_scale = expected
                        .abs()
                        .max()
                        .to_data()
                        .convert::<f32>()
                        .into_vec::<f32>()
                        .expect("gradient scale")[0]
                        .max(1.0e-7);
                    assert!(
                        max_error / reference_scale < 2.0e-4,
                        "{} relative max gradient error: {}",
                        $name,
                        max_error / reference_scale
                    );
                } else {
                    let actual_scale = actual
                        .abs()
                        .max()
                        .to_data()
                        .convert::<f32>()
                        .into_vec::<f32>()
                        .expect("inactive gradient scale")[0];
                    assert!(
                        actual_scale < 1.0e-8,
                        "{} should be an explicit zero gradient, got {}",
                        $name,
                        actual_scale
                    );
                }
            }};
        }
        assert_gradient_close!("embedding", ids.embedding, 2);
        assert_gradient_close!("shared encoder", ids.encoder, 3);
        assert_gradient_close!("shared value encoder", ids.encoder_v, 3);
        assert_gradient_close!("shared decoder", ids.decoder, 2);
        assert_gradient_close!("norm gamma", ids.norm_gamma, 1);
        assert_gradient_close!("norm beta", ids.norm_beta, 1);
        assert_gradient_close!("norm alpha", ids.norm_alpha, 1);
        assert_gradient_close!("norm shift", ids.norm_shift, 1);
        assert_gradient_close!("language head", ids.lm_head, 2);
        assert_eq!(local.report.global_backward_calls, 0);
    }

    #[test]
    fn amortized_adjoint_exact_anchor_matches_fixed_prediction() {
        let device = Default::default();
        TestBackend::seed(&device, 20260808);
        let model = crate::train::test_support::deterministic_matrix_parameters(model(&device));
        let (inputs, targets) = batch(&device);
        let fixed = local_predictive_coding_train_step(
            &model,
            inputs.clone(),
            targets.clone(),
            None,
            &LocalPredictiveCodingConfig {
                solver: LocalPredictiveCodingSolver::FixedPrediction,
                factor_reduction: PredictiveCodingFactorReduction::Sum,
                ..LocalPredictiveCodingConfig::default()
            },
            &LocalPredictiveCodingProfile::default(),
        );
        let amortized = amortized_adjoint_predictive_coding_train_step(
            &model,
            inputs,
            targets,
            None,
            model.init_state_ephemeral(),
            None,
            0,
            &LocalPredictiveCodingConfig {
                solver: LocalPredictiveCodingSolver::AmortizedAdjoint,
                factor_reduction: PredictiveCodingFactorReduction::Sum,
                direct_feedback: burn_pc::PcDirectFeedbackConfig {
                    initialization: burn_pc::PcFeedbackInitialization::Identity,
                    ..burn_pc::PcDirectFeedbackConfig::default()
                },
                amortized_adjoint: burn_pc::PcAmortizedAdjointConfig {
                    enabled: true,
                    teacher_every_updates: 1,
                    ..burn_pc::PcAmortizedAdjointConfig::default()
                },
                ..LocalPredictiveCodingConfig::default()
            },
            &LocalPredictiveCodingProfile::default(),
        );
        let fixed_loss = burn_pc::diagnostic_scalar_f32(fixed.loss.inner());
        let amortized_loss = burn_pc::diagnostic_scalar_f32(amortized.loss.inner());
        assert!(
            (fixed_loss - amortized_loss).abs() < 1.0e-6,
            "fixed={fixed_loss} amortized={amortized_loss}"
        );

        let ids = model
            .predictive_coding_parameter_ids()
            .expect("supported PC model");
        macro_rules! assert_gradient_close {
            ($name:literal, $id:expr, $rank:literal) => {{
                let fixed_gradient = fixed.grads.get::<PlainBackend, $rank>($id).expect(concat!(
                    "fixed ",
                    $name,
                    " gradient"
                ));
                let amortized_gradient = amortized
                    .grads
                    .get::<PlainBackend, $rank>($id)
                    .expect(concat!("amortized ", $name, " gradient"));
                let max_error = max_abs_diff(fixed_gradient.clone(), amortized_gradient);
                let reference_scale = fixed_gradient
                    .abs()
                    .max()
                    .to_data()
                    .convert::<f32>()
                    .into_vec::<f32>()
                    .expect("gradient scale")[0]
                    .max(1.0e-7);
                assert!(
                    max_error / reference_scale < 2.0e-4,
                    "{} relative max gradient error: {}",
                    $name,
                    max_error / reference_scale
                );
            }};
        }
        assert_gradient_close!("embedding", ids.embedding, 2);
        assert_gradient_close!("shared encoder", ids.encoder, 3);
        assert_gradient_close!("shared value encoder", ids.encoder_v, 3);
        assert_gradient_close!("shared decoder", ids.decoder, 2);
        assert_gradient_close!("norm gamma", ids.norm_gamma, 1);
        assert_gradient_close!("norm beta", ids.norm_beta, 1);
        assert_gradient_close!("norm alpha", ids.norm_alpha, 1);
        assert_gradient_close!("norm shift", ids.norm_shift, 1);
        assert_gradient_close!("language head", ids.lm_head, 2);
        assert_eq!(amortized.report.global_backward_calls, 0);
        assert_eq!(amortized.report.adjoint_teacher_updates, 2);
        assert_eq!(amortized.report.parameter_updates, 1);
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
    fn error_equilibrium_descends_error_energy_without_global_backward() {
        let device = Default::default();
        TestBackend::seed(&device, 20260807);
        let model = model(&device);
        let config = LocalPredictiveCodingConfig {
            solver: LocalPredictiveCodingSolver::ErrorEquilibrium,
            inference: burn_pc::PcInferenceConfig {
                steps: 8,
                step_size: 0.01,
                max_grad_norm: None,
                gradient_norm_scope: burn_pc::PcGradientNormScope::PerRow,
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

        let before = report.energy_before.expect("initial ePC energy");
        let after = report.energy_after.expect("relaxed ePC energy");
        assert!(
            after < before,
            "error-coordinate inference must descend energy: before={before} after={after}"
        );
        assert_eq!(report.solver, LocalPredictiveCodingSolver::ErrorEquilibrium);
        assert_eq!(report.global_backward_calls, 0);
        assert_eq!(report.parameter_updates, 1);
        assert_eq!(report.gradient_tensors, 9);
    }

    #[test]
    fn error_equilibrium_keeps_transient_errors_out_of_stream_state() {
        let device = Default::default();
        TestBackend::seed(&device, 20260807);
        let model = model(&device);
        let (inputs, targets) = batch(&device);
        let fixed = local_predictive_coding_train_step(
            &model,
            inputs.clone(),
            targets.clone(),
            None,
            &LocalPredictiveCodingConfig {
                solver: LocalPredictiveCodingSolver::FixedPrediction,
                ..LocalPredictiveCodingConfig::default()
            },
            &LocalPredictiveCodingProfile::default(),
        );
        let epc = local_predictive_coding_train_step(
            &model,
            inputs,
            targets,
            None,
            &LocalPredictiveCodingConfig {
                solver: LocalPredictiveCodingSolver::ErrorEquilibrium,
                inference: burn_pc::PcInferenceConfig {
                    steps: 4,
                    step_size: 0.01,
                    max_grad_norm: None,
                    ..burn_pc::PcInferenceConfig::default()
                },
                ..LocalPredictiveCodingConfig::default()
            },
            &LocalPredictiveCodingProfile::default(),
        );

        assert_eq!(fixed.terminal_state.position, epc.terminal_state.position);
        assert_eq!(
            fixed.terminal_state.layers.len(),
            epc.terminal_state.layers.len()
        );
        for (layer, (fixed_layer, epc_layer)) in fixed
            .terminal_state
            .layers
            .into_iter()
            .zip(epc.terminal_state.layers)
            .enumerate()
        {
            match (fixed_layer.rho, epc_layer.rho) {
                (Some(fixed_rho), Some(epc_rho)) => {
                    let difference = max_abs_diff(fixed_rho.inner(), epc_rho.inner());
                    assert!(
                        difference < 1.0e-6,
                        "transient ePC errors changed layer {layer} stream rho: {difference}"
                    );
                }
                (None, None) => {}
                _ => panic!("solver controls disagree on layer {layer} rho presence"),
            }
        }
    }

    #[test]
    fn derivative_api_rejects_mu_pc_on_non_error_solver() {
        let device = Default::default();
        let model = model(&device);
        let (inputs, targets) = batch(&device);
        let error = local_predictive_coding_derivatives(
            &model,
            inputs,
            targets,
            None,
            &LocalPredictiveCodingConfig {
                solver: LocalPredictiveCodingSolver::FixedPrediction,
                parameterization: burn_pc::PcParameterizationKind::MuPc,
                ..LocalPredictiveCodingConfig::default()
            },
        )
        .expect_err("muPC must not silently alter a control solver");
        assert!(error.contains("requires solver=error_equilibrium"));
    }

    #[test]
    fn local_pc_derivatives_are_invariant_to_batch_duplication() {
        let device = Default::default();
        let model = model(&device);
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
        let configs = [
            (
                "state_equilibrium",
                LocalPredictiveCodingConfig {
                    inference: burn_pc::PcInferenceConfig {
                        steps: 4,
                        step_size: 0.05,
                        max_grad_norm: None,
                        ..burn_pc::PcInferenceConfig::default()
                    },
                    ..LocalPredictiveCodingConfig::default()
                },
            ),
            (
                "error_equilibrium",
                LocalPredictiveCodingConfig {
                    solver: LocalPredictiveCodingSolver::ErrorEquilibrium,
                    prediction_precision: 10.0,
                    inference: burn_pc::PcInferenceConfig {
                        steps: 1,
                        step_size: 0.1,
                        max_grad_norm: None,
                        ..burn_pc::PcInferenceConfig::default()
                    },
                    ..LocalPredictiveCodingConfig::default()
                },
            ),
        ];
        for (solver, config) in configs {
            let single = local_predictive_coding_train_step(
                &model,
                single_inputs.clone(),
                single_targets.clone(),
                None,
                &config,
                &LocalPredictiveCodingProfile::default(),
            );
            let doubled = local_predictive_coding_train_step(
                &model,
                double_inputs.clone(),
                double_targets.clone(),
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
                "{solver} encoder derivative changed after duplicating the batch: {encoder_diff}"
            );
            assert!(
                head_diff < 1.0e-5,
                "{solver} head derivative changed after duplicating the batch: {head_diff}"
            );
        }
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

    #[test]
    fn checkpoint_identity_covers_graph_and_inference_dynamics() {
        let config = LocalPredictiveCodingConfig::default();
        let manifest = dragon_predictive_coding_checkpoint_manifest(3, &config)
            .expect("predictive-coding manifest");
        assert_eq!(
            manifest.execution_contract,
            burn_pc::PcExecutionContract::strict_local()
        );

        let wider_graph = dragon_predictive_coding_checkpoint_manifest(4, &config)
            .expect("wider predictive-coding graph");
        assert_ne!(manifest.graph_digest, wider_graph.graph_digest);

        let mut identity_feedback = config.clone();
        identity_feedback.direct_feedback.initialization =
            burn_pc::PcFeedbackInitialization::Identity;
        let identity_feedback = dragon_predictive_coding_checkpoint_manifest(3, &identity_feedback)
            .expect("identity-feedback predictive-coding program");
        assert_eq!(manifest.graph_digest, identity_feedback.graph_digest);
        assert_ne!(manifest.program_digest, identity_feedback.program_digest);

        let mut changed = config;
        changed.inference.step_size = 0.1;
        let changed = dragon_predictive_coding_checkpoint_manifest(3, &changed)
            .expect("changed predictive-coding program");
        assert_eq!(manifest.graph_digest, changed.graph_digest);
        assert_ne!(manifest.program_digest, changed.program_digest);

        let verifier_terminal = LocalPredictiveCodingConfig {
            terminal_criterion:
                crate::config::LocalPredictiveCodingTerminalCriterion::RuliadVerifierSet,
            ..LocalPredictiveCodingConfig::default()
        };
        let verifier_terminal = dragon_predictive_coding_checkpoint_manifest(3, &verifier_terminal)
            .expect("verifier-terminal predictive-coding program");
        assert_eq!(manifest.graph_digest, verifier_terminal.graph_digest);
        assert_ne!(manifest.program_digest, verifier_terminal.program_digest);

        let calibrated_adjoint = LocalPredictiveCodingConfig {
            solver: LocalPredictiveCodingSolver::DirectKolenPollack,
            amortized_adjoint: burn_pc::PcAmortizedAdjointConfig {
                enabled: true,
                ..burn_pc::PcAmortizedAdjointConfig::default()
            },
            ..LocalPredictiveCodingConfig::default()
        };
        let calibrated_adjoint =
            dragon_predictive_coding_checkpoint_manifest(3, &calibrated_adjoint)
                .expect("calibrated-adjoint predictive-coding program");
        assert_eq!(manifest.graph_digest, calibrated_adjoint.graph_digest);
        assert_ne!(manifest.program_digest, calibrated_adjoint.program_digest);
    }
}
