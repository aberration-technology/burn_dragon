use std::cell::Cell;

use burn::optim::GradientsParams;
use burn::tensor::{Tensor, backend::AutodiffBackend};

use super::*;

#[derive(Debug, Clone)]
struct DragonAlmLinearization<B: Backend> {
    trace: DragonPredictiveCodingLayerTrace<B>,
    terminal_grad: Tensor<B, 4>,
}

fn unpack_activities<B: Backend>(
    clamped: &Tensor<B, 4>,
    inferred: Tensor<B, 4>,
    layers: usize,
) -> Vec<Tensor<B, 4>> {
    let [batch, streams, time, dim] = clamped.shape().dims::<4>();
    let mut activities = Vec::with_capacity(layers + 1);
    activities.push(clamped.clone());
    for layer in 0..layers {
        let start = layer * batch;
        activities.push(inferred.clone().slice([
            start..start + batch,
            0..streams,
            0..time,
            0..dim,
        ]));
    }
    activities
}

fn synchronized_metric<B: Backend>(enabled: bool, tensor: Tensor<B, 1>) -> Option<f64> {
    enabled.then(|| f64::from(burn_pc::diagnostic_scalar_f32(tensor)))
}

pub(super) fn augmented_lagrangian_train_step<B: AutodiffBackend>(
    plain: &DragonModel<B::InnerBackend>,
    context: LocalPredictiveCodingContext<B::InnerBackend>,
    config: &LocalPredictiveCodingConfig,
    started: Instant,
    profile: &LocalPredictiveCodingProfile,
) -> LocalPredictiveCodingDerivatives<B>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let LocalPredictiveCodingContext {
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
    let layers = traces.len();
    let clamped = activities
        .first()
        .expect("validated PC model has a clamped activity")
        .clone();
    let [batch, streams, time, dim] = clamped.shape().dims::<4>();
    let initial_rhos = traces
        .iter()
        .map(|trace| trace.initial_rho.clone())
        .collect::<Vec<_>>();
    let inferred = Tensor::cat(activities.iter().skip(1).cloned().collect(), 0);
    let initial_terminal = criterion.activity_factor(
        plain,
        plain.predictive_coding_hidden_from_activity(
            activities
                .last()
                .expect("validated PC model has a terminal activity")
                .clone(),
        ),
    );
    let feedforward_loss = initial_terminal.loss;
    let state = burn_pc::PcAlmState::new(
        inferred.clone(),
        Tensor::<B::InnerBackend, 4>::zeros(inferred.shape(), &inferred.device()),
    );
    let local_vjp_calls = Cell::new(0usize);

    let result = burn_pc::run_pc_alm_inference(
        state,
        &config.augmented_lagrangian,
        |inferred| {
            let current = unpack_activities(&clamped, inferred.clone(), layers);
            let trace = forward_trace_batch(
                plain,
                &current,
                &initial_rhos,
                neuron_mask.as_ref(),
                activity_mask.as_ref(),
            );
            let terminal_activity = current
                .last()
                .expect("validated PC model has a terminal activity");
            let terminal = criterion.activity_factor(
                plain,
                plain.predictive_coding_hidden_from_activity(terminal_activity.clone()),
            );
            local_vjp_calls.set(local_vjp_calls.get().saturating_add(1));
            let terminal_grad = (terminal.grad_hidden * terminal.normalization.reshape([1, 1, 1]))
                .reshape(terminal_activity.shape());
            burn_pc::PcAlmLinearization::new(
                inferred - trace.next.clone(),
                DragonAlmLinearization {
                    trace,
                    terminal_grad,
                },
            )
        },
        |_inferred, composite, linearization| {
            let internal_child_grads = (layers > 1).then(|| {
                local_vjp_calls.set(
                    local_vjp_calls
                        .get()
                        .saturating_add(layers.saturating_sub(1)),
                );
                layer_activity_vjp(
                    plain,
                    0,
                    &slice_trace_batch(&linearization.trace, batch, layers * batch),
                    slice_batch(composite.clone(), batch, layers * batch).mul_scalar(-1.0),
                    neuron_mask.as_ref(),
                    activity_mask.as_ref(),
                )
            });
            let mut gradients = Vec::with_capacity(layers);
            for layer in 0..layers {
                let offset = layer * batch;
                let own = slice_batch(composite.clone(), offset, offset + batch);
                let child = if layer + 1 == layers {
                    linearization.terminal_grad.clone()
                } else {
                    internal_child_grads
                        .as_ref()
                        .expect("non-terminal PC activity has a child factor")
                        .clone()
                        .slice([offset..offset + batch, 0..streams, 0..time, 0..dim])
                };
                gradients.push(apply_activity_mask(
                    (own + child).mul_scalar(scale),
                    activity_mask.as_ref(),
                ));
            }
            Tensor::cat(gradients, 0)
        },
    );

    let final_activities = unpack_activities(&clamped, result.state.activity.clone(), layers);
    let terminal_activity = final_activities
        .last()
        .expect("validated PC model has a terminal activity");
    let terminal = criterion.parameter_factor(
        plain,
        plain.predictive_coding_hidden_from_activity(terminal_activity.clone()),
    );
    let normalization = terminal.supervised_tokens.clone().clamp_min(1.0);
    let parameter_signal =
        result.composite_signal.clone().mul_scalar(-scale) / normalization.reshape([1, 1, 1, 1]);
    let batched_vjp = layer_parameter_vjp(
        plain,
        0,
        &result.final_context.trace,
        parameter_signal,
        neuron_mask.as_ref(),
        activity_mask.as_ref(),
    );
    local_vjp_calls.set(local_vjp_calls.get().saturating_add(layers));
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
    local_vjp_calls.set(local_vjp_calls.get().saturating_add(2));

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

    let sync = config.sync_diagnostics;
    let report = LocalPredictiveCodingStepReport {
        solver: LocalPredictiveCodingSolver::AugmentedLagrangian,
        inference_steps: result.metrics.primal_steps_run,
        dual_steps: result.metrics.dual_steps_run,
        factors,
        local_vjp_calls: local_vjp_calls.get(),
        temporal_state_vjp_calls: 0,
        fused_temporal_vjp_calls: 0,
        global_backward_calls: 0,
        gradient_tensors: grads.len(),
        direct_forward_updates: 0,
        feedback_parameter_updates: 0,
        adjoint_teacher_updates: 0,
        adjoint_local_updates: 0,
        parameter_updates: 1,
        energy_before: None,
        energy_after: None,
        grad_norm_mean: synchronized_metric(sync, result.metrics.primal_grad_norm),
        grad_norm_max: synchronized_metric(sync, result.metrics.primal_grad_norm_max),
        delta_rms_mean: synchronized_metric(sync, result.metrics.primal_delta_rms),
        clip_fraction_mean: synchronized_metric(sync, result.metrics.primal_clip_fraction),
        constraint_rms: synchronized_metric(sync, result.metrics.constraint_rms),
        dual_rms: synchronized_metric(sync, result.metrics.dual_rms),
        composite_signal_rms: synchronized_metric(sync, result.metrics.composite_signal_rms),
        elapsed_ns: started.elapsed().as_nanos(),
    };
    validate_step_execution_contract(config, &report);
    profile.record(report);
    LocalPredictiveCodingDerivatives {
        grads,
        loss: Tensor::<B, 1>::from_inner(feedforward_loss),
        supervised_tokens: Tensor::<B, 1>::from_inner(terminal.supervised_tokens),
        terminal_state: ModelState::<B>::from_inner_cloned(terminal_state),
        initial_rho_adjoints: split_optional_rho_adjoints(
            batched_vjp.grad_initial_rho,
            layers,
            batch,
        )
        .into_iter()
        .map(|adjoint| adjoint.map(Tensor::<B, 4>::from_inner))
        .collect(),
        dkp_feedback: None,
        report,
    }
}
