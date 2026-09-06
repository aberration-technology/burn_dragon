use burn::optim::GradientsParams;
use burn::tensor::Tensor;
use burn::tensor::backend::AutodiffBackend;
use burn_dragon_core::ModelState;
use burn_dragon_time::Instant;

use super::criterion::LocalPcTerminalActivityFactor;
use super::{
    LocalPredictiveCodingContext, LocalPredictiveCodingDerivatives, LocalPredictiveCodingProfile,
    LocalPredictiveCodingStepReport, apply_activity_mask, concatenate_traces, factor_scale,
    forward_trace_batch, layer_activity_vjp, layer_parameter_vjp, prediction_energy,
    prediction_error, prediction_error_gradient, register_terminal_head_derivatives, slice_batch,
    slice_trace_batch, validate_step_execution_contract,
};
use crate::config::{LocalPredictiveCodingConfig, LocalPredictiveCodingSolver};

fn equilibrium_energy<B: burn::tensor::backend::Backend>(
    model: &burn_dragon_core::DragonModel<B>,
    activities: &[Tensor<B, 4>],
    initial_rhos: &[Option<Tensor<B, 4>>],
    criterion: &super::LocalPcTerminalCriterion<B>,
    config: &LocalPredictiveCodingConfig,
    neuron_mask: Option<&Tensor<B, 4>>,
    activity_mask: Option<&Tensor<B, 4>>,
) -> Tensor<B, 1>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let trace = forward_trace_batch(model, activities, initial_rhos, neuron_mask, activity_mask);
    let inferred = Tensor::cat(activities.iter().skip(1).cloned().collect(), 0);
    let hidden = model.predictive_coding_hidden_from_activity(
        activities
            .last()
            .expect("terminal equilibrium activity")
            .clone(),
    );
    let terminal = criterion.activity_factor(model, hidden);
    (prediction_energy(
        trace.next,
        inferred,
        config.prediction_precision,
        terminal.normalization,
    ) + terminal.loss)
        .mul_scalar(factor_scale(
            config,
            model.predictive_coding_layer_count() + 1,
        ))
}

#[allow(clippy::too_many_arguments)]
fn synchronous_update<B: burn::tensor::backend::Backend>(
    model: &burn_dragon_core::DragonModel<B>,
    activities: &mut [Tensor<B, 4>],
    trace: &burn_dragon_core::DragonPredictiveCodingLayerTrace<B>,
    terminal: LocalPcTerminalActivityFactor<B>,
    config: &LocalPredictiveCodingConfig,
    scale: f32,
    neuron_mask: Option<&Tensor<B, 4>>,
    activity_mask: Option<&Tensor<B, 4>>,
) -> usize
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let layers = model.predictive_coding_layer_count();
    let [batch, streams, time, dim] = activities[0].shape().dims::<4>();
    let terminal_grad = (terminal.grad_hidden * terminal.normalization.reshape([1, 1, 1])).reshape(
        activities
            .last()
            .expect("terminal synchronous activity")
            .shape(),
    );
    let inferred = Tensor::cat(activities.iter().skip(1).cloned().collect(), 0);
    let errors = prediction_error(
        trace.next.clone(),
        inferred,
        config.prediction_precision,
        scale,
    );
    let internal_child_grads = (layers > 1).then(|| {
        layer_activity_vjp(
            model,
            0,
            &slice_trace_batch(trace, batch, layers * batch),
            slice_batch(errors.clone(), batch, layers * batch),
            neuron_mask,
            activity_mask,
        )
    });
    let mut updates = Vec::with_capacity(layers);
    for (activity_index, activity) in activities.iter().enumerate().take(layers + 1).skip(1) {
        let own_offset = (activity_index - 1) * batch;
        let own = slice_batch(errors.clone(), own_offset, own_offset + batch).mul_scalar(-1.0);
        let child = if activity_index == layers {
            terminal_grad.clone().mul_scalar(scale)
        } else {
            let offset = (activity_index - 1) * batch;
            internal_child_grads
                .as_ref()
                .expect("non-terminal synchronous activity has a child factor")
                .clone()
                .slice([offset..offset + batch, 0..streams, 0..time, 0..dim])
        };
        updates.push(apply_activity_mask(
            burn_pc::pc_sgd_update(activity.clone(), own + child, &config.inference),
            activity_mask,
        ));
    }
    for (activity, update) in activities.iter_mut().skip(1).zip(updates) {
        *activity = update.detach();
    }
    layers.saturating_sub(1)
}

#[allow(clippy::too_many_arguments)]
fn reverse_gauss_seidel_update<B: burn::tensor::backend::Backend>(
    model: &burn_dragon_core::DragonModel<B>,
    activities: &mut [Tensor<B, 4>],
    trace: &burn_dragon_core::DragonPredictiveCodingLayerTrace<B>,
    terminal: LocalPcTerminalActivityFactor<B>,
    config: &LocalPredictiveCodingConfig,
    scale: f32,
    neuron_mask: Option<&Tensor<B, 4>>,
    activity_mask: Option<&Tensor<B, 4>>,
) -> usize
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let layers = model.predictive_coding_layer_count();
    let [batch, _, _, _] = activities[0].shape().dims::<4>();
    let terminal_grad = (terminal.grad_hidden * terminal.normalization.reshape([1, 1, 1])).reshape(
        activities
            .last()
            .expect("terminal Gauss-Seidel activity")
            .shape(),
    );
    let mut vjp_calls = 0usize;
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
            vjp_calls = vjp_calls.saturating_add(1);
            layer_activity_vjp(
                model,
                activity_index,
                &slice_trace_batch(trace, child_offset, child_offset + batch),
                child_error,
                neuron_mask,
                activity_mask,
            )
        };
        activities[activity_index] = apply_activity_mask(
            burn_pc::pc_sgd_update(
                activities[activity_index].clone(),
                own + child,
                &config.inference,
            ),
            activity_mask,
        )
        .detach();
    }
    vjp_calls
}

pub(super) fn train_step<B: AutodiffBackend>(
    plain: &burn_dragon_core::DragonModel<B::InnerBackend>,
    context: LocalPredictiveCodingContext<B::InnerBackend>,
    config: &LocalPredictiveCodingConfig,
    started: Instant,
    profile: &LocalPredictiveCodingProfile,
) -> LocalPredictiveCodingDerivatives<B>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    debug_assert!(matches!(
        config.solver,
        LocalPredictiveCodingSolver::SynchronousEquilibrium
            | LocalPredictiveCodingSolver::ReverseGaussSeidel
    ));
    let LocalPredictiveCodingContext {
        parameter_ids,
        inputs,
        criterion,
        mut activities,
        traces: feedforward_traces,
        neuron_mask,
        activity_mask,
        terminal_state,
        factors,
        scale,
    } = context;
    let layers = feedforward_traces.len();
    let initial_rhos = feedforward_traces
        .iter()
        .map(|trace| trace.initial_rho.clone())
        .collect::<Vec<_>>();
    let feedforward_hidden = plain.predictive_coding_hidden_from_activity(
        activities
            .last()
            .expect("terminal feedforward equilibrium activity")
            .clone(),
    );
    let feedforward_terminal = criterion.activity_factor(plain, feedforward_hidden);
    let feedforward_loss = feedforward_terminal.loss.clone();
    let supervised_tokens = feedforward_terminal.normalization.clone().reshape([1]);
    let energy_before = config.sync_diagnostics.then(|| {
        burn_pc::diagnostic_scalar_f32(equilibrium_energy(
            plain,
            &activities,
            &initial_rhos,
            &criterion,
            config,
            neuron_mask.as_ref(),
            activity_mask.as_ref(),
        )) as f64
    });
    let mut feedforward_trace = Some(concatenate_traces(&feedforward_traces));
    let mut feedforward_terminal = Some(feedforward_terminal);
    let mut local_vjp_calls = 0usize;

    for _ in 0..config.inference.steps {
        let trace = feedforward_trace.take().unwrap_or_else(|| {
            forward_trace_batch(
                plain,
                &activities,
                &initial_rhos,
                neuron_mask.as_ref(),
                activity_mask.as_ref(),
            )
        });
        let terminal = feedforward_terminal.take().unwrap_or_else(|| {
            criterion.activity_factor(
                plain,
                plain.predictive_coding_hidden_from_activity(
                    activities
                        .last()
                        .expect("terminal inferred equilibrium activity")
                        .clone(),
                ),
            )
        });
        local_vjp_calls = local_vjp_calls.saturating_add(1);
        local_vjp_calls = local_vjp_calls.saturating_add(match config.solver {
            LocalPredictiveCodingSolver::SynchronousEquilibrium => synchronous_update(
                plain,
                &mut activities,
                &trace,
                terminal,
                config,
                scale,
                neuron_mask.as_ref(),
                activity_mask.as_ref(),
            ),
            LocalPredictiveCodingSolver::ReverseGaussSeidel => reverse_gauss_seidel_update(
                plain,
                &mut activities,
                &trace,
                terminal,
                config,
                scale,
                neuron_mask.as_ref(),
                activity_mask.as_ref(),
            ),
            _ => unreachable!("validated state-equilibrium solver"),
        });
    }

    let energy_after = config.sync_diagnostics.then(|| {
        burn_pc::diagnostic_scalar_f32(equilibrium_energy(
            plain,
            &activities,
            &initial_rhos,
            &criterion,
            config,
            neuron_mask.as_ref(),
            activity_mask.as_ref(),
        )) as f64
    });
    let trace = forward_trace_batch(
        plain,
        &activities,
        &initial_rhos,
        neuron_mask.as_ref(),
        activity_mask.as_ref(),
    );
    let terminal_hidden = plain.predictive_coding_hidden_from_activity(
        activities
            .last()
            .expect("terminal settled equilibrium activity")
            .clone(),
    );
    let terminal = criterion.parameter_factor(plain, terminal_hidden);
    let normalization = terminal.supervised_tokens.clone().clamp_min(1.0);
    let errors = prediction_error_gradient(
        trace.next.clone(),
        Tensor::cat(activities.iter().skip(1).cloned().collect(), 0),
        config.prediction_precision,
        scale,
        normalization,
    );
    let batched_vjp = layer_parameter_vjp(
        plain,
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
    register_terminal_head_derivatives(
        &mut grads,
        &parameter_ids,
        terminal.grad_lm_head,
        terminal.grad_sequence_score_head,
        scale,
        None,
    );

    let report = LocalPredictiveCodingStepReport {
        solver: config.solver,
        inference_steps: config.inference.steps,
        dual_steps: 0,
        factors,
        local_vjp_calls,
        temporal_state_vjp_calls: 0,
        fused_temporal_vjp_calls: 0,
        global_backward_calls: 0,
        gradient_tensors: grads.len(),
        direct_forward_updates: 0,
        feedback_parameter_updates: 0,
        adjoint_teacher_updates: 0,
        adjoint_local_updates: 0,
        parameter_updates: 1,
        energy_before,
        energy_after,
        grad_norm_mean: None,
        grad_norm_max: None,
        delta_rms_mean: None,
        clip_fraction_mean: None,
        constraint_rms: None,
        dual_rms: None,
        composite_signal_rms: None,
        elapsed_ns: started.elapsed().as_nanos(),
    };
    validate_step_execution_contract(config, &report);
    profile.record(report);
    LocalPredictiveCodingDerivatives {
        grads,
        loss: Tensor::<B, 1>::from_inner(feedforward_loss),
        supervised_tokens: Tensor::<B, 1>::from_inner(supervised_tokens),
        terminal_state: ModelState::<B>::from_inner_cloned(terminal_state),
        initial_rho_adjoints: Vec::new(),
        dkp_feedback: None,
        report,
    }
}
