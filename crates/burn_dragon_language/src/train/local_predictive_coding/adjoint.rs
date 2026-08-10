use super::*;
use crate::config::LocalPredictiveCodingAdjointConditioning;

pub(super) struct ParallelAdjointResolution<B: Backend> {
    pub(super) layer_signals: Tensor<B, 3>,
    pub(super) feedback: Option<Tensor<B, 3>>,
    pub(super) signal_vjp_calls: usize,
    pub(super) teacher_due: bool,
}

fn initial_amortized_feedback<B: Backend>(
    layers: usize,
    dim: usize,
    config: &LocalPredictiveCodingConfig,
    device: &B::Device,
) -> Tensor<B, 3> {
    match config.amortized_adjoint.predictor {
        burn_pc::PcAdjointPredictorKind::DirectLinear => {
            super::initial_dkp_feedback(layers, dim, config.direct_feedback.initialization, device)
        }
        burn_pc::PcAdjointPredictorKind::ResidualConditioned => Tensor::zeros(
            [
                layers,
                dim,
                dim * config.amortized_adjoint.predictor.feature_multiplier(),
            ],
            device,
        ),
    }
}

fn conditioning<B: Backend>(
    trace: &DragonPredictiveCodingLayerTrace<B>,
    terminal_activity: Tensor<B, 4>,
    layers: usize,
    batch: usize,
    kind: LocalPredictiveCodingAdjointConditioning,
) -> Tensor<B, 3> {
    let [trace_batch, streams, time, dim] = trace.next.shape().dims::<4>();
    assert_eq!(trace_batch, layers * batch);
    let conditioning = match kind {
        LocalPredictiveCodingAdjointConditioning::LocalResidual => trace.residual_delta.clone(),
        LocalPredictiveCodingAdjointConditioning::TerminalDisplacement => {
            terminal_activity.repeat_dim(0, layers) - trace.next.clone()
        }
    };
    conditioning.reshape([layers, batch * streams * time, dim])
}

/// Exact local-VJP teacher for every shared-depth output activity.
pub(super) fn exact_layer_output_adjoint_batch<B: Backend>(
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
            gradient = super::layer_activity_vjp(
                model,
                layer,
                &super::slice_trace_batch(batched_trace, layer * batch, (layer + 1) * batch),
                gradient,
                None,
                None,
            );
        }
    }
    reversed.reverse();
    Tensor::cat(reversed, 0)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn resolve_parallel_adjoint<B: Backend>(
    model: &DragonModel<B>,
    batched_trace: &DragonPredictiveCodingLayerTrace<B>,
    terminal_gradient: Tensor<B, 4>,
    terminal_signal: Tensor<B, 3>,
    terminal_activity: Tensor<B, 4>,
    feedback: Option<Tensor<B, 3>>,
    feedback_updates: u64,
    config: &LocalPredictiveCodingConfig,
    profile: &LocalPredictiveCodingProfile,
    layers: usize,
    batch: usize,
) -> ParallelAdjointResolution<B>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let [_, streams, time, dim] = terminal_activity.shape().dims::<4>();
    match config.solver {
        LocalPredictiveCodingSolver::FirstOrderAdjoint => {
            let local_terminal_adjoint = super::layer_activity_vjp(
                model,
                0,
                batched_trace,
                terminal_gradient.repeat_dim(0, layers),
                None,
                None,
            )
            .reshape([layers, batch * streams * time, dim]);
            ParallelAdjointResolution {
                layer_signals: burn_pc::first_order_residual_adjoint_batched(
                    local_terminal_adjoint,
                    terminal_signal,
                ),
                feedback: None,
                signal_vjp_calls: layers,
                teacher_due: false,
            }
        }
        LocalPredictiveCodingSolver::AmortizedAdjoint => {
            let feedback = feedback.unwrap_or_else(|| {
                initial_amortized_feedback::<B>(layers, dim, config, &terminal_activity.device())
            });
            let feedback_features = dim * config.amortized_adjoint.predictor.feature_multiplier();
            assert_eq!(
                feedback.shape().dims::<3>(),
                [layers, dim, feedback_features],
                "amortized-adjoint feedback checkpoint geometry must match model depth and embedding"
            );
            let conditioning = matches!(
                config.amortized_adjoint.predictor,
                burn_pc::PcAdjointPredictorKind::ResidualConditioned
            )
            .then(|| {
                conditioning(
                    batched_trace,
                    terminal_activity,
                    layers,
                    batch,
                    config.adjoint_conditioning,
                )
            });
            let teacher_due = config.amortized_adjoint.teacher_due(feedback_updates);
            if teacher_due {
                let teacher_signal = exact_layer_output_adjoint_batch(
                    model,
                    batched_trace,
                    terminal_gradient,
                    layers,
                    batch,
                );
                let calibration = burn_pc::calibrate_amortized_adjoint_batched(
                    feedback,
                    terminal_signal,
                    conditioning,
                    teacher_signal.clone(),
                    &config.amortized_adjoint,
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
                ParallelAdjointResolution {
                    layer_signals: teacher_signal,
                    feedback: Some(calibration.feedback),
                    signal_vjp_calls: layers.saturating_sub(1),
                    teacher_due: true,
                }
            } else {
                let layer_signals = match config.amortized_adjoint.predictor {
                    burn_pc::PcAdjointPredictorKind::DirectLinear => {
                        burn_pc::direct_feedback_signal_batched(
                            terminal_signal,
                            feedback.clone(),
                            config.direct_feedback.signal_scale,
                        )
                    }
                    burn_pc::PcAdjointPredictorKind::ResidualConditioned => {
                        burn_pc::predict_amortized_adjoint_batched(
                            feedback.clone(),
                            terminal_signal,
                            conditioning,
                            &config.amortized_adjoint,
                        )
                    }
                };
                ParallelAdjointResolution {
                    layer_signals,
                    feedback: Some(feedback),
                    signal_vjp_calls: 0,
                    teacher_due: false,
                }
            }
        }
        _ => unreachable!("validated parallel-adjoint solver"),
    }
}
