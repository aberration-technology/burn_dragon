use std::sync::{Arc, Mutex};

use burn::module::AutodiffModule;
use burn::optim::GradientsParams;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor};
use burn_dragon_core::{DragonModel, DragonPredictiveCodingLayerTrace};
use burn_dragon_time::Instant;

use crate::config::{LocalPredictiveCodingConfig, PredictiveCodingFactorReduction};

#[derive(Debug)]
pub(crate) struct LocalPredictiveCodingTrainStep<B: AutodiffBackend> {
    pub grads: GradientsParams,
    pub loss: Tensor<B, 1>,
    pub report: LocalPredictiveCodingStepReport,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LocalPredictiveCodingStepReport {
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

fn forward_trace_batch<B: Backend>(
    model: &DragonModel<B>,
    activities: &[Tensor<B, 4>],
) -> DragonPredictiveCodingLayerTrace<B>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let layers = model.predictive_coding_layer_count();
    model.predictive_coding_forward_layer(
        Tensor::cat(activities.iter().take(layers).cloned().collect(), 0),
        0,
    )
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
        attention_readout: slice_batch(trace.attention_readout.clone(), start, end),
        residual_delta: slice_batch(trace.residual_delta.clone(), start, end),
        x_neuron: slice_batch(trace.x_neuron.clone(), start, end),
        y_gate: slice_batch(trace.y_gate.clone(), start, end),
        y_neuron: slice_batch(trace.y_neuron.clone(), start, end),
        next: slice_batch(trace.next.clone(), start, end),
    }
}

fn total_energy<B: Backend>(
    model: &DragonModel<B>,
    activities: &[Tensor<B, 4>],
    targets: Tensor<B, 2, Int>,
    loss_mask: Option<Tensor<B, 2, Int>>,
    config: &LocalPredictiveCodingConfig,
) -> Tensor<B, 1>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let trace = forward_trace_batch(model, activities);
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
) -> LocalPredictiveCodingTrainStep<B>
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
    let inputs = inputs.inner();
    let targets = targets.inner();
    let loss_mask = loss_mask.map(Tensor::inner);
    let initial = plain.predictive_coding_initial_activity(inputs.clone());
    let layers = plain.predictive_coding_layer_count();
    let graph = dragon_predictive_coding_graph(layers);
    debug_assert!(graph.validate().is_ok());
    let factors = layers + 1;
    let scale = factor_scale(config, factors);

    let mut activities = Vec::with_capacity(layers + 1);
    activities.push(initial);
    for layer in 0..layers {
        let trace = plain.predictive_coding_forward_layer(activities[layer].clone(), layer);
        activities.push(trace.next.detach());
    }

    let energy_before = config.sync_diagnostics.then(|| {
        burn_pc::diagnostic_scalar_f32(total_energy(
            &plain,
            &activities,
            targets.clone(),
            loss_mask.clone(),
            config,
        )) as f64
    });

    let mut local_vjp_calls = 0usize;
    let mut feedforward_loss = None;
    for _ in 0..config.inference.steps {
        let trace = forward_trace_batch(&plain, &activities);
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
        let terminal_grad = (terminal.grad_hidden * terminal.normalization.reshape([1, 1, 1]))
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
            plain.predictive_coding_layer_activity_vjp(
                0,
                &slice_trace_batch(&trace, batch, layers * batch),
                slice_batch(errors.clone(), batch, layers * batch),
            )
        });
        local_vjp_calls = local_vjp_calls.saturating_add(layers.saturating_sub(1));

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
            *activity = update.detach();
        }
    }

    let energy_after = config.sync_diagnostics.then(|| {
        burn_pc::diagnostic_scalar_f32(total_energy(
            &plain,
            &activities,
            targets.clone(),
            loss_mask.clone(),
            config,
        )) as f64
    });

    let trace = forward_trace_batch(&plain, &activities);
    let terminal_hidden = plain.predictive_coding_hidden_from_activity(
        activities.last().expect("terminal PC activity").clone(),
    );
    let terminal = plain.predictive_coding_head_vjp(terminal_hidden, targets, loss_mask.clone());
    let normalization = loss_mask.map_or_else(
        || {
            let [batch, time] = inputs.shape().dims::<2>();
            Tensor::<B::InnerBackend, 1>::full([1], (batch * time) as f32, &inputs.device())
        },
        |mask| mask.float().sum().clamp_min(1.0).reshape([1]),
    );
    let errors = prediction_error_gradient(
        trace.next.clone(),
        Tensor::cat(activities.iter().skip(1).cloned().collect(), 0),
        config.prediction_precision,
        scale,
        normalization,
    );
    let batched_vjp = plain.predictive_coding_layer_vjp(0, &trace, errors);
    local_vjp_calls = local_vjp_calls.saturating_add(layers);
    let [batch, streams, time, dim] = activities[0].shape().dims::<4>();
    let initial_vjp = plain.predictive_coding_initial_vjp(
        inputs,
        batched_vjp
            .grad_input
            .slice([0..batch, 0..streams, 0..time, 0..dim]),
    );
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
    LocalPredictiveCodingTrainStep {
        grads,
        loss,
        report,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::optim::{AdamWConfig, Optimizer};
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
