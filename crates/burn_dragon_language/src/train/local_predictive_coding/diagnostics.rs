use burn::module::ParamId;
use burn::optim::GradientsParams;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor};
use burn_dragon_core::{DragonModel, ModelState};
use serde::Serialize;

use super::{
    LocalPredictiveCodingDerivatives, LocalPredictiveCodingProfile,
    LocalPredictiveCodingStepReport, local_predictive_coding_train_step,
};
use crate::config::LocalPredictiveCodingConfig;

const NORM_EPSILON: f64 = 1.0e-30;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct PredictiveCodingGradientFidelity {
    pub parameter_family: String,
    pub elements: usize,
    /// Whether Burn's exact backward graph emitted this parameter gradient.
    /// A missing reference gradient is compared as an exact zero tensor. This
    /// is `None` for the aggregate row.
    pub reference_gradient_present: Option<bool>,
    pub dot_product: f64,
    pub pc_norm: f64,
    pub reference_norm: f64,
    pub cosine: Option<f64>,
    pub pc_to_reference_norm_ratio: Option<f64>,
    pub relative_l2_error: Option<f64>,
    /// Scalar multiplying the PC derivative that minimizes squared error to
    /// the reference gradient. A negative value means the update directions
    /// disagree overall.
    pub least_squares_scale: Option<f64>,
    /// Fraction of elements whose PC/reference product is non-negative.
    pub nonnegative_product_fraction: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LocalPredictiveCodingGradientFidelityReport {
    pub pc_loss: f64,
    pub reference_loss: f64,
    pub loss_absolute_error: f64,
    pub reference_backward_calls: usize,
    pub pc_gradient_tensors: usize,
    pub reference_gradient_tensors: usize,
    pub pc_step: LocalPredictiveCodingStepReport,
    pub global: PredictiveCodingGradientFidelity,
    pub parameter_families: Vec<PredictiveCodingGradientFidelity>,
}

#[derive(Debug, Clone, Copy, Default)]
struct RawGradientStatistics {
    elements: usize,
    reference_gradient_present: bool,
    dot_product: f64,
    pc_squared_norm: f64,
    reference_squared_norm: f64,
    squared_error: f64,
    nonnegative_products: f64,
}

impl RawGradientStatistics {
    fn merge(&mut self, other: Self) {
        self.elements = self.elements.saturating_add(other.elements);
        self.dot_product += other.dot_product;
        self.pc_squared_norm += other.pc_squared_norm;
        self.reference_squared_norm += other.reference_squared_norm;
        self.squared_error += other.squared_error;
        self.nonnegative_products += other.nonnegative_products;
    }

    fn into_fidelity(
        self,
        parameter_family: impl Into<String>,
        reference_gradient_present: Option<bool>,
    ) -> PredictiveCodingGradientFidelity {
        let pc_norm = self.pc_squared_norm.max(0.0).sqrt();
        let reference_norm = self.reference_squared_norm.max(0.0).sqrt();
        let cosine_denominator = pc_norm * reference_norm;
        PredictiveCodingGradientFidelity {
            parameter_family: parameter_family.into(),
            elements: self.elements,
            reference_gradient_present,
            dot_product: self.dot_product,
            pc_norm,
            reference_norm,
            cosine: finite_ratio(self.dot_product, cosine_denominator),
            pc_to_reference_norm_ratio: finite_ratio(pc_norm, reference_norm),
            relative_l2_error: finite_ratio(self.squared_error.max(0.0).sqrt(), reference_norm),
            least_squares_scale: finite_ratio(self.dot_product, self.pc_squared_norm),
            nonnegative_product_fraction: if self.elements == 0 {
                0.0
            } else {
                self.nonnegative_products / self.elements as f64
            },
        }
    }
}

fn finite_ratio(numerator: f64, denominator: f64) -> Option<f64> {
    if denominator.abs() <= NORM_EPSILON {
        return None;
    }
    let value = numerator / denominator;
    value.is_finite().then_some(value)
}

fn gradient_statistics<B: Backend, const D: usize>(
    pc: &GradientsParams,
    reference: &GradientsParams,
    id: ParamId,
    family: &str,
) -> Result<RawGradientStatistics, String> {
    let pc = pc.get::<B, D>(id);
    let reference = reference.get::<B, D>(id);
    if pc.is_none() && reference.is_none() {
        return Ok(RawGradientStatistics::default());
    }
    let pc = pc.ok_or_else(|| {
        format!("local PC did not emit the {family} derivative required by global autodiff")
    })?;
    let reference_gradient_present = reference.is_some();
    let reference = reference.unwrap_or_else(|| Tensor::zeros(pc.shape(), &pc.device()));
    if pc.shape() != reference.shape() {
        return Err(format!(
            "gradient shape mismatch for {family}: pc={:?} reference={:?}",
            pc.shape(),
            reference.shape()
        ));
    }

    let elements = pc.shape().num_elements();
    let product = pc.clone() * reference.clone();
    let summary = Tensor::cat(
        vec![
            product.clone().sum().reshape([1]),
            pc.clone().square().sum().reshape([1]),
            reference.clone().square().sum().reshape([1]),
            (pc - reference).square().sum().reshape([1]),
            product.greater_equal_elem(0.0).float().sum().reshape([1]),
        ],
        0,
    )
    .to_data()
    .convert::<f32>()
    .into_vec::<f32>()
    .map_err(|error| format!("failed to read {family} gradient statistics: {error}"))?;
    let [
        dot_product,
        pc_squared_norm,
        reference_squared_norm,
        squared_error,
        nonnegative_products,
    ] = summary.as_slice()
    else {
        return Err(format!(
            "expected five gradient statistics for {family}, got {}",
            summary.len()
        ));
    };
    let values = [
        *dot_product,
        *pc_squared_norm,
        *reference_squared_norm,
        *squared_error,
        *nonnegative_products,
    ];
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "non-finite gradient statistics for {family}: {values:?}"
        ));
    }
    Ok(RawGradientStatistics {
        elements,
        reference_gradient_present,
        dot_product: f64::from(*dot_product),
        pc_squared_norm: f64::from(*pc_squared_norm),
        reference_squared_norm: f64::from(*reference_squared_norm),
        squared_error: f64::from(*squared_error),
        nonnegative_products: f64::from(*nonnegative_products),
    })
}

/// Compare one canonical local-PC derivative step with an exact global
/// backward pass of the same masked next-token objective.
///
/// This is an offline diagnostic. It intentionally executes one global
/// backward pass and must not be called from the training step or telemetry
/// hot path.
#[derive(Debug, Default)]
struct GradientFidelityContext<B: AutodiffBackend> {
    initial_state: Option<ModelState<B>>,
    masks: super::LocalPredictiveCodingContextMasks<B>,
}

pub fn local_predictive_coding_gradient_fidelity<B: AutodiffBackend>(
    model: &DragonModel<B>,
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
    loss_mask: Option<Tensor<B, 2, Int>>,
    config: &LocalPredictiveCodingConfig,
) -> Result<LocalPredictiveCodingGradientFidelityReport, String>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    local_predictive_coding_gradient_fidelity_impl(
        model,
        inputs,
        targets,
        loss_mask,
        GradientFidelityContext::default(),
        config,
    )
}

/// Compare recurrent local-factor derivatives with exact global
/// backpropagation through the same chunk and detached incoming rho.
pub fn local_predictive_coding_gradient_fidelity_with_state<B: AutodiffBackend>(
    model: &DragonModel<B>,
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
    loss_mask: Option<Tensor<B, 2, Int>>,
    initial_state: ModelState<B>,
    config: &LocalPredictiveCodingConfig,
) -> Result<LocalPredictiveCodingGradientFidelityReport, String>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    local_predictive_coding_gradient_fidelity_impl(
        model,
        inputs,
        targets,
        loss_mask,
        GradientFidelityContext {
            initial_state: Some(initial_state),
            ..GradientFidelityContext::default()
        },
        config,
    )
}

/// Compare context-masked local-PC derivatives with exact backpropagation
/// through the same context-masked Dragon forward.
pub fn local_predictive_coding_gradient_fidelity_with_neuron_mask<B: AutodiffBackend>(
    model: &DragonModel<B>,
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
    loss_mask: Option<Tensor<B, 2, Int>>,
    neuron_mask: Tensor<B, 4>,
    config: &LocalPredictiveCodingConfig,
) -> Result<LocalPredictiveCodingGradientFidelityReport, String>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    local_predictive_coding_gradient_fidelity_impl(
        model,
        inputs,
        targets,
        loss_mask,
        GradientFidelityContext {
            masks: super::LocalPredictiveCodingContextMasks {
                neuron: Some(neuron_mask),
                activity: None,
            },
            ..GradientFidelityContext::default()
        },
        config,
    )
}

/// Compare context-selected subnetwork PC derivatives with exact
/// backpropagation through the same rho and residual activity masks.
pub fn local_predictive_coding_gradient_fidelity_with_subnetwork_masks<B: AutodiffBackend>(
    model: &DragonModel<B>,
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
    loss_mask: Option<Tensor<B, 2, Int>>,
    neuron_mask: Tensor<B, 4>,
    activity_mask: Tensor<B, 4>,
    config: &LocalPredictiveCodingConfig,
) -> Result<LocalPredictiveCodingGradientFidelityReport, String>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    local_predictive_coding_gradient_fidelity_impl(
        model,
        inputs,
        targets,
        loss_mask,
        GradientFidelityContext {
            masks: super::LocalPredictiveCodingContextMasks {
                neuron: Some(neuron_mask),
                activity: Some(activity_mask),
            },
            ..GradientFidelityContext::default()
        },
        config,
    )
}

/// Compare the exact sparse verifier terminal used by training with one
/// global-autodiff evaluation of the same rows and categorical-set objective.
///
/// This diagnostic deliberately performs a global backward pass. Keep it out
/// of the training hot path and use it only for numerical validation.
pub(crate) fn local_predictive_coding_verifier_gradient_fidelity<B: AutodiffBackend>(
    model: &DragonModel<B>,
    prepared: super::verifier::PreparedRuliadVerifierTerminal<B::InnerBackend>,
    config: &LocalPredictiveCodingConfig,
) -> Result<LocalPredictiveCodingGradientFidelityReport, String>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    model.predictive_coding_support()?;
    let reference = super::verifier::lift_ruliad_verifier_terminal::<B>(prepared.clone());
    let pc_step = super::local_predictive_coding_verifier_train_step(
        model,
        prepared,
        config,
        &LocalPredictiveCodingProfile::default(),
    );
    let reference_loss = reference
        .criterion
        .verifier_autodiff_loss_from_hidden(model, model.forward_hidden(reference.inputs))
        .ok_or_else(|| "Ruliad verifier terminal did not produce an autodiff loss".to_string())?;
    gradient_fidelity_report(model, pc_step, reference_loss)
}

fn local_predictive_coding_gradient_fidelity_impl<B: AutodiffBackend>(
    model: &DragonModel<B>,
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
    loss_mask: Option<Tensor<B, 2, Int>>,
    context: GradientFidelityContext<B>,
    config: &LocalPredictiveCodingConfig,
) -> Result<LocalPredictiveCodingGradientFidelityReport, String>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let GradientFidelityContext {
        initial_state,
        masks:
            super::LocalPredictiveCodingContextMasks {
                neuron: neuron_mask,
                activity: activity_mask,
            },
    } = context;
    model.predictive_coding_support()?;
    config
        .inference
        .validate("local_predictive_coding_gradient_fidelity.inference")
        .map_err(|error| error.to_string())?;
    config
        .augmented_lagrangian
        .validate("local_predictive_coding_gradient_fidelity.augmented_lagrangian")
        .map_err(|error| error.to_string())?;
    let input_shape = inputs.shape();
    if targets.shape() != input_shape {
        return Err(format!(
            "target shape {:?} does not match input shape {input_shape:?}",
            targets.shape()
        ));
    }
    if let Some(mask) = loss_mask.as_ref()
        && mask.shape() != input_shape
    {
        return Err(format!(
            "loss-mask shape {:?} does not match input shape {input_shape:?}",
            mask.shape()
        ));
    }
    if input_shape.num_elements() == 0 {
        return Err("gradient fidelity requires at least one token".to_string());
    }

    if initial_state.is_some() && (neuron_mask.is_some() || activity_mask.is_some()) {
        return Err(
            "stateful gradient fidelity currently requires dense context masks".to_string(),
        );
    }
    let pc_step = match (
        initial_state.clone(),
        neuron_mask.clone(),
        activity_mask.clone(),
    ) {
        (Some(state), None, None) => {
            super::local_predictive_coding_train_step_with_state_and_context_masks(
                model,
                inputs.clone(),
                targets.clone(),
                loss_mask.clone(),
                Some(state),
                super::LocalPredictiveCodingContextMasks::default(),
                config,
                &LocalPredictiveCodingProfile::default(),
            )
        }
        (None, None, None) => local_predictive_coding_train_step(
            model,
            inputs.clone(),
            targets.clone(),
            loss_mask.clone(),
            config,
            &LocalPredictiveCodingProfile::default(),
        ),
        (None, Some(neuron_mask), activity_mask) => {
            super::local_predictive_coding_train_step_with_context_masks(
                model,
                inputs.clone(),
                targets.clone(),
                loss_mask.clone(),
                super::LocalPredictiveCodingContextMasks {
                    neuron: Some(neuron_mask),
                    activity: activity_mask,
                },
                config,
                &LocalPredictiveCodingProfile::default(),
            )
        }
        (None, None, Some(_)) => {
            return Err("activity-only context masks are not a supported diagnostic".to_string());
        }
        (Some(_), _, _) => unreachable!("stateful masks rejected above"),
    };
    let reference_logits = match (initial_state, neuron_mask, activity_mask) {
        (Some(mut state), None, None) => {
            state.detach_in_place();
            model.forward_with_state(inputs, &mut state)
        }
        (None, Some(neuron_mask), Some(activity_mask)) => model
            .predictive_coding_forward_with_subnetwork_masks(inputs, neuron_mask, activity_mask)
            .expect("validated context subnetwork masks"),
        (None, Some(neuron_mask), None) => model
            .predictive_coding_forward_with_neuron_mask(inputs, neuron_mask)
            .expect("validated context neuron mask"),
        (None, None, None) => model.forward(inputs),
        (None, None, Some(_)) => unreachable!("validated diagnostic context masks"),
        (Some(_), _, _) => unreachable!("stateful masks rejected above"),
    };
    let reference_loss = burn_dragon_core::objective::masked_token_mean(
        model.language_token_losses_from_logits(reference_logits, targets),
        loss_mask,
    );
    gradient_fidelity_report(model, pc_step, reference_loss)
}

fn gradient_fidelity_report<B: AutodiffBackend>(
    model: &DragonModel<B>,
    pc_step: LocalPredictiveCodingDerivatives<B>,
    reference_loss: Tensor<B, 1>,
) -> Result<LocalPredictiveCodingGradientFidelityReport, String> {
    let pc_loss = f64::from(burn_pc::diagnostic_scalar_f32(pc_step.loss.inner()));
    let reference_loss_value = f64::from(burn_pc::diagnostic_scalar_f32(
        reference_loss.clone().inner(),
    ));
    if !pc_loss.is_finite() || !reference_loss_value.is_finite() {
        return Err(format!(
            "non-finite objective values: pc={pc_loss} reference={reference_loss_value}"
        ));
    }
    let reference_grads = GradientsParams::from_grads(reference_loss.backward(), model);
    let parameter_ids = model.predictive_coding_parameter_ids()?;

    let mut global = RawGradientStatistics::default();
    let mut parameter_families = Vec::with_capacity(15);
    macro_rules! compare_family {
        ($name:literal, $id:expr, $rank:literal) => {{
            let raw = gradient_statistics::<B::InnerBackend, $rank>(
                &pc_step.grads,
                &reference_grads,
                $id,
                $name,
            )?;
            global.merge(raw);
            let reference_gradient_present = raw.reference_gradient_present;
            parameter_families.push(raw.into_fidelity($name, Some(reference_gradient_present)));
        }};
    }
    compare_family!("embedding", parameter_ids.embedding, 2);
    compare_family!("shared_encoder", parameter_ids.encoder, 3);
    compare_family!("shared_value_encoder", parameter_ids.encoder_v, 3);
    compare_family!("shared_decoder", parameter_ids.decoder, 2);
    compare_family!("norm_gamma", parameter_ids.norm_gamma, 1);
    compare_family!("norm_beta", parameter_ids.norm_beta, 1);
    compare_family!("norm_alpha", parameter_ids.norm_alpha, 1);
    compare_family!("norm_shift", parameter_ids.norm_shift, 1);
    compare_family!("language_head", parameter_ids.lm_head, 2);
    if let Some(score) = parameter_ids.sequence_score_head {
        compare_family!("sequence_query_weight", score.query_weight, 2);
        compare_family!("sequence_query_bias", score.query_bias, 1);
        compare_family!("sequence_candidate_weight", score.candidate_weight, 2);
        compare_family!("sequence_candidate_bias", score.candidate_bias, 1);
        compare_family!("sequence_score_weight", score.score_weight, 2);
        compare_family!("sequence_score_bias", score.score_bias, 1);
    }

    Ok(LocalPredictiveCodingGradientFidelityReport {
        pc_loss,
        reference_loss: reference_loss_value,
        loss_absolute_error: (pc_loss - reference_loss_value).abs(),
        reference_backward_calls: 1,
        pc_gradient_tensors: pc_step.grads.len(),
        reference_gradient_tensors: reference_grads.len(),
        pc_step: pc_step.report,
        global: global.into_fidelity("all_parameters", None),
        parameter_families,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::AutodiffModule;
    use burn::optim::{GradientsParams, Optimizer, SgdConfig};
    use burn::tensor::TensorData;
    use burn_autodiff::Autodiff;
    use burn_dragon_core::{DragonConfig, RotaryEmbedding, SequenceTrainingExecutor};
    use burn_ndarray::NdArray;

    type TestBackend = Autodiff<NdArray<f32>>;

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
        config.fused_kernels.rotary_embedding = RotaryEmbedding::Alibi;
        DragonModel::new(config, device)
    }

    fn batch(
        device: &burn::tensor::Device<TestBackend>,
    ) -> (
        Tensor<TestBackend, 2, Int>,
        Tensor<TestBackend, 2, Int>,
        Tensor<TestBackend, 2, Int>,
    ) {
        (
            Tensor::from_data(
                TensorData::new(vec![1_i64, 2, 3, 1, 2, 3, 1, 2], [2, 4]),
                device,
            ),
            Tensor::from_data(
                TensorData::new(vec![2_i64, 3, 1, 2, 3, 1, 2, 3], [2, 4]),
                device,
            ),
            Tensor::from_data(
                TensorData::new(vec![1_i64, 1, 0, 1, 1, 0, 1, 1], [2, 4]),
                device,
            ),
        )
    }

    #[test]
    fn fidelity_report_matches_masked_objective_and_resolves_all_pc_families() {
        let device = Default::default();
        let model = model(&device);
        let (inputs, targets, mask) = batch(&device);
        let report = local_predictive_coding_gradient_fidelity(
            &model,
            inputs,
            targets,
            Some(mask),
            &LocalPredictiveCodingConfig {
                inference: burn_pc::PcInferenceConfig {
                    steps: 4,
                    step_size: 0.05,
                    max_grad_norm: None,
                    ..burn_pc::PcInferenceConfig::default()
                },
                ..LocalPredictiveCodingConfig::default()
            },
        )
        .expect("gradient fidelity report");

        assert!(report.loss_absolute_error < 1.0e-6);
        assert_eq!(report.reference_backward_calls, 1);
        assert_eq!(report.pc_step.global_backward_calls, 0);
        assert_eq!(report.pc_gradient_tensors, 9);
        assert_eq!(report.parameter_families.len(), 9);
        assert!(report.global.cosine.is_some_and(|cosine| cosine > 0.0));
        for family in &report.parameter_families {
            assert!(family.dot_product.is_finite());
            assert!(family.pc_norm.is_finite());
            assert!(family.reference_norm.is_finite());
            if family.reference_norm > 1.0e-8 && family.pc_norm > 1.0e-8 {
                assert!(
                    family
                        .cosine
                        .is_some_and(|cosine| cosine.abs() <= 1.0 + 1.0e-6),
                    "invalid local derivative cosine for {}: {:?}",
                    family.parameter_family,
                    family.cosine
                );
            }
        }
        for inactive in ["norm_alpha", "norm_shift"] {
            let family = report
                .parameter_families
                .iter()
                .find(|family| family.parameter_family == inactive)
                .expect("inactive norm family");
            assert_eq!(family.reference_gradient_present, Some(false));
            assert_eq!(family.reference_norm, 0.0);
            assert_eq!(family.pc_norm, 0.0);
        }
    }

    #[test]
    fn fixed_prediction_matches_reference_through_shared_layer_uses() {
        let device = Default::default();
        let mut config = DragonConfig {
            n_layer: 4,
            n_embd: 8,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 16,
            dropout: 0.0,
            ..DragonConfig::default()
        };
        config.sequence_kernel.executor = SequenceTrainingExecutor::DenseScoreShortContext;
        config.fused_kernels.rotary_embedding = RotaryEmbedding::Alibi;
        let model = crate::train::test_support::deterministic_matrix_parameters(DragonModel::new(
            config, &device,
        ));
        let (inputs, targets, mask) = batch(&device);
        let report = local_predictive_coding_gradient_fidelity(
            &model,
            inputs,
            targets,
            Some(mask),
            &LocalPredictiveCodingConfig {
                solver: crate::config::LocalPredictiveCodingSolver::FixedPrediction,
                ..LocalPredictiveCodingConfig::default()
            },
        )
        .expect("fixed-prediction gradient fidelity report");

        assert_eq!(
            report.pc_step.solver,
            crate::config::LocalPredictiveCodingSolver::FixedPrediction
        );
        assert_eq!(report.pc_step.inference_steps, 1);
        assert_eq!(report.pc_step.global_backward_calls, 0);
        assert_eq!(report.pc_gradient_tensors, 9);
        assert!(report.loss_absolute_error < 1.0e-6);
        assert!(
            report.global.cosine.is_some_and(|cosine| cosine > 0.999_99),
            "global fidelity: {:?}",
            report.global
        );
        assert!(
            report
                .global
                .relative_l2_error
                .is_some_and(|error| error < 1.0e-4),
            "global fidelity: {:?}",
            report.global
        );
        for family in &report.parameter_families {
            if family.reference_norm > 1.0e-8 {
                assert!(
                    family.cosine.is_some_and(|cosine| cosine > 0.999_9),
                    "shared-weight family mismatch: {family:?}"
                );
                assert!(
                    family.relative_l2_error.is_some_and(|error| error < 1.0e-3),
                    "shared-weight family mismatch: {family:?}"
                );
            }
        }
    }

    #[test]
    fn recurrent_fixed_prediction_matches_detached_tbptt_reference() {
        let device = Default::default();
        TestBackend::seed(&device, 20260805);
        let mut config = DragonConfig {
            n_layer: 4,
            n_embd: 8,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 16,
            dropout: 0.0,
            ..DragonConfig::default()
        };
        config.sequence_kernel.executor = SequenceTrainingExecutor::DenseScoreShortContext;
        config.fused_kernels.rotary_embedding = RotaryEmbedding::Alibi;
        let model = DragonModel::<TestBackend>::new(config, &device);
        let prefix = Tensor::from_data(
            TensorData::new(vec![4_i64, 5, 6, 7, 7, 6, 5, 4], [2, 4]),
            &device,
        );
        let mut incoming = model.init_state();
        let _ = model.forward_with_state(prefix, &mut incoming);
        incoming.detach_in_place();
        let (inputs, targets, mask) = batch(&device);
        let pc_config = LocalPredictiveCodingConfig {
            solver: crate::config::LocalPredictiveCodingSolver::FixedPrediction,
            ..LocalPredictiveCodingConfig::default()
        };
        let report = local_predictive_coding_gradient_fidelity_with_state(
            &model,
            inputs.clone(),
            targets.clone(),
            Some(mask.clone()),
            incoming.clone(),
            &pc_config,
        )
        .expect("recurrent fixed-prediction fidelity report");
        assert!(report.loss_absolute_error < 1.0e-6);
        assert!(
            report.global.cosine.is_some_and(|cosine| cosine > 0.999_99),
            "recurrent fixed-prediction fidelity: {:?}",
            report.global
        );
        assert!(
            report
                .global
                .relative_l2_error
                .is_some_and(|error| error < 1.0e-4),
            "recurrent fixed-prediction fidelity: {:?}",
            report.global
        );

        for config in [
            pc_config,
            LocalPredictiveCodingConfig {
                solver: crate::config::LocalPredictiveCodingSolver::LayerLocalPrediction,
                factor_reduction: crate::config::PredictiveCodingFactorReduction::Mean,
                ..LocalPredictiveCodingConfig::default()
            },
        ] {
            let local = super::super::local_predictive_coding_derivatives_with_state(
                &model,
                inputs.clone(),
                targets.clone(),
                Some(mask.clone()),
                incoming.clone(),
                &config,
            )
            .expect("recurrent local derivatives");
            let mut reference_state = incoming.clone();
            let _ = model.forward_with_state(inputs.clone(), &mut reference_state);
            assert_eq!(
                local.terminal_state.position, reference_state.position,
                "solver={:?}",
                config.solver
            );
            for (layer, (local_layer, reference_layer)) in local
                .terminal_state
                .layers
                .iter()
                .zip(&reference_state.layers)
                .enumerate()
            {
                let local_rho = local_layer.rho.clone().expect("local terminal rho");
                let reference_rho = reference_layer.rho.clone().expect("reference terminal rho");
                let max_error = (local_rho - reference_rho)
                    .abs()
                    .max()
                    .inner()
                    .to_data()
                    .convert::<f32>()
                    .into_vec::<f32>()
                    .expect("terminal rho error")[0];
                assert!(
                    max_error < 1.0e-5,
                    "solver={:?} layer {layer} terminal rho mismatch: {max_error}",
                    config.solver
                );
            }
        }
    }

    #[test]
    fn two_chunk_state_adjoint_composition_matches_full_recurrent_backpropagation() {
        let device = Default::default();
        TestBackend::seed(&device, 20260811);
        let mut config = DragonConfig {
            n_layer: 3,
            n_embd: 8,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 16,
            dropout: 0.0,
            ..DragonConfig::default()
        };
        config.sequence_kernel.executor = SequenceTrainingExecutor::DenseScoreShortContext;
        config.fused_kernels.rotary_embedding = RotaryEmbedding::Alibi;
        let model = DragonModel::<TestBackend>::new(config, &device);
        let inputs = Tensor::from_data(
            TensorData::new(
                vec![1_i64, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1],
                [2, 8],
            ),
            &device,
        );
        let targets = Tensor::from_data(
            TensorData::new(
                vec![2_i64, 3, 4, 5, 6, 7, 8, 9, 7, 6, 5, 4, 3, 2, 1, 0],
                [2, 8],
            ),
            &device,
        );
        let mask = Tensor::from_data(
            TensorData::new(
                vec![1_i64, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1],
                [2, 8],
            ),
            &device,
        );
        let input_chunks = [
            inputs.clone().slice([0..2, 0..4]),
            inputs.slice([0..2, 4..8]),
        ];
        let target_chunks = [
            targets.clone().slice([0..2, 0..4]),
            targets.slice([0..2, 4..8]),
        ];
        let mask_chunks = [mask.clone().slice([0..2, 0..4]), mask.slice([0..2, 4..8])];
        let initial_state = model.init_state();
        let pc_config = LocalPredictiveCodingConfig {
            solver: crate::config::LocalPredictiveCodingSolver::FixedPrediction,
            ..LocalPredictiveCodingConfig::default()
        };

        let first_observation = super::super::local_predictive_coding_derivatives_with_state(
            &model,
            input_chunks[0].clone(),
            target_chunks[0].clone(),
            Some(mask_chunks[0].clone()),
            initial_state.clone(),
            &pc_config,
        )
        .expect("first recurrent local factor");
        let second = super::super::local_predictive_coding_derivatives_with_state(
            &model,
            input_chunks[1].clone(),
            target_chunks[1].clone(),
            Some(mask_chunks[1].clone()),
            first_observation.terminal_state,
            &pc_config,
        )
        .expect("second recurrent local factor");
        assert_eq!(second.initial_rho_adjoints.len(), 3);
        assert!(second.initial_rho_adjoints.iter().all(Option::is_some));
        let first = super::super::local_predictive_coding_derivatives_with_state_adjoint(
            &model,
            input_chunks[0].clone(),
            target_chunks[0].clone(),
            Some(mask_chunks[0].clone()),
            initial_state,
            second.initial_rho_adjoints.clone(),
            &pc_config,
        )
        .expect("first factor with future rho adjoint");
        assert_eq!(first.report.temporal_state_vjp_calls, 3);
        assert_eq!(first.report.global_backward_calls, 0);
        assert_eq!(second.report.global_backward_calls, 0);

        let mut combined = GradientsParams::new();
        let ids = model.predictive_coding_parameter_ids().expect("PC ids");
        macro_rules! combine_family {
            ($id:expr, $rank:literal) => {{
                let id = $id;
                combined.register(
                    id,
                    first
                        .grads
                        .get::<NdArray<f32>, $rank>(id)
                        .expect("first local gradient")
                        + second
                            .grads
                            .get::<NdArray<f32>, $rank>(id)
                            .expect("second local gradient"),
                );
            }};
        }
        combine_family!(ids.embedding, 2);
        combine_family!(ids.encoder, 3);
        combine_family!(ids.encoder_v, 3);
        combine_family!(ids.decoder, 2);
        combine_family!(ids.norm_gamma, 1);
        combine_family!(ids.norm_beta, 1);
        combine_family!(ids.norm_alpha, 1);
        combine_family!(ids.norm_shift, 1);
        combine_family!(ids.lm_head, 2);

        let mut reference_state = model.init_state();
        let logits_first = model.forward_with_state(input_chunks[0].clone(), &mut reference_state);
        let logits_second = model.forward_with_state(input_chunks[1].clone(), &mut reference_state);
        let first_loss = burn_dragon_core::objective::masked_token_mean(
            model.language_token_losses_from_logits(logits_first, target_chunks[0].clone()),
            Some(mask_chunks[0].clone()),
        );
        let second_loss = burn_dragon_core::objective::masked_token_mean(
            model.language_token_losses_from_logits(logits_second, target_chunks[1].clone()),
            Some(mask_chunks[1].clone()),
        );
        let reference = GradientsParams::from_grads((first_loss + second_loss).backward(), &model);
        let mut aggregate = RawGradientStatistics::default();
        macro_rules! compare_family {
            ($name:literal, $id:expr, $rank:literal) => {{
                aggregate.merge(
                    gradient_statistics::<NdArray<f32>, $rank>(&combined, &reference, $id, $name)
                        .expect("two-chunk family statistics"),
                );
            }};
        }
        compare_family!("embedding", ids.embedding, 2);
        compare_family!("shared_encoder", ids.encoder, 3);
        compare_family!("shared_value_encoder", ids.encoder_v, 3);
        compare_family!("shared_decoder", ids.decoder, 2);
        compare_family!("norm_gamma", ids.norm_gamma, 1);
        compare_family!("norm_beta", ids.norm_beta, 1);
        compare_family!("norm_alpha", ids.norm_alpha, 1);
        compare_family!("norm_shift", ids.norm_shift, 1);
        compare_family!("language_head", ids.lm_head, 2);
        let fidelity = aggregate.into_fidelity("two_chunk_parameters", None);
        assert!(
            fidelity.cosine.is_some_and(|cosine| cosine > 0.999_9),
            "two-chunk temporal fidelity: {fidelity:?}"
        );
        assert!(
            fidelity
                .relative_l2_error
                .is_some_and(|error| error < 1.0e-3),
            "two-chunk temporal fidelity: {fidelity:?}"
        );
    }

    #[test]
    fn reverse_gauss_seidel_propagates_credit_within_one_sweep() {
        let device = Default::default();
        let mut config = DragonConfig {
            n_layer: 4,
            n_embd: 8,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 16,
            dropout: 0.0,
            ..DragonConfig::default()
        };
        config.sequence_kernel.executor = SequenceTrainingExecutor::DenseScoreShortContext;
        config.fused_kernels.rotary_embedding = RotaryEmbedding::Alibi;
        let model = crate::train::test_support::deterministic_matrix_parameters(DragonModel::new(
            config, &device,
        ));
        let (inputs, targets, mask) = batch(&device);
        let inference = burn_pc::PcInferenceConfig {
            steps: 1,
            step_size: 1.0,
            max_grad_norm: None,
            ..burn_pc::PcInferenceConfig::default()
        };
        let synchronous = local_predictive_coding_gradient_fidelity(
            &model,
            inputs.clone(),
            targets.clone(),
            Some(mask.clone()),
            &LocalPredictiveCodingConfig {
                solver: crate::config::LocalPredictiveCodingSolver::SynchronousEquilibrium,
                inference,
                sync_diagnostics: true,
                ..LocalPredictiveCodingConfig::default()
            },
        )
        .expect("synchronous fidelity report");
        let reverse = local_predictive_coding_gradient_fidelity(
            &model,
            inputs,
            targets,
            Some(mask),
            &LocalPredictiveCodingConfig {
                solver: crate::config::LocalPredictiveCodingSolver::ReverseGaussSeidel,
                inference,
                sync_diagnostics: true,
                ..LocalPredictiveCodingConfig::default()
            },
        )
        .expect("reverse Gauss-Seidel fidelity report");
        let synchronous_cosine = synchronous.global.cosine.expect("synchronous cosine");
        let reverse_cosine = reverse.global.cosine.expect("reverse cosine");
        assert!(
            reverse_cosine > 0.9 && reverse_cosine > synchronous_cosine + 0.15,
            "reverse sweep should propagate terminal credit through shared depth: synchronous={synchronous_cosine} reverse={reverse_cosine}"
        );
        assert!(
            reverse.pc_step.energy_after.expect("energy after")
                < reverse.pc_step.energy_before.expect("energy before"),
            "reverse activity sweep must descend the joint energy"
        );
        assert_eq!(reverse.pc_step.global_backward_calls, 0);
    }

    #[test]
    fn context_neuron_masked_fixed_prediction_matches_masked_backpropagation() {
        let device = Default::default();
        TestBackend::seed(&device, 73);
        let model = model(&device);
        let (inputs, targets, mask) = batch(&device);
        let neuron_mask = Tensor::from_data(
            TensorData::new(
                vec![1.0_f32, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1, 1, 1, 8],
            ),
            &device,
        );
        let report = local_predictive_coding_gradient_fidelity_with_neuron_mask(
            &model,
            inputs,
            targets,
            Some(mask),
            neuron_mask,
            &LocalPredictiveCodingConfig {
                solver: crate::config::LocalPredictiveCodingSolver::FixedPrediction,
                ..LocalPredictiveCodingConfig::default()
            },
        )
        .expect("context-masked fidelity report");
        assert!(report.loss_absolute_error < 1.0e-6);
        assert!(
            report.global.cosine.is_some_and(|cosine| cosine > 0.999_99),
            "masked fixed-prediction fidelity: {:?}",
            report.global
        );
        assert!(
            report
                .global
                .relative_l2_error
                .is_some_and(|error| error < 1.0e-4),
            "masked fixed-prediction fidelity: {:?}",
            report.global
        );
    }

    #[test]
    fn context_subnetwork_fixed_prediction_matches_masked_backpropagation() {
        let device = Default::default();
        TestBackend::seed(&device, 79);
        let model = model(&device);
        let (inputs, targets, mask) = batch(&device);
        let disjoint_mask = || {
            Tensor::from_data(
                TensorData::new(
                    vec![1.0_f32, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [1, 1, 1, 8],
                ),
                &device,
            )
        };
        let report = local_predictive_coding_gradient_fidelity_with_subnetwork_masks(
            &model,
            inputs,
            targets,
            Some(mask),
            disjoint_mask(),
            disjoint_mask(),
            &LocalPredictiveCodingConfig {
                solver: crate::config::LocalPredictiveCodingSolver::FixedPrediction,
                ..LocalPredictiveCodingConfig::default()
            },
        )
        .expect("context-selected subnetwork fidelity report");
        assert!(report.loss_absolute_error < 1.0e-6);
        assert!(
            report.global.cosine.is_some_and(|cosine| cosine > 0.999_99),
            "subnetwork fixed-prediction fidelity: {:?}",
            report.global
        );
        assert!(
            report
                .global
                .relative_l2_error
                .is_some_and(|error| error < 1.0e-4),
            "subnetwork fixed-prediction fidelity: {:?}",
            report.global
        );
    }

    #[test]
    fn disjoint_subnetwork_step_preserves_inactive_context_logits() {
        let device = Default::default();
        TestBackend::seed(&device, 83);
        let mut model = model(&device);
        let (inputs, targets, loss_mask) = batch(&device);
        let context_mask = |first_half: bool| {
            let values = if first_half {
                vec![1.0_f32, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
            } else {
                vec![0.0_f32, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
            };
            Tensor::from_data(TensorData::new(values, [1, 1, 1, 8]), &device)
        };
        let before = model
            .valid()
            .predictive_coding_forward_with_subnetwork_masks(
                inputs.clone().inner(),
                context_mask(true).inner(),
                context_mask(true).inner(),
            )
            .expect("task A forward");
        let task_b_logits = model
            .predictive_coding_forward_with_subnetwork_masks(
                inputs.clone(),
                context_mask(false),
                context_mask(false),
            )
            .expect("task B forward");
        let loss = burn_dragon_core::objective::masked_token_mean(
            model.language_token_losses_from_logits(task_b_logits, targets),
            Some(loss_mask),
        );
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        let mut optimizer = SgdConfig::new().init::<TestBackend, DragonModel<TestBackend>>();
        model = optimizer.step(0.1, model, grads);
        let after = model
            .valid()
            .predictive_coding_forward_with_subnetwork_masks(
                inputs.clone().inner(),
                context_mask(true).inner(),
                context_mask(true).inner(),
            )
            .expect("task A forward after task B update");
        let max_delta = (after - before)
            .abs()
            .max()
            .into_data()
            .to_vec::<f32>()
            .expect("logit delta")[0];
        assert!(
            max_delta < 1.0e-6,
            "disjoint task-B update changed task-A logits by {max_delta}"
        );
    }
}
