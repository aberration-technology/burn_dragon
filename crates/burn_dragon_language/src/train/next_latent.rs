use crate::config::SelfDistillationKlKind;
use crate::train::objective::{
    log_probs_from_logits, masked_token_mean, self_distillation_per_token_from_log_probs,
};
use crate::train::prelude::{BackendTrait, Tensor};

mod mask;
pub(crate) use mask::NextLatentTokenLayout;

/// Average horizons with valid support, without coupling different loss families.
#[derive(Default)]
pub(crate) struct HorizonMean<B: BackendTrait> {
    sum: Option<Tensor<B, 1>>,
    horizons: Option<Tensor<B, 1>>,
}

impl<B: BackendTrait> HorizonMean<B> {
    pub(crate) fn new() -> Self {
        Self {
            sum: None,
            horizons: None,
        }
    }

    pub(crate) fn push(&mut self, values: Tensor<B, 2>, mask: Tensor<B, 2, burn::tensor::Int>) {
        let (mean, count) =
            crate::train::objective::masked_token_mean_with_count(values, Some(mask));
        let active = count.greater_elem(0.0).float();
        self.sum = Some(match self.sum.take() {
            Some(sum) => sum + mean,
            None => mean,
        });
        self.horizons = Some(match self.horizons.take() {
            Some(count) => count + active,
            None => active,
        });
    }

    pub(crate) fn finish(self) -> Option<Tensor<B, 1>> {
        Some(self.sum? / self.horizons?.clamp_min(1.0))
    }
}

fn smooth_l1<B: BackendTrait, const D: usize>(
    prediction: Tensor<B, D>,
    target: Tensor<B, D>,
    beta: f32,
) -> Tensor<B, D> {
    let beta = beta.max(1.0e-8);
    let abs_error = (prediction - target).abs();
    let linear = abs_error.clone().add_scalar(-beta).clamp_min(0.0);
    let quadratic = abs_error - linear.clone();
    quadratic.powf_scalar(2.0).mul_scalar(0.5 / beta) + linear
}

pub(crate) fn smooth_l1_per_token<B: BackendTrait>(
    prediction: Tensor<B, 3>,
    target: Tensor<B, 3>,
    beta: f32,
) -> Tensor<B, 2> {
    let [batch, time, _] = prediction.dims();
    smooth_l1(prediction, target, beta)
        .mean_dim(2)
        .reshape([batch, time])
}

pub(crate) fn smooth_l1_mean<B: BackendTrait, const D: usize>(
    prediction: Tensor<B, D>,
    target: Tensor<B, D>,
    beta: f32,
) -> Tensor<B, 1> {
    smooth_l1(prediction, target, beta).mean().reshape([1])
}

pub(crate) fn token_kl_per_token<B: BackendTrait>(
    student_logits: Tensor<B, 3>,
    teacher_logits: Tensor<B, 3>,
) -> Tensor<B, 2> {
    self_distillation_per_token_from_log_probs(
        log_probs_from_logits(student_logits),
        log_probs_from_logits(teacher_logits.detach()),
        SelfDistillationKlKind::Forward,
    )
}

pub(crate) fn token_kl_mean_from_logits<B: BackendTrait>(
    student_logits: Tensor<B, 3>,
    teacher_logits: Tensor<B, 3>,
) -> Tensor<B, 1> {
    masked_token_mean(token_kl_per_token(student_logits, teacher_logits), None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn device() -> burn::tensor::Device<TestBackend> {
        Default::default()
    }

    fn scalar(value: Tensor<TestBackend, 1>) -> f32 {
        value
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("scalar")[0]
    }

    #[test]
    fn smooth_l1_mean_matches_hand_fixture() {
        let prediction = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![0.5, 3.0], [1, 2, 1]),
            &device(),
        );
        let target = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![0.0, 1.0], [1, 2, 1]),
            &device(),
        );
        let loss = scalar(smooth_l1_mean(prediction, target, 1.0));
        assert!((loss - 0.8125).abs() < 1.0e-6, "loss={loss}");
    }

    #[test]
    fn token_kl_is_zero_for_identical_logits() {
        let logits = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![1.0, 0.0, -1.0, 2.0], [1, 2, 2]),
            &device(),
        );
        let loss = scalar(token_kl_mean_from_logits(logits.clone(), logits));
        assert!(loss.abs() < 1.0e-6, "loss={loss}");
    }
}
