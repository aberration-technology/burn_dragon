use crate::config::SelfDistillationKlKind;
use crate::train::objective::{
    log_probs_from_logits, masked_token_mean, self_distillation_per_token_from_log_probs,
};
use crate::train::prelude::{BackendTrait, Tensor};

pub(crate) fn smooth_l1_mean<B: BackendTrait, const D: usize>(
    prediction: Tensor<B, D>,
    target: Tensor<B, D>,
    beta: f32,
) -> Tensor<B, 1> {
    let beta = beta.max(1.0e-8);
    let abs_error = (prediction - target).abs();
    let linear = abs_error.clone().add_scalar(-beta).clamp_min(0.0);
    let quadratic = abs_error - linear.clone();
    let loss = quadratic.powf_scalar(2.0).mul_scalar(0.5 / beta) + linear;
    loss.mean().reshape([1])
}

pub(crate) fn token_kl_mean_from_logits<B: BackendTrait>(
    student_logits: Tensor<B, 3>,
    teacher_logits: Tensor<B, 3>,
) -> Tensor<B, 1> {
    let student_log_probs = log_probs_from_logits(student_logits);
    let teacher_log_probs = log_probs_from_logits(teacher_logits.detach());
    let per_token = self_distillation_per_token_from_log_probs(
        student_log_probs,
        teacher_log_probs,
        SelfDistillationKlKind::Forward,
    );
    masked_token_mean(per_token, None)
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
