use crate::config::{SelfDistillationKlKind, TrainingObjectiveConfig, TrainingObjectiveKind};
use burn::tensor::activation;
use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Int, Tensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveTrainerKind {
    SingleDevice,
    Browser,
    Ddp,
    Pipeline,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveSupport {
    FullySupported,
    ConfigAndNumericsOnly,
}

pub fn objective_support(
    objective: &TrainingObjectiveConfig,
    trainer: ObjectiveTrainerKind,
) -> ObjectiveSupport {
    match (objective.kind(), trainer) {
        (TrainingObjectiveKind::NextToken, _) => ObjectiveSupport::FullySupported,
        (TrainingObjectiveKind::Sdft | TrainingObjectiveKind::Sdpo, _) => {
            ObjectiveSupport::ConfigAndNumericsOnly
        }
    }
}

pub fn ensure_objective_supported(
    objective: &TrainingObjectiveConfig,
    trainer: ObjectiveTrainerKind,
) -> anyhow::Result<()> {
    if objective_support(objective, trainer) == ObjectiveSupport::FullySupported {
        return Ok(());
    }
    let objective = match objective.kind() {
        TrainingObjectiveKind::NextToken => "next_token",
        TrainingObjectiveKind::Sdft => "sdft",
        TrainingObjectiveKind::Sdpo => "sdpo",
    };
    let trainer = match trainer {
        ObjectiveTrainerKind::SingleDevice => "single-device",
        ObjectiveTrainerKind::Browser => "browser",
        ObjectiveTrainerKind::Ddp => "ddp",
        ObjectiveTrainerKind::Pipeline => "pipeline",
    };
    Err(anyhow::anyhow!(
        "training.objective.type={objective} is configured, but {trainer} training is only wired for the next_token objective; SDFT/SDPO config and numerical kernels are available, and the rollout objective driver must be enabled before running this objective"
    ))
}

pub fn log_probs_from_logits<B: BackendTrait>(logits: Tensor<B, 3>) -> Tensor<B, 3> {
    let [batch, time, vocab] = logits.shape().dims();
    activation::log_softmax(logits.reshape([batch * time, vocab]), 1).reshape([batch, time, vocab])
}

pub fn selected_token_log_probs<B: BackendTrait>(
    log_probs: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
) -> Tensor<B, 2> {
    let [batch, time, _vocab] = log_probs.shape().dims();
    log_probs
        .gather(2, targets.reshape([batch, time, 1]))
        .reshape([batch, time])
}

pub fn self_distillation_loss_from_log_probs<B: BackendTrait>(
    student_log_probs: Tensor<B, 3>,
    teacher_log_probs: Tensor<B, 3>,
    mask: Option<Tensor<B, 2, Int>>,
    kind: SelfDistillationKlKind,
) -> Tensor<B, 1> {
    let per_token = match kind {
        SelfDistillationKlKind::Forward => {
            kl_per_token(teacher_log_probs.clone(), student_log_probs.clone())
        }
        SelfDistillationKlKind::Reverse => kl_per_token(student_log_probs, teacher_log_probs),
        SelfDistillationKlKind::JensenShannon => {
            let student_prob = student_log_probs.clone().exp();
            let teacher_prob = teacher_log_probs.clone().exp();
            let mixture_log_probs = (student_prob + teacher_prob)
                .mul_scalar(0.5)
                .clamp_min(1e-12)
                .log();
            let teacher_kl = kl_per_token(teacher_log_probs, mixture_log_probs.clone());
            let student_kl = kl_per_token(student_log_probs, mixture_log_probs);
            (teacher_kl + student_kl).mul_scalar(0.5)
        }
    };
    masked_token_mean(per_token, mask)
}

pub fn self_distillation_loss_from_logits<B: BackendTrait>(
    student_logits: Tensor<B, 3>,
    teacher_logits: Tensor<B, 3>,
    mask: Option<Tensor<B, 2, Int>>,
    kind: SelfDistillationKlKind,
) -> Tensor<B, 1> {
    self_distillation_loss_from_log_probs(
        log_probs_from_logits(student_logits),
        log_probs_from_logits(teacher_logits),
        mask,
        kind,
    )
}

pub fn sdpo_token_advantage<B: BackendTrait>(
    teacher_token_log_probs: Tensor<B, 2>,
    student_token_log_probs: Tensor<B, 2>,
    mask: Option<Tensor<B, 2, Int>>,
    normalize: bool,
    epsilon: f32,
) -> Tensor<B, 2> {
    let advantage = teacher_token_log_probs - student_token_log_probs;
    let advantage = if let Some(mask) = mask {
        advantage * mask.float()
    } else {
        advantage
    };
    if !normalize {
        return advantage;
    }
    let [batch, time] = advantage.shape().dims();
    let mean = advantage
        .clone()
        .mean_dim(0)
        .mean_dim(1)
        .repeat_dim(0, batch)
        .repeat_dim(1, time);
    let centered = advantage - mean;
    let var = centered
        .clone()
        .powf_scalar(2.0)
        .mean_dim(0)
        .mean_dim(1)
        .repeat_dim(0, batch)
        .repeat_dim(1, time);
    centered / var.add_scalar(epsilon.max(1e-12)).sqrt()
}

pub fn clipped_policy_loss<B: BackendTrait>(
    log_prob_new: Tensor<B, 2>,
    log_prob_old: Tensor<B, 2>,
    advantage: Tensor<B, 2>,
    mask: Option<Tensor<B, 2, Int>>,
    clip_range: Option<f32>,
    weight: f32,
) -> Tensor<B, 1> {
    let weight = weight.max(0.0);
    if weight <= 0.0 {
        return Tensor::<B, 1>::zeros([1], &log_prob_new.device());
    }
    let objective = if let Some(clip) = clip_range.filter(|clip| *clip > 0.0) {
        let log_ratio = (log_prob_new - log_prob_old)
            .clamp_min(-20.0)
            .clamp_max(20.0);
        let ratio = log_ratio.exp();
        let clipped = ratio.clone().clamp_min(1.0 - clip).clamp_max(1.0 + clip);
        let surrogate = ratio * advantage.clone();
        let surrogate_clipped = clipped * advantage;
        let use_clipped = surrogate_clipped.clone().lower_equal(surrogate.clone());
        surrogate.mask_where(use_clipped, surrogate_clipped)
    } else {
        log_prob_new * advantage
    };
    masked_token_mean(objective.mul_scalar(-weight), mask)
}

fn kl_per_token<B: BackendTrait>(
    left_log_probs: Tensor<B, 3>,
    right_log_probs: Tensor<B, 3>,
) -> Tensor<B, 2> {
    let [batch, time, _vocab] = left_log_probs.shape().dims();
    (left_log_probs.clone().exp() * (left_log_probs - right_log_probs))
        .sum_dim(2)
        .reshape([batch, time])
}

fn masked_token_mean<B: BackendTrait>(
    values: Tensor<B, 2>,
    mask: Option<Tensor<B, 2, Int>>,
) -> Tensor<B, 1> {
    if let Some(mask) = mask {
        let mask = mask.float();
        return (values * mask.clone())
            .sum()
            .div(mask.sum().clamp_min(1.0))
            .reshape([1]);
    }
    values.mean().reshape([1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn device() -> <TestBackend as BackendTrait>::Device {
        Default::default()
    }

    fn tensor3(values: Vec<f32>, shape: [usize; 3]) -> Tensor<TestBackend, 3> {
        Tensor::<TestBackend, 3>::from_data(TensorData::new(values, shape), &device())
    }

    fn tensor2(values: Vec<i64>, shape: [usize; 2]) -> Tensor<TestBackend, 2, Int> {
        Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(values, shape), &device())
    }

    fn scalar(value: Tensor<TestBackend, 1>) -> f32 {
        value
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("scalar vec")[0]
    }

    #[test]
    fn forward_kl_matches_hand_fixture() {
        let student = log_probs_from_logits(tensor3(vec![0.0, 0.0], [1, 1, 2]));
        let teacher = log_probs_from_logits(tensor3(vec![2.0, 0.0], [1, 1, 2]));
        let loss = scalar(self_distillation_loss_from_log_probs(
            student,
            teacher,
            None,
            SelfDistillationKlKind::Forward,
        ));
        let p0 = 2.0_f32.exp() / (2.0_f32.exp() + 1.0);
        let p1 = 1.0 / (2.0_f32.exp() + 1.0);
        let expected = p0 * (p0.ln() - 0.5_f32.ln()) + p1 * (p1.ln() - 0.5_f32.ln());
        assert!((loss - expected).abs() < 1e-5);
    }

    #[test]
    fn mask_excludes_tokens_from_distillation_mean() {
        let student = log_probs_from_logits(tensor3(vec![0.0, 0.0, 8.0, -8.0], [1, 2, 2]));
        let teacher = log_probs_from_logits(tensor3(vec![2.0, 0.0, -8.0, 8.0], [1, 2, 2]));
        let masked = scalar(self_distillation_loss_from_log_probs(
            student.clone(),
            teacher.clone(),
            Some(tensor2(vec![1, 0], [1, 2])),
            SelfDistillationKlKind::Forward,
        ));
        let first_only = scalar(self_distillation_loss_from_log_probs(
            student.slice([0..1, 0..1, 0..2]),
            teacher.slice([0..1, 0..1, 0..2]),
            None,
            SelfDistillationKlKind::Forward,
        ));
        assert!((masked - first_only).abs() < 1e-5);
    }

    #[test]
    fn selected_token_log_probs_gathers_targets() {
        let log_probs = log_probs_from_logits(tensor3(vec![0.0, 2.0, 3.0, 1.0], [1, 2, 2]));
        let selected = selected_token_log_probs(log_probs, tensor2(vec![0, 0], [1, 2]))
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("selected vec");
        assert_eq!(selected.len(), 2);
        assert!(selected[0] < selected[1]);
    }

    #[test]
    fn clipped_policy_loss_limits_positive_ratio() {
        let log_prob_new =
            Tensor::<TestBackend, 2>::from_data(TensorData::new(vec![0.3], [1, 1]), &device());
        let log_prob_old = Tensor::<TestBackend, 2>::zeros([1, 1], &device());
        let advantage = Tensor::<TestBackend, 2>::ones([1, 1], &device());
        let loss = scalar(clipped_policy_loss(
            log_prob_new,
            log_prob_old,
            advantage,
            None,
            Some(0.2),
            1.0,
        ));
        assert!((loss + 1.2).abs() < 1e-3);
    }

    #[test]
    fn sdpo_advantage_normalizes_teacher_student_delta() {
        let teacher =
            Tensor::<TestBackend, 2>::from_data(TensorData::new(vec![0.0, 2.0], [1, 2]), &device());
        let student = Tensor::<TestBackend, 2>::zeros([1, 2], &device());
        let advantage = sdpo_token_advantage(teacher, student, None, true, 1e-6)
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("advantage vec");
        assert!(advantage[0] < 0.0);
        assert!(advantage[1] > 0.0);
        assert!((advantage[0] + advantage[1]).abs() < 1e-5);
    }
}
