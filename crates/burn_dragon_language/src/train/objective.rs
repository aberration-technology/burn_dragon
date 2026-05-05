use crate::config::TrainingObjectiveConfig;

pub use burn_dragon_core::objective::{
    ObjectiveSupport, ObjectiveTrainerKind, SelectedTokenDistillationHiddenBatch,
    SelectedTokenSdpoLossConfig, SelfDistillationObjectiveKind, WindowSelfDistillationObjective,
    WindowSelfDistillationSmokeObjective, clipped_policy_loss,
    ensure_objective_supported as ensure_objective_kind_supported, log_probs_from_logits,
    masked_token_mean, objective_support as objective_kind_support, sdpo_token_advantage,
    selected_token_distillation_loss_from_hidden, selected_token_log_prob_mse_loss,
    selected_token_log_probs, selected_token_log_probs_from_hidden,
    selected_token_sdpo_loss_from_hidden, self_distillation_loss_from_log_probs,
    self_distillation_loss_from_logits, self_distillation_per_token_from_log_probs,
    window_sdft_loss, window_sdpo_loss, window_self_distillation_loss,
    window_self_distillation_smoke_loss, window_smoke_sdft_loss, window_smoke_sdpo_loss,
};

pub fn objective_support(
    objective: &TrainingObjectiveConfig,
    trainer: ObjectiveTrainerKind,
) -> ObjectiveSupport {
    objective_kind_support(objective.kind(), trainer)
}

pub fn ensure_objective_supported(
    objective: &TrainingObjectiveConfig,
    trainer: ObjectiveTrainerKind,
) -> anyhow::Result<()> {
    ensure_objective_kind_supported(objective.kind(), trainer).map_err(anyhow::Error::msg)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RolloutObjectiveRuntimeConstraints {
    pub uses_flat_token_logits: bool,
    pub distributed_pipeline: bool,
    pub tbptt_enabled: bool,
}

pub fn ensure_rollout_objective_runtime(
    objective: &TrainingObjectiveConfig,
    constraints: RolloutObjectiveRuntimeConstraints,
) -> anyhow::Result<()> {
    if objective.is_next_token() {
        return Ok(());
    }
    let kind = objective.kind();
    if !constraints.uses_flat_token_logits {
        return Err(anyhow::anyhow!(
            "training.objective.type={:?} requires language_head.type=\"standard_token_classification\" for paper-aligned full-logit distillation",
            kind
        ));
    }
    if constraints.distributed_pipeline {
        return Err(anyhow::anyhow!(
            "training.objective.type={:?} with distributed process-group pipeline training is not wired yet; use single-process pipeline, single-device, or DDP objective training",
            kind
        ));
    }
    if constraints.tbptt_enabled {
        return Err(anyhow::anyhow!(
            "training.objective.type={:?} does not yet support tbptt_chunk_size or tbptt_persist_across_steps",
            kind
        ));
    }
    Ok(())
}

pub fn assert_flat_logits_for_rollout_objective(
    objective: &TrainingObjectiveConfig,
    uses_factorized_language_head: bool,
) {
    assert!(
        objective.is_next_token() || !uses_factorized_language_head,
        "paper-aligned SDFT/SDPO rollout objectives require flat token logits; factorized heads only expose selected-token smoke losses"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SelfDistillationKlKind;
    use burn::tensor::TensorData;
    use burn::tensor::backend::Backend as BackendTrait;
    use burn::tensor::{Int, Tensor};
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
    fn support_matrix_distinguishes_active_native_rollouts_from_guarded_trainers() {
        let sdft = TrainingObjectiveConfig::Sdft(Default::default());
        let sdpo = TrainingObjectiveConfig::Sdpo(Default::default());
        let composite = TrainingObjectiveConfig::SdftSdpo(Default::default());

        assert_eq!(
            objective_support(
                &TrainingObjectiveConfig::NextToken,
                ObjectiveTrainerKind::Browser
            ),
            ObjectiveSupport::FullySupported
        );
        assert_eq!(
            objective_support(&sdft, ObjectiveTrainerKind::SingleDevice),
            ObjectiveSupport::FullySupported
        );
        assert_eq!(
            objective_support(&sdpo, ObjectiveTrainerKind::Ddp),
            ObjectiveSupport::FullySupported
        );
        assert_eq!(
            objective_support(&sdft, ObjectiveTrainerKind::Pipeline),
            ObjectiveSupport::FullySupported
        );
        assert_eq!(
            objective_support(&sdpo, ObjectiveTrainerKind::Browser),
            ObjectiveSupport::ConfigAndNumericsOnly
        );
        assert_eq!(
            objective_support(&composite, ObjectiveTrainerKind::SingleDevice),
            ObjectiveSupport::FullySupported
        );
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

    #[test]
    fn selected_token_distillation_matches_teacher_log_prob() {
        let student =
            Tensor::<TestBackend, 2>::from_data(TensorData::new(vec![-2.0], [1, 1]), &device());
        let teacher =
            Tensor::<TestBackend, 2>::from_data(TensorData::new(vec![-0.5], [1, 1]), &device());
        let loss = scalar(selected_token_log_prob_mse_loss(student, teacher, None));
        assert!((loss - 2.25).abs() < 1e-5);
    }
}
