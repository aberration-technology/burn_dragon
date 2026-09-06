use super::*;
use burn::optim::{AdamWConfig, SgdConfig};
use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;

type TestBackend = Autodiff<NdArray<f32>>;
type TestInnerBackend = NdArray<f32>;

fn tensor_scalar(tensor: Tensor<TestBackend, 1>) -> f32 {
    tensor
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("scalar tensor")[0]
}

fn tiny_model_config() -> DragonConfig {
    DragonConfig {
        n_layer: 1,
        n_embd: 8,
        n_head: 1,
        mlp_internal_dim_multiplier: 1,
        dropout: 0.0,
        vocab_size: 16,
        ..Default::default()
    }
}

#[test]
fn causal_predictive_coding_cadence_uses_post_observation_chunks() {
    let due = |step, chunk| {
        predictive_coding_chunk_due(
            PredictiveCodingObservationContract::ObservedPrefix,
            step,
            chunk,
            4,
            2,
        )
    };

    assert_eq!(
        (0..4).map(|chunk| due(0, chunk)).collect::<Vec<_>>(),
        vec![false, true, false, true]
    );
    assert_eq!(
        (0..4).map(|chunk| due(1, chunk)).collect::<Vec<_>>(),
        vec![false, true, false, true]
    );
}

#[test]
fn causal_predictive_coding_sparse_cadence_crosses_step_boundaries() {
    let due = |step, chunk| {
        predictive_coding_chunk_due(
            PredictiveCodingObservationContract::ObservedPrefix,
            step,
            chunk,
            4,
            8,
        )
    };

    assert!((0..4).all(|chunk| !due(0, chunk)));
    assert_eq!(
        (0..4).map(|chunk| due(1, chunk)).collect::<Vec<_>>(),
        vec![false, false, false, true]
    );
}

#[test]
fn oracle_negative_control_preserves_historical_cadence_phase() {
    let due = |chunk| {
        predictive_coding_chunk_due(
            PredictiveCodingObservationContract::OracleNextTokenNegativeControl,
            0,
            chunk,
            4,
            2,
        )
    };

    assert_eq!(
        (0..4).map(due).collect::<Vec<_>>(),
        vec![true, false, true, false]
    );
}

#[test]
fn stochastic_step_streams_are_reproducible_and_domain_separated() {
    let base = 1_337;
    let main = stochastic_step_seed(base, 19, STOCHASTIC_STREAM_MAIN);
    assert_eq!(main, stochastic_step_seed(base, 19, STOCHASTIC_STREAM_MAIN));
    assert_ne!(main, stochastic_step_seed(base, 20, STOCHASTIC_STREAM_MAIN));
    assert_ne!(
        main,
        stochastic_step_seed(base, 19, STOCHASTIC_STREAM_PROOF_POLICY)
    );
    assert_ne!(
        stochastic_step_seed(base, 19, STOCHASTIC_STREAM_PROOF_POLICY),
        stochastic_step_seed(base, 19, STOCHASTIC_STREAM_VERIFIER_POLICY)
    );
}

#[test]
fn streaming_state_is_pipeline_owned_and_clone_shared() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let model_a = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_tbptt_persist_across_steps(true);
    let model_b = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_tbptt_persist_across_steps(true);
    let mut state = model_a.model.init_state();
    state.position = 17;
    model_a.store_step_state(state);

    assert_eq!(
        model_a
            .peek_step_state_for_test()
            .expect("model a state")
            .position,
        17
    );
    assert!(
        model_b.peek_step_state_for_test().is_none(),
        "independent pipelines must not share recurrent state"
    );
    let cloned = model_a.clone();
    assert_eq!(
        cloned
            .peek_step_state_for_test()
            .expect("cloned learner state")
            .position,
        17,
        "Burn learner clones must retain the same pipeline runtime cell"
    );
    assert_eq!(model_a.load_step_state(true, 4).position, 0);
    assert!(model_a.peek_step_state_for_test().is_none());
}

#[test]
fn feedforward_local_pc_solvers_carry_and_reset_tbptt_rho_state() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let batch = |reset_stream_state| SequenceBatch {
        inputs: Tensor::from_data(
            TensorData::new(vec![1_i64, 2, 3, 4, 5, 6, 7, 8], [1, 8]),
            &device,
        ),
        targets: Tensor::from_data(
            TensorData::new(vec![2_i64, 3, 4, 5, 6, 7, 8, 9], [1, 8]),
            &device,
        ),
        loss_mask: None,
        supervised_token_count: None,
        summary_event_mask: None,
        ruliad_policy_batch: None,
        absolute_step: None,
        reset_stream_state,
    };

    for solver in [
        LocalPredictiveCodingSolver::FixedPrediction,
        LocalPredictiveCodingSolver::LayerLocalPrediction,
    ] {
        let mut model_config = tiny_model_config();
        model_config.n_layer = 2;
        model_config.sequence_kernel =
            burn_dragon_core::SequenceKernelConfig::dense_score_short_context();
        model_config.fused_kernels.rotary_embedding = burn_dragon_core::RotaryEmbedding::Alibi;
        let factor_reduction =
            if matches!(solver, LocalPredictiveCodingSolver::LayerLocalPrediction) {
                PredictiveCodingFactorReduction::Mean
            } else {
                PredictiveCodingFactorReduction::Sum
            };
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(model_config, &device))
            .with_training_algorithm(TrainingAlgorithm::PredictiveCoding)
            .with_local_predictive_coding(LocalPredictiveCodingConfig {
                solver,
                factor_reduction,
                ..LocalPredictiveCodingConfig::default()
            })
            .with_tbptt_chunk_size(Some(2))
            .with_tbptt_persist_across_steps(true);

        let first = burn_train::TrainStep::step(&model, batch(true));
        assert_eq!(first.grads.len(), 9, "solver={solver:?}");
        let first_state = model
            .peek_step_state_for_test()
            .expect("persistent local PC state after first step");
        assert_eq!(first_state.position, 8, "solver={solver:?}");
        assert!(
            first_state.layers.iter().all(|layer| layer.rho.is_some()),
            "solver={solver:?}"
        );

        let second = burn_train::TrainStep::step(&model, batch(false));
        assert_eq!(second.grads.len(), 9, "solver={solver:?}");
        assert_eq!(
            model
                .peek_step_state_for_test()
                .expect("persistent local PC state after second step")
                .position,
            16,
            "solver={solver:?}"
        );

        let reset = burn_train::TrainStep::step(&model, batch(true));
        assert_eq!(reset.grads.len(), 9, "solver={solver:?}");
        assert_eq!(
            model
                .peek_step_state_for_test()
                .expect("reset local PC state")
                .position,
            8,
            "solver={solver:?}"
        );
    }
}

#[test]
fn incremental_local_pc_interleaves_one_optimizer_update_per_inference_step() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let mut model_config = tiny_model_config();
    model_config.sequence_kernel =
        burn_dragon_core::SequenceKernelConfig::dense_score_short_context();
    model_config.fused_kernels.rotary_embedding = burn_dragon_core::RotaryEmbedding::Alibi;
    let config = LocalPredictiveCodingConfig {
        solver: LocalPredictiveCodingSolver::ReverseGaussSeidel,
        learning_schedule: burn_pc::PcLearningSchedule::Incremental,
        inference: burn_pc::PcInferenceConfig {
            steps: 3,
            ..burn_pc::PcInferenceConfig::default()
        },
        incremental_parameter_step_scale: 0.25,
        ..LocalPredictiveCodingConfig::default()
    };
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(model_config, &device))
        .with_training_algorithm(TrainingAlgorithm::PredictiveCoding)
        .with_local_predictive_coding(config);
    let profile = model.local_predictive_coding_profile();
    let batch = SequenceBatch {
        inputs: Tensor::from_data(TensorData::new(vec![1_i64, 2, 3, 4], [1, 4]), &device),
        targets: Tensor::from_data(TensorData::new(vec![2_i64, 3, 4, 5], [1, 4]), &device),
        loss_mask: None,
        supervised_token_count: None,
        summary_event_mask: None,
        ruliad_policy_batch: None,
        absolute_step: None,
        reset_stream_state: true,
    };
    let step = burn_train::TrainStep::step(&model, batch);
    assert_eq!(
        step.grads.len(),
        0,
        "iPC owns updates at the optimizer boundary"
    );
    let mut optimizer = SgdConfig::new().init::<TestBackend, LanguageTrainModel<TestBackend>>();
    let model = burn_train::TrainStep::optimize::<TestBackend, _>(
        model,
        &mut optimizer,
        1.0e-3,
        step.grads,
    );

    let snapshot = profile.snapshot();
    assert_eq!(snapshot.steps, 1);
    assert_eq!(snapshot.inference_steps, 3);
    assert_eq!(snapshot.global_backward_calls, 0);
    assert_eq!(snapshot.gradient_tensors, 27);
    assert_eq!(snapshot.parameter_updates, 3);
    assert!(
        model
            .incremental_predictive_coding_runtime
            .inner
            .lock()
            .expect("incremental PC runtime lock")
            .is_none(),
        "the optimizer boundary must consume staged run-local state"
    );
    assert_eq!(
        model.gradient_scale_step_index(),
        2,
        "the zero-based update index must advance once per inference phase"
    );
}

#[test]
fn direct_kolen_pollack_owns_two_ordered_updates_and_persists_feedback() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let mut model_config = tiny_model_config();
    model_config.n_layer = 2;
    model_config.sequence_kernel =
        burn_dragon_core::SequenceKernelConfig::dense_score_short_context();
    model_config.fused_kernels.rotary_embedding = burn_dragon_core::RotaryEmbedding::Alibi;
    let config = LocalPredictiveCodingConfig {
        solver: LocalPredictiveCodingSolver::DirectKolenPollack,
        inference: burn_pc::PcInferenceConfig {
            steps: 1,
            max_grad_norm: None,
            ..burn_pc::PcInferenceConfig::default()
        },
        direct_feedback: burn_pc::PcDirectFeedbackConfig {
            preliminary_step_size: 0.25,
            feedback_step_size: 0.1,
            ..burn_pc::PcDirectFeedbackConfig::default()
        },
        tied_consensus: burn_pc::PcTiedConsensusConfig {
            damping: 0.0,
            ..burn_pc::PcTiedConsensusConfig::default()
        },
        ..LocalPredictiveCodingConfig::default()
    };
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(model_config, &device))
        .with_training_algorithm(TrainingAlgorithm::PredictiveCoding)
        .with_local_predictive_coding(config)
        .with_tbptt_chunk_size(Some(2));
    let encoder_before = model.model.shared_lowrank_effective_weights().encoder;
    let profile = model.local_predictive_coding_profile();
    let batch = || SequenceBatch {
        inputs: Tensor::from_data(TensorData::new(vec![1_i64, 2, 3, 4], [1, 4]), &device),
        targets: Tensor::from_data(TensorData::new(vec![2_i64, 3, 4, 5], [1, 4]), &device),
        loss_mask: None,
        supervised_token_count: None,
        summary_event_mask: None,
        ruliad_policy_batch: None,
        absolute_step: None,
        reset_stream_state: true,
    };

    let step = burn_train::TrainStep::step(&model, batch());
    assert_eq!(step.grads.len(), 0, "DKP stages optimizer-owned updates");
    let mut optimizer = SgdConfig::new().init::<TestBackend, LanguageTrainModel<TestBackend>>();
    let model = burn_train::TrainStep::optimize::<TestBackend, _>(
        model,
        &mut optimizer,
        1.0e-3,
        step.grads,
    );
    let first_feedback = model
        .dkp_feedback_for_checkpoint()
        .expect("DKP feedback after first update");
    let first_feedback_tensor = first_feedback.feedback.clone();
    assert_eq!(first_feedback.updates, 2);
    assert_eq!(first_feedback.feedback.shape().dims::<3>(), [2, 8, 8]);
    let encoder_delta = tensor_scalar(
        (model.model.shared_lowrank_effective_weights().encoder - encoder_before)
            .abs()
            .max()
            .reshape([1]),
    );
    assert!(
        encoder_delta > 0.0,
        "the ordered DKP phases must update tied forward parameters"
    );
    assert_eq!(model.gradient_scale_step_index(), 3);

    let step = burn_train::TrainStep::step(&model, batch());
    let model = burn_train::TrainStep::optimize::<TestBackend, _>(
        model,
        &mut optimizer,
        1.0e-3,
        step.grads,
    );
    let second_feedback = model
        .dkp_feedback_for_checkpoint()
        .expect("persistent DKP feedback after second update");
    assert_eq!(second_feedback.updates, 4);
    let feedback_delta = tensor_scalar(
        (second_feedback.feedback - first_feedback_tensor)
            .abs()
            .max()
            .reshape([1]),
    );
    assert!(
        feedback_delta > 0.0,
        "the feedback side channel must learn across optimizer steps"
    );
    assert_eq!(model.gradient_scale_step_index(), 7);

    let snapshot = profile.snapshot();
    assert_eq!(snapshot.steps, 4);
    assert_eq!(snapshot.global_backward_calls, 0);
    assert_eq!(snapshot.direct_forward_updates, 8);
    assert_eq!(snapshot.feedback_parameter_updates, 8);
    assert_eq!(snapshot.parameter_updates, 8);
    assert_eq!(snapshot.inference_steps, 4);
}

#[test]
fn amortized_dkp_alternates_exact_local_teachers_with_cheap_feedback_updates() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let mut model_config = tiny_model_config();
    model_config.n_layer = 2;
    model_config.sequence_kernel =
        burn_dragon_core::SequenceKernelConfig::dense_score_short_context();
    model_config.fused_kernels.rotary_embedding = burn_dragon_core::RotaryEmbedding::Alibi;
    let config = LocalPredictiveCodingConfig {
        solver: LocalPredictiveCodingSolver::DirectKolenPollack,
        inference: burn_pc::PcInferenceConfig {
            steps: 1,
            max_grad_norm: None,
            ..burn_pc::PcInferenceConfig::default()
        },
        amortized_adjoint: burn_pc::PcAmortizedAdjointConfig {
            enabled: true,
            teacher_every_updates: 2,
            calibration: burn_pc::PcAdjointCalibrationConfig {
                learning_rate: 0.1,
                max_update_norm: Some(1.0),
                ..burn_pc::PcAdjointCalibrationConfig::default()
            },
            ..burn_pc::PcAmortizedAdjointConfig::default()
        },
        ..LocalPredictiveCodingConfig::default()
    };
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(model_config, &device))
        .with_training_algorithm(TrainingAlgorithm::PredictiveCoding)
        .with_local_predictive_coding(config);
    let profile = model.local_predictive_coding_profile();
    let batch = || SequenceBatch {
        inputs: Tensor::from_data(TensorData::new(vec![1_i64, 2, 3, 4], [1, 4]), &device),
        targets: Tensor::from_data(TensorData::new(vec![2_i64, 3, 4, 5], [1, 4]), &device),
        loss_mask: None,
        supervised_token_count: None,
        summary_event_mask: None,
        ruliad_policy_batch: None,
        absolute_step: None,
        reset_stream_state: true,
    };
    let mut optimizer = SgdConfig::new().init::<TestBackend, LanguageTrainModel<TestBackend>>();

    let step = burn_train::TrainStep::step(&model, batch());
    let model = burn_train::TrainStep::optimize::<TestBackend, _>(
        model,
        &mut optimizer,
        1.0e-3,
        step.grads,
    );
    let after_teacher = profile.snapshot();
    assert_eq!(after_teacher.adjoint_teacher_updates, 2);
    assert_eq!(after_teacher.adjoint_local_updates, 0);
    assert_eq!(after_teacher.global_backward_calls, 0);

    let step = burn_train::TrainStep::step(&model, batch());
    let model = burn_train::TrainStep::optimize::<TestBackend, _>(
        model,
        &mut optimizer,
        1.0e-3,
        step.grads,
    );
    let after_local = profile.snapshot();
    assert_eq!(after_local.adjoint_teacher_updates, 2);
    assert_eq!(after_local.adjoint_local_updates, 2);
    assert_eq!(after_local.global_backward_calls, 0);
    let feedback = model
        .dkp_feedback_for_checkpoint()
        .expect("amortized feedback checkpoint");
    assert_eq!(feedback.updates, 2);
    assert!(tensor_scalar(feedback.feedback.abs().max().reshape([1])).is_finite());
}

#[test]
fn amortized_adjoint_emits_one_outer_update_and_persists_its_feedback_bank() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let mut model_config = tiny_model_config();
    model_config.n_layer = 2;
    model_config.sequence_kernel =
        burn_dragon_core::SequenceKernelConfig::dense_score_short_context();
    model_config.fused_kernels.rotary_embedding = burn_dragon_core::RotaryEmbedding::Alibi;
    let config = LocalPredictiveCodingConfig {
        solver: LocalPredictiveCodingSolver::AmortizedAdjoint,
        factor_reduction: PredictiveCodingFactorReduction::Sum,
        direct_feedback: burn_pc::PcDirectFeedbackConfig {
            initialization: burn_pc::PcFeedbackInitialization::Identity,
            ..burn_pc::PcDirectFeedbackConfig::default()
        },
        amortized_adjoint: burn_pc::PcAmortizedAdjointConfig {
            enabled: true,
            teacher_every_updates: 2,
            predictor: burn_pc::PcAdjointPredictorKind::ResidualConditioned,
            calibration: burn_pc::PcAdjointCalibrationConfig {
                learning_rate: 0.1,
                max_update_norm: Some(1.0),
                ..burn_pc::PcAdjointCalibrationConfig::default()
            },
            ..burn_pc::PcAmortizedAdjointConfig::default()
        },
        ..LocalPredictiveCodingConfig::default()
    };
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(model_config, &device))
        .with_training_algorithm(TrainingAlgorithm::PredictiveCoding)
        .with_local_predictive_coding(config);
    let profile = model.local_predictive_coding_profile();
    let batch = || SequenceBatch {
        inputs: Tensor::from_data(TensorData::new(vec![1_i64, 2, 3, 4], [1, 4]), &device),
        targets: Tensor::from_data(TensorData::new(vec![2_i64, 3, 4, 5], [1, 4]), &device),
        loss_mask: None,
        supervised_token_count: None,
        summary_event_mask: None,
        ruliad_policy_batch: None,
        absolute_step: None,
        reset_stream_state: true,
    };
    let mut optimizer = SgdConfig::new().init::<TestBackend, LanguageTrainModel<TestBackend>>();

    let teacher_step = burn_train::TrainStep::step(&model, batch());
    assert_eq!(teacher_step.grads.len(), 9);
    let model = burn_train::TrainStep::optimize::<TestBackend, _>(
        model,
        &mut optimizer,
        1.0e-3,
        teacher_step.grads,
    );
    let first_feedback = model
        .dkp_feedback_for_checkpoint()
        .expect("amortized-adjoint feedback after exact anchor");
    assert_eq!(first_feedback.updates, 1);
    let after_teacher = profile.snapshot();
    assert_eq!(after_teacher.adjoint_teacher_updates, 2);
    assert_eq!(after_teacher.adjoint_local_updates, 0);
    assert_eq!(after_teacher.parameter_updates, 1);
    assert_eq!(after_teacher.global_backward_calls, 0);

    let local_step = burn_train::TrainStep::step(&model, batch());
    assert_eq!(local_step.grads.len(), 9);
    let model = burn_train::TrainStep::optimize::<TestBackend, _>(
        model,
        &mut optimizer,
        1.0e-3,
        local_step.grads,
    );
    let second_feedback = model
        .dkp_feedback_for_checkpoint()
        .expect("amortized-adjoint feedback after local use");
    assert_eq!(second_feedback.updates, 2);
    let after_local = profile.snapshot();
    assert_eq!(after_local.adjoint_teacher_updates, 2);
    assert_eq!(after_local.adjoint_local_updates, 2);
    assert_eq!(after_local.parameter_updates, 2);
    assert_eq!(after_local.global_backward_calls, 0);
}

#[test]
fn first_order_adjoint_emits_one_update_without_feedback_state() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let mut model_config = tiny_model_config();
    model_config.n_layer = 2;
    model_config.sequence_kernel =
        burn_dragon_core::SequenceKernelConfig::dense_score_short_context();
    model_config.fused_kernels.rotary_embedding = burn_dragon_core::RotaryEmbedding::Alibi;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(model_config, &device))
        .with_training_algorithm(TrainingAlgorithm::PredictiveCoding)
        .with_local_predictive_coding(LocalPredictiveCodingConfig {
            solver: LocalPredictiveCodingSolver::FirstOrderAdjoint,
            factor_reduction: PredictiveCodingFactorReduction::Sum,
            ..LocalPredictiveCodingConfig::default()
        });
    let profile = model.local_predictive_coding_profile();
    let output = burn_train::TrainStep::step(
        &model,
        SequenceBatch {
            inputs: Tensor::from_data(TensorData::new(vec![1_i64, 2, 3, 4], [1, 4]), &device),
            targets: Tensor::from_data(TensorData::new(vec![2_i64, 3, 4, 5], [1, 4]), &device),
            loss_mask: None,
            supervised_token_count: None,
            summary_event_mask: None,
            ruliad_policy_batch: None,
            absolute_step: None,
            reset_stream_state: true,
        },
    );
    assert_eq!(output.grads.len(), 9);
    assert!(model.dkp_feedback_for_checkpoint().is_none());
    let snapshot = profile.snapshot();
    assert_eq!(snapshot.adjoint_teacher_updates, 0);
    assert_eq!(snapshot.adjoint_local_updates, 2);
    assert_eq!(snapshot.parameter_updates, 1);
    assert_eq!(snapshot.global_backward_calls, 0);
}

#[test]
fn empty_verifier_terminal_panel_falls_back_and_records_the_skip() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let mut model_config = tiny_model_config();
    model_config.sequence_kernel =
        burn_dragon_core::SequenceKernelConfig::dense_score_short_context();
    model_config.fused_kernels.rotary_embedding = burn_dragon_core::RotaryEmbedding::Alibi;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(model_config, &device))
        .with_training_algorithm(TrainingAlgorithm::PredictiveCoding)
        .with_local_predictive_coding(LocalPredictiveCodingConfig {
            solver: LocalPredictiveCodingSolver::FixedPrediction,
            terminal_criterion:
                crate::config::LocalPredictiveCodingTerminalCriterion::RuliadVerifierSet,
            ..LocalPredictiveCodingConfig::default()
        })
        .with_ruliad_supervision(RuliadSupervisionConfig {
            proof_policy: crate::config::RuliadProofPolicyTrainingConfig {
                enabled: true,
                mode: crate::config::RuliadProofPolicyTrainingMode::StaticExpert,
                every_steps: 1,
                start_after_steps: 0,
                stratified_difficulty_levels: 1,
                ..crate::config::RuliadProofPolicyTrainingConfig::default()
            },
            ..RuliadSupervisionConfig::default()
        });
    let profile = model.local_predictive_coding_profile();
    let output = burn_train::TrainStep::step(
        &model,
        SequenceBatch {
            inputs: Tensor::from_data(TensorData::new(vec![1_i64, 2, 3, 4], [1, 4]), &device),
            targets: Tensor::from_data(TensorData::new(vec![2_i64, 3, 4, 5], [1, 4]), &device),
            loss_mask: None,
            supervised_token_count: None,
            summary_event_mask: None,
            ruliad_policy_batch: None,
            absolute_step: None,
            reset_stream_state: true,
        },
    );
    assert!(!output.grads.is_empty());
    let snapshot = profile.snapshot();
    assert_eq!(snapshot.structured_terminal_steps, 0);
    assert_eq!(snapshot.structured_terminal_skipped_steps, 1);
    assert_eq!(snapshot.global_backward_calls, 0);
}

#[test]
fn required_verifier_terminal_fails_after_recording_a_missing_batch() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let directory = tempfile::tempdir().expect("telemetry directory");
    let telemetry_path = directory.path().join("proof-policy.jsonl");
    let mut model_config = tiny_model_config();
    model_config.sequence_kernel =
        burn_dragon_core::SequenceKernelConfig::dense_score_short_context();
    model_config.fused_kernels.rotary_embedding = burn_dragon_core::RotaryEmbedding::Alibi;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(model_config, &device))
        .with_training_algorithm(TrainingAlgorithm::PredictiveCoding)
        .with_local_predictive_coding(LocalPredictiveCodingConfig {
            solver: LocalPredictiveCodingSolver::FixedPrediction,
            terminal_criterion:
                crate::config::LocalPredictiveCodingTerminalCriterion::RuliadVerifierSet,
            ..LocalPredictiveCodingConfig::default()
        })
        .with_ruliad_supervision(RuliadSupervisionConfig {
            proof_policy: crate::config::RuliadProofPolicyTrainingConfig {
                enabled: true,
                require_scheduled_update: true,
                mode: crate::config::RuliadProofPolicyTrainingMode::StaticExpert,
                every_steps: 1,
                start_after_steps: 0,
                stratified_difficulty_levels: 1,
                ..crate::config::RuliadProofPolicyTrainingConfig::default()
            },
            ..RuliadSupervisionConfig::default()
        })
        .with_ruliad_proof_policy_telemetry_path(Some(telemetry_path.clone()));

    let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        burn_train::TrainStep::step(
            &model,
            SequenceBatch {
                inputs: Tensor::from_data(TensorData::new(vec![1_i64, 2, 3, 4], [1, 4]), &device),
                targets: Tensor::from_data(TensorData::new(vec![2_i64, 3, 4, 5], [1, 4]), &device),
                loss_mask: None,
                supervised_token_count: None,
                summary_event_mask: None,
                ruliad_policy_batch: None,
                absolute_step: Some(0),
                reset_stream_state: true,
            },
        )
    }));
    assert!(failure.is_err(), "required objective must fail closed");
    let event: serde_json::Value = serde_json::from_str(
        std::fs::read_to_string(telemetry_path)
            .expect("skip telemetry")
            .lines()
            .next()
            .expect("skip event"),
    )
    .expect("skip JSON");
    assert_eq!(event["skip_reason"], "missing_policy_batch");
}

#[test]
fn model_visited_verifier_terminal_preserves_stream_state_for_backprop_and_local_pc() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let mut model_config = tiny_model_config();
    model_config.vocab_size = 272;
    model_config.sequence_kernel =
        burn_dragon_core::SequenceKernelConfig::dense_score_short_context();
    model_config.fused_kernels.rotary_embedding = burn_dragon_core::RotaryEmbedding::Alibi;
    let proof_policy = crate::config::RuliadProofPolicyTrainingConfig {
        enabled: true,
        mode: crate::config::RuliadProofPolicyTrainingMode::Dagger,
        scoring: crate::config::RuliadProofPolicyScoring::CompletionLikelihood,
        gradient_scope: crate::config::RuliadProofPolicyGradientScope::FullModel,
        normalization: crate::config::RuliadProofPolicyNormalization::PrefixConditional,
        candidate_symmetry: crate::config::RuliadProofPolicyCandidateSymmetry::BalancedRotation,
        presentation_risk: crate::config::RuliadProofPolicyPresentationRisk::Mean,
        weight: 1.0,
        every_steps: 16,
        start_after_steps: 16,
        rollout_steps: 2,
        max_rows_per_update: 2,
        max_presentation_rows_per_update: 64,
        candidates: 4,
        max_completion_tokens: 128,
        stratified_difficulty_levels: 1,
        ..crate::config::RuliadProofPolicyTrainingConfig::default()
    };
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        model_config.clone(),
        &device,
    ))
    .with_training_algorithm(TrainingAlgorithm::Backpropagation)
    .with_local_predictive_coding(LocalPredictiveCodingConfig {
        terminal_criterion:
            crate::config::LocalPredictiveCodingTerminalCriterion::RuliadVerifierSet,
        ..LocalPredictiveCodingConfig::default()
    })
    .with_ruliad_supervision(RuliadSupervisionConfig {
        proof_policy,
        ..RuliadSupervisionConfig::default()
    })
    .with_tbptt_chunk_size(Some(64))
    .with_tbptt_persist_across_steps(true);
    let bundle = burn_dragon_universality::ruliad::formal::generate_formal_bundle(
        41,
        burn_dragon_universality::ruliad::formal::RuliadFormalGeneratorConfig {
            rewrite_depth: 2,
            leaf_count: 3,
            context_depth: 1,
            distractor_axioms: 1,
            ..Default::default()
        },
    )
    .expect("formal bundle");
    let proof_step_index = 1.min(bundle.certificate.step_count().saturating_sub(1));
    let actions = burn_dragon_universality::ruliad::oracle_proof_action_set(
        &bundle.problem,
        &bundle.certificate,
        proof_step_index,
        4,
    )
    .expect("proof actions");
    let answer_contract =
        burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep;
    let item = burn_dragon_universality::RuliadEvalItem {
        oracle_hash: bundle.problem.canonical_hash().expect("problem hash"),
        sample_index: 41,
        split: burn_dragon_universality::SampleSplit::Train,
        family: "formal_proof".to_string(),
        task_kind: burn_dragon_universality::RuliadTaskKind::SelectProofAction
            .label()
            .to_string(),
        math_domains: vec!["formal_proof".to_string()],
        reasoning_modes: vec!["proof_construction".to_string()],
        prompt: burn_dragon_universality::ruliad::ruliad_proof_action_prompt(
            &bundle.problem,
            &actions,
        )
        .expect("proof prompt"),
        expected_answer: burn_dragon_universality::ruliad::proof_action_answer(
            &actions,
            actions.selected_index,
            answer_contract,
        )
        .expect("proof answer"),
        difficulty_level: Some(0),
        spec: Some(burn_dragon_universality::RuliadSampleSpec::FormalProof {
            problem: bundle.problem,
            certificate: bundle.certificate,
            candidate: None,
            proof_step_index: Some(proof_step_index),
            action_presentation_rotation: Some(0),
            action_candidate_count: Some(actions.candidates.len()),
            action_answer_contract: answer_contract,
            task: burn_dragon_universality::RuliadTaskKind::SelectProofAction,
        }),
    };
    let policy_batch = Arc::new(crate::dataset::RuliadPolicyBatch {
        samples: vec![crate::dataset::RuliadPolicySample {
            item,
            prompt_tokens: vec![1],
        }],
        tokenization: burn_dragon_universality::RuliadTokenizationConfig::StructuredSymbolic {
            vocab_size: 272,
            eos_id: Some(271),
        },
        stop_token_id: Some(271),
        sampling_metadata: None,
    });
    let profile = model.local_predictive_coding_profile();
    model.gradient_scale_step.store(7, Ordering::Relaxed);
    let output = burn_train::TrainStep::step(
        &model,
        SequenceBatch {
            inputs: Tensor::zeros([1, 512], &device),
            targets: Tensor::zeros([1, 512], &device),
            loss_mask: None,
            supervised_token_count: None,
            summary_event_mask: None,
            ruliad_policy_batch: Some(policy_batch.clone()),
            absolute_step: Some(16),
            reset_stream_state: true,
        },
    );

    assert!(!output.grads.is_empty());
    let snapshot = profile.snapshot();
    assert_eq!(snapshot.steps, 1);
    assert_eq!(snapshot.global_backward_calls, 1);
    assert_eq!(snapshot.local_vjp_calls, 0);
    assert_eq!(snapshot.structured_terminal_steps, 1);
    assert_eq!(snapshot.structured_terminal_skipped_steps, 0);
    assert!(snapshot.structured_terminal_groups >= 1);
    assert!(snapshot.structured_terminal_rows > 0);
    assert_eq!(
        model
            .peek_step_state_for_test()
            .expect("backprop verifier update must preserve stream continuity")
            .position,
        512
    );

    let local_model =
        LanguageTrainModel::new(DragonModel::<TestBackend>::new(model_config, &device))
            .with_training_algorithm(TrainingAlgorithm::PredictiveCoding)
            .with_local_predictive_coding(LocalPredictiveCodingConfig {
                solver: LocalPredictiveCodingSolver::FixedPrediction,
                terminal_criterion:
                    crate::config::LocalPredictiveCodingTerminalCriterion::RuliadVerifierSet,
                ..LocalPredictiveCodingConfig::default()
            })
            .with_ruliad_supervision(RuliadSupervisionConfig {
                proof_policy,
                ..RuliadSupervisionConfig::default()
            })
            .with_tbptt_chunk_size(Some(64))
            .with_tbptt_persist_across_steps(true);
    let local_profile = local_model.local_predictive_coding_profile();
    local_model.gradient_scale_step.store(7, Ordering::Relaxed);
    let local_output = burn_train::TrainStep::step(
        &local_model,
        SequenceBatch {
            inputs: Tensor::zeros([1, 512], &device),
            targets: Tensor::zeros([1, 512], &device),
            loss_mask: None,
            supervised_token_count: None,
            summary_event_mask: None,
            ruliad_policy_batch: Some(policy_batch),
            absolute_step: Some(16),
            reset_stream_state: true,
        },
    );
    assert!(!local_output.grads.is_empty());
    let local_snapshot = local_profile.snapshot();
    assert_eq!(local_snapshot.global_backward_calls, 0);
    assert_eq!(local_snapshot.structured_terminal_steps, 1);
    assert_eq!(local_snapshot.structured_terminal_skipped_steps, 0);
    assert_eq!(
        local_model
            .peek_step_state_for_test()
            .expect("local-PC verifier update must preserve stream continuity")
            .position,
        512
    );
}

#[test]
fn local_pc_semantic_terminal_telemetry_reports_executed_scope() {
    let config = crate::config::RuliadProofPolicyTrainingConfig {
        enabled: true,
        scoring: crate::config::RuliadProofPolicyScoring::SemanticEnergy,
        target: crate::config::RuliadProofPolicyTarget::VerifiedProgressDistribution,
        gradient_scope: crate::config::RuliadProofPolicyGradientScope::ScoreHeadOnly,
        normalization: crate::config::RuliadProofPolicyNormalization::CandidateConditional,
        counterfactual_targets_per_state: 1,
        max_rows_per_update: 2,
        max_presentation_rows_per_update: 8,
        ..Default::default()
    };
    let stats = crate::train::local_predictive_coding::RuliadVerifierPanelStats {
        policy_batch_fingerprint: 71,
        objective_panel_fingerprint: 113,
        answer_contract: "semantic_step",
        configured_mode: "static_expert",
        effective_mode: "static_expert",
        semantic_states: 2,
        base_semantic_states: 1,
        counterfactual_semantic_states: 1,
        supervised_action_tokens: 40,
        candidate_target_tokens: 40,
        equivalent_target_tokens: 10,
        ..Default::default()
    };

    let consolidation = crate::config::RuliadConsolidationConfig {
        enabled: true,
        initial_unique_steps: 4,
        hold_steps: 20,
        novelty_interval_steps: 4,
        seed: 19,
    };
    let mut policy_batch = prompt_value_binding_policy_batch();
    let coordinate = consolidation.coordinate(16);
    policy_batch.sampling_metadata = Some(crate::dataset::RuliadPolicySamplingMetadata {
        logical_epoch_index: 3,
        logical_selection_step: 16,
        generation_epoch_index: 0,
        generation_step: coordinate.generation_step,
        released_unique_steps: coordinate.released_unique_steps,
        novel: coordinate.novel,
        consolidation_enabled: consolidation.enabled,
    });
    let telemetry = RuliadProofPolicyDaggerTelemetry::from_verifier_panel(&stats, config, 16, 2)
        .with_policy_sampling(Some(&policy_batch));
    assert_eq!(
        telemetry.objective,
        "semantic_sequence_energy_counterfactual_v1"
    );
    assert_eq!(telemetry.gradient_scope, "score_head_only");
    assert_eq!(telemetry.target, "verified_progress_distribution");
    assert_eq!(telemetry.supervised_action_tokens, 40);
    assert_eq!(telemetry.candidate_target_tokens, 40);
    assert_eq!(telemetry.equivalent_target_tokens, 10);
    assert_eq!(telemetry.policy_batch_fingerprint, 71);
    assert_eq!(telemetry.objective_panel_fingerprint, 113);
    assert_eq!(telemetry.mean_candidate_targets_per_row, 20.0);
    assert_eq!(telemetry.mean_equivalent_targets_per_row, 5.0);
    assert_eq!(telemetry.prefix_branch_rows, 0);
    assert!(telemetry.consolidation_enabled);
    assert_eq!(telemetry.consolidation_logical_epoch_index, 3);
    assert_eq!(telemetry.consolidation_logical_selection_step, 16);
    assert_eq!(telemetry.consolidation_generation_epoch_index, 0);
    assert!(!telemetry.consolidation_novel);
    assert_eq!(telemetry.consolidation_released_unique_steps, 4);
    assert!(telemetry.consolidation_generation_step < 4);
}

#[test]
fn incremental_local_pc_tbptt_updates_each_chunk_and_carries_rho() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let mut model_config = tiny_model_config();
    model_config.n_layer = 2;
    model_config.sequence_kernel =
        burn_dragon_core::SequenceKernelConfig::dense_score_short_context();
    model_config.fused_kernels.rotary_embedding = burn_dragon_core::RotaryEmbedding::Alibi;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(model_config, &device))
        .with_training_algorithm(TrainingAlgorithm::PredictiveCoding)
        .with_local_predictive_coding(LocalPredictiveCodingConfig {
            solver: LocalPredictiveCodingSolver::SynchronousEquilibrium,
            learning_schedule: burn_pc::PcLearningSchedule::Incremental,
            inference: burn_pc::PcInferenceConfig {
                steps: 2,
                ..burn_pc::PcInferenceConfig::default()
            },
            ..LocalPredictiveCodingConfig::default()
        })
        .with_tbptt_chunk_size(Some(2))
        .with_tbptt_persist_across_steps(true);
    let profile = model.local_predictive_coding_profile();
    let batch = SequenceBatch {
        inputs: Tensor::from_data(TensorData::new(vec![1_i64, 2, 3, 4], [1, 4]), &device),
        targets: Tensor::from_data(TensorData::new(vec![2_i64, 3, 4, 5], [1, 4]), &device),
        loss_mask: Some(Tensor::from_data(
            TensorData::new(vec![1_i64, 0, 1, 1], [1, 4]),
            &device,
        )),
        supervised_token_count: Some(3),
        summary_event_mask: None,
        ruliad_policy_batch: None,
        absolute_step: None,
        reset_stream_state: true,
    };
    let step = burn_train::TrainStep::step(&model, batch);
    let mut optimizer = SgdConfig::new().init::<TestBackend, LanguageTrainModel<TestBackend>>();
    let model = burn_train::TrainStep::optimize::<TestBackend, _>(
        model,
        &mut optimizer,
        1.0e-3,
        step.grads,
    );

    let snapshot = profile.snapshot();
    assert_eq!(snapshot.steps, 2, "one report per TBPTT factor");
    assert_eq!(snapshot.inference_steps, 4);
    assert_eq!(snapshot.gradient_tensors, 36);
    assert_eq!(snapshot.parameter_updates, 4);
    assert_eq!(
        model.gradient_scale_step_index(),
        3,
        "two chunks times two inference phases produce four updates"
    );
    let state = model
        .peek_step_state_for_test()
        .expect("incremental PC stores persistent rho state");
    assert_eq!(state.position, 4);
    assert!(state.layers.iter().all(|layer| layer.rho.is_some()));
}

#[test]
fn local_predictive_coding_tbptt_uses_supervised_token_loss_weighting() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let mut config = tiny_model_config();
    config.n_layer = 2;
    config.sequence_kernel = burn_dragon_core::SequenceKernelConfig::dense_score_short_context();
    config.fused_kernels.rotary_embedding = burn_dragon_core::RotaryEmbedding::Alibi;
    config.fused_kernels.relu_threshold = -0.25;
    let base =
        crate::train::test_support::deterministic_matrix_parameters(
            DragonModel::<TestBackend>::new(config, &device),
        );
    let make_model = |model| {
        LanguageTrainModel::new(model)
            .with_training_algorithm(TrainingAlgorithm::PredictiveCoding)
            .with_local_predictive_coding(LocalPredictiveCodingConfig {
                solver: LocalPredictiveCodingSolver::FixedPrediction,
                ..LocalPredictiveCodingConfig::default()
            })
    };
    let chunked = make_model(base).with_tbptt_chunk_size(Some(2));
    let batch = || SequenceBatch {
        inputs: Tensor::from_data(
            TensorData::new(vec![1_i64, 2, 3, 4, 5, 6, 7, 8], [1, 8]),
            &device,
        ),
        targets: Tensor::from_data(
            TensorData::new(vec![2_i64, 3, 4, 5, 6, 7, 8, 9], [1, 8]),
            &device,
        ),
        loss_mask: Some(Tensor::from_data(
            TensorData::new(vec![1_i64, 0, 0, 0, 1, 1, 1, 1], [1, 8]),
            &device,
        )),
        supervised_token_count: Some(5),
        summary_event_mask: None,
        ruliad_policy_batch: None,
        absolute_step: None,
        reset_stream_state: true,
    };

    let source = batch();
    let mut state = chunked.model.init_state_ephemeral();
    let mut weighted_loss = 0.0_f32;
    let mut supervised_tokens = 0.0_f32;
    let config = LocalPredictiveCodingConfig {
        solver: LocalPredictiveCodingSolver::FixedPrediction,
        ..LocalPredictiveCodingConfig::default()
    };
    for start in (0..8).step_by(2) {
        let end = start + 2;
        let step = crate::train::local_predictive_coding_derivatives_with_state(
            &chunked.model,
            LanguageTrainModel::<TestBackend>::slice_tokens(source.inputs.clone(), 1, start, end),
            LanguageTrainModel::<TestBackend>::slice_tokens(source.targets.clone(), 1, start, end),
            source
                .loss_mask
                .clone()
                .map(|mask| LanguageTrainModel::<TestBackend>::slice_tokens(mask, 1, start, end)),
            state,
            &config,
        )
        .expect("manual recurrent local-PC factor");
        let chunk_loss = burn_pc::diagnostic_scalar_f32(step.loss.inner());
        let chunk_tokens = burn_pc::diagnostic_scalar_f32(step.supervised_tokens.inner());
        weighted_loss += chunk_loss * chunk_tokens;
        supervised_tokens += chunk_tokens;
        state = step.terminal_state;
    }
    let expected_loss = weighted_loss / supervised_tokens.max(1.0);
    let chunked_loss = scalar_loss(burn_train::TrainStep::step(&chunked, batch()));
    assert!(
        (expected_loss - chunked_loss).abs() < 1.0e-5,
        "expected={expected_loss} chunked={chunked_loss}"
    );
}

#[test]
fn backprop_tbptt_matches_global_masked_objective_with_uneven_supervision() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let mut config = tiny_model_config();
    config.n_layer = 2;
    config.sequence_kernel = burn_dragon_core::SequenceKernelConfig::dense_score_short_context();
    config.fused_kernels.rotary_embedding = burn_dragon_core::RotaryEmbedding::Alibi;
    config.fused_kernels.relu_threshold = -0.25;
    let base =
        crate::train::test_support::deterministic_matrix_parameters(
            DragonModel::<TestBackend>::new(config, &device),
        );
    let reference = LanguageTrainModel::new(base.clone());
    let chunked = LanguageTrainModel::new(base).with_tbptt_chunk_size(Some(2));
    let batch = || SequenceBatch {
        inputs: Tensor::from_data(
            TensorData::new(vec![1_i64, 2, 3, 4, 5, 6, 7, 8], [1, 8]),
            &device,
        ),
        targets: Tensor::from_data(
            TensorData::new(vec![2_i64, 3, 4, 5, 6, 7, 8, 9], [1, 8]),
            &device,
        ),
        loss_mask: Some(Tensor::from_data(
            TensorData::new(vec![1_i64, 0, 0, 0, 1, 1, 1, 1], [1, 8]),
            &device,
        )),
        supervised_token_count: Some(5),
        summary_event_mask: None,
        ruliad_policy_batch: None,
        absolute_step: None,
        reset_stream_state: true,
    };

    let source = batch();
    let mut state = reference.model.init_state_ephemeral();
    let mut hidden_chunks = Vec::new();
    for start in (0..8).step_by(2) {
        let end = start + 2;
        hidden_chunks.push(reference.model.forward_hidden_with_state(
            LanguageTrainModel::<TestBackend>::slice_tokens(source.inputs.clone(), 1, start, end),
            &mut state,
        ));
        if end < 8 {
            state.detach_in_place();
        }
    }
    let reference_loss = reference.language_loss_from_hidden(
        Tensor::cat(hidden_chunks, 1),
        source.targets,
        source.loss_mask,
    );
    let expected_loss = tensor_scalar(reference_loss.clone());
    let reference_grads = GradientsParams::from_grads(reference_loss.backward(), &reference);

    let output = burn_train::TrainStep::step(&chunked, batch());
    let actual_loss = {
        let synced = output.item.sync();
        let loss: LossValue<TestInnerBackend> = synced.adapt();
        loss.value()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("chunked loss")[0]
    };
    assert!(
        (expected_loss - actual_loss).abs() < 1.0e-5,
        "global={expected_loss} chunked={actual_loss}"
    );

    let parameter_ids = reference
        .model
        .predictive_coding_parameter_ids()
        .expect("test model parameter ids");
    let chunked_grads = output.grads;
    macro_rules! assert_gradient_close {
        ($name:literal, $id:expr, $rank:literal) => {{
            match (
                reference_grads.get::<TestInnerBackend, $rank>($id),
                chunked_grads.get::<TestInnerBackend, $rank>($id),
            ) {
                (Some(expected), Some(actual)) => {
                    let max_error = (expected.clone() - actual)
                        .abs()
                        .max()
                        .to_data()
                        .convert::<f32>()
                        .into_vec::<f32>()
                        .expect("gradient error")[0];
                    let reference_scale = expected
                        .abs()
                        .max()
                        .to_data()
                        .convert::<f32>()
                        .into_vec::<f32>()
                        .expect("gradient scale")[0]
                        .max(1.0e-7);
                    assert!(
                        max_error / reference_scale < 1.0e-4,
                        "{} relative max gradient error: {}",
                        $name,
                        max_error / reference_scale
                    );
                }
                (None, None) => {}
                (expected, actual) => panic!(
                    "{} gradient presence mismatch: reference={} chunked={}",
                    $name,
                    expected.is_some(),
                    actual.is_some()
                ),
            }
        }};
    }
    assert_gradient_close!("embedding", parameter_ids.embedding, 2);
    assert_gradient_close!("shared encoder", parameter_ids.encoder, 3);
    assert_gradient_close!("shared value encoder", parameter_ids.encoder_v, 3);
    assert_gradient_close!("shared decoder", parameter_ids.decoder, 2);
    assert_gradient_close!("norm gamma", parameter_ids.norm_gamma, 1);
    assert_gradient_close!("norm beta", parameter_ids.norm_beta, 1);
    assert_gradient_close!("norm alpha", parameter_ids.norm_alpha, 1);
    assert_gradient_close!("norm shift", parameter_ids.norm_shift, 1);
    assert_gradient_close!("language head", parameter_ids.lm_head, 2);
}

#[test]
fn exact_temporal_pc_windows_match_bounded_recurrent_backprop_with_uneven_masks() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let mut config = tiny_model_config();
    config.n_layer = 2;
    config.sequence_kernel = burn_dragon_core::SequenceKernelConfig::dense_score_short_context();
    config.fused_kernels.rotary_embedding = burn_dragon_core::RotaryEmbedding::Alibi;
    // Keep exact temporal-gradient comparisons away from ReLU's undefined
    // zero subgradient. Projection support itself is covered in kernel tests.
    config.fused_kernels.relu_threshold = -0.25;
    let base =
        crate::train::test_support::deterministic_matrix_parameters(
            DragonModel::<TestBackend>::new(config, &device),
        );
    let batch = || SequenceBatch {
        inputs: Tensor::from_data(
            TensorData::new(vec![1_i64, 2, 3, 4, 5, 6, 7, 8], [1, 8]),
            &device,
        ),
        targets: Tensor::from_data(
            TensorData::new(vec![2_i64, 3, 4, 5, 6, 7, 8, 9], [1, 8]),
            &device,
        ),
        // The second chunk has no direct objective. It must still transport
        // the later chunks' rho adjoint through the recurrent state.
        loss_mask: Some(Tensor::from_data(
            TensorData::new(vec![1_i64, 0, 0, 0, 1, 1, 1, 1], [1, 8]),
            &device,
        )),
        supervised_token_count: Some(5),
        summary_event_mask: None,
        ruliad_policy_batch: None,
        absolute_step: None,
        reset_stream_state: true,
    };

    for window_chunks in [2, 4] {
        let reference = LanguageTrainModel::new(base.clone());
        let exact = LanguageTrainModel::new(base.clone())
            .with_training_algorithm(TrainingAlgorithm::PredictiveCoding)
            .with_local_predictive_coding(LocalPredictiveCodingConfig {
                solver: LocalPredictiveCodingSolver::FixedPrediction,
                factor_reduction: PredictiveCodingFactorReduction::Sum,
                temporal_credit: burn_pc::PcTemporalCreditConfig {
                    mode: burn_pc::PcTemporalCreditMode::ExactWindow,
                    window_chunks,
                },
                ..LocalPredictiveCodingConfig::default()
            })
            .with_tbptt_chunk_size(Some(2));
        let bounded_backprop = LanguageTrainModel::new(base.clone())
            .with_tbptt_chunk_size(Some(2))
            .with_tbptt_credit_window_chunks(window_chunks);

        let source = batch();
        let mut state = reference.model.init_state_ephemeral();
        let mut hidden_chunks = Vec::new();
        for (chunk_index, start) in (0..8).step_by(2).enumerate() {
            hidden_chunks.push(reference.model.forward_hidden_with_state(
                LanguageTrainModel::<TestBackend>::slice_tokens(
                    source.inputs.clone(),
                    1,
                    start,
                    start + 2,
                ),
                &mut state,
            ));
            if (chunk_index + 1) % window_chunks == 0 && start + 2 < 8 {
                state.detach_in_place();
            }
        }
        let reference_loss = reference.language_loss_from_hidden(
            Tensor::cat(hidden_chunks, 1),
            source.targets,
            source.loss_mask,
        );
        let expected_loss = tensor_scalar(reference_loss.clone());
        let reference_grads = GradientsParams::from_grads(reference_loss.backward(), &reference);

        let bounded_output = burn_train::TrainStep::step(&bounded_backprop, batch());
        let bounded_loss = {
            let synced = bounded_output.item.sync();
            let loss: LossValue<TestInnerBackend> = synced.adapt();
            loss.value()
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("bounded-backprop loss")[0]
        };
        assert!(
            (expected_loss - bounded_loss).abs() < 1.0e-5,
            "window={window_chunks} global={expected_loss} bounded_backprop={bounded_loss}"
        );

        let output = burn_train::TrainStep::step(&exact, batch());
        let actual_loss = {
            let synced = output.item.sync();
            let loss: LossValue<TestInnerBackend> = synced.adapt();
            loss.value()
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("exact-window loss")[0]
        };
        assert!(
            (expected_loss - actual_loss).abs() < 1.0e-5,
            "window={window_chunks} global={expected_loss} exact_window={actual_loss}"
        );

        let parameter_ids = reference
            .model
            .predictive_coding_parameter_ids()
            .expect("test model parameter ids");
        macro_rules! assert_gradient_close {
            ($implementation:literal, $name:literal, $id:expr, $rank:literal, $actual_grads:expr) => {{
                match (
                    reference_grads.get::<TestInnerBackend, $rank>($id),
                    $actual_grads.get::<TestInnerBackend, $rank>($id),
                ) {
                    (Some(expected), Some(actual)) => {
                        let max_error = (expected.clone() - actual)
                            .abs()
                            .max()
                            .to_data()
                            .convert::<f32>()
                            .into_vec::<f32>()
                            .expect("gradient error")[0];
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
                            "window={} {} {} relative max gradient error: {}",
                            window_chunks,
                            $implementation,
                            $name,
                            max_error / reference_scale
                        );
                    }
                    (None, Some(actual)) => {
                        let actual_scale = actual
                            .abs()
                            .max()
                            .to_data()
                            .convert::<f32>()
                            .into_vec::<f32>()
                            .expect("inactive gradient scale")[0];
                        assert!(
                            actual_scale < 1.0e-7,
                            "window={} {} {} should be inactive but gradient scale is {}",
                            window_chunks,
                            $implementation,
                            $name,
                            actual_scale
                        );
                    }
                    (None, None) => {}
                    (Some(_), None) => panic!(
                        "window={} {} {} gradient is missing",
                        window_chunks, $implementation, $name
                    ),
                }
            }};
        }
        macro_rules! assert_all_gradients_close {
            ($implementation:literal, $grads:expr) => {{
                assert_gradient_close!(
                    $implementation,
                    "embedding",
                    parameter_ids.embedding,
                    2,
                    $grads
                );
                assert_gradient_close!(
                    $implementation,
                    "shared encoder",
                    parameter_ids.encoder,
                    3,
                    $grads
                );
                assert_gradient_close!(
                    $implementation,
                    "shared value encoder",
                    parameter_ids.encoder_v,
                    3,
                    $grads
                );
                assert_gradient_close!(
                    $implementation,
                    "shared decoder",
                    parameter_ids.decoder,
                    2,
                    $grads
                );
                assert_gradient_close!(
                    $implementation,
                    "norm gamma",
                    parameter_ids.norm_gamma,
                    1,
                    $grads
                );
                assert_gradient_close!(
                    $implementation,
                    "norm beta",
                    parameter_ids.norm_beta,
                    1,
                    $grads
                );
                assert_gradient_close!(
                    $implementation,
                    "norm alpha",
                    parameter_ids.norm_alpha,
                    1,
                    $grads
                );
                assert_gradient_close!(
                    $implementation,
                    "norm shift",
                    parameter_ids.norm_shift,
                    1,
                    $grads
                );
                assert_gradient_close!(
                    $implementation,
                    "language head",
                    parameter_ids.lm_head,
                    2,
                    $grads
                );
            }};
        }
        assert_all_gradients_close!("bounded_backprop", bounded_output.grads);
        assert_all_gradients_close!("exact_pc", output.grads);

        let profile = exact.local_predictive_coding_profile().snapshot();
        assert_eq!(profile.steps, 4);
        assert_eq!(
            profile.temporal_state_vjp_calls,
            (2 * (4 - 4_usize.div_ceil(window_chunks))) as u64
        );
        assert_eq!(
            profile.fused_temporal_vjp_calls,
            profile.temporal_state_vjp_calls
        );
        assert_eq!(profile.global_backward_calls, 0);
    }
}

#[test]
fn sequence_state_diagnostics_detect_redundant_rho_slots() {
    let device = burn::tensor::Device::<TestInnerBackend>::default();
    let mut state = ModelState::<TestInnerBackend>::new(1);
    state.layers[0].rho = Some(Tensor::from_data(
        TensorData::new(
            vec![1.0f32, -1.0, 0.5, -0.5, 1.0, -1.0, 0.5, -0.5],
            [1, 1, 2, 4],
        ),
        &device,
    ));

    let diagnostics = LanguageTrainModel::<TestInnerBackend>::sequence_state_diagnostics(&state, 2)
        .expect("rho diagnostics");
    assert_eq!(diagnostics.rho_layers, 1);
    assert!((diagnostics.rho_rms - 0.790_569_4).abs() < 1.0e-5);
    assert!(diagnostics.rho_slot_variance_ratio.abs() < 1.0e-6);
    assert!((diagnostics.rho_slot_redundancy - 1.0).abs() < 1.0e-5);
}

#[test]
fn sequence_state_diagnostics_detect_distinct_rho_slots() {
    let device = burn::tensor::Device::<TestInnerBackend>::default();
    let mut state = ModelState::<TestInnerBackend>::new(1);
    state.layers[0].rho = Some(Tensor::from_data(
        TensorData::new(
            vec![1.0f32, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
            [1, 1, 2, 4],
        ),
        &device,
    ));

    let diagnostics = LanguageTrainModel::<TestInnerBackend>::sequence_state_diagnostics(&state, 2)
        .expect("rho diagnostics");
    assert!(diagnostics.rho_slot_variance_ratio > 0.49);
    assert!(diagnostics.rho_slot_redundancy < 1.0e-5);
}

#[test]
fn terminal_sequence_state_elision_requires_a_stateless_training_contract() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let mut config = tiny_model_config();
    config.sequence_kernel = burn_dragon_core::SequenceKernelConfig::dense_score_short_context();

    let baseline =
        LanguageTrainModel::new(DragonModel::<TestBackend>::new(config.clone(), &device));
    assert!(
        !baseline.load_step_state(false, 4).layers[0].retain_terminal_sequence_state,
        "an unchunked nonpersistent dense-score step should elide unused terminal state"
    );

    let retained =
        LanguageTrainModel::new(DragonModel::<TestBackend>::new(config.clone(), &device))
            .with_ephemeral_terminal_sequence_state_retention(true);
    assert!(retained.load_step_state(false, 4).layers[0].retain_terminal_sequence_state);

    let chunked = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config.clone(), &device))
        .with_tbptt_chunk_size(Some(2));
    assert!(chunked.load_step_state(false, 4).layers[0].retain_terminal_sequence_state);

    let persistent =
        LanguageTrainModel::new(DragonModel::<TestBackend>::new(config.clone(), &device))
            .with_tbptt_persist_across_steps(true);
    assert!(persistent.load_step_state(false, 4).layers[0].retain_terminal_sequence_state);

    let mut pipeline_config = config.clone();
    pipeline_config.n_layer = 2;
    let pipeline =
        LanguageTrainModel::new(DragonModel::<TestBackend>::new(pipeline_config, &device))
            .with_pipeline_plan(Some(tiny_pipeline_plan()));
    assert!(pipeline.load_step_state(false, 4).layers[0].retain_terminal_sequence_state);

    let predictive_coding = PredictiveCodingConfig {
        enabled: true,
        ..Default::default()
    };
    let predictive =
        LanguageTrainModel::new(DragonModel::<TestBackend>::new(config.clone(), &device))
            .with_predictive_coding(predictive_coding);
    assert!(predictive.load_step_state(false, 4).layers[0].retain_terminal_sequence_state);

    let latent_reasoning = LatentReasoningTrainingConfig {
        enabled: true,
        sigreg: crate::config::LatentReasoningSigRegConfig {
            target: crate::config::LatentReasoningSigRegTarget::RhoMemorySlots,
            ..Default::default()
        },
        ..Default::default()
    };
    let rho_regularized =
        LanguageTrainModel::new(DragonModel::<TestBackend>::new(config.clone(), &device))
            .with_latent_reasoning(latent_reasoning);
    assert!(rho_regularized.load_step_state(false, 4).layers[0].retain_terminal_sequence_state);

    let dragon_state_reasoning = LatentReasoningTrainingConfig {
        enabled: true,
        dragon_state: crate::config::DragonStateConsistencyConfig {
            enabled: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let dragon_state =
        LanguageTrainModel::new(DragonModel::<TestBackend>::new(config.clone(), &device))
            .with_latent_reasoning(dragon_state_reasoning);
    assert!(dragon_state.load_step_state(false, 4).layers[0].retain_terminal_sequence_state);

    let mut summary_memory_config = config;
    summary_memory_config.summary_memory.enabled = true;
    let summary_memory = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        summary_memory_config,
        &device,
    ));
    assert!(summary_memory.load_step_state(false, 4).layers[0].retain_terminal_sequence_state);

    let mut reference_config = tiny_model_config();
    reference_config.sequence_kernel = burn_dragon_core::SequenceKernelConfig::default();
    let reference =
        LanguageTrainModel::new(DragonModel::<TestBackend>::new(reference_config, &device));
    assert!(reference.load_step_state(false, 4).layers[0].retain_terminal_sequence_state);

    let mut multi_step_config = tiny_model_config();
    multi_step_config.sequence_kernel =
        burn_dragon_core::SequenceKernelConfig::dense_score_short_context();
    multi_step_config.rollout_fast_steps_per_slow_step = 2;
    let multi_step =
        LanguageTrainModel::new(DragonModel::<TestBackend>::new(multi_step_config, &device));
    assert!(multi_step.load_step_state(false, 4).layers[0].retain_terminal_sequence_state);

    let mut y_neuron_config = tiny_model_config();
    y_neuron_config.sequence_kernel =
        burn_dragon_core::SequenceKernelConfig::dense_score_short_context();
    y_neuron_config.y_neuron_recurrence.enabled = true;
    let y_neuron =
        LanguageTrainModel::new(DragonModel::<TestBackend>::new(y_neuron_config, &device));
    assert!(y_neuron.load_step_state(false, 4).layers[0].retain_terminal_sequence_state);

    let mut hierarchical_config = tiny_model_config();
    hierarchical_config.sequence_kernel =
        burn_dragon_core::SequenceKernelConfig::dense_score_short_context();
    hierarchical_config.hierarchical_dragon.enabled = true;
    let hierarchical = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        hierarchical_config,
        &device,
    ));
    assert!(hierarchical.load_step_state(false, 4).layers[0].retain_terminal_sequence_state);

    let mut clocked_config = tiny_model_config();
    clocked_config.sequence_kernel =
        burn_dragon_core::SequenceKernelConfig::dense_score_short_context();
    clocked_config.clocked_slow_memory.enabled = true;
    let clocked = LanguageTrainModel::new(DragonModel::<TestBackend>::new(clocked_config, &device));
    assert!(clocked.load_step_state(false, 4).layers[0].retain_terminal_sequence_state);
}

fn ruliad_test_score(
    status: burn_dragon_universality::ruliad::RuliadAnswerStatus,
    partial_progress_ppm: usize,
    completion_quality_ppm: usize,
) -> burn_dragon_universality::ruliad::RuliadReasoningScore {
    burn_dragon_universality::ruliad::RuliadReasoningScore {
        version: 1,
        status,
        correct_field_count: 0,
        expected_field_count: 1,
        observed_field_count: 0,
        partial_progress_ppm,
        certificate_valid_prefix_steps: 0,
        certificate_expected_steps: 0,
        certificate_prefix_ppm: 0,
        generated_token_count: 8,
        hash_canary: false,
        answer_terminated: true,
        completion_quality_ppm,
    }
}

fn tiny_factorized_model_config() -> DragonConfig {
    let mut config = tiny_model_config();
    config.vocab_size = 32;
    config.language_head = burn_dragon_core::LanguageHeadConfig::NcaFactorizedPatch {
        state_count: 2,
        patch_size: 2,
        frame_special_tokens: true,
        eos_id: Some(31),
    };
    config
}

fn tiny_pipeline_plan() -> PipelinePlan {
    build_pipeline_plan(
        2,
        &burn_dragon_train::ParallelPipelineConfig {
            enabled: true,
            stage_count: 2,
            virtual_stages_per_rank: 1,
            schedule: burn_dragon_train::PipelineScheduleKind::Interleaved1f1b,
            microbatches: 2,
            ..Default::default()
        },
    )
    .expect("pipeline plan")
}

fn batch(device: &burn::tensor::Device<TestBackend>) -> SequenceBatch<TestBackend> {
    SequenceBatch::new(
        Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![0, 1, 2, 3, 4, 5, 6, 7], [2, 4]),
            device,
        ),
        Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1, 2, 3, 4, 5, 6, 7, 8], [2, 4]),
            device,
        ),
        None,
    )
}

#[test]
fn all_active_context_stream_validation_matches_dense_tbptt() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7_311);
    let mut config = tiny_model_config();
    config.sequence_kernel.executor =
        burn_dragon_core::SequenceTrainingExecutor::DenseScoreShortContext;
    config.fused_kernels.rotary_embedding = burn_dragon_core::RotaryEmbedding::Alibi;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_tbptt_chunk_size(Some(2));
    model
        .model
        .predictive_coding_support()
        .expect("PC-compatible test model");
    let mut dense_state = model.model.init_state();
    let mut context_state = model.model.init_state();
    let dense = model.step_with_stream_state(batch(&device), &mut dense_state);
    let routed = model.step_with_predictive_context_stream_state(
        batch(&device),
        Tensor::ones([1, 1, 1, 8], &device),
        Tensor::ones([1, 1, 1, 8], &device),
        &mut context_state,
    );
    let dense_loss: LossValue<TestBackend> = dense.adapt();
    let routed_loss: LossValue<TestBackend> = routed.adapt();
    let loss_diff = (dense_loss.value() - routed_loss.value())
        .abs()
        .max()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("loss difference")[0];
    assert!(
        loss_diff < 1.0e-5,
        "routed stream loss mismatch: {loss_diff}"
    );
    assert_eq!(dense_state.position, context_state.position);
    let rho_diff = (dense_state.layers[0].rho.clone().expect("dense rho")
        - context_state.layers[0].rho.clone().expect("context rho"))
    .abs()
    .max()
    .to_data()
    .convert::<f32>()
    .into_vec::<f32>()
    .expect("rho difference")[0];
    assert!(rho_diff < 1.0e-5, "routed stream rho mismatch: {rho_diff}");
}

fn scalar_loss(output: TrainOutput<LanguageModelTrainItem<TestBackend>>) -> f32 {
    let synced = output.item.sync();
    let loss: LossValue<TestInnerBackend> = synced.adapt();
    loss.value()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("loss vec")[0]
}

#[test]
fn oracle_predictive_coding_negative_control_corrects_state() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 11);
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_tbptt_chunk_size(Some(2))
    .with_predictive_coding(PredictiveCodingConfig {
        enabled: true,
        observation_contract: PredictiveCodingObservationContract::OracleNextTokenNegativeControl,
        allow_oracle_target_leak: true,
        steps: 1,
        step_size: 0.01,
        sync_diagnostics: true,
        ..Default::default()
    });
    let batch = batch(&device);
    let [batch_size, _block_size] = batch.inputs.shape().dims();
    let mut state = model.model.init_state_ephemeral();
    let first_inputs =
        LanguageTrainModel::<TestBackend>::slice_tokens(batch.inputs.clone(), batch_size, 0, 2);
    let _ = model
        .model
        .forward_hidden_with_state(first_inputs, &mut state);
    state.detach_in_place();

    let second_inputs =
        LanguageTrainModel::<TestBackend>::slice_tokens(batch.inputs, batch_size, 2, 4);
    let second_targets =
        LanguageTrainModel::<TestBackend>::slice_tokens(batch.targets, batch_size, 2, 4);
    let (_corrected_state, report) = model.correct_state_with_oracle_predictive_coding(
        state,
        second_inputs,
        second_targets,
        None,
        None,
    );

    assert!(
        report.chunks_seen > 0,
        "PC should observe at least one TBPTT state handoff, report={report:?}"
    );
    assert!(
        report.chunks_corrected > 0,
        "PC should correct at least one recurrent state, report={report:?}"
    );
    assert!(
        report.chunks_corrected <= report.chunks_seen,
        "corrected chunks should be bounded by observed chunks, report={report:?}"
    );
    assert!(
        report
            .energy_before
            .zip(report.energy_after)
            .is_some_and(|(before, after)| before.is_finite() && after.is_finite()),
        "PC should record finite before/after energy, report={report:?}"
    );
}

#[test]
fn observed_prefix_predictive_coding_uses_no_future_target() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 13);
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_tbptt_chunk_size(Some(2))
    .with_predictive_coding(PredictiveCodingConfig {
        enabled: true,
        observation_contract: PredictiveCodingObservationContract::ObservedPrefix,
        steps: 1,
        step_size: 0.01,
        sync_diagnostics: true,
        ..Default::default()
    });
    let batch = batch(&device);
    let [batch_size, _block_size] = batch.inputs.shape().dims();
    let mut state = model.model.init_state_ephemeral();
    let first_inputs =
        LanguageTrainModel::<TestBackend>::slice_tokens(batch.inputs.clone(), batch_size, 0, 2);
    let _ = model
        .model
        .forward_hidden_with_state(first_inputs, &mut state);
    state.detach_in_place();
    let observed_inputs =
        LanguageTrainModel::<TestBackend>::slice_tokens(batch.inputs, batch_size, 2, 4);
    let (corrected_state, report) =
        model.correct_state_from_observed_prefix(state, observed_inputs, None, None);

    assert!(report.chunks_corrected > 0, "report={report:?}");
    assert!(
        report
            .energy_before
            .zip(report.energy_after)
            .is_some_and(|(before, after)| {
                before.is_finite() && after.is_finite() && after <= before + 1.0e-4
            }),
        "observed-prefix inference should descend its causal energy: {report:?}"
    );
    assert!(
        LanguageTrainModel::<TestBackend>::predictive_coding_state_has_latents(
            &corrected_state,
            PredictiveCodingStateScope::Core,
        )
    );
}

#[test]
fn observed_prefix_empty_entry_replays_instead_of_resetting_state() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 17);
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_predictive_coding(PredictiveCodingConfig {
        enabled: true,
        observation_contract: PredictiveCodingObservationContract::ObservedPrefix,
        ..Default::default()
    });
    let observed_inputs = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![0, 1, 2, 3], [2, 2]),
        &device,
    );

    let (replayed, report) = model.correct_state_from_observed_prefix(
        model.model.init_state_ephemeral(),
        observed_inputs,
        None,
        None,
    );

    assert_eq!(report.skipped_empty_state, 1);
    assert_eq!(report.chunks_corrected, 0);
    assert_eq!(replayed.position, 2);
    assert!(
        LanguageTrainModel::<TestBackend>::predictive_coding_state_has_latents(
            &replayed,
            PredictiveCodingStateScope::Core,
        )
    );
}

#[test]
fn predictive_coding_amortization_constraint_detects_state_drift() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 19);
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_predictive_coding(PredictiveCodingConfig {
        enabled: true,
        amortization_tolerance: 0.0,
        ..Default::default()
    });
    let mut student = model.model.init_state_ephemeral();
    let inputs = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![0, 1, 2, 3], [2, 2]),
        &device,
    );
    model.model.forward_hidden_with_state(inputs, &mut student);
    let teacher = student.detached_clone();
    let (same, components) = model.predictive_coding_amortization_constraint(&student, &teacher);
    let same = scalar_tensor_to_f64(same.expect("same-state constraint").detach().inner());
    assert!(components > 0);
    assert!(same <= 1.0e-8, "same-state constraint={same}");

    let mut drifted = teacher;
    for layer in &mut drifted.layers {
        layer.rho = layer.rho.take().map(|rho| rho.add_scalar(1.0).detach());
        layer.y_neuron_state = layer
            .y_neuron_state
            .take()
            .map(|state| state.add_scalar(1.0).detach());
    }
    let (drift, drift_components) =
        model.predictive_coding_amortization_constraint(&student, &drifted);
    let drift = scalar_tensor_to_f64(drift.expect("drift constraint").detach().inner());
    assert_eq!(drift_components, components);
    assert!(drift > 1.0e-4, "drift constraint={drift}");
}

#[test]
fn predictive_coding_amortization_has_finite_zero_error_gradient() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let student = Tensor::<TestBackend, 3>::zeros([2, 2, 4], &device).require_grad();
    let teacher = Tensor::<TestBackend, 3>::zeros([2, 2, 4], &device);
    let mut total = None;
    let mut components = 0;
    let mut sample_indices = PredictiveCodingSampleIndexCache::new();
    accumulate_predictive_coding_amortization_constraint(
        &mut total,
        &mut components,
        &Some(student.clone()),
        &Some(teacher),
        PredictiveCodingAmortizationConstraint {
            sample_axis: 2,
            max_slots: 4,
            sample_offset: 0,
            tolerance: 0.0,
            eps: 1.0e-8,
        },
        &mut sample_indices,
    );

    let grads = total.expect("constraint").backward();
    let grad = student.grad(&grads).expect("student state gradient");
    let values = grad
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("gradient values");

    assert_eq!(components, 1);
    assert!(values.iter().all(|value| value.is_finite()));
    assert!(values.iter().all(|value| value.abs() <= 1.0e-8));
}

#[test]
fn observed_prefix_train_step_amortizes_without_online_state_replacement() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 23);
    crate::train::profile::reset_predictive_coding();
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_tbptt_chunk_size(Some(2))
    .with_predictive_coding(PredictiveCodingConfig {
        enabled: true,
        observation_contract: PredictiveCodingObservationContract::ObservedPrefix,
        parameter_update: PredictiveCodingParameterUpdate::Optimizer,
        steps: 1,
        step_size: 0.01,
        ..Default::default()
    });

    let loss = scalar_loss(TrainStep::step(&model, batch(&device)));
    let profile = crate::train::profile::take_predictive_coding();

    assert!(loss.is_finite());
    assert!(profile.chunks_corrected > 0, "profile={profile:?}");
    assert!(
        profile.amortization_components > 0,
        "causal PC must constrain the ordinary deployment transition: {profile:?}"
    );
}

fn require_grad_param_count<B: BackendTrait>(model: &DragonModel<B>) -> usize {
    #[derive(Default)]
    struct RequireGradCounter {
        count: usize,
    }

    impl<B: BackendTrait> burn::module::ModuleVisitor<B> for RequireGradCounter {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            self.count += usize::from(param.val().is_require_grad());
        }
    }

    let mut counter = RequireGradCounter::default();
    model.visit(&mut counter);
    counter.count
}

#[test]
fn teacher_runtime_detaches_trainable_parameters() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let model = DragonModel::<TestBackend>::new(tiny_model_config(), &device);
    assert!(
        require_grad_param_count(&model) > 0,
        "training model should own trainable autodiff parameters"
    );

    let teacher = TeacherModelRuntime::new(model);
    assert_eq!(
        require_grad_param_count(&teacher.model),
        0,
        "teacher snapshots must not build parameter-gradient graphs"
    );
}

#[test]
fn predictive_coding_all_scope_covers_every_slow_state_family() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let mut state = ModelState::<TestBackend>::new(1);
    let layer = &mut state.layers[0];
    layer.slow_rho = Some(Tensor::zeros([1, 1, 2, 2], &device));
    layer.slow_rho_norm = Some(Tensor::zeros([1, 1, 2], &device));
    layer.slow_sequence_aux = Some(Tensor::zeros([1, 1, 2, 2], &device));
    layer.slow_mamba_angle_state = Some(Tensor::zeros([1, 1, 2], &device));
    layer.slow_mamba_k_state = Some(Tensor::zeros([1, 1, 2], &device));
    layer.slow_mamba_v_state = Some(Tensor::zeros([1, 1, 2], &device));
    layer.hierarchical_slow_hidden = Some(Tensor::zeros([1, 1, 2, 2], &device));

    assert!(
        !LanguageTrainModel::<TestBackend>::predictive_coding_state_has_latents(
            &state,
            PredictiveCodingStateScope::Core,
        )
    );
    assert!(
        LanguageTrainModel::<TestBackend>::predictive_coding_state_has_latents(
            &state,
            PredictiveCodingStateScope::All,
        )
    );

    let snapshot = predictive_coding_state_snapshot(&state, PredictiveCodingStateScope::All);
    let names = snapshot
        .rank3
        .iter()
        .map(|(name, _)| *name)
        .chain(snapshot.rank4.iter().map(|(name, _)| *name))
        .collect::<HashSet<_>>();
    for required in [
        "slow_rho",
        "slow_sequence_aux",
        "slow_mamba_angle_state",
        "slow_mamba_k_state",
        "slow_mamba_v_state",
        "hierarchical_slow_hidden",
    ] {
        assert!(names.contains(required), "missing state field {required}");
    }

    assert!(
        LanguageTrainModel::<TestBackend>::attach_predictive_coding_state_latents(
            &mut state,
            PredictiveCodingStateScope::All,
        )
    );
    let layer = &state.layers[0];
    assert!(layer.slow_rho.as_ref().is_some_and(Tensor::is_require_grad));
    assert!(
        layer
            .slow_mamba_k_state
            .as_ref()
            .is_some_and(Tensor::is_require_grad)
    );
    assert!(
        layer
            .hierarchical_slow_hidden
            .as_ref()
            .is_some_and(Tensor::is_require_grad)
    );
    assert!(layer.slow_rho_norm.is_none());
}

#[test]
fn predictive_coding_rotating_sampler_covers_all_slots() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let tensor = Tensor::<TestBackend, 3>::from_data(
        TensorData::new((0..10).map(|value| value as f32).collect(), [1, 1, 10]),
        &device,
    );
    let mut cache = PredictiveCodingSampleIndexCache::new();
    let mut covered = HashSet::new();

    for offset in 0..10 {
        let (student, teacher) = rotating_sample_state_axis_pair(
            tensor.clone(),
            tensor.clone(),
            2,
            3,
            offset,
            &mut cache,
        );
        let student = student
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("sampled student");
        let teacher = teacher
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("sampled teacher");
        assert_eq!(student, teacher);
        covered.extend(student.into_iter().map(|value| value as usize));
    }

    assert_eq!(covered, (0..10).collect::<HashSet<_>>());
}

#[test]
fn neuron_scale_3d_gradient_scaling_preserves_headed_tail_semantics() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let tensor = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 1, 4]),
        &device,
    );
    let scaled = scale_3d_latent_tail(tensor, 2, 4, 0.5, 2.0)
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("scaled 3d gradient");
    assert_eq!(scaled, vec![0.5, 1.0, 6.0, 8.0, 2.5, 3.0, 14.0, 16.0]);
}

#[test]
fn neuron_scale_2d_gradient_scaling_preserves_headed_tail_semantics() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let tensor = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(
            (1..=16).map(|value| value as f32).collect::<Vec<_>>(),
            [8, 2],
        ),
        &device,
    );
    let scaled = scale_2d_headed_latent_rows(tensor, 2, 4, 0.5, 2.0)
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("scaled 2d gradient");
    assert_eq!(
        scaled,
        vec![
            0.5, 1.0, 1.5, 2.0, 10.0, 12.0, 14.0, 16.0, 4.5, 5.0, 5.5, 6.0, 26.0, 28.0, 30.0, 32.0,
        ]
    );
}

#[test]
fn output_degeneracy_step_reports_overconfident_argmax() {
    let step = output_degeneracy_step_from_row(&[12.0, -8.0, -9.0, -10.0]).expect("finite step");
    assert_eq!(step.argmax, 0);
    assert!(
        step.entropy_bits < 0.001,
        "unexpected entropy: {}",
        step.entropy_bits
    );
    assert!(
        step.max_probability > 0.999,
        "unexpected max probability: {}",
        step.max_probability
    );
}

#[test]
fn output_degeneracy_accumulator_tracks_repetition_and_eos() {
    let mut accumulator = OutputDegeneracyAccumulator::new(Some(2));
    for argmax in [2, 2, 3, 3] {
        accumulator.record(OutputDegeneracyStep {
            argmax,
            entropy_bits: 0.25,
            max_probability: 0.9,
        });
        accumulator.record_generated_token(argmax as i64);
    }
    let stats = accumulator.finish();
    assert_eq!(stats.token_count, 4);
    assert_eq!(stats.argmax_unique_fraction, 0.5);
    assert_eq!(stats.eos_fraction, 0.5);
    assert!((stats.repetition_fraction - (2.0 / 3.0)).abs() < 1e-12);
    assert_eq!(stats.distinct_1_fraction, 0.5);
    assert_eq!(stats.distinct_2_fraction, 1.0);
    assert_eq!(stats.period_2_fraction, 0.0);
}

#[test]
fn output_degeneracy_accumulator_ignores_eos_padding_after_payload() {
    let eos_id = 99usize;
    let mut accumulator = OutputDegeneracyAccumulator::new(Some(eos_id as i64));
    for argmax in (0usize..24).chain(std::iter::repeat_n(eos_id, 40)) {
        accumulator.record(OutputDegeneracyStep {
            argmax,
            entropy_bits: if argmax == eos_id { 0.01 } else { 2.0 },
            max_probability: if argmax == eos_id { 0.99 } else { 0.3 },
        });
        accumulator.record_generated_token(argmax as i64);
    }
    let stats = accumulator.finish();
    assert_eq!(stats.token_count, 64);
    assert_eq!(
        stats.eos_fraction, 0.0,
        "EOS padding after a payload should not trip EOS collapse"
    );
    assert!(
        stats.repetition_fraction < 0.01,
        "payload repetition should be scored before EOS padding: {}",
        stats.repetition_fraction
    );
    assert!(
        stats.entropy_bits > 1.9,
        "payload entropy should be scored before EOS padding: {}",
        stats.entropy_bits
    );
    assert_eq!(stats.distinct_1_fraction, 1.0);
}

#[test]
fn output_degeneracy_accumulator_tracks_long_period_cycles() {
    let mut accumulator = OutputDegeneracyAccumulator::new(None);
    for index in 0..128 {
        let argmax = index % 37;
        accumulator.record(OutputDegeneracyStep {
            argmax,
            entropy_bits: 4.0,
            max_probability: 0.25,
        });
        accumulator.record_generated_token(argmax as i64);
    }
    let stats = accumulator.finish();
    assert!(
        stats.max_period_2_to_16_fraction < 0.05,
        "period-2..16 should not catch a period-37 loop: {}",
        stats.max_period_2_to_16_fraction
    );
    assert_eq!(stats.dominant_period_2_to_64, 37);
    assert!(
        stats.max_period_2_to_64_fraction > 0.95,
        "expected high extended long-cycle fraction, got {}",
        stats.max_period_2_to_64_fraction
    );
    assert!(
        stats.period_2_fraction < 0.01 && stats.period_3_fraction < 0.01,
        "period-2/3 should not catch a period-37 loop"
    );
}

#[test]
fn output_degeneracy_ignores_single_comparison_long_period_aliases() {
    let mut tokens: Vec<i64> = (0..32).collect();
    tokens[31] = tokens[0];

    assert_eq!(period_fraction(&tokens, 31), 0.0);
    let (dominant_period, max_fraction) = dominant_period_fraction(&tokens, 2..=64);

    assert_ne!(
        dominant_period, 31,
        "period-31 only has one comparison over a 32-token probe"
    );
    assert!(
        max_fraction < 1.0,
        "single-comparison alias should not produce a perfect period score"
    );
}

#[test]
fn validation_degeneracy_probe_rolls_out_generated_tokens() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ));
    let (_loss, stats) = model.validation_loss_and_output_degeneracy(batch(&device), 3, None);
    let stats = stats.expect("free-running degeneracy stats");
    assert_eq!(stats.token_count, 6);
    assert!(stats.entropy_bits.is_finite());
    assert!(stats.mean_max_probability.is_finite());
    assert!((0.0..=1.0).contains(&stats.argmax_unique_fraction));
    assert!((0.0..=1.0).contains(&stats.repetition_fraction));
    assert_eq!(stats.generated_tokens.len(), 6);
    assert!((0.0..=1.0).contains(&stats.distinct_1_fraction));
    assert!((0.0..=1.0).contains(&stats.distinct_2_fraction));
    assert!((0.0..=1.0).contains(&stats.period_2_fraction));
    assert!((0.0..=1.0).contains(&stats.period_3_fraction));
}

#[test]
fn validation_degeneracy_prompts_cover_header_and_interior_windows() {
    let starts = (0..4)
        .map(|index| validation_degeneracy_prompt_start(index, 4, 224))
        .collect::<Vec<_>>();
    assert_eq!(starts[0], 0);
    assert!(starts[1] >= 64, "{starts:?}");
    assert!(starts[3] <= 224, "{starts:?}");
    assert!(starts.windows(2).all(|window| window[0] <= window[1]));
}

#[test]
fn rollout_unlikelihood_prompt_rotates_away_from_header() {
    let first = rollout_prompt_start(0, 1, 256, 32);
    let later = rollout_prompt_start(1, 1, 256, 32);
    assert_eq!(first, 32);
    assert_ne!(first, later);
    assert!(later > 0);
}

#[test]
fn selected_token_logits_gathers_raw_logits_not_log_probs() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let logits = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![1.0, 2.0, 9.0, -1.0, 4.0, 3.0, 7.0, 8.0], [1, 2, 4]),
        &device,
    );
    let targets =
        Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![2, 0], [1, 2]), &device);
    let selected = selected_token_logits(logits, targets)
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("selected logits");
    assert_eq!(selected, vec![9.0, 4.0]);
}

#[test]
fn causal_input_corruption_replaces_inputs_with_fixed_token() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_input_corruption(CausalInputCorruptionConfig {
        enabled: true,
        probability: 1.0,
        replacement_token_id: Some(3),
        ..Default::default()
    });
    let inputs = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![0, 1, 2, 4, 5, 6], [2, 3]),
        &device,
    );
    let corrupted = model.corrupt_causal_inputs(inputs);
    let values = corrupted
        .to_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .expect("corrupted inputs");
    assert_eq!(values, vec![3; 6]);
}

#[test]
fn causal_input_corruption_respects_warmup() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_input_corruption(CausalInputCorruptionConfig {
        enabled: true,
        probability: 1.0,
        warmup_steps: 10,
        replacement_token_id: Some(3),
        ..Default::default()
    });
    let inputs = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![0, 1, 2, 4, 5, 6], [2, 3]),
        &device,
    );
    let corrupted = model.corrupt_causal_inputs(inputs);
    let values = corrupted
        .to_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .expect("corrupted inputs");
    assert_eq!(values, vec![0, 1, 2, 4, 5, 6]);
}

#[test]
fn next_token_loss_honors_optional_target_mask() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ));
    let logits = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![8.0, 0.0, 0.0, 8.0], [1, 2, 2]),
        &device,
    );
    let clean_inputs =
        Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![0, 1], [1, 2]), &device);
    let targets =
        Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![0, 0], [1, 2]), &device);
    let first_only_mask =
        Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![1, 0], [1, 2]), &device);

    let unmasked = tensor_scalar(model.next_token_loss_from_logits(
        logits.clone(),
        targets.clone(),
        clean_inputs.clone(),
        None,
        None,
    ));
    let masked = tensor_scalar(model.next_token_loss_from_logits(
        logits,
        targets,
        clean_inputs,
        Some(first_only_mask),
        None,
    ));

    assert!(unmasked > masked + 3.0);
    assert!(
        masked < 1.0e-3,
        "masked loss should keep only the confident first token"
    );
}

#[test]
fn ruliad_answer_ranking_penalizes_corrupt_answer_logits() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_ruliad_supervision(RuliadSupervisionConfig {
        mode: RuliadSupervisionMode::AnswerCompletion,
        answer_ranking: RuliadAnswerRankingConfig {
            enabled: true,
            weight: 1.0,
            margin: 0.5,
            corrupt_offset: 1,
        },
        ..Default::default()
    });
    let targets =
        Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![1, 2], [1, 2]), &device);
    let mask =
        Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![1, 1], [1, 2]), &device);
    let preferred = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(
            vec![
                0.0, 5.0, -2.0, 0.0, //
                0.0, 0.0, 5.0, -2.0,
            ],
            [1, 2, 4],
        ),
        &device,
    );
    let inverted = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(
            vec![
                0.0, -2.0, 5.0, 0.0, //
                0.0, 0.0, -2.0, 5.0,
            ],
            [1, 2, 4],
        ),
        &device,
    );

    let preferred_loss = tensor_scalar(
        model
            .ruliad_answer_ranking_loss_from_logits(preferred, targets.clone(), Some(mask.clone()))
            .expect("preferred ranking loss"),
    );
    let inverted_loss = tensor_scalar(
        model
            .ruliad_answer_ranking_loss_from_logits(inverted, targets, Some(mask))
            .expect("inverted ranking loss"),
    );

    assert!(
        inverted_loss > preferred_loss + 5.0,
        "ranking loss should reward oracle answer logits over corrupt answer logits: preferred={preferred_loss} inverted={inverted_loss}"
    );
}

#[test]
fn answer_prefix_input_mask_shifts_target_answer_mask_right() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let mask = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![0, 1, 1, 0, 1], [1, 5]),
        &device,
    );
    let shifted = answer_prefix_input_mask(mask)
        .to_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .expect("shifted mask");
    assert_eq!(shifted, vec![0, 0, 1, 1, 0]);
}

#[test]
fn ruliad_answer_denoising_corrupts_only_answer_prefix_inputs() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ));
    let inputs = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![10, 11, 12, 13, 14], [1, 5]),
        &device,
    );
    let target_mask = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![0, 1, 1, 0, 1], [1, 5]),
        &device,
    );
    let prefix_mask = answer_prefix_input_mask(target_mask);
    let corrupted = model
        .corrupt_ruliad_answer_prefix_inputs(
            inputs,
            prefix_mask,
            RuliadAnswerDenoisingConfig {
                enabled: true,
                weight: 1.0,
                probability: 1.0,
                corrupt_offset: 1,
                ..Default::default()
            },
        )
        .to_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .expect("corrupted inputs");
    assert_eq!(corrupted, vec![10, 11, 13, 14, 14]);
}

#[test]
fn ruliad_answer_denoising_loss_is_finite_for_masked_answer_batch() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_ruliad_supervision(RuliadSupervisionConfig {
        mode: RuliadSupervisionMode::AnswerCompletion,
        answer_denoising: RuliadAnswerDenoisingConfig {
            enabled: true,
            weight: 0.5,
            probability: 1.0,
            corrupt_offset: 1,
            ..Default::default()
        },
        ..Default::default()
    });
    let inputs = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![0, 1, 2, 3, 4, 5], [1, 6]),
        &device,
    );
    let targets = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![1, 2, 3, 4, 5, 6], [1, 6]),
        &device,
    );
    let mask = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![0, 1, 1, 1, 0, 0], [1, 6]),
        &device,
    );
    let loss = tensor_scalar(
        model
            .ruliad_answer_denoising_loss(inputs, targets, Some(mask))
            .expect("denoising loss"),
    );
    assert!(loss.is_finite(), "denoising loss should be finite: {loss}");
    assert!(loss > 0.0, "denoising loss should be non-zero: {loss}");
}

#[test]
fn ruliad_structured_answer_recovery_loss_trains_oracle_after_wrong_prefix() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 29);
    let mut config = tiny_model_config();
    config.vocab_size = 257;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_ruliad_supervision(RuliadSupervisionConfig {
            mode: RuliadSupervisionMode::AnswerCompletion,
            answer_denoising: RuliadAnswerDenoisingConfig {
                enabled: true,
                weight: 0.0,
                structured_recovery_weight: 0.25,
                structured_recovery_every_steps: 2,
                structured_recovery_start_after_steps: 4,
                structured_recovery_max_completion_tokens: 24,
                structured_recovery_negative_count: 1,
                structured_recovery_template_negative_count: 1,
                ..Default::default()
            },
            ..Default::default()
        });
    let item = burn_dragon_universality::RuliadEvalItem {
        oracle_hash: "h0".to_string(),
        sample_index: 43,
        split: burn_dragon_universality::SampleSplit::Train,
        family: "trajectory_category".to_string(),
        task_kind: "eca_summary".to_string(),
        math_domains: vec!["category".to_string(), "finite_state".to_string()],
        reasoning_modes: vec!["symbolic_execution".to_string()],
        prompt: "?:eca\n!:".to_string(),
        expected_answer: "xlen=44;xalpha=01;xcounts=20,24;xedge=01".to_string(),
        difficulty_level: Some(0),
        spec: None,
    };
    let policy_batch = crate::dataset::RuliadPolicyBatch {
        samples: vec![crate::dataset::RuliadPolicySample {
            item,
            prompt_tokens: vec![1, 2, 3],
        }],
        tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
            vocab_size: 257,
            eos_id: None,
        },
        stop_token_id: None,
        sampling_metadata: None,
    };

    model.gradient_scale_step.store(3, Ordering::Relaxed);
    assert!(
        model
            .ruliad_structured_answer_recovery_loss(&policy_batch, &device, 64)
            .is_none(),
        "structured recovery should respect start_after_steps"
    );
    model.gradient_scale_step.store(5, Ordering::Relaxed);
    assert!(
        model
            .ruliad_structured_answer_recovery_loss(&policy_batch, &device, 64)
            .is_none(),
        "structured recovery should respect every_steps cadence"
    );
    model.gradient_scale_step.store(6, Ordering::Relaxed);
    let loss = model
        .ruliad_structured_answer_recovery_loss(&policy_batch, &device, 64)
        .expect("structured answer recovery loss");
    let loss = tensor_scalar(loss);
    assert!(loss.is_finite(), "recovery loss should be finite: {loss}");
    assert!(loss > 0.0, "recovery loss should be non-zero: {loss}");
}

#[test]
fn ruliad_structured_answer_recovery_loss_writes_activity_telemetry() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 31);
    let dir = tempfile::tempdir().expect("tempdir");
    let telemetry_path = dir
        .path()
        .join("events")
        .join("ruliad_structured_recovery.jsonl");
    let mut config = tiny_model_config();
    config.vocab_size = 257;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_ruliad_supervision(RuliadSupervisionConfig {
            mode: RuliadSupervisionMode::AnswerCompletion,
            answer_denoising: RuliadAnswerDenoisingConfig {
                enabled: true,
                weight: 0.0,
                structured_recovery_weight: 0.25,
                structured_recovery_every_steps: 1,
                structured_recovery_start_after_steps: 0,
                structured_recovery_max_completion_tokens: 24,
                structured_recovery_negative_count: 2,
                structured_recovery_template_negative_count: 2,
                structured_recovery_schema_negative_count: 2,
                ..Default::default()
            },
            ..Default::default()
        })
        .with_ruliad_structured_recovery_telemetry_path(Some(telemetry_path.clone()));
    let item = burn_dragon_universality::RuliadEvalItem {
        oracle_hash: "h0".to_string(),
        sample_index: 44,
        split: burn_dragon_universality::SampleSplit::Train,
        family: "proof_tree".to_string(),
        task_kind: "prove_theorem".to_string(),
        math_domains: vec!["category".to_string(), "formal_proof".to_string()],
        reasoning_modes: vec!["equational".to_string()],
        prompt: "?:ss\n!:".to_string(),
        expected_answer: "ok=1;l=17;r=17".to_string(),
        difficulty_level: Some(0),
        spec: None,
    };
    let policy_batch = crate::dataset::RuliadPolicyBatch {
        samples: vec![crate::dataset::RuliadPolicySample {
            item,
            prompt_tokens: vec![1, 2, 3],
        }],
        tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
            vocab_size: 257,
            eos_id: None,
        },
        stop_token_id: None,
        sampling_metadata: None,
    };

    let loss = model
        .ruliad_structured_answer_recovery_loss(&policy_batch, &device, 64)
        .expect("structured answer recovery loss");
    let loss = tensor_scalar(loss);
    assert!(loss.is_finite(), "recovery loss should be finite: {loss}");

    let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
    let event: serde_json::Value =
        serde_json::from_str(content.lines().next().expect("telemetry line"))
            .expect("telemetry json");
    assert_eq!(event["sample_groups"].as_u64(), Some(1));
    assert_eq!(event["field_negative_recovery_rows"].as_u64(), Some(2));
    assert_eq!(event["template_negative_recovery_rows"].as_u64(), Some(2));
    let schema_rows = event["schema_negative_recovery_rows"]
        .as_u64()
        .expect("schema recovery rows");
    assert!(
        schema_rows > 0,
        "schema-collapse recovery rows should be present: {event}"
    );
    assert_eq!(
        event["recovery_rows"].as_u64(),
        Some(4 + schema_rows),
        "recovery rows should include field, template, and schema negatives"
    );
}

#[test]
fn ruliad_answer_contract_loss_trains_full_oracle_contract_and_respects_cadence() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 41);
    let mut config = tiny_model_config();
    config.vocab_size = 257;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_ruliad_supervision(RuliadSupervisionConfig {
            mode: RuliadSupervisionMode::AnswerCompletion,
            answer_contract: crate::config::train::RuliadAnswerContractConfig {
                enabled: true,
                weight: 0.25,
                premature_close_unlikelihood_weight: 0.5,
                every_steps: 2,
                start_after_steps: 4,
                max_completion_tokens: 24,
                max_rows_per_step: 1,
                prompt_schema_max_rows_per_step: 0,
                schema_token_weight: 2.0,
                schema_start_token_weight: 8.0,
                value_token_weight: 1.0,
                other_token_weight: 1.0,
                prompt_schema_value_weight: 0.0,
            },
            ..Default::default()
        });
    let item = burn_dragon_universality::RuliadEvalItem {
        oracle_hash: "h0".to_string(),
        sample_index: 46,
        split: burn_dragon_universality::SampleSplit::Train,
        family: "proof_tree".to_string(),
        task_kind: "prove_theorem".to_string(),
        math_domains: vec!["category".to_string(), "formal_proof".to_string()],
        reasoning_modes: vec!["equational".to_string()],
        prompt: "?:ss\n!:".to_string(),
        expected_answer: "ok=1;l=17;r=17".to_string(),
        difficulty_level: Some(0),
        spec: None,
    };
    let policy_batch = crate::dataset::RuliadPolicyBatch {
        samples: vec![crate::dataset::RuliadPolicySample {
            item,
            prompt_tokens: vec![1, 2, 3],
        }],
        tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
            vocab_size: 257,
            eos_id: None,
        },
        stop_token_id: None,
        sampling_metadata: None,
    };

    model.gradient_scale_step.store(3, Ordering::Relaxed);
    assert!(
        model
            .ruliad_answer_contract_loss(&policy_batch, &device, 64)
            .is_none(),
        "answer contract loss should respect start_after_steps"
    );
    model.gradient_scale_step.store(5, Ordering::Relaxed);
    assert!(
        model
            .ruliad_answer_contract_loss(&policy_batch, &device, 64)
            .is_none(),
        "answer contract loss should respect every_steps cadence"
    );
    model.gradient_scale_step.store(6, Ordering::Relaxed);
    let loss = model
        .ruliad_answer_contract_loss(&policy_batch, &device, 64)
        .expect("answer contract loss");
    let loss = tensor_scalar(loss);
    assert!(loss.is_finite(), "contract loss should be finite: {loss}");
    assert!(loss > 0.0, "contract loss should be non-zero: {loss}");
}

#[test]
fn ruliad_answer_contract_loss_writes_activity_telemetry_and_caps_rows() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 43);
    let dir = tempfile::tempdir().expect("tempdir");
    let telemetry_path = dir
        .path()
        .join("events")
        .join("ruliad_answer_contract.jsonl");
    let mut config = tiny_model_config();
    config.vocab_size = 272;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_ruliad_supervision(RuliadSupervisionConfig {
            mode: RuliadSupervisionMode::AnswerCompletion,
            answer_contract: crate::config::train::RuliadAnswerContractConfig {
                enabled: true,
                weight: 0.25,
                premature_close_unlikelihood_weight: 0.5,
                every_steps: 1,
                start_after_steps: 0,
                max_completion_tokens: 24,
                max_rows_per_step: 1,
                prompt_schema_max_rows_per_step: 1,
                schema_token_weight: 2.0,
                schema_start_token_weight: 8.0,
                value_token_weight: 1.0,
                other_token_weight: 1.0,
                prompt_schema_value_weight: 2.0,
            },
            ..Default::default()
        })
        .with_ruliad_answer_contract_telemetry_path(Some(telemetry_path.clone()));
    let make_item = |sample_index, answer: &str| burn_dragon_universality::RuliadEvalItem {
        oracle_hash: format!("h{sample_index}"),
        sample_index,
        split: burn_dragon_universality::SampleSplit::Train,
        family: "proof_tree".to_string(),
        task_kind: "prove_theorem".to_string(),
        math_domains: vec!["category".to_string(), "formal_proof".to_string()],
        reasoning_modes: vec!["equational".to_string()],
        prompt: "?:ss\n!:".to_string(),
        expected_answer: answer.to_string(),
        difficulty_level: Some(0),
        spec: None,
    };
    let policy_batch = crate::dataset::RuliadPolicyBatch {
        samples: vec![
            crate::dataset::RuliadPolicySample {
                item: make_item(47, "ok=1;l=17;r=17"),
                prompt_tokens: vec![1, 2, 3],
            },
            crate::dataset::RuliadPolicySample {
                item: make_item(48, "nflen=3;nfalpha=ABC;nfcounts=1,1,1;nfedge=AB"),
                prompt_tokens: vec![1, 2, 3],
            },
        ],
        tokenization: burn_dragon_universality::RuliadTokenizationConfig::StructuredSymbolic {
            vocab_size: 272,
            eos_id: Some(271),
        },
        stop_token_id: Some(265),
        sampling_metadata: None,
    };

    let loss = model
        .ruliad_answer_contract_loss(&policy_batch, &device, 64)
        .expect("answer contract loss");
    let loss = tensor_scalar(loss);
    assert!(loss.is_finite(), "contract loss should be finite: {loss}");

    let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
    let event: serde_json::Value =
        serde_json::from_str(content.lines().next().expect("telemetry line"))
            .expect("contract telemetry json");
    assert_eq!(event["policy_batch_present"].as_bool(), Some(true));
    assert_eq!(event["oracle_rows"].as_u64(), Some(1));
    assert!(
        event["sample_groups"].as_u64().unwrap_or_default() >= 1,
        "contract objective should report active sample groups: {event}"
    );
    assert!(
        event["prompt_schema_sample_groups"]
            .as_u64()
            .unwrap_or_default()
            >= 1,
        "contract objective should report active prompt-schema sample groups: {event}"
    );
    assert_eq!(event["prompt_schema_rows"].as_u64(), Some(1));
    assert_eq!(event["prompt_schema_max_rows_per_step"].as_u64(), Some(1));
    assert!(
        event["schema_tokens"].as_u64().unwrap_or_default() > 0,
        "contract objective should supervise schema tokens: {event}"
    );
    assert!(
        event["schema_start_tokens"].as_u64().unwrap_or_default() > 0,
        "contract objective should identify schema-start tokens: {event}"
    );
    assert!(
        event["value_tokens"].as_u64().unwrap_or_default() > 0,
        "contract objective should supervise value tokens: {event}"
    );
    assert!(
        event["prompt_schema_value_tokens"]
            .as_u64()
            .unwrap_or_default()
            > 0,
        "contract objective should supervise schema-forced value tokens: {event}"
    );
    assert!(
        event["premature_close_tokens"].as_u64().unwrap_or_default() > 0,
        "contract objective should penalize premature close markers: {event}"
    );
}

#[test]
fn ruliad_verifier_policy_loss_builds_from_policy_metadata() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_ruliad_supervision(RuliadSupervisionConfig {
        verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
            enabled: true,
            weight: 0.1,
            group_size: 2,
            max_completion_tokens: 2,
            every_steps: 1,
            top_k: 1,
            ..Default::default()
        },
        ..Default::default()
    });
    let item = burn_dragon_universality::RuliadEvalItem {
        oracle_hash: "h0".to_string(),
        sample_index: 0,
        split: burn_dragon_universality::SampleSplit::Train,
        family: "law".to_string(),
        task_kind: "category_law".to_string(),
        math_domains: vec!["category".to_string()],
        reasoning_modes: vec!["equational".to_string()],
        prompt: "?:q\n!:".to_string(),
        expected_answer: "ok=1".to_string(),
        difficulty_level: Some(0),
        spec: None,
    };
    let policy_batch = crate::dataset::RuliadPolicyBatch {
        samples: vec![crate::dataset::RuliadPolicySample {
            item,
            prompt_tokens: vec![1, 2, 3],
        }],
        tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
            vocab_size: 257,
            eos_id: None,
        },
        stop_token_id: None,
        sampling_metadata: None,
    };
    let loss = model
        .ruliad_verifier_policy_loss(&policy_batch, &device, 8)
        .expect("verifier policy loss");
    let loss = tensor_scalar(loss);
    assert!(
        loss.is_finite(),
        "verifier policy loss should be finite: {loss}"
    );
}

#[test]
fn ruliad_verifier_policy_loss_respects_start_after_steps() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_ruliad_supervision(RuliadSupervisionConfig {
        verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
            enabled: true,
            weight: 0.1,
            every_steps: 1,
            start_after_steps: 4,
            ..Default::default()
        },
        ..Default::default()
    });
    model.gradient_scale_step.store(3, Ordering::Relaxed);
    assert_eq!(model.ruliad_verifier_reward_weight(), 0.0);
    model.gradient_scale_step.store(4, Ordering::Relaxed);
    assert_eq!(model.ruliad_verifier_reward_weight(), 0.1);
}

#[test]
fn ruliad_policy_telemetry_marks_saturated_updates_as_skipped() {
    let mut telemetry = RuliadPolicyRewardTelemetryAccumulator::new(
        crate::config::train::RuliadVerifierRewardMode::VpoIndependent,
        64,
    );
    telemetry.record_rewards_and_advantages(&[0.0, 1.0, -1.0], &[0.0, 1.0, -1.0], 0.2);
    assert!(
        telemetry.advantage_clip_fraction() > 0.5,
        "test should exercise a saturated policy update"
    );
    telemetry.mark_skipped("advantage_clip_fraction>0.500000");
    let telemetry = telemetry.finish().expect("telemetry");
    assert!(!telemetry.policy_update_applied);
    assert_eq!(
        telemetry.policy_skip_reason.as_deref(),
        Some("advantage_clip_fraction>0.500000")
    );
}

#[test]
fn ruliad_policy_telemetry_reports_gated_groups_without_rows() {
    let mut telemetry = RuliadPolicyRewardTelemetryAccumulator::new(
        crate::config::train::RuliadVerifierRewardMode::VpoIndependent,
        64,
    );
    telemetry.record_gated_group(4);
    telemetry.mark_skipped("positive_advantage_gate");
    let telemetry = telemetry.finish().expect("telemetry");
    assert_eq!(telemetry.completion_rows, 0);
    assert_eq!(telemetry.gated_sample_groups, 1);
    assert_eq!(telemetry.gated_completion_rows, 4);
    assert!(!telemetry.policy_update_applied);
    assert_eq!(
        telemetry.policy_skip_reason.as_deref(),
        Some("positive_advantage_gate")
    );
}

fn strict_policy_advantage_gate_config() -> crate::config::train::RuliadVerifierRewardConfig {
    crate::config::train::RuliadVerifierRewardConfig {
        positive_advantage_requires_correctness: true,
        positive_advantage_min_partial_progress_ppm: 500_000,
        positive_advantage_min_completion_quality_ppm: 750_000,
        ..Default::default()
    }
}

fn policy_score(
    status: burn_dragon_universality::ruliad::RuliadAnswerStatus,
    partial_progress_ppm: usize,
    completion_quality_ppm: usize,
) -> burn_dragon_universality::ruliad::RuliadReasoningScore {
    let expected_field_count = 4;
    let correct_field_count = partial_progress_ppm
        .saturating_mul(expected_field_count)
        .div_ceil(1_000_000)
        .min(expected_field_count);
    burn_dragon_universality::ruliad::RuliadReasoningScore {
        version: burn_dragon_universality::ruliad::RULIAD_REASONING_SCORE_VERSION,
        status,
        correct_field_count,
        expected_field_count,
        observed_field_count: correct_field_count,
        partial_progress_ppm,
        certificate_valid_prefix_steps: 0,
        certificate_expected_steps: 0,
        certificate_prefix_ppm: 0,
        generated_token_count: if partial_progress_ppm > 0 { 8 } else { 1 },
        hash_canary: false,
        answer_terminated: status
            != burn_dragon_universality::ruliad::RuliadAnswerStatus::Malformed,
        completion_quality_ppm,
    }
}

#[test]
fn ruliad_rollout_recovery_signal_accepts_wrong_and_malformed_corruptions() {
    let partial = policy_score(
        burn_dragon_universality::ruliad::RuliadAnswerStatus::Partial,
        500_000,
        1_000_000,
    );
    assert!(
        LanguageTrainModel::<TestBackend>::ruliad_score_has_rollout_recovery_signal(
            &partial, 500_000, 750_000,
        )
    );

    let mut schema_wrong = policy_score(
        burn_dragon_universality::ruliad::RuliadAnswerStatus::SchemaValidWrong,
        0,
        1_000_000,
    );
    schema_wrong.observed_field_count = 1;
    assert!(
        LanguageTrainModel::<TestBackend>::ruliad_score_has_rollout_recovery_signal(
            &schema_wrong,
            500_000,
            750_000,
        )
    );

    let malformed = policy_score(
        burn_dragon_universality::ruliad::RuliadAnswerStatus::Malformed,
        0,
        1_000_000,
    );
    assert!(
        LanguageTrainModel::<TestBackend>::ruliad_score_has_rollout_recovery_signal(
            &malformed, 0, 0,
        )
    );
    assert!(
        !LanguageTrainModel::<TestBackend>::ruliad_score_has_rollout_recovery_signal(
            &malformed, 0, 1_000_001,
        )
    );
}

#[test]
fn ruliad_policy_advantage_guard_blocks_positive_wrong_schema() {
    let config = strict_policy_advantage_gate_config();
    let scores = vec![
        policy_score(
            burn_dragon_universality::ruliad::RuliadAnswerStatus::Partial,
            500_000,
            1_000_000,
        ),
        policy_score(
            burn_dragon_universality::ruliad::RuliadAnswerStatus::SchemaValidWrong,
            0,
            1_000_000,
        ),
    ];
    let mut advantages = [-0.4, 0.9];
    assert!(
        LanguageTrainModel::<TestBackend>::constrain_ruliad_policy_advantages(
            &scores,
            &mut advantages,
            config,
        )
    );
    assert_eq!(advantages[0], -0.4);
    assert_eq!(advantages[1], 0.0);
}

#[test]
fn ruliad_policy_advantage_guard_skips_all_wrong_groups() {
    let config = strict_policy_advantage_gate_config();
    let scores = vec![
        policy_score(
            burn_dragon_universality::ruliad::RuliadAnswerStatus::SchemaValidWrong,
            0,
            1_000_000,
        ),
        policy_score(
            burn_dragon_universality::ruliad::RuliadAnswerStatus::Malformed,
            0,
            1_000_000,
        ),
    ];
    let mut advantages = [0.9, -0.9];
    assert!(
        !LanguageTrainModel::<TestBackend>::constrain_ruliad_policy_advantages(
            &scores,
            &mut advantages,
            config,
        )
    );
}

#[test]
fn ruliad_policy_advantage_guard_skips_weak_partial_groups() {
    let config = strict_policy_advantage_gate_config();
    let scores = vec![
        policy_score(
            burn_dragon_universality::ruliad::RuliadAnswerStatus::Partial,
            250_000,
            1_000_000,
        ),
        policy_score(
            burn_dragon_universality::ruliad::RuliadAnswerStatus::SchemaValidWrong,
            0,
            1_000_000,
        ),
    ];
    let mut advantages = [0.9, -0.9];
    assert!(
        !LanguageTrainModel::<TestBackend>::constrain_ruliad_policy_advantages(
            &scores,
            &mut advantages,
            config,
        )
    );
}

#[test]
fn ruliad_policy_advantage_guard_skips_low_quality_partials() {
    let config = strict_policy_advantage_gate_config();
    let scores = vec![policy_score(
        burn_dragon_universality::ruliad::RuliadAnswerStatus::Partial,
        500_000,
        250_000,
    )];
    let mut advantages = [0.9];
    assert!(
        !LanguageTrainModel::<TestBackend>::constrain_ruliad_policy_advantages(
            &scores,
            &mut advantages,
            config,
        )
    );
}

#[test]
fn ruliad_verifier_policy_loss_supports_vpo_mode() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 11);
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_ruliad_supervision(RuliadSupervisionConfig {
        verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
            enabled: true,
            mode: crate::config::train::RuliadVerifierRewardMode::VpoIndependent,
            weight: 0.1,
            group_size: 2,
            max_completion_tokens: 2,
            every_steps: 1,
            top_k: 1,
            kl_weight: 0.0,
            vpo_scalarizations: 4,
            ..Default::default()
        },
        ..Default::default()
    });
    let vpo_config = crate::config::train::RuliadVerifierRewardConfig {
        enabled: true,
        mode: crate::config::train::RuliadVerifierRewardMode::VpoIndependent,
        vpo_correctness_mass_floor: 0.70,
        vpo_schema_quality_mass_floor: 0.10,
        vpo_completion_health_mass_floor: 0.10,
        vpo_compactness_max_weight: 0.05,
        ..Default::default()
    };
    let scalarizations = model.ruliad_vpo_scalarizations(17, 4, vpo_config);
    assert_eq!(scalarizations.len(), 4);
    for scalarization in scalarizations {
        assert!(
            scalarization.iter().all(|weight| *weight >= 0.0),
            "VPO scalarization weights should be non-negative"
        );
        let sum = scalarization.iter().sum::<f32>();
        assert!(
            (sum - 1.0).abs() < 1.0e-5,
            "VPO scalarization should sum to one, got {sum}"
        );
        let correctness_mass = scalarization[0..=4].iter().sum::<f32>();
        let schema_mass = scalarization[6];
        let health_mass = scalarization[8..=9].iter().sum::<f32>();
        assert!(
            correctness_mass >= 0.70 - 1.0e-5,
            "correctness mass floor should hold, got {correctness_mass}"
        );
        assert!(
            schema_mass >= 0.10 - 1.0e-5,
            "schema-quality mass floor should hold, got {schema_mass}"
        );
        assert!(
            health_mass >= 0.10 - 1.0e-5,
            "health mass floor should hold, got {health_mass}"
        );
        assert!(
            scalarization[5] <= 0.05 + 1.0e-5,
            "compactness weight should be capped"
        );
    }
    let item = burn_dragon_universality::RuliadEvalItem {
        oracle_hash: "h0".to_string(),
        sample_index: 17,
        split: burn_dragon_universality::SampleSplit::Train,
        family: "law".to_string(),
        task_kind: "category_law".to_string(),
        math_domains: vec!["category".to_string()],
        reasoning_modes: vec!["equational".to_string()],
        prompt: "?:q\n!:".to_string(),
        expected_answer: "ok=1".to_string(),
        difficulty_level: Some(0),
        spec: None,
    };
    let policy_batch = crate::dataset::RuliadPolicyBatch {
        samples: vec![crate::dataset::RuliadPolicySample {
            item,
            prompt_tokens: vec![1, 2, 3],
        }],
        tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
            vocab_size: 257,
            eos_id: None,
        },
        stop_token_id: None,
        sampling_metadata: None,
    };
    let loss = model
        .ruliad_verifier_policy_loss(&policy_batch, &device, 8)
        .expect("VPO verifier policy loss");
    let loss = tensor_scalar(loss);
    assert!(
        loss.is_finite(),
        "VPO verifier policy loss should be finite: {loss}"
    );
}

#[test]
fn ruliad_verifier_policy_loss_can_include_oracle_candidate() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 17);
    let dir = tempfile::tempdir().expect("tempdir");
    let telemetry_path = dir
        .path()
        .join("events")
        .join("ruliad_verifier_policy.jsonl");
    let mut config = tiny_model_config();
    config.vocab_size = 257;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_ruliad_supervision(RuliadSupervisionConfig {
            verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                enabled: true,
                mode: crate::config::train::RuliadVerifierRewardMode::VpoIndependent,
                weight: 0.1,
                group_size: 2,
                max_completion_tokens: 16,
                every_steps: 1,
                top_k: 1,
                kl_weight: 0.0,
                vpo_scalarizations: 4,
                positive_advantage_requires_correctness: true,
                positive_advantage_min_partial_progress_ppm: 500_000,
                positive_advantage_min_completion_quality_ppm: 750_000,
                include_oracle_candidate: true,
                ..Default::default()
            },
            ..Default::default()
        })
        .with_ruliad_policy_telemetry_path(Some(telemetry_path.clone()));
    let item = burn_dragon_universality::RuliadEvalItem {
        oracle_hash: "h0".to_string(),
        sample_index: 29,
        split: burn_dragon_universality::SampleSplit::Train,
        family: "law".to_string(),
        task_kind: "category_law".to_string(),
        math_domains: vec!["category".to_string()],
        reasoning_modes: vec!["equational".to_string()],
        prompt: "?:q\n!:".to_string(),
        expected_answer: "ok=1".to_string(),
        difficulty_level: Some(0),
        spec: None,
    };
    let policy_batch = crate::dataset::RuliadPolicyBatch {
        samples: vec![crate::dataset::RuliadPolicySample {
            item,
            prompt_tokens: vec![1, 2, 3],
        }],
        tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
            vocab_size: 257,
            eos_id: None,
        },
        stop_token_id: None,
        sampling_metadata: None,
    };
    let loss = model
        .ruliad_verifier_policy_loss(&policy_batch, &device, 32)
        .expect("oracle VPO verifier policy loss");
    assert!(tensor_scalar(loss).is_finite());
    let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
    let value: serde_json::Value =
        serde_json::from_str(content.lines().next().expect("telemetry line"))
            .expect("telemetry json");
    assert_eq!(value["oracle_sample_groups"], 1);
    assert_eq!(value["oracle_completion_rows"], 1);
    assert_eq!(value["oracle_truncated_completion_rows"], 0);
    assert_eq!(value["policy_update_applied"], true);
    assert!(
        value["vector_semantic_match_mean"]
            .as_f64()
            .expect("semantic mean")
            > 0.0,
        "oracle candidate should provide a correctness-positive row"
    );
}

#[test]
fn ruliad_rollout_recovery_accepts_malformed_and_missing_corruptions() {
    use burn_dragon_universality::ruliad::RuliadAnswerStatus;

    let min_partial = 500_000;
    let min_quality = 750_000;
    let accepts = |status, partial, quality| {
        LanguageTrainModel::<TestBackend>::ruliad_score_has_rollout_recovery_signal(
            &ruliad_test_score(status, partial, quality),
            min_partial,
            min_quality,
        )
    };

    assert!(accepts(
        RuliadAnswerStatus::SchemaValidWrong,
        0,
        min_quality
    ));
    assert!(accepts(RuliadAnswerStatus::Malformed, 0, min_quality));
    assert!(accepts(RuliadAnswerStatus::Missing, 0, min_quality));
    assert!(accepts(
        RuliadAnswerStatus::Partial,
        min_partial,
        min_quality
    ));
    assert!(!accepts(
        RuliadAnswerStatus::Partial,
        min_partial.saturating_sub(1),
        min_quality
    ));
    assert!(!accepts(
        RuliadAnswerStatus::Malformed,
        0,
        min_quality.saturating_sub(1)
    ));
    assert!(!accepts(
        RuliadAnswerStatus::VerifierMatch,
        1_000_000,
        min_quality
    ));
    assert!(!accepts(
        RuliadAnswerStatus::SemanticMatch,
        1_000_000,
        min_quality
    ));
}

#[test]
fn ruliad_structured_negative_answers_mutate_prompt_bound_fields() {
    let negatives =
        LanguageTrainModel::<TestBackend>::ruliad_structured_negative_answers("ok=1;l=17;r=17", 3);

    assert_eq!(negatives.len(), 3);
    assert!(negatives.iter().all(|answer| answer != "ok=1;l=17;r=17"));
    assert!(
        negatives.iter().any(|answer| answer.starts_with("ok=0")),
        "boolean fields should be mutated into plausible wrong answers: {negatives:?}"
    );
    assert!(
        negatives
            .iter()
            .any(|answer| answer.contains("l=19") || answer.contains("l=18")),
        "numeric fields should be mutated without destroying the answer schema: {negatives:?}"
    );
}

#[test]
fn ruliad_structured_negative_answers_include_template_collapse_hard_negatives() {
    let proof_negatives =
        LanguageTrainModel::<TestBackend>::ruliad_structured_negative_answers_with_templates(
            "ok=1;l=17;r=17",
            1,
            2,
        );
    let proof_texts = proof_negatives
        .iter()
        .map(|(answer, _kind)| answer.as_str())
        .collect::<Vec<_>>();
    assert!(
        proof_texts.contains(&"ok=1;l=5;r=5"),
        "proof hard negatives should target the observed l=5/r=5 attractor: {proof_texts:?}"
    );
    assert!(
        proof_texts.contains(&"ok=1;l=1;r=1"),
        "proof hard negatives should target the observed l/r collapse: {proof_texts:?}"
    );
    assert!(
        proof_negatives
            .iter()
            .any(|(_answer, kind)| *kind == RuliadStructuredNegativeKind::TemplateCollapse),
        "template rows should be tracked separately"
    );

    let automaton_negatives =
        LanguageTrainModel::<TestBackend>::ruliad_structured_negative_answers_with_templates(
            "acc=1", 0, 2,
        )
        .into_iter()
        .map(|(answer, _kind)| answer)
        .collect::<Vec<_>>();
    assert_eq!(automaton_negatives, vec!["acc=0".to_string()]);

    let eca_negatives =
        LanguageTrainModel::<TestBackend>::ruliad_structured_negative_answers_with_templates(
            "xlen=44;xalpha=01;xcounts=20,24;xedge=01",
            0,
            3,
        )
        .into_iter()
        .map(|(answer, _kind)| answer)
        .collect::<Vec<_>>();
    assert!(
        eca_negatives
            .iter()
            .any(|answer| answer == "xlen=13;xalpha=abc;nfcounts=1,1,0;nfedge=ba"),
        "ECA hard negatives should include the observed mixed x/nf lowercase attractor: {eca_negatives:?}"
    );

    let normal_form_negatives =
        LanguageTrainModel::<TestBackend>::ruliad_structured_negative_answers_with_templates(
            "nflen=44;nfalpha=01;nfcounts=20,24;nfedge=01",
            0,
            3,
        )
        .into_iter()
        .map(|(answer, _kind)| answer)
        .collect::<Vec<_>>();
    assert!(
        normal_form_negatives
            .iter()
            .all(|answer| answer.contains("nfedge=") && !answer.contains("xedge=")),
        "normal-form hard negatives should preserve the nfedge schema: {normal_form_negatives:?}"
    );
    assert!(
        normal_form_negatives
            .iter()
            .any(|answer| answer == "nflen=5;nfalpha=abc;nfcounts=1,1,0;nfedge=ba"),
        "normal-form hard negatives should include the observed lowercase normal-form attractor: {normal_form_negatives:?}"
    );
}

#[test]
fn ruliad_structured_negative_answers_with_schema_include_contract_sibling_negatives() {
    let negatives =
        LanguageTrainModel::<TestBackend>::ruliad_structured_negative_answers_with_schema(
            "xlen=44;xalpha=01;xcounts=20,24;xedge=01",
            0,
            0,
            8,
        );
    assert_eq!(negatives.len(), 8);
    assert!(
        negatives
            .iter()
            .all(|(_answer, kind)| *kind == RuliadStructuredNegativeKind::SchemaCollapse),
        "schema rows should be tracked separately: {negatives:?}"
    );
    let texts = negatives
        .iter()
        .map(|(answer, _kind)| answer.as_str())
        .collect::<Vec<_>>();
    assert!(
        texts.iter().any(|answer| answer.contains("nfalpha=")),
        "schema-collapse negatives should expose sibling normal-form keys: {texts:?}"
    );
    assert!(
        texts.contains(&"xlen=44;xalpha=01;xcounts=20,24"),
        "schema-collapse negatives should include missing-tail-field answers: {texts:?}"
    );
    assert!(
        texts.contains(&"xlen=44"),
        "schema-collapse negatives should include first-field-only answer collapse: {texts:?}"
    );
    assert!(
        texts.contains(&"ok=1;l=1;r=1"),
        "schema-collapse negatives should include the observed ok/l/r cross-contract prototype: {texts:?}"
    );
    assert!(
        texts.contains(&"acc=1"),
        "schema-collapse negatives should include compact cross-contract prototypes: {texts:?}"
    );
    assert!(
        texts
            .iter()
            .all(|answer| *answer != "xlen=44;xalpha=01;xcounts=20,24;xedge=01"),
        "schema-collapse negatives must not duplicate the oracle answer: {texts:?}"
    );
}

#[test]
fn ruliad_structured_proof_step_negatives_preserve_the_wire_contract() {
    let answer = "g4|a:r0|f|1.1";
    let negatives =
        LanguageTrainModel::<TestBackend>::ruliad_structured_negative_answers_with_templates(
            answer, 4, 1,
        );

    assert_eq!(negatives.len(), 5, "{negatives:?}");
    assert_eq!(
        negatives
            .iter()
            .filter(|(_, kind)| *kind == RuliadStructuredNegativeKind::TemplateCollapse)
            .count(),
        1
    );
    assert_eq!(
        negatives
            .iter()
            .filter(|(_, kind)| *kind == RuliadStructuredNegativeKind::FieldMutation)
            .count(),
        4
    );
    assert!(negatives.iter().all(|(candidate, _)| {
        candidate != answer
            && burn_dragon_universality::ruliad::wire::decode_model_proof_step(candidate).is_some()
    }));
    let oracle_fields = answer.split('|').collect::<Vec<_>>();
    for (field_index, oracle_field) in oracle_fields.iter().enumerate() {
        assert!(negatives.iter().any(|(candidate, kind)| {
            *kind == RuliadStructuredNegativeKind::FieldMutation
                && candidate
                    .split('|')
                    .nth(field_index)
                    .is_some_and(|field| field != *oracle_field)
        }));
    }
}

#[test]
fn ruliad_answer_value_completion_mask_marks_only_answer_values() {
    let tokenizer = burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer::from_config(
        &burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
            vocab_size: 257,
            eos_id: None,
        },
    )
    .expect("tokenizer");
    let answer = "ok=1;l=17;r=17";
    let completion = tokenizer.encode_payload(&format!("{answer}\n[/R2]"));
    let mask = LanguageTrainModel::<TestBackend>::ruliad_answer_value_completion_mask(
        &tokenizer,
        answer,
        completion.len(),
    );
    let marked = completion
        .iter()
        .zip(mask.iter())
        .filter_map(|(token, active)| active.then_some(*token))
        .filter_map(char::from_u32)
        .collect::<String>();

    assert_eq!(marked, "11717");

    let answer = "g4|a:r0|f|1.1";
    let completion = tokenizer.encode_payload(&format!("{answer}\n[/R3]"));
    let mask = LanguageTrainModel::<TestBackend>::ruliad_answer_value_completion_mask(
        &tokenizer,
        answer,
        completion.len(),
    );
    let marked = completion
        .iter()
        .zip(mask.iter())
        .filter_map(|(token, active)| active.then_some(*token))
        .filter_map(char::from_u32)
        .collect::<String>();

    assert_eq!(marked, "4r0f1.1");
    assert_eq!(
        LanguageTrainModel::<TestBackend>::ruliad_answer_contract(answer).as_deref(),
        Some("proof_action_step")
    );
}

#[test]
fn ruliad_answer_key_completion_mask_marks_only_answer_keys() {
    let tokenizer = burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer::from_config(
        &burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
            vocab_size: 257,
            eos_id: None,
        },
    )
    .expect("tokenizer");
    let answer = "ok=1;l=17;r=17";
    let completion = tokenizer.encode_payload(&format!("{answer}\n[/R2]"));
    let mask = LanguageTrainModel::<TestBackend>::ruliad_answer_key_completion_mask(
        &tokenizer,
        answer,
        completion.len(),
    );
    let marked = completion
        .iter()
        .zip(mask.iter())
        .filter_map(|(token, active)| active.then_some(*token))
        .filter_map(char::from_u32)
        .collect::<String>();

    assert_eq!(marked, "oklr");
}

#[test]
fn ruliad_answer_schema_completion_mask_marks_keys_and_field_separators() {
    let tokenizer = burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer::from_config(
        &burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
            vocab_size: 257,
            eos_id: None,
        },
    )
    .expect("tokenizer");
    let answer = "ok=1;l=17;r=17";
    let completion = tokenizer.encode_payload(&format!("{answer}\n[/R2]"));
    let mask = LanguageTrainModel::<TestBackend>::ruliad_answer_schema_completion_mask(
        &tokenizer,
        answer,
        completion.len(),
    );
    let marked = completion
        .iter()
        .zip(mask.iter())
        .filter_map(|(token, active)| active.then_some(*token))
        .filter_map(char::from_u32)
        .collect::<String>();

    assert_eq!(marked, "ok=;l=;r=");
}

#[test]
fn ruliad_answer_schema_start_completion_mask_marks_first_key_bytes() {
    let tokenizer = burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer::from_config(
        &burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
            vocab_size: 257,
            eos_id: None,
        },
    )
    .expect("tokenizer");
    let answer = "xlen=14;xalpha=01;xcounts=8,6;xedge=01";
    let completion = tokenizer.encode_payload(&format!("{answer}\n[/R2]"));
    let mask = LanguageTrainModel::<TestBackend>::ruliad_answer_schema_start_completion_mask(
        &tokenizer,
        answer,
        completion.len(),
    );
    let marked = completion
        .iter()
        .zip(mask.iter())
        .filter_map(|(token, active)| active.then_some(*token))
        .filter_map(char::from_u32)
        .collect::<String>();

    assert_eq!(marked, "xxxx");
}

#[test]
fn ruliad_prompt_schema_value_rows_train_values_under_supplied_keys() {
    let tokenizer = burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer::from_config(
        &burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
            vocab_size: 257,
            eos_id: None,
        },
    )
    .expect("tokenizer");
    let prompt = tokenizer
        .encode_payload("?:prove\n!:")
        .into_iter()
        .map(i64::from)
        .collect::<Vec<_>>();

    let rows = LanguageTrainModel::<TestBackend>::ruliad_prompt_schema_value_completion_rows(
        &tokenizer,
        &prompt,
        "ok=1;l=17;r=17",
        burn_dragon_universality::ruliad::RULIAD_V2_DOCUMENT_CLOSE_MARKER,
        32,
        96,
        8,
    );

    assert_eq!(rows.len(), 3);
    let decoded_targets = rows
        .iter()
        .map(|(_inputs, targets, mask, active)| {
            assert_eq!(*active, mask.iter().filter(|value| **value > 0.0).count());
            let tokens = targets
                .iter()
                .zip(mask.iter())
                .filter_map(|(token, active)| (*active > 0.0).then_some(*token as u32))
                .collect::<Vec<_>>();
            tokenizer.decode_payload(&tokens, true)
        })
        .collect::<Vec<_>>();

    assert_eq!(
        decoded_targets,
        vec!["1;", "17;", "17\n[/R2]"],
        "schema-forced value rows should target field values and close markers"
    );
}

#[test]
fn ruliad_prompt_schema_value_rows_train_semantic_proof_step_fields() {
    let tokenizer = burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer::from_config(
        &burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
            vocab_size: 257,
            eos_id: None,
        },
    )
    .expect("tokenizer");
    let prompt = tokenizer
        .encode_payload("?:select;g=3;dst=x;at=1.1\n!:")
        .into_iter()
        .map(i64::from)
        .collect::<Vec<_>>();

    let rows = LanguageTrainModel::<TestBackend>::ruliad_prompt_schema_value_completion_rows(
        &tokenizer,
        &prompt,
        "g3|a:r0|f|1.1",
        burn_dragon_universality::ruliad::RULIAD_V2_DOCUMENT_CLOSE_MARKER,
        32,
        96,
        8,
    );

    assert_eq!(rows.len(), 4);
    let decoded_targets = rows
        .iter()
        .map(|(_inputs, targets, mask, active)| {
            assert_eq!(*active, mask.iter().filter(|value| **value > 0.0).count());
            let tokens = targets
                .iter()
                .zip(mask.iter())
                .filter_map(|(token, active)| (*active > 0.0).then_some(*token as u32))
                .collect::<Vec<_>>();
            tokenizer.decode_payload(&tokens, true)
        })
        .collect::<Vec<_>>();

    assert_eq!(decoded_targets, vec!["3|", "r0|", "f|", "1.1\n[/R2]"]);
}

fn prompt_value_binding_policy_batch() -> crate::dataset::RuliadPolicyBatch {
    let tokenization = burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
        vocab_size: 257,
        eos_id: None,
    };
    let tokenizer =
        burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer::from_config(&tokenization)
            .expect("tokenizer");
    let item = burn_dragon_universality::RuliadEvalItem {
        oracle_hash: "binding-test".to_string(),
        sample_index: 1,
        split: burn_dragon_universality::SampleSplit::Train,
        family: "formal_proof".to_string(),
        task_kind: "select_proof_action".to_string(),
        math_domains: vec!["formal_proof".to_string()],
        reasoning_modes: vec!["equational".to_string()],
        prompt: "?:select;g=3;dst=x;at=1.1\n!:".to_string(),
        expected_answer: "g3|a:r0|f|1.1".to_string(),
        difficulty_level: Some(0),
        spec: None,
    };
    crate::dataset::RuliadPolicyBatch {
        samples: vec![crate::dataset::RuliadPolicySample {
            prompt_tokens: tokenizer
                .encode_payload(&item.prompt)
                .into_iter()
                .map(i64::from)
                .collect(),
            item,
        }],
        tokenization,
        stop_token_id: None,
        sampling_metadata: None,
    }
}

fn prompt_value_binding_model_config() -> DragonConfig {
    let mut config = tiny_model_config();
    config.vocab_size = 257;
    config.sequence_kernel = burn_dragon_core::SequenceKernelConfig::dense_score_short_context();
    config.fused_kernels.rotary_embedding = burn_dragon_core::RotaryEmbedding::Alibi;
    config
}

fn scheduled_ruliad_policy_batch() -> crate::dataset::RuliadPolicyBatch {
    let tokenization = burn_dragon_universality::RuliadTokenizationConfig::StructuredSymbolic {
        vocab_size: 272,
        eos_id: Some(271),
    };
    let tokenizer =
        burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer::from_config(&tokenization)
            .expect("tokenizer");
    let bundle = burn_dragon_universality::ruliad::formal::generate_formal_bundle(
        43,
        burn_dragon_universality::ruliad::formal::RuliadFormalGeneratorConfig {
            rewrite_depth: 2,
            leaf_count: 3,
            context_depth: 1,
            distractor_axioms: 1,
            ..Default::default()
        },
    )
    .expect("formal bundle");
    let proof_step_index = 1.min(bundle.certificate.step_count().saturating_sub(1));
    let actions = burn_dragon_universality::ruliad::oracle_proof_action_set(
        &bundle.problem,
        &bundle.certificate,
        proof_step_index,
        4,
    )
    .expect("proof action set");
    let answer_contract =
        burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep;
    let prompt =
        burn_dragon_universality::ruliad::ruliad_proof_action_prompt(&bundle.problem, &actions)
            .expect("proof prompt");
    let expected_answer = burn_dragon_universality::ruliad::proof_action_answer(
        &actions,
        actions.selected_index,
        answer_contract,
    )
    .expect("proof action answer");
    let item = burn_dragon_universality::RuliadEvalItem {
        oracle_hash: bundle.problem.canonical_hash().expect("problem hash"),
        sample_index: 43,
        split: burn_dragon_universality::SampleSplit::Train,
        family: "formal_proof".to_string(),
        task_kind: burn_dragon_universality::RuliadTaskKind::SelectProofAction
            .label()
            .to_string(),
        math_domains: vec!["formal_proof".to_string()],
        reasoning_modes: vec!["proof_construction".to_string()],
        prompt: prompt.clone(),
        expected_answer,
        difficulty_level: Some(0),
        spec: Some(burn_dragon_universality::RuliadSampleSpec::FormalProof {
            problem: bundle.problem,
            certificate: bundle.certificate,
            candidate: None,
            proof_step_index: Some(proof_step_index),
            action_presentation_rotation: Some(0),
            action_candidate_count: Some(actions.candidates.len()),
            action_answer_contract: answer_contract,
            task: burn_dragon_universality::RuliadTaskKind::SelectProofAction,
        }),
    };
    crate::dataset::RuliadPolicyBatch {
        samples: vec![crate::dataset::RuliadPolicySample {
            prompt_tokens: tokenizer
                .encode_payload(&prompt)
                .into_iter()
                .map(i64::from)
                .collect(),
            item,
        }],
        tokenization,
        stop_token_id: Some(271),
        sampling_metadata: None,
    }
}

fn scheduled_ruliad_supervision() -> RuliadSupervisionConfig {
    RuliadSupervisionConfig {
        mode: RuliadSupervisionMode::AnswerCompletion,
        prompt_value_binding: crate::config::RuliadPromptValueBindingConfig {
            enabled: true,
            every_steps: 2,
            phase_steps: 1,
            max_rows_per_step: 8,
            ..Default::default()
        },
        proof_policy: crate::config::RuliadProofPolicyTrainingConfig {
            enabled: true,
            mode: crate::config::RuliadProofPolicyTrainingMode::StaticExpert,
            scoring: crate::config::RuliadProofPolicyScoring::CompletionLikelihood,
            gradient_scope: crate::config::RuliadProofPolicyGradientScope::FullModel,
            normalization: crate::config::RuliadProofPolicyNormalization::PrefixConditional,
            candidate_symmetry: crate::config::RuliadProofPolicyCandidateSymmetry::BalancedRotation,
            presentation_risk: crate::config::RuliadProofPolicyPresentationRisk::Mean,
            every_steps: 4,
            start_after_steps: 0,
            max_rows_per_update: 2,
            max_presentation_rows_per_update: 8,
            counterfactual_targets_per_state: 1,
            candidates: 4,
            max_completion_tokens: 32,
            stratified_difficulty_levels: 1,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn scheduled_ruliad_model_config() -> DragonConfig {
    let mut config = prompt_value_binding_model_config();
    config.n_layer = 2;
    config.n_head = 2;
    config.mlp_internal_dim_multiplier = 2;
    config.vocab_size = 272;
    config.fused_kernels.relu_threshold = -0.25;
    config
}

fn scheduled_score_head_ruliad_supervision(
    scoring: crate::config::RuliadProofPolicyScoring,
    target: crate::config::RuliadProofPolicyTarget,
) -> RuliadSupervisionConfig {
    let mut supervision = scheduled_ruliad_supervision();
    supervision.prompt_value_binding.enabled = false;
    supervision.proof_policy.scoring = scoring;
    supervision.proof_policy.target = target;
    supervision.proof_policy.gradient_scope =
        crate::config::RuliadProofPolicyGradientScope::ScoreHeadOnly;
    supervision.proof_policy.normalization =
        crate::config::RuliadProofPolicyNormalization::CandidateConditional;
    supervision.proof_policy.counterfactual_objective =
        crate::config::RuliadProofPolicyCounterfactualObjective::Independent;
    supervision
}

fn scheduled_score_head_ruliad_model_config() -> DragonConfig {
    let mut config = scheduled_ruliad_model_config();
    config.sequence_score_head.enabled = true;
    config.sequence_score_head.projection_dim = 8;
    config
}

fn model_parameter_values(model: &LanguageTrainModel<TestBackend>) -> Vec<f32> {
    #[derive(Default)]
    struct ParameterCollector {
        values: Vec<f32>,
    }

    impl burn::module::ModuleVisitor<TestBackend> for ParameterCollector {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<TestBackend, D>>) {
            self.values.extend(
                param
                    .val()
                    .to_data()
                    .convert::<f32>()
                    .into_vec::<f32>()
                    .expect("parameter values"),
            );
        }
    }

    let mut collector = ParameterCollector::default();
    model.visit(&mut collector);
    collector.values
}

fn maximum_absolute_slice_difference(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f32::max)
}

#[derive(Debug)]
struct GradientComparison {
    cosine: f64,
    norm_ratio: f64,
    relative_l2_error: f64,
    parameter_tensors: usize,
    presence_mismatches: usize,
    presence_mismatch_ids: Vec<String>,
}

fn compare_model_gradients(
    model: &LanguageTrainModel<TestBackend>,
    candidate: &GradientsParams,
    reference: &GradientsParams,
) -> GradientComparison {
    #[derive(Default)]
    struct ComparisonVisitor<'a> {
        candidate: Option<&'a GradientsParams>,
        reference: Option<&'a GradientsParams>,
        dot: f64,
        candidate_squared_norm: f64,
        reference_squared_norm: f64,
        squared_error: f64,
        parameter_tensors: usize,
        presence_mismatches: usize,
        presence_mismatch_ids: Vec<String>,
    }

    impl burn::module::ModuleVisitor<TestBackend> for ComparisonVisitor<'_> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<TestBackend, D>>) {
            let candidate = self
                .candidate
                .expect("candidate gradients")
                .get::<TestInnerBackend, D>(param.id);
            let reference = self
                .reference
                .expect("reference gradients")
                .get::<TestInnerBackend, D>(param.id);
            self.parameter_tensors = self.parameter_tensors.saturating_add(1);
            self.presence_mismatches = self
                .presence_mismatches
                .saturating_add(usize::from(candidate.is_some() != reference.is_some()));
            if candidate.is_some() != reference.is_some() {
                self.presence_mismatch_ids.push(format!(
                    "{:?}:candidate={}:reference={}",
                    param.id,
                    candidate.is_some(),
                    reference.is_some()
                ));
            }

            let shape = param.val().shape();
            let device = param.val().device();
            let candidate = candidate.unwrap_or_else(|| Tensor::zeros(shape.clone(), &device));
            let reference = reference.unwrap_or_else(|| Tensor::zeros(shape, &device));
            let summary = Tensor::cat(
                vec![
                    (candidate.clone() * reference.clone()).sum().reshape([1]),
                    candidate.clone().square().sum().reshape([1]),
                    reference.clone().square().sum().reshape([1]),
                    (candidate - reference).square().sum().reshape([1]),
                ],
                0,
            )
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("gradient comparison summary");
            self.dot += f64::from(summary[0]);
            self.candidate_squared_norm += f64::from(summary[1]);
            self.reference_squared_norm += f64::from(summary[2]);
            self.squared_error += f64::from(summary[3]);
        }
    }

    let mut visitor = ComparisonVisitor {
        candidate: Some(candidate),
        reference: Some(reference),
        ..ComparisonVisitor::default()
    };
    model.visit(&mut visitor);
    let candidate_norm = visitor.candidate_squared_norm.max(0.0).sqrt();
    let reference_norm = visitor.reference_squared_norm.max(0.0).sqrt();
    GradientComparison {
        cosine: visitor.dot / (candidate_norm * reference_norm).max(1.0e-30),
        norm_ratio: candidate_norm / reference_norm.max(1.0e-30),
        relative_l2_error: visitor.squared_error.max(0.0).sqrt() / reference_norm.max(1.0e-30),
        parameter_tensors: visitor.parameter_tensors,
        presence_mismatches: visitor.presence_mismatches,
        presence_mismatch_ids: visitor.presence_mismatch_ids,
    }
}

fn recurrent_rho_values(state: &ModelState<TestBackend>) -> Vec<f32> {
    let mut values = Vec::new();
    for layer in &state.layers {
        values.extend(
            layer
                .rho
                .as_ref()
                .expect("linear-attention rho")
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("rho values"),
        );
        if let Some(rho_norm) = layer.rho_norm.as_ref() {
            values.extend(
                rho_norm
                    .to_data()
                    .convert::<f32>()
                    .into_vec::<f32>()
                    .expect("rho norm values"),
            );
        }
    }
    values
}

#[test]
fn context_only_stream_chunks_advance_state_without_weight_decay_updates() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 20260811);
    let base =
        crate::train::test_support::deterministic_matrix_parameters(
            DragonModel::<TestBackend>::new(scheduled_ruliad_model_config(), &device),
        );

    for algorithm in [
        TrainingAlgorithm::Backpropagation,
        TrainingAlgorithm::PredictiveCoding,
    ] {
        let mut model = LanguageTrainModel::new(base.clone())
            .with_training_algorithm(algorithm)
            .with_local_predictive_coding(LocalPredictiveCodingConfig {
                solver: LocalPredictiveCodingSolver::FixedPrediction,
                ..Default::default()
            })
            .with_tbptt_chunk_size(Some(8))
            .with_tbptt_persist_across_steps(true);
        let before = model_parameter_values(&model);
        let batch = SequenceBatch::new(
            Tensor::from_data(
                TensorData::new(vec![1_i64, 2, 3, 4, 5, 6, 7, 8], [1, 8]),
                &device,
            ),
            Tensor::from_data(
                TensorData::new(vec![2_i64, 3, 4, 5, 6, 7, 8, 9], [1, 8]),
                &device,
            ),
            None,
        )
        .with_loss_mask(Some(Tensor::zeros([1, 8], &device)))
        .with_supervised_token_count(Some(0))
        .with_reset_stream_state(true);
        let output = burn_train::TrainStep::step(&model, batch);
        assert!(
            output.grads.is_empty(),
            "{algorithm:?} must not expose zero-gradient parameter tensors"
        );
        assert_eq!(
            model
                .peek_step_state_for_test()
                .expect("context carry state")
                .position,
            8
        );
        let mut optimizer = AdamWConfig::new()
            .with_weight_decay(0.5)
            .init::<TestBackend, LanguageTrainModel<TestBackend>>();
        model = burn_train::TrainStep::optimize::<TestBackend, _>(
            model,
            &mut optimizer,
            3.0e-4,
            output.grads,
        );
        assert_eq!(
            maximum_absolute_slice_difference(&before, &model_parameter_values(&model)),
            0.0,
            "{algorithm:?} context-only chunks must not decay trained parameters"
        );
    }
}

#[test]
fn fixed_prediction_matches_stateful_production_objective_schedule() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 20260808);
    let base =
        crate::train::test_support::deterministic_matrix_parameters(
            DragonModel::<TestBackend>::new(scheduled_ruliad_model_config(), &device),
        );
    let make_model = |model, algorithm| {
        LanguageTrainModel::new(model)
            .with_training_algorithm(algorithm)
            .with_local_predictive_coding(LocalPredictiveCodingConfig {
                solver: LocalPredictiveCodingSolver::FixedPrediction,
                terminal_criterion:
                    crate::config::LocalPredictiveCodingTerminalCriterion::RuliadVerifierSet,
                factor_reduction: PredictiveCodingFactorReduction::Sum,
                ..Default::default()
            })
            .with_ruliad_supervision(scheduled_ruliad_supervision())
            .with_tbptt_chunk_size(Some(16))
            .with_tbptt_persist_across_steps(true)
            .with_stochastic_seed(20260808)
    };
    let mut backprop = make_model(base.clone(), TrainingAlgorithm::Backpropagation);
    let mut fixed = make_model(base, TrainingAlgorithm::PredictiveCoding);
    let fixed_profile = fixed.local_predictive_coding_profile();
    let mut backprop_optimizer =
        AdamWConfig::new().init::<TestBackend, LanguageTrainModel<TestBackend>>();
    let mut fixed_optimizer =
        AdamWConfig::new().init::<TestBackend, LanguageTrainModel<TestBackend>>();
    let policy_batch = Arc::new(scheduled_ruliad_policy_batch());
    let learning_rate = 3.0e-4;
    let block_size = 128;

    for step_index in 0..8 {
        let values = (0..block_size)
            .map(|index| ((index * 17 + step_index * 29) % 270 + 1) as i64)
            .collect::<Vec<_>>();
        let targets = values
            .iter()
            .enumerate()
            .map(|(index, value)| ((value + index as i64 + 1) % 270) + 1)
            .collect::<Vec<_>>();
        let make_batch = || {
            let context_only = step_index == 4;
            let batch = SequenceBatch::new(
                Tensor::from_data(TensorData::new(values.clone(), [1, block_size]), &device),
                Tensor::from_data(TensorData::new(targets.clone(), [1, block_size]), &device),
                None,
            )
            .with_ruliad_policy_batch(Some(policy_batch.clone()))
            .with_absolute_step(step_index)
            .with_reset_stream_state(step_index == 0);
            if context_only {
                batch
                    .with_loss_mask(Some(Tensor::zeros([1, block_size], &device)))
                    .with_supervised_token_count(Some(0))
            } else {
                batch
            }
        };

        let backprop_output = burn_train::TrainStep::step(&backprop, make_batch());
        let fixed_output = burn_train::TrainStep::step(&fixed, make_batch());
        let backprop_loss = scalar_loss(TrainOutput {
            grads: GradientsParams::new(),
            item: backprop_output.item,
        });
        let fixed_loss = scalar_loss(TrainOutput {
            grads: GradientsParams::new(),
            item: fixed_output.item,
        });
        assert!(
            (backprop_loss - fixed_loss).abs() < 2.0e-5,
            "objective loss diverged at schedule step {step_index}: backprop={backprop_loss} fixed={fixed_loss}"
        );

        backprop = burn_train::TrainStep::optimize::<TestBackend, _>(
            backprop,
            &mut backprop_optimizer,
            learning_rate,
            backprop_output.grads,
        );
        fixed = burn_train::TrainStep::optimize::<TestBackend, _>(
            fixed,
            &mut fixed_optimizer,
            learning_rate,
            fixed_output.grads,
        );

        let parameter_difference = maximum_absolute_slice_difference(
            &model_parameter_values(&backprop),
            &model_parameter_values(&fixed),
        );
        assert!(
            parameter_difference < 3.0e-4,
            "parameter trajectory diverged at schedule step {step_index}: max_abs={parameter_difference}"
        );
        let backprop_state = backprop
            .peek_step_state_for_test()
            .expect("backprop persistent state");
        let fixed_state = fixed
            .peek_step_state_for_test()
            .expect("fixed-prediction persistent state");
        assert_eq!(backprop_state.position, fixed_state.position);
        let rho_difference = maximum_absolute_slice_difference(
            &recurrent_rho_values(&backprop_state),
            &recurrent_rho_values(&fixed_state),
        );
        assert!(
            rho_difference < 3.0e-4,
            "recurrent trajectory diverged at schedule step {step_index}: max_abs={rho_difference}"
        );
    }

    let snapshot = fixed_profile.snapshot();
    assert_eq!(snapshot.global_backward_calls, 0, "snapshot={snapshot:?}");
    assert_eq!(snapshot.structured_terminal_steps, 2);
    assert_eq!(backprop.gradient_scale_step_index(), 7);
    assert_eq!(fixed.gradient_scale_step_index(), 7);
}

#[test]
fn residual_decoder_calibration_reaches_the_production_verifier_objective() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 20260811);
    let telemetry_dir = tempfile::tempdir().expect("telemetry dir");
    let telemetry_path = telemetry_dir.path().join("policy.jsonl");
    let mut supervision = scheduled_ruliad_supervision();
    supervision.prompt_value_binding.enabled = false;
    supervision.proof_policy.mode =
        crate::config::RuliadProofPolicyTrainingMode::StaticThenPairedDagger;
    supervision.proof_policy.scoring = crate::config::RuliadProofPolicyScoring::ResidualEnergy;
    supervision.proof_policy.decoder_calibration_steps = 4;
    supervision.proof_policy.target =
        crate::config::RuliadProofPolicyTarget::VerifiedProgressDistribution;
    supervision.proof_policy.gradient_scope =
        crate::config::RuliadProofPolicyGradientScope::FullModel;
    supervision.proof_policy.normalization =
        crate::config::RuliadProofPolicyNormalization::CandidateConditional;
    supervision.proof_policy.dagger_start_after_steps = 4;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        scheduled_score_head_ruliad_model_config(),
        &device,
    ))
    .with_ruliad_supervision(supervision)
    .with_ruliad_proof_policy_telemetry_path(Some(telemetry_path.clone()))
    .with_stochastic_seed(20260811);
    let policy_batch = scheduled_ruliad_policy_batch();

    for step_index in [0, 4] {
        let objective = model
            .ruliad_proof_policy_objective_at_step(&policy_batch, &device, 128, step_index)
            .unwrap_or_else(|| panic!("scheduled objective at step {step_index}"));
        assert!(tensor_scalar(objective.loss).is_finite());
    }

    let events = std::fs::read_to_string(telemetry_path)
        .expect("policy telemetry")
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("telemetry json"))
        .collect::<Vec<_>>();
    assert_eq!(events.len(), 2, "events={events:#?}");
    assert_eq!(events[0]["step_index"], 0);
    assert_eq!(events[0]["objective"], "vocabulary_marginal_equivalent_v1");
    assert_eq!(events[0]["target"], "expert_set");
    assert_eq!(events[0]["mode"], "static_expert");
    assert_eq!(events[1]["step_index"], 4);
    assert_eq!(
        events[1]["objective"],
        "autoregressive_residual_energy_counterfactual_v1"
    );
    assert_eq!(events[1]["target"], "verified_progress_distribution");
    assert_eq!(events[1]["mode"], "paired_dagger");
}

#[test]
fn factorized_joint_materializes_both_normalized_training_factors() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 20260812);
    let telemetry_dir = tempfile::tempdir().expect("telemetry dir");
    let telemetry_path = telemetry_dir.path().join("policy.jsonl");
    let mut supervision = scheduled_score_head_ruliad_supervision(
        crate::config::RuliadProofPolicyScoring::ResidualEnergy,
        crate::config::RuliadProofPolicyTarget::ExpertSet,
    );
    supervision.proof_policy.counterfactual_objective =
        crate::config::RuliadProofPolicyCounterfactualObjective::FactorizedJoint;
    supervision.proof_policy.every_steps = 4;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        scheduled_score_head_ruliad_model_config(),
        &device,
    ))
    .with_ruliad_supervision(supervision)
    .with_ruliad_proof_policy_telemetry_path(Some(telemetry_path.clone()))
    .with_stochastic_seed(20260812);
    let policy_batch = scheduled_ruliad_policy_batch();

    for step_index in [0, 4] {
        let objective = model
            .ruliad_proof_policy_objective_at_step(&policy_batch, &device, 128, step_index)
            .unwrap_or_else(|| panic!("scheduled objective at step {step_index}"));
        assert!(tensor_scalar(objective.loss).is_finite());
    }

    let events = std::fs::read_to_string(telemetry_path)
        .expect("policy telemetry")
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("telemetry json"))
        .collect::<Vec<_>>();
    assert_eq!(events.len(), 2, "events={events:#?}");
    assert_eq!(events[0]["step_index"], 0);
    assert_eq!(events[0]["objective"], "vocabulary_marginal_equivalent_v1");
    assert_eq!(events[0]["gradient_scope"], "full_model");
    assert_eq!(events[1]["step_index"], 4);
    assert_eq!(
        events[1]["objective"],
        "residual_energy_target_group_conditional_v1"
    );
    assert_eq!(events[1]["gradient_scope"], "score_head_only");
    assert_eq!(events[1]["target_group_conditional_groups"], 1);
}

#[test]
fn static_residual_policy_panel_identity_matches_global_and_local_pc_builders() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let telemetry_dir = tempfile::tempdir().expect("telemetry dir");
    let telemetry_path = telemetry_dir.path().join("policy.jsonl");
    let mut supervision = scheduled_score_head_ruliad_supervision(
        crate::config::RuliadProofPolicyScoring::ResidualEnergy,
        crate::config::RuliadProofPolicyTarget::ExpertSet,
    );
    supervision.proof_policy.mode = crate::config::RuliadProofPolicyTrainingMode::StaticExpert;
    supervision.proof_policy.counterfactual_objective =
        crate::config::RuliadProofPolicyCounterfactualObjective::Independent;
    supervision.proof_policy.max_rows_per_update = 2;
    supervision.proof_policy.max_presentation_rows_per_update = 8;
    let policy = supervision.proof_policy;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        scheduled_score_head_ruliad_model_config(),
        &device,
    ))
    .with_ruliad_supervision(supervision)
    .with_ruliad_proof_policy_telemetry_path(Some(telemetry_path.clone()));
    let policy_batch = scheduled_ruliad_policy_batch();

    let objective = model
        .ruliad_proof_policy_objective_at_step(&policy_batch, &device, 128, 0)
        .expect("global residual policy objective");
    assert!(tensor_scalar(objective.loss).is_finite());
    let event: serde_json::Value = serde_json::from_str(
        std::fs::read_to_string(telemetry_path)
            .expect("global policy telemetry")
            .lines()
            .next()
            .expect("global policy event"),
    )
    .expect("global policy telemetry JSON");
    let global_fingerprint = event["objective_panel_fingerprint"]
        .as_u64()
        .expect("global objective-panel fingerprint");
    let local = crate::train::local_predictive_coding::prepare_ruliad_verifier_terminal::<
        TestInnerBackend,
    >(&policy_batch, policy, 128, 272, &device)
    .expect("local-PC residual policy objective");

    assert_ne!(global_fingerprint, 0);
    assert_eq!(global_fingerprint, local.stats.objective_panel_fingerprint);
}

#[test]
fn static_residual_policy_gradients_match_global_and_local_pc_builders() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 20260812);
    let mut supervision = scheduled_score_head_ruliad_supervision(
        crate::config::RuliadProofPolicyScoring::ResidualEnergy,
        crate::config::RuliadProofPolicyTarget::ExpertSet,
    );
    supervision.proof_policy.mode = crate::config::RuliadProofPolicyTrainingMode::StaticExpert;
    supervision.proof_policy.gradient_scope =
        crate::config::RuliadProofPolicyGradientScope::PolicyPath;
    supervision.proof_policy.counterfactual_objective =
        crate::config::RuliadProofPolicyCounterfactualObjective::Independent;
    let policy = supervision.proof_policy;
    let model =
        LanguageTrainModel::new(crate::train::test_support::deterministic_matrix_parameters(
            DragonModel::<TestBackend>::new(scheduled_score_head_ruliad_model_config(), &device),
        ))
        .with_ruliad_supervision(supervision);
    let policy_batch = scheduled_ruliad_policy_batch();

    let global = model
        .ruliad_proof_policy_objective_at_step(&policy_batch, &device, 128, 0)
        .expect("global residual policy objective");
    let global_loss = tensor_scalar(global.loss.clone());
    let global_grads = GradientsParams::from_grads(global.loss.backward(), &model);
    let prepared = crate::train::local_predictive_coding::prepare_ruliad_verifier_terminal::<
        TestInnerBackend,
    >(&policy_batch, policy, 128, 272, &device)
    .expect("local-PC residual policy objective");
    let mut local =
        crate::train::local_predictive_coding::local_predictive_coding_verifier_train_step(
            &model.model,
            prepared,
            &crate::config::LocalPredictiveCodingConfig {
                solver: crate::config::LocalPredictiveCodingSolver::FixedPrediction,
                terminal_criterion:
                    crate::config::LocalPredictiveCodingTerminalCriterion::RuliadVerifierSet,
                factor_reduction: crate::config::PredictiveCodingFactorReduction::Sum,
                ..Default::default()
            },
            &crate::train::local_predictive_coding::LocalPredictiveCodingProfile::default(),
        );
    let local_loss = tensor_scalar(local.loss);
    for parameter_id in model
        .model
        .predictive_coding_structurally_inactive_parameter_ids()
        .expect("validated predictive-coding model")
    {
        let _ = local.grads.remove::<TestInnerBackend, 1>(parameter_id);
    }
    let comparison = compare_model_gradients(&model, &local.grads, &global_grads);

    assert!(
        (local_loss - global_loss).abs() < 2.0e-6,
        "local={local_loss} global={global_loss}"
    );
    assert_eq!(local.report.global_backward_calls, 0);
    assert_eq!(comparison.presence_mismatches, 0, "{comparison:?}");
    assert!(comparison.cosine > 0.999_98, "{comparison:?}");
    assert!(
        (comparison.norm_ratio - 1.0).abs() < 5.0e-4,
        "{comparison:?}"
    );
    assert!(comparison.relative_l2_error < 5.0e-4, "{comparison:?}");
}

#[test]
fn exact_temporal_fixed_prediction_matches_persistent_joint_language_and_verifier_updates() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 20260809);
    let base =
        crate::train::test_support::deterministic_matrix_parameters(
            DragonModel::<TestBackend>::new(scheduled_ruliad_model_config(), &device),
        );
    let mut supervision = scheduled_ruliad_supervision();
    supervision.prompt_value_binding.enabled = false;
    let make_model = |model, algorithm| {
        let exact_temporal_credit = matches!(algorithm, TrainingAlgorithm::PredictiveCoding);
        LanguageTrainModel::new(model)
            .with_training_algorithm(algorithm)
            .with_local_predictive_coding(LocalPredictiveCodingConfig {
                solver: LocalPredictiveCodingSolver::FixedPrediction,
                terminal_criterion:
                    crate::config::LocalPredictiveCodingTerminalCriterion::RuliadVerifierSetJoint,
                factor_reduction: PredictiveCodingFactorReduction::Sum,
                temporal_credit: if exact_temporal_credit {
                    burn_pc::PcTemporalCreditConfig {
                        mode: burn_pc::PcTemporalCreditMode::ExactWindow,
                        window_chunks: 2,
                    }
                } else {
                    burn_pc::PcTemporalCreditConfig::default()
                },
                ..Default::default()
            })
            .with_ruliad_supervision(supervision)
            .with_tbptt_chunk_size(Some(if exact_temporal_credit { 64 } else { 128 }))
            .with_tbptt_persist_across_steps(true)
            .with_stochastic_seed(20260809)
    };
    let mut backprop = make_model(base.clone(), TrainingAlgorithm::Backpropagation);
    let mut fixed = make_model(base, TrainingAlgorithm::PredictiveCoding);
    let fixed_profile = fixed.local_predictive_coding_profile();
    let mut backprop_optimizer =
        AdamWConfig::new().init::<TestBackend, LanguageTrainModel<TestBackend>>();
    let mut fixed_optimizer =
        AdamWConfig::new().init::<TestBackend, LanguageTrainModel<TestBackend>>();
    let policy_batch = Arc::new(scheduled_ruliad_policy_batch());
    let block_size = 128;

    for step_index in 0..8 {
        let values = (0..block_size)
            .map(|index| ((index * 19 + step_index * 31) % 270 + 1) as i64)
            .collect::<Vec<_>>();
        let targets = values
            .iter()
            .enumerate()
            .map(|(index, value)| ((value + index as i64 + 3) % 270) + 1)
            .collect::<Vec<_>>();
        let make_batch = || {
            SequenceBatch::new(
                Tensor::from_data(TensorData::new(values.clone(), [1, block_size]), &device),
                Tensor::from_data(TensorData::new(targets.clone(), [1, block_size]), &device),
                None,
            )
            .with_ruliad_policy_batch(Some(policy_batch.clone()))
            .with_absolute_step(step_index)
            .with_reset_stream_state(step_index == 0)
        };

        let backprop_output = burn_train::TrainStep::step(&backprop, make_batch());
        let fixed_output = burn_train::TrainStep::step(&fixed, make_batch());
        let backprop_loss = scalar_loss(TrainOutput {
            grads: GradientsParams::new(),
            item: backprop_output.item,
        });
        let fixed_loss = scalar_loss(TrainOutput {
            grads: GradientsParams::new(),
            item: fixed_output.item,
        });
        assert!(
            (backprop_loss - fixed_loss).abs() < 2.0e-5,
            "joint objective loss diverged at step {step_index}: backprop={backprop_loss} fixed={fixed_loss}"
        );

        backprop = burn_train::TrainStep::optimize::<TestBackend, _>(
            backprop,
            &mut backprop_optimizer,
            3.0e-4,
            backprop_output.grads,
        );
        fixed = burn_train::TrainStep::optimize::<TestBackend, _>(
            fixed,
            &mut fixed_optimizer,
            3.0e-4,
            fixed_output.grads,
        );
        let parameter_difference = maximum_absolute_slice_difference(
            &model_parameter_values(&backprop),
            &model_parameter_values(&fixed),
        );
        assert!(
            parameter_difference < 3.0e-4,
            "joint parameter trajectory diverged at step {step_index}: max_abs={parameter_difference}"
        );
        let backprop_state = backprop
            .peek_step_state_for_test()
            .expect("joint backprop persistent state");
        let fixed_state = fixed
            .peek_step_state_for_test()
            .expect("joint fixed-prediction persistent state");
        assert_eq!(backprop_state.position, (step_index + 1) * block_size);
        assert_eq!(backprop_state.position, fixed_state.position);
        let rho_difference = maximum_absolute_slice_difference(
            &recurrent_rho_values(&backprop_state),
            &recurrent_rho_values(&fixed_state),
        );
        assert!(
            rho_difference < 3.0e-4,
            "joint recurrent trajectory diverged at step {step_index}: max_abs={rho_difference}"
        );
    }

    let snapshot = fixed_profile.snapshot();
    assert_eq!(snapshot.global_backward_calls, 0, "snapshot={snapshot:?}");
    assert_eq!(snapshot.structured_terminal_steps, 2);
    assert_eq!(snapshot.optimizer_updates, 8);
}

#[test]
fn exact_temporal_full_model_residual_verifier_matches_joint_backprop_gradients() {
    assert_exact_temporal_residual_verifier_matches_joint_backprop_gradients(
        crate::config::RuliadProofPolicyGradientScope::FullModel,
    );
}

#[test]
fn exact_temporal_policy_path_residual_verifier_matches_joint_backprop_gradients() {
    assert_exact_temporal_residual_verifier_matches_joint_backprop_gradients(
        crate::config::RuliadProofPolicyGradientScope::PolicyPath,
    );
}

fn assert_exact_temporal_residual_verifier_matches_joint_backprop_gradients(
    gradient_scope: crate::config::RuliadProofPolicyGradientScope,
) {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 20260812);
    let base =
        crate::train::test_support::deterministic_matrix_parameters(
            DragonModel::<TestBackend>::new(scheduled_score_head_ruliad_model_config(), &device),
        );
    let mut supervision = scheduled_score_head_ruliad_supervision(
        crate::config::RuliadProofPolicyScoring::ResidualEnergy,
        crate::config::RuliadProofPolicyTarget::VerifiedProgressDistribution,
    );
    supervision.proof_policy.gradient_scope = gradient_scope;
    supervision.proof_policy.every_steps = 1;
    let make_model = |model, algorithm| {
        let exact_temporal_credit = matches!(algorithm, TrainingAlgorithm::PredictiveCoding);
        LanguageTrainModel::new(model)
            .with_training_algorithm(algorithm)
            .with_local_predictive_coding(LocalPredictiveCodingConfig {
                solver: LocalPredictiveCodingSolver::FixedPrediction,
                terminal_criterion:
                    crate::config::LocalPredictiveCodingTerminalCriterion::RuliadVerifierSetJoint,
                factor_reduction: PredictiveCodingFactorReduction::Sum,
                temporal_credit: if exact_temporal_credit {
                    burn_pc::PcTemporalCreditConfig {
                        mode: burn_pc::PcTemporalCreditMode::ExactWindow,
                        window_chunks: 8,
                    }
                } else {
                    burn_pc::PcTemporalCreditConfig::default()
                },
                ..Default::default()
            })
            .with_ruliad_supervision(supervision)
            .with_tbptt_chunk_size(Some(if exact_temporal_credit { 16 } else { 128 }))
            .with_tbptt_persist_across_steps(true)
            .with_stochastic_seed(20260812)
    };
    let mut backprop = make_model(base.clone(), TrainingAlgorithm::Backpropagation);
    let mut fixed = make_model(base, TrainingAlgorithm::PredictiveCoding);
    let policy_batch = Arc::new(scheduled_ruliad_policy_batch());
    let block_size = 128;
    let values = (0..block_size)
        .map(|index| ((index * 23 + 17) % 270 + 1) as i64)
        .collect::<Vec<_>>();
    let targets = values
        .iter()
        .enumerate()
        .map(|(index, value)| ((value + index as i64 + 5) % 270) + 1)
        .collect::<Vec<_>>();
    let mask = (0..block_size)
        .map(|index| i64::from(index % 7 != 2 && index % 11 != 5))
        .collect::<Vec<_>>();
    let supervised_tokens = mask.iter().filter(|value| **value != 0).count();
    let make_batch = || {
        SequenceBatch::new(
            Tensor::from_data(TensorData::new(values.clone(), [1, block_size]), &device),
            Tensor::from_data(TensorData::new(targets.clone(), [1, block_size]), &device),
            None,
        )
        .with_loss_mask(Some(Tensor::from_data(
            TensorData::new(mask.clone(), [1, block_size]),
            &device,
        )))
        .with_supervised_token_count(Some(supervised_tokens))
        .with_ruliad_policy_batch(Some(policy_batch.clone()))
        .with_absolute_step(0)
        .with_reset_stream_state(true)
    };

    let reference = burn_train::TrainStep::step(&backprop, make_batch());
    let candidate = burn_train::TrainStep::step(&fixed, make_batch());
    let reference_loss = scalar_loss(TrainOutput {
        grads: GradientsParams::new(),
        item: reference.item,
    });
    let candidate_loss = scalar_loss(TrainOutput {
        grads: GradientsParams::new(),
        item: candidate.item,
    });
    assert!(
        (reference_loss - candidate_loss).abs() < 2.0e-5,
        "{gradient_scope:?} residual-energy loss mismatch: reference={reference_loss} candidate={candidate_loss}"
    );

    let fidelity = compare_model_gradients(&fixed, &candidate.grads, &reference.grads);
    let parameter_ids = fixed
        .model
        .predictive_coding_parameter_ids()
        .expect("production PC parameter ids");
    assert_eq!(
        fidelity.presence_mismatches, 0,
        "{fidelity:?} ids={parameter_ids:?}"
    );
    assert!(fidelity.parameter_tensors >= 15, "{fidelity:?}");
    assert!(fidelity.cosine > 0.999_8, "{fidelity:?}");
    assert!((fidelity.norm_ratio - 1.0).abs() < 5.0e-4, "{fidelity:?}");
    assert!(fidelity.relative_l2_error < 5.0e-4, "{fidelity:?}");
    let mut backprop_optimizer =
        AdamWConfig::new().init::<TestBackend, LanguageTrainModel<TestBackend>>();
    let mut fixed_optimizer =
        AdamWConfig::new().init::<TestBackend, LanguageTrainModel<TestBackend>>();
    backprop = burn_train::TrainStep::optimize::<TestBackend, _>(
        backprop,
        &mut backprop_optimizer,
        3.0e-4,
        reference.grads,
    );
    fixed = burn_train::TrainStep::optimize::<TestBackend, _>(
        fixed,
        &mut fixed_optimizer,
        3.0e-4,
        candidate.grads,
    );
    let parameter_difference = maximum_absolute_slice_difference(
        &model_parameter_values(&backprop),
        &model_parameter_values(&fixed),
    );
    assert!(
        parameter_difference < 2.0e-6,
        "{gradient_scope:?} residual-energy AdamW update mismatch: max_abs={parameter_difference}"
    );
    let profile = fixed.local_predictive_coding_profile().snapshot();
    assert_eq!(profile.global_backward_calls, 0, "{profile:?}");
    assert_eq!(profile.temporal_state_vjp_calls, 14, "{profile:?}");
    assert_eq!(profile.structured_terminal_steps, 1, "{profile:?}");
}

fn assert_fixed_prediction_matches_joint_score_head_trajectory(
    scoring: crate::config::RuliadProofPolicyScoring,
    target: crate::config::RuliadProofPolicyTarget,
) {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 20260811);
    let base =
        crate::train::test_support::deterministic_matrix_parameters(
            DragonModel::<TestBackend>::new(scheduled_score_head_ruliad_model_config(), &device),
        );
    let make_model = |model, algorithm| {
        LanguageTrainModel::new(model)
            .with_training_algorithm(algorithm)
            .with_local_predictive_coding(LocalPredictiveCodingConfig {
                solver: LocalPredictiveCodingSolver::FixedPrediction,
                terminal_criterion:
                    crate::config::LocalPredictiveCodingTerminalCriterion::RuliadVerifierSetJoint,
                factor_reduction: PredictiveCodingFactorReduction::Sum,
                ..Default::default()
            })
            .with_ruliad_supervision(scheduled_score_head_ruliad_supervision(scoring, target))
            .with_tbptt_chunk_size(Some(128))
            .with_stochastic_seed(20260811)
    };
    let mut backprop = make_model(base.clone(), TrainingAlgorithm::Backpropagation);
    let mut fixed = make_model(base, TrainingAlgorithm::PredictiveCoding);
    let fixed_profile = fixed.local_predictive_coding_profile();
    let mut backprop_optimizer =
        AdamWConfig::new().init::<TestBackend, LanguageTrainModel<TestBackend>>();
    let mut fixed_optimizer =
        AdamWConfig::new().init::<TestBackend, LanguageTrainModel<TestBackend>>();
    let policy_batch = Arc::new(scheduled_ruliad_policy_batch());
    let block_size = 128;
    let updates = 32;

    for step_index in 0..updates {
        let values = (0..block_size)
            .map(|index| ((index * 23 + step_index * 37) % 270 + 1) as i64)
            .collect::<Vec<_>>();
        let targets = values
            .iter()
            .enumerate()
            .map(|(index, value)| ((value + index as i64 + 5) % 270) + 1)
            .collect::<Vec<_>>();
        let make_batch = || {
            SequenceBatch::new(
                Tensor::from_data(TensorData::new(values.clone(), [1, block_size]), &device),
                Tensor::from_data(TensorData::new(targets.clone(), [1, block_size]), &device),
                None,
            )
            .with_ruliad_policy_batch(Some(policy_batch.clone()))
            .with_absolute_step(step_index)
            .with_reset_stream_state(true)
        };

        let backprop_output = burn_train::TrainStep::step(&backprop, make_batch());
        let fixed_output = burn_train::TrainStep::step(&fixed, make_batch());
        let backprop_loss = scalar_loss(TrainOutput {
            grads: GradientsParams::new(),
            item: backprop_output.item,
        });
        let fixed_loss = scalar_loss(TrainOutput {
            grads: GradientsParams::new(),
            item: fixed_output.item,
        });
        assert!(
            (backprop_loss - fixed_loss).abs() < 2.0e-5,
            "joint {scoring:?}/{target:?} loss diverged at step {step_index}: backprop={backprop_loss} fixed={fixed_loss}"
        );

        backprop = burn_train::TrainStep::optimize::<TestBackend, _>(
            backprop,
            &mut backprop_optimizer,
            3.0e-4,
            backprop_output.grads,
        );
        fixed = burn_train::TrainStep::optimize::<TestBackend, _>(
            fixed,
            &mut fixed_optimizer,
            3.0e-4,
            fixed_output.grads,
        );
        let parameter_difference = maximum_absolute_slice_difference(
            &model_parameter_values(&backprop),
            &model_parameter_values(&fixed),
        );
        assert!(
            parameter_difference < 5.0e-6,
            "joint {scoring:?}/{target:?} parameter trajectory diverged at step {step_index}: max_abs={parameter_difference}"
        );
    }

    let snapshot = fixed_profile.snapshot();
    assert_eq!(snapshot.global_backward_calls, 0, "snapshot={snapshot:?}");
    assert_eq!(snapshot.structured_terminal_steps, 8);
    assert_eq!(snapshot.optimizer_updates, updates as u64);
}

#[test]
fn fixed_prediction_matches_joint_semantic_score_head_trajectory() {
    assert_fixed_prediction_matches_joint_score_head_trajectory(
        crate::config::RuliadProofPolicyScoring::SemanticEnergy,
        crate::config::RuliadProofPolicyTarget::ExpertSet,
    );
}

#[test]
fn fixed_prediction_matches_joint_residual_score_head_trajectory() {
    assert_fixed_prediction_matches_joint_score_head_trajectory(
        crate::config::RuliadProofPolicyScoring::ResidualEnergy,
        crate::config::RuliadProofPolicyTarget::ExpertSet,
    );
}

#[test]
fn fixed_prediction_matches_joint_verified_progress_trajectory() {
    assert_fixed_prediction_matches_joint_score_head_trajectory(
        crate::config::RuliadProofPolicyScoring::ResidualEnergy,
        crate::config::RuliadProofPolicyTarget::VerifiedProgressDistribution,
    );
}

#[test]
fn prompt_value_binding_primary_step_uses_local_pc_without_global_backward() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 41);
    let policy_batch = Arc::new(prompt_value_binding_policy_batch());
    let dir = tempfile::tempdir().expect("tempdir");
    let telemetry_path = dir.path().join("prompt_value_binding.jsonl");
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        prompt_value_binding_model_config(),
        &device,
    ))
    .with_training_algorithm(TrainingAlgorithm::PredictiveCoding)
    .with_local_predictive_coding(LocalPredictiveCodingConfig {
        solver: LocalPredictiveCodingSolver::FixedPrediction,
        ..Default::default()
    })
    .with_ruliad_supervision(RuliadSupervisionConfig {
        mode: RuliadSupervisionMode::AnswerCompletion,
        prompt_value_binding: crate::config::RuliadPromptValueBindingConfig {
            enabled: true,
            every_steps: 2,
            phase_steps: 1,
            max_rows_per_step: 8,
            ..Default::default()
        },
        ..Default::default()
    })
    .with_ruliad_prompt_value_binding_telemetry_path(Some(telemetry_path.clone()));
    let profile = model.local_predictive_coding_profile();
    let train_batch = SequenceBatch::new(
        Tensor::from_data(
            TensorData::new(vec![1_i64, 2, 3, 4, 5, 6, 7, 8], [1, 8]),
            &device,
        ),
        Tensor::from_data(
            TensorData::new(vec![2_i64, 3, 4, 5, 6, 7, 8, 9], [1, 8]),
            &device,
        ),
        None,
    )
    .with_ruliad_policy_batch(Some(policy_batch))
    .with_absolute_step(1);

    let output = burn_train::TrainStep::step(&model, train_batch);
    assert!(!output.grads.is_empty());
    let loss: LossValue<TestInnerBackend> = output.item.sync().adapt();
    let loss = loss
        .value()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("loss")[0];
    assert!(loss.is_finite());
    let snapshot = profile.snapshot();
    assert_eq!(snapshot.global_backward_calls, 0);
    assert_eq!(snapshot.steps, 1);
    let line = std::fs::read_to_string(telemetry_path).expect("binding telemetry");
    let event: serde_json::Value =
        serde_json::from_str(line.lines().next().expect("telemetry line")).expect("json");
    assert_eq!(event["algorithm"], "predictive_coding");
    assert_eq!(event["prompt_context"], "dataset_prompt");
    assert_eq!(event["objective"], "schema_values");
    assert_eq!(event["global_backward_calls"], 0);
    assert_eq!(event["rows"], 4);
    assert!(event["active_tokens"].as_u64().unwrap_or_default() > 0);
}

fn assert_prompt_value_binding_fixed_prediction_matches_global_backpropagation(
    objective: crate::config::RuliadPromptValueBindingObjective,
    expected_rows: usize,
) {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 43);
    let model =
        crate::train::test_support::deterministic_matrix_parameters(
            DragonModel::<TestBackend>::new(prompt_value_binding_model_config(), &device),
        );
    let learner =
        LanguageTrainModel::new(model.clone()).with_ruliad_supervision(RuliadSupervisionConfig {
            mode: RuliadSupervisionMode::AnswerCompletion,
            prompt_value_binding: crate::config::RuliadPromptValueBindingConfig {
                enabled: true,
                objective,
                every_steps: 2,
                phase_steps: 1,
                max_rows_per_step: 8,
                ..Default::default()
            },
            ..Default::default()
        });
    let batch = prompt_value_binding_policy_batch();
    let prepared = learner
        .prepare_ruliad_prompt_value_binding_batch(&batch, &device, 96, 0)
        .expect("prompt-value binding batch");
    assert_eq!(prepared.rows, expected_rows);
    assert!(prepared.active_tokens > 0);

    let report = crate::train::local_predictive_coding::local_predictive_coding_gradient_fidelity(
        &model,
        prepared.inputs,
        prepared.targets,
        Some(prepared.loss_mask),
        &LocalPredictiveCodingConfig {
            solver: LocalPredictiveCodingSolver::FixedPrediction,
            factor_reduction: PredictiveCodingFactorReduction::Sum,
            ..Default::default()
        },
    )
    .expect("prompt-value binding gradient fidelity");

    assert!(report.loss_absolute_error < 1.0e-6, "{report:?}");
    assert_eq!(report.pc_step.global_backward_calls, 0);
    assert!(
        report.global.cosine.is_some_and(|cosine| cosine > 0.999_99),
        "{report:?}"
    );
    assert!(
        report
            .global
            .relative_l2_error
            .is_some_and(|error| error < 1.0e-4),
        "{report:?}"
    );
}

#[test]
fn prompt_value_binding_fixed_prediction_matches_global_backpropagation() {
    assert_prompt_value_binding_fixed_prediction_matches_global_backpropagation(
        crate::config::RuliadPromptValueBindingObjective::SchemaValues,
        4,
    );
}

#[test]
fn prompt_full_completion_fixed_prediction_matches_global_backpropagation() {
    assert_prompt_value_binding_fixed_prediction_matches_global_backpropagation(
        crate::config::RuliadPromptValueBindingObjective::FullCompletion,
        1,
    );
}

#[test]
fn prompt_value_binding_can_share_the_exact_proof_policy_context() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let batch = scheduled_ruliad_policy_batch();
    let tokenizer = burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer::from_config(
        &batch.tokenization,
    )
    .expect("tokenizer");
    let mut supervision = scheduled_ruliad_supervision();
    supervision.prompt_value_binding.context =
        crate::config::RuliadPromptValueBindingContext::ProofPolicy;
    supervision.proof_policy.prompt_context =
        crate::config::RuliadProofPolicyPromptContext::LocalActionState;
    let learner = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        prompt_value_binding_model_config(),
        &device,
    ))
    .with_ruliad_supervision(supervision);

    let (prompt_tokens, answer) = learner
        .ruliad_prompt_value_binding_target(&batch.samples[0], &tokenizer, 1)
        .expect("proof-policy binding target");
    let prompt = tokenizer.decode_payload(
        &prompt_tokens
            .iter()
            .map(|token| u32::try_from(*token).expect("token id"))
            .collect::<Vec<_>>(),
        true,
    );
    assert!(prompt.ends_with("\n!:"), "{prompt}");
    assert!(!prompt.contains("[R3 "), "{prompt}");
    assert!(
        burn_dragon_universality::ruliad::wire::decode_model_proof_step(&answer).is_some(),
        "{answer}"
    );

    let prepared = learner
        .prepare_ruliad_prompt_value_binding_batch(&batch, &device, 256, 1)
        .expect("proof-policy prompt-value rows");
    assert_eq!(prepared.rows, 4);
    assert!(prepared.active_tokens > 0);
}

#[test]
fn proof_policy_prompt_binding_can_supervise_the_complete_deployed_action() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let batch = scheduled_ruliad_policy_batch();
    let tokenizer = burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer::from_config(
        &batch.tokenization,
    )
    .expect("tokenizer");
    let mut supervision = scheduled_ruliad_supervision();
    supervision.prompt_value_binding.context =
        crate::config::RuliadPromptValueBindingContext::ProofPolicy;
    supervision.prompt_value_binding.objective =
        crate::config::RuliadPromptValueBindingObjective::FullCompletion;
    supervision.proof_policy.prompt_context =
        crate::config::RuliadProofPolicyPromptContext::LocalActionState;
    supervision.proof_policy.scoring = crate::config::RuliadProofPolicyScoring::ResidualEnergy;
    supervision.proof_policy.gradient_scope =
        crate::config::RuliadProofPolicyGradientScope::PolicyPath;
    let learner = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        prompt_value_binding_model_config(),
        &device,
    ))
    .with_ruliad_supervision(supervision);

    let (prompt_tokens, answer) = learner
        .ruliad_prompt_value_binding_target(&batch.samples[0], &tokenizer, 1)
        .expect("proof-policy binding target");
    let (_, answer_later) = learner
        .ruliad_prompt_value_binding_target(&batch.samples[0], &tokenizer, 4)
        .expect("phase-independent deployed binding target");
    assert_eq!(answer, answer_later);
    let row = LanguageTrainModel::<TestBackend>::ruliad_prompt_full_completion_row(
        &tokenizer,
        &prompt_tokens,
        &answer,
        batch.samples[0].item.document_close_marker(),
        64,
        256,
    )
    .expect("full policy completion row");
    let decoded_targets = row
        .1
        .iter()
        .zip(&row.2)
        .filter_map(|(token, active)| (*active > 0.0).then_some(*token as u32))
        .collect::<Vec<_>>();
    let expected_targets = tokenizer.encode_payload(&format!(
        "{answer}\n{}",
        batch.samples[0].item.document_close_marker()
    ));
    assert_eq!(decoded_targets, expected_targets);
    assert!(
        tokenizer
            .decode_payload(&decoded_targets, true)
            .starts_with(&answer),
        "full-completion row must retain the semantic action"
    );
    assert_eq!(row.3, decoded_targets.len());

    let prepared = learner
        .prepare_ruliad_prompt_value_binding_batch(&batch, &device, 256, 1)
        .expect("full proof-policy completion batch");
    assert_eq!(prepared.rows, batch.samples.len());
    assert!(prepared.active_tokens >= row.3);
}

#[test]
fn prompt_schema_row_budget_is_spread_across_samples_first() {
    let groups = vec![vec!["a0", "a1"], vec!["b0", "b1"], vec!["c0", "c1"]];

    assert_eq!(
        take_rows_round_robin(&groups, 4),
        vec![(0, "a0"), (1, "b0"), (2, "c0"), (0, "a1")]
    );
}

#[test]
fn ruliad_schema_collapse_negative_answers_cover_sibling_contracts() {
    let eca_negatives = LanguageTrainModel::<TestBackend>::ruliad_schema_collapse_negative_answers(
        "xlen=14;xalpha=01;xcounts=8,6;xedge=01",
    );
    assert!(
        eca_negatives
            .iter()
            .any(|answer| answer == "xlen=14;xalpha=01;xcounts=8,6"),
        "ECA schema negatives should include tail-field omission: {eca_negatives:?}"
    );
    assert!(
        eca_negatives
            .iter()
            .any(|answer| answer == "xlen=14;nfalpha=01;nfcounts=8,6;xedge=01"),
        "ECA schema negatives should include the observed x/nf mixed-key collapse: {eca_negatives:?}"
    );
    assert!(
        eca_negatives
            .iter()
            .any(|answer| answer == "nflen=14;nfalpha=01;nfcounts=8,6;nfedge=01"),
        "ECA schema negatives should include the full sibling rewrite contract: {eca_negatives:?}"
    );

    let proof_negatives =
        LanguageTrainModel::<TestBackend>::ruliad_schema_collapse_negative_answers(
            "ok=1;l=17;r=17",
        );
    assert_eq!(
        &proof_negatives[..2],
        ["ok=1;l=17".to_string(), "ok=1".to_string()],
        "proof-specific truncation negatives should remain first"
    );
    assert!(
        proof_negatives
            .iter()
            .any(|answer| answer.starts_with("xlen=")),
        "proof negatives should include the ECA sibling contract: {proof_negatives:?}"
    );
    assert!(
        proof_negatives
            .iter()
            .any(|answer| answer.starts_with("nflen=")),
        "proof negatives should include the normal-form sibling contract: {proof_negatives:?}"
    );
    assert!(
        proof_negatives.iter().any(|answer| answer == "acc=0"),
        "proof negatives should include the acceptance sibling contract: {proof_negatives:?}"
    );
}

#[test]
fn ruliad_trim_prompt_for_completion_preserves_maximum_context() {
    let prompt = vec![10, 11, 12, 13, 14, 15, 16, 17];
    let trimmed =
        LanguageTrainModel::<TestBackend>::ruliad_trim_prompt_for_completion(&prompt, 3, 7);
    assert_eq!(trimmed, vec![14, 15, 16, 17]);

    let untrimmed =
        LanguageTrainModel::<TestBackend>::ruliad_trim_prompt_for_completion(&prompt, 2, 16);
    assert_eq!(untrimmed, prompt);

    let overlong_completion =
        LanguageTrainModel::<TestBackend>::ruliad_trim_prompt_for_completion(&prompt, 99, 7);
    assert_eq!(overlong_completion, vec![17]);
}

#[test]
fn ruliad_structured_answer_contrast_loss_supports_structured_symbolic_tokenizer() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 29);
    let mut config = tiny_model_config();
    config.vocab_size = 512;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_ruliad_supervision(RuliadSupervisionConfig {
            verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                enabled: true,
                structured_contrast_weight: 0.25,
                structured_contrast_every_steps: 1,
                structured_negative_count: 2,
                structured_template_negative_count: 1,
                max_completion_tokens: 32,
                ..Default::default()
            },
            ..Default::default()
        });
    let tokenizer = burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer::from_config(
        &burn_dragon_universality::RuliadTokenizationConfig::StructuredSymbolic {
            vocab_size: 512,
            eos_id: None,
        },
    )
    .expect("tokenizer");
    let prompt_tokens = tokenizer
        .encode_payload("?:ss\n!:")
        .into_iter()
        .map(i64::from)
        .collect::<Vec<_>>();
    let item = burn_dragon_universality::RuliadEvalItem {
        oracle_hash: "h0".to_string(),
        sample_index: 43,
        split: burn_dragon_universality::SampleSplit::Train,
        family: "formal_proof".to_string(),
        task_kind: "select_proof_action".to_string(),
        math_domains: vec!["category".to_string()],
        reasoning_modes: vec!["equational".to_string()],
        prompt: "?:ss\n!:".to_string(),
        expected_answer: "g4|a:r0|f|1.1".to_string(),
        difficulty_level: Some(0),
        spec: None,
    };
    let policy_batch = crate::dataset::RuliadPolicyBatch {
        samples: vec![crate::dataset::RuliadPolicySample {
            item,
            prompt_tokens,
        }],
        tokenization: burn_dragon_universality::RuliadTokenizationConfig::StructuredSymbolic {
            vocab_size: 512,
            eos_id: None,
        },
        stop_token_id: None,
        sampling_metadata: None,
    };

    let loss = model
        .ruliad_structured_answer_contrast_loss(&policy_batch, &device, 64)
        .expect("structured symbolic contrast loss");

    let loss = tensor_scalar(loss);
    assert!(loss.is_finite(), "contrast loss should be finite: {loss}");
    assert!(loss > 0.0, "contrast loss should be non-zero: {loss}");
}

#[test]
fn ruliad_verifier_policy_loss_can_include_structured_negative_candidates() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 19);
    let dir = tempfile::tempdir().expect("tempdir");
    let telemetry_path = dir
        .path()
        .join("events")
        .join("ruliad_verifier_policy.jsonl");
    let mut config = tiny_model_config();
    config.vocab_size = 257;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_ruliad_supervision(RuliadSupervisionConfig {
            verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                enabled: true,
                mode: crate::config::train::RuliadVerifierRewardMode::VpoIndependent,
                weight: 0.1,
                group_size: 2,
                max_completion_tokens: 24,
                every_steps: 1,
                top_k: 1,
                kl_weight: 0.0,
                vpo_scalarizations: 4,
                positive_advantage_requires_correctness: true,
                positive_advantage_min_partial_progress_ppm: 500_000,
                positive_advantage_min_completion_quality_ppm: 750_000,
                include_oracle_candidate: true,
                include_structured_negative_candidates: true,
                structured_negative_count: 2,
                ..Default::default()
            },
            ..Default::default()
        })
        .with_ruliad_policy_telemetry_path(Some(telemetry_path.clone()));
    let item = burn_dragon_universality::RuliadEvalItem {
        oracle_hash: "h0".to_string(),
        sample_index: 31,
        split: burn_dragon_universality::SampleSplit::Train,
        family: "formal_proof".to_string(),
        task_kind: "select_proof_action".to_string(),
        math_domains: vec!["category".to_string(), "formal_proof".to_string()],
        reasoning_modes: vec!["equational".to_string()],
        prompt: "?:ss\n!:".to_string(),
        expected_answer: "ok=1;l=17;r=17".to_string(),
        difficulty_level: Some(0),
        spec: None,
    };
    let policy_batch = crate::dataset::RuliadPolicyBatch {
        samples: vec![crate::dataset::RuliadPolicySample {
            item,
            prompt_tokens: vec![1, 2, 3],
        }],
        tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
            vocab_size: 257,
            eos_id: None,
        },
        stop_token_id: None,
        sampling_metadata: None,
    };
    let loss = model
        .ruliad_verifier_policy_loss(&policy_batch, &device, 48)
        .expect("structured-negative VPO verifier policy loss");
    assert!(tensor_scalar(loss).is_finite());
    let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
    let value: serde_json::Value =
        serde_json::from_str(content.lines().next().expect("telemetry line"))
            .expect("telemetry json");

    assert_eq!(value["oracle_completion_rows"], 1);
    assert_eq!(value["structured_negative_completion_rows"], 2);
    assert_eq!(value["policy_update_applied"], true);
    assert!(
        value["completion_rows"].as_u64().expect("completion rows") >= 3,
        "oracle plus structured negatives should contribute trainable policy rows"
    );
    assert!(
        value["vector_schema_quality_mean"]
            .as_f64()
            .expect("schema quality")
            > 0.0,
        "structured negatives should remain parseable enough to teach field binding"
    );
}

#[test]
fn ruliad_verifier_rollout_imitation_writes_skip_telemetry_for_wrong_generations() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 20);
    let dir = tempfile::tempdir().expect("tempdir");
    let telemetry_path = dir
        .path()
        .join("events")
        .join("ruliad_verifier_rollout_imitation.jsonl");
    let mut config = tiny_model_config();
    config.vocab_size = 257;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_ruliad_supervision(RuliadSupervisionConfig {
            verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                enabled: true,
                weight: 0.0,
                group_size: 2,
                max_completion_tokens: 8,
                top_k: 1,
                rollout_imitation_weight: 0.05,
                rollout_imitation_every_steps: 1,
                rollout_imitation_min_partial_progress_ppm: 500_000,
                rollout_imitation_min_completion_quality_ppm: 750_000,
                ..Default::default()
            },
            ..Default::default()
        })
        .with_ruliad_verifier_rollout_telemetry_path(Some(telemetry_path.clone()));
    let item = burn_dragon_universality::RuliadEvalItem {
        oracle_hash: "h0".to_string(),
        sample_index: 33,
        split: burn_dragon_universality::SampleSplit::Train,
        family: "law".to_string(),
        task_kind: "category_law".to_string(),
        math_domains: vec!["category".to_string()],
        reasoning_modes: vec!["equational".to_string()],
        prompt: "?:q\n!:".to_string(),
        expected_answer: "ok=1".to_string(),
        difficulty_level: Some(0),
        spec: None,
    };
    let policy_batch = crate::dataset::RuliadPolicyBatch {
        samples: vec![crate::dataset::RuliadPolicySample {
            item,
            prompt_tokens: vec![1, 2, 3],
        }],
        tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
            vocab_size: 257,
            eos_id: None,
        },
        stop_token_id: None,
        sampling_metadata: None,
    };

    assert!(
        model
            .ruliad_verifier_rollout_imitation_loss(&policy_batch, &device, 16)
            .is_none(),
        "wrong generated completions should not be reinforced"
    );
    let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
    let value: serde_json::Value =
        serde_json::from_str(content.lines().next().expect("telemetry line"))
            .expect("telemetry json");
    let skip_reason = value["skip_reason"].as_str();
    assert!(
        matches!(
            skip_reason,
            Some("no_candidate_completion") | Some("rollout_health_gate")
        ),
        "unexpected skip reason: {skip_reason:?}"
    );
    assert_eq!(value["accepted_completion_rows"].as_u64(), Some(0));
    assert_eq!(value["accepted_imitation_rows"].as_u64(), Some(0));
    assert_eq!(value["accepted_recovery_rows"].as_u64(), Some(0));
    let candidate_rows = value["candidate_completion_rows"]
        .as_u64()
        .expect("candidate rows");
    if skip_reason == Some("rollout_health_gate") {
        assert!(candidate_rows > 0);
    } else {
        assert_eq!(candidate_rows, 0);
    }
    assert_eq!(value["health_gate_passed"].as_bool(), Some(false));
    assert!(
        value["generated_completion_rows"]
            .as_u64()
            .expect("generated rows")
            > 0
    );
}

#[test]
fn ruliad_verifier_rollout_recovery_accepts_generated_malformed_prefixes() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 23);
    let dir = tempfile::tempdir().expect("tempdir");
    let telemetry_path = dir
        .path()
        .join("events")
        .join("ruliad_verifier_rollout_recovery.jsonl");
    let mut config = tiny_model_config();
    config.vocab_size = 257;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_ruliad_supervision(RuliadSupervisionConfig {
            verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                enabled: true,
                weight: 0.0,
                group_size: 1,
                max_completion_tokens: 1,
                top_k: 1,
                rollout_recovery_weight: 0.05,
                rollout_imitation_weight: 0.0,
                rollout_imitation_every_steps: 1,
                rollout_imitation_min_partial_progress_ppm: 0,
                rollout_imitation_min_completion_quality_ppm: 0,
                rollout_imitation_max_rows_per_step: 1,
                ..Default::default()
            },
            ..Default::default()
        })
        .with_ruliad_verifier_rollout_telemetry_path(Some(telemetry_path.clone()));
    let item = burn_dragon_universality::RuliadEvalItem {
        oracle_hash: "h0".to_string(),
        sample_index: 37,
        split: burn_dragon_universality::SampleSplit::Train,
        family: "rewrite".to_string(),
        task_kind: "rewrite_normal_form".to_string(),
        math_domains: vec!["symbolic_rewriting".to_string()],
        reasoning_modes: vec!["normalization".to_string()],
        prompt: "?:q\n!:".to_string(),
        expected_answer: "nflen=3;nfalpha=ABC;nfcounts=1,1,1;nfedge=AB".to_string(),
        difficulty_level: Some(0),
        spec: None,
    };
    let policy_batch = crate::dataset::RuliadPolicyBatch {
        samples: vec![crate::dataset::RuliadPolicySample {
            item,
            prompt_tokens: vec![1, 2, 3],
        }],
        tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
            vocab_size: 257,
            eos_id: None,
        },
        stop_token_id: None,
        sampling_metadata: None,
    };

    let loss = model
        .ruliad_verifier_rollout_imitation_loss(&policy_batch, &device, 16)
        .expect("malformed rollout should create an oracle recovery row");
    assert!(tensor_scalar(loss).is_finite());
    let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
    let value: serde_json::Value =
        serde_json::from_str(content.lines().next().expect("telemetry line"))
            .expect("telemetry json");
    assert_eq!(value["accepted_imitation_rows"].as_u64(), Some(0));
    assert_eq!(value["accepted_recovery_rows"].as_u64(), Some(1));
    let malformed = value["recovery_malformed_rows"]
        .as_u64()
        .unwrap_or_default();
    let missing = value["recovery_missing_rows"].as_u64().unwrap_or_default();
    let schema_wrong = value["recovery_schema_wrong_rows"]
        .as_u64()
        .unwrap_or_default();
    let partial = value["recovery_partial_rows"].as_u64().unwrap_or_default();
    assert_eq!(malformed + missing + schema_wrong + partial, 1);
}

#[test]
fn ruliad_proof_policy_masks_only_the_action_bearing_token() {
    let prompt = [1, 2, 3];
    let completion = [4, 5, 6, 7];
    let (_, targets, mask) =
        LanguageTrainModel::<TestBackend>::ruliad_policy_row_from_completion_token(
            &prompt,
            &completion,
            2,
        )
        .expect("action-token policy row");
    assert_eq!(mask.iter().filter(|value| **value > 0.0).count(), 1);
    assert_eq!(mask[4], 1.0);
    assert_eq!(targets[4], 6);
}

#[test]
fn verifier_equivalent_action_loss_marginalizes_all_valid_tokens() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let logits = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![0.0, 0.0, 0.0, 0.0], [1, 1, 4]),
        &device,
    );
    let one_valid = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![1.0, 0.0, 0.0, 0.0], [1, 1, 4]),
        &device,
    );
    let two_valid = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![1.0, 1.0, 0.0, 0.0], [1, 1, 4]),
        &device,
    );
    let candidates = Tensor::<TestBackend, 3>::ones([1, 1, 4], &device);

    let one_loss = tensor_scalar(verifier_equivalent_action_loss(
        logits.clone(),
        candidates.clone(),
        one_valid,
        crate::config::RuliadProofPolicyNormalization::CandidateConditional,
        1.0,
    ));
    let two_loss = tensor_scalar(verifier_equivalent_action_loss(
        logits,
        candidates,
        two_valid,
        crate::config::RuliadProofPolicyNormalization::CandidateConditional,
        1.0,
    ));
    assert!((one_loss - 4.0f32.ln()).abs() < 1.0e-5, "{one_loss}");
    assert!((two_loss - 2.0f32.ln()).abs() < 1.0e-5, "{two_loss}");
    assert!(two_loss < one_loss);
}

#[test]
fn verifier_equivalent_action_loss_ignores_non_candidate_logits() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let candidate_mask = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![1.0, 1.0, 0.0], [1, 1, 3]),
        &device,
    );
    let equivalent_mask = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![1.0, 0.0, 0.0], [1, 1, 3]),
        &device,
    );
    let baseline = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![0.0, 0.0, 0.0], [1, 1, 3]),
        &device,
    );
    let dominant_non_candidate = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![0.0, 0.0, 20.0], [1, 1, 3]),
        &device,
    );

    let baseline_loss = tensor_scalar(verifier_equivalent_action_loss(
        baseline,
        candidate_mask.clone(),
        equivalent_mask.clone(),
        crate::config::RuliadProofPolicyNormalization::CandidateConditional,
        1.0,
    ));
    let perturbed_loss = tensor_scalar(verifier_equivalent_action_loss(
        dominant_non_candidate,
        candidate_mask,
        equivalent_mask,
        crate::config::RuliadProofPolicyNormalization::CandidateConditional,
        1.0,
    ));
    assert!((baseline_loss - 2.0f32.ln()).abs() < 1.0e-5);
    assert!((perturbed_loss - baseline_loss).abs() < 1.0e-5);
}

#[test]
fn vocabulary_marginal_action_loss_penalizes_non_candidate_probability() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let candidate_mask = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![1.0, 1.0, 0.0], [1, 1, 3]),
        &device,
    );
    let equivalent_mask = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![1.0, 0.0, 0.0], [1, 1, 3]),
        &device,
    );
    let baseline = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![0.0, 0.0, 0.0], [1, 1, 3]),
        &device,
    );
    let dominant_non_candidate = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![0.0, 0.0, 20.0], [1, 1, 3]),
        &device,
    );

    let baseline_loss = tensor_scalar(verifier_equivalent_action_loss(
        baseline,
        candidate_mask.clone(),
        equivalent_mask.clone(),
        crate::config::RuliadProofPolicyNormalization::VocabularyMarginal,
        1.0,
    ));
    let perturbed_loss = tensor_scalar(verifier_equivalent_action_loss(
        dominant_non_candidate,
        candidate_mask,
        equivalent_mask,
        crate::config::RuliadProofPolicyNormalization::VocabularyMarginal,
        1.0,
    ));
    assert!((baseline_loss - 3.0f32.ln()).abs() < 1.0e-5);
    assert!(perturbed_loss > baseline_loss + 10.0, "{perturbed_loss}");
}

#[test]
fn semantic_sequence_policy_loss_marginalizes_verifier_equivalent_actions() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let scores = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(vec![0.4f32.ln(), 0.1f32.ln(), 0.1f32.ln()], [1, 3]),
        &device,
    );
    let equivalent =
        Tensor::<TestBackend, 2>::from_data(TensorData::new(vec![1.0, 0.0, 1.0], [1, 3]), &device);
    let weights = Tensor::<TestBackend, 1>::ones([1], &device);
    let support = Tensor::<TestBackend, 2>::ones([1, 3], &device);
    let conditional = tensor_scalar(grouped_verifier_equivalent_sequence_loss(
        scores.clone(),
        scores.clone(),
        support.clone(),
        equivalent.clone(),
        weights.clone(),
        GroupedVerifierSequenceLossConfig {
            normalization: crate::config::RuliadProofPolicyNormalization::CandidateConditional,
            presentation_risk: crate::config::RuliadProofPolicyPresentationRisk::Mean,
            presentation_group_size: 1,
            weight: 1.0,
        },
    ));
    let marginal = tensor_scalar(grouped_verifier_equivalent_sequence_loss(
        scores.clone(),
        scores,
        support,
        equivalent,
        weights,
        GroupedVerifierSequenceLossConfig {
            normalization: crate::config::RuliadProofPolicyNormalization::VocabularyMarginal,
            presentation_risk: crate::config::RuliadProofPolicyPresentationRisk::Mean,
            presentation_group_size: 1,
            weight: 1.0,
        },
    ));
    let expected_conditional = -(5.0f32 / 6.0).ln();
    let expected_marginal = -0.5f32.ln();
    assert!(
        (conditional - expected_conditional).abs() < 1.0e-5,
        "conditional={conditional}"
    );
    assert!(
        (marginal - expected_marginal).abs() < 1.0e-5,
        "marginal={marginal}"
    );
    assert!(marginal > conditional);
}

#[test]
fn target_group_conditional_loss_rewards_prompt_dependent_target_reversal() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let support = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], [2, 4]),
        &device,
    );
    let equivalent = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [2, 4]),
        &device,
    );
    let weights = Tensor::<TestBackend, 1>::ones([2], &device);
    let config = GroupedVerifierSequenceLossConfig {
        normalization: crate::config::RuliadProofPolicyNormalization::CandidateConditional,
        presentation_risk: crate::config::RuliadProofPolicyPresentationRisk::Mean,
        presentation_group_size: 1,
        weight: 1.0,
    };
    let target_independent = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(vec![2.0, 0.0, -4.0, -4.0, 2.0, 0.0, -4.0, -4.0], [2, 4]),
        &device,
    );
    let conditioned = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(vec![2.0, 0.0, -4.0, -4.0, 0.0, 2.0, -4.0, -4.0], [2, 4]),
        &device,
    );
    let prior_loss = tensor_scalar(grouped_verifier_equivalent_sequence_loss(
        target_independent.clone(),
        target_independent,
        support.clone(),
        equivalent.clone(),
        weights.clone(),
        config,
    ));
    let conditioned_loss = tensor_scalar(grouped_verifier_equivalent_sequence_loss(
        conditioned.clone(),
        conditioned,
        support,
        equivalent,
        weights,
        config,
    ));
    assert!(prior_loss > std::f32::consts::LN_2, "{prior_loss}");
    assert!(conditioned_loss < 0.2, "{conditioned_loss}");
    assert!(conditioned_loss < prior_loss - 0.8);
}

#[test]
fn worst_presentation_risk_targets_each_groups_weakest_orbit_member() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let logits = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(
            vec![
                0.9f32.ln(),
                0.1f32.ln(),
                0.6f32.ln(),
                0.4f32.ln(),
                0.8f32.ln(),
                0.2f32.ln(),
                0.2f32.ln(),
                0.8f32.ln(),
            ],
            [4, 1, 2],
        ),
        &device,
    );
    let candidates = Tensor::<TestBackend, 3>::ones([4, 1, 2], &device);
    let equivalent = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [4, 1, 2]),
        &device,
    );
    let row_weights =
        Tensor::<TestBackend, 1>::from_data(TensorData::new(vec![0.5; 4], [4]), &device);
    let mean = tensor_scalar(grouped_verifier_equivalent_action_loss(
        logits.clone(),
        candidates.clone(),
        equivalent.clone(),
        row_weights.clone(),
        crate::config::RuliadProofPolicyNormalization::VocabularyMarginal,
        crate::config::RuliadProofPolicyPresentationRisk::Mean,
        2,
        1.0,
    ));
    let worst = tensor_scalar(grouped_verifier_equivalent_action_loss(
        logits,
        candidates,
        equivalent,
        row_weights,
        crate::config::RuliadProofPolicyNormalization::VocabularyMarginal,
        crate::config::RuliadProofPolicyPresentationRisk::Worst,
        2,
        1.0,
    ));

    let expected_worst = -(0.6f32.ln() + 0.2f32.ln()) / 2.0;
    assert!((worst - expected_worst).abs() < 1.0e-5, "{worst}");
    assert!(worst > mean, "mean={mean} worst={worst}");
}

#[test]
fn ruliad_proof_policy_dagger_labels_model_visited_state_with_expert_action() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 29);
    let dir = tempfile::tempdir().expect("tempdir");
    let telemetry_path = dir
        .path()
        .join("events")
        .join("ruliad_proof_policy_dagger.jsonl");
    let mut model_config = tiny_model_config();
    model_config.vocab_size = 272;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(model_config, &device))
        .with_ruliad_supervision(RuliadSupervisionConfig {
            proof_policy: crate::config::RuliadProofPolicyTrainingConfig {
                enabled: true,
                require_scheduled_update: false,
                decoder_calibration_steps: 0,
                mode: crate::config::RuliadProofPolicyTrainingMode::Dagger,
                scoring: crate::config::RuliadProofPolicyScoring::CompletionLikelihood,
                prompt_context: crate::config::RuliadProofPolicyPromptContext::FullProblemSuffix,
                target: crate::config::RuliadProofPolicyTarget::ExpertSet,
                gradient_scope: crate::config::RuliadProofPolicyGradientScope::FullModel,
                normalization: crate::config::RuliadProofPolicyNormalization::VocabularyMarginal,
                candidate_symmetry:
                    crate::config::RuliadProofPolicyCandidateSymmetry::BalancedRotation,
                presentation_risk: crate::config::RuliadProofPolicyPresentationRisk::Mean,
                weight: 1.0,
                every_steps: 1,
                start_after_steps: 0,
                dagger_start_after_steps: 1,
                stratified_difficulty_levels: 0,
                rollout_steps: 2,
                max_rows_per_update: 2,
                max_presentation_rows_per_update: 32,
                counterfactual_targets_per_state: 0,
                counterfactual_objective:
                    crate::config::RuliadProofPolicyCounterfactualObjective::Independent,
                candidates: 4,
                max_completion_tokens: 16,
            },
            ..Default::default()
        })
        .with_ruliad_proof_policy_telemetry_path(Some(telemetry_path.clone()));
    let bundle = burn_dragon_universality::ruliad::formal::generate_formal_bundle(
        29,
        burn_dragon_universality::ruliad::formal::RuliadFormalGeneratorConfig {
            rewrite_depth: 2,
            leaf_count: 3,
            context_depth: 1,
            distractor_axioms: 1,
            ..Default::default()
        },
    )
    .expect("formal bundle");
    let proof_step_index = 1.min(bundle.certificate.step_count().saturating_sub(1));
    assert!(proof_step_index > 0, "fixture needs a nonzero proof step");
    let actions = burn_dragon_universality::ruliad::oracle_proof_action_set(
        &bundle.problem,
        &bundle.certificate,
        proof_step_index,
        4,
    )
    .expect("oracle action set");
    let problem_hash = bundle.problem.canonical_hash().expect("problem hash");
    let item = burn_dragon_universality::RuliadEvalItem {
        oracle_hash: problem_hash,
        sample_index: 29,
        split: burn_dragon_universality::SampleSplit::Train,
        family: "formal_proof".to_string(),
        task_kind: burn_dragon_universality::RuliadTaskKind::SelectProofAction
            .label()
            .to_string(),
        math_domains: vec!["formal_proof".to_string()],
        reasoning_modes: vec!["proof_construction".to_string()],
        prompt: burn_dragon_universality::ruliad::ruliad_proof_action_prompt(
            &bundle.problem,
            &actions,
        )
        .expect("policy prompt"),
        expected_answer: format!("c={}", actions.selected_index),
        difficulty_level: Some(0),
        spec: Some(burn_dragon_universality::RuliadSampleSpec::FormalProof {
            problem: bundle.problem,
            certificate: bundle.certificate,
            candidate: None,
            proof_step_index: Some(proof_step_index),
            action_presentation_rotation: Some(0),
            action_candidate_count: Some(actions.candidates.len()),
            action_answer_contract: Default::default(),
            task: burn_dragon_universality::RuliadTaskKind::SelectProofAction,
        }),
    };
    let mut policy_batch = crate::dataset::RuliadPolicyBatch {
        samples: vec![crate::dataset::RuliadPolicySample {
            item,
            prompt_tokens: vec![1],
        }],
        tokenization: burn_dragon_universality::RuliadTokenizationConfig::StructuredSymbolic {
            vocab_size: 272,
            eos_id: Some(271),
        },
        stop_token_id: Some(271),
        sampling_metadata: None,
    };
    policy_batch.samples.push(policy_batch.samples[0].clone());

    let objective = model
        .ruliad_proof_policy_objective(&policy_batch, &device, 512)
        .expect("DAgger expert correction loss");
    assert!(objective.semantic_states > 0);
    assert!(objective.decision_rows > 0);
    assert!(objective.padded_tokens >= objective.decision_rows);
    assert!(tensor_scalar(objective.loss).is_finite());
    let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
    let value: serde_json::Value =
        serde_json::from_str(content.lines().next().expect("telemetry line"))
            .expect("telemetry json");
    assert_eq!(value["version"], RULIAD_PROOF_POLICY_TELEMETRY_VERSION);
    assert_eq!(value["answer_contract"], "presentation_index");
    assert_eq!(value["objective"], "vocabulary_marginal_equivalent_v1");
    assert_eq!(value["target"], "expert_set");
    assert_eq!(value["presentation_risk"], "mean");
    assert_eq!(value["configured_mode"], "dagger");
    assert_eq!(value["mode"], "dagger");
    assert_eq!(value["candidate_symmetry"], "balanced_rotation");
    assert_eq!(value["available_sample_groups"], 2);
    assert_eq!(value["sample_groups"], 1);
    assert_eq!(value["nonzero_start_trajectories"], 1);
    assert_eq!(value["mean_start_step"], proof_step_index as f64);
    assert!(value["visited_states"].as_u64().unwrap_or_default() >= 1);
    assert_eq!(value["semantic_state_rows"], value["expert_rows"]);
    assert!(value["expert_rows"].as_u64().unwrap_or_default() >= 1);
    assert_eq!(value["static_expert_rows"], 0);
    assert!(value["dagger_expert_rows"].as_u64().unwrap_or_default() >= 1);
    assert_eq!(value["supervised_action_tokens"], value["expert_rows"]);
    assert_eq!(value["supervised_presentation_rows"], value["expert_rows"]);
    assert_eq!(value["mean_presentations_per_state"], 1.0);
    assert!(value["model_scoring_batches"].as_u64().unwrap_or_default() >= 1);
    assert_eq!(value["maximum_model_scoring_batch_rows"], 1);
    assert!(
        value["model_scoring_padded_tokens"]
            .as_u64()
            .unwrap_or_default()
            > 0
    );
    assert!(value["sampling_model_materialize_ms"].is_number());
    assert!(value["state_prepare_ms"].is_number());
    assert!(value["rollout_cpu_prepare_ms"].is_number());
    assert!(value["model_scoring_ms"].is_number());
    assert_eq!(value["trajectory_budget"], 1);
    assert_eq!(value["semantic_row_budget"], 2);
    assert_eq!(value["max_rows_per_update"], 2);
    assert_eq!(value["max_presentation_rows_per_update"], 32);
    assert!(value["rollout_depth_reached"].as_u64().unwrap_or_default() >= 2);
    assert!(
        value["model_visited_expert_rows"]
            .as_u64()
            .unwrap_or_default()
            >= 1
    );
    assert!(
        value["equivalent_target_tokens"]
            .as_u64()
            .unwrap_or_default()
            >= value["expert_rows"].as_u64().unwrap_or_default()
    );
    assert!(
        value["candidate_target_tokens"]
            .as_u64()
            .unwrap_or_default()
            >= value["equivalent_target_tokens"]
                .as_u64()
                .unwrap_or_default()
    );
    assert!(
        value["mean_candidate_targets_per_row"]
            .as_f64()
            .unwrap_or_default()
            >= value["mean_equivalent_targets_per_row"]
                .as_f64()
                .unwrap_or_default()
    );
    assert!(
        value["mean_equivalent_targets_per_row"]
            .as_f64()
            .unwrap_or_default()
            >= 1.0
    );
    assert!(value["expert_selected_index_histogram"].is_object());
    assert!(value["expert_equivalent_index_histogram"].is_object());
    assert!(value["model_selected_index_histogram"].is_object());
    assert_eq!(value["difficulty_sample_groups"]["0"], 1);
    assert!(
        value["difficulty_visited_states"]["0"]
            .as_u64()
            .unwrap_or_default()
            >= 1
    );

    let static_telemetry_path = dir
        .path()
        .join("events")
        .join("ruliad_proof_policy_static.jsonl");
    let mut static_model_config = tiny_model_config();
    static_model_config.vocab_size = 272;
    let static_model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        static_model_config,
        &device,
    ))
    .with_ruliad_supervision(RuliadSupervisionConfig {
        proof_policy: crate::config::RuliadProofPolicyTrainingConfig {
            enabled: true,
            require_scheduled_update: false,
            decoder_calibration_steps: 0,
            mode: crate::config::RuliadProofPolicyTrainingMode::StaticExpert,
            scoring: crate::config::RuliadProofPolicyScoring::CompletionLikelihood,
            prompt_context: crate::config::RuliadProofPolicyPromptContext::FullProblemSuffix,
            target: crate::config::RuliadProofPolicyTarget::ExpertSet,
            gradient_scope: crate::config::RuliadProofPolicyGradientScope::FullModel,
            normalization: crate::config::RuliadProofPolicyNormalization::CandidateConditional,
            candidate_symmetry:
                crate::config::RuliadProofPolicyCandidateSymmetry::CyclicOrbitAverage,
            presentation_risk: crate::config::RuliadProofPolicyPresentationRisk::Mean,
            weight: 1.0,
            every_steps: 1,
            start_after_steps: 0,
            dagger_start_after_steps: 1,
            stratified_difficulty_levels: 0,
            rollout_steps: 8,
            max_rows_per_update: 2,
            max_presentation_rows_per_update: 8,
            counterfactual_targets_per_state: 0,
            counterfactual_objective:
                crate::config::RuliadProofPolicyCounterfactualObjective::Independent,
            candidates: 4,
            max_completion_tokens: 16,
        },
        ..Default::default()
    })
    .with_ruliad_proof_policy_telemetry_path(Some(static_telemetry_path.clone()));
    let static_loss = static_model
        .ruliad_proof_policy_dagger_loss(&policy_batch, &device, 512)
        .expect("static expert policy loss");
    assert!(tensor_scalar(static_loss).is_finite());
    let static_content =
        std::fs::read_to_string(static_telemetry_path).expect("static telemetry sidecar");
    let static_value: serde_json::Value = serde_json::from_str(
        static_content
            .lines()
            .next()
            .expect("static telemetry line"),
    )
    .expect("static telemetry json");
    assert_eq!(
        static_value["version"],
        RULIAD_PROOF_POLICY_TELEMETRY_VERSION
    );
    assert_eq!(static_value["answer_contract"], "presentation_index");
    assert_eq!(static_value["presentation_risk"], "mean");
    assert_eq!(static_value["configured_mode"], "static_expert");
    assert_eq!(static_value["mode"], "static_expert");
    assert_eq!(static_value["candidate_symmetry"], "cyclic_orbit_average");
    assert_eq!(static_value["rollout_steps"], 1);
    assert_eq!(static_value["configured_rollout_steps"], 8);
    assert_eq!(static_value["model_scoring_batches"], 0);
    assert_eq!(static_value["semantic_row_budget"], 2);
    assert_eq!(static_value["max_presentation_rows_per_update"], 8);
    assert!(static_value["expert_rows"].as_u64().unwrap_or_default() >= 1);
    assert!(
        static_value["static_expert_rows"]
            .as_u64()
            .unwrap_or_default()
            >= 1
    );
    assert_eq!(static_value["dagger_expert_rows"], 0);
    assert!(
        static_value["supervised_presentation_rows"]
            .as_u64()
            .unwrap_or_default()
            >= static_value["expert_rows"]
                .as_u64()
                .unwrap_or_default()
                .saturating_mul(2)
    );
    assert!(
        static_value["mean_presentations_per_state"]
            .as_f64()
            .unwrap_or_default()
            >= 2.0
    );
    assert!(
        static_value["supervised_presentation_rows"]
            .as_u64()
            .unwrap_or_default()
            <= 8
    );

    let semantic_telemetry_path = dir
        .path()
        .join("events")
        .join("ruliad_proof_policy_semantic.jsonl");
    let mut semantic_batch = policy_batch.clone();
    for sample in &mut semantic_batch.samples {
        let Some(burn_dragon_universality::RuliadSampleSpec::FormalProof {
            action_answer_contract,
            ..
        }) = sample.item.spec.as_mut()
        else {
            panic!("formal proof fixture");
        };
        *action_answer_contract =
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep;
    }
    let mut semantic_model_config = tiny_model_config();
    semantic_model_config.vocab_size = 272;
    let semantic_model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        semantic_model_config,
        &device,
    ))
    .with_ruliad_supervision(RuliadSupervisionConfig {
        proof_policy: crate::config::RuliadProofPolicyTrainingConfig {
            enabled: true,
            require_scheduled_update: false,
            decoder_calibration_steps: 0,
            mode: crate::config::RuliadProofPolicyTrainingMode::StaticExpert,
            scoring: crate::config::RuliadProofPolicyScoring::CompletionLikelihood,
            prompt_context: crate::config::RuliadProofPolicyPromptContext::FullProblemSuffix,
            target: crate::config::RuliadProofPolicyTarget::ExpertSet,
            gradient_scope: crate::config::RuliadProofPolicyGradientScope::FullModel,
            normalization: crate::config::RuliadProofPolicyNormalization::CandidateConditional,
            candidate_symmetry:
                crate::config::RuliadProofPolicyCandidateSymmetry::CyclicOrbitAverage,
            presentation_risk: crate::config::RuliadProofPolicyPresentationRisk::Worst,
            weight: 1.0,
            every_steps: 1,
            start_after_steps: 0,
            dagger_start_after_steps: 1,
            stratified_difficulty_levels: 0,
            rollout_steps: 1,
            max_rows_per_update: 1,
            max_presentation_rows_per_update: 8,
            counterfactual_targets_per_state: 0,
            counterfactual_objective:
                crate::config::RuliadProofPolicyCounterfactualObjective::Independent,
            candidates: 4,
            max_completion_tokens: 128,
        },
        ..Default::default()
    })
    .with_ruliad_proof_policy_telemetry_path(Some(semantic_telemetry_path.clone()));
    let semantic_loss = semantic_model
        .ruliad_proof_policy_dagger_loss(&semantic_batch, &device, 512)
        .expect("semantic proof-step policy loss");
    assert!(tensor_scalar(semantic_loss.clone()).is_finite());
    let _semantic_gradients = semantic_loss.backward();
    let semantic_content =
        std::fs::read_to_string(semantic_telemetry_path).expect("semantic telemetry sidecar");
    let semantic_value: serde_json::Value = serde_json::from_str(
        semantic_content
            .lines()
            .next()
            .expect("semantic telemetry line"),
    )
    .expect("semantic telemetry json");
    assert_eq!(
        semantic_value["version"],
        RULIAD_PROOF_POLICY_TELEMETRY_VERSION
    );
    assert_eq!(semantic_value["answer_contract"], "semantic_step");
    assert_eq!(semantic_value["presentation_risk"], "worst");
    assert!(
        semantic_value["supervised_action_tokens"]
            .as_u64()
            .unwrap_or_default()
            > semantic_value["supervised_presentation_rows"]
                .as_u64()
                .unwrap_or_default()
    );

    let energy_telemetry_path = dir
        .path()
        .join("events")
        .join("ruliad_proof_policy_semantic_energy.jsonl");
    let mut energy_model_config = tiny_model_config();
    energy_model_config.vocab_size = 272;
    energy_model_config.sequence_score_head.enabled = true;
    let energy_model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        energy_model_config,
        &device,
    ))
    .with_ruliad_supervision(RuliadSupervisionConfig {
        proof_policy: crate::config::RuliadProofPolicyTrainingConfig {
            enabled: true,
            require_scheduled_update: false,
            decoder_calibration_steps: 0,
            mode: crate::config::RuliadProofPolicyTrainingMode::StaticExpert,
            scoring: crate::config::RuliadProofPolicyScoring::SemanticEnergy,
            prompt_context: crate::config::RuliadProofPolicyPromptContext::FullProblemSuffix,
            target: crate::config::RuliadProofPolicyTarget::ExpertSet,
            gradient_scope: crate::config::RuliadProofPolicyGradientScope::FullModel,
            normalization: crate::config::RuliadProofPolicyNormalization::CandidateConditional,
            candidate_symmetry: crate::config::RuliadProofPolicyCandidateSymmetry::BalancedRotation,
            presentation_risk: crate::config::RuliadProofPolicyPresentationRisk::Mean,
            weight: 1.0,
            every_steps: 1,
            start_after_steps: 0,
            dagger_start_after_steps: 1,
            stratified_difficulty_levels: 0,
            rollout_steps: 1,
            max_rows_per_update: 2,
            max_presentation_rows_per_update: 2,
            counterfactual_targets_per_state: 1,
            counterfactual_objective:
                crate::config::RuliadProofPolicyCounterfactualObjective::Independent,
            candidates: 4,
            max_completion_tokens: 128,
        },
        ..Default::default()
    })
    .with_ruliad_proof_policy_telemetry_path(Some(energy_telemetry_path.clone()));
    let energy_loss = energy_model
        .ruliad_proof_policy_dagger_loss(&policy_batch, &device, 512)
        .expect("semantic-energy proof policy loss");
    assert!(tensor_scalar(energy_loss.clone()).is_finite());
    let _energy_gradients = energy_loss.backward();
    let energy_content =
        std::fs::read_to_string(energy_telemetry_path).expect("energy telemetry sidecar");
    let energy_value: serde_json::Value = serde_json::from_str(
        energy_content
            .lines()
            .next()
            .expect("energy telemetry line"),
    )
    .expect("energy telemetry json");
    assert_eq!(
        energy_value["version"],
        RULIAD_PROOF_POLICY_TELEMETRY_VERSION
    );
    assert_eq!(energy_value["answer_contract"], "semantic_step");
    assert_eq!(energy_value["gradient_scope"], "full_model");
    assert_eq!(energy_value["target"], "expert_set");
    assert_eq!(
        energy_value["objective"],
        "semantic_sequence_energy_counterfactual_v1"
    );
    assert_eq!(
        energy_value["configured_counterfactual_targets_per_state"],
        1
    );
    assert_eq!(energy_value["target_variants_per_state"], 2);
    assert_eq!(energy_value["base_semantic_row_budget"], 1);
    assert_eq!(energy_value["base_semantic_state_rows"], 1);
    assert_eq!(energy_value["counterfactual_semantic_state_rows"], 1);
    assert_eq!(energy_value["counterfactual_target_shortfall"], 0);
    assert_eq!(energy_value["semantic_state_rows"], 2);

    let language_head_telemetry_path = dir
        .path()
        .join("events")
        .join("ruliad_proof_policy_language_head.jsonl");
    let mut language_head_model_config = tiny_model_config();
    language_head_model_config.vocab_size = 272;
    language_head_model_config.tie_input_output_embeddings = false;
    let language_head_model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        language_head_model_config,
        &device,
    ))
    .with_ruliad_supervision(RuliadSupervisionConfig {
        proof_policy: crate::config::RuliadProofPolicyTrainingConfig {
            enabled: true,
            require_scheduled_update: false,
            decoder_calibration_steps: 0,
            mode: crate::config::RuliadProofPolicyTrainingMode::StaticExpert,
            scoring: crate::config::RuliadProofPolicyScoring::CompletionLikelihood,
            prompt_context: crate::config::RuliadProofPolicyPromptContext::FullProblemSuffix,
            target: crate::config::RuliadProofPolicyTarget::ExpertSet,
            gradient_scope: crate::config::RuliadProofPolicyGradientScope::LanguageHeadOnly,
            normalization: crate::config::RuliadProofPolicyNormalization::CandidateConditional,
            candidate_symmetry: crate::config::RuliadProofPolicyCandidateSymmetry::BalancedRotation,
            presentation_risk: crate::config::RuliadProofPolicyPresentationRisk::Mean,
            weight: 1.0,
            every_steps: 1,
            start_after_steps: 0,
            dagger_start_after_steps: 1,
            stratified_difficulty_levels: 0,
            rollout_steps: 1,
            max_rows_per_update: 2,
            max_presentation_rows_per_update: 2,
            counterfactual_targets_per_state: 1,
            counterfactual_objective:
                crate::config::RuliadProofPolicyCounterfactualObjective::TargetGroupConditional,
            candidates: 4,
            max_completion_tokens: 128,
        },
        ..Default::default()
    })
    .with_ruliad_proof_policy_telemetry_path(Some(language_head_telemetry_path.clone()));
    let language_head_loss = language_head_model
        .ruliad_proof_policy_dagger_loss(&semantic_batch, &device, 512)
        .expect("language-head-only counterfactual proof policy loss");
    assert!(tensor_scalar(language_head_loss.clone()).is_finite());
    let _language_head_gradients = language_head_loss.backward();
    let language_head_content = std::fs::read_to_string(language_head_telemetry_path)
        .expect("language-head telemetry sidecar");
    let language_head_value: serde_json::Value = serde_json::from_str(
        language_head_content
            .lines()
            .next()
            .expect("language-head telemetry line"),
    )
    .expect("language-head telemetry json");
    assert_eq!(
        language_head_value["version"],
        RULIAD_PROOF_POLICY_TELEMETRY_VERSION
    );
    assert_eq!(language_head_value["answer_contract"], "semantic_step");
    assert_eq!(language_head_value["gradient_scope"], "language_head_only");
    assert_eq!(
        language_head_value["objective"],
        "completion_target_group_conditional_v1"
    );
    assert_eq!(
        language_head_value["counterfactual_objective"],
        "target_group_conditional"
    );
    assert_eq!(
        language_head_value["configured_counterfactual_targets_per_state"],
        1
    );
    assert_eq!(language_head_value["target_variants_per_state"], 2);
    assert_eq!(language_head_value["target_group_conditional_groups"], 1);
    assert_eq!(language_head_value["target_group_conditional_rows"], 2);
    assert_eq!(language_head_value["base_semantic_state_rows"], 1);
    assert_eq!(language_head_value["counterfactual_semantic_state_rows"], 1);
    assert_eq!(language_head_value["counterfactual_target_shortfall"], 0);

    let prefix_telemetry_path = dir
        .path()
        .join("events")
        .join("ruliad_proof_policy_semantic_prefix.jsonl");
    let mut prefix_model_config = tiny_model_config();
    prefix_model_config.vocab_size = 272;
    let prefix_model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        prefix_model_config,
        &device,
    ))
    .with_ruliad_supervision(RuliadSupervisionConfig {
        proof_policy: crate::config::RuliadProofPolicyTrainingConfig {
            enabled: true,
            require_scheduled_update: false,
            decoder_calibration_steps: 0,
            mode: crate::config::RuliadProofPolicyTrainingMode::StaticExpert,
            scoring: crate::config::RuliadProofPolicyScoring::CompletionLikelihood,
            prompt_context: crate::config::RuliadProofPolicyPromptContext::FullProblemSuffix,
            target: crate::config::RuliadProofPolicyTarget::ExpertSet,
            gradient_scope: crate::config::RuliadProofPolicyGradientScope::FullModel,
            normalization: crate::config::RuliadProofPolicyNormalization::PrefixConditional,
            candidate_symmetry: crate::config::RuliadProofPolicyCandidateSymmetry::BalancedRotation,
            presentation_risk: crate::config::RuliadProofPolicyPresentationRisk::Mean,
            weight: 1.0,
            every_steps: 1,
            start_after_steps: 0,
            dagger_start_after_steps: 1,
            stratified_difficulty_levels: 0,
            rollout_steps: 1,
            max_rows_per_update: 2,
            max_presentation_rows_per_update: 2,
            counterfactual_targets_per_state: 0,
            counterfactual_objective:
                crate::config::RuliadProofPolicyCounterfactualObjective::Independent,
            candidates: 4,
            max_completion_tokens: 128,
        },
        ..Default::default()
    })
    .with_ruliad_proof_policy_telemetry_path(Some(prefix_telemetry_path.clone()));
    let prefix_loss = prefix_model
        .ruliad_proof_policy_dagger_loss(&semantic_batch, &device, 512)
        .expect("semantic prefix policy loss");
    assert!(tensor_scalar(prefix_loss.clone()).is_finite());
    let _gradients = prefix_loss.backward();
    let prefix_content =
        std::fs::read_to_string(prefix_telemetry_path).expect("prefix telemetry sidecar");
    let prefix_value: serde_json::Value = serde_json::from_str(
        prefix_content
            .lines()
            .next()
            .expect("prefix telemetry line"),
    )
    .expect("prefix telemetry json");
    assert_eq!(
        prefix_value["version"],
        RULIAD_PROOF_POLICY_TELEMETRY_VERSION
    );
    assert_eq!(prefix_value["answer_contract"], "semantic_step");
    assert_eq!(
        prefix_value["objective"],
        "prefix_conditional_equivalent_v1"
    );
    assert!(
        prefix_value["prefix_branch_rows"]
            .as_u64()
            .unwrap_or_default()
            > 0
    );
    assert!(
        prefix_value["prefix_candidate_tokens"]
            .as_u64()
            .unwrap_or_default()
            > prefix_value["prefix_equivalent_tokens"]
                .as_u64()
                .unwrap_or_default()
    );
}

#[test]
fn ruliad_proof_policy_dagger_accepts_every_production_cadence_panel() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 20260831);
    let proof_policy = crate::config::RuliadProofPolicyTrainingConfig {
        enabled: true,
        require_scheduled_update: false,
        decoder_calibration_steps: 0,
        mode: crate::config::RuliadProofPolicyTrainingMode::Dagger,
        scoring: crate::config::RuliadProofPolicyScoring::CompletionLikelihood,
        prompt_context: crate::config::RuliadProofPolicyPromptContext::FullProblemSuffix,
        target: crate::config::RuliadProofPolicyTarget::ExpertSet,
        gradient_scope: crate::config::RuliadProofPolicyGradientScope::FullModel,
        normalization: crate::config::RuliadProofPolicyNormalization::PrefixConditional,
        candidate_symmetry: crate::config::RuliadProofPolicyCandidateSymmetry::BalancedRotation,
        presentation_risk: crate::config::RuliadProofPolicyPresentationRisk::Mean,
        weight: 1.0,
        every_steps: 16,
        start_after_steps: 0,
        dagger_start_after_steps: 128,
        stratified_difficulty_levels: 4,
        rollout_steps: 4,
        max_rows_per_update: 16,
        max_presentation_rows_per_update: 128,
        counterfactual_targets_per_state: 0,
        counterfactual_objective:
            crate::config::RuliadProofPolicyCounterfactualObjective::Independent,
        candidates: 4,
        max_completion_tokens: 128,
    };
    let mut model_config = tiny_model_config();
    model_config.vocab_size = 272;
    model_config.sequence_kernel =
        burn_dragon_core::SequenceKernelConfig::dense_score_short_context();
    model_config.fused_kernels.rotary_embedding = burn_dragon_core::RotaryEmbedding::Alibi;
    let telemetry_dir = tempfile::tempdir().expect("telemetry dir");
    let telemetry_path = telemetry_dir.path().join("policy.jsonl");
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(model_config, &device))
        .with_ruliad_supervision(RuliadSupervisionConfig {
            proof_policy,
            ..Default::default()
        })
        .with_ruliad_proof_policy_telemetry_path(Some(telemetry_path.clone()));
    let tokenizer = crate::tokenizer::TokenizerConfig {
        vocab_path: None,
        kind: crate::tokenizer::TokenizerKind::Pretokenized(
            crate::tokenizer::PretokenizedTokenizerConfig {
                vocab_size: 272,
                bos_id: None,
                eos_id: Some(271),
                pad_id: None,
                unk_id: None,
            },
        ),
    };
    let corpus_config = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../burn_dragon_p2p/deploy/profiles/ruliad-r3.semantic-action.corpus.toml");
    let dataset = crate::dataset::UniversalityDataset::new_ruliad_on_the_fly(
        &corpus_config,
        128,
        32,
        &tokenizer,
    )
    .expect("load production verifier corpus")
    .with_ruliad_supervision(RuliadSupervisionConfig {
        proof_policy,
        ..Default::default()
    });
    let dataset = crate::dataset::Dataset::from_universality(dataset);

    for step in [0, 16, 32, 48] {
        let policy_batch =
            crate::dataset::TokenSequenceDataset::source_selected_ruliad_policy_batch(
                &dataset,
                crate::dataset::DatasetSplit::Train,
                0,
                step,
                32,
                4,
            )
            .unwrap_or_else(|| panic!("missing production policy panel at step {step}"));
        let objective = model
            .ruliad_proof_policy_objective_at_step(&policy_batch, &device, 128, step)
            .unwrap_or_else(|| {
                let telemetry = std::fs::read_to_string(&telemetry_path).unwrap_or_default();
                panic!("unusable production policy panel at step {step}: {telemetry}")
            });
        assert!(objective.semantic_states > 0, "step={step}");
        assert!(objective.decision_rows > 0, "step={step}");
        assert!(tensor_scalar(objective.loss).is_finite(), "step={step}");
    }

    let hybrid_telemetry_path = telemetry_dir.path().join("hybrid-policy.jsonl");
    let mut hybrid_model_config = tiny_model_config();
    hybrid_model_config.vocab_size = 272;
    hybrid_model_config.sequence_kernel =
        burn_dragon_core::SequenceKernelConfig::dense_score_short_context();
    hybrid_model_config.fused_kernels.rotary_embedding = burn_dragon_core::RotaryEmbedding::Alibi;
    hybrid_model_config.sequence_score_head.enabled = true;
    let hybrid_model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        hybrid_model_config,
        &device,
    ))
    .with_ruliad_supervision(RuliadSupervisionConfig {
        proof_policy,
        proof_policy_semantic_refresh: crate::config::RuliadProofPolicySemanticRefreshConfig {
            enabled: true,
            every_steps: 64,
            start_after_steps: 64,
            counterfactual_targets_per_state: 1,
        },
        ..Default::default()
    })
    .with_ruliad_proof_policy_telemetry_path(Some(hybrid_telemetry_path.clone()));
    for step in [64, 80] {
        let policy_batch =
            crate::dataset::TokenSequenceDataset::source_selected_ruliad_policy_batch(
                &dataset,
                crate::dataset::DatasetSplit::Train,
                0,
                step,
                32,
                4,
            )
            .unwrap_or_else(|| panic!("missing hybrid policy panel at step {step}"));
        let objective = hybrid_model
            .ruliad_proof_policy_objective_at_step(&policy_batch, &device, 128, step)
            .unwrap_or_else(|| panic!("unusable hybrid policy panel at step {step}"));
        assert!(tensor_scalar(objective.loss).is_finite(), "step={step}");
    }
    let objectives = std::fs::read_to_string(hybrid_telemetry_path)
        .expect("hybrid telemetry")
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("telemetry json"))
        .map(|value| value["objective"].as_str().unwrap_or_default().to_string())
        .collect::<Vec<_>>();
    assert_eq!(
        objectives,
        vec![
            "semantic_sequence_energy_counterfactual_v1",
            "prefix_conditional_equivalent_v1",
        ]
    );
}

#[test]
fn ruliad_proof_policy_batch_plan_pairs_expert_and_model_visited_rows() {
    let plan = RuliadProofPolicyBatchPlan::new(
        crate::config::RuliadProofPolicyEffectiveMode::PairedDagger,
        32,
        4,
        4,
    );
    assert_eq!(plan.static_row_budget, 16);
    assert_eq!(plan.dagger_row_budget, 16);
    assert_eq!(plan.dagger_trajectory_budget, 4);
    assert_eq!(plan.trajectory_budget(), 20);
    assert_eq!(plan.rollout_steps, 4);
    assert_eq!(plan.dagger_trajectories_for_samples(1), 1);
    assert_eq!(plan.dagger_depth_for_count(0, 1), 4);
    assert_eq!(plan.rollout_steps_for_dagger_count(1), 4);
    assert_eq!(
        (0..plan.dagger_trajectory_budget)
            .map(|index| plan.dagger_depth(index))
            .sum::<usize>(),
        plan.dagger_row_budget
    );

    let uneven = RuliadProofPolicyBatchPlan::new(
        crate::config::RuliadProofPolicyEffectiveMode::PairedDagger,
        10,
        4,
        2,
    );
    assert_eq!(uneven.static_row_budget, 5);
    assert_eq!(uneven.dagger_row_budget, 5);
    assert_eq!(uneven.dagger_trajectory_budget, 2);
    assert_eq!(uneven.rollout_steps, 3);
    assert_eq!(uneven.dagger_depth(0), 3);
    assert_eq!(uneven.dagger_depth(1), 2);

    let bounded_causal = RuliadProofPolicyBatchPlan::new(
        crate::config::RuliadProofPolicyEffectiveMode::PairedDagger,
        4,
        2,
        1,
    );
    assert_eq!(bounded_causal.static_row_budget, 2);
    assert_eq!(bounded_causal.dagger_row_budget, 2);
    assert_eq!(bounded_causal.dagger_trajectory_budget, 1);
    assert_eq!(bounded_causal.rollout_steps, 2);
    assert_eq!(bounded_causal.dagger_depth(0), 2);
}

#[test]
fn paired_dagger_objective_executes_model_visited_rows_with_batch_one() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 20260812);
    let telemetry_dir = tempfile::tempdir().expect("telemetry directory");
    let telemetry_path = telemetry_dir.path().join("paired-dagger.jsonl");
    let mut supervision = scheduled_score_head_ruliad_supervision(
        crate::config::RuliadProofPolicyScoring::ResidualEnergy,
        crate::config::RuliadProofPolicyTarget::ExpertSet,
    );
    supervision.proof_policy.mode =
        crate::config::RuliadProofPolicyTrainingMode::StaticThenPairedDagger;
    supervision.proof_policy.dagger_start_after_steps = 0;
    supervision.proof_policy.rollout_steps = 4;
    supervision.proof_policy.max_rows_per_update = 32;
    supervision.proof_policy.max_presentation_rows_per_update = 32;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        scheduled_score_head_ruliad_model_config(),
        &device,
    ))
    .with_ruliad_supervision(supervision)
    .with_ruliad_proof_policy_telemetry_path(Some(telemetry_path.clone()));

    let objective = model
        .ruliad_proof_policy_objective_at_step(&scheduled_ruliad_policy_batch(), &device, 512, 0)
        .expect("batch-one paired DAgger objective");
    assert!(tensor_scalar(objective.loss).is_finite());

    let event: serde_json::Value = serde_json::from_str(
        std::fs::read_to_string(telemetry_path)
            .expect("paired DAgger telemetry")
            .lines()
            .next()
            .expect("telemetry event"),
    )
    .expect("telemetry JSON");
    assert_eq!(event["mode"], "paired_dagger");
    assert!(event["model_scoring_batches"].as_u64().unwrap_or_default() > 0);
    assert!(
        event["model_visited_expert_rows"]
            .as_u64()
            .unwrap_or_default()
            > 0
    );
    assert!(event["rollout_depth_reached"].as_u64().unwrap_or_default() > 1);
}

#[test]
fn ruliad_structured_answer_contrast_loss_scores_oracle_against_field_negatives() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 21);
    let mut config = tiny_model_config();
    config.vocab_size = 257;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_ruliad_supervision(RuliadSupervisionConfig {
            verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                enabled: true,
                structured_contrast_weight: 0.25,
                structured_contrast_every_steps: 2,
                structured_contrast_start_after_steps: 4,
                structured_contrast_margin: 0.25,
                structured_negative_count: 2,
                structured_template_negative_count: 2,
                max_completion_tokens: 24,
                ..Default::default()
            },
            ..Default::default()
        });
    let item = burn_dragon_universality::RuliadEvalItem {
        oracle_hash: "h0".to_string(),
        sample_index: 37,
        split: burn_dragon_universality::SampleSplit::Train,
        family: "eca".to_string(),
        task_kind: "multi_step_state".to_string(),
        math_domains: vec!["computation".to_string()],
        reasoning_modes: vec!["iterated".to_string()],
        prompt: "?:eca\n!:".to_string(),
        expected_answer: "xlen=44;xalpha=01;xcounts=20,24;xedge=01".to_string(),
        difficulty_level: Some(0),
        spec: None,
    };
    let policy_batch = crate::dataset::RuliadPolicyBatch {
        samples: vec![crate::dataset::RuliadPolicySample {
            item,
            prompt_tokens: vec![1, 2, 3],
        }],
        tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
            vocab_size: 257,
            eos_id: None,
        },
        stop_token_id: None,
        sampling_metadata: None,
    };

    model.gradient_scale_step.store(3, Ordering::Relaxed);
    assert!(
        model
            .ruliad_structured_answer_contrast_loss(&policy_batch, &device, 64)
            .is_none(),
        "contrast loss should respect start_after_steps"
    );
    model.gradient_scale_step.store(5, Ordering::Relaxed);
    assert!(
        model
            .ruliad_structured_answer_contrast_loss(&policy_batch, &device, 64)
            .is_none(),
        "contrast loss should respect every_steps cadence"
    );
    model.gradient_scale_step.store(6, Ordering::Relaxed);
    let loss = model
        .ruliad_structured_answer_contrast_loss(&policy_batch, &device, 64)
        .expect("structured answer contrast loss");

    let loss = tensor_scalar(loss);
    assert!(loss.is_finite(), "contrast loss should be finite: {loss}");
    assert!(loss > 0.0, "contrast loss should be non-zero: {loss}");
}

#[test]
fn ruliad_structured_answer_contrast_loss_scores_schema_negatives() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 35);
    let dir = tempfile::tempdir().expect("tempdir");
    let telemetry_path = dir
        .path()
        .join("events")
        .join("ruliad_structured_contrast.jsonl");
    let mut config = tiny_model_config();
    config.vocab_size = 257;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_ruliad_supervision(RuliadSupervisionConfig {
            verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                enabled: true,
                structured_contrast_weight: 0.25,
                structured_contrast_every_steps: 1,
                structured_negative_count: 0,
                structured_template_negative_count: 0,
                structured_schema_negative_count: 4,
                max_completion_tokens: 32,
                ..Default::default()
            },
            ..Default::default()
        })
        .with_ruliad_structured_contrast_telemetry_path(Some(telemetry_path.clone()));
    let item = burn_dragon_universality::RuliadEvalItem {
        oracle_hash: "h0".to_string(),
        sample_index: 56,
        split: burn_dragon_universality::SampleSplit::Train,
        family: "eca".to_string(),
        task_kind: "multi_step_state".to_string(),
        math_domains: vec!["computation".to_string()],
        reasoning_modes: vec!["iterated".to_string()],
        prompt: "?:eca\n!:".to_string(),
        expected_answer: "xlen=14;xalpha=01;xcounts=8,6;xedge=01".to_string(),
        difficulty_level: Some(0),
        spec: None,
    };
    let policy_batch = crate::dataset::RuliadPolicyBatch {
        samples: vec![crate::dataset::RuliadPolicySample {
            item,
            prompt_tokens: vec![1, 2, 3],
        }],
        tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
            vocab_size: 257,
            eos_id: None,
        },
        stop_token_id: None,
        sampling_metadata: None,
    };

    let loss = model
        .ruliad_structured_answer_contrast_loss(&policy_batch, &device, 64)
        .expect("schema-only structured answer contrast loss");
    assert!(tensor_scalar(loss).is_finite());
    let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
    let value: serde_json::Value =
        serde_json::from_str(content.lines().next().expect("telemetry line"))
            .expect("telemetry json");
    assert_eq!(value["field_negative_completion_rows"], 0);
    assert_eq!(value["template_negative_completion_rows"], 0);
    assert!(
        value["schema_negative_completion_rows"]
            .as_u64()
            .expect("schema rows")
            > 0
    );
    assert!(
        value["contrast_discriminative_tokens"]
            .as_u64()
            .expect("schema discriminative tokens")
            > 0
    );
}

#[test]
fn ruliad_field_binding_contrast_loss_scores_prompt_counterfactuals() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 24);
    let mut config = tiny_model_config();
    config.vocab_size = 257;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_ruliad_supervision(RuliadSupervisionConfig {
            verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                enabled: true,
                weight: 0.0,
                field_binding_contrast_weight: 0.25,
                field_binding_contrast_every_steps: 2,
                field_binding_contrast_start_after_steps: 4,
                field_binding_contrast_margin: 0.25,
                field_binding_contrast_pair_weight: 1.0,
                field_binding_contrast_max_pairs: 4,
                max_completion_tokens: 24,
                ..Default::default()
            },
            ..Default::default()
        });
    let item_a = burn_dragon_universality::RuliadEvalItem {
        oracle_hash: "h0".to_string(),
        sample_index: 43,
        split: burn_dragon_universality::SampleSplit::Train,
        family: "formal_proof".to_string(),
        task_kind: "select_proof_action".to_string(),
        math_domains: vec!["category".to_string()],
        reasoning_modes: vec!["equational".to_string()],
        prompt: "?:a\n!:".to_string(),
        expected_answer: "g4|a:r0|f|1.1".to_string(),
        difficulty_level: Some(0),
        spec: None,
    };
    let item_b = burn_dragon_universality::RuliadEvalItem {
        oracle_hash: "h1".to_string(),
        sample_index: 44,
        split: burn_dragon_universality::SampleSplit::Train,
        family: "formal_proof".to_string(),
        task_kind: "select_proof_action".to_string(),
        math_domains: vec!["category".to_string()],
        reasoning_modes: vec!["equational".to_string()],
        prompt: "?:b\n!:".to_string(),
        expected_answer: "g7|l:3|r|0.2".to_string(),
        difficulty_level: Some(0),
        spec: None,
    };
    let policy_batch = crate::dataset::RuliadPolicyBatch {
        samples: vec![
            crate::dataset::RuliadPolicySample {
                item: item_a,
                prompt_tokens: vec![1, 2, 3],
            },
            crate::dataset::RuliadPolicySample {
                item: item_b,
                prompt_tokens: vec![1, 2, 4],
            },
        ],
        tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
            vocab_size: 257,
            eos_id: None,
        },
        stop_token_id: None,
        sampling_metadata: None,
    };

    model.gradient_scale_step.store(3, Ordering::Relaxed);
    assert!(
        model
            .ruliad_field_binding_contrast_loss(&policy_batch, &device, 64)
            .is_none(),
        "field-binding contrast should respect start_after_steps"
    );
    model.gradient_scale_step.store(5, Ordering::Relaxed);
    assert!(
        model
            .ruliad_field_binding_contrast_loss(&policy_batch, &device, 64)
            .is_none(),
        "field-binding contrast should respect every_steps cadence"
    );
    model.gradient_scale_step.store(6, Ordering::Relaxed);
    let loss = model
        .ruliad_field_binding_contrast_loss(&policy_batch, &device, 64)
        .expect("field-binding contrast loss");

    let loss = tensor_scalar(loss);
    assert!(
        loss.is_finite(),
        "field-binding contrast loss should be finite: {loss}"
    );
    assert!(
        loss > 0.0,
        "field-binding contrast loss should be non-zero: {loss}"
    );
}

#[test]
fn ruliad_field_binding_contrast_loss_writes_activity_and_skip_telemetry() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 25);
    let dir = tempfile::tempdir().expect("tempdir");
    let telemetry_path = dir
        .path()
        .join("events")
        .join("ruliad_field_binding_contrast.jsonl");
    let mut config = tiny_model_config();
    config.vocab_size = 257;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_ruliad_supervision(RuliadSupervisionConfig {
            verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                enabled: true,
                weight: 0.0,
                field_binding_contrast_weight: 0.25,
                field_binding_contrast_every_steps: 1,
                field_binding_contrast_pair_weight: 1.0,
                field_binding_contrast_max_pairs: 2,
                max_completion_tokens: 24,
                ..Default::default()
            },
            ..Default::default()
        })
        .with_ruliad_field_binding_contrast_telemetry_path(Some(telemetry_path.clone()));
    let item_a = burn_dragon_universality::RuliadEvalItem {
        oracle_hash: "h0".to_string(),
        sample_index: 45,
        split: burn_dragon_universality::SampleSplit::Train,
        family: "proof_tree".to_string(),
        task_kind: "prove_theorem".to_string(),
        math_domains: vec!["category".to_string()],
        reasoning_modes: vec!["equational".to_string()],
        prompt: "?:a\n!:".to_string(),
        expected_answer: "ok=1;l=17;r=17".to_string(),
        difficulty_level: Some(0),
        spec: None,
    };
    let item_b = burn_dragon_universality::RuliadEvalItem {
        oracle_hash: "h1".to_string(),
        sample_index: 46,
        split: burn_dragon_universality::SampleSplit::Train,
        family: "proof_tree".to_string(),
        task_kind: "prove_theorem".to_string(),
        math_domains: vec!["category".to_string()],
        reasoning_modes: vec!["equational".to_string()],
        prompt: "?:b\n!:".to_string(),
        expected_answer: "ok=1;l=19;r=19".to_string(),
        difficulty_level: Some(0),
        spec: None,
    };
    let policy_batch = crate::dataset::RuliadPolicyBatch {
        samples: vec![
            crate::dataset::RuliadPolicySample {
                item: item_a.clone(),
                prompt_tokens: vec![1, 2, 3],
            },
            crate::dataset::RuliadPolicySample {
                item: item_b,
                prompt_tokens: vec![1, 2, 4],
            },
        ],
        tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
            vocab_size: 257,
            eos_id: None,
        },
        stop_token_id: None,
        sampling_metadata: None,
    };
    let loss = model
        .ruliad_field_binding_contrast_loss(&policy_batch, &device, 64)
        .expect("field-binding contrast loss");
    assert!(tensor_scalar(loss).is_finite());

    let item_c = burn_dragon_universality::RuliadEvalItem {
        oracle_hash: "h2".to_string(),
        sample_index: 47,
        split: burn_dragon_universality::SampleSplit::Train,
        family: "custom".to_string(),
        task_kind: "field_binding".to_string(),
        math_domains: vec!["category".to_string()],
        reasoning_modes: vec!["equational".to_string()],
        prompt: "?:c\n!:".to_string(),
        expected_answer: "v=17".to_string(),
        difficulty_level: Some(0),
        spec: None,
    };
    let one_sample_batch = crate::dataset::RuliadPolicyBatch {
        samples: vec![crate::dataset::RuliadPolicySample {
            item: item_c,
            prompt_tokens: vec![1, 2, 3],
        }],
        tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
            vocab_size: 257,
            eos_id: None,
        },
        stop_token_id: None,
        sampling_metadata: None,
    };
    assert!(
        model
            .ruliad_field_binding_contrast_loss(&one_sample_batch, &device, 64)
            .is_none(),
        "single oracle sample without a template schema should not produce a counterfactual pair"
    );

    let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
    let lines = content.lines().collect::<Vec<_>>();
    assert_eq!(lines.len(), 2);
    let active: serde_json::Value = serde_json::from_str(lines[0]).expect("active telemetry json");
    assert_eq!(active["version"], 3);
    assert_eq!(active["objective"], RULIAD_FIELD_BINDING_OBJECTIVE);
    assert_eq!(active["sample_groups"], 2);
    assert_eq!(
        active["oracle_prompt_count"], 2,
        "the bounded contrast batch should cover both prompts before reusing either"
    );
    assert!(
        active["prompt_pairs"].as_u64().expect("prompt pairs") >= 2,
        "template hard negatives may add extra field-binding rows"
    );
    assert!(
        active["contrast_pairs"].as_u64().expect("contrast pairs") >= 2,
        "template hard negatives may add extra contrast pairs"
    );
    assert!(
        active["candidate_pairs"].as_u64().expect("candidate pairs") >= 2,
        "template hard negatives may add extra candidates"
    );
    assert!(
        active["negative_pool_size"]
            .as_u64()
            .expect("negative pool size")
            > 2,
        "template hard negatives should be included in the pool"
    );
    assert_eq!(active["replay_pool_size"], 0);
    assert_eq!(active["replay_contrast_pairs"], 0);
    assert!(
        active["contrast_discriminative_tokens"]
            .as_u64()
            .expect("discriminative tokens")
            > 0
    );
    assert!(
        active["rank_metric_pairs"].as_u64().expect("rank pairs") >= 2,
        "active field-binding telemetry should rank natural and/or template pairs"
    );
    assert!(
        active["rank_metric_tokens"].as_u64().expect("rank tokens") > 0,
        "active field-binding telemetry should include rank-token evidence"
    );
    let positive_fraction = active["positive_token_fraction"]
        .as_f64()
        .expect("positive token fraction");
    assert!(
        (0.0..=1.0).contains(&positive_fraction),
        "positive token fraction should be bounded: {positive_fraction}"
    );
    assert!(
        active["logit_margin_mean"]
            .as_f64()
            .expect("margin mean")
            .is_finite(),
        "rank telemetry should include a finite margin mean"
    );
    assert!(
        active["sequence_rank_metric_pairs"]
            .as_u64()
            .expect("sequence rank pairs")
            >= 2
    );
    let positive_sequence_fraction = active["positive_sequence_fraction"]
        .as_f64()
        .expect("positive sequence fraction");
    assert!((0.0..=1.0).contains(&positive_sequence_fraction));
    assert!(
        active["sequence_log_probability_margin_mean"]
            .as_f64()
            .expect("sequence log-probability margin")
            .is_finite()
    );
    let skipped: serde_json::Value = serde_json::from_str(lines[1]).expect("skip telemetry json");
    assert_eq!(
        skipped["skip_reason"].as_str(),
        Some("no_counterfactual_pairs")
    );
    assert_eq!(skipped["contrast_pairs"], 0);
    assert_eq!(skipped["rank_metric_tokens"], 0);
    assert_eq!(skipped["sequence_rank_metric_pairs"], 0);
    assert!(skipped["logit_margin_mean"].is_null());
}

#[test]
fn ruliad_field_binding_contrast_never_uses_presented_actions_as_negatives() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 41);
    let dir = tempfile::tempdir().expect("tempdir");
    let telemetry_path = dir
        .path()
        .join("events")
        .join("ruliad_field_binding_contrast.jsonl");
    let mut model_config = tiny_model_config();
    model_config.vocab_size = 257;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(model_config, &device))
        .with_ruliad_supervision(RuliadSupervisionConfig {
            verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                enabled: true,
                weight: 0.0,
                field_binding_contrast_weight: 0.25,
                field_binding_contrast_every_steps: 1,
                field_binding_contrast_max_pairs: 4,
                max_completion_tokens: 64,
                ..Default::default()
            },
            ..Default::default()
        })
        .with_ruliad_field_binding_contrast_telemetry_path(Some(telemetry_path.clone()));
    let bundle = burn_dragon_universality::ruliad::formal::generate_formal_bundle(
        41,
        burn_dragon_universality::ruliad::formal::RuliadFormalGeneratorConfig {
            rewrite_depth: 2,
            leaf_count: 3,
            context_depth: 1,
            distractor_axioms: 1,
            ..Default::default()
        },
    )
    .expect("formal bundle");
    let proof_step_index = 0;
    let actions = burn_dragon_universality::ruliad::oracle_proof_action_set(
        &bundle.problem,
        &bundle.certificate,
        proof_step_index,
        4,
    )
    .expect("oracle action set");
    let contract = burn_dragon_universality::RuliadProofActionAnswerContract::SemanticStep;
    let oracle_answer = burn_dragon_universality::ruliad::proof_action_answer(
        &actions,
        actions.selected_index,
        contract,
    )
    .expect("oracle answer");
    let distractor_index = (0..actions.candidates.len())
        .find(|index| *index != actions.selected_index)
        .expect("distractor action");
    let distractor_answer =
        burn_dragon_universality::ruliad::proof_action_answer(&actions, distractor_index, contract)
            .expect("distractor answer");
    let problem_hash = bundle.problem.canonical_hash().expect("problem hash");
    let item = burn_dragon_universality::RuliadEvalItem {
        oracle_hash: problem_hash,
        sample_index: 41,
        split: burn_dragon_universality::SampleSplit::Train,
        family: "formal_proof".to_string(),
        task_kind: burn_dragon_universality::RuliadTaskKind::SelectProofAction
            .label()
            .to_string(),
        math_domains: vec!["formal_proof".to_string()],
        reasoning_modes: vec!["proof_construction".to_string()],
        prompt: burn_dragon_universality::ruliad::ruliad_proof_action_prompt(
            &bundle.problem,
            &actions,
        )
        .expect("policy prompt"),
        expected_answer: oracle_answer,
        difficulty_level: Some(0),
        spec: Some(burn_dragon_universality::RuliadSampleSpec::FormalProof {
            problem: bundle.problem,
            certificate: bundle.certificate,
            candidate: None,
            proof_step_index: Some(proof_step_index),
            action_presentation_rotation: Some(0),
            action_candidate_count: Some(actions.candidates.len()),
            action_answer_contract: contract,
            task: burn_dragon_universality::RuliadTaskKind::SelectProofAction,
        }),
    };
    let mut distractor_item = item.clone();
    distractor_item.sample_index = 42;
    distractor_item.expected_answer = distractor_answer;
    let policy_batch = crate::dataset::RuliadPolicyBatch {
        samples: vec![
            crate::dataset::RuliadPolicySample {
                item,
                prompt_tokens: vec![1, 2, 3],
            },
            crate::dataset::RuliadPolicySample {
                item: distractor_item,
                prompt_tokens: vec![1, 2, 4],
            },
        ],
        tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
            vocab_size: 257,
            eos_id: None,
        },
        stop_token_id: None,
        sampling_metadata: None,
    };

    assert!(
        model
            .ruliad_field_binding_contrast_loss(&policy_batch, &device, 128)
            .is_none(),
        "presented distractors must not produce a negative training pair"
    );
    let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
    let value: serde_json::Value =
        serde_json::from_str(content.lines().next().expect("telemetry row"))
            .expect("telemetry json");
    assert_eq!(value["candidate_pairs"], 0);
    assert!(
        value["filtered_presented_action_candidates"]
            .as_u64()
            .expect("filtered candidates")
            >= 2
    );
}

#[test]
fn ruliad_field_binding_contrast_uses_template_negatives_for_single_sample() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 33);
    let dir = tempfile::tempdir().expect("tempdir");
    let telemetry_path = dir
        .path()
        .join("events")
        .join("ruliad_field_binding_contrast.jsonl");
    let mut config = tiny_model_config();
    config.vocab_size = 257;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_ruliad_supervision(RuliadSupervisionConfig {
            verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                enabled: true,
                weight: 0.0,
                field_binding_contrast_weight: 0.25,
                field_binding_contrast_every_steps: 1,
                field_binding_contrast_max_pairs: 4,
                field_binding_contrast_replay_capacity: 0,
                max_completion_tokens: 24,
                ..Default::default()
            },
            ..Default::default()
        })
        .with_ruliad_field_binding_contrast_telemetry_path(Some(telemetry_path.clone()));
    let item = burn_dragon_universality::RuliadEvalItem {
        oracle_hash: "h0".to_string(),
        sample_index: 54,
        split: burn_dragon_universality::SampleSplit::Train,
        family: "proof_tree".to_string(),
        task_kind: "prove_theorem".to_string(),
        math_domains: vec!["category".to_string()],
        reasoning_modes: vec!["equational".to_string()],
        prompt: "?:single\n!:".to_string(),
        expected_answer: "ok=1;l=17;r=17".to_string(),
        difficulty_level: Some(0),
        spec: None,
    };
    let policy_batch = crate::dataset::RuliadPolicyBatch {
        samples: vec![crate::dataset::RuliadPolicySample {
            item,
            prompt_tokens: vec![1, 2, 3],
        }],
        tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
            vocab_size: 257,
            eos_id: None,
        },
        stop_token_id: None,
        sampling_metadata: None,
    };

    let loss = model
        .ruliad_field_binding_contrast_loss(&policy_batch, &device, 64)
        .expect("template hard negatives should provide a single-sample contrast pair");
    assert!(tensor_scalar(loss).is_finite());
    let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
    let active: serde_json::Value =
        serde_json::from_str(content.lines().next().expect("telemetry line"))
            .expect("field-binding telemetry json");
    assert_eq!(active["sample_groups"], 1);
    assert!(
        active["contrast_pairs"].as_u64().expect("contrast pairs") > 0,
        "template hard negatives should create contrast rows"
    );
    assert!(
        active["negative_pool_size"]
            .as_u64()
            .expect("negative pool size")
            > 1,
        "template hard negatives should augment the natural single-answer pool"
    );
    assert_eq!(active["replay_pool_size"], 0);
    assert_eq!(active["replay_contrast_pairs"], 0);
}

#[test]
fn ruliad_field_binding_contrast_uses_schema_negatives_for_single_sample() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 34);
    let dir = tempfile::tempdir().expect("tempdir");
    let telemetry_path = dir
        .path()
        .join("events")
        .join("ruliad_field_binding_contrast.jsonl");
    let mut config = tiny_model_config();
    config.vocab_size = 257;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_ruliad_supervision(RuliadSupervisionConfig {
            verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                enabled: true,
                weight: 0.0,
                field_binding_contrast_weight: 0.25,
                field_binding_contrast_every_steps: 1,
                field_binding_contrast_max_pairs: 4,
                field_binding_contrast_replay_capacity: 0,
                max_completion_tokens: 32,
                ..Default::default()
            },
            ..Default::default()
        })
        .with_ruliad_field_binding_contrast_telemetry_path(Some(telemetry_path.clone()));
    let item = burn_dragon_universality::RuliadEvalItem {
        oracle_hash: "h0".to_string(),
        sample_index: 55,
        split: burn_dragon_universality::SampleSplit::Train,
        family: "eca".to_string(),
        task_kind: "multi_step_state".to_string(),
        math_domains: vec!["computation".to_string()],
        reasoning_modes: vec!["iterated".to_string()],
        prompt: "?:eca\n!:".to_string(),
        expected_answer: "xlen=14;xalpha=01;xcounts=8,6;xedge=01".to_string(),
        difficulty_level: Some(0),
        spec: None,
    };
    let policy_batch = crate::dataset::RuliadPolicyBatch {
        samples: vec![crate::dataset::RuliadPolicySample {
            item,
            prompt_tokens: vec![1, 2, 3],
        }],
        tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
            vocab_size: 257,
            eos_id: None,
        },
        stop_token_id: None,
        sampling_metadata: None,
    };

    let loss = model
        .ruliad_field_binding_contrast_loss(&policy_batch, &device, 64)
        .expect("schema hard negatives should provide a single-sample contrast pair");
    assert!(tensor_scalar(loss).is_finite());
    let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
    let active: serde_json::Value =
        serde_json::from_str(content.lines().next().expect("telemetry line"))
            .expect("field-binding telemetry json");
    assert_eq!(active["sample_groups"], 1);
    assert!(
        active["contrast_discriminative_tokens"]
            .as_u64()
            .expect("discriminative key tokens")
            >= 4,
        "schema negatives should activate key-token contrast"
    );
    assert!(
        active["negative_pool_size"]
            .as_u64()
            .expect("negative pool size")
            > 1,
        "schema hard negatives should augment the natural single-answer pool"
    );
}

#[test]
fn ruliad_field_binding_contrast_prioritizes_prompt_coverage_over_global_byte_distance() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 32);
    let dir = tempfile::tempdir().expect("tempdir");
    let telemetry_path = dir
        .path()
        .join("events")
        .join("ruliad_field_binding_contrast.jsonl");
    let mut config = tiny_model_config();
    config.vocab_size = 257;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_ruliad_supervision(RuliadSupervisionConfig {
            verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                enabled: true,
                weight: 0.0,
                field_binding_contrast_weight: 0.25,
                field_binding_contrast_every_steps: 1,
                field_binding_contrast_max_pairs: 1,
                max_completion_tokens: 24,
                ..Default::default()
            },
            ..Default::default()
        })
        .with_ruliad_field_binding_contrast_telemetry_path(Some(telemetry_path.clone()));
    let make_item = |sample_index, answer: &str| burn_dragon_universality::RuliadEvalItem {
        oracle_hash: format!("h{sample_index}"),
        sample_index,
        split: burn_dragon_universality::SampleSplit::Train,
        family: "proof_tree".to_string(),
        task_kind: "prove_theorem".to_string(),
        math_domains: vec!["category".to_string()],
        reasoning_modes: vec!["equational".to_string()],
        prompt: format!("?:sample{sample_index}\n!:"),
        expected_answer: answer.to_string(),
        difficulty_level: Some(0),
        spec: None,
    };
    let policy_batch = crate::dataset::RuliadPolicyBatch {
        samples: vec![
            crate::dataset::RuliadPolicySample {
                item: make_item(51, "ok=1;l=17;r=17"),
                prompt_tokens: vec![1, 2, 3],
            },
            crate::dataset::RuliadPolicySample {
                item: make_item(52, "ok=1;l=19;r=19"),
                prompt_tokens: vec![1, 2, 4],
            },
            crate::dataset::RuliadPolicySample {
                item: make_item(53, "ok=0;l=00;r=00"),
                prompt_tokens: vec![1, 2, 5],
            },
        ],
        tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
            vocab_size: 257,
            eos_id: None,
        },
        stop_token_id: None,
        sampling_metadata: None,
    };

    let loss = model
        .ruliad_field_binding_contrast_loss(&policy_batch, &device, 64)
        .expect("field-binding contrast loss");
    assert!(tensor_scalar(loss).is_finite());
    let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
    let active: serde_json::Value =
        serde_json::from_str(content.lines().next().expect("telemetry line"))
            .expect("telemetry json");

    assert_eq!(active["contrast_pairs"], 1);
    assert_eq!(
        active["contrast_discriminative_tokens"], 1,
        "the bounded pair should supervise only the causally valid first divergence"
    );
    assert_eq!(active["oracle_prompt_count"], 1);
}

#[test]
fn ruliad_field_binding_contrast_uses_replay_for_single_sample_batches() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 26);
    let dir = tempfile::tempdir().expect("tempdir");
    let telemetry_path = dir
        .path()
        .join("events")
        .join("ruliad_field_binding_contrast.jsonl");
    let mut config = tiny_model_config();
    config.vocab_size = 257;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_ruliad_supervision(RuliadSupervisionConfig {
            verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                enabled: true,
                weight: 0.0,
                field_binding_contrast_weight: 0.25,
                field_binding_contrast_every_steps: 1,
                field_binding_contrast_max_pairs: 4,
                field_binding_contrast_replay_capacity: 4,
                max_completion_tokens: 24,
                ..Default::default()
            },
            ..Default::default()
        })
        .with_ruliad_field_binding_contrast_telemetry_path(Some(telemetry_path.clone()));
    let item_a = burn_dragon_universality::RuliadEvalItem {
        oracle_hash: "h0".to_string(),
        sample_index: 47,
        split: burn_dragon_universality::SampleSplit::Train,
        family: "proof_tree".to_string(),
        task_kind: "prove_theorem".to_string(),
        math_domains: vec!["category".to_string()],
        reasoning_modes: vec!["equational".to_string()],
        prompt: "?:a\n!:".to_string(),
        expected_answer: "v=17".to_string(),
        difficulty_level: Some(0),
        spec: None,
    };
    let item_b = burn_dragon_universality::RuliadEvalItem {
        oracle_hash: "h1".to_string(),
        sample_index: 48,
        split: burn_dragon_universality::SampleSplit::Train,
        family: "proof_tree".to_string(),
        task_kind: "prove_theorem".to_string(),
        math_domains: vec!["category".to_string()],
        reasoning_modes: vec!["equational".to_string()],
        prompt: "?:b\n!:".to_string(),
        expected_answer: "v=19".to_string(),
        difficulty_level: Some(0),
        spec: None,
    };
    let tokenization = burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
        vocab_size: 257,
        eos_id: None,
    };
    let first_batch = crate::dataset::RuliadPolicyBatch {
        samples: vec![crate::dataset::RuliadPolicySample {
            item: item_a,
            prompt_tokens: vec![1, 2, 3],
        }],
        tokenization: tokenization.clone(),
        stop_token_id: None,
        sampling_metadata: None,
    };
    assert!(
        model
            .ruliad_field_binding_contrast_loss(&first_batch, &device, 64)
            .is_none(),
        "first single-sample batch should fill replay but have no contrast pair"
    );

    let second_batch = crate::dataset::RuliadPolicyBatch {
        samples: vec![crate::dataset::RuliadPolicySample {
            item: item_b,
            prompt_tokens: vec![1, 2, 4],
        }],
        tokenization,
        stop_token_id: None,
        sampling_metadata: None,
    };
    let loss = model
        .ruliad_field_binding_contrast_loss(&second_batch, &device, 64)
        .expect("replay should provide a counterfactual pair");
    assert!(tensor_scalar(loss).is_finite());

    let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
    let lines = content.lines().collect::<Vec<_>>();
    assert_eq!(lines.len(), 2);
    let replay_active: serde_json::Value =
        serde_json::from_str(lines[1]).expect("replay telemetry json");
    assert_eq!(replay_active["sample_groups"], 1);
    assert_eq!(replay_active["contrast_pairs"], 1);
    assert_eq!(replay_active["candidate_pairs"], 1);
    assert_eq!(replay_active["replay_pool_size"], 1);
    assert_eq!(replay_active["replay_contrast_pairs"], 1);
    assert!(
        replay_active["rank_metric_tokens"]
            .as_u64()
            .expect("rank tokens")
            > 0
    );
}

#[test]
fn ruliad_generated_attractor_replay_tracks_repeated_wrong_answers() {
    let mut replay = RuliadGeneratedAttractorReplay::default();
    let key = RuliadGeneratedAttractorKey {
        family: "proof_tree".to_string(),
        task_kind: "prove_theorem".to_string(),
        contract: "ok;l;r".to_string(),
        answer: "ok=1;l=5;r=5".to_string(),
    };
    assert!(replay.record(
        key.clone(),
        burn_dragon_universality::ruliad::RuliadAnswerStatus::SchemaValidWrong,
        1,
        8,
    ));
    assert!(
        replay
            .candidates_for(RuliadGeneratedAttractorQuery {
                family: "proof_tree",
                task_kind: "prove_theorem",
                expected_contract: "ok;l;r",
                expected_answer: "ok=1;l=17;r=17",
                min_count: 2,
                max_candidates: 4,
                min_distinct_answers: 1,
                max_dominant_fraction: 1.0,
            },)
            .is_empty()
    );
    assert!(replay.record(
        key,
        burn_dragon_universality::ruliad::RuliadAnswerStatus::Partial,
        2,
        8,
    ));
    let candidates = replay.candidates_for(RuliadGeneratedAttractorQuery {
        family: "proof_tree",
        task_kind: "prove_theorem",
        expected_contract: "ok;l;r",
        expected_answer: "ok=1;l=17;r=17",
        min_count: 2,
        max_candidates: 4,
        min_distinct_answers: 1,
        max_dominant_fraction: 1.0,
    });
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].count, 2);
    assert_eq!(candidates[0].key.answer, "ok=1;l=5;r=5");
    assert!(
        replay
            .candidates_for(RuliadGeneratedAttractorQuery {
                family: "proof_tree",
                task_kind: "prove_theorem",
                expected_contract: "ok;l;r",
                expected_answer: "ok=1;l=5;r=5",
                min_count: 2,
                max_candidates: 4,
                min_distinct_answers: 1,
                max_dominant_fraction: 1.0,
            },)
            .is_empty()
    );
    let summary = replay.summary(2);
    assert_eq!(summary.pool_size, 1);
    assert_eq!(summary.active_count, 1);
    assert_eq!(summary.active_observation_count, 2);
    assert_eq!(summary.dominant_count, 2);
    assert_eq!(summary.distinct_answers, 1);
}

#[test]
fn ruliad_generated_attractor_replay_requires_diverse_answers() {
    let mut replay = RuliadGeneratedAttractorReplay::default();
    let key_a = RuliadGeneratedAttractorKey {
        family: "proof_tree".to_string(),
        task_kind: "prove_theorem".to_string(),
        contract: "ok;l;r".to_string(),
        answer: "ok=1;l=5;r=5".to_string(),
    };
    let key_b = RuliadGeneratedAttractorKey {
        family: "proof_tree".to_string(),
        task_kind: "prove_theorem".to_string(),
        contract: "ok;l;r".to_string(),
        answer: "ok=1;l=9;r=9".to_string(),
    };
    for step_index in 1..=3 {
        assert!(replay.record(
            key_a.clone(),
            burn_dragon_universality::ruliad::RuliadAnswerStatus::SchemaValidWrong,
            step_index,
            8,
        ));
    }
    assert!(replay.record(
        key_b.clone(),
        burn_dragon_universality::ruliad::RuliadAnswerStatus::SchemaValidWrong,
        4,
        8,
    ));

    let dominated_summary = replay.summary(1);
    assert_eq!(
        dominated_summary.diversity_skip_reason(2, 0.5),
        Some("generated_attractor_dominant_answer")
    );
    assert!(
        replay
            .candidates_for(RuliadGeneratedAttractorQuery {
                family: "proof_tree",
                task_kind: "prove_theorem",
                expected_contract: "ok;l;r",
                expected_answer: "ok=1;l=17;r=17",
                min_count: 1,
                max_candidates: 4,
                min_distinct_answers: 2,
                max_dominant_fraction: 0.5,
            },)
            .is_empty()
    );

    for step_index in 5..=6 {
        assert!(replay.record(
            key_b.clone(),
            burn_dragon_universality::ruliad::RuliadAnswerStatus::Partial,
            step_index,
            8,
        ));
    }
    let balanced_summary = replay.summary(1);
    assert_eq!(balanced_summary.dominant_fraction(), 0.5);
    assert_eq!(balanced_summary.diversity_skip_reason(2, 0.5), None);
    let candidates = replay.candidates_for(RuliadGeneratedAttractorQuery {
        family: "proof_tree",
        task_kind: "prove_theorem",
        expected_contract: "ok;l;r",
        expected_answer: "ok=1;l=17;r=17",
        min_count: 1,
        max_candidates: 4,
        min_distinct_answers: 2,
        max_dominant_fraction: 0.5,
    });
    assert_eq!(candidates.len(), 2);
}

#[test]
fn ruliad_field_binding_contrast_uses_generated_attractor_replay() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 38);
    let dir = tempfile::tempdir().expect("tempdir");
    let telemetry_path = dir
        .path()
        .join("events")
        .join("ruliad_field_binding_contrast.jsonl");
    let attractor_telemetry_path = dir
        .path()
        .join("events")
        .join("ruliad_generated_attractor_replay.jsonl");
    let mut config = tiny_model_config();
    config.vocab_size = 257;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_ruliad_supervision(RuliadSupervisionConfig {
            verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                enabled: true,
                weight: 0.01,
                field_binding_contrast_weight: 0.25,
                field_binding_contrast_every_steps: 1,
                field_binding_contrast_max_pairs: 16,
                field_binding_contrast_replay_capacity: 0,
                generated_attractor_replay_capacity: 8,
                generated_attractor_replay_min_count: 1,
                generated_attractor_replay_max_candidates: 4,
                generated_attractor_replay_min_distinct_answers: 1,
                generated_attractor_replay_max_dominant_fraction: 1.0,
                max_completion_tokens: 24,
                ..Default::default()
            },
            ..Default::default()
        })
        .with_ruliad_field_binding_contrast_telemetry_path(Some(telemetry_path.clone()))
        .with_ruliad_generated_attractor_telemetry_path(Some(attractor_telemetry_path.clone()));
    let item = burn_dragon_universality::RuliadEvalItem {
        oracle_hash: "h0".to_string(),
        sample_index: 61,
        split: burn_dragon_universality::SampleSplit::Train,
        family: "proof_tree".to_string(),
        task_kind: "prove_theorem".to_string(),
        math_domains: vec!["category".to_string()],
        reasoning_modes: vec!["equational".to_string()],
        prompt: "?:single\n!:".to_string(),
        expected_answer: "ok=1;l=17;r=17".to_string(),
        difficulty_level: Some(0),
        spec: None,
    };
    let sample = crate::dataset::RuliadPolicySample {
        item,
        prompt_tokens: vec![1, 2, 3],
    };
    let score = burn_dragon_universality::ruliad::score_ruliad_item_completion(
        &sample.item,
        Some("ok=1;l=5;r=5\n[/R2]"),
    );
    assert!(model.record_ruliad_generated_attractor(&sample, "ok=1;l=5;r=5\n[/R2]", &score, 3,));
    let policy_batch = crate::dataset::RuliadPolicyBatch {
        samples: vec![sample],
        tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
            vocab_size: 257,
            eos_id: None,
        },
        stop_token_id: None,
        sampling_metadata: None,
    };
    let loss = model
        .ruliad_field_binding_contrast_loss(&policy_batch, &device, 64)
        .expect("generated attractor should provide a contrast pair");
    assert!(tensor_scalar(loss).is_finite());
    let content = std::fs::read_to_string(&telemetry_path).expect("field telemetry");
    let active: serde_json::Value =
        serde_json::from_str(content.lines().next().expect("telemetry line"))
            .expect("field-binding telemetry json");
    assert_eq!(active["sample_groups"], 1);
    assert!(
        active["generated_attractor_negative_pool_size"]
            .as_u64()
            .expect("generated attractor pool")
            >= 1
    );
    assert!(
        active["generated_attractor_contrast_pairs"]
            .as_u64()
            .expect("generated attractor pairs")
            >= 1
    );
    let attractor_content =
        std::fs::read_to_string(&attractor_telemetry_path).expect("attractor telemetry");
    let replay_event: serde_json::Value =
        serde_json::from_str(attractor_content.lines().next().expect("attractor line"))
            .expect("attractor telemetry json");
    assert_eq!(replay_event["source"], "field_binding");
    assert!(
        replay_event["selected_field_binding_pairs"]
            .as_u64()
            .expect("selected field-binding pairs")
            >= 1
    );
}

#[test]
fn ruliad_verifier_policy_loss_uses_generated_attractor_replay_candidates() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 39);
    let dir = tempfile::tempdir().expect("tempdir");
    let telemetry_path = dir
        .path()
        .join("events")
        .join("ruliad_verifier_policy.jsonl");
    let attractor_telemetry_path = dir
        .path()
        .join("events")
        .join("ruliad_generated_attractor_replay.jsonl");
    let mut config = tiny_model_config();
    config.vocab_size = 257;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_ruliad_supervision(RuliadSupervisionConfig {
            verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                enabled: true,
                mode: crate::config::train::RuliadVerifierRewardMode::VpoIndependent,
                weight: 0.01,
                group_size: 2,
                every_steps: 1,
                include_oracle_candidate: true,
                generated_attractor_replay_capacity: 8,
                generated_attractor_replay_min_count: 1,
                generated_attractor_replay_max_candidates: 4,
                generated_attractor_replay_min_distinct_answers: 1,
                generated_attractor_replay_max_dominant_fraction: 1.0,
                max_completion_tokens: 24,
                ..Default::default()
            },
            ..Default::default()
        })
        .with_ruliad_policy_telemetry_path(Some(telemetry_path.clone()))
        .with_ruliad_generated_attractor_telemetry_path(Some(attractor_telemetry_path.clone()));
    let item = burn_dragon_universality::RuliadEvalItem {
        oracle_hash: "h0".to_string(),
        sample_index: 62,
        split: burn_dragon_universality::SampleSplit::Train,
        family: "proof_tree".to_string(),
        task_kind: "prove_theorem".to_string(),
        math_domains: vec!["category".to_string()],
        reasoning_modes: vec!["equational".to_string()],
        prompt: "?:single\n!:".to_string(),
        expected_answer: "ok=1;l=17;r=17".to_string(),
        difficulty_level: Some(0),
        spec: None,
    };
    let sample = crate::dataset::RuliadPolicySample {
        item,
        prompt_tokens: vec![1, 2, 3],
    };
    let score = burn_dragon_universality::ruliad::score_ruliad_item_completion(
        &sample.item,
        Some("ok=1;l=5;r=5\n[/R2]"),
    );
    assert!(model.record_ruliad_generated_attractor(&sample, "ok=1;l=5;r=5\n[/R2]", &score, 4,));
    let policy_batch = crate::dataset::RuliadPolicyBatch {
        samples: vec![sample],
        tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
            vocab_size: 257,
            eos_id: None,
        },
        stop_token_id: None,
        sampling_metadata: None,
    };
    let loss = model
        .ruliad_verifier_policy_loss(&policy_batch, &device, 64)
        .expect("policy loss should include generated-attractor candidate");
    assert!(tensor_scalar(loss).is_finite());
    let content = std::fs::read_to_string(&telemetry_path).expect("policy telemetry");
    let active: serde_json::Value =
        serde_json::from_str(content.lines().next().expect("telemetry line"))
            .expect("policy telemetry json");
    assert!(
        active["generated_attractor_completion_rows"]
            .as_u64()
            .expect("generated attractor candidate rows")
            >= 1
    );
    let attractor_content =
        std::fs::read_to_string(&attractor_telemetry_path).expect("attractor telemetry");
    let replay_event: serde_json::Value =
        serde_json::from_str(attractor_content.lines().next().expect("attractor line"))
            .expect("attractor telemetry json");
    assert_eq!(replay_event["source"], "policy");
    assert!(
        replay_event["selected_candidate_rows"]
            .as_u64()
            .expect("selected candidates")
            >= 1
    );
}

#[test]
fn ruliad_structured_answer_contrast_loss_writes_activity_telemetry() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 23);
    let dir = tempfile::tempdir().expect("tempdir");
    let telemetry_path = dir
        .path()
        .join("events")
        .join("ruliad_structured_contrast.jsonl");
    let mut config = tiny_model_config();
    config.vocab_size = 257;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_ruliad_supervision(RuliadSupervisionConfig {
            verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                enabled: true,
                structured_contrast_weight: 0.25,
                structured_contrast_every_steps: 1,
                structured_negative_count: 2,
                structured_template_negative_count: 2,
                structured_schema_negative_count: 2,
                max_completion_tokens: 24,
                ..Default::default()
            },
            ..Default::default()
        })
        .with_ruliad_structured_contrast_telemetry_path(Some(telemetry_path.clone()));
    let item = burn_dragon_universality::RuliadEvalItem {
        oracle_hash: "h0".to_string(),
        sample_index: 41,
        split: burn_dragon_universality::SampleSplit::Train,
        family: "proof_tree".to_string(),
        task_kind: "prove_theorem".to_string(),
        math_domains: vec!["category".to_string(), "formal_proof".to_string()],
        reasoning_modes: vec!["equational".to_string()],
        prompt: "?:ss\n!:".to_string(),
        expected_answer: "ok=1;l=17;r=17".to_string(),
        difficulty_level: Some(0),
        spec: None,
    };
    let policy_batch = crate::dataset::RuliadPolicyBatch {
        samples: vec![crate::dataset::RuliadPolicySample {
            item,
            prompt_tokens: vec![1, 2, 3],
        }],
        tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
            vocab_size: 257,
            eos_id: None,
        },
        stop_token_id: None,
        sampling_metadata: None,
    };

    let loss = model
        .ruliad_structured_answer_contrast_loss(&policy_batch, &device, 64)
        .expect("structured answer contrast loss");
    assert!(tensor_scalar(loss).is_finite());
    let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
    let value: serde_json::Value =
        serde_json::from_str(content.lines().next().expect("telemetry line"))
            .expect("telemetry json");
    assert_eq!(value["sample_groups"], 1);
    assert_eq!(value["oracle_completion_rows"], 1);
    assert_eq!(value["field_negative_completion_rows"], 2);
    assert_eq!(value["template_negative_completion_rows"], 2);
    assert_eq!(value["schema_negative_completion_rows"], 2);
    assert_eq!(value["contrast_pairs"], 6);
    assert!(
        value["contrast_discriminative_tokens"]
            .as_u64()
            .expect("discriminative tokens")
            > 0
    );
}

#[test]
fn ruliad_verifier_policy_loss_writes_reward_telemetry() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 13);
    let dir = tempfile::tempdir().expect("tempdir");
    let telemetry_path = dir
        .path()
        .join("events")
        .join("ruliad_verifier_policy.jsonl");
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_ruliad_supervision(RuliadSupervisionConfig {
        verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
            enabled: true,
            mode: crate::config::train::RuliadVerifierRewardMode::VpoIndependent,
            weight: 0.1,
            group_size: 2,
            max_completion_tokens: 2,
            every_steps: 1,
            top_k: 1,
            kl_weight: 0.0,
            vpo_scalarizations: 4,
            ..Default::default()
        },
        ..Default::default()
    })
    .with_ruliad_policy_telemetry_path(Some(telemetry_path.clone()));
    let item = burn_dragon_universality::RuliadEvalItem {
        oracle_hash: "h0".to_string(),
        sample_index: 23,
        split: burn_dragon_universality::SampleSplit::Train,
        family: "law".to_string(),
        task_kind: "category_law".to_string(),
        math_domains: vec!["category".to_string()],
        reasoning_modes: vec!["equational".to_string()],
        prompt: "?:q\n!:".to_string(),
        expected_answer: "ok=1".to_string(),
        difficulty_level: Some(0),
        spec: None,
    };
    let policy_batch = crate::dataset::RuliadPolicyBatch {
        samples: vec![crate::dataset::RuliadPolicySample {
            item,
            prompt_tokens: vec![1, 2, 3],
        }],
        tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
            vocab_size: 257,
            eos_id: None,
        },
        stop_token_id: None,
        sampling_metadata: None,
    };
    let loss = model
        .ruliad_verifier_policy_loss(&policy_batch, &device, 8)
        .expect("VPO verifier policy loss");
    assert!(tensor_scalar(loss).is_finite());
    let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
    let line = content.lines().next().expect("telemetry line");
    let value: serde_json::Value = serde_json::from_str(line).expect("telemetry json");
    assert_eq!(value["mode"], "vpo_independent");
    assert_eq!(value["scalarization_count"], 4);
    assert_eq!(value["completion_rows"], 2);
    assert!(
        value["reward_mean"]
            .as_f64()
            .expect("reward mean")
            .is_finite(),
        "reward mean should be finite"
    );
}

#[test]
fn dynamics_anchor_penalizes_teacher_distribution_drift() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let plain = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ));
    let anchored = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_dynamics_anchor(DynamicsAnchorConfig {
        enabled: true,
        weight: 1.0,
        teacher_update_rate: 0.0,
        kl: SelfDistillationKlKind::Forward,
        ..Default::default()
    });
    let student_logits = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![8.0, 0.0, 8.0, 0.0], [1, 2, 2]),
        &device,
    );
    let teacher_logits = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![0.0, 8.0, 0.0, 8.0], [1, 2, 2]),
        &device,
    );
    let clean_inputs =
        Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![0, 1], [1, 2]), &device);
    let targets =
        Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![0, 0], [1, 2]), &device);

    let ce = tensor_scalar(plain.next_token_loss_from_logits(
        student_logits.clone(),
        targets.clone(),
        clean_inputs.clone(),
        None,
        None,
    ));
    let anchored_loss = tensor_scalar(anchored.next_token_loss_from_logits(
        student_logits,
        targets,
        clean_inputs,
        None,
        Some(teacher_logits),
    ));

    assert!(
        anchored_loss > ce + 6.0,
        "anchor should add KL pressure when student diverges from teacher: ce={ce} anchored={anchored_loss}"
    );
}

#[test]
fn dynamics_anchor_context_mask_uses_unsupervised_tokens() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_dynamics_anchor(DynamicsAnchorConfig {
        enabled: true,
        weight: 1.0,
        mask: DynamicsAnchorMask::ContextTokens,
        ..Default::default()
    });
    let target_mask =
        Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![1, 0, 1], [1, 3]), &device);
    let context_mask = model
        .dynamics_anchor_mask(Some(target_mask))
        .expect("context mask")
        .to_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .expect("mask values");

    assert_eq!(context_mask, vec![0, 1, 0]);
}

#[test]
fn repeat_unlikelihood_penalizes_wrong_copy_predictions() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let plain = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ));
    let repeat_penalized = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_repeat_unlikelihood(RepeatUnlikelihoodConfig {
        enabled: true,
        weight: 0.5,
        ..Default::default()
    });
    let logits = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![5.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0], [1, 2, 4]),
        &device,
    );
    let clean_inputs =
        Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![0, 1], [1, 2]), &device);
    let targets =
        Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![1, 2], [1, 2]), &device);
    let ce = plain.next_token_loss_from_logits(
        logits.clone(),
        targets.clone(),
        clean_inputs.clone(),
        None,
        None,
    );
    let penalized =
        repeat_penalized.next_token_loss_from_logits(logits, targets, clean_inputs, None, None);
    let ce_value = ce.to_data().convert::<f32>().into_vec::<f32>().expect("ce")[0];
    let penalized_value = penalized
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("penalized")[0];
    assert!(
        penalized_value > ce_value,
        "repeat unlikelihood should increase loss for wrong-copy logits: ce={ce_value} penalized={penalized_value}"
    );
}

#[test]
fn logit_entropy_floor_penalizes_overconfident_logits() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let plain = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ));
    let entropy_penalized = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_logit_entropy_floor(LogitEntropyFloorConfig {
        enabled: true,
        weight: 0.5,
        target_entropy_bits: 2.0,
        ..Default::default()
    });
    let logits = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![8.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0], [1, 2, 4]),
        &device,
    );
    let clean_inputs =
        Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![0, 1], [1, 2]), &device);
    let targets =
        Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![0, 1], [1, 2]), &device);
    let ce = plain.next_token_loss_from_logits(
        logits.clone(),
        targets.clone(),
        clean_inputs.clone(),
        None,
        None,
    );
    let penalized =
        entropy_penalized.next_token_loss_from_logits(logits, targets, clean_inputs, None, None);
    let ce_value = ce.to_data().convert::<f32>().into_vec::<f32>().expect("ce")[0];
    let penalized_value = penalized
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("penalized")[0];
    assert!(
        penalized_value > ce_value,
        "entropy floor should increase loss for overconfident logits: ce={ce_value} penalized={penalized_value}"
    );
}

#[test]
fn logit_entropy_floor_respects_every_steps() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let plain = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ));
    let throttled = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_logit_entropy_floor(LogitEntropyFloorConfig {
        enabled: true,
        weight: 0.5,
        target_entropy_bits: 2.0,
        every_steps: 4,
        ..Default::default()
    });
    let logits = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![8.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0], [1, 2, 4]),
        &device,
    );
    let clean_inputs =
        Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![0, 1], [1, 2]), &device);
    let targets =
        Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![0, 1], [1, 2]), &device);
    let ce = tensor_scalar(plain.next_token_loss_from_logits(
        logits.clone(),
        targets.clone(),
        clean_inputs.clone(),
        None,
        None,
    ));
    throttled
        .gradient_scale_step
        .store(2, std::sync::atomic::Ordering::Relaxed);
    let off_cadence = tensor_scalar(throttled.next_token_loss_from_logits(
        logits.clone(),
        targets.clone(),
        clean_inputs.clone(),
        None,
        None,
    ));
    throttled
        .gradient_scale_step
        .store(4, std::sync::atomic::Ordering::Relaxed);
    let on_cadence = tensor_scalar(throttled.next_token_loss_from_logits(
        logits,
        targets,
        clean_inputs,
        None,
        None,
    ));
    assert!(
        (off_cadence - ce).abs() < 1.0e-5,
        "off-cadence entropy loss should match CE: ce={ce} off={off_cadence}"
    );
    assert!(
        on_cadence > ce,
        "on-cadence entropy loss should add penalty: ce={ce} on={on_cadence}"
    );
}

#[test]
fn logit_entropy_floor_does_not_penalize_logits_above_floor() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let plain = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ));
    let entropy_penalized = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_logit_entropy_floor(LogitEntropyFloorConfig {
        enabled: true,
        weight: 0.5,
        target_entropy_bits: 1.0,
        ..Default::default()
    });
    let logits = Tensor::<TestBackend, 3>::zeros([1, 2, 4], &device);
    let clean_inputs =
        Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![0, 1], [1, 2]), &device);
    let targets =
        Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![0, 1], [1, 2]), &device);
    let ce = plain.next_token_loss_from_logits(
        logits.clone(),
        targets.clone(),
        clean_inputs.clone(),
        None,
        None,
    );
    let penalized =
        entropy_penalized.next_token_loss_from_logits(logits, targets, clean_inputs, None, None);
    let ce_value = ce.to_data().convert::<f32>().into_vec::<f32>().expect("ce")[0];
    let penalized_value = penalized
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("penalized")[0];
    assert!(
        (penalized_value - ce_value).abs() < 1.0e-5,
        "entropy floor should not penalize logits already above the floor: ce={ce_value} penalized={penalized_value}"
    );
}

#[test]
fn marginal_entropy_floor_penalizes_collapsed_batch_distribution() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let collapsed = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(
            vec![
                8.0, 0.0, 0.0, 0.0, //
                8.0, 0.0, 0.0, 0.0, //
                8.0, 0.0, 0.0, 0.0, //
                8.0, 0.0, 0.0, 0.0,
            ],
            [1, 4, 4],
        ),
        &device,
    );
    let diverse = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(
            vec![
                8.0, 0.0, 0.0, 0.0, //
                0.0, 8.0, 0.0, 0.0, //
                0.0, 0.0, 8.0, 0.0, //
                0.0, 0.0, 0.0, 8.0,
            ],
            [1, 4, 4],
        ),
        &device,
    );
    let collapsed_loss =
        tensor_scalar(marginal_entropy_floor_loss_from_logits(collapsed, 2.0).expect("loss"));
    let diverse_loss =
        tensor_scalar(marginal_entropy_floor_loss_from_logits(diverse, 2.0).expect("loss"));
    assert!(
        collapsed_loss > diverse_loss + 1.0,
        "marginal entropy should penalize collapsed predicted support: collapsed={collapsed_loss} diverse={diverse_loss}"
    );
}

#[test]
fn target_marginal_coverage_penalizes_missing_batch_targets() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let targets = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![0, 1, 2, 3], [1, 4]),
        &device,
    );
    let collapsed = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(
            vec![
                6.0, 0.0, 0.0, 0.0, //
                6.0, 0.0, 0.0, 0.0, //
                6.0, 0.0, 0.0, 0.0, //
                6.0, 0.0, 0.0, 0.0,
            ],
            [1, 4, 4],
        ),
        &device,
    );
    let covered = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(
            vec![
                6.0, 0.0, 0.0, 0.0, //
                0.0, 6.0, 0.0, 0.0, //
                0.0, 0.0, 6.0, 0.0, //
                0.0, 0.0, 0.0, 6.0,
            ],
            [1, 4, 4],
        ),
        &device,
    );
    let collapsed_loss = tensor_scalar(
        target_marginal_coverage_loss_from_logits(collapsed, targets.clone(), 1.0e-8)
            .expect("collapsed loss"),
    );
    let covered_loss = tensor_scalar(
        target_marginal_coverage_loss_from_logits(covered, targets, 1.0e-8).expect("covered loss"),
    );
    assert!(
        collapsed_loss > covered_loss + 2.0,
        "target marginal coverage should penalize missing target support: collapsed={collapsed_loss} covered={covered_loss}"
    );
}

#[test]
fn logit_entropy_floor_target_coverage_increases_training_loss() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let plain = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ));
    let coverage_penalized = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_logit_entropy_floor(LogitEntropyFloorConfig {
        enabled: true,
        target_coverage_weight: 0.5,
        ..Default::default()
    });
    let logits = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(
            vec![
                6.0, 0.0, 0.0, 0.0, //
                6.0, 0.0, 0.0, 0.0, //
                6.0, 0.0, 0.0, 0.0, //
                6.0, 0.0, 0.0, 0.0,
            ],
            [1, 4, 4],
        ),
        &device,
    );
    let clean_inputs = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![0, 1, 2, 3], [1, 4]),
        &device,
    );
    let targets = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![0, 1, 2, 3], [1, 4]),
        &device,
    );
    let ce = plain.next_token_loss_from_logits(
        logits.clone(),
        targets.clone(),
        clean_inputs.clone(),
        None,
        None,
    );
    let penalized =
        coverage_penalized.next_token_loss_from_logits(logits, targets, clean_inputs, None, None);
    let ce_value = tensor_scalar(ce);
    let penalized_value = tensor_scalar(penalized);
    assert!(
        penalized_value > ce_value,
        "target coverage should increase loss for collapsed marginal support: ce={ce_value} penalized={penalized_value}"
    );
}

#[test]
fn repeat_unlikelihood_penalizes_configured_history_lags() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let immediate_only = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_repeat_unlikelihood(RepeatUnlikelihoodConfig {
        enabled: true,
        weight: 0.5,
        ..Default::default()
    });
    let lagged = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_repeat_unlikelihood(RepeatUnlikelihoodConfig {
        enabled: true,
        weight: 0.5,
        history_lags: vec![2],
        ..Default::default()
    });
    let logits = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(
            vec![
                0.0, 0.0, 0.0, 0.0, //
                5.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0,
            ],
            [1, 3, 4],
        ),
        &device,
    );
    let clean_inputs =
        Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![0, 1, 2], [1, 3]), &device);
    let targets =
        Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![1, 2, 3], [1, 3]), &device);
    let immediate = immediate_only.next_token_loss_from_logits(
        logits.clone(),
        targets.clone(),
        clean_inputs.clone(),
        None,
        None,
    );
    let lagged = lagged.next_token_loss_from_logits(logits, targets, clean_inputs, None, None);
    let immediate_value = immediate
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("immediate")[0];
    let lagged_value = lagged
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("lagged")[0];
    assert!(
        lagged_value > immediate_value,
        "configured history lag should add unlikelihood loss: immediate={immediate_value} lagged={lagged_value}"
    );
}

#[test]
fn repeat_cycle_lags_respect_budget_and_rotate() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_repeat_unlikelihood(RepeatUnlikelihoodConfig {
        enabled: true,
        cycle_weight: 0.5,
        cycle_min_lag: 2,
        cycle_max_lag: 16,
        cycle_lags_per_step: 4,
        ..Default::default()
    });
    let first = model.repeat_cycle_lags(16);
    assert_eq!(first.len(), 4);
    assert!(first.iter().all(|lag| (2..=16).contains(lag)));
    model
        .gradient_scale_step
        .store(1, std::sync::atomic::Ordering::Relaxed);
    let second = model.repeat_cycle_lags(16);
    assert_eq!(second.len(), 4);
    assert_ne!(first, second);
}

#[test]
fn repeat_unlikelihood_respects_every_steps() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let plain = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ));
    let throttled = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_repeat_unlikelihood(RepeatUnlikelihoodConfig {
        enabled: true,
        weight: 0.5,
        every_steps: 4,
        ..Default::default()
    });
    let logits = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![5.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0], [1, 2, 4]),
        &device,
    );
    let clean_inputs =
        Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![0, 1], [1, 2]), &device);
    let targets =
        Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![1, 2], [1, 2]), &device);
    let ce = tensor_scalar(plain.next_token_loss_from_logits(
        logits.clone(),
        targets.clone(),
        clean_inputs.clone(),
        None,
        None,
    ));
    throttled
        .gradient_scale_step
        .store(2, std::sync::atomic::Ordering::Relaxed);
    let off_cadence = tensor_scalar(throttled.next_token_loss_from_logits(
        logits.clone(),
        targets.clone(),
        clean_inputs.clone(),
        None,
        None,
    ));
    throttled
        .gradient_scale_step
        .store(4, std::sync::atomic::Ordering::Relaxed);
    let on_cadence = tensor_scalar(throttled.next_token_loss_from_logits(
        logits,
        targets,
        clean_inputs,
        None,
        None,
    ));
    assert!(
        (off_cadence - ce).abs() < 1.0e-5,
        "off-cadence repeat loss should match CE: ce={ce} off={off_cadence}"
    );
    assert!(
        on_cadence > ce,
        "on-cadence repeat loss should add penalty: ce={ce} on={on_cadence}"
    );
}

#[test]
fn repeat_cycle_unlikelihood_penalizes_wrong_cycle_predictions() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let plain = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ));
    let cycle_penalized = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_repeat_unlikelihood(RepeatUnlikelihoodConfig {
        enabled: true,
        cycle_weight: 0.5,
        cycle_margin_weight: 0.5,
        cycle_margin: 0.05,
        cycle_min_lag: 2,
        cycle_max_lag: 2,
        cycle_lags_per_step: 1,
        ..Default::default()
    });
    let logits = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(
            vec![
                0.0, 0.0, 0.0, 0.0, //
                5.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0,
            ],
            [1, 3, 4],
        ),
        &device,
    );
    let clean_inputs =
        Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![0, 1, 2], [1, 3]), &device);
    let targets =
        Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![1, 2, 3], [1, 3]), &device);
    let ce = plain.next_token_loss_from_logits(
        logits.clone(),
        targets.clone(),
        clean_inputs.clone(),
        None,
        None,
    );
    let penalized =
        cycle_penalized.next_token_loss_from_logits(logits, targets, clean_inputs, None, None);
    let ce_value = ce.to_data().convert::<f32>().into_vec::<f32>().expect("ce")[0];
    let penalized_value = penalized
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("penalized")[0];
    assert!(
        penalized_value > ce_value,
        "cycle unlikelihood should increase loss for wrong-cycle logits: ce={ce_value} penalized={penalized_value}"
    );
}

#[test]
fn greedy_rollout_recovery_only_skips_stable_hot_path() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_greedy_rollout_unlikelihood(GreedyRolloutUnlikelihoodConfig {
        enabled: true,
        recovery_only: true,
        weight: 0.5,
        prompt_tokens: 1,
        rollout_tokens: 1,
        history_tokens: 1,
        batch_prompts: 1,
        every_steps: 1,
        ..Default::default()
    });
    let clean_inputs = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![0, 1, 2, 3], [1, 4]),
        &device,
    );

    assert!(
        model
            .greedy_rollout_unlikelihood_loss(clean_inputs.clone())
            .is_none(),
        "recovery-only rollout must not run during stable training"
    );
    model.set_recovery_auxiliary_active(true);
    assert!(
        model
            .greedy_rollout_unlikelihood_loss(clean_inputs)
            .is_some(),
        "recovery-only rollout should run when dynamics enters recovery"
    );
}

#[test]
fn greedy_rollout_sequence_recovery_runs_without_step_penalties() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_greedy_rollout_unlikelihood(GreedyRolloutUnlikelihoodConfig {
        enabled: true,
        sequence_recovery_weight: 0.5,
        prompt_tokens: 2,
        rollout_tokens: 2,
        history_tokens: 2,
        batch_prompts: 1,
        every_steps: 1,
        ..Default::default()
    });
    let clean_inputs = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![0, 1, 2, 3, 4], [1, 5]),
        &device,
    );

    let loss = model
        .greedy_rollout_unlikelihood_loss(clean_inputs)
        .expect("sequence recovery should produce a rollout loss");
    let loss = scalar_tensor_to_f64(loss.inner());
    assert!(
        loss.is_finite(),
        "unexpected sequence recovery loss: {loss}"
    );
}

#[test]
fn sdft_train_step_runs_rollout_objective() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_training_objective(TrainingObjectiveConfig::Sdft(SdftObjectiveConfig {
        max_completion_tokens: 2,
        top_k: Some(1),
        ..Default::default()
    }));
    let loss = scalar_loss(TrainStep::step(&model, batch(&device)));
    assert!(loss.is_finite(), "unexpected SDFT loss: {loss}");
}

#[test]
fn latent_reasoning_train_step_runs_next_token_objective() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let mut config = tiny_model_config();
    config.latent_reasoning.enabled = true;
    config.latent_reasoning.max_steps = 2;
    config.latent_reasoning.min_steps = 1;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_latent_reasoning(LatentReasoningTrainingConfig {
            enabled: true,
            jepa_future_offsets: vec![1],
            ..Default::default()
        });
    let loss = scalar_loss(TrainStep::step(&model, batch(&device)));
    assert!(loss.is_finite(), "unexpected latent reasoning loss: {loss}");
}

#[test]
fn train_step_writes_recovery_skip_telemetry_without_policy_batch() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 17);
    let dir = tempfile::tempdir().expect("tempdir");
    let telemetry_path = dir
        .path()
        .join("events")
        .join("ruliad_structured_recovery.jsonl");
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_ruliad_supervision(RuliadSupervisionConfig {
        mode: RuliadSupervisionMode::AnswerCompletion,
        answer_denoising: RuliadAnswerDenoisingConfig {
            enabled: true,
            weight: 0.0,
            structured_recovery_weight: 0.25,
            structured_recovery_every_steps: 1,
            structured_recovery_start_after_steps: 0,
            structured_recovery_max_completion_tokens: 24,
            structured_recovery_negative_count: 1,
            structured_recovery_template_negative_count: 1,
            ..Default::default()
        },
        ..Default::default()
    })
    .with_ruliad_structured_recovery_telemetry_path(Some(telemetry_path.clone()));

    let loss = scalar_loss(TrainStep::step(&model, batch(&device)));
    assert!(loss.is_finite(), "unexpected train loss: {loss}");
    let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
    let event: serde_json::Value =
        serde_json::from_str(content.lines().next().expect("telemetry line"))
            .expect("telemetry json");
    assert_eq!(event["policy_batch_present"].as_bool(), Some(false));
    assert_eq!(event["skip_reason"].as_str(), Some("missing_policy_batch"));
}

#[test]
fn train_step_runs_structured_recovery_with_tbptt_policy_batch() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 19);
    let dir = tempfile::tempdir().expect("tempdir");
    let telemetry_path = dir
        .path()
        .join("events")
        .join("ruliad_structured_recovery.jsonl");
    let mut config = tiny_model_config();
    config.vocab_size = 257;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_tbptt_chunk_size(Some(4))
        .with_ruliad_supervision(RuliadSupervisionConfig {
            mode: RuliadSupervisionMode::AnswerCompletion,
            answer_denoising: RuliadAnswerDenoisingConfig {
                enabled: true,
                weight: 0.0,
                structured_recovery_weight: 0.25,
                structured_recovery_every_steps: 1,
                structured_recovery_start_after_steps: 0,
                structured_recovery_max_completion_tokens: 24,
                structured_recovery_negative_count: 1,
                structured_recovery_template_negative_count: 1,
                ..Default::default()
            },
            ..Default::default()
        })
        .with_ruliad_structured_recovery_telemetry_path(Some(telemetry_path.clone()));
    let item = burn_dragon_universality::RuliadEvalItem {
        oracle_hash: "h0".to_string(),
        sample_index: 45,
        split: burn_dragon_universality::SampleSplit::Train,
        family: "proof_tree".to_string(),
        task_kind: "prove_theorem".to_string(),
        math_domains: vec!["category".to_string(), "formal_proof".to_string()],
        reasoning_modes: vec!["equational".to_string()],
        prompt: "?:ss\n!:".to_string(),
        expected_answer: "ok=1;l=17;r=17".to_string(),
        difficulty_level: Some(0),
        spec: None,
    };
    let policy_batch = Arc::new(crate::dataset::RuliadPolicyBatch {
        samples: vec![crate::dataset::RuliadPolicySample {
            item,
            prompt_tokens: vec![1, 2, 3],
        }],
        tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
            vocab_size: 257,
            eos_id: None,
        },
        stop_token_id: None,
        sampling_metadata: None,
    });
    let train_batch = SequenceBatch::new(
        Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(
                vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                [2, 8],
            ),
            &device,
        ),
        Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(
                vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                [2, 8],
            ),
            &device,
        ),
        None,
    )
    .with_ruliad_policy_batch(Some(policy_batch));

    let loss = scalar_loss(TrainStep::step(&model, train_batch));
    assert!(loss.is_finite(), "unexpected train loss: {loss}");
    let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
    let event: serde_json::Value =
        serde_json::from_str(content.lines().next().expect("telemetry line"))
            .expect("telemetry json");
    assert_eq!(event["policy_batch_present"].as_bool(), Some(true));
    assert!(
        event["recovery_rows"].as_u64().unwrap_or_default() > 0,
        "expected active recovery rows, event={event}"
    );
}

#[test]
fn train_step_runs_field_binding_contrast_with_tbptt_policy_batch() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 37);
    let dir = tempfile::tempdir().expect("tempdir");
    let telemetry_path = dir
        .path()
        .join("events")
        .join("ruliad_field_binding_contrast.jsonl");
    let mut config = tiny_model_config();
    config.vocab_size = 257;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_tbptt_chunk_size(Some(8))
        .with_tbptt_persist_across_steps(true)
        .with_ruliad_supervision(RuliadSupervisionConfig {
            mode: RuliadSupervisionMode::AnswerCompletion,
            verifier_reward: crate::config::train::RuliadVerifierRewardConfig {
                enabled: true,
                weight: 0.0,
                field_binding_contrast_weight: 0.25,
                field_binding_contrast_every_steps: 1,
                field_binding_contrast_start_after_steps: 0,
                field_binding_contrast_max_pairs: 4,
                field_binding_contrast_replay_capacity: 0,
                max_completion_tokens: 24,
                ..Default::default()
            },
            ..Default::default()
        })
        .with_ruliad_field_binding_contrast_telemetry_path(Some(telemetry_path.clone()));
    let item = burn_dragon_universality::RuliadEvalItem {
        oracle_hash: "h0".to_string(),
        sample_index: 56,
        split: burn_dragon_universality::SampleSplit::Train,
        family: "proof_tree".to_string(),
        task_kind: "prove_theorem".to_string(),
        math_domains: vec!["category".to_string(), "formal_proof".to_string()],
        reasoning_modes: vec!["equational".to_string()],
        prompt: "?:fb\n!:".to_string(),
        expected_answer: "ok=1;l=17;r=17".to_string(),
        difficulty_level: Some(0),
        spec: None,
    };
    let policy_batch = Arc::new(crate::dataset::RuliadPolicyBatch {
        samples: vec![crate::dataset::RuliadPolicySample {
            item,
            prompt_tokens: vec![1, 2, 3],
        }],
        tokenization: burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
            vocab_size: 257,
            eos_id: None,
        },
        stop_token_id: None,
        sampling_metadata: None,
    });
    let inputs = (0..64)
        .map(|value| (value % 128) as i64)
        .collect::<Vec<_>>();
    let targets = (1..65)
        .map(|value| (value % 128) as i64)
        .collect::<Vec<_>>();
    let train_batch = SequenceBatch::new(
        Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(inputs, [2, 32]), &device),
        Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(targets, [2, 32]), &device),
        None,
    )
    .with_ruliad_policy_batch(Some(policy_batch));

    let loss = scalar_loss(TrainStep::step(&model, train_batch));
    assert!(loss.is_finite(), "unexpected train loss: {loss}");
    let content = std::fs::read_to_string(&telemetry_path).expect("telemetry sidecar");
    let event: serde_json::Value =
        serde_json::from_str(content.lines().next().expect("telemetry line"))
            .expect("field-binding telemetry json");
    assert_eq!(event["sample_groups"], 1);
    assert!(
        event["contrast_pairs"].as_u64().unwrap_or_default() > 0,
        "expected active field-binding contrast rows under TBPTT, event={event}"
    );
    assert!(
        event["rank_metric_tokens"].as_u64().unwrap_or_default() > 0,
        "expected field-binding rank telemetry under TBPTT, event={event}"
    );
}

#[test]
fn latent_energy_margin_loss_prefers_lower_positive_energy() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let low_positive =
        Tensor::<TestBackend, 3>::from_data(TensorData::new(vec![0.0, 0.1], [1, 2, 1]), &device);
    let high_negative =
        Tensor::<TestBackend, 3>::from_data(TensorData::new(vec![3.0, 2.5], [1, 2, 1]), &device);
    let high_positive =
        Tensor::<TestBackend, 3>::from_data(TensorData::new(vec![3.0, 2.5], [1, 2, 1]), &device);
    let low_negative =
        Tensor::<TestBackend, 3>::from_data(TensorData::new(vec![0.0, 0.1], [1, 2, 1]), &device);

    let preferred = tensor_scalar(latent_energy_contrastive_margin_loss(
        low_positive,
        high_negative,
        1.0,
    ));
    let inverted = tensor_scalar(latent_energy_contrastive_margin_loss(
        high_positive,
        low_negative,
        1.0,
    ));
    assert!(
        inverted > preferred + 2.0,
        "contrastive energy should prefer low positives: preferred={preferred} inverted={inverted}"
    );
}

#[test]
fn latent_energy_monotonic_penalty_catches_ascending_energy() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let previous =
        Tensor::<TestBackend, 3>::from_data(TensorData::new(vec![1.0, 1.0], [1, 2, 1]), &device);
    let descending =
        Tensor::<TestBackend, 3>::from_data(TensorData::new(vec![0.75, 0.5], [1, 2, 1]), &device);
    let ascending =
        Tensor::<TestBackend, 3>::from_data(TensorData::new(vec![1.25, 1.5], [1, 2, 1]), &device);

    let descending = tensor_scalar(latent_energy_monotonic_penalty(
        previous.clone(),
        descending,
        0.0,
    ));
    let ascending = tensor_scalar(latent_energy_monotonic_penalty(previous, ascending, 0.0));
    assert!(
        descending <= 1.0e-6,
        "descending energy should have no monotonic penalty: {descending}"
    );
    assert!(
        ascending > 0.25,
        "ascending energy should be penalized: {ascending}"
    );
}

#[test]
fn latent_energy_contractivity_penalty_catches_large_hidden_drift() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let target =
        Tensor::<TestBackend, 3>::from_data(TensorData::new(vec![1.0, -1.0], [1, 1, 2]), &device);
    let close =
        Tensor::<TestBackend, 3>::from_data(TensorData::new(vec![1.05, -0.95], [1, 1, 2]), &device);
    let far =
        Tensor::<TestBackend, 3>::from_data(TensorData::new(vec![3.0, -3.0], [1, 1, 2]), &device);

    let close = tensor_scalar(latent_energy_contractivity_penalty(
        close,
        target.clone(),
        0.5,
    ));
    let far = tensor_scalar(latent_energy_contractivity_penalty(far, target, 0.5));
    assert!(
        close <= 1.0e-6,
        "nearby hidden states should fit within the trust radius: {close}"
    );
    assert!(
        far > close + 1.0,
        "large hidden drift should be penalized: close={close} far={far}"
    );
}

#[test]
fn latent_energy_model_train_step_runs() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let mut config = tiny_model_config();
    config.latent_reasoning.enabled = true;
    config.latent_reasoning.max_steps = 2;
    config.latent_reasoning.min_steps = 2;
    config.latent_reasoning.energy_head = true;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_latent_reasoning(LatentReasoningTrainingConfig {
            enabled: true,
            jepa_future_offsets: vec![usize::MAX],
            energy_model: crate::config::LatentEnergyModelConfig {
                enabled: true,
                max_rollout_steps_for_loss: 2,
                ..Default::default()
            },
            sigreg: LatentReasoningSigRegConfig {
                enabled: false,
                ..Default::default()
            },
            constraint_balancer: LatentReasoningConstraintBalancerConfig {
                normalized_aux_scale: 0.01,
                ..Default::default()
            },
            ..Default::default()
        });
    let loss = scalar_loss(TrainStep::step(&model, batch(&device)));
    assert!(loss.is_finite(), "unexpected latent EBM loss: {loss}");
}

#[test]
fn latent_step_contract_train_step_runs_and_records_components() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    crate::train::profile::reset();
    let mut config = tiny_model_config();
    config.latent_reasoning.enabled = true;
    config.latent_reasoning.max_steps = 2;
    config.latent_reasoning.min_steps = 2;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_latent_reasoning(LatentReasoningTrainingConfig {
            enabled: true,
            jepa_future_offsets: vec![usize::MAX],
            step_contract: LatentStepContractConfig {
                enabled: true,
                max_rollout_steps_for_loss: 2,
                ce_weight: 0.1,
                monotonic_ce_weight: 0.5,
                contractive_weight: 0.05,
                ..Default::default()
            },
            sigreg: LatentReasoningSigRegConfig {
                enabled: false,
                ..Default::default()
            },
            constraint_balancer: LatentReasoningConstraintBalancerConfig {
                normalized_aux_scale: 0.01,
                ..Default::default()
            },
            ..Default::default()
        });
    let loss = scalar_loss(TrainStep::step(&model, batch(&device)));
    assert!(
        loss.is_finite(),
        "unexpected latent step contract loss: {loss}"
    );
    let snapshot = crate::train::profile::take_latent_reasoning();
    assert!(
        snapshot.step_contract_components > 0,
        "step contract should record active components: {snapshot:?}"
    );
}

#[test]
fn latent_reasoning_step_diagnostics_are_finite() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let mut config = tiny_model_config();
    config.latent_reasoning.enabled = true;
    config.latent_reasoning.max_steps = 3;
    config.latent_reasoning.min_steps = 3;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device));

    let diagnostics = model
        .latent_reasoning_step_diagnostics(batch(&device))
        .expect("latent diagnostics");
    assert_eq!(diagnostics.step_loss.len(), 3);
    assert_eq!(diagnostics.step_ce_delta.len(), 3);
    assert_eq!(diagnostics.step_ce_monotonic_violation_rate.len(), 3);
    assert_eq!(diagnostics.step_entropy_bits.len(), 3);
    assert_eq!(diagnostics.step_delta_rms.len(), 3);
    assert_eq!(diagnostics.step_raw_cosine.len(), 3);
    for value in [
        diagnostics.raw_loss,
        diagnostics.final_loss,
        diagnostics.raw_entropy_bits,
        diagnostics.final_entropy_bits,
        diagnostics.final_delta_rms,
        diagnostics.final_raw_cosine,
    ]
    .into_iter()
    .chain(diagnostics.step_loss)
    .chain(diagnostics.step_ce_delta)
    .chain(diagnostics.step_ce_monotonic_violation_rate)
    .chain(diagnostics.step_entropy_bits)
    .chain(diagnostics.step_delta_rms)
    .chain(diagnostics.step_raw_cosine)
    {
        assert!(
            value.is_finite(),
            "diagnostic value was not finite: {value}"
        );
    }
}

#[test]
fn latent_reasoning_step_diagnostics_include_energy_when_head_enabled() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let mut config = tiny_model_config();
    config.latent_reasoning.enabled = true;
    config.latent_reasoning.max_steps = 3;
    config.latent_reasoning.min_steps = 3;
    config.latent_reasoning.energy_head = true;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device));

    let diagnostics = model
        .latent_reasoning_step_diagnostics(batch(&device))
        .expect("latent diagnostics");
    assert_eq!(diagnostics.step_energy_mean.len(), 3);
    assert_eq!(diagnostics.step_energy_delta.len(), 3);
    assert_eq!(diagnostics.step_energy_monotonic_violation_rate.len(), 3);
    assert!(diagnostics.best_energy_step.is_some());
    for value in diagnostics
        .step_energy_mean
        .into_iter()
        .chain(diagnostics.step_energy_delta)
        .chain(diagnostics.step_energy_monotonic_violation_rate)
    {
        assert!(
            value.is_finite(),
            "energy diagnostic value was not finite: {value}"
        );
    }
}

#[test]
fn latent_reasoning_auxiliary_scale_respects_start_after_steps() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let mut config = tiny_model_config();
    config.latent_reasoning.enabled = true;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_latent_reasoning(LatentReasoningTrainingConfig {
            enabled: true,
            every_steps: 1,
            jepa_future_offsets: vec![1],
            constraint_balancer: LatentReasoningConstraintBalancerConfig {
                normalized_aux_scale: 0.25,
                start_after_steps: 2,
                ..Default::default()
            },
            ..Default::default()
        });

    model.gradient_scale_step.store(0, Ordering::Relaxed);
    assert_eq!(model.latent_reasoning_auxiliary_scale(), None);
    model.gradient_scale_step.store(1, Ordering::Relaxed);
    assert_eq!(model.latent_reasoning_auxiliary_scale(), None);
    model.gradient_scale_step.store(2, Ordering::Relaxed);
    assert_eq!(model.latent_reasoning_auxiliary_scale(), Some(0.25));
}

#[test]
fn proof_policy_schedule_accepts_authoritative_batch_step() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_ruliad_supervision(RuliadSupervisionConfig {
        proof_policy: crate::config::RuliadProofPolicyTrainingConfig {
            enabled: true,
            weight: 1.0,
            every_steps: 16,
            start_after_steps: 16,
            ..Default::default()
        },
        ..Default::default()
    });

    model.gradient_scale_step.store(7, Ordering::Relaxed);
    assert_eq!(model.ruliad_proof_policy_dagger_weight(), 0.0);
    assert_eq!(model.ruliad_proof_policy_dagger_weight_at_step(16), 1.0);
    assert_eq!(model.ruliad_proof_policy_dagger_weight_at_step(17), 0.0);
}

#[test]
fn latent_reasoning_auxiliary_scale_can_wait_for_capability_gate() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let mut config = tiny_model_config();
    config.latent_reasoning.enabled = true;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_latent_reasoning(LatentReasoningTrainingConfig {
            enabled: true,
            every_steps: 1,
            start_after_capability_gate_passed: true,
            jepa_future_offsets: vec![1],
            constraint_balancer: LatentReasoningConstraintBalancerConfig {
                normalized_aux_scale: 0.25,
                ..Default::default()
            },
            ..Default::default()
        });

    model.gradient_scale_step.store(32, Ordering::Relaxed);
    assert_eq!(model.latent_reasoning_auxiliary_scale(), None);
    model.set_latent_reasoning_capability_gate_open(true);
    assert_eq!(model.latent_reasoning_auxiliary_scale(), Some(0.25));
}

#[test]
fn latent_reasoning_auxiliary_scale_respects_per_objective_every_steps() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let mut config = tiny_model_config();
    config.latent_reasoning.enabled = true;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_latent_reasoning(LatentReasoningTrainingConfig {
            enabled: true,
            every_steps: 8,
            jepa_every_steps: Some(8),
            jepa_future_offsets: vec![1],
            next_latent: NextLatentPredictionConfig {
                enabled: true,
                every_steps: Some(16),
                start_after_steps: Some(8),
                ..Default::default()
            },
            constraint_balancer: LatentReasoningConstraintBalancerConfig {
                normalized_aux_scale: 0.25,
                ..Default::default()
            },
            ..Default::default()
        });

    model.gradient_scale_step.store(7, Ordering::Relaxed);
    assert_eq!(
        model.latent_reasoning_auxiliary_scale_for_every_steps(
            model.latent_reasoning_jepa_every_steps()
        ),
        Some(0.25)
    );
    assert_eq!(
        model.latent_reasoning_auxiliary_scale_for_schedule(
            model.latent_reasoning_next_latent_every_steps(),
            model.latent_reasoning_next_latent_start_after_steps(),
            model.latent_reasoning_next_latent_start_policy()
        ),
        None
    );
    model.gradient_scale_step.store(15, Ordering::Relaxed);
    assert_eq!(
        model.latent_reasoning_auxiliary_scale_for_schedule(
            model.latent_reasoning_next_latent_every_steps(),
            model.latent_reasoning_next_latent_start_after_steps(),
            model.latent_reasoning_next_latent_start_policy()
        ),
        Some(0.25)
    );
}

#[test]
fn latent_reasoning_start_policy_can_gate_specific_objectives() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let mut config = tiny_model_config();
    config.latent_reasoning.enabled = true;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_latent_reasoning(LatentReasoningTrainingConfig {
            enabled: true,
            every_steps: 1,
            jepa_start_policy: Some(LatentReasoningAuxiliaryStartPolicy::FixedStep),
            jepa_future_offsets: vec![1],
            next_latent: NextLatentPredictionConfig {
                enabled: true,
                start_policy: Some(LatentReasoningAuxiliaryStartPolicy::FixedStepAndCapabilityGate),
                ..Default::default()
            },
            constraint_balancer: LatentReasoningConstraintBalancerConfig {
                normalized_aux_scale: 0.25,
                start_after_steps: 4,
                ..Default::default()
            },
            ..Default::default()
        });

    model.gradient_scale_step.store(4, Ordering::Relaxed);
    assert_eq!(
        model.latent_reasoning_auxiliary_scale_for_schedule(
            model.latent_reasoning_jepa_every_steps(),
            model.latent_reasoning_jepa_start_after_steps(),
            model.latent_reasoning_jepa_start_policy()
        ),
        Some(0.25)
    );
    assert_eq!(
        model.latent_reasoning_auxiliary_scale_for_schedule(
            model.latent_reasoning_next_latent_every_steps(),
            model.latent_reasoning_next_latent_start_after_steps(),
            model.latent_reasoning_next_latent_start_policy()
        ),
        None
    );
    model.set_latent_reasoning_capability_gate_open(true);
    assert_eq!(
        model.latent_reasoning_auxiliary_scale_for_schedule(
            model.latent_reasoning_next_latent_every_steps(),
            model.latent_reasoning_next_latent_start_after_steps(),
            model.latent_reasoning_next_latent_start_policy()
        ),
        Some(0.25)
    );
}

#[test]
fn latent_reasoning_capability_gate_policy_can_ignore_fixed_step_start() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let mut config = tiny_model_config();
    config.latent_reasoning.enabled = true;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_latent_reasoning(LatentReasoningTrainingConfig {
            enabled: true,
            every_steps: 1,
            next_latent: NextLatentPredictionConfig {
                enabled: true,
                start_after_steps: Some(512),
                start_policy: Some(LatentReasoningAuxiliaryStartPolicy::CapabilityGate),
                ..Default::default()
            },
            constraint_balancer: LatentReasoningConstraintBalancerConfig {
                normalized_aux_scale: 0.25,
                ..Default::default()
            },
            ..Default::default()
        });

    model.gradient_scale_step.store(0, Ordering::Relaxed);
    assert_eq!(
        model.latent_reasoning_auxiliary_scale_for_schedule(
            model.latent_reasoning_next_latent_every_steps(),
            model.latent_reasoning_next_latent_start_after_steps(),
            model.latent_reasoning_next_latent_start_policy()
        ),
        None
    );
    model.set_latent_reasoning_capability_gate_open(true);
    assert_eq!(
        model.latent_reasoning_auxiliary_scale_for_schedule(
            model.latent_reasoning_next_latent_every_steps(),
            model.latent_reasoning_next_latent_start_after_steps(),
            model.latent_reasoning_next_latent_start_policy()
        ),
        Some(0.25)
    );
}

#[test]
fn latent_reasoning_global_capability_gate_remains_compatibility_default() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let mut config = tiny_model_config();
    config.latent_reasoning.enabled = true;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_latent_reasoning(LatentReasoningTrainingConfig {
            enabled: true,
            every_steps: 1,
            start_after_capability_gate_passed: true,
            jepa_future_offsets: vec![1],
            constraint_balancer: LatentReasoningConstraintBalancerConfig {
                normalized_aux_scale: 0.25,
                start_after_steps: 2,
                ..Default::default()
            },
            ..Default::default()
        });

    model.gradient_scale_step.store(2, Ordering::Relaxed);
    assert_eq!(
        model.latent_reasoning_jepa_start_policy(),
        LatentReasoningAuxiliaryStartPolicy::FixedStepAndCapabilityGate
    );
    assert_eq!(model.latent_reasoning_auxiliary_scale(), None);
    model.set_latent_reasoning_capability_gate_open(true);
    assert_eq!(model.latent_reasoning_auxiliary_scale(), Some(0.25));
}

#[test]
fn next_latent_train_step_runs_without_inference_latent_reasoning() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let mut config = tiny_model_config();
    config.next_latent_transition.enabled = true;
    let mut model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_latent_reasoning(LatentReasoningTrainingConfig {
            enabled: true,
            jepa_future_offsets: vec![usize::MAX],
            next_latent: NextLatentPredictionConfig {
                enabled: true,
                horizon: 2,
                regression_weight: 1.0,
                token_kl_weight: 0.01,
                smooth_l1_beta: 1.0,
                detach_action_embedding: true,
                ..Default::default()
            },
            sigreg: LatentReasoningSigRegConfig {
                enabled: false,
                ..Default::default()
            },
            constraint_balancer: LatentReasoningConstraintBalancerConfig {
                normalized_aux_scale: 0.01,
                ..Default::default()
            },
            ..Default::default()
        });
    model.next_latent_token_layout = Some(Default::default());
    assert!(!model.model.latent_reasoning_enabled());
    assert!(model.model.next_latent_transition_enabled());
    let loss = scalar_loss(TrainStep::step(&model, batch(&device)));
    assert!(loss.is_finite(), "unexpected NextLat loss: {loss}");
}

#[test]
fn dragon_state_consistency_train_step_runs_without_latent_reasoning_architecture() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let mut config = tiny_model_config();
    config.latent_reasoning.enabled = false;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_latent_reasoning(LatentReasoningTrainingConfig {
            enabled: true,
            jepa_future_offsets: vec![usize::MAX],
            dragon_state: DragonStateConsistencyConfig {
                enabled: true,
                rho_weight: 1.0,
                rho_energy_weight: 0.25,
                smooth_l1_beta: 1.0,
                max_rho_slots: 4,
                ..Default::default()
            },
            constraint_balancer: LatentReasoningConstraintBalancerConfig {
                normalized_aux_scale: 0.01,
                ..Default::default()
            },
            ..Default::default()
        });
    assert!(!model.model.latent_reasoning_enabled());
    let loss = scalar_loss(TrainStep::step(&model, batch(&device)));
    assert!(
        loss.is_finite(),
        "unexpected Dragon state consistency loss: {loss}"
    );
}

#[test]
fn latent_reasoning_rho_memory_sigreg_train_step_runs() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let mut config = tiny_model_config();
    config.latent_reasoning.enabled = true;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_latent_reasoning(LatentReasoningTrainingConfig {
            enabled: true,
            jepa_future_offsets: vec![usize::MAX],
            sigreg: LatentReasoningSigRegConfig {
                enabled: true,
                target: crate::config::LatentReasoningSigRegTarget::RhoMemorySlots,
                ..Default::default()
            },
            constraint_balancer: LatentReasoningConstraintBalancerConfig {
                normalized_aux_scale: 0.01,
                ..Default::default()
            },
            ..Default::default()
        });
    let loss = scalar_loss(TrainStep::step(&model, batch(&device)));
    assert!(
        loss.is_finite(),
        "unexpected rho-memory latent reasoning loss: {loss}"
    );
}

#[test]
fn rho_memory_sigreg_train_step_runs_without_latent_reasoning_architecture() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let mut config = tiny_model_config();
    config.latent_reasoning.enabled = false;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_latent_reasoning(LatentReasoningTrainingConfig {
            enabled: true,
            jepa_future_offsets: vec![usize::MAX],
            sigreg: LatentReasoningSigRegConfig {
                enabled: true,
                target: LatentReasoningSigRegTarget::RhoMemorySlots,
                ..Default::default()
            },
            constraint_balancer: LatentReasoningConstraintBalancerConfig {
                normalized_aux_scale: 0.01,
                ..Default::default()
            },
            ..Default::default()
        });
    let loss = scalar_loss(TrainStep::step(&model, batch(&device)));
    assert!(
        loss.is_finite(),
        "unexpected rho-memory regularized base Dragon loss: {loss}"
    );
}

#[test]
fn rho_memory_sigreg_penalizes_redundant_slots() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_latent_reasoning(LatentReasoningTrainingConfig {
        enabled: true,
        sigreg: LatentReasoningSigRegConfig {
            enabled: true,
            target: LatentReasoningSigRegTarget::RhoMemorySlots,
            ..Default::default()
        },
        ..Default::default()
    });
    let mut duplicate_state = model.model.init_state_ephemeral();
    duplicate_state.layers[0].rho = Some(Tensor::<TestBackend, 4>::from_data(
        TensorData::new(vec![1.0, -1.0, 0.0, 1.0, -1.0, 0.0], [1, 1, 2, 3]),
        &device,
    ));
    let mut orthogonal_state = model.model.init_state_ephemeral();
    orthogonal_state.layers[0].rho = Some(Tensor::<TestBackend, 4>::from_data(
        TensorData::new(vec![1.0, -1.0, 0.0, 1.0, 1.0, -2.0], [1, 1, 2, 3]),
        &device,
    ));

    let duplicate = tensor_scalar(
        model
            .sigreg_loss_from_rho_memory_state(&duplicate_state)
            .expect("duplicate rho loss"),
    );
    let orthogonal = tensor_scalar(
        model
            .sigreg_loss_from_rho_memory_state(&orthogonal_state)
            .expect("orthogonal rho loss"),
    );

    assert!(
        duplicate > orthogonal + 0.5,
        "duplicate slots should be penalized more strongly: duplicate={duplicate} orthogonal={orthogonal}"
    );
    assert!(
        orthogonal < 1.0e-5,
        "centered orthogonal slots should have near-zero redundancy penalty: {orthogonal}"
    );
}

#[test]
fn rho_memory_sigreg_samples_slots_deterministically() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_latent_reasoning(LatentReasoningTrainingConfig {
        enabled: true,
        sigreg: LatentReasoningSigRegConfig {
            enabled: true,
            target: LatentReasoningSigRegTarget::RhoMemorySlots,
            max_rho_slots: 3,
            ..Default::default()
        },
        ..Default::default()
    });
    let rho = Tensor::<TestBackend, 4>::from_data(
        TensorData::new(vec![0.0, 1.0, 2.0, 3.0, 4.0], [1, 1, 5, 1]),
        &device,
    );

    let sampled = model.sigreg_sample_rho_slots(rho, 5);
    assert_eq!(sampled.shape().dims::<4>(), [1, 1, 3, 1]);
    let values = sampled
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("sampled rho");
    assert_eq!(values, vec![0.0, 2.0, 4.0]);
}

#[test]
fn dragon_state_consistency_is_zero_for_matching_rho_and_positive_for_drift() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_latent_reasoning(LatentReasoningTrainingConfig {
        enabled: true,
        dragon_state: DragonStateConsistencyConfig {
            enabled: true,
            rho_weight: 1.0,
            rho_energy_weight: 1.0,
            smooth_l1_beta: 1.0,
            max_rho_slots: 2,
            ..Default::default()
        },
        ..Default::default()
    });
    let mut student_state = model.model.init_state_ephemeral();
    student_state.layers[0].rho = Some(Tensor::<TestBackend, 4>::from_data(
        TensorData::new(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [1, 1, 2, 3]),
        &device,
    ));
    let teacher_state = student_state.clone();

    let (matching_loss, matching_components) =
        model.dragon_state_consistency_loss(&student_state, &teacher_state);
    assert_eq!(matching_components, 2);
    let matching_loss = tensor_scalar(matching_loss.expect("matching rho loss"));
    assert!(
        matching_loss.abs() < 1.0e-6,
        "matching rho state should have zero consistency loss: {matching_loss}"
    );

    let mut drifted_teacher_state = model.model.init_state_ephemeral();
    drifted_teacher_state.layers[0].rho = Some(Tensor::<TestBackend, 4>::from_data(
        TensorData::new(vec![1.0, 0.0, 0.0, 0.0, -1.0, 0.0], [1, 1, 2, 3]),
        &device,
    ));
    let (drift_loss, drift_components) =
        model.dragon_state_consistency_loss(&student_state, &drifted_teacher_state);
    assert_eq!(drift_components, 2);
    let drift_loss = tensor_scalar(drift_loss.expect("drift rho loss"));
    assert!(
        drift_loss > matching_loss + 0.1,
        "drifted rho rows should be penalized: matching={matching_loss} drift={drift_loss}"
    );
}

#[test]
fn sigreg_combined_target_enables_hidden_and_rho_losses() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_latent_reasoning(LatentReasoningTrainingConfig {
        enabled: true,
        sigreg: LatentReasoningSigRegConfig {
            enabled: true,
            target: LatentReasoningSigRegTarget::HiddenAndRhoMemorySlots,
            ..Default::default()
        },
        ..Default::default()
    });
    let hidden = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![0.0, 0.1, 0.2, 0.3], [1, 2, 2]),
        &device,
    );
    let mut state = model.model.init_state_ephemeral();
    state.layers[0].rho = Some(Tensor::<TestBackend, 4>::from_data(
        TensorData::new(vec![1.0, -1.0, 0.0, 1.0, -1.0, 0.0], [1, 1, 2, 3]),
        &device,
    ));

    assert!(model.sigreg_loss_from_hidden(hidden).is_some());
    assert!(model.sigreg_loss_from_rho_memory_state(&state).is_some());
}

#[test]
fn sdpo_train_step_runs_rollout_objective() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_training_objective(TrainingObjectiveConfig::Sdpo(SdpoObjectiveConfig {
        group_size: 2,
        max_completion_tokens: 2,
        top_k: Some(1),
        ..Default::default()
    }));
    let loss = scalar_loss(TrainStep::step(&model, batch(&device)));
    assert!(loss.is_finite(), "unexpected SDPO loss: {loss}");
}

#[test]
#[should_panic(expected = "paper-aligned SDFT/SDPO rollout objectives require flat token logits")]
fn sdft_train_step_guards_factorized_language_head() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_factorized_model_config(),
        &device,
    ))
    .with_training_objective(TrainingObjectiveConfig::Sdft(SdftObjectiveConfig {
        max_completion_tokens: 2,
        top_k: Some(1),
        ..Default::default()
    }));
    let _ = TrainStep::step(&model, batch(&device));
}

#[test]
fn sdft_train_step_updates_teacher_runtime() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_training_objective(TrainingObjectiveConfig::Sdft(SdftObjectiveConfig {
        max_completion_tokens: 2,
        top_k: Some(1),
        teacher_update_rate: 0.5,
        ..Default::default()
    }));
    let _ = scalar_loss(TrainStep::step(&model, batch(&device)));
    let update_count = model
        .teacher_update_count_for_test()
        .expect("teacher update count");
    assert_eq!(update_count, 1);
}

#[test]
fn rollout_teacher_context_contains_gold_demonstration() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ));
    let inputs = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![0, 1, 2, 3], [1, 4]),
        &device,
    );
    let targets = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![1, 2, 9, 10], [1, 4]),
        &device,
    );
    let rollout = model.rollout_score_batch(
        &model.model,
        inputs,
        targets,
        RolloutScoreConfig {
            max_completion_tokens: 2,
            group_size: 1,
            temperature: 1.0,
            top_k: Some(1),
            num_loss_tokens_to_skip: 0,
            max_reprompt_len: usize::MAX,
            reprompt_truncation: RepromptTruncation::Right,
        },
    );
    let teacher_inputs = rollout
        .teacher_inputs
        .to_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .expect("teacher input vec");
    assert_eq!(teacher_inputs[0], 2);
    assert_eq!(teacher_inputs[1], 9);
}

#[test]
fn sdft_sdpo_composite_train_step_runs() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        tiny_model_config(),
        &device,
    ))
    .with_training_objective(TrainingObjectiveConfig::SdftSdpo(SdftSdpoObjectiveConfig {
        sdft: SdftObjectiveConfig {
            max_completion_tokens: 2,
            top_k: Some(1),
            ..Default::default()
        },
        sdpo: SdpoObjectiveConfig {
            group_size: 2,
            max_completion_tokens: 2,
            top_k: Some(1),
            ..Default::default()
        },
        ..Default::default()
    }));
    let loss = scalar_loss(TrainStep::step(&model, batch(&device)));
    assert!(loss.is_finite(), "unexpected composite loss: {loss}");
}

#[test]
fn sdpo_train_step_runs_with_single_process_pipeline_plan() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let mut config = tiny_model_config();
    config.n_layer = 2;
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device))
        .with_pipeline_plan(Some(tiny_pipeline_plan()))
        .with_training_objective(TrainingObjectiveConfig::Sdpo(SdpoObjectiveConfig {
            group_size: 2,
            max_completion_tokens: 2,
            top_k: Some(1),
            ..Default::default()
        }));
    let loss = scalar_loss(TrainStep::step(&model, batch(&device)));
    assert!(loss.is_finite(), "unexpected pipeline SDPO loss: {loss}");
}
