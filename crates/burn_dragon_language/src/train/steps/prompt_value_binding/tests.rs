use super::*;

type TestBackend = burn_autodiff::Autodiff<burn_ndarray::NdArray<f32>>;

#[test]
fn required_binding_records_failure_before_a_context_only_batch_can_skip() {
    for algorithm in [
        TrainingAlgorithm::Backpropagation,
        TrainingAlgorithm::PredictiveCoding,
    ] {
        let device = Default::default();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("binding.jsonl");
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            DragonConfig {
                n_embd: 8,
                n_head: 1,
                n_layer: 1,
                mlp_internal_dim_multiplier: 1,
                vocab_size: 16,
                dropout: 0.0,
                ..Default::default()
            },
            &device,
        ))
        .with_training_algorithm(algorithm)
        .with_ruliad_supervision(RuliadSupervisionConfig {
            prompt_value_binding: crate::config::RuliadPromptValueBindingConfig {
                enabled: true,
                require_scheduled_update: true,
                every_steps: 1,
                phase_steps: 0,
                ..Default::default()
            },
            ..Default::default()
        })
        .with_ruliad_prompt_value_binding_telemetry_path(Some(path.clone()));
        let batch = SequenceBatch {
            inputs: Tensor::zeros([1, 4], &device),
            targets: Tensor::zeros([1, 4], &device),
            loss_mask: Some(Tensor::zeros([1, 4], &device)),
            supervised_token_count: Some(0),
            summary_event_mask: None,
            ruliad_policy_batch: None,
            absolute_step: Some(0),
            reset_stream_state: true,
        };
        let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            burn_train::TrainStep::step(&model, batch)
        }));
        assert!(failure.is_err());
        let text = std::fs::read_to_string(path).unwrap();
        let event: serde_json::Value = serde_json::from_str(text.trim()).unwrap();
        assert_eq!(event["skip_reason"], "missing_or_empty_policy_batch");
        assert_eq!(event["active_tokens"], 0);
    }
}
