use super::*;
use crate::config::TrainingObjectiveConfig;
use crate::config::load_training_config;
use crate::config::train::RuliadSupervisionMode;
use crate::inference::build_model_config;
use burn_dragon_core::{
    HierarchicalDragonSharing, RotaryEmbedding, SequenceKernelConfig, SequenceMemorySystem,
};
use burn_dragon_train::OptimizerKind;
use std::path::{Path, PathBuf};

fn parse_config(extra_training: &str) -> TrainingConfig {
    let toml = format!(
        r#"
[dataset]
cache_dir = "target/test-cache"
type = "nemotron_climb_mix"
max_records = 4

[training]
block_size = 8
batch_size = 2
max_iters = 1
log_frequency = 1
{extra_training}

[optimizer]
learning_rate = 0.001
weight_decay = 0.0

[generation]
prompt = ""
"#
    );
    toml::from_str(&toml).expect("training config should parse")
}

#[test]
fn default_objective_is_next_token() {
    let config = parse_config("");
    assert!(config.training.objective.is_next_token());
    config.validate().expect("default objective validates");
}

#[test]
fn validation_defaults_to_a_training_seed_independent_fixed_holdout() {
    let first = parse_config("seed = 1");
    let second = parse_config("seed = 2");
    assert_eq!(
        first.training.validation.sampling,
        crate::config::TrainingValidationSampling::FixedHoldout
    );
    assert_eq!(
        first.training.validation.objective,
        crate::config::TrainingValidationObjective::FixedHoldout
    );
    assert_eq!(
        first.training.validation.seed,
        second.training.validation.seed
    );
    assert_ne!(first.training.seed, second.training.seed);
}

#[test]
fn live_source_selected_validation_cannot_masquerade_as_a_fixed_holdout() {
    let config =
        parse_config("\n[training.validation]\nsampling = \"live_source_selection\"\nseed = 19");
    assert!(
        config
            .training
            .validation
            .sampling
            .uses_live_source_selection()
    );
    assert_eq!(config.training.validation.seed, 19);
    let error = config
        .validate()
        .expect_err("a live source sample is not a fixed holdout");
    assert!(error.to_string().contains("objective=fixed_holdout"));
}

#[test]
fn source_weighted_validation_objective_requires_ruliad_batches() {
    let mut config = parse_config("");
    config.training.validation.objective =
        crate::config::TrainingValidationObjective::SourceWeighted;
    config.training.events.source_weighted_validation_batches = 0;
    let error = config
        .validate()
        .expect_err("source-weighted objective requires an enabled probe");
    assert!(
        error
            .to_string()
            .contains("source_weighted_validation_batches")
    );

    config.training.events.source_weighted_validation_batches = 2;
    let error = config
        .validate()
        .expect_err("source-weighted objective requires a Ruliad source");
    assert!(error.to_string().contains("universality_ruliad"));

    config.dataset.source = crate::config::DatasetSourceConfig::UniversalityRuliad {
        config: PathBuf::from("ruliad.toml"),
    };
    config
        .validate()
        .expect("Ruliad source-weighted validation should validate");
}

#[test]
fn stream_warm_validation_objective_requires_a_carry_probe() {
    let mut config = parse_config("");
    config.training.validation.objective = crate::config::TrainingValidationObjective::StreamWarm;
    let error = config
        .validate()
        .expect_err("stream-warm objective needs recurrent carry evidence");
    assert!(error.to_string().contains("sequence_state_probe"));

    config.training.sequence_state_probe.enabled = true;
    config
        .validate()
        .expect("an explicit sequence-state probe supplies stream-warm validation");
}

#[test]
fn persisted_ruliad_validation_panel_requires_an_explicit_path_contract() {
    let mut config = parse_config("");
    config.training.validation.ruliad_panel.mode =
        crate::config::RuliadValidationPanelMode::CreateOrReuse;
    assert!(
        config
            .validate()
            .expect_err("persisted panel without path must fail closed")
            .to_string()
            .contains("ruliad_panel.path is required")
    );

    config.training.validation.ruliad_panel.path =
        Some(PathBuf::from("target/test-ruliad-panel.json"));
    config
        .validate()
        .expect("create-or-reuse panel with explicit path should validate");

    config.training.validation.ruliad_panel.mode =
        crate::config::RuliadValidationPanelMode::Dynamic;
    assert!(
        config
            .validate()
            .expect_err("dynamic panel must not pretend to persist a path")
            .to_string()
            .contains("path requires mode")
    );
}

#[test]
fn explicit_streaming_batching_is_independent_of_state_persistence() {
    let config = parse_config("sequence_batching = \"streaming\"");
    config
        .validate()
        .expect("ordered streaming batches should support a reset-state control");
    assert!(
        config
            .training
            .sequence_batching
            .uses_streaming_loader(config.training.tbptt_persist_across_steps)
    );
}

#[test]
fn persistent_state_rejects_random_batch_order() {
    let config = parse_config(
        "tbptt_chunk_size = 4\ntbptt_persist_across_steps = true\nsequence_batching = \"random\"",
    );
    let error = config
        .validate()
        .expect_err("persistent state cannot follow unrelated random windows");
    assert!(error.to_string().contains("sequence_batching=random"));
}

#[test]
fn sequence_state_probe_supports_matched_stateless_and_persistent_arms() {
    let config = parse_config(
        "sequence_batching = \"streaming\"\n\n[training.sequence_state_probe]\nenabled = true\npaired_batches = 2\nmax_rho_slots = 8",
    );
    config
        .validate()
        .expect("stateless training should still support carried-state evaluation");
    let config = parse_config(
        "tbptt_chunk_size = 4\ntbptt_persist_across_steps = true\nsequence_batching = \"streaming\"\n\n[training.sequence_state_probe]\nenabled = true\npaired_batches = 2\nmax_rho_slots = 8",
    );
    config
        .validate()
        .expect("persistent stream carry diagnostics should validate");
}

fn external_evaluator_config() -> TrainingConfig {
    let mut config = parse_config("");
    config.training.validation.execution =
        crate::config::TrainingValidationExecution::ExternalEvaluator;
    config.training.gates.enabled = false;
    config.training.dynamics.enabled = false;
    config.training.neuron_scaling.enabled = false;
    config.training.events.ruliad_correctness_probe_items = 0;
    config.training.events.source_weighted_validation_batches = 0;
    config.training.ruliad_policy_probe.enabled = false;
    config
}

#[test]
fn external_evaluator_contract_validates_when_local_consumers_are_disabled() {
    external_evaluator_config()
        .validate()
        .expect("external evaluator contract should validate");
}

#[test]
fn external_evaluator_contract_rejects_local_validation_consumers() {
    let cases = [
        (
            "gates",
            Box::new(|config: &mut TrainingConfig| config.training.gates.enabled = true)
                as Box<dyn Fn(&mut TrainingConfig)>,
        ),
        (
            "dynamics",
            Box::new(|config: &mut TrainingConfig| config.training.dynamics.enabled = true),
        ),
        (
            "source_weighted_validation_batches",
            Box::new(|config: &mut TrainingConfig| {
                config.training.events.source_weighted_validation_batches = 1;
            }),
        ),
        (
            "ruliad_correctness_probe_items",
            Box::new(|config: &mut TrainingConfig| {
                config.training.events.ruliad_correctness_probe_items = 1;
            }),
        ),
    ];

    for (expected, mutate) in cases {
        let mut config = external_evaluator_config();
        mutate(&mut config);
        let error = config
            .validate()
            .expect_err("local validation consumer should be rejected");
        assert!(
            error.to_string().contains(expected),
            "unexpected error for {expected}: {error}"
        );
    }
}

#[test]
fn latent_reasoning_jepa_training_requires_model_modules() {
    let mut config = parse_config("");
    config.training.latent_reasoning.enabled = true;
    config.training.latent_reasoning.jepa_future_offsets = vec![1];

    let err = config
        .validate()
        .expect_err("latent JEPA training should require model latent reasoning");
    assert!(
        err.to_string()
            .contains("training.latent_reasoning JEPA offsets"),
        "unexpected error: {err}"
    );
}

#[test]
fn latent_sigreg_only_training_validates_without_model_modules() {
    let mut config = parse_config("");
    config.training.latent_reasoning.enabled = true;
    config.training.latent_reasoning.jepa_future_offsets = vec![usize::MAX];
    config.training.latent_reasoning.sigreg.enabled = true;
    config.training.latent_reasoning.sigreg.target =
        crate::config::LatentReasoningSigRegTarget::RhoMemorySlots;

    config
        .validate()
        .expect("SIGReg-only latent regularization should not require model latent modules");
}

#[test]
fn hierarchical_dragon_training_config_validates() {
    let config = parse_config(
        r#"
[model.hierarchical_dragon]
enabled = true
last_layers = 1
fast_cycles = 2
slow_cycles = 1
rho_sharing = "split"
weight_sharing = "shared"
slow_to_fast_scale = 0.1
fast_to_slow_scale = 0.1
"#,
    );

    config
        .validate()
        .expect("hierarchical Dragon profile should validate");
}

#[test]
fn hierarchical_dragon_rejects_zero_cycles() {
    let config = parse_config(
        r#"
[model.hierarchical_dragon]
enabled = true
fast_cycles = 0
"#,
    );

    let err = config
        .validate()
        .expect_err("zero fast cycles should be rejected");
    assert!(
        err.to_string()
            .contains("model.hierarchical_dragon.fast_cycles"),
        "unexpected error: {err}"
    );
}

#[test]
fn hierarchical_dragon_rejects_pipeline_parallelism() {
    let config = parse_config(
        r#"
[parallel.pipeline]
enabled = true
stage_count = 2
microbatches = 2

[model.hierarchical_dragon]
enabled = true
"#,
    );

    let err = config
        .validate()
        .expect_err("pipeline hierarchy should be rejected");
    assert!(
        err.to_string().contains("parallel.pipeline.enabled"),
        "unexpected error: {err}"
    );
}

#[test]
fn next_latent_training_requires_transition_head() {
    let mut config = parse_config("");
    config.training.latent_reasoning.enabled = true;
    config.training.latent_reasoning.jepa_future_offsets = vec![usize::MAX];
    config.training.latent_reasoning.sigreg.enabled = false;
    config.training.latent_reasoning.next_latent.enabled = true;

    let err = config
        .validate()
        .expect_err("NextLat training should require a transition head");
    assert!(
        err.to_string()
            .contains("training.latent_reasoning.next_latent.enabled"),
        "unexpected error: {err}"
    );
}

#[test]
fn next_latent_training_does_not_require_inference_latent_reasoning() {
    let mut config = parse_config("");
    config.model.next_latent_transition = Some(Default::default());
    config
        .model
        .next_latent_transition
        .as_mut()
        .expect("next latent transition config")
        .enabled = true;
    config.training.latent_reasoning.enabled = true;
    config.training.latent_reasoning.jepa_future_offsets = vec![usize::MAX];
    config.training.latent_reasoning.sigreg.enabled = false;
    config.training.latent_reasoning.next_latent.enabled = true;

    config
        .validate()
        .expect("NextLat transition training should not require model.latent_reasoning");
}

#[test]
fn dragon_state_training_does_not_require_inference_latent_reasoning_or_transition_head() {
    let mut config = parse_config("");
    config.training.latent_reasoning.enabled = true;
    config.training.latent_reasoning.jepa_future_offsets = vec![usize::MAX];
    config.training.latent_reasoning.sigreg.enabled = false;
    config.training.latent_reasoning.dragon_state.enabled = true;

    config
        .validate()
        .expect("Dragon state consistency should only require recurrent Dragon state");
}

#[test]
fn step_contract_training_requires_inference_latent_reasoning() {
    let mut config = parse_config("");
    config.training.latent_reasoning.enabled = true;
    config.training.latent_reasoning.jepa_future_offsets = vec![usize::MAX];
    config.training.latent_reasoning.sigreg.enabled = false;
    config.training.latent_reasoning.step_contract.enabled = true;

    let err = config
        .validate()
        .expect_err("step contract training should require latent reasoning architecture");
    assert!(
        err.to_string()
            .contains("training.latent_reasoning.step_contract.enabled"),
        "unexpected error: {err}"
    );
}

#[test]
fn latent_reasoning_training_validates_with_model_modules() {
    let mut config = parse_config("");
    config.model.latent_reasoning = Some(Default::default());
    config
        .model
        .latent_reasoning
        .as_mut()
        .expect("latent config")
        .enabled = true;
    config.training.latent_reasoning.enabled = true;
    config.training.latent_reasoning.jepa_future_offsets = vec![1, 2];

    config
        .validate()
        .expect("latent reasoning training should validate with model modules enabled");
}

#[test]
fn latent_energy_model_training_requires_model_energy_head() {
    let mut config = parse_config("");
    config.model.latent_reasoning = Some(Default::default());
    config
        .model
        .latent_reasoning
        .as_mut()
        .expect("latent config")
        .enabled = true;
    config.training.latent_reasoning.enabled = true;
    config.training.latent_reasoning.jepa_future_offsets = vec![usize::MAX];
    config.training.latent_reasoning.sigreg.enabled = false;
    config.training.latent_reasoning.energy_model.enabled = true;

    let err = config
        .validate()
        .expect_err("latent EBM training should require model energy head");
    assert!(
        err.to_string()
            .contains("model.latent_reasoning.energy_head"),
        "unexpected error: {err}"
    );
}

#[test]
fn latent_energy_model_training_validates_with_model_energy_head() {
    let mut config = parse_config("");
    config.model.latent_reasoning = Some(Default::default());
    let latent = config
        .model
        .latent_reasoning
        .as_mut()
        .expect("latent config");
    latent.enabled = true;
    latent.energy_head = true;
    config.training.latent_reasoning.enabled = true;
    config.training.latent_reasoning.jepa_future_offsets = vec![usize::MAX];
    config.training.latent_reasoning.sigreg.enabled = false;
    config.training.latent_reasoning.energy_model.enabled = true;

    config
        .validate()
        .expect("latent EBM training should validate with model energy head");
}

#[test]
fn latent_reasoning_eval_step_sweep_requires_model_modules() {
    let mut config = parse_config("");
    config.training.latent_reasoning.eval_step_sweep = vec![1, 2, 4];

    let err = config
        .validate()
        .expect_err("eval step sweep should require model latent reasoning");
    assert!(
        err.to_string()
            .contains("training.latent_reasoning.eval_step_sweep"),
        "unexpected error: {err}"
    );
}

#[test]
fn latent_reasoning_eval_step_sweep_validates_with_model_modules() {
    let mut config = parse_config("");
    config.model.latent_reasoning = Some(Default::default());
    config
        .model
        .latent_reasoning
        .as_mut()
        .expect("latent config")
        .enabled = true;
    config.training.latent_reasoning.eval_step_sweep = vec![1, 2, 4];

    config
        .validate()
        .expect("eval step sweep should validate with model latent reasoning");
}

#[test]
fn latent_reasoning_eval_step_sweep_rejects_zero() {
    let mut config = parse_config("");
    config.model.latent_reasoning = Some(Default::default());
    config
        .model
        .latent_reasoning
        .as_mut()
        .expect("latent config")
        .enabled = true;
    config.training.latent_reasoning.eval_step_sweep = vec![1, 0, 4];

    let err = config
        .validate()
        .expect_err("zero eval step should fail validation");
    assert!(
        err.to_string()
            .contains("eval_step_sweep must contain only positive"),
        "unexpected error: {err}"
    );
}

#[test]
fn latent_reasoning_start_policy_toml_values_parse() {
    let config = parse_config(
        r#"
[training.latent_reasoning]
enabled = true
jepa_start_policy = "fixed_step_and_capability_gate"

[training.latent_reasoning.next_latent]
enabled = true
start_policy = "capability_gate"
"#,
    );

    assert_eq!(
        config.training.latent_reasoning.jepa_start_policy,
        Some(crate::config::LatentReasoningAuxiliaryStartPolicy::FixedStepAndCapabilityGate)
    );
    assert_eq!(
        config.training.latent_reasoning.next_latent.start_policy,
        Some(crate::config::LatentReasoningAuxiliaryStartPolicy::CapabilityGate)
    );
}

#[test]
fn latent_reasoning_training_rejects_zero_every_steps() {
    let mut config = parse_config("");
    config.model.latent_reasoning = Some(Default::default());
    config
        .model
        .latent_reasoning
        .as_mut()
        .expect("latent config")
        .enabled = true;
    config.training.latent_reasoning.enabled = true;
    config.training.latent_reasoning.every_steps = 0;

    let err = config
        .validate()
        .expect_err("zero latent every_steps should fail validation");
    assert!(
        err.to_string()
            .contains("training.latent_reasoning.every_steps"),
        "unexpected error: {err}"
    );
}

#[test]
fn latent_reasoning_training_rejects_zero_per_objective_every_steps() {
    let mut config = parse_config("");
    config.model.latent_reasoning = Some(Default::default());
    config
        .model
        .latent_reasoning
        .as_mut()
        .expect("latent config")
        .enabled = true;
    config.model.next_latent_transition = Some(Default::default());
    config
        .model
        .next_latent_transition
        .as_mut()
        .expect("next latent config")
        .enabled = true;
    config.training.latent_reasoning.enabled = true;
    config.training.latent_reasoning.jepa_future_offsets = vec![1];
    config.training.latent_reasoning.next_latent.enabled = true;
    config.training.latent_reasoning.next_latent.every_steps = Some(0);

    let err = config
        .validate()
        .expect_err("zero NextLat every_steps should fail validation");
    assert!(
        err.to_string()
            .contains("training.latent_reasoning.next_latent.every_steps"),
        "unexpected error: {err}"
    );
}

#[test]
fn predictive_coding_validates_for_single_tbptt_next_token_training() {
    let mut config = parse_config("");
    config.training.tbptt_chunk_size = Some(4);
    config.training.predictive_coding.enabled = true;

    config
        .validate()
        .expect("predictive coding should validate for local TBPTT next-token training");
}

#[test]
fn predictive_coding_rejects_invalid_amortization_contract() {
    let mut config = parse_config("");
    config.training.tbptt_chunk_size = Some(4);
    config.training.predictive_coding.enabled = true;
    config.training.predictive_coding.amortization_tolerance = f32::NAN;

    let err = config
        .validate()
        .expect_err("non-finite amortization tolerance should fail validation");
    assert!(err.to_string().contains("amortization_tolerance"));

    config.training.predictive_coding.amortization_tolerance = 0.05;
    config
        .training
        .predictive_coding
        .amortization_max_state_slots = 0;
    let err = config
        .validate()
        .expect_err("empty amortization sample should fail validation");
    assert!(err.to_string().contains("amortization_max_state_slots"));
}

#[test]
fn predictive_coding_rejects_unacknowledged_oracle_target_control() {
    let mut config = parse_config("");
    config.training.tbptt_chunk_size = Some(4);
    config.training.predictive_coding.enabled = true;
    config.training.predictive_coding.observation_contract =
        PredictiveCodingObservationContract::OracleNextTokenNegativeControl;

    let err = config
        .validate()
        .expect_err("oracle target leakage must require explicit acknowledgement");
    assert!(
        err.to_string().contains("allow_oracle_target_leak=true"),
        "unexpected error: {err}"
    );
}

#[test]
fn predictive_coding_oracle_target_control_is_explicitly_available_for_ablations() {
    let mut config = parse_config("");
    config.training.tbptt_chunk_size = Some(4);
    config.training.predictive_coding.enabled = true;
    config.training.predictive_coding.observation_contract =
        PredictiveCodingObservationContract::OracleNextTokenNegativeControl;
    config.training.predictive_coding.allow_oracle_target_leak = true;

    config
        .validate()
        .expect("acknowledged oracle negative control should remain reproducible");
}

#[test]
fn observed_prefix_predictive_coding_rejects_block_backward() {
    let mut config = parse_config("");
    config.training.tbptt_chunk_size = Some(4);
    config.training.predictive_coding.enabled = true;
    config.training.predictive_coding.backward_mode = PredictiveCodingBackwardMode::Block;

    let err = config
        .validate()
        .expect_err("causal correction must follow each completed chunk");
    assert!(
        err.to_string().contains("requires backward_mode=chunked"),
        "unexpected error: {err}"
    );
}

#[test]
fn retired_predictive_coding_optimizer_points_to_the_algorithm_contract() {
    let mut config = parse_config("");
    config.optimizer.name = OptimizerKind::PredictiveCoding;
    let err = config
        .validate()
        .expect_err("retired optimizer route must be rejected");
    assert!(
        err.to_string()
            .contains("training.algorithm=predictive_coding"),
        "unexpected error: {err}"
    );
}

#[test]
fn local_predictive_coding_algorithm_validates_with_plain_vjp_contract() {
    let mut config = parse_config("");
    config.training.algorithm = TrainingAlgorithm::PredictiveCoding;
    config.model.dropout = Some(0.0);
    config.model.sequence_kernel = Some(SequenceKernelConfig::dense_score_short_context());
    config.model.rotary_embedding = Some(RotaryEmbedding::Alibi);

    config
        .validate()
        .expect("canonical local predictive-coding contract should validate");
    assert_eq!(
        config.resolved_training_algorithm(),
        TrainingAlgorithm::PredictiveCoding
    );
}

#[test]
fn local_predictive_coding_algorithm_validates_with_recurrent_executor() {
    let mut config = parse_config("");
    config.training.algorithm = TrainingAlgorithm::PredictiveCoding;
    config.model.dropout = Some(0.0);
    config.model.sequence_kernel = Some(SequenceKernelConfig::reference(
        SequenceMemorySystem::LinearAttention,
    ));
    config.model.rotary_embedding = Some(RotaryEmbedding::Alibi);

    config
        .validate()
        .expect("production recurrent local predictive-coding contract should validate");
}

#[test]
fn local_predictive_coding_validates_recurrent_tbptt_contract() {
    let mut config = parse_config("");
    config.training.algorithm = TrainingAlgorithm::PredictiveCoding;
    config.training.tbptt_chunk_size = Some(4);
    config.training.tbptt_persist_across_steps = true;
    config.model.dropout = Some(0.0);
    config.model.sequence_kernel = Some(SequenceKernelConfig::dense_score_short_context());
    config.model.rotary_embedding = Some(RotaryEmbedding::Alibi);

    config
        .validate()
        .expect("local predictive coding should support detached recurrent TBPTT factors");
}

#[test]
fn bounded_backprop_temporal_credit_requires_an_explicit_tbptt_contract() {
    let mut config = parse_config("");
    config.training.tbptt_credit_window_chunks = 2;
    let error = config
        .validate()
        .expect_err("bounded temporal credit requires chunk geometry");
    assert!(
        error
            .to_string()
            .contains("tbptt_credit_window_chunks > 1 requires training.tbptt_chunk_size"),
        "unexpected error: {error}"
    );

    config.training.tbptt_chunk_size = Some(4);
    config
        .validate()
        .expect("bounded backprop temporal credit should validate");

    config.training.algorithm = TrainingAlgorithm::PredictiveCoding;
    config.model.dropout = Some(0.0);
    config.model.sequence_kernel = Some(SequenceKernelConfig::dense_score_short_context());
    config.model.rotary_embedding = Some(RotaryEmbedding::Alibi);
    let error = config
        .validate()
        .expect_err("local PC owns its temporal-credit contract");
    assert!(
        error
            .to_string()
            .contains("requires training.tbptt_credit_window_chunks=1"),
        "unexpected error: {error}"
    );
}

#[test]
fn exact_temporal_credit_validates_only_bounded_fixed_prediction_tbptt() {
    let mut config = parse_config("");
    config.training.algorithm = TrainingAlgorithm::PredictiveCoding;
    config.training.tbptt_chunk_size = Some(4);
    config.training.local_predictive_coding.solver =
        crate::config::LocalPredictiveCodingSolver::FixedPrediction;
    config.training.local_predictive_coding.temporal_credit = burn_pc::PcTemporalCreditConfig {
        mode: burn_pc::PcTemporalCreditMode::ExactWindow,
        window_chunks: 3,
    };
    config.model.dropout = Some(0.0);
    config.model.sequence_kernel = Some(SequenceKernelConfig::dense_score_short_context());
    config.model.rotary_embedding = Some(RotaryEmbedding::Alibi);
    config
        .validate()
        .expect("bounded exact temporal credit should validate");

    let mut missing_tbptt = config.clone();
    missing_tbptt.training.tbptt_chunk_size = None;
    assert!(
        missing_tbptt
            .validate()
            .expect_err("exact temporal credit needs explicit chunks")
            .to_string()
            .contains("requires training.tbptt_chunk_size")
    );

    let mut unsupported_solver = config.clone();
    unsupported_solver.training.local_predictive_coding.solver =
        crate::config::LocalPredictiveCodingSolver::ErrorEquilibrium;
    assert!(
        unsupported_solver
            .validate()
            .expect_err("exact temporal credit needs an exact local VJP solver")
            .to_string()
            .contains("requires solver=fixed_prediction")
    );

    config.training.predictive_context_routing.enabled = true;
    assert!(
        config
            .validate()
            .expect_err("routed state needs an explicit temporal ownership contract")
            .to_string()
            .contains("does not yet compose with predictive_context_routing")
    );
}

#[test]
fn incremental_local_predictive_coding_validates_only_interleaved_solvers() {
    let mut config = parse_config("");
    config.training.algorithm = TrainingAlgorithm::PredictiveCoding;
    config.training.local_predictive_coding.learning_schedule =
        burn_pc::PcLearningSchedule::Incremental;
    config.training.local_predictive_coding.solver =
        crate::config::LocalPredictiveCodingSolver::ReverseGaussSeidel;
    config
        .training
        .local_predictive_coding
        .incremental_parameter_step_scale = 0.25;
    config.model.dropout = Some(0.0);
    config.model.sequence_kernel = Some(SequenceKernelConfig::dense_score_short_context());
    config.model.rotary_embedding = Some(RotaryEmbedding::Alibi);
    config
        .validate()
        .expect("incremental local PC should own its optimizer schedule");

    config.training.local_predictive_coding.solver =
        crate::config::LocalPredictiveCodingSolver::FixedPrediction;
    assert!(
        config
            .validate()
            .expect_err("fixed prediction has no inferred activity schedule")
            .to_string()
            .contains("synchronous_equilibrium or reverse_gauss_seidel")
    );
    config.training.local_predictive_coding.solver =
        crate::config::LocalPredictiveCodingSolver::ReverseGaussSeidel;
    config
        .training
        .local_predictive_coding
        .incremental_parameter_step_scale = 0.0;
    assert!(
        config
            .validate()
            .expect_err("zero iPC parameter scale must fail closed")
            .to_string()
            .contains("incremental_parameter_step_scale")
    );
}

#[test]
fn local_fixed_prediction_solver_validates_with_plain_vjp_contract() {
    let mut config = parse_config("");
    config.training.algorithm = TrainingAlgorithm::PredictiveCoding;
    config.training.local_predictive_coding.solver =
        crate::config::LocalPredictiveCodingSolver::FixedPrediction;
    config.model.dropout = Some(0.0);
    config.model.sequence_kernel = Some(SequenceKernelConfig::dense_score_short_context());
    config.model.rotary_embedding = Some(RotaryEmbedding::Alibi);

    config
        .validate()
        .expect("fixed-prediction local PC contract should validate");
}

#[test]
fn local_error_equilibrium_validates_standard_and_mu_pc_contracts() {
    let mut config = parse_config("");
    config.training.algorithm = TrainingAlgorithm::PredictiveCoding;
    config.training.local_predictive_coding.solver =
        crate::config::LocalPredictiveCodingSolver::ErrorEquilibrium;
    config.model.dropout = Some(0.0);
    config.model.sequence_kernel = Some(SequenceKernelConfig::dense_score_short_context());
    config.model.rotary_embedding = Some(RotaryEmbedding::Alibi);

    config
        .validate()
        .expect("standard error-equilibrium contract should validate");
    config.training.local_predictive_coding.parameterization =
        burn_pc::PcParameterizationKind::MuPc;
    config
        .validate()
        .expect("muPC requires and inherits depth-scaled initialization");

    config.model.initialization = Some(burn_dragon_core::DragonInitializationConfig {
        residual_scaling: burn_dragon_core::DragonResidualScalingConfig {
            kind: DragonResidualScalingKind::Disabled,
            ..burn_dragon_core::DragonResidualScalingConfig::default()
        },
        ..burn_dragon_core::DragonInitializationConfig::default()
    });
    assert!(
        config
            .validate()
            .expect_err("muPC must fail closed without depth scaling")
            .to_string()
            .contains("residual_scaling.kind=depth_scaled")
    );
}

#[test]
fn direct_kolen_pollack_validates_two_phase_tied_contract() {
    let mut config = parse_config("");
    config.training.algorithm = TrainingAlgorithm::PredictiveCoding;
    config.training.local_predictive_coding.solver =
        crate::config::LocalPredictiveCodingSolver::DirectKolenPollack;
    config.model.dropout = Some(0.0);
    config.model.sequence_kernel = Some(SequenceKernelConfig::dense_score_short_context());
    config.model.rotary_embedding = Some(RotaryEmbedding::Alibi);
    config
        .validate()
        .expect("DKP should validate with tied consensus and optimizer-owned decay");

    config
        .training
        .local_predictive_coding
        .direct_feedback
        .forward_weight_decay = 0.1;
    assert!(
        config
            .validate()
            .expect_err("DKP forward decay has one owner")
            .to_string()
            .contains("outer optimizer owns")
    );

    config
        .training
        .local_predictive_coding
        .direct_feedback
        .forward_weight_decay = 0.0;
    config
        .training
        .local_predictive_coding
        .amortized_adjoint
        .enabled = true;
    config
        .validate()
        .expect("DKP should accept a periodic exact-local adjoint teacher");

    config.training.local_predictive_coding.solver =
        crate::config::LocalPredictiveCodingSolver::FixedPrediction;
    assert!(
        config
            .validate()
            .expect_err("amortized feedback must not be ignored by another solver")
            .to_string()
            .contains("requires a direct-feedback solver")
    );
}

#[test]
fn amortized_adjoint_validates_single_update_exact_anchor_contract() {
    let mut config = parse_config("");
    config.training.algorithm = TrainingAlgorithm::PredictiveCoding;
    config.training.local_predictive_coding.solver =
        crate::config::LocalPredictiveCodingSolver::AmortizedAdjoint;
    config
        .training
        .local_predictive_coding
        .amortized_adjoint
        .enabled = true;
    config.training.local_predictive_coding.factor_reduction =
        crate::config::PredictiveCodingFactorReduction::Sum;
    config.model.dropout = Some(0.0);
    config.model.sequence_kernel = Some(SequenceKernelConfig::dense_score_short_context());
    config.model.rotary_embedding = Some(RotaryEmbedding::Alibi);
    config
        .validate()
        .expect("amortized adjoint should validate with exact anchors enabled");

    config
        .training
        .local_predictive_coding
        .amortized_adjoint
        .predictor = burn_pc::PcAdjointPredictorKind::ResidualConditioned;
    config
        .validate()
        .expect("residual-conditioned adjoint should preserve unit-scale credit");
    config.training.local_predictive_coding.adjoint_conditioning =
        crate::config::LocalPredictiveCodingAdjointConditioning::TerminalDisplacement;
    config
        .validate()
        .expect("terminal displacement is valid for residual-conditioned adjoints");
    config
        .training
        .local_predictive_coding
        .direct_feedback
        .signal_scale = 0.5;
    assert!(
        config
            .validate()
            .expect_err("residual identity path cannot be rescaled")
            .to_string()
            .contains("signal_scale=1")
    );
    config
        .training
        .local_predictive_coding
        .direct_feedback
        .signal_scale = 1.0;

    config
        .training
        .local_predictive_coding
        .amortized_adjoint
        .predictor = burn_pc::PcAdjointPredictorKind::DirectLinear;
    assert!(
        config
            .validate()
            .expect_err("terminal displacement must not be silently ignored")
            .to_string()
            .contains("terminal_displacement requires")
    );
    config
        .training
        .local_predictive_coding
        .amortized_adjoint
        .predictor = burn_pc::PcAdjointPredictorKind::ResidualConditioned;

    config
        .training
        .local_predictive_coding
        .amortized_adjoint
        .enabled = false;
    assert!(
        config
            .validate()
            .expect_err("amortized adjoint requires its teacher schedule")
            .to_string()
            .contains("amortized_adjoint.enabled=true")
    );
    config
        .training
        .local_predictive_coding
        .amortized_adjoint
        .enabled = true;
    config.training.local_predictive_coding.factor_reduction =
        crate::config::PredictiveCodingFactorReduction::Mean;
    assert!(
        config
            .validate()
            .expect_err("exact anchors cannot be depth averaged")
            .to_string()
            .contains("factor_reduction=sum")
    );
}

#[test]
fn first_order_adjoint_validates_teacher_free_residual_contract() {
    let mut config = parse_config("");
    config.training.algorithm = TrainingAlgorithm::PredictiveCoding;
    config.training.local_predictive_coding.solver =
        crate::config::LocalPredictiveCodingSolver::FirstOrderAdjoint;
    config.training.local_predictive_coding.factor_reduction =
        crate::config::PredictiveCodingFactorReduction::Sum;
    config.model.dropout = Some(0.0);
    config.model.sequence_kernel = Some(SequenceKernelConfig::dense_score_short_context());
    config.model.rotary_embedding = Some(RotaryEmbedding::Alibi);
    config
        .validate()
        .expect("first-order adjoint should validate without a feedback teacher");

    config.training.local_predictive_coding.factor_reduction =
        crate::config::PredictiveCodingFactorReduction::Mean;
    assert!(
        config
            .validate()
            .expect_err("first-order terminal credit cannot be depth averaged")
            .to_string()
            .contains("factor_reduction=sum")
    );
}

#[test]
fn local_layer_prediction_validates_only_with_its_normalized_contract() {
    let mut config = parse_config("");
    config.training.algorithm = TrainingAlgorithm::PredictiveCoding;
    config.training.local_predictive_coding.solver =
        crate::config::LocalPredictiveCodingSolver::LayerLocalPrediction;
    config.training.local_predictive_coding.factor_reduction =
        crate::config::PredictiveCodingFactorReduction::Mean;
    config.model.dropout = Some(0.0);
    config.model.sequence_kernel = Some(SequenceKernelConfig::dense_score_short_context());
    config.model.rotary_embedding = Some(RotaryEmbedding::Alibi);

    config
        .validate()
        .expect("normalized layer-local prediction contract should validate");

    config.training.local_predictive_coding.factor_reduction =
        crate::config::PredictiveCodingFactorReduction::Sum;
    assert!(
        config
            .validate()
            .expect_err("unnormalized layer-local factors must fail closed")
            .to_string()
            .contains("factor_reduction=mean")
    );
    config.training.local_predictive_coding.factor_reduction =
        crate::config::PredictiveCodingFactorReduction::Mean;
    config.training.local_predictive_coding.sync_diagnostics = true;
    assert!(
        config
            .validate()
            .expect_err("undefined equilibrium diagnostics must fail closed")
            .to_string()
            .contains("sync_diagnostics=false")
    );
}

#[test]
fn predictive_context_routing_accepts_feedforward_local_solvers() {
    let mut config = parse_config("");
    config.training.algorithm = TrainingAlgorithm::PredictiveCoding;
    config.training.local_predictive_coding.solver =
        crate::config::LocalPredictiveCodingSolver::FixedPrediction;
    config.training.predictive_context_routing.enabled = true;
    config.training.dynamics.enabled = false;
    config.model.dropout = Some(0.0);
    config.model.sequence_kernel = Some(SequenceKernelConfig::dense_score_short_context());
    config.model.rotary_embedding = Some(RotaryEmbedding::Alibi);
    config
        .validate()
        .expect("bounded predictive context routing contract should validate");

    config.training.local_predictive_coding.solver =
        crate::config::LocalPredictiveCodingSolver::LayerLocalPrediction;
    config.training.local_predictive_coding.factor_reduction =
        crate::config::PredictiveCodingFactorReduction::Mean;
    config
        .validate()
        .expect("routed layer-local prediction contract should validate");

    config.training.local_predictive_coding.solver =
        crate::config::LocalPredictiveCodingSolver::SynchronousEquilibrium;
    assert!(
        config
            .validate()
            .expect_err("non-triangular context learner must fail closed")
            .to_string()
            .contains("feed-forward local solver")
    );

    config.training.local_predictive_coding.solver =
        crate::config::LocalPredictiveCodingSolver::FixedPrediction;
    config.training.ruliad_policy_probe.enabled = true;
    assert!(
        config
            .validate()
            .expect_err("routed hidden-state policy probe must fail closed")
            .to_string()
            .contains("hidden-state Ruliad policy probes")
    );
    config.training.ruliad_policy_probe.enabled = false;
    config.training.events.ruliad_contract_probe_enabled = true;
    assert!(
        config
            .validate()
            .expect_err("routed constrained decoder must fail closed")
            .to_string()
            .contains("constrained Ruliad contract decoding")
    );
    config.training.events.ruliad_contract_probe_enabled = false;
    config.training.gradient_accumulation_steps = 2;
    assert!(
        config
            .validate()
            .expect_err("cross-context gradient accumulation must fail closed")
            .to_string()
            .contains("gradient_accumulation_steps=1")
    );
}

#[test]
fn local_predictive_coding_rejects_dropout_mismatch() {
    let mut config = parse_config("");
    config.training.algorithm = TrainingAlgorithm::PredictiveCoding;
    config.model.dropout = Some(0.1);
    config.model.sequence_kernel = Some(SequenceKernelConfig::dense_score_short_context());
    config.model.rotary_embedding = Some(RotaryEmbedding::Alibi);

    let err = config
        .validate()
        .expect_err("plain local VJP path must not silently drop configured dropout");
    assert!(
        err.to_string().contains("dropout=0"),
        "unexpected error: {err}"
    );
}

#[test]
fn local_predictive_coding_rejects_ignored_ruliad_auxiliaries() {
    let mut config = parse_config("");
    config.training.algorithm = TrainingAlgorithm::PredictiveCoding;
    config.model.dropout = Some(0.0);
    config.model.sequence_kernel = Some(SequenceKernelConfig::dense_score_short_context());
    config.model.rotary_embedding = Some(RotaryEmbedding::Alibi);
    config.training.ruliad_supervision.mode = RuliadSupervisionMode::AnswerCompletion;
    config.training.ruliad_supervision.answer_contract.enabled = true;

    let err = config
        .validate_local_predictive_coding()
        .expect_err("local PC must not silently skip Ruliad auxiliary objectives");
    assert!(
        err.to_string().contains("Ruliad auxiliary objectives"),
        "unexpected error: {err}"
    );
}

fn verifier_terminal_config() -> TrainingConfig {
    let mut config = parse_config("");
    config.training.algorithm = TrainingAlgorithm::PredictiveCoding;
    config.training.local_predictive_coding.solver =
        crate::config::LocalPredictiveCodingSolver::FixedPrediction;
    config.training.local_predictive_coding.terminal_criterion =
        crate::config::LocalPredictiveCodingTerminalCriterion::RuliadVerifierSet;
    config.model.dropout = Some(0.0);
    config.model.sequence_kernel = Some(SequenceKernelConfig::dense_score_short_context());
    config.model.rotary_embedding = Some(RotaryEmbedding::Alibi);
    config.training.ruliad_supervision.proof_policy =
        crate::config::RuliadProofPolicyTrainingConfig {
            enabled: true,
            mode: crate::config::RuliadProofPolicyTrainingMode::StaticExpert,
            scoring: crate::config::RuliadProofPolicyScoring::CompletionLikelihood,
            gradient_scope: crate::config::RuliadProofPolicyGradientScope::FullModel,
            normalization: crate::config::RuliadProofPolicyNormalization::PrefixConditional,
            presentation_risk: crate::config::RuliadProofPolicyPresentationRisk::Mean,
            weight: 1.0,
            counterfactual_targets_per_state: 0,
            stratified_difficulty_levels: 1,
            ..crate::config::RuliadProofPolicyTrainingConfig::default()
        };
    config
}

#[test]
fn local_predictive_coding_accepts_explicit_static_verifier_terminal() {
    verifier_terminal_config()
        .validate()
        .expect("static verifier-set terminal should be an executable local factor");
}

#[test]
fn local_predictive_coding_accepts_counterfactual_deployed_decoder_terminal() {
    let mut config = verifier_terminal_config();
    config
        .training
        .ruliad_supervision
        .proof_policy
        .counterfactual_targets_per_state = 1;
    config
        .validate()
        .expect("local PC should accept complete counterfactual target groups on the action trie");
}

#[test]
fn backpropagation_accepts_the_matched_static_verifier_terminal() {
    let mut config = verifier_terminal_config();
    config.training.algorithm = TrainingAlgorithm::Backpropagation;
    config
        .validate()
        .expect("global autodiff should accept the same static verifier factor");
}

#[test]
fn backpropagation_accepts_model_visited_verifier_terminals() {
    for mode in [
        crate::config::RuliadProofPolicyTrainingMode::Dagger,
        crate::config::RuliadProofPolicyTrainingMode::StaticThenPairedDagger,
    ] {
        let mut config = verifier_terminal_config();
        config.training.algorithm = TrainingAlgorithm::Backpropagation;
        config.training.ruliad_supervision.proof_policy.mode = mode;
        config
            .validate()
            .expect("global autodiff should support verifier-labelled model-visited panels");
    }
}

#[test]
fn backpropagation_accepts_counterfactual_semantic_energy_verifier_terminal() {
    let mut config = verifier_terminal_config();
    config.training.algorithm = TrainingAlgorithm::Backpropagation;
    config.model.sequence_score_head = Some(burn_dragon_core::SequenceScoreHeadConfig {
        enabled: true,
        projection_dim: 64,
    });
    let policy = &mut config.training.ruliad_supervision.proof_policy;
    policy.mode = crate::config::RuliadProofPolicyTrainingMode::Dagger;
    policy.scoring = crate::config::RuliadProofPolicyScoring::SemanticEnergy;
    policy.normalization = crate::config::RuliadProofPolicyNormalization::CandidateConditional;
    policy.counterfactual_targets_per_state = 1;

    config
        .validate()
        .expect("global autodiff should accept counterfactual semantic-energy trajectory panels");
}

#[test]
fn fixed_prediction_accepts_exact_semantic_energy_verifier_terminal() {
    let mut config = verifier_terminal_config();
    config.model.sequence_score_head = Some(burn_dragon_core::SequenceScoreHeadConfig {
        enabled: true,
        projection_dim: 64,
    });
    let policy = &mut config.training.ruliad_supervision.proof_policy;
    policy.mode = crate::config::RuliadProofPolicyTrainingMode::Dagger;
    policy.scoring = crate::config::RuliadProofPolicyScoring::SemanticEnergy;
    policy.gradient_scope = crate::config::RuliadProofPolicyGradientScope::FullModel;
    policy.normalization = crate::config::RuliadProofPolicyNormalization::CandidateConditional;
    policy.counterfactual_targets_per_state = 1;

    config
        .validate()
        .expect("fixed-prediction PC should accept its analytic sequence-energy VJP");

    let mut unsupported_solver = config.clone();
    unsupported_solver.training.local_predictive_coding.solver =
        crate::config::LocalPredictiveCodingSolver::ErrorEquilibrium;
    let error = unsupported_solver
        .validate()
        .expect_err("equilibrium PC does not register sequence-score-head derivatives");
    assert!(error.to_string().contains("fixed-prediction PC"), "{error}");

    let mut residual = config;
    residual.training.ruliad_supervision.proof_policy.scoring =
        crate::config::RuliadProofPolicyScoring::ResidualEnergy;
    let error = residual
        .validate()
        .expect_err("residual energy also needs an analytic autoregressive-prior VJP");
    assert!(error.to_string().contains("fixed-prediction PC"), "{error}");
}

#[test]
fn semantic_refresh_replaces_sparse_global_verifier_terminals() {
    let mut config = verifier_terminal_config();
    config.training.algorithm = TrainingAlgorithm::Backpropagation;
    config.model.sequence_score_head = Some(burn_dragon_core::SequenceScoreHeadConfig {
        enabled: true,
        projection_dim: 64,
    });
    config
        .training
        .ruliad_supervision
        .proof_policy_semantic_refresh = crate::config::RuliadProofPolicySemanticRefreshConfig {
        enabled: true,
        every_steps: 64,
        start_after_steps: 64,
        counterfactual_targets_per_state: 1,
    };
    config
        .validate()
        .expect("global verifier terminal should accept sparse semantic refreshes");

    let supervision = config.training.ruliad_supervision;
    assert_eq!(
        supervision.proof_policy_for_step(48).scoring,
        crate::config::RuliadProofPolicyScoring::CompletionLikelihood
    );
    let refresh = supervision.proof_policy_for_step(64);
    assert_eq!(
        refresh.scoring,
        crate::config::RuliadProofPolicyScoring::SemanticEnergy
    );
    assert_eq!(
        refresh.normalization,
        crate::config::RuliadProofPolicyNormalization::CandidateConditional
    );
    assert_eq!(refresh.counterfactual_targets_per_state, 1);

    config.training.algorithm = TrainingAlgorithm::PredictiveCoding;
    assert!(
        config
            .validate()
            .expect_err("local PC must reject unimplemented semantic refresh factors")
            .to_string()
            .contains("semantic-energy refresh")
    );
}

#[test]
fn local_predictive_coding_verifier_terminal_accepts_dagger_and_rejects_unsupported_solver() {
    for mode in [
        crate::config::RuliadProofPolicyTrainingMode::Dagger,
        crate::config::RuliadProofPolicyTrainingMode::StaticThenPairedDagger,
    ] {
        let mut config = verifier_terminal_config();
        config.training.ruliad_supervision.proof_policy.mode = mode;
        if mode == crate::config::RuliadProofPolicyTrainingMode::StaticThenPairedDagger {
            config
                .training
                .ruliad_supervision
                .proof_policy
                .start_after_steps = 0;
            config
                .training
                .ruliad_supervision
                .proof_policy
                .dagger_start_after_steps = 256;
        }
        config
            .validate()
            .expect("local PC must accept verifier-labelled model-visited panels");
    }

    let mut config = verifier_terminal_config();
    config.training.local_predictive_coding.solver =
        crate::config::LocalPredictiveCodingSolver::DirectKolenPollack;
    assert!(
        config
            .validate()
            .expect_err("verifier terminal must reject unsupported credit solvers")
            .to_string()
            .contains("or augmented_lagrangian")
    );

    let mut config = verifier_terminal_config();
    config.training.local_predictive_coding.solver =
        crate::config::LocalPredictiveCodingSolver::AugmentedLagrangian;
    config
        .validate()
        .expect("PC-ALM must accept the same typed verifier terminal factor");

    let mut config = verifier_terminal_config();
    config
        .training
        .ruliad_supervision
        .proof_policy
        .normalization = crate::config::RuliadProofPolicyNormalization::CandidateConditional;
    let error = config
        .validate()
        .expect_err("sequence-level normalization is not the trie terminal objective");
    assert!(
        error.to_string().contains("prefix-conditional"),
        "unexpected error: {error}"
    );
}

#[test]
fn predictive_coding_requires_tbptt() {
    let mut config = parse_config("");
    config.training.tbptt_chunk_size = None;
    config.training.predictive_coding.enabled = true;

    let err = config
        .validate()
        .expect_err("predictive coding without TBPTT should be rejected");
    assert!(
        err.to_string()
            .contains("predictive_coding.enabled requires training.tbptt_chunk_size"),
        "unexpected error: {err}"
    );
}

#[test]
fn predictive_coding_rejects_pipeline_training() {
    let mut config = parse_config("");
    config.training.tbptt_chunk_size = Some(4);
    config.training.predictive_coding.enabled = true;
    config.parallel.pipeline.enabled = true;
    config.parallel.pipeline.stage_count = 1;
    config.parallel.pipeline.microbatches = 1;

    let err = config
        .validate()
        .expect_err("predictive coding should not run in pipeline mode");
    assert!(
        err.to_string()
            .contains("predictive_coding.enabled does not support parallel.pipeline.enabled"),
        "unexpected error: {err}"
    );
}

#[test]
fn predictive_coding_high_latent_rejects_unsafe_fixed_large_batch() {
    let mut config = parse_config("");
    config.training.tbptt_chunk_size = Some(4);
    config.training.predictive_coding.enabled = true;
    config.training.batch_size = 2;
    config.training.auto_batch_size.enabled = false;
    config.model.latent_total = Some(16_384);

    let err = config
        .validate()
        .expect_err("high-latent PC should require batch one or auto batch sizing");
    assert!(
        err.to_string()
            .contains("predictive_coding with fixed training.batch_size > 1"),
        "unexpected error: {err}"
    );
}

#[test]
fn ruliad_policy_batch_is_required_only_by_active_auxiliary_consumers() {
    let mut supervision = crate::RuliadSupervisionConfig::default();
    assert!(!supervision.needs_ruliad_policy_batch());

    supervision.verifier_reward.enabled = true;
    supervision.verifier_reward.weight = 0.0;
    supervision.verifier_reward.structured_contrast_weight = 0.0;
    supervision.verifier_reward.field_binding_contrast_weight = 0.0;
    supervision.verifier_reward.rollout_imitation_weight = 0.0;
    assert!(
        !supervision.needs_ruliad_policy_batch(),
        "enabling verifier config alone should not change loader shape"
    );

    supervision.verifier_reward.field_binding_contrast_weight = 0.01;
    assert!(supervision.needs_ruliad_policy_batch());

    supervision.verifier_reward.field_binding_contrast_weight = 0.0;
    supervision.verifier_reward.rollout_imitation_weight = 0.01;
    assert!(supervision.needs_ruliad_policy_batch());

    supervision.verifier_reward.rollout_imitation_weight = 0.0;
    supervision.answer_denoising.enabled = true;
    supervision.answer_denoising.structured_recovery_weight = 0.25;
    assert!(supervision.needs_ruliad_policy_batch());

    supervision.answer_denoising.enabled = false;
    supervision.answer_denoising.structured_recovery_weight = 0.0;
    supervision.answer_contract.enabled = true;
    supervision.answer_contract.weight = 0.25;
    assert!(supervision.needs_ruliad_policy_batch());
}

#[test]
fn ruliad_policy_batch_schedule_matches_active_auxiliary_cadence() {
    let mut supervision = crate::RuliadSupervisionConfig::default();
    supervision.proof_policy.enabled = true;
    supervision.proof_policy.weight = 0.25;
    supervision.proof_policy.start_after_steps = 4;
    supervision.proof_policy.every_steps = 3;

    for step in 0..10 {
        assert_eq!(
            supervision.needs_ruliad_policy_batch_at_step(step),
            matches!(step, 6 | 9),
            "step={step}"
        );
    }

    supervision.verifier_reward.enabled = true;
    supervision.verifier_reward.weight = 0.05;
    supervision.verifier_reward.start_after_steps = 2;
    supervision.verifier_reward.every_steps = 4;
    assert!(!supervision.needs_ruliad_policy_batch_at_step(2));
    assert!(!supervision.needs_ruliad_policy_batch_at_step(3));
    assert!(supervision.needs_ruliad_policy_batch_at_step(4));
    assert!(supervision.needs_ruliad_policy_batch_at_step(8));
}

#[test]
fn ruliad_1m_baseline_profile_validates_and_stays_small() {
    let config = load_profile("ruliad-1m.training.toml");
    config.validate().expect("ruliad-1m profile validates");
    assert!(matches!(
        &config.dataset.source,
        DatasetSourceConfig::UniversalityRuliad { .. }
    ));
    assert!(matches!(config.optimizer.name, OptimizerKind::Adamw));
    assert_eq!(
        config.training.ruliad_supervision.mode,
        RuliadSupervisionMode::AnswerCompletion
    );
    let estimated = estimate_profile_parameter_budget(&config);
    assert!(
        (750_000..=2_000_000).contains(&estimated),
        "ruliad-1m profile should stay in the fast diagnostic range, estimated params={estimated}"
    );
}

#[test]
fn ruliad_corpus_profiles_warm_start_without_hard_frontier_cap() {
    for profile in ["ruliad-1m.corpus.toml", "ruliad-r1.corpus.toml"] {
        let config = burn_dragon_universality::load_ruliad_config(&profile_path(profile))
            .unwrap_or_else(|err| panic!("load {profile}: {err}"));
        assert!(config.source_selection.enabled, "{profile}");
        assert!(
            config.source_selection.frontier_extension.enabled,
            "{profile}"
        );
        assert_eq!(
            config
                .source_selection
                .frontier_extension
                .max_materialized_levels,
            0,
            "{profile} should keep the live frontier unbounded"
        );
        assert!(
            config.source_selection.cold_start.enabled,
            "{profile} should warm-start cold models on easy buckets"
        );
        assert!(
            config.source_selection.cold_start.max_difficulty_level
                < config.source_selection.difficulty_levels.max,
            "{profile} cold-start cap should be below the initial materialized frontier"
        );
        assert_eq!(
            config.source_selection.cold_start.max_difficulty_level,
            config.source_selection.difficulty_levels.min,
            "{profile} should bootstrap from the easiest difficulty bucket"
        );
        assert!(
            config.source_selection.cold_start.release_requires_mastery,
            "{profile} should release cold-start difficulty by capability, not time alone"
        );
    }
}

#[test]
fn ruliad_1m_la16k_verifier_proxy_profiles_validate() {
    for (profile, ranking, denoising, structured_recovery) in [
        (
            "ruliad-1m-la-16k.answer-completion.self-recovery.training.toml",
            false,
            false,
            false,
        ),
        (
            "ruliad-1m-la-16k.answer-completion-ranking.self-recovery.training.toml",
            true,
            false,
            false,
        ),
        (
            "ruliad-1m-la-16k.answer-completion-recovery-denoising.self-recovery.training.toml",
            false,
            true,
            true,
        ),
        (
            "ruliad-1m-la-16k.answer-completion-denoising.self-recovery.training.toml",
            false,
            true,
            false,
        ),
        (
            "ruliad-1m-la-16k.answer-completion-ranking-denoising.self-recovery.training.toml",
            true,
            true,
            false,
        ),
        (
            "ruliad-1m-la-16k.field-binding-recovery.training.toml",
            false,
            true,
            true,
        ),
    ] {
        let config = load_profile(profile);
        config
            .validate()
            .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
        assert_eq!(
            config.training.ruliad_supervision.mode,
            RuliadSupervisionMode::AnswerCompletion,
            "{profile}"
        );
        assert!(
            config.training.ruliad_supervision.uses_target_loss_mask(),
            "{profile} should expose answer masks for verifier-proxy objectives"
        );
        assert_eq!(
            config.training.ruliad_supervision.answer_ranking.enabled, ranking,
            "{profile}"
        );
        assert_eq!(
            config.training.ruliad_supervision.answer_denoising.enabled, denoising,
            "{profile}"
        );
        assert_eq!(
            config
                .training
                .ruliad_supervision
                .answer_denoising
                .structured_recovery_weight
                > 0.0,
            structured_recovery,
            "{profile}"
        );
        if structured_recovery {
            if !profile.contains("field-binding") {
                assert!(
                    config.training.tbptt_chunk_size.is_none(),
                    "{profile} structured recovery must run in the non-TBPTT train path"
                );
                assert!(
                    !config.training.tbptt_persist_across_steps,
                    "{profile} structured recovery must run in the non-TBPTT train path"
                );
            }
            assert!(
                config
                    .training
                    .ruliad_supervision
                    .answer_denoising
                    .structured_recovery_schema_negative_count
                    > 0,
                "{profile} should include schema-collapse recovery negatives"
            );
        }
        assert_eq!(
            config
                .training
                .ruliad_supervision
                .needs_ruliad_policy_batch(),
            structured_recovery,
            "{profile}"
        );
    }
}

#[test]
fn ruliad_1m_la16k_verifier_reward_profile_validates() {
    let config = load_profile("ruliad-1m-la-16k.verifier-reward.training.toml");
    config
        .validate()
        .expect("ruliad verifier-reward profile should validate");
    assert!(config.training.ruliad_supervision.verifier_reward.enabled);
    assert_eq!(
        config.training.ruliad_supervision.mode,
        RuliadSupervisionMode::AnswerCompletion
    );
    assert_eq!(
        config.training.ruliad_supervision.verifier_reward.mode,
        RuliadVerifierRewardMode::Scalar
    );
    assert!(config.training.tbptt_chunk_size.is_none());
    assert!(!config.training.tbptt_persist_across_steps);
    assert!(config.training.objective.is_next_token());
}

#[test]
fn ruliad_1m_la16k_verifier_vpo_profile_validates() {
    for (profile, include_oracle_candidate, include_structured_negatives, structured_contrast) in [
        (
            "ruliad-1m-la-16k.verifier-vpo.training.toml",
            false,
            false,
            false,
        ),
        (
            "ruliad-1m-la-16k.verifier-vpo-oracle.training.toml",
            true,
            false,
            false,
        ),
        (
            "ruliad-1m-la-16k.verifier-vpo-oracle-structured.training.toml",
            true,
            true,
            false,
        ),
        (
            "ruliad-1m-la-16k.verifier-vpo-oracle-structured-contrast.training.toml",
            true,
            true,
            true,
        ),
    ] {
        let config = load_profile(profile);
        config
            .validate()
            .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
        assert!(
            config.training.ruliad_supervision.verifier_reward.enabled,
            "{profile}"
        );
        assert_eq!(
            config.training.ruliad_supervision.verifier_reward.mode,
            RuliadVerifierRewardMode::VpoIndependent,
            "{profile}"
        );
        assert_eq!(
            config.training.ruliad_supervision.mode,
            RuliadSupervisionMode::AnswerCompletion,
            "{profile}"
        );
        assert!(
            config
                .training
                .ruliad_supervision
                .verifier_reward
                .vpo_scalarizations
                > 0,
            "{profile}"
        );
        assert!(
            config
                .training
                .ruliad_supervision
                .verifier_reward
                .vpo_correctness_mass_floor
                >= 0.70,
            "{profile}"
        );
        assert!(
            config
                .training
                .ruliad_supervision
                .verifier_reward
                .vpo_schema_quality_mass_floor
                >= 0.10,
            "{profile}"
        );
        assert!(
            config
                .training
                .ruliad_supervision
                .verifier_reward
                .vpo_compactness_max_weight
                <= 0.05,
            "{profile}"
        );
        assert!(
            config
                .training
                .ruliad_supervision
                .verifier_reward
                .positive_advantage_requires_correctness,
            "{profile}"
        );
        assert!(
            config
                .training
                .ruliad_supervision
                .verifier_reward
                .positive_advantage_min_partial_progress_ppm
                >= 500_000,
            "{profile}"
        );
        assert!(
            config
                .training
                .ruliad_supervision
                .verifier_reward
                .positive_advantage_min_completion_quality_ppm
                >= 750_000,
            "{profile}"
        );
        assert_eq!(
            config
                .training
                .ruliad_supervision
                .verifier_reward
                .start_after_steps,
            512,
            "{profile}"
        );
        assert_eq!(
            config
                .training
                .ruliad_supervision
                .verifier_reward
                .max_advantage_clip_fraction,
            Some(0.95),
            "{profile}"
        );
        assert!(
            config
                .training
                .ruliad_supervision
                .verifier_reward
                .clip_range
                >= 1.0,
            "{profile}"
        );
        assert_eq!(
            config
                .training
                .ruliad_supervision
                .verifier_reward
                .include_oracle_candidate,
            include_oracle_candidate,
            "{profile}"
        );
        assert_eq!(
            config
                .training
                .ruliad_supervision
                .verifier_reward
                .include_structured_negative_candidates,
            include_structured_negatives,
            "{profile}"
        );
        if include_structured_negatives {
            assert!(
                config
                    .training
                    .ruliad_supervision
                    .verifier_reward
                    .structured_negative_count
                    > 0,
                "{profile}"
            );
            assert!(
                config
                    .training
                    .ruliad_supervision
                    .verifier_reward
                    .structured_template_negative_count
                    > 0,
                "{profile}"
            );
            assert!(
                config
                    .training
                    .ruliad_supervision
                    .verifier_reward
                    .structured_schema_negative_count
                    > 0,
                "{profile}"
            );
        }
        assert_eq!(
            config
                .training
                .ruliad_supervision
                .verifier_reward
                .structured_contrast_weight
                > 0.0,
            structured_contrast,
            "{profile}"
        );
        assert!(config.training.tbptt_chunk_size.is_none(), "{profile}");
        assert!(!config.training.tbptt_persist_across_steps, "{profile}");
        assert!(config.training.objective.is_next_token(), "{profile}");
    }
}

#[test]
fn ruliad_1m_la16k_structured_contrast_profile_validates_without_sampled_policy() {
    let profile = "ruliad-1m-la-16k.structured-contrast.training.toml";
    let config = load_profile(profile);
    config
        .validate()
        .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
    let verifier_reward = config.training.ruliad_supervision.verifier_reward;

    assert!(verifier_reward.enabled);
    assert_eq!(verifier_reward.weight, 0.0);
    assert!(verifier_reward.structured_negative_count > 0);
    assert_eq!(verifier_reward.structured_template_negative_count, 0);
    assert!(verifier_reward.structured_schema_negative_count > 0);
    assert!(verifier_reward.structured_contrast_weight > 0.0);
    assert_eq!(verifier_reward.structured_contrast_start_after_steps, 0);
    assert_eq!(
        config.training.ruliad_supervision.mode,
        RuliadSupervisionMode::AnswerCompletion
    );
    assert!(config.training.tbptt_chunk_size.is_none());
    assert!(!config.training.tbptt_persist_across_steps);
    assert!(
        config
            .training
            .ruliad_supervision
            .needs_ruliad_policy_batch()
    );
}

#[test]
fn ruliad_1m_la16k_field_binding_profile_validates_without_tbptt() {
    let profile =
        "ruliad-1m-la-16k.verifier-vpo-oracle-structured-contrast-field-binding.training.toml";
    let config = load_profile(profile);
    config
        .validate()
        .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
    let verifier_reward = config.training.ruliad_supervision.verifier_reward;

    assert!(verifier_reward.enabled);
    assert!(verifier_reward.field_binding_contrast_weight > 0.0);
    assert!(verifier_reward.structured_schema_negative_count > 0);
    assert!(verifier_reward.structured_contrast_weight > 0.0);
    assert_eq!(verifier_reward.field_binding_contrast_start_after_steps, 0);
    assert_eq!(verifier_reward.field_binding_contrast_every_steps, 8);
    assert_eq!(verifier_reward.field_binding_contrast_pair_weight, 0.5);
    assert_eq!(verifier_reward.field_binding_contrast_max_pairs, 24);
    assert_eq!(verifier_reward.field_binding_contrast_replay_capacity, 64);
    assert_eq!(verifier_reward.generated_attractor_replay_capacity, 128);
    assert_eq!(verifier_reward.generated_attractor_replay_min_count, 2);
    assert_eq!(verifier_reward.generated_attractor_replay_max_candidates, 4);
    assert_eq!(
        verifier_reward.generated_attractor_replay_min_distinct_answers,
        2
    );
    assert_eq!(
        verifier_reward.generated_attractor_replay_max_dominant_fraction,
        0.5
    );
    assert_eq!(
        config.training.ruliad_supervision.mode,
        RuliadSupervisionMode::AnswerCompletion
    );
    assert!(config.training.tbptt_chunk_size.is_none());
    assert!(!config.training.tbptt_persist_across_steps);
    assert!(
        config
            .training
            .ruliad_supervision
            .needs_ruliad_policy_batch()
    );
}

#[test]
fn ruliad_1m_la16k_field_binding_only_profile_validates_without_policy_reward() {
    let profile = "ruliad-1m-la-16k.field-binding-contrast.training.toml";
    let config = load_profile(profile);
    config
        .validate()
        .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
    let verifier_reward = config.training.ruliad_supervision.verifier_reward;

    assert!(verifier_reward.enabled);
    assert_eq!(verifier_reward.weight, 0.0);
    assert!(verifier_reward.field_binding_contrast_weight > 0.0);
    assert_eq!(verifier_reward.field_binding_contrast_every_steps, 4);
    assert_eq!(verifier_reward.field_binding_contrast_pair_weight, 1.0);
    assert_eq!(verifier_reward.field_binding_contrast_max_pairs, 16);
    assert_eq!(verifier_reward.field_binding_contrast_replay_capacity, 128);
    assert!(config.training.tbptt_chunk_size.is_none());
    assert!(!config.training.tbptt_persist_across_steps);
    assert!(
        config
            .training
            .ruliad_supervision
            .needs_ruliad_policy_batch()
    );
}

#[test]
fn ruliad_1m_la64k_field_binding_profile_validates_with_tbptt() {
    let profile = "ruliad-1m-la-64k.field-binding-contrast.training.toml";
    let config = load_profile(profile);
    config
        .validate()
        .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
    let verifier_reward = config.training.ruliad_supervision.verifier_reward;

    assert!(verifier_reward.enabled);
    assert_eq!(verifier_reward.weight, 0.0);
    assert_eq!(verifier_reward.field_binding_contrast_weight, 0.05);
    assert_eq!(verifier_reward.field_binding_contrast_every_steps, 8);
    assert_eq!(verifier_reward.field_binding_contrast_pair_weight, 0.5);
    assert_eq!(verifier_reward.field_binding_contrast_max_pairs, 8);
    assert_eq!(verifier_reward.field_binding_contrast_replay_capacity, 64);
    assert_eq!(config.model.latent_total, Some(65_536));
    assert_eq!(config.training.tbptt_chunk_size, Some(128));
    assert!(config.training.tbptt_persist_across_steps);
    assert!(
        config
            .training
            .ruliad_supervision
            .needs_ruliad_policy_batch()
    );
}

#[test]
fn ruliad_1m_la64k_structured_recovery_profile_validates_with_tbptt() {
    let profile = "ruliad-1m-la-64k.answer-completion-recovery.training.toml";
    let config = load_profile(profile);
    config
        .validate()
        .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
    let denoising = config.training.ruliad_supervision.answer_denoising;

    assert!(denoising.enabled);
    assert_eq!(denoising.weight, 0.0);
    assert_eq!(denoising.structured_recovery_weight, 0.25);
    assert_eq!(denoising.structured_recovery_every_steps, 4);
    assert_eq!(denoising.structured_recovery_schema_negative_count, 4);
    assert_eq!(
        config.training.ruliad_supervision.mode,
        RuliadSupervisionMode::AnswerCompletion
    );
    assert_eq!(config.model.latent_total, Some(65_536));
    assert_eq!(config.training.tbptt_chunk_size, Some(128));
    assert!(config.training.tbptt_persist_across_steps);
    assert!(
        config
            .training
            .ruliad_supervision
            .needs_ruliad_policy_batch()
    );
}

#[test]
fn ruliad_1m_la64k_answer_contract_profile_validates_with_tbptt() {
    let profile = "ruliad-1m-la-64k.answer-contract.training.toml";
    let config = load_profile(profile);
    config
        .validate()
        .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
    let contract = config.training.ruliad_supervision.answer_contract;

    assert!(contract.enabled);
    assert_eq!(contract.weight, 0.25);
    assert_eq!(contract.premature_close_unlikelihood_weight, 0.5);
    assert_eq!(contract.every_steps, 1);
    assert_eq!(contract.max_completion_tokens, 64);
    assert_eq!(contract.max_rows_per_step, 8);
    assert_eq!(
        config.training.ruliad_supervision.mode,
        RuliadSupervisionMode::AnswerCompletion
    );
    assert_eq!(config.model.latent_total, Some(65_536));
    assert_eq!(
        config
            .model
            .latent_reasoning
            .as_ref()
            .expect("answer-contract profile should configure latent reasoning")
            .max_steps,
        2
    );
    assert_eq!(config.training.tbptt_chunk_size, Some(128));
    assert!(config.training.tbptt_persist_across_steps);
    assert!(
        config
            .training
            .ruliad_supervision
            .needs_ruliad_policy_batch()
    );
}

#[test]
fn ruliad_1m_la64k_answer_contract_schema_profile_validates_with_tbptt() {
    let profile = "ruliad-1m-la-64k.answer-contract-schema.training.toml";
    let config = load_profile(profile);
    config
        .validate()
        .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
    let supervision = config.training.ruliad_supervision;
    let contract = supervision.answer_contract;

    assert!(contract.enabled);
    assert_eq!(contract.weight, 0.25);
    assert_eq!(contract.premature_close_unlikelihood_weight, 1.0);
    assert_eq!(contract.schema_token_weight, 4.0);
    assert_eq!(contract.schema_start_token_weight, 0.0);
    assert_eq!(contract.value_token_weight, 1.0);
    assert_eq!(contract.other_token_weight, 0.25);
    assert_eq!(supervision.answer_close_marker_stride, 4);
    assert_eq!(supervision.answer_schema_token_weight, 4);
    assert_eq!(supervision.answer_schema_start_token_weight, 1);
    assert_eq!(supervision.answer_value_token_weight, 1);
    assert_eq!(config.model.latent_total, Some(65_536));
    assert_eq!(config.training.tbptt_chunk_size, Some(128));
    assert!(config.training.tbptt_persist_across_steps);
    assert!(supervision.needs_ruliad_policy_batch());
}

#[test]
fn ruliad_1m_la64k_answer_contract_schema_start_profile_validates_with_tbptt() {
    let profile = "ruliad-1m-la-64k.answer-contract-schema-start.training.toml";
    let config = load_profile(profile);
    config
        .validate()
        .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
    let supervision = config.training.ruliad_supervision;
    let contract = supervision.answer_contract;

    assert!(contract.enabled);
    assert_eq!(contract.weight, 0.25);
    assert_eq!(contract.schema_token_weight, 4.0);
    assert_eq!(contract.schema_start_token_weight, 16.0);
    assert_eq!(contract.value_token_weight, 1.0);
    assert_eq!(supervision.answer_close_marker_stride, 4);
    assert_eq!(supervision.answer_schema_token_weight, 4);
    assert_eq!(supervision.answer_schema_start_token_weight, 12);
    assert_eq!(supervision.answer_value_token_weight, 1);
    assert_eq!(config.model.latent_total, Some(65_536));
    assert_eq!(config.training.tbptt_chunk_size, Some(128));
    assert!(config.training.tbptt_persist_across_steps);
    assert!(supervision.needs_ruliad_policy_batch());
}

#[test]
fn ruliad_1m_la64k_answer_contract_schema_trace_answer_profile_validates_with_tbptt() {
    let profile = "ruliad-1m-la-64k.answer-contract-schema-trace-answer.training.toml";
    let config = load_profile(profile);
    config
        .validate()
        .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
    let supervision = config.training.ruliad_supervision;
    let contract = supervision.answer_contract;

    assert_eq!(supervision.mode, RuliadSupervisionMode::TraceAndAnswer);
    assert!(supervision.mask_high_entropy_spans);
    assert!(supervision.uses_answer_target_mask());
    assert!(supervision.uses_trace_answer_target_mask());
    assert!(contract.enabled);
    assert_eq!(contract.weight, 0.25);
    assert_eq!(contract.schema_token_weight, 4.0);
    assert_eq!(contract.schema_start_token_weight, 0.0);
    assert_eq!(contract.value_token_weight, 1.0);
    assert_eq!(supervision.answer_close_marker_stride, 4);
    assert_eq!(supervision.answer_schema_token_weight, 4);
    assert_eq!(supervision.answer_schema_start_token_weight, 1);
    assert_eq!(supervision.answer_value_token_weight, 1);
    assert_eq!(config.model.latent_total, Some(65_536));
    assert_eq!(config.training.tbptt_chunk_size, Some(128));
    assert!(config.training.tbptt_persist_across_steps);
    assert!(supervision.uses_target_loss_mask());
    assert!(supervision.needs_ruliad_policy_batch());
}

#[test]
fn ruliad_1m_la64k_answer_contract_schema_mixed_trace_profile_validates_with_tbptt() {
    let profile = "ruliad-1m-la-64k.answer-contract-schema-mixed-trace.training.toml";
    let config = load_profile(profile);
    config
        .validate()
        .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
    let supervision = config.training.ruliad_supervision;
    let contract = supervision.answer_contract;

    assert_eq!(supervision.mode, RuliadSupervisionMode::Mixed);
    assert!(supervision.mask_high_entropy_spans);
    assert!(contract.enabled);
    assert_eq!(contract.weight, 0.25);
    assert_eq!(contract.schema_token_weight, 4.0);
    assert_eq!(contract.schema_start_token_weight, 0.0);
    assert_eq!(contract.value_token_weight, 1.0);
    assert_eq!(supervision.answer_close_marker_stride, 4);
    assert_eq!(supervision.answer_schema_token_weight, 4);
    assert_eq!(supervision.answer_schema_start_token_weight, 1);
    assert_eq!(supervision.answer_value_token_weight, 1);
    assert_eq!(config.model.latent_total, Some(65_536));
    assert_eq!(config.training.tbptt_chunk_size, Some(128));
    assert!(config.training.tbptt_persist_across_steps);
    assert!(supervision.uses_target_loss_mask());
    assert!(supervision.needs_ruliad_policy_batch());
}

#[test]
fn ruliad_1m_la64k_answer_contract_schema_field_binding_profile_validates_with_tbptt() {
    let profile = "ruliad-1m-la-64k.answer-contract-schema-field-binding.training.toml";
    let config = load_profile(profile);
    config
        .validate()
        .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
    let supervision = config.training.ruliad_supervision;
    let contract = supervision.answer_contract;
    let verifier_reward = supervision.verifier_reward;

    assert!(contract.enabled);
    assert_eq!(contract.weight, 0.25);
    assert_eq!(contract.premature_close_unlikelihood_weight, 1.0);
    assert_eq!(contract.schema_token_weight, 4.0);
    assert_eq!(contract.schema_start_token_weight, 0.0);
    assert_eq!(contract.value_token_weight, 1.0);
    assert_eq!(supervision.answer_close_marker_stride, 4);
    assert_eq!(supervision.answer_schema_token_weight, 4);
    assert_eq!(supervision.answer_schema_start_token_weight, 1);
    assert_eq!(supervision.answer_value_token_weight, 1);
    assert!(verifier_reward.enabled);
    assert_eq!(verifier_reward.weight, 0.0);
    assert_eq!(verifier_reward.field_binding_contrast_weight, 0.05);
    assert_eq!(verifier_reward.field_binding_contrast_every_steps, 8);
    assert_eq!(verifier_reward.field_binding_contrast_pair_weight, 0.5);
    assert_eq!(verifier_reward.field_binding_contrast_max_pairs, 8);
    assert_eq!(verifier_reward.field_binding_contrast_replay_capacity, 64);
    assert_eq!(config.model.latent_total, Some(65_536));
    assert_eq!(config.training.tbptt_chunk_size, Some(128));
    assert!(config.training.tbptt_persist_across_steps);
    assert!(supervision.needs_ruliad_policy_batch());
}

#[test]
fn ruliad_1m_la64k_answer_contract_value_binding_profile_validates_with_tbptt() {
    let profile = "ruliad-1m-la-64k.answer-contract-value-binding.training.toml";
    let config = load_profile(profile);
    config
        .validate()
        .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
    let supervision = config.training.ruliad_supervision;
    let contract = supervision.answer_contract;
    let verifier_reward = supervision.verifier_reward;

    assert!(contract.enabled);
    assert_eq!(contract.weight, 0.25);
    assert_eq!(contract.schema_token_weight, 4.0);
    assert_eq!(contract.value_token_weight, 1.0);
    assert_eq!(contract.prompt_schema_value_weight, 4.0);
    assert_eq!(contract.prompt_schema_max_rows_per_step, 4);
    assert!(verifier_reward.enabled);
    assert_eq!(verifier_reward.field_binding_contrast_weight, 0.05);
    assert_eq!(
        verifier_reward.field_binding_contrast_rank_metric_every_steps,
        8
    );
    assert_eq!(config.model.latent_total, Some(65_536));
    assert_eq!(config.training.tbptt_chunk_size, Some(128));
    assert!(config.training.tbptt_persist_across_steps);
    assert!(supervision.needs_ruliad_policy_batch());
}

#[test]
fn ruliad_1m_la64k_answer_contract_values_profile_validates_with_tbptt() {
    let profile = "ruliad-1m-la-64k.answer-contract-values.training.toml";
    let config = load_profile(profile);
    config
        .validate()
        .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
    let supervision = config.training.ruliad_supervision;
    let contract = supervision.answer_contract;

    assert!(contract.enabled);
    assert_eq!(contract.weight, 0.50);
    assert_eq!(contract.premature_close_unlikelihood_weight, 0.75);
    assert_eq!(contract.schema_token_weight, 1.0);
    assert_eq!(contract.value_token_weight, 8.0);
    assert_eq!(contract.other_token_weight, 0.25);
    assert_eq!(supervision.answer_close_marker_stride, 1);
    assert_eq!(supervision.answer_close_marker_weight, 2);
    assert_eq!(supervision.answer_schema_token_weight, 1);
    assert_eq!(supervision.answer_value_token_weight, 6);
    assert_eq!(config.model.latent_total, Some(65_536));
    assert_eq!(config.training.tbptt_chunk_size, Some(128));
    assert!(config.training.tbptt_persist_across_steps);
    assert!(supervision.needs_ruliad_policy_batch());
}

#[test]
fn ruliad_1m_la16k_verifier_rollout_imitation_profile_validates_without_tbptt() {
    let profile = "ruliad-1m-la-16k.verifier-rollout-imitation.training.toml";
    let config = load_profile(profile);
    config
        .validate()
        .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
    let verifier_reward = config.training.ruliad_supervision.verifier_reward;

    assert!(verifier_reward.enabled);
    assert_eq!(verifier_reward.weight, 0.0);
    assert_eq!(verifier_reward.structured_contrast_weight, 0.0);
    assert!(verifier_reward.rollout_imitation_weight > 0.0);
    assert_eq!(verifier_reward.rollout_imitation_start_after_steps, 128);
    assert_eq!(
        verifier_reward.rollout_imitation_min_verifier_rate_ppm,
        100_000
    );
    assert_eq!(
        verifier_reward.rollout_imitation_max_schema_wrong_rate_ppm,
        250_000
    );
    assert_eq!(
        config.training.ruliad_supervision.mode,
        RuliadSupervisionMode::AnswerCompletion
    );
    assert!(config.training.tbptt_chunk_size.is_none());
    assert!(!config.training.tbptt_persist_across_steps);
    assert!(
        config
            .training
            .ruliad_supervision
            .needs_ruliad_policy_batch()
    );
}

#[test]
fn ruliad_1m_jepa_default_profiles_validate() {
    for profile in [
        "ruliad-1m.jepa.training.toml",
        "ruliad-1m-la-16k.jepa.training.toml",
        "ruliad-1m-la-32k.jepa.training.toml",
        "ruliad-1m-la-64k.jepa.training.toml",
    ] {
        let config = load_profile(profile);
        config
            .validate()
            .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
        assert!(
            config
                .model
                .latent_reasoning
                .as_ref()
                .is_some_and(|latent| latent.enabled),
            "{profile} should enable the JEPA latent reasoning model module"
        );
        assert!(
            config.training.latent_reasoning.enabled,
            "{profile} should enable JEPA latent training"
        );
        assert_eq!(
            config.training.latent_reasoning.jepa_future_offsets,
            vec![1],
            "{profile} should use JEPA-only future hidden prediction"
        );
        assert!(
            !config.training.latent_reasoning.next_latent.enabled,
            "{profile} should not enable NextLat by default"
        );
    }
}

#[test]
fn ruliad_10m_screening_profiles_validate_capability_gates() {
    for profile in [
        "ruliad-r1.jepa-10m-screening.toml",
        "ruliad-r1.jepa-nextlat-10m-screening.toml",
    ] {
        let config = load_profile(profile);
        config
            .validate()
            .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
        assert!(config.training.latent_reasoning.enabled, "{profile}");
        assert_eq!(
            config.training.events.ruliad_correctness_probe_items, 128,
            "{profile}"
        );
        assert!(
            config.training.events.source_selection_capability_feedback,
            "{profile} should feed capability probes back into live source selection by default"
        );
        assert_eq!(
            config
                .training
                .gates
                .capability_zero_verifier_patience_epochs,
            8,
            "{profile}"
        );
        assert_eq!(
            config.training.gates.capability_grace_epochs, 3,
            "{profile}"
        );
        assert_eq!(
            config.training.gates.capability_regression_patience_epochs, 2,
            "{profile}"
        );
        assert!(
            config.training.gates.capability_required_after_first_pass,
            "{profile}"
        );
        assert_eq!(
            config.training.gates.capability_schema_wrong_max_rate, 0.50,
            "{profile}"
        );
        assert_eq!(
            config.training.gates.capability_malformed_max_rate, 0.02,
            "{profile}"
        );
        assert_eq!(
            config.training.gates.capability_missing_max_rate, 0.02,
            "{profile}"
        );
        assert_eq!(
            config.training.gates.capability_completion_health_min_rate, 0.40,
            "{profile}"
        );
        assert_eq!(
            config.training.gates.capability_output_entropy_min_bits, 1.25,
            "{profile}"
        );
        assert_eq!(
            config.training.gates.capability_distinct_2_min_fraction, 0.30,
            "{profile}"
        );
        assert_eq!(
            config
                .training
                .gates
                .capability_answer_distinct_min_fraction,
            0.20,
            "{profile}"
        );
        assert_eq!(
            config
                .training
                .gates
                .capability_field_value_distinct_ratio_min,
            0.35,
            "{profile}"
        );
        assert_eq!(
            config.training.gates.capability_field_value_dominance_max, 0.85,
            "{profile}"
        );
    }
}

#[test]
fn ruliad_latent_energy_ablation_profile_validates() {
    for profile in [
        "ruliad-r1.jepa-nextlat-energy-probe128-fixed-ablation.toml",
        "ruliad-r1.jepa-nextlat-energy-contrastive-probe128-fixed-ablation.toml",
        "ruliad-r1.jepa-nextlat-energy-stability-probe128-fixed-ablation.toml",
    ] {
        let config = load_profile(profile);
        config
            .validate()
            .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
        let latent = config
            .model
            .latent_reasoning
            .as_ref()
            .unwrap_or_else(|| panic!("{profile} should configure latent reasoning"));
        assert!(latent.enabled, "{profile}");
        assert!(latent.energy_head, "{profile}");
        assert!(config.training.latent_reasoning.enabled, "{profile}");
        assert!(
            config.training.latent_reasoning.energy_model.enabled,
            "{profile} should enable latent EBM training"
        );
        assert_eq!(
            config.training.latent_reasoning.eval_step_sweep,
            vec![1, 2, 4, 8],
            "{profile}"
        );
    }
}

#[test]
fn ruliad_step_contract_ablation_profile_validates() {
    let profile = "ruliad-r1.jepa-nextlat-step-contract-probe128-fixed-ablation.toml";
    let config = load_profile(profile);
    config
        .validate()
        .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
    let latent = config
        .model
        .latent_reasoning
        .as_ref()
        .unwrap_or_else(|| panic!("{profile} should configure latent reasoning"));
    assert!(latent.enabled, "{profile}");
    assert!(!latent.energy_head, "{profile}");
    assert!(config.training.latent_reasoning.enabled, "{profile}");
    assert!(
        config.training.latent_reasoning.step_contract.enabled,
        "{profile} should enable latent step contract training"
    );
    assert_eq!(
        config.training.latent_reasoning.eval_step_sweep,
        vec![1, 2, 4, 8],
        "{profile}"
    );
}

#[test]
fn ruliad_hierarchical_dragon_ablation_profiles_validate() {
    for (profile, rho_sharing, weight_sharing) in [
        (
            "ruliad-r1.hdragon-shared-rho-shared-weights-probe128-fixed-ablation.toml",
            HierarchicalDragonSharing::Shared,
            HierarchicalDragonSharing::Shared,
        ),
        (
            "ruliad-r1.hdragon-split-rho-shared-weights-probe128-fixed-ablation.toml",
            HierarchicalDragonSharing::Split,
            HierarchicalDragonSharing::Shared,
        ),
        (
            "ruliad-r1.hdragon-split-rho-split-weights-probe128-fixed-ablation.toml",
            HierarchicalDragonSharing::Split,
            HierarchicalDragonSharing::Split,
        ),
    ] {
        let config = load_profile(profile);
        config
            .validate()
            .unwrap_or_else(|err| panic!("{profile} should validate: {err}"));
        let hierarchy = config
            .model
            .hierarchical_dragon
            .as_ref()
            .unwrap_or_else(|| panic!("{profile} should configure hierarchical Dragon"));
        assert!(hierarchy.enabled, "{profile}");
        assert_eq!(hierarchy.rho_sharing, rho_sharing, "{profile}");
        assert_eq!(hierarchy.weight_sharing, weight_sharing, "{profile}");
        assert_eq!(hierarchy.last_layers, Some(1), "{profile}");
        assert!(config.training.latent_reasoning.enabled, "{profile}");
        assert!(
            config.training.latent_reasoning.next_latent.enabled,
            "{profile} should inherit NextLat training"
        );
    }
}

#[test]
fn ruliad_1m_high_neuron_sweep_profiles_resolve_expected_long_context_shape() {
    for (profile, latent_total, batch_size) in [
        ("ruliad-1m-la-16k.training.toml", 16_384, 1),
        ("ruliad-1m-la-32k.training.toml", 32_768, 1),
        ("ruliad-1m-la-64k.training.toml", 65_536, 1),
    ] {
        let config = load_profile(profile);
        config.validate().unwrap_or_else(|err| {
            panic!("{profile} should validate as a safe high-neuron sweep profile: {err}")
        });
        assert!(matches!(
            &config.dataset.source,
            DatasetSourceConfig::UniversalityRuliad { .. }
        ));
        assert!(matches!(config.optimizer.name, OptimizerKind::Adamw));
        assert_eq!(config.optimizer.learning_rate, 3.0e-4, "{profile}");
        assert_eq!(config.model.n_layer, Some(4), "{profile}");
        assert_eq!(config.model.n_embd, Some(256), "{profile}");
        assert_eq!(config.model.n_head, Some(4), "{profile}");
        assert_eq!(config.model.latent_total, Some(latent_total), "{profile}");
        assert_eq!(config.training.block_size, 256, "{profile}");
        assert_eq!(config.training.tbptt_chunk_size, Some(128), "{profile}");
        assert!(config.training.tbptt_persist_across_steps, "{profile}");
        assert_eq!(
            config.training.min_logical_block_size,
            Some(512),
            "{profile}"
        );
        assert_eq!(
            config.training.ruliad_supervision.mode,
            RuliadSupervisionMode::AnswerWindow,
            "{profile}"
        );
        assert!(
            config.training.input_corruption.enabled,
            "{profile} should use input corruption as cheap continual-learning regularization"
        );
        assert!(
            config.training.logit_entropy_floor.enabled,
            "{profile} should keep a minimum token-distribution entropy floor"
        );
        assert!(
            config.training.repeat_unlikelihood.enabled,
            "{profile} should penalize short-period repetition"
        );
        assert!(
            config.training.greedy_rollout_unlikelihood.enabled
                && config.training.greedy_rollout_unlikelihood.recovery_only,
            "{profile} should keep expensive rollout anti-collapse pressure recovery-only"
        );
        assert!(
            config.training.dynamics_anchor.enabled && config.training.dynamics_anchor.weight > 0.0,
            "{profile} should constrain next-token distribution drift with an EMA dynamics anchor"
        );
        assert!(
            !config.training.auto_batch_size.enabled,
            "{profile} should rely on the guarded sweep wrapper, not startup auto-batch probing"
        );
        assert_eq!(config.training.batch_size, batch_size, "{profile}");
        assert_eq!(
            config.training.gates.degeneracy_eos_max_fraction, 0.20,
            "{profile}"
        );
        assert_eq!(
            config.training.gates.degeneracy_period_3_max_fraction, 0.25,
            "{profile}"
        );
        assert_eq!(
            config.training.gates.degeneracy_period_2_to_64_max_fraction, 0.25,
            "{profile}"
        );

        let model = build_model_config(&config.model, config.training.block_size);
        assert_eq!(model.latent_total(), latent_total, "{profile}");
        assert_eq!(
            model.sequence_kernel.memory_system,
            SequenceMemorySystem::LinearAttention,
            "{profile}"
        );
        assert_eq!(
            model.fused_kernels.rotary_embedding,
            RotaryEmbedding::Alibi,
            "{profile}"
        );
    }
}

fn load_profile(file_name: &str) -> TrainingConfig {
    let profile_path = profile_path(file_name);
    load_training_config(std::slice::from_ref(&profile_path))
        .unwrap_or_else(|err| panic!("load {}: {err}", profile_path.display()))
}

#[test]
fn ruliad_r3_profile_streams_the_full_formal_proof_contract() {
    let config = load_profile("ruliad-r3.training.toml");
    config.validate().expect("R3 profile should validate");

    assert_eq!(config.training.tbptt_chunk_size, Some(512));
    assert!(config.training.tbptt_persist_across_steps);
    assert_eq!(
        config.training.ruliad_supervision.mode,
        RuliadSupervisionMode::TraceAndAnswer
    );
    assert!(config.training.ruliad_supervision.mask_high_entropy_spans);
}

#[test]
fn ruliad_r3_stateful_tbptt_profiles_form_a_matched_factorial_ablation() {
    use crate::config::SequenceBatchingMode;

    let arms = [
        ("ruliad-r3.stateful-tbptt-block512-reset.toml", 512, false),
        ("ruliad-r3.stateful-tbptt-block512-carry.toml", 512, true),
        ("ruliad-r3.stateful-tbptt-chunk128-reset.toml", 128, false),
        ("ruliad-r3.stateful-tbptt-chunk128-carry.toml", 128, true),
        ("ruliad-r3.stateful-tbptt-chunk64-carry.toml", 64, true),
    ];
    let mut shared_contract = None;
    for (profile, chunk_size, persist) in arms {
        let config = load_profile(profile);
        config
            .validate()
            .unwrap_or_else(|error| panic!("{profile} should validate: {error}"));
        assert_eq!(config.training.block_size, 512, "{profile}");
        assert_eq!(
            config.training.tbptt_chunk_size,
            Some(chunk_size),
            "{profile}"
        );
        assert_eq!(
            config.training.tbptt_persist_across_steps, persist,
            "{profile}"
        );
        assert_eq!(
            config.training.sequence_batching,
            SequenceBatchingMode::Streaming,
            "{profile}"
        );
        assert!(config.training.sequence_state_probe.enabled, "{profile}");
        assert!(
            config.training.ruliad_supervision.balance_trace_answer_mass,
            "{profile}"
        );
        assert!(!config.training.auto_batch_size.enabled, "{profile}");
        assert!(!config.training.continual_backprop.enabled, "{profile}");
        assert!(!config.training.neuron_scaling.enabled, "{profile}");
        assert!(!config.training.gates.enabled, "{profile}");
        assert!(!config.training.dynamics.enabled, "{profile}");
        let contract = (
            config.dataset.clone(),
            config.model.clone(),
            config.optimizer.clone(),
            config.training.ruliad_supervision,
            config.training.ruliad_probe_generation,
            config.training.objective.clone(),
            config.training.batch_size,
            config.training.seed,
        );
        if let Some(expected) = shared_contract.as_ref() {
            assert_eq!(
                &contract, expected,
                "{profile} changed a controlled variable"
            );
        } else {
            shared_contract = Some(contract);
        }
    }

    let corpus_path = profile_path("ruliad-r3.stateful-tbptt.corpus.toml");
    let corpus = burn_dragon_universality::load_ruliad_config(&corpus_path)
        .unwrap_or_else(|error| panic!("load {}: {error}", corpus_path.display()));
    assert_eq!(corpus.serialization.document_mode.label(), "single_sample");
    assert_eq!(corpus.serialization.document_chunks.min, 1);
    assert_eq!(corpus.serialization.document_chunks.max, 1);
    assert_eq!(corpus.serialization.document_tokens, 6145);
    assert!(corpus.source_selection.enabled);
    assert!(!corpus.source_selection.feedback_updates_enabled);
}

#[test]
fn ruliad_r3_typed_policy_profile_has_a_long_run_semantic_action_contract() {
    use crate::config::{RuliadProofPolicyCandidateSymmetry, RuliadProofPolicyTrainingMode};

    let config = load_profile("ruliad-r3.typed-policy.training.toml");
    config
        .validate()
        .expect("R3 typed-policy profile should validate");

    assert_eq!(config.training.max_iters, 1_000_000);
    assert!(config.training.auto_batch_size.enabled);
    assert_eq!(config.training.tbptt_chunk_size, Some(512));
    assert!(!config.training.tbptt_persist_across_steps);
    assert_eq!(
        config.training.ruliad_supervision.mode,
        RuliadSupervisionMode::AnswerCompletion
    );
    let policy = config.training.ruliad_supervision.proof_policy;
    assert!(policy.enabled);
    assert_eq!(policy.mode, RuliadProofPolicyTrainingMode::StaticExpert);
    assert_eq!(policy.every_steps, 2);
    assert_eq!(policy.start_after_steps, 128);
    assert_eq!(policy.max_rows_per_update, 8);
    assert_eq!(
        policy.candidate_symmetry,
        RuliadProofPolicyCandidateSymmetry::BalancedRotation
    );
    assert_eq!(
        config.training.ruliad_policy_probe.candidate_symmetry,
        RuliadProofPolicyCandidateSymmetry::CyclicOrbitAverage
    );
    assert_eq!(
        config
            .training
            .ruliad_policy_probe
            .effective_closed_loop_every_epochs(),
        16
    );
    assert!(config.training.ruliad_policy_probe.promotion_gate.enabled);
}

#[test]
fn ruliad_r3_semantic_energy_profile_decouples_policy_from_language_serialization() {
    use crate::config::RuliadProofPolicyScoring;

    let config = load_profile("ruliad-r3.action-policy-semantic-energy-fixed-ablation.toml");
    config
        .validate()
        .expect("R3 semantic-energy profile should validate");
    assert!(
        config
            .model
            .sequence_score_head
            .is_some_and(|head| head.enabled)
    );
    assert_eq!(
        config.training.ruliad_supervision.proof_policy.scoring,
        RuliadProofPolicyScoring::SemanticEnergy
    );
    assert_eq!(
        config.training.ruliad_policy_probe.scoring,
        RuliadProofPolicyScoring::SemanticEnergy
    );
    let proof_policy = config.training.ruliad_supervision.proof_policy;
    assert_eq!(proof_policy.counterfactual_targets_per_state, 1);
    assert_eq!(proof_policy.target_variants_per_state(), 2);
    assert_eq!(proof_policy.semantic_rows_per_update(), 8);
    assert_eq!(proof_policy.base_semantic_rows_per_update(), 4);
    assert!(
        build_model_config(&config.model, config.training.block_size)
            .sequence_score_head
            .enabled
    );
    assert_eq!(
        build_model_config(&config.model, config.training.block_size)
            .sequence_score_head
            .projection_dim,
        64
    );
}

#[test]
fn ruliad_r3_semantic_energy_head_only_profile_is_explicit_and_valid() {
    use crate::config::{RuliadProofPolicyGradientScope, RuliadProofPolicyScoring};

    let config =
        load_profile("ruliad-r3.action-policy-semantic-energy-head-only-fixed-ablation.toml");
    config
        .validate()
        .expect("R3 head-only semantic-energy profile should validate");
    let policy = config.training.ruliad_supervision.proof_policy;
    assert_eq!(policy.scoring, RuliadProofPolicyScoring::SemanticEnergy);
    assert_eq!(
        policy.gradient_scope,
        RuliadProofPolicyGradientScope::ScoreHeadOnly
    );

    let fullrate =
        load_profile("ruliad-r3.action-policy-semantic-energy-head-only-fullrate-ablation.toml");
    fullrate
        .validate()
        .expect("R3 full-rate head-only semantic-energy profile should validate");
    let policy = fullrate.training.ruliad_supervision.proof_policy;
    assert_eq!(policy.scoring, RuliadProofPolicyScoring::SemanticEnergy);
    assert_eq!(
        policy.gradient_scope,
        RuliadProofPolicyGradientScope::ScoreHeadOnly
    );
    assert_eq!(policy.weight, 1.0);
    assert_eq!(policy.every_steps, 1);
    assert_eq!(policy.start_after_steps, 0);
    assert_eq!(policy.max_rows_per_update, 32);
    assert_eq!(policy.max_presentation_rows_per_update, 32);
}

#[test]
fn score_head_only_gradient_scope_requires_semantic_energy() {
    use crate::config::{RuliadProofPolicyGradientScope, RuliadProofPolicyScoring};

    let mut config = load_profile("ruliad-r3.action-policy-semantic-energy-fixed-ablation.toml");
    let policy = &mut config.training.ruliad_supervision.proof_policy;
    policy.scoring = RuliadProofPolicyScoring::CompletionLikelihood;
    policy.gradient_scope = RuliadProofPolicyGradientScope::ScoreHeadOnly;
    let error = config
        .validate()
        .expect_err("head-only scope must not silently target the language head");
    assert!(
        error
            .to_string()
            .contains("gradient_scope=score_head_only requires scoring=semantic_energy"),
        "{error}"
    );
}

#[test]
fn residual_energy_accepts_score_head_only_counterfactual_training_and_probe() {
    use crate::config::{RuliadProofPolicyGradientScope, RuliadProofPolicyScoring};

    let mut config = load_profile("ruliad-r3.action-policy-semantic-energy-fixed-ablation.toml");
    let policy = &mut config.training.ruliad_supervision.proof_policy;
    policy.scoring = RuliadProofPolicyScoring::ResidualEnergy;
    policy.gradient_scope = RuliadProofPolicyGradientScope::ScoreHeadOnly;
    policy.counterfactual_targets_per_state = 1;
    config.training.ruliad_policy_probe.scoring = RuliadProofPolicyScoring::ResidualEnergy;

    config
        .validate()
        .expect("residual energy should accept a detached autoregressive prior and score-head-only correction");
}

#[test]
fn language_head_only_profile_is_explicit_untied_and_valid() {
    use crate::config::{RuliadProofPolicyGradientScope, RuliadProofPolicyScoring};

    let config = load_profile("ruliad-r3.semantic-action-language-head-only-fixed-ablation.toml");
    config
        .validate()
        .expect("R3 language-head-only completion profile should validate");
    let policy = config.training.ruliad_supervision.proof_policy;
    assert_eq!(
        policy.scoring,
        RuliadProofPolicyScoring::CompletionLikelihood
    );
    assert_eq!(
        policy.gradient_scope,
        RuliadProofPolicyGradientScope::LanguageHeadOnly
    );
    assert_eq!(policy.counterfactual_targets_per_state, 1);
    assert_eq!(policy.target_variants_per_state(), 2);
    assert_eq!(policy.base_semantic_rows_per_update(), 4);
    assert!(!config.model.tie_input_output_embeddings.unwrap_or(false));
}

#[test]
fn language_head_only_gradient_scope_rejects_tied_embeddings_and_energy_scoring() {
    use crate::config::{RuliadProofPolicyGradientScope, RuliadProofPolicyScoring};

    let mut tied = load_profile("ruliad-r3.semantic-action-language-head-only-fixed-ablation.toml");
    tied.model.tie_input_output_embeddings = Some(true);
    let error = tied
        .validate()
        .expect_err("language-head-only scope must not update tied input embeddings");
    assert!(
        error
            .to_string()
            .contains("language_head_only requires model.tie_input_output_embeddings=false"),
        "{error}"
    );

    let mut energy =
        load_profile("ruliad-r3.semantic-action-language-head-only-fixed-ablation.toml");
    energy.model.sequence_score_head = Some(burn_dragon_core::SequenceScoreHeadConfig {
        enabled: true,
        ..Default::default()
    });
    energy.training.ruliad_supervision.proof_policy.scoring =
        RuliadProofPolicyScoring::SemanticEnergy;
    energy
        .training
        .ruliad_supervision
        .proof_policy
        .gradient_scope = RuliadProofPolicyGradientScope::LanguageHeadOnly;
    let error = energy
        .validate()
        .expect_err("language-head-only scope must target completion likelihood");
    assert!(
        error
            .to_string()
            .contains("language_head_only requires scoring=completion_likelihood"),
        "{error}"
    );

    let mut factorized =
        load_profile("ruliad-r3.semantic-action-language-head-only-fixed-ablation.toml");
    factorized.model.language_head =
        Some(burn_dragon_core::LanguageHeadConfig::NcaFactorizedPatch {
            state_count: 2,
            patch_size: 2,
            frame_special_tokens: false,
            eos_id: None,
        });
    let error = factorized
        .validate()
        .expect_err("language-head-only scope requires a flat token projection");
    assert!(
        error.to_string().contains(
            "language_head_only requires model.language_head.type=standard_token_classification"
        ),
        "{error}"
    );

    let mut conditioned =
        load_profile("ruliad-r3.semantic-action-language-head-only-fixed-ablation.toml");
    let latent = burn_dragon_core::LatentReasoningConfig {
        step_conditioned_decoder: true,
        ..Default::default()
    };
    conditioned.model.latent_reasoning = Some(latent);
    let error = conditioned
        .validate()
        .expect_err("language-head-only scope must not update latent-step conditioning");
    assert!(
        error.to_string().contains(
            "language_head_only requires model.latent_reasoning.step_conditioned_decoder=false"
        ),
        "{error}"
    );
}

#[test]
fn paired_dagger_validation_preserves_causal_and_model_visited_rows() {
    use crate::config::{
        RuliadProofPolicyEffectiveMode, RuliadProofPolicyScoring, RuliadProofPolicyTrainingMode,
    };

    let mut config = load_profile("ruliad-r3.action-policy-semantic-energy-fixed-ablation.toml");
    let proof_policy = &mut config.training.ruliad_supervision.proof_policy;
    proof_policy.mode = RuliadProofPolicyTrainingMode::StaticThenPairedDagger;
    proof_policy.dagger_start_after_steps = 512;
    proof_policy.stratified_difficulty_levels = 1;
    proof_policy.rollout_steps = 2;
    config
        .validate()
        .expect("bounded semantic-energy paired DAgger should validate");
    let policy = config.training.ruliad_supervision.proof_policy;
    assert_eq!(policy.scoring, RuliadProofPolicyScoring::SemanticEnergy);
    assert_eq!(
        policy.mode,
        RuliadProofPolicyTrainingMode::StaticThenPairedDagger
    );
    assert_eq!(
        policy.effective_mode(511),
        RuliadProofPolicyEffectiveMode::StaticExpert
    );
    assert_eq!(
        policy.effective_mode(512),
        RuliadProofPolicyEffectiveMode::PairedDagger
    );
    assert_eq!(policy.stratified_difficulty_levels, 1);
    assert_eq!(policy.rollout_steps, 2);
    assert_eq!(policy.counterfactual_targets_per_state, 1);
    assert_eq!(policy.semantic_rows_per_update(), 8);
    assert_eq!(policy.base_semantic_rows_per_update(), 4);

    let mut invalid = config;
    invalid
        .training
        .ruliad_supervision
        .proof_policy
        .stratified_difficulty_levels = 2;
    let error = invalid
        .validate()
        .expect_err("paired DAgger must reserve one visited state per trajectory");
    assert!(
        error
            .to_string()
            .contains("exceeds the paired DAgger trajectory budget"),
        "{error}"
    );
}

#[test]
fn semantic_energy_rejects_an_empty_compatibility_projection() {
    let mut config = load_profile("ruliad-r3.action-policy-semantic-energy-fixed-ablation.toml");
    config
        .model
        .sequence_score_head
        .as_mut()
        .expect("semantic-energy head")
        .projection_dim = 0;
    let error = config
        .validate()
        .expect_err("zero-rank compatibility head must be rejected");
    assert!(
        error
            .to_string()
            .contains("sequence_score_head.projection_dim must be > 0"),
        "{error}"
    );
}

#[test]
fn ruliad_counterfactual_policy_requires_energy_candidates_and_complete_groups() {
    use crate::config::RuliadProofPolicyScoring;

    let mut config = load_profile("ruliad-r3.action-policy-semantic-energy-fixed-ablation.toml");
    config.training.ruliad_supervision.proof_policy.scoring =
        RuliadProofPolicyScoring::CompletionLikelihood;
    let error = config
        .validate()
        .expect_err("candidate-normalized full-model completion counterfactuals must be rejected");
    assert!(
        error
            .to_string()
            .contains("full-model prefix-conditional deployed-decoder completion")
    );

    let mut config = load_profile("ruliad-r3.action-policy-semantic-energy-fixed-ablation.toml");
    config
        .training
        .ruliad_supervision
        .proof_policy
        .counterfactual_targets_per_state = 4;
    let error = config
        .validate()
        .expect_err("counterfactual targets must leave an original candidate class");
    assert!(error.to_string().contains("must be less than candidates"));

    let mut config = load_profile("ruliad-r3.action-policy-semantic-energy-fixed-ablation.toml");
    config
        .training
        .ruliad_supervision
        .proof_policy
        .max_rows_per_update = 1;
    let error = config
        .validate()
        .expect_err("row budget must fit a complete target pair");
    assert!(
        error
            .to_string()
            .contains("target-variant presentation group")
    );
}

#[test]
fn ruliad_action_policy_profiles_load_with_explicit_search_contracts() {
    use crate::config::RuliadProofPolicyCandidateSymmetry::{BalancedRotation, CyclicOrbitAverage};
    for (profile, beam_width, dagger, symmetry) in [
        (
            "ruliad-r3.action-policy-fixed-ablation.toml",
            1,
            false,
            BalancedRotation,
        ),
        (
            "ruliad-r3.semantic-action-fixed-ablation.toml",
            1,
            false,
            BalancedRotation,
        ),
        (
            "ruliad-r3.semantic-action-static-fixed-ablation.toml",
            1,
            true,
            BalancedRotation,
        ),
        (
            "ruliad-r3.semantic-action-static-every-step-fixed-ablation.toml",
            1,
            true,
            BalancedRotation,
        ),
        (
            "ruliad-r3.semantic-action-static-every-two-steps-fixed-ablation.toml",
            1,
            true,
            BalancedRotation,
        ),
        (
            "ruliad-r3.semantic-action-static-prefix-fixed-ablation.toml",
            1,
            true,
            BalancedRotation,
        ),
        (
            "ruliad-r3.semantic-action-static-marginal-fixed-ablation.toml",
            1,
            true,
            BalancedRotation,
        ),
        (
            "ruliad-r3.action-policy-beam4-fixed-ablation.toml",
            4,
            false,
            BalancedRotation,
        ),
        (
            "ruliad-r3.action-policy-dagger-fixed-ablation.toml",
            1,
            true,
            BalancedRotation,
        ),
        (
            "ruliad-r3.action-policy-dagger-marginal-fixed-ablation.toml",
            1,
            true,
            BalancedRotation,
        ),
        (
            "ruliad-r3.action-policy-static-marginal-fixed-ablation.toml",
            1,
            true,
            BalancedRotation,
        ),
        (
            "ruliad-r3.action-policy-static-orbit-marginal-fixed-ablation.toml",
            1,
            true,
            CyclicOrbitAverage,
        ),
        (
            "ruliad-r3.action-policy-static-orbit-worst-marginal-fixed-ablation.toml",
            1,
            true,
            CyclicOrbitAverage,
        ),
        (
            "ruliad-r3.action-policy-bc-paired-dagger-marginal-fixed-ablation.toml",
            1,
            true,
            BalancedRotation,
        ),
        (
            "ruliad-r3.action-policy-bc-paired-dagger-orbit-marginal-fixed-ablation.toml",
            1,
            true,
            CyclicOrbitAverage,
        ),
        (
            "ruliad-r3.action-policy-dagger-beam4-fixed-ablation.toml",
            4,
            true,
            BalancedRotation,
        ),
        (
            "ruliad-r3.action-policy-promotion-audit.toml",
            1,
            false,
            BalancedRotation,
        ),
        (
            "ruliad-r3.action-policy-beam4-promotion-audit.toml",
            4,
            false,
            BalancedRotation,
        ),
    ] {
        let config = load_profile(profile);
        config
            .validate()
            .unwrap_or_else(|error| panic!("validate {profile}: {error}"));
        assert!(config.training.ruliad_probe_generation.enabled, "{profile}");
        assert_eq!(
            config.training.ruliad_probe_generation.max_batch_rows, 64,
            "{profile}"
        );
        assert_eq!(
            config.training.ruliad_probe_generation.minimum_batch_rows, 2,
            "{profile}"
        );
        assert_eq!(
            config
                .training
                .ruliad_probe_generation
                .maximum_prompt_position_span,
            32,
            "{profile}"
        );
        assert_eq!(
            config.training.ruliad_probe_generation.device_buffer_tokens, 4,
            "{profile}"
        );
        assert_eq!(config.training.ruliad_policy_probe.beam_width, beam_width);
        assert_eq!(config.training.ruliad_policy_probe.scoring_batch_rows, 32);
        assert_eq!(
            config.training.ruliad_policy_probe.scoring_token_budget,
            32_768
        );
        assert_eq!(
            config.training.ruliad_policy_probe.scoring_pipeline_depth,
            2
        );
        assert_eq!(
            config.training.ruliad_policy_probe.candidate_symmetry, symmetry,
            "{profile}"
        );
        assert_eq!(
            config.training.ruliad_supervision.proof_policy.enabled,
            dagger
        );
        if dagger {
            assert_eq!(
                config.training.ruliad_supervision.proof_policy.weight, 0.25,
                "{profile}"
            );
        }
        let promotion_audit = profile.contains("promotion-audit");
        assert_eq!(
            config.training.ruliad_policy_probe.every_epochs,
            if promotion_audit { 1 } else { 4 },
            "{profile}"
        );
        assert_eq!(
            config
                .training
                .ruliad_policy_probe
                .effective_closed_loop_every_epochs(),
            if promotion_audit { 1 } else { 4 },
            "ablation profiles must retain matched closed-loop cadence: {profile}"
        );
        assert_eq!(
            config.training.ruliad_policy_probe.items,
            if promotion_audit { 64 } else { 16 },
            "{profile}"
        );
        assert_eq!(
            config.training.ruliad_policy_probe.max_steps,
            if promotion_audit { 256 } else { 64 },
            "{profile}"
        );
    }

    let marginal = load_profile("ruliad-r3.action-policy-dagger-marginal-fixed-ablation.toml");
    assert_eq!(
        marginal
            .training
            .ruliad_supervision
            .proof_policy
            .normalization,
        crate::config::RuliadProofPolicyNormalization::VocabularyMarginal
    );
    let semantic_marginal =
        load_profile("ruliad-r3.semantic-action-static-marginal-fixed-ablation.toml");
    assert_eq!(
        semantic_marginal
            .training
            .ruliad_supervision
            .proof_policy
            .normalization,
        crate::config::RuliadProofPolicyNormalization::VocabularyMarginal
    );
    assert_eq!(
        semantic_marginal
            .training
            .ruliad_supervision
            .proof_policy
            .max_rows_per_update,
        8
    );
    assert_eq!(
        semantic_marginal
            .training
            .ruliad_supervision
            .proof_policy
            .max_completion_tokens,
        64
    );
    let semantic_prefix =
        load_profile("ruliad-r3.semantic-action-static-prefix-fixed-ablation.toml");
    assert_eq!(
        semantic_prefix
            .training
            .ruliad_supervision
            .proof_policy
            .normalization,
        crate::config::RuliadProofPolicyNormalization::PrefixConditional
    );
    let semantic_every_step =
        load_profile("ruliad-r3.semantic-action-static-every-step-fixed-ablation.toml");
    assert_eq!(
        semantic_every_step
            .training
            .ruliad_supervision
            .proof_policy
            .every_steps,
        1
    );
    assert_eq!(
        semantic_every_step
            .training
            .ruliad_supervision
            .proof_policy
            .start_after_steps,
        0
    );
    let semantic_every_two_steps =
        load_profile("ruliad-r3.semantic-action-static-every-two-steps-fixed-ablation.toml");
    assert_eq!(
        semantic_every_two_steps
            .training
            .ruliad_supervision
            .proof_policy
            .every_steps,
        2
    );
    assert_eq!(
        semantic_every_two_steps
            .training
            .ruliad_supervision
            .proof_policy
            .start_after_steps,
        0
    );
    assert_eq!(
        marginal
            .training
            .ruliad_supervision
            .proof_policy
            .candidate_symmetry,
        crate::config::RuliadProofPolicyCandidateSymmetry::BalancedRotation
    );
    let static_marginal =
        load_profile("ruliad-r3.action-policy-static-marginal-fixed-ablation.toml");
    assert_eq!(
        static_marginal
            .training
            .ruliad_supervision
            .proof_policy
            .mode,
        crate::config::RuliadProofPolicyTrainingMode::StaticExpert
    );
    assert_eq!(
        static_marginal
            .training
            .ruliad_supervision
            .proof_policy
            .every_steps,
        4
    );
    let orbit = load_profile("ruliad-r3.action-policy-static-orbit-marginal-fixed-ablation.toml");
    assert_eq!(
        orbit
            .training
            .ruliad_supervision
            .proof_policy
            .candidate_symmetry,
        CyclicOrbitAverage
    );
    assert_eq!(
        orbit
            .training
            .ruliad_supervision
            .proof_policy
            .max_presentation_rows_per_update,
        32
    );
    assert_eq!(
        orbit
            .training
            .ruliad_supervision
            .proof_policy
            .semantic_rows_per_update(),
        8
    );
    let worst_orbit =
        load_profile("ruliad-r3.action-policy-static-orbit-worst-marginal-fixed-ablation.toml");
    assert_eq!(
        worst_orbit
            .training
            .ruliad_supervision
            .proof_policy
            .presentation_risk,
        crate::config::RuliadProofPolicyPresentationRisk::Worst
    );
    let scheduled =
        load_profile("ruliad-r3.action-policy-bc-paired-dagger-marginal-fixed-ablation.toml");
    assert_eq!(
        scheduled.training.ruliad_supervision.proof_policy.mode,
        crate::config::RuliadProofPolicyTrainingMode::StaticThenPairedDagger
    );
    assert_eq!(
        scheduled
            .training
            .ruliad_supervision
            .proof_policy
            .dagger_start_after_steps,
        768
    );
    assert_eq!(
        scheduled
            .training
            .ruliad_supervision
            .proof_policy
            .effective_mode(767),
        crate::config::RuliadProofPolicyEffectiveMode::StaticExpert
    );
    assert_eq!(
        scheduled
            .training
            .ruliad_supervision
            .proof_policy
            .effective_mode(768),
        crate::config::RuliadProofPolicyEffectiveMode::PairedDagger
    );
}

#[test]
fn ruliad_action_policy_probe_rejects_zero_cadence() {
    let mut config = load_profile("ruliad-r3.action-policy-fixed-ablation.toml");
    config.training.ruliad_policy_probe.every_epochs = 0;

    let error = config.validate().expect_err("zero cadence must fail");
    assert!(
        error
            .to_string()
            .contains("ruliad_policy_probe.every_epochs must be > 0"),
        "{error}"
    );
}

#[test]
fn ruliad_action_policy_probe_rejects_zero_closed_loop_cadence() {
    let mut config = load_profile("ruliad-r3.action-policy-fixed-ablation.toml");
    config.training.ruliad_policy_probe.closed_loop_every_epochs = Some(0);

    let error = config
        .validate()
        .expect_err("zero closed-loop cadence must fail");
    assert!(
        error
            .to_string()
            .contains("ruliad_policy_probe.closed_loop_every_epochs must be > 0"),
        "{error}"
    );
}

#[test]
fn ruliad_probe_generation_rejects_unbounded_or_empty_batches() {
    let mut config = load_profile("ruliad-r3.action-policy-fixed-ablation.toml");
    config.training.ruliad_probe_generation.minimum_batch_rows = config
        .training
        .ruliad_probe_generation
        .max_batch_rows
        .saturating_add(1);
    let error = config
        .validate()
        .expect_err("minimum rows above maximum must fail");
    assert!(
        error
            .to_string()
            .contains("minimum_batch_rows must be in 1..=max_batch_rows"),
        "{error}"
    );

    config.training.ruliad_probe_generation.minimum_batch_rows = 2;
    config
        .training
        .ruliad_probe_generation
        .maximum_prompt_position_span = 0;
    let error = config.validate().expect_err("zero prompt span must fail");
    assert!(
        error
            .to_string()
            .contains("maximum_prompt_position_span must be > 0"),
        "{error}"
    );

    config
        .training
        .ruliad_probe_generation
        .maximum_prompt_position_span = 32;
    config.training.ruliad_probe_generation.device_buffer_tokens = 0;
    let error = config.validate().expect_err("zero device buffer must fail");
    assert!(
        error
            .to_string()
            .contains("device_buffer_tokens must be > 0"),
        "{error}"
    );

    config.training.ruliad_probe_generation.device_buffer_tokens = 4;
    config.training.ruliad_probe_generation.max_in_flight_rows = 0;
    let error = config
        .validate()
        .expect_err("zero in-flight row bound must fail");
    assert!(
        error.to_string().contains("max_in_flight_rows must be > 0"),
        "{error}"
    );
}

#[test]
fn ruliad_static_then_paired_dagger_requires_an_aligned_later_transition() {
    let mut config =
        load_profile("ruliad-r3.action-policy-bc-paired-dagger-marginal-fixed-ablation.toml");
    config
        .training
        .ruliad_supervision
        .proof_policy
        .dagger_start_after_steps = 128;
    let error = config
        .validate()
        .expect_err("DAgger transition must follow static warmup");
    assert!(error.to_string().contains("must exceed start_after_steps"));

    config
        .training
        .ruliad_supervision
        .proof_policy
        .dagger_start_after_steps = 769;
    let error = config
        .validate()
        .expect_err("DAgger transition must align to policy cadence");
    assert!(error.to_string().contains("must align with every_steps"));

    config
        .training
        .ruliad_supervision
        .proof_policy
        .dagger_start_after_steps = 768;
    config
        .training
        .ruliad_supervision
        .proof_policy
        .max_rows_per_update = 1;
    let error = config
        .validate()
        .expect_err("paired DAgger needs both row populations");
    assert!(error.to_string().contains("must be at least 2"));
}

#[test]
fn ruliad_orbit_policy_requires_a_complete_bounded_presentation_set() {
    let mut config =
        load_profile("ruliad-r3.action-policy-static-orbit-marginal-fixed-ablation.toml");
    let proof_policy = &mut config.training.ruliad_supervision.proof_policy;
    proof_policy.max_presentation_rows_per_update = proof_policy.candidates - 1;

    let error = config
        .validate()
        .expect_err("an incomplete orbit must not be materialized");
    assert!(error.to_string().contains("fit one complete"), "{error}");
}

#[test]
fn ruliad_worst_presentation_risk_requires_an_exact_orbit() {
    let mut config =
        load_profile("ruliad-r3.action-policy-static-orbit-worst-marginal-fixed-ablation.toml");
    assert!(config.validate().is_ok());
    config
        .training
        .ruliad_supervision
        .proof_policy
        .candidate_symmetry = crate::config::RuliadProofPolicyCandidateSymmetry::BalancedRotation;

    let error = config
        .validate()
        .expect_err("worst presentation risk needs a complete orbit");
    assert!(
        error
            .to_string()
            .contains("presentation_risk=worst requires"),
        "{error}"
    );
}

#[test]
fn ruliad_paired_orbit_policy_requires_both_semantic_populations() {
    let mut config =
        load_profile("ruliad-r3.action-policy-bc-paired-dagger-orbit-marginal-fixed-ablation.toml");
    let proof_policy = &mut config.training.ruliad_supervision.proof_policy;
    proof_policy.rollout_steps = 1;
    proof_policy.max_presentation_rows_per_update = proof_policy.candidates;

    let error = config
        .validate()
        .expect_err("paired DAgger needs two complete semantic orbits");
    assert!(
        error.to_string().contains("at least 2 base semantic rows"),
        "{error}"
    );
}

#[test]
fn random_scaffold_ruliad_matrix_profiles_load_and_validate() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../config/language/experiments/random_scaffold");
    for profile in [
        "ruliad-screen.dense.toml",
        "ruliad-screen.rank1.toml",
        "ruliad-screen.rank4.toml",
        "ruliad-screen.rank8.toml",
        "ruliad-screen.rank16.toml",
        "ruliad-screen.rs-rank8.toml",
        "ruliad-screen.rs-rank16.toml",
        "ruliad-screen.rs-rank32.toml",
        "ruliad-screen.rs-rank64.toml",
        "ruliad-screen.rank8-fixed-gain.toml",
        "ruliad-screen.rademacher-rank8.toml",
        "ruliad-parity.dense.toml",
        "ruliad-parity.rank8.toml",
        "ruliad-parity.rs-rank16.toml",
        "ruliad-parity.rs-rank32.toml",
    ] {
        let path = root.join(profile);
        let config = load_training_config(std::slice::from_ref(&path))
            .unwrap_or_else(|error| panic!("load {}: {error}", path.display()));
        config
            .validate()
            .unwrap_or_else(|error| panic!("validate {}: {error}", path.display()));
    }
}

#[test]
fn local_predictive_coding_profiles_load_with_canonical_factor_contract() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../config/language/experiments/predictive_coding");
    for profile in ["local-pc-smoke.toml", "local-pc-1m.toml"] {
        let path = root.join(profile);
        let config = load_training_config(std::slice::from_ref(&path))
            .unwrap_or_else(|error| panic!("load {}: {error}", path.display()));
        config
            .validate()
            .unwrap_or_else(|error| panic!("validate {}: {error}", path.display()));
        assert_eq!(
            config.resolved_training_algorithm(),
            TrainingAlgorithm::PredictiveCoding
        );
        assert_eq!(config.optimizer.name, OptimizerKind::Adamw);
        assert_eq!(
            config.training.local_predictive_coding.factor_reduction,
            crate::config::PredictiveCodingFactorReduction::Sum
        );
        assert_eq!(
            config.training.local_predictive_coding.solver,
            crate::config::LocalPredictiveCodingSolver::SynchronousEquilibrium
        );
        assert_eq!(
            config
                .training
                .local_predictive_coding
                .inference
                .gradient_norm_scope,
            burn_pc::PcGradientNormScope::PerRow
        );
        assert!(!config.training.local_predictive_coding.sync_diagnostics);
        assert_eq!(
            config.training.validation.sampling,
            crate::config::TrainingValidationSampling::FixedHoldout
        );
    }
}

#[test]
fn local_pc_closed_loop_profile_composes_an_accelerated_mastery_gated_corpus() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let profile = workspace.join(
        "config/language/experiments/predictive_coding/local-pc-verifier-1m-closed-loop.toml",
    );
    let config = load_training_config(std::slice::from_ref(&profile))
        .unwrap_or_else(|error| panic!("load {}: {error}", profile.display()));
    config
        .validate()
        .unwrap_or_else(|error| panic!("validate {}: {error}", profile.display()));
    assert_eq!(
        config
            .training
            .ruliad_policy_probe
            .effective_closed_loop_every_epochs(),
        1
    );
    assert_eq!(
        config
            .training
            .ruliad_policy_probe
            .checkpoint_capability_contract,
        crate::config::RuliadCheckpointCapabilityContract::ClosedLoopPolicy
    );
    assert!(config.training.ruliad_policy_probe.promotion_gate.enabled);
    let DatasetSourceConfig::UniversalityRuliad { config: corpus } = &config.dataset.source else {
        panic!("closed-loop profile must use a Ruliad corpus");
    };
    let corpus = burn_dragon_universality::load_ruliad_config(&workspace.join(corpus))
        .expect("load composed closed-loop corpus");
    assert_eq!(corpus.source_selection.cold_start.hold_steps, 64);
    assert_eq!(corpus.source_selection.cold_start.ramp_steps, 256);
    assert_eq!(
        corpus
            .source_selection
            .cold_start
            .mastery_min_feedback_count,
        4
    );
    assert!(corpus.source_selection.cold_start.release_requires_mastery);
    assert_eq!(
        corpus
            .source_selection
            .frontier_extension
            .max_materialized_levels,
        0,
        "closed-loop smoke must retain the unbounded production frontier"
    );
}

#[test]
fn closed_loop_checkpoint_contract_fails_closed_on_missing_gate_or_probe_cadence() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let profile = workspace.join(
        "config/language/experiments/predictive_coding/local-pc-verifier-1m-closed-loop.toml",
    );
    let config = load_training_config(std::slice::from_ref(&profile))
        .unwrap_or_else(|error| panic!("load {}: {error}", profile.display()));

    let mut missing_gate = config.clone();
    missing_gate
        .training
        .ruliad_policy_probe
        .promotion_gate
        .enabled = false;
    assert!(
        missing_gate
            .validate()
            .expect_err("policy promotion contract requires a gate")
            .to_string()
            .contains("promotion_gate.enabled=true")
    );

    let mut missing_probe = config.clone();
    missing_probe.training.ruliad_policy_probe.enabled = false;
    assert!(
        missing_probe
            .validate()
            .expect_err("policy promotion contract requires the policy probe")
            .to_string()
            .contains("ruliad_policy_probe.enabled=true")
    );

    let mut sparse_probe = config;
    sparse_probe
        .training
        .ruliad_policy_probe
        .closed_loop_every_epochs = Some(2);
    sparse_probe
        .training
        .events
        .ruliad_correctness_probe_every_epochs = 1;
    assert!(
        sparse_probe
            .validate()
            .expect_err("every promotion validation needs a policy result")
            .to_string()
            .contains("closed-loop cadence must divide")
    );

    let config = load_training_config(std::slice::from_ref(&profile))
        .unwrap_or_else(|error| panic!("load {}: {error}", profile.display()));
    for invalid_z in [0.0, -1.0, f64::INFINITY, f64::NAN] {
        let mut invalid_confidence = config.clone();
        invalid_confidence
            .training
            .ruliad_policy_probe
            .promotion_gate
            .regression_confidence_z = invalid_z;
        assert!(
            invalid_confidence
                .validate()
                .expect_err("invalid policy regression confidence must fail closed")
                .to_string()
                .contains("regression_confidence_z")
        );
    }
}

#[test]
fn local_fixed_prediction_overlay_selects_the_control_solver() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../config/language/experiments/predictive_coding");
    let paths = [
        root.join("local-pc-1m.toml"),
        root.join("pc-fixed-prediction.overlay.toml"),
    ];
    let config = load_training_config(&paths).expect("load fixed-prediction PC overlay");
    config
        .validate()
        .expect("validate fixed-prediction PC overlay");
    assert_eq!(
        config.training.local_predictive_coding.solver,
        crate::config::LocalPredictiveCodingSolver::FixedPrediction
    );
}

#[test]
fn local_incremental_research_overlay_selects_the_interleaved_contract() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../config/language/experiments/predictive_coding");
    let paths = [
        root.join("local-pc-1m.toml"),
        root.join("pc-incremental-rgs-research.overlay.toml"),
    ];
    let config = load_training_config(&paths).expect("load incremental PC research overlay");
    config
        .validate()
        .expect("validate incremental PC research overlay");
    assert_eq!(
        config.training.local_predictive_coding.solver,
        crate::config::LocalPredictiveCodingSolver::ReverseGaussSeidel
    );
    assert_eq!(
        config.training.local_predictive_coding.learning_schedule,
        burn_pc::PcLearningSchedule::Incremental
    );
    assert_eq!(config.training.local_predictive_coding.inference.steps, 1);
    assert_eq!(
        config.training.local_predictive_coding.inference.step_size,
        0.2
    );
}

#[test]
fn local_layer_prediction_overlay_selects_the_normalized_solver() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../config/language/experiments/predictive_coding");
    let paths = [
        root.join("local-pc-1m.toml"),
        root.join("pc-layer-local-prediction.overlay.toml"),
    ];
    let config = load_training_config(&paths).expect("load layer-local PC overlay");
    config.validate().expect("validate layer-local PC overlay");
    assert_eq!(
        config.training.local_predictive_coding.solver,
        crate::config::LocalPredictiveCodingSolver::LayerLocalPrediction
    );
    assert_eq!(
        config.training.local_predictive_coding.factor_reduction,
        crate::config::PredictiveCodingFactorReduction::Mean
    );
    assert!(!config.training.local_predictive_coding.sync_diagnostics);
}

#[test]
fn routed_layer_prediction_overlays_use_reserve_validated_discovery_defaults() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../config/language/experiments/predictive_coding");
    let paths = [
        root.join("local-pc-1m.toml"),
        root.join("pc-layer-local-prediction.overlay.toml"),
        root.join("pc-context-routing.overlay.toml"),
    ];
    let config = load_training_config(&paths).expect("load routed layer-local PC overlays");
    config
        .validate()
        .expect("validate routed layer-local PC overlays");
    let routing = &config.training.predictive_context_routing;
    assert!(routing.enabled);
    assert_eq!(routing.bank.calibration_update_rate, 0.5);
    assert_eq!(routing.bank.novelty_standard_deviations, 3.0);
    assert_eq!(
        routing.bank.capacity_policy,
        burn_pc::PredictiveContextCapacityPolicy::Reject
    );
}

#[test]
fn local_recurrent_predictive_coding_overlay_loads_and_validates() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../config/language/experiments/predictive_coding");
    let paths = [
        root.join("local-pc-smoke.toml"),
        root.join("pc-fixed-prediction.overlay.toml"),
        root.join("pc-recurrent-tbptt.overlay.toml"),
    ];
    let config = load_training_config(&paths).expect("load recurrent local-PC overlays");
    config
        .validate()
        .expect("validate recurrent local-PC overlays");
    assert_eq!(config.training.tbptt_chunk_size, Some(8));
    assert!(config.training.tbptt_persist_across_steps);
    assert_eq!(
        config.training.local_predictive_coding.solver,
        crate::config::LocalPredictiveCodingSolver::FixedPrediction
    );
}

#[test]
fn ruliad_64k_dynamics_anchor_rejects_unsafe_fixed_large_batch() {
    let mut config = load_profile("ruliad-1m-la-64k.training.toml");
    config.training.batch_size = 32;
    config.training.auto_batch_size.enabled = false;
    config.training.dynamics_anchor.enabled = true;

    let err = config
        .validate()
        .expect_err("64k anchored profile should reject fixed batch sizes above one");
    assert!(
        err.to_string()
            .contains("dynamics_anchor with fixed training.batch_size > 1"),
        "unexpected error: {err}"
    );
}

fn profile_path(file_name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../burn_dragon_p2p/deploy/profiles")
        .join(file_name)
}

#[test]
fn eggroll_optimizer_config_validates_for_single_next_token_training() {
    let mut config = parse_config("");
    config.optimizer.name = OptimizerKind::Eggroll;
    config.optimizer.eggroll.population.population_size = 2;
    config.optimizer.eggroll.population.population_chunk_size = 2;
    config
        .validate()
        .expect("minimal single-device EGGROLL config should validate");
}

fn estimate_profile_parameter_budget(config: &TrainingConfig) -> usize {
    let layers = config.model.n_layer.unwrap_or(8);
    let width = config.model.n_embd.unwrap_or(512);
    let latent = config.model.latent_total.unwrap_or(width * 2);
    let vocab = config.dataset.tokenizer.vocab_size();
    let embeddings = vocab.saturating_mul(width).saturating_mul(2);
    let per_layer = width
        .saturating_mul(width)
        .saturating_mul(4)
        .saturating_add(width.saturating_mul(latent).saturating_mul(4));
    embeddings.saturating_add(layers.saturating_mul(per_layer))
}

#[test]
fn eggroll_optimizer_rejects_gradient_accumulation() {
    let mut config = parse_config("");
    config.optimizer.name = OptimizerKind::Eggroll;
    config.training.gradient_accumulation_steps = 2;
    let err = config
        .validate()
        .expect_err("EGGROLL gradient accumulation should fail");
    assert!(
        err.to_string().contains("gradient_accumulation_steps = 1"),
        "unexpected error: {err}"
    );
}

#[test]
fn eggroll_optimizer_rejects_gradient_correction() {
    let mut config = parse_config("");
    config.optimizer.name = OptimizerKind::Eggroll;
    config.optimizer.eggroll.gradient_learning_rate = Some(1.0e-3);
    let err = config
        .validate()
        .expect_err("EGGROLL gradient correction should fail");
    assert!(
        err.to_string().contains("gradient_learning_rate"),
        "unexpected error: {err}"
    );
}

#[test]
fn eggroll_optimizer_rejects_continual_backprop() {
    let mut config = parse_config("");
    config.optimizer.name = OptimizerKind::Eggroll;
    config.training.continual_backprop.enabled = true;
    let err = config
        .validate()
        .expect_err("EGGROLL continual backprop should fail");
    assert!(
        err.to_string().contains("continual_backprop.enabled"),
        "unexpected error: {err}"
    );
}

#[test]
fn eggroll_optimizer_rejects_neuron_scaling() {
    let mut config = parse_config("");
    config.optimizer.name = OptimizerKind::Eggroll;
    config.training.neuron_scaling.enabled = true;
    let err = config
        .validate()
        .expect_err("EGGROLL neuron scaling should fail");
    assert!(
        err.to_string().contains("neuron_scaling.enabled"),
        "unexpected error: {err}"
    );
}

#[test]
fn ruliad_answer_completion_requires_ruliad_dataset() {
    let mut config = parse_config("");
    config.training.ruliad_supervision.mode = RuliadSupervisionMode::AnswerCompletion;
    let err = config
        .validate()
        .expect_err("answer-completion supervision should reject non-ruliad datasets");
    assert!(
        err.to_string().contains("universality_ruliad"),
        "unexpected error: {err}"
    );
}

#[test]
fn ruliad_answer_completion_validates_for_ruliad_adamw() {
    let mut config = parse_config("");
    config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
        config: "target/test-ruliad.toml".into(),
    };
    config.training.ruliad_supervision.mode = RuliadSupervisionMode::AnswerCompletion;
    config
        .validate()
        .expect("ruliad answer-completion AdamW config should validate");
}

#[test]
fn ruliad_answer_ranking_requires_answer_target_mask_mode() {
    let mut config = parse_config("");
    config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
        config: "target/test-ruliad.toml".into(),
    };
    config.training.ruliad_supervision.mode = RuliadSupervisionMode::FullDocument;
    config.training.ruliad_supervision.answer_ranking.enabled = true;
    let err = config
        .validate()
        .expect_err("answer ranking should require an answer target mask mode");
    assert!(
        err.to_string().contains("answer target masks"),
        "unexpected error: {err}"
    );
}

#[test]
fn ruliad_answer_ranking_validates_for_ruliad_answer_completion() {
    let mut config = parse_config("");
    config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
        config: "target/test-ruliad.toml".into(),
    };
    config.training.ruliad_supervision.mode = RuliadSupervisionMode::AnswerCompletion;
    config.training.ruliad_supervision.answer_ranking.enabled = true;
    config
        .validate()
        .expect("ruliad answer ranking should validate with answer-completion masks");
}

#[test]
fn ruliad_answer_ranking_rejects_invalid_parameters() {
    let mut config = parse_config("");
    config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
        config: "target/test-ruliad.toml".into(),
    };
    config.training.ruliad_supervision.mode = RuliadSupervisionMode::AnswerCompletion;
    config.training.ruliad_supervision.answer_ranking.enabled = true;
    config
        .training
        .ruliad_supervision
        .answer_ranking
        .corrupt_offset = 0;
    let err = config
        .validate()
        .expect_err("zero corrupt offset should fail");
    assert!(
        err.to_string().contains("corrupt_offset"),
        "unexpected error: {err}"
    );
}

#[test]
fn ruliad_answer_denoising_requires_answer_target_mask_mode() {
    let mut config = parse_config("");
    config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
        config: "target/test-ruliad.toml".into(),
    };
    config.training.ruliad_supervision.mode = RuliadSupervisionMode::FullDocument;
    config.training.ruliad_supervision.answer_denoising.enabled = true;
    let err = config
        .validate()
        .expect_err("answer denoising should require an answer target mask mode");
    assert!(
        err.to_string().contains("answer target masks"),
        "unexpected error: {err}"
    );
}

#[test]
fn ruliad_answer_denoising_validates_for_ruliad_answer_completion() {
    let mut config = parse_config("");
    config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
        config: "target/test-ruliad.toml".into(),
    };
    config.training.ruliad_supervision.mode = RuliadSupervisionMode::AnswerCompletion;
    config.training.ruliad_supervision.answer_denoising.enabled = true;
    config
        .validate()
        .expect("ruliad answer denoising should validate with answer-completion masks");
}

#[test]
fn ruliad_answer_denoising_rejects_invalid_parameters() {
    let mut config = parse_config("");
    config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
        config: "target/test-ruliad.toml".into(),
    };
    config.training.ruliad_supervision.mode = RuliadSupervisionMode::AnswerCompletion;
    config.training.ruliad_supervision.answer_denoising.enabled = true;
    config
        .training
        .ruliad_supervision
        .answer_denoising
        .probability = 1.5;
    let err = config
        .validate()
        .expect_err("invalid denoising probability should fail");
    assert!(
        err.to_string().contains("answer_denoising.probability"),
        "unexpected error: {err}"
    );
}

#[test]
fn ruliad_verifier_reward_requires_ruliad_dataset() {
    let mut config = parse_config("");
    config.training.ruliad_supervision.verifier_reward.enabled = true;
    let err = config
        .validate()
        .expect_err("verifier reward should require ruliad data");
    assert!(
        err.to_string().contains("universality_ruliad"),
        "unexpected error: {err}"
    );
}

#[test]
fn ruliad_verifier_reward_rejects_invalid_parameters() {
    let mut config = parse_config("");
    config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
        config: "target/test-ruliad.toml".into(),
    };
    config.training.ruliad_supervision.verifier_reward.enabled = true;
    config
        .training
        .ruliad_supervision
        .verifier_reward
        .group_size = 1;
    let err = config
        .validate()
        .expect_err("single-sample verifier reward group should fail");
    assert!(
        err.to_string().contains("verifier_reward.group_size"),
        "unexpected error: {err}"
    );
}

#[test]
fn ruliad_generated_attractor_replay_rejects_invalid_diversity_guard() {
    let mut config = parse_config("");
    config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
        config: "target/test-ruliad.toml".into(),
    };
    let verifier_reward = &mut config.training.ruliad_supervision.verifier_reward;
    verifier_reward.enabled = true;
    verifier_reward.generated_attractor_replay_capacity = 8;
    verifier_reward.field_binding_contrast_weight = 0.01;
    verifier_reward.field_binding_contrast_pair_weight = 0.5;
    verifier_reward.generated_attractor_replay_min_distinct_answers = 0;
    let err = config
        .validate()
        .expect_err("zero generated-attractor distinct-answer guard should fail");
    assert!(
        err.to_string()
            .contains("generated_attractor_replay_min_distinct_answers"),
        "unexpected error: {err}"
    );

    let mut config = parse_config("");
    config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
        config: "target/test-ruliad.toml".into(),
    };
    let verifier_reward = &mut config.training.ruliad_supervision.verifier_reward;
    verifier_reward.enabled = true;
    verifier_reward.generated_attractor_replay_capacity = 8;
    verifier_reward.field_binding_contrast_weight = 0.01;
    verifier_reward.field_binding_contrast_pair_weight = 0.5;
    verifier_reward.generated_attractor_replay_max_dominant_fraction = 1.25;
    let err = config
        .validate()
        .expect_err("dominant generated-attractor fraction above one should fail");
    assert!(
        err.to_string()
            .contains("generated_attractor_replay_max_dominant_fraction"),
        "unexpected error: {err}"
    );
}

#[test]
fn ruliad_verifier_reward_validates_for_local_ruliad_next_token() {
    let mut config = parse_config("");
    config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
        config: "target/test-ruliad.toml".into(),
    };
    config.training.ruliad_supervision.verifier_reward.enabled = true;
    config
        .validate()
        .expect("verifier reward should validate for local ruliad next-token training");
}

#[test]
fn ruliad_verifier_reward_vpo_rejects_zero_scalarizations() {
    let mut config = parse_config("");
    config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
        config: "target/test-ruliad.toml".into(),
    };
    config.training.ruliad_supervision.verifier_reward.enabled = true;
    config.training.ruliad_supervision.verifier_reward.mode =
        RuliadVerifierRewardMode::VpoIndependent;
    config
        .training
        .ruliad_supervision
        .verifier_reward
        .vpo_scalarizations = 0;
    let err = config
        .validate()
        .expect_err("zero VPO scalarization count should fail");
    assert!(
        err.to_string().contains("vpo_scalarizations"),
        "unexpected error: {err}"
    );
}

#[test]
fn ruliad_supervision_rejects_invalid_answer_value_weight() {
    let mut config = parse_config("");
    config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
        config: "target/test-ruliad.toml".into(),
    };
    config.training.ruliad_supervision.mode = RuliadSupervisionMode::AnswerCompletion;
    config.training.ruliad_supervision.answer_value_token_weight = 0;
    let err = config
        .validate()
        .expect_err("zero answer value token weight should fail");
    assert!(
        err.to_string().contains("answer_value_token_weight"),
        "unexpected error: {err}"
    );
}

#[test]
fn ruliad_supervision_rejects_invalid_answer_schema_weight() {
    let mut config = parse_config("");
    config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
        config: "target/test-ruliad.toml".into(),
    };
    config.training.ruliad_supervision.mode = RuliadSupervisionMode::AnswerCompletion;
    config
        .training
        .ruliad_supervision
        .answer_schema_token_weight = 0;
    let err = config
        .validate()
        .expect_err("zero answer schema token weight should fail");
    assert!(
        err.to_string().contains("answer_schema_token_weight"),
        "unexpected error: {err}"
    );
}

#[test]
fn ruliad_supervision_rejects_invalid_answer_schema_start_weight() {
    let mut config = parse_config("");
    config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
        config: "target/test-ruliad.toml".into(),
    };
    config.training.ruliad_supervision.mode = RuliadSupervisionMode::AnswerCompletion;
    config
        .training
        .ruliad_supervision
        .answer_schema_start_token_weight = 0;
    let err = config
        .validate()
        .expect_err("zero answer schema start token weight should fail");
    assert!(
        err.to_string().contains("answer_schema_start_token_weight"),
        "unexpected error: {err}"
    );
}

#[test]
fn ruliad_answer_contract_rejects_invalid_prompt_schema_value_weight() {
    let mut config = parse_config("");
    config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
        config: "target/test-ruliad.toml".into(),
    };
    config.training.ruliad_supervision.mode = RuliadSupervisionMode::AnswerCompletion;
    config.training.ruliad_supervision.answer_contract.enabled = true;
    config.training.ruliad_supervision.answer_contract.weight = 0.25;
    config
        .training
        .ruliad_supervision
        .answer_contract
        .prompt_schema_value_weight = -1.0;
    let err = config
        .validate()
        .expect_err("negative prompt-schema value weight should fail");
    assert!(
        err.to_string().contains("prompt_schema_value_weight"),
        "unexpected error: {err}"
    );
}

#[test]
fn ruliad_supervision_rejects_invalid_answer_close_marker_weight() {
    let mut config = parse_config("");
    config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
        config: "target/test-ruliad.toml".into(),
    };
    config.training.ruliad_supervision.mode = RuliadSupervisionMode::AnswerCompletion;
    config
        .training
        .ruliad_supervision
        .answer_close_marker_weight = 0;
    let err = config
        .validate()
        .expect_err("zero answer close marker weight should fail");
    assert!(
        err.to_string().contains("answer_close_marker_weight"),
        "unexpected error: {err}"
    );
}

#[test]
fn ruliad_verifier_reward_vpo_rejects_invalid_mass_floors() {
    let mut config = parse_config("");
    config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
        config: "target/test-ruliad.toml".into(),
    };
    config.training.ruliad_supervision.verifier_reward.enabled = true;
    config.training.ruliad_supervision.verifier_reward.mode =
        RuliadVerifierRewardMode::VpoIndependent;
    config
        .training
        .ruliad_supervision
        .verifier_reward
        .vpo_correctness_mass_floor = 0.8;
    config
        .training
        .ruliad_supervision
        .verifier_reward
        .vpo_completion_health_mass_floor = 0.3;
    let err = config
        .validate()
        .expect_err("VPO mass floors above one should fail");
    assert!(
        err.to_string().contains("mass floors"),
        "unexpected error: {err}"
    );
}

#[test]
fn ruliad_verifier_reward_rejects_invalid_advantage_clip_gate() {
    let mut config = parse_config("");
    config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
        config: "target/test-ruliad.toml".into(),
    };
    config.training.ruliad_supervision.verifier_reward.enabled = true;
    config
        .training
        .ruliad_supervision
        .verifier_reward
        .max_advantage_clip_fraction = Some(1.25);
    let err = config
        .validate()
        .expect_err("invalid advantage clip fraction should fail");
    assert!(
        err.to_string().contains("max_advantage_clip_fraction"),
        "unexpected error: {err}"
    );
}

#[test]
fn ruliad_verifier_reward_rejects_streaming_tbptt() {
    let mut config = parse_config("");
    config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
        config: "target/test-ruliad.toml".into(),
    };
    config.training.ruliad_supervision.verifier_reward.enabled = true;
    config.training.tbptt_persist_across_steps = true;
    let err = config
        .validate()
        .expect_err("verifier reward should reject persistent TBPTT");
    assert!(
        err.to_string().contains("tbptt_persist_across_steps"),
        "unexpected error: {err}"
    );
}

#[test]
fn ruliad_verifier_reward_rejects_tbptt_chunking() {
    let mut config = parse_config("");
    config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
        config: "target/test-ruliad.toml".into(),
    };
    config.training.ruliad_supervision.verifier_reward.enabled = true;
    config.training.tbptt_chunk_size = Some(4);
    let err = config
        .validate()
        .expect_err("verifier reward should reject TBPTT chunking");
    assert!(
        err.to_string().contains("tbptt_chunk_size"),
        "unexpected error: {err}"
    );
}

#[test]
fn ruliad_structured_contrast_rejects_tbptt_chunking() {
    let mut config = parse_config("");
    config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
        config: "target/test-ruliad.toml".into(),
    };
    config.training.ruliad_supervision.verifier_reward.enabled = true;
    config.training.ruliad_supervision.verifier_reward.weight = 0.0;
    config
        .training
        .ruliad_supervision
        .verifier_reward
        .structured_contrast_weight = 0.01;
    config
        .training
        .ruliad_supervision
        .verifier_reward
        .structured_negative_count = 1;
    config.training.tbptt_chunk_size = Some(4);
    let err = config
        .validate()
        .expect_err("structured contrast should reject TBPTT chunking");
    assert!(
        err.to_string().contains("structured_contrast_weight > 0")
            && err.to_string().contains("tbptt_chunk_size"),
        "unexpected error: {err}"
    );
}

#[test]
fn ruliad_rollout_imitation_rejects_tbptt_chunking() {
    let mut config = parse_config("");
    config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
        config: "target/test-ruliad.toml".into(),
    };
    config.training.ruliad_supervision.verifier_reward.enabled = true;
    config.training.ruliad_supervision.verifier_reward.weight = 0.0;
    config
        .training
        .ruliad_supervision
        .verifier_reward
        .rollout_imitation_weight = 0.01;
    config.training.tbptt_chunk_size = Some(4);
    let err = config
        .validate()
        .expect_err("rollout imitation should reject TBPTT chunking");
    assert!(
        err.to_string().contains("rollout_imitation_weight > 0")
            && err.to_string().contains("tbptt_chunk_size"),
        "unexpected error: {err}"
    );
}

#[test]
fn ruliad_field_binding_contrast_accepts_tbptt_chunking() {
    let mut config = parse_config("");
    config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
        config: "target/test-ruliad.toml".into(),
    };
    config.training.ruliad_supervision.verifier_reward.enabled = true;
    config.training.ruliad_supervision.verifier_reward.weight = 0.0;
    config
        .training
        .ruliad_supervision
        .verifier_reward
        .field_binding_contrast_weight = 0.01;
    config.training.tbptt_chunk_size = Some(4);
    config.training.tbptt_persist_across_steps = true;
    config.validate().expect(
        "field-binding contrast runs as an auxiliary policy-batch forward and should allow TBPTT",
    );
}

#[test]
fn ruliad_structured_recovery_accepts_tbptt_chunking() {
    let mut config = parse_config("");
    config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
        config: "target/test-ruliad.toml".into(),
    };
    config.training.ruliad_supervision.mode = RuliadSupervisionMode::AnswerCompletion;
    config.training.ruliad_supervision.answer_denoising.enabled = true;
    config
        .training
        .ruliad_supervision
        .answer_denoising
        .structured_recovery_weight = 0.01;
    config
        .training
        .ruliad_supervision
        .answer_denoising
        .structured_recovery_negative_count = 1;
    config.training.tbptt_chunk_size = Some(4);
    config.training.tbptt_persist_across_steps = true;
    config
        .validate()
        .expect("structured recovery should run as an auxiliary forward with TBPTT");
}

#[test]
fn ruliad_structured_recovery_accepts_schema_negatives_without_field_mutations() {
    let mut config = parse_config("");
    config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
        config: "target/test-ruliad.toml".into(),
    };
    config.training.ruliad_supervision.mode = RuliadSupervisionMode::AnswerCompletion;
    config.training.ruliad_supervision.answer_denoising.enabled = true;
    config
        .training
        .ruliad_supervision
        .answer_denoising
        .structured_recovery_weight = 0.01;
    config
        .training
        .ruliad_supervision
        .answer_denoising
        .structured_recovery_schema_negative_count = 1;
    config
        .validate()
        .expect("schema-only structured recovery should validate");
}

#[test]
fn ruliad_structured_contrast_accepts_schema_negatives_without_field_mutations() {
    let mut config = parse_config("");
    config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
        config: "target/test-ruliad.toml".into(),
    };
    config.training.ruliad_supervision.mode = RuliadSupervisionMode::AnswerCompletion;
    config.training.ruliad_supervision.verifier_reward.enabled = true;
    config.training.ruliad_supervision.verifier_reward.weight = 0.0;
    config
        .training
        .ruliad_supervision
        .verifier_reward
        .structured_contrast_weight = 0.01;
    config
        .training
        .ruliad_supervision
        .verifier_reward
        .structured_negative_count = 0;
    config
        .training
        .ruliad_supervision
        .verifier_reward
        .structured_template_negative_count = 0;
    config
        .training
        .ruliad_supervision
        .verifier_reward
        .structured_schema_negative_count = 1;
    config
        .validate()
        .expect("schema-only structured contrast should validate");
}

#[test]
fn source_selection_state_path_requires_ruliad_dataset() {
    let mut config = parse_config("");
    config.training.source_selection_state_path = Some("target/source-state.json".into());
    let err = config
        .validate()
        .expect_err("source-selection handoff should reject non-ruliad datasets");
    assert!(
        err.to_string().contains("universality_ruliad"),
        "unexpected error: {err}"
    );
}

#[test]
fn source_selection_state_path_validates_for_ruliad_dataset() {
    let mut config = parse_config("");
    config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
        config: "target/test-ruliad.toml".into(),
    };
    config.training.source_selection_state_path = Some("target/source-state.json".into());
    config
        .validate()
        .expect("ruliad source-selection state should validate");
}

#[test]
fn ruliad_source_feedback_override_requires_ruliad_dataset() {
    let mut config = parse_config("");
    config
        .dataset
        .ruliad_source_selection_feedback_updates_enabled = Some(false);
    let err = config
        .validate()
        .expect_err("source-feedback override should reject non-ruliad datasets");
    assert!(
        err.to_string().contains("universality_ruliad"),
        "unexpected error: {err}"
    );
}

#[test]
fn ruliad_source_feedback_override_validates_for_ruliad_dataset() {
    let mut config = parse_config("");
    config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
        config: "target/test-ruliad.toml".into(),
    };
    config
        .dataset
        .ruliad_source_selection_feedback_updates_enabled = Some(false);
    config
        .validate()
        .expect("ruliad source-feedback override should validate");
}

#[test]
fn ruliad_answer_completion_rejects_pure_eggroll_dense_ce_path() {
    let mut config = parse_config("");
    config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
        config: "target/test-ruliad.toml".into(),
    };
    config.training.ruliad_supervision.mode = RuliadSupervisionMode::AnswerCompletion;
    config.optimizer.name = OptimizerKind::Eggroll;
    config.optimizer.eggroll.population.population_size = 2;
    config.optimizer.eggroll.population.population_chunk_size = 2;
    let err = config
        .validate()
        .expect_err("answer-completion supervision should reject current pure EGGROLL path");
    assert!(
        err.to_string().contains("ruliad_supervision"),
        "unexpected error: {err}"
    );
}

#[test]
fn auto_batch_size_config_validates() {
    parse_config(
        r#"
[training.auto_batch_size]
enabled = true
min_batch_size = 1
max_batch_size = 32
target_device_memory_mb = 90000
probe_steps = 1
recompute_on_neuron_scale = true
"#,
    )
    .validate()
    .expect("auto batch config should validate");
}

#[test]
fn auto_batch_size_rejects_inverted_bounds() {
    let config = parse_config(
        r#"
[training.auto_batch_size]
enabled = true
min_batch_size = 8
max_batch_size = 4
"#,
    );
    let err = config
        .validate()
        .expect_err("inverted auto batch bounds should fail");
    assert!(
        err.to_string()
            .contains("auto_batch_size.max_batch_size must be >= min_batch_size"),
        "unexpected error: {err}"
    );
}

#[test]
fn auto_batch_size_rejects_probe_cap_below_min_batch() {
    let config = parse_config(
        r#"
[training.auto_batch_size]
enabled = true
min_batch_size = 8
max_probe_batch_size = 4
"#,
    );
    let err = config
        .validate()
        .expect_err("probe cap below min batch should fail");
    assert!(
        err.to_string()
            .contains("auto_batch_size.max_probe_batch_size must be >= min_batch_size"),
        "unexpected error: {err}"
    );
}

#[test]
fn auto_batch_size_rejects_host_fraction_above_ninety_percent() {
    let config = parse_config(
        r#"
[training.auto_batch_size]
enabled = true
max_system_memory_fraction = 0.95
"#,
    );
    let err = config
        .validate()
        .expect_err("host memory fraction above 90% should fail");
    assert!(
        err.to_string()
            .contains("auto_batch_size.max_system_memory_fraction"),
        "unexpected error: {err}"
    );
}

#[test]
fn greedy_rollout_unlikelihood_rejects_zero_history() {
    let config = parse_config(
        r#"
[training.greedy_rollout_unlikelihood]
enabled = true
weight = 0.5
history_tokens = 0
"#,
    );
    let err = config
        .validate()
        .expect_err("zero rollout history should fail");
    assert!(
        err.to_string()
            .contains("greedy_rollout_unlikelihood.history_tokens must be > 0"),
        "unexpected error: {err}"
    );
}

#[test]
fn greedy_rollout_unlikelihood_rejects_negative_recovery_weight() {
    let config = parse_config(
        r#"
[training.greedy_rollout_unlikelihood]
enabled = true
weight = 0.5
recovery_weight = -1.0
"#,
    );
    let err = config
        .validate()
        .expect_err("negative rollout recovery weight should fail");
    assert!(
        err.to_string()
            .contains("greedy_rollout_unlikelihood.recovery_weight must be finite and >= 0"),
        "unexpected error: {err}"
    );
}

#[test]
fn greedy_rollout_unlikelihood_rejects_negative_sequence_recovery_weight() {
    let config = parse_config(
        r#"
[training.greedy_rollout_unlikelihood]
enabled = true
weight = 0.5
sequence_recovery_weight = -1.0
"#,
    );
    let err = config
        .validate()
        .expect_err("negative rollout sequence recovery weight should fail");
    assert!(
        err.to_string().contains(
            "greedy_rollout_unlikelihood.sequence_recovery_weight must be finite and >= 0"
        ),
        "unexpected error: {err}"
    );
}

#[test]
fn greedy_rollout_unlikelihood_rejects_invalid_cycle_lag_range() {
    let config = parse_config(
        r#"
[training.greedy_rollout_unlikelihood]
enabled = true
cycle_weight = 0.5
cycle_min_lag = 32
cycle_max_lag = 16
"#,
    );
    let err = config
        .validate()
        .expect_err("invalid rollout cycle lag range should fail");
    assert!(
        err.to_string()
            .contains("greedy_rollout_unlikelihood.cycle_max_lag"),
        "unexpected error: {err}"
    );
}

#[test]
fn greedy_rollout_unlikelihood_rejects_invalid_margin() {
    let config = parse_config(
        r#"
[training.greedy_rollout_unlikelihood]
enabled = true
weight = 0.5
margin_weight = -1.0
"#,
    );
    let err = config
        .validate()
        .expect_err("negative rollout margin weight should fail");
    assert!(
        err.to_string()
            .contains("greedy_rollout_unlikelihood.margin_weight must be finite and >= 0"),
        "unexpected error: {err}"
    );

    let config = parse_config(
        r#"
[training.greedy_rollout_unlikelihood]
enabled = true
weight = 0.5
margin = -0.25
"#,
    );
    let err = config
        .validate()
        .expect_err("negative rollout margin should fail");
    assert!(
        err.to_string()
            .contains("greedy_rollout_unlikelihood.margin must be finite and >= 0"),
        "unexpected error: {err}"
    );
}

#[test]
fn degeneracy_gates_reject_invalid_period_thresholds() {
    let config = parse_config(
        r#"
[training.gates]
degeneracy_distinct_2_min_fraction = 1.1
"#,
    );
    let err = config
        .validate()
        .expect_err("invalid distinct-2 threshold should fail");
    assert!(
        err.to_string()
            .contains("degeneracy_distinct_2_min_fraction"),
        "unexpected error: {err}"
    );

    let config = parse_config(
        r#"
[training.gates]
degeneracy_period_2_max_fraction = -0.1
"#,
    );
    let err = config
        .validate()
        .expect_err("invalid period threshold should fail");
    assert!(
        err.to_string().contains("degeneracy_period_2_max_fraction"),
        "unexpected error: {err}"
    );

    let config = parse_config(
        r#"
[training.gates]
degeneracy_period_2_to_16_max_fraction = 1.1
"#,
    );
    let err = config
        .validate()
        .expect_err("invalid long-cycle period threshold should fail");
    assert!(
        err.to_string()
            .contains("degeneracy_period_2_to_16_max_fraction"),
        "unexpected error: {err}"
    );

    let config = parse_config(
        r#"
[training.gates]
degeneracy_period_2_to_64_max_fraction = 1.1
"#,
    );
    let err = config
        .validate()
        .expect_err("invalid extended long-cycle period threshold should fail");
    assert!(
        err.to_string()
            .contains("degeneracy_period_2_to_64_max_fraction"),
        "unexpected error: {err}"
    );
}

#[test]
fn capability_gates_reject_invalid_thresholds() {
    let config = parse_config(
        r#"
[training.gates]
capability_zero_verifier_patience_epochs = 0
"#,
    );
    let err = config
        .validate()
        .expect_err("zero capability patience should fail");
    assert!(
        err.to_string()
            .contains("capability_zero_verifier_patience_epochs"),
        "unexpected error: {err}"
    );

    let config = parse_config(
        r#"
[training.gates]
capability_regression_patience_epochs = 0
"#,
    );
    let err = config
        .validate()
        .expect_err("zero capability regression patience should fail");
    assert!(
        err.to_string()
            .contains("capability_regression_patience_epochs"),
        "unexpected error: {err}"
    );

    let config = parse_config(
        r#"
[training.gates]
capability_schema_wrong_max_rate = 1.1
"#,
    );
    let err = config
        .validate()
        .expect_err("invalid capability schema threshold should fail");
    assert!(
        err.to_string().contains("capability_schema_wrong_max_rate"),
        "unexpected error: {err}"
    );

    let config = parse_config(
        r#"
[training.gates]
capability_malformed_max_rate = -0.1
"#,
    );
    let err = config
        .validate()
        .expect_err("invalid capability malformed threshold should fail");
    assert!(
        err.to_string().contains("capability_malformed_max_rate"),
        "unexpected error: {err}"
    );

    let config = parse_config(
        r#"
[training.gates]
capability_missing_max_rate = 1.1
"#,
    );
    let err = config
        .validate()
        .expect_err("invalid capability missing threshold should fail");
    assert!(
        err.to_string().contains("capability_missing_max_rate"),
        "unexpected error: {err}"
    );

    let config = parse_config(
        r#"
[training.gates]
capability_completion_health_min_rate = 1.1
"#,
    );
    let err = config
        .validate()
        .expect_err("invalid capability completion-health threshold should fail");
    assert!(
        err.to_string()
            .contains("capability_completion_health_min_rate"),
        "unexpected error: {err}"
    );

    let config = parse_config(
        r#"
[training.gates]
capability_distinct_2_min_fraction = -0.1
"#,
    );
    let err = config
        .validate()
        .expect_err("invalid capability distinct-2 threshold should fail");
    assert!(
        err.to_string()
            .contains("capability_distinct_2_min_fraction"),
        "unexpected error: {err}"
    );

    let config = parse_config(
        r#"
[training.gates]
capability_answer_distinct_min_fraction = -0.1
"#,
    );
    let err = config
        .validate()
        .expect_err("invalid capability answer distinct threshold should fail");
    assert!(
        err.to_string()
            .contains("capability_answer_distinct_min_fraction"),
        "unexpected error: {err}"
    );

    let config = parse_config(
        r#"
[training.gates]
capability_field_value_distinct_ratio_min = 1.1
"#,
    );
    let err = config
        .validate()
        .expect_err("invalid capability field-value distinct threshold should fail");
    assert!(
        err.to_string()
            .contains("capability_field_value_distinct_ratio_min"),
        "unexpected error: {err}"
    );

    let config = parse_config(
        r#"
[training.gates]
capability_field_value_dominance_max = -0.1
"#,
    );
    let err = config
        .validate()
        .expect_err("invalid capability field-value dominance threshold should fail");
    assert!(
        err.to_string()
            .contains("capability_field_value_dominance_max"),
        "unexpected error: {err}"
    );
}

#[test]
fn repeat_unlikelihood_rejects_zero_history_lag() {
    let config = parse_config(
        r#"
[training.repeat_unlikelihood]
enabled = true
weight = 0.1
history_lags = [1, 0, 8]
"#,
    );
    let err = config.validate().expect_err("zero history lag should fail");
    assert!(
        err.to_string().contains("repeat_unlikelihood.history_lags"),
        "unexpected error: {err}"
    );
}

#[test]
fn repeat_unlikelihood_rejects_invalid_cycle_lag_range() {
    let config = parse_config(
        r#"
[training.repeat_unlikelihood]
enabled = true
cycle_weight = 0.5
cycle_min_lag = 32
cycle_max_lag = 16
"#,
    );
    let err = config
        .validate()
        .expect_err("invalid repeat cycle lag range should fail");
    assert!(
        err.to_string()
            .contains("repeat_unlikelihood.cycle_max_lag"),
        "unexpected error: {err}"
    );
}

#[test]
fn repeat_unlikelihood_rejects_zero_cycle_lags_per_step_when_enabled() {
    let config = parse_config(
        r#"
[training.repeat_unlikelihood]
enabled = true
cycle_weight = 0.5
cycle_lags_per_step = 0
"#,
    );
    let err = config
        .validate()
        .expect_err("zero repeat cycle lags per step should fail when cycle loss is enabled");
    assert!(
        err.to_string()
            .contains("repeat_unlikelihood.cycle_lags_per_step"),
        "unexpected error: {err}"
    );
}

#[test]
fn repeat_unlikelihood_rejects_zero_every_steps() {
    let config = parse_config(
        r#"
[training.repeat_unlikelihood]
enabled = true
weight = 0.5
every_steps = 0
"#,
    );
    let err = config
        .validate()
        .expect_err("zero repeat cadence should fail");
    assert!(
        err.to_string().contains("repeat_unlikelihood.every_steps"),
        "unexpected error: {err}"
    );
}

#[test]
fn logit_entropy_floor_rejects_negative_target() {
    let config = parse_config(
        r#"
[training.logit_entropy_floor]
enabled = true
weight = 0.1
target_entropy_bits = -1.0
"#,
    );
    let err = config
        .validate()
        .expect_err("negative entropy floor target should fail");
    assert!(
        err.to_string()
            .contains("logit_entropy_floor.target_entropy_bits"),
        "unexpected error: {err}"
    );
}

#[test]
fn logit_entropy_floor_rejects_invalid_marginal_fields() {
    let config = parse_config(
        r#"
[training.logit_entropy_floor]
enabled = true
marginal_weight = -0.1
"#,
    );
    let err = config
        .validate()
        .expect_err("negative marginal weight should fail");
    assert!(
        err.to_string()
            .contains("logit_entropy_floor.marginal_weight"),
        "unexpected error: {err}"
    );

    let config = parse_config(
        r#"
[training.logit_entropy_floor]
enabled = true
target_marginal_entropy_bits = -1.0
"#,
    );
    let err = config
        .validate()
        .expect_err("negative marginal entropy target should fail");
    assert!(
        err.to_string()
            .contains("logit_entropy_floor.target_marginal_entropy_bits"),
        "unexpected error: {err}"
    );
}

#[test]
fn logit_entropy_floor_rejects_zero_every_steps() {
    let config = parse_config(
        r#"
[training.logit_entropy_floor]
enabled = true
weight = 0.1
target_entropy_bits = 2.0
every_steps = 0
"#,
    );
    let err = config
        .validate()
        .expect_err("zero entropy cadence should fail");
    assert!(
        err.to_string().contains("logit_entropy_floor.every_steps"),
        "unexpected error: {err}"
    );
}

#[test]
fn logit_entropy_floor_rejects_invalid_target_coverage_fields() {
    let config = parse_config(
        r#"
[training.logit_entropy_floor]
enabled = true
target_coverage_weight = -0.1
"#,
    );
    let err = config
        .validate()
        .expect_err("negative target coverage weight should fail");
    assert!(
        err.to_string()
            .contains("logit_entropy_floor.target_coverage_weight"),
        "unexpected error: {err}"
    );

    let config = parse_config(
        r#"
[training.logit_entropy_floor]
enabled = true
target_coverage_epsilon = 1.0
"#,
    );
    let err = config
        .validate()
        .expect_err("invalid target coverage epsilon should fail");
    assert!(
        err.to_string()
            .contains("logit_entropy_floor.target_coverage_epsilon"),
        "unexpected error: {err}"
    );
}

#[test]
fn tied_input_output_embeddings_rejects_factorized_head() {
    let config = parse_config(
        r#"
[model]
tie_input_output_embeddings = true

[model.language_head]
type = "nca_factorized_patch"
state_count = 2
patch_size = 2
"#,
    );
    let err = config
        .validate()
        .expect_err("tied embeddings require flat token head");
    assert!(
        err.to_string()
            .contains("model.tie_input_output_embeddings requires"),
        "unexpected error: {err}"
    );
}

#[test]
fn neuron_scaling_config_validates_across_memory_kernels() {
    let cases = [
        r#"
[training.neuron_scaling]
enabled = true
max_latent_total = 64

[model]
n_layer = 1
n_embd = 16
n_head = 2
latent_total = 32
"#,
        r#"
[training.neuron_scaling]
enabled = true
max_latent_total = 64

[model]
n_layer = 1
n_embd = 16
n_head = 2
latent_total = 32
sequence_kernel = { memory_system = "linear_attention", executor = "dense_score_short_context" }
"#,
        r#"
[training.neuron_scaling]
enabled = true
max_latent_total = 64

[model]
n_layer = 1
n_embd = 16
n_head = 2
latent_total = 32
sequence_kernel = "mamba3_state_space_duality"

[model.mamba]
headdim = 8
chunk_size = 4
"#,
        r#"
[training.neuron_scaling]
enabled = true
max_latent_total = 64

[model]
n_layer = 1
n_embd = 16
n_head = 2
latent_total = 32
sequence_kernel = "gated_deltanet2"
"#,
        r#"
[training.neuron_scaling]
enabled = true
max_latent_total = 64

[model]
n_layer = 1
n_embd = 16
n_head = 2
latent_total = 32
sequence_kernel = { memory_system = "gated_deltanet2", executor = "gated_delta_chunk_wy" }

[model.gated_deltanet2]
implementation = "upstream_full"
chunk_size = 4
"#,
    ];

    for case in cases {
        parse_config(case)
            .validate()
            .unwrap_or_else(|err| panic!("neuron scaling config should validate: {err}"));
    }
}

#[test]
fn neuron_scaling_rejects_max_below_current_latent_total() {
    let config = parse_config(
        r#"
[training.neuron_scaling]
enabled = true
max_latent_total = 16

[model]
n_layer = 1
n_embd = 16
n_head = 2
latent_total = 32
"#,
    );

    let err = config
        .validate()
        .expect_err("max below current should fail");
    assert!(
        err.to_string()
            .contains("max_latent_total must be >= resolved model.latent_total"),
        "unexpected error: {err}"
    );
}

#[test]
fn neuron_scaling_rejects_max_not_divisible_by_head_count() {
    let config = parse_config(
        r#"
[training.neuron_scaling]
enabled = true
max_latent_total = 64

[model]
n_layer = 1
n_embd = 16
n_head = 3
latent_total = 48
"#,
    );

    let err = config
        .validate()
        .expect_err("head-incompatible max should fail");
    assert!(
        err.to_string()
            .contains("max_latent_total must be divisible by model.n_head"),
        "unexpected error: {err}"
    );
}

#[test]
fn neuron_scaling_rejects_non_single_parallel_mode() {
    let config = parse_config(
        r#"
[training.neuron_scaling]
enabled = true
max_latent_total = 80

[model]
n_layer = 1
n_embd = 10
n_head = 2
latent_total = 40

[parallel]
mode = "tensor_parallel_neuron"
world_size = 4

[parallel.data]
size = 1

[parallel.tensor]
size = 4
"#,
    );

    let err = config
        .validate()
        .expect_err("tensor-parallel neuron scaling should fail");
    assert!(
        err.to_string()
            .contains("neuron_scaling.enabled currently requires parallel.mode=single"),
        "unexpected error: {err}"
    );
}

#[test]
fn sdft_objective_config_validates() {
    let config = parse_config(
        r#"
[training.objective]
type = "sdft"
max_completion_tokens = 4
teacher_update_rate = 0.25
"#,
    );
    assert!(matches!(
        config.training.objective,
        TrainingObjectiveConfig::Sdft(_)
    ));
    config.validate().expect("sdft objective validates");
}

#[test]
fn sdpo_rejects_invalid_alpha() {
    let config = parse_config(
        r#"
[training.objective]
type = "sdpo"
alpha = 1.25
"#,
    );
    let err = config
        .validate()
        .expect_err("invalid sdpo alpha should fail");
    assert!(
        err.to_string().contains("training.objective.alpha"),
        "unexpected error: {err}"
    );
}

#[test]
fn sdft_rejects_unwired_top_entropy_quantile() {
    let config = parse_config(
        r#"
[training.objective]
type = "sdft"
top_entropy_quantile = 0.25
"#,
    );
    let err = config
        .validate()
        .expect_err("unwired SDFT entropy mask should fail");
    assert!(
        err.to_string().contains("top_entropy_quantile"),
        "unexpected error: {err}"
    );
}

#[test]
fn sdpo_rejects_unwired_reward_feedback_fields() {
    let config = parse_config(
        r#"
[training.objective]
type = "sdpo"
success_reward_threshold = 1.0
include_environment_feedback = true
"#,
    );
    let err = config
        .validate()
        .expect_err("unwired SDPO reward/feedback fields should fail");
    assert!(
        err.to_string().contains("success_reward_threshold"),
        "unexpected error: {err}"
    );
}

#[test]
fn sdpo_rejects_unwired_topk_fields() {
    let config = parse_config(
        r#"
[training.objective]
type = "sdpo"
distillation_topk = 100
"#,
    );
    let err = config
        .validate()
        .expect_err("unwired SDPO top-k distillation should fail");
    assert!(
        err.to_string().contains("distillation_topk"),
        "unexpected error: {err}"
    );
}

#[test]
fn sdft_sdpo_composite_objective_config_validates() {
    let config = parse_config(
        r#"
[training.objective]
type = "sdft_sdpo"
sdft_weight = 0.25
sdpo_weight = 0.75

[training.objective.sdft]
max_completion_tokens = 2
generate_from_teacher = true

[training.objective.sdpo]
group_size = 2
max_completion_tokens = 2
alpha = 0.25
"#,
    );
    assert!(matches!(
        config.training.objective,
        TrainingObjectiveConfig::SdftSdpo(_)
    ));
    config
        .validate()
        .expect("composite SDFT/SDPO objective validates");
}

#[test]
fn reservoir_model_initialization_config_validates() {
    let config = parse_config(
        r#"
[model]
n_layer = 1
n_embd = 32
n_head = 4
latent_total = 64

[model.initialization]
kind = "reservoir"

[model.initialization.reservoir]
seed = 1337
density = 0.08
encoder_value_scale = 0.70
decoder_scale = 1.00

[model.initialization.topology_prior]
kind = "modular_bridges"
community_count = 4
bridge_fraction = 0.03
intra_community_gain = 1.5
inter_community_gain = 0.5
bridge_gain = 1.0
"#,
    );
    config
        .validate()
        .expect("reservoir model initialization validates");
}

#[test]
fn legacy_gdpo_flag_is_mutually_exclusive_with_objective_switch() {
    let config = parse_config(
        r#"
[training.gdpo]
enabled = true
"#,
    );
    let err = config
        .validate()
        .expect_err("legacy gdpo objective flag should fail");
    assert!(
        err.to_string().contains("training.gdpo.enabled"),
        "unexpected error: {err}"
    );
}
