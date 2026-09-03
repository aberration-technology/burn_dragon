use super::*;
use crate::model::init::{
    DragonInitializationConfig, DragonInitializationKind, DragonReservoirInitializationConfig,
};
use burn::optim::GradientsParams;
use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;

type TestBackend = NdArray<f32>;
type TestAutodiffBackend = Autodiff<TestBackend>;

fn tensor_values<const D: usize>(tensor: Tensor<TestBackend, D>) -> Vec<f32> {
    tensor
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("tensor values")
}

fn tiny_scaling_source_config(sequence_kernel: SequenceKernelConfig) -> DragonConfig {
    DragonConfig {
        n_layer: 1,
        n_embd: 16,
        n_head: 2,
        mlp_internal_dim_multiplier: 2,
        vocab_size: 32,
        dropout: 0.0,
        sequence_kernel,
        ..Default::default()
    }
}

fn assert_widened_forward_is_finite(model: &DragonModel<TestBackend>) {
    let device = burn::tensor::Device::<TestBackend>::default();
    let tokens = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![1_i64, 2, 3], [1, 3]),
        &device,
    );
    let logits = model.forward(tokens);
    assert_eq!(logits.shape().dims(), [1, 3, 32]);
    assert!(tensor_values(logits).iter().all(|value| value.is_finite()));
}

fn max_abs_diff(lhs: Vec<f32>, rhs: Vec<f32>) -> f32 {
    assert_eq!(lhs.len(), rhs.len(), "tensor length mismatch");
    lhs.into_iter()
        .zip(rhs)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn optional_sequence_score_head_preserves_shared_initialization_and_backend_rng() {
    const ISOLATED_ENV: &str = "BURN_DRAGON_ISOLATED_SCORE_HEAD_RNG_TEST";
    if std::env::var_os(ISOLATED_ENV).is_none() {
        // NdArray's RNG is process-global, so a seed-sequence assertion must not share a
        // process with parallel tests that initialize unrelated models.
        let status = std::process::Command::new(
                std::env::current_exe().expect("current core test executable"),
            )
            .arg("model::dragon::tests::optional_sequence_score_head_preserves_shared_initialization_and_backend_rng")
            .arg("--exact")
            .env(ISOLATED_ENV, "1")
            .status()
            .expect("run isolated score-head RNG test");
        assert!(status.success(), "isolated score-head RNG test failed");
        return;
    }

    let device = burn::tensor::Device::<TestBackend>::default();
    let mut config = tiny_scaling_source_config(SequenceKernelConfig::default());
    config.latent_reasoning.enabled = true;
    config.latent_reasoning.max_steps = 2;
    let tokens = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![1_i64, 2, 3, 4], [1, 4]),
        &device,
    );

    TestBackend::seed(&device, 4_211);
    let without_head = DragonModel::<TestBackend>::new(config.clone(), &device);
    let without_head_logits = tensor_values(without_head.forward(tokens.clone()));
    let without_head_rng = tensor_values(Tensor::<TestBackend, 1>::random(
        [16],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    ));

    TestBackend::seed(&device, 4_211);
    config.sequence_score_head.enabled = true;
    let with_head = DragonModel::<TestBackend>::new(config, &device);
    let with_head_logits = tensor_values(with_head.forward(tokens));
    let with_head_rng = tensor_values(Tensor::<TestBackend, 1>::random(
        [16],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    ));

    assert_eq!(
        max_abs_diff(without_head_logits, with_head_logits),
        0.0,
        "optional score head perturbed shared Dragon initialization"
    );
    assert_eq!(
        max_abs_diff(without_head_rng, with_head_rng),
        0.0,
        "optional score head advanced the backend-global RNG"
    );
}

fn tiny_random_scaffold_config(seed: u64) -> DragonConfig {
    let mut config = tiny_scaling_source_config(SequenceKernelConfig::default());
    config.dropout = 0.0;
    config.random_scaffold.enabled = true;
    config.random_scaffold.seed = seed;
    config.random_scaffold.rank = 4;
    config.random_scaffold.alpha = 16.0;
    config
}

#[test]
fn random_scaffold_zero_b_is_end_to_end_function_preserving() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let mut model = DragonModel::<TestBackend>::new(tiny_random_scaffold_config(31), &device);
    let adapters = model.random_scaffold_adapters.take();
    let raw = model.shared_lowrank_weights();
    model.random_scaffold_adapters = adapters;
    let effective = model.shared_lowrank_effective_weights();
    assert_eq!(
        tensor_values(raw.encoder),
        tensor_values(effective.encoder),
        "zero-B encoder adapter changed the scaffold"
    );
    assert_eq!(
        tensor_values(raw.encoder_v),
        tensor_values(effective.encoder_v),
        "zero-B encoder-v adapter changed the scaffold"
    );
    assert_eq!(
        tensor_values(raw.decoder),
        tensor_values(effective.decoder),
        "zero-B decoder adapter changed the scaffold"
    );
    let tokens = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![1_i64, 2, 3, 4], [1, 4]),
        &device,
    );
    let adapters = model.random_scaffold_adapters.take();
    let expected = model.forward(tokens.clone());
    model.random_scaffold_adapters = adapters;
    let actual = model.forward(tokens);
    let diff = max_abs_diff(tensor_values(expected), tensor_values(actual));
    assert!(diff <= 1.0e-6, "zero-B scaffold adapter drifted by {diff}");
}

#[test]
fn random_scaffold_record_round_trip_preserves_logits_and_manifest() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let config = tiny_random_scaffold_config(37);
    let model = DragonModel::<TestBackend>::new(config.clone(), &device);
    let tokens = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![1_i64, 2, 3, 4], [1, 4]),
        &device,
    );
    let expected = model.forward(tokens.clone());
    let expected_identity = model
        .random_scaffold_report()
        .expect("scaffold report")
        .manifest
        .canonical_identity()
        .expect("manifest identity");
    let record = model.into_record();
    let reloaded = DragonModel::<TestBackend>::new(config, &device).load_record(record);
    let actual = reloaded.forward(tokens);
    let actual_identity = reloaded
        .random_scaffold_report()
        .expect("scaffold report")
        .manifest
        .canonical_identity()
        .expect("manifest identity");
    let diff = max_abs_diff(tensor_values(expected), tensor_values(actual));
    assert!(diff <= 1.0e-6, "scaffold checkpoint drifted by {diff}");
    assert_eq!(actual_identity, expected_identity);
}

#[test]
fn random_scaffold_inference_materialization_preserves_logits() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let model = DragonModel::<TestBackend>::new(tiny_random_scaffold_config(39), &device);
    let tokens = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![1_i64, 2, 3, 4], [1, 4]),
        &device,
    );
    let expected = model.forward(tokens.clone());
    let materialized = model.materialize_random_scaffold_for_inference();
    assert!(!materialized.uses_random_scaffold());
    let actual = materialized.forward(tokens);
    let diff = max_abs_diff(tensor_values(expected), tensor_values(actual));
    assert!(
        diff <= 1.0e-6,
        "materialized scaffold inference drifted by {diff}"
    );
}

#[test]
fn random_scaffold_backward_excludes_base_and_reaches_adapter() {
    let device = burn::tensor::Device::<TestAutodiffBackend>::default();
    let model = DragonModel::<TestAutodiffBackend>::new(tiny_random_scaffold_config(41), &device);
    let tokens = Tensor::<TestAutodiffBackend, 2, Int>::from_data(
        TensorData::new(vec![1_i64, 2, 3, 4], [1, 4]),
        &device,
    );
    let loss = model.forward(tokens).powf_scalar(2.0).mean();
    let grads = GradientsParams::from_grads(loss.backward(), &model);
    let base_ids = model.shared_lowrank_param_ids();
    assert!(
        grads.get::<TestBackend, 3>(base_ids.encoder).is_none(),
        "immutable encoder scaffold received a gradient"
    );
    assert!(
        grads.get::<TestBackend, 3>(base_ids.encoder_v).is_none(),
        "immutable encoder-v scaffold received a gradient"
    );
    assert!(
        grads.get::<TestBackend, 2>(base_ids.decoder).is_none(),
        "immutable decoder scaffold received a gradient"
    );

    let adapters = model
        .random_scaffold_adapters
        .as_ref()
        .expect("scaffold adapters");
    assert!(
        grads
            .get::<TestBackend, 3>(adapters.fast.encoder.b.id)
            .is_some(),
        "encoder adapter B did not receive a gradient"
    );
    assert!(
        grads
            .get::<TestBackend, 2>(adapters.fast.decoder.b.id)
            .is_some(),
        "decoder adapter B did not receive a gradient"
    );
    assert!(
        grads
            .get::<TestBackend, 1>(adapters.fast.encoder.gain.id)
            .is_some(),
        "scaffold gain did not receive a gradient"
    );
}

#[test]
fn tied_language_head_projects_with_input_embeddings() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let mut config = tiny_scaling_source_config(SequenceKernelConfig::default());
    config.tie_input_output_embeddings = true;
    let model = DragonModel::<TestBackend>::new(config, &device);
    let hidden = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(
            (0..16).map(|value| value as f32 / 16.0).collect(),
            [1, 1, 16],
        ),
        &device,
    );
    let logits = model.logits_from_hidden(hidden.clone());
    let expected = hidden
        .reshape([1, 16])
        .matmul(model.embed.weight.val().transpose())
        .reshape([1, 1, 32]);
    let diff = max_abs_diff(tensor_values(logits), tensor_values(expected));
    assert!(diff <= 1e-6, "tied logits drifted by {diff}");
}

#[test]
fn latent_reasoning_forward_returns_finite_reasoned_hidden() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let mut config = tiny_scaling_source_config(SequenceKernelConfig::default());
    config.latent_reasoning.enabled = true;
    config.latent_reasoning.max_steps = 2;
    config.latent_reasoning.min_steps = 1;
    config.latent_reasoning.adaptive_halting = true;
    config.latent_reasoning.energy_head = true;
    let model = DragonModel::<TestBackend>::new(config, &device);
    let tokens = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![1_i64, 2, 3, 4], [1, 4]),
        &device,
    );

    let raw = model.forward_hidden_raw(tokens.clone());
    let reasoned = model.forward_hidden(tokens.clone());
    let logits = model.forward(tokens.clone());
    let output = model.reason_hidden(raw.clone());

    assert_eq!(raw.shape().dims(), [1, 4, 16]);
    assert_eq!(reasoned.shape().dims(), [1, 4, 16]);
    assert_eq!(logits.shape().dims(), [1, 4, 32]);
    assert_eq!(output.steps_used, 2);
    assert_eq!(output.energies.len(), 2);
    assert_eq!(output.stop_probs.len(), 2);
    assert!(tensor_values(logits).iter().all(|value| value.is_finite()));
    assert!(
        tensor_values(reasoned)
            .iter()
            .all(|value| value.is_finite()),
        "reasoned hidden contains non-finite values"
    );
}

#[test]
fn latent_reasoning_default_refiner_starts_as_identity_residual() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let mut config = tiny_scaling_source_config(SequenceKernelConfig::default());
    config.latent_reasoning.enabled = true;
    let model = DragonModel::<TestBackend>::new(config, &device);
    let tokens = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![1_i64, 2, 3, 4], [1, 4]),
        &device,
    );

    let raw = model.forward_hidden_raw(tokens);
    let output = model.reason_hidden(raw.clone());
    let diff = max_abs_diff(tensor_values(raw), tensor_values(output.final_hidden));

    assert!(
        diff <= 1e-6,
        "zero-initialized latent residual should preserve hidden at init; diff={diff}"
    );
    assert_eq!(output.steps_used, 1);
    assert_eq!(output.energies.len(), 0);
    assert_eq!(output.stop_probs.len(), 0);
}

#[test]
fn latent_residual_refinement_gate_scales_learned_updates() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let hidden = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(
            (0..32).map(|value| value as f32 / 16.0 - 1.0).collect(),
            [1, 2, 16],
        ),
        &device,
    );

    TestBackend::seed(&device, 13);
    let mut gated_config = tiny_scaling_source_config(SequenceKernelConfig::default());
    gated_config.latent_reasoning.enabled = true;
    gated_config.latent_reasoning.max_steps = 1;
    gated_config.latent_reasoning.min_steps = 1;
    gated_config.latent_reasoning.residual_refinement_gate = true;
    gated_config.latent_reasoning.residual_refinement_gate_init = 0.25;
    let mut gated_model = DragonModel::<TestBackend>::new(gated_config, &device);

    let out = gated_model
        .latent_refiner_out
        .as_mut()
        .expect("latent refiner output");
    let [rows, cols] = out.weight.val().shape().dims();
    out.weight =
        Param::from_tensor(Tensor::<TestBackend, 2>::ones([rows, cols], &device).mul_scalar(0.01));
    if let Some(bias) = out.bias.as_mut() {
        let [dim] = bias.val().shape().dims();
        *bias = Param::from_tensor(Tensor::<TestBackend, 1>::zeros([dim], &device));
    }

    let gate = gated_model.latent_refiner_gate.take();
    let open_delta = gated_model.reason_hidden(hidden.clone()).final_hidden - hidden.clone();
    gated_model.latent_refiner_gate = gate;
    let gated_delta = gated_model.reason_hidden(hidden.clone()).final_hidden - hidden;
    let expected = open_delta.mul_scalar(0.25);
    let diff = max_abs_diff(tensor_values(gated_delta), tensor_values(expected));

    assert!(
        diff <= 1.0e-5,
        "residual refinement gate should scale update by init multiplier; diff={diff}"
    );
}

#[test]
fn latent_step_conditioned_decoder_starts_neutral_and_can_shift_step_logits() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let mut config = tiny_scaling_source_config(SequenceKernelConfig::default());
    config.latent_reasoning.enabled = true;
    config.latent_reasoning.max_steps = 2;
    config.latent_reasoning.min_steps = 2;
    config.latent_reasoning.step_conditioned_decoder = true;
    let mut model = DragonModel::<TestBackend>::new(config, &device);
    let hidden = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(
            (0..32).map(|value| value as f32 / 32.0).collect(),
            [1, 2, 16],
        ),
        &device,
    );

    let neutral_step0 = model.logits_from_hidden_for_latent_step(hidden.clone(), 0);
    let neutral_step2 = model.logits_from_hidden_for_latent_step(hidden.clone(), 2);
    let neutral_diff = max_abs_diff(tensor_values(neutral_step0), tensor_values(neutral_step2));
    assert!(
        neutral_diff <= 1.0e-6,
        "zero-initialized step decoder should preserve logits; diff={neutral_diff}"
    );

    let mut values = vec![0.0f32; 3 * 16];
    for index in 0..16 {
        values[2 * 16 + index] = 0.05;
    }
    model.latent_step_decoder_embedding = Some(Param::from_tensor(
        Tensor::<TestBackend, 2>::from_data(TensorData::new(values, [3, 16]), &device),
    ));

    let shifted_step0 = model.logits_from_hidden_for_latent_step(hidden.clone(), 0);
    let shifted_step2 = model.logits_from_hidden_for_latent_step(hidden, 2);
    let shifted_diff = max_abs_diff(tensor_values(shifted_step0), tensor_values(shifted_step2));
    assert!(
        shifted_diff > 1.0e-5,
        "nonzero step decoder embedding should change step-specific logits"
    );
}

#[test]
fn next_latent_transition_starts_as_identity_delta() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let mut config = tiny_scaling_source_config(SequenceKernelConfig::default());
    config.next_latent_transition.enabled = true;
    let model = DragonModel::<TestBackend>::new(config, &device);
    let tokens = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![1_i64, 2, 3, 4], [1, 4]),
        &device,
    );

    let raw = model.forward_hidden_raw(tokens.clone());
    let hidden = model.forward_hidden(tokens.clone());
    let context = hidden.clone().slice([0..1, 0..3, 0..16]);
    let action_tokens = tokens.slice([0..1, 1..4]);
    let action_embedding = model.embed_tokens(action_tokens);
    let prediction = model
        .next_latent_prediction_from_hidden_action(context.clone(), action_embedding)
        .expect("next latent transition enabled");
    let diff = max_abs_diff(tensor_values(context), tensor_values(prediction));
    let forward_diff = max_abs_diff(tensor_values(raw), tensor_values(hidden));

    assert!(
        diff <= 1.0e-6,
        "zero-initialized transition drifted by {diff}"
    );
    assert!(
        forward_diff <= 1.0e-6,
        "NextLat transition should not alter forward_hidden; diff={forward_diff}"
    );
}

#[test]
fn hierarchical_dragon_split_rho_shared_weights_forward_persists_slow_state() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let mut config = tiny_scaling_source_config(SequenceKernelConfig::default());
    config.hierarchical_dragon.enabled = true;
    config.hierarchical_dragon.last_layers = Some(1);
    config.hierarchical_dragon.fast_cycles = 1;
    config.hierarchical_dragon.slow_cycles = 1;
    config.hierarchical_dragon.rho_sharing = HierarchicalDragonSharing::Split;
    config.hierarchical_dragon.weight_sharing = HierarchicalDragonSharing::Shared;
    config.hierarchical_dragon.slow_to_fast_scale = 0.1;
    config.hierarchical_dragon.fast_to_slow_scale = 0.1;
    let model = DragonModel::<TestBackend>::new(config, &device);
    let mut state = model.init_state();
    let tokens = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![1_i64, 2, 3, 4], [1, 4]),
        &device,
    );

    let logits = model.forward_with_state(tokens, &mut state);

    assert_eq!(logits.shape().dims(), [1, 4, 32]);
    assert!(tensor_values(logits).iter().all(|value| value.is_finite()));
    assert!(state.layers[0].rho.is_some(), "fast rho should be written");
    assert!(
        state.layers[0].slow_rho.is_some(),
        "split slow rho should be written"
    );
    assert!(
        state.layers[0].hierarchical_slow_hidden.is_some(),
        "slow hidden summary should be retained"
    );
    assert!(model.slow_encoder.is_none());
    assert!(model.slow_encoder_v.is_none());
    assert!(model.slow_decoder.is_none());
    assert!(!model.supports_shared_lowrank_population_forward());
    assert!(!model.supports_shared_lowrank_continual_backprop());
}

#[test]
fn hierarchical_dragon_split_weights_forward_and_record_round_trip() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let mut config = tiny_scaling_source_config(SequenceKernelConfig::default());
    config.hierarchical_dragon.enabled = true;
    config.hierarchical_dragon.last_layers = Some(1);
    config.hierarchical_dragon.fast_cycles = 1;
    config.hierarchical_dragon.slow_cycles = 1;
    config.hierarchical_dragon.rho_sharing = HierarchicalDragonSharing::Split;
    config.hierarchical_dragon.weight_sharing = HierarchicalDragonSharing::Split;
    let model = DragonModel::<TestBackend>::new(config.clone(), &device);
    let tokens = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![1_i64, 2, 3, 4], [1, 4]),
        &device,
    );

    let logits = model.forward(tokens.clone());
    let record = model.clone().into_record();
    let reloaded = DragonModel::<TestBackend>::new(config, &device).load_record(record);
    let reloaded_logits = reloaded.forward(tokens);
    let diff = max_abs_diff(
        tensor_values(logits.clone()),
        tensor_values(reloaded_logits),
    );

    assert_eq!(logits.shape().dims(), [1, 4, 32]);
    assert!(tensor_values(logits).iter().all(|value| value.is_finite()));
    assert!(model.slow_encoder.is_some());
    assert!(model.slow_encoder_v.is_some());
    assert!(model.slow_decoder.is_some());
    assert!(
        diff <= 1.0e-6,
        "split hierarchical record round-trip drifted by {diff}"
    );
}

#[test]
fn shared_lowrank_population_forward_matches_base_for_single_member() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let config = tiny_scaling_source_config(SequenceKernelConfig::default());
    let model = DragonModel::<TestBackend>::new(config, &device);
    let tokens = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![1_i64, 2, 3, 4, 5, 6], [2, 3]),
        &device,
    );
    let base = model.shared_lowrank_weights();
    let population = SharedLowrankPopulationWeights {
        encoder: base.encoder.reshape([
            1,
            model.n_head,
            model.n_embd,
            model.latent_per_head_capacity(),
        ]),
        encoder_v: base.encoder_v.reshape([
            1,
            model.n_head,
            model.n_embd,
            model.latent_per_head_capacity(),
        ]),
        decoder: base
            .decoder
            .reshape([1, model.latent_total_capacity(), model.n_embd]),
    };

    let expected = model.forward(tokens.clone());
    let actual = model.forward_with_shared_lowrank_population(tokens, population);
    let diff = max_abs_diff(tensor_values(expected), tensor_values(actual));
    assert!(diff <= 1e-5, "population forward drifted by {diff}");
}

#[test]
fn shared_lowrank_population_forward_keeps_members_independent() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let config = tiny_scaling_source_config(SequenceKernelConfig::default());
    let model = DragonModel::<TestBackend>::new(config, &device);
    let tokens = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![1_i64, 2, 3, 4, 5, 6], [2, 3]),
        &device,
    );
    let base = model.shared_lowrank_weights();
    let population = SharedLowrankPopulationWeights {
        encoder: Tensor::cat(
            vec![
                base.encoder.clone().reshape([
                    1,
                    model.n_head,
                    model.n_embd,
                    model.latent_per_head_capacity(),
                ]),
                base.encoder.reshape([
                    1,
                    model.n_head,
                    model.n_embd,
                    model.latent_per_head_capacity(),
                ]),
            ],
            0,
        ),
        encoder_v: Tensor::cat(
            vec![
                base.encoder_v.clone().reshape([
                    1,
                    model.n_head,
                    model.n_embd,
                    model.latent_per_head_capacity(),
                ]),
                base.encoder_v.reshape([
                    1,
                    model.n_head,
                    model.n_embd,
                    model.latent_per_head_capacity(),
                ]),
            ],
            0,
        ),
        decoder: Tensor::cat(
            vec![
                base.decoder
                    .clone()
                    .reshape([1, model.latent_total_capacity(), model.n_embd]),
                base.decoder
                    .reshape([1, model.latent_total_capacity(), model.n_embd]),
            ],
            0,
        ),
    };

    let expected = model.forward(tokens.clone());
    let stacked = model.forward_with_shared_lowrank_population(tokens, population);
    let first = stacked.clone().slice_dim(0, 0..2);
    let second = stacked.slice_dim(0, 2..4);
    let first_diff = max_abs_diff(tensor_values(expected.clone()), tensor_values(first));
    let second_diff = max_abs_diff(tensor_values(expected), tensor_values(second));
    assert!(
        first_diff <= 1e-5,
        "first population drifted by {first_diff}"
    );
    assert!(
        second_diff <= 1e-5,
        "second population drifted by {second_diff}"
    );
}

#[test]
fn shared_lowrank_population_forward_does_not_couple_different_members() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let config = tiny_scaling_source_config(SequenceKernelConfig::default());
    let model = DragonModel::<TestBackend>::new(config, &device);
    let tokens = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![1_i64, 2, 3, 4, 5, 6], [2, 3]),
        &device,
    );
    let base = model.shared_lowrank_weights();
    let shifted_encoder = base.encoder.clone().add_scalar(1.0e-3);
    let shifted_encoder_v = base.encoder_v.clone().sub_scalar(1.0e-3);
    let shifted_decoder = base.decoder.clone().add_scalar(1.0e-3);
    let population = SharedLowrankPopulationWeights {
        encoder: Tensor::cat(
            vec![
                base.encoder.clone().reshape([
                    1,
                    model.n_head,
                    model.n_embd,
                    model.latent_per_head_capacity(),
                ]),
                shifted_encoder.reshape([
                    1,
                    model.n_head,
                    model.n_embd,
                    model.latent_per_head_capacity(),
                ]),
            ],
            0,
        ),
        encoder_v: Tensor::cat(
            vec![
                base.encoder_v.clone().reshape([
                    1,
                    model.n_head,
                    model.n_embd,
                    model.latent_per_head_capacity(),
                ]),
                shifted_encoder_v.reshape([
                    1,
                    model.n_head,
                    model.n_embd,
                    model.latent_per_head_capacity(),
                ]),
            ],
            0,
        ),
        decoder: Tensor::cat(
            vec![
                base.decoder
                    .clone()
                    .reshape([1, model.latent_total_capacity(), model.n_embd]),
                shifted_decoder.reshape([1, model.latent_total_capacity(), model.n_embd]),
            ],
            0,
        ),
    };

    let expected = model.forward(tokens.clone());
    let stacked = model.forward_with_shared_lowrank_population(tokens, population);
    let first = stacked.slice_dim(0, 0..2);
    let diff = max_abs_diff(tensor_values(expected), tensor_values(first));
    assert!(diff <= 1e-5, "base population was coupled by {diff}");
}

#[test]
fn shared_lowrank_population_forward_single_head_keeps_members_independent() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let mut config = tiny_scaling_source_config(SequenceKernelConfig::default());
    config.n_embd = 8;
    config.n_head = 1;
    config.mlp_internal_dim_multiplier = 1;
    config.vocab_size = 16;
    let model = DragonModel::<TestBackend>::new(config, &device);
    let tokens = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![1_i64, 2, 3, 4, 5, 6, 7, 8], [2, 4]),
        &device,
    );
    let base = model.shared_lowrank_weights();
    let shifted_encoder = base.encoder.clone().add_scalar(1.0e-3);
    let shifted_encoder_v = base.encoder_v.clone().sub_scalar(1.0e-3);
    let shifted_decoder = base.decoder.clone().add_scalar(1.0e-3);
    let population = SharedLowrankPopulationWeights {
        encoder: Tensor::cat(
            vec![
                base.encoder.clone().reshape([
                    1,
                    model.n_head,
                    model.n_embd,
                    model.latent_per_head_capacity(),
                ]),
                shifted_encoder.reshape([
                    1,
                    model.n_head,
                    model.n_embd,
                    model.latent_per_head_capacity(),
                ]),
            ],
            0,
        ),
        encoder_v: Tensor::cat(
            vec![
                base.encoder_v.clone().reshape([
                    1,
                    model.n_head,
                    model.n_embd,
                    model.latent_per_head_capacity(),
                ]),
                shifted_encoder_v.reshape([
                    1,
                    model.n_head,
                    model.n_embd,
                    model.latent_per_head_capacity(),
                ]),
            ],
            0,
        ),
        decoder: Tensor::cat(
            vec![
                base.decoder
                    .clone()
                    .reshape([1, model.latent_total_capacity(), model.n_embd]),
                shifted_decoder.reshape([1, model.latent_total_capacity(), model.n_embd]),
            ],
            0,
        ),
    };

    let expected = model.forward(tokens.clone());
    let stacked = model.forward_with_shared_lowrank_population(tokens, population);
    let first = stacked.slice_dim(0, 0..2);
    let diff = max_abs_diff(tensor_values(expected), tensor_values(first));
    assert!(
        diff <= 1e-5,
        "single-head base population was coupled by {diff}"
    );
}

#[test]
fn linear_attention_incremental_forward_matches_full_sequence() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let config = tiny_scaling_source_config(SequenceKernelConfig::reference(
        SequenceMemorySystem::LinearAttention,
    ));
    let model = DragonModel::<TestBackend>::new(config, &device);
    let tokens = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![1_i64, 2, 3, 4, 5, 6], [1, 6]),
        &device,
    );

    let full_logits = model.forward(tokens.clone());
    let mut state = model.init_state();
    let mut pieces = Vec::new();
    for index in 0..6 {
        let token = tokens.clone().slice([0..1, index..index + 1]);
        pieces.push(model.forward_with_state(token, &mut state));
    }
    let incremental_logits = Tensor::cat(pieces, 1);
    let diff = max_abs_diff(
        tensor_values(full_logits),
        tensor_values(incremental_logits),
    );
    assert!(
        diff <= 1.0e-4,
        "linear-attention incremental logits drifted from full sequence by {diff}"
    );
    assert_eq!(state.position, 6);
}

fn assert_widened_forward_matches_source(
    source: &DragonModel<TestBackend>,
    widened: &DragonModel<TestBackend>,
    tolerance: f32,
) {
    let device = burn::tensor::Device::<TestBackend>::default();
    let tokens = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![1_i64, 2, 3, 4], [1, 4]),
        &device,
    );
    let embedding_weight_diff = max_abs_diff(
        tensor_values(source.embed.weight.val()),
        tensor_values(widened.embed.weight.val()),
    );
    assert!(
        embedding_weight_diff <= tolerance,
        "widened model changed embedding weights before training: max_abs_diff={embedding_weight_diff} tolerance={tolerance}"
    );
    let source_embedded = tensor_values(source.embed_tokens(tokens.clone()));
    let widened_embedded = tensor_values(widened.embed_tokens(tokens.clone()));
    let embedded_diff = max_abs_diff(source_embedded, widened_embedded);
    assert!(
        embedded_diff <= tolerance,
        "widened model changed embeddings before training: max_abs_diff={embedded_diff} tolerance={tolerance}"
    );
    let source_hidden = tensor_values(source.forward_hidden(tokens.clone()));
    let widened_hidden = tensor_values(widened.forward_hidden(tokens.clone()));
    let hidden_diff = max_abs_diff(source_hidden, widened_hidden);
    assert!(
        hidden_diff <= tolerance,
        "widened model changed hidden states before training: max_abs_diff={hidden_diff} tolerance={tolerance}"
    );
    let source_logits = tensor_values(source.forward(tokens.clone()));
    let widened_logits = tensor_values(widened.forward(tokens));
    let diff = max_abs_diff(source_logits, widened_logits);
    assert!(
        diff <= tolerance,
        "widened model changed logits before training: max_abs_diff={diff} tolerance={tolerance}"
    );
}

fn assert_widened_record_round_trip_matches_source(
    source: &DragonModel<TestBackend>,
    widened: &DragonModel<TestBackend>,
    target_config: DragonConfig,
    tolerance: f32,
) {
    let device = burn::tensor::Device::<TestBackend>::default();
    let record = widened.clone().into_record();
    let reloaded = DragonModel::<TestBackend>::new(target_config, &device).load_record(record);
    assert_widened_forward_matches_source(source, &reloaded, tolerance);
}

fn assert_shared_lowrank_prefix_preserved(
    source: &DragonModel<TestBackend>,
    widened: &DragonModel<TestBackend>,
) {
    let old_latent_per_head = source.latent_per_head_capacity();
    assert_eq!(
        tensor_values(source.encoder.val()),
        tensor_values(widened.encoder.val().slice([
            0..source.n_head,
            0..source.n_embd,
            0..old_latent_per_head
        ]))
    );
    assert_eq!(
        tensor_values(source.encoder_v.val()),
        tensor_values(widened.encoder_v.val().slice([
            0..source.n_head,
            0..source.n_embd,
            0..old_latent_per_head
        ]))
    );
    for head in 0..source.n_head {
        let source_start = head * old_latent_per_head;
        let widened_start = head * widened.latent_per_head_capacity();
        assert_eq!(
            tensor_values(source.decoder.val().slice([
                source_start..source_start + old_latent_per_head,
                0..source.n_embd
            ])),
            tensor_values(widened.decoder.val().slice([
                widened_start..widened_start + old_latent_per_head,
                0..source.n_embd
            ]))
        );
    }
    assert!(
        tensor_values(widened.encoder.val().slice([
            0..source.n_head,
            0..source.n_embd,
            old_latent_per_head..widened.latent_per_head_capacity()
        ]))
        .iter()
        .all(|value| *value == 0.0),
        "widened query encoder tail should start as a no-op"
    );
}

fn assert_slow_lowrank_prefix_preserved(
    source: &DragonModel<TestBackend>,
    widened: &DragonModel<TestBackend>,
) {
    let old_latent_per_head = source.latent_per_head_capacity();
    let source_encoder = source.slow_encoder.as_ref().expect("source slow encoder");
    let source_encoder_v = source
        .slow_encoder_v
        .as_ref()
        .expect("source slow encoder_v");
    let source_decoder = source.slow_decoder.as_ref().expect("source slow decoder");
    let widened_encoder = widened.slow_encoder.as_ref().expect("widened slow encoder");
    let widened_encoder_v = widened
        .slow_encoder_v
        .as_ref()
        .expect("widened slow encoder_v");
    let widened_decoder = widened.slow_decoder.as_ref().expect("widened slow decoder");

    assert_eq!(
        tensor_values(source_encoder.val()),
        tensor_values(widened_encoder.val().slice([
            0..source.n_head,
            0..source.n_embd,
            0..old_latent_per_head
        ]))
    );
    assert_eq!(
        tensor_values(source_encoder_v.val()),
        tensor_values(widened_encoder_v.val().slice([
            0..source.n_head,
            0..source.n_embd,
            0..old_latent_per_head
        ]))
    );
    for head in 0..source.n_head {
        let source_start = head * old_latent_per_head;
        let widened_start = head * widened.latent_per_head_capacity();
        assert_eq!(
            tensor_values(source_decoder.val().slice([
                source_start..source_start + old_latent_per_head,
                0..source.n_embd
            ])),
            tensor_values(widened_decoder.val().slice([
                widened_start..widened_start + old_latent_per_head,
                0..source.n_embd
            ]))
        );
    }
    assert!(
        tensor_values(widened_encoder.val().slice([
            0..source.n_head,
            0..source.n_embd,
            old_latent_per_head..widened.latent_per_head_capacity()
        ]))
        .iter()
        .all(|value| *value == 0.0),
        "widened slow query encoder tail should start as a no-op"
    );
}

#[test]
fn tiny_reservoir_model_constructs_and_runs_forward() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let config = DragonConfig {
        n_layer: 1,
        n_embd: 16,
        n_head: 2,
        mlp_internal_dim_multiplier: 2,
        vocab_size: 32,
        dropout: 0.0,
        initialization: DragonInitializationConfig {
            kind: DragonInitializationKind::Reservoir,
            reservoir: DragonReservoirInitializationConfig {
                seed: 7,
                density: 0.2,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let model = DragonModel::<TestBackend>::new(config, &device);
    let tokens = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![1_i64, 2, 3], [1, 3]),
        &device,
    );
    let logits = model.forward(tokens);
    assert_eq!(logits.shape().dims(), [1, 3, 32]);
    let values = logits
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("logits");
    assert!(values.iter().all(|value| value.is_finite()));
}

#[test]
fn tiny_gated_deltanet2_model_constructs_and_runs_forward() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let config = DragonConfig {
        n_layer: 1,
        n_embd: 16,
        n_head: 2,
        mlp_internal_dim_multiplier: 2,
        vocab_size: 32,
        dropout: 0.0,
        sequence_kernel: SequenceKernelConfig::reference(SequenceMemorySystem::GatedDeltaNet2),
        ..Default::default()
    };
    let model = DragonModel::<TestBackend>::new(config, &device);
    let tokens = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![1_i64, 2, 3], [1, 3]),
        &device,
    );
    let logits = model.forward(tokens);
    assert_eq!(logits.shape().dims(), [1, 3, 32]);
    let values = logits
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("logits");
    assert!(values.iter().all(|value| value.is_finite()));
}

#[test]
fn hierarchical_dragon_split_rho_mamba3_forward_persists_slow_state() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let mut config = tiny_scaling_source_config(SequenceKernelConfig::reference(
        SequenceMemorySystem::Mamba3StateSpaceDuality,
    ));
    config.mamba = super::super::sequence::mamba::MambaSequenceConfig {
        headdim: 8,
        chunk_size: 4,
        ..Default::default()
    };
    config.hierarchical_dragon.enabled = true;
    config.hierarchical_dragon.last_layers = Some(1);
    config.hierarchical_dragon.fast_cycles = 1;
    config.hierarchical_dragon.slow_cycles = 1;
    config.hierarchical_dragon.rho_sharing = HierarchicalDragonSharing::Split;
    let model = DragonModel::<TestBackend>::new(config, &device);
    let mut state = model.init_state();
    let tokens = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![1_i64, 2, 3], [1, 3]),
        &device,
    );

    let logits = model.forward_with_state(tokens, &mut state);

    assert_eq!(logits.shape().dims(), [1, 3, 32]);
    assert!(tensor_values(logits).iter().all(|value| value.is_finite()));
    assert!(state.layers[0].rho.is_some(), "fast Mamba3 state");
    assert!(state.layers[0].slow_rho.is_some(), "slow Mamba3 state");
    assert!(
        state.layers[0].slow_mamba_angle_state.is_some(),
        "slow Mamba3 angle state"
    );
}

#[test]
fn hierarchical_dragon_split_rho_gdn2_forward_persists_slow_state() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let mut config = tiny_scaling_source_config(SequenceKernelConfig::reference(
        SequenceMemorySystem::GatedDeltaNet2,
    ));
    config.hierarchical_dragon.enabled = true;
    config.hierarchical_dragon.last_layers = Some(1);
    config.hierarchical_dragon.fast_cycles = 1;
    config.hierarchical_dragon.slow_cycles = 1;
    config.hierarchical_dragon.rho_sharing = HierarchicalDragonSharing::Split;
    let model = DragonModel::<TestBackend>::new(config, &device);
    let mut state = model.init_state();
    let tokens = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![1_i64, 2, 3], [1, 3]),
        &device,
    );

    let logits = model.forward_with_state(tokens, &mut state);

    assert_eq!(logits.shape().dims(), [1, 3, 32]);
    assert!(tensor_values(logits).iter().all(|value| value.is_finite()));
    assert!(state.layers[0].rho.is_some(), "fast GDN2 state");
    assert!(state.layers[0].slow_rho.is_some(), "slow GDN2 state");
}

#[test]
fn widen_latent_total_supports_linear_attention() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let source_config = tiny_scaling_source_config(SequenceKernelConfig::reference(
        SequenceMemorySystem::LinearAttention,
    ));
    let target_config = DragonConfig {
        mlp_internal_dim_multiplier: 4,
        ..source_config.clone()
    };
    let source = DragonModel::<TestBackend>::new(source_config, &device);
    let (widened, report) = source
        .widen_latent_total(target_config.clone(), &device)
        .expect("widen");
    assert_eq!(report.old_latent_total, 32);
    assert_eq!(report.new_latent_total, 64);
    assert_eq!(widened.latent_total_capacity(), 64);
    assert_shared_lowrank_prefix_preserved(&source, &widened);
    assert_widened_forward_matches_source(&source, &widened, 1.0e-5);
    assert_widened_record_round_trip_matches_source(&source, &widened, target_config, 1.0e-5);
    assert_widened_forward_is_finite(&widened);
}

#[test]
fn widen_latent_total_supports_split_hierarchical_dragon_weights() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let mut source_config = tiny_scaling_source_config(SequenceKernelConfig::reference(
        SequenceMemorySystem::LinearAttention,
    ));
    source_config.hierarchical_dragon.enabled = true;
    source_config.hierarchical_dragon.last_layers = Some(1);
    source_config.hierarchical_dragon.fast_cycles = 1;
    source_config.hierarchical_dragon.slow_cycles = 1;
    source_config.hierarchical_dragon.rho_sharing = HierarchicalDragonSharing::Split;
    source_config.hierarchical_dragon.weight_sharing = HierarchicalDragonSharing::Split;
    let target_config = DragonConfig {
        mlp_internal_dim_multiplier: 4,
        ..source_config.clone()
    };
    let source = DragonModel::<TestBackend>::new(source_config, &device);
    let (widened, report) = source
        .widen_latent_total(target_config.clone(), &device)
        .expect("widen split hierarchy");

    assert_eq!(report.old_latent_total, 32);
    assert_eq!(report.new_latent_total, 64);
    assert_shared_lowrank_prefix_preserved(&source, &widened);
    assert_slow_lowrank_prefix_preserved(&source, &widened);
    assert_widened_forward_matches_source(&source, &widened, 1.0e-5);
    assert_widened_record_round_trip_matches_source(&source, &widened, target_config, 1.0e-5);
    assert_widened_forward_is_finite(&widened);
}

#[test]
fn widen_latent_total_supports_dense_score_short_context() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let source_config =
        tiny_scaling_source_config(SequenceKernelConfig::dense_score_short_context());
    let target_config = DragonConfig {
        mlp_internal_dim_multiplier: 4,
        ..source_config.clone()
    };
    let source = DragonModel::<TestBackend>::new(source_config, &device);
    let (widened, report) = source
        .widen_latent_total(target_config.clone(), &device)
        .expect("widen");
    assert_eq!(report.new_latent_total, 64);
    assert_shared_lowrank_prefix_preserved(&source, &widened);
    assert_widened_forward_matches_source(&source, &widened, 1.0e-5);
    assert_widened_record_round_trip_matches_source(&source, &widened, target_config, 1.0e-5);
    assert_widened_forward_is_finite(&widened);
}

#[test]
fn widen_latent_total_supports_mamba3_and_preserves_mamba_params() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let source_config = DragonConfig {
        sequence_kernel: SequenceKernelConfig::reference(
            SequenceMemorySystem::Mamba3StateSpaceDuality,
        ),
        mamba: super::super::sequence::mamba::MambaSequenceConfig {
            headdim: 8,
            chunk_size: 4,
            ..Default::default()
        },
        ..tiny_scaling_source_config(SequenceKernelConfig::reference(
            SequenceMemorySystem::Mamba3StateSpaceDuality,
        ))
    };
    let target_config = DragonConfig {
        mlp_internal_dim_multiplier: 4,
        ..source_config.clone()
    };
    let source = DragonModel::<TestBackend>::new(source_config, &device);
    let source_mamba = source.mamba.as_ref().expect("source mamba").mamba3();
    let source_in_proj = tensor_values(source_mamba.in_proj_tensor());
    let source_dt_bias = tensor_values(source_mamba.dt_bias_tensor());
    let source_out_proj = tensor_values(source_mamba.out_proj_tensor());

    let (widened, report) = source
        .widen_latent_total(target_config.clone(), &device)
        .expect("widen");
    assert_eq!(report.new_latent_total, 64);
    assert_shared_lowrank_prefix_preserved(&source, &widened);
    let widened_mamba = widened.mamba.as_ref().expect("widened mamba").mamba3();
    assert_eq!(
        source_in_proj,
        tensor_values(widened_mamba.in_proj_tensor())
    );
    assert_eq!(
        source_dt_bias,
        tensor_values(widened_mamba.dt_bias_tensor())
    );
    assert_eq!(
        source_out_proj,
        tensor_values(widened_mamba.out_proj_tensor())
    );
    assert_widened_forward_matches_source(&source, &widened, 1.0e-5);
    assert_widened_record_round_trip_matches_source(&source, &widened, target_config, 1.0e-5);
    assert_widened_forward_is_finite(&widened);
}

#[test]
fn widen_latent_total_supports_gdn2_adapter_and_preserves_latent_prefix() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 31);
    let source_config = tiny_scaling_source_config(SequenceKernelConfig::reference(
        SequenceMemorySystem::GatedDeltaNet2,
    ));
    let target_config = DragonConfig {
        mlp_internal_dim_multiplier: 4,
        ..source_config.clone()
    };
    let source = DragonModel::<TestBackend>::new(source_config, &device);
    let source_gdn2 = source.gated_deltanet2.as_ref().expect("source gdn2");
    let source_key = tensor_values(source_gdn2.key_proj_tensor());

    let (widened, report) = source
        .widen_latent_total(target_config.clone(), &device)
        .expect("widen");
    assert_eq!(report.new_latent_total, 64);
    assert_shared_lowrank_prefix_preserved(&source, &widened);
    let widened_key_prefix = widened
        .gated_deltanet2
        .as_ref()
        .expect("widened gdn2")
        .key_proj_tensor()
        .slice([0..source.n_head, 0..source.n_embd, 0..16]);
    assert_eq!(source_key, tensor_values(widened_key_prefix));
    assert_widened_forward_matches_source(&source, &widened, 5.0e-4);
    assert_widened_record_round_trip_matches_source(&source, &widened, target_config, 5.0e-4);
    assert_widened_forward_is_finite(&widened);
}

#[test]
fn widen_latent_total_supports_upstream_gdn2_and_preserves_headed_prefix() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let source_config = DragonConfig {
        sequence_kernel: SequenceKernelConfig::gated_delta_chunk_wy(),
        gated_deltanet2: super::super::sequence::gdn2::GatedDeltaNet2Config {
            implementation: GatedDeltaNet2Implementation::UpstreamFull,
            chunk_size: 4,
            ..Default::default()
        },
        ..tiny_scaling_source_config(SequenceKernelConfig::gated_delta_chunk_wy())
    };
    let target_config = DragonConfig {
        mlp_internal_dim_multiplier: 4,
        ..source_config.clone()
    };
    let source = DragonModel::<TestBackend>::new(source_config, &device);
    let source_upstream = source
        .gated_deltanet2_upstream
        .as_ref()
        .expect("source upstream gdn2");

    let (widened, report) = source
        .widen_latent_total(target_config.clone(), &device)
        .expect("widen");
    assert_eq!(report.new_latent_total, 64);
    assert_shared_lowrank_prefix_preserved(&source, &widened);
    let widened_upstream = widened
        .gated_deltanet2_upstream
        .as_ref()
        .expect("widened upstream gdn2");
    for head in 0..source.n_head {
        let source_start = head * 16;
        let widened_start = head * 32;
        assert_eq!(
            tensor_values(
                source_upstream
                    .query
                    .weight
                    .val()
                    .slice([0..source.n_embd, source_start..source_start + 16])
            ),
            tensor_values(
                widened_upstream
                    .query
                    .weight
                    .val()
                    .slice([0..source.n_embd, widened_start..widened_start + 16])
            )
        );
    }
    assert_widened_forward_matches_source(&source, &widened, 1.0e-4);
    assert_widened_record_round_trip_matches_source(&source, &widened, target_config, 1.0e-4);
    assert_widened_forward_is_finite(&widened);
}

#[test]
fn tiny_upstream_gated_deltanet2_model_constructs_and_runs_forward() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let config = DragonConfig {
        n_layer: 1,
        n_embd: 16,
        n_head: 2,
        mlp_internal_dim_multiplier: 2,
        vocab_size: 32,
        dropout: 0.0,
        sequence_kernel: SequenceKernelConfig::gated_delta_chunk_wy(),
        gated_deltanet2: super::super::sequence::gdn2::GatedDeltaNet2Config {
            implementation: GatedDeltaNet2Implementation::UpstreamFull,
            chunk_size: 4,
            ..Default::default()
        },
        ..Default::default()
    };
    let model = DragonModel::<TestBackend>::new(config, &device);
    let tokens = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![1_i64, 2, 3], [1, 3]),
        &device,
    );
    let logits = model.forward(tokens);
    assert_eq!(logits.shape().dims(), [1, 3, 32]);
    let values = logits
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("logits");
    assert!(values.iter().all(|value| value.is_finite()));
}
