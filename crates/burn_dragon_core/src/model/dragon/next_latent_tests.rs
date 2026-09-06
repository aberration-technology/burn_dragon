use super::*;
use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;

type B = Autodiff<NdArray<f32>>;

#[test]
fn next_latent_frozen_decoder_preserves_values_and_input_vjp() {
    let device = Default::default();
    for tied in [false, true] {
        let mut config = DragonConfig {
            n_layer: 1,
            n_embd: 8,
            n_head: 1,
            mlp_internal_dim_multiplier: 1,
            vocab_size: 16,
            dropout: 0.0,
            tie_input_output_embeddings: tied,
            ..Default::default()
        };
        config.latent_reasoning.enabled = true;
        config.latent_reasoning.step_conditioned_decoder = true;
        config.latent_reasoning.step_conditioned_decoder_scale = 1.0;
        let model = DragonModel::<B>::new(config, &device);
        let hidden = Tensor::<B, 3>::ones([1, 2, 8], &device).require_grad();
        let regular = model.logits_from_hidden(hidden.clone());
        let frozen = model.logits_from_hidden_with_frozen_head(hidden.clone());
        let diff = (regular.clone() - frozen.clone()).abs().max().into_scalar();
        assert!(diff < 1e-6, "decoder values differ: {diff}");
        let grads = frozen.sum().backward();
        let frozen_input_grad = hidden.grad(&grads).expect("input derivative");
        assert!(model.embed.weight.val().grad(&grads).is_none());
        if let Some(head) = &model.lm_head {
            assert!(head.val().grad(&grads).is_none());
        }
        assert!(
            model
                .latent_step_decoder_embedding
                .as_ref()
                .unwrap()
                .val()
                .grad(&grads)
                .is_none()
        );
        let regular_grads = regular.sum().backward();
        let regular_input_grad = hidden.grad(&regular_grads).unwrap();
        let diff = (frozen_input_grad - regular_input_grad)
            .abs()
            .max()
            .into_scalar();
        assert!(diff < 1e-6, "input VJP differs: {diff}");
        assert!(
            model
                .latent_step_decoder_embedding
                .as_ref()
                .unwrap()
                .val()
                .grad(&regular_grads)
                .is_some()
        );
    }
}

#[test]
fn next_latent_tied_decoder_still_allows_action_embedding_gradients() {
    let device = Default::default();
    let mut config = DragonConfig {
        n_layer: 1,
        n_embd: 8,
        n_head: 1,
        mlp_internal_dim_multiplier: 1,
        vocab_size: 16,
        dropout: 0.0,
        tie_input_output_embeddings: true,
        ..Default::default()
    };
    config.next_latent_transition.enabled = true;
    config.next_latent_transition.zero_init_output = false;
    let model = DragonModel::<B>::new(config, &device);
    let hidden = Tensor::<B, 3>::ones([1, 1, 8], &device).require_grad();
    let token = Tensor::<B, 2, Int>::from_data([[2]], &device);
    let predicted = model
        .next_latent_prediction_from_hidden_action(hidden.clone(), model.embed_tokens(token))
        .unwrap();
    let grads = model
        .logits_from_hidden_with_frozen_head(predicted)
        .powf_scalar(2.0)
        .mean()
        .backward();
    assert!(hidden.grad(&grads).is_some());
    let grad = model
        .embed
        .weight
        .val()
        .grad(&grads)
        .expect("action derivative");
    assert!(grad.abs().sum().into_scalar() > 0.0);
    assert!(
        model
            .next_latent_transition_out
            .as_ref()
            .unwrap()
            .weight
            .val()
            .grad(&grads)
            .is_some()
    );
}
