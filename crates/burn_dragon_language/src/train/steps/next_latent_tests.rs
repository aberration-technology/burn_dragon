use super::*;
use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;

type B = Autodiff<NdArray<f32>>;

fn model() -> LanguageTrainModel<B> {
    let mut config = DragonConfig {
        n_layer: 1,
        n_embd: 8,
        n_head: 1,
        mlp_internal_dim_multiplier: 1,
        vocab_size: 16,
        dropout: 0.0,
        ..Default::default()
    };
    config.next_latent_transition.enabled = true;
    let mut model = LanguageTrainModel::new(DragonModel::<B>::new(config, &Default::default()))
        .with_latent_reasoning(LatentReasoningTrainingConfig {
            enabled: true,
            every_steps: 1,
            jepa_future_offsets: vec![],
            next_latent: NextLatentPredictionConfig {
                enabled: true,
                horizon: 2,
                ..Default::default()
            },
            sigreg: LatentReasoningSigRegConfig {
                enabled: false,
                ..Default::default()
            },
            ..Default::default()
        });
    model.next_latent_token_layout = Some(crate::train::next_latent::NextLatentTokenLayout {
        bos: Some(14),
        eos: Some(9),
        pad: Some(15),
    });
    model
}

fn hidden() -> Tensor<B, 3> {
    Tensor::from_data(
        TensorData::new((0..6).flat_map(|t| [t as f32; 8]).collect(), [1, 6, 8]),
        &Default::default(),
    )
}

fn tokens() -> Tensor<B, 2, Int> {
    Tensor::from_data([[1, 2, 9, 3, 4, 15]], &Default::default())
}

#[test]
fn next_latent_credit_window_matches_manual_recurrent_objective() {
    let model = model()
        .with_tbptt_chunk_size(Some(3))
        .with_tbptt_credit_window_chunks(2);
    let inputs = Tensor::<B, 2, Int>::from_data([[1, 2, 3, 4, 5, 6]], &Default::default());
    let targets = Tensor::<B, 2, Int>::from_data([[2, 3, 4, 5, 6, 7]], &Default::default());
    let mut state = model.model.init_state();
    let mut losses = Vec::new();
    for start in [0, 3] {
        let chunk = inputs.clone().slice([0..1, start..start + 3]);
        let hidden = model
            .model
            .forward_hidden_with_state(chunk.clone(), &mut state);
        losses.push(
            model
                .next_token_loss_parts_from_hidden(
                    hidden,
                    targets.clone().slice([0..1, start..start + 3]),
                    chunk,
                    None,
                    None,
                )
                .total()
                .mul_scalar(0.5),
        );
    }
    let expected = losses.remove(0) + losses.remove(0);
    let expected_value = expected.clone().into_scalar();
    let expected_grads = GradientsParams::from_grads(expected.backward(), &model);
    let output = burn_train::TrainStep::step(
        &model,
        SequenceBatch::new(inputs, targets, None).with_reset_stream_state(true),
    );
    let item = output.item.sync();
    let loss: LossValue<NdArray<f32>> = item.adapt();
    assert!((loss.value().into_scalar() - expected_value).abs() < 1e-5);
    let ids = model.model.predictive_coding_parameter_ids().unwrap();
    for id in [ids.embedding, ids.lm_head] {
        let expected = expected_grads.get::<NdArray<f32>, 2>(id).unwrap();
        let actual = output.grads.get::<NdArray<f32>, 2>(id).unwrap();
        assert!((expected - actual).abs().max().into_scalar() < 1e-5);
    }
}

#[test]
fn next_latent_experiment_profiles_preserve_the_matched_backbone_and_token_budget() {
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../config/language/experiments/next_latent");
    let base = crate::load_training_config(&[directory.join("credit-base.toml")]).unwrap();
    assert_eq!(base.training.block_size * base.training.batch_size, 8192);
    for overlay in [
        "credit4.overlay.toml",
        "h2.overlay.toml",
        "answer-balanced.overlay.toml",
    ] {
        let variant = crate::load_training_config(&[
            directory.join("credit-base.toml"),
            directory.join(overlay),
        ])
        .unwrap();
        assert_eq!(variant.model, base.model);
        assert_eq!(variant.dataset, base.dataset);
        assert_eq!(variant.training.batch_size, base.training.batch_size);
        assert_eq!(variant.training.block_size, base.training.block_size);
    }
}

#[test]
fn next_latent_masks_cross_document_horizons_and_excludes_empty_horizons() {
    let model = model();
    // Identity predictor: only 1->2 and 3->4 are eligible; no h2 pair survives.
    let (loss, _) = model.next_latent_auxiliary_loss(hidden(), hidden(), tokens(), None);
    assert!((loss.unwrap().into_scalar() - 0.5).abs() < 1e-6);
}

#[test]
fn next_latent_kl_does_not_rescale_hidden_regression() {
    let mut model = model();
    model.latent_reasoning.next_latent.token_kl_weight = 0.7;
    let both = model
        .next_latent_auxiliary_loss(hidden(), hidden(), tokens(), None)
        .0
        .unwrap()
        .into_scalar();
    model.latent_reasoning.next_latent.regression_weight = 0.0;
    let kl = model
        .next_latent_auxiliary_loss(hidden(), hidden(), tokens(), None)
        .0
        .unwrap()
        .into_scalar();
    assert!((both - kl - 0.5).abs() < 1e-5, "both={both} kl={kl}");
}

#[test]
fn next_latent_prompt_mask_only_masks_token_kl() {
    let mut model = model();
    model.latent_reasoning.next_latent.token_kl_weight = 1.0;
    let mask = Some(Tensor::zeros([1, 6], &Default::default()));
    let loss = model
        .next_latent_auxiliary_loss(hidden(), hidden(), tokens(), mask)
        .0
        .unwrap();
    assert!((loss.into_scalar() - 0.5).abs() < 1e-6);
}

#[test]
fn next_latent_target_is_detached_but_source_is_trained() {
    let model = model();
    let source = hidden().require_grad();
    let target = (hidden() + 1.0).require_grad();
    let loss = model
        .next_latent_auxiliary_loss(source.clone(), target.clone(), tokens(), None)
        .0
        .unwrap();
    let grads = loss.backward();
    assert!(target.grad(&grads).is_none());
    assert!(source.grad(&grads).unwrap().abs().sum().into_scalar() > 0.0);
}

#[test]
fn next_latent_all_padding_and_single_token_are_finite() {
    let mut model = model();
    model.latent_reasoning.next_latent.token_kl_weight = 1.0;
    let loss = model
        .next_latent_auxiliary_loss(
            hidden(),
            hidden(),
            Tensor::full([1, 6], 15, &Default::default()),
            None,
        )
        .0
        .unwrap();
    assert_eq!(loss.into_scalar(), 0.0);
    let one = Tensor::zeros([1, 1, 8], &Default::default());
    assert!(
        model
            .next_latent_auxiliary_loss(
                one.clone(),
                one,
                Tensor::zeros([1, 1], &Default::default()),
                None
            )
            .0
            .is_none()
    );
}

#[test]
fn next_latent_is_not_diluted_by_other_auxiliaries() {
    let mut model = model();
    let next = model
        .latent_reasoning_auxiliary_loss(hidden(), tokens(), None, None)
        .unwrap()
        .into_scalar();
    model.latent_reasoning.sigreg.enabled = true;
    model.latent_reasoning.sigreg.target = crate::config::LatentReasoningSigRegTarget::Hidden;
    let both = model
        .latent_reasoning_auxiliary_loss(hidden(), tokens(), None, None)
        .unwrap()
        .into_scalar();
    model.latent_reasoning.next_latent.enabled = false;
    let other = model
        .latent_reasoning_auxiliary_loss(hidden(), tokens(), None, None)
        .unwrap()
        .into_scalar();
    assert!((both - next - other).abs() < 1e-5);
}

#[test]
fn next_latent_missing_tokenizer_fails_closed() {
    let mut model = model();
    model.next_latent_token_layout = None;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        model.next_latent_auxiliary_loss(hidden(), hidden(), tokens(), None)
    }));
    assert!(result.is_err());
}
