use std::hint::black_box;

use burn::tensor::{Int, Tensor, TensorData};
#[cfg(feature = "cuda")]
use burn_cuda::Cuda;
use burn_dragon_core::{
    DragonConfig, DragonModel, SharedLowrankPopulationFactors, SharedLowrankPopulationWeights,
    SharedLowrankWeights,
};
#[cfg(not(feature = "cuda"))]
use burn_ndarray::NdArray;
use criterion::Criterion;

#[cfg(feature = "cuda")]
type BenchBackend = Cuda<f32, i32>;
#[cfg(not(feature = "cuda"))]
type BenchBackend = NdArray<f32>;

fn device() -> burn::tensor::Device<BenchBackend> {
    Default::default()
}

fn model_config() -> DragonConfig {
    DragonConfig {
        n_layer: 2,
        n_embd: 64,
        n_head: 4,
        mlp_internal_dim_multiplier: 2,
        dropout: 0.0,
        vocab_size: 128,
        ..DragonConfig::default()
    }
}

fn tokens(batch: usize, time: usize, vocab: usize) -> Tensor<BenchBackend, 2, Int> {
    let values = (0..batch * time)
        .map(|idx| (idx % vocab) as i64)
        .collect::<Vec<_>>();
    Tensor::<BenchBackend, 2, Int>::from_data(TensorData::new(values, [batch, time]), &device())
}

fn population_weights(
    model: &DragonModel<BenchBackend>,
    population: usize,
) -> SharedLowrankPopulationWeights<BenchBackend> {
    let base = model.shared_lowrank_weights();
    let [heads, embd, latent] = base.encoder.shape().dims::<3>();
    let [decoder_rows, decoder_cols] = base.decoder.shape().dims::<2>();
    SharedLowrankPopulationWeights {
        encoder: Tensor::cat(
            (0..population)
                .map(|idx| {
                    base.encoder
                        .clone()
                        .add_scalar(idx as f64 * 1.0e-4)
                        .reshape([1, heads, embd, latent])
                })
                .collect(),
            0,
        ),
        encoder_v: Tensor::cat(
            (0..population)
                .map(|idx| {
                    base.encoder_v
                        .clone()
                        .sub_scalar(idx as f64 * 1.0e-4)
                        .reshape([1, heads, embd, latent])
                })
                .collect(),
            0,
        ),
        decoder: Tensor::cat(
            (0..population)
                .map(|idx| {
                    base.decoder
                        .clone()
                        .add_scalar(idx as f64 * 1.0e-4)
                        .reshape([1, decoder_rows, decoder_cols])
                })
                .collect(),
            0,
        ),
    }
}

fn eggroll_population_weights(
    model: &DragonModel<BenchBackend>,
    population: usize,
) -> SharedLowrankPopulationWeights<BenchBackend> {
    let base = model.shared_lowrank_weights();
    let ids = model.shared_lowrank_param_ids();
    let pair_count = population / 2;
    let generation = 7;
    let sigma = 1.0e-2;
    SharedLowrankPopulationWeights {
        encoder: burn_eggroll::perturb_matrix_3d_antithetic_population_with_mode(
            base.encoder,
            sigma,
            burn_eggroll::MatrixNoisePopulationSpec::new(
                1337,
                ids.encoder.val(),
                generation,
                0,
                pair_count,
                4,
            ),
            burn_eggroll::MatrixNoiseMode::default(),
        ),
        encoder_v: burn_eggroll::perturb_matrix_3d_antithetic_population_with_mode(
            base.encoder_v,
            sigma,
            burn_eggroll::MatrixNoisePopulationSpec::new(
                1337,
                ids.encoder_v.val(),
                generation,
                0,
                pair_count,
                4,
            ),
            burn_eggroll::MatrixNoiseMode::default(),
        ),
        decoder: burn_eggroll::perturb_matrix_2d_antithetic_population_with_mode(
            base.decoder,
            sigma,
            burn_eggroll::MatrixNoisePopulationSpec::new(
                1337,
                ids.decoder.val(),
                generation,
                0,
                pair_count,
                4,
            ),
            burn_eggroll::MatrixNoiseMode::default(),
        ),
    }
}

fn eggroll_population_factors(
    model: &DragonModel<BenchBackend>,
    population: usize,
) -> SharedLowrankPopulationFactors<BenchBackend> {
    let base = model.shared_lowrank_weights();
    let ids = model.shared_lowrank_param_ids();
    let pair_count = population / 2;
    let generation = 7;
    let sigma = 1.0e-2;
    let [heads, embd, latent_capacity] = base.encoder.shape().dims::<3>();
    let [decoder_rows, decoder_cols] = base.decoder.shape().dims::<2>();
    let device = base.encoder.device();
    let encoder = burn_eggroll::low_rank_factors_3d_antithetic_population_with_mode(
        heads,
        embd,
        latent_capacity,
        burn_eggroll::MatrixNoisePopulationSpec::new(
            1337,
            ids.encoder.val(),
            generation,
            0,
            pair_count,
            4,
        ),
        burn_eggroll::MatrixNoiseMode::default(),
        &device,
    );
    let encoder_v = burn_eggroll::low_rank_factors_3d_antithetic_population_with_mode(
        heads,
        embd,
        latent_capacity,
        burn_eggroll::MatrixNoisePopulationSpec::new(
            1337,
            ids.encoder_v.val(),
            generation,
            0,
            pair_count,
            4,
        ),
        burn_eggroll::MatrixNoiseMode::default(),
        &device,
    );
    let decoder = burn_eggroll::low_rank_factors_2d_antithetic_population_with_mode(
        decoder_rows,
        decoder_cols,
        burn_eggroll::MatrixNoisePopulationSpec::new(
            1337,
            ids.decoder.val(),
            generation,
            0,
            pair_count,
            4,
        ),
        burn_eggroll::MatrixNoiseMode::default(),
        &device,
    );

    SharedLowrankPopulationFactors {
        encoder_a: encoder.a,
        encoder_b: encoder.b,
        encoder_v_a: encoder_v.a,
        encoder_v_b: encoder_v.b,
        decoder_a: decoder.a,
        decoder_b: decoder.b,
        signs: encoder.signs,
        encoder_scale: encoder.scale,
        encoder_v_scale: encoder_v.scale,
        decoder_scale: decoder.scale,
        sigma,
    }
}

fn member_weights(
    population: &SharedLowrankPopulationWeights<BenchBackend>,
    member: usize,
    base: &SharedLowrankWeights<BenchBackend>,
) -> SharedLowrankWeights<BenchBackend> {
    SharedLowrankWeights {
        encoder: population
            .encoder
            .clone()
            .slice_dim(0, member..member + 1)
            .reshape(base.encoder.shape().dims::<3>()),
        encoder_v: population
            .encoder_v
            .clone()
            .slice_dim(0, member..member + 1)
            .reshape(base.encoder_v.shape().dims::<3>()),
        decoder: population
            .decoder
            .clone()
            .slice_dim(0, member..member + 1)
            .reshape(base.decoder.shape().dims::<2>()),
    }
}

fn bench_population_forward(c: &mut Criterion) {
    let device = device();
    let model = DragonModel::<BenchBackend>::new(model_config(), &device);
    let batch = 8;
    let time = 32;
    let tokens = tokens(batch, time, model.vocab_size());
    let baseline_population_weights = population_weights(&model, 16);
    let base = model.shared_lowrank_weights();

    c.bench_function("eggroll_population/member_loop_baseline_p16_b8_t32", |b| {
        b.iter(|| {
            let logits = (0..16)
                .map(|member| {
                    let weights = member_weights(&baseline_population_weights, member, &base);
                    model
                        .clone()
                        .with_shared_lowrank_weights(weights)
                        .forward(black_box(tokens.clone()))
                })
                .collect::<Vec<_>>();
            black_box(Tensor::cat(logits, 0).to_data())
        })
    });

    for population in [16usize, 64, 256] {
        let weights = population_weights(&model, population);
        c.bench_function(
            &format!("eggroll_population/stacked_grouped_p{population}_b8_t32"),
            |b| {
                b.iter(|| {
                    black_box(
                        model
                            .forward_with_shared_lowrank_population(
                                black_box(tokens.clone()),
                                black_box(weights.clone()),
                            )
                            .to_data(),
                    )
                })
            },
        );

        let eggroll_weights = eggroll_population_weights(&model, population);
        c.bench_function(
            &format!("eggroll_population/materialized_antithetic_p{population}_b8_t32"),
            |b| {
                b.iter(|| {
                    black_box(
                        model
                            .forward_with_shared_lowrank_population(
                                black_box(tokens.clone()),
                                black_box(eggroll_weights.clone()),
                            )
                            .to_data(),
                    )
                })
            },
        );

        let factors = eggroll_population_factors(&model, population);
        c.bench_function(
            &format!("eggroll_population/factorized_p{population}_b8_t32"),
            |b| {
                b.iter(|| {
                    black_box(
                        model
                            .forward_with_shared_lowrank_population_factors(
                                black_box(tokens.clone()),
                                black_box(factors.clone()),
                            )
                            .to_data(),
                    )
                })
            },
        );
    }

    for population in [64usize, 256] {
        c.bench_function(
            &format!("eggroll_population/build_antithetic_lowrank_p{population}"),
            |b| b.iter(|| black_box(eggroll_population_weights(&model, population))),
        );
        c.bench_function(
            &format!("eggroll_population/build_antithetic_factors_p{population}"),
            |b| b.iter(|| black_box(eggroll_population_factors(&model, population))),
        );
    }
}

fn cargo_test_invocation() -> bool {
    std::env::args_os().skip(1).any(|arg| {
        arg.to_str()
            .is_some_and(|arg| arg == "--test-threads" || arg.starts_with("--test-threads="))
    })
}

fn main() {
    if cargo_test_invocation() {
        return;
    }

    let mut criterion = Criterion::default().configure_from_args();
    bench_population_forward(&mut criterion);
    criterion.final_summary();
}
