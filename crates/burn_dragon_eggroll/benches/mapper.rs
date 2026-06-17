use burn::module::{Module, Param};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use burn_dragon_eggroll::{
    AntitheticFitness, EggrollModuleOptimizerState, apply_antithetic_update,
    apply_antithetic_update_with_allowed_param_ids, perturb_module,
    perturb_module_with_allowed_param_ids,
};
use burn_eggroll::{AntitheticSign, EggrollConfig, PopulationConfig};
use burn_ndarray::NdArray;
use criterion::{Criterion, criterion_group, criterion_main};
use std::collections::BTreeSet;
use std::hint::black_box;

type BenchBackend = NdArray<f32>;

#[derive(Module, Debug)]
struct BenchModule<B: Backend> {
    projection: Param<Tensor<B, 2>>,
    headed: Param<Tensor<B, 3>>,
    scale: Param<Tensor<B, 1>>,
}

fn device() -> burn::tensor::Device<BenchBackend> {
    Default::default()
}

fn bench_module() -> BenchModule<BenchBackend> {
    let device = device();
    BenchModule {
        projection: Param::from_data(TensorData::new(vec![0.01; 256 * 256], [256, 256]), &device),
        headed: Param::from_data(
            TensorData::new(vec![0.01; 8 * 128 * 128], [8, 128, 128]),
            &device,
        ),
        scale: Param::from_data(TensorData::new(vec![1.0; 4096], [4096]), &device),
    }
}

fn config() -> EggrollConfig {
    EggrollConfig {
        sigma: 2.5e-3,
        population: PopulationConfig {
            population_size: 8,
            population_chunk_size: 8,
            rank: 2,
            seed: 19,
            matrix_noise: Default::default(),
        },
        ..EggrollConfig::default()
    }
}

fn bench_mapper(c: &mut Criterion) {
    let config = config();
    let module = bench_module();
    let allowed = BTreeSet::from([module.projection.id.val(), module.headed.id.val()]);
    c.bench_function("dragon_eggroll/perturb_module", |b| {
        b.iter(|| {
            perturb_module::<BenchBackend, _>(
                black_box(module.clone()),
                black_box(&config),
                7,
                3,
                AntitheticSign::Plus,
            )
        })
    });
    c.bench_function("dragon_eggroll/perturb_module_scoped", |b| {
        b.iter(|| {
            perturb_module_with_allowed_param_ids::<BenchBackend, _>(
                black_box(module.clone()),
                black_box(&config),
                7,
                3,
                AntitheticSign::Plus,
                Some(black_box(&allowed)),
            )
        })
    });

    let fitness = [
        AntitheticFitness {
            pair_index: 0,
            plus: -1.0,
            minus: -1.1,
        },
        AntitheticFitness {
            pair_index: 1,
            plus: -0.9,
            minus: -1.0,
        },
        AntitheticFitness {
            pair_index: 2,
            plus: -0.8,
            minus: -0.9,
        },
        AntitheticFitness {
            pair_index: 3,
            plus: -0.7,
            minus: -0.8,
        },
    ];
    c.bench_function("dragon_eggroll/apply_antithetic_update", |b| {
        b.iter_batched(
            || {
                (
                    module.clone(),
                    EggrollModuleOptimizerState::<BenchBackend>::new(),
                )
            },
            |(module, mut state)| {
                apply_antithetic_update(
                    black_box(module),
                    black_box(&config),
                    7,
                    black_box(&fitness),
                    black_box(&mut state),
                )
                .expect("eggroll update")
            },
            criterion::BatchSize::SmallInput,
        )
    });
    c.bench_function("dragon_eggroll/apply_antithetic_update_scoped", |b| {
        b.iter_batched(
            || {
                (
                    module.clone(),
                    EggrollModuleOptimizerState::<BenchBackend>::new(),
                )
            },
            |(module, mut state)| {
                apply_antithetic_update_with_allowed_param_ids(
                    black_box(module),
                    black_box(&config),
                    7,
                    black_box(&fitness),
                    black_box(&mut state),
                    Some(black_box(&allowed)),
                )
                .expect("eggroll update")
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, bench_mapper);
criterion_main!(benches);
