use burn::tensor::{Tensor, TensorData};
use burn_dragon_core::gated_deltanet2_reference;
use burn_ndarray::NdArray;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

type BenchBackend = NdArray<f32>;

fn tensor4(shape: [usize; 4], stride: usize, modulus: usize) -> Tensor<BenchBackend, 4> {
    let len = shape.iter().product::<usize>();
    let values = (0..len)
        .map(|index| ((index * stride) % modulus) as f32 / modulus.max(1) as f32)
        .collect::<Vec<_>>();
    Tensor::<BenchBackend, 4>::from_data(TensorData::new(values, shape), &Default::default())
}

fn bench_reference(c: &mut Criterion) {
    let mut group = c.benchmark_group("gdn2_sequence_reference");
    let full_nca_shape = std::env::var("BURN_DRAGON_GDN2_BENCH_FULL").is_ok();
    let (batch, heads, latent, dense, times): (usize, usize, usize, usize, &[usize]) =
        if full_nca_shape {
            (6, 8, 128, 512, &[64, 128, 512])
        } else {
            (1, 4, 32, 128, &[16, 32, 64])
        };
    for &time in times {
        let query = tensor4([batch, heads, time, latent], 3, 251);
        let key = tensor4([batch, heads, time, latent], 5, 257);
        let value = tensor4([batch, 1, time, dense], 7, 263);
        let erase =
            Tensor::<BenchBackend, 4>::ones([batch, heads, time, latent], &Default::default())
                .mul_scalar(0.5);
        let write =
            Tensor::<BenchBackend, 4>::ones([batch, heads, time, dense], &Default::default());
        let log_decay =
            Tensor::<BenchBackend, 4>::zeros([batch, heads, time, latent], &Default::default());
        group.bench_with_input(BenchmarkId::from_parameter(time), &time, |b, _| {
            b.iter(|| {
                let _ = gated_deltanet2_reference(
                    query.clone(),
                    key.clone(),
                    value.clone(),
                    erase.clone(),
                    write.clone(),
                    log_decay.clone(),
                    None,
                    true,
                    1.0e-6,
                );
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_reference);
criterion_main!(benches);
