use burn::tensor::{Int, Tensor, TensorData};
use burn_dragon_language::SelfDistillationKlKind;
use burn_dragon_language::loss::language_model_loss;
use burn_dragon_language::train::self_distillation_loss_from_logits;
use burn_ndarray::NdArray;
use criterion::Criterion;
use std::hint::black_box;

type BenchBackend = NdArray<f32>;

fn device() -> burn::tensor::Device<BenchBackend> {
    Default::default()
}

fn logits(batch: usize, time: usize, vocab: usize, offset: f32) -> Tensor<BenchBackend, 3> {
    let count = batch * time * vocab;
    let values = (0..count)
        .map(|idx| ((idx % vocab) as f32 * 0.001) + offset)
        .collect::<Vec<_>>();
    Tensor::<BenchBackend, 3>::from_data(TensorData::new(values, [batch, time, vocab]), &device())
}

fn targets(batch: usize, time: usize, vocab: usize) -> Tensor<BenchBackend, 2, Int> {
    let values = (0..batch * time)
        .map(|idx| (idx % vocab) as i64)
        .collect::<Vec<_>>();
    Tensor::<BenchBackend, 2, Int>::from_data(TensorData::new(values, [batch, time]), &device())
}

fn criterion_benchmark(c: &mut Criterion) {
    let batch = 8;
    let time = 64;
    let vocab = 256;
    let student_logits = logits(batch, time, vocab, 0.0);
    let teacher_logits = logits(batch, time, vocab, 0.01);
    let targets = targets(batch, time, vocab);

    c.bench_function("next_token_ce_flat_logits", |b| {
        b.iter(|| {
            black_box(language_model_loss::<BenchBackend>(
                student_logits.clone(),
                targets.clone(),
            ))
            .to_data()
        })
    });

    c.bench_function("sdft_forward_kl_flat_logits", |b| {
        b.iter(|| {
            black_box(self_distillation_loss_from_logits::<BenchBackend>(
                student_logits.clone(),
                teacher_logits.clone(),
                None,
                SelfDistillationKlKind::Forward,
            ))
            .to_data()
        })
    });

    c.bench_function("sdpo_js_distillation_flat_logits", |b| {
        b.iter(|| {
            black_box(self_distillation_loss_from_logits::<BenchBackend>(
                student_logits.clone(),
                teacher_logits.clone(),
                None,
                SelfDistillationKlKind::JensenShannon,
            ))
            .to_data()
        })
    });
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
    criterion_benchmark(&mut criterion);
    criterion.final_summary();
}
