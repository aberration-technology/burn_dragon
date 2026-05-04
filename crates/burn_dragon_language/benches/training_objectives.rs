use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Int, Tensor, TensorData};
use burn_dragon_language::SelfDistillationKlKind;
use burn_dragon_language::loss::language_model_loss;
use burn_dragon_language::train::{
    clipped_policy_loss, selected_token_log_probs, self_distillation_loss_from_logits,
};
use burn_ndarray::NdArray;
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

type BenchBackend = NdArray<f32>;

fn device() -> <BenchBackend as BackendTrait>::Device {
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

    let old_log_probs = selected_token_log_probs(
        burn_dragon_language::train::log_probs_from_logits(teacher_logits.clone()),
        targets.clone(),
    );
    let new_log_probs = selected_token_log_probs(
        burn_dragon_language::train::log_probs_from_logits(student_logits.clone()),
        targets,
    );
    let advantage = Tensor::<BenchBackend, 2>::ones([batch, time], &device());

    c.bench_function("sdpo_clipped_policy_loss", |b| {
        b.iter(|| {
            black_box(clipped_policy_loss::<BenchBackend>(
                new_log_probs.clone(),
                old_log_probs.clone(),
                advantage.clone(),
                None,
                Some(0.2),
                1.0,
            ))
            .to_data()
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
