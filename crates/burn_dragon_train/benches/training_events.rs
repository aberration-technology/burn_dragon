use burn_dragon_train::train::events::{
    TrainingAppBuilder, TrainingAppConfig, TrainingMetricSample, TrainingMetricSplit,
    TrainingRunContext,
};
use burn_dragon_train::{TrainingEventsConfig, TrainingGatesConfig};
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn event_runtime_metric_step(c: &mut Criterion) {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let mut events = TrainingEventsConfig::default();
    events.flush_every_steps = usize::MAX;
    let mut runtime = TrainingAppBuilder::new(TrainingAppConfig {
        run: TrainingRunContext::new("bench", "bench", tempdir.path(), 1024),
        events,
        gates: TrainingGatesConfig::default(),
        bus: Default::default(),
    })
    .build()
    .expect("event runtime");
    let mut step = 0usize;

    c.bench_function("training_event_metric_step", |b| {
        b.iter(|| {
            let current_step = step;
            step = step.wrapping_add(1);
            runtime.write_metric_sample(black_box(TrainingMetricSample {
                run_id: "bench".to_string(),
                split: TrainingMetricSplit::Train,
                epoch: 1 + current_step / 1024,
                step_in_epoch: 1 + current_step % 1024,
                absolute_step: current_step,
                name: "Loss".to_string(),
                value: 1.0,
                running_value: 1.0,
            }));
            runtime.update();
        });
    });
}

criterion_group!(benches, event_runtime_metric_step);
criterion_main!(benches);
