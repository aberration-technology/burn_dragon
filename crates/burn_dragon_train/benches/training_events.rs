use burn_dragon_train::train::events::{
    App, TrainingAppExt, TrainingEventBus, TrainingEventBusConfig, TrainingMetricSample,
    TrainingMetricSplit, TrainingPlugins, TrainingRunContext, TrainingRunOptions, TrainingRuntime,
    TrainingRuntimeThread,
};
use burn_dragon_train::{TrainingEventsConfig, TrainingGatesConfig};
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn metric_sample(step: usize) -> TrainingMetricSample {
    TrainingMetricSample {
        run_id: "bench".into(),
        split: TrainingMetricSplit::Train,
        epoch: 1 + step / 1024,
        step_in_epoch: 1 + step % 1024,
        absolute_step: step,
        name: "Loss".to_string(),
        value: 1.0,
        running_value: 1.0,
    }
}

fn no_event_metric_step(c: &mut Criterion) {
    let mut step = 0usize;
    c.bench_function("training_event/no_event_metric_step", |b| {
        b.iter(|| {
            let current_step = step;
            step = step.wrapping_add(1);
            black_box(metric_sample(current_step));
        });
    });
}

fn threaded_event_bus_metric_step(c: &mut Criterion) {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let mut events = TrainingEventsConfig::default();
    events.flush_every_steps = usize::MAX;
    let run_dir = tempdir.path().to_owned();
    let runtime = TrainingRuntimeThread::spawn(
        move || {
            let mut app = App::new();
            app.add_plugins(TrainingPlugins);
            app.try_add_training_run_with(
                TrainingRunContext::new("bench", "bench", run_dir, 1024),
                TrainingRunOptions {
                    sinks: events,
                    gates: TrainingGatesConfig::default(),
                    ..TrainingRunOptions::default()
                },
            )?;
            Ok(app)
        },
        TrainingEventBusConfig::default(),
    )
    .expect("event runtime");
    let bus: TrainingEventBus = runtime.bus();
    let mut step = 0usize;

    c.bench_function("training_event/threaded_event_bus_metric_step", |b| {
        b.iter(|| {
            let current_step = step;
            step = step.wrapping_add(1);
            bus.send_metric_sample(black_box(metric_sample(current_step)))
                .expect("send metric sample");
        });
    });
    runtime.shutdown().expect("shutdown event runtime");
}

fn event_runtime_metric_step(c: &mut Criterion) {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let mut events = TrainingEventsConfig::default();
    events.flush_every_steps = usize::MAX;
    let mut app = App::new();
    app.add_plugins(TrainingPlugins);
    app.try_add_training_run_with(
        TrainingRunContext::new("bench", "bench", tempdir.path(), 1024),
        TrainingRunOptions {
            sinks: events,
            gates: TrainingGatesConfig::default(),
            ..TrainingRunOptions::default()
        },
    )
    .expect("training run");
    let mut runtime = TrainingRuntime::new(app);
    let mut step = 0usize;

    c.bench_function("training_event_metric_step", |b| {
        b.iter(|| {
            let current_step = step;
            step = step.wrapping_add(1);
            runtime.write_message(black_box(metric_sample(current_step)));
            runtime.update();
        });
    });
}

criterion_group!(
    benches,
    no_event_metric_step,
    threaded_event_bus_metric_step,
    event_runtime_metric_step
);
criterion_main!(benches);
