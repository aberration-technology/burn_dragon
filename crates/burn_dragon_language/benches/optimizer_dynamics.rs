use std::hint::black_box;

use burn_dragon_language::train::optimizer_dynamics::{
    OptimizerDynamicsConfig, OptimizerDynamicsKind, OptimizerDynamicsReport, run_optimizer_dynamics,
};
use criterion::Criterion;
use tempfile::tempdir;

fn assert_report_sane(report: &OptimizerDynamicsReport, min_loss_delta: f32) {
    assert!(
        report.initial_loss.is_finite(),
        "initial loss must be finite: {report:?}"
    );
    assert!(
        report.final_loss.is_finite(),
        "final loss must be finite: {report:?}"
    );
    assert!(
        report.loss_delta() >= min_loss_delta,
        "optimizer should learn during dynamics bench: {report:?}"
    );
}

fn bench_optimizer_dynamics(c: &mut Criterion) {
    let config = OptimizerDynamicsConfig {
        epochs: 6,
        max_iters: 24,
        log_frequency: usize::MAX,
        ..OptimizerDynamicsConfig::default()
    };

    c.bench_function("optimizer_dynamics/adamw_tiny_next_token", |b| {
        b.iter(|| {
            let dir = tempdir().expect("tempdir");
            let report = run_optimizer_dynamics(OptimizerDynamicsKind::AdamW, &config, dir.path())
                .expect("adamw dynamics");
            assert_report_sane(&report, 0.40);
            black_box(report)
        })
    });

    c.bench_function(
        "optimizer_dynamics/eggroll_rank_sgd_pop8_rank2_tiny_next_token",
        |b| {
            b.iter(|| {
                let dir = tempdir().expect("tempdir");
                let report =
                    run_optimizer_dynamics(OptimizerDynamicsKind::Eggroll, &config, dir.path())
                        .expect("eggroll dynamics");
                assert_report_sane(&report, 0.10);
                black_box(report)
            })
        },
    );
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
    bench_optimizer_dynamics(&mut criterion);
    criterion.final_summary();
}
