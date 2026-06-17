#[cfg(feature = "train")]
use std::path::PathBuf;
#[cfg(feature = "train")]
use std::time::Instant;

#[cfg(feature = "train")]
use anyhow::{Context, Result, anyhow};
#[cfg(feature = "train")]
use burn_autodiff::Autodiff;
#[cfg(feature = "train")]
use burn_dragon_language::train::{
    AdamwEggrollPipelineReport, OptimizerPipelinePhaseConfig, ScopedRunEnv,
    latest_model_checkpoint_epoch, load_adamw_eggroll_pipeline_config, load_phase_training_config,
    plan_adamw_eggroll_runs, prepare_eggroll_continuation_config, validate_adamw_warmup_config,
    write_pipeline_report,
};
#[cfg(feature = "train")]
use burn_dragon_language::{TrainingConfig, train};
#[cfg(feature = "train")]
use burn_ndarray::NdArray;

#[cfg(feature = "train")]
#[derive(Debug, Default)]
struct PhaseOverrides {
    max_iters: Option<usize>,
    checkpoint_interval_iters: Option<usize>,
}

#[cfg(feature = "train")]
#[derive(Debug)]
struct RunArgs {
    backend: String,
    pipeline_config: PathBuf,
    adamw_overrides: PhaseOverrides,
    eggroll_overrides: PhaseOverrides,
}

#[cfg(feature = "train")]
fn parse_usize_arg(args: &mut impl Iterator<Item = String>, name: &str) -> Result<usize> {
    args.next()
        .ok_or_else(|| anyhow!("{name} requires a value"))?
        .parse::<usize>()
        .map_err(|err| anyhow!("{name} requires a positive integer: {err}"))
}

#[cfg(feature = "train")]
fn parse_args() -> Result<RunArgs> {
    let mut backend = String::from("cpu");
    let mut pipeline_config = None;
    let mut adamw_overrides = PhaseOverrides::default();
    let mut eggroll_overrides = PhaseOverrides::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--backend" => {
                backend = args
                    .next()
                    .ok_or_else(|| anyhow!("--backend requires a value"))?;
            }
            "--pipeline-config" | "--config" => {
                let path = args
                    .next()
                    .ok_or_else(|| anyhow!("{arg} requires a path"))?;
                pipeline_config = Some(PathBuf::from(path));
            }
            "--adamw-max-iters" => {
                adamw_overrides.max_iters = Some(parse_usize_arg(&mut args, "--adamw-max-iters")?)
            }
            "--eggroll-max-iters" => {
                eggroll_overrides.max_iters =
                    Some(parse_usize_arg(&mut args, "--eggroll-max-iters")?)
            }
            "--adamw-checkpoint-interval-iters" => {
                adamw_overrides.checkpoint_interval_iters = Some(parse_usize_arg(
                    &mut args,
                    "--adamw-checkpoint-interval-iters",
                )?)
            }
            "--eggroll-checkpoint-interval-iters" => {
                eggroll_overrides.checkpoint_interval_iters = Some(parse_usize_arg(
                    &mut args,
                    "--eggroll-checkpoint-interval-iters",
                )?)
            }
            "--help" | "-h" => {
                println!(
                    "usage: cargo run -p burn_dragon_language --example train_language_pipeline --features train[,cuda] -- --backend <cpu|cuda> --pipeline-config <path> [--adamw-max-iters N] [--eggroll-max-iters N] [--adamw-checkpoint-interval-iters N] [--eggroll-checkpoint-interval-iters N]"
                );
                std::process::exit(0);
            }
            value if value.starts_with('-') => {
                return Err(anyhow!("unknown argument {value}"));
            }
            value => {
                if pipeline_config.is_some() {
                    return Err(anyhow!(
                        "unexpected positional argument {value}; only one pipeline config is supported"
                    ));
                }
                pipeline_config = Some(PathBuf::from(value));
            }
        }
    }
    let pipeline_config =
        pipeline_config.ok_or_else(|| anyhow!("--pipeline-config path is required"))?;
    Ok(RunArgs {
        backend,
        pipeline_config,
        adamw_overrides,
        eggroll_overrides,
    })
}

#[cfg(feature = "train")]
fn apply_cli_phase_overrides(phase: &mut OptimizerPipelinePhaseConfig, overrides: &PhaseOverrides) {
    if let Some(max_iters) = overrides.max_iters {
        phase.max_iters = Some(max_iters);
    }
    if let Some(checkpoint_interval_iters) = overrides.checkpoint_interval_iters {
        phase.checkpoint_interval_iters = Some(checkpoint_interval_iters);
    }
}

#[cfg(feature = "train")]
fn load_pipeline_for_args(
    args: &RunArgs,
) -> Result<(
    burn_dragon_language::train::AdamwEggrollPipelineConfig,
    burn_dragon_language::train::AdamwEggrollPipelineRunPlan,
)> {
    let mut pipeline = load_adamw_eggroll_pipeline_config(&args.pipeline_config)?;
    apply_cli_phase_overrides(&mut pipeline.adamw, &args.adamw_overrides);
    apply_cli_phase_overrides(&mut pipeline.eggroll, &args.eggroll_overrides);
    let plan = plan_adamw_eggroll_runs(&pipeline);
    Ok((pipeline, plan))
}

#[cfg(feature = "train")]
fn run_adamw_cpu_phase(
    config: &TrainingConfig,
    run_dir: &std::path::Path,
    run_name: &str,
) -> Result<()> {
    let dataset = train::prepare_dataset(&config.dataset, &config.training)?;
    let _run_env = ScopedRunEnv::set(run_dir, run_name)?;
    train::train_backend::<Autodiff<NdArray<f32>>, _>(config, dataset, "cpu", |_| {})
}

#[cfg(feature = "train")]
fn run_eggroll_cpu_phase(
    config: &TrainingConfig,
    run_dir: &std::path::Path,
    run_name: &str,
) -> Result<()> {
    let dataset = train::prepare_dataset(&config.dataset, &config.training)?;
    let _run_env = ScopedRunEnv::set(run_dir, run_name)?;
    train::train_backend_forward_eggroll::<NdArray<f32>, _>(config, dataset, "cpu", |_| {})
}

#[cfg(all(feature = "train", feature = "cuda"))]
fn run_adamw_cuda_phase(
    config: &TrainingConfig,
    run_dir: &std::path::Path,
    run_name: &str,
) -> Result<()> {
    let dataset = train::prepare_dataset(&config.dataset, &config.training)?;
    let _run_env = ScopedRunEnv::set(run_dir, run_name)?;
    train::train_backend::<Autodiff<burn_cuda::Cuda<f32>>, _>(config, dataset, "cuda", |_| {})
}

#[cfg(all(feature = "train", feature = "cuda"))]
fn run_eggroll_cuda_phase(
    config: &TrainingConfig,
    run_dir: &std::path::Path,
    run_name: &str,
) -> Result<()> {
    let dataset = train::prepare_dataset(&config.dataset, &config.training)?;
    let _run_env = ScopedRunEnv::set(run_dir, run_name)?;
    train::train_backend_forward_eggroll::<burn_cuda::Cuda<f32>, _>(config, dataset, "cuda", |_| {})
}

#[cfg(all(feature = "train", not(feature = "cuda")))]
fn run_adamw_cuda_phase(
    _config: &TrainingConfig,
    _run_dir: &std::path::Path,
    _run_name: &str,
) -> Result<()> {
    Err(anyhow!(
        "the train_language_pipeline example was built without the cuda feature"
    ))
}

#[cfg(all(feature = "train", not(feature = "cuda")))]
fn run_eggroll_cuda_phase(
    _config: &TrainingConfig,
    _run_dir: &std::path::Path,
    _run_name: &str,
) -> Result<()> {
    Err(anyhow!(
        "the train_language_pipeline example was built without the cuda feature"
    ))
}

#[cfg(feature = "train")]
fn run_pipeline(args: &RunArgs) -> Result<()> {
    let (pipeline, plan) = load_pipeline_for_args(args)?;
    std::fs::create_dir_all(&plan.run_root)
        .with_context(|| format!("create pipeline run root {}", plan.run_root.display()))?;

    eprintln!(
        "adamw->eggroll pipeline start backend={} adamw_run={} eggroll_run={}",
        args.backend,
        plan.adamw_run_dir.display(),
        plan.eggroll_run_dir.display()
    );

    let adamw_config = load_phase_training_config(&pipeline.adamw)?;
    validate_adamw_warmup_config(&adamw_config)?;
    match args.backend.as_str() {
        "cpu" => run_adamw_cpu_phase(&adamw_config, &plan.adamw_run_dir, &plan.adamw_run_name)?,
        "cuda" => run_adamw_cuda_phase(&adamw_config, &plan.adamw_run_dir, &plan.adamw_run_name)?,
        other => return Err(anyhow!("unsupported backend {other}")),
    }

    let checkpoint_dir = plan.adamw_checkpoint_dir();
    let checkpoint_epoch = latest_model_checkpoint_epoch(&checkpoint_dir)?;
    eprintln!(
        "adamw warmup complete checkpoint_dir={} epoch={}",
        checkpoint_dir.display(),
        checkpoint_epoch
    );

    let eggroll_config = prepare_eggroll_continuation_config(
        load_phase_training_config(&pipeline.eggroll)?,
        &checkpoint_dir,
        Some(checkpoint_epoch),
    )?;
    let report = AdamwEggrollPipelineReport {
        plan: plan.clone(),
        adamw_checkpoint_dir: checkpoint_dir,
        adamw_checkpoint_epoch: checkpoint_epoch,
    };
    let report_path = plan
        .run_root
        .join(format!("{}-pipeline-report.json", plan.run_prefix));
    write_pipeline_report(&report_path, &report)?;
    eprintln!(
        "eggroll continuation start report={}",
        report_path.display()
    );

    match args.backend.as_str() {
        "cpu" => run_eggroll_cpu_phase(
            &eggroll_config,
            &plan.eggroll_run_dir,
            &plan.eggroll_run_name,
        ),
        "cuda" => run_eggroll_cuda_phase(
            &eggroll_config,
            &plan.eggroll_run_dir,
            &plan.eggroll_run_name,
        ),
        other => Err(anyhow!("unsupported backend {other}")),
    }
}

#[cfg(feature = "train")]
fn main() -> Result<()> {
    let args = parse_args()?;
    let started = Instant::now();
    run_pipeline(&args)?;
    eprintln!(
        "train_language_pipeline complete backend={} elapsed_ms={}",
        args.backend,
        started.elapsed().as_millis()
    );
    Ok(())
}

#[cfg(not(feature = "train"))]
fn main() {
    eprintln!("the train_language_pipeline example requires the train feature");
    std::process::exit(2);
}
