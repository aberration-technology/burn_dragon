//! Local-only training and run monitoring commands.

use super::*;

pub(super) fn train_local(args: TrainLocalArgs) -> Result<()> {
    ensure_training_backend_runtime_accessible(args.backend)?;
    apply_local_run_env(&args)?;
    let mut config = load_training_config(&args.training_config_paths)?;
    args.training_overrides.apply_to(&mut config)?;
    eprintln!(
        "starting burn_dragon local training: backend={} configs={} max_iters={} batch_size={} block_size={}",
        args.backend.as_label(),
        args.training_config_paths
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(","),
        config.training.max_iters,
        config.training.batch_size,
        config.training.block_size,
    );
    match args.backend {
        BackendArg::Cpu => {
            if train::optimizer_uses_forward_only_eggroll(&config) {
                train_local_backend_forward_eggroll::<NdArray<f32>>(&config, "cpu")
            } else {
                train_local_backend::<Autodiff<NdArray<f32>>>(&config, "cpu")
            }
        }
        #[cfg(feature = "wgpu")]
        BackendArg::Wgpu => {
            if train::optimizer_uses_forward_only_eggroll(&config) {
                train_local_backend_forward_eggroll::<burn_wgpu::Wgpu<f32>>(&config, "wgpu")
            } else {
                train_local_backend::<Autodiff<burn_wgpu::Wgpu<f32>>>(&config, "wgpu")
            }
        }
        #[cfg(feature = "cuda")]
        BackendArg::Cuda => {
            if train::optimizer_uses_forward_only_eggroll(&config) {
                train_local_backend_forward_eggroll::<burn_cuda::Cuda<f32>>(&config, "cuda")
            } else {
                train_local_backend::<Autodiff<burn_cuda::Cuda<f32>>>(&config, "cuda")
            }
        }
        #[cfg(feature = "rocm")]
        BackendArg::Rocm => {
            if train::optimizer_uses_forward_only_eggroll(&config) {
                train_local_backend_forward_eggroll::<burn_rocm::Rocm<f32>>(&config, "rocm")
            } else {
                train_local_backend::<Autodiff<burn_rocm::Rocm<f32>>>(&config, "rocm")
            }
        }
        #[cfg(not(feature = "wgpu"))]
        BackendArg::Wgpu => bail!("this binary was built without the `wgpu` feature"),
        #[cfg(not(feature = "cuda"))]
        BackendArg::Cuda => bail!("this binary was built without the `cuda` feature"),
        #[cfg(not(feature = "rocm"))]
        BackendArg::Rocm => bail!("this binary was built without the `rocm` feature"),
    }
}

pub(super) fn monitor_run(args: MonitorRunArgs) -> Result<()> {
    let run_name = args.run_name.unwrap_or_else(|| {
        args.run_dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("monitored-run")
            .to_string()
    });
    burn_dragon_train::train::events::monitor_run(
        burn_dragon_train::train::events::MonitorRunOptions {
            run_dir: args.run_dir,
            run_name,
            follow: args.follow,
            poll_interval: Duration::from_secs(args.poll_interval_secs.max(1)),
            sinks: burn_dragon_train::TrainingEventsConfig::default(),
            gates: burn_dragon_train::TrainingGatesConfig::default(),
        },
    )
}

pub(super) fn apply_local_run_env(args: &TrainLocalArgs) -> Result<()> {
    set_process_env(
        "DragonModel_TRAINING_PROGRESS_RENDERER",
        args.progress.as_env(),
    );
    if let Some(run_root) = &args.run_root {
        set_process_env_path("BURN_DRAGON_RUN_ROOT", run_root);
    }
    match (&args.run_dir, &args.run_name) {
        (Some(run_dir), Some(run_name)) => {
            set_process_env_path("BURN_DRAGON_RUN_DIR", run_dir);
            set_process_env("BURN_DRAGON_RUN_NAME", run_name);
        }
        (None, None) => {}
        _ => bail!("--run-dir and --run-name must be provided together"),
    }
    Ok(())
}

pub(super) fn set_process_env_path(key: &str, value: &Path) {
    // SAFETY: CLI startup is single-threaded here; no other Rust threads are reading env vars yet.
    unsafe {
        env::set_var(key, value);
    }
}

pub(super) fn set_process_env(key: &str, value: &str) {
    // SAFETY: CLI startup is single-threaded here; no other Rust threads are reading env vars yet.
    unsafe {
        env::set_var(key, value);
    }
}

pub(super) fn train_local_backend<B>(config: &TrainingConfig, backend_label: &str) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone + 'static,
{
    let dataset = train::prepare_dataset(&config.dataset, &config.training)?;
    train::train_backend::<B, _>(config, dataset, backend_label, |_| {})
}

pub(super) fn train_local_backend_forward_eggroll<B>(
    config: &TrainingConfig,
    backend_label: &str,
) -> Result<()>
where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone + 'static,
{
    let dataset = train::prepare_dataset(&config.dataset, &config.training)?;
    train::train_backend_forward_eggroll::<B, _>(config, dataset, backend_label, |_| {})
}
