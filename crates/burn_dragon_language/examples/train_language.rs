#[cfg(feature = "train")]
use std::path::PathBuf;
#[cfg(feature = "train")]
use std::time::Instant;

#[cfg(feature = "train")]
use anyhow::{Result, anyhow};
#[cfg(feature = "train")]
use burn_autodiff::Autodiff;
#[cfg(feature = "train")]
use burn_dragon_language::{TrainingConfig, load_training_config, train};
#[cfg(feature = "train")]
use burn_ndarray::NdArray;

#[cfg(feature = "train")]
#[derive(Debug, Default)]
struct TrainingOverrides {
    n_layer: Option<usize>,
    n_embd: Option<usize>,
    n_head: Option<usize>,
    latent_total: Option<usize>,
    block_size: Option<usize>,
    batch_size: Option<usize>,
    max_iters: Option<usize>,
    checkpoint_interval_iters: Option<usize>,
}

#[cfg(feature = "train")]
#[derive(Debug)]
struct RunArgs {
    backend: String,
    config_paths: Vec<PathBuf>,
    overrides: TrainingOverrides,
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
    let mut config_paths = Vec::new();
    let mut overrides = TrainingOverrides::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--backend" => {
                backend = args
                    .next()
                    .ok_or_else(|| anyhow!("--backend requires a value"))?;
            }
            "--config" | "--training-config" => {
                let path = args
                    .next()
                    .ok_or_else(|| anyhow!("{arg} requires a path"))?;
                config_paths.push(PathBuf::from(path));
            }
            "--n-layer" => overrides.n_layer = Some(parse_usize_arg(&mut args, "--n-layer")?),
            "--n-embd" => overrides.n_embd = Some(parse_usize_arg(&mut args, "--n-embd")?),
            "--n-head" => overrides.n_head = Some(parse_usize_arg(&mut args, "--n-head")?),
            "--latent-total" => {
                overrides.latent_total = Some(parse_usize_arg(&mut args, "--latent-total")?)
            }
            "--block-size" => {
                overrides.block_size = Some(parse_usize_arg(&mut args, "--block-size")?)
            }
            "--batch-size" => {
                overrides.batch_size = Some(parse_usize_arg(&mut args, "--batch-size")?)
            }
            "--max-iters" => overrides.max_iters = Some(parse_usize_arg(&mut args, "--max-iters")?),
            "--checkpoint-interval-iters" => {
                overrides.checkpoint_interval_iters =
                    Some(parse_usize_arg(&mut args, "--checkpoint-interval-iters")?)
            }
            "--help" | "-h" => {
                println!(
                    "usage: cargo run -p burn_dragon_language --example train_language --features train[,cuda] -- --backend <cpu|cuda> --config <path> [--config <path>...] [--n-layer N] [--n-embd N] [--n-head N] [--latent-total N] [--block-size N] [--batch-size N] [--max-iters N] [--checkpoint-interval-iters N]"
                );
                std::process::exit(0);
            }
            value if value.starts_with('-') => {
                return Err(anyhow!("unknown argument {value}"));
            }
            value => config_paths.push(PathBuf::from(value)),
        }
    }
    if config_paths.is_empty() {
        return Err(anyhow!("at least one --config path is required"));
    }
    Ok(RunArgs {
        backend,
        config_paths,
        overrides,
    })
}

#[cfg(feature = "train")]
fn apply_overrides(config: &mut TrainingConfig, overrides: &TrainingOverrides) -> Result<()> {
    if let Some(n_layer) = overrides.n_layer {
        config.model.n_layer = Some(n_layer);
    }
    if let Some(n_embd) = overrides.n_embd {
        config.model.n_embd = Some(n_embd);
    }
    if let Some(n_head) = overrides.n_head {
        config.model.n_head = Some(n_head);
    }
    if let Some(latent_total) = overrides.latent_total {
        config.model.latent_total = Some(latent_total);
        if let Some(n_embd) = overrides.n_embd.or(config.model.n_embd) {
            if latent_total % n_embd != 0 {
                return Err(anyhow!(
                    "--latent-total must be divisible by the resolved --n-embd/model.n_embd (got latent_total={latent_total} n_embd={n_embd})"
                ));
            }
            config.model.mlp_internal_dim_multiplier = Some(latent_total / n_embd);
        }
    }
    if let Some(block_size) = overrides.block_size {
        config.training.block_size = block_size;
        config.model.block_size = Some(block_size);
    }
    if let Some(batch_size) = overrides.batch_size {
        config.training.batch_size = batch_size;
    }
    if let Some(max_iters) = overrides.max_iters {
        config.training.max_iters = max_iters;
    }
    if let Some(checkpoint_interval_iters) = overrides.checkpoint_interval_iters {
        config.training.checkpoint_interval_iters = checkpoint_interval_iters;
    }
    config.validate()?;
    Ok(())
}

#[cfg(feature = "train")]
fn load_config(config_paths: &[PathBuf], overrides: &TrainingOverrides) -> Result<TrainingConfig> {
    let mut config = load_training_config(config_paths)?;
    apply_overrides(&mut config, overrides)?;
    Ok(config)
}

#[cfg(feature = "train")]
fn train_cpu(args: &RunArgs) -> Result<()> {
    let config = load_config(&args.config_paths, &args.overrides)?;
    let dataset = train::prepare_dataset(&config.dataset, &config.training)?;
    train::train_backend::<Autodiff<NdArray<f32>>, _>(&config, dataset, "cpu", |_| {})
}

#[cfg(all(feature = "train", feature = "cuda"))]
fn train_cuda(args: &RunArgs) -> Result<()> {
    let config = load_config(&args.config_paths, &args.overrides)?;
    let dataset = train::prepare_dataset(&config.dataset, &config.training)?;
    train::train_backend::<Autodiff<burn_cuda::Cuda<f32>>, _>(&config, dataset, "cuda", |_| {})
}

#[cfg(all(feature = "train", not(feature = "cuda")))]
fn train_cuda(_args: &RunArgs) -> Result<()> {
    Err(anyhow!(
        "the train_language example was built without the cuda feature"
    ))
}

#[cfg(feature = "train")]
fn main() -> Result<()> {
    let args = parse_args()?;
    let started = Instant::now();
    eprintln!(
        "train_language start backend={} configs={} overrides={:?}",
        args.backend,
        args.config_paths
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(","),
        args.overrides
    );
    match args.backend.as_str() {
        "cpu" => train_cpu(&args)?,
        "cuda" => train_cuda(&args)?,
        other => return Err(anyhow!("unsupported backend {other}")),
    }
    eprintln!(
        "train_language complete backend={} elapsed_ms={}",
        args.backend,
        started.elapsed().as_millis()
    );
    Ok(())
}

#[cfg(not(feature = "train"))]
fn main() {
    eprintln!("the train_language example requires the train feature");
    std::process::exit(2);
}
