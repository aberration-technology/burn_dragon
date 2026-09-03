#[cfg(feature = "train")]
use std::fs;
#[cfg(feature = "train")]
use std::path::{Path, PathBuf};

#[cfg(feature = "train")]
use anyhow::{Context, Result, anyhow};
#[cfg(feature = "train")]
use burn::tensor::backend::Backend;
#[cfg(feature = "train")]
use burn_dragon_language::train::{
    LanguageTrainModel, RuliadEvaluationSuiteOptions, evaluate_ruliad_model_suite,
    latest_model_checkpoint_epoch, prepare_dataset,
};
#[cfg(feature = "train")]
use burn_dragon_language::{
    load_language_core_from_checkpoint, load_training_config_for_checkpoint,
};
#[cfg(feature = "train")]
use burn_ndarray::NdArray;
#[cfg(feature = "train")]
use serde::Serialize;

#[cfg(feature = "train")]
#[derive(Debug)]
struct Args {
    backend: String,
    checkpoint: PathBuf,
    epoch: Option<usize>,
    output: Option<PathBuf>,
    free_run_items: usize,
    policy_items: usize,
    difficulty_levels: usize,
    batch_size: Option<usize>,
    include_closed_loop_rollout: bool,
}

#[cfg(feature = "train")]
fn parse_usize(args: &mut impl Iterator<Item = String>, name: &str) -> Result<usize> {
    args.next()
        .ok_or_else(|| anyhow!("{name} requires a value"))?
        .parse::<usize>()
        .with_context(|| format!("{name} requires a non-negative integer"))
}

#[cfg(feature = "train")]
fn parse_args() -> Result<Args> {
    let mut backend = "cpu".to_string();
    let mut checkpoint = None;
    let mut epoch = None;
    let mut output = None;
    let mut free_run_items = 32;
    let mut policy_items = 32;
    let mut difficulty_levels = 4;
    let mut batch_size = None;
    let mut include_closed_loop_rollout = true;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--backend" => {
                backend = args
                    .next()
                    .ok_or_else(|| anyhow!("--backend requires a value"))?;
            }
            "--checkpoint" => {
                checkpoint = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| anyhow!("--checkpoint requires a path"))?,
                ));
            }
            "--epoch" => epoch = Some(parse_usize(&mut args, "--epoch")?),
            "--output" => {
                output = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| anyhow!("--output requires a path"))?,
                ));
            }
            "--free-run-items" => free_run_items = parse_usize(&mut args, "--free-run-items")?,
            "--policy-items" => policy_items = parse_usize(&mut args, "--policy-items")?,
            "--difficulty-levels" => {
                difficulty_levels = parse_usize(&mut args, "--difficulty-levels")?
            }
            "--batch-size" => batch_size = Some(parse_usize(&mut args, "--batch-size")?),
            "--no-closed-loop-rollout" => include_closed_loop_rollout = false,
            "--help" | "-h" => {
                println!(
                    "usage: evaluate_ruliad_checkpoint --backend <cpu|cuda> --checkpoint <run-or-checkpoint-dir> [--epoch N] [--output report.json] [--free-run-items N] [--policy-items N] [--difficulty-levels N] [--batch-size N] [--no-closed-loop-rollout]"
                );
                std::process::exit(0);
            }
            other => return Err(anyhow!("unknown argument {other}")),
        }
    }
    Ok(Args {
        backend,
        checkpoint: checkpoint.ok_or_else(|| anyhow!("--checkpoint is required"))?,
        epoch,
        output,
        free_run_items,
        policy_items,
        difficulty_levels,
        batch_size,
        include_closed_loop_rollout,
    })
}

#[cfg(feature = "train")]
fn checkpoint_dir(path: &Path) -> PathBuf {
    if path.join("checkpoint").is_dir() {
        path.join("checkpoint")
    } else {
        path.to_path_buf()
    }
}

#[cfg(feature = "train")]
#[derive(Serialize)]
struct CheckpointEvaluationDocument {
    version: u32,
    backend: String,
    checkpoint: PathBuf,
    checkpoint_epoch: usize,
    git_commit: Option<String>,
    evaluation: burn_dragon_language::train::RuliadEvaluationSuiteReport,
}

#[cfg(feature = "train")]
fn evaluate<B>(args: &Args) -> Result<CheckpointEvaluationDocument>
where
    B: Backend + Clone + 'static,
    B::Device: Default + Clone,
{
    let checkpoint = checkpoint_dir(&args.checkpoint);
    let checkpoint_epoch = match args.epoch {
        Some(epoch) => epoch,
        None => latest_model_checkpoint_epoch(&checkpoint)?,
    };
    let config =
        load_training_config_for_checkpoint(&[], Some(&checkpoint), args.backend.as_str())?;
    let dataset = prepare_dataset(&config.dataset, &config.training)?;
    let device = B::Device::default();
    B::seed(&device, config.training.seed);
    let model = load_language_core_from_checkpoint::<B>(
        &checkpoint,
        Some(checkpoint_epoch),
        &[],
        args.backend.as_str(),
        &device,
    )?;
    let model = LanguageTrainModel::new(model)
        .with_training_objectives(&config.training)
        .with_tbptt_chunk_size(config.training.tbptt_chunk_size)
        .with_tbptt_credit_window_chunks(config.training.tbptt_credit_window_chunks)
        .with_tbptt_persist_across_steps(config.training.tbptt_persist_across_steps);
    let options = RuliadEvaluationSuiteOptions {
        panel_seed: config.training.validation.seed,
        free_run_items: args.free_run_items,
        policy_items: args.policy_items,
        difficulty_levels: args.difficulty_levels,
        training_batch_size: args.batch_size.unwrap_or(config.training.batch_size).max(1),
        include_closed_loop_rollout: args.include_closed_loop_rollout,
        epoch: checkpoint_epoch,
        absolute_step: checkpoint_epoch.saturating_mul(config.training.checkpoint_interval_iters),
        dataset_name: "ruliad_checkpoint_evaluation".to_string(),
    };
    let evaluation = evaluate_ruliad_model_suite(
        dataset.as_ref(),
        &model,
        &config.training,
        &options,
        &device,
    )?;
    Ok(CheckpointEvaluationDocument {
        version: 1,
        backend: args.backend.clone(),
        checkpoint,
        checkpoint_epoch,
        git_commit: option_env!("BURN_DRAGON_GIT_COMMIT").map(str::to_string),
        evaluation,
    })
}

#[cfg(feature = "train")]
fn main() -> Result<()> {
    let args = parse_args()?;
    let report = match args.backend.as_str() {
        "cpu" => evaluate::<NdArray<f32>>(&args)?,
        #[cfg(feature = "cuda")]
        "cuda" => evaluate::<burn_cuda::Cuda<f32>>(&args)?,
        #[cfg(not(feature = "cuda"))]
        "cuda" => return Err(anyhow!("the example was built without the cuda feature")),
        other => return Err(anyhow!("unsupported backend {other}")),
    };
    let json = serde_json::to_string_pretty(&report).context("serialize evaluation report")?;
    if let Some(path) = args.output.as_ref() {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create output directory {}", parent.display()))?;
        }
        fs::write(path, format!("{json}\n"))
            .with_context(|| format!("write {}", path.display()))?;
    } else {
        println!("{json}");
    }
    Ok(())
}

#[cfg(not(feature = "train"))]
fn main() {
    eprintln!("the evaluate_ruliad_checkpoint example requires the train feature");
    std::process::exit(2);
}
