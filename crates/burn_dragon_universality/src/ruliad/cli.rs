use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

use crate::ruliad::config::{
    LeanMode, RuliadCorpusConfig, RuliadSerializationConfig, RuliadTokenizationConfig,
    default_ruliad_families, load_ruliad_config,
};
use crate::ruliad::generate::generate_ruliad_corpus;
use crate::ruliad::search::{RuliadFrontierSampler, RuliadSamplerCandidate, RuliadSamplerConfig};
use crate::ruliad::verification::verify_manifest;

#[derive(Debug, Parser)]
#[command(name = "bd-ruliad")]
#[command(about = "Generate and verify burn_dragon ruliad corpora")]
pub struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Generate {
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        #[arg(short = 'n', long)]
        samples: Option<usize>,
        #[arg(long)]
        proof_tasks: Option<PathBuf>,
        #[arg(long)]
        lean_limit: Option<usize>,
    },
    Verify {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long, default_value = "off")]
        lean_mode: LeanMode,
        #[arg(long)]
        lean_project: Option<PathBuf>,
    },
    InspectSampler {
        #[arg(long, default_value_t = 128)]
        candidates: usize,
    },
}

pub fn main() -> Result<()> {
    let args = Args::parse();
    match args.command {
        Command::Generate {
            config,
            out,
            samples,
            proof_tasks,
            lean_limit,
        } => {
            let mut config = if let Some(config) = config {
                load_ruliad_config(&config)?
            } else {
                RuliadCorpusConfig {
                    output_dir: out
                        .parent()
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from(".")),
                    seed: 1337,
                    name: "ruliad-cli".to_string(),
                    train_samples: samples.unwrap_or(64),
                    validation_samples: samples.unwrap_or(64).saturating_div(4).max(1),
                    chunk_token_capacity: 1_048_576,
                    serialization: RuliadSerializationConfig::default(),
                    tokenization: RuliadTokenizationConfig::default(),
                    source_selection: crate::ruliad::config::RuliadSourceSelectionConfig::default(),
                    families: default_ruliad_families(),
                    proof_tasks: None,
                    lean_task_limit: None,
                }
            };
            config.output_dir = out
                .parent()
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("."));
            if let Some(samples) = samples {
                config.train_samples = samples;
                config.validation_samples = samples.saturating_div(4).max(1);
            }
            if let Some(proof_tasks) = proof_tasks {
                config.proof_tasks = Some(proof_tasks);
            }
            if lean_limit.is_some() {
                config.lean_task_limit = lean_limit;
            }
            let report = generate_ruliad_corpus(&config)?;
            if report.manifest_path != out {
                std::fs::copy(&report.manifest_path, &out)?;
            }
            println!(
                "{}",
                serde_json::json!({
                    "manifest_path": out,
                    "sample_records_path": report.sample_records_path,
                    "samples": report.sample_count,
                    "tokens": report.token_count,
                })
            );
        }
        Command::Verify {
            manifest,
            lean_mode,
            lean_project,
        } => {
            let report = verify_manifest(&manifest, lean_mode, lean_project.as_deref())?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            if !report.failed.is_empty() {
                std::process::exit(1);
            }
        }
        Command::InspectSampler { candidates } => {
            let candidates = (0..candidates)
                .map(|index| RuliadSamplerCandidate {
                    oracle_hash: format!("candidate-{index}"),
                    family: if index % 11 == 0 {
                        "hash_noise".to_string()
                    } else {
                        "eca".to_string()
                    },
                    task_kind: if index % 11 == 0 {
                        "hash_canary".to_string()
                    } else {
                        "multi_step_state".to_string()
                    },
                    prior: 1.0,
                    cost: 1.0 + (index % 5) as f32,
                    loss_ema: 1.0 + (index % 7) as f32 * 0.5,
                    previous_loss_ema: 2.0 + (index % 7) as f32 * 0.5,
                    gradient_alignment: 0.0,
                    is_hash_noise: index % 11 == 0,
                })
                .collect::<Vec<_>>();
            let sampler = RuliadFrontierSampler::new(RuliadSamplerConfig::default(), candidates);
            println!("{}", serde_json::to_string_pretty(&sampler.snapshot())?);
        }
    }
    Ok(())
}
