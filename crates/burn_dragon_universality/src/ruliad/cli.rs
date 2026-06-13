use std::path::PathBuf;

use anyhow::{Result, anyhow};
use clap::{Parser, Subcommand};

use crate::ruliad::config::{
    LeanMode, RuliadCorpusConfig, RuliadSerializationConfig, RuliadTokenizationConfig,
    default_ruliad_families, load_ruliad_config,
};
use crate::ruliad::eval::{
    RuliadDiagnosticThresholds, RuliadEvalBaseline, RuliadEvalConfig, baseline_completions,
    build_eval_items_from_manifest, diagnose_config, diagnose_manifest, evaluate_completions,
    read_completion_records, write_eval_items_jsonl,
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
    Diagnose {
        #[arg(long)]
        manifest: Option<PathBuf>,
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long, default_value_t = 128)]
        samples: usize,
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long, default_value_t = 0.0)]
        min_task_share: f32,
        #[arg(long, default_value_t = 0.0)]
        max_duplicate_oracle_hash_rate: f32,
        #[arg(long)]
        relaxed_semantics: bool,
        #[arg(long)]
        fail_on_gates: bool,
    },
    Eval {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        completions: Option<PathBuf>,
        #[arg(long, default_value = "oracle")]
        baseline: RuliadEvalBaseline,
        #[arg(long)]
        split: Option<String>,
        #[arg(long)]
        max_items: Option<usize>,
        #[arg(long, default_value_t = true)]
        include_hash_canaries: bool,
        #[arg(long)]
        items_out: Option<PathBuf>,
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long, default_value_t = 0.0)]
        min_semantic_accuracy: f32,
        #[arg(long)]
        max_semantic_accuracy: Option<f32>,
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
        Command::Diagnose {
            manifest,
            config,
            samples,
            out,
            min_task_share,
            max_duplicate_oracle_hash_rate,
            relaxed_semantics,
            fail_on_gates,
        } => {
            let thresholds = RuliadDiagnosticThresholds {
                min_task_share,
                max_duplicate_oracle_hash_rate,
                require_all_semantics: !relaxed_semantics,
            };
            let report = match (manifest, config) {
                (Some(manifest), None) => diagnose_manifest(&manifest, thresholds)?,
                (None, Some(config_path)) => {
                    let config = load_ruliad_config(&config_path)?;
                    diagnose_config(&config, samples, thresholds)?
                }
                (Some(_), Some(_)) => {
                    return Err(anyhow!(
                        "diagnose accepts either --manifest or --config, not both"
                    ));
                }
                (None, None) => {
                    return Err(anyhow!("diagnose requires --manifest or --config"));
                }
            };
            write_json_report(out, &report)?;
            if fail_on_gates && !report.gate_failures.is_empty() {
                return Err(anyhow!(
                    "ruliad diagnostic gates failed: {}",
                    report.gate_failures.join(", ")
                ));
            }
        }
        Command::Eval {
            manifest,
            completions,
            baseline,
            split,
            max_items,
            include_hash_canaries,
            items_out,
            out,
            min_semantic_accuracy,
            max_semantic_accuracy,
        } => {
            let eval_config = RuliadEvalConfig {
                split: split.as_deref().map(parse_split).transpose()?.flatten(),
                max_items,
                include_hash_canaries,
            };
            let items = build_eval_items_from_manifest(&manifest, &eval_config)?;
            if let Some(items_out) = items_out {
                write_eval_items_jsonl(&items_out, &items)?;
            }
            let completions = if let Some(completions) = completions {
                read_completion_records(&completions)?
            } else {
                baseline_completions(&items, baseline)
            };
            let dataset_name = crate::manifest::load_manifest(&manifest)?.dataset_name;
            let report = evaluate_completions(dataset_name, &items, &completions);
            write_json_report(out, &report)?;
            if report.semantic_accuracy < min_semantic_accuracy {
                return Err(anyhow!(
                    "ruliad eval semantic_accuracy {:.6} below minimum {:.6}",
                    report.semantic_accuracy,
                    min_semantic_accuracy
                ));
            }
            if let Some(max_semantic_accuracy) = max_semantic_accuracy
                && report.semantic_accuracy > max_semantic_accuracy
            {
                return Err(anyhow!(
                    "ruliad eval semantic_accuracy {:.6} above maximum {:.6}",
                    report.semantic_accuracy,
                    max_semantic_accuracy
                ));
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
                    difficulty_level: index % 4,
                    params_hash: format!("{index:016x}"),
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

fn parse_split(value: &str) -> Result<Option<crate::manifest::SampleSplit>> {
    match value {
        "all" => Ok(None),
        "train" => Ok(Some(crate::manifest::SampleSplit::Train)),
        "validation" | "val" => Ok(Some(crate::manifest::SampleSplit::Validation)),
        other => Err(anyhow!(
            "invalid split `{other}`; expected train, validation, or all"
        )),
    }
}

fn write_json_report<T: serde::Serialize>(out: Option<PathBuf>, report: &T) -> Result<()> {
    let payload = serde_json::to_string_pretty(report)?;
    if let Some(out) = out {
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(out, payload)?;
    } else {
        println!("{payload}");
    }
    Ok(())
}
