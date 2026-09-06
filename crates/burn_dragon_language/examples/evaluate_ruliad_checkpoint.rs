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
    config::{RuliadProofPolicyPromptContext, RuliadProofPolicyScoring},
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
    policy_scoring: Option<RuliadProofPolicyScoring>,
    policy_prompt_context: Option<RuliadProofPolicyPromptContext>,
    policy_max_steps: Option<usize>,
    panel_seed: Option<u64>,
    evaluation_corpus: Option<PathBuf>,
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
    let mut policy_scoring = None;
    let mut policy_prompt_context = None;
    let mut policy_max_steps = None;
    let mut panel_seed = None;
    let mut evaluation_corpus = None;
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
            "--evaluation-corpus" => {
                evaluation_corpus =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        anyhow!("--evaluation-corpus requires a path")
                    })?));
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
            "--panel-seed" => {
                panel_seed = Some(
                    args.next()
                        .ok_or_else(|| anyhow!("--panel-seed requires a value"))?
                        .parse::<u64>()
                        .context("--panel-seed requires a non-negative integer")?,
                )
            }
            "--difficulty-levels" => {
                difficulty_levels = parse_usize(&mut args, "--difficulty-levels")?
            }
            "--batch-size" => batch_size = Some(parse_usize(&mut args, "--batch-size")?),
            "--policy-scoring" => {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow!("--policy-scoring requires a value"))?;
                policy_scoring = Some(match value.as_str() {
                    "completion_likelihood" => RuliadProofPolicyScoring::CompletionLikelihood,
                    "semantic_energy" => RuliadProofPolicyScoring::SemanticEnergy,
                    "residual_energy" => RuliadProofPolicyScoring::ResidualEnergy,
                    _ => {
                        return Err(anyhow!(
                            "--policy-scoring must be completion_likelihood, semantic_energy, or residual_energy"
                        ));
                    }
                });
            }
            "--policy-prompt-context" => {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow!("--policy-prompt-context requires a value"))?;
                policy_prompt_context = Some(match value.as_str() {
                    "full_problem_suffix" => RuliadProofPolicyPromptContext::FullProblemSuffix,
                    "local_action_state" => RuliadProofPolicyPromptContext::LocalActionState,
                    "exact_action_state" => RuliadProofPolicyPromptContext::ExactActionState,
                    _ => {
                        return Err(anyhow!(
                            "--policy-prompt-context must be full_problem_suffix, local_action_state, or exact_action_state"
                        ));
                    }
                });
            }
            "--policy-max-steps" => {
                policy_max_steps = Some(parse_usize(&mut args, "--policy-max-steps")?)
            }
            "--no-closed-loop-rollout" => include_closed_loop_rollout = false,
            "--help" | "-h" => {
                println!(
                    "usage: evaluate_ruliad_checkpoint --backend <cpu|cuda> --checkpoint <run-or-checkpoint-dir> [--epoch N] [--output report.json] [--free-run-items N] [--policy-items N] [--panel-seed N] [--difficulty-levels N] [--batch-size N] [--evaluation-corpus path.toml] [--policy-scoring <completion_likelihood|semantic_energy|residual_energy>] [--policy-prompt-context <full_problem_suffix|local_action_state|exact_action_state>] [--policy-max-steps <0|N>] [--no-closed-loop-rollout]"
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
        policy_scoring,
        policy_prompt_context,
        policy_max_steps,
        panel_seed,
        evaluation_corpus,
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
    policy_scoring: String,
    policy_prompt_context: String,
    policy_max_steps: usize,
    options: RuliadEvaluationSuiteOptions,
    corpus_semantic_fingerprint: Option<String>,
    corpus_override: Option<EvaluationCorpusOverride>,
    evaluation: burn_dragon_language::train::RuliadEvaluationSuiteReport,
}

#[cfg(feature = "train")]
#[derive(Serialize)]
struct EvaluationCorpusOverride {
    checkpoint_config_corpus: PathBuf,
    checkpoint_config_corpus_fingerprint: String,
    evaluation_corpus: PathBuf,
    evaluation_corpus_fingerprint: String,
}

#[cfg(feature = "train")]
fn check_corpus_tokenization(
    original: &burn_dragon_universality::ruliad::config::RuliadCorpusConfig,
    requested: &burn_dragon_universality::ruliad::config::RuliadCorpusConfig,
) -> Result<()> {
    if original.tokenization != requested.tokenization {
        return Err(anyhow!(
            "evaluation corpus must preserve checkpoint corpus tokenization"
        ));
    }
    Ok(())
}

#[cfg(feature = "train")]
fn override_evaluation_corpus(
    config: &mut burn_dragon_language::config::TrainingConfig,
    requested: Option<&Path>,
) -> Result<Option<EvaluationCorpusOverride>> {
    use burn_dragon_language::config::DatasetSourceConfig;
    use burn_dragon_universality::ruliad::{
        config::load_ruliad_config, contract::RuliadSemanticContract,
    };

    let Some(requested) = requested else {
        return Ok(None);
    };
    let DatasetSourceConfig::UniversalityRuliad { config: path } = &mut config.dataset.source
    else {
        return Err(anyhow!(
            "--evaluation-corpus requires a Ruliad checkpoint dataset"
        ));
    };
    let original = load_ruliad_config(path)?;
    let replacement = load_ruliad_config(requested)?;
    check_corpus_tokenization(&original, &replacement)?;
    let identity = EvaluationCorpusOverride {
        checkpoint_config_corpus: path.clone(),
        checkpoint_config_corpus_fingerprint: RuliadSemanticContract::from_config(
            &original,
            path.parent(),
        )?
        .canonical_hash()?,
        evaluation_corpus: requested.to_path_buf(),
        evaluation_corpus_fingerprint: RuliadSemanticContract::from_config(
            &replacement,
            requested.parent(),
        )?
        .canonical_hash()?,
    };
    *path = requested.to_path_buf();
    // Fixed evaluation panels must not restore the old source-selection catalog.
    config.training.source_selection_state_path = None;
    Ok(Some(identity))
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
    let mut config =
        load_training_config_for_checkpoint(&[], Some(&checkpoint), args.backend.as_str())?;
    if let Some(scoring) = args.policy_scoring {
        if scoring.uses_sequence_score_head()
            && !config
                .model
                .sequence_score_head
                .as_ref()
                .is_some_and(|head| head.enabled)
        {
            return Err(anyhow!(
                "policy scorer {} requires a checkpoint with an enabled sequence score head",
                scoring.as_str()
            ));
        }
        config.training.ruliad_policy_probe.scoring = scoring;
    }
    if let Some(max_steps) = args.policy_max_steps {
        config.training.ruliad_policy_probe.max_steps = max_steps;
    }
    if let Some(context) = args.policy_prompt_context {
        config.training.ruliad_policy_probe.prompt_context = context;
    }
    let policy_scoring = config.training.ruliad_policy_probe.scoring;
    let policy_max_steps = config.training.ruliad_policy_probe.max_steps;
    let corpus_override =
        override_evaluation_corpus(&mut config, args.evaluation_corpus.as_deref())?;
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
        panel_seed: args.panel_seed.unwrap_or(config.training.validation.seed),
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
        version: 5,
        backend: args.backend.clone(),
        checkpoint,
        checkpoint_epoch,
        git_commit: option_env!("BURN_DRAGON_GIT_COMMIT").map(str::to_string),
        policy_scoring: policy_scoring.as_str().to_string(),
        policy_prompt_context: config
            .training
            .ruliad_policy_probe
            .prompt_context
            .as_str()
            .to_string(),
        policy_max_steps,
        options,
        corpus_semantic_fingerprint: dataset.ruliad_semantic_fingerprint()?,
        corpus_override,
        evaluation,
    })
}

#[cfg(all(test, feature = "train"))]
mod tests {
    use super::*;

    #[test]
    fn evaluation_override_is_explicit_and_preserves_model_and_objective() {
        let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../config/language/experiments/next_latent");
        let mut original =
            burn_dragon_language::load_training_config(&[directory.join("credit-base.toml")])
                .unwrap();
        original.dataset.source =
            burn_dragon_language::config::DatasetSourceConfig::UniversalityRuliad {
                config: directory.join("corpus.toml"),
            };
        let mut evaluation = original.clone();
        assert!(
            override_evaluation_corpus(&mut evaluation, None)
                .unwrap()
                .is_none()
        );
        assert_eq!(evaluation, original);
        let identity = override_evaluation_corpus(
            &mut evaluation,
            Some(&directory.join("in-distribution.corpus.toml")),
        )
        .unwrap()
        .unwrap();
        assert_ne!(
            identity.checkpoint_config_corpus_fingerprint,
            identity.evaluation_corpus_fingerprint
        );
        assert_eq!(evaluation.model, original.model);
        assert_eq!(evaluation.training, original.training);
        evaluation.training.source_selection_state_path = Some("old-sampler.json".into());
        override_evaluation_corpus(
            &mut evaluation,
            Some(&directory.join("in-distribution.corpus.toml")),
        )
        .unwrap();
        assert!(evaluation.training.source_selection_state_path.is_none());

        evaluation.dataset.source =
            burn_dragon_language::config::DatasetSourceConfig::UniversalityManifest {
                manifest: "unused.json".into(),
            };
        assert!(
            override_evaluation_corpus(
                &mut evaluation,
                Some(&directory.join("in-distribution.corpus.toml"))
            )
            .is_err()
        );
    }

    #[test]
    fn evaluation_override_rejects_changed_token_mapping() {
        use burn_dragon_universality::ruliad::config::{
            RuliadTokenizationConfig, load_ruliad_config,
        };
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../config/language/experiments/next_latent/corpus.toml");
        let original = load_ruliad_config(&path).unwrap();
        let mut requested = original.clone();
        // Equal vocabulary sizes do not imply equal token semantics.
        requested.tokenization = RuliadTokenizationConfig::Symbolic {
            vocab_size: 272,
            eos_id: Some(271),
        };
        assert!(check_corpus_tokenization(&original, &requested).is_err());
    }
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
