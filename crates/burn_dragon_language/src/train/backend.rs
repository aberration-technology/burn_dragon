use crate::checkpoint::{RUN_DIR_ENV, RUN_NAME_ENV};
use crate::train::prelude::*;
use crate::train::schedule::{
    ForwardEggrollTrainEnvironment, TrainEnvironment, autotune_eggroll_population_chunk_size,
    resolve_lr_scheduler, resolve_train_schedule, train_with_dynamic_neuron_scaling_scheduler,
    train_with_eggroll_forward_only, train_with_scheduler,
};
use crate::train::startup_autotune::{
    resolve_gradient_accumulation_steps, resolve_startup_batch_size,
    resolve_startup_batch_size_forward_only,
};
use crate::train::utils::{build_training_execution_form, write_run_config};
use crate::train::{resolve_dragon_language_optimizer, validate_dragon_continual_backprop};
use crate::{ensure_random_scaffold_run_manifest, write_training_snapshot};
use burn_dragon_core::{SequenceMemorySystem, SequenceTrainingExecutor};
use serde::Serialize;
use std::fs;

use burn_dragon_time::Instant;
use tracing::warn;

const PROCESS_GROUP_RUN_DIR_ENV: &str = "BURN_DRAGON_PROCESS_GROUP_RUN_DIR";
const PROCESS_GROUP_RUN_NAME_ENV: &str = "BURN_DRAGON_PROCESS_GROUP_RUN_NAME";
const CUDA_LINEAR_DENSE_SCORE_AUTO_BLOCK_LIMIT: usize = 2048;

fn cuda_mamba_training_geometry_summary(
    model_config: &DragonConfig,
    micro_batch_size: usize,
    training_kernel_block_size: usize,
) -> Option<String> {
    if !matches!(
        model_config.sequence_kernel.memory_system,
        SequenceMemorySystem::Mamba3StateSpaceDuality
    ) {
        return None;
    }

    let resolved = model_config.mamba.resolve(
        model_config.n_embd,
        SequenceMemorySystem::Mamba3StateSpaceDuality,
    );
    Some(format!(
        "cuda mamba3 geometry: micro_batch={} kernel_block={} tokens/micro_batch={} d_inner={} headdim={} nheads={} ngroups={} d_state={} rope_angles={} chunk_size={}",
        micro_batch_size,
        training_kernel_block_size,
        micro_batch_size.saturating_mul(training_kernel_block_size),
        resolved.d_inner,
        resolved.headdim,
        resolved.nheads,
        resolved.ngroups,
        resolved.d_state,
        resolved.num_rope_angles,
        resolved.chunk_size,
    ))
}

fn resolve_run_root() -> PathBuf {
    crate::checkpoint::resolve_run_root()
}

fn resolve_checkpoint_steps_per_epoch(
    training: &TrainingHyperparameters,
    dataset_steps_per_epoch: usize,
) -> usize {
    match training.epochs {
        Some(_) => dataset_steps_per_epoch.max(1),
        None => training
            .checkpoint_interval_iters
            .min(training.max_iters.max(1))
            .max(1),
    }
}

fn derive_run_name(run_dir: &Path) -> Result<String> {
    run_dir
        .file_name()
        .and_then(|name| name.to_str())
        .map(ToOwned::to_owned)
        .ok_or_else(|| anyhow!("failed to derive run name from {}", run_dir.display()))
}

fn resolve_run_artifacts(
    parallel_runtime: &ParallelRuntime,
    run_root: &Path,
    training: &TrainingHyperparameters,
) -> Result<(PathBuf, String)> {
    if let Some(resume_run_dir) = &training.resume_run_dir {
        let run_dir = resume_run_dir.clone();
        let run_name = derive_run_name(&run_dir)?;
        if !parallel_runtime.is_process_group_launch() {
            if !run_dir.is_dir() {
                return Err(anyhow!(
                    "training.resume_run_dir does not exist or is not a directory: {}",
                    run_dir.display()
                ));
            }
            return Ok((run_dir, run_name));
        }
        let env_run_dir = std::env::var_os(PROCESS_GROUP_RUN_DIR_ENV)
            .map(PathBuf::from)
            .ok_or_else(|| {
                anyhow!(
                    "parallel.mode=ddp process-group launches require {PROCESS_GROUP_RUN_DIR_ENV}"
                )
            })?;
        let env_run_name = std::env::var(PROCESS_GROUP_RUN_NAME_ENV).map_err(|_| {
            anyhow!("parallel.mode=ddp process-group launches require {PROCESS_GROUP_RUN_NAME_ENV}")
        })?;
        if env_run_dir != run_dir || env_run_name != run_name {
            return Err(anyhow!(
                "process-group resume requires launcher env run_dir/run_name to match training.resume_run_dir (env={} name={}, resume={} name={})",
                env_run_dir.display(),
                env_run_name,
                run_dir.display(),
                run_name
            ));
        }
        return Ok((run_dir, run_name));
    }

    let env_run_dir = std::env::var_os(RUN_DIR_ENV).map(PathBuf::from);
    let env_run_name = std::env::var(RUN_NAME_ENV).ok();
    if !parallel_runtime.is_process_group_launch() {
        match (env_run_dir, env_run_name) {
            (Some(run_dir), Some(run_name)) => {
                fs::create_dir_all(&run_dir).with_context(|| {
                    format!(
                        "failed to create preassigned run directory {}",
                        run_dir.display()
                    )
                })?;
                return Ok((run_dir, run_name));
            }
            (None, None) => return create_run_dir(run_root),
            _ => {
                return Err(anyhow!(
                    "single-process launches require both {RUN_DIR_ENV} and {RUN_NAME_ENV} when either one is set"
                ));
            }
        }
    }

    let run_dir = std::env::var_os(PROCESS_GROUP_RUN_DIR_ENV)
        .map(PathBuf::from)
        .ok_or_else(|| {
            anyhow!("parallel.mode=ddp process-group launches require {PROCESS_GROUP_RUN_DIR_ENV}")
        })?;
    let run_name = std::env::var(PROCESS_GROUP_RUN_NAME_ENV).map_err(|_| {
        anyhow!("parallel.mode=ddp process-group launches require {PROCESS_GROUP_RUN_NAME_ENV}")
    })?;
    Ok((run_dir, run_name))
}

fn resolve_resume_checkpoint_epoch(
    training: &TrainingHyperparameters,
    run_dir: &Path,
) -> Result<Option<usize>> {
    let Some(_) = training.resume_run_dir else {
        return Ok(None);
    };
    let checkpoint_dir = run_dir.join("checkpoint");
    let (_, epoch) = crate::checkpoint::resolve_checkpoint_base(
        &checkpoint_dir,
        training.resume_checkpoint_epoch,
    )
    .with_context(|| {
        format!(
            "failed to resolve resume checkpoint in {}",
            checkpoint_dir.display()
        )
    })?;
    Ok(Some(epoch))
}

fn configure_resume_source_selection_state(
    config: &mut TrainingConfig,
    dataset: &Dataset,
    run_dir: &Path,
    resume_checkpoint_epoch: Option<usize>,
) -> Result<bool> {
    let Some(epoch) = resume_checkpoint_epoch else {
        return Ok(false);
    };
    if !dataset.uses_live_source_selection() {
        return Ok(false);
    }
    let checkpoint_path = run_dir
        .join("checkpoint")
        .join(format!("source-selection-state-{epoch}.json"));
    if !checkpoint_path.is_file() {
        if matches!(
            config.training.launch_mode,
            burn_dragon_train::train::pipeline::TrainingLaunchMode::ResumeExactRun
        ) {
            return Err(anyhow!(
                "exact resume of live source selection requires {}",
                checkpoint_path.display()
            ));
        }
        warn!(
            "resume checkpoint {epoch} has no source-selection state at {}; live curriculum will use configured initialization",
            checkpoint_path.display()
        );
        return Ok(false);
    }
    config.training.source_selection_state_path = Some(checkpoint_path.clone());
    info!(
        "resume source-selection state: epoch={epoch} path={}",
        checkpoint_path.display()
    );
    Ok(true)
}

fn validate_resolved_optimizer_owned_batching(config: &TrainingConfig) -> Result<()> {
    if matches!(
        config.training.algorithm,
        TrainingAlgorithm::PredictiveCoding
    ) && config.training.gradient_accumulation_steps != 1
    {
        return Err(anyhow!(
            "startup batching resolved gradient_accumulation_steps={} for predictive coding; local parameter updates require one microbatch per optimizer boundary",
            config.training.gradient_accumulation_steps
        ));
    }
    if config.training.predictive_context_routing.enabled
        && config.training.gradient_accumulation_steps != 1
    {
        return Err(anyhow!(
            "startup batching resolved gradient_accumulation_steps={} for predictive context routing; routed gradients must not cross context-owned optimizer steps",
            config.training.gradient_accumulation_steps
        ));
    }
    Ok(())
}

fn initialize_model_from_checkpoint<B: BackendTrait>(
    resolved_config: &TrainingConfig,
    training: &TrainingHyperparameters,
    model: &mut DragonModel<B>,
    device: &B::Device,
    backend_name: &str,
) -> Result<()> {
    let Some(checkpoint_path) = &training.init_checkpoint_path else {
        return Ok(());
    };
    *model = crate::checkpoint::apply_init_checkpoint_to_language_core(
        model,
        resolved_config,
        checkpoint_path,
        training.init_checkpoint_epoch,
        backend_name,
        device,
    )?;
    Ok(())
}

fn train_with_resolved_scheduler<B>(
    context: &TrainEnvironment<'_, B>,
    model: LanguageTrainModel<B>,
    optimizer: crate::train::continual_backprop::LanguageOptimizer<B>,
    scheduler: ResolvedLrScheduler,
) -> Result<DragonModel<ValidBackend<B>>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let use_dynamic_scaling = context.training.neuron_scaling.enabled;
    if use_dynamic_scaling && context.parallel_runtime.mode != ParallelismKind::Single {
        return Err(anyhow!(
            "training.neuron_scaling.enabled currently requires parallel.mode=single; in-process Dragon neuron scaling mutates model and optimizer state between validation epochs"
        ));
    }
    let source_selection_uses_live_policy = context
        .source_selection_dataset
        .as_ref()
        .is_some_and(|dataset| dataset.uses_live_source_selection());
    let use_event_scheduler = use_event_scheduler_for_training(
        context.training,
        context.parallel_runtime.mode,
        source_selection_uses_live_policy,
    );
    match scheduler {
        ResolvedLrScheduler::Constant(lr) => {
            if use_event_scheduler {
                train_with_dynamic_neuron_scaling_scheduler(context, model, optimizer, lr)
            } else {
                train_with_scheduler(context, model, optimizer, lr)
            }
        }
        ResolvedLrScheduler::Cosine(scheduler) => {
            if use_event_scheduler {
                train_with_dynamic_neuron_scaling_scheduler(context, model, optimizer, scheduler)
            } else {
                train_with_scheduler(context, model, optimizer, scheduler)
            }
        }
        ResolvedLrScheduler::Linear(scheduler) => {
            if use_event_scheduler {
                train_with_dynamic_neuron_scaling_scheduler(context, model, optimizer, scheduler)
            } else {
                train_with_scheduler(context, model, optimizer, scheduler)
            }
        }
        ResolvedLrScheduler::Exponential(scheduler) => {
            if use_event_scheduler {
                train_with_dynamic_neuron_scaling_scheduler(context, model, optimizer, scheduler)
            } else {
                train_with_scheduler(context, model, optimizer, scheduler)
            }
        }
        ResolvedLrScheduler::Step(scheduler) => {
            if use_event_scheduler {
                train_with_dynamic_neuron_scaling_scheduler(context, model, optimizer, scheduler)
            } else {
                train_with_scheduler(context, model, optimizer, scheduler)
            }
        }
        ResolvedLrScheduler::Noam(scheduler) => {
            if use_event_scheduler {
                train_with_dynamic_neuron_scaling_scheduler(context, model, optimizer, scheduler)
            } else {
                train_with_scheduler(context, model, optimizer, scheduler)
            }
        }
    }
}

fn use_event_scheduler_for_training(
    training: &TrainingHyperparameters,
    parallel_mode: ParallelismKind,
    source_selection_uses_live_policy: bool,
) -> bool {
    let local_runtime_objectives = training.dynamics.enabled
        || training.sequence_state_probe.enabled
        || training.predictive_context_routing.enabled
        || (matches!(training.algorithm, TrainingAlgorithm::PredictiveCoding)
            && training.tbptt_persist_across_steps)
        || source_selection_uses_live_policy;

    !training.validation.execution.is_local()
        || training.neuron_scaling.enabled
        || training.predictive_coding.enabled
        || (parallel_mode == ParallelismKind::Single && local_runtime_objectives)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load_profile(file_name: &str) -> TrainingConfig {
        let profile_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../burn_dragon_p2p/deploy/profiles")
            .join(file_name);
        crate::config::train::load_training_config(std::slice::from_ref(&profile_path))
            .unwrap_or_else(|err| panic!("load {}: {err}", profile_path.display()))
    }

    #[test]
    fn ruliad_policy_runtime_objectives_attach_batch_metadata() {
        let control =
            load_profile("ruliad-1m-la-16k.answer-completion.self-recovery.training.toml");
        assert!(
            !control
                .training
                .ruliad_supervision
                .needs_ruliad_policy_batch()
        );

        let training = load_profile(
            "ruliad-1m-la-16k.answer-completion-recovery-denoising.self-recovery.training.toml",
        );
        assert!(
            training
                .training
                .ruliad_supervision
                .needs_ruliad_policy_batch()
        );

        let contract = load_profile("ruliad-1m-la-64k.answer-contract.training.toml");
        assert!(
            contract
                .training
                .ruliad_supervision
                .needs_ruliad_policy_batch()
        );
    }

    #[test]
    fn verifier_reward_runtime_objective_attaches_batch_metadata() {
        let training = load_profile("ruliad-1m-la-16k.verifier-reward.training.toml");
        assert!(
            training
                .training
                .ruliad_supervision
                .needs_ruliad_policy_batch()
        );
    }

    #[test]
    fn sequence_state_probe_selects_the_event_scheduler() {
        let training = load_profile("ruliad-r3.stateful-tbptt-block512-reset.toml");
        assert!(training.training.sequence_state_probe.enabled);
        assert!(use_event_scheduler_for_training(
            &training.training,
            ParallelismKind::Single,
            false,
        ));
    }

    #[test]
    fn live_source_selection_uses_checkpoint_complete_event_scheduler() {
        let profile_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../config/language/experiments/predictive_coding/local-pc-smoke.toml");
        let config = crate::config::train::load_training_config(&[profile_path])
            .expect("load local PC smoke profile");
        assert_eq!(config.training.events.source_weighted_validation_batches, 0);
        assert!(!config.training.tbptt_persist_across_steps);
        assert!(!use_event_scheduler_for_training(
            &config.training,
            ParallelismKind::Single,
            false,
        ));
        assert!(use_event_scheduler_for_training(
            &config.training,
            ParallelismKind::Single,
            true,
        ));
    }

    #[test]
    fn offloaded_duty_profile_uses_external_evaluator_scheduler() {
        let config = load_profile("ruliad-r3.typed-policy-offloaded-duty-ablation.toml");
        config
            .validate()
            .expect("offloaded duty profile should satisfy the external evaluator contract");
        assert_eq!(
            config.training.validation.execution,
            crate::config::TrainingValidationExecution::ExternalEvaluator
        );
        assert!(use_event_scheduler_for_training(
            &config.training,
            config.parallel.mode,
            true,
        ));
    }

    #[test]
    fn resolved_batching_rejects_gradient_accumulation_for_predictive_coding() {
        let mut config =
            load_profile("ruliad-1m-la-16k.answer-completion.self-recovery.training.toml");
        config.training.algorithm = TrainingAlgorithm::PredictiveCoding;
        config.training.gradient_accumulation_steps = 2;
        let error = validate_resolved_optimizer_owned_batching(&config)
            .expect_err("startup autotune must not queue PC-owned updates");
        assert!(error.to_string().contains("one microbatch"));
    }
}

#[derive(Debug, Serialize)]
struct PreStepValidationReport {
    split: &'static str,
    mean_loss: f64,
    num_batches: usize,
    init_checkpoint_path: String,
    init_checkpoint_epoch: Option<usize>,
    init_transfer_interface_checkpoint_path: Option<String>,
    init_transfer_interface_checkpoint_epoch: Option<usize>,
    init_transfer_preserve_interface_input_embedding: bool,
    init_transfer_preserve_interface_output_head: bool,
    init_transfer_backbone_blend_alpha: Option<f32>,
    init_transfer_backbone_grad_scale: Option<f32>,
    init_transfer_backbone_grad_scale_steps: Option<usize>,
    init_transfer_fresh_top_layers: Option<usize>,
    init_transfer_preserve_fresh_decoder: bool,
    init_transfer_preserve_fresh_norm: bool,
    init_transfer_match_fresh_rms: bool,
}

fn mean_scalar_from_valid_loss<B: BackendTrait>(tensor: Tensor<B, 1>) -> f64 {
    let values = tensor
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("loss tensor to vec");
    if values.is_empty() {
        0.0
    } else {
        values.iter().map(|value| *value as f64).sum::<f64>() / values.len() as f64
    }
}

fn maybe_write_pre_step_validation_report<B>(
    training: &TrainingHyperparameters,
    parallel_runtime: &ParallelRuntime,
    run_dir: &Path,
    model: &LanguageTrainModel<B>,
    valid_loader: &Arc<dyn DataLoader<ValidBackend<B>, SequenceBatch<ValidBackend<B>>>>,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    if !training.validation.execution.is_local() {
        return Ok(());
    }
    let Some(init_checkpoint_path) = training.init_checkpoint_path.as_ref() else {
        return Ok(());
    };
    if !parallel_runtime.is_primary() {
        return Ok(());
    }

    let valid_model = model.valid().materialize_random_scaffold_for_inference();
    let iterator = valid_loader.iter();
    let mut total = 0.0;
    let mut count = 0usize;

    for item in iterator {
        let output = valid_model.step(item);
        let loss_value: LossValue<ValidBackend<B>> = output.adapt();
        total += mean_scalar_from_valid_loss(loss_value.value());
        count += 1;
    }

    let mean_loss = if count == 0 {
        0.0
    } else {
        total / count as f64
    };
    let report = PreStepValidationReport {
        split: "val",
        mean_loss,
        num_batches: count,
        init_checkpoint_path: init_checkpoint_path.display().to_string(),
        init_checkpoint_epoch: training.init_checkpoint_epoch,
        init_transfer_interface_checkpoint_path: training
            .init_transfer
            .interface_checkpoint_path
            .as_ref()
            .map(|path| path.display().to_string()),
        init_transfer_interface_checkpoint_epoch: training.init_transfer.interface_checkpoint_epoch,
        init_transfer_preserve_interface_input_embedding: training
            .init_transfer
            .preserve_interface_input_embedding,
        init_transfer_preserve_interface_output_head: training
            .init_transfer
            .preserve_interface_output_head,
        init_transfer_backbone_blend_alpha: training.init_transfer.backbone_blend_alpha,
        init_transfer_backbone_grad_scale: training.init_transfer.backbone_grad_scale,
        init_transfer_backbone_grad_scale_steps: training.init_transfer.backbone_grad_scale_steps,
        init_transfer_fresh_top_layers: training.init_transfer.fresh_top_layers,
        init_transfer_preserve_fresh_decoder: training.init_transfer.preserve_fresh_decoder,
        init_transfer_preserve_fresh_norm: training.init_transfer.preserve_fresh_norm,
        init_transfer_match_fresh_rms: training.init_transfer.match_fresh_rms,
    };
    let payload =
        serde_json::to_string_pretty(&report).context("serialize pre-step validation report")?;
    let path = run_dir.join("pre_step_validation.json");
    fs::write(&path, payload)
        .with_context(|| format!("write pre-step validation report to {}", path.display()))?;
    info!("pre-step validation before optimizer step 1: mean_loss={mean_loss:.6} batches={count}");
    Ok(())
}

fn maybe_write_pre_step_validation_report_forward_only<B>(
    training: &TrainingHyperparameters,
    parallel_runtime: &ParallelRuntime,
    run_dir: &Path,
    model: &LanguageTrainModel<B>,
    valid_loader: &Arc<dyn DataLoader<B, SequenceBatch<B>>>,
) -> Result<()>
where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone,
{
    if !training.validation.execution.is_local() {
        return Ok(());
    }
    let Some(init_checkpoint_path) = training.init_checkpoint_path.as_ref() else {
        return Ok(());
    };
    if !parallel_runtime.is_primary() {
        return Ok(());
    }

    let iterator = valid_loader.iter();
    let mut total = 0.0;
    let mut count = 0usize;

    for item in iterator {
        let output = ValidStep::step(model, item);
        let loss_value: LossValue<B> = output.adapt();
        total += mean_scalar_from_valid_loss(loss_value.value());
        count += 1;
    }

    let mean_loss = if count == 0 {
        0.0
    } else {
        total / count as f64
    };
    let report = PreStepValidationReport {
        split: "val",
        mean_loss,
        num_batches: count,
        init_checkpoint_path: init_checkpoint_path.display().to_string(),
        init_checkpoint_epoch: training.init_checkpoint_epoch,
        init_transfer_interface_checkpoint_path: training
            .init_transfer
            .interface_checkpoint_path
            .as_ref()
            .map(|path| path.display().to_string()),
        init_transfer_interface_checkpoint_epoch: training.init_transfer.interface_checkpoint_epoch,
        init_transfer_preserve_interface_input_embedding: training
            .init_transfer
            .preserve_interface_input_embedding,
        init_transfer_preserve_interface_output_head: training
            .init_transfer
            .preserve_interface_output_head,
        init_transfer_backbone_blend_alpha: training.init_transfer.backbone_blend_alpha,
        init_transfer_backbone_grad_scale: training.init_transfer.backbone_grad_scale,
        init_transfer_backbone_grad_scale_steps: training.init_transfer.backbone_grad_scale_steps,
        init_transfer_fresh_top_layers: training.init_transfer.fresh_top_layers,
        init_transfer_preserve_fresh_decoder: training.init_transfer.preserve_fresh_decoder,
        init_transfer_preserve_fresh_norm: training.init_transfer.preserve_fresh_norm,
        init_transfer_match_fresh_rms: training.init_transfer.match_fresh_rms,
    };
    let payload =
        serde_json::to_string_pretty(&report).context("serialize pre-step validation report")?;
    let path = run_dir.join("pre_step_validation.json");
    fs::write(&path, payload)
        .with_context(|| format!("write pre-step validation report to {}", path.display()))?;
    info!("pre-step validation before optimizer step 1: mean_loss={mean_loss:.6} batches={count}");
    Ok(())
}

pub fn optimizer_uses_forward_only_eggroll(config: &TrainingConfig) -> bool {
    matches!(
        config.resolved_training_algorithm(),
        TrainingAlgorithm::Eggroll
    )
}

pub fn training_uses_local_predictive_coding(config: &TrainingConfig) -> bool {
    matches!(
        config.resolved_training_algorithm(),
        TrainingAlgorithm::PredictiveCoding
    )
}

pub(crate) fn resolve_effective_training_sequence_kernel(
    configured_kernel: SequenceKernelConfig,
    training_override: Option<SequenceKernelConfig>,
    backend_name: &str,
    training_kernel_block_size: usize,
) -> (
    SequenceKernelConfig,
    Option<SequenceKernelConfig>,
    Option<&'static str>,
) {
    if let Some(explicit) = training_override {
        return (explicit, Some(explicit), None);
    }

    if backend_name.eq_ignore_ascii_case("cuda")
        && configured_kernel
            == SequenceKernelConfig::reference(SequenceMemorySystem::LinearAttention)
        && training_kernel_block_size <= CUDA_LINEAR_DENSE_SCORE_AUTO_BLOCK_LIMIT
    {
        let promoted = SequenceKernelConfig::dense_score_short_context();
        return (
            promoted,
            Some(promoted),
            Some(
                "auto-promoted short-context CUDA linear-attention training to dense_score_short_context",
            ),
        );
    }

    (configured_kernel, None, None)
}

pub fn train_backend_forward_eggroll<B, Init>(
    config: &TrainingConfig,
    dataset: Arc<Dataset>,
    backend_name: &str,
    init_backend: Init,
) -> Result<()>
where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone + 'static,
    Init: Fn(&B::Device),
{
    if !optimizer_uses_forward_only_eggroll(config) {
        return Err(anyhow!(
            "train_backend_forward_eggroll requires training.algorithm=eggroll (or auto with optimizer.name=eggroll)"
        ));
    }
    if config.optimizer.eggroll.gradient_learning_rate.is_some() {
        return Err(anyhow!(
            "optimizer.name=eggroll does not support optimizer.eggroll.gradient_learning_rate; choose pure EGGROLL or optimizer.name=adamw"
        ));
    }
    if config.optimizer.eggroll.gradient_weight_decay.is_some() {
        return Err(anyhow!(
            "optimizer.name=eggroll does not support optimizer.eggroll.gradient_weight_decay; choose pure EGGROLL or optimizer.name=adamw"
        ));
    }
    if config.training.continual_backprop.enabled
        || config.training.neuron_scaling.enabled
        || config.parallel.pipeline.enabled
        || config.parallel.mode != ParallelismKind::Single
        || config.training.tbptt_persist_across_steps
        || config.training.predictive_coding.enabled
        || !config.training.objective.is_next_token()
    {
        return Err(anyhow!(
            "forward-only EGGROLL supports single-device next-token training without pipeline, continual backprop, neuron scaling, predictive coding, or persistent TBPTT"
        ));
    }

    let parallel_runtime = resolve_parallel_runtime(&config.parallel)?;
    info!("parallel runtime: {}", parallel_runtime.summary());

    let primary_device = B::Device::default();
    let devices = resolve_training_devices::<B>(&parallel_runtime, &primary_device)?;
    for device in &devices {
        B::seed(device, config.training.seed);
        init_backend(device);
    }
    let device = devices
        .first()
        .cloned()
        .expect("at least one training device");
    info!(
        "resolved training devices: {} (eggroll forward-only plain backend, autodiff disabled)",
        devices.len()
    );

    let mut resolved_config = config.clone();
    let run_root = resolve_run_root();
    let (run_dir, run_name) =
        resolve_run_artifacts(&parallel_runtime, &run_root, &resolved_config.training)?;
    let resume_checkpoint_epoch =
        resolve_resume_checkpoint_epoch(&resolved_config.training, &run_dir)?;
    let resume_source_selection = configure_resume_source_selection_state(
        &mut resolved_config,
        &dataset,
        &run_dir,
        resume_checkpoint_epoch,
    )?;
    let startup_autotune = resolve_startup_batch_size_forward_only::<B>(
        &resolved_config,
        &dataset,
        backend_name,
        &device,
    )?;
    if let Some(report) = &startup_autotune {
        resolved_config.training.batch_size = report.resolved_batch_size;
        resolved_config.training.gradient_accumulation_steps = 1;
    }
    if startup_autotune.is_none() {
        resolved_config.training.gradient_accumulation_steps = 1;
    }
    if resolved_config
        .training
        .target_effective_batch_size
        .is_some()
    {
        warn!(
            "optimizer.name=eggroll ignores training.target_effective_batch_size; forward-only EGGROLL uses batch_size directly and gradient_accumulation_steps=1"
        );
    }
    let eggroll_auto_population = resolved_config
        .optimizer
        .apply_auto_eggroll_population(resolved_config.training.batch_size);
    resolved_config.optimizer.apply_effective_eggroll_config();
    validate_resolved_optimizer_owned_batching(&resolved_config)?;

    let datasets = if resolved_config.training.batch_size == config.training.batch_size
        && !resume_source_selection
    {
        crate::train::utils::PreparedDatasets {
            train: Arc::clone(&dataset),
            valid: Arc::clone(&dataset),
        }
    } else {
        crate::train::utils::prepare_datasets(&resolved_config.dataset, &resolved_config.training)?
    };

    let training = &resolved_config.training;
    let training_kernel_block_size =
        crate::train::utils::effective_training_kernel_block_size(training);

    let tokenizer = datasets.train.tokenizer();
    let mut model_config = build_model_config_with_tokenizer(
        &resolved_config.model,
        training_kernel_block_size,
        tokenizer.as_ref(),
    )?;
    let configured_sequence_kernel = model_config.sequence_kernel;
    let (effective_sequence_kernel, effective_training_sequence_kernel_override, promotion_reason) =
        resolve_effective_training_sequence_kernel(
            configured_sequence_kernel,
            training.sequence_kernel_override,
            backend_name,
            training_kernel_block_size,
        );
    model_config.sequence_kernel = effective_sequence_kernel;
    apply_wgpu_fused_core_override(
        &mut model_config,
        backend_name,
        WgpuFusedCoreOverride {
            recurrent: resolved_config.wgpu.training.fused_core_recurrent,
            rollout: resolved_config.wgpu.training.fused_core_rollout,
        },
    );
    info!(
        "training path fingerprint: backend={} backend_mode=eggroll_forward_only autodiff=false execution_form={} launch_mode={:?} effective_sequence_kernel={:?} sequence_kernel_override={:?} tbptt_chunk_size={:?} kernel_block_size={} pipeline_enabled={}",
        backend_name,
        build_training_execution_form(&resolved_config),
        training.launch_mode,
        model_config.sequence_kernel,
        effective_training_sequence_kernel_override,
        training.tbptt_chunk_size,
        training_kernel_block_size,
        resolved_config.parallel.pipeline.enabled,
    );
    if let Some(reason) = promotion_reason {
        info!(
            "training sequence kernel promotion: configured={:?} effective={:?} reason={reason}",
            configured_sequence_kernel, model_config.sequence_kernel,
        );
    }
    if backend_name.eq_ignore_ascii_case("cuda") && model_config.fused_kernels.enabled {
        warn!(
            "cuda language training still mixes burn_dragon_kernel fused kernels with generic Burn tensor ops; only selected recurrent/projection paths are accelerated today"
        );
    }
    if backend_name.eq_ignore_ascii_case("cuda")
        && matches!(
            model_config.sequence_kernel.memory_system,
            SequenceMemorySystem::Mamba3StateSpaceDuality
        )
        && let Some(summary) = cuda_mamba_training_geometry_summary(
            &model_config,
            resolved_config.training.batch_size,
            training_kernel_block_size,
        )
    {
        info!("{summary}");
    }
    if matches!(
        model_config.sequence_kernel.memory_system,
        SequenceMemorySystem::GatedDeltaNet2
    ) {
        info!(
            "gated_deltanet2 config: chunk_size={} qk_l2_norm={} erase_gate={:?} write_gate={:?} decay_gate={:?} allow_neg_eigval={}",
            model_config.gated_deltanet2.chunk_size,
            model_config.gated_deltanet2.qk_l2_norm,
            model_config.gated_deltanet2.erase_gate,
            model_config.gated_deltanet2.write_gate,
            model_config.gated_deltanet2.decay_gate,
            model_config.gated_deltanet2.allow_neg_eigval,
        );
        if matches!(
            model_config.sequence_kernel.executor,
            SequenceTrainingExecutor::GatedDeltaChunkWy
        ) {
            info!(
                "gated_deltanet2 chunk-WY executor uses the non-autodiff eval recurrence in EGGROLL forward-only training"
            );
        }
    }

    ensure_objective_supported(
        &resolved_config.training.objective,
        ObjectiveTrainerKind::SingleDevice,
    )?;
    ensure_rollout_objective_runtime(
        &resolved_config.training.objective,
        RolloutObjectiveRuntimeConstraints {
            uses_flat_token_logits: model_config.language_head.uses_flat_token_logits(),
            distributed_pipeline: false,
            tbptt_enabled: training.tbptt_chunk_size.is_some(),
        },
    )?;
    let summary_event_token_ids = model_config.summary_memory.write_trigger_token_ids.clone();

    let dataset_steps_per_epoch = datasets.train.steps_per_epoch(DatasetSplit::Train);
    let checkpoint_steps_per_epoch =
        resolve_checkpoint_steps_per_epoch(training, dataset_steps_per_epoch);
    let schedule = resolve_train_schedule(training, checkpoint_steps_per_epoch)?;
    let steps_per_epoch = schedule.steps_per_epoch;
    let total_epochs = schedule.total_epochs;
    let total_steps = schedule.total_steps;
    let resume_consumed_steps = resume_checkpoint_epoch
        .unwrap_or_default()
        .saturating_mul(steps_per_epoch);

    info!(
        "train schedule: dataset_steps_per_epoch={dataset_steps_per_epoch}, logical_steps_per_epoch={steps_per_epoch}, checkpoint_interval_iters={}, total_steps={total_steps}, epochs={total_epochs}, source={}",
        training.checkpoint_interval_iters,
        schedule.source.as_str()
    );
    let train_loader: Arc<dyn DataLoader<B, SequenceBatch<B>>> = if training
        .sequence_batching
        .uses_streaming_loader(training.tbptt_persist_across_steps)
    {
        Arc::new(
            StreamingDataLoader::<B>::new(
                Arc::clone(&datasets.train),
                DatasetSplit::Train,
                &device,
                steps_per_epoch,
                Some(total_steps),
                training.min_logical_block_size,
                training.seed,
            )
            .with_initial_consumed_steps(resume_consumed_steps)
            .with_ruliad_policy_supervision(training.ruliad_supervision)
            .with_ruliad_policy_stratified_difficulty_levels(
                training
                    .ruliad_supervision
                    .proof_policy
                    .stratified_difficulty_levels,
            )
            .with_summary_event_token_ids(summary_event_token_ids.clone()),
        )
    } else {
        Arc::new(
            RandomDataLoader::<B>::new(
                Arc::clone(&datasets.train),
                DatasetSplit::Train,
                &device,
                steps_per_epoch,
                Some(total_steps),
            )
            .with_seed(training.seed)
            .with_initial_consumed_steps(resume_consumed_steps)
            .with_ruliad_policy_supervision(training.ruliad_supervision)
            .with_ruliad_policy_stratified_difficulty_levels(
                training
                    .ruliad_supervision
                    .proof_policy
                    .stratified_difficulty_levels,
            )
            .with_summary_event_token_ids(summary_event_token_ids.clone()),
        )
    };

    let val_steps_per_epoch = datasets.valid.steps_per_epoch(DatasetSplit::Val);
    let valid_steps =
        resolve_valid_steps_per_epoch(steps_per_epoch, training.log_frequency, val_steps_per_epoch);
    info!(
        "validation schedule: steps_per_epoch={valid_steps} logical_train_steps_per_epoch={steps_per_epoch} sampling={:?} seed={}",
        training.validation.sampling, training.validation.seed,
    );
    let valid_loader: Arc<dyn DataLoader<B, SequenceBatch<B>>> = Arc::new(
        RandomDataLoader::<B>::new(
            Arc::clone(&datasets.valid),
            DatasetSplit::Val,
            &device,
            valid_steps,
            None,
        )
        .with_seed(training.validation.seed)
        .with_source_selection_enabled(training.validation.sampling.uses_live_source_selection())
        .with_summary_event_token_ids(summary_event_token_ids.clone()),
    );

    let mut base_model = DragonModel::<B>::new(model_config.clone(), &device);
    initialize_model_from_checkpoint(
        &resolved_config,
        training,
        &mut base_model,
        &device,
        backend_name,
    )?;
    ensure_random_scaffold_run_manifest(&base_model, &run_dir, parallel_runtime.is_primary())?;
    if let Some(report) = base_model.random_scaffold_report() {
        info!(
            "random scaffold: identity={} frozen_projection_params={} trainable_adapter_params={} fp32_full_bytes={} fp32_adapter_bytes={} byte_reduction={:.2}%",
            report
                .manifest
                .canonical_identity()
                .context("random scaffold identity")?,
            report.frozen_scaffold_elements,
            report.trainable_adapter_elements,
            report.fp32_size.full_model_bytes(),
            report.fp32_size.adapter_payload_bytes(),
            report.fp32_size.byte_reduction_fraction() * 100.0,
        );
    }
    let ruliad_policy_telemetry_path = training
        .ruliad_supervision
        .verifier_reward
        .enabled
        .then(|| run_dir.join("events").join("ruliad_verifier_policy.jsonl"));
    let ruliad_structured_recovery_telemetry_path =
        (training.ruliad_supervision.answer_denoising.enabled
            && training
                .ruliad_supervision
                .answer_denoising
                .structured_recovery_weight
                > f32::EPSILON)
            .then(|| {
                run_dir
                    .join("events")
                    .join("ruliad_structured_recovery.jsonl")
            });
    let ruliad_answer_contract_telemetry_path =
        (training.ruliad_supervision.answer_contract.enabled
            && training.ruliad_supervision.answer_contract.weight > f32::EPSILON)
            .then(|| run_dir.join("events").join("ruliad_answer_contract.jsonl"));
    let ruliad_structured_contrast_telemetry_path = training
        .ruliad_supervision
        .verifier_reward
        .enabled
        .then(|| {
            run_dir
                .join("events")
                .join("ruliad_structured_contrast.jsonl")
        });
    let ruliad_field_binding_contrast_telemetry_path =
        (training.ruliad_supervision.verifier_reward.enabled
            && training
                .ruliad_supervision
                .verifier_reward
                .field_binding_contrast_weight
                > f32::EPSILON)
            .then(|| {
                run_dir
                    .join("events")
                    .join("ruliad_field_binding_contrast.jsonl")
            });
    let ruliad_generated_attractor_telemetry_path =
        (training.ruliad_supervision.verifier_reward.enabled
            && training
                .ruliad_supervision
                .verifier_reward
                .generated_attractor_replay_capacity
                > 0)
        .then(|| {
            run_dir
                .join("events")
                .join("ruliad_generated_attractor_replay.jsonl")
        });
    let prepared_model = LanguageTrainModel::new(base_model)
        .with_training_objectives(training)
        .with_ruliad_policy_telemetry_path(ruliad_policy_telemetry_path)
        .with_ruliad_structured_recovery_telemetry_path(ruliad_structured_recovery_telemetry_path)
        .with_ruliad_answer_contract_telemetry_path(ruliad_answer_contract_telemetry_path)
        .with_ruliad_structured_contrast_telemetry_path(ruliad_structured_contrast_telemetry_path)
        .with_ruliad_field_binding_contrast_telemetry_path(
            ruliad_field_binding_contrast_telemetry_path,
        )
        .with_ruliad_generated_attractor_telemetry_path(ruliad_generated_attractor_telemetry_path)
        .with_tbptt_chunk_size(training.tbptt_chunk_size);
    let model = prepared_model;
    let eggroll_chunk_autotune =
        autotune_eggroll_population_chunk_size(&resolved_config.optimizer, &model, &train_loader)
            .context("autotune EGGROLL population chunk size")?;
    if let Some(report) = &eggroll_chunk_autotune {
        resolved_config
            .optimizer
            .eggroll
            .population
            .population_chunk_size = report.selected_population_chunk_size;
        info!(
            "eggroll chunk autotune: selected_population_chunk_size={} configured_population_chunk_size={} population_size={} candidates={}",
            report.selected_population_chunk_size,
            report.configured_population_chunk_size,
            report.population_size,
            report
                .candidates
                .iter()
                .map(|candidate| format!(
                    "{}:{}evals:{:.1}ms:{:.2}eval/s",
                    candidate.population_chunk_size,
                    candidate.evaluated_population_size,
                    candidate.elapsed_ms,
                    candidate.forward_evaluations_per_second
                ))
                .collect::<Vec<_>>()
                .join(",")
        );
    }
    let optimizer_cfg = &resolved_config.optimizer;

    if parallel_runtime.is_primary() {
        fs::create_dir_all(&run_dir).with_context(|| {
            format!(
                "failed to create run directory before writing artifacts: {}",
                run_dir.display()
            )
        })?;
        write_latest_run(&run_root, &run_name)?;
        if let Some(report) = &eggroll_chunk_autotune {
            let path = run_dir.join("eggroll_chunk_autotune.json");
            fs::write(
                &path,
                serde_json::to_string_pretty(report).context("serialize eggroll chunk autotune")?,
            )
            .with_context(|| {
                format!("write EGGROLL chunk autotune report to {}", path.display())
            })?;
        }
        write_run_config(
            &resolved_config,
            &model_config,
            &run_dir,
            &run_name,
            backend_name,
            effective_training_sequence_kernel_override,
            startup_autotune.as_ref(),
        )?;
        write_training_snapshot(&resolved_config, &run_dir, dataset.tokenizer().as_ref())?;
    }
    maybe_write_pre_step_validation_report_forward_only(
        training,
        &parallel_runtime,
        &run_dir,
        &model,
        &valid_loader,
    )?;
    info!("run name: {run_name}");
    if let Some(report) = &startup_autotune {
        info!(
            "startup autotune: source={} backend={} backend_mode=eggroll_forward_only target_device_memory_mb={} resolved_batch_size={} resolved_gradient_accumulation_steps={} resolved_effective_batch_size={} eggroll_gradient_accumulation_steps=1 probes={}",
            report.config_source,
            report.backend_name,
            report.target_device_memory_mb,
            report.resolved_batch_size,
            report.resolved_gradient_accumulation_steps,
            report.resolved_effective_batch_size,
            report
                .probes
                .iter()
                .map(|probe| match (probe.reserved_mb, probe.in_use_mb) {
                    (Some(reserved), Some(in_use)) => format!(
                        "bs{}:{}:{reserved:.1}/{in_use:.1}MiB",
                        probe.batch_size, probe.status
                    ),
                    (Some(reserved), None) =>
                        format!("bs{}:{}:{reserved:.1}MiB", probe.batch_size, probe.status),
                    _ => format!("bs{}:{}", probe.batch_size, probe.status),
                })
                .collect::<Vec<_>>()
                .join(",")
        );
    }
    if let Some(report) = &eggroll_auto_population {
        info!(
            "eggroll auto population: batch_size={} population_per_batch={} configured_population_size={} configured_chunk_size={} rank={} min_population_size={} max_population_size={} resolved_population_size={} resolved_chunk_size={} forward_evaluations_per_update={}",
            report.batch_size,
            report.population_per_batch,
            report.configured_population_size,
            report.configured_population_chunk_size,
            report.configured_rank,
            report.min_population_size,
            report.max_population_size,
            report.resolved_population_size,
            report.resolved_population_chunk_size,
            report.resolved_population_size,
        );
    }
    info!(
        "training batching: micro_batch_size={} gradient_accumulation_steps={} effective_batch_size={} tbptt_chunk_size={} tbptt_persist_across_steps={} sequence_batching={:?} streaming_loader={} min_logical_block_size={}",
        resolved_config.training.batch_size,
        resolved_config.training.gradient_accumulation_steps,
        resolved_config
            .training
            .batch_size
            .saturating_mul(resolved_config.training.gradient_accumulation_steps),
        resolved_config
            .training
            .tbptt_chunk_size
            .map(|value| value.to_string())
            .unwrap_or_else(|| "disabled".to_string()),
        resolved_config.training.tbptt_persist_across_steps,
        resolved_config.training.sequence_batching,
        resolved_config
            .training
            .sequence_batching
            .uses_streaming_loader(resolved_config.training.tbptt_persist_across_steps),
        resolved_config
            .training
            .min_logical_block_size
            .map(|value| value.to_string())
            .unwrap_or_else(|| "disabled".to_string())
    );
    info!(
        "optimizer fingerprint: name={:?} backend_mode=eggroll_forward_only autodiff=false learning_rate={} weight_decay={} eggroll_interval_steps={} eggroll_population_size={} eggroll_rank={}",
        optimizer_cfg.name,
        optimizer_cfg.learning_rate,
        optimizer_cfg.weight_decay,
        optimizer_cfg.eggroll.interval_steps,
        optimizer_cfg.eggroll.population.population_size,
        optimizer_cfg.eggroll.population.rank,
    );

    let context = ForwardEggrollTrainEnvironment {
        parallel_runtime: &parallel_runtime,
        run_dir: &run_dir,
        run_name: &run_name,
        backend_name,
        training,
        resume_checkpoint_epoch,
        model_config: &model_config,
        device: &device,
        train_loader,
        valid_loader,
        source_selection_dataset: datasets
            .train
            .uses_live_source_selection()
            .then(|| Arc::clone(&datasets.train)),
        summary_event_token_ids,
        epochs: total_epochs,
    };
    let _model = train_with_eggroll_forward_only(&context, optimizer_cfg, model)?;

    info!("Training complete on {backend_name} with EGGROLL forward-only backend");
    Ok(())
}

pub fn train_backend<B, Init>(
    config: &TrainingConfig,
    dataset: Arc<Dataset>,
    backend_name: &str,
    init_backend: Init,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone + 'static,
    Init: Fn(&B::Device),
{
    if optimizer_uses_forward_only_eggroll(config) {
        return Err(anyhow!(
            "training.algorithm=eggroll must use train_backend_forward_eggroll so autodiff is not enabled"
        ));
    }

    let stage_profile = crate::train::profile::enabled();

    let parallel_runtime = resolve_parallel_runtime(&config.parallel)?;
    info!("parallel runtime: {}", parallel_runtime.summary());

    let primary_device = B::Device::default();
    let devices = resolve_training_devices::<B>(&parallel_runtime, &primary_device)?;
    for device in &devices {
        B::seed(device, config.training.seed);
        init_backend(device);
    }
    let device = devices
        .first()
        .cloned()
        .expect("at least one training device");
    info!("resolved training devices: {}", devices.len());

    let mut resolved_config = config.clone();
    let run_root = resolve_run_root();
    let (run_dir, run_name) =
        resolve_run_artifacts(&parallel_runtime, &run_root, &resolved_config.training)?;
    let resume_checkpoint_epoch =
        resolve_resume_checkpoint_epoch(&resolved_config.training, &run_dir)?;
    let resume_source_selection = configure_resume_source_selection_state(
        &mut resolved_config,
        &dataset,
        &run_dir,
        resume_checkpoint_epoch,
    )?;
    let startup_autotune =
        resolve_startup_batch_size::<B>(&resolved_config, &dataset, backend_name, &device)?;
    if let Some(report) = &startup_autotune {
        resolved_config.training.batch_size = report.resolved_batch_size;
        resolved_config.training.gradient_accumulation_steps =
            report.resolved_gradient_accumulation_steps;
    }
    if startup_autotune.is_none() {
        resolved_config.training.gradient_accumulation_steps = resolve_gradient_accumulation_steps(
            resolved_config.training.batch_size,
            resolved_config.training.gradient_accumulation_steps,
            resolved_config.training.target_effective_batch_size,
        );
    }
    if matches!(resolved_config.optimizer.name, OptimizerKind::Eggroll) {
        resolved_config.optimizer.apply_effective_eggroll_config();
    }
    validate_resolved_optimizer_owned_batching(&resolved_config)?;

    let datasets = if resolved_config.training.batch_size == config.training.batch_size
        && !resume_source_selection
    {
        crate::train::utils::PreparedDatasets {
            train: Arc::clone(&dataset),
            valid: Arc::clone(&dataset),
        }
    } else {
        crate::train::utils::prepare_datasets(&resolved_config.dataset, &resolved_config.training)?
    };

    let training = &resolved_config.training;
    let optimizer_cfg = &resolved_config.optimizer;
    let training_kernel_block_size =
        crate::train::utils::effective_training_kernel_block_size(training);

    let tokenizer = datasets.train.tokenizer();
    let mut model_config = build_model_config_with_tokenizer(
        &resolved_config.model,
        training_kernel_block_size,
        tokenizer.as_ref(),
    )?;
    let configured_sequence_kernel = model_config.sequence_kernel;
    let (effective_sequence_kernel, effective_training_sequence_kernel_override, promotion_reason) =
        resolve_effective_training_sequence_kernel(
            configured_sequence_kernel,
            training.sequence_kernel_override,
            backend_name,
            training_kernel_block_size,
        );
    model_config.sequence_kernel = effective_sequence_kernel;
    apply_wgpu_fused_core_override(
        &mut model_config,
        backend_name,
        WgpuFusedCoreOverride {
            recurrent: resolved_config.wgpu.training.fused_core_recurrent,
            rollout: resolved_config.wgpu.training.fused_core_rollout,
        },
    );
    let (backend_mode, autodiff_graph) = if training_uses_local_predictive_coding(&resolved_config)
    {
        ("predictive_coding_local_vjp", false)
    } else {
        ("global_backpropagation", true)
    };
    info!(
        "training path fingerprint: backend={} backend_mode={} autodiff_graph={} execution_form={} launch_mode={:?} effective_sequence_kernel={:?} sequence_kernel_override={:?} tbptt_chunk_size={:?} kernel_block_size={} pipeline_enabled={}",
        backend_name,
        backend_mode,
        autodiff_graph,
        build_training_execution_form(&resolved_config),
        training.launch_mode,
        model_config.sequence_kernel,
        effective_training_sequence_kernel_override,
        training.tbptt_chunk_size,
        training_kernel_block_size,
        resolved_config.parallel.pipeline.enabled,
    );
    if let Some(reason) = promotion_reason {
        info!(
            "training sequence kernel promotion: configured={:?} effective={:?} reason={reason}",
            configured_sequence_kernel, model_config.sequence_kernel,
        );
    }
    if backend_name.eq_ignore_ascii_case("cuda") && model_config.fused_kernels.enabled {
        warn!(
            "cuda language training still mixes burn_dragon_kernel fused kernels with generic Burn tensor ops; only selected recurrent/projection paths are accelerated today"
        );
    }
    if backend_name.eq_ignore_ascii_case("cuda")
        && matches!(
            model_config.sequence_kernel.memory_system,
            SequenceMemorySystem::Mamba3StateSpaceDuality
        )
    {
        if let Some(summary) = cuda_mamba_training_geometry_summary(
            &model_config,
            resolved_config.training.batch_size,
            training_kernel_block_size,
        ) {
            info!("{summary}");
        }
        warn!(
            "cuda mamba3 training defaults to the tensorized custom analytical backward wrapper over the chunked SISO path; set BURN_DRAGON_MAMBA3_CUDA_TENSORIZED_TRAIN_WRAPPER=0 to force the direct graph baseline"
        );
    }
    if matches!(
        model_config.sequence_kernel.memory_system,
        SequenceMemorySystem::GatedDeltaNet2
    ) {
        info!(
            "gated_deltanet2 config: chunk_size={} qk_l2_norm={} erase_gate={:?} write_gate={:?} decay_gate={:?} allow_neg_eigval={}",
            model_config.gated_deltanet2.chunk_size,
            model_config.gated_deltanet2.qk_l2_norm,
            model_config.gated_deltanet2.erase_gate,
            model_config.gated_deltanet2.write_gate,
            model_config.gated_deltanet2.decay_gate,
            model_config.gated_deltanet2.allow_neg_eigval,
        );
        if matches!(
            model_config.sequence_kernel.executor,
            SequenceTrainingExecutor::GatedDeltaChunkWy
        ) {
            let message = "gated_deltanet2 chunk-WY executor uses the CUDA fused WY solver/backward on CUDA autodiff calls; WGPU autodiff uses the chunked analytic wrapper, and non-autodiff eval calls use the direct recurrence";
            info!("{message}");
            eprintln!("{message}");
        }
    }
    let pipeline_plan = if resolved_config.parallel.pipeline.enabled {
        let pipeline_plan =
            build_pipeline_plan(model_config.n_layer, &resolved_config.parallel.pipeline)?;
        info!("resolved pipeline plan: {}", pipeline_plan.summary());
        if resolved_config.parallel.pipeline.communication
            == burn_dragon_train::PipelineCommunicationKind::BlockResidualCache
            && resolved_config.model.residual_connector
                == Some(burn_dragon_core::ResidualConnectorKind::BlockAttentionResidual)
        {
            let layers_per_block = resolved_config
                .model
                .block_attention_residual
                .as_ref()
                .map(|cfg| cfg.layers_per_block.max(1))
                .unwrap_or(1);
            let payload_bytes = model_config
                .n_embd
                .saturating_mul(training.block_size.max(1))
                .saturating_mul(std::mem::size_of::<f32>());
            let communication = simulate_pipeline_communication(
                &pipeline_plan,
                resolved_config.parallel.pipeline.communication,
                &resolved_config.parallel.pipeline.cache,
                layers_per_block,
                payload_bytes,
            )?;
            info!(
                "resolved pipeline communication: requested_bytes={} transmitted_bytes={} bytes_saved={} cache_hits={} cache_misses={} backward_reuse_hits={} hit_rate={:.3}",
                communication.raw_payload_bytes_requested,
                communication.payload_bytes_transmitted,
                communication.bytes_saved(),
                communication.cache_hits,
                communication.cache_misses,
                communication.backward_reuse_hits,
                communication.cache_hit_rate(),
            );
            if parallel_runtime.mode != ParallelismKind::Single {
                warn!(
                    "parallel.pipeline.communication=block_residual_cache currently reports simulated savings, but live distributed pipeline transport still sends full pipeline states until compressed block-residual transport is implemented"
                );
            }
        }
        if parallel_runtime.mode != ParallelismKind::Single {
            let layout =
                resolve_pipeline_parallel_layout(&parallel_runtime, &resolved_config.parallel)?
                    .ok_or_else(|| {
                        anyhow!("parallel.pipeline.enabled requires a resolved DDP pipeline layout")
                    })?;
            let assignment = layout.assignment(parallel_runtime.global_rank).clone();
            let workload = build_pipeline_rank_workload(
                &pipeline_plan,
                assignment.global_rank,
                assignment.pipeline_stage_id,
                assignment.data_parallel_rank,
            );
            info!(
                "resolved distributed pipeline rank workload: {} rank={} stage={} dp_rank={} assignments={} forward_events={} backward_events={}",
                layout.summary(),
                assignment.global_rank,
                assignment.pipeline_stage_id,
                assignment.data_parallel_rank,
                workload.stage_assignments.len(),
                workload.forward_events.len(),
                workload.backward_events.len(),
            );
            if parallel_runtime.mode != ParallelismKind::Ddp
                || !parallel_runtime.is_process_group_launch()
            {
                return Err(anyhow!(
                    "parallel.pipeline.enabled distributed execution currently requires a process-group DDP launch"
                ));
            }
        }
        if training.tbptt_chunk_size.is_some() || training.tbptt_persist_across_steps {
            return Err(anyhow!(
                "parallel.pipeline.enabled does not yet support tbptt chunking or persistent stream state"
            ));
        }
        if model_config.rollout_fast_steps_per_slow_step != 1 {
            return Err(anyhow!(
                "parallel.pipeline.enabled requires rollout_fast_steps_per_slow_step = 1 (got {})",
                model_config.rollout_fast_steps_per_slow_step
            ));
        }
        if model_config.y_neuron_recurrence.enabled {
            return Err(anyhow!(
                "parallel.pipeline.enabled does not yet support y_neuron_recurrence"
            ));
        }
        Some(pipeline_plan)
    } else {
        None
    };
    let objective_trainer = if resolved_config.parallel.pipeline.enabled {
        ObjectiveTrainerKind::Pipeline
    } else {
        match parallel_runtime.mode {
            ParallelismKind::Single => ObjectiveTrainerKind::SingleDevice,
            ParallelismKind::Ddp => ObjectiveTrainerKind::Ddp,
            mode => {
                return Err(anyhow!(
                    "parallel.mode={mode:?} is not wired into objective-aware language training yet"
                ));
            }
        }
    };
    ensure_objective_supported(&resolved_config.training.objective, objective_trainer)?;
    ensure_rollout_objective_runtime(
        &resolved_config.training.objective,
        RolloutObjectiveRuntimeConstraints {
            uses_flat_token_logits: model_config.language_head.uses_flat_token_logits(),
            distributed_pipeline: resolved_config.parallel.pipeline.enabled
                && parallel_runtime.mode != ParallelismKind::Single,
            tbptt_enabled: training.tbptt_chunk_size.is_some()
                || training.tbptt_persist_across_steps,
        },
    )?;
    let summary_event_token_ids = model_config.summary_memory.write_trigger_token_ids.clone();

    let dataset_steps_per_epoch = datasets.train.steps_per_epoch(DatasetSplit::Train);
    let checkpoint_steps_per_epoch =
        resolve_checkpoint_steps_per_epoch(training, dataset_steps_per_epoch);
    let schedule = resolve_train_schedule(training, checkpoint_steps_per_epoch)?;
    let steps_per_epoch = schedule.steps_per_epoch;
    let total_epochs = schedule.total_epochs;
    let total_steps = schedule.total_steps;
    let resume_consumed_steps = resume_checkpoint_epoch
        .unwrap_or_default()
        .saturating_mul(steps_per_epoch);

    info!(
        "train schedule: dataset_steps_per_epoch={dataset_steps_per_epoch}, logical_steps_per_epoch={steps_per_epoch}, checkpoint_interval_iters={}, total_steps={total_steps}, epochs={total_epochs}, source={}",
        training.checkpoint_interval_iters,
        schedule.source.as_str()
    );
    let train_loader: Arc<dyn DataLoader<B, SequenceBatch<B>>> = if training
        .sequence_batching
        .uses_streaming_loader(training.tbptt_persist_across_steps)
    {
        Arc::new(
            StreamingDataLoader::<B>::new(
                Arc::clone(&datasets.train),
                DatasetSplit::Train,
                &device,
                steps_per_epoch,
                Some(total_steps),
                training.min_logical_block_size,
                training.seed,
            )
            .with_initial_consumed_steps(resume_consumed_steps)
            .with_ruliad_policy_supervision(training.ruliad_supervision)
            .with_ruliad_policy_stratified_difficulty_levels(
                training
                    .ruliad_supervision
                    .proof_policy
                    .stratified_difficulty_levels,
            )
            .with_summary_event_token_ids(summary_event_token_ids.clone()),
        )
    } else {
        Arc::new(
            RandomDataLoader::<B>::new(
                Arc::clone(&datasets.train),
                DatasetSplit::Train,
                &device,
                steps_per_epoch,
                Some(total_steps),
            )
            .with_seed(training.seed)
            .with_initial_consumed_steps(resume_consumed_steps)
            .with_ruliad_policy_supervision(training.ruliad_supervision)
            .with_ruliad_policy_stratified_difficulty_levels(
                training
                    .ruliad_supervision
                    .proof_policy
                    .stratified_difficulty_levels,
            )
            .with_summary_event_token_ids(summary_event_token_ids.clone()),
        )
    };

    let val_steps_per_epoch = datasets.valid.steps_per_epoch(DatasetSplit::Val);
    let valid_steps =
        resolve_valid_steps_per_epoch(steps_per_epoch, training.log_frequency, val_steps_per_epoch);
    info!(
        "validation schedule: steps_per_epoch={valid_steps} logical_train_steps_per_epoch={steps_per_epoch} sampling={:?} seed={}",
        training.validation.sampling, training.validation.seed,
    );

    let valid_device = device.clone();
    let valid_loader: Arc<dyn DataLoader<ValidBackend<B>, SequenceBatch<ValidBackend<B>>>> =
        Arc::new(
            RandomDataLoader::<ValidBackend<B>>::new(
                Arc::clone(&datasets.valid),
                DatasetSplit::Val,
                &valid_device,
                valid_steps,
                None,
            )
            .with_seed(training.validation.seed)
            .with_source_selection_enabled(
                training.validation.sampling.uses_live_source_selection(),
            )
            .with_summary_event_token_ids(summary_event_token_ids.clone()),
        );

    let mut base_model = DragonModel::<B>::new(model_config.clone(), &device);
    let fresh_model = base_model.clone();
    initialize_model_from_checkpoint(
        &resolved_config,
        training,
        &mut base_model,
        &device,
        backend_name,
    )?;
    ensure_random_scaffold_run_manifest(&base_model, &run_dir, parallel_runtime.is_primary())?;
    if let Some(report) = base_model.random_scaffold_report() {
        info!(
            "random scaffold: identity={} frozen_projection_params={} trainable_adapter_params={} fp32_full_bytes={} fp32_adapter_bytes={} byte_reduction={:.2}%",
            report
                .manifest
                .canonical_identity()
                .context("random scaffold identity")?,
            report.frozen_scaffold_elements,
            report.trainable_adapter_elements,
            report.fp32_size.full_model_bytes(),
            report.fp32_size.adapter_payload_bytes(),
            report.fp32_size.byte_reduction_fraction() * 100.0,
        );
    }
    validate_dragon_continual_backprop(training, &base_model, parallel_runtime.world_size)?;
    let ruliad_policy_telemetry_path = training
        .ruliad_supervision
        .verifier_reward
        .enabled
        .then(|| run_dir.join("events").join("ruliad_verifier_policy.jsonl"));
    let ruliad_structured_recovery_telemetry_path =
        (training.ruliad_supervision.answer_denoising.enabled
            && training
                .ruliad_supervision
                .answer_denoising
                .structured_recovery_weight
                > f32::EPSILON)
            .then(|| {
                run_dir
                    .join("events")
                    .join("ruliad_structured_recovery.jsonl")
            });
    let ruliad_answer_contract_telemetry_path =
        (training.ruliad_supervision.answer_contract.enabled
            && training.ruliad_supervision.answer_contract.weight > f32::EPSILON)
            .then(|| run_dir.join("events").join("ruliad_answer_contract.jsonl"));
    let ruliad_structured_contrast_telemetry_path = training
        .ruliad_supervision
        .verifier_reward
        .enabled
        .then(|| {
            run_dir
                .join("events")
                .join("ruliad_structured_contrast.jsonl")
        });
    let ruliad_field_binding_contrast_telemetry_path =
        (training.ruliad_supervision.verifier_reward.enabled
            && training
                .ruliad_supervision
                .verifier_reward
                .field_binding_contrast_weight
                > f32::EPSILON)
            .then(|| {
                run_dir
                    .join("events")
                    .join("ruliad_field_binding_contrast.jsonl")
            });
    let ruliad_generated_attractor_telemetry_path =
        (training.ruliad_supervision.verifier_reward.enabled
            && training
                .ruliad_supervision
                .verifier_reward
                .generated_attractor_replay_capacity
                > 0)
        .then(|| {
            run_dir
                .join("events")
                .join("ruliad_generated_attractor_replay.jsonl")
        });
    let ruliad_verifier_rollout_telemetry_path =
        (training.ruliad_supervision.verifier_reward.enabled
            && (training
                .ruliad_supervision
                .verifier_reward
                .rollout_imitation_weight
                > f32::EPSILON
                || training
                    .ruliad_supervision
                    .verifier_reward
                    .rollout_recovery_weight
                    > f32::EPSILON))
            .then(|| {
                run_dir
                    .join("events")
                    .join("ruliad_verifier_rollout_imitation.jsonl")
            });
    let ruliad_proof_policy_telemetry_path =
        training.ruliad_supervision.proof_policy.enabled.then(|| {
            run_dir
                .join("events")
                .join("ruliad_proof_policy_dagger.jsonl")
        });
    let prepared_model = LanguageTrainModel::new(base_model)
        .with_training_configuration(training, total_steps)
        .with_ruliad_policy_telemetry_path(ruliad_policy_telemetry_path)
        .with_ruliad_structured_recovery_telemetry_path(ruliad_structured_recovery_telemetry_path)
        .with_ruliad_answer_contract_telemetry_path(ruliad_answer_contract_telemetry_path)
        .with_ruliad_structured_contrast_telemetry_path(ruliad_structured_contrast_telemetry_path)
        .with_ruliad_field_binding_contrast_telemetry_path(
            ruliad_field_binding_contrast_telemetry_path,
        )
        .with_ruliad_generated_attractor_telemetry_path(ruliad_generated_attractor_telemetry_path)
        .with_ruliad_verifier_rollout_telemetry_path(ruliad_verifier_rollout_telemetry_path)
        .with_ruliad_proof_policy_telemetry_path(ruliad_proof_policy_telemetry_path)
        .with_pipeline_plan(pipeline_plan.clone());
    let model = prepared_model;
    let optim =
        resolve_dragon_language_optimizer::<B>(training, optimizer_cfg, total_steps, fresh_model)?;
    let scheduler_iters = match schedule.source {
        ScheduleSource::Epochs => Some(total_steps),
        ScheduleSource::MaxIters => None,
    };
    let scheduler =
        resolve_lr_scheduler(optimizer_cfg, total_steps, scheduler_iters, &model_config)?;
    if parallel_runtime.is_primary() {
        write_latest_run(&run_root, &run_name)?;
        write_run_config(
            &resolved_config,
            &model_config,
            &run_dir,
            &run_name,
            backend_name,
            effective_training_sequence_kernel_override,
            startup_autotune.as_ref(),
        )?;
        write_training_snapshot(&resolved_config, &run_dir, dataset.tokenizer().as_ref())?;
    }
    maybe_write_pre_step_validation_report(
        training,
        &parallel_runtime,
        &run_dir,
        &model,
        &valid_loader,
    )?;
    info!("run name: {run_name}");
    if let Some(report) = &startup_autotune {
        info!(
            "startup autotune: source={} backend={} target_device_memory_mb={} resolved_batch_size={} resolved_gradient_accumulation_steps={} resolved_effective_batch_size={} probes={}",
            report.config_source,
            report.backend_name,
            report.target_device_memory_mb,
            report.resolved_batch_size,
            report.resolved_gradient_accumulation_steps,
            report.resolved_effective_batch_size,
            report
                .probes
                .iter()
                .map(|probe| match (probe.reserved_mb, probe.in_use_mb) {
                    (Some(reserved), Some(in_use)) => format!(
                        "bs{}:{}:{reserved:.1}/{in_use:.1}MiB",
                        probe.batch_size, probe.status
                    ),
                    (Some(reserved), None) =>
                        format!("bs{}:{}:{reserved:.1}MiB", probe.batch_size, probe.status),
                    _ => format!("bs{}:{}", probe.batch_size, probe.status),
                })
                .collect::<Vec<_>>()
                .join(",")
        );
    }
    info!(
        "training batching: micro_batch_size={} gradient_accumulation_steps={} effective_batch_size={} tbptt_chunk_size={} tbptt_persist_across_steps={} sequence_batching={:?} streaming_loader={} min_logical_block_size={}",
        resolved_config.training.batch_size,
        resolved_config.training.gradient_accumulation_steps,
        resolved_config
            .training
            .batch_size
            .saturating_mul(resolved_config.training.gradient_accumulation_steps),
        resolved_config
            .training
            .tbptt_chunk_size
            .map(|value| value.to_string())
            .unwrap_or_else(|| "disabled".to_string()),
        resolved_config.training.tbptt_persist_across_steps,
        resolved_config.training.sequence_batching,
        resolved_config
            .training
            .sequence_batching
            .uses_streaming_loader(resolved_config.training.tbptt_persist_across_steps),
        resolved_config
            .training
            .min_logical_block_size
            .map(|value| value.to_string())
            .unwrap_or_else(|| "disabled".to_string())
    );
    info!(
        "optimizer fingerprint: name={:?} schedule_mode={:?} learning_rate={} weight_decay={} weight_decay_final={:?} module_lr_scales={:?} continual_backprop_lr_coupling={:?} continual_backprop_lr_coupling_power={}",
        optimizer_cfg.name,
        optimizer_cfg.schedule_mode,
        optimizer_cfg.learning_rate,
        optimizer_cfg.weight_decay,
        optimizer_cfg.weight_decay_final,
        training.module_lr_scales,
        training.continual_backprop.lr_coupling,
        training.continual_backprop.lr_coupling_power,
    );
    let neuron_scaling_slot = training
        .neuron_scaling
        .enabled
        .then(crate::train::neuron_scaling::NeuronScaleRequestSlot::default);
    let context = TrainEnvironment {
        parallel_runtime: &parallel_runtime,
        parallel_config: &resolved_config.parallel,
        run_dir: &run_dir,
        run_name: &run_name,
        backend_name,
        training,
        resume_checkpoint_epoch,
        model_config: &model_config,
        device: &device,
        devices: &devices,
        train_dataset: Some(Arc::clone(&datasets.train)),
        valid_dataset: Some(Arc::clone(&datasets.valid)),
        train_loader,
        valid_loader,
        source_selection_dataset: datasets
            .train
            .uses_live_source_selection()
            .then(|| Arc::clone(&datasets.train)),
        summary_event_token_ids,
        neuron_scaling_slot: neuron_scaling_slot.clone(),
        epochs: total_epochs,
        total_steps,
        valid_steps,
    };
    if stage_profile {
        crate::train::profile::reset();
    }
    let train_wall_start = stage_profile.then(Instant::now);
    let _model = train_with_resolved_scheduler(&context, model, optim, scheduler)?;

    info!("Training complete on {backend_name}");

    if let Some(start) = train_wall_start {
        let elapsed_ns = start.elapsed().as_nanos();
        let snapshot = crate::train::profile::snapshot();
        let train_tokens = (snapshot.train_steps as u128)
            .saturating_mul(resolved_config.training.batch_size as u128)
            .saturating_mul(resolved_config.training.block_size as u128);
        let main_model_step_ns = snapshot
            .forward_ns
            .saturating_add(snapshot.loss_backward_ns)
            .saturating_add(snapshot.local_learning_ns);
        let model_step_ns = main_model_step_ns.saturating_add(snapshot.auxiliary_objective_ns);
        let train_compute_ns = model_step_ns.saturating_add(snapshot.optimizer_ns);
        let accounted_ns = train_compute_ns
            .saturating_add(snapshot.metric_sync_ns)
            .saturating_add(snapshot.validation_ns)
            .saturating_add(snapshot.checkpoint_ns)
            .saturating_add(snapshot.dataloader_foreground_wait_ns);
        let wall_tokens_per_second = tokens_per_second(train_tokens, elapsed_ns);
        let model_tokens_per_second = tokens_per_second(train_tokens, model_step_ns);
        let main_model_tokens_per_second = tokens_per_second(train_tokens, main_model_step_ns);
        let model_duty_fraction = fraction(model_step_ns, elapsed_ns);
        let auxiliary_objective_fraction = fraction(snapshot.auxiliary_objective_ns, elapsed_ns);
        let proof_policy_fraction = fraction(snapshot.proof_policy_ns, elapsed_ns);
        let train_compute_fraction = fraction(train_compute_ns, elapsed_ns);
        let optimizer_fraction = fraction(snapshot.optimizer_ns, elapsed_ns);
        let metric_sync_fraction = fraction(snapshot.metric_sync_ns, elapsed_ns);
        let accounted_fraction = fraction(accounted_ns, elapsed_ns);
        let validation_fraction = fraction(snapshot.validation_ns, elapsed_ns);
        let checkpoint_fraction = fraction(snapshot.checkpoint_ns, elapsed_ns);
        let dataloader_wall_fraction = if elapsed_ns == 0 {
            0.0
        } else {
            snapshot.dataloader_cpu_ns as f64 / elapsed_ns as f64
        };
        let dataloader_foreground_wait_fraction = if elapsed_ns == 0 {
            0.0
        } else {
            snapshot.dataloader_foreground_wait_ns as f64 / elapsed_ns as f64
        };
        info!(
            "[stage-profile][training] total_ns={elapsed_ns} train_tokens={train_tokens} wall_tokens_per_second={wall_tokens_per_second:.3} model_tokens_per_second={model_tokens_per_second:.3} main_model_tokens_per_second={main_model_tokens_per_second:.3} model_duty_fraction={model_duty_fraction:.6} auxiliary_objective_fraction={auxiliary_objective_fraction:.6} proof_policy_fraction={proof_policy_fraction:.6} train_compute_fraction={train_compute_fraction:.6} optimizer_fraction={optimizer_fraction:.6} metric_sync_fraction={metric_sync_fraction:.6} accounted_fraction={accounted_fraction:.6} validation_fraction={validation_fraction:.6} checkpoint_fraction={checkpoint_fraction:.6} dataloader_cpu_thread_fraction={dataloader_wall_fraction:.6} dataloader_foreground_wait_fraction={dataloader_foreground_wait_fraction:.6} dataloader_cpu_ns={} dataloader_foreground_wait_ns={} dataloader_tensor_copy_ns={} dataloader_host_to_device_copy_bytes={} host_sync_points={} forward_ns={} local_learning_ns={} auxiliary_objective_ns={} proof_policy_ns={} loss_backward_ns={} optimizer_ns={} metric_sync_ns={} validation_ns={} checkpoint_ns={} embed_probe_ns={} first_layer_forward_probe_ns={} first_layer_probe_ns={} logits_loss_probe_ns={} hidden_logits_loss_probe_ns={} hidden_model_forward_probe_ns={} hidden_model_probe_ns={} detail_probe_steps={} train_steps={} max_step_reserved_before_bytes={} max_step_in_use_before_bytes={} max_step_reserved_after_forward_bytes={} max_step_in_use_after_forward_bytes={} max_step_reserved_after_backward_bytes={} max_step_in_use_after_backward_bytes={}",
            snapshot.dataloader_cpu_ns,
            snapshot.dataloader_foreground_wait_ns,
            snapshot.dataloader_tensor_copy_ns,
            snapshot.dataloader_host_to_device_copy_bytes,
            snapshot.host_sync_points,
            snapshot.forward_ns,
            snapshot.local_learning_ns,
            snapshot.auxiliary_objective_ns,
            snapshot.proof_policy_ns,
            snapshot.loss_backward_ns,
            snapshot.optimizer_ns,
            snapshot.metric_sync_ns,
            snapshot.validation_ns,
            snapshot.checkpoint_ns,
            snapshot.embed_probe_ns,
            snapshot.first_layer_forward_probe_ns,
            snapshot.first_layer_probe_ns,
            snapshot.logits_loss_probe_ns,
            snapshot.hidden_logits_loss_probe_ns,
            snapshot.hidden_model_forward_probe_ns,
            snapshot.hidden_model_probe_ns,
            snapshot.detail_probe_steps,
            snapshot.train_steps,
            snapshot.max_step_reserved_before_bytes,
            snapshot.max_step_in_use_before_bytes,
            snapshot.max_step_reserved_after_forward_bytes,
            snapshot.max_step_in_use_after_forward_bytes,
            snapshot.max_step_reserved_after_backward_bytes,
            snapshot.max_step_in_use_after_backward_bytes,
        );
        eprintln!(
            "[stage-profile][training] total_ns={elapsed_ns} train_tokens={train_tokens} wall_tokens_per_second={wall_tokens_per_second:.3} model_tokens_per_second={model_tokens_per_second:.3} main_model_tokens_per_second={main_model_tokens_per_second:.3} model_duty_fraction={model_duty_fraction:.6} auxiliary_objective_fraction={auxiliary_objective_fraction:.6} proof_policy_fraction={proof_policy_fraction:.6} train_compute_fraction={train_compute_fraction:.6} optimizer_fraction={optimizer_fraction:.6} metric_sync_fraction={metric_sync_fraction:.6} accounted_fraction={accounted_fraction:.6} validation_fraction={validation_fraction:.6} checkpoint_fraction={checkpoint_fraction:.6} dataloader_cpu_thread_fraction={dataloader_wall_fraction:.6} dataloader_foreground_wait_fraction={dataloader_foreground_wait_fraction:.6} dataloader_cpu_ns={} dataloader_foreground_wait_ns={} dataloader_tensor_copy_ns={} dataloader_host_to_device_copy_bytes={} host_sync_points={} forward_ns={} local_learning_ns={} auxiliary_objective_ns={} proof_policy_ns={} loss_backward_ns={} optimizer_ns={} metric_sync_ns={} validation_ns={} checkpoint_ns={} embed_probe_ns={} first_layer_forward_probe_ns={} first_layer_probe_ns={} logits_loss_probe_ns={} hidden_logits_loss_probe_ns={} hidden_model_forward_probe_ns={} hidden_model_probe_ns={} detail_probe_steps={} train_steps={} max_step_reserved_before_bytes={} max_step_in_use_before_bytes={} max_step_reserved_after_forward_bytes={} max_step_in_use_after_forward_bytes={} max_step_reserved_after_backward_bytes={} max_step_in_use_after_backward_bytes={}",
            snapshot.dataloader_cpu_ns,
            snapshot.dataloader_foreground_wait_ns,
            snapshot.dataloader_tensor_copy_ns,
            snapshot.dataloader_host_to_device_copy_bytes,
            snapshot.host_sync_points,
            snapshot.forward_ns,
            snapshot.local_learning_ns,
            snapshot.auxiliary_objective_ns,
            snapshot.proof_policy_ns,
            snapshot.loss_backward_ns,
            snapshot.optimizer_ns,
            snapshot.metric_sync_ns,
            snapshot.validation_ns,
            snapshot.checkpoint_ns,
            snapshot.embed_probe_ns,
            snapshot.first_layer_forward_probe_ns,
            snapshot.first_layer_probe_ns,
            snapshot.logits_loss_probe_ns,
            snapshot.hidden_logits_loss_probe_ns,
            snapshot.hidden_model_forward_probe_ns,
            snapshot.hidden_model_probe_ns,
            snapshot.detail_probe_steps,
            snapshot.train_steps,
            snapshot.max_step_reserved_before_bytes,
            snapshot.max_step_in_use_before_bytes,
            snapshot.max_step_reserved_after_forward_bytes,
            snapshot.max_step_in_use_after_forward_bytes,
            snapshot.max_step_reserved_after_backward_bytes,
            snapshot.max_step_in_use_after_backward_bytes,
        );
    }

    Ok(())
}

fn tokens_per_second(tokens: u128, elapsed_ns: u128) -> f64 {
    if elapsed_ns == 0 {
        return 0.0;
    }
    (tokens as f64) / (elapsed_ns as f64 / 1_000_000_000.0)
}

fn fraction(part: u128, total: u128) -> f64 {
    if total == 0 {
        0.0
    } else {
        part as f64 / total as f64
    }
}
