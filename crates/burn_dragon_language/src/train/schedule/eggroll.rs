//! AdamW and forward-only Eggroll schedule entry points.

use super::*;

pub(crate) fn train_with_scheduler<B, O, S>(
    env: &TrainEnvironment<'_, B>,
    model: LanguageTrainModel<B>,
    optimizer: O,
    scheduler: S,
) -> Result<DragonModel<ValidBackend<B>>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    O: Optimizer<LanguageTrainModel<B>, B> + 'static,
    S: LrScheduler + 'static,
{
    fs::create_dir_all(env.run_dir)?;
    let pc_manifest = prepare_local_predictive_coding_contract(env, &model)?;

    let source_selection_dataset = env.source_selection_dataset.as_ref().cloned();
    let train_loss_metric_every = crate::train::events::train_loss_metric_frequency(
        env.training,
        source_selection_dataset.as_ref(),
    );
    #[cfg(feature = "ddp")]
    if env.parallel_runtime.mode == ParallelismKind::Ddp
        && env.parallel_runtime.is_process_group_launch()
    {
        return train_with_process_group_scheduler(env, model, optimizer, scheduler);
    }
    let training_strategy = match env.parallel_runtime.mode {
        ParallelismKind::Single => {
            LearningStrategy::Default(ExecutionStrategy::single(env.device.clone()))
        }
        ParallelismKind::Ddp => LearningStrategy::Default(ExecutionStrategy::multi(
            env.devices.to_vec(),
            MultiDeviceOptim::OptimMainDevice,
        )),
        mode => {
            return Err(anyhow!(
                "parallel.mode={mode:?} is not wired into language training yet"
            ));
        }
    };
    let event_handles =
        crate::train::events::build_training_event_handles_with_local_predictive_coding(
            env.run_name,
            env.run_dir,
            env.train_loader.num_items(),
            env.training,
            source_selection_dataset,
            env.neuron_scaling_slot
                .as_ref()
                .map(|slot| (env.model_config.latent_total(), slot.clone())),
            None,
            Some(model.local_predictive_coding_profile()),
        )?;

    let builder = SupervisedTraining::new(
        env.run_dir,
        Arc::clone(&env.train_loader),
        Arc::clone(&env.valid_loader),
    )
    .num_epochs(env.epochs)
    .grads_accumulation(env.training.gradient_accumulation_steps.max(1))
    .with_training_strategy(training_strategy)
    .with_application_logger(None)
    .with_interrupter(event_handles.interrupter)
    .with_metric_logger(event_handles.metric_logger)
    .with_file_checkpointer(BinFileRecorder::<FullPrecisionSettings>::new())
    .with_checkpointing_strategy(FileMetricBestCheckpointingStrategy::new(
        env.run_dir,
        &LossMetric::<ValidBackend<B>>::new(),
        burn_train::metric::store::Direction::Lowest,
        burn_train::metric::store::Split::Valid,
    ));
    let builder = builder.metric_train_numeric(ScalarMetric::<
        ValidBackend<B>,
        LossValue<ValidBackend<B>>,
    >::new_every("Loss", train_loss_metric_every));
    let builder = builder
        .metric_valid_numeric(LossMetric::<ValidBackend<B>>::new())
        .metric_train_numeric(LearningRateMetric::new())
        .metric_train(DeviceMetric::new("device", env.backend_name))
        .metric_valid(DeviceMetric::new("device", env.backend_name));
    let builder = if quiet_progress_renderer_enabled() {
        builder.renderer(QuietMetricsRenderer)
    } else {
        builder
    };
    #[cfg(feature = "rerun")]
    let builder = crate::train::rerun::attach_metric_loggers(builder, env.run_dir);
    let builder = builder.summary();
    let builder = match env.resume_checkpoint_epoch {
        Some(checkpoint) => builder.checkpoint(checkpoint),
        None => builder,
    };

    info!("run name: {}", env.run_name);
    info!(
        "training strategy: mode={:?} replicas={}",
        env.parallel_runtime.mode,
        env.devices.len()
    );
    info!(
        "checkpoint policy: logical_epoch_steps={} keep_last={} keep_best_valid_loss=true",
        env.train_loader.num_items(),
        CHECKPOINT_KEEP_LAST
    );

    let learner = burn_train::Learner::new(model, optimizer, scheduler);
    let TrainingResult { model, .. } = builder.launch(learner);
    synchronize_predictive_coding_checkpoint_manifests(env.run_dir, pc_manifest.as_ref())?;

    log_theoretical_profile(
        env.model_config,
        env.training
            .batch_size
            .saturating_mul(env.training.gradient_accumulation_steps.max(1)),
        env.training.block_size,
        env.backend_name,
    );

    Ok(model.model)
}

pub(crate) fn autotune_eggroll_population_chunk_size<B>(
    optimizer_cfg: &OptimizerConfig,
    model: &LanguageTrainModel<B>,
    train_loader: &Arc<dyn DataLoader<B, SequenceBatch<B>>>,
) -> Result<Option<EggrollChunkAutotuneReport>>
where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone,
{
    if !matches!(optimizer_cfg.name, OptimizerKind::Eggroll)
        || !optimizer_cfg.eggroll_auto_population.enabled
        || !optimizer_cfg.eggroll_auto_population.chunk_autotune.enabled
    {
        return Ok(None);
    }

    let Some(batch) = train_loader.iter().next() else {
        return Ok(None);
    };
    let eggroll = optimizer_cfg.effective_eggroll_config();
    let candidates = resolve_eggroll_chunk_autotune_candidates(optimizer_cfg);
    if candidates.is_empty() {
        return Ok(None);
    }
    let execution_plan = resolve_eggroll_population_execution_plan(optimizer_cfg, model)?;

    let mut reports = Vec::with_capacity(candidates.len());
    for population_chunk_size in candidates {
        let report = measure_eggroll_chunk_candidate(
            &execution_plan,
            model,
            batch.clone(),
            &eggroll,
            population_chunk_size,
            optimizer_cfg
                .eggroll_auto_population
                .chunk_autotune
                .max_probe_population_size,
        )?;
        reports.push(report);
    }
    let Some(selected) = reports
        .iter()
        .filter(|report| {
            report.forward_evaluations_per_second.is_finite()
                && report.forward_evaluations_per_second > 0.0
                && report.mean_loss.is_finite()
        })
        .max_by(|left, right| {
            left.forward_evaluations_per_second
                .partial_cmp(&right.forward_evaluations_per_second)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    else {
        return Ok(None);
    };

    Ok(Some(EggrollChunkAutotuneReport {
        selected_population_chunk_size: selected.population_chunk_size,
        configured_population_chunk_size: eggroll.population.population_chunk_size,
        population_size: eggroll.population.population_size,
        max_probe_population_size: optimizer_cfg
            .eggroll_auto_population
            .chunk_autotune
            .max_probe_population_size,
        candidates: reports,
    }))
}

pub(crate) fn train_with_eggroll_forward_only<B>(
    env: &ForwardEggrollTrainEnvironment<'_, B>,
    optimizer_cfg: &OptimizerConfig,
    mut model: LanguageTrainModel<B>,
) -> Result<DragonModel<B>>
where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone,
{
    fs::create_dir_all(env.run_dir)?;
    let eggroll = optimizer_cfg.effective_eggroll_config();

    let source_selection_dataset = env.source_selection_dataset.as_ref().cloned();
    let event_handles = crate::train::events::build_training_event_handles(
        env.run_name,
        env.run_dir,
        env.train_loader.num_items(),
        env.training,
        source_selection_dataset,
        None,
        None,
    )?;
    let bus = event_handles.metric_logger.bus();
    crate::train::profile::reset_predictive_coding();
    let emit_step_events = env.training.events.persist_step_events;
    let steps_per_epoch = env.train_loader.num_items().max(1);
    let start_epoch = env
        .resume_checkpoint_epoch
        .map(|epoch| epoch + 1)
        .unwrap_or(1);
    let mut best_valid_loss: Option<f64> = None;
    let mut best_valid_epoch: Option<usize> = None;
    let mut best_ruliad_competence: Option<RuliadCompetenceKey> = None;
    let mut best_ruliad_policy_competence: Option<RuliadPolicyCompetenceKey> = None;
    let mut best_ruliad_recovery_competence: Option<RuliadCompetenceKey> = None;
    let mut best_ruliad_policy_recovery_competence: Option<RuliadPolicyCompetenceKey> = None;
    let mut best_ruliad_checkpoint_epoch: Option<usize> = None;
    if let Some(epoch) = env.resume_checkpoint_epoch {
        model.model =
            load_dragon_model_checkpoint(env.run_dir, epoch, env.model_config, env.device)?;
        let historical_best = historical_best_validation(env.run_dir, epoch);
        best_valid_loss = historical_best.best_loss;
        best_valid_epoch = historical_best.best_checkpoint_epoch;
        info!(
            "EGGROLL forward-only resumed model checkpoint epoch={} with fresh optimizer state historical_best_valid_loss={:?} historical_best_checkpoint_epoch={:?}",
            epoch, best_valid_loss, best_valid_epoch
        );
    }

    let pair_count = eggroll.population.pair_count().max(1);
    let chunk_pair_count = (eggroll.population.population_chunk_size.max(2) / 2)
        .max(1)
        .min(pair_count);
    let eggroll_execution_plan = resolve_eggroll_population_execution_plan(optimizer_cfg, &model)?;
    let mut optimizer_state = burn_dragon_eggroll::EggrollModuleOptimizerState::<B>::new();

    info!(
        "training strategy: mode={:?} optimizer=eggroll backend_mode=forward_only population_size={} pair_count={} population_chunk_size={} chunk_pair_count={} rank={} sigma={} interval_steps={} start_epoch={} eggroll_executor={} eggroll_scope={}",
        env.parallel_runtime.mode,
        eggroll.population.population_size,
        pair_count,
        eggroll.population.population_chunk_size,
        chunk_pair_count,
        eggroll.population.rank,
        eggroll.sigma,
        eggroll.interval_steps,
        start_epoch,
        eggroll_execution_plan.executor_name(),
        eggroll_execution_plan.scope_name()
    );

    for epoch in start_epoch..=env.epochs {
        if let Some(reason) = training_interruption_reason(&event_handles.interrupter) {
            info!("Training interrupted: {reason}");
            break;
        }

        let mut iterator = env.train_loader.iter();
        let mut iteration = 0usize;
        let mut stop_requested = false;
        while let Some(batch) = iterator.next() {
            if let Some(reason) = training_interruption_reason(&event_handles.interrupter) {
                info!("Training interrupted during epoch {epoch}: {reason}");
                stop_requested = true;
                break;
            }
            iteration += 1;
            let absolute_step = epoch
                .saturating_sub(1)
                .saturating_mul(steps_per_epoch)
                .saturating_add(iteration.saturating_sub(1));
            if emit_step_events {
                let _ = bus.send_step_started(StepStarted {
                    run_id: env.run_name.to_string().into(),
                    absolute_step,
                    epoch,
                });
            }

            let eggroll_step_active = absolute_step.is_multiple_of(eggroll.interval_steps.max(1));
            let (mean_train_loss, metrics, eggroll_step_timing) = if eggroll_step_active {
                let eggroll_step_start = burn_dragon_time::Instant::now();
                let candidate_eval_start = burn_dragon_time::Instant::now();
                let mut losses = Vec::with_capacity(pair_count * 2);
                let mut fitness = Vec::with_capacity(pair_count);
                for chunk_start in (0..pair_count).step_by(chunk_pair_count) {
                    let chunk_end = chunk_start.saturating_add(chunk_pair_count).min(pair_count);
                    let pairs_in_chunk = chunk_end.saturating_sub(chunk_start);
                    let chunk_losses = evaluate_eggroll_population_chunk(
                        &eggroll_execution_plan,
                        &model,
                        batch.clone(),
                        &eggroll,
                        absolute_step as u64,
                        chunk_start,
                        pairs_in_chunk,
                    )?;
                    for (offset, pair_index) in (chunk_start..chunk_end).enumerate() {
                        let plus_loss = chunk_losses[offset * 2];
                        let minus_loss = chunk_losses[offset * 2 + 1];
                        losses.push(plus_loss);
                        losses.push(minus_loss);
                        fitness.push(burn_dragon_eggroll::AntitheticFitness {
                            pair_index: pair_index as u64,
                            plus: -(plus_loss as f32),
                            minus: -(minus_loss as f32),
                        });
                    }
                }
                let candidate_eval_ms = candidate_eval_start.elapsed().as_millis() as f64;

                let mean_train_loss = if losses.is_empty() {
                    f64::NAN
                } else {
                    losses.iter().sum::<f64>() / losses.len() as f64
                };
                let update_start = burn_dragon_time::Instant::now();
                let (updated, metrics) = apply_eggroll_population_update(
                    &eggroll_execution_plan,
                    model,
                    &eggroll,
                    absolute_step as u64,
                    &fitness,
                    &mut optimizer_state,
                )?;
                let update_ms = update_start.elapsed().as_millis() as f64;
                model = updated;
                (
                    mean_train_loss,
                    Some(metrics),
                    Some(EggrollStepTiming {
                        total_ms: eggroll_step_start.elapsed().as_millis() as f64,
                        candidate_eval_ms,
                        update_ms,
                    }),
                )
            } else {
                (f64::NAN, None, None)
            };

            if source_selection_telemetry_due_for(
                env.training,
                env.source_selection_dataset.as_ref(),
                absolute_step,
            ) && mean_train_loss.is_finite()
            {
                emit_source_selection_telemetry_sample(
                    env.run_name,
                    env.source_selection_dataset.as_ref(),
                    absolute_step,
                    mean_train_loss,
                    &bus,
                );
            }
            let log_train_metrics = iteration.is_multiple_of(env.training.log_frequency.max(1))
                || iteration == steps_per_epoch;
            if log_train_metrics {
                let progress = iterator.progress();
                if let Some(metrics) = &metrics {
                    info!(
                        "train epoch={} step={}/{} loss={:.4} eggroll_fitness_mean={:.4} eggroll_fitness_std={:.4}",
                        epoch,
                        progress.items_processed,
                        progress.items_total,
                        mean_train_loss,
                        metrics.fitness_mean,
                        metrics.fitness_std
                    );
                } else {
                    info!(
                        "train epoch={} step={}/{} loss={:.4} eggroll_interval_skip=true",
                        epoch, progress.items_processed, progress.items_total, mean_train_loss
                    );
                }
            }
            if mean_train_loss.is_finite() {
                let _ = bus.send_metric_sample(TrainingMetricSample {
                    run_id: env.run_name.to_string().into(),
                    split: TrainingMetricSplit::Train,
                    epoch,
                    step_in_epoch: iteration,
                    absolute_step,
                    name: "Loss".to_string(),
                    value: mean_train_loss,
                    running_value: mean_train_loss,
                });
            }
            if let Some(metrics) = &metrics {
                let timing = eggroll_step_timing.unwrap_or_default();
                let elapsed_ms = timing.total_ms;
                let forward_evaluations = eggroll.population.population_size as f64;
                let forward_evaluations_per_second = if timing.candidate_eval_ms > 0.0 {
                    forward_evaluations * 1000.0 / timing.candidate_eval_ms
                } else {
                    0.0
                };
                let candidate_eval_fraction = if elapsed_ms > 0.0 {
                    timing.candidate_eval_ms / elapsed_ms
                } else {
                    0.0
                };
                let update_fraction = if elapsed_ms > 0.0 {
                    timing.update_ms / elapsed_ms
                } else {
                    0.0
                };
                let _ = bus.send_metric_sample(TrainingMetricSample {
                    run_id: env.run_name.to_string().into(),
                    split: TrainingMetricSplit::Train,
                    epoch,
                    step_in_epoch: iteration,
                    absolute_step,
                    name: "EGGROLL Fitness Std".to_string(),
                    value: metrics.fitness_std as f64,
                    running_value: metrics.fitness_std as f64,
                });
                let _ = bus.send_metric_sample(TrainingMetricSample {
                    run_id: env.run_name.to_string().into(),
                    split: TrainingMetricSplit::Train,
                    epoch,
                    step_in_epoch: iteration,
                    absolute_step,
                    name: "EGGROLL Coefficient RMS".to_string(),
                    value: metrics.coefficient_rms as f64,
                    running_value: metrics.coefficient_rms as f64,
                });
                let _ = bus.send_metric_sample(TrainingMetricSample {
                    run_id: env.run_name.to_string().into(),
                    split: TrainingMetricSplit::Train,
                    epoch,
                    step_in_epoch: iteration,
                    absolute_step,
                    name: "EGGROLL Coefficient Clip Fraction".to_string(),
                    value: metrics.coefficient_clip_fraction as f64,
                    running_value: metrics.coefficient_clip_fraction as f64,
                });
                let _ = bus.send_metric_sample(TrainingMetricSample {
                    run_id: env.run_name.to_string().into(),
                    split: TrainingMetricSplit::Train,
                    epoch,
                    step_in_epoch: iteration,
                    absolute_step,
                    name: "EGGROLL Step Milliseconds".to_string(),
                    value: elapsed_ms,
                    running_value: elapsed_ms,
                });
                let _ = bus.send_metric_sample(TrainingMetricSample {
                    run_id: env.run_name.to_string().into(),
                    split: TrainingMetricSplit::Train,
                    epoch,
                    step_in_epoch: iteration,
                    absolute_step,
                    name: "EGGROLL Candidate Eval Milliseconds".to_string(),
                    value: timing.candidate_eval_ms,
                    running_value: timing.candidate_eval_ms,
                });
                let _ = bus.send_metric_sample(TrainingMetricSample {
                    run_id: env.run_name.to_string().into(),
                    split: TrainingMetricSplit::Train,
                    epoch,
                    step_in_epoch: iteration,
                    absolute_step,
                    name: "EGGROLL Update Milliseconds".to_string(),
                    value: timing.update_ms,
                    running_value: timing.update_ms,
                });
                let _ = bus.send_metric_sample(TrainingMetricSample {
                    run_id: env.run_name.to_string().into(),
                    split: TrainingMetricSplit::Train,
                    epoch,
                    step_in_epoch: iteration,
                    absolute_step,
                    name: "EGGROLL Candidate Eval Fraction".to_string(),
                    value: candidate_eval_fraction,
                    running_value: candidate_eval_fraction,
                });
                let _ = bus.send_metric_sample(TrainingMetricSample {
                    run_id: env.run_name.to_string().into(),
                    split: TrainingMetricSplit::Train,
                    epoch,
                    step_in_epoch: iteration,
                    absolute_step,
                    name: "EGGROLL Update Fraction".to_string(),
                    value: update_fraction,
                    running_value: update_fraction,
                });
                let _ = bus.send_metric_sample(TrainingMetricSample {
                    run_id: env.run_name.to_string().into(),
                    split: TrainingMetricSplit::Train,
                    epoch,
                    step_in_epoch: iteration,
                    absolute_step,
                    name: "EGGROLL Forward Evaluations Per Second".to_string(),
                    value: forward_evaluations_per_second,
                    running_value: forward_evaluations_per_second,
                });
                let _ = bus.send_metric_sample(TrainingMetricSample {
                    run_id: env.run_name.to_string().into(),
                    split: TrainingMetricSplit::Train,
                    epoch,
                    step_in_epoch: iteration,
                    absolute_step,
                    name: "EGGROLL Population Size".to_string(),
                    value: eggroll.population.population_size as f64,
                    running_value: eggroll.population.population_size as f64,
                });
                let _ = bus.send_metric_sample(TrainingMetricSample {
                    run_id: env.run_name.to_string().into(),
                    split: TrainingMetricSplit::Train,
                    epoch,
                    step_in_epoch: iteration,
                    absolute_step,
                    name: "EGGROLL Population Chunk Size".to_string(),
                    value: eggroll.population.population_chunk_size as f64,
                    running_value: eggroll.population.population_chunk_size as f64,
                });
                let _ = bus.send_metric_sample(TrainingMetricSample {
                    run_id: env.run_name.to_string().into(),
                    split: TrainingMetricSplit::Train,
                    epoch,
                    step_in_epoch: iteration,
                    absolute_step,
                    name: "EGGROLL Stacked Tensorized Executor Active".to_string(),
                    value: 1.0,
                    running_value: 1.0,
                });
                let scope_id = match eggroll_execution_plan.scope {
                    EggrollPerturbationScope::DragonCoreProjection => 1.0,
                };
                let _ = bus.send_metric_sample(TrainingMetricSample {
                    run_id: env.run_name.to_string().into(),
                    split: TrainingMetricSplit::Train,
                    epoch,
                    step_in_epoch: iteration,
                    absolute_step,
                    name: "EGGROLL Perturbation Scope ID".to_string(),
                    value: scope_id,
                    running_value: scope_id,
                });
            }
            if emit_step_events {
                let _ = bus.send_step_finished(StepFinished {
                    run_id: env.run_name.to_string().into(),
                    absolute_step,
                    epoch,
                    loss: mean_train_loss.is_finite().then_some(mean_train_loss),
                });
            }
        }
        drop(iterator);

        if stop_requested {
            break;
        }

        let _ = bus.send_epoch_summary(TrainingEpochSummary {
            run_id: env.run_name.to_string().into(),
            split: TrainingMetricSplit::Train,
            epoch,
        });
        if !env.training.validation.execution.is_local() {
            let absolute_step = epoch_end_absolute_step(epoch, steps_per_epoch, iteration);
            let checkpoint_started = burn_dragon_time::Instant::now();
            save_dragon_model_checkpoint(env.run_dir, epoch, &model.model)?;
            save_source_selection_state_checkpoint(
                env.run_dir,
                epoch,
                absolute_step,
                env.source_selection_dataset.as_ref(),
            )?;
            prune_dragon_model_checkpoints(env.run_dir, epoch, &[None, None])?;
            let _ = bus.send_checkpoint(CheckpointEvent {
                run_id: env.run_name.to_string().into(),
                checkpoint_id: format!("model-{epoch}"),
                epoch: Some(epoch),
                absolute_step: Some(absolute_step),
                promoted: false,
            });
            let _ = bus.flush();
            crate::train::profile::record_checkpoint(checkpoint_started.elapsed().as_nanos());
            info!(
                "validation deferred to external evaluator epoch={epoch}; candidate checkpoint is unpromoted"
            );
            continue;
        }
        let validation_started = burn_dragon_time::Instant::now();
        let absolute_step = epoch_end_absolute_step(epoch, steps_per_epoch, iteration);
        let validation = run_dynamic_validation_forward_only(
            env,
            &model,
            epoch,
            steps_per_epoch,
            absolute_step,
            &bus,
        )?;
        crate::train::profile::record_validation(validation_started.elapsed().as_nanos());
        let valid_loss = validation.primary_loss();
        info!("valid epoch={} loss={valid_loss:.4}", epoch);
        let checkpoint_promoted = should_promote_checkpoint(
            &validation,
            best_valid_loss,
            best_ruliad_competence,
            best_ruliad_policy_competence,
            env.training,
        );
        if checkpoint_promoted {
            best_valid_loss = Some(valid_loss);
            best_valid_epoch = Some(epoch);
            if let Some(competence) = validation
                .ruliad_eval_report
                .as_ref()
                .and_then(ruliad_competence_key)
            {
                best_ruliad_competence = Some(competence);
            }
            if let Some(competence) = validation
                .ruliad_policy_rollout
                .as_ref()
                .and_then(ruliad_policy_competence_key)
            {
                best_ruliad_policy_competence = Some(competence);
            }
        }
        let checkpoint_started = burn_dragon_time::Instant::now();
        save_dragon_model_checkpoint(env.run_dir, epoch, &model.model)?;
        if update_ruliad_recovery_competence(
            &validation,
            env.training
                .ruliad_policy_probe
                .checkpoint_capability_contract,
            env.training.ruliad_policy_probe.promotion_gate,
            &env.training.gates,
            &mut best_ruliad_recovery_competence,
            &mut best_ruliad_policy_recovery_competence,
        ) {
            best_ruliad_checkpoint_epoch = Some(epoch);
        }
        save_source_selection_state_checkpoint(
            env.run_dir,
            epoch,
            absolute_step,
            env.source_selection_dataset.as_ref(),
        )?;
        prune_dragon_model_checkpoints(
            env.run_dir,
            epoch,
            &[best_valid_epoch, best_ruliad_checkpoint_epoch],
        )?;
        let _ = bus.send_checkpoint(CheckpointEvent {
            run_id: env.run_name.to_string().into(),
            checkpoint_id: format!("model-{epoch}"),
            epoch: Some(epoch),
            absolute_step: Some(absolute_step),
            promoted: checkpoint_promoted,
        });
        let _ = bus.flush();
        crate::train::profile::record_checkpoint(checkpoint_started.elapsed().as_nanos());
    }

    log_theoretical_profile(
        env.model_config,
        env.training.batch_size,
        env.training.block_size,
        env.backend_name,
    );

    Ok(model.model)
}

pub(super) fn eggroll_batch_loss_tensor<B>(
    model: &LanguageTrainModel<B>,
    batch: SequenceBatch<B>,
) -> Tensor<B, 1>
where
    B: BackendTrait,
{
    let output = ValidStep::step(model, batch);
    let loss_value: LossValue<B> = output.adapt();
    loss_value.value()
}

pub(super) fn scalar_values_from_loss_tensors<B>(loss_tensors: Vec<Tensor<B, 1>>) -> Vec<f64>
where
    B: BackendTrait,
{
    if loss_tensors.is_empty() {
        return Vec::new();
    }
    let values = Tensor::cat(loss_tensors, 0)
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("loss tensor to vec");
    values.into_iter().map(|value| value as f64).collect()
}

pub(super) fn resolve_eggroll_population_execution_plan<B>(
    optimizer_cfg: &OptimizerConfig,
    model: &LanguageTrainModel<B>,
) -> Result<EggrollPopulationExecutionPlan>
where
    B: BackendTrait,
{
    let execution = &optimizer_cfg.eggroll_population_execution;
    let scope = execution.perturbation_scope;

    if let Some(reason) = eggroll_population_execution_unsupported_reason(model) {
        return Err(anyhow!(
            "optimizer.eggroll_population_execution unsupported: {reason}"
        ));
    }

    Ok(EggrollPopulationExecutionPlan {
        backend: execution.backend,
        scope,
        population_tile_size: execution.population_tile_size,
    })
}

pub(super) fn eggroll_population_execution_unsupported_reason<B>(
    model: &LanguageTrainModel<B>,
) -> Option<String>
where
    B: BackendTrait,
{
    if !matches!(model.objective, TrainingObjectiveConfig::NextToken) {
        return Some(
            "EGGROLL stacked tensorized executor currently supports objective=next_token"
                .to_string(),
        );
    }
    if model.tbptt_chunk_size.is_some() || model.tbptt_persist_across_steps {
        return Some(
            "EGGROLL stacked tensorized executor currently does not support TBPTT".to_string(),
        );
    }
    if model.pipeline_plan.is_some() {
        return Some(
            "EGGROLL stacked tensorized executor currently does not support pipeline execution"
                .to_string(),
        );
    }
    if !model.model.supports_shared_lowrank_population_forward() {
        return Some(
            "EGGROLL stacked tensorized executor requires flat logits, rollout_fast_steps_per_slow_step=1, and y-neuron recurrence disabled"
                .to_string(),
        );
    }
    None
}

pub(super) fn evaluate_eggroll_population_chunk<B>(
    plan: &EggrollPopulationExecutionPlan,
    model: &LanguageTrainModel<B>,
    batch: SequenceBatch<B>,
    eggroll: &burn_eggroll::EggrollConfig,
    generation: u64,
    pair_start: usize,
    pair_count: usize,
) -> Result<Vec<f64>>
where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone,
{
    evaluate_eggroll_population_chunk_stacked_tensorized(
        plan, model, batch, eggroll, generation, pair_start, pair_count,
    )
}

pub(super) fn apply_eggroll_population_update<B>(
    _plan: &EggrollPopulationExecutionPlan,
    model: LanguageTrainModel<B>,
    eggroll: &burn_eggroll::EggrollConfig,
    generation: u64,
    fitness: &[burn_dragon_eggroll::AntitheticFitness],
    state: &mut burn_dragon_eggroll::EggrollModuleOptimizerState<B>,
) -> Result<(LanguageTrainModel<B>, burn_eggroll::EggrollMetrics)>
where
    B: BackendTrait + Clone,
{
    apply_shared_lowrank_eggroll_update(model, eggroll, generation, fitness, state)
}

pub(super) fn apply_shared_lowrank_eggroll_update<B>(
    model: LanguageTrainModel<B>,
    eggroll: &burn_eggroll::EggrollConfig,
    generation: u64,
    fitness: &[burn_dragon_eggroll::AntitheticFitness],
    state: &mut burn_dragon_eggroll::EggrollModuleOptimizerState<B>,
) -> Result<(LanguageTrainModel<B>, burn_eggroll::EggrollMetrics)>
where
    B: BackendTrait,
{
    let coefficients =
        burn_dragon_eggroll::pair_gradient_coefficients(eggroll, generation, fitness)?;
    let population = fitness
        .iter()
        .flat_map(|item| [item.plus, item.minus])
        .collect::<Vec<_>>();
    let metrics = burn_eggroll::eggroll_metrics(
        generation,
        population.len(),
        eggroll.population.rank,
        eggroll.effective_sigma(generation),
        &population,
        &coefficients
            .iter()
            .map(|coefficient| coefficient.coefficient)
            .collect::<Vec<_>>(),
        eggroll.coefficient_clip,
    );
    let ids = model.model.shared_lowrank_param_ids();
    let perturbation_keys = shared_lowrank_eggroll_perturbation_keys();
    let weights = model.model.shared_lowrank_weights();
    let next = SharedLowrankWeights {
        encoder: burn_dragon_eggroll::apply_antithetic_update_to_tensor_with_coefficients(
            weights.encoder,
            ids.encoder.val(),
            perturbation_keys.encoder,
            eggroll,
            generation,
            &coefficients,
            state,
        ),
        encoder_v: burn_dragon_eggroll::apply_antithetic_update_to_tensor_with_coefficients(
            weights.encoder_v,
            ids.encoder_v.val(),
            perturbation_keys.encoder_v,
            eggroll,
            generation,
            &coefficients,
            state,
        ),
        decoder: burn_dragon_eggroll::apply_antithetic_update_to_tensor_with_coefficients(
            weights.decoder,
            ids.decoder.val(),
            perturbation_keys.decoder,
            eggroll,
            generation,
            &coefficients,
            state,
        ),
    };
    Ok((
        model.map_model(|dragon| dragon.with_shared_lowrank_weights(next)),
        metrics,
    ))
}

#[derive(Clone, Copy)]
pub(super) struct SharedLowrankEggrollPerturbationKeys {
    pub(super) encoder: u64,
    pub(super) encoder_v: u64,
    pub(super) decoder: u64,
}

pub(super) fn shared_lowrank_eggroll_perturbation_keys() -> SharedLowrankEggrollPerturbationKeys {
    SharedLowrankEggrollPerturbationKeys {
        encoder: burn_dragon_eggroll::stable_parameter_key("encoder"),
        encoder_v: burn_dragon_eggroll::stable_parameter_key("encoder_v"),
        decoder: burn_dragon_eggroll::stable_parameter_key("decoder"),
    }
}

pub(super) fn evaluate_eggroll_population_chunk_stacked_tensorized<B>(
    plan: &EggrollPopulationExecutionPlan,
    model: &LanguageTrainModel<B>,
    batch: SequenceBatch<B>,
    eggroll: &burn_eggroll::EggrollConfig,
    generation: u64,
    pair_start: usize,
    pair_count: usize,
) -> Result<Vec<f64>>
where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone,
{
    if batch.summary_event_mask.is_some() {
        return Err(anyhow!(
            "EGGROLL stacked tensorized population evaluator does not support summary_event_mask batches"
        ));
    }
    if batch.loss_mask.is_some() {
        return Err(anyhow!(
            "EGGROLL stacked tensorized population evaluator does not support target loss masks; use AdamW answer-only warmup or the verifier-ranked EGGROLL path"
        ));
    }
    let tile_pair_count = plan
        .population_tile_size
        .map(|tile| make_even_population_size(tile).saturating_div(2).max(1))
        .unwrap_or(pair_count.max(1))
        .min(pair_count.max(1));
    if tile_pair_count < pair_count {
        let mut losses = Vec::with_capacity(pair_count * 2);
        let mut local_start = pair_start;
        let pair_end = pair_start.saturating_add(pair_count);
        while local_start < pair_end {
            let local_count = (pair_end - local_start).min(tile_pair_count);
            losses.extend(evaluate_eggroll_population_chunk_stacked_tensorized(
                plan,
                model,
                batch.clone(),
                eggroll,
                generation,
                local_start,
                local_count,
            )?);
            local_start = local_start.saturating_add(local_count);
        }
        return Ok(losses);
    }

    let logits = match plan.backend {
        EggrollPopulationExecutionBackend::Factorized => {
            let lowrank = build_shared_lowrank_population_factors(
                model, eggroll, generation, pair_start, pair_count,
            );
            model
                .model
                .forward_with_shared_lowrank_population_factors(batch.inputs.clone(), lowrank)
        }
        EggrollPopulationExecutionBackend::Auto
        | EggrollPopulationExecutionBackend::Reference
        | EggrollPopulationExecutionBackend::Cuda => {
            let lowrank = build_shared_lowrank_population_weights(
                model, eggroll, generation, pair_start, pair_count,
            );
            model
                .model
                .forward_with_shared_lowrank_population(batch.inputs.clone(), lowrank)
        }
    };
    Ok(scalar_values_from_loss_tensors(vec![
        population_next_token_losses(model, logits, batch.targets, pair_count * 2),
    ]))
}

pub(super) fn build_shared_lowrank_population_factors<B>(
    model: &LanguageTrainModel<B>,
    eggroll: &burn_eggroll::EggrollConfig,
    generation: u64,
    pair_start: usize,
    pair_count: usize,
) -> SharedLowrankPopulationFactors<B>
where
    B: BackendTrait,
{
    let base = model.model.shared_lowrank_weights();
    let perturbation_keys = shared_lowrank_eggroll_perturbation_keys();
    let sigma = eggroll.effective_sigma(generation);
    let encoder_spec = burn_eggroll::MatrixNoisePopulationSpec::new(
        eggroll.population.seed,
        perturbation_keys.encoder,
        generation,
        pair_start as u64,
        pair_count,
        eggroll.population.rank,
    );
    let encoder_v_spec = burn_eggroll::MatrixNoisePopulationSpec::new(
        eggroll.population.seed,
        perturbation_keys.encoder_v,
        generation,
        pair_start as u64,
        pair_count,
        eggroll.population.rank,
    );
    let decoder_spec = burn_eggroll::MatrixNoisePopulationSpec::new(
        eggroll.population.seed,
        perturbation_keys.decoder,
        generation,
        pair_start as u64,
        pair_count,
        eggroll.population.rank,
    );
    let [heads, embd, latent_capacity] = base.encoder.shape().dims::<3>();
    let [decoder_rows, decoder_cols] = base.decoder.shape().dims::<2>();
    let device = base.encoder.device();
    let encoder = burn_eggroll::low_rank_factors_3d_antithetic_population_with_mode(
        heads,
        embd,
        latent_capacity,
        encoder_spec,
        eggroll.population.matrix_noise,
        &device,
    );
    let encoder_v = burn_eggroll::low_rank_factors_3d_antithetic_population_with_mode(
        heads,
        embd,
        latent_capacity,
        encoder_v_spec,
        eggroll.population.matrix_noise,
        &device,
    );
    let decoder = burn_eggroll::low_rank_factors_2d_antithetic_population_with_mode(
        decoder_rows,
        decoder_cols,
        decoder_spec,
        eggroll.population.matrix_noise,
        &device,
    );

    SharedLowrankPopulationFactors {
        encoder_a: encoder.a,
        encoder_b: encoder.b,
        encoder_v_a: encoder_v.a,
        encoder_v_b: encoder_v.b,
        decoder_a: decoder.a,
        decoder_b: decoder.b,
        signs: encoder.signs,
        encoder_scale: encoder.scale,
        encoder_v_scale: encoder_v.scale,
        decoder_scale: decoder.scale,
        sigma,
    }
}

pub(super) fn build_shared_lowrank_population_weights<B>(
    model: &LanguageTrainModel<B>,
    eggroll: &burn_eggroll::EggrollConfig,
    generation: u64,
    pair_start: usize,
    pair_count: usize,
) -> SharedLowrankPopulationWeights<B>
where
    B: BackendTrait,
{
    let base = model.model.shared_lowrank_weights();
    let perturbation_keys = shared_lowrank_eggroll_perturbation_keys();
    let sigma = eggroll.effective_sigma(generation);
    let encoder_spec = burn_eggroll::MatrixNoisePopulationSpec::new(
        eggroll.population.seed,
        perturbation_keys.encoder,
        generation,
        pair_start as u64,
        pair_count,
        eggroll.population.rank,
    );
    let encoder_v_spec = burn_eggroll::MatrixNoisePopulationSpec::new(
        eggroll.population.seed,
        perturbation_keys.encoder_v,
        generation,
        pair_start as u64,
        pair_count,
        eggroll.population.rank,
    );
    let decoder_spec = burn_eggroll::MatrixNoisePopulationSpec::new(
        eggroll.population.seed,
        perturbation_keys.decoder,
        generation,
        pair_start as u64,
        pair_count,
        eggroll.population.rank,
    );

    SharedLowrankPopulationWeights {
        encoder: burn_eggroll::perturb_matrix_3d_antithetic_population_with_mode(
            base.encoder,
            sigma,
            encoder_spec,
            eggroll.population.matrix_noise,
        ),
        encoder_v: burn_eggroll::perturb_matrix_3d_antithetic_population_with_mode(
            base.encoder_v,
            sigma,
            encoder_v_spec,
            eggroll.population.matrix_noise,
        ),
        decoder: burn_eggroll::perturb_matrix_2d_antithetic_population_with_mode(
            base.decoder,
            sigma,
            decoder_spec,
            eggroll.population.matrix_noise,
        ),
    }
}

pub(super) fn population_next_token_losses<B>(
    model: &LanguageTrainModel<B>,
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
    population: usize,
) -> Tensor<B, 1>
where
    B: BackendTrait,
{
    let [stacked_batch, _time, _vocab] = logits.shape().dims::<3>();
    assert_eq!(
        stacked_batch % population,
        0,
        "stacked population batch must divide evenly"
    );
    let base_batch = stacked_batch / population;
    let targets = Tensor::cat(
        (0..population).map(|_| targets.clone()).collect::<Vec<_>>(),
        0,
    );
    model
        .model
        .language_token_losses_from_logits(logits, targets)
        .mean_dim(1)
        .reshape([population, base_batch])
        .mean_dim(1)
        .reshape([population])
}

pub(super) fn resolve_eggroll_chunk_autotune_candidates(
    optimizer_cfg: &OptimizerConfig,
) -> Vec<usize> {
    let population_size = optimizer_cfg.eggroll.population.population_size.max(2);
    let configured = optimizer_cfg
        .eggroll
        .population
        .population_chunk_size
        .max(2)
        .min(population_size);
    let max_probe = optimizer_cfg
        .eggroll_auto_population
        .chunk_autotune
        .max_probe_population_size
        .max(2)
        .min(population_size);
    let configured = make_even_population_size(configured.min(max_probe));
    let mut candidates = if optimizer_cfg
        .eggroll_auto_population
        .chunk_autotune
        .candidates
        .is_empty()
    {
        vec![16, 32, 64, 128, configured]
    } else {
        let mut candidates = optimizer_cfg
            .eggroll_auto_population
            .chunk_autotune
            .candidates
            .clone();
        candidates.push(configured);
        candidates
    };
    candidates = candidates
        .into_iter()
        .map(|candidate| make_even_population_size(candidate.max(2).min(max_probe)))
        .filter(|candidate| *candidate >= 2 && *candidate <= max_probe)
        .collect();
    candidates.sort_unstable();
    candidates.dedup();
    candidates
}

pub(super) fn make_even_population_size(value: usize) -> usize {
    value.saturating_sub(value % 2).max(2)
}

pub(super) fn measure_eggroll_chunk_candidate<B>(
    plan: &EggrollPopulationExecutionPlan,
    model: &LanguageTrainModel<B>,
    batch: SequenceBatch<B>,
    eggroll: &burn_eggroll::EggrollConfig,
    population_chunk_size: usize,
    max_probe_population_size: usize,
) -> Result<EggrollChunkAutotuneCandidateReport>
where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone,
{
    let evaluated_population_size = make_even_population_size(
        population_chunk_size
            .min(max_probe_population_size)
            .min(eggroll.population.population_size)
            .max(2),
    );
    let pair_count = (evaluated_population_size / 2).max(1);
    let started = burn_dragon_time::Instant::now();
    let losses = evaluate_eggroll_population_chunk(plan, model, batch, eggroll, 0, 0, pair_count)?;
    let elapsed_ms = started.elapsed().as_millis() as f64;
    let mean_loss = if losses.is_empty() {
        f64::NAN
    } else {
        losses.iter().sum::<f64>() / losses.len() as f64
    };
    let forward_evaluations_per_second = if elapsed_ms > 0.0 {
        losses.len() as f64 * 1000.0 / elapsed_ms
    } else {
        0.0
    };
    Ok(EggrollChunkAutotuneCandidateReport {
        population_chunk_size,
        evaluated_population_size: losses.len(),
        elapsed_ms,
        forward_evaluations_per_second,
        mean_loss,
    })
}
