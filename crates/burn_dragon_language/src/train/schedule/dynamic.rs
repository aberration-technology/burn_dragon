//! Dynamic loaders, training loops, and validation dispatch.

use super::*;

const METRIC_VALIDATION_SUPERVISED_TOKENS: &str = "Validation Supervised Tokens";
const METRIC_VALIDATION_SUPERVISED_BATCHES: &str = "Validation Supervised Batches";
const METRIC_VALIDATION_EMPTY_SUPERVISION_BATCHES: &str = "Validation Empty Supervision Batches";

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct SupervisedValidationLoss {
    weighted_loss: f64,
    supervised_tokens: usize,
    supervised_batches: usize,
    empty_batches: usize,
}

impl SupervisedValidationLoss {
    fn observe(&mut self, loss: f64, supervised_tokens: usize) {
        if supervised_tokens == 0 {
            self.empty_batches = self.empty_batches.saturating_add(1);
            return;
        }
        self.weighted_loss += loss * supervised_tokens as f64;
        self.supervised_tokens = self.supervised_tokens.saturating_add(supervised_tokens);
        self.supervised_batches = self.supervised_batches.saturating_add(1);
    }

    fn mean(self) -> Option<f64> {
        (self.supervised_tokens > 0).then(|| self.weighted_loss / self.supervised_tokens as f64)
    }
}

fn sequence_batch_supervised_tokens<B: BackendTrait>(batch: &SequenceBatch<B>) -> usize {
    let [batch_size, time] = batch.targets.shape().dims();
    let Some(mask) = batch.loss_mask.as_ref() else {
        return batch_size.saturating_mul(time);
    };
    mask.clone()
        .sum()
        .to_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .expect("validation supervision mask to vec")
        .into_iter()
        .map(|value| value.max(0) as usize)
        .sum()
}

fn emit_validation_supervision_metrics(
    run_name: &str,
    epoch: usize,
    absolute_step: usize,
    summary: SupervisedValidationLoss,
    bus: &TrainingEventBus,
) {
    for (name, value) in [
        (
            METRIC_VALIDATION_SUPERVISED_TOKENS,
            summary.supervised_tokens as f64,
        ),
        (
            METRIC_VALIDATION_SUPERVISED_BATCHES,
            summary.supervised_batches as f64,
        ),
        (
            METRIC_VALIDATION_EMPTY_SUPERVISION_BATCHES,
            summary.empty_batches as f64,
        ),
    ] {
        let _ = bus.send_metric_sample(TrainingMetricSample {
            run_id: run_name.to_string().into(),
            split: TrainingMetricSplit::Valid,
            epoch,
            step_in_epoch: summary
                .supervised_batches
                .saturating_add(summary.empty_batches),
            absolute_step,
            name: name.to_string(),
            value,
            running_value: value,
        });
    }
}

pub(super) fn build_dynamic_train_loader<B>(
    env: &TrainEnvironment<'_, B>,
    batch_size: usize,
    consumed_steps: usize,
) -> Arc<dyn DataLoader<B, SequenceBatch<B>>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let batch_size = batch_size.max(1);
    let Some(train_dataset) = env.train_dataset.as_ref() else {
        return Arc::clone(&env.train_loader);
    };
    if env
        .training
        .sequence_batching
        .uses_streaming_loader(env.training.tbptt_persist_across_steps)
    {
        Arc::new(
            StreamingDataLoader::<B>::new(
                Arc::clone(train_dataset),
                DatasetSplit::Train,
                env.device,
                env.train_loader.num_items().max(1),
                Some(env.total_steps),
                env.training.min_logical_block_size,
                env.training.seed,
            )
            .with_batch_size(batch_size)
            .with_initial_consumed_steps(consumed_steps)
            .with_ruliad_policy_supervision(env.training.ruliad_supervision)
            .with_ruliad_policy_stratified_difficulty_levels(
                env.training
                    .ruliad_supervision
                    .proof_policy
                    .stratified_difficulty_levels,
            )
            .with_summary_event_token_ids(env.summary_event_token_ids.clone()),
        )
    } else {
        Arc::new(
            RandomDataLoader::<B>::new(
                Arc::clone(train_dataset),
                DatasetSplit::Train,
                env.device,
                env.train_loader.num_items().max(1),
                Some(env.total_steps),
            )
            .with_seed(env.training.seed)
            .with_batch_size(batch_size)
            .with_initial_consumed_steps(consumed_steps)
            .with_ruliad_policy_supervision(env.training.ruliad_supervision)
            .with_ruliad_policy_stratified_difficulty_levels(
                env.training
                    .ruliad_supervision
                    .proof_policy
                    .stratified_difficulty_levels,
            )
            .with_summary_event_token_ids(env.summary_event_token_ids.clone()),
        )
    }
}

pub(super) fn build_dynamic_valid_loader<B>(
    env: &TrainEnvironment<'_, B>,
    batch_size: usize,
) -> Arc<dyn DataLoader<ValidBackend<B>, SequenceBatch<ValidBackend<B>>>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let Some(valid_dataset) = env.valid_dataset.as_ref() else {
        return Arc::clone(&env.valid_loader);
    };
    Arc::new(
        RandomDataLoader::<ValidBackend<B>>::new(
            Arc::clone(valid_dataset),
            DatasetSplit::Val,
            env.device,
            env.valid_steps.max(1),
            None,
        )
        .with_seed(env.training.validation.seed)
        .with_source_selection_enabled(
            env.training
                .validation
                .sampling
                .uses_live_source_selection(),
        )
        .with_batch_size(batch_size.max(1))
        .with_summary_event_token_ids(env.summary_event_token_ids.clone()),
    )
}

pub(crate) fn train_with_dynamic_neuron_scaling_scheduler<B, S>(
    env: &TrainEnvironment<'_, B>,
    mut model: LanguageTrainModel<B>,
    mut optimizer: crate::train::continual_backprop::LanguageOptimizer<B>,
    mut scheduler: S,
) -> Result<DragonModel<ValidBackend<B>>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    S: LrScheduler + Clone + 'static,
{
    fs::create_dir_all(env.run_dir)?;
    prepare_local_predictive_coding_contract(env, &model)?;
    let source_selection_dataset = env.source_selection_dataset.clone();
    let dynamics_control_slot = DragonDynamicsControlSlot::default();
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
            Some(dynamics_control_slot.clone()),
            Some(model.local_predictive_coding_profile()),
        )?;
    let bus = event_handles.metric_logger.bus();
    let emit_step_events = env.training.events.persist_step_events;
    let steps_per_epoch = env.train_loader.num_items().max(1);
    let start_epoch = env
        .resume_checkpoint_epoch
        .map(|epoch| epoch + 1)
        .unwrap_or(1);
    let mut active_batch_size = env.training.batch_size.max(1);
    let mut active_grad_accumulation = env.training.gradient_accumulation_steps.max(1);
    let mut active_train_loader = Arc::clone(&env.train_loader);
    let mut active_valid_loader = Arc::clone(&env.valid_loader);
    let mut current_model_config = env.model_config.clone();
    let mut scale_generation = 0usize;
    let mut stability = ContinualLearningStabilityState::default();
    let mut dynamics_control = ActiveDynamicsControl::default();
    let mut last_cbp_telemetry_step = 0usize;
    let mut best_valid_loss: Option<f64> = None;
    let mut best_valid_epoch: Option<usize> = None;
    let mut context_routing = env
        .training
        .predictive_context_routing
        .enabled
        .then(|| {
            crate::train::PredictiveContextRoutingRuntime::new(
                env.training.predictive_context_routing.clone(),
                &current_model_config,
                env.training.seed,
                env.device,
                optimizer.clone(),
                env.run_dir.join("checkpoint"),
            )
        })
        .transpose()?;

    if let Some(epoch) = env.resume_checkpoint_epoch {
        let require_exact = matches!(
            env.training.launch_mode,
            burn_dragon_train::train::pipeline::TrainingLaunchMode::ResumeExactRun
        );
        let (loaded_model, loaded_model_config) = load_dragon_training_state_checkpoint(
            env.run_dir,
            epoch,
            env.model_config,
            env.device,
            &mut optimizer,
            &mut scheduler,
            &mut dynamics_control,
        )?;
        model.model = loaded_model;
        let runtime_restored = load_runtime_state_checkpoint(
            env.run_dir,
            epoch,
            &model,
            env.device,
            require_exact,
            env.training.predictive_context_routing.enabled,
        )?;
        let context_routing_restored = if let Some(routing) = context_routing.as_mut() {
            routing.restore_checkpoint(env.run_dir, epoch, require_exact)?
        } else {
            false
        };
        if let Some(loaded_model_config) = loaded_model_config {
            current_model_config = loaded_model_config;
        }
        optimizer.refresh_continual_backprop_fresh_model(DragonModel::<B>::new(
            current_model_config.clone(),
            env.device,
        ));
        let historical_best = historical_best_validation(env.run_dir, epoch);
        if let Some(restored) =
            load_continual_learning_stability_checkpoint(env.run_dir, epoch, require_exact)?
        {
            stability = restored;
            best_valid_loss = stability.best_valid_loss.or(historical_best.best_loss);
            best_valid_epoch = stability
                .best_checkpoint_epoch
                .or(historical_best.best_checkpoint_epoch);
        } else {
            best_valid_loss = historical_best.best_loss;
            best_valid_epoch = historical_best.best_checkpoint_epoch;
            stability.best_valid_loss = historical_best.best_loss;
            stability.best_checkpoint_epoch = historical_best.best_checkpoint_epoch;
        }
        info!(
            "resumed dynamic training checkpoint epoch={} runtime_state_restored={} context_routing_restored={} historical_best_valid_loss={:?} historical_best_checkpoint_epoch={:?}",
            epoch, runtime_restored, context_routing_restored, best_valid_loss, best_valid_epoch
        );
    }

    let dynamic_neuron_scaling = env.neuron_scaling_slot.is_some();
    let update_parameters = parameter_updates_enabled(env.training);
    info!("run name: {}", env.run_name);
    info!(
        "training strategy: mode={:?} replicas={} event_scheduler=true dynamic_neuron_scaling={} parameter_updates={} start_epoch={}",
        env.parallel_runtime.mode,
        env.devices.len(),
        dynamic_neuron_scaling,
        update_parameters,
        start_epoch
    );
    info!(
        "checkpoint policy: logical_epoch_steps={} keep_last={} keep_best_valid_loss=true event_scheduler=true dynamic_neuron_scaling={}",
        env.train_loader.num_items(),
        CHECKPOINT_KEEP_LAST,
        dynamic_neuron_scaling
    );

    for epoch in start_epoch..=env.epochs {
        if let Some(reason) = training_interruption_reason(&event_handles.interrupter) {
            info!("Training interrupted: {reason}");
            break;
        }

        let mut iterator = active_train_loader.iter();
        let mut iteration = 0usize;
        let mut accumulator = GradientsAccumulator::new();
        let mut accumulation_current = 0usize;
        let mut last_lr = 0.0;
        let mut stop_requested = false;
        let mut token_exposure = TrainingTokenExposure::default();

        while let Some(item) = iterator.next() {
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

            model.set_recovery_auxiliary_active(dynamics_control.recovery_auxiliary_active());
            let [rows, time] = item.inputs.dims();
            token_exposure.observe(
                rows.saturating_mul(time),
                if item.loss_mask.is_none() {
                    Some(rows.saturating_mul(time))
                } else {
                    item.supervised_token_count
                },
            );
            let reset_stream_state = item.reset_stream_state;
            let (item, selected_context) = if let Some(routing) = context_routing.as_mut() {
                let decision = routing.route(&model, &item, absolute_step)?;
                let masks = routing.masks(decision.identity)?;
                let state = routing.take_stream_state(decision.identity, reset_stream_state)?;
                let step =
                    model.predictive_context_train_step(item, masks.neuron, masks.activity, state);
                routing.store_stream_state(decision.identity, step.terminal_state)?;
                emit_predictive_context_routing_metrics(
                    env,
                    epoch,
                    iteration,
                    absolute_step,
                    routing.known_contexts(),
                    &decision,
                    &bus,
                );
                (step.output, Some(decision.identity))
            } else {
                (burn_train::TrainStep::step(&model, item), None)
            };
            let source_selection_due = source_selection_telemetry_due(env, absolute_step);
            let log_train_metrics = iteration.is_multiple_of(env.training.log_frequency.max(1))
                || iteration == steps_per_epoch;
            let mean_train_loss = if source_selection_due || log_train_metrics {
                let metric_sync_started =
                    crate::train::profile::enabled().then(burn_dragon_time::Instant::now);
                let train_output = item.item.sync();
                if let Some(started) = metric_sync_started {
                    crate::train::profile::record_metric_sync(started.elapsed().as_nanos());
                }
                let loss_value: LossValue<ValidBackend<B>> = train_output.adapt();
                Some(mean_scalar_from_loss(loss_value.value()))
            } else {
                None
            };
            if source_selection_due && let Some(mean_train_loss) = mean_train_loss {
                emit_source_selection_telemetry(env, absolute_step, mean_train_loss, &bus);
            }
            if update_parameters {
                accumulator.accumulate(&model, item.grads);
                accumulation_current += 1;
            } else {
                last_lr = 0.0;
            }

            if update_parameters && active_grad_accumulation <= accumulation_current {
                if apply_pending_dynamics_control(
                    env,
                    &dynamics_control_slot,
                    &mut dynamics_control,
                    &mut optimizer,
                    &model,
                ) == DynamicsControlOutcome::Stop
                {
                    stop_requested = true;
                    break;
                }
                let lr = scheduler.step() * dynamics_control.lr_scale;
                let grads = accumulator.grads();
                let optimizer_started = (crate::train::profile::enabled()
                    && !model.uses_incremental_predictive_coding())
                .then(burn_dragon_time::Instant::now);
                model = if let (Some(routing), Some(identity)) =
                    (context_routing.as_mut(), selected_context)
                {
                    routing.optimizer_mut(identity)?.step(lr, model, grads)
                } else {
                    burn_train::TrainStep::optimize::<B, _>(model, &mut optimizer, lr, grads)
                };
                if let Some(started) = optimizer_started {
                    crate::train::profile::record_optimizer(started.elapsed().as_nanos());
                }
                accumulation_current = 0;
                last_lr = lr;
                emit_continual_backprop_telemetry(
                    env,
                    &optimizer,
                    epoch,
                    absolute_step,
                    &bus,
                    &mut last_cbp_telemetry_step,
                );
            }

            if log_train_metrics && let Some(mean_train_loss) = mean_train_loss {
                let metric = TrainingMetricSample {
                    run_id: env.run_name.to_string().into(),
                    split: TrainingMetricSplit::Train,
                    epoch,
                    step_in_epoch: iteration,
                    absolute_step,
                    name: train_loss_metric_name(env.training).to_string(),
                    value: mean_train_loss,
                    running_value: mean_train_loss,
                };
                let _ = bus.send_metric_sample(metric.clone());
                if let Some(name) = train_objective_metric_name(env.training, absolute_step) {
                    let _ = bus.send_metric_sample(TrainingMetricSample {
                        name: name.to_string(),
                        ..metric
                    });
                }
                let _ = bus.send_metric_sample(TrainingMetricSample {
                    run_id: env.run_name.to_string().into(),
                    split: TrainingMetricSplit::Train,
                    epoch,
                    step_in_epoch: iteration,
                    absolute_step,
                    name: "Learning Rate".to_string(),
                    value: last_lr,
                    running_value: last_lr,
                });
                emit_predictive_coding_telemetry(
                    env,
                    epoch,
                    iteration,
                    absolute_step,
                    model.gradient_scale_step_index(),
                    &bus,
                );
                emit_latent_reasoning_telemetry(env, epoch, iteration, absolute_step, &bus);
                for (name, value) in token_exposure.metrics() {
                    let _ = bus.send_metric_sample(TrainingMetricSample {
                        run_id: env.run_name.to_string().into(),
                        split: TrainingMetricSplit::Train,
                        epoch,
                        step_in_epoch: iteration,
                        absolute_step,
                        name: name.to_string(),
                        value,
                        running_value: value,
                    });
                }
            }
            if emit_step_events {
                let _ = bus.send_step_finished(StepFinished {
                    run_id: env.run_name.to_string().into(),
                    absolute_step,
                    epoch,
                    loss: mean_train_loss,
                });
            }

            if log_train_metrics {
                let progress = iterator.progress();
                info!(
                    "train epoch={} step={}/{} loss={:.4} lr={:.6} global_progress={}/{}",
                    epoch,
                    progress.items_processed,
                    progress.items_total,
                    mean_train_loss.unwrap_or(f64::NAN),
                    last_lr,
                    epoch,
                    env.epochs
                );
            }
        }

        if stop_requested {
            info!("stopping training after dynamics control request");
            break;
        }

        if update_parameters && accumulation_current > 0 {
            if apply_pending_dynamics_control(
                env,
                &dynamics_control_slot,
                &mut dynamics_control,
                &mut optimizer,
                &model,
            ) == DynamicsControlOutcome::Stop
            {
                info!("stopping training after dynamics control request");
                break;
            }
            let lr = scheduler.step() * dynamics_control.lr_scale;
            let grads = accumulator.grads();
            let optimizer_started = (crate::train::profile::enabled()
                && !model.uses_incremental_predictive_coding())
            .then(burn_dragon_time::Instant::now);
            model = if let Some(routing) = context_routing.as_mut() {
                let identity = routing.current_identity().ok_or_else(|| {
                    anyhow!("predictive context gradient remainder has no selected context")
                })?;
                routing.optimizer_mut(identity)?.step(lr, model, grads)
            } else {
                burn_train::TrainStep::optimize::<B, _>(model, &mut optimizer, lr, grads)
            };
            if let Some(started) = optimizer_started {
                crate::train::profile::record_optimizer(started.elapsed().as_nanos());
            }
            let absolute_step = epoch_end_absolute_step(epoch, steps_per_epoch, iteration);
            emit_continual_backprop_telemetry(
                env,
                &optimizer,
                epoch,
                absolute_step,
                &bus,
                &mut last_cbp_telemetry_step,
            );
        }
        drop(iterator);
        // Preserve causal ordering across independently typed ECS messages at
        // the epoch boundary without adding a barrier to the per-step path.
        let _ = bus.flush();
        let _ = bus.send_epoch_summary(TrainingEpochSummary {
            run_id: env.run_name.to_string().into(),
            split: TrainingMetricSplit::Train,
            epoch,
        });

        if !env.training.validation.execution.is_local() {
            let absolute_step = epoch_end_absolute_step(epoch, steps_per_epoch, iteration);
            let checkpoint_started = burn_dragon_time::Instant::now();
            save_dragon_training_state_checkpoint(
                DragonTrainingCheckpointContext {
                    run_dir: env.run_dir,
                    epoch,
                    completed_steps: absolute_step.saturating_add(1),
                    model_config: &current_model_config,
                    dynamics_control: &dynamics_control,
                },
                &model,
                &optimizer,
                &scheduler,
            )?;
            if let Some(routing) = context_routing.as_ref() {
                routing.save_checkpoint(env.run_dir, epoch)?;
            }
            save_source_selection_state_checkpoint(
                env.run_dir,
                epoch,
                absolute_step,
                env.source_selection_dataset.as_ref(),
            )?;
            let _ = bus.send_checkpoint(CheckpointEvent {
                run_id: env.run_name.to_string().into(),
                checkpoint_id: format!("model-{epoch}"),
                epoch: Some(epoch),
                absolute_step: Some(absolute_step),
                promoted: false,
            });
            let _ = bus.flush();
            save_continual_learning_stability_checkpoint(env.run_dir, epoch, &stability)?;
            crate::train::events::save_training_event_state_checkpoint(
                &event_handles,
                env.run_name,
                env.run_dir,
                epoch,
            )?;
            prune_dragon_model_checkpoints(env.run_dir, epoch, &[None, None])?;
            crate::train::profile::record_checkpoint(checkpoint_started.elapsed().as_nanos());
            info!(
                "validation deferred to external evaluator epoch={epoch}; candidate checkpoint is unpromoted"
            );
            continue;
        }

        let validation_started = burn_dragon_time::Instant::now();
        let absolute_step = epoch_end_absolute_step(epoch, steps_per_epoch, iteration);
        let validation = run_dynamic_validation(
            DynamicValidation {
                env,
                valid_loader: &active_valid_loader,
                model: &model,
                batch_size: active_batch_size,
                bus: &bus,
                context_routing: context_routing.as_ref(),
            },
            epoch,
            absolute_step,
        )?;
        crate::train::profile::record_validation(validation_started.elapsed().as_nanos());
        let valid_loss = validation.primary_loss();
        info!(
            "valid epoch={} primary_loss={valid_loss:.4} cold_window_loss={:.4}",
            epoch, validation.loss
        );
        if let Some(source_weighted_loss) = validation.source_weighted_loss {
            info!(
                "valid epoch={} source_weighted_loss={source_weighted_loss:.4}",
                epoch
            );
        }
        if let Some(stream_warm_loss) = validation.stream_warm_loss {
            info!(
                "valid epoch={} stream_warm_loss={stream_warm_loss:.4}",
                epoch
            );
        }
        let checkpoint_promoted = should_promote_checkpoint(
            &validation,
            best_valid_loss,
            stability.best_ruliad_competence,
            stability.best_ruliad_policy_competence,
            env.training,
        );
        if checkpoint_promoted {
            best_valid_loss = Some(valid_loss);
            best_valid_epoch = Some(epoch);
            stability.best_checkpoint_epoch = Some(epoch);
            if let Some(competence) = validation
                .ruliad_eval_report
                .as_ref()
                .and_then(ruliad_competence_key)
            {
                stability.best_ruliad_competence = Some(competence);
            }
            if let Some(competence) = validation
                .ruliad_policy_rollout
                .as_ref()
                .and_then(ruliad_policy_competence_key)
            {
                stability.best_ruliad_policy_competence = Some(competence);
            }
        }
        let checkpoint_started = burn_dragon_time::Instant::now();
        save_dragon_training_state_checkpoint(
            DragonTrainingCheckpointContext {
                run_dir: env.run_dir,
                epoch,
                completed_steps: absolute_step.saturating_add(1),
                model_config: &current_model_config,
                dynamics_control: &dynamics_control,
            },
            &model,
            &optimizer,
            &scheduler,
        )?;
        if let Some(routing) = context_routing.as_ref() {
            routing.save_checkpoint(env.run_dir, epoch)?;
        }
        save_source_selection_state_checkpoint(
            env.run_dir,
            epoch,
            absolute_step,
            env.source_selection_dataset.as_ref(),
        )?;
        update_ruliad_recovery_checkpoint(env, &validation, epoch, &mut stability);
        apply_continual_learning_stability_policy(
            env,
            validation,
            epoch,
            absolute_step,
            &mut stability,
            &bus,
        );
        let _ = bus.send_checkpoint(CheckpointEvent {
            run_id: env.run_name.to_string().into(),
            checkpoint_id: format!("model-{epoch}"),
            epoch: Some(epoch),
            absolute_step: Some(absolute_step),
            promoted: checkpoint_promoted,
        });
        let _ = bus.flush();
        save_continual_learning_stability_checkpoint(env.run_dir, epoch, &stability)?;
        crate::train::events::save_training_event_state_checkpoint(
            &event_handles,
            env.run_name,
            env.run_dir,
            epoch,
        )?;
        prune_dragon_model_checkpoints(
            env.run_dir,
            epoch,
            &[best_valid_epoch, stability.best_ruliad_checkpoint_epoch],
        )?;
        crate::train::profile::record_checkpoint(checkpoint_started.elapsed().as_nanos());
        if handle_post_validation_dynamics_control(
            env,
            &dynamics_control_slot,
            DynamicTrainingState {
                active: &mut dynamics_control,
                optimizer: &mut optimizer,
                scheduler: &mut scheduler,
                model: &mut model,
                model_config: &mut current_model_config,
            },
            epoch,
        )? == DynamicsControlOutcome::Stop
        {
            info!("stopping training after dynamics control request");
            break;
        }

        if let Some(request) = env
            .neuron_scaling_slot
            .as_ref()
            .and_then(|slot| slot.take())
        {
            if let Some((old_capacity_units, new_capacity_units)) = apply_dynamic_neuron_scale(
                env,
                DynamicNeuronScaleState {
                    model: &mut model,
                    optimizer: &mut optimizer,
                    model_config: &mut current_model_config,
                    scale_generation: &mut scale_generation,
                    batch_size: active_batch_size,
                    gradient_accumulation_steps: active_grad_accumulation,
                },
                request,
                TrainingEventContext {
                    epoch,
                    absolute_step,
                    bus: &bus,
                },
            )? {
                let next_batch_size =
                    crate::train::startup_autotune::resolve_scaled_auto_batch_size(
                        &env.training.auto_batch_size,
                        active_batch_size,
                        old_capacity_units,
                        new_capacity_units,
                    );
                let next_grad_accumulation =
                    crate::train::startup_autotune::resolve_gradient_accumulation_steps(
                        next_batch_size,
                        env.training.gradient_accumulation_steps,
                        env.training.target_effective_batch_size,
                    );
                if next_batch_size != active_batch_size
                    || next_grad_accumulation != active_grad_accumulation
                {
                    active_batch_size = next_batch_size;
                    active_grad_accumulation = next_grad_accumulation;
                    let consumed_steps = epoch.saturating_mul(steps_per_epoch);
                    active_train_loader =
                        build_dynamic_train_loader(env, active_batch_size, consumed_steps);
                    active_valid_loader = build_dynamic_valid_loader(env, active_batch_size);
                    info!(
                        "auto batch after neuron scale: batch_size={} gradient_accumulation_steps={} effective_batch_size={} consumed_steps={}",
                        active_batch_size,
                        active_grad_accumulation,
                        active_batch_size.saturating_mul(active_grad_accumulation),
                        consumed_steps
                    );
                    emit_policy_gate(
                        env,
                        &bus,
                        "auto_batch_size_after_neuron_scale",
                        epoch,
                        absolute_step,
                        format!(
                            "auto batch selected batch_size={} gradient_accumulation_steps={} after capacity {} -> {}",
                            active_batch_size,
                            active_grad_accumulation,
                            old_capacity_units,
                            new_capacity_units
                        ),
                    );
                }
            }
            let _ = bus.flush();
        }
    }

    log_theoretical_profile(
        &current_model_config,
        active_batch_size.saturating_mul(active_grad_accumulation),
        env.training.block_size,
        env.backend_name,
    );

    Ok(model.valid().model)
}

pub(super) fn run_dynamic_validation<B>(
    validation: DynamicValidation<'_, '_, B>,
    epoch: usize,
    training_absolute_step: usize,
) -> Result<DynamicValidationReport>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let DynamicValidation {
        env,
        valid_loader,
        model,
        batch_size,
        bus,
        context_routing,
    } = validation;
    let steps_per_epoch = env.train_loader.num_items().max(1);
    let valid_model = model.valid().materialize_random_scaffold_for_inference();
    let iterator = valid_loader.iter();
    let mut supervised_loss = SupervisedValidationLoss::default();
    let mut count = 0usize;
    let mut output_degeneracy = None;
    let mut latent_eval_sweep_emitted = false;
    let probe_enabled = epoch.is_multiple_of(env.training.events.degeneracy_probe_every_epochs);
    let probe_absolute_step = training_absolute_step;
    for item in iterator {
        let supervised_tokens = sequence_batch_supervised_tokens(&item);
        let eval_sweep_enabled = context_routing.is_none()
            && !latent_eval_sweep_emitted
            && !latent_eval_step_sweep_for_model(env.training, &valid_model).is_empty();
        let degeneracy_probe_enabled = probe_enabled && output_degeneracy.is_none();
        let item_for_eval_sweep = item.clone();
        let (loss_tensor, degeneracy) = if let Some(routing) = context_routing {
            let probe_tokens = if degeneracy_probe_enabled {
                env.training.events.degeneracy_probe_tokens
            } else {
                0
            };
            let (loss, identity, degeneracy) = routing.validation_loss(
                &valid_model,
                item,
                probe_tokens,
                dataset_eos_id(env.source_selection_dataset.as_ref()),
            )?;
            let _ = bus.send_metric_sample(TrainingMetricSample {
                run_id: env.run_name.to_string().into(),
                split: TrainingMetricSplit::Valid,
                epoch,
                step_in_epoch: count.saturating_add(1),
                absolute_step: probe_absolute_step,
                name: "Predictive Context Index".to_string(),
                value: identity.context_index as f64,
                running_value: identity.context_index as f64,
            });
            (loss, degeneracy)
        } else if degeneracy_probe_enabled {
            valid_model.validation_loss_and_output_degeneracy(
                item,
                env.training.events.degeneracy_probe_tokens,
                dataset_eos_id(env.source_selection_dataset.as_ref()),
            )
        } else {
            let output = valid_model.step(item);
            let loss_value: LossValue<ValidBackend<B>> = output.adapt();
            (loss_value.value(), None)
        };
        let loss = mean_scalar_from_loss(loss_tensor);
        count += 1;
        supervised_loss.observe(loss, supervised_tokens);
        let absolute_step = training_absolute_step;
        if let Some(degeneracy) = degeneracy {
            emit_output_degeneracy(env, epoch, probe_absolute_step, &degeneracy, bus);
            output_degeneracy = Some(degeneracy);
        }
        if eval_sweep_enabled {
            emit_latent_eval_step_validation_sweep(LatentEvalSweep {
                run_name: env.run_name,
                training: env.training,
                source_selection_dataset: env.source_selection_dataset.as_ref(),
                model: &valid_model,
                batch: item_for_eval_sweep,
                eos_id: dataset_eos_id(env.source_selection_dataset.as_ref()),
                include_degeneracy: degeneracy_probe_enabled,
                event: TrainingEventContext {
                    epoch,
                    absolute_step: probe_absolute_step,
                    bus,
                },
            });
            latent_eval_sweep_emitted = true;
        }
        if supervised_tokens > 0 {
            let running_loss = supervised_loss
                .mean()
                .expect("a supervised validation batch has a mean loss");
            let _ = bus.send_metric_sample(TrainingMetricSample {
                run_id: env.run_name.to_string().into(),
                split: TrainingMetricSplit::Valid,
                epoch,
                step_in_epoch: count,
                absolute_step,
                name: "Loss".to_string(),
                value: loss,
                running_value: running_loss,
            });
            emit_teacher_forced_validation_metric(
                env.run_name,
                env.source_selection_dataset.as_ref(),
                count,
                loss,
                running_loss,
                TrainingEventContext {
                    epoch,
                    absolute_step,
                    bus,
                },
            );
        }
    }
    let mean = supervised_loss.mean().unwrap_or(f64::NAN);
    emit_validation_supervision_metrics(
        env.run_name,
        epoch,
        training_absolute_step,
        supervised_loss,
        bus,
    );
    if env.training.tbptt_persist_across_steps && mean.is_finite() {
        let absolute_step = training_absolute_step;
        let _ = bus.send_metric_sample(TrainingMetricSample {
            run_id: env.run_name.to_string().into(),
            split: TrainingMetricSplit::Valid,
            epoch,
            step_in_epoch: count.saturating_add(1),
            absolute_step,
            name: METRIC_RANDOM_COLD_LOSS.to_string(),
            value: mean,
            running_value: mean,
        });
    }
    let source_weighted_loss = run_source_weighted_validation(
        env,
        &valid_model,
        steps_per_epoch,
        batch_size,
        context_routing,
        TrainingEventContext {
            epoch,
            absolute_step: training_absolute_step,
            bus,
        },
    )?;
    let correctness_request = RuliadCorrectnessValidation {
        run_name: env.run_name,
        run_dir: env.run_dir,
        training: env.training,
        dataset: env
            .source_selection_dataset
            .as_ref()
            .or(env.valid_dataset.as_ref()),
        model: &valid_model,
        training_batch_size: batch_size,
        device: env.device,
        output_degeneracy: output_degeneracy.as_ref(),
        event: TrainingEventContext {
            epoch,
            absolute_step: training_absolute_step,
            bus,
        },
    };
    let ruliad_validation = if let Some(routing) = context_routing {
        let router = routing.validation_router();
        run_routed_ruliad_correctness_validation(correctness_request, &router)?
    } else {
        run_ruliad_correctness_validation(correctness_request)?
    };
    let ruliad_eval_report = ruliad_validation
        .as_ref()
        .map(|validation| validation.free_run.clone());
    let ruliad_constrained_policy = ruliad_validation
        .as_ref()
        .and_then(|validation| validation.constrained_policy.clone());
    let ruliad_policy_rollout =
        ruliad_validation.and_then(|validation| validation.closed_loop_policy);
    if let Some(report) = ruliad_eval_report.as_ref() {
        let _ = emit_ruliad_capability_gate_metrics(
            env.run_name,
            report,
            output_degeneracy.as_ref(),
            &env.training.gates,
            env.training
                .ruliad_policy_probe
                .checkpoint_capability_contract
                .requires_free_run(),
            TrainingEventContext {
                epoch,
                absolute_step: training_absolute_step,
                bus,
            },
        );
    }
    if ruliad_eval_report.is_some() || ruliad_policy_rollout.is_some() {
        let deployment_capability_gate = ruliad_deployment_capability_gate_status(
            ruliad_eval_report.as_ref(),
            ruliad_policy_rollout.as_ref(),
            ruliad_constrained_policy.as_ref(),
            output_degeneracy.as_ref(),
            env.training,
        );
        emit_ruliad_deployment_capability_gate_metrics(
            env.run_name,
            epoch,
            training_absolute_step,
            &deployment_capability_gate,
            bus,
        );
        if deployment_capability_gate.passed {
            model.set_latent_reasoning_capability_gate_open(true);
        }
    }
    let stream_warm_report =
        if env.training.tbptt_persist_across_steps || env.training.sequence_state_probe.enabled {
            Some(run_stream_warm_validation(
                env,
                model,
                batch_size,
                context_routing,
                TrainingEventContext {
                    epoch,
                    absolute_step: training_absolute_step,
                    bus,
                },
            )?)
        } else {
            None
        };
    let stream_warm_loss = stream_warm_report.and_then(|report| report.warm_loss);
    if let Some(report) = stream_warm_report
        && let (Some(cold), Some(gain), Some(relative_gain)) = (
            report.paired_cold_loss,
            report.carry_nll_gain,
            report.carry_relative_gain,
        )
    {
        info!(
            "valid epoch={} stream_carry paired_batches={} cold_loss={cold:.6} nll_gain={gain:.6} relative_gain={relative_gain:.4}",
            epoch, report.paired_batches
        );
    }
    if let Some(source_weighted_loss) = source_weighted_loss
        && mean.is_finite()
    {
        let delta = source_weighted_loss - mean;
        let ratio = if mean.abs() <= f64::EPSILON {
            0.0
        } else {
            source_weighted_loss / mean
        };
        let absolute_step = training_absolute_step;
        for (name, value) in [
            ("Source Weighted Loss Delta", delta),
            ("Source Weighted Loss Ratio", ratio),
        ] {
            let _ = bus.send_metric_sample(TrainingMetricSample {
                run_id: env.run_name.to_string().into(),
                split: TrainingMetricSplit::Valid,
                epoch,
                step_in_epoch: count.saturating_add(1),
                absolute_step,
                name: name.to_string(),
                value,
                running_value: value,
            });
        }
    }
    let _ = bus.send_epoch_summary(TrainingEpochSummary {
        run_id: env.run_name.to_string().into(),
        split: TrainingMetricSplit::Valid,
        epoch,
    });
    let validation_objective = env.training.validation.objective;
    let validation_objective_loss = select_validation_objective_loss(
        validation_objective,
        mean,
        source_weighted_loss,
        stream_warm_loss,
    )?;
    emit_validation_objective_loss(
        env.run_name,
        epoch,
        training_absolute_step,
        validation_objective,
        validation_objective_loss,
        bus,
    );
    let _ = bus.send_validation_finished(ValidationFinished {
        run_id: env.run_name.to_string().into(),
        epoch,
        absolute_step: Some(training_absolute_step),
        objective: validation_objective.as_str().to_string(),
        loss: Some(validation_objective_loss),
    });
    Ok(DynamicValidationReport {
        objective: validation_objective,
        loss: mean,
        source_weighted_loss,
        stream_warm_loss,
        output_degeneracy,
        ruliad_eval_report,
        ruliad_policy_rollout,
        ruliad_constrained_policy,
    })
}

pub(super) fn run_stream_warm_validation<B>(
    env: &TrainEnvironment<'_, B>,
    model: &LanguageTrainModel<B>,
    batch_size: usize,
    context_routing: Option<&crate::train::PredictiveContextRoutingRuntime<B>>,
    event: TrainingEventContext<'_>,
) -> Result<StreamWarmValidationReport>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    // Validation advances its own batch counter, never the training clock.
    let TrainingEventContext {
        epoch,
        absolute_step,
        bus,
    } = event;
    let Some(valid_dataset) = env.valid_dataset.as_ref() else {
        return Ok(StreamWarmValidationReport::default());
    };
    let loader = StreamingDataLoader::<ValidBackend<B>>::new(
        Arc::clone(valid_dataset),
        DatasetSplit::Val,
        env.device,
        env.valid_steps.max(1),
        None,
        env.training.min_logical_block_size,
        env.training.validation.seed,
    )
    .with_source_selection_enabled(
        env.training
            .validation
            .sampling
            .uses_live_source_selection(),
    )
    .with_batch_size(batch_size.max(1))
    .with_summary_event_token_ids(env.summary_event_token_ids.clone());
    let valid_model = model.valid().materialize_random_scaffold_for_inference();
    let mut state = valid_model.model.init_state();
    let context_router = context_routing.map(|routing| routing.validation_router());
    let mut context_states = context_router
        .as_ref()
        .map(|router| {
            (0..router.known_contexts())
                .map(|_| valid_model.model.init_state())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let iterator = loader.iter();
    let mut count = 0usize;
    let mut warm = SupervisedValidationLoss::default();
    let mut paired_warm = SupervisedValidationLoss::default();
    let mut paired_cold = SupervisedValidationLoss::default();
    let mut paired_count = 0usize;
    let probe = &env.training.sequence_state_probe;
    for item in iterator {
        let supervised_tokens = sequence_batch_supervised_tokens(&item);
        let paired_item = (probe.enabled
            && supervised_tokens > 0
            && !item.reset_stream_state
            && paired_count < probe.paired_batches)
            .then(|| item.clone());
        let (output, selected_route) = if let Some(router) = context_router.as_ref() {
            let route = router.select_batch(&valid_model, &item)?;
            let route_state = context_states
                .get_mut(route.identity.context_index)
                .ok_or_else(|| anyhow!("routed validation selected missing context state"))?;
            let output = valid_model.step_with_predictive_context_stream_state(
                item,
                route.masks.neuron.clone(),
                route.masks.activity.clone(),
                route_state,
            );
            (output, Some(route))
        } else {
            (valid_model.step_with_stream_state(item, &mut state), None)
        };
        let loss_value: LossValue<ValidBackend<B>> = output.adapt();
        let loss = mean_scalar_from_loss(loss_value.value());
        count += 1;
        warm.observe(loss, supervised_tokens);
        if supervised_tokens > 0 {
            let _ = bus.send_metric_sample(TrainingMetricSample {
                run_id: env.run_name.to_string().into(),
                split: TrainingMetricSplit::Valid,
                epoch,
                step_in_epoch: count,
                absolute_step,
                name: METRIC_STREAM_WARM_LOSS.to_string(),
                value: loss,
                running_value: warm.mean().expect("nonempty supervised batch"),
            });
        }
        if let Some(cold_item) = paired_item {
            let cold_output = if let Some(route) = selected_route {
                let mut cold_state = valid_model.model.init_state();
                valid_model.step_with_predictive_context_stream_state(
                    cold_item,
                    route.masks.neuron,
                    route.masks.activity,
                    &mut cold_state,
                )
            } else {
                valid_model.step(cold_item)
            };
            let cold_loss_value: LossValue<ValidBackend<B>> = cold_output.adapt();
            let cold_loss = mean_scalar_from_loss(cold_loss_value.value());
            paired_warm.observe(loss, supervised_tokens);
            paired_cold.observe(cold_loss, supervised_tokens);
            paired_count = paired_count.saturating_add(1);
        }
    }

    let mut report = StreamWarmValidationReport {
        warm_loss: warm.mean(),
        paired_batches: paired_count,
        ..Default::default()
    };
    if paired_count > 0 {
        let paired_warm_loss = paired_warm.mean().expect("nonempty paired supervision");
        let paired_cold_loss = paired_cold.mean().expect("nonempty paired supervision");
        let carry_nll_gain = paired_cold_loss - paired_warm_loss;
        let carry_relative_gain = if paired_cold_loss.abs() <= f64::EPSILON {
            0.0
        } else {
            carry_nll_gain / paired_cold_loss.abs()
        };
        report.paired_warm_loss = Some(paired_warm_loss);
        report.paired_cold_loss = Some(paired_cold_loss);
        report.carry_nll_gain = Some(carry_nll_gain);
        report.carry_relative_gain = Some(carry_relative_gain);
        for (name, value) in [
            (METRIC_STREAM_PAIRED_WARM_LOSS, paired_warm_loss),
            (METRIC_STREAM_PAIRED_COLD_LOSS, paired_cold_loss),
            (METRIC_STREAM_CARRY_NLL_GAIN, carry_nll_gain),
            (METRIC_STREAM_CARRY_RELATIVE_GAIN, carry_relative_gain),
            (METRIC_STREAM_CARRY_PROBE_BATCHES, paired_count as f64),
        ] {
            let _ = bus.send_metric_sample(TrainingMetricSample {
                run_id: env.run_name.to_string().into(),
                split: TrainingMetricSplit::Valid,
                epoch,
                step_in_epoch: count.saturating_add(1),
                absolute_step,
                name: name.to_string(),
                value,
                running_value: value,
            });
        }
    }
    for (name, value) in [
        ("Stream Validation Contract Version", 2.0),
        (
            "Stream Validation Supervision Weight",
            warm.supervised_tokens as f64,
        ),
        ("Stream Validation Empty Batches", warm.empty_batches as f64),
        (
            "Stream Paired Supervision Weight",
            paired_warm.supervised_tokens as f64,
        ),
    ] {
        let _ = bus.send_metric_sample(TrainingMetricSample {
            run_id: env.run_name.to_string().into(),
            split: TrainingMetricSplit::Valid,
            epoch,
            step_in_epoch: count.saturating_add(1),
            absolute_step,
            name: name.to_string(),
            value,
            running_value: value,
        });
    }
    let diagnostics = if !probe.enabled {
        None
    } else if context_router.is_some() {
        let diagnostics = context_states
            .iter()
            .filter_map(|state| {
                LanguageTrainModel::<ValidBackend<B>>::sequence_state_diagnostics(
                    state,
                    probe.max_rho_slots,
                )
            })
            .collect::<Vec<_>>();
        (!diagnostics.is_empty()).then(|| {
            let count = diagnostics.len() as f64;
            crate::train::steps::SequenceStateDiagnostics {
                rho_layers: diagnostics
                    .iter()
                    .map(|item| item.rho_layers)
                    .max()
                    .unwrap_or_default(),
                rho_rms: diagnostics.iter().map(|item| item.rho_rms).sum::<f64>() / count,
                rho_slot_variance_ratio: diagnostics
                    .iter()
                    .map(|item| item.rho_slot_variance_ratio)
                    .sum::<f64>()
                    / count,
                rho_slot_redundancy: diagnostics
                    .iter()
                    .map(|item| item.rho_slot_redundancy)
                    .sum::<f64>()
                    / count,
            }
        })
    } else {
        LanguageTrainModel::<ValidBackend<B>>::sequence_state_diagnostics(
            &state,
            probe.max_rho_slots,
        )
    };
    if let Some(diagnostics) = diagnostics {
        for (name, value) in [
            (METRIC_RHO_RMS, diagnostics.rho_rms),
            (
                METRIC_RHO_SLOT_VARIANCE_RATIO,
                diagnostics.rho_slot_variance_ratio,
            ),
            (METRIC_RHO_SLOT_REDUNDANCY, diagnostics.rho_slot_redundancy),
            (METRIC_RHO_LAYERS, diagnostics.rho_layers as f64),
        ] {
            let _ = bus.send_metric_sample(TrainingMetricSample {
                run_id: env.run_name.to_string().into(),
                split: TrainingMetricSplit::Valid,
                epoch,
                step_in_epoch: count.saturating_add(1),
                absolute_step,
                name: name.to_string(),
                value,
                running_value: value,
            });
        }
    }
    Ok(report)
}

pub(super) fn run_dynamic_validation_forward_only<B>(
    env: &ForwardEggrollTrainEnvironment<'_, B>,
    model: &LanguageTrainModel<B>,
    epoch: usize,
    steps_per_epoch: usize,
    training_absolute_step: usize,
    bus: &TrainingEventBus,
) -> Result<DynamicValidationReport>
where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone,
{
    let iterator = env.valid_loader.iter();
    let mut supervised_loss = SupervisedValidationLoss::default();
    let mut count = 0usize;
    let mut output_degeneracy = None;
    let mut latent_eval_sweep_emitted = false;
    let probe_enabled = epoch.is_multiple_of(env.training.events.degeneracy_probe_every_epochs);
    let probe_absolute_step = training_absolute_step;
    for item in iterator {
        let supervised_tokens = sequence_batch_supervised_tokens(&item);
        let eval_sweep_enabled = !latent_eval_sweep_emitted
            && !latent_eval_step_sweep_for_model(env.training, model).is_empty();
        let degeneracy_probe_enabled = probe_enabled && output_degeneracy.is_none();
        let item_for_eval_sweep = item.clone();
        let (loss_tensor, degeneracy) = if degeneracy_probe_enabled {
            model.validation_loss_and_output_degeneracy(
                item,
                env.training.events.degeneracy_probe_tokens,
                dataset_eos_id(env.source_selection_dataset.as_ref()),
            )
        } else {
            let output = ValidStep::step(model, item);
            let loss_value: LossValue<B> = output.adapt();
            (loss_value.value(), None)
        };
        let loss = mean_scalar_from_loss(loss_tensor);
        count += 1;
        supervised_loss.observe(loss, supervised_tokens);
        let absolute_step = training_absolute_step;
        if let Some(degeneracy) = degeneracy {
            emit_output_degeneracy_sample(
                env.run_name,
                env.source_selection_dataset.as_ref(),
                epoch,
                probe_absolute_step,
                &degeneracy,
                bus,
            );
            output_degeneracy = Some(degeneracy);
        }
        if eval_sweep_enabled {
            emit_latent_eval_step_validation_sweep(LatentEvalSweep {
                run_name: env.run_name,
                training: env.training,
                source_selection_dataset: env.source_selection_dataset.as_ref(),
                model,
                batch: item_for_eval_sweep,
                eos_id: dataset_eos_id(env.source_selection_dataset.as_ref()),
                include_degeneracy: degeneracy_probe_enabled,
                event: TrainingEventContext {
                    epoch,
                    absolute_step: probe_absolute_step,
                    bus,
                },
            });
            latent_eval_sweep_emitted = true;
        }
        if supervised_tokens > 0 {
            let running_loss = supervised_loss
                .mean()
                .expect("a supervised validation batch has a mean loss");
            let _ = bus.send_metric_sample(TrainingMetricSample {
                run_id: env.run_name.to_string().into(),
                split: TrainingMetricSplit::Valid,
                epoch,
                step_in_epoch: count,
                absolute_step,
                name: "Loss".to_string(),
                value: loss,
                running_value: running_loss,
            });
            emit_teacher_forced_validation_metric(
                env.run_name,
                env.source_selection_dataset.as_ref(),
                count,
                loss,
                running_loss,
                TrainingEventContext {
                    epoch,
                    absolute_step,
                    bus,
                },
            );
        }
    }
    let mean = supervised_loss.mean().unwrap_or(f64::NAN);
    emit_validation_supervision_metrics(
        env.run_name,
        epoch,
        training_absolute_step,
        supervised_loss,
        bus,
    );
    let source_weighted_loss = run_source_weighted_validation_forward_only(
        env,
        model,
        steps_per_epoch,
        TrainingEventContext {
            epoch,
            absolute_step: training_absolute_step,
            bus,
        },
    )?;
    let ruliad_validation = run_ruliad_correctness_validation(RuliadCorrectnessValidation {
        run_name: env.run_name,
        run_dir: env.run_dir,
        training: env.training,
        dataset: env.source_selection_dataset.as_ref(),
        model,
        training_batch_size: env.training.batch_size,
        device: env.device,
        output_degeneracy: output_degeneracy.as_ref(),
        event: TrainingEventContext {
            epoch,
            absolute_step: training_absolute_step,
            bus,
        },
    })?;
    let ruliad_eval_report = ruliad_validation
        .as_ref()
        .map(|validation| validation.free_run.clone());
    let ruliad_constrained_policy = ruliad_validation
        .as_ref()
        .and_then(|validation| validation.constrained_policy.clone());
    let ruliad_policy_rollout =
        ruliad_validation.and_then(|validation| validation.closed_loop_policy);
    if let Some(report) = ruliad_eval_report.as_ref() {
        let _ = emit_ruliad_capability_gate_metrics(
            env.run_name,
            report,
            output_degeneracy.as_ref(),
            &env.training.gates,
            env.training
                .ruliad_policy_probe
                .checkpoint_capability_contract
                .requires_free_run(),
            TrainingEventContext {
                epoch,
                absolute_step: training_absolute_step,
                bus,
            },
        );
    }
    if ruliad_eval_report.is_some() || ruliad_policy_rollout.is_some() {
        let deployment_capability_gate = ruliad_deployment_capability_gate_status(
            ruliad_eval_report.as_ref(),
            ruliad_policy_rollout.as_ref(),
            ruliad_constrained_policy.as_ref(),
            output_degeneracy.as_ref(),
            env.training,
        );
        emit_ruliad_deployment_capability_gate_metrics(
            env.run_name,
            epoch,
            training_absolute_step,
            &deployment_capability_gate,
            bus,
        );
        if deployment_capability_gate.passed {
            model.set_latent_reasoning_capability_gate_open(true);
        }
    }
    if let Some(source_weighted_loss) = source_weighted_loss
        && mean.is_finite()
    {
        let delta = source_weighted_loss - mean;
        let ratio = if mean.abs() <= f64::EPSILON {
            0.0
        } else {
            source_weighted_loss / mean
        };
        let absolute_step = training_absolute_step;
        for (name, value) in [
            ("Source Weighted Loss Delta", delta),
            ("Source Weighted Loss Ratio", ratio),
        ] {
            let _ = bus.send_metric_sample(TrainingMetricSample {
                run_id: env.run_name.to_string().into(),
                split: TrainingMetricSplit::Valid,
                epoch,
                step_in_epoch: count.saturating_add(1),
                absolute_step,
                name: name.to_string(),
                value,
                running_value: value,
            });
        }
    }
    let _ = bus.send_epoch_summary(TrainingEpochSummary {
        run_id: env.run_name.to_string().into(),
        split: TrainingMetricSplit::Valid,
        epoch,
    });
    let validation_objective = env.training.validation.objective;
    let validation_objective_loss =
        select_validation_objective_loss(validation_objective, mean, source_weighted_loss, None)?;
    emit_validation_objective_loss(
        env.run_name,
        epoch,
        training_absolute_step,
        validation_objective,
        validation_objective_loss,
        bus,
    );
    let _ = bus.send_validation_finished(ValidationFinished {
        run_id: env.run_name.to_string().into(),
        epoch,
        absolute_step: Some(training_absolute_step),
        objective: validation_objective.as_str().to_string(),
        loss: Some(validation_objective_loss),
    });
    Ok(DynamicValidationReport {
        objective: validation_objective,
        loss: mean,
        source_weighted_loss,
        stream_warm_loss: None,
        output_degeneracy,
        ruliad_eval_report,
        ruliad_policy_rollout,
        ruliad_constrained_policy,
    })
}

#[cfg(test)]
mod validation_accounting_tests {
    use super::SupervisedValidationLoss;

    #[test]
    fn masked_validation_ignores_empty_batches_and_weights_by_supervised_tokens() {
        let mut summary = SupervisedValidationLoss::default();
        summary.observe(0.0, 0);
        summary.observe(2.0, 2);
        summary.observe(1.0, 6);

        assert_eq!(summary.supervised_tokens, 8);
        assert_eq!(summary.supervised_batches, 2);
        assert_eq!(summary.empty_batches, 1);
        assert_eq!(summary.mean(), Some(1.25));
    }

    #[test]
    fn empty_masked_validation_has_no_loss_measurement() {
        let mut summary = SupervisedValidationLoss::default();
        summary.observe(0.0, 0);

        assert_eq!(summary.mean(), None);
    }

    #[test]
    fn paired_carry_gain_uses_identical_nonempty_token_weights() {
        let mut warm = SupervisedValidationLoss::default();
        let mut cold = SupervisedValidationLoss::default();
        for (warm_loss, cold_loss, tokens) in [(0.0, 0.0, 0), (2.0, 3.0, 2), (1.0, 1.0, 6)] {
            warm.observe(warm_loss, tokens);
            cold.observe(cold_loss, tokens);
        }
        assert_eq!(warm.supervised_batches, 2);
        assert_eq!(warm.supervised_tokens, cold.supervised_tokens);
        assert_eq!(cold.mean().unwrap() - warm.mean().unwrap(), 0.25);
    }
}
