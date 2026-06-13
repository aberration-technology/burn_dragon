use crate::train::prelude::*;
use crate::train::utils::log_theoretical_profile;
#[cfg(feature = "ddp")]
use burn::tensor::TensorPrimitive;
use burn_dragon_train::train::events::{
    CheckpointEvent, ContinualBackpropSample, ModelScaleApplied, ModelScaleRequest,
    ModelScaleSkipped, OutputDegeneracySample, StepFinished, StepStarted, TrainingEpochSummary,
    TrainingEventBus, TrainingGateAction, TrainingGateEvent, TrainingGateSeverity,
    TrainingMetricSample, TrainingMetricSplit, ValidationFinished,
};
use std::collections::BTreeSet;
#[cfg(feature = "ddp")]
use std::collections::HashMap;
#[cfg(feature = "ddp")]
use std::marker::PhantomData;

const CHECKPOINT_KEEP_LAST: usize = 2;

#[derive(Clone, Debug, Default)]
struct ContinualLearningStabilityState {
    best_valid_loss: Option<f64>,
    consecutive_validation_regressions: usize,
    consecutive_output_degeneracy: usize,
}

#[derive(Clone, Debug, Default)]
struct DynamicValidationReport {
    loss: f64,
    source_weighted_loss: Option<f64>,
    output_degeneracy: Option<crate::train::steps::OutputDegeneracyStats>,
}

struct QuietMetricsRenderer;

impl burn_train::renderer::MetricsRendererTraining for QuietMetricsRenderer {
    fn update_train(&mut self, _state: burn_train::renderer::MetricState) {}

    fn update_valid(&mut self, _state: burn_train::renderer::MetricState) {}

    fn render_train(
        &mut self,
        _item: burn_train::renderer::TrainingProgress,
        _progress_indicators: Vec<burn_train::renderer::ProgressType>,
    ) {
    }

    fn render_valid(
        &mut self,
        _item: burn_train::renderer::TrainingProgress,
        _progress_indicators: Vec<burn_train::renderer::ProgressType>,
    ) {
    }

    fn on_train_end(
        &mut self,
        _summary: Option<burn_train::LearnerSummary>,
    ) -> std::result::Result<(), Box<dyn core::error::Error>> {
        Ok(())
    }
}

impl burn_train::renderer::MetricsRendererEvaluation for QuietMetricsRenderer {
    fn update_test(
        &mut self,
        _name: burn_train::renderer::EvaluationName,
        _state: burn_train::renderer::MetricState,
    ) {
    }

    fn render_test(
        &mut self,
        _item: burn_train::renderer::EvaluationProgress,
        _progress_indicators: Vec<burn_train::renderer::ProgressType>,
    ) {
    }

    fn on_test_end(
        &mut self,
        _summary: Option<burn_train::LearnerSummary>,
    ) -> std::result::Result<(), Box<dyn core::error::Error>> {
        Ok(())
    }
}

impl burn_train::renderer::MetricsRenderer for QuietMetricsRenderer {
    fn manual_close(&mut self) {}

    fn register_metric(&mut self, _definition: burn_train::metric::MetricDefinition) {}
}

fn quiet_progress_renderer_enabled() -> bool {
    std::env::var("DragonModel_TRAINING_PROGRESS_RENDERER")
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "quiet" | "none" | "off"
            )
        })
        .unwrap_or(false)
}

struct FileMetricBestCheckpointingStrategy {
    run_dir: PathBuf,
    metric_name: String,
    direction: burn_train::metric::store::Direction,
    split: burn_train::metric::store::Split,
    best_epoch: Option<usize>,
    best_value: Option<f64>,
}

impl FileMetricBestCheckpointingStrategy {
    fn new<M>(
        run_dir: &Path,
        metric: &M,
        direction: burn_train::metric::store::Direction,
        split: burn_train::metric::store::Split,
    ) -> Self
    where
        M: burn_train::metric::Metric,
    {
        Self {
            run_dir: run_dir.to_path_buf(),
            metric_name: metric.name().to_string(),
            direction,
            split,
            best_epoch: None,
            best_value: None,
        }
    }

    fn is_better(&self, candidate: f64, current: f64) -> bool {
        match self.direction {
            burn_train::metric::store::Direction::Lowest => candidate < current,
            burn_train::metric::store::Direction::Highest => candidate > current,
        }
    }

    fn checkpoint_path(&self, epoch: usize) -> PathBuf {
        self.run_dir
            .join("checkpoint")
            .join(format!("model-{epoch}.bin"))
    }

    fn metric_log_path(&self, epoch: usize) -> PathBuf {
        let split_dir = match self.split {
            burn_train::metric::store::Split::Train => "train",
            burn_train::metric::store::Split::Valid => "valid",
            burn_train::metric::store::Split::Test(_) => "test",
        };
        self.run_dir
            .join(split_dir)
            .join(format!("epoch-{epoch}"))
            .join(format!("{}.log", self.metric_name))
    }

    fn checkpoint_exists(&self, epoch: usize) -> bool {
        self.checkpoint_path(epoch).is_file()
    }

    fn existing_checkpoint_epochs(&self) -> BTreeSet<usize> {
        let checkpoint_dir = self.run_dir.join("checkpoint");
        let Ok(entries) = fs::read_dir(checkpoint_dir) else {
            return BTreeSet::new();
        };

        entries
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                let name = path.file_name()?.to_str()?;
                name.strip_prefix("model-")?
                    .strip_suffix(".bin")?
                    .parse::<usize>()
                    .ok()
            })
            .collect()
    }

    fn metric_mean_from_log(&self, epoch: usize) -> Option<f64> {
        let path = self.metric_log_path(epoch);
        let content = fs::read_to_string(path).ok()?;
        let mut sum = 0.0;
        let mut count = 0usize;

        for line in content.lines() {
            let field = line.split(',').next()?.trim();
            let value = field.parse::<f64>().ok()?;
            sum += value;
            count += 1;
        }

        (count > 0).then_some(sum / count as f64)
    }

    fn update_best_candidate(&mut self, epoch: usize, value: f64) -> Option<usize> {
        let should_replace = self
            .best_value
            .is_none_or(|current| self.is_better(value, current));

        if !should_replace {
            return None;
        }

        let previous_best = self.best_epoch.replace(epoch);
        self.best_value = Some(value);
        previous_best.filter(|previous_best| *previous_best != epoch)
    }

    fn refresh_best_from_history(&mut self, last_epoch: usize) {
        self.best_epoch = None;
        self.best_value = None;

        for epoch in 1..=last_epoch {
            if let Some(value) = self.metric_mean_from_log(epoch) {
                let _ = self.update_best_candidate(epoch, value);
            }
        }
    }

    fn refresh_best_from_store(
        &mut self,
        store: &burn_train::metric::store::EventStoreClient,
    ) -> bool {
        let Some(best_epoch) = store.find_epoch(
            &self.metric_name,
            burn_train::metric::store::Aggregate::Mean,
            self.direction,
            &self.split,
        ) else {
            return false;
        };

        self.best_epoch = Some(best_epoch);
        self.best_value = store.find_metric(
            &self.metric_name,
            best_epoch,
            burn_train::metric::store::Aggregate::Mean,
            &self.split,
        );
        true
    }

    fn checkpoint_actions_after_refresh(
        &self,
        epoch: usize,
    ) -> Vec<burn_train::checkpoint::CheckpointingAction> {
        let mut keep_epochs = BTreeSet::new();
        keep_epochs.extend(epoch.saturating_sub(CHECKPOINT_KEEP_LAST - 1).max(1)..=epoch);
        if let Some(best_epoch) = self.best_epoch {
            keep_epochs.insert(best_epoch);
        }

        let existing_epochs = self.existing_checkpoint_epochs();
        let mut actions = vec![burn_train::checkpoint::CheckpointingAction::Save];
        actions.extend(
            existing_epochs
                .into_iter()
                .filter(|existing_epoch| !keep_epochs.contains(existing_epoch))
                .map(burn_train::checkpoint::CheckpointingAction::Delete),
        );
        actions
    }

    fn actions_for_epoch(
        &mut self,
        epoch: usize,
    ) -> Vec<burn_train::checkpoint::CheckpointingAction> {
        self.refresh_best_from_history(epoch);
        self.checkpoint_actions_after_refresh(epoch)
    }

    fn actions_for_epoch_with_store(
        &mut self,
        epoch: usize,
        store: &burn_train::metric::store::EventStoreClient,
    ) -> Vec<burn_train::checkpoint::CheckpointingAction> {
        if !self.refresh_best_from_store(store) {
            self.refresh_best_from_history(epoch);
        }
        self.checkpoint_actions_after_refresh(epoch)
    }
}

impl burn_train::checkpoint::CheckpointingStrategy for FileMetricBestCheckpointingStrategy {
    fn checkpointing(
        &mut self,
        epoch: usize,
        store: &burn_train::metric::store::EventStoreClient,
    ) -> Vec<burn_train::checkpoint::CheckpointingAction> {
        self.actions_for_epoch_with_store(epoch, store)
    }
}

pub struct TrainEnvironment<'a, B>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    pub parallel_runtime: &'a ParallelRuntime,
    pub parallel_config: &'a ParallelConfig,
    pub run_dir: &'a Path,
    pub run_name: &'a str,
    pub backend_name: &'a str,
    pub training: &'a TrainingHyperparameters,
    pub resume_checkpoint_epoch: Option<usize>,
    pub model_config: &'a DragonConfig,
    pub device: &'a B::Device,
    pub devices: &'a [B::Device],
    pub train_dataset: Option<Arc<Dataset>>,
    pub valid_dataset: Option<Arc<Dataset>>,
    pub train_loader: Arc<dyn DataLoader<B, SequenceBatch<B>>>,
    pub valid_loader: Arc<dyn DataLoader<ValidBackend<B>, SequenceBatch<ValidBackend<B>>>>,
    pub source_selection_dataset: Option<Arc<Dataset>>,
    pub summary_event_token_ids: Option<Vec<u32>>,
    pub neuron_scaling_slot: Option<crate::train::neuron_scaling::NeuronScaleRequestSlot>,
    pub epochs: usize,
    pub total_steps: usize,
    pub valid_steps: usize,
}

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
    let event_handles = crate::train::events::build_training_event_handles(
        env.run_name,
        env.run_dir,
        env.train_loader.num_items(),
        env.training,
        source_selection_dataset,
        env.neuron_scaling_slot
            .as_ref()
            .map(|slot| (env.model_config.latent_total(), slot.clone())),
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

fn build_dynamic_train_loader<B>(
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
    if env.training.tbptt_persist_across_steps {
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
            .with_batch_size(batch_size)
            .with_initial_consumed_steps(consumed_steps)
            .with_summary_event_token_ids(env.summary_event_token_ids.clone()),
        )
    }
}

fn build_dynamic_valid_loader<B>(
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
    S: LrScheduler + 'static,
{
    fs::create_dir_all(env.run_dir)?;
    let source_selection_dataset = env.source_selection_dataset.as_ref().cloned();
    let event_handles = crate::train::events::build_training_event_handles(
        env.run_name,
        env.run_dir,
        env.train_loader.num_items(),
        env.training,
        source_selection_dataset,
        env.neuron_scaling_slot
            .as_ref()
            .map(|slot| (env.model_config.latent_total(), slot.clone())),
    )?;
    let bus = event_handles.metric_logger.bus();
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
    let mut last_cbp_telemetry_step = 0usize;
    let mut best_valid_loss: Option<f64> = None;
    let mut best_valid_epoch: Option<usize> = None;

    if let Some(epoch) = env.resume_checkpoint_epoch {
        model.model =
            load_dragon_model_checkpoint(env.run_dir, epoch, env.model_config, env.device)?;
    }

    let dynamic_neuron_scaling = env.neuron_scaling_slot.is_some();
    info!("run name: {}", env.run_name);
    info!(
        "training strategy: mode={:?} replicas={} event_scheduler=true dynamic_neuron_scaling={} start_epoch={}",
        env.parallel_runtime.mode,
        env.devices.len(),
        dynamic_neuron_scaling,
        start_epoch
    );
    info!(
        "checkpoint policy: logical_epoch_steps={} keep_last={} keep_best_valid_loss=true event_scheduler=true dynamic_neuron_scaling={}",
        env.train_loader.num_items(),
        CHECKPOINT_KEEP_LAST,
        dynamic_neuron_scaling
    );

    for epoch in start_epoch..=env.epochs {
        if event_handles.interrupter.should_stop() {
            let reason = event_handles
                .interrupter
                .get_message()
                .unwrap_or_else(|| "training interrupted".to_string());
            info!("Training interrupted: {reason}");
            break;
        }

        let mut iterator = active_train_loader.iter();
        let mut iteration = 0usize;
        let mut accumulator = GradientsAccumulator::new();
        let mut accumulation_current = 0usize;
        let mut last_lr = 0.0;

        while let Some(item) = iterator.next() {
            iteration += 1;
            let absolute_step = epoch
                .saturating_sub(1)
                .saturating_mul(steps_per_epoch)
                .saturating_add(iteration.saturating_sub(1));
            let _ = bus.send_step_started(StepStarted {
                run_id: env.run_name.to_string(),
                absolute_step,
                epoch,
            });

            let item = burn_train::TrainStep::step(&model, item);
            let train_output = item.item.sync();
            let loss_value: LossValue<ValidBackend<B>> = train_output.adapt();
            let mean_train_loss = mean_scalar_from_loss(loss_value.value());
            accumulator.accumulate(&model, item.grads);
            accumulation_current += 1;

            if active_grad_accumulation <= accumulation_current {
                let lr = scheduler.step();
                let grads = accumulator.grads();
                model = optimizer.step(lr, model, grads);
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

            let _ = bus.send_metric_sample(TrainingMetricSample {
                run_id: env.run_name.to_string(),
                split: TrainingMetricSplit::Train,
                epoch,
                step_in_epoch: iteration,
                absolute_step,
                name: "Loss".to_string(),
                value: mean_train_loss,
                running_value: mean_train_loss,
            });
            let _ = bus.send_metric_sample(TrainingMetricSample {
                run_id: env.run_name.to_string(),
                split: TrainingMetricSplit::Train,
                epoch,
                step_in_epoch: iteration,
                absolute_step,
                name: "Learning Rate".to_string(),
                value: last_lr,
                running_value: last_lr,
            });
            let _ = bus.send_step_finished(StepFinished {
                run_id: env.run_name.to_string(),
                absolute_step,
                epoch,
                loss: Some(mean_train_loss),
            });

            if iteration % env.training.log_frequency.max(1) == 0 || iteration == steps_per_epoch {
                let progress = iterator.progress();
                info!(
                    "train epoch={} step={}/{} loss={:.4} lr={:.6} global_progress={}/{}",
                    epoch,
                    progress.items_processed,
                    progress.items_total,
                    mean_train_loss,
                    last_lr,
                    epoch,
                    env.epochs
                );
            }
        }

        if accumulation_current > 0 {
            let lr = scheduler.step();
            let grads = accumulator.grads();
            model = optimizer.step(lr, model, grads);
            let absolute_step = epoch.saturating_mul(steps_per_epoch).saturating_sub(1);
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
        let _ = bus.send_epoch_summary(TrainingEpochSummary {
            run_id: env.run_name.to_string(),
            split: TrainingMetricSplit::Train,
            epoch,
        });

        let validation = run_dynamic_validation(
            env,
            &active_valid_loader,
            &model,
            epoch,
            steps_per_epoch,
            active_batch_size,
            &bus,
        )?;
        let valid_loss = validation.loss;
        info!("valid epoch={} loss={valid_loss:.4}", epoch);
        if let Some(source_weighted_loss) = validation.source_weighted_loss {
            info!(
                "valid epoch={} source_weighted_loss={source_weighted_loss:.4}",
                epoch
            );
        }
        let checkpoint_promoted = best_valid_loss.is_none_or(|best| valid_loss < best);
        if checkpoint_promoted {
            best_valid_loss = Some(valid_loss);
            best_valid_epoch = Some(epoch);
        }
        save_dragon_model_checkpoint(env.run_dir, epoch, &model.model)?;
        prune_dragon_model_checkpoints(env.run_dir, epoch, best_valid_epoch)?;
        let absolute_step = epoch.saturating_mul(steps_per_epoch).saturating_sub(1);
        apply_continual_learning_stability_policy(
            env,
            validation,
            epoch,
            absolute_step,
            &mut stability,
            &bus,
        );
        let _ = bus.send_checkpoint(CheckpointEvent {
            run_id: env.run_name.to_string(),
            checkpoint_id: format!("model-{epoch}"),
            epoch: Some(epoch),
            absolute_step: Some(absolute_step),
            promoted: checkpoint_promoted,
        });
        let _ = bus.flush();

        if let Some(request) = env
            .neuron_scaling_slot
            .as_ref()
            .and_then(|slot| slot.take())
        {
            if let Some((old_capacity_units, new_capacity_units)) = apply_dynamic_neuron_scale(
                env,
                &mut model,
                &mut optimizer,
                &mut current_model_config,
                &mut scale_generation,
                request,
                epoch,
                absolute_step,
                &bus,
                active_batch_size,
                active_grad_accumulation,
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
            let pause_steps = env
                .training
                .neuron_scaling
                .stabilization
                .freeze_base_steps
                .saturating_add(
                    env.training
                        .neuron_scaling
                        .stabilization
                        .unfreeze_ramp_steps,
                )
                .saturating_add(env.training.continual_backprop.cooldown_steps);
            optimizer.pause_continual_backprop_for_steps(pause_steps, "neuron_scale_stabilization");
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

fn run_dynamic_validation<B>(
    env: &TrainEnvironment<'_, B>,
    valid_loader: &Arc<dyn DataLoader<ValidBackend<B>, SequenceBatch<ValidBackend<B>>>>,
    model: &LanguageTrainModel<B>,
    epoch: usize,
    steps_per_epoch: usize,
    batch_size: usize,
    bus: &TrainingEventBus,
) -> Result<DynamicValidationReport>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let valid_model = model.valid();
    let mut iterator = valid_loader.iter();
    let mut total = 0.0;
    let mut count = 0usize;
    let mut output_degeneracy = None;
    let probe_enabled = epoch.is_multiple_of(env.training.events.degeneracy_probe_every_epochs);
    while let Some(item) = iterator.next() {
        let (loss_tensor, degeneracy) = if probe_enabled && output_degeneracy.is_none() {
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
        total += loss;
        if let Some(degeneracy) = degeneracy {
            let absolute_step = epoch
                .saturating_sub(1)
                .saturating_mul(steps_per_epoch)
                .saturating_add(count.saturating_sub(1));
            emit_output_degeneracy(env, epoch, absolute_step, &degeneracy, bus);
            output_degeneracy = Some(degeneracy);
        }
        let _ = bus.send_metric_sample(TrainingMetricSample {
            run_id: env.run_name.to_string(),
            split: TrainingMetricSplit::Valid,
            epoch,
            step_in_epoch: count,
            absolute_step: epoch
                .saturating_sub(1)
                .saturating_mul(steps_per_epoch)
                .saturating_add(count.saturating_sub(1)),
            name: "Loss".to_string(),
            value: loss,
            running_value: total / count as f64,
        });
    }
    let mean = if count == 0 {
        0.0
    } else {
        total / count as f64
    };
    let source_weighted_loss =
        run_source_weighted_validation(env, &valid_model, epoch, steps_per_epoch, batch_size, bus)?;
    if let Some(source_weighted_loss) = source_weighted_loss {
        let delta = source_weighted_loss - mean;
        let ratio = if mean.abs() <= f64::EPSILON {
            0.0
        } else {
            source_weighted_loss / mean
        };
        let absolute_step = epoch
            .saturating_sub(1)
            .saturating_mul(steps_per_epoch)
            .saturating_add(count);
        for (name, value) in [
            ("Source Weighted Loss Delta", delta),
            ("Source Weighted Loss Ratio", ratio),
        ] {
            let _ = bus.send_metric_sample(TrainingMetricSample {
                run_id: env.run_name.to_string(),
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
        run_id: env.run_name.to_string(),
        split: TrainingMetricSplit::Valid,
        epoch,
    });
    let _ = bus.send_validation_finished(ValidationFinished {
        run_id: env.run_name.to_string(),
        epoch,
        loss: Some(mean),
    });
    Ok(DynamicValidationReport {
        loss: mean,
        source_weighted_loss,
        output_degeneracy,
    })
}

fn run_source_weighted_validation<B>(
    env: &TrainEnvironment<'_, B>,
    valid_model: &LanguageTrainModel<ValidBackend<B>>,
    epoch: usize,
    steps_per_epoch: usize,
    batch_size: usize,
    bus: &TrainingEventBus,
) -> Result<Option<f64>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let requested_batches = env.training.events.source_weighted_validation_batches;
    if requested_batches == 0 {
        return Ok(None);
    }
    let Some(dataset) = env.source_selection_dataset.as_ref() else {
        return Ok(None);
    };
    if !dataset.uses_live_source_selection() {
        return Ok(None);
    }

    let base_absolute_step = epoch.saturating_sub(1).saturating_mul(steps_per_epoch);
    let mut total = 0.0;
    let mut count = 0usize;
    for batch_index in 0..requested_batches {
        let absolute_step = base_absolute_step.saturating_add(batch_index);
        let Some(batch) = dataset.sample_source_weighted_validation_batch::<ValidBackend<B>>(
            epoch,
            absolute_step,
            batch_size,
            env.summary_event_token_ids.as_deref(),
            env.device,
        ) else {
            break;
        };
        let output = valid_model.step(batch);
        let loss_value: LossValue<ValidBackend<B>> = output.adapt();
        let loss = mean_scalar_from_loss(loss_value.value());
        count += 1;
        total += loss;
        let _ = bus.send_metric_sample(TrainingMetricSample {
            run_id: env.run_name.to_string(),
            split: TrainingMetricSplit::Valid,
            epoch,
            step_in_epoch: count,
            absolute_step,
            name: "Source Weighted Loss".to_string(),
            value: loss,
            running_value: total / count as f64,
        });
    }

    Ok((count > 0).then_some(total / count as f64))
}

fn dataset_eos_id(dataset: Option<&Arc<Dataset>>) -> Option<i64> {
    dataset
        .and_then(|dataset| dataset.tokenizer().eos_id())
        .map(i64::from)
}

fn emit_output_degeneracy<B>(
    env: &TrainEnvironment<'_, B>,
    epoch: usize,
    absolute_step: usize,
    stats: &crate::train::steps::OutputDegeneracyStats,
    bus: &TrainingEventBus,
) where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let _ = bus.send_output_degeneracy_sample(OutputDegeneracySample {
        run_id: env.run_name.to_string(),
        split: TrainingMetricSplit::Valid,
        epoch,
        absolute_step,
        token_count: stats.token_count,
        entropy_bits: stats.entropy_bits,
        mean_max_probability: stats.mean_max_probability,
        argmax_unique_fraction: stats.argmax_unique_fraction,
        eos_fraction: stats.eos_fraction,
        repetition_fraction: stats.repetition_fraction,
    });
    for (name, value) in [
        ("Output Entropy Bits", stats.entropy_bits),
        ("Output Mean Max Probability", stats.mean_max_probability),
        (
            "Output Argmax Unique Fraction",
            stats.argmax_unique_fraction,
        ),
        ("Output EOS Fraction", stats.eos_fraction),
        ("Output Repetition Fraction", stats.repetition_fraction),
    ] {
        let _ = bus.send_metric_sample(TrainingMetricSample {
            run_id: env.run_name.to_string(),
            split: TrainingMetricSplit::Valid,
            epoch,
            step_in_epoch: 0,
            absolute_step,
            name: name.to_string(),
            value,
            running_value: value,
        });
    }
}

fn emit_continual_backprop_telemetry<B>(
    env: &TrainEnvironment<'_, B>,
    optimizer: &crate::train::continual_backprop::LanguageOptimizer<B>,
    epoch: usize,
    absolute_step: usize,
    bus: &TrainingEventBus,
    last_emitted_optimizer_step: &mut usize,
) where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let Some(telemetry) = optimizer.continual_backprop_telemetry() else {
        return;
    };
    if telemetry.optimizer_step == 0 || telemetry.optimizer_step == *last_emitted_optimizer_step {
        return;
    }
    if telemetry.replacement_count == 0
        && absolute_step % env.training.events.continual_backprop_every_steps.max(1) != 0
    {
        return;
    }
    *last_emitted_optimizer_step = telemetry.optimizer_step;
    let _ = bus.send_continual_backprop_sample(ContinualBackpropSample {
        run_id: env.run_name.to_string(),
        epoch: Some(epoch),
        absolute_step,
        optimizer_step: telemetry.optimizer_step,
        feature_count: telemetry.feature_count,
        eligible_count: telemetry.eligible_count,
        replacement_count: telemetry.replacement_count,
        replacement_budget: telemetry.replacement_budget as f64,
        lr_multiplier: telemetry.lr_multiplier as f64,
        paused: telemetry.paused,
        pause_reason: telemetry.pause_reason,
        utility_min: telemetry.utility_min as f64,
        utility_mean: telemetry.utility_mean as f64,
        utility_max: telemetry.utility_max as f64,
        age_mean: telemetry.age_mean as f64,
        age_max: telemetry.age_max as f64,
    });
}

fn apply_continual_learning_stability_policy<B>(
    env: &TrainEnvironment<'_, B>,
    validation: DynamicValidationReport,
    epoch: usize,
    absolute_step: usize,
    state: &mut ContinualLearningStabilityState,
    bus: &TrainingEventBus,
) where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let valid_loss = validation.loss;
    let improved = state.best_valid_loss.is_none_or(|best| {
        valid_loss < best * (1.0 - env.training.gates.plateau_min_relative_improvement)
    });
    if improved {
        state.best_valid_loss = Some(valid_loss);
        state.consecutive_validation_regressions = 0;
    } else if let Some(best) = state.best_valid_loss {
        if valid_loss > best * (1.0 + env.training.gates.validation_regression_max_relative) {
            state.consecutive_validation_regressions =
                state.consecutive_validation_regressions.saturating_add(1);
        } else {
            state.consecutive_validation_regressions = 0;
        }
        if state.consecutive_validation_regressions
            >= env.training.gates.validation_regression_patience_epochs
        {
            emit_policy_gate(
                env,
                bus,
                "continual_learning_validation_regression",
                epoch,
                absolute_step,
                format!(
                    "validation regression detected while leaving continual backprop active: best {:.6}, current {:.6}",
                    best, valid_loss
                ),
            );
        }
    }

    let Some(degeneracy) = validation.output_degeneracy else {
        return;
    };
    let degenerate = degeneracy.entropy_bits < env.training.gates.degeneracy_entropy_min_bits
        || degeneracy.mean_max_probability > env.training.gates.degeneracy_max_probability_max
        || degeneracy.argmax_unique_fraction
            < env.training.gates.degeneracy_argmax_unique_min_fraction
        || degeneracy.repetition_fraction > env.training.gates.degeneracy_repetition_max_fraction;
    if degenerate {
        state.consecutive_output_degeneracy = state.consecutive_output_degeneracy.saturating_add(1);
    } else {
        state.consecutive_output_degeneracy = 0;
    }
    if state.consecutive_output_degeneracy >= env.training.gates.degeneracy_patience {
        emit_policy_gate(
            env,
            bus,
            "continual_learning_output_degeneracy",
            epoch,
            absolute_step,
            format!(
                "output degeneracy detected while leaving continual backprop active: entropy {:.3}, max_prob {:.3}, unique {:.3}, repetition {:.3}",
                degeneracy.entropy_bits,
                degeneracy.mean_max_probability,
                degeneracy.argmax_unique_fraction,
                degeneracy.repetition_fraction
            ),
        );
    }
}

fn emit_policy_gate<B>(
    env: &TrainEnvironment<'_, B>,
    bus: &TrainingEventBus,
    gate: &str,
    epoch: usize,
    absolute_step: usize,
    message: String,
) where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let _ = bus.send_gate_event(TrainingGateEvent {
        run_id: env.run_name.to_string(),
        gate: gate.to_string(),
        action: TrainingGateAction::Alert,
        severity: TrainingGateSeverity::Warning,
        epoch: Some(epoch),
        absolute_step: Some(absolute_step),
        message,
    });
}

fn mean_scalar_from_loss<B: BackendTrait>(tensor: Tensor<B, 1>) -> f64 {
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

fn save_dragon_model_checkpoint<B: AutodiffBackend + Clone + 'static>(
    run_dir: &Path,
    epoch: usize,
    model: &DragonModel<B>,
) -> Result<()> {
    let checkpoint_dir = run_dir.join("checkpoint");
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    FileCheckpointer::new(recorder, &checkpoint_dir, "model")
        .save(epoch, model.clone().into_record())
        .with_context(|| {
            format!(
                "failed to save dynamic Dragon model checkpoint {epoch} in {}",
                checkpoint_dir.display()
            )
        })?;
    Ok(())
}

fn prune_dragon_model_checkpoints(
    run_dir: &Path,
    current_epoch: usize,
    best_epoch: Option<usize>,
) -> Result<()> {
    let checkpoint_dir = run_dir.join("checkpoint");
    let Ok(entries) = fs::read_dir(&checkpoint_dir) else {
        return Ok(());
    };
    let mut keep_epochs = BTreeSet::new();
    keep_epochs.extend(
        current_epoch
            .saturating_sub(CHECKPOINT_KEEP_LAST - 1)
            .max(1)..=current_epoch,
    );
    if let Some(best_epoch) = best_epoch {
        keep_epochs.insert(best_epoch);
    }

    for entry in entries {
        let path = entry?.path();
        let Some(epoch) = path
            .file_name()
            .and_then(|name| name.to_str())
            .and_then(|name| name.strip_prefix("model-"))
            .and_then(|name| name.strip_suffix(".bin"))
            .and_then(|epoch| epoch.parse::<usize>().ok())
        else {
            continue;
        };
        if !keep_epochs.contains(&epoch) {
            fs::remove_file(&path)
                .with_context(|| format!("failed to prune checkpoint {}", path.display()))?;
        }
    }
    Ok(())
}

fn load_dragon_model_checkpoint<B: AutodiffBackend + Clone + 'static>(
    run_dir: &Path,
    epoch: usize,
    model_config: &DragonConfig,
    device: &B::Device,
) -> Result<DragonModel<B>> {
    let checkpoint_dir = run_dir.join("checkpoint");
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let record = FileCheckpointer::new(recorder, &checkpoint_dir, "model")
        .restore(epoch, device)
        .with_context(|| {
            format!(
                "failed to restore dynamic Dragon model checkpoint {epoch} from {}",
                checkpoint_dir.display()
            )
        })?;
    Ok(DragonModel::<B>::new(model_config.clone(), device).load_record(record))
}

fn apply_dynamic_neuron_scale<B>(
    env: &TrainEnvironment<'_, B>,
    model: &mut LanguageTrainModel<B>,
    optimizer: &mut crate::train::continual_backprop::LanguageOptimizer<B>,
    current_model_config: &mut DragonConfig,
    scale_generation: &mut usize,
    request: ModelScaleRequest,
    epoch: usize,
    absolute_step: usize,
    bus: &TrainingEventBus,
    batch_size: usize,
    gradient_accumulation_steps: usize,
) -> Result<Option<(usize, usize)>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let current_latent_total = model.model.latent_total_capacity();
    let skip = |reason: String, bus: &TrainingEventBus| {
        let _ = bus.send_model_scale_skipped(ModelScaleSkipped {
            run_id: env.run_name.to_string(),
            epoch: Some(epoch),
            absolute_step: Some(absolute_step),
            from_capacity_units: current_latent_total,
            requested_capacity_units: Some(request.to_capacity_units),
            reason,
        });
    };

    if request.run_id != env.run_name {
        skip(
            format!(
                "scale request run_id {} does not match active run {}",
                request.run_id, env.run_name
            ),
            bus,
        );
        return Ok(None);
    }
    if request.from_capacity_units != current_latent_total {
        skip(
            format!(
                "scale request source capacity {} does not match active latent_total {}",
                request.from_capacity_units, current_latent_total
            ),
            bus,
        );
        return Ok(None);
    }
    if request.to_capacity_units <= current_latent_total {
        skip(
            format!(
                "scale request target {} must exceed current latent_total {}",
                request.to_capacity_units, current_latent_total
            ),
            bus,
        );
        return Ok(None);
    }
    if request.to_capacity_units > env.training.neuron_scaling.max_latent_total {
        skip(
            format!(
                "scale request target {} exceeds configured max_latent_total {}",
                request.to_capacity_units, env.training.neuron_scaling.max_latent_total
            ),
            bus,
        );
        return Ok(None);
    }
    if request.to_capacity_units % current_model_config.n_embd != 0 {
        skip(
            format!(
                "scale request target {} is not divisible by n_embd {}",
                request.to_capacity_units, current_model_config.n_embd
            ),
            bus,
        );
        return Ok(None);
    }
    if request.to_capacity_units % current_model_config.n_head != 0 {
        skip(
            format!(
                "scale request target {} is not divisible by n_head {}",
                request.to_capacity_units, current_model_config.n_head
            ),
            bus,
        );
        return Ok(None);
    }
    if request.to_capacity_units % env.parallel_config.tensor.size != 0 {
        skip(
            format!(
                "scale request target {} is not divisible by tensor parallel size {}",
                request.to_capacity_units, env.parallel_config.tensor.size
            ),
            bus,
        );
        return Ok(None);
    }

    let mut target_config = current_model_config.clone();
    target_config.mlp_internal_dim_multiplier = request.to_capacity_units / target_config.n_embd;
    let (widened, report) = model
        .model
        .widen_latent_total(target_config.clone(), env.device)
        .map_err(|message| anyhow!("failed to widen Dragon latent_total in-process: {message}"))?;
    model.model = widened;
    *model = model
        .clone()
        .with_continual_backprop(&env.training.continual_backprop)
        .with_gradient_scale_schedule(
            env.training,
            env.epochs
                .saturating_mul(env.train_loader.num_items().max(1)),
        )
        .with_neuron_scale_stabilization(
            report.old_latent_total,
            report.new_latent_total,
            &env.training.neuron_scaling.stabilization,
        );
    optimizer.prepare_after_neuron_scale(model);
    *current_model_config = target_config;
    *scale_generation = scale_generation.saturating_add(1);

    let _ = bus.send_model_scale_applied(ModelScaleApplied {
        run_id: env.run_name.to_string(),
        epoch: Some(epoch),
        absolute_step: Some(absolute_step),
        from_capacity_units: report.old_latent_total,
        to_capacity_units: report.new_latent_total,
        scale_generation: *scale_generation,
        batch_size: Some(batch_size),
        gradient_accumulation_steps: Some(gradient_accumulation_steps),
        message: format!(
            "applied Dragon neuron scaling {} -> {} in-process; optimizer_state_policy=drop_widened_param_moments; stabilization_freeze_base_steps={} stabilization_unfreeze_ramp_steps={}",
            report.old_latent_total,
            report.new_latent_total,
            env.training.neuron_scaling.stabilization.freeze_base_steps,
            env.training.neuron_scaling.stabilization.unfreeze_ramp_steps,
        ),
    });
    info!(
        "applied in-process Dragon neuron scaling {} -> {} at epoch={} absolute_step={}",
        report.old_latent_total, report.new_latent_total, epoch, absolute_step
    );
    Ok(Some((report.old_latent_total, report.new_latent_total)))
}

#[cfg(feature = "ddp")]
struct CollectiveSessionGuard<B: BackendTrait> {
    peer_id: PeerId,
    _marker: PhantomData<B>,
}

#[cfg(feature = "ddp")]
impl<B: BackendTrait> CollectiveSessionGuard<B> {
    fn register(
        peer_id: PeerId,
        device: B::Device,
        config: burn_collective::CollectiveConfig,
    ) -> Result<Self> {
        info!("registering DDP collective session for peer_id={peer_id}");
        register::<B>(peer_id, device, config)
            .map_err(|err| anyhow!("failed to register DDP collective session: {err:?}"))?;
        info!("registered DDP collective session for peer_id={peer_id}");
        Ok(Self {
            peer_id,
            _marker: PhantomData,
        })
    }
}

#[cfg(feature = "ddp")]
impl<B: BackendTrait> Drop for CollectiveSessionGuard<B> {
    fn drop(&mut self) {
        let _ = finish_collective::<B>(self.peer_id);
    }
}

#[cfg(feature = "ddp")]
fn shard_bounds(
    total_items: usize,
    shard_index: usize,
    shard_count: usize,
) -> Result<(usize, usize)> {
    if shard_count == 0 {
        return Err(anyhow!("cannot shard a dataloader across zero ranks"));
    }
    if shard_index >= shard_count {
        return Err(anyhow!(
            "rank-local dataloader shard {shard_index} is out of range for shard_count={shard_count}"
        ));
    }
    if total_items < shard_count {
        return Err(anyhow!(
            "rank-local dataloader sharding requires at least one step per rank (steps={total_items}, world_size={shard_count})"
        ));
    }

    let base = total_items / shard_count;
    let remainder = total_items % shard_count;
    let start = shard_index * base + shard_index.min(remainder);
    let width = base + usize::from(shard_index < remainder);
    Ok((start, start + width))
}

#[cfg(feature = "ddp")]
fn shard_dataloader<B, I>(
    loader: Arc<dyn DataLoader<B, I>>,
    shard_index: usize,
    shard_count: usize,
    label: &str,
) -> Result<Arc<dyn DataLoader<B, I>>>
where
    B: BackendTrait + 'static,
    I: 'static,
{
    if shard_count <= 1 {
        return Ok(loader);
    }

    let total_items = loader.num_items();
    let (start, end) = shard_bounds(total_items, shard_index, shard_count)
        .with_context(|| format!("failed to shard {label} dataloader"))?;
    Ok(loader.slice(start, end))
}

#[cfg(feature = "ddp")]
fn mean_scalar_from_tensor<B: BackendTrait>(tensor: Tensor<B, 1>) -> f64 {
    tensor
        .mean()
        .into_data()
        .iter::<f64>()
        .next()
        .unwrap_or(0.0)
}

#[cfg(feature = "ddp")]
fn reduce_mean_scalar<B: BackendTrait>(peer_id: PeerId, tensor: Tensor<B, 1>) -> Result<f64> {
    let reduced = all_reduce::<B>(
        peer_id,
        tensor.into_primitive().tensor(),
        ReduceOperation::Mean,
    )
    .map_err(|err| anyhow!("failed to all-reduce scalar metric: {err:?}"))?;
    Ok(mean_scalar_from_tensor(Tensor::<B, 1>::from_primitive(
        TensorPrimitive::Float(reduced),
    )))
}

#[cfg(feature = "ddp")]
fn process_group_peer_id(runtime: &ParallelRuntime) -> PeerId {
    runtime.global_rank.into()
}

#[cfg(feature = "ddp")]
fn process_group_data_shard(
    runtime: &ParallelRuntime,
    config: &ParallelConfig,
) -> Result<(
    usize,
    usize,
    Option<PipelineRankAssignment>,
    Option<PipelineParallelLayout>,
)> {
    let layout = resolve_pipeline_parallel_layout(runtime, config)?;
    if let Some(layout) = layout {
        let assignment = layout.assignment(runtime.global_rank).clone();
        return Ok((
            assignment.data_parallel_rank,
            layout.data_parallel_size,
            Some(assignment),
            Some(layout),
        ));
    }

    Ok((runtime.global_rank, runtime.world_size, None, None))
}

#[cfg(feature = "ddp")]
fn all_reduce_gradients_in_module_order<B, M>(
    module: &M,
    grads: &mut GradientsParams,
    peer_id: PeerId,
    op: ReduceOperation,
) -> Result<()>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    struct GradientAllReduceVisitor<'a, B: AutodiffBackend> {
        grads: &'a mut GradientsParams,
        peer_id: PeerId,
        op: ReduceOperation,
        trace_grads: bool,
        index: usize,
        error: Option<anyhow::Error>,
        _marker: PhantomData<B>,
    }

    impl<B: AutodiffBackend> burn::module::ModuleVisitor<B> for GradientAllReduceVisitor<'_, B> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            if self.error.is_some() {
                return;
            }

            self.index += 1;
            let grad_index = self.index;

            let grad = match self.grads.remove::<B::InnerBackend, D>(param.id) {
                Some(grad) => grad,
                None => {
                    if self.trace_grads && grad_index <= 12 {
                        info!(
                            "process-group DDP peer_id={} gradient[{grad_index}] missing, zero-filling shape={:?}",
                            self.peer_id,
                            param.val().shape().dims::<D>()
                        );
                    }
                    param.val().inner().zeros_like()
                }
            };

            if self.trace_grads && grad_index <= 12 {
                info!(
                    "process-group DDP peer_id={} gradient[{grad_index}] entering all-reduce shape={:?}",
                    self.peer_id,
                    grad.shape().dims::<D>()
                );
            }

            match all_reduce::<B::InnerBackend>(
                self.peer_id,
                grad.into_primitive().tensor(),
                self.op,
            ) {
                Ok(reduced) => {
                    if self.trace_grads && grad_index <= 12 {
                        info!(
                            "process-group DDP peer_id={} gradient[{grad_index}] completed all-reduce",
                            self.peer_id
                        );
                    }
                    self.grads.register(
                        param.id,
                        Tensor::<B::InnerBackend, D>::from_primitive(TensorPrimitive::Float(
                            reduced,
                        )),
                    )
                }
                Err(err) => {
                    self.error = Some(anyhow!(
                        "failed to all-reduce process-group DDP gradients: {err:?}"
                    ));
                }
            }
        }
    }

    let trace_grads = true;
    let mut visitor = GradientAllReduceVisitor::<B> {
        grads,
        peer_id,
        op,
        trace_grads,
        index: 0,
        error: None,
        _marker: PhantomData,
    };
    module.visit(&mut visitor);

    if let Some(err) = visitor.error {
        return Err(err);
    }

    Ok(())
}

#[cfg(feature = "ddp")]
fn scale_gradients_in_module_order<B, M>(module: &M, grads: &mut GradientsParams, scale: f32)
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    if (scale - 1.0).abs() <= f32::EPSILON {
        return;
    }

    struct GradientScaleVisitor<'a, B: AutodiffBackend> {
        grads: &'a mut GradientsParams,
        scale: f32,
        _marker: PhantomData<B>,
    }

    impl<B: AutodiffBackend> burn::module::ModuleVisitor<B> for GradientScaleVisitor<'_, B> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            if let Some(grad) = self.grads.remove::<B::InnerBackend, D>(param.id) {
                self.grads.register(param.id, grad.mul_scalar(self.scale));
            }
        }
    }

    let mut visitor = GradientScaleVisitor::<B> {
        grads,
        scale,
        _marker: PhantomData,
    };
    module.visit(&mut visitor);
}

#[cfg(feature = "ddp")]
fn reduce_sum_scalar<B: BackendTrait>(peer_id: PeerId, tensor: Tensor<B, 1>) -> Result<f64> {
    let reduced = all_reduce::<B>(
        peer_id,
        tensor.into_primitive().tensor(),
        ReduceOperation::Sum,
    )
    .map_err(|err| anyhow!("failed to all-reduce scalar sum: {err:?}"))?;
    Ok(mean_scalar_from_tensor(Tensor::<B, 1>::from_primitive(
        TensorPrimitive::Float(reduced),
    )))
}

#[cfg(feature = "ddp")]
fn scalar_tensor<B: BackendTrait>(device: &B::Device, value: f32) -> Tensor<B, 1> {
    Tensor::<B, 1>::from_floats([value], device)
}

#[cfg(feature = "ddp")]
fn scalar_flag<B: BackendTrait>(device: &B::Device, enabled: bool) -> Tensor<B, 1> {
    scalar_tensor::<B>(device, if enabled { 1.0 } else { 0.0 })
}

#[cfg(feature = "ddp")]
fn broadcast_float_tensor_rooted<B: BackendTrait, const D: usize>(
    peer_id: PeerId,
    global_rank: usize,
    root_rank: usize,
    tensor: Option<Tensor<B, D>>,
) -> Result<Tensor<B, D>> {
    let root_tensor = if global_rank == root_rank {
        Some(
            tensor
                .ok_or_else(|| anyhow!("collective root rank {root_rank} is missing a tensor"))?
                .into_primitive()
                .tensor(),
        )
    } else {
        None
    };
    let broadcasted = broadcast::<B>(peer_id, root_tensor).map_err(|err| {
        anyhow!("failed to broadcast rooted tensor from rank {root_rank}: {err:?}")
    })?;
    Ok(Tensor::<B, D>::from_primitive(TensorPrimitive::Float(
        broadcasted,
    )))
}

#[cfg(feature = "ddp")]
fn broadcast_usize_rooted<B: BackendTrait>(
    peer_id: PeerId,
    global_rank: usize,
    root_rank: usize,
    device: &B::Device,
    value: Option<usize>,
) -> Result<usize> {
    let tensor = broadcast_float_tensor_rooted::<B, 1>(
        peer_id,
        global_rank,
        root_rank,
        value.map(|value| scalar_tensor::<B>(device, value as f32)),
    )?;
    Ok(mean_scalar_from_tensor(tensor).round().max(0.0) as usize)
}

#[cfg(feature = "ddp")]
fn broadcast_bool_rooted<B: BackendTrait>(
    peer_id: PeerId,
    global_rank: usize,
    root_rank: usize,
    device: &B::Device,
    value: Option<bool>,
) -> Result<bool> {
    let tensor = broadcast_float_tensor_rooted::<B, 1>(
        peer_id,
        global_rank,
        root_rank,
        value.map(|value| scalar_flag::<B>(device, value)),
    )?;
    Ok(mean_scalar_from_tensor(tensor) >= 0.5)
}

#[cfg(feature = "ddp")]
fn broadcast_int_tensor_rooted<B: AutodiffBackend, const D: usize>(
    peer_id: PeerId,
    global_rank: usize,
    root_rank: usize,
    tensor: Option<Tensor<B, D, Int>>,
) -> Result<Tensor<B, D, Int>> {
    let broadcasted = broadcast_float_tensor_rooted::<B::InnerBackend, D>(
        peer_id,
        global_rank,
        root_rank,
        tensor.map(|tensor| tensor.float().inner()),
    )?;
    Ok(Tensor::<B, D>::from_inner(broadcasted).int())
}

#[cfg(feature = "ddp")]
fn broadcast_optional_int_tensor_rooted<B: AutodiffBackend, const D: usize>(
    peer_id: PeerId,
    global_rank: usize,
    root_rank: usize,
    device: &B::Device,
    tensor: Option<Tensor<B, D, Int>>,
) -> Result<Option<Tensor<B, D, Int>>> {
    let has_tensor = broadcast_bool_rooted::<B::InnerBackend>(
        peer_id,
        global_rank,
        root_rank,
        device,
        Some(tensor.is_some()),
    )?;
    if !has_tensor {
        return Ok(None);
    }
    broadcast_int_tensor_rooted(peer_id, global_rank, root_rank, tensor).map(Some)
}

#[cfg(feature = "ddp")]
fn broadcast_sequence_batch_rooted<B: AutodiffBackend>(
    peer_id: PeerId,
    global_rank: usize,
    root_rank: usize,
    device: &B::Device,
    batch: Option<SequenceBatch<B>>,
) -> Result<SequenceBatch<B>> {
    let inputs = broadcast_int_tensor_rooted(
        peer_id,
        global_rank,
        root_rank,
        batch.as_ref().map(|batch| batch.inputs.clone()),
    )?;
    let targets = broadcast_int_tensor_rooted(
        peer_id,
        global_rank,
        root_rank,
        batch.as_ref().map(|batch| batch.targets.clone()),
    )?;
    let summary_event_mask = broadcast_optional_int_tensor_rooted(
        peer_id,
        global_rank,
        root_rank,
        device,
        batch
            .as_ref()
            .and_then(|batch| batch.summary_event_mask.clone()),
    )?;
    let reset_stream_state = broadcast_bool_rooted::<B::InnerBackend>(
        peer_id,
        global_rank,
        root_rank,
        device,
        Some(batch.as_ref().is_some_and(|batch| batch.reset_stream_state)),
    )?;

    Ok(SequenceBatch {
        inputs,
        targets,
        summary_event_mask,
        reset_stream_state,
    })
}

#[cfg(feature = "ddp")]
fn detach_pipeline_state_to_inner<B: AutodiffBackend>(
    state: &LanguagePipelineState<B>,
) -> LanguagePipelineState<B::InnerBackend> {
    LanguagePipelineState::from_parts(
        state.current().clone().detach().inner(),
        state
            .residual_history()
            .iter()
            .cloned()
            .map(|tensor| tensor.detach().inner())
            .collect(),
    )
}

#[cfg(feature = "ddp")]
fn attach_pipeline_state_require_grad<B: AutodiffBackend>(
    state: LanguagePipelineState<B::InnerBackend>,
) -> LanguagePipelineState<B> {
    let (current, residual_history) = state.into_parts();
    LanguagePipelineState::from_parts(
        Tensor::<B, 4>::from_inner(current).require_grad(),
        residual_history
            .into_iter()
            .map(|tensor| Tensor::<B, 4>::from_inner(tensor).require_grad())
            .collect(),
    )
}

#[cfg(feature = "ddp")]
fn broadcast_pipeline_state_rooted<B: AutodiffBackend>(
    peer_id: PeerId,
    global_rank: usize,
    root_rank: usize,
    device: &B::Device,
    state: Option<&LanguagePipelineState<B>>,
) -> Result<LanguagePipelineState<B::InnerBackend>> {
    let history_len = broadcast_usize_rooted::<B::InnerBackend>(
        peer_id,
        global_rank,
        root_rank,
        device,
        state.map(|state| state.residual_history().len()),
    )?;
    let current = broadcast_float_tensor_rooted::<B::InnerBackend, 4>(
        peer_id,
        global_rank,
        root_rank,
        state.map(|state| state.current().clone().detach().inner()),
    )?;
    let residual_history = (0..history_len)
        .map(|index| {
            broadcast_float_tensor_rooted::<B::InnerBackend, 4>(
                peer_id,
                global_rank,
                root_rank,
                state.map(|state| state.residual_history()[index].clone().detach().inner()),
            )
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(LanguagePipelineState::from_parts(current, residual_history))
}

#[cfg(feature = "ddp")]
fn broadcast_pipeline_state_inner_rooted<B: BackendTrait>(
    peer_id: PeerId,
    global_rank: usize,
    root_rank: usize,
    device: &B::Device,
    state: Option<&LanguagePipelineState<B>>,
) -> Result<LanguagePipelineState<B>> {
    let history_len = broadcast_usize_rooted::<B>(
        peer_id,
        global_rank,
        root_rank,
        device,
        state.map(|state| state.residual_history().len()),
    )?;
    let current = broadcast_float_tensor_rooted::<B, 4>(
        peer_id,
        global_rank,
        root_rank,
        state.map(|state| state.current().clone()),
    )?;
    let residual_history = (0..history_len)
        .map(|index| {
            broadcast_float_tensor_rooted::<B, 4>(
                peer_id,
                global_rank,
                root_rank,
                state.map(|state| state.residual_history()[index].clone()),
            )
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(LanguagePipelineState::from_parts(current, residual_history))
}

#[cfg(feature = "ddp")]
fn pipeline_surrogate_loss<B: AutodiffBackend>(
    output_state: &LanguagePipelineState<B>,
    grad_state: LanguagePipelineState<B::InnerBackend>,
) -> Tensor<B, 1> {
    let (grad_current, grad_history) = grad_state.into_parts();
    assert_eq!(
        output_state.residual_history().len(),
        grad_history.len(),
        "pipeline residual history length mismatch"
    );

    let mut surrogate = output_state
        .current()
        .clone()
        .mul(Tensor::<B, 4>::from_inner(grad_current))
        .sum();
    for (residual, grad) in output_state
        .residual_history()
        .iter()
        .zip(grad_history.into_iter())
    {
        surrogate = surrogate + residual.clone().mul(Tensor::<B, 4>::from_inner(grad)).sum();
    }
    surrogate
}

#[cfg(feature = "ddp")]
fn pipeline_input_grad_state<B: AutodiffBackend>(
    input_state: &LanguagePipelineState<B>,
    grads: &mut B::Gradients,
) -> LanguagePipelineState<B::InnerBackend> {
    LanguagePipelineState::from_parts(
        input_state
            .current()
            .grad_remove(grads)
            .unwrap_or_else(|| input_state.current().clone().inner().zeros_like()),
        input_state
            .residual_history()
            .iter()
            .map(|tensor| {
                tensor
                    .grad_remove(grads)
                    .unwrap_or_else(|| tensor.clone().inner().zeros_like())
            })
            .collect(),
    )
}

#[cfg(feature = "ddp")]
fn slice_batch_int<B: BackendTrait>(
    tensor: Tensor<B, 2, Int>,
    range: std::ops::Range<usize>,
) -> Tensor<B, 2, Int> {
    let [_batch, block_size] = tensor.shape().dims();
    tensor.slice([range.start..range.end, 0..block_size])
}

#[cfg(feature = "ddp")]
fn pipeline_replica_root_rank(layout: &PipelineParallelLayout, data_parallel_rank: usize) -> usize {
    data_parallel_rank * layout.stage_count
}

#[cfg(feature = "ddp")]
fn global_rank_for_virtual_stage(
    plan: &PipelinePlan,
    layout: &PipelineParallelLayout,
    data_parallel_rank: usize,
    virtual_stage_id: usize,
) -> usize {
    let physical_stage_id = plan.assignment(virtual_stage_id).physical_stage_id;
    data_parallel_rank * layout.stage_count + physical_stage_id
}

#[cfg(feature = "ddp")]
struct DistributedPipelineForwardCache<B: AutodiffBackend> {
    input_state: Option<LanguagePipelineState<B>>,
    output_state: Option<LanguagePipelineState<B>>,
    loss: Option<Tensor<B, 1>>,
}

#[cfg(feature = "ddp")]
fn save_process_group_checkpoint<B, O, S>(
    run_dir: &Path,
    epoch: usize,
    learner: &burn_train::Learner<
        burn_train::LearningComponentsMarker<B, S, LanguageTrainModel<B>, O>,
    >,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
    O: Optimizer<LanguageTrainModel<B>, B> + 'static,
    S: LrScheduler + 'static,
{
    let checkpoint_dir = run_dir.join("checkpoint");
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    FileCheckpointer::new(recorder, &checkpoint_dir, "model")
        .save(epoch, learner.model().model.into_record())
        .with_context(|| {
            format!(
                "failed to save process-group model checkpoint {epoch} in {}",
                checkpoint_dir.display()
            )
        })?;
    Ok(())
}

#[cfg(feature = "ddp")]
fn load_process_group_checkpoint<B, O, S>(
    run_dir: &Path,
    epoch: usize,
    device: &B::Device,
    mut learner: burn_train::Learner<
        burn_train::LearningComponentsMarker<B, S, LanguageTrainModel<B>, O>,
    >,
) -> Result<burn_train::Learner<burn_train::LearningComponentsMarker<B, S, LanguageTrainModel<B>, O>>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    O: Optimizer<LanguageTrainModel<B>, B> + 'static,
    S: LrScheduler + 'static,
{
    let checkpoint_dir = run_dir.join("checkpoint");
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let model_record = FileCheckpointer::new(recorder.clone(), &checkpoint_dir, "model")
        .restore(epoch, device)
        .with_context(|| {
            format!(
                "failed to restore process-group model checkpoint {epoch} from {}",
                checkpoint_dir.display()
            )
        })?;
    learner.load_model(model_record);

    let optim_path = checkpoint_dir.join(format!("optim-{epoch}.bin"));
    if optim_path.is_file() {
        let optim_record = FileCheckpointer::new(recorder.clone(), &checkpoint_dir, "optim")
            .restore(epoch, device)
            .with_context(|| {
                format!(
                    "failed to restore process-group optimizer checkpoint {epoch} from {}",
                    checkpoint_dir.display()
                )
            })?;
        learner.load_optim(optim_record);
    }

    let scheduler_path = checkpoint_dir.join(format!("scheduler-{epoch}.bin"));
    if scheduler_path.is_file() {
        let scheduler_record = FileCheckpointer::new(recorder, &checkpoint_dir, "scheduler")
            .restore(epoch, device)
            .with_context(|| {
                format!(
                    "failed to restore process-group scheduler checkpoint {epoch} from {}",
                    checkpoint_dir.display()
                )
            })?;
        learner.load_scheduler(scheduler_record);
    }

    Ok(learner)
}

#[cfg(feature = "ddp")]
fn run_process_group_validation<B, O, S>(
    env: &TrainEnvironment<'_, B>,
    learner: &burn_train::Learner<
        burn_train::LearningComponentsMarker<B, S, LanguageTrainModel<B>, O>,
    >,
) -> Option<f64>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    O: Optimizer<LanguageTrainModel<B>, B> + 'static,
    S: LrScheduler + 'static,
{
    if !env.parallel_runtime.is_primary() {
        return None;
    }

    let model = learner.model().valid();
    let mut iterator = env.valid_loader.iter();
    let mut total = 0.0;
    let mut count = 0usize;

    while let Some(item) = iterator.next() {
        let output = model.step(item);
        let loss_value: LossValue<ValidBackend<B>> = output.adapt();
        total += mean_scalar_from_tensor(loss_value.value());
        count += 1;
    }

    (count > 0).then_some(total / count as f64)
}

#[cfg(feature = "ddp")]
struct DistributedPipelineTrainStepResult {
    grads: GradientsParams,
    mean_train_loss: f64,
}

#[cfg(feature = "ddp")]
fn distributed_pipeline_train_step<B>(
    peer_id: PeerId,
    model: &LanguageTrainModel<B>,
    batch: SequenceBatch<B>,
    layout: &PipelineParallelLayout,
    assignment: &PipelineRankAssignment,
    device: &B::Device,
) -> Result<DistributedPipelineTrainStepResult>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let plan = model
        .pipeline_plan
        .as_ref()
        .ok_or_else(|| anyhow!("distributed pipeline step requires a pipeline plan"))?;
    let [batch_size, _block_size] = batch.inputs.shape().dims();
    let ranges = split_microbatch_ranges(batch_size, plan.microbatches)?;
    let chunk_inputs = ranges
        .iter()
        .cloned()
        .map(|range| slice_batch_int(batch.inputs.clone(), range))
        .collect::<Vec<_>>();
    let chunk_targets = ranges
        .iter()
        .cloned()
        .map(|range| slice_batch_int(batch.targets.clone(), range))
        .collect::<Vec<_>>();
    let chunk_masks = ranges
        .iter()
        .cloned()
        .map(|range| {
            batch
                .summary_event_mask
                .clone()
                .map(|mask| slice_batch_int(mask, range))
        })
        .collect::<Vec<_>>();
    let mut chunk_states = (0..plan.microbatches)
        .map(|_| model.model.init_state())
        .collect::<Vec<ModelState<B>>>();
    let mut forward_cache = HashMap::<(usize, usize), DistributedPipelineForwardCache<B>>::new();
    let mut incoming_forward =
        HashMap::<(usize, usize), LanguagePipelineState<B::InnerBackend>>::new();
    let mut incoming_backward =
        HashMap::<(usize, usize), LanguagePipelineState<B::InnerBackend>>::new();
    let mut local_accumulator = GradientsAccumulator::new();
    let mut local_loss: Option<Tensor<B::InnerBackend, 1>> = None;
    let last_virtual_stage_id = plan.total_virtual_stages.saturating_sub(1);

    for event in &plan.events {
        let microbatch_id = event.microbatch_id;
        let local_forward_output = if event.kind
            == burn_dragon_train::train::pipeline::PipelineEventKind::Forward
            && event.physical_stage_id == assignment.pipeline_stage_id
        {
            let input_state = if event.virtual_stage_id == 0 {
                model
                    .model
                    .begin_language_pipeline(chunk_inputs[microbatch_id].clone())
            } else {
                let input_state = incoming_forward
                    .remove(&(event.virtual_stage_id, microbatch_id))
                    .ok_or_else(|| {
                        anyhow!(
                            "missing forward pipeline state for virtual_stage={} microbatch={microbatch_id}",
                            event.virtual_stage_id
                        )
                    })?;
                attach_pipeline_state_require_grad::<B>(input_state)
            };
            let cached_input = (event.virtual_stage_id > 0).then_some(input_state.clone());
            let output_state = model.model.forward_language_pipeline_stage_with_state(
                input_state,
                &mut chunk_states[microbatch_id],
                plan.assignment(event.virtual_stage_id).layer_range.clone(),
                chunk_masks[microbatch_id].clone(),
            );

            if event.virtual_stage_id == last_virtual_stage_id {
                let hidden = model.model.finish_language_pipeline_hidden_with_state(
                    output_state,
                    &mut chunk_states[microbatch_id],
                );
                let weight = ranges[microbatch_id].len() as f32 / batch_size as f32;
                let loss = model
                    .model
                    .language_loss_from_hidden(hidden, chunk_targets[microbatch_id].clone())
                    .mul_scalar(weight);
                local_loss = Some(match local_loss {
                    Some(accumulated) => accumulated + loss.clone().detach().inner(),
                    None => loss.clone().detach().inner(),
                });
                forward_cache.insert(
                    (event.virtual_stage_id, microbatch_id),
                    DistributedPipelineForwardCache {
                        input_state: cached_input,
                        output_state: None,
                        loss: Some(loss),
                    },
                );
                None
            } else {
                forward_cache.insert(
                    (event.virtual_stage_id, microbatch_id),
                    DistributedPipelineForwardCache {
                        input_state: cached_input,
                        output_state: Some(output_state.clone()),
                        loss: None,
                    },
                );
                Some(output_state)
            }
        } else {
            None
        };

        if event.kind == burn_dragon_train::train::pipeline::PipelineEventKind::Forward
            && event.virtual_stage_id < last_virtual_stage_id
        {
            for replica_id in 0..layout.data_parallel_size {
                let sender_rank =
                    global_rank_for_virtual_stage(plan, layout, replica_id, event.virtual_stage_id);
                let receiver_rank = global_rank_for_virtual_stage(
                    plan,
                    layout,
                    replica_id,
                    event.virtual_stage_id + 1,
                );

                if sender_rank == receiver_rank {
                    if assignment.data_parallel_rank == replica_id
                        && assignment.global_rank == receiver_rank
                    {
                        let forwarded = detach_pipeline_state_to_inner(
                            local_forward_output.as_ref().ok_or_else(|| {
                                anyhow!(
                                    "missing local forward state for virtual_stage={} microbatch={microbatch_id}",
                                    event.virtual_stage_id
                                )
                            })?,
                        );
                        incoming_forward
                            .insert((event.virtual_stage_id + 1, microbatch_id), forwarded);
                    }
                    continue;
                }

                let broadcasted = broadcast_pipeline_state_rooted(
                    peer_id,
                    assignment.global_rank,
                    sender_rank,
                    device,
                    (assignment.data_parallel_rank == replica_id
                        && assignment.global_rank == sender_rank)
                        .then_some(local_forward_output.as_ref())
                        .flatten(),
                )?;
                if assignment.data_parallel_rank == replica_id
                    && assignment.global_rank == receiver_rank
                {
                    incoming_forward
                        .insert((event.virtual_stage_id + 1, microbatch_id), broadcasted);
                }
            }
        }

        let local_backward_grad = if event.kind
            == burn_dragon_train::train::pipeline::PipelineEventKind::Backward
            && event.physical_stage_id == assignment.pipeline_stage_id
        {
            let cached = forward_cache
                .remove(&(event.virtual_stage_id, microbatch_id))
                .ok_or_else(|| {
                    anyhow!(
                        "missing backward cache for virtual_stage={} microbatch={microbatch_id}",
                        event.virtual_stage_id
                    )
                })?;

            let mut grads = if event.virtual_stage_id == last_virtual_stage_id {
                cached
                    .loss
                    .ok_or_else(|| {
                        anyhow!(
                            "missing terminal loss for virtual_stage={} microbatch={microbatch_id}",
                            event.virtual_stage_id
                        )
                    })?
                    .backward()
            } else {
                let output_state = cached.output_state.as_ref().ok_or_else(|| {
                    anyhow!(
                        "missing stage output for virtual_stage={} microbatch={microbatch_id}",
                        event.virtual_stage_id
                    )
                })?;
                let grad_state = incoming_backward
                        .remove(&(event.virtual_stage_id, microbatch_id))
                        .ok_or_else(|| {
                            anyhow!(
                                "missing backward pipeline gradient for virtual_stage={} microbatch={microbatch_id}",
                                event.virtual_stage_id
                            )
                        })?;
                pipeline_surrogate_loss(output_state, grad_state).backward()
            };

            let input_grad = cached
                .input_state
                .as_ref()
                .map(|input_state| pipeline_input_grad_state(input_state, &mut grads));
            local_accumulator.accumulate(model, GradientsParams::from_grads(grads, model));
            input_grad
        } else {
            None
        };

        if event.kind == burn_dragon_train::train::pipeline::PipelineEventKind::Backward
            && event.virtual_stage_id > 0
        {
            for replica_id in 0..layout.data_parallel_size {
                let sender_rank =
                    global_rank_for_virtual_stage(plan, layout, replica_id, event.virtual_stage_id);
                let receiver_rank = global_rank_for_virtual_stage(
                    plan,
                    layout,
                    replica_id,
                    event.virtual_stage_id - 1,
                );

                if sender_rank == receiver_rank {
                    if assignment.data_parallel_rank == replica_id
                        && assignment.global_rank == receiver_rank
                    {
                        let grad_state = local_backward_grad.clone().ok_or_else(|| {
                            anyhow!(
                                "missing local backward gradient for virtual_stage={} microbatch={microbatch_id}",
                                event.virtual_stage_id
                            )
                        })?;
                        incoming_backward
                            .insert((event.virtual_stage_id - 1, microbatch_id), grad_state);
                    }
                    continue;
                }

                let broadcasted = broadcast_pipeline_state_inner_rooted::<B::InnerBackend>(
                    peer_id,
                    assignment.global_rank,
                    sender_rank,
                    device,
                    (assignment.data_parallel_rank == replica_id
                        && assignment.global_rank == sender_rank)
                        .then_some(local_backward_grad.as_ref())
                        .flatten(),
                )?;
                if assignment.data_parallel_rank == replica_id
                    && assignment.global_rank == receiver_rank
                {
                    incoming_backward
                        .insert((event.virtual_stage_id - 1, microbatch_id), broadcasted);
                }
            }
        }
    }

    let reduced_loss = reduce_sum_scalar::<B::InnerBackend>(
        peer_id,
        if assignment.is_last_stage() {
            local_loss.unwrap_or_else(|| Tensor::<B::InnerBackend, 1>::zeros([1], device))
        } else {
            Tensor::<B::InnerBackend, 1>::zeros([1], device)
        },
    )?;

    Ok(DistributedPipelineTrainStepResult {
        grads: local_accumulator.grads(),
        mean_train_loss: reduced_loss / layout.data_parallel_size as f64,
    })
}

#[cfg(feature = "ddp")]
fn train_with_collective_pipeline_scheduler<B, O, S>(
    env: &TrainEnvironment<'_, B>,
    mut learner: burn_train::Learner<
        burn_train::LearningComponentsMarker<B, S, LanguageTrainModel<B>, O>,
    >,
    local_train_loader: Arc<dyn DataLoader<B, SequenceBatch<B>>>,
    peer_id: PeerId,
    layout: PipelineParallelLayout,
    assignment: PipelineRankAssignment,
) -> Result<DragonModel<ValidBackend<B>>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    O: Optimizer<LanguageTrainModel<B>, B> + 'static,
    S: LrScheduler + 'static,
{
    let global_train_steps = env.train_loader.num_items();
    if global_train_steps % layout.data_parallel_size != 0 {
        return Err(anyhow!(
            "parallel.pipeline.enabled process-group execution requires env.train_loader.num_items() divisible by parallel.data.size so every replica executes the same number of collectives (got {} steps across {} replicas)",
            global_train_steps,
            layout.data_parallel_size
        ));
    }

    let local_train_steps = local_train_loader.num_items();
    let expected_local_train_steps = global_train_steps / layout.data_parallel_size;
    if local_train_steps != expected_local_train_steps {
        return Err(anyhow!(
            "parallel.pipeline.enabled process-group execution expected {} local steps for dp_rank={} but resolved {}",
            expected_local_train_steps,
            assignment.data_parallel_rank,
            local_train_steps
        ));
    }
    let metric_every = env.training.log_frequency.max(1);
    let grad_accumulation = env.training.gradient_accumulation_steps.max(1);
    let logical_replica_count = layout.data_parallel_size;
    let start_epoch = env
        .resume_checkpoint_epoch
        .map(|epoch| epoch + 1)
        .unwrap_or(1);

    for epoch in start_epoch..=env.epochs {
        info!(
            "Executing process-group pipeline epoch {} on global_rank={} stage={} dp_rank={}",
            epoch,
            assignment.global_rank,
            assignment.pipeline_stage_id,
            assignment.data_parallel_rank
        );

        let mut iterator = local_train_loader.iter();
        let mut iteration = 0usize;
        let mut accumulator = GradientsAccumulator::new();
        let mut accumulation_current = 0usize;

        while iteration < local_train_steps {
            let mut batch = None;
            for replica_id in 0..layout.data_parallel_size {
                let batch_root_rank = pipeline_replica_root_rank(&layout, replica_id);
                let replica_root_batch = if assignment.data_parallel_rank == replica_id
                    && assignment.global_rank == batch_root_rank
                {
                    iterator.next()
                } else {
                    None
                };
                let replica_batch = broadcast_sequence_batch_rooted(
                    peer_id,
                    assignment.global_rank,
                    batch_root_rank,
                    env.device,
                    replica_root_batch,
                )?;
                if assignment.data_parallel_rank == replica_id {
                    batch = Some(replica_batch);
                }
            }
            let batch = batch.ok_or_else(|| {
                anyhow!(
                    "missing local replica batch for dp_rank={} at iteration={iteration}",
                    assignment.data_parallel_rank
                )
            })?;

            iteration += 1;
            for _ in 0..logical_replica_count {
                learner.lr_step();
            }

            let step = distributed_pipeline_train_step(
                peer_id,
                &learner.model(),
                batch,
                &layout,
                &assignment,
                env.device,
            )?;

            accumulator.accumulate(&learner.model(), step.grads);
            accumulation_current += 1;

            if grad_accumulation <= accumulation_current {
                let mut grads = accumulator.grads();
                all_reduce_gradients_in_module_order(
                    &learner.model(),
                    &mut grads,
                    peer_id,
                    ReduceOperation::Sum,
                )?;
                scale_gradients_in_module_order::<B, _>(
                    &learner.model(),
                    &mut grads,
                    1.0 / layout.data_parallel_size as f32,
                );
                learner.optimizer_step(grads);
                accumulation_current = 0;
            }

            if env.parallel_runtime.is_primary()
                && (iteration % metric_every == 0 || iteration == local_train_steps)
            {
                let progress = iterator.progress();
                let global_iteration = epoch
                    .saturating_sub(1)
                    .saturating_mul(logical_replica_count.saturating_mul(local_train_steps))
                    .saturating_add(iteration.saturating_mul(logical_replica_count));
                info!(
                    "train epoch={} local_step={}/{} global_iteration={} loss={:.4} lr={:.6} global_progress={}/{}",
                    epoch,
                    progress.items_processed,
                    progress.items_total,
                    global_iteration,
                    step.mean_train_loss,
                    learner.lr_current(),
                    epoch,
                    env.epochs
                );
            }
        }

        if env.parallel_runtime.is_primary() {
            if let Some(valid_loss) = run_process_group_validation::<B, O, S>(env, &learner) {
                info!("valid epoch={} loss={valid_loss:.4}", epoch);
            }
            save_process_group_checkpoint::<B, O, S>(env.run_dir, epoch, &learner)?;
        }
    }

    Ok(learner.model().valid().model)
}

#[cfg(feature = "ddp")]
fn train_with_collective_scheduler<B, O, S>(
    env: &TrainEnvironment<'_, B>,
    model: LanguageTrainModel<B>,
    optimizer: O,
    scheduler: S,
    collective: burn_collective::CollectiveConfig,
    peer_id: PeerId,
) -> Result<DragonModel<ValidBackend<B>>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    O: Optimizer<LanguageTrainModel<B>, B> + 'static,
    S: LrScheduler + 'static,
{
    let _session = CollectiveSessionGuard::<B::InnerBackend>::register(
        peer_id,
        env.device.clone(),
        collective,
    )?;

    let (data_shard_index, data_shard_count, pipeline_assignment, pipeline_layout) =
        process_group_data_shard(env.parallel_runtime, env.parallel_config)?;

    let local_train_loader = shard_dataloader(
        Arc::clone(&env.train_loader),
        data_shard_index,
        data_shard_count,
        "train",
    )?;

    let metric_every = env.training.log_frequency.max(1);
    let grad_accumulation = env.training.gradient_accumulation_steps.max(1);
    let local_train_steps = local_train_loader.num_items();
    let mut learner = burn_train::Learner::new(model, optimizer, scheduler);
    if let Some(checkpoint) = env.resume_checkpoint_epoch {
        learner =
            load_process_group_checkpoint::<B, O, S>(env.run_dir, checkpoint, env.device, learner)?;
    }
    let start_epoch = env
        .resume_checkpoint_epoch
        .map(|epoch| epoch + 1)
        .unwrap_or(1);

    info!(
        "training strategy: mode={:?} replicas={} local_rank={} global_rank={} local_train_steps={} start_epoch={}",
        env.parallel_runtime.mode,
        env.parallel_runtime.world_size,
        env.parallel_runtime.local_rank,
        env.parallel_runtime.global_rank,
        local_train_steps,
        start_epoch
    );
    if let (Some(layout), Some(assignment)) = (&pipeline_layout, &pipeline_assignment) {
        info!(
            "process-group pipeline topology: {} rank={} stage={} dp_rank={} predecessor={:?} successor={:?} pipeline_group={:?} dp_group={:?}",
            layout.summary(),
            assignment.global_rank,
            assignment.pipeline_stage_id,
            assignment.data_parallel_rank,
            assignment.predecessor_global_rank,
            assignment.successor_global_rank,
            assignment.pipeline_group_ranks,
            assignment.data_parallel_group_ranks,
        );
    }

    if let (Some(layout), Some(assignment)) = (pipeline_layout.clone(), pipeline_assignment.clone())
    {
        return train_with_collective_pipeline_scheduler(
            env,
            learner,
            local_train_loader,
            peer_id,
            layout,
            assignment,
        );
    }

    for epoch in start_epoch..=env.epochs {
        info!(
            "Executing process-group DDP epoch {} on global_rank={}",
            epoch, env.parallel_runtime.global_rank
        );

        let mut iterator = local_train_loader.iter();
        let mut iteration = 0usize;
        let mut accumulator = GradientsAccumulator::new();
        let mut accumulation_current = 0usize;
        let logical_replica_count = env.parallel_runtime.world_size;
        while let Some(item) = iterator.next() {
            iteration += 1;
            for _ in 0..logical_replica_count {
                learner.lr_step();
            }

            let item = learner.train_step(item);
            let train_output = item.item.sync();
            let loss_value: LossValue<ValidBackend<B>> = train_output.adapt();
            info!(
                "process-group DDP rank={} iteration={} entering scalar loss all-reduce",
                env.parallel_runtime.global_rank, iteration
            );
            let mean_train_loss =
                reduce_mean_scalar::<ValidBackend<B>>(peer_id, loss_value.value())?;
            info!(
                "process-group DDP rank={} iteration={} completed scalar loss all-reduce",
                env.parallel_runtime.global_rank, iteration
            );
            if let Some(dataset) = env
                .source_selection_dataset
                .as_ref()
                .filter(|dataset| dataset.uses_live_source_selection())
            {
                let absolute_step = epoch
                    .saturating_sub(1)
                    .saturating_mul(local_train_steps)
                    .saturating_add(iteration.saturating_sub(1));
                let _ = dataset.record_source_selection_loss(absolute_step, mean_train_loss as f32);
            }

            accumulator.accumulate(&learner.model(), item.grads);
            accumulation_current += 1;

            if grad_accumulation <= accumulation_current {
                info!(
                    "process-group DDP rank={} iteration={} entering gradient all-reduce",
                    env.parallel_runtime.global_rank, iteration
                );
                let mut grads = accumulator.grads();
                // Fresh multi-process launches instantiate random ParamIds per rank, so
                // cross-rank gradient sync must follow deterministic module traversal order.
                all_reduce_gradients_in_module_order(
                    &learner.model(),
                    &mut grads,
                    peer_id,
                    ReduceOperation::Mean,
                )?;
                info!(
                    "process-group DDP rank={} iteration={} completed gradient all-reduce",
                    env.parallel_runtime.global_rank, iteration
                );
                learner.optimizer_step(grads);
                accumulation_current = 0;
            }

            if env.parallel_runtime.is_primary()
                && (iteration % metric_every == 0 || iteration == local_train_steps)
            {
                let progress = iterator.progress();
                let global_iteration = epoch
                    .saturating_sub(1)
                    .saturating_mul(logical_replica_count.saturating_mul(local_train_steps))
                    .saturating_add(iteration.saturating_mul(logical_replica_count));
                info!(
                    "train epoch={} local_step={}/{} global_iteration={} loss={:.4} lr={:.6} global_progress={}/{}",
                    epoch,
                    progress.items_processed,
                    progress.items_total,
                    global_iteration,
                    mean_train_loss,
                    learner.lr_current(),
                    epoch,
                    env.epochs
                );
            }
        }

        if env.parallel_runtime.is_primary() {
            if let Some(valid_loss) = run_process_group_validation::<B, O, S>(env, &learner) {
                info!("valid epoch={} loss={valid_loss:.4}", epoch);
            }
            save_process_group_checkpoint::<B, O, S>(env.run_dir, epoch, &learner)?;
        }
    }

    Ok(learner.model().valid().model)
}

#[cfg(feature = "ddp")]
fn train_with_process_group_scheduler<B, O, S>(
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
    let collective = resolve_collective_config(env.parallel_runtime, env.parallel_config)?;
    train_with_collective_scheduler::<B, O, S>(
        env,
        model,
        optimizer,
        scheduler,
        collective,
        process_group_peer_id(env.parallel_runtime),
    )
}

pub fn resolve_lr_scheduler(
    optimizer_cfg: &OptimizerConfig,
    total_steps: usize,
    override_num_iters: Option<usize>,
    model_config: &DragonConfig,
) -> Result<ResolvedLrScheduler> {
    burn_dragon_train::train::pipeline::resolve_lr_scheduler(
        optimizer_cfg,
        total_steps,
        override_num_iters,
        model_config.n_embd,
    )
}

pub fn resolve_train_schedule(
    training: &TrainingHyperparameters,
    steps_per_epoch: usize,
) -> Result<TrainSchedule> {
    burn_dragon_train::train::pipeline::resolve_train_schedule(
        training.epochs,
        training.max_iters,
        steps_per_epoch,
        "training",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::data::dataloader::{DataLoaderIterator, Progress};
    #[cfg(feature = "ddp")]
    use burn::module::list_param_ids;
    use burn::tensor::TensorData;
    use burn_autodiff::Autodiff;
    #[cfg(feature = "ddp")]
    use burn_collective::reset_collective;
    use burn_ndarray::NdArray;
    use burn_train::checkpoint::CheckpointingAction;
    #[cfg(feature = "ddp")]
    use std::sync::{Mutex, OnceLock};
    #[cfg(feature = "ddp")]
    use tempfile::tempdir;

    type TestBackend = Autodiff<NdArray<f32>>;
    type TestValidBackend = ValidBackend<TestBackend>;

    #[test]
    fn file_metric_best_strategy_tracks_best_value() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut strategy = FileMetricBestCheckpointingStrategy::new(
            dir.path(),
            &LossMetric::<TestValidBackend>::new(),
            burn_train::metric::store::Direction::Lowest,
            burn_train::metric::store::Split::Valid,
        );

        let previous_best = strategy.update_best_candidate(1, 3.5);

        assert_eq!(previous_best, None);
        assert_eq!(strategy.best_epoch, Some(1));
        assert_eq!(strategy.best_value, Some(3.5));
    }

    #[test]
    fn file_metric_best_strategy_replaces_only_on_improvement() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut strategy = FileMetricBestCheckpointingStrategy::new(
            dir.path(),
            &LossMetric::<TestValidBackend>::new(),
            burn_train::metric::store::Direction::Lowest,
            burn_train::metric::store::Split::Valid,
        );
        strategy.best_epoch = Some(2);
        strategy.best_value = Some(3.2);

        let worse_previous_best = strategy.update_best_candidate(3, 3.3);
        assert_eq!(worse_previous_best, None);
        assert_eq!(strategy.best_epoch, Some(2));
        assert_eq!(strategy.best_value, Some(3.2));

        let better_previous_best = strategy.update_best_candidate(4, 3.1);
        assert_eq!(better_previous_best, Some(2));
        assert_eq!(strategy.best_epoch, Some(4));
        assert_eq!(strategy.best_value, Some(3.1));
    }

    fn write_metric_log(run_dir: &Path, split: &str, epoch: usize, values: &[f64]) {
        let epoch_dir = run_dir.join(split).join(format!("epoch-{epoch}"));
        fs::create_dir_all(&epoch_dir).expect("create epoch dir");
        let path = epoch_dir.join("Loss.log");
        let content = values
            .iter()
            .map(|value| format!("{value},1"))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(path, content).expect("write metric log");
    }

    fn apply_checkpoint_actions(run_dir: &Path, epoch: usize, actions: &[CheckpointingAction]) {
        let checkpoint_dir = run_dir.join("checkpoint");
        fs::create_dir_all(&checkpoint_dir).expect("create checkpoint dir");
        for action in actions {
            match action {
                CheckpointingAction::Save => {
                    for prefix in ["model", "optim", "scheduler"] {
                        fs::write(
                            checkpoint_dir.join(format!("{prefix}-{epoch}.bin")),
                            format!("{prefix}-{epoch}"),
                        )
                        .expect("write checkpoint file");
                    }
                }
                CheckpointingAction::Delete(epoch) => {
                    for prefix in ["model", "optim", "scheduler"] {
                        let path = checkpoint_dir.join(format!("{prefix}-{epoch}.bin"));
                        if path.exists() {
                            fs::remove_file(path).expect("remove checkpoint file");
                        }
                    }
                }
            }
        }
    }

    fn retained_model_epochs(run_dir: &Path) -> Vec<usize> {
        let checkpoint_dir = run_dir.join("checkpoint");
        let mut epochs = fs::read_dir(&checkpoint_dir)
            .expect("read checkpoint dir")
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                let name = path.file_name()?.to_str()?;
                let epoch = name
                    .strip_prefix("model-")?
                    .strip_suffix(".bin")?
                    .parse::<usize>()
                    .ok()?;
                Some(epoch)
            })
            .collect::<Vec<_>>();
        epochs.sort_unstable();
        epochs
    }

    #[test]
    fn file_metric_best_strategy_preserves_old_best_outside_keep_last_window() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut strategy = FileMetricBestCheckpointingStrategy::new(
            dir.path(),
            &LossMetric::<TestValidBackend>::new(),
            burn_train::metric::store::Direction::Lowest,
            burn_train::metric::store::Split::Valid,
        );

        let means = [
            2.0, 1.9, 1.8, 1.7, 1.6, 1.55, 1.53, 1.52, 1.515, 1.51, 1.509, 1.508, 1.507, 1.506,
            1.505, 1.504, 1.503, 1.502, 1.497, 1.501, 1.510, 1.512, 1.511, 1.499, 1.513, 1.514,
            1.502, 1.520, 1.506, 1.530,
        ];

        for (index, mean) in means.iter().enumerate() {
            let epoch = index + 1;
            write_metric_log(dir.path(), "valid", epoch, &[*mean]);
            let actions = strategy.actions_for_epoch(epoch);
            apply_checkpoint_actions(dir.path(), epoch, &actions);
        }

        assert_eq!(strategy.best_epoch, Some(19));
        assert_eq!(retained_model_epochs(dir.path()), vec![19, 29, 30]);
    }

    #[test]
    fn dynamic_scheduler_checkpoint_pruning_keeps_recent_and_best() {
        let dir = tempfile::tempdir().expect("tempdir");
        let checkpoint_dir = dir.path().join("checkpoint");
        fs::create_dir_all(&checkpoint_dir).expect("checkpoint dir");
        for epoch in 1..=5 {
            fs::write(checkpoint_dir.join(format!("model-{epoch}.bin")), "model")
                .expect("checkpoint");
        }

        prune_dragon_model_checkpoints(dir.path(), 5, Some(2)).expect("prune checkpoints");

        assert_eq!(retained_model_epochs(dir.path()), vec![2, 4, 5]);
    }

    #[test]
    fn file_metric_best_strategy_deletes_old_best_after_replacement() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut strategy = FileMetricBestCheckpointingStrategy::new(
            dir.path(),
            &LossMetric::<TestValidBackend>::new(),
            burn_train::metric::store::Direction::Lowest,
            burn_train::metric::store::Split::Valid,
        );

        for (epoch, mean) in [(1, 3.0), (2, 2.0), (3, 2.5), (4, 1.5)] {
            write_metric_log(dir.path(), "valid", epoch, &[mean]);
            let actions = strategy.actions_for_epoch(epoch);
            apply_checkpoint_actions(dir.path(), epoch, &actions);
        }

        assert_eq!(strategy.best_epoch, Some(4));
        assert_eq!(retained_model_epochs(dir.path()), vec![3, 4]);
    }

    #[test]
    fn file_metric_best_strategy_rehydrates_history_when_resuming() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut strategy = FileMetricBestCheckpointingStrategy::new(
            dir.path(),
            &LossMetric::<TestValidBackend>::new(),
            burn_train::metric::store::Direction::Lowest,
            burn_train::metric::store::Split::Valid,
        );

        for (epoch, mean) in [(1, 3.0), (2, 1.5), (3, 2.0), (4, 2.1), (5, 2.2), (6, 2.3)] {
            write_metric_log(dir.path(), "valid", epoch, &[mean]);
        }
        for epoch in [2, 5, 6] {
            apply_checkpoint_actions(dir.path(), epoch, &[CheckpointingAction::Save]);
        }

        write_metric_log(dir.path(), "valid", 7, &[2.4]);
        let actions = strategy.actions_for_epoch(7);
        apply_checkpoint_actions(dir.path(), 7, &actions);

        assert_eq!(strategy.best_epoch, Some(2));
        assert_eq!(retained_model_epochs(dir.path()), vec![2, 6, 7]);
    }

    #[test]
    fn file_metric_best_strategy_recomputes_history_when_new_best_log_arrives_late() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut strategy = FileMetricBestCheckpointingStrategy::new(
            dir.path(),
            &LossMetric::<TestValidBackend>::new(),
            burn_train::metric::store::Direction::Lowest,
            burn_train::metric::store::Split::Valid,
        );

        for epoch in 1..=23 {
            let mean = if epoch == 23 {
                1.50
            } else {
                2.0 + epoch as f64 * 0.01
            };
            write_metric_log(dir.path(), "valid", epoch, &[mean]);
            let actions = strategy.actions_for_epoch(epoch);
            apply_checkpoint_actions(dir.path(), epoch, &actions);
        }

        for epoch in 24..=28 {
            write_metric_log(dir.path(), "valid", epoch, &[1.60 + epoch as f64 * 0.001]);
            let actions = strategy.actions_for_epoch(epoch);
            apply_checkpoint_actions(dir.path(), epoch, &actions);
        }

        let actions = strategy.actions_for_epoch(29);
        apply_checkpoint_actions(dir.path(), 29, &actions);
        write_metric_log(dir.path(), "valid", 29, &[1.48]);

        write_metric_log(dir.path(), "valid", 30, &[1.49]);
        let actions = strategy.actions_for_epoch(30);
        apply_checkpoint_actions(dir.path(), 30, &actions);

        assert_eq!(strategy.best_epoch, Some(29));
        assert_eq!(retained_model_epochs(dir.path()), vec![29, 30]);
    }

    #[derive(Clone)]
    struct StaticSequenceLoader<B: BackendTrait> {
        items: Vec<SequenceBatch<B>>,
    }

    impl<B: BackendTrait> StaticSequenceLoader<B> {
        fn new(items: Vec<SequenceBatch<B>>) -> Self {
            Self { items }
        }
    }

    struct StaticSequenceIterator<B: BackendTrait> {
        items: Vec<SequenceBatch<B>>,
        index: usize,
    }

    impl<B: BackendTrait> Iterator for StaticSequenceIterator<B> {
        type Item = SequenceBatch<B>;

        fn next(&mut self) -> Option<Self::Item> {
            let item = self.items.get(self.index).cloned();
            if item.is_some() {
                self.index += 1;
            }
            item
        }
    }

    impl<B: BackendTrait> DataLoaderIterator<SequenceBatch<B>> for StaticSequenceIterator<B> {
        fn progress(&self) -> Progress {
            Progress::new(self.index, self.items.len())
        }
    }

    impl<B> DataLoader<B, SequenceBatch<B>> for StaticSequenceLoader<B>
    where
        B: BackendTrait + 'static,
    {
        fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<SequenceBatch<B>> + 'a> {
            Box::new(StaticSequenceIterator {
                items: self.items.clone(),
                index: 0,
            })
        }

        fn num_items(&self) -> usize {
            self.items.len()
        }

        fn to_device(&self, _device: &B::Device) -> Arc<dyn DataLoader<B, SequenceBatch<B>>> {
            Arc::new(self.clone())
        }

        fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<B, SequenceBatch<B>>> {
            let len = self.items.len();
            let start = start.min(len);
            let end = end.min(len);
            Arc::new(Self {
                items: self.items[start..end].to_vec(),
            })
        }
    }

    fn make_batch<B: BackendTrait>(
        device: &B::Device,
        inputs: &[i64],
        targets: &[i64],
        shape: [usize; 2],
    ) -> SequenceBatch<B> {
        SequenceBatch::new(
            Tensor::<B, 2, Int>::from_data(TensorData::new(inputs.to_vec(), shape), device),
            Tensor::<B, 2, Int>::from_data(TensorData::new(targets.to_vec(), shape), device),
            None,
        )
    }

    fn tiny_model_config() -> DragonConfig {
        DragonConfig {
            n_layer: 1,
            n_embd: 8,
            n_head: 1,
            mlp_internal_dim_multiplier: 1,
            dropout: 0.0,
            vocab_size: 16,
            ..Default::default()
        }
    }

    fn tiny_training_hparams() -> TrainingHyperparameters {
        TrainingHyperparameters {
            block_size: 4,
            tbptt_chunk_size: None,
            tbptt_persist_across_steps: false,
            min_logical_block_size: None,
            batch_size: 2,
            seed: 1337,
            gradient_accumulation_steps: 1,
            target_effective_batch_size: None,
            epochs: Some(1),
            max_iters: 2,
            checkpoint_interval_iters: 2000,
            log_frequency: 1,
            launch_mode: burn_dragon_train::train::pipeline::TrainingLaunchMode::Fresh,
            resume_run_dir: None,
            resume_checkpoint_epoch: None,
            init_checkpoint_path: None,
            init_checkpoint_epoch: None,
            init_transfer: Default::default(),
            continual_backprop: Default::default(),
            module_lr_scales: Vec::new(),
            context_strategy: ContextStrategyConfig::Infinite,
            sequence_kernel_override: None,
            objective: Default::default(),
            gdpo: None,
            events: Default::default(),
            gates: Default::default(),
            neuron_scaling: Default::default(),
            auto_batch_size: Default::default(),
        }
    }

    fn tiny_training_hparams_with_epochs(
        epochs: usize,
        resume_checkpoint_epoch: Option<usize>,
    ) -> TrainingHyperparameters {
        let mut training = tiny_training_hparams();
        training.epochs = Some(epochs);
        training.resume_checkpoint_epoch = resume_checkpoint_epoch;
        training
    }

    fn objective_training_hparams(objective: TrainingObjectiveConfig) -> TrainingHyperparameters {
        let mut training = tiny_training_hparams();
        training.objective = objective;
        training
    }

    fn tiny_language_optimizer(
        training: &TrainingHyperparameters,
        model_config: &DragonConfig,
        device: &burn::tensor::Device<TestBackend>,
    ) -> crate::train::continual_backprop::LanguageOptimizer<TestBackend> {
        let optimizer_cfg = OptimizerConfig {
            name: OptimizerKind::Adamw,
            learning_rate: 1e-3,
            weight_decay: 0.0,
            weight_decay_final: None,
            lr_schedule: None,
            schedule_mode: OptimizerScheduleMode::DragonReference,
            grad_clip_norm: None,
            grad_clip_value: None,
        };
        let fresh_model = DragonModel::<TestBackend>::new(model_config.clone(), device);
        crate::train::continual_backprop::resolve_dragon_language_optimizer::<TestBackend>(
            training,
            &optimizer_cfg,
            1,
            fresh_model,
        )
        .expect("optimizer")
    }

    fn single_device_scheduler_smoke(objective: TrainingObjectiveConfig, run_name: &str) -> f32 {
        let dir = tempfile::tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");
        let parallel_config = burn_dragon_train::ParallelConfig::default();
        let parallel_runtime =
            resolve_parallel_runtime(&parallel_config).expect("resolve single runtime");

        let primary_device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&primary_device, 11);
        let valid_device = burn::tensor::Device::<TestValidBackend>::default();
        let train_batches = vec![
            make_batch::<TestBackend>(
                &primary_device,
                &[0, 1, 2, 3, 4, 5, 6, 7],
                &[1, 2, 3, 4, 5, 6, 7, 0],
                [2, 4],
            ),
            make_batch::<TestBackend>(
                &primary_device,
                &[7, 6, 5, 4, 3, 2, 1, 0],
                &[6, 5, 4, 3, 2, 1, 0, 7],
                [2, 4],
            ),
        ];
        let valid_batches = vec![make_batch::<TestValidBackend>(
            &valid_device,
            &[0, 0, 1, 1, 2, 2, 3, 3],
            &[0, 1, 1, 2, 2, 3, 3, 0],
            [2, 4],
        )];

        let training = objective_training_hparams(objective.clone());
        let model_config = tiny_model_config();
        let devices = vec![primary_device];
        let env = TrainEnvironment {
            parallel_runtime: &parallel_runtime,
            parallel_config: &parallel_config,
            run_dir: &run_dir,
            run_name,
            backend_name: "cpu",
            training: &training,
            resume_checkpoint_epoch: None,
            model_config: &model_config,
            device: &primary_device,
            devices: &devices,
            train_dataset: None,
            valid_dataset: None,
            train_loader: Arc::new(StaticSequenceLoader::new(train_batches)),
            valid_loader: Arc::new(StaticSequenceLoader::new(valid_batches)),
            source_selection_dataset: None,
            summary_event_token_ids: None,
            neuron_scaling_slot: None,
            epochs: 1,
            total_steps: 2,
            valid_steps: 1,
        };
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            model_config.clone(),
            &primary_device,
        ))
        .with_training_objective(objective);
        let optimizer = AdamWConfig::new()
            .with_weight_decay(0.0)
            .init::<TestBackend, LanguageTrainModel<TestBackend>>();

        let trained =
            train_with_scheduler(&env, model, optimizer, 1e-3).expect("objective scheduler train");
        assert!(run_dir.join("checkpoint").join("model-1.bin").is_file());

        let probe = make_batch::<TestValidBackend>(
            &valid_device,
            &[1, 2, 3, 4, 4, 3, 2, 1],
            &[2, 3, 4, 5, 3, 2, 1, 0],
            [2, 4],
        );
        language_model_loss::<TestValidBackend>(trained.forward(probe.inputs), probe.targets)
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("loss vec")[0]
    }

    #[test]
    fn train_with_scheduler_accepts_next_token_objective_toggle() {
        let loss = single_device_scheduler_smoke(
            TrainingObjectiveConfig::NextToken,
            "single-next-token-objective-smoke",
        );
        assert!(loss.is_finite(), "next_token smoke loss must be finite");
    }

    #[test]
    fn train_with_scheduler_accepts_sdft_objective_toggle() {
        let loss = single_device_scheduler_smoke(
            TrainingObjectiveConfig::Sdft(SdftObjectiveConfig {
                max_completion_tokens: 2,
                top_k: Some(1),
                generate_from_teacher: true,
                num_loss_tokens_to_skip: 1,
                ..Default::default()
            }),
            "single-sdft-objective-smoke",
        );
        assert!(loss.is_finite(), "SDFT smoke loss must be finite");
    }

    #[test]
    fn train_with_scheduler_accepts_sdpo_objective_toggle() {
        let loss = single_device_scheduler_smoke(
            TrainingObjectiveConfig::Sdpo(SdpoObjectiveConfig {
                group_size: 2,
                max_completion_tokens: 2,
                top_k: Some(1),
                ..Default::default()
            }),
            "single-sdpo-objective-smoke",
        );
        assert!(loss.is_finite(), "SDPO smoke loss must be finite");
    }

    #[test]
    fn train_with_scheduler_accepts_composite_sdft_sdpo_objective_toggle() {
        let loss = single_device_scheduler_smoke(
            TrainingObjectiveConfig::SdftSdpo(SdftSdpoObjectiveConfig {
                sdft: SdftObjectiveConfig {
                    max_completion_tokens: 2,
                    top_k: Some(1),
                    ..Default::default()
                },
                sdpo: SdpoObjectiveConfig {
                    group_size: 2,
                    max_completion_tokens: 2,
                    top_k: Some(1),
                    ..Default::default()
                },
                ..Default::default()
            }),
            "single-sdft-sdpo-objective-smoke",
        );
        assert!(
            loss.is_finite(),
            "composite SDFT/SDPO smoke loss must be finite"
        );
    }

    #[test]
    fn dynamic_neuron_scale_widens_model_in_process() {
        let dir = tempfile::tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");
        let parallel_config = burn_dragon_train::ParallelConfig::default();
        let parallel_runtime =
            resolve_parallel_runtime(&parallel_config).expect("resolve single runtime");
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let valid_device = burn::tensor::Device::<TestValidBackend>::default();
        let mut training = tiny_training_hparams();
        training.neuron_scaling.enabled = true;
        training.neuron_scaling.max_latent_total = 16;
        training.neuron_scaling.stabilization.freeze_base_steps = 1;
        training.neuron_scaling.stabilization.unfreeze_ramp_steps = 1;
        let model_config = tiny_model_config();
        let devices = vec![device.clone()];
        let train_batches = vec![make_batch::<TestBackend>(
            &device,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[1, 2, 3, 4, 5, 6, 7, 0],
            [2, 4],
        )];
        let valid_batches = vec![make_batch::<TestValidBackend>(
            &valid_device,
            &[0, 0, 1, 1, 2, 2, 3, 3],
            &[0, 1, 1, 2, 2, 3, 3, 0],
            [2, 4],
        )];
        let env = TrainEnvironment {
            parallel_runtime: &parallel_runtime,
            parallel_config: &parallel_config,
            run_dir: &run_dir,
            run_name: "dynamic-scale-smoke",
            backend_name: "cpu",
            training: &training,
            resume_checkpoint_epoch: None,
            model_config: &model_config,
            device: &device,
            devices: &devices,
            train_dataset: None,
            valid_dataset: None,
            train_loader: Arc::new(StaticSequenceLoader::new(train_batches)),
            valid_loader: Arc::new(StaticSequenceLoader::new(valid_batches)),
            source_selection_dataset: None,
            summary_event_token_ids: None,
            neuron_scaling_slot: None,
            epochs: 1,
            total_steps: 1,
            valid_steps: 1,
        };
        let mut model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            model_config.clone(),
            &device,
        ))
        .with_gradient_scale_schedule(&training, 1);
        let mut optimizer = tiny_language_optimizer(&training, &model_config, &device);
        let handles = crate::train::events::build_training_event_handles(
            "dynamic-scale-smoke",
            &run_dir,
            1,
            &training,
            None,
            None,
        )
        .expect("event handles");
        let bus = handles.metric_logger.bus();
        let mut current_model_config = model_config.clone();
        let mut scale_generation = 0usize;

        let scale_result = apply_dynamic_neuron_scale(
            &env,
            &mut model,
            &mut optimizer,
            &mut current_model_config,
            &mut scale_generation,
            ModelScaleRequest {
                run_id: "dynamic-scale-smoke".to_string(),
                epoch: Some(1),
                absolute_step: Some(0),
                from_capacity_units: 8,
                to_capacity_units: 16,
                reason: "test plateau".to_string(),
            },
            1,
            0,
            &bus,
            training.batch_size,
            training.gradient_accumulation_steps,
        )
        .expect("apply scale");

        let _ = bus.flush();
        assert_eq!(scale_result, Some((8, 16)));
        assert_eq!(model.model.latent_total_capacity(), 16);
        assert_eq!(current_model_config.latent_total(), 16);
        assert_eq!(scale_generation, 1);
    }

    #[test]
    fn dynamic_neuron_scaling_scheduler_consumes_request_in_process() {
        let dir = tempfile::tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");
        let parallel_config = burn_dragon_train::ParallelConfig::default();
        let parallel_runtime =
            resolve_parallel_runtime(&parallel_config).expect("resolve single runtime");
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 13);
        let valid_device = burn::tensor::Device::<TestValidBackend>::default();
        let mut training = tiny_training_hparams();
        training.neuron_scaling.enabled = true;
        training.neuron_scaling.max_latent_total = 16;
        let model_config = tiny_model_config();
        let devices = vec![device.clone()];
        let request_slot = crate::train::neuron_scaling::NeuronScaleRequestSlot::default();
        assert!(request_slot.set_if_empty(ModelScaleRequest {
            run_id: "dynamic-scale-loop-smoke".to_string(),
            epoch: Some(1),
            absolute_step: Some(0),
            from_capacity_units: 8,
            to_capacity_units: 16,
            reason: "test plateau".to_string(),
        }));
        let train_batches = vec![make_batch::<TestBackend>(
            &device,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[1, 2, 3, 4, 5, 6, 7, 0],
            [2, 4],
        )];
        let valid_batches = vec![make_batch::<TestValidBackend>(
            &valid_device,
            &[0, 0, 1, 1, 2, 2, 3, 3],
            &[0, 1, 1, 2, 2, 3, 3, 0],
            [2, 4],
        )];
        let env = TrainEnvironment {
            parallel_runtime: &parallel_runtime,
            parallel_config: &parallel_config,
            run_dir: &run_dir,
            run_name: "dynamic-scale-loop-smoke",
            backend_name: "cpu",
            training: &training,
            resume_checkpoint_epoch: None,
            model_config: &model_config,
            device: &device,
            devices: &devices,
            train_dataset: None,
            valid_dataset: None,
            train_loader: Arc::new(StaticSequenceLoader::new(train_batches)),
            valid_loader: Arc::new(StaticSequenceLoader::new(valid_batches)),
            source_selection_dataset: None,
            summary_event_token_ids: None,
            neuron_scaling_slot: Some(request_slot.clone()),
            epochs: 1,
            total_steps: 1,
            valid_steps: 1,
        };
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            model_config.clone(),
            &device,
        ))
        .with_gradient_scale_schedule(&training, 1);
        let optimizer = tiny_language_optimizer(&training, &model_config, &device);

        let trained = train_with_dynamic_neuron_scaling_scheduler(&env, model, optimizer, 1e-3)
            .expect("dynamic scaling train");

        assert_eq!(trained.latent_total_capacity(), 16);
        assert!(request_slot.take().is_none());
        assert!(run_dir.join("checkpoint").join("model-1.bin").is_file());
    }

    #[cfg(feature = "ddp")]
    fn collective_test_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[cfg(feature = "ddp")]
    fn flatten_gradients_in_module_order<B, M>(module: &M, mut grads: GradientsParams) -> Vec<f32>
    where
        B: AutodiffBackend,
        M: AutodiffModule<B>,
    {
        #[derive(Default)]
        struct GradientCollector {
            values: Vec<f32>,
        }

        struct GradientCollectorVisitor<'a> {
            collector: &'a mut GradientCollector,
            grads: &'a mut GradientsParams,
        }

        impl<B: AutodiffBackend> burn::module::ModuleVisitor<B> for GradientCollectorVisitor<'_> {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
                let grad = self
                    .grads
                    .remove::<B::InnerBackend, D>(param.id)
                    .unwrap_or_else(|| param.val().inner().zeros_like());
                let values = grad
                    .to_data()
                    .convert::<f32>()
                    .into_vec::<f32>()
                    .expect("gradient data");
                self.collector.values.extend(values);
            }
        }

        let mut collector = GradientCollector::default();
        let mut visitor = GradientCollectorVisitor {
            collector: &mut collector,
            grads: &mut grads,
        };
        module.visit(&mut visitor);
        collector.values
    }

    #[cfg(feature = "ddp")]
    fn mean_abs_diff(left: &[f32], right: &[f32]) -> f32 {
        left.iter()
            .zip(right.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .sum::<f32>()
            / left.len().max(1) as f32
    }

    #[cfg(feature = "ddp")]
    fn l2_norm(values: &[f32]) -> f32 {
        values.iter().map(|value| value * value).sum::<f32>().sqrt()
    }

    #[cfg(feature = "ddp")]
    fn stage_split_surrogate_gradients(
        split_model: LanguageTrainModel<TestBackend>,
        plan: &PipelinePlan,
        batch: SequenceBatch<TestBackend>,
    ) -> Vec<f32> {
        let [batch_size, _] = batch.inputs.shape().dims();
        let ranges = split_microbatch_ranges(batch_size, plan.microbatches).expect("ranges");
        let chunk_inputs = ranges
            .iter()
            .cloned()
            .map(|range| slice_batch_int(batch.inputs.clone(), range))
            .collect::<Vec<_>>();
        let chunk_targets = ranges
            .iter()
            .cloned()
            .map(|range| slice_batch_int(batch.targets.clone(), range))
            .collect::<Vec<_>>();
        let chunk_masks = ranges
            .iter()
            .cloned()
            .map(|range| {
                batch
                    .summary_event_mask
                    .clone()
                    .map(|mask| slice_batch_int(mask, range))
            })
            .collect::<Vec<_>>();
        let mut chunk_states = (0..plan.microbatches)
            .map(|_| split_model.model.init_state())
            .collect::<Vec<_>>();
        let mut accumulator = GradientsAccumulator::new();
        let last_virtual_stage_id = plan.total_virtual_stages.saturating_sub(1);

        for microbatch_id in 0..plan.microbatches {
            let stage0_output = split_model
                .model
                .forward_language_pipeline_stage_with_state(
                    split_model
                        .model
                        .begin_language_pipeline(chunk_inputs[microbatch_id].clone()),
                    &mut chunk_states[microbatch_id],
                    plan.assignment(0).layer_range.clone(),
                    chunk_masks[microbatch_id].clone(),
                );
            let stage1_input = attach_pipeline_state_require_grad::<TestBackend>(
                detach_pipeline_state_to_inner(&stage0_output),
            );
            let stage1_input_for_grad = stage1_input.clone();
            let stage1_output = split_model
                .model
                .forward_language_pipeline_stage_with_state(
                    stage1_input,
                    &mut chunk_states[microbatch_id],
                    plan.assignment(last_virtual_stage_id).layer_range.clone(),
                    chunk_masks[microbatch_id].clone(),
                );
            let hidden = split_model
                .model
                .finish_language_pipeline_hidden_with_state(
                    stage1_output,
                    &mut chunk_states[microbatch_id],
                );
            let weight = ranges[microbatch_id].len() as f32 / batch_size as f32;
            let loss = split_model
                .model
                .language_loss_from_hidden(hidden, chunk_targets[microbatch_id].clone())
                .mul_scalar(weight);
            let mut stage1_grads = loss.backward();
            let grad_to_stage0 =
                pipeline_input_grad_state(&stage1_input_for_grad, &mut stage1_grads);
            accumulator.accumulate(
                &split_model,
                GradientsParams::from_grads(stage1_grads, &split_model),
            );

            let stage0_surrogate = pipeline_surrogate_loss(&stage0_output, grad_to_stage0);
            accumulator.accumulate(
                &split_model,
                GradientsParams::from_grads(stage0_surrogate.backward(), &split_model),
            );
        }

        flatten_gradients_in_module_order::<TestBackend, _>(&split_model, accumulator.grads())
    }

    #[cfg(feature = "ddp")]
    #[test]
    fn train_with_scheduler_executes_local_ddp_on_ndarray() {
        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");

        let parallel_config = burn_dragon_train::ParallelConfig {
            mode: ParallelismKind::Ddp,
            world_size: 2,
            data: burn_dragon_train::ParallelDataConfig {
                size: 2,
                ..Default::default()
            },
            ..Default::default()
        };
        let parallel_runtime =
            resolve_parallel_runtime(&parallel_config).expect("resolve local ddp runtime");

        let primary_device = burn::tensor::Device::<TestBackend>::default();
        let devices =
            resolve_training_devices::<TestBackend>(&parallel_runtime, &primary_device).unwrap();
        assert_eq!(devices.len(), 2, "expected 2 local replicas");

        let valid_device = burn::tensor::Device::<TestValidBackend>::default();
        let train_batches = vec![
            make_batch::<TestBackend>(
                &primary_device,
                &[0, 1, 2, 3, 4, 5, 6, 7],
                &[1, 2, 3, 4, 5, 6, 7, 0],
                [2, 4],
            ),
            make_batch::<TestBackend>(
                &primary_device,
                &[7, 6, 5, 4, 3, 2, 1, 0],
                &[6, 5, 4, 3, 2, 1, 0, 7],
                [2, 4],
            ),
        ];
        let valid_batches = vec![make_batch::<TestValidBackend>(
            &valid_device,
            &[0, 0, 1, 1, 2, 2, 3, 3],
            &[0, 1, 1, 2, 2, 3, 3, 0],
            [2, 4],
        )];

        let training = tiny_training_hparams();
        let model_config = tiny_model_config();
        let env = TrainEnvironment {
            parallel_runtime: &parallel_runtime,
            parallel_config: &parallel_config,
            run_dir: &run_dir,
            run_name: "ddp-ndarray-smoke",
            backend_name: "cpu",
            training: &training,
            resume_checkpoint_epoch: None,
            model_config: &model_config,
            device: &primary_device,
            devices: &devices,
            train_dataset: None,
            valid_dataset: None,
            train_loader: Arc::new(StaticSequenceLoader::new(train_batches)),
            valid_loader: Arc::new(StaticSequenceLoader::new(valid_batches)),
            source_selection_dataset: None,
            summary_event_token_ids: None,
            neuron_scaling_slot: None,
            epochs: 1,
            total_steps: 2,
            valid_steps: 1,
        };

        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            model_config.clone(),
            &primary_device,
        ));
        let optimizer = AdamWConfig::new()
            .with_weight_decay(0.0)
            .init::<TestBackend, LanguageTrainModel<TestBackend>>();

        let trained = train_with_scheduler(&env, model, optimizer, 1e-3).expect("ddp train");
        let probe = make_batch::<TestValidBackend>(
            &valid_device,
            &[1, 2, 3, 4, 4, 3, 2, 1],
            &[2, 3, 4, 5, 3, 2, 1, 0],
            [2, 4],
        );
        let loss =
            language_model_loss::<TestValidBackend>(trained.forward(probe.inputs), probe.targets)
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("loss vec")[0];

        assert!(loss.is_finite(), "ddp smoke loss must be finite");
    }

    #[test]
    fn train_with_scheduler_retains_best_valid_and_last_checkpoints() {
        let dir = tempfile::tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");

        let parallel_config = burn_dragon_train::ParallelConfig::default();
        let parallel_runtime =
            resolve_parallel_runtime(&parallel_config).expect("resolve single runtime");

        let primary_device = burn::tensor::Device::<TestBackend>::default();
        let valid_device = burn::tensor::Device::<TestValidBackend>::default();
        let train_batches = vec![
            make_batch::<TestBackend>(
                &primary_device,
                &[0, 1, 2, 3, 4, 5, 6, 7],
                &[1, 2, 3, 4, 5, 6, 7, 0],
                [2, 4],
            ),
            make_batch::<TestBackend>(
                &primary_device,
                &[7, 6, 5, 4, 3, 2, 1, 0],
                &[6, 5, 4, 3, 2, 1, 0, 7],
                [2, 4],
            ),
        ];
        let valid_batches = vec![make_batch::<TestValidBackend>(
            &valid_device,
            &[0, 0, 1, 1, 2, 2, 3, 3],
            &[0, 1, 1, 2, 2, 3, 3, 0],
            [2, 4],
        )];

        let training = tiny_training_hparams_with_epochs(4, None);
        let model_config = tiny_model_config();
        let devices = vec![primary_device];
        let env = TrainEnvironment {
            parallel_runtime: &parallel_runtime,
            parallel_config: &parallel_config,
            run_dir: &run_dir,
            run_name: "single-retention-smoke",
            backend_name: "cpu",
            training: &training,
            resume_checkpoint_epoch: None,
            model_config: &model_config,
            device: &primary_device,
            devices: &devices,
            train_dataset: None,
            valid_dataset: None,
            train_loader: Arc::new(StaticSequenceLoader::new(train_batches)),
            valid_loader: Arc::new(StaticSequenceLoader::new(valid_batches)),
            source_selection_dataset: None,
            summary_event_token_ids: None,
            neuron_scaling_slot: None,
            epochs: 4,
            total_steps: 8,
            valid_steps: 1,
        };
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            model_config.clone(),
            &primary_device,
        ));
        let optimizer = AdamWConfig::new()
            .with_weight_decay(0.0)
            .init::<TestBackend, LanguageTrainModel<TestBackend>>();

        let _trained =
            train_with_scheduler(&env, model, optimizer, 1e-3).expect("single-device train");

        let retained = retained_model_epochs(&run_dir);
        assert!(
            retained.contains(&3),
            "third epoch should be kept as recent"
        );
        assert!(retained.contains(&4), "last epoch should be kept as recent");
        assert!(
            retained.len() <= CHECKPOINT_KEEP_LAST + 1,
            "retention should keep the recent window plus at most one older best checkpoint"
        );
        assert!(
            retained.iter().all(|epoch| (1..=4).contains(epoch)),
            "retained epochs must come from completed checkpoints"
        );
    }

    #[cfg(feature = "ddp")]
    #[test]
    fn shard_bounds_evenly_distribute_remainder_steps() {
        assert_eq!(shard_bounds(5, 0, 2).expect("rank0"), (0, 3));
        assert_eq!(shard_bounds(5, 1, 2).expect("rank1"), (3, 5));
        assert!(shard_bounds(1, 1, 2).is_err());
    }

    #[cfg(feature = "ddp")]
    #[test]
    fn gradient_mean_matches_combined_batch_reference_in_module_order() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let config = tiny_model_config();
        let reference = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device));
        let combined_model = reference.clone();
        let shard_a_model = reference.clone();
        let shard_b_model = reference;

        let shard_a = make_batch::<TestBackend>(
            &device,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[1, 2, 3, 4, 5, 6, 7, 0],
            [2, 4],
        );
        let shard_b = make_batch::<TestBackend>(
            &device,
            &[7, 6, 5, 4, 3, 2, 1, 0],
            &[6, 5, 4, 3, 2, 1, 0, 7],
            [2, 4],
        );
        let combined = make_batch::<TestBackend>(
            &device,
            &[0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0],
            &[1, 2, 3, 4, 5, 6, 7, 0, 6, 5, 4, 3, 2, 1, 0, 7],
            [4, 4],
        );

        let combined_grads = flatten_gradients_in_module_order::<TestBackend, _>(
            &combined_model,
            burn_train::TrainStep::step(&combined_model, combined).grads,
        );
        let shard_a_grads = flatten_gradients_in_module_order::<TestBackend, _>(
            &shard_a_model,
            burn_train::TrainStep::step(&shard_a_model, shard_a).grads,
        );
        let shard_b_grads = flatten_gradients_in_module_order::<TestBackend, _>(
            &shard_b_model,
            burn_train::TrainStep::step(&shard_b_model, shard_b).grads,
        );

        assert_eq!(combined_grads.len(), shard_a_grads.len());
        assert_eq!(combined_grads.len(), shard_b_grads.len());

        let averaged_shards = shard_a_grads
            .iter()
            .zip(shard_b_grads.iter())
            .map(|(lhs, rhs)| (lhs + rhs) * 0.5)
            .collect::<Vec<_>>();

        let mean_abs = mean_abs_diff(&combined_grads, &averaged_shards);
        let combined_norm = l2_norm(&combined_grads);
        let averaged_norm = l2_norm(&averaged_shards);

        assert!(
            mean_abs <= 1.0e-5,
            "combined-batch reference and mean rank-local gradients drifted: mean_abs_diff={mean_abs}"
        );
        assert!(
            (combined_norm - averaged_norm).abs() <= 1.0e-5,
            "gradient norms drifted: combined_norm={combined_norm} averaged_norm={averaged_norm}"
        );
    }

    #[cfg(feature = "ddp")]
    #[test]
    fn train_with_collective_scheduler_runs_single_rank_and_writes_checkpoint() {
        let _lock = collective_test_lock().lock().expect("collective lock");
        reset_collective::<TestValidBackend>();

        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");
        let parallel_config = burn_dragon_train::ParallelConfig {
            mode: ParallelismKind::Ddp,
            world_size: 1,
            data: burn_dragon_train::ParallelDataConfig {
                size: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let parallel_runtime = ParallelRuntime {
            mode: ParallelismKind::Ddp,
            world_size: 1,
            global_rank: 0,
            local_rank: 0,
            data_parallel_size: 1,
            local_data_parallel_size: 1,
            tensor_parallel_size: 1,
            process_group_launch: false,
        };

        let primary_device = burn::tensor::Device::<TestBackend>::default();
        let valid_device = burn::tensor::Device::<TestValidBackend>::default();
        let train_batches = vec![
            make_batch::<TestBackend>(
                &primary_device,
                &[0, 1, 2, 3, 4, 5, 6, 7],
                &[1, 2, 3, 4, 5, 6, 7, 0],
                [2, 4],
            ),
            make_batch::<TestBackend>(
                &primary_device,
                &[7, 6, 5, 4, 3, 2, 1, 0],
                &[6, 5, 4, 3, 2, 1, 0, 7],
                [2, 4],
            ),
        ];
        let valid_batches = vec![make_batch::<TestValidBackend>(
            &valid_device,
            &[0, 0, 1, 1, 2, 2, 3, 3],
            &[0, 1, 1, 2, 2, 3, 3, 0],
            [2, 4],
        )];

        let training = tiny_training_hparams();
        let model_config = tiny_model_config();
        let devices = vec![primary_device.clone()];
        let env = TrainEnvironment {
            parallel_runtime: &parallel_runtime,
            parallel_config: &parallel_config,
            run_dir: &run_dir,
            run_name: "collective-single-rank",
            backend_name: "cpu",
            training: &training,
            resume_checkpoint_epoch: None,
            model_config: &model_config,
            device: &primary_device,
            devices: &devices,
            train_dataset: None,
            valid_dataset: None,
            train_loader: Arc::new(StaticSequenceLoader::new(train_batches)),
            valid_loader: Arc::new(StaticSequenceLoader::new(valid_batches)),
            source_selection_dataset: None,
            summary_event_token_ids: None,
            neuron_scaling_slot: None,
            epochs: 1,
            total_steps: 2,
            valid_steps: 1,
        };
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            model_config.clone(),
            &primary_device,
        ));
        let optimizer = AdamWConfig::new()
            .with_weight_decay(0.0)
            .init::<TestBackend, LanguageTrainModel<TestBackend>>();
        let collective =
            resolve_collective_config(&parallel_runtime, &parallel_config).expect("collective");

        let trained =
            train_with_collective_scheduler(&env, model, optimizer, 1e-3, collective, 0.into())
                .expect("collective train");
        let probe = make_batch::<TestValidBackend>(
            &valid_device,
            &[1, 2, 3, 4, 4, 3, 2, 1],
            &[2, 3, 4, 5, 3, 2, 1, 0],
            [2, 4],
        );
        let loss =
            language_model_loss::<TestValidBackend>(trained.forward(probe.inputs), probe.targets)
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("loss vec")[0];

        assert!(loss.is_finite());
        assert!(run_dir.join("checkpoint").join("model-1.bin").is_file());

        reset_collective::<TestValidBackend>();
    }

    #[cfg(feature = "ddp")]
    #[test]
    fn train_with_collective_scheduler_resumes_from_checkpoint_family() {
        let _lock = collective_test_lock().lock().expect("collective lock");
        reset_collective::<TestValidBackend>();

        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");
        let parallel_config = burn_dragon_train::ParallelConfig {
            mode: ParallelismKind::Ddp,
            world_size: 1,
            data: burn_dragon_train::ParallelDataConfig {
                size: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let parallel_runtime = ParallelRuntime {
            mode: ParallelismKind::Ddp,
            world_size: 1,
            global_rank: 0,
            local_rank: 0,
            data_parallel_size: 1,
            local_data_parallel_size: 1,
            tensor_parallel_size: 1,
            process_group_launch: false,
        };

        let primary_device = burn::tensor::Device::<TestBackend>::default();
        let valid_device = burn::tensor::Device::<TestValidBackend>::default();
        let train_loader: Arc<dyn DataLoader<TestBackend, SequenceBatch<TestBackend>>> =
            Arc::new(StaticSequenceLoader::new(vec![
                make_batch::<TestBackend>(
                    &primary_device,
                    &[0, 1, 2, 3, 4, 5, 6, 7],
                    &[1, 2, 3, 4, 5, 6, 7, 0],
                    [2, 4],
                ),
                make_batch::<TestBackend>(
                    &primary_device,
                    &[7, 6, 5, 4, 3, 2, 1, 0],
                    &[6, 5, 4, 3, 2, 1, 0, 7],
                    [2, 4],
                ),
            ]));
        let valid_loader: Arc<dyn DataLoader<TestValidBackend, SequenceBatch<TestValidBackend>>> =
            Arc::new(StaticSequenceLoader::new(vec![make_batch::<
                TestValidBackend,
            >(
                &valid_device,
                &[0, 0, 1, 1, 2, 2, 3, 3],
                &[0, 1, 1, 2, 2, 3, 3, 0],
                [2, 4],
            )]));
        let devices = vec![primary_device.clone()];
        let model_config = tiny_model_config();
        let collective =
            resolve_collective_config(&parallel_runtime, &parallel_config).expect("collective");

        let training_first = tiny_training_hparams_with_epochs(1, None);
        let env_first = TrainEnvironment {
            parallel_runtime: &parallel_runtime,
            parallel_config: &parallel_config,
            run_dir: &run_dir,
            run_name: "collective-resume",
            backend_name: "cpu",
            training: &training_first,
            resume_checkpoint_epoch: None,
            model_config: &model_config,
            device: &primary_device,
            devices: &devices,
            train_dataset: None,
            valid_dataset: None,
            train_loader: Arc::clone(&train_loader),
            valid_loader: Arc::clone(&valid_loader),
            source_selection_dataset: None,
            summary_event_token_ids: None,
            neuron_scaling_slot: None,
            epochs: 1,
            total_steps: 2,
            valid_steps: 1,
        };
        let model_first = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            model_config.clone(),
            &primary_device,
        ));
        let optimizer_first = AdamWConfig::new()
            .with_weight_decay(0.0)
            .init::<TestBackend, LanguageTrainModel<TestBackend>>();
        train_with_collective_scheduler(
            &env_first,
            model_first,
            optimizer_first,
            1e-3,
            collective.clone(),
            0.into(),
        )
        .expect("first collective train");
        assert!(run_dir.join("checkpoint").join("model-1.bin").is_file());

        reset_collective::<TestValidBackend>();

        let training_resume = tiny_training_hparams_with_epochs(2, Some(1));
        let env_resume = TrainEnvironment {
            parallel_runtime: &parallel_runtime,
            parallel_config: &parallel_config,
            run_dir: &run_dir,
            run_name: "collective-resume",
            backend_name: "cpu",
            training: &training_resume,
            resume_checkpoint_epoch: Some(1),
            model_config: &model_config,
            device: &primary_device,
            devices: &devices,
            train_dataset: None,
            valid_dataset: None,
            train_loader,
            valid_loader,
            source_selection_dataset: None,
            summary_event_token_ids: None,
            neuron_scaling_slot: None,
            epochs: 2,
            total_steps: 4,
            valid_steps: 1,
        };
        let model_resume = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            model_config.clone(),
            &primary_device,
        ));
        let optimizer_resume = AdamWConfig::new()
            .with_weight_decay(0.0)
            .init::<TestBackend, LanguageTrainModel<TestBackend>>();
        let resumed = train_with_collective_scheduler(
            &env_resume,
            model_resume,
            optimizer_resume,
            1e-3,
            collective,
            0.into(),
        )
        .expect("resumed collective train");

        let probe = make_batch::<TestValidBackend>(
            &valid_device,
            &[1, 2, 3, 4, 4, 3, 2, 1],
            &[2, 3, 4, 5, 3, 2, 1, 0],
            [2, 4],
        );
        let loss =
            language_model_loss::<TestValidBackend>(resumed.forward(probe.inputs), probe.targets)
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("loss vec")[0];

        assert!(loss.is_finite());
        assert!(run_dir.join("checkpoint").join("model-2.bin").is_file());

        reset_collective::<TestValidBackend>();
    }

    #[cfg(feature = "ddp")]
    #[test]
    fn pipeline_stage_surrogate_backward_matches_full_pipeline_gradients() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let mut config = tiny_model_config();
        config.n_layer = 2;
        let pipeline = burn_dragon_train::ParallelPipelineConfig {
            enabled: true,
            stage_count: 2,
            virtual_stages_per_rank: 1,
            schedule: burn_dragon_train::PipelineScheduleKind::Interleaved1f1b,
            microbatches: 2,
            ..Default::default()
        };
        let plan = build_pipeline_plan(config.n_layer, &pipeline).expect("plan");
        let reference_model =
            LanguageTrainModel::new(DragonModel::<TestBackend>::new(config.clone(), &device))
                .with_pipeline_plan(Some(plan.clone()));
        let split_model = reference_model.clone();

        let batch = make_batch::<TestBackend>(
            &device,
            &[0, 1, 2, 3, 7, 6, 5, 4],
            &[1, 2, 3, 4, 6, 5, 4, 3],
            [2, 4],
        );
        let reference_grads = flatten_gradients_in_module_order::<TestBackend, _>(
            &reference_model,
            burn_train::TrainStep::step(&reference_model, batch.clone()).grads,
        );
        let split_grads = stage_split_surrogate_gradients(split_model, &plan, batch);
        let mean_abs = mean_abs_diff(&reference_grads, &split_grads);
        let reference_norm = l2_norm(&reference_grads);
        let split_norm = l2_norm(&split_grads);

        assert!(
            mean_abs <= 1.0e-5,
            "surrogate split pipeline gradients drifted from full pipeline reference: mean_abs_diff={mean_abs}"
        );
        assert!(
            (reference_norm - split_norm).abs() <= 1.0e-5,
            "split pipeline gradient norm drifted from reference: reference_norm={reference_norm} split_norm={split_norm}"
        );
    }

    #[cfg(feature = "ddp")]
    #[test]
    fn pipeline_stage_surrogate_mean_across_replicas_matches_full_batch_gradients() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let mut config = tiny_model_config();
        config.n_layer = 2;
        let pipeline = burn_dragon_train::ParallelPipelineConfig {
            enabled: true,
            stage_count: 2,
            virtual_stages_per_rank: 1,
            schedule: burn_dragon_train::PipelineScheduleKind::Interleaved1f1b,
            microbatches: 2,
            ..Default::default()
        };
        let plan = build_pipeline_plan(config.n_layer, &pipeline).expect("plan");
        let reference_model =
            LanguageTrainModel::new(DragonModel::<TestBackend>::new(config.clone(), &device))
                .with_pipeline_plan(Some(plan.clone()));

        let replica_a = make_batch::<TestBackend>(
            &device,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[1, 2, 3, 4, 5, 6, 7, 0],
            [2, 4],
        );
        let replica_b = make_batch::<TestBackend>(
            &device,
            &[7, 6, 5, 4, 3, 2, 1, 0],
            &[6, 5, 4, 3, 2, 1, 0, 7],
            [2, 4],
        );
        let combined = make_batch::<TestBackend>(
            &device,
            &[0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0],
            &[1, 2, 3, 4, 5, 6, 7, 0, 6, 5, 4, 3, 2, 1, 0, 7],
            [4, 4],
        );

        let combined_grads = flatten_gradients_in_module_order::<TestBackend, _>(
            &reference_model,
            burn_train::TrainStep::step(&reference_model, combined).grads,
        );
        let replica_a_grads =
            stage_split_surrogate_gradients(reference_model.clone(), &plan, replica_a);
        let replica_b_grads =
            stage_split_surrogate_gradients(reference_model.clone(), &plan, replica_b);
        let averaged_grads = replica_a_grads
            .iter()
            .zip(replica_b_grads.iter())
            .map(|(lhs, rhs)| (lhs + rhs) * 0.5)
            .collect::<Vec<_>>();

        let mean_abs = mean_abs_diff(&combined_grads, &averaged_grads);
        let combined_norm = l2_norm(&combined_grads);
        let averaged_norm = l2_norm(&averaged_grads);

        assert!(
            mean_abs <= 1.0e-5,
            "replica-averaged split pipeline gradients drifted from combined-batch reference: mean_abs_diff={mean_abs}"
        );
        assert!(
            (combined_norm - averaged_norm).abs() <= 1.0e-5,
            "replica-averaged split pipeline gradient norm drifted from combined-batch reference: combined_norm={combined_norm} averaged_norm={averaged_norm}"
        );
    }

    #[cfg(feature = "ddp")]
    #[test]
    fn process_group_peer_id_uses_global_rank() {
        let runtime = ParallelRuntime {
            mode: ParallelismKind::Ddp,
            world_size: 4,
            global_rank: 3,
            local_rank: 1,
            data_parallel_size: 4,
            local_data_parallel_size: 1,
            tensor_parallel_size: 1,
            process_group_launch: true,
        };

        assert_eq!(process_group_peer_id(&runtime), 3usize.into());
    }

    #[cfg(feature = "ddp")]
    #[test]
    fn process_group_data_shard_uses_data_parallel_rank_when_pipeline_enabled() {
        let runtime = ParallelRuntime {
            mode: ParallelismKind::Ddp,
            world_size: 4,
            global_rank: 3,
            local_rank: 1,
            data_parallel_size: 2,
            local_data_parallel_size: 1,
            tensor_parallel_size: 1,
            process_group_launch: true,
        };
        let config = burn_dragon_train::ParallelConfig {
            mode: ParallelismKind::Ddp,
            world_size: 4,
            data: burn_dragon_train::ParallelDataConfig {
                size: 2,
                ..Default::default()
            },
            pipeline: burn_dragon_train::ParallelPipelineConfig {
                enabled: true,
                stage_count: 2,
                virtual_stages_per_rank: 1,
                ..Default::default()
            },
            ..Default::default()
        };

        let (shard_index, shard_count, assignment, layout) =
            process_group_data_shard(&runtime, &config).expect("pipeline shard");

        assert_eq!(shard_index, 1);
        assert_eq!(shard_count, 2);
        let assignment = assignment.expect("rank assignment");
        let layout = layout.expect("layout");
        assert_eq!(assignment.pipeline_stage_id, 1);
        assert_eq!(assignment.data_parallel_rank, 1);
        assert_eq!(assignment.pipeline_group_ranks, vec![2, 3]);
        assert_eq!(assignment.data_parallel_group_ranks, vec![1, 3]);
        assert_eq!(
            layout.summary(),
            "pipeline_layout=replica_major stage_count=2 virtual_stages_per_rank=1 data_parallel_size=2 world_size=4"
        );
    }

    #[cfg(feature = "ddp")]
    #[test]
    fn fresh_models_use_random_param_ids_but_stable_module_traversal_shapes() {
        #[derive(Default)]
        struct ShapeCollector {
            shapes: Vec<Vec<usize>>,
        }

        impl<B: BackendTrait> burn::module::ModuleVisitor<B> for ShapeCollector {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
                self.shapes.push(param.val().shape().dims::<D>().into());
            }
        }

        let device = burn::tensor::Device::<TestBackend>::default();
        let config = tiny_model_config();
        let model_a =
            LanguageTrainModel::new(DragonModel::<TestBackend>::new(config.clone(), &device));
        let model_b = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device));

        let ids_a = list_param_ids(&model_a);
        let ids_b = list_param_ids(&model_b);
        let mut shapes_a = ShapeCollector::default();
        let mut shapes_b = ShapeCollector::default();
        model_a.visit(&mut shapes_a);
        model_b.visit(&mut shapes_b);

        assert_eq!(ids_a.len(), ids_b.len());
        assert_ne!(
            ids_a, ids_b,
            "fresh models should not rely on matching ParamIds"
        );
        assert_eq!(shapes_a.shapes, shapes_b.shapes);
    }
}
