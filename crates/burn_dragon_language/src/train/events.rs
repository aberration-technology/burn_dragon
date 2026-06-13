use std::sync::Arc;

use anyhow::Result;
use burn_dragon_train::train::events::{
    BurnInterrupterControl, TrainingAppBuilder, TrainingAppConfig, TrainingEventMetricLogger,
    TrainingRunContext,
};
use burn_ecs::prelude::{
    App, IntoScheduleConfigs, MessageReader, MessageWriter, Plugin, Res, SourceSelectionSample,
    TrainingMetricSample, TrainingMetricSplit, TrainingSet, Update,
};

use crate::config::TrainingHyperparameters;
use crate::dataset::Dataset;
use crate::train::neuron_scaling::{DragonNeuronScalingPlugin, NeuronScaleRequestSlot};

#[derive(Clone)]
pub struct RuliadSourceSelectionResource {
    dataset: Arc<Dataset>,
    source_selection_every_steps: usize,
}

impl burn_ecs::prelude::Resource for RuliadSourceSelectionResource {}

impl RuliadSourceSelectionResource {
    pub fn new(dataset: Arc<Dataset>, source_selection_every_steps: usize) -> Self {
        Self {
            dataset,
            source_selection_every_steps: source_selection_every_steps.max(1),
        }
    }
}

pub struct TrainingEventHandles {
    pub interrupter: burn_train::Interrupter,
    pub metric_logger: TrainingEventMetricLogger,
}

pub fn train_loss_metric_frequency(
    training: &TrainingHyperparameters,
    source_selection_dataset: Option<&Arc<Dataset>>,
) -> usize {
    if source_selection_dataset
        .as_ref()
        .is_some_and(|dataset| dataset.uses_live_source_selection())
    {
        training.events.source_selection_every_steps.max(1)
    } else {
        training.log_frequency.max(1)
    }
}

pub fn build_training_event_handles(
    run_name: &str,
    run_dir: &std::path::Path,
    steps_per_epoch: usize,
    training: &TrainingHyperparameters,
    source_selection_dataset: Option<Arc<Dataset>>,
    neuron_scaling_slot: Option<(usize, NeuronScaleRequestSlot)>,
) -> Result<TrainingEventHandles> {
    let interrupter = burn_train::Interrupter::new();
    let mut event_app = TrainingAppBuilder::new(TrainingAppConfig {
        run: TrainingRunContext::new(run_name, run_name, run_dir, steps_per_epoch),
        events: training.events.clone(),
        gates: training.gates.clone(),
        bus: Default::default(),
    })
    .with_control(BurnInterrupterControl::new(interrupter.clone()));

    if let Some(dataset) =
        source_selection_dataset.filter(|dataset| dataset.uses_live_source_selection())
    {
        let source_selection_every_steps = training.events.source_selection_every_steps;
        event_app = event_app.with_plugin(RuliadSourceSelectionTelemetryPlugin::new(
            dataset,
            source_selection_every_steps,
        ));
    }

    if training.neuron_scaling.enabled
        && let Some((current_latent_total, request_slot)) = neuron_scaling_slot
    {
        event_app = event_app.with_plugin(DragonNeuronScalingPlugin::new(
            training.neuron_scaling.clone(),
            current_latent_total,
            request_slot,
        ));
    }

    let event_thread = event_app.spawn_threaded()?;
    let metric_logger =
        TrainingEventMetricLogger::with_thread(event_thread, run_name, steps_per_epoch);
    Ok(TrainingEventHandles {
        interrupter,
        metric_logger,
    })
}

pub struct RuliadSourceSelectionTelemetryPlugin {
    source_selection: RuliadSourceSelectionResource,
}

impl RuliadSourceSelectionTelemetryPlugin {
    pub fn new(dataset: Arc<Dataset>, source_selection_every_steps: usize) -> Self {
        Self {
            source_selection: RuliadSourceSelectionResource::new(
                dataset,
                source_selection_every_steps,
            ),
        }
    }
}

impl Plugin for RuliadSourceSelectionTelemetryPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(self.source_selection.clone())
            .add_systems(
                Update,
                record_ruliad_source_selection_from_loss.in_set(TrainingSet::Telemetry),
            );
    }
}

fn record_ruliad_source_selection_from_loss(
    mut metrics: MessageReader<TrainingMetricSample>,
    source_selection: Res<RuliadSourceSelectionResource>,
    mut source_selection_events: MessageWriter<SourceSelectionSample>,
) {
    for sample in metrics.read() {
        if sample.split != TrainingMetricSplit::Train || sample.name != "Loss" {
            continue;
        }
        if sample.absolute_step % source_selection.source_selection_every_steps != 0 {
            continue;
        }
        let snapshot = source_selection
            .dataset
            .record_source_selection_loss(sample.absolute_step, sample.value as f32)
            .or_else(|| source_selection.dataset.source_selection_snapshot());
        let Some(snapshot) = snapshot else {
            continue;
        };
        source_selection_events.write(SourceSelectionSample {
            run_id: sample.run_id.clone(),
            absolute_step: sample.absolute_step,
            loss: Some(sample.value as f32),
            entropy_bits: snapshot.sampler_entropy_bits as f64,
            hash_noise_probability: snapshot.hash_noise_probability as f64,
            mean_loss: snapshot.mean_loss as f64,
            mean_learning_progress: snapshot.mean_learning_progress as f64,
            verifier_failures: snapshot.verifier_failures as u64,
        });
    }
}
