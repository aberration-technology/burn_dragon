use std::sync::Arc;

use anyhow::Result;
use burn_dragon_train::train::events::{
    BurnInterrupterControl, TrainingAppBuilder, TrainingAppConfig, TrainingEventMetricLogger,
    TrainingRunContext,
};
use burn_ecs::prelude::{
    App, IntoScheduleConfigs, MessageReader, MessageWriter, Plugin, Res,
    SourceSelectionBucketMetric, SourceSelectionGroupMetric, SourceSelectionSample,
    TrainingMetricSample, TrainingMetricSplit, TrainingSet, Update,
};

use crate::config::TrainingHyperparameters;
use crate::dataset::Dataset;
use crate::train::dynamics::{DragonDynamicsControlPlugin, DragonDynamicsControlSlot};
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
    dynamics_control_slot: Option<DragonDynamicsControlSlot>,
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

    if training.dynamics.enabled {
        event_app = event_app.with_plugin(
            burn_dragon_train::train::events::DynamicsEquilibriumPlugin::new(
                training.dynamics.clone(),
            ),
        );
        if let Some(slot) = dynamics_control_slot {
            event_app = event_app.with_plugin(DragonDynamicsControlPlugin::new(slot));
        }
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
        if sample.split != TrainingMetricSplit::Train
            || (sample.name != "Loss" && sample.name != "Stream Warm Loss")
        {
            continue;
        }
        if sample.absolute_step % source_selection.source_selection_every_steps != 0 {
            continue;
        }
        let recorded_snapshot = source_selection
            .dataset
            .record_source_selection_loss(sample.absolute_step, sample.value as f32);
        let loss = recorded_snapshot.as_ref().map(|_| sample.value as f32);
        let snapshot =
            recorded_snapshot.or_else(|| source_selection.dataset.source_selection_snapshot());
        let Some(snapshot) = snapshot else {
            continue;
        };
        source_selection_events.write(source_selection_sample_from_snapshot(
            sample.run_id.clone(),
            sample.absolute_step,
            loss,
            &snapshot,
        ));
    }
}

pub(crate) fn source_selection_sample_from_snapshot(
    run_id: String,
    absolute_step: usize,
    loss: Option<f32>,
    snapshot: &burn_dragon_universality::RuliadMetricSnapshot,
) -> SourceSelectionSample {
    SourceSelectionSample {
        run_id,
        absolute_step,
        loss,
        entropy_bits: snapshot.sampler_entropy_bits as f64,
        active_candidate_count: snapshot.active_candidate_count,
        active_max_entropy_bits: snapshot.active_max_entropy_bits as f64,
        normalized_entropy: snapshot.normalized_sampler_entropy as f64,
        hash_noise_probability: snapshot.hash_noise_probability as f64,
        mean_loss: snapshot.mean_loss as f64,
        mean_learning_progress: snapshot.mean_learning_progress as f64,
        frontier_loss: snapshot.frontier_loss as f64,
        target_loss: snapshot.target_loss as f64,
        target_difficulty_score: snapshot.target_difficulty_score as f64,
        max_difficulty_level: snapshot.max_difficulty_level,
        materialized_frontier_edge: snapshot.max_difficulty_level,
        mean_difficulty_level: snapshot.mean_difficulty_level as f64,
        normalized_difficulty_score: snapshot.normalized_difficulty_score as f64,
        max_difficulty_probability: snapshot.max_difficulty_probability as f64,
        mastered_probability: snapshot.mastered_probability as f64,
        capability_feedback_probability: snapshot.capability_feedback_probability as f64,
        capability_verifier_ema: snapshot.capability_verifier_ema as f64,
        capability_completion_health_ema: snapshot.capability_completion_health_ema as f64,
        capability_schema_wrong_ema: snapshot.capability_schema_wrong_ema as f64,
        capability_malformed_ema: snapshot.capability_malformed_ema as f64,
        capability_missing_ema: snapshot.capability_missing_ema as f64,
        capability_lagging_probability: snapshot.capability_lagging_probability as f64,
        frontier_extension_count: snapshot.frontier_extension_count,
        frontier_saturated: snapshot.frontier_saturated,
        unbounded_frontier: snapshot.frontier_unbounded,
        top_buckets: snapshot
            .top_buckets
            .iter()
            .map(|bucket| SourceSelectionBucketMetric {
                label: bucket.label.clone(),
                family: bucket.family.clone(),
                task_kind: bucket.task_kind.clone(),
                difficulty_level: bucket.difficulty_level,
                probability: bucket.probability as f64,
                loss_ema: bucket.loss_ema as f64,
                previous_loss_ema: bucket.previous_loss_ema as f64,
                learning_progress: bucket.learning_progress as f64,
                mastered: bucket.mastered,
                capability_feedback_count: bucket.capability_feedback_count,
                capability_verifier_ema: bucket.capability_verifier_ema as f64,
                capability_completion_health_ema: bucket.capability_completion_health_ema as f64,
                capability_schema_wrong_ema: bucket.capability_schema_wrong_ema as f64,
                capability_malformed_ema: bucket.capability_malformed_ema as f64,
                capability_missing_ema: bucket.capability_missing_ema as f64,
                capability_lagging: bucket.capability_lagging,
            })
            .collect(),
        difficulty_buckets: snapshot
            .difficulty_buckets
            .iter()
            .map(source_selection_group_metric)
            .collect(),
        family_buckets: snapshot
            .family_buckets
            .iter()
            .map(source_selection_group_metric)
            .collect(),
        task_buckets: snapshot
            .task_buckets
            .iter()
            .map(source_selection_group_metric)
            .collect(),
        contract_buckets: snapshot
            .contract_buckets
            .iter()
            .map(source_selection_group_metric)
            .collect(),
        verifier_failures: snapshot.verifier_failures as u64,
    }
}

fn source_selection_group_metric(
    group: &burn_dragon_universality::RuliadGroupMetric,
) -> SourceSelectionGroupMetric {
    SourceSelectionGroupMetric {
        label: group.label.clone(),
        candidate_count: group.candidate_count,
        probability: group.probability as f64,
        mean_loss: group.mean_loss as f64,
        learning_progress: group.learning_progress as f64,
        mastered_probability: group.mastered_probability as f64,
        mean_difficulty_level: group.mean_difficulty_level as f64,
        capability_feedback_probability: group.capability_feedback_probability as f64,
        capability_verifier_ema: group.capability_verifier_ema as f64,
        capability_completion_health_ema: group.capability_completion_health_ema as f64,
        capability_schema_wrong_ema: group.capability_schema_wrong_ema as f64,
        capability_malformed_ema: group.capability_malformed_ema as f64,
        capability_missing_ema: group.capability_missing_ema as f64,
        capability_lagging_probability: group.capability_lagging_probability as f64,
    }
}
