use std::sync::Arc;

use anyhow::Result;
use burn_dragon_train::train::events::{
    BurnInterrupterControl, DynamicsEquilibriumPlugin, ModelCapacityConfig, ModelCapacityState,
    TrainingEventBusConfig, TrainingEventMetricLogger, TrainingRunContext, TrainingRunOptions,
    TrainingRuntimeThread,
};
use burn_ecs::bevy_ecs;
use burn_ecs::prelude::{
    App, Component, IntoScheduleConfigs, MessageReader, MessageWriter, Plugin, Query, Res,
    SourceSelectionBucketMetric, SourceSelectionCapabilityCoverageMetric,
    SourceSelectionGroupMetric, SourceSelectionSample, TrainingAppExt, TrainingMetricSample,
    TrainingMetricSplit, TrainingPlugins, TrainingRunId, TrainingRunRegistry, TrainingSet, Update,
};

use crate::config::TrainingHyperparameters;
use crate::dataset::Dataset;
use crate::train::dynamics::{DragonDynamicsControlPlugin, DragonDynamicsControlSlot};
use crate::train::neuron_scaling::{DragonNeuronScalingPlugin, NeuronScaleRequestSlot};

#[derive(Clone, Component)]
pub struct RuliadSourceSelectionConfig {
    dataset: Arc<Dataset>,
    source_selection_every_steps: usize,
}

impl RuliadSourceSelectionConfig {
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
    let control = BurnInterrupterControl::new(interrupter.clone());
    let run = TrainingRunContext::new(run_name, run_name, run_dir, steps_per_epoch);
    let source_selection = source_selection_dataset
        .filter(|dataset| dataset.uses_live_source_selection())
        .map(|dataset| {
            RuliadSourceSelectionConfig::new(dataset, training.events.source_selection_every_steps)
        });
    let neuron_scaling = (training
        .neuron_scaling
        .enabled
        .then_some(training.neuron_scaling.clone()))
    .zip(neuron_scaling_slot);
    let capacity = neuron_scaling
        .as_ref()
        .map(|(config, (current_latent_total, _))| ModelCapacityConfig {
            policy: crate::train::neuron_scaling::capacity_policy_from_neuron_scaling(config),
            capacity: Some(ModelCapacityState::new(
                *current_latent_total,
                config.max_latent_total.max(*current_latent_total),
            )),
        });
    let options = TrainingRunOptions {
        sinks: training.events.clone(),
        gates: training.gates.clone(),
        dynamics: training
            .dynamics
            .enabled
            .then_some(training.dynamics.clone()),
        capacity,
    };
    let dynamics_enabled = training.dynamics.enabled;
    let event_thread = TrainingRuntimeThread::spawn(
        move || {
            let mut app = App::new();
            app.add_plugins(TrainingPlugins)
                .insert_training_control(control);
            if source_selection.is_some() {
                app.add_plugins(RuliadSourceSelectionTelemetryPlugin);
            }
            if neuron_scaling.is_some() {
                app.add_plugins(DragonNeuronScalingPlugin);
            }
            if dynamics_enabled {
                app.add_plugins(DynamicsEquilibriumPlugin);
                if dynamics_control_slot.is_some() {
                    app.add_plugins(DragonDynamicsControlPlugin);
                }
            }
            let run_entity = app.try_add_training_run_with(run, options)?;
            if let Some(source_selection) = source_selection {
                app.world_mut()
                    .entity_mut(run_entity)
                    .insert(source_selection);
            }
            if let Some((config, (_, request_slot))) = neuron_scaling {
                app.world_mut().entity_mut(run_entity).insert(
                    crate::train::neuron_scaling::DragonNeuronScalingState::new(
                        config,
                        request_slot,
                    ),
                );
            }
            if let Some(slot) = dynamics_control_slot {
                app.world_mut().entity_mut(run_entity).insert(slot);
            }
            Ok(app)
        },
        TrainingEventBusConfig::default(),
    )?;
    let metric_logger =
        TrainingEventMetricLogger::with_thread(event_thread, run_name, steps_per_epoch);
    Ok(TrainingEventHandles {
        interrupter,
        metric_logger,
    })
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RuliadSourceSelectionTelemetryPlugin;

impl Plugin for RuliadSourceSelectionTelemetryPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            record_ruliad_source_selection_from_loss.in_set(TrainingSet::Telemetry),
        );
    }
}

fn record_ruliad_source_selection_from_loss(
    mut metrics: MessageReader<TrainingMetricSample>,
    registry: Res<TrainingRunRegistry>,
    source_selection_runs: Query<&RuliadSourceSelectionConfig>,
    mut source_selection_events: MessageWriter<SourceSelectionSample>,
) {
    for sample in metrics.read() {
        if sample.split != TrainingMetricSplit::Train
            || (sample.name != "Loss" && sample.name != "Stream Warm Loss")
        {
            continue;
        }
        let Some(source_selection) = registry.get_query(&sample.run_id, &source_selection_runs)
        else {
            continue;
        };
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
    run_id: impl Into<TrainingRunId>,
    absolute_step: usize,
    loss: Option<f32>,
    snapshot: &burn_dragon_universality::RuliadMetricSnapshot,
) -> SourceSelectionSample {
    let run_id = run_id.into();
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
        capability_frontier_allowed_max_difficulty: snapshot
            .capability_frontier_allowed_max_difficulty,
        capability_frontier_coverage: snapshot
            .capability_frontier_coverage
            .iter()
            .map(source_selection_capability_coverage_metric)
            .collect(),
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

fn source_selection_capability_coverage_metric(
    coverage: &burn_dragon_universality::RuliadCapabilityCoverageMetric,
) -> SourceSelectionCapabilityCoverageMetric {
    SourceSelectionCapabilityCoverageMetric {
        difficulty_level: coverage.difficulty_level,
        candidate_coverage: coverage.candidate_coverage as f64,
        family_coverage: coverage.family_coverage as f64,
        task_coverage: coverage.task_coverage as f64,
        contract_coverage: coverage.contract_coverage as f64,
        observed_items: coverage.observed_items,
        mastered: coverage.mastered,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_frontier_coverage_survives_the_ecs_event_boundary() {
        let metric = source_selection_capability_coverage_metric(
            &burn_dragon_universality::RuliadCapabilityCoverageMetric {
                difficulty_level: 7,
                candidate_coverage: 0.75,
                family_coverage: 1.0,
                task_coverage: 0.5,
                contract_coverage: 0.25,
                observed_items: 128,
                mastered: false,
            },
        );

        assert_eq!(metric.difficulty_level, 7);
        assert_eq!(metric.observed_items, 128);
        assert_eq!(metric.candidate_coverage, 0.75);
        assert_eq!(metric.family_coverage, 1.0);
        assert_eq!(metric.task_coverage, 0.5);
        assert_eq!(metric.contract_coverage, 0.25);
        assert!(!metric.mastered);
    }
}
