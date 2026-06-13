use std::sync::Arc;

use burn_ecs::prelude::{
    App, IntoScheduleConfigs, MessageReader, MessageWriter, Plugin, Res, SourceSelectionSample,
    TrainingMetricSample, TrainingMetricSplit, TrainingSet, Update,
};

use crate::dataset::Dataset;

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
