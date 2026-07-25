use std::sync::{Arc, Mutex};

use burn_dragon_train::train::events::{
    CapacityPlateauDetected, CapacityScalingPolicy, ModelScaleRequest,
};
use burn_ecs::bevy_ecs;
use burn_ecs::prelude::{
    App, Component, IntoScheduleConfigs, MessageReader, MessageWriter, Plugin, Query, Res,
    TrainingRunRegistry, TrainingSet, Update,
};

use crate::config::train::{NeuronScalingConfig, NeuronScalingGrowth};

#[derive(Clone, Debug, Default)]
pub struct NeuronScaleRequestSlot {
    inner: Arc<Mutex<Option<ModelScaleRequest>>>,
}

impl NeuronScaleRequestSlot {
    pub fn take(&self) -> Option<ModelScaleRequest> {
        self.inner.lock().ok().and_then(|mut guard| guard.take())
    }

    pub(crate) fn set_if_empty(&self, request: ModelScaleRequest) -> bool {
        let Ok(mut guard) = self.inner.lock() else {
            return false;
        };
        if guard.is_some() {
            return false;
        }
        *guard = Some(request);
        true
    }
}

#[derive(Clone, Component)]
pub(crate) struct DragonNeuronScalingState {
    config: NeuronScalingConfig,
    request_slot: NeuronScaleRequestSlot,
}

impl DragonNeuronScalingState {
    pub(crate) fn new(config: NeuronScalingConfig, request_slot: NeuronScaleRequestSlot) -> Self {
        Self {
            config,
            request_slot,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DragonNeuronScalingPlugin;

impl Plugin for DragonNeuronScalingPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(burn_dragon_train::train::events::CapacityPlateauPlugin)
            .add_systems(
                Update,
                request_neuron_scale_on_capacity_plateau.in_set(TrainingSet::Control),
            );
    }
}

pub fn capacity_policy_from_neuron_scaling(config: &NeuronScalingConfig) -> CapacityScalingPolicy {
    CapacityScalingPolicy {
        enabled: config.enabled,
        capacity_patience_epochs: config.capacity_patience_epochs,
        min_steps_between_scales: config.min_steps_between_scales,
        max_scale_events: config.max_scale_events,
        require_source_selection: config.require_live_source_selection,
        ..CapacityScalingPolicy::default()
    }
}

pub fn next_latent_total(
    current_latent_total: usize,
    config: &NeuronScalingConfig,
) -> Option<usize> {
    if !config.enabled || current_latent_total >= config.max_latent_total {
        return None;
    }
    let next = match config.growth {
        NeuronScalingGrowth::Double => current_latent_total.saturating_mul(2),
    };
    Some(next.min(config.max_latent_total)).filter(|next| *next > current_latent_total)
}

fn request_neuron_scale_on_capacity_plateau(
    mut plateaus: MessageReader<CapacityPlateauDetected>,
    registry: Res<TrainingRunRegistry>,
    scaling_runs: Query<&DragonNeuronScalingState>,
    mut requests: MessageWriter<ModelScaleRequest>,
) {
    for plateau in plateaus.read() {
        let Some(scaling) = registry.get_query(&plateau.run_id, &scaling_runs) else {
            continue;
        };
        let Some(target) = next_latent_total(plateau.current_capacity_units, &scaling.config)
        else {
            continue;
        };
        let request = ModelScaleRequest {
            run_id: plateau.run_id.clone(),
            epoch: Some(plateau.epoch),
            absolute_step: Some(plateau.absolute_step),
            from_capacity_units: plateau.current_capacity_units,
            to_capacity_units: target,
            reason: plateau.message.clone(),
        };
        if !scaling.request_slot.set_if_empty(request.clone()) {
            continue;
        }
        requests.write(request);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ecs::prelude::{App, TrainingAppExt, TrainingPlugins, TrainingRunConfig};

    #[test]
    fn next_latent_total_doubles_until_cap() {
        let config = NeuronScalingConfig {
            enabled: true,
            max_latent_total: 8192,
            ..NeuronScalingConfig::default()
        };
        assert_eq!(next_latent_total(1024, &config), Some(2048));
        assert_eq!(next_latent_total(4096, &config), Some(8192));
        assert_eq!(next_latent_total(8192, &config), None);
    }

    #[test]
    fn capacity_policy_tracks_neuron_scaling_config() {
        let config = NeuronScalingConfig {
            enabled: true,
            min_steps_between_scales: 123,
            max_scale_events: 3,
            capacity_patience_epochs: 4,
            require_live_source_selection: false,
            ..NeuronScalingConfig::default()
        };

        let policy = capacity_policy_from_neuron_scaling(&config);
        assert!(policy.enabled);
        assert_eq!(policy.min_steps_between_scales, 123);
        assert_eq!(policy.max_scale_events, 3);
        assert_eq!(policy.capacity_patience_epochs, 4);
        assert!(!policy.require_source_selection);
    }

    #[test]
    fn scale_requests_are_isolated_by_training_run() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut app = App::new();
        app.add_plugins(TrainingPlugins)
            .add_plugins(DragonNeuronScalingPlugin);
        let run_a = app
            .try_add_training_run(TrainingRunConfig::new(
                "run-a",
                "run-a",
                dir.path().join("a"),
                1,
            ))
            .expect("run a");
        let run_b = app
            .try_add_training_run(TrainingRunConfig::new(
                "run-b",
                "run-b",
                dir.path().join("b"),
                1,
            ))
            .expect("run b");
        let config = NeuronScalingConfig {
            enabled: true,
            max_latent_total: 8192,
            ..NeuronScalingConfig::default()
        };
        let slot_a = NeuronScaleRequestSlot::default();
        let slot_b = NeuronScaleRequestSlot::default();
        app.world_mut()
            .entity_mut(run_a)
            .insert(DragonNeuronScalingState::new(
                config.clone(),
                slot_a.clone(),
            ));
        app.world_mut()
            .entity_mut(run_b)
            .insert(DragonNeuronScalingState::new(config, slot_b.clone()));
        app.world_mut().write_message(CapacityPlateauDetected {
            run_id: "run-a".into(),
            epoch: 4,
            absolute_step: 400,
            current_capacity_units: 1024,
            max_capacity_units: 8192,
            best_valid_epoch: 1,
            best_valid_loss: 1.0,
            current_valid_loss: 1.0,
            stagnant_epochs: 3,
            mean_learning_progress: Some(0.0),
            entropy_bits: Some(4.0),
            hash_noise_probability: Some(0.0),
            message: "test capacity plateau".into(),
        });

        app.update();

        let request = slot_a.take().expect("run-a scale request");
        assert_eq!(request.to_capacity_units, 2048);
        assert!(slot_b.take().is_none());
    }
}
