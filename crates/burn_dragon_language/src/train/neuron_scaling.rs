use std::sync::{Arc, Mutex};

use burn_dragon_train::train::events::{
    CapacityPlateauDetected, CapacityScalingPolicy, ModelCapacityState, ModelScaleRequest,
};
use burn_ecs::prelude::{
    App, IntoScheduleConfigs, MessageReader, MessageWriter, Plugin, Res, Resource, TrainingSet,
    Update,
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

#[derive(Clone)]
struct DragonNeuronScalingResource {
    config: NeuronScalingConfig,
    request_slot: NeuronScaleRequestSlot,
}

impl Resource for DragonNeuronScalingResource {}

pub struct DragonNeuronScalingPlugin {
    resource: DragonNeuronScalingResource,
    initial_capacity: ModelCapacityState,
}

impl DragonNeuronScalingPlugin {
    pub fn new(
        config: NeuronScalingConfig,
        current_latent_total: usize,
        request_slot: NeuronScaleRequestSlot,
    ) -> Self {
        let max_latent_total = config.max_latent_total.max(current_latent_total);
        Self {
            resource: DragonNeuronScalingResource {
                config,
                request_slot,
            },
            initial_capacity: ModelCapacityState::new(current_latent_total, max_latent_total),
        }
    }
}

impl Plugin for DragonNeuronScalingPlugin {
    fn build(&self, app: &mut App) {
        let policy = capacity_policy_from_neuron_scaling(&self.resource.config);
        app.add_plugins(
            burn_dragon_train::train::events::CapacityPlateauPlugin::new(
                policy,
                self.initial_capacity.clone(),
            ),
        )
        .insert_resource(self.resource.clone())
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
    scaling: Res<DragonNeuronScalingResource>,
    mut requests: MessageWriter<ModelScaleRequest>,
) {
    for plateau in plateaus.read() {
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
}
