use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow};
use burn_dragon_core::DragonConfig;
use burn_dragon_train::train::events::{
    CapacityPlateauDetected, CapacityScalingPolicy, ModelCapacityState, ModelScaleRequest,
    TrainingControlResource,
};
use burn_ecs::prelude::{
    App, IntoScheduleConfigs, MessageReader, MessageWriter, Plugin, Res, Resource, TrainingSet,
    Update,
};
use serde::{Deserialize, Serialize};

use crate::config::train::{
    NeuronScalingBatchFitConfig, NeuronScalingConfig, NeuronScalingGrowth,
    NeuronScalingStabilizationConfig,
};

pub const NEURON_SCALE_REQUEST_ARTIFACT: &str = "events/neuron-scale-request.json";

#[derive(Clone, Debug, Default)]
pub struct NeuronScaleRequestSlot {
    inner: Arc<Mutex<Option<ModelScaleRequest>>>,
}

impl NeuronScaleRequestSlot {
    pub fn take(&self) -> Option<ModelScaleRequest> {
        self.inner.lock().ok().and_then(|mut guard| guard.take())
    }

    fn set_if_empty(&self, request: ModelScaleRequest) -> bool {
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

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct NeuronScaleModelTarget {
    pub current_latent_total: usize,
    pub target_latent_total: usize,
    pub target_mlp_internal_dim_multiplier: usize,
    pub n_embd: usize,
    pub n_head: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct NeuronScaleRequestArtifact {
    pub request: ModelScaleRequest,
    pub target: NeuronScaleModelTarget,
    pub max_latent_total: usize,
    pub max_scale_events: usize,
    pub batch_fit: NeuronScalingBatchFitConfig,
    pub stabilization: NeuronScalingStabilizationConfig,
    pub resume_checkpoint_epoch: Option<usize>,
    pub note: String,
}

fn latest_available_checkpoint_epoch(
    run_dir: &Path,
    requested_epoch: Option<usize>,
) -> Option<usize> {
    let checkpoint_dir = run_dir.join("checkpoint");
    let entries = fs::read_dir(checkpoint_dir).ok()?;
    entries
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?;
            let epoch = name
                .strip_prefix("model-")?
                .strip_suffix(".bin")?
                .parse::<usize>()
                .ok()?;
            if requested_epoch.is_some_and(|requested| epoch > requested) {
                return None;
            }
            Some(epoch)
        })
        .max()
}

pub fn write_neuron_scale_request_artifact(
    run_dir: &Path,
    model_config: &DragonConfig,
    config: &NeuronScalingConfig,
    request: ModelScaleRequest,
) -> Result<PathBuf> {
    if request.to_capacity_units % model_config.n_embd != 0 {
        return Err(anyhow!(
            "requested Dragon latent_total {} is not divisible by n_embd {}",
            request.to_capacity_units,
            model_config.n_embd
        ));
    }

    let artifact = NeuronScaleRequestArtifact {
        target: NeuronScaleModelTarget {
            current_latent_total: request.from_capacity_units,
            target_latent_total: request.to_capacity_units,
            target_mlp_internal_dim_multiplier: request.to_capacity_units / model_config.n_embd,
            n_embd: model_config.n_embd,
            n_head: model_config.n_head,
        },
        max_latent_total: config.max_latent_total,
        max_scale_events: config.max_scale_events,
        batch_fit: config.batch_fit.clone(),
        stabilization: config.stabilization.clone(),
        resume_checkpoint_epoch: latest_available_checkpoint_epoch(run_dir, request.epoch),
        note: "pending Dragon neuron scaling; start a scaled continuation with model.latent_total set to target_latent_total and training.init_checkpoint_path pointing at this run's checkpoint directory so checkpoint widening is applied before continuing".to_string(),
        request,
    };

    let path = run_dir.join(NEURON_SCALE_REQUEST_ARTIFACT);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let payload = serde_json::to_string_pretty(&artifact)
        .context("serialize neuron scale request artifact")?;
    fs::write(&path, payload).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(path)
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
    control: Res<TrainingControlResource>,
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
        requests.write(request.clone());
        if let Some(handle) = &control.handle {
            handle.request_restart(&format!(
                "capacity plateau requested Dragon neuron scaling {} -> {}",
                request.from_capacity_units, request.to_capacity_units
            ));
        }
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

    #[test]
    fn write_neuron_scale_request_artifact_records_target_multiplier() {
        let dir = tempfile::tempdir().expect("tempdir");
        let checkpoint_dir = dir.path().join("checkpoint");
        std::fs::create_dir_all(&checkpoint_dir).expect("checkpoint dir");
        std::fs::write(checkpoint_dir.join("model-2.bin"), b"checkpoint").expect("checkpoint file");
        std::fs::write(checkpoint_dir.join("model-5.bin"), b"future checkpoint")
            .expect("future checkpoint file");
        let model_config = DragonConfig {
            n_embd: 8,
            n_head: 1,
            mlp_internal_dim_multiplier: 2,
            ..DragonConfig::default()
        };
        let config = NeuronScalingConfig {
            enabled: true,
            max_latent_total: 64,
            max_scale_events: 2,
            ..NeuronScalingConfig::default()
        };
        let request = ModelScaleRequest {
            run_id: "run".to_string(),
            epoch: Some(3),
            absolute_step: Some(48),
            from_capacity_units: 16,
            to_capacity_units: 32,
            reason: "capacity plateau".to_string(),
        };

        let path = write_neuron_scale_request_artifact(dir.path(), &model_config, &config, request)
            .expect("write scale request");
        assert_eq!(path, dir.path().join(NEURON_SCALE_REQUEST_ARTIFACT));

        let payload = std::fs::read_to_string(path).expect("read scale request");
        let artifact: NeuronScaleRequestArtifact =
            serde_json::from_str(&payload).expect("parse scale request");
        assert_eq!(artifact.target.current_latent_total, 16);
        assert_eq!(artifact.target.target_latent_total, 32);
        assert_eq!(artifact.target.target_mlp_internal_dim_multiplier, 4);
        assert_eq!(artifact.max_latent_total, 64);
        assert_eq!(artifact.max_scale_events, 2);
        assert_eq!(artifact.resume_checkpoint_epoch, Some(2));
    }

    #[test]
    fn write_neuron_scale_request_artifact_handles_missing_checkpoint() {
        let dir = tempfile::tempdir().expect("tempdir");
        let model_config = DragonConfig {
            n_embd: 8,
            n_head: 1,
            mlp_internal_dim_multiplier: 2,
            ..DragonConfig::default()
        };
        let config = NeuronScalingConfig {
            enabled: true,
            max_latent_total: 64,
            max_scale_events: 2,
            ..NeuronScalingConfig::default()
        };
        let request = ModelScaleRequest {
            run_id: "run".to_string(),
            epoch: Some(3),
            absolute_step: Some(48),
            from_capacity_units: 16,
            to_capacity_units: 32,
            reason: "capacity plateau".to_string(),
        };

        let path = write_neuron_scale_request_artifact(dir.path(), &model_config, &config, request)
            .expect("write scale request");
        let payload = std::fs::read_to_string(path).expect("read scale request");
        let artifact: NeuronScaleRequestArtifact =
            serde_json::from_str(&payload).expect("parse scale request");
        assert_eq!(artifact.resume_checkpoint_epoch, None);
    }
}
