use std::sync::{Arc, Mutex};

use burn_dragon_train::train::events::{DynamicsControlEvent, DynamicsMode};
use burn_ecs::bevy_ecs;
use burn_ecs::prelude::{Component, IntoScheduleConfigs};

#[derive(Clone, Component, Default)]
pub struct DragonDynamicsControlSlot {
    inner: Arc<Mutex<Option<DynamicsControlEvent>>>,
}

impl DragonDynamicsControlSlot {
    pub fn store(&self, event: DynamicsControlEvent) {
        let mut pending = self
            .inner
            .lock()
            .expect("dragon dynamics control slot lock poisoned");
        if pending
            .as_ref()
            .is_none_or(|current| event.mode.control_priority() >= current.mode.control_priority())
        {
            *pending = Some(event);
        }
    }

    pub fn take(&self) -> Option<DynamicsControlEvent> {
        self.inner
            .lock()
            .expect("dragon dynamics control slot lock poisoned")
            .take()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DragonDynamicsControlPlugin;

impl burn_ecs::prelude::Plugin for DragonDynamicsControlPlugin {
    fn build(&self, app: &mut burn_ecs::prelude::App) {
        app.add_systems(
            burn_ecs::prelude::Update,
            capture_dynamics_controls.in_set(burn_ecs::prelude::TrainingSet::Sinks),
        );
    }
}

fn capture_dynamics_controls(
    mut controls: burn_ecs::prelude::MessageReader<DynamicsControlEvent>,
    registry: burn_ecs::prelude::Res<burn_ecs::prelude::TrainingRunRegistry>,
    slots: burn_ecs::prelude::Query<&DragonDynamicsControlSlot>,
) {
    for event in controls.read() {
        if let Some(slot) = registry.get_query(&event.run_id, &slots) {
            slot.store(event.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ecs::prelude::{App, TrainingAppExt, TrainingPlugins, TrainingRunConfig};

    fn control_event(mode: DynamicsMode) -> DynamicsControlEvent {
        DynamicsControlEvent {
            run_id: "run".into(),
            epoch: Some(1),
            absolute_step: Some(1),
            mode,
            lr_scale: 1.0,
            continual_backprop_scale: 1.0,
            max_replacements_per_interval: None,
            source_difficulty_pressure: 1.0,
            hash_noise_max_probability: 0.01,
            rollback_to_epoch: None,
            stop_if_repeated: false,
            reason: format!("{mode:?}"),
        }
    }

    #[test]
    fn dynamics_slot_keeps_recovery_when_stable_arrives_later() {
        let slot = DragonDynamicsControlSlot::default();
        slot.store(control_event(DynamicsMode::HardRecovery));
        slot.store(control_event(DynamicsMode::Stable));

        let event = slot.take().expect("pending control");
        assert_eq!(event.mode, DynamicsMode::HardRecovery);
    }

    #[test]
    fn dynamics_slot_allows_stronger_recovery_to_replace_pending_control() {
        let slot = DragonDynamicsControlSlot::default();
        slot.store(control_event(DynamicsMode::PlasticityRecovery));
        slot.store(control_event(DynamicsMode::RollbackRecovery));

        let event = slot.take().expect("pending control");
        assert_eq!(event.mode, DynamicsMode::RollbackRecovery);
    }

    #[test]
    fn dynamics_controls_are_isolated_by_training_run() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut app = App::new();
        app.add_plugins(TrainingPlugins)
            .add_plugins(DragonDynamicsControlPlugin);
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
        let slot_a = DragonDynamicsControlSlot::default();
        let slot_b = DragonDynamicsControlSlot::default();
        app.world_mut().entity_mut(run_a).insert(slot_a.clone());
        app.world_mut().entity_mut(run_b).insert(slot_b.clone());
        let mut event = control_event(DynamicsMode::PlasticityRecovery);
        event.run_id = "run-a".into();
        app.world_mut().write_message(event);

        app.update();

        assert_eq!(
            slot_a.take().expect("run-a control").mode,
            DynamicsMode::PlasticityRecovery
        );
        assert!(slot_b.take().is_none());
    }
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct ActiveDynamicsControl {
    pub mode: DynamicsMode,
    pub lr_scale: f64,
    pub continual_backprop_scale: f32,
    pub max_replacements_per_interval: Option<usize>,
    pub source_difficulty_pressure: f64,
    pub hash_noise_max_probability: f64,
    pub last_reason: String,
}

impl Default for ActiveDynamicsControl {
    fn default() -> Self {
        Self {
            mode: DynamicsMode::Stable,
            lr_scale: 1.0,
            continual_backprop_scale: 1.0,
            max_replacements_per_interval: None,
            source_difficulty_pressure: 1.0,
            hash_noise_max_probability: 0.01,
            last_reason: "initial stable control".to_string(),
        }
    }
}

impl ActiveDynamicsControl {
    pub fn recovery_auxiliary_active(&self) -> bool {
        matches!(
            self.mode,
            DynamicsMode::PlasticityRecovery
                | DynamicsMode::ValidationRecovery
                | DynamicsMode::RollbackRecovery
                | DynamicsMode::HardRecovery
                | DynamicsMode::HardCollapse
        )
    }

    pub fn apply_event(&mut self, event: &DynamicsControlEvent) {
        self.mode = event.mode;
        self.lr_scale = event.lr_scale.clamp(0.0, 4.0);
        self.continual_backprop_scale = event.continual_backprop_scale.clamp(0.0, 8.0) as f32;
        self.max_replacements_per_interval = event.max_replacements_per_interval;
        self.source_difficulty_pressure = event.source_difficulty_pressure.clamp(0.0, 8.0);
        self.hash_noise_max_probability = event.hash_noise_max_probability.clamp(0.0, 1.0);
        self.last_reason = event.reason.clone();
    }
}
