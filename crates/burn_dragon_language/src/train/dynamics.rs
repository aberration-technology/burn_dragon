use std::sync::{Arc, Mutex};

use burn_dragon_train::train::events::{DynamicsControlEvent, DynamicsMode};
use burn_ecs::prelude::IntoScheduleConfigs;

#[derive(Clone, Default)]
pub struct DragonDynamicsControlSlot {
    inner: Arc<Mutex<Option<DynamicsControlEvent>>>,
}

impl DragonDynamicsControlSlot {
    pub fn store(&self, event: DynamicsControlEvent) {
        *self
            .inner
            .lock()
            .expect("dragon dynamics control slot lock poisoned") = Some(event);
    }

    pub fn take(&self) -> Option<DynamicsControlEvent> {
        self.inner
            .lock()
            .expect("dragon dynamics control slot lock poisoned")
            .take()
    }
}

impl burn_ecs::prelude::Resource for DragonDynamicsControlSlot {}

pub struct DragonDynamicsControlPlugin {
    slot: DragonDynamicsControlSlot,
}

impl DragonDynamicsControlPlugin {
    pub fn new(slot: DragonDynamicsControlSlot) -> Self {
        Self { slot }
    }
}

impl burn_ecs::prelude::Plugin for DragonDynamicsControlPlugin {
    fn build(&self, app: &mut burn_ecs::prelude::App) {
        app.insert_resource(self.slot.clone()).add_systems(
            burn_ecs::prelude::Update,
            capture_dynamics_controls.in_set(burn_ecs::prelude::TrainingSet::Sinks),
        );
    }
}

fn capture_dynamics_controls(
    mut controls: burn_ecs::prelude::MessageReader<DynamicsControlEvent>,
    slot: burn_ecs::prelude::Res<DragonDynamicsControlSlot>,
) {
    for event in controls.read() {
        slot.store(event.clone());
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
