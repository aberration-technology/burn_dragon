use std::sync::{Mutex, OnceLock};

#[derive(Clone, Copy, Debug, Default)]
pub struct TrainProfileSnapshot {
    pub dataloader_cpu_ns: u128,
    pub dataloader_foreground_wait_ns: u128,
    pub dataloader_tensor_copy_ns: u128,
    pub dataloader_host_to_device_copy_bytes: u128,
    pub host_sync_points: u64,
    pub forward_ns: u128,
    pub loss_backward_ns: u128,
    pub embed_probe_ns: u128,
    pub first_layer_forward_probe_ns: u128,
    pub first_layer_probe_ns: u128,
    pub logits_loss_probe_ns: u128,
    pub hidden_logits_loss_probe_ns: u128,
    pub hidden_model_forward_probe_ns: u128,
    pub hidden_model_probe_ns: u128,
    pub detail_probe_steps: u64,
    pub train_steps: u64,
    pub max_step_reserved_before_bytes: u64,
    pub max_step_in_use_before_bytes: u64,
    pub max_step_reserved_after_forward_bytes: u64,
    pub max_step_in_use_after_forward_bytes: u64,
    pub max_step_reserved_after_backward_bytes: u64,
    pub max_step_in_use_after_backward_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PredictiveCodingProfileSnapshot {
    pub chunks_seen: usize,
    pub chunks_corrected: usize,
    pub inference_steps: usize,
    pub skipped_empty_state: usize,
    pub energy_before_sum: f64,
    pub energy_after_sum: f64,
    pub energy_delta_sum: f64,
    pub energy_samples: usize,
    pub grad_norm_sum: f64,
    pub grad_norm_max: f64,
    pub grad_norm_samples: usize,
    pub delta_rms_sum: f64,
    pub delta_rms_samples: usize,
    pub elapsed_ns: u128,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LatentReasoningProfileSnapshot {
    pub loss_calls: usize,
    pub next_latent_components: usize,
    pub dragon_state_components: usize,
    pub jepa_components: usize,
    pub sigreg_components: usize,
    pub configured_steps_sum: usize,
    pub configured_steps_samples: usize,
}

impl LatentReasoningProfileSnapshot {
    pub(crate) fn has_activity(self) -> bool {
        self.loss_calls > 0
            || self.next_latent_components > 0
            || self.dragon_state_components > 0
            || self.jepa_components > 0
            || self.sigreg_components > 0
    }

    pub(crate) fn configured_steps_mean(self) -> Option<f64> {
        (self.configured_steps_samples > 0)
            .then(|| self.configured_steps_sum as f64 / self.configured_steps_samples as f64)
    }
}

impl PredictiveCodingProfileSnapshot {
    pub(crate) fn has_activity(self) -> bool {
        self.chunks_seen > 0 || self.inference_steps > 0 || self.elapsed_ns > 0
    }

    pub(crate) fn energy_before_mean(self) -> Option<f64> {
        (self.energy_samples > 0).then(|| self.energy_before_sum / self.energy_samples as f64)
    }

    pub(crate) fn energy_after_mean(self) -> Option<f64> {
        (self.energy_samples > 0).then(|| self.energy_after_sum / self.energy_samples as f64)
    }

    pub(crate) fn energy_delta_mean(self) -> Option<f64> {
        (self.energy_samples > 0).then(|| self.energy_delta_sum / self.energy_samples as f64)
    }

    pub(crate) fn grad_norm_mean(self) -> Option<f64> {
        (self.grad_norm_samples > 0).then(|| self.grad_norm_sum / self.grad_norm_samples as f64)
    }

    pub(crate) fn grad_norm_max(self) -> Option<f64> {
        (self.grad_norm_samples > 0).then_some(self.grad_norm_max)
    }

    pub(crate) fn delta_rms_mean(self) -> Option<f64> {
        (self.delta_rms_samples > 0).then(|| self.delta_rms_sum / self.delta_rms_samples as f64)
    }

    pub(crate) fn elapsed_ms(self) -> f64 {
        self.elapsed_ns as f64 / 1_000_000.0
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct TrainProfileState {
    dataloader_cpu_ns: u128,
    dataloader_foreground_wait_ns: u128,
    dataloader_tensor_copy_ns: u128,
    dataloader_host_to_device_copy_bytes: u128,
    host_sync_points: u64,
    forward_ns: u128,
    loss_backward_ns: u128,
    embed_probe_ns: u128,
    first_layer_forward_probe_ns: u128,
    first_layer_probe_ns: u128,
    logits_loss_probe_ns: u128,
    hidden_logits_loss_probe_ns: u128,
    hidden_model_forward_probe_ns: u128,
    hidden_model_probe_ns: u128,
    detail_probe_steps: u64,
    train_steps: u64,
    max_step_reserved_before_bytes: u64,
    max_step_in_use_before_bytes: u64,
    max_step_reserved_after_forward_bytes: u64,
    max_step_in_use_after_forward_bytes: u64,
    max_step_reserved_after_backward_bytes: u64,
    max_step_in_use_after_backward_bytes: u64,
    predictive_coding: PredictiveCodingProfileSnapshot,
    latent_reasoning: LatentReasoningProfileSnapshot,
}

static TRAIN_PROFILE: OnceLock<Mutex<TrainProfileState>> = OnceLock::new();

pub fn enabled() -> bool {
    std::env::var_os("DragonModel_STAGE_PROFILE").is_some()
}

pub fn detail_enabled() -> bool {
    std::env::var_os("DragonModel_STAGE_PROFILE_DETAIL").is_some()
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
}

pub fn detail_interval_steps() -> usize {
    env_usize("DragonModel_STAGE_PROFILE_DETAIL_EVERY").unwrap_or(64)
}

pub fn detail_max_steps() -> Option<usize> {
    env_usize("DragonModel_STAGE_PROFILE_DETAIL_MAX_STEPS")
}

pub(crate) fn detail_due_for(
    step_index: usize,
    interval_steps: usize,
    max_steps: Option<usize>,
) -> bool {
    if max_steps.is_some_and(|max_steps| step_index >= max_steps) {
        return false;
    }
    step_index.is_multiple_of(interval_steps.max(1))
}

pub fn detail_due(step_index: usize) -> bool {
    detail_enabled() && detail_due_for(step_index, detail_interval_steps(), detail_max_steps())
}

pub fn memory_enabled() -> bool {
    std::env::var_os("DragonModel_STAGE_PROFILE_MEMORY").is_some()
}

fn state() -> &'static Mutex<TrainProfileState> {
    TRAIN_PROFILE.get_or_init(|| Mutex::new(TrainProfileState::default()))
}

fn record(mutator: impl FnOnce(&mut TrainProfileState)) {
    if let Ok(mut profile) = state().lock() {
        mutator(&mut profile);
    }
}

pub fn reset() {
    if let Ok(mut profile) = state().lock() {
        *profile = TrainProfileState::default();
    }
}

pub(crate) fn reset_predictive_coding() {
    if let Ok(mut profile) = state().lock() {
        profile.predictive_coding = PredictiveCodingProfileSnapshot::default();
    }
}

pub(crate) fn take_predictive_coding() -> PredictiveCodingProfileSnapshot {
    if let Ok(mut profile) = state().lock() {
        let snapshot = profile.predictive_coding;
        profile.predictive_coding = PredictiveCodingProfileSnapshot::default();
        return snapshot;
    }
    PredictiveCodingProfileSnapshot::default()
}

pub(crate) fn take_latent_reasoning() -> LatentReasoningProfileSnapshot {
    if let Ok(mut profile) = state().lock() {
        let snapshot = profile.latent_reasoning;
        profile.latent_reasoning = LatentReasoningProfileSnapshot::default();
        return snapshot;
    }
    LatentReasoningProfileSnapshot::default()
}

pub fn snapshot() -> TrainProfileSnapshot {
    if let Ok(profile) = state().lock() {
        return TrainProfileSnapshot {
            dataloader_cpu_ns: profile.dataloader_cpu_ns,
            dataloader_foreground_wait_ns: profile.dataloader_foreground_wait_ns,
            dataloader_tensor_copy_ns: profile.dataloader_tensor_copy_ns,
            dataloader_host_to_device_copy_bytes: profile.dataloader_host_to_device_copy_bytes,
            host_sync_points: profile.host_sync_points,
            forward_ns: profile.forward_ns,
            loss_backward_ns: profile.loss_backward_ns,
            embed_probe_ns: profile.embed_probe_ns,
            first_layer_forward_probe_ns: profile.first_layer_forward_probe_ns,
            first_layer_probe_ns: profile.first_layer_probe_ns,
            logits_loss_probe_ns: profile.logits_loss_probe_ns,
            hidden_logits_loss_probe_ns: profile.hidden_logits_loss_probe_ns,
            hidden_model_forward_probe_ns: profile.hidden_model_forward_probe_ns,
            hidden_model_probe_ns: profile.hidden_model_probe_ns,
            detail_probe_steps: profile.detail_probe_steps,
            train_steps: profile.train_steps,
            max_step_reserved_before_bytes: profile.max_step_reserved_before_bytes,
            max_step_in_use_before_bytes: profile.max_step_in_use_before_bytes,
            max_step_reserved_after_forward_bytes: profile.max_step_reserved_after_forward_bytes,
            max_step_in_use_after_forward_bytes: profile.max_step_in_use_after_forward_bytes,
            max_step_reserved_after_backward_bytes: profile.max_step_reserved_after_backward_bytes,
            max_step_in_use_after_backward_bytes: profile.max_step_in_use_after_backward_bytes,
        };
    }
    TrainProfileSnapshot::default()
}

pub(crate) fn record_predictive_coding(
    chunks_seen: usize,
    chunks_corrected: usize,
    inference_steps: usize,
    skipped_empty_state: usize,
    energy_before: Option<f64>,
    energy_after: Option<f64>,
    grad_norm_mean: Option<f64>,
    grad_norm_max: Option<f64>,
    delta_rms_mean: Option<f64>,
    elapsed_ns: u128,
) {
    record(|profile| {
        let pc = &mut profile.predictive_coding;
        pc.chunks_seen = pc.chunks_seen.saturating_add(chunks_seen);
        pc.chunks_corrected = pc.chunks_corrected.saturating_add(chunks_corrected);
        pc.inference_steps = pc.inference_steps.saturating_add(inference_steps);
        pc.skipped_empty_state = pc.skipped_empty_state.saturating_add(skipped_empty_state);
        pc.elapsed_ns = pc.elapsed_ns.saturating_add(elapsed_ns);
        if let (Some(before), Some(after)) = (energy_before, energy_after)
            && before.is_finite()
            && after.is_finite()
        {
            pc.energy_before_sum += before;
            pc.energy_after_sum += after;
            pc.energy_delta_sum += after - before;
            pc.energy_samples = pc.energy_samples.saturating_add(1);
        }
        if let Some(grad_norm_mean) = grad_norm_mean
            && grad_norm_mean.is_finite()
        {
            pc.grad_norm_sum += grad_norm_mean;
            pc.grad_norm_samples = pc.grad_norm_samples.saturating_add(1);
        }
        if let Some(grad_norm_max) = grad_norm_max
            && grad_norm_max.is_finite()
        {
            pc.grad_norm_max = pc.grad_norm_max.max(grad_norm_max);
        }
        if let Some(delta_rms_mean) = delta_rms_mean
            && delta_rms_mean.is_finite()
        {
            pc.delta_rms_sum += delta_rms_mean;
            pc.delta_rms_samples = pc.delta_rms_samples.saturating_add(1);
        }
    });
}

pub(crate) fn record_latent_reasoning(
    next_latent_components: usize,
    dragon_state_components: usize,
    jepa_components: usize,
    sigreg_components: usize,
    configured_steps: usize,
) {
    record(|profile| {
        let latent = &mut profile.latent_reasoning;
        latent.loss_calls = latent.loss_calls.saturating_add(1);
        latent.next_latent_components = latent
            .next_latent_components
            .saturating_add(next_latent_components);
        latent.dragon_state_components = latent
            .dragon_state_components
            .saturating_add(dragon_state_components);
        latent.jepa_components = latent.jepa_components.saturating_add(jepa_components);
        latent.sigreg_components = latent.sigreg_components.saturating_add(sigreg_components);
        latent.configured_steps_sum = latent.configured_steps_sum.saturating_add(configured_steps);
        latent.configured_steps_samples = latent.configured_steps_samples.saturating_add(1);
    });
}

pub fn record_dataloader(
    cpu_ns: u128,
    tensor_copy_ns: u128,
    host_to_device_copy_bytes: u128,
    host_sync_points: u64,
) {
    record(|profile| {
        profile.dataloader_cpu_ns = profile.dataloader_cpu_ns.saturating_add(cpu_ns);
        profile.dataloader_tensor_copy_ns = profile
            .dataloader_tensor_copy_ns
            .saturating_add(tensor_copy_ns);
        profile.dataloader_host_to_device_copy_bytes = profile
            .dataloader_host_to_device_copy_bytes
            .saturating_add(host_to_device_copy_bytes);
        profile.host_sync_points = profile.host_sync_points.saturating_add(host_sync_points);
    });
}

pub fn record_dataloader_foreground_wait(wait_ns: u128) {
    record(|profile| {
        profile.dataloader_foreground_wait_ns = profile
            .dataloader_foreground_wait_ns
            .saturating_add(wait_ns);
    });
}

pub fn record_train_step(forward_ns: u128, loss_backward_ns: u128) {
    record(|profile| {
        profile.forward_ns = profile.forward_ns.saturating_add(forward_ns);
        profile.loss_backward_ns = profile.loss_backward_ns.saturating_add(loss_backward_ns);
        profile.train_steps = profile.train_steps.saturating_add(1);
    });
}

pub fn record_train_step_memory(
    before_reserved_bytes: u64,
    before_in_use_bytes: u64,
    after_forward_reserved_bytes: u64,
    after_forward_in_use_bytes: u64,
    after_backward_reserved_bytes: u64,
    after_backward_in_use_bytes: u64,
) {
    record(|profile| {
        profile.max_step_reserved_before_bytes = profile
            .max_step_reserved_before_bytes
            .max(before_reserved_bytes);
        profile.max_step_in_use_before_bytes = profile
            .max_step_in_use_before_bytes
            .max(before_in_use_bytes);
        profile.max_step_reserved_after_forward_bytes = profile
            .max_step_reserved_after_forward_bytes
            .max(after_forward_reserved_bytes);
        profile.max_step_in_use_after_forward_bytes = profile
            .max_step_in_use_after_forward_bytes
            .max(after_forward_in_use_bytes);
        profile.max_step_reserved_after_backward_bytes = profile
            .max_step_reserved_after_backward_bytes
            .max(after_backward_reserved_bytes);
        profile.max_step_in_use_after_backward_bytes = profile
            .max_step_in_use_after_backward_bytes
            .max(after_backward_in_use_bytes);
    });
}

pub fn record_detail_probe(
    embed_probe_ns: u128,
    first_layer_forward_probe_ns: u128,
    first_layer_probe_ns: u128,
    logits_loss_probe_ns: u128,
    hidden_logits_loss_probe_ns: u128,
    hidden_model_forward_probe_ns: u128,
    hidden_model_probe_ns: u128,
) {
    record(|profile| {
        profile.embed_probe_ns = profile.embed_probe_ns.saturating_add(embed_probe_ns);
        profile.first_layer_forward_probe_ns = profile
            .first_layer_forward_probe_ns
            .saturating_add(first_layer_forward_probe_ns);
        profile.first_layer_probe_ns = profile
            .first_layer_probe_ns
            .saturating_add(first_layer_probe_ns);
        profile.logits_loss_probe_ns = profile
            .logits_loss_probe_ns
            .saturating_add(logits_loss_probe_ns);
        profile.hidden_logits_loss_probe_ns = profile
            .hidden_logits_loss_probe_ns
            .saturating_add(hidden_logits_loss_probe_ns);
        profile.hidden_model_forward_probe_ns = profile
            .hidden_model_forward_probe_ns
            .saturating_add(hidden_model_forward_probe_ns);
        profile.hidden_model_probe_ns = profile
            .hidden_model_probe_ns
            .saturating_add(hidden_model_probe_ns);
        profile.detail_probe_steps = profile.detail_probe_steps.saturating_add(1);
    });
}

#[cfg(test)]
mod tests {
    use super::detail_due_for;

    #[test]
    fn detail_due_respects_interval_and_max_steps() {
        assert!(detail_due_for(0, 64, None));
        assert!(!detail_due_for(1, 64, None));
        assert!(detail_due_for(64, 64, None));
        assert!(!detail_due_for(64, 64, Some(64)));
        assert!(detail_due_for(63, 1, Some(64)));
    }
}
