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
