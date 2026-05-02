use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::DragonExperimentKind;

#[cfg(feature = "native")]
mod native;
#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
mod web;

#[cfg(feature = "native")]
pub use native::{
    NativeDowngradeObservation, NativeDowngradeScope, apply_native_downgrade_state,
    clear_native_downgrade, load_matching_native_downgrade, persist_native_downgrade,
};
#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
pub use web::{
    apply_browser_downgrade_state, clear_browser_downgrade, load_browser_downgrade,
    persist_browser_downgrade,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DragonCapabilityDowngradeRecord {
    pub scope_fingerprint: String,
    pub experiment_kind: DragonExperimentKind,
    pub backend_label: String,
    pub downgrade_to: String,
    pub observed_training_bytes: u64,
    pub trainer_budget_bytes: Option<u64>,
    pub reason: String,
    pub source: String,
    pub observed_at: DateTime<Utc>,
    pub failure_count: u32,
}

#[derive(Serialize)]
struct DragonCapabilityScope<'a, M> {
    experiment_kind: DragonExperimentKind,
    backend_label: &'a str,
    model_config: &'a M,
    batch_size: usize,
    block_size: usize,
}

pub fn capability_scope_fingerprint<M: Serialize>(
    experiment_kind: DragonExperimentKind,
    backend_label: &str,
    model_config: &M,
    batch_size: usize,
    block_size: usize,
) -> String {
    let scope = DragonCapabilityScope {
        experiment_kind,
        backend_label,
        model_config,
        batch_size,
        block_size,
    };
    let encoded = serde_json::to_vec(&scope).expect("capability scope should serialize");
    let mut hasher = Sha256::new();
    hasher.update(encoded);
    format!("{:x}", hasher.finalize())
}

fn record_is_still_binding(
    record: &DragonCapabilityDowngradeRecord,
    current_trainer_budget_bytes: Option<u64>,
) -> bool {
    if !is_probable_trainer_fit_failure(&record.reason) {
        return false;
    }
    let failed_budget_bytes = record
        .trainer_budget_bytes
        .unwrap_or(record.observed_training_bytes)
        .max(record.observed_training_bytes);
    match current_trainer_budget_bytes {
        Some(current_budget) => current_budget <= failed_budget_bytes,
        None => true,
    }
}

pub(crate) fn is_probable_trainer_fit_failure(message: &str) -> bool {
    let message = message.to_ascii_lowercase();
    if is_transient_runtime_or_control_plane_failure(&message) {
        return false;
    }
    [
        "out of memory",
        "out_of_memory",
        "vram",
        "gpu memory",
        "device lost",
        "failed to allocate",
        "failed allocation",
        "insufficient memory",
        "allocation failed",
        "allocator",
        "buffer allocation",
        "cuda error",
        "wgpu error",
        "webgpu device lost",
    ]
    .iter()
    .any(|needle| message.contains(needle))
        || contains_oom_token(&message)
}

fn contains_oom_token(message: &str) -> bool {
    message
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .any(|token| token == "oom")
}

fn is_transient_runtime_or_control_plane_failure(message: &str) -> bool {
    [
        "http client error",
        "http status",
        "502",
        "503",
        "504",
        "bad gateway",
        "service unavailable",
        "gateway timeout",
        "receipts/browser",
        "/receipts/",
        "failed to synchronize browser runtime",
        "failed to negotiate transport",
        "failed to connect to destination",
        "failed to dial",
        "resource limit exceeded",
        "timeout has been reached",
    ]
    .iter()
    .any(|needle| message.contains(needle))
}

#[cfg(test)]
mod tests {
    use chrono::Utc;

    use super::*;

    fn record_with_reason(reason: &str) -> DragonCapabilityDowngradeRecord {
        DragonCapabilityDowngradeRecord {
            scope_fingerprint: "scope".into(),
            experiment_kind: DragonExperimentKind::NcaPrepretraining,
            backend_label: "wgpu".into(),
            downgrade_to: "browser_verifier".into(),
            observed_training_bytes: 1024,
            trainer_budget_bytes: Some(512),
            reason: reason.into(),
            source: "runtime".into(),
            observed_at: Utc::now(),
            failure_count: 1,
        }
    }

    #[test]
    fn trainer_fit_failure_classifier_rejects_transient_receipt_errors() {
        assert!(!is_probable_trainer_fit_failure(
            "http client error: HTTP status server error (502 Bad Gateway) for url (https://edge.dragon.aberration.technology/receipts/browser)"
        ));
        assert!(!record_is_still_binding(
            &record_with_reason(
                "http client error: HTTP status server error (502 Bad Gateway) for url (https://edge.dragon.aberration.technology/receipts/browser)"
            ),
            Some(512),
        ));
    }

    #[test]
    fn trainer_fit_failure_classifier_rejects_transport_resource_limits_with_peer_id_noise() {
        assert!(!is_probable_trainer_fit_failure(
            "Failed to negotiate transport protocol(s): [(/ip4/3.149.166.58/udp/443/webrtc-direct/certhash/uEiBIQQvRGIR6ld6a-VTmYxgsVlaOOMfJtcsf5LvtFwh7mQ/p2p/12D3KooWCkxZ42qCD3mSzPeAazTE9cCrFtidxAQKisQgMiXtVFxB/p2p-circuit/p2p/12D3KooWRBYrqJ8PvwQsJ523gNCUHJ4YJyo6p91QhC3DMssfitWe: : Failed to connect to destination.: Failed to connect to destination.: Remote reported resource limit exceeded.)]"
        ));
        assert!(!record_is_still_binding(
            &record_with_reason(
                "Failed to negotiate transport protocol(s): /p2p/12D3KooWExample resource limit exceeded"
            ),
            Some(512),
        ));
    }

    #[test]
    fn trainer_fit_failure_classifier_accepts_memory_and_device_failures() {
        assert!(is_probable_trainer_fit_failure(
            "CUDA error: out of memory while allocating optimizer state"
        ));
        assert!(is_probable_trainer_fit_failure(
            "trainer failed with OOM while allocating activation buffer"
        ));
        assert!(is_probable_trainer_fit_failure(
            "CUDA_ERROR_OUT_OF_MEMORY during forward pass"
        ));
        assert!(is_probable_trainer_fit_failure(
            "webgpu device lost after failed to allocate buffer"
        ));
        assert!(record_is_still_binding(
            &record_with_reason("webgpu device lost after failed to allocate buffer"),
            Some(512),
        ));
    }
}
