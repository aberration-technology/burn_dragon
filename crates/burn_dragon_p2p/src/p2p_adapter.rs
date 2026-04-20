#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
use anyhow::{Result, anyhow};

#[cfg(feature = "wasm-ui")]
use burn_p2p::{BrowserEdgeSnapshot, ClientReleaseManifest, ExperimentScope};
#[cfg(feature = "wasm-ui")]
use burn_p2p_browser::{
    BrowserAppConnectConfig, BrowserAppTarget, BrowserCapabilityReport, BrowserEnrollmentConfig,
    BrowserRuntimeRole,
};
#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
use burn_p2p_core::{BrowserSeedAdvertisement, SchemaEnvelope, SignedPayload};

#[cfg(feature = "wasm-ui")]
use std::collections::BTreeSet;

#[cfg(feature = "wasm-ui")]
pub fn browser_runtime_role_label(role: &BrowserRuntimeRole) -> &'static str {
    match role {
        BrowserRuntimeRole::BrowserTrainerWgpu => "browser_trainer_wgpu",
        BrowserRuntimeRole::BrowserVerifier => "browser_verifier",
        BrowserRuntimeRole::BrowserObserver => "browser_observer",
        BrowserRuntimeRole::BrowserFallback => "browser_fallback",
        BrowserRuntimeRole::Viewer => "viewer",
    }
}

#[cfg(feature = "wasm-ui")]
pub fn browser_non_trainer_role_target(
    can_validate: bool,
) -> (BrowserRuntimeRole, BrowserAppTarget) {
    if can_validate {
        (
            BrowserRuntimeRole::BrowserVerifier,
            BrowserAppTarget::Validate,
        )
    } else {
        (
            BrowserRuntimeRole::BrowserObserver,
            BrowserAppTarget::Observe,
        )
    }
}

#[cfg(feature = "wasm-ui")]
pub fn browser_app_target_for_role(role: &BrowserRuntimeRole) -> BrowserAppTarget {
    match role {
        BrowserRuntimeRole::BrowserVerifier => BrowserAppTarget::Validate,
        BrowserRuntimeRole::BrowserTrainerWgpu => BrowserAppTarget::Train,
        _ => BrowserAppTarget::Observe,
    }
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
pub fn browser_enrollment_config_from_snapshot(
    snapshot: &BrowserEdgeSnapshot,
    release_manifest: &ClientReleaseManifest,
    requested_scopes: BTreeSet<ExperimentScope>,
    session_ttl_secs: i64,
) -> Result<BrowserEnrollmentConfig> {
    BrowserEnrollmentConfig::from_edge_snapshot(
        snapshot,
        release_manifest.target_artifact_id.clone(),
        release_manifest.target_artifact_hash.clone(),
        requested_scopes,
        session_ttl_secs,
    )
    .map_err(|error| anyhow!("failed to build browser enrollment config: {error}"))
}

#[cfg(all(feature = "wasm-ui", feature = "wasm-peer", target_arch = "wasm32"))]
pub fn build_browser_app_connect_config(
    edge_base_url: String,
    capability: BrowserCapabilityReport,
    target: BrowserAppTarget,
    seed_node_urls: Vec<String>,
    selection: Option<(String, Option<String>)>,
    bootstrap_snapshot: Option<BrowserEdgeSnapshot>,
    signed_seed_advertisement: Option<SignedPayload<SchemaEnvelope<BrowserSeedAdvertisement>>>,
) -> BrowserAppConnectConfig {
    let mut connect = BrowserAppConnectConfig::new(edge_base_url, capability, target)
        .with_seed_node_urls(seed_node_urls)
        .with_bootstrap_material(bootstrap_snapshot, signed_seed_advertisement);
    if let Some((experiment_id, revision_id)) = selection {
        connect = connect.with_selection(experiment_id, revision_id);
    }
    connect
}
