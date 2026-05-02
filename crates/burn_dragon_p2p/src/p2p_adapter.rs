#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
use anyhow::{Result, anyhow};

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
use burn_p2p::WorkloadTrainingLease;
#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
use burn_p2p::{ClientReleaseManifest, ExperimentScope};
#[cfg(all(feature = "wasm-ui", feature = "wasm-peer", target_arch = "wasm32"))]
use burn_p2p_browser::{BrowserAppClientView, BrowserAppConnectConfig, BrowserAppController};
#[cfg(feature = "wasm-ui")]
use burn_p2p_browser::{BrowserAppTarget, BrowserRuntimeRole};
#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
use burn_p2p_browser::{
    BrowserCapabilityReport, BrowserEdgeSnapshot, BrowserEnrollmentConfig,
    BrowserSeedAdvertisement, BrowserTransportKind, BrowserTransportPolicy, SchemaEnvelope,
    SignedPayload,
};

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
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
    BrowserEnrollmentConfig::from_edge_snapshot_for_release(
        snapshot,
        release_manifest,
        requested_scopes,
        session_ttl_secs,
    )
    .map_err(|error| anyhow!("failed to build browser enrollment config: {error}"))
}

#[cfg(all(feature = "wasm-ui", feature = "wasm-peer", target_arch = "wasm32"))]
pub struct DragonBrowserAppHandle(BrowserAppController);

#[cfg(all(feature = "wasm-ui", feature = "wasm-peer", target_arch = "wasm32"))]
impl DragonBrowserAppHandle {
    pub async fn connect(connect: BrowserAppConnectConfig) -> Result<Self> {
        BrowserAppController::connect_with(connect)
            .await
            .map(Self)
            .map_err(|error| anyhow!("failed to connect browser app controller: {error}"))
    }

    pub async fn refresh(&mut self) -> Result<BrowserAppClientView> {
        self.0
            .refresh()
            .await
            .map(|_| self.0.view())
            .map_err(|error| anyhow!("failed to refresh browser app controller: {error}"))
    }

    pub fn view(&self) -> BrowserAppClientView {
        self.0.view()
    }

    pub fn active_training_lease(&self) -> Option<&WorkloadTrainingLease> {
        self.0.active_training_lease()
    }

    pub fn effective_active_training_lease(&self) -> Option<WorkloadTrainingLease> {
        self.0.effective_active_training_lease()
    }
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
pub fn browser_trainer_transport_policy() -> BrowserTransportPolicy {
    BrowserTransportPolicy {
        preferred: vec![BrowserTransportKind::WebRtcDirect],
        observer_fallback: BrowserTransportKind::WebRtcDirect,
        allow_suspend_resume: true,
    }
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
