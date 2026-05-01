use anyhow::{Result, anyhow};
use burn_p2p::{
    AuthProvider, ClientPlatform, ClientReleaseManifest, ContentId, ExperimentDirectoryEntry,
    ExperimentScope, ProjectFamilyId, StudyId,
};
use burn_p2p_admin::AdminResult;
use burn_p2p_app::{
    AdminSessionCard, AdminSessionSummaryView, DirectoryEntryDraftPanel, DirectoryEntryDraftView,
    DirectoryMutationResultView, ExperimentDirectoryEntryView, ExperimentDirectoryListPanel,
    ExperimentDirectoryListView, RolloutPreviewPanel, RolloutPreviewView,
    RolloutSubmissionStatusPanel,
};
use burn_p2p_browser::{
    BrowserAppClientView, BrowserEdgeSnapshot, BrowserSeedAdvertisement, BrowserSessionState,
    SchemaEnvelope, SignedPayload, browser_transport_kind,
};
use burn_p2p_core::BrowserSeedTransportKind;
use dioxus::prelude::*;
#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
use gloo_timers::future::TimeoutFuture;
use log::{error, info, warn};
use std::{cell::RefCell, collections::BTreeSet};
use url::form_urlencoded;

use crate::admin::{
    fetch_directory_entries, fetch_signed_directory_entries, rollout_directory_entries,
    upsert_directory_entry,
};
use crate::auth::{
    begin_browser_github_login, complete_browser_github_login, fetch_edge_snapshot,
    load_browser_session, provider_code_from_window_location, reset_browser_runtime_state,
};
use crate::build_info;
use crate::capability::{
    DragonBrowserCapabilityDecision, DragonBrowserHostCapabilityProbe, decide_browser_capability,
    detect_browser_host_capabilities,
};
use crate::capability_state::apply_browser_downgrade_state;
#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
use crate::capability_state::clear_browser_downgrade;
use crate::config::{DragonBrowserAppConfig, DragonPeerNetworkConfig};
use crate::p2p_adapter::{
    DragonBrowserAppHandle, browser_runtime_role_label, build_browser_app_connect_config,
};
#[cfg(feature = "wasm-peer")]
use crate::profile::{
    DragonExperimentProfile, browser_training_config_from_profile, find_matching_entry,
};
#[cfg(feature = "wasm-peer")]
use crate::wasm::training::{
    DragonBrowserTrainingResult, run_browser_training_with_release_manifest,
};

#[cfg(feature = "wasm-peer")]
pub mod training;

mod ui_state;
#[cfg(test)]
use ui_state::{
    DragonHeroTone, DragonReadinessStepId, DragonStepStatus, DragonUiEventCandidate,
    DragonUiEventKind, dragon_browser_training_action_ready, dragon_live_notice,
};
use ui_state::{
    DragonMetricCardView, DragonPeerUiContext, DragonReadinessStepView,
    DragonTrainingActionContext, DragonUiEvent, browser_app_refresh_interval_millis,
    dragon_global_training_detail, dragon_global_training_summary, dragon_local_training_detail,
    dragon_local_training_summary, dragon_network_detail, dragon_peer_ui_state,
    dragon_push_ui_event, dragon_runtime_mode_detail, dragon_runtime_mode_summary,
    dragon_session_metric_view, dragon_slice_progress_summary, dragon_training_action_state,
    dragon_transport_summary, dragon_ui_now_ms, dragon_window_progress_detail,
    dragon_window_summary,
};

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
const BROWSER_APP_REFRESH_INTERVAL_MILLIS: u32 = 1_000;
#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
const BROWSER_APP_CONNECTING_REFRESH_INTERVAL_MILLIS: u32 = 4_000;
#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
const BROWSER_APP_DEGRADED_REFRESH_INTERVAL_MILLIS: u32 = 8_000;
#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
const HERO_RATTLE_INTERVAL_MILLIS: u32 = 80;
const HERO_RATTLE_FRAMES: &[&str] = &[
    "⠉⠉", "⠈⠙", "⠀⠹", "⠀⢸", "⠀⣰", "⢀⣠", "⣀⣀", "⣄⡀", "⣆⠀", "⡇⠀", "⠏⠀", "⠋⠁",
];
const DRAGON_UI_EVENT_LIMIT: usize = 5;

#[cfg(feature = "wasm-peer")]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
enum DragonLocalTrainingState {
    #[default]
    Idle,
    Starting,
    SyncingCheckpoint,
    TrainingWindow,
    Stopping,
    Failed {
        message: String,
    },
    Stopped,
}

#[cfg(feature = "wasm-peer")]
impl DragonLocalTrainingState {
    fn is_active(&self) -> bool {
        matches!(
            self,
            Self::Starting | Self::SyncingCheckpoint | Self::TrainingWindow | Self::Stopping
        )
    }

    fn failure_message(&self) -> Option<&str> {
        match self {
            Self::Failed { message } => Some(message.as_str()),
            _ => None,
        }
    }
}

thread_local! {
    static DRAGON_BROWSER_APP_CONTROLLER: RefCell<Option<DragonBrowserAppHandle>> = const { RefCell::new(None) };
}

fn current_app_semver() -> semver::Version {
    semver::Version::parse(env!("CARGO_PKG_VERSION")).expect("valid burn_dragon version")
}

fn browser_view_log_summary(view: &BrowserAppClientView) -> String {
    let artifact_sync = view.network.swarm_status.artifact_sync.as_ref();
    format!(
        "runtime={} detail={} desired_transport={:?} connected_transport={:?} direct_peers={} can_train={} assignment={} head_present={} head_artifact_ready={} head_artifact_source={:?} head_artifact_route={:?} head_artifact_error={:?} cached_microshards={}",
        view.runtime_label,
        view.runtime_detail,
        view.network.swarm_status.desired_transport,
        view.network.swarm_status.connected_transport,
        view.network.direct_peers,
        view.training.can_train,
        view.training.active_assignment.is_some(),
        view.training.latest_head_id.is_some(),
        view.training.active_head_artifact_ready,
        view.training.active_head_artifact_source,
        artifact_sync.and_then(|diagnostic| diagnostic.route_label.as_deref()),
        artifact_sync.and_then(|diagnostic| diagnostic.last_error.as_deref()),
        view.training.cached_microshards,
    )
}

fn browser_transport_family_label(
    family: Option<&burn_p2p_core::BrowserTransportFamily>,
) -> Option<String> {
    family.map(|transport| browser_transport_kind(transport).label().to_owned())
}

fn active_direct_transport_error(view: &BrowserAppClientView) -> Option<&str> {
    let error = view.network.swarm_status.last_error.as_deref()?;
    if view.network.swarm_status.connected_transport.is_some() || view.network.direct_peers > 0 {
        return None;
    }
    Some(error)
}

fn dragon_browser_training_requires_active_head_artifact(config: &DragonBrowserAppConfig) -> bool {
    #[cfg(feature = "wasm-peer")]
    {
        config
            .training
            .as_ref()
            .and_then(|training| training.live_participant.as_ref())
            .is_none_or(|live| live.load_active_head_artifact)
    }
    #[cfg(not(feature = "wasm-peer"))]
    {
        let _ = config;
        true
    }
}

fn browser_view_machine_state_json(view: &BrowserAppClientView) -> String {
    serde_json::to_string_pretty(&serde_json::json!({
        "runtime_label": &view.runtime_label,
        "runtime_detail": &view.runtime_detail,
        "desired_transport": browser_transport_family_label(view.network.swarm_status.desired_transport.as_ref()),
        "connected_transport": browser_transport_family_label(view.network.swarm_status.connected_transport.as_ref()),
        "direct_peers": view.network.direct_peers,
        "connected_peer_ids": &view.network.swarm_status.connected_peer_ids,
        "can_train": view.training.can_train,
        "assignment": view.training.active_assignment.is_some(),
        "head_present": view.training.latest_head_id.is_some(),
        "head_artifact_ready": view.training.active_head_artifact_ready,
        "head_artifact_source": &view.training.active_head_artifact_source,
        "head_artifact_error": &view.training.active_head_artifact_error,
        "head_artifact_sync": &view.network.swarm_status.artifact_sync,
        "cached_microshards": view.training.cached_microshards,
        "last_error": active_direct_transport_error(view),
        "network_transport": &view.network.transport,
    }))
    .expect("browser machine state json")
}

fn browser_scope_summary(scopes: &BTreeSet<ExperimentScope>) -> String {
    if scopes.is_empty() {
        return "none".into();
    }
    scopes
        .iter()
        .map(|scope| format!("{scope:?}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn browser_host_capability_summary(probe: &DragonBrowserHostCapabilityProbe) -> String {
    format!(
        "navigator_gpu={} worker_gpu={} dedicated_worker={} persistent_storage={} web_rtc={} web_transport={} system_memory_mib={}",
        probe.navigator_gpu_exposed,
        probe.worker_gpu_exposed,
        probe.dedicated_worker_exposed,
        probe.persistent_storage_exposed,
        probe.web_rtc_exposed,
        probe.web_transport_exposed,
        probe.system_memory_bytes.unwrap_or_default() / (1024 * 1024),
    )
}

fn browser_capability_decision_summary(decision: &DragonBrowserCapabilityDecision) -> String {
    let trainer_budget_mib = decision
        .trainer_memory_budget_bytes
        .map(|bytes| (bytes / (1024 * 1024)).to_string())
        .unwrap_or_else(|| "n/a".into());
    let estimated_training_mib = decision
        .footprint
        .as_ref()
        .map(|footprint| (footprint.estimated_training_bytes / (1024 * 1024)).to_string())
        .unwrap_or_else(|| "n/a".into());
    format!(
        "recommended_role={} connect_target={:?} can_train={} trainer_budget_mib={} estimated_training_mib={} downgrade_reason={}",
        browser_runtime_role_label(&decision.capability.recommended_role),
        decision.connect_target,
        decision.can_train,
        trainer_budget_mib,
        estimated_training_mib,
        decision.downgrade_reason.as_deref().unwrap_or("none"),
    )
}

fn browser_capability_decision_for_config(
    config: &DragonBrowserAppConfig,
) -> (
    DragonBrowserHostCapabilityProbe,
    DragonBrowserCapabilityDecision,
) {
    let browser_host_capabilities = detect_browser_host_capabilities();
    let browser_capability_decision =
        match (config.training.as_ref(), resolved_edge_base_url(config)) {
            (Some(training), Ok(edge_base_url)) => apply_browser_downgrade_state(
                &edge_base_url,
                training,
                browser_backend_label(training),
                decide_browser_capability(Some(training), &browser_host_capabilities),
            ),
            (Some(training), Err(_)) => {
                decide_browser_capability(Some(training), &browser_host_capabilities)
            }
            (None, _) => decide_browser_capability(None, &browser_host_capabilities),
        };
    (browser_host_capabilities, browser_capability_decision)
}

fn browser_session_scope_summary(session: &BrowserSessionState) -> String {
    session
        .session
        .as_ref()
        .map(|session| browser_scope_summary(&session.claims.granted_scopes))
        .unwrap_or_else(|| "none".into())
}

#[cfg(feature = "wasm-peer")]
fn browser_backend_label(config: &crate::config::DragonBrowserTrainingConfig) -> &'static str {
    config.execution_backend.backend_label()
}

fn window_query_string() -> Result<String> {
    web_sys::window()
        .ok_or_else(|| anyhow!("window unavailable"))?
        .location()
        .search()
        .map_err(|error| anyhow!("failed to inspect browser query params: {error:?}"))
}

fn window_query_flag(name: &str) -> bool {
    let Ok(query) = window_query_string() else {
        return false;
    };
    form_urlencoded::parse(query.trim_start_matches('?').as_bytes()).any(|(key, value)| {
        key == name && matches!(value.as_ref(), "" | "1" | "true" | "yes" | "on" | "open")
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DragonBrowserTransportOverride {
    WebRtcDirect,
    WebTransport,
    Wss,
}

impl DragonBrowserTransportOverride {
    fn matches_seed(self, seed: &str) -> bool {
        match self {
            Self::WebRtcDirect => seed.contains("/webrtc-direct/"),
            Self::WebTransport => seed.contains("/webtransport"),
            Self::Wss => seed.contains("/wss") || seed.contains("/ws"),
        }
    }

    fn transport_policy_preference(self) -> Vec<BrowserSeedTransportKind> {
        match self {
            Self::WebRtcDirect => vec![BrowserSeedTransportKind::WebRtcDirect],
            Self::WebTransport => vec![BrowserSeedTransportKind::WebTransport],
            Self::Wss => vec![BrowserSeedTransportKind::WssFallback],
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::WebRtcDirect => "webrtc-direct",
            Self::WebTransport => "webtransport",
            Self::Wss => "wss",
        }
    }
}

fn browser_transport_override_from_label(label: &str) -> Option<DragonBrowserTransportOverride> {
    match label.trim().to_ascii_lowercase().as_str() {
        "webrtc" | "webrtc-direct" | "direct" => Some(DragonBrowserTransportOverride::WebRtcDirect),
        "webtransport" | "web-transport" => Some(DragonBrowserTransportOverride::WebTransport),
        "wss" | "ws" | "websocket" | "websocket-fallback" => {
            Some(DragonBrowserTransportOverride::Wss)
        }
        "auto" | "" => None,
        _ => None,
    }
}

fn browser_transport_override_from_query(query: &str) -> Option<DragonBrowserTransportOverride> {
    for (key, value) in form_urlencoded::parse(query.trim_start_matches('?').as_bytes()) {
        if matches!(
            key.as_ref(),
            "transport" | "transport_mode" | "peer_transport" | "browser_transport"
        ) {
            return browser_transport_override_from_label(&value);
        }
    }
    None
}

fn browser_transport_override() -> Option<DragonBrowserTransportOverride> {
    if let Ok(query) = window_query_string()
        && let Some(transport) = browser_transport_override_from_query(&query)
    {
        return Some(transport);
    }
    None
}

fn filter_seed_urls_for_transport(
    seeds: Vec<String>,
    transport: DragonBrowserTransportOverride,
) -> Vec<String> {
    seeds
        .into_iter()
        .filter(|seed| transport.matches_seed(seed))
        .collect()
}

fn filter_signed_seed_advertisement_for_transport(
    signed_seed_advertisement: &mut SignedPayload<SchemaEnvelope<BrowserSeedAdvertisement>>,
    transport: DragonBrowserTransportOverride,
) {
    let payload = &mut signed_seed_advertisement.payload.payload;
    for seed in &mut payload.seeds {
        seed.multiaddrs
            .retain(|multiaddr| transport.matches_seed(multiaddr));
    }
    payload.seeds.retain(|seed| !seed.multiaddrs.is_empty());
    payload.transport_policy.preferred = transport.transport_policy_preference();
    payload.transport_policy.allow_fallback_wss =
        matches!(transport, DragonBrowserTransportOverride::Wss);
}

fn prefer_webrtc_direct_bootstrap_when_configured(
    seed_node_urls: &mut Vec<String>,
    signed_seed_advertisement: &mut Option<SignedPayload<SchemaEnvelope<BrowserSeedAdvertisement>>>,
) {
    let direct = DragonBrowserTransportOverride::WebRtcDirect;
    if !seed_node_urls.iter().any(|seed| direct.matches_seed(seed)) {
        return;
    }

    let filtered_seed_node_urls = filter_seed_urls_for_transport(seed_node_urls.clone(), direct);
    if !filtered_seed_node_urls.is_empty() {
        *seed_node_urls = filtered_seed_node_urls;
    }

    let mut signed_seed_retained = true;
    if let Some(advertisement) = signed_seed_advertisement.as_mut() {
        filter_signed_seed_advertisement_for_transport(advertisement, direct);
        signed_seed_retained = !advertisement.payload.payload.seeds.is_empty();
    }
    if !signed_seed_retained {
        warn!(
            "signed browser seed advertisement contained no webrtc-direct seeds; using site-config seed bootstrap only"
        );
        *signed_seed_advertisement = None;
    }
}

fn callback_site_root_pathname(pathname: &str) -> Option<String> {
    let (prefix, _) = pathname.split_once("/callback/")?;
    let prefix = prefix.trim_end_matches('/');
    Some(if prefix.is_empty() {
        "/".to_owned()
    } else {
        format!("{prefix}/")
    })
}

fn normalized_browser_callback_url(pathname: &str, search: &str, hash: &str) -> String {
    let normalized_pathname =
        callback_site_root_pathname(pathname).unwrap_or_else(|| pathname.to_owned());
    let mut filtered = form_urlencoded::Serializer::new(String::new());
    for (key, value) in form_urlencoded::parse(search.trim_start_matches('?').as_bytes()) {
        if key == "code" || key == "state" {
            continue;
        }
        filtered.append_pair(&key, &value);
    }
    let query = filtered.finish();
    if query.is_empty() {
        format!("{normalized_pathname}{hash}")
    } else {
        format!("{normalized_pathname}?{query}{hash}")
    }
}

fn normalize_provider_callback_window_location() -> Result<()> {
    let window = web_sys::window().ok_or_else(|| anyhow!("window unavailable"))?;
    let location = window.location();
    let pathname = location
        .pathname()
        .map_err(|error| anyhow!("failed to inspect browser pathname: {error:?}"))?;
    let search = location
        .search()
        .map_err(|error| anyhow!("failed to inspect browser query params: {error:?}"))?;
    let hash = location
        .hash()
        .map_err(|error| anyhow!("failed to inspect browser hash: {error:?}"))?;
    let next_url = normalized_browser_callback_url(&pathname, &search, &hash);
    window
        .history()
        .map_err(|error| anyhow!("failed to access browser history: {error:?}"))?
        .replace_state_with_url(&wasm_bindgen::JsValue::NULL, "", Some(&next_url))
        .map_err(|error| anyhow!("failed to replace browser callback URL: {error:?}"))?;
    Ok(())
}

fn config_with_window_network_overrides(
    config: &DragonBrowserAppConfig,
) -> Result<DragonBrowserAppConfig> {
    let query = window_query_string()?;
    Ok(config.clone().with_network_overrides(
        DragonPeerNetworkConfig::parse_edge_base_url_query(&query),
        DragonPeerNetworkConfig::parse_seed_node_query(&query),
    ))
}

fn resolved_edge_base_url(config: &DragonBrowserAppConfig) -> Result<String> {
    config_with_window_network_overrides(config)?
        .effective_edge_base_url()
        .map(str::to_owned)
        .ok_or_else(|| anyhow!("no edge base URL configured"))
}

fn can_use_embedded_browser_bootstrap(
    bootstrap_config: &DragonBrowserAppConfig,
    connect_config: &DragonBrowserAppConfig,
) -> bool {
    bootstrap_config.effective_edge_base_url() == connect_config.effective_edge_base_url()
        && bootstrap_config.effective_seed_node_urls() == connect_config.effective_seed_node_urls()
}

fn connect_config(
    bootstrap_config: &DragonBrowserAppConfig,
    connect_config: &DragonBrowserAppConfig,
    edge_snapshot: Option<&BrowserEdgeSnapshot>,
    signed_seed_advertisement: Option<&SignedPayload<SchemaEnvelope<BrowserSeedAdvertisement>>>,
) -> Result<burn_p2p_browser::BrowserAppConnectConfig> {
    let config = connect_config.clone();
    let (_capability_probe, capability_decision) = browser_capability_decision_for_config(&config);
    let transport_override = browser_transport_override();
    let (bootstrap_snapshot, mut signed_seed_advertisement) =
        if can_use_embedded_browser_bootstrap(bootstrap_config, &config)
            || (transport_override.is_some()
                && bootstrap_config.effective_edge_base_url() == config.effective_edge_base_url())
        {
            (edge_snapshot.cloned(), signed_seed_advertisement.cloned())
        } else {
            (None, None)
        };
    if signed_seed_advertisement
        .as_ref()
        .is_some_and(|advertisement| {
            advertisement.payload.payload.expires_at
                <= chrono::Utc::now() + chrono::Duration::seconds(30)
        })
    {
        warn!(
            "embedded browser seed advertisement expired or near expiry; fetching fresh signed seeds from edge"
        );
        signed_seed_advertisement = None;
    }
    let mut seed_node_urls = config.effective_seed_node_urls().to_vec();
    if transport_override.is_none() {
        prefer_webrtc_direct_bootstrap_when_configured(
            &mut seed_node_urls,
            &mut signed_seed_advertisement,
        );
    } else if let Some(transport_override) = transport_override {
        let filtered_seed_node_urls =
            filter_seed_urls_for_transport(seed_node_urls.clone(), transport_override);
        if filtered_seed_node_urls.is_empty() {
            warn!(
                "browser transport override {} matched no configured seed addrs; keeping default seed set",
                transport_override.label()
            );
        } else {
            seed_node_urls = filtered_seed_node_urls;
            if let Some(signed_seed_advertisement) = signed_seed_advertisement.as_mut() {
                filter_signed_seed_advertisement_for_transport(
                    signed_seed_advertisement,
                    transport_override,
                );
            }
            info!(
                "browser transport override active: {} seed_count={}",
                transport_override.label(),
                seed_node_urls.len()
            );
        }
    }
    Ok(build_browser_app_connect_config(
        resolved_edge_base_url(&config)?,
        capability_decision.capability,
        capability_decision.connect_target,
        seed_node_urls,
        config.selected_experiment(),
        bootstrap_snapshot,
        signed_seed_advertisement,
    ))
}

fn browser_release_manifest_from_snapshot(snapshot: &BrowserEdgeSnapshot) -> ClientReleaseManifest {
    let target_artifact_hash = snapshot
        .allowed_target_artifact_hashes
        .iter()
        .next()
        .cloned()
        .or_else(|| {
            snapshot
                .trust_bundle
                .as_ref()
                .and_then(|bundle| bundle.allowed_target_artifact_hashes.iter().next().cloned())
        })
        .unwrap_or_else(|| ContentId::new("dragon-browser-client-artifact"));
    let release_train_hash = snapshot
        .required_release_train_hash
        .clone()
        .or_else(|| {
            snapshot
                .trust_bundle
                .as_ref()
                .map(|bundle| bundle.required_release_train_hash.clone())
        })
        .unwrap_or_else(|| ContentId::new("dragon-browser-client-train"));
    let project_family_id = snapshot
        .trust_bundle
        .as_ref()
        .map(|bundle| bundle.project_family_id.clone())
        .unwrap_or_else(|| ProjectFamilyId::new("burn-dragon-language"));
    let app_semver = current_app_semver();

    ClientReleaseManifest {
        project_family_id,
        release_train_hash,
        target_artifact_id: "browser-wasm".into(),
        target_artifact_hash,
        target_platform: ClientPlatform::Browser,
        app_semver,
        git_commit: build_info::embedded_git_commit_or_unknown(),
        cargo_lock_hash: ContentId::new("dragon-browser-site-lock"),
        burn_version_string: "0.21.0-pre.3".into(),
        enabled_features_hash: ContentId::new("dragon-browser-site-features"),
        protocol_major: snapshot.protocol_major,
        supported_workloads: Vec::new(),
        built_at: chrono::Utc::now(),
    }
}

async fn resolve_browser_release_manifest(
    config: &DragonBrowserAppConfig,
    release_manifest: Option<&ClientReleaseManifest>,
    edge_snapshot: Option<&BrowserEdgeSnapshot>,
) -> Result<ClientReleaseManifest> {
    if let Some(release_manifest) = release_manifest {
        return Ok(release_manifest.clone());
    }
    if let Some(snapshot) = edge_snapshot {
        return Ok(browser_release_manifest_from_snapshot(snapshot));
    }

    let edge_base_url = resolved_edge_base_url(config)?;
    let snapshot = fetch_edge_snapshot(&edge_base_url).await?;
    Ok(browser_release_manifest_from_snapshot(&snapshot))
}

#[cfg(feature = "wasm-peer")]
async fn resolve_browser_training_config(
    bootstrap_config: &DragonBrowserAppConfig,
    config: &DragonBrowserAppConfig,
    edge_snapshot: Option<&BrowserEdgeSnapshot>,
    signed_seed_advertisement: Option<&SignedPayload<SchemaEnvelope<BrowserSeedAdvertisement>>>,
) -> Result<crate::config::DragonBrowserTrainingConfig> {
    if let Some(mut training) = config.training.clone() {
        if training.training_lease.is_none() {
            training.training_lease = active_training_lease(
                bootstrap_config,
                config,
                edge_snapshot,
                signed_seed_advertisement,
            )
            .await?;
        }
        return Ok(training);
    }

    let mut training = resolve_browser_training_config_from_directory(config, edge_snapshot)
        .await?
        .ok_or_else(|| {
            anyhow!("selected experiment does not publish a browser training profile")
        })?;
    training.training_lease = active_training_lease(
        bootstrap_config,
        config,
        edge_snapshot,
        signed_seed_advertisement,
    )
    .await?;
    Ok(training)
}

#[cfg(feature = "wasm-peer")]
async fn resolve_browser_training_config_from_directory(
    config: &DragonBrowserAppConfig,
    edge_snapshot: Option<&BrowserEdgeSnapshot>,
) -> Result<Option<crate::config::DragonBrowserTrainingConfig>> {
    let snapshot = match resolved_edge_base_url(config) {
        Ok(edge_base_url) => match fetch_edge_snapshot(&edge_base_url).await {
            Ok(snapshot) => snapshot,
            Err(error) => {
                if let Some(snapshot) = edge_snapshot {
                    warn!(
                        "failed to refresh browser training profile from live edge; using embedded snapshot: {error}"
                    );
                    snapshot.clone()
                } else {
                    return Err(error);
                }
            }
        },
        Err(error) => {
            if let Some(snapshot) = edge_snapshot {
                warn!(
                    "failed to resolve edge URL for live browser training profile; using embedded snapshot: {error}"
                );
                snapshot.clone()
            } else {
                return Err(error);
            }
        }
    };
    let entry = find_matching_entry(
        &snapshot.directory.entries,
        config.selected_experiment_id.as_deref(),
        config.selected_revision_id.as_deref(),
        None,
    )?
    .ok_or_else(|| anyhow!("no Dragon experiment entry was found on the current edge"))?;
    let profile = DragonExperimentProfile::from_entry_metadata(entry)?
        .ok_or_else(|| anyhow!("selected experiment does not publish a Dragon training profile"))?;
    browser_training_config_from_profile(entry, &profile)
}

#[cfg(feature = "wasm-peer")]
async fn resolve_browser_app_runtime_config(
    config: &DragonBrowserAppConfig,
    edge_snapshot: Option<&BrowserEdgeSnapshot>,
) -> Result<DragonBrowserAppConfig> {
    let mut effective_config = config_with_window_network_overrides(config)?;
    if effective_config.training.is_none() {
        match resolve_browser_training_config_from_directory(&effective_config, edge_snapshot).await
        {
            Ok(training) => effective_config.training = training,
            Err(error) => warn!("browser training profile resolution failed: {error}"),
        }
    }
    Ok(effective_config)
}

#[cfg(feature = "wasm-peer")]
async fn active_training_lease(
    bootstrap_config: &DragonBrowserAppConfig,
    config: &DragonBrowserAppConfig,
    edge_snapshot: Option<&BrowserEdgeSnapshot>,
    signed_seed_advertisement: Option<&SignedPayload<SchemaEnvelope<BrowserSeedAdvertisement>>>,
) -> Result<Option<burn_p2p::WorkloadTrainingLease>> {
    let live_controller_present =
        DRAGON_BROWSER_APP_CONTROLLER.with(|slot| slot.borrow().as_ref().is_some());
    let live_training_lease = DRAGON_BROWSER_APP_CONTROLLER.with(|slot| {
        slot.borrow()
            .as_ref()
            .and_then(|controller| controller.active_training_lease().cloned())
    });
    if live_controller_present {
        return Ok(live_training_lease);
    }

    let controller = DragonBrowserAppHandle::connect(connect_config(
        bootstrap_config,
        config,
        edge_snapshot,
        signed_seed_advertisement,
    )?)
    .await?;
    Ok(controller.active_training_lease().cloned())
}

fn auth_provider_label(provider: &AuthProvider) -> String {
    match provider {
        AuthProvider::GitHub => "GitHub".into(),
        AuthProvider::Oidc { issuer } => format!("OIDC ({issuer})"),
        AuthProvider::OAuth { provider } => format!("OAuth ({provider})"),
        AuthProvider::External { authority } => format!("External ({authority})"),
        AuthProvider::Static { authority } => format!("Static ({authority})"),
    }
}

fn admin_session_summary_view(
    session: Option<&BrowserSessionState>,
    study_id: &str,
) -> AdminSessionSummaryView {
    let rollout_enabled = session_has_admin_scope(session, study_id);
    let Some(session_state) = session else {
        return AdminSessionSummaryView {
            session_label: "no active operator session".into(),
            principal_label: None,
            provider_label: None,
            session_id: None,
            rollout_enabled,
        };
    };
    let session_id = session_state
        .session_id()
        .map(|session_id| session_id.as_str().to_owned());
    let Some(session) = session_state.session.as_ref() else {
        return AdminSessionSummaryView {
            session_label: "edge session pending claims".into(),
            principal_label: None,
            provider_label: None,
            session_id,
            rollout_enabled,
        };
    };
    let claims = &session.claims;
    AdminSessionSummaryView {
        session_label: if rollout_enabled {
            "admin session ready".into()
        } else {
            "authenticated session".into()
        },
        principal_label: Some(claims.principal_id.as_str().to_owned()),
        provider_label: Some(auth_provider_label(&claims.provider)),
        session_id,
        rollout_enabled,
    }
}

fn directory_list_view(
    entries: &[ExperimentDirectoryEntry],
    selected_experiment_id: Option<String>,
    selected_revision_id: Option<String>,
) -> ExperimentDirectoryListView {
    ExperimentDirectoryListView::from_entries(
        "/directory",
        "/directory/signed",
        selected_experiment_id,
        selected_revision_id,
        entries,
    )
}

fn rollout_preview_view(entries: &[ExperimentDirectoryEntry]) -> RolloutPreviewView {
    let summary_label = match entries.len() {
        1 => "1 directory entry ready for rollout".into(),
        count => format!("{count} directory entries ready for rollout"),
    };
    RolloutPreviewView {
        summary_label,
        submit_path: "/admin".into(),
        requires_session: true,
        entries: entries
            .iter()
            .map(ExperimentDirectoryEntryView::from)
            .collect(),
    }
}

fn rollout_result_view(result: &AdminResult) -> Option<DirectoryMutationResultView> {
    match result {
        AdminResult::AuthPolicyRolledOut {
            minimum_revocation_epoch,
            directory_entries,
            trusted_issuers,
            reenrollment_required,
        } => Some(DirectoryMutationResultView {
            status_label: minimum_revocation_epoch
                .as_ref()
                .map(|epoch| format!("auth policy rolled out at epoch {}", epoch.0))
                .unwrap_or_else(|| "auth policy rolled out".into()),
            directory_entries: *directory_entries,
            trusted_issuers: *trusted_issuers,
            reenrollment_required: *reenrollment_required,
        }),
        _ => None,
    }
}

const DEFAULT_ADMIN_STUDY_ID: &str = "burn-dragon-mainnet";

fn admin_requested_scopes(
    config: &DragonBrowserAppConfig,
    study_id: &str,
) -> std::collections::BTreeSet<ExperimentScope> {
    let mut scopes = config.requested_scopes.clone();
    let study_id = study_id.trim();
    if !study_id.is_empty() {
        scopes.insert(ExperimentScope::Admin {
            study_id: StudyId::new(study_id.to_owned()),
        });
    }
    scopes
}

fn granted_admin_studies(session: Option<&BrowserSessionState>) -> Vec<String> {
    session
        .and_then(|session| session.session.as_ref())
        .map(|session| {
            session
                .claims
                .granted_scopes
                .iter()
                .filter_map(|scope| match scope {
                    ExperimentScope::Admin { study_id } => Some(study_id.as_str().to_owned()),
                    _ => None,
                })
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn session_has_admin_scope(session: Option<&BrowserSessionState>, study_id: &str) -> bool {
    let study_id = study_id.trim();
    !study_id.is_empty()
        && granted_admin_studies(session)
            .iter()
            .any(|value| value == study_id)
}

fn browser_session_is_authenticated(session: &BrowserSessionState) -> bool {
    session.session.is_some()
}

fn directory_entries_to_json(entries: &[ExperimentDirectoryEntry]) -> Result<String> {
    serde_json::to_string_pretty(entries).map_err(Into::into)
}

fn directory_entry_to_json(entry: &ExperimentDirectoryEntry) -> Result<String> {
    serde_json::to_string_pretty(entry).map_err(Into::into)
}

fn parse_directory_entries_json(input: &str) -> Result<Vec<ExperimentDirectoryEntry>> {
    let input = input.trim();
    if input.is_empty() {
        return Ok(Vec::new());
    }
    serde_json::from_str(input).map_err(|error| anyhow!("invalid directory JSON: {error}"))
}

fn parse_directory_entry_json(input: &str) -> Result<ExperimentDirectoryEntry> {
    serde_json::from_str(input.trim()).map_err(|error| anyhow!("invalid entry JSON: {error}"))
}

fn find_directory_entry(
    entries: &[ExperimentDirectoryEntry],
    study_id: &str,
    experiment_id: &str,
) -> Option<ExperimentDirectoryEntry> {
    let study_id = study_id.trim();
    let experiment_id = experiment_id.trim();
    entries
        .iter()
        .find(|entry| {
            entry.experiment_id.as_str() == experiment_id
                && (study_id.is_empty() || entry.study_id.as_str() == study_id)
        })
        .cloned()
}

pub async fn connect_browser_app(
    bootstrap_config: &DragonBrowserAppConfig,
    config: &DragonBrowserAppConfig,
    edge_snapshot: Option<&BrowserEdgeSnapshot>,
    signed_seed_advertisement: Option<&SignedPayload<SchemaEnvelope<BrowserSeedAdvertisement>>>,
) -> Result<BrowserAppClientView> {
    let effective_config = resolve_browser_app_runtime_config(config, edge_snapshot).await?;
    let edge_base_url = resolved_edge_base_url(&effective_config)?;
    let (browser_host_capabilities, browser_capability_decision) =
        browser_capability_decision_for_config(&effective_config);
    info!(
        "browser capability assessment: edge_url={} requested_scopes=[{}] {} {}",
        edge_base_url,
        browser_scope_summary(&effective_config.requested_scopes),
        browser_host_capability_summary(&browser_host_capabilities),
        browser_capability_decision_summary(&browser_capability_decision),
    );
    info!(
        "browser app connect: edge_url={} seed_count={} auth_required={}",
        edge_base_url,
        effective_config.effective_seed_node_urls().len(),
        effective_config.require_edge_auth
    );
    let controller = DragonBrowserAppHandle::connect(connect_config(
        bootstrap_config,
        &effective_config,
        edge_snapshot,
        signed_seed_advertisement,
    )?)
    .await?;
    let view = controller.view();
    info!("browser app connected: {}", browser_view_log_summary(&view));
    DRAGON_BROWSER_APP_CONTROLLER.with(|slot| {
        *slot.borrow_mut() = Some(controller);
    });
    Ok(view)
}

fn clear_live_browser_app_controller() {
    DRAGON_BROWSER_APP_CONTROLLER.with(|slot| {
        let _ = slot.borrow_mut().take();
    });
}

pub async fn refresh_browser_app(
    bootstrap_config: &DragonBrowserAppConfig,
    config: &DragonBrowserAppConfig,
    edge_snapshot: Option<&BrowserEdgeSnapshot>,
    signed_seed_advertisement: Option<&SignedPayload<SchemaEnvelope<BrowserSeedAdvertisement>>>,
) -> Result<BrowserAppClientView> {
    let effective_config = resolve_browser_app_runtime_config(config, edge_snapshot).await?;
    let mut controller = if let Some(controller) =
        DRAGON_BROWSER_APP_CONTROLLER.with(|slot| slot.borrow_mut().take())
    {
        controller
    } else {
        DragonBrowserAppHandle::connect(connect_config(
            bootstrap_config,
            &effective_config,
            edge_snapshot,
            signed_seed_advertisement,
        )?)
        .await?
    };
    let refresh_result = controller.refresh().await;
    DRAGON_BROWSER_APP_CONTROLLER.with(|slot| {
        *slot.borrow_mut() = Some(controller);
    });
    let view = refresh_result?;
    if let Some(error) = retained_refresh_transport_warning(&view) {
        warn!("browser app refresh retained transport error: {error}");
    }
    Ok(view)
}

fn retained_refresh_transport_warning(view: &BrowserAppClientView) -> Option<&str> {
    active_direct_transport_error(view)
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
fn spawn_browser_app_refresh_loop(
    bootstrap_config: DragonBrowserAppConfig,
    config: DragonBrowserAppConfig,
    edge_snapshot: Option<BrowserEdgeSnapshot>,
    signed_seed_advertisement: Option<SignedPayload<SchemaEnvelope<BrowserSeedAdvertisement>>>,
    mut current_view: Signal<Option<BrowserAppClientView>>,
    mut status: Signal<String>,
    mut checkpoint_wait_generation: Signal<u64>,
) {
    let next_generation = (*checkpoint_wait_generation.read()).saturating_add(1);
    checkpoint_wait_generation.set(next_generation);
    spawn(async move {
        loop {
            let refresh_interval_millis = {
                let view = current_view.read();
                browser_app_refresh_interval_millis(view.as_ref())
            };
            TimeoutFuture::new(refresh_interval_millis).await;
            if *checkpoint_wait_generation.read() != next_generation {
                break;
            }
            match refresh_browser_app(
                &bootstrap_config,
                &config,
                edge_snapshot.as_ref(),
                signed_seed_advertisement.as_ref(),
            )
            .await
            {
                Ok(view) => {
                    current_view.set(Some(view));
                    status.set(String::new());
                }
                Err(error) => {
                    warn!("browser app refresh failed: {error}");
                    status.set(error.to_string());
                }
            }
        }
    });
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
fn spawn_hero_rattle_loop(
    active: bool,
    mut hero_rattle_index: Signal<usize>,
    mut hero_rattle_generation: Signal<u64>,
) {
    let next_generation = (*hero_rattle_generation.read()).saturating_add(1);
    hero_rattle_generation.set(next_generation);
    if !active {
        hero_rattle_index.set(0);
        return;
    }
    spawn(async move {
        loop {
            TimeoutFuture::new(HERO_RATTLE_INTERVAL_MILLIS).await;
            if *hero_rattle_generation.read() != next_generation {
                break;
            }
            let next_index = (*hero_rattle_index.read() + 1) % HERO_RATTLE_FRAMES.len();
            hero_rattle_index.set(next_index);
        }
    });
}

pub async fn resume_or_complete_browser_auth(
    config: &DragonBrowserAppConfig,
    release_manifest: Option<&ClientReleaseManifest>,
    edge_snapshot: Option<&BrowserEdgeSnapshot>,
) -> Result<Option<BrowserSessionState>> {
    let edge_base_url = resolved_edge_base_url(config)?;
    if let Some(provider_code) = provider_code_from_window_location() {
        let release_manifest =
            resolve_browser_release_manifest(config, release_manifest, edge_snapshot).await?;
        let session = complete_browser_github_login(
            &edge_base_url,
            &release_manifest,
            config.requested_scopes.clone(),
            3600,
            &provider_code,
        )
        .await?;
        info!(
            "browser auth completed: principal={} reenrollment_required={} granted_scopes=[{}]",
            session
                .session
                .as_ref()
                .map(|session| session.claims.principal_id.as_str())
                .unwrap_or("anonymous"),
            session.reenrollment_required,
            browser_session_scope_summary(&session),
        );
        let _ = normalize_provider_callback_window_location();
        return Ok(Some(session));
    }
    if config.require_edge_auth {
        let session = load_browser_session(&edge_base_url).await?;
        if browser_session_is_authenticated(&session) {
            info!(
                "browser auth resumed: principal={} reenrollment_required={} granted_scopes=[{}]",
                session
                    .session
                    .as_ref()
                    .map(|session| session.claims.principal_id.as_str())
                    .unwrap_or("anonymous"),
                session.reenrollment_required,
                browser_session_scope_summary(&session),
            );
        }
        return Ok(browser_session_is_authenticated(&session).then_some(session));
    }
    Ok(None)
}

pub async fn start_browser_github_auth_with_scopes(
    config: &DragonBrowserAppConfig,
    release_manifest: Option<&ClientReleaseManifest>,
    edge_snapshot: Option<&BrowserEdgeSnapshot>,
    requested_scopes: std::collections::BTreeSet<ExperimentScope>,
) -> Result<()> {
    let edge_base_url = resolved_edge_base_url(config)?;
    let release_manifest =
        resolve_browser_release_manifest(config, release_manifest, edge_snapshot).await?;
    let login = begin_browser_github_login(
        &edge_base_url,
        &release_manifest,
        requested_scopes,
        3600,
        None,
    )
    .await?;
    let authorize_url = login
        .authorize_url
        .ok_or_else(|| anyhow!("edge did not return a browser authorize URL"))?;
    web_sys::window()
        .ok_or_else(|| anyhow!("window unavailable"))?
        .location()
        .set_href(&authorize_url)
        .map_err(|error| anyhow!("failed to redirect to edge auth: {error:?}"))?;
    Ok(())
}

pub async fn start_browser_github_auth(
    config: &DragonBrowserAppConfig,
    release_manifest: Option<&ClientReleaseManifest>,
    edge_snapshot: Option<&BrowserEdgeSnapshot>,
) -> Result<()> {
    start_browser_github_auth_with_scopes(
        config,
        release_manifest,
        edge_snapshot,
        config.requested_scopes.clone(),
    )
    .await
}

#[derive(Props, Clone, PartialEq)]
pub struct DragonBrowserAppProps {
    pub config: DragonBrowserAppConfig,
    pub release_manifest: Option<ClientReleaseManifest>,
    pub edge_snapshot: Option<BrowserEdgeSnapshot>,
    pub signed_seed_advertisement: Option<SignedPayload<SchemaEnvelope<BrowserSeedAdvertisement>>>,
}

#[component]
pub fn DragonBrowserApp(props: DragonBrowserAppProps) -> Element {
    let initial_config = config_with_window_network_overrides(&props.config)
        .unwrap_or_else(|_| props.config.clone());
    let runtime_config = use_signal(|| initial_config.clone());
    let mut edge_url = use_signal(|| {
        initial_config
            .effective_edge_base_url()
            .unwrap_or_default()
            .to_owned()
    });
    let mut seed_node_urls = use_signal(|| initial_config.effective_seed_node_urls().join(", "));
    let status = use_signal(String::new);
    let current_view = use_signal(|| None::<BrowserAppClientView>);
    let session_state = use_signal(|| None::<BrowserSessionState>);
    let last_logged_browser_status = use_signal(String::new);
    let last_logged_transport_error = use_signal(|| None::<String>);
    let last_logged_runtime_summary = use_signal(|| None::<String>);
    let mut admin_study_id = use_signal(|| DEFAULT_ADMIN_STUDY_ID.to_owned());
    let mut admin_experiment_id = use_signal(|| {
        initial_config
            .selected_experiment_id
            .clone()
            .unwrap_or_else(|| "nca-prepretraining".into())
    });
    let mut admin_directory_json = use_signal(String::new);
    let mut admin_entry_json = use_signal(String::new);
    let mut admin_status = use_signal(String::new);
    let admin_rollout_result = use_signal(|| None::<DirectoryMutationResultView>);
    let debug_controls_enabled = window_query_flag("debug");
    let mut show_connection_settings = use_signal(|| false);
    let mut show_admin_tools = use_signal(|| window_query_flag("admin"));
    let auth_bootstrap_started = use_signal(|| false);
    let auth_bootstrap_pending = use_signal(|| true);
    let checkpoint_wait_generation = use_signal(|| 0_u64);
    #[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
    let hero_rattle_index = use_signal(|| 0_usize);
    #[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
    let hero_rattle_generation = use_signal(|| 0_u64);
    #[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
    let hero_rattle_state = use_signal(|| None::<bool>);
    let ui_events = use_signal(Vec::<DragonUiEvent>::new);
    let last_ui_event_key = use_signal(String::new);
    #[cfg(feature = "wasm-peer")]
    let local_training = use_signal(|| None::<DragonBrowserTrainingResult>);
    #[cfg(feature = "wasm-peer")]
    let local_training_state = use_signal(DragonLocalTrainingState::default);
    #[cfg(feature = "wasm-peer")]
    let local_training_stop_requested = use_signal(|| false);

    {
        let config = initial_config.clone();
        let bootstrap_config = props.config.clone();
        let release_manifest = props.release_manifest.clone();
        let edge_snapshot = props.edge_snapshot.clone();
        let signed_seed_advertisement = props.signed_seed_advertisement.clone();
        let mut session_state = session_state;
        let mut current_view = current_view;
        let mut status = status;
        let mut runtime_config = runtime_config;
        let mut auth_bootstrap_started = auth_bootstrap_started;
        let mut auth_bootstrap_pending = auth_bootstrap_pending;
        use_effect(move || {
            if *auth_bootstrap_started.read() {
                return;
            }
            auth_bootstrap_started.set(true);
            let config = config.clone();
            let bootstrap_config = bootstrap_config.clone();
            let release_manifest = release_manifest.clone();
            let edge_snapshot = edge_snapshot.clone();
            let signed_seed_advertisement = signed_seed_advertisement.clone();
            spawn(async move {
                match resume_or_complete_browser_auth(
                    &config,
                    release_manifest.as_ref(),
                    edge_snapshot.as_ref(),
                )
                .await
                {
                    Ok(Some(session)) => {
                        session_state.set(Some(session));
                        let connect_config = match resolve_browser_app_runtime_config(
                            &config,
                            edge_snapshot.as_ref(),
                        )
                        .await
                        {
                            Ok(config) => {
                                runtime_config.set(config.clone());
                                config
                            }
                            Err(error) => {
                                warn!("browser runtime config resolution failed: {error}");
                                config.clone()
                            }
                        };
                        if let Ok(view) = connect_browser_app(
                            &bootstrap_config,
                            &connect_config,
                            edge_snapshot.as_ref(),
                            signed_seed_advertisement.as_ref(),
                        )
                        .await
                        {
                            current_view.set(Some(view));
                            spawn_browser_app_refresh_loop(
                                bootstrap_config.clone(),
                                connect_config.clone(),
                                edge_snapshot.clone(),
                                signed_seed_advertisement.clone(),
                                current_view,
                                status,
                                checkpoint_wait_generation,
                            );
                        }
                        if provider_code_from_window_location().is_some() {
                            status.set(String::new());
                        }
                    }
                    Ok(None) => {}
                    Err(error) => {
                        if config.require_edge_auth
                            || provider_code_from_window_location().is_some()
                        {
                            status.set(error.to_string());
                        }
                    }
                }
                auth_bootstrap_pending.set(false);
            });
        });
    }

    let connect_action = {
        let props = props.clone();
        move |_| {
            let mut next_config = props.config.clone();
            next_config = next_config.with_network_overrides(
                Some(edge_url.read().clone()),
                DragonPeerNetworkConfig::parse_seed_node_list(&seed_node_urls.read()),
            );
            let bootstrap_config = props.config.clone();
            let mut status = status;
            let mut current_view = current_view;
            let mut session_state = session_state;
            let mut runtime_config = runtime_config;
            let checkpoint_wait_generation = checkpoint_wait_generation;
            let edge_snapshot = props.edge_snapshot.clone();
            let signed_seed_advertisement = props.signed_seed_advertisement.clone();
            spawn(async move {
                status.set("Connecting…".into());
                let next_config =
                    match resolve_browser_app_runtime_config(&next_config, edge_snapshot.as_ref())
                        .await
                    {
                        Ok(config) => {
                            runtime_config.set(config.clone());
                            config
                        }
                        Err(error) => {
                            warn!("browser runtime config resolution failed: {error}");
                            next_config
                        }
                    };
                match connect_browser_app(
                    &bootstrap_config,
                    &next_config,
                    edge_snapshot.as_ref(),
                    signed_seed_advertisement.as_ref(),
                )
                .await
                {
                    Ok(view) => {
                        current_view.set(Some(view));
                        let session = match resolved_edge_base_url(&next_config) {
                            Ok(edge_base_url) => load_browser_session(&edge_base_url)
                                .await
                                .ok()
                                .filter(browser_session_is_authenticated),
                            Err(_) => None,
                        };
                        if let Some(session) = session.as_ref() {
                            info!(
                                "browser auth loaded for connect: principal={} reenrollment_required={} granted_scopes=[{}]",
                                session
                                    .session
                                    .as_ref()
                                    .map(|session| session.claims.principal_id.as_str())
                                    .unwrap_or("anonymous"),
                                session.reenrollment_required,
                                browser_session_scope_summary(session),
                            );
                        }
                        session_state.set(session);
                        status.set(String::new());
                        spawn_browser_app_refresh_loop(
                            bootstrap_config.clone(),
                            next_config,
                            edge_snapshot.clone(),
                            signed_seed_advertisement.clone(),
                            current_view,
                            status,
                            checkpoint_wait_generation,
                        );
                    }
                    Err(error) => status.set(error.to_string()),
                }
            });
        }
    };

    let github_login_action = {
        let props = props.clone();
        move |_| {
            let mut next_config = props.config.clone();
            next_config = next_config.with_network_overrides(
                Some(edge_url.read().clone()),
                DragonPeerNetworkConfig::parse_seed_node_list(&seed_node_urls.read()),
            );
            let release_manifest = props.release_manifest.clone();
            let edge_snapshot = props.edge_snapshot.clone();
            let mut status = status;
            spawn(async move {
                status.set("Starting sign-in…".into());
                if let Err(error) = start_browser_github_auth(
                    &next_config,
                    release_manifest.as_ref(),
                    edge_snapshot.as_ref(),
                )
                .await
                {
                    status.set(error.to_string());
                }
            });
        }
    };

    let admin_github_login_action = {
        let props = props.clone();
        move |_| {
            let mut next_config = props.config.clone();
            next_config = next_config.with_network_overrides(
                Some(edge_url.read().clone()),
                DragonPeerNetworkConfig::parse_seed_node_list(&seed_node_urls.read()),
            );
            let requested_scopes =
                admin_requested_scopes(&next_config, admin_study_id.read().as_str());
            let release_manifest = props.release_manifest.clone();
            let edge_snapshot = props.edge_snapshot.clone();
            let mut admin_status = admin_status;
            spawn(async move {
                admin_status.set("Starting admin sign-in…".into());
                if let Err(error) = start_browser_github_auth_with_scopes(
                    &next_config,
                    release_manifest.as_ref(),
                    edge_snapshot.as_ref(),
                    requested_scopes,
                )
                .await
                {
                    admin_status.set(error.to_string());
                }
            });
        }
    };

    let admin_load_directory_action = {
        let props = props.clone();
        move |_| {
            let mut next_config = props.config.clone();
            next_config = next_config.with_network_overrides(
                Some(edge_url.read().clone()),
                DragonPeerNetworkConfig::parse_seed_node_list(&seed_node_urls.read()),
            );
            let selected_study = admin_study_id.read().clone();
            let selected_experiment = admin_experiment_id.read().clone();
            let session = session_state.read().clone();
            let mut admin_status = admin_status;
            let mut admin_directory_json = admin_directory_json;
            let mut admin_entry_json = admin_entry_json;
            spawn(async move {
                admin_status.set("Loading directory…".into());
                let edge_base_url = match resolved_edge_base_url(&next_config) {
                    Ok(edge_base_url) => edge_base_url,
                    Err(error) => {
                        admin_status.set(error.to_string());
                        return;
                    }
                };
                let directory_result = if let Some(session_id) =
                    session.as_ref().and_then(|session| {
                        session
                            .session_id()
                            .map(|session_id| session_id.as_str().to_owned())
                    }) {
                    fetch_signed_directory_entries(&edge_base_url, &session_id).await
                } else {
                    fetch_directory_entries(&edge_base_url).await
                };
                match directory_result {
                    Ok(entries) => {
                        let directory_json = match directory_entries_to_json(&entries) {
                            Ok(directory_json) => directory_json,
                            Err(error) => {
                                admin_status.set(error.to_string());
                                return;
                            }
                        };
                        let selected_entry =
                            find_directory_entry(&entries, &selected_study, &selected_experiment);
                        admin_directory_json.set(directory_json);
                        if let Some(entry) = selected_entry {
                            match directory_entry_to_json(&entry) {
                                Ok(entry_json) => admin_entry_json.set(entry_json),
                                Err(error) => {
                                    admin_status.set(error.to_string());
                                    return;
                                }
                            }
                        }
                        admin_status.set(format!("Loaded {} directory entries", entries.len()));
                    }
                    Err(error) => admin_status.set(error.to_string()),
                }
            });
        }
    };

    let admin_load_selected_entry_action = move |_| {
        let selected_study = admin_study_id.read().clone();
        let selected_experiment = admin_experiment_id.read().clone();
        let directory_json = admin_directory_json.read().clone();
        match parse_directory_entries_json(&directory_json).and_then(|entries| {
            find_directory_entry(&entries, &selected_study, &selected_experiment).ok_or_else(|| {
                anyhow!(
                    "no directory entry found for study `{}` and experiment `{}`",
                    selected_study,
                    selected_experiment
                )
            })
        }) {
            Ok(entry) => match directory_entry_to_json(&entry) {
                Ok(entry_json) => {
                    admin_study_id.set(entry.study_id.as_str().to_owned());
                    admin_experiment_id.set(entry.experiment_id.as_str().to_owned());
                    admin_entry_json.set(entry_json);
                    admin_status.set("Loaded selected entry into the editor".into());
                }
                Err(error) => admin_status.set(error.to_string()),
            },
            Err(error) => admin_status.set(error.to_string()),
        }
    };

    let admin_upsert_editor_entry_action = move |_| {
        let directory_json = admin_directory_json.read().clone();
        let entry_json = admin_entry_json.read().clone();
        match parse_directory_entry_json(&entry_json) {
            Ok(entry) => match parse_directory_entries_json(&directory_json) {
                Ok(mut entries) => {
                    upsert_directory_entry(&mut entries, entry.clone());
                    match directory_entries_to_json(&entries) {
                        Ok(directory_json) => {
                            admin_study_id.set(entry.study_id.as_str().to_owned());
                            admin_experiment_id.set(entry.experiment_id.as_str().to_owned());
                            admin_directory_json.set(directory_json);
                            admin_status.set("Updated local directory draft".into());
                        }
                        Err(error) => admin_status.set(error.to_string()),
                    }
                }
                Err(error) => admin_status.set(error.to_string()),
            },
            Err(error) => admin_status.set(error.to_string()),
        }
    };

    let admin_rollout_directory_action = {
        let props = props.clone();
        move |_| {
            let mut next_config = props.config.clone();
            next_config = next_config.with_network_overrides(
                Some(edge_url.read().clone()),
                DragonPeerNetworkConfig::parse_seed_node_list(&seed_node_urls.read()),
            );
            let selected_study = admin_study_id.read().clone();
            let selected_experiment = admin_experiment_id.read().clone();
            let directory_json = admin_directory_json.read().clone();
            let entry_json = admin_entry_json.read().clone();
            let session = session_state.read().clone();
            let mut admin_status = admin_status;
            let mut admin_directory_json = admin_directory_json;
            let mut admin_entry_json = admin_entry_json;
            let mut admin_rollout_result = admin_rollout_result;
            let mut current_view = current_view;
            let bootstrap_config = props.config.clone();
            let edge_snapshot = props.edge_snapshot.clone();
            let signed_seed_advertisement = props.signed_seed_advertisement.clone();
            spawn(async move {
                admin_status.set("Rolling out directory…".into());
                if selected_study.trim().is_empty() {
                    admin_status.set("Admin study id is required before rollout".into());
                    return;
                }
                let edge_base_url = match resolved_edge_base_url(&next_config) {
                    Ok(edge_base_url) => edge_base_url,
                    Err(error) => {
                        admin_status.set(error.to_string());
                        return;
                    }
                };
                let Some(session_id) = session.as_ref().and_then(|session| {
                    session
                        .session_id()
                        .map(|session_id| session_id.as_str().to_owned())
                }) else {
                    admin_status.set("No authenticated browser session id found".into());
                    return;
                };
                if !session_has_admin_scope(session.as_ref(), &selected_study) {
                    admin_status.set(format!(
                        "Current session does not grant admin scope for study `{}`",
                        selected_study
                    ));
                    return;
                }
                let mut entries = match parse_directory_entries_json(&directory_json) {
                    Ok(entries) => entries,
                    Err(error) => {
                        admin_status.set(error.to_string());
                        return;
                    }
                };
                if !entry_json.trim().is_empty() {
                    let entry = match parse_directory_entry_json(&entry_json) {
                        Ok(entry) => entry,
                        Err(error) => {
                            admin_status.set(error.to_string());
                            return;
                        }
                    };
                    upsert_directory_entry(&mut entries, entry);
                }
                if entries.is_empty() {
                    admin_status.set("Directory draft is empty".into());
                    return;
                }
                let rollout_result =
                    match rollout_directory_entries(&edge_base_url, &session_id, entries.clone())
                        .await
                    {
                        Ok(result) => result,
                        Err(error) => {
                            admin_status.set(error.to_string());
                            return;
                        }
                    };
                if let Some(result_view) = rollout_result_view(&rollout_result) {
                    admin_rollout_result.set(Some(result_view));
                }
                match fetch_signed_directory_entries(&edge_base_url, &session_id).await {
                    Ok(entries) => {
                        match directory_entries_to_json(&entries) {
                            Ok(directory_json) => admin_directory_json.set(directory_json),
                            Err(error) => {
                                admin_status.set(error.to_string());
                                return;
                            }
                        }
                        if let Some(entry) =
                            find_directory_entry(&entries, &selected_study, &selected_experiment)
                        {
                            match directory_entry_to_json(&entry) {
                                Ok(entry_json) => admin_entry_json.set(entry_json),
                                Err(error) => {
                                    admin_status.set(error.to_string());
                                    return;
                                }
                            }
                        }
                        if let Ok(view) = refresh_browser_app(
                            &bootstrap_config,
                            &next_config,
                            edge_snapshot.as_ref(),
                            signed_seed_advertisement.as_ref(),
                        )
                        .await
                        {
                            current_view.set(Some(view));
                        }
                        admin_status.set(format!("Rolled out {} directory entries", entries.len()));
                    }
                    Err(error) => admin_status.set(error.to_string()),
                }
            });
        }
    };

    let complete_callback_action = {
        let props = props.clone();
        move |_| {
            let mut next_config = props.config.clone();
            next_config = next_config.with_network_overrides(
                Some(edge_url.read().clone()),
                DragonPeerNetworkConfig::parse_seed_node_list(&seed_node_urls.read()),
            );
            let release_manifest = props.release_manifest.clone();
            let edge_snapshot = props.edge_snapshot.clone();
            let mut status = status;
            let mut session_state = session_state;
            spawn(async move {
                status.set("Completing sign-in…".into());
                match resume_or_complete_browser_auth(
                    &next_config,
                    release_manifest.as_ref(),
                    edge_snapshot.as_ref(),
                )
                .await
                {
                    Ok(Some(session)) => {
                        session_state.set(Some(session));
                        status.set(String::new());
                    }
                    Ok(None) => status.set(String::new()),
                    Err(error) => status.set(error.to_string()),
                }
            });
        }
    };

    let view = current_view.read().clone();
    let callback_available = provider_code_from_window_location().is_some();
    let auth_required = props.config.require_edge_auth;
    let admin_granted_studies = granted_admin_studies(session_state.read().as_ref());
    let admin_granted_studies_label = admin_granted_studies.join(", ");
    let admin_scope_ready = session_has_admin_scope(
        session_state.read().as_ref(),
        admin_study_id.read().as_str(),
    );
    let admin_scope_label = if admin_scope_ready { "yes" } else { "no" };
    let admin_session_card_view = admin_session_summary_view(
        session_state.read().as_ref(),
        admin_study_id.read().as_str(),
    );
    let admin_directory_entries = parse_directory_entries_json(&admin_directory_json.read()).ok();
    let admin_entry_draft_view = parse_directory_entry_json(&admin_entry_json.read())
        .ok()
        .map(|entry| DirectoryEntryDraftView::from_entry(&entry));
    let admin_directory_list_view = admin_directory_entries.as_ref().map(|entries| {
        directory_list_view(
            entries,
            Some(admin_experiment_id.read().clone()).filter(|value| !value.trim().is_empty()),
            admin_entry_draft_view
                .as_ref()
                .map(|draft| draft.revision_id.clone()),
        )
    });
    let admin_rollout_preview = admin_directory_entries
        .as_ref()
        .filter(|entries| !entries.is_empty())
        .map(|entries| rollout_preview_view(entries));
    let admin_rollout_status_view = admin_rollout_result.read().clone();
    let show_connection_settings_active = *show_connection_settings.read();
    let show_admin_tools_active = *show_admin_tools.read();
    let display_config = runtime_config.read().clone();
    let (browser_host_capabilities, browser_capability_decision) =
        browser_capability_decision_for_config(&display_config);
    let browser_can_attempt_dynamic_training = display_config.training.is_some()
        || (browser_host_capabilities.navigator_gpu_exposed
            && browser_host_capabilities.worker_gpu_exposed
            && browser_host_capabilities.dedicated_worker_exposed);
    let browser_downgrade_reason = browser_capability_decision.downgrade_reason.clone();
    let capability_budget_label = browser_capability_decision
        .trainer_memory_budget_bytes
        .map(|bytes| format!("{} MiB", bytes / (1024 * 1024)))
        .unwrap_or_else(|| "n/a".into());
    let capability_window_label = browser_capability_decision
        .training_budget
        .as_ref()
        .map(|budget| budget.max_window_secs.to_string())
        .unwrap_or_else(|| "n/a".into());
    let capability_checkpoint_label = browser_capability_decision
        .footprint
        .as_ref()
        .map(|footprint| {
            format!(
                "{} MiB",
                footprint.estimated_checkpoint_bytes / (1024 * 1024)
            )
        })
        .unwrap_or_else(|| "n/a".into());
    let capability_shard_label = browser_capability_decision
        .footprint
        .as_ref()
        .map(|footprint| format!("{} MiB", footprint.estimated_shard_bytes / (1024 * 1024)))
        .unwrap_or_else(|| "n/a".into());
    let capability_training_label = if browser_capability_decision.can_train {
        "train-ready".to_owned()
    } else {
        "blocked".to_owned()
    };
    let capability_reason_label = browser_downgrade_reason.clone().unwrap_or_else(|| {
        if browser_capability_decision.can_train {
            "WebGPU and worker requirements satisfied".into()
        } else if display_config.training.is_none() {
            if view.is_some() {
                "browser training is not configured for this deployment".into()
            } else {
                "resolving browser training profile from the live network".into()
            }
        } else {
            "browser trainer capability check blocked local training".into()
        }
    });
    let capability_navigator_gpu_label = if browser_host_capabilities.navigator_gpu_exposed {
        "available"
    } else {
        "not available"
    }
    .to_owned();
    let capability_worker_gpu_label = if browser_host_capabilities.worker_gpu_exposed {
        "available"
    } else {
        "not available"
    }
    .to_owned();
    let capability_worker_label = if browser_host_capabilities.dedicated_worker_exposed {
        "available"
    } else {
        "not available"
    }
    .to_owned();
    let capability_storage_label = if browser_host_capabilities.persistent_storage_exposed {
        "available"
    } else {
        "not available"
    }
    .to_owned();
    let active_head_label = view
        .as_ref()
        .and_then(|view| {
            view.training
                .latest_head_id
                .clone()
                .or_else(|| view.training.last_artifact_id.clone())
        })
        .unwrap_or_else(|| "awaiting checkpoint".into());
    let network_summary = dragon_network_detail(view.as_ref());
    let transport_summary = dragon_transport_summary(view.as_ref());
    let has_session = session_state
        .read()
        .as_ref()
        .and_then(|session| session.session.as_ref())
        .is_some();
    let session_metric_view = dragon_session_metric_view(
        &admin_session_card_view,
        view.as_ref().map(|view| view.session_label.as_str()),
        has_session,
    );
    let auth_bootstrap_pending_active = *auth_bootstrap_pending.read();
    let has_connected_view = view.is_some();
    let public_landing = !auth_bootstrap_pending_active && !has_session && !has_connected_view;
    let needs_sign_in = !auth_bootstrap_pending_active && auth_required && !has_session;
    let ready_to_connect = !auth_bootstrap_pending_active && !needs_sign_in && !has_connected_view;
    let raw_status_message = status.read().clone();
    let status_message = if public_landing
        && (raw_status_message.contains("failed to fetch edge snapshot")
            || raw_status_message.contains("failed to decode edge snapshot")
            || raw_status_message.contains("empty response body")
            || raw_status_message.contains("Failed to fetch")
            || raw_status_message.contains("tls")
            || raw_status_message.contains("connection"))
    {
        String::from("the edge is unavailable right now. try again soon.")
    } else if raw_status_message.contains("/metrics/catchup/")
        || raw_status_message.contains("metrics indexer disabled")
    {
        String::from("connect is unavailable right now. try again soon.")
    } else {
        raw_status_message
    };
    let show_public_retry =
        !auth_bootstrap_pending_active && callback_available && !status_message.is_empty();
    #[cfg(feature = "wasm-peer")]
    let local_training_state_active = local_training_state.read().clone();
    #[cfg(feature = "wasm-peer")]
    let local_training_pending_active = local_training_state_active.is_active();
    #[cfg(feature = "wasm-peer")]
    let local_training_failure = local_training_state_active
        .failure_message()
        .map(str::to_owned);
    #[cfg(not(feature = "wasm-peer"))]
    let local_training_pending_active = false;
    #[cfg(not(feature = "wasm-peer"))]
    let local_training_failure: Option<String> = None;
    let direct_transport_ready = view
        .as_ref()
        .is_some_and(|view| view.network.direct_peers > 0);
    let edge_configured = resolved_edge_base_url(&initial_config).is_ok();
    let requires_active_head_artifact =
        dragon_browser_training_requires_active_head_artifact(&initial_config);
    let training_action_state = dragon_training_action_state(DragonTrainingActionContext {
        view: view.as_ref(),
        browser_can_attempt_dynamic_training,
        edge_configured,
        direct_transport_ready,
        requires_active_head_artifact,
        local_training_pending: local_training_pending_active,
        local_training_failure: local_training_failure.as_deref(),
        downgrade_reason: browser_downgrade_reason.as_deref(),
    });
    let peer_ui_context = DragonPeerUiContext {
        view: view.as_ref(),
        status_message: &status_message,
        has_session,
        auth_bootstrap_pending: auth_bootstrap_pending_active,
        needs_sign_in,
        ready_to_connect,
        edge_configured,
        browser_can_attempt_dynamic_training,
        direct_transport_ready,
        requires_active_head_artifact,
        local_training_pending: local_training_pending_active,
        local_training_failure: local_training_failure.as_deref(),
        downgrade_reason: browser_downgrade_reason.as_deref(),
        training_action_state: training_action_state.as_ref(),
        session_metric: session_metric_view,
    };
    let peer_ui_state = dragon_peer_ui_state(&peer_ui_context);
    let hero_rattle_active = peer_ui_state.hero.animate
        || status_message.starts_with("Connecting")
        || status_message.starts_with("Starting sign-in");
    #[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
    {
        let mut hero_rattle_state = hero_rattle_state;
        use_effect(move || {
            if *hero_rattle_state.read() == Some(hero_rattle_active) {
                return;
            }
            hero_rattle_state.set(Some(hero_rattle_active));
            spawn_hero_rattle_loop(
                hero_rattle_active,
                hero_rattle_index,
                hero_rattle_generation,
            );
        });
    }
    {
        let mut ui_events = ui_events;
        let mut last_ui_event_key = last_ui_event_key;
        let event_candidate = peer_ui_state.event_candidate.clone();
        use_effect(move || {
            if *last_ui_event_key.read() == event_candidate.key {
                return;
            }
            last_ui_event_key.set(event_candidate.key.clone());
            let current_events = ui_events.read().clone();
            let next_events =
                dragon_push_ui_event(current_events.clone(), &event_candidate, dragon_ui_now_ms());
            if next_events != current_events {
                ui_events.set(next_events);
            }
        });
    }
    let activity_events = ui_events.read().clone();
    let direct_transport_error = view
        .as_ref()
        .and_then(|view| active_direct_transport_error(view).map(str::to_owned));
    let show_reset_browser_state_button = debug_controls_enabled
        && (direct_transport_error.is_some() || browser_downgrade_reason.is_some());
    let browser_machine_state_json = view.as_ref().map(browser_view_machine_state_json);
    let runtime_mode_summary = dragon_runtime_mode_summary(
        view.as_ref(),
        direct_transport_ready,
        training_action_state.as_ref(),
        false,
        local_training_pending_active,
    );
    let runtime_mode_detail = dragon_runtime_mode_detail(
        view.as_ref(),
        direct_transport_ready,
        training_action_state.as_ref(),
        local_training_pending_active,
        browser_downgrade_reason.as_deref(),
    );
    let local_training_summary =
        dragon_local_training_summary(view.as_ref(), local_training_pending_active);
    let local_training_detail =
        dragon_local_training_detail(view.as_ref(), training_action_state.as_ref());
    let global_training_summary = dragon_global_training_summary(view.as_ref());
    let global_training_detail = dragon_global_training_detail(view.as_ref());
    let window_summary = dragon_window_summary(view.as_ref(), local_training_pending_active);
    let slice_progress_summary = dragon_slice_progress_summary(view.as_ref());
    let window_progress_detail = dragon_window_progress_detail(view.as_ref(), &window_summary);
    #[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
    {
        let mut last_logged_browser_status = last_logged_browser_status;
        let status_message = status_message.clone();
        use_effect(move || {
            let previous = last_logged_browser_status.read().clone();
            if previous == status_message {
                return;
            }
            last_logged_browser_status.set(status_message.clone());
            if !status_message.is_empty() {
                warn!("browser ui status: {status_message}");
            }
        });
    }
    #[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
    {
        let mut last_logged_transport_error = last_logged_transport_error;
        let mut last_logged_runtime_summary = last_logged_runtime_summary;
        let view = view.clone();
        let logged_direct_transport_error = direct_transport_error.clone();
        use_effect(move || {
            let runtime_summary = view.as_ref().map(browser_view_log_summary);
            if *last_logged_runtime_summary.read() != runtime_summary {
                last_logged_runtime_summary.set(runtime_summary.clone());
                if let Some(runtime_summary) = runtime_summary {
                    info!("browser runtime state: {runtime_summary}");
                }
            }
            let transport_error = logged_direct_transport_error.clone();
            if *last_logged_transport_error.read() != transport_error {
                last_logged_transport_error.set(transport_error.clone());
                if let Some(transport_error) = transport_error {
                    error!("browser direct transport error: {transport_error}");
                }
            }
        });
    }
    #[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
    let hero_rattle_frame =
        HERO_RATTLE_FRAMES[*hero_rattle_index.read() % HERO_RATTLE_FRAMES.len()];
    #[cfg(not(all(feature = "wasm-ui", target_arch = "wasm32")))]
    let hero_rattle_frame = HERO_RATTLE_FRAMES[0];
    #[cfg(feature = "wasm-peer")]
    let train_action = {
        let props = props.clone();
        move |_| {
            let mut next_config = props.config.clone();
            next_config = next_config.with_network_overrides(
                Some(edge_url.read().clone()),
                DragonPeerNetworkConfig::parse_seed_node_list(&seed_node_urls.read()),
            );
            let bootstrap_config = props.config.clone();
            let release_manifest = props.release_manifest.clone();
            let edge_snapshot = props.edge_snapshot.clone();
            let signed_seed_advertisement = props.signed_seed_advertisement.clone();
            let mut status = status;
            let mut current_view = current_view;
            let mut local_training = local_training;
            let mut local_training_state = local_training_state;
            let mut local_training_stop_requested = local_training_stop_requested;
            if local_training_state.read().is_active() {
                local_training_stop_requested.set(true);
                local_training_state.set(DragonLocalTrainingState::Stopping);
                status.set("Stopping browser training after the current window…".into());
                return;
            }
            spawn(async move {
                local_training_state.set(DragonLocalTrainingState::Starting);
                let release_manifest = match resolve_browser_release_manifest(
                    &next_config,
                    release_manifest.as_ref(),
                    edge_snapshot.as_ref(),
                )
                .await
                {
                    Ok(release_manifest) => release_manifest,
                    Err(error) => {
                        let message = error.to_string();
                        error!("browser training start failed: {message}");
                        status.set(message.clone());
                        local_training_state.set(DragonLocalTrainingState::Failed { message });
                        return;
                    }
                };
                local_training_stop_requested.set(false);
                let edge_base_url = match resolved_edge_base_url(&next_config) {
                    Ok(edge_base_url) => edge_base_url,
                    Err(error) => {
                        let message = error.to_string();
                        error!("browser training edge resolution failed: {message}");
                        status.set(message.clone());
                        local_training_state.set(DragonLocalTrainingState::Failed { message });
                        return;
                    }
                };
                let mut completed_windows = 0_u64;
                let mut failed = false;
                loop {
                    if *local_training_stop_requested.read() {
                        break;
                    }
                    local_training_state.set(DragonLocalTrainingState::SyncingCheckpoint);
                    let training = match resolve_browser_training_config(
                        &bootstrap_config,
                        &next_config,
                        edge_snapshot.as_ref(),
                        signed_seed_advertisement.as_ref(),
                    )
                    .await
                    {
                        Ok(training) => training,
                        Err(error) => {
                            let message = error.to_string();
                            error!("browser training config resolution failed: {message}");
                            status.set(message.clone());
                            local_training_state.set(DragonLocalTrainingState::Failed { message });
                            failed = true;
                            break;
                        }
                    };
                    let next_window = completed_windows.saturating_add(1);
                    local_training_state.set(DragonLocalTrainingState::TrainingWindow);
                    status.set(format!("Running browser training window {}…", next_window));
                    match run_browser_training_with_release_manifest(
                        &edge_base_url,
                        &training,
                        &release_manifest,
                    )
                    .await
                    {
                        Ok(result) => {
                            completed_windows = completed_windows.saturating_add(1);
                            let status_message = if result.train_loss_observed {
                                format!(
                                    "Browser training window {} complete: mean train loss {:.4}",
                                    completed_windows, result.train_loss_mean
                                )
                            } else {
                                format!(
                                    "Browser training window {} complete: WebGPU window finished",
                                    completed_windows
                                )
                            };
                            status.set(status_message);
                            local_training.set(Some(result));
                            if let Ok(view) = refresh_browser_app(
                                &bootstrap_config,
                                &next_config,
                                edge_snapshot.as_ref(),
                                signed_seed_advertisement.as_ref(),
                            )
                            .await
                            {
                                current_view.set(Some(view));
                            }
                        }
                        Err(error) => {
                            let message = error.to_string();
                            error!("browser training window failed: {message}");
                            status.set(message.clone());
                            local_training_state.set(DragonLocalTrainingState::Failed { message });
                            if let Ok(view) = refresh_browser_app(
                                &bootstrap_config,
                                &next_config,
                                edge_snapshot.as_ref(),
                                signed_seed_advertisement.as_ref(),
                            )
                            .await
                            {
                                current_view.set(Some(view));
                            }
                            failed = true;
                            break;
                        }
                    }
                    #[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
                    gloo_timers::future::TimeoutFuture::new(25).await;
                }
                if *local_training_stop_requested.read() {
                    local_training_state.set(DragonLocalTrainingState::Stopped);
                    if completed_windows == 0 {
                        status.set("Browser training stopped before a window completed".into());
                    } else {
                        status.set(format!(
                            "Browser training stopped after {completed_windows} window(s)"
                        ));
                    }
                } else if !failed {
                    local_training_state.set(DragonLocalTrainingState::Idle);
                }
                local_training_stop_requested.set(false);
            });
        }
    };

    #[cfg(all(feature = "wasm-peer", feature = "wasm-ui", target_arch = "wasm32"))]
    let reset_browser_state_action = {
        let props = props.clone();
        move |_| {
            let mut next_config = props.config.clone();
            next_config = next_config.with_network_overrides(
                Some(edge_url.read().clone()),
                DragonPeerNetworkConfig::parse_seed_node_list(&seed_node_urls.read()),
            );
            let mut status = status;
            let mut current_view = current_view;
            let mut session_state = session_state;
            let mut checkpoint_wait_generation = checkpoint_wait_generation;
            spawn(async move {
                let edge_base_url = match resolved_edge_base_url(&next_config) {
                    Ok(edge_base_url) => edge_base_url,
                    Err(error) => {
                        status.set(error.to_string());
                        return;
                    }
                };
                let downgrade_clear_result = next_config.training.as_ref().map(|training| {
                    clear_browser_downgrade(
                        &edge_base_url,
                        training,
                        browser_backend_label(training),
                    )
                });
                let runtime_reset_result = reset_browser_runtime_state(&edge_base_url).await;
                match (downgrade_clear_result.transpose(), runtime_reset_result) {
                    (Ok(_), Ok(())) => {
                        clear_live_browser_app_controller();
                        let next_generation =
                            (*checkpoint_wait_generation.read()).saturating_add(1);
                        checkpoint_wait_generation.set(next_generation);
                        current_view.set(None);
                        session_state.set(None);
                        status.set(
                            "reset local browser state. reconnect to retry browser training."
                                .into(),
                        );
                    }
                    (Err(error), _) | (_, Err(error)) => status.set(error.to_string()),
                }
            });
        }
    };
    #[cfg(not(all(feature = "wasm-peer", feature = "wasm-ui", target_arch = "wasm32")))]
    let reset_browser_state_action = move |_| {};

    #[cfg(feature = "wasm-peer")]
    let train_button = {
        let training_action_state = training_action_state.clone();
        if has_connected_view {
            if let Some(training_action_state) = training_action_state {
                let button_label = training_action_state.label;
                let button_detail = training_action_state.detail.clone();
                rsx! {
                    if training_action_state.enabled {
                        button {
                            r#type: "button",
                            class: "action-button action-button-primary",
                            onclick: train_action,
                            "{button_label}"
                        }
                        p { class: "dragon-live-action-note", "{button_detail}" }
                    } else {
                        div { class: "dragon-live-action-status",
                            span { "{button_label}" }
                            p { "{button_detail}" }
                        }
                        if show_reset_browser_state_button {
                            button {
                                r#type: "button",
                                class: "action-button action-button-secondary dragon-live-reset-button",
                                onclick: reset_browser_state_action,
                                "reset local state"
                            }
                        }
                    }
                }
            } else {
                rsx! {}
            }
        } else {
            rsx! {}
        }
    };
    #[cfg(not(feature = "wasm-peer"))]
    let train_button = rsx! {};

    #[cfg(feature = "wasm-peer")]
    let local_training_section = if let Some(result) = local_training.read().clone() {
        let eval_loss_label = result
            .eval_loss
            .map(|value| format!("{value:.4}"))
            .unwrap_or_else(|| "n/a".into());
        let train_loss_label = if result.train_loss_observed {
            format!("{:.4}", result.train_loss_mean)
        } else {
            "not sampled".into()
        };
        let tokens_per_second_label = result
            .tokens_per_second
            .map(|value| format!("{value:.1}"))
            .unwrap_or_else(|| "n/a".into());
        let train_batches_label = result.train_batches.to_string();
        let live_training_details = result.live_participant.map(|live| {
            let receipt_state = if live.receipt_submission_accepted {
                "accepted".to_owned()
            } else if live.receipt_submission_deferred {
                "pending retry".to_owned()
            } else {
                "not accepted".to_owned()
            };
            let artifact_state = if live.artifact_published {
                "published"
            } else {
                "not published"
            };
            let update_state = if live.update_announced {
                "announced"
            } else {
                "not announced"
            };
            (
                receipt_state,
                live.accepted_receipt_ids.join(", "),
                live.pending_receipt_count,
                live.receipt_submission_error,
                live.runtime_state.unwrap_or_else(|| "n/a".into()),
                artifact_state,
                update_state,
            )
        });
        rsx! {
            section { class: "panel compact-panel",
                SectionHeader {
                    eyebrow: "local",
                    title: "browser training",
                    detail: "latest browser training window executed in this tab.",
                }
                div { class: "keyvalue-list",
                    div { class: "keyvalue-row",
                        span { "experiment" }
                        strong { "{result.experiment_kind_label}" }
                    }
                    div { class: "keyvalue-row",
                        span { "backend" }
                        strong { "{result.backend}" }
                    }
                    div { class: "keyvalue-row",
                        span { "train loss" }
                        strong { "{train_loss_label}" }
                    }
                    div { class: "keyvalue-row",
                        span { "eval loss" }
                        strong { "{eval_loss_label}" }
                    }
                    div { class: "keyvalue-row",
                        span { "train batches" }
                        strong { "{train_batches_label}" }
                    }
                    div { class: "keyvalue-row",
                        span { "tokens/sec" }
                        strong { "{tokens_per_second_label}" }
                    }
                }
                if let Some((receipt_state, accepted_receipts_label, pending_receipt_count, receipt_submission_error, runtime_state_label, artifact_state, update_state)) = live_training_details {
                    div { class: "keyvalue-list",
                        div { class: "keyvalue-row",
                            span { "receipt state" }
                            strong { "{receipt_state}" }
                        }
                        div { class: "keyvalue-row",
                            span { "artifact" }
                            strong { "{artifact_state}" }
                        }
                        div { class: "keyvalue-row",
                            span { "p2p update" }
                            strong { "{update_state}" }
                        }
                        div { class: "keyvalue-row",
                            span { "accepted receipts" }
                            strong { "{accepted_receipts_label}" }
                        }
                        div { class: "keyvalue-row",
                            span { "pending receipts" }
                            strong { "{pending_receipt_count}" }
                        }
                        if let Some(receipt_submission_error) = receipt_submission_error {
                            div { class: "keyvalue-row",
                                span { "receipt retry" }
                                strong { "{receipt_submission_error}" }
                            }
                        }
                        div { class: "keyvalue-row",
                            span { "runtime state" }
                            strong { "{runtime_state_label}" }
                        }
                    }
                }
            }
        }
    } else {
        rsx! {}
    };
    #[cfg(not(feature = "wasm-peer"))]
    let local_training_section = rsx! {};
    let footer_build_rev = build_info::footer_build_rev(props.release_manifest.as_ref());

    rsx! {
        main { class: "browser-app-shell dragon-browser-shell",
            section { class: "panel hero browser-hero",
                div { class: "browser-hero-grid",
                    div { class: "browser-hero-copy",
                        div { class: "dragon-eyebrow-row",
                            div { class: "eyebrow", "burn_dragon" }
                            if hero_rattle_active {
                                span {
                                    class: "dragon-eyebrow-rattle",
                                    "{hero_rattle_frame}"
                                }
                            }
                        }
                        h1 { class: "app-title", "train the dragon" }
                    }
                }
                if needs_sign_in || ready_to_connect || (debug_controls_enabled && has_connected_view) {
                    div { class: "browser-hero-bar",
                        div { class: "dragon-connection-editor",
                            div { class: "browser-action-row",
                                if needs_sign_in {
                                    if callback_available {
                                        if show_public_retry {
                                            ActionButton {
                                                label: "try again",
                                                tone: "secondary",
                                                onclick: complete_callback_action,
                                            }
                                        }
                                    } else {
                                        ActionButton {
                                            label: "get started",
                                            tone: "primary",
                                            onclick: github_login_action,
                                        }
                                    }
                                } else if ready_to_connect {
                                    ActionButton {
                                        label: "connect",
                                        tone: "primary",
                                        onclick: connect_action,
                                    }
                                }
                                if debug_controls_enabled {
                                    if (ready_to_connect || has_connected_view) && show_connection_settings_active {
                                        button {
                                            r#type: "button",
                                            class: "action-button action-button-secondary",
                                            onclick: move |_| show_connection_settings.set(false),
                                            "hide debug"
                                        }
                                    } else if ready_to_connect || has_connected_view {
                                        button {
                                            r#type: "button",
                                            class: "action-button action-button-secondary",
                                            onclick: move |_| show_connection_settings.set(true),
                                            "debug"
                                        }
                                    }
                                }
                            }
                            if debug_controls_enabled && show_connection_settings_active {
                                div { class: "edge-editor dragon-advanced-settings",
                                    EdgeConnectField {
                                        label: "edge url",
                                        value: edge_url.read().clone(),
                                        oninput: move |value| edge_url.set(value),
                                    }
                                    EdgeConnectField {
                                        label: "seed node urls",
                                        value: seed_node_urls.read().clone(),
                                        oninput: move |value| seed_node_urls.set(value),
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if !status_message.is_empty() {
                ActivityNotice {
                    label: String::from("status"),
                    detail: status_message.clone(),
                    tone: "accent",
                }
            }
            if debug_controls_enabled && (has_session || has_connected_view) && browser_capability_decision.downgrade_reason.is_some() {
                ActivityNotice {
                    label: String::from("capability policy"),
                    detail: browser_capability_decision.downgrade_reason.clone().unwrap_or_default(),
                    tone: "neutral",
                }
            }
            if has_connected_view {
                div { class: "dragon-live-actions",
                    {train_button}
                }
            }
            ReadinessLadder { steps: peer_ui_state.readiness.clone() }
            MetricGrid { metrics: peer_ui_state.metrics.clone() }
            ActivityFeed { events: activity_events }
            if has_connected_view {
                div { class: "dragon-live-stats dragon-canary-diagnostics", "aria-hidden": "true",
                    StatTile {
                        label: "status",
                        value: runtime_mode_summary.clone(),
                        detail: Some(runtime_mode_detail.clone()),
                    }
                    StatTile {
                        label: "transport",
                        value: transport_summary.clone(),
                        detail: Some(network_summary.clone()),
                    }
                    StatTile {
                        label: "local train",
                        value: local_training_summary.clone(),
                        detail: Some(local_training_detail.clone()),
                    }
                    StatTile {
                        label: "global train",
                        value: global_training_summary.clone(),
                        detail: Some(global_training_detail.clone()),
                    }
                    StatTile {
                        label: "window",
                        value: slice_progress_summary.clone(),
                        detail: Some(window_progress_detail.clone()),
                    }
                    StatTile {
                        label: "peers",
                        value: network_summary.clone(),
                        detail: Some(transport_summary.clone()),
                    }
                }
                if let Some(machine_state) = browser_machine_state_json.as_ref() {
                    pre { class: "dragon-live-machine-state dragon-canary-diagnostics", "aria-hidden": "true", "{machine_state}" }
                }
            }
            if has_connected_view || debug_controls_enabled {
                details { class: "panel dragon-diagnostics-drawer", open: debug_controls_enabled,
                    summary { class: "dragon-diagnostics-summary",
                        span { "advanced diagnostics" }
                        small { "state · network · training · session" }
                    }
                    div { class: "dragon-diagnostics-grid",
                        section { class: "dragon-diagnostics-section dragon-diagnostics-section-state",
                            SectionHeader {
                                eyebrow: "state",
                                title: "machine state",
                                detail: "raw browser peer state for operators.",
                            }
                            if let Some(machine_state) = browser_machine_state_json.as_ref() {
                                pre { class: "operator-raw dragon-machine-state", "{machine_state}" }
                            } else {
                                EmptyState {
                                    title: "no runtime state",
                                    detail: "connect the browser peer to inspect raw state.",
                                }
                            }
                        }
                        section { class: "dragon-diagnostics-section dragon-diagnostics-section-network",
                            SectionHeader {
                                eyebrow: "network",
                                title: "transport",
                                detail: "edge and direct peer transport details.",
                            }
                            div { class: "keyvalue-list dragon-live-keyvalues",
                                div { class: "keyvalue-row",
                                    span { "edge url" }
                                    strong { code { "{edge_url.read().clone()}" } }
                                }
                                div { class: "keyvalue-row",
                                    span { "seed urls" }
                                    strong { code { "{seed_node_urls.read().clone()}" } }
                                }
                                div { class: "keyvalue-row",
                                    span { "transport" }
                                    strong { "{transport_summary}" }
                                }
                                div { class: "keyvalue-row",
                                    span { "peers" }
                                    strong { "{network_summary}" }
                                }
                                if let Some(error) = direct_transport_error.as_ref() {
                                    div { class: "keyvalue-row",
                                        span { "last error" }
                                        strong { code { "{error}" } }
                                    }
                                }
                            }
                        }
                        section { class: "dragon-diagnostics-section dragon-diagnostics-section-training",
                            SectionHeader {
                                eyebrow: "training",
                                title: "local slice",
                                detail: "assignment, checkpoint, and local training counters.",
                            }
                            if let Some(view) = view.clone() {
                                div { class: "keyvalue-list dragon-live-keyvalues",
                                    div { class: "keyvalue-row",
                                        span { "assignment" }
                                        strong { "{view.training.active_assignment.is_some().then_some(\"assigned\").unwrap_or(\"waiting\")}" }
                                    }
                                    div { class: "keyvalue-row",
                                        span { "head" }
                                        strong { code { "{active_head_label}" } }
                                    }
                                    div { class: "keyvalue-row",
                                        span { "cached microshards" }
                                        strong { "{view.training.cached_microshards}" }
                                    }
                                    div { class: "keyvalue-row",
                                        span { "accepted samples" }
                                        strong { "{view.training.accepted_samples.map(|value| value.to_string()).unwrap_or_else(|| \"n/a\".into())}" }
                                    }
                                    div { class: "keyvalue-row",
                                        span { "optimizer steps" }
                                        strong { "{view.training.optimizer_steps.map(|value| value.to_string()).unwrap_or_else(|| \"n/a\".into())}" }
                                    }
                                }
                            } else {
                                EmptyState {
                                    title: "no training state",
                                    detail: "training details appear after connect.",
                                }
                            }
                        }
                        section { class: "dragon-diagnostics-section dragon-diagnostics-section-capability",
                            SectionHeader {
                                eyebrow: "capability",
                                title: "browser budget",
                                detail: "policy and memory estimates used by the browser trainer.",
                            }
                            div { class: "browser-metric-band dragon-metric-band",
                                StatTile {
                                    label: "recommended role",
                                    value: browser_runtime_role_label(&browser_capability_decision.capability.recommended_role).replace('_', " "),
                                    detail: Some("capability".into()),
                                }
                                StatTile {
                                    label: "memory budget",
                                    value: capability_budget_label,
                                    detail: Some("trainer".into()),
                                }
                                StatTile {
                                    label: "checkpoint",
                                    value: capability_checkpoint_label,
                                    detail: Some("budget".into()),
                                }
                                StatTile {
                                    label: "window",
                                    value: capability_window_label.clone(),
                                    detail: Some("secs".into()),
                                }
                                StatTile {
                                    label: "shard",
                                    value: capability_shard_label,
                                    detail: Some("budget".into()),
                                }
                            }
                            div { class: "keyvalue-list dragon-live-keyvalues",
                                div { class: "keyvalue-row",
                                    span { "training decision" }
                                    strong { "{capability_training_label}" }
                                }
                                div { class: "keyvalue-row",
                                    span { "why" }
                                    strong { "{capability_reason_label}" }
                                }
                                div { class: "keyvalue-row",
                                    span { "page webgpu" }
                                    strong { "{capability_navigator_gpu_label}" }
                                }
                                div { class: "keyvalue-row",
                                    span { "worker webgpu" }
                                    strong { "{capability_worker_gpu_label}" }
                                }
                                div { class: "keyvalue-row",
                                    span { "dedicated worker" }
                                    strong { "{capability_worker_label}" }
                                }
                                div { class: "keyvalue-row",
                                    span { "persistent storage" }
                                    strong { "{capability_storage_label}" }
                                }
                            }
                        }
                    }
                }
            }
            {local_training_section}
            if show_admin_tools_active {
                div { class: "surface-layout browser-surface-layout dragon-admin-surface",
                    section { class: "panel primary-panel browser-focus-panel",
                        SectionHeader {
                            eyebrow: "admin",
                            title: "directory rollout",
                            detail: "load, edit, preview, and publish signed experiment directory entries.",
                        }
                        AdminSessionCard { session: admin_session_card_view }
                        div { class: "dragon-operator-summary keyvalue-list",
                            div { class: "keyvalue-row",
                                span { "selected study" }
                                strong { "{admin_study_id}" }
                            }
                            div { class: "keyvalue-row",
                                span { "admin scope ready" }
                                strong { "{admin_scope_label}" }
                            }
                            if !admin_granted_studies.is_empty() {
                                div { class: "keyvalue-row",
                                    span { "granted studies" }
                                    strong { "{admin_granted_studies_label}" }
                                }
                            }
                        }
                        if !admin_status.read().is_empty() {
                            ActivityNotice {
                                label: String::from("status"),
                                detail: admin_status.read().clone(),
                                tone: "neutral",
                            }
                        }
                        div { class: "dragon-editor-grid",
                            EditorInputField {
                                label: "admin study id",
                                value: admin_study_id.read().clone(),
                                oninput: move |value| admin_study_id.set(value),
                            }
                            EditorInputField {
                                label: "experiment id",
                                value: admin_experiment_id.read().clone(),
                                oninput: move |value| admin_experiment_id.set(value),
                            }
                        }
                        div { class: "browser-action-row dragon-admin-actions",
                            ActionButton {
                                label: "sign in as admin",
                                tone: "secondary",
                                onclick: admin_github_login_action,
                            }
                            ActionButton {
                                label: "load directory",
                                tone: "secondary",
                                onclick: admin_load_directory_action,
                            }
                            ActionButton {
                                label: "load selected entry",
                                tone: "secondary",
                                onclick: admin_load_selected_entry_action,
                            }
                            ActionButton {
                                label: "upsert editor entry",
                                tone: "secondary",
                                onclick: admin_upsert_editor_entry_action,
                            }
                            ActionButton {
                                label: "roll out directory",
                                tone: "primary",
                                onclick: admin_rollout_directory_action,
                            }
                        }
                        div { class: "dragon-editor-grid dragon-editor-grid-wide",
                            EditorTextareaField {
                                label: "directory json",
                                value: admin_directory_json.read().clone(),
                                rows: "18",
                                oninput: move |value| admin_directory_json.set(value),
                            }
                            EditorTextareaField {
                                label: "entry editor json",
                                value: admin_entry_json.read().clone(),
                                rows: "16",
                                oninput: move |value| admin_entry_json.set(value),
                            }
                        }
                        div { class: "dragon-panel-stack",
                            if let Some(view) = admin_directory_list_view.clone() {
                                ExperimentDirectoryListPanel { view }
                            }
                            if let Some(draft) = admin_entry_draft_view.clone() {
                                DirectoryEntryDraftPanel { draft }
                            }
                            if let Some(view) = admin_rollout_preview.clone() {
                                RolloutPreviewPanel { view }
                            }
                            if let Some(view) = admin_rollout_status_view.clone() {
                                RolloutSubmissionStatusPanel { view }
                            }
                        }
                    }
                    aside { class: "support-stack",
                        section { class: "panel compact-panel",
                            SectionHeader {
                                eyebrow: "admin",
                                title: "close admin tools",
                                detail: "return to the main contributor view.",
                            }
                            div { class: "browser-action-row" ,
                                button {
                                    r#type: "button",
                                    class: "action-button action-button-secondary",
                                    onclick: move |_| show_admin_tools.set(false),
                                    "hide admin tools"
                                }
                            }
                        }
                    }
                }
            }
            footer { class: "dragon-site-footer",
                span { class: "dragon-site-footer-build", "rev {footer_build_rev}" }
                ul { class: "dragon-site-footer-links",
                    li {
                        a {
                            href: "https://aberration.technology",
                            "aberration"
                        }
                    }
                    li {
                        a {
                            href: "https://github.com/aberration-technology",
                            "code"
                        }
                    }
                }
            }
        }
    }
}

#[component]
fn ReadinessLadder(steps: Vec<DragonReadinessStepView>) -> Element {
    rsx! {
        section { class: "dragon-readiness-shell", "aria-label": "browser peer readiness",
            ol { class: "dragon-readiness",
                for step in steps {
                    {
                        let status = step.status.class();
                        let marker = step.status.marker();
                        rsx! {
                            li { class: "dragon-step dragon-step-{status}",
                                span { class: "dragon-step-marker", "{marker}" }
                                span { class: "dragon-step-label", "{step.label}" }
                                span { class: "dragon-step-detail", "{step.detail}" }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[component]
fn MetricGrid(metrics: Vec<DragonMetricCardView>) -> Element {
    rsx! {
        section { class: "dragon-metrics-grid", "aria-label": "browser peer metrics",
            for metric in metrics {
                MetricCard { view: metric }
            }
        }
    }
}

#[component]
fn MetricCard(view: DragonMetricCardView) -> Element {
    let tone = view.tone.class();
    let metric_kind = view.title;
    rsx! {
        article { class: "dragon-card dragon-metric dragon-metric-{tone} dragon-metric-card-{metric_kind}",
            div { class: "dragon-card-title", "{view.title}" }
            div { class: "dragon-card-value", title: "{view.value}", "{view.value}" }
            div { class: "dragon-card-detail", title: "{view.detail}", "{view.detail}" }
        }
    }
}

#[component]
fn ActivityFeed(events: Vec<DragonUiEvent>) -> Element {
    rsx! {
        section { class: "panel compact-panel dragon-activity-panel",
            SectionHeader {
                eyebrow: "log",
                title: "state changes",
                detail: "latest peer transitions.",
            }
            if events.is_empty() {
                EmptyState {
                    title: "waiting",
                    detail: "new transitions appear here.",
                }
            } else {
                ol { class: "dragon-activity-feed",
                    for event in events {
                        {
                            let kind = event.kind.class();
                            let time = dragon_format_ui_event_time(event.at_ms);
                            rsx! {
                                li { class: "dragon-activity-event dragon-activity-event-{kind}",
                                    time { "{time}" }
                                    span { class: "dragon-activity-label", "{event.label}" }
                                    if let Some(detail) = event.detail {
                                        span { class: "dragon-activity-detail", "{detail}" }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn dragon_format_ui_event_time(at_ms: f64) -> String {
    let total_seconds = ((at_ms / 1000.0).floor() as u64) % 86_400;
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;
    format!("{hours:02}:{minutes:02}:{seconds:02}")
}

#[component]
fn SectionHeader(eyebrow: &'static str, title: &'static str, detail: String) -> Element {
    rsx! {
        header { class: "section-header",
            div { class: "eyebrow", "{eyebrow}" }
            h2 { class: "browser-focus-title", "{title}" }
            if !detail.trim().is_empty() {
                p { class: "section-detail", "{detail}" }
            }
        }
    }
}

#[component]
fn ActionButton(
    label: &'static str,
    tone: &'static str,
    onclick: EventHandler<MouseEvent>,
) -> Element {
    rsx! {
        button {
            r#type: "button",
            class: "action-button action-button-{tone}",
            onclick: move |event| onclick.call(event),
            "{label}"
        }
    }
}

#[component]
fn ActivityNotice(label: String, detail: String, tone: &'static str) -> Element {
    rsx! {
        div { class: "activity-notice activity-notice-{tone}",
            span { class: "activity-notice-label", "{label}" }
            p { class: "activity-notice-detail", "{detail}" }
        }
    }
}

#[component]
fn StatTile(label: &'static str, value: String, detail: Option<String>) -> Element {
    rsx! {
        div { class: "stat-tile",
            span { "{label}" }
            strong { "{value}" }
            if let Some(detail) = detail {
                p { class: "stat-detail", "{detail}" }
            }
        }
    }
}

#[component]
fn LandingCard(eyebrow: &'static str, title: &'static str, detail: &'static str) -> Element {
    rsx! {
        article { class: "dragon-landing-card",
            div { class: "eyebrow", "{eyebrow}" }
            h3 { class: "browser-focus-title", "{title}" }
            p { class: "section-detail", "{detail}" }
        }
    }
}

#[component]
fn EmptyState(title: &'static str, detail: &'static str) -> Element {
    rsx! {
        div { class: "empty-state",
            h3 { "{title}" }
            p { class: "section-detail", "{detail}" }
        }
    }
}

#[component]
fn EdgeConnectField(label: &'static str, value: String, oninput: EventHandler<String>) -> Element {
    rsx! {
        label { class: "edge-connect",
            span { class: "toolbar-meta-label", "{label}" }
            input {
                value: "{value}",
                oninput: move |event| oninput.call(event.value()),
            }
        }
    }
}

#[component]
fn EditorInputField(label: &'static str, value: String, oninput: EventHandler<String>) -> Element {
    rsx! {
        label { class: "dragon-editor-field",
            span { class: "toolbar-meta-label", "{label}" }
            input {
                class: "dragon-text-input",
                value: "{value}",
                oninput: move |event| oninput.call(event.value()),
            }
        }
    }
}

#[component]
fn EditorTextareaField(
    label: &'static str,
    value: String,
    rows: &'static str,
    oninput: EventHandler<String>,
) -> Element {
    rsx! {
        label { class: "dragon-editor-field",
            span { class: "toolbar-meta-label", "{label}" }
            textarea {
                class: "dragon-textarea",
                value: "{value}",
                rows: "{rows}",
                oninput: move |event| oninput.call(event.value()),
            }
        }
    }
}

#[cfg(test)]
mod tests;
