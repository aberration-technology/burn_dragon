use anyhow::{Result, anyhow};
use burn_p2p::{
    AuthProvider, BrowserEdgeSnapshot, ClientPlatform, ClientReleaseManifest, ContentId,
    ExperimentDirectoryEntry, ExperimentScope, ProjectFamilyId, StudyId,
};
use burn_p2p_admin::AdminResult;
use burn_p2p_app::{
    AdminSessionCard, DirectoryEntryDraftPanel, ExperimentDirectoryListPanel, RolloutPreviewPanel,
    RolloutSubmissionStatusPanel,
};
use burn_p2p_browser::{
    BrowserAppConnectConfig, BrowserAppController, BrowserSessionState, browser_transport_kind,
};
use burn_p2p_core::{BrowserSeedAdvertisement, SchemaEnvelope, SignedPayload};
use burn_p2p_views::{
    AdminSessionSummaryView, BrowserAppClientView, DirectoryEntryDraftView,
    DirectoryMutationResultView, ExperimentDirectoryEntryView, ExperimentDirectoryListView,
    RolloutPreviewView,
};
use dioxus::prelude::*;
#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
use gloo_timers::future::TimeoutFuture;
use std::cell::RefCell;
use url::form_urlencoded;

use crate::admin::{
    fetch_directory_entries, fetch_signed_directory_entries, rollout_directory_entries,
    upsert_directory_entry,
};
use crate::auth::{
    begin_browser_github_login, complete_browser_github_login, fetch_edge_snapshot,
    load_browser_session, provider_code_from_window_location,
};
use crate::capability::{decide_browser_capability, detect_browser_host_capabilities};
use crate::capability_state::apply_browser_downgrade_state;
use crate::config::{DragonBrowserAppConfig, DragonPeerNetworkConfig};
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

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
const BROWSER_APP_REFRESH_INTERVAL_MILLIS: u32 = 1_000;
#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
const HERO_RATTLE_INTERVAL_MILLIS: u32 = 80;
const HERO_RATTLE_FRAMES: &[&str] = &[
    "⠉⠉", "⠈⠙", "⠀⠹", "⠀⢸", "⠀⣰", "⢀⣠", "⣀⣀", "⣄⡀", "⣆⠀", "⡇⠀", "⠏⠀", "⠋⠁",
];

thread_local! {
    static DRAGON_BROWSER_APP_CONTROLLER: RefCell<Option<BrowserAppController>> = const { RefCell::new(None) };
}

fn current_app_semver() -> semver::Version {
    semver::Version::parse(env!("CARGO_PKG_VERSION")).expect("valid burn_dragon version")
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
) -> Result<BrowserAppConnectConfig> {
    let config = connect_config.clone();
    let capability_decision = match config.training.as_ref() {
        Some(training) => apply_browser_downgrade_state(
            &resolved_edge_base_url(&config)?,
            training,
            browser_backend_label(training),
            decide_browser_capability(Some(training), &detect_browser_host_capabilities()),
        ),
        None => decide_browser_capability(None, &detect_browser_host_capabilities()),
    };
    let mut connect = BrowserAppConnectConfig::new(
        resolved_edge_base_url(&config)?,
        capability_decision.capability,
        capability_decision.connect_target,
    )
    .with_seed_node_urls(config.effective_seed_node_urls().to_vec());
    if can_use_embedded_browser_bootstrap(bootstrap_config, &config) {
        connect = connect
            .with_bootstrap_material(edge_snapshot.cloned(), signed_seed_advertisement.cloned());
    }
    if let Some((experiment_id, revision_id)) = config.selected_experiment() {
        connect = connect.with_selection(experiment_id, revision_id);
    }
    Ok(connect)
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
        git_commit: "browser-site".into(),
        cargo_lock_hash: ContentId::new("dragon-browser-site-lock"),
        burn_version_string: "0.21.0-pre.3".into(),
        enabled_features_hash: ContentId::new("dragon-browser-site-features"),
        // BrowserEdgeSnapshot does not currently expose protocol_major. Dragon's
        // network/deploy surface still defaults to protocol 0, so keep the
        // synthesized browser-site release manifest aligned with that until the
        // edge snapshot exposes the network manifest protocol directly.
        protocol_major: 0,
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

    let snapshot = if let Some(snapshot) = edge_snapshot {
        snapshot.clone()
    } else {
        let edge_base_url = resolved_edge_base_url(config)?;
        fetch_edge_snapshot(&edge_base_url).await?
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
    let mut training = browser_training_config_from_profile(entry, &profile)?.ok_or_else(|| {
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
async fn active_training_lease(
    bootstrap_config: &DragonBrowserAppConfig,
    config: &DragonBrowserAppConfig,
    edge_snapshot: Option<&BrowserEdgeSnapshot>,
    signed_seed_advertisement: Option<&SignedPayload<SchemaEnvelope<BrowserSeedAdvertisement>>>,
) -> Result<Option<burn_p2p::WorkloadTrainingLease>> {
    let controller = BrowserAppController::connect_with(connect_config(
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

#[derive(Clone, Debug, PartialEq, Eq)]
struct DragonLiveNotice {
    label: &'static str,
    detail: String,
    tone: &'static str,
}

fn dragon_live_notice(
    view: Option<&BrowserAppClientView>,
    local_training_pending: bool,
) -> Option<DragonLiveNotice> {
    if local_training_pending {
        return Some(DragonLiveNotice {
            label: "training",
            detail: "running a local training window in this tab".into(),
            tone: "accent",
        });
    }

    let view = view?;
    if view.runtime_label.starts_with("joining ") {
        return Some(DragonLiveNotice {
            label: "connecting",
            detail: view.runtime_detail.clone(),
            tone: "accent",
        });
    }
    if view.runtime_label.starts_with("catchup ") {
        return Some(DragonLiveNotice {
            label: "syncing",
            detail: view.runtime_detail.clone(),
            tone: "neutral",
        });
    }
    if view.runtime_label == "blocked" {
        return Some(DragonLiveNotice {
            label: "blocked",
            detail: view.runtime_detail.clone(),
            tone: "neutral",
        });
    }
    if view.training.can_train
        && view.network.swarm_status.connected_transport.is_none()
        && view.network.swarm_status.desired_transport.is_some()
    {
        let transport = dragon_transport_target_label(view);
        if let Some(error) = view.network.swarm_status.last_error.as_ref() {
            return Some(DragonLiveNotice {
                label: "waiting",
                detail: format!("{transport} unavailable: {error}"),
                tone: "neutral",
            });
        }
        return Some(DragonLiveNotice {
            label: "connecting",
            detail: format!("waiting for {transport}"),
            tone: "accent",
        });
    }

    match (
        view.training.can_train,
        view.training.active_assignment.as_ref(),
        view.training.latest_head_id.as_ref(),
        view.training.cached_microshards,
        view.training.throughput_summary.as_ref(),
    ) {
        (true, None, _, _, _) => Some(DragonLiveNotice {
            label: "waiting",
            detail: "waiting for work".into(),
            tone: "neutral",
        }),
        (true, Some(_), None, _, _) => Some(DragonLiveNotice {
            label: "syncing",
            detail: "waiting for checkpoint sync".into(),
            tone: "neutral",
        }),
        (true, Some(_), Some(_), 0, _) => Some(DragonLiveNotice {
            label: "syncing",
            detail: "downloading assigned slice".into(),
            tone: "accent",
        }),
        (true, Some(_), Some(_), _, None) => Some(DragonLiveNotice {
            label: "waiting",
            detail: "waiting for the first training window".into(),
            tone: "neutral",
        }),
        _ => None,
    }
}

fn dragon_window_summary(
    view: Option<&BrowserAppClientView>,
    local_training_pending: bool,
) -> String {
    if local_training_pending {
        return "running".into();
    }
    let Some(view) = view else {
        return "pending".into();
    };
    match (
        view.training.last_window_secs,
        view.training.max_window_secs,
    ) {
        (Some(last), Some(max)) => format!("{last}s of {max}s"),
        (Some(last), None) => format!("{last}s last"),
        (None, Some(max)) => format!("up to {max}s"),
        (None, None) => "pending".into(),
    }
}

fn parse_leading_rate_per_second(summary: &str) -> Option<f64> {
    summary.split_whitespace().next()?.parse::<f64>().ok()
}

fn format_compact_duration(seconds: u64) -> String {
    match seconds {
        0 => "<1s".into(),
        1..=59 => format!("{seconds}s"),
        60..=3599 => {
            let minutes = seconds / 60;
            let remainder = seconds % 60;
            if remainder == 0 {
                format!("{minutes}m")
            } else {
                format!("{minutes}m {remainder}s")
            }
        }
        _ => {
            let hours = seconds / 3600;
            let remainder = seconds % 3600;
            let minutes = remainder / 60;
            if minutes == 0 {
                format!("{hours}h")
            } else {
                format!("{hours}h {minutes}m")
            }
        }
    }
}

fn dragon_window_eta_summary(view: Option<&BrowserAppClientView>) -> Option<String> {
    let view = view?;
    let remaining = view.training.slice_remaining_samples?;
    let throughput = view.training.throughput_summary.as_deref()?;
    let rate = parse_leading_rate_per_second(throughput)?;
    if !rate.is_finite() || rate <= 0.0 {
        return None;
    }
    let eta_seconds = ((remaining as f64) / rate).ceil() as u64;
    Some(format_compact_duration(eta_seconds))
}

fn dragon_slice_progress_summary(view: Option<&BrowserAppClientView>) -> String {
    let Some(view) = view else {
        return "pending".into();
    };
    match (
        view.training.accepted_samples,
        view.training.slice_target_samples,
        view.training.slice_remaining_samples,
    ) {
        (Some(done), Some(target), Some(remaining)) => {
            format!("{done}/{target} · {remaining} left")
        }
        (Some(done), Some(target), None) => format!("{done}/{target}"),
        _ => view.training.slice_status.clone(),
    }
}

fn dragon_transport_summary(view: Option<&BrowserAppClientView>) -> String {
    let Some(view) = view else {
        return "offline".into();
    };
    let transport = dragon_transport_target_label(view);
    if transport.is_empty() {
        return "offline".into();
    }
    if view.network.direct_peers > 0 {
        let peer_label = if view.network.direct_peers == 1 {
            "peer"
        } else {
            "peers"
        };
        return format!("{transport} · {} {peer_label}", view.network.direct_peers);
    }
    if view.network.swarm_status.connected_transport.is_none()
        && view.network.swarm_status.desired_transport.is_some()
    {
        return format!("dialing {transport}");
    }
    transport.to_owned()
}

fn dragon_transport_target_label(view: &BrowserAppClientView) -> String {
    if let Some(connected) = view.network.swarm_status.connected_transport.as_ref() {
        return browser_transport_kind(connected).label().into();
    }
    if let Some(desired) = view.network.swarm_status.desired_transport.as_ref() {
        return browser_transport_kind(desired).label().into();
    }
    let fallback = view.network.transport.trim();
    if fallback.is_empty() {
        "offline".into()
    } else {
        fallback.to_owned()
    }
}

fn dragon_network_detail(view: Option<&BrowserAppClientView>) -> String {
    let Some(view) = view else {
        return "edge snapshot only".into();
    };
    if view.network.direct_peers > 0 {
        if view.network.estimated_network_size > view.network.direct_peers {
            return format!(
                "{} direct · ~{} visible",
                view.network.direct_peers, view.network.estimated_network_size
            );
        }
        let peer_label = if view.network.direct_peers == 1 {
            "direct peer"
        } else {
            "direct peers"
        };
        return format!("{} {peer_label}", view.network.direct_peers);
    }
    if view.network.swarm_status.connected_transport.is_none()
        && view.network.swarm_status.desired_transport.is_some()
    {
        return "awaiting direct peer handshake".into();
    }
    if view.network.estimated_network_size > 0 {
        return format!(
            "~{} visible from the current network view",
            view.network.estimated_network_size
        );
    }
    "edge snapshot only".into()
}

fn dragon_window_progress_detail(
    view: Option<&BrowserAppClientView>,
    window_summary: &str,
) -> String {
    let Some(view) = view else {
        return window_summary.into();
    };
    match (
        view.training.slice_remaining_samples,
        view.training.slice_target_samples,
    ) {
        (Some(remaining), Some(_target)) => {
            if let Some(eta) = dragon_window_eta_summary(Some(view)) {
                format!("{remaining} left · eta {eta}")
            } else {
                format!("{remaining} left · {window_summary}")
            }
        }
        (Some(remaining), None) => {
            if let Some(eta) = dragon_window_eta_summary(Some(view)) {
                format!("{remaining} left · eta {eta}")
            } else {
                format!("{remaining} left")
            }
        }
        _ => window_summary.into(),
    }
}

fn dragon_local_training_summary(
    view: Option<&BrowserAppClientView>,
    local_training_pending: bool,
    window_summary: &str,
) -> String {
    if local_training_pending {
        return "training…".into();
    }
    view.and_then(|view| view.training.throughput_summary.clone())
        .unwrap_or_else(|| window_summary.to_owned())
}

fn dragon_local_training_detail(
    view: Option<&BrowserAppClientView>,
    window_summary: &str,
) -> String {
    let Some(view) = view else {
        return window_summary.into();
    };
    if let Some(loss) = view.training.last_loss.as_ref() {
        return format!("loss {loss} · {window_summary}");
    }
    window_summary.into()
}

fn dragon_global_training_summary(view: Option<&BrowserAppClientView>) -> String {
    view.and_then(|view| {
        view.network
            .performance
            .as_ref()
            .map(|performance| performance.training_throughput.clone())
    })
    .unwrap_or_else(|| "pending".into())
}

fn dragon_global_training_detail(view: Option<&BrowserAppClientView>) -> String {
    let Some(performance) = view.and_then(|view| view.network.performance.as_ref()) else {
        return "pending".into();
    };
    format!("validation {}", performance.validation_throughput)
}

fn browser_runtime_role_label(role: &burn_p2p_browser::BrowserRuntimeRole) -> &'static str {
    match role {
        burn_p2p_browser::BrowserRuntimeRole::BrowserTrainerWgpu => "browser_trainer_wgpu",
        burn_p2p_browser::BrowserRuntimeRole::BrowserVerifier => "browser_verifier",
        burn_p2p_browser::BrowserRuntimeRole::BrowserObserver => "browser_observer",
        burn_p2p_browser::BrowserRuntimeRole::BrowserFallback => "browser_fallback",
        burn_p2p_browser::BrowserRuntimeRole::Viewer => "viewer",
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
    let controller = BrowserAppController::connect_with(connect_config(
        bootstrap_config,
        config,
        edge_snapshot,
        signed_seed_advertisement,
    )?)
    .await?;
    let view = controller.view();
    DRAGON_BROWSER_APP_CONTROLLER.with(|slot| {
        *slot.borrow_mut() = Some(controller);
    });
    Ok(view)
}

pub async fn refresh_browser_app(
    bootstrap_config: &DragonBrowserAppConfig,
    config: &DragonBrowserAppConfig,
    edge_snapshot: Option<&BrowserEdgeSnapshot>,
    signed_seed_advertisement: Option<&SignedPayload<SchemaEnvelope<BrowserSeedAdvertisement>>>,
) -> Result<BrowserAppClientView> {
    let mut controller = if let Some(controller) =
        DRAGON_BROWSER_APP_CONTROLLER.with(|slot| slot.borrow_mut().take())
    {
        controller
    } else {
        BrowserAppController::connect_with(connect_config(
            bootstrap_config,
            config,
            edge_snapshot,
            signed_seed_advertisement,
        )?)
        .await?
    };
    let refresh_result = controller.refresh().await.map(|_| controller.view());
    DRAGON_BROWSER_APP_CONTROLLER.with(|slot| {
        *slot.borrow_mut() = Some(controller);
    });
    Ok(refresh_result?)
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
            TimeoutFuture::new(BROWSER_APP_REFRESH_INTERVAL_MILLIS).await;
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
        let _ = normalize_provider_callback_window_location();
        return Ok(Some(session));
    }
    if config.require_edge_auth {
        let session = load_browser_session(&edge_base_url).await?;
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
    #[cfg(feature = "wasm-peer")]
    let local_training = use_signal(|| None::<DragonBrowserTrainingResult>);
    #[cfg(feature = "wasm-peer")]
    let local_training_pending = use_signal(|| false);

    {
        let config = initial_config.clone();
        let bootstrap_config = props.config.clone();
        let release_manifest = props.release_manifest.clone();
        let edge_snapshot = props.edge_snapshot.clone();
        let signed_seed_advertisement = props.signed_seed_advertisement.clone();
        let mut session_state = session_state;
        let mut current_view = current_view;
        let mut status = status;
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
                        if let Ok(view) = connect_browser_app(
                            &bootstrap_config,
                            &config,
                            edge_snapshot.as_ref(),
                            signed_seed_advertisement.as_ref(),
                        )
                        .await
                        {
                            current_view.set(Some(view));
                            spawn_browser_app_refresh_loop(
                                bootstrap_config.clone(),
                                config.clone(),
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
            let checkpoint_wait_generation = checkpoint_wait_generation;
            let edge_snapshot = props.edge_snapshot.clone();
            let signed_seed_advertisement = props.signed_seed_advertisement.clone();
            spawn(async move {
                status.set("Connecting…".into());
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
    let browser_host_capabilities = detect_browser_host_capabilities();
    let browser_capability_decision = match (
        props.config.training.as_ref(),
        resolved_edge_base_url(&initial_config),
    ) {
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
    let browser_can_attempt_dynamic_training = props.config.training.is_some()
        || (browser_host_capabilities.navigator_gpu_exposed
            && browser_host_capabilities.worker_gpu_exposed
            && browser_host_capabilities.dedicated_worker_exposed);
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
    let active_head_label = view
        .as_ref()
        .and_then(|view| {
            view.training
                .latest_head_id
                .clone()
                .or_else(|| view.training.last_artifact_id.clone())
        })
        .unwrap_or_else(|| "awaiting checkpoint".into());
    let has_active_checkpoint = view.as_ref().is_some_and(|view| {
        view.training.latest_head_id.is_some() || view.training.last_artifact_id.is_some()
    });
    let network_summary = dragon_network_detail(view.as_ref());
    let transport_summary = dragon_transport_summary(view.as_ref());
    let has_session = session_state
        .read()
        .as_ref()
        .and_then(|session| session.session.as_ref())
        .is_some();
    let auth_bootstrap_pending_active = *auth_bootstrap_pending.read();
    let has_connected_view = view.is_some();
    let public_landing = !auth_bootstrap_pending_active && !has_session && !has_connected_view;
    let needs_sign_in = !auth_bootstrap_pending_active && auth_required && !has_session;
    let ready_to_connect = !auth_bootstrap_pending_active && !needs_sign_in && !has_connected_view;
    let hero_title = "train the dragon".to_owned();
    let hero_subtitle = String::new();
    let raw_status_message = status.read().clone();
    let status_message = if public_landing
        && (raw_status_message.contains("failed to fetch edge snapshot")
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
    let local_training_pending_active = *local_training_pending.read();
    #[cfg(not(feature = "wasm-peer"))]
    let local_training_pending_active = false;
    let live_notice = dragon_live_notice(view.as_ref(), local_training_pending_active);
    let hero_rattle_active = auth_bootstrap_pending_active
        || status_message.starts_with("Connecting")
        || status_message.starts_with("Starting sign-in")
        || live_notice
            .as_ref()
            .is_some_and(|notice| notice.label != "blocked");
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
    #[cfg(feature = "wasm-peer")]
    let direct_transport_ready = view
        .as_ref()
        .is_some_and(|view| view.network.direct_peers > 0);
    let connected_panel_title = "connected";
    let connected_panel_detail = "status, throughput, peers, and remaining work.".to_owned();
    let live_status_label = if local_training_pending_active {
        "training"
    } else if let Some(notice) = live_notice.as_ref() {
        notice.label
    } else if has_active_checkpoint {
        "ready"
    } else {
        "connected"
    };
    let runtime_mode_summary = view
        .as_ref()
        .map(|view| view.runtime_label.clone())
        .unwrap_or_else(|| {
            if auth_bootstrap_pending_active {
                "bootstrapping".into()
            } else {
                "idle".into()
            }
        });
    let slice_progress_summary = dragon_slice_progress_summary(view.as_ref());
    let window_summary = dragon_window_summary(view.as_ref(), local_training_pending_active);
    let local_training_summary = dragon_local_training_summary(
        view.as_ref(),
        local_training_pending_active,
        &window_summary,
    );
    let local_training_detail = dragon_local_training_detail(view.as_ref(), &window_summary);
    let global_training_summary = dragon_global_training_summary(view.as_ref());
    let global_training_detail = dragon_global_training_detail(view.as_ref());
    let window_progress_detail = dragon_window_progress_detail(view.as_ref(), &window_summary);
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
            let mut local_training_pending = local_training_pending;
            spawn(async move {
                let release_manifest = match resolve_browser_release_manifest(
                    &next_config,
                    release_manifest.as_ref(),
                    edge_snapshot.as_ref(),
                )
                .await
                {
                    Ok(release_manifest) => release_manifest,
                    Err(error) => {
                        status.set(error.to_string());
                        return;
                    }
                };
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
                        status.set(error.to_string());
                        return;
                    }
                };
                local_training_pending.set(true);
                status.set("Running browser training…".into());
                let edge_base_url = match resolved_edge_base_url(&next_config) {
                    Ok(edge_base_url) => edge_base_url,
                    Err(error) => {
                        local_training_pending.set(false);
                        status.set(error.to_string());
                        return;
                    }
                };
                match run_browser_training_with_release_manifest(
                    &edge_base_url,
                    &training,
                    &release_manifest,
                )
                .await
                {
                    Ok(result) => {
                        status.set(format!(
                            "Browser training complete: mean train loss {:.4}",
                            result.train_loss_mean
                        ));
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
                        status.set(error.to_string());
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
                }
                local_training_pending.set(false);
            });
        }
    };

    #[cfg(feature = "wasm-peer")]
    let train_button = {
        let has_training_config = has_connected_view
            && has_active_checkpoint
            && direct_transport_ready
            && resolved_edge_base_url(&initial_config).is_ok()
            && browser_can_attempt_dynamic_training;
        rsx! {
            if has_training_config {
                button {
                    r#type: "button",
                    class: "action-button action-button-primary",
                    disabled: local_training_pending_active,
                    onclick: train_action,
                    if local_training_pending_active {
                        "training…"
                    } else {
                        "run browser training"
                    }
                }
            }
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
        let train_loss_label = format!("{:.4}", result.train_loss_mean);
        let tokens_per_second_label = result
            .tokens_per_second
            .map(|value| format!("{value:.1}"))
            .unwrap_or_else(|| "n/a".into());
        let train_batches_label = result.train_batches.to_string();
        let live_training_details = result.live_participant.map(|live| {
            (
                live.receipt_submission_accepted,
                live.accepted_receipt_ids.join(", "),
                live.runtime_state.unwrap_or_else(|| "n/a".into()),
            )
        });
        rsx! {
            section { class: "panel compact-panel",
                SectionHeader {
                    eyebrow: "local",
                    title: "browser training",
                    detail: "latest local-only training window executed in this tab.",
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
                if let Some((receipt_submission_accepted, accepted_receipts_label, runtime_state_label)) = live_training_details {
                    div { class: "keyvalue-list",
                        div { class: "keyvalue-row",
                            span { "receipt accepted" }
                            strong { "{receipt_submission_accepted}" }
                        }
                        div { class: "keyvalue-row",
                            span { "accepted receipts" }
                            strong { "{accepted_receipts_label}" }
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
                        h1 { class: "app-title", "{hero_title}" }
                        if !hero_subtitle.is_empty() {
                            p { class: "app-subtitle", "{hero_subtitle}" }
                        }
                    }
                }
                if !status_message.is_empty() {
                    ActivityNotice {
                        label: String::from("status"),
                        detail: status_message,
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
            if has_connected_view {
                div { class: "dragon-live-shell-wrap",
                    section { class: "panel primary-panel browser-focus-panel dragon-live-shell",
                        SectionHeader {
                            eyebrow: "live",
                            title: connected_panel_title,
                            detail: connected_panel_detail,
                        }
                        if let Some(view) = view.clone() {
                            div { class: "dragon-panel-stack dragon-live-summary",
                                if let Some(notice) = live_notice.as_ref() {
                                    ActivityNotice {
                                        label: notice.label.to_owned(),
                                        detail: notice.detail.clone(),
                                        tone: notice.tone,
                                    }
                                }
                                div { class: "dragon-live-status-row",
                                    span {
                                        class: "dragon-live-status-pill dragon-live-status-pill-{live_status_label}",
                                        "{live_status_label}"
                                    }
                                }
                                div { class: "dragon-live-stats",
                                    StatTile {
                                        label: "status",
                                        value: runtime_mode_summary,
                                        detail: Some(view.runtime_detail.clone()),
                                    }
                                    StatTile {
                                        label: "local train",
                                        value: local_training_summary,
                                        detail: Some(local_training_detail),
                                    }
                                    StatTile {
                                        label: "global train",
                                        value: global_training_summary,
                                        detail: Some(global_training_detail),
                                    }
                                    StatTile {
                                        label: "window",
                                        value: slice_progress_summary,
                                        detail: Some(window_progress_detail),
                                    }
                                    StatTile {
                                        label: "peers",
                                        value: network_summary.clone(),
                                        detail: Some(transport_summary.clone()),
                                    }
                                }
                                if has_active_checkpoint {
                                    div { class: "dragon-live-actions browser-action-row",
                                        {train_button}
                                    }
                                }
                                if debug_controls_enabled {
                                    div { class: "keyvalue-list",
                                        div { class: "keyvalue-row",
                                            span { "optimizer steps" }
                                            strong { "{view.training.optimizer_steps.map(|value| value.to_string()).unwrap_or_else(|| \"n/a\".into())}" }
                                        }
                                        div { class: "keyvalue-row",
                                            span { "accepted samples" }
                                            strong { "{view.training.accepted_samples.map(|value| value.to_string()).unwrap_or_else(|| \"n/a\".into())}" }
                                        }
                                        div { class: "keyvalue-row",
                                            span { "head" }
                                            strong { "{active_head_label}" }
                                        }
                                    }
                                }
                            }
                        }
                        if debug_controls_enabled && props.config.training.is_some() {
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
fn SectionHeader(eyebrow: &'static str, title: &'static str, detail: String) -> Element {
    rsx! {
        header { class: "section-header",
            div { class: "eyebrow", "{eyebrow}" }
            h2 { class: "browser-focus-title", "{title}" }
            p { class: "section-detail", "{detail}" }
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
mod tests {
    use super::{
        browser_session_is_authenticated, dragon_global_training_detail,
        dragon_global_training_summary, dragon_live_notice, dragon_local_training_detail,
        dragon_local_training_summary, dragon_network_detail, dragon_slice_progress_summary,
        dragon_transport_summary, dragon_window_progress_detail, dragon_window_summary,
        normalized_browser_callback_url,
    };
    use burn_p2p::{
        AuthProvider, ContentId, NetworkId, PeerRoleSet, PrincipalClaims, PrincipalId,
        PrincipalSession,
    };
    use burn_p2p_browser::{BrowserSessionState, BrowserTransportKind};
    use burn_p2p_core::{BrowserSwarmStatus, BrowserTransportFamily};
    use burn_p2p_views::{
        BrowserAppClientView, BrowserAppNetworkView, BrowserAppPerformanceView, BrowserAppSurface,
        BrowserAppTrainingView, BrowserAppValidationView, BrowserAppViewerView,
    };
    use chrono::Utc;
    use std::collections::{BTreeMap, BTreeSet};
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    fn sample_browser_view() -> BrowserAppClientView {
        BrowserAppClientView {
            network_id: "burn-dragon-mainnet".into(),
            default_surface: BrowserAppSurface::Train,
            runtime_label: "ready".into(),
            runtime_detail: "browser runtime ready".into(),
            capability_summary: "full".into(),
            session_label: "authenticated".into(),
            selected_experiment: None,
            viewer: BrowserAppViewerView {
                visible_experiments: 0,
                visible_heads: 0,
                leaderboard_entries: 0,
                signed_directory_ready: false,
                signed_leaderboard_ready: false,
                experiments_preview: Vec::new(),
                leaderboard_preview: Vec::new(),
            },
            validation: BrowserAppValidationView {
                validate_available: false,
                can_validate: false,
                current_head_id: None,
                metrics_sync_at: None,
                pending_receipts: 0,
                validation_status: None,
                checked_chunks: None,
                emitted_receipt_id: None,
                evaluation_summary: None,
                metric_preview: Vec::new(),
            },
            training: BrowserAppTrainingView {
                train_available: true,
                can_train: true,
                active_assignment: None,
                active_training_lease: None,
                slice_status: "pending".into(),
                latest_head_id: None,
                cached_chunk_artifacts: 0,
                cached_microshards: 0,
                pending_receipts: 0,
                max_window_secs: None,
                last_window_secs: None,
                optimizer_steps: None,
                accepted_samples: None,
                slice_target_samples: None,
                slice_remaining_samples: None,
                last_loss: None,
                publish_latency_ms: None,
                throughput_summary: None,
                last_artifact_id: None,
                last_receipt_id: None,
            },
            network: BrowserAppNetworkView {
                edge_base_url: "https://edge.example".into(),
                transport: BrowserTransportKind::WebRtcDirect.label().into(),
                node_state: "IdleReady".into(),
                direct_peers: 0,
                observed_peers: 0,
                estimated_network_size: 0,
                accepted_receipts: 0,
                certified_merges: 0,
                in_flight_transfers: 0,
                network_note: "test".into(),
                swarm_status: BrowserSwarmStatus::default(),
                metrics_live_ready: false,
                last_directory_sync_at: None,
                last_error: None,
                performance: None,
                diffusion: None,
            },
        }
    }

    #[test]
    fn callback_url_normalizes_to_site_root() {
        assert_eq!(
            normalized_browser_callback_url("/callback/github", "?code=abc&state=xyz", ""),
            "/"
        );
        assert_eq!(
            normalized_browser_callback_url(
                "/repo/callback/github",
                "?code=abc&edge=https%3A%2F%2Fedge.example",
                "#frag",
            ),
            "/repo/?edge=https%3A%2F%2Fedge.example#frag"
        );
    }

    #[test]
    fn browser_session_authentication_requires_session_claims() {
        assert!(!browser_session_is_authenticated(
            &BrowserSessionState::default()
        ));

        let now = Utc::now();
        let session = BrowserSessionState {
            session: Some(PrincipalSession {
                session_id: ContentId::new("session-browser-test"),
                network_id: NetworkId::new("burn-dragon-mainnet"),
                claims: PrincipalClaims {
                    principal_id: PrincipalId::new("principal-browser-test"),
                    provider: AuthProvider::Static {
                        authority: "test".into(),
                    },
                    display_name: "Browser Test".into(),
                    org_memberships: BTreeSet::new(),
                    group_memberships: BTreeSet::new(),
                    granted_roles: PeerRoleSet::default(),
                    granted_scopes: BTreeSet::new(),
                    custom_claims: BTreeMap::new(),
                    issued_at: now,
                    expires_at: now,
                },
                issued_at: now,
                expires_at: now,
            }),
            ..BrowserSessionState::default()
        };
        assert!(browser_session_is_authenticated(&session));
    }

    #[wasm_bindgen_test]
    fn dragon_live_notice_reports_direct_connecting_state_truthfully() {
        let mut view = sample_browser_view();
        view.network.swarm_status.desired_transport = Some(BrowserTransportFamily::WebRtcDirect);

        let notice = dragon_live_notice(Some(&view), false).expect("direct handoff notice");
        assert_eq!(notice.label, "connecting");
        assert_eq!(notice.detail, "waiting for webrtc-direct");
        assert_eq!(notice.tone, "accent");
    }

    #[wasm_bindgen_test]
    fn dragon_live_notice_reports_checkpoint_and_slice_wait_states() {
        let mut view = sample_browser_view();
        view.training.active_assignment = Some("assignment-1".into());

        let checkpoint_notice = dragon_live_notice(Some(&view), false).expect("checkpoint notice");
        assert_eq!(checkpoint_notice.label, "syncing");
        assert_eq!(checkpoint_notice.detail, "waiting for checkpoint sync");

        view.training.latest_head_id = Some("head-1".into());
        let slice_notice = dragon_live_notice(Some(&view), false).expect("slice notice");
        assert_eq!(slice_notice.label, "syncing");
        assert_eq!(slice_notice.detail, "downloading assigned slice");

        view.training.cached_microshards = 4;
        let window_notice = dragon_live_notice(Some(&view), false).expect("window notice");
        assert_eq!(window_notice.label, "waiting");
        assert_eq!(
            window_notice.detail,
            "waiting for the first training window"
        );
    }

    #[wasm_bindgen_test]
    fn dragon_live_notice_prefers_local_training_state() {
        let notice = dragon_live_notice(Some(&sample_browser_view()), true).expect("training");
        assert_eq!(notice.label, "training");
        assert_eq!(notice.detail, "running a local training window in this tab");
        assert_eq!(notice.tone, "accent");
    }

    #[wasm_bindgen_test]
    fn dragon_transport_and_progress_summaries_reflect_truthful_runtime_state() {
        let mut view = sample_browser_view();
        view.network.direct_peers = 2;
        view.training.last_window_secs = Some(9);
        view.training.max_window_secs = Some(30);
        view.training.accepted_samples = Some(96);
        view.training.slice_target_samples = Some(128);
        view.training.slice_remaining_samples = Some(32);

        assert_eq!(
            dragon_transport_summary(Some(&view)),
            "webrtc-direct · 2 peers"
        );
        assert_eq!(dragon_window_summary(Some(&view), false), "9s of 30s");
        assert_eq!(
            dragon_slice_progress_summary(Some(&view)),
            "96/128 · 32 left"
        );
    }

    #[wasm_bindgen_test]
    fn dragon_connected_summary_prefers_high_signal_metrics() {
        let mut view = sample_browser_view();
        view.training.last_window_secs = Some(9);
        view.training.max_window_secs = Some(30);
        view.training.accepted_samples = Some(96);
        view.training.slice_target_samples = Some(128);
        view.training.slice_remaining_samples = Some(32);
        view.training.throughput_summary = Some("8.0 sample/s".into());
        view.training.last_loss = Some("0.421".into());
        view.network.direct_peers = 3;
        view.network.estimated_network_size = 9;
        view.network.performance = Some(BrowserAppPerformanceView {
            scope_summary: "visible peers".into(),
            captured_at: "2026-04-18T00:00:00Z".into(),
            training_throughput: "128.0 sample/s".into(),
            validation_throughput: "16.0 sample/s".into(),
            wait_time: "2s".into(),
            idle_time: "1s".into(),
        });

        assert_eq!(
            dragon_local_training_summary(Some(&view), false, "9s of 30s"),
            "8.0 sample/s"
        );
        assert_eq!(
            dragon_local_training_detail(Some(&view), "9s of 30s"),
            "loss 0.421 · 9s of 30s"
        );
        assert_eq!(
            dragon_global_training_summary(Some(&view)),
            "128.0 sample/s"
        );
        assert_eq!(
            dragon_global_training_detail(Some(&view)),
            "validation 16.0 sample/s"
        );
        assert_eq!(
            dragon_window_progress_detail(Some(&view), "9s of 30s"),
            "32 left · eta 4s"
        );
        assert_eq!(dragon_network_detail(Some(&view)), "3 direct · ~9 visible");
    }
}
