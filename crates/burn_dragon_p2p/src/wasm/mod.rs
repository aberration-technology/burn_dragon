use anyhow::{Result, anyhow};
use burn_p2p::{
    AuthProvider, BrowserEdgeSnapshot, ClientPlatform, ClientReleaseManifest, ContentId,
    ExperimentDirectoryEntry, ExperimentId, ExperimentScope, ProjectFamilyId, StudyId,
};
use burn_p2p_admin::AdminResult;
use burn_p2p_app::{
    AdminSessionCard, AuthSessionCard, DirectoryEntryDraftPanel, ExperimentDirectoryListPanel,
    LifecycleAssignmentStatusCard, RolloutPreviewPanel, RolloutSubmissionStatusPanel,
    RuntimeCapabilityCard, TrainingResultPanel, TransportHealthPanel,
};
use burn_p2p_browser::{BrowserAppConnectConfig, BrowserAppController, BrowserSessionState};
use burn_p2p_views::{
    AdminSessionSummaryView, BrowserAppClientView, ContributionIdentityPanel,
    DirectoryEntryDraftView, DirectoryMutationResultView, ExperimentDirectoryEntryView,
    ExperimentDirectoryListView, LifecycleAssignmentStatusView, RolloutPreviewView,
    RuntimeCapabilitySummaryView, TrainingResultSummaryView,
};
use dioxus::prelude::*;
use url::form_urlencoded;

use crate::admin::{fetch_directory_entries, rollout_directory_entries, upsert_directory_entry};
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

fn connect_config(config: &DragonBrowserAppConfig) -> Result<BrowserAppConnectConfig> {
    let config = config_with_window_network_overrides(config)?;
    let capability_decision = match config.training.as_ref() {
        Some(training) => apply_browser_downgrade_state(
            &resolved_edge_base_url(&config)?,
            training,
            browser_backend_label(training),
            decide_browser_capability(Some(training), &detect_browser_host_capabilities()),
        ),
        None => decide_browser_capability(None, &detect_browser_host_capabilities()),
    };
    let connect = BrowserAppConnectConfig::new(
        resolved_edge_base_url(&config)?,
        capability_decision.capability,
        capability_decision.connect_target,
    );
    if let Some((experiment_id, revision_id)) = config.selected_experiment() {
        Ok(connect.with_selection(experiment_id, revision_id))
    } else {
        Ok(connect)
    }
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
    let app_semver = semver::Version::parse(env!("CARGO_PKG_VERSION")).unwrap_or_else(|_| {
        semver::Version::parse("0.21.0-pre.15").expect("valid burn_dragon version")
    });

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
) -> Result<ClientReleaseManifest> {
    if let Some(release_manifest) = release_manifest {
        return Ok(release_manifest.clone());
    }

    let edge_base_url = resolved_edge_base_url(config)?;
    let snapshot = fetch_edge_snapshot(&edge_base_url).await?;
    Ok(browser_release_manifest_from_snapshot(&snapshot))
}

#[cfg(feature = "wasm-peer")]
async fn resolve_browser_training_config(
    config: &DragonBrowserAppConfig,
) -> Result<crate::config::DragonBrowserTrainingConfig> {
    if let Some(mut training) = config.training.clone() {
        if training.training_lease.is_none() {
            training.training_lease = active_training_lease(config).await?;
        }
        return Ok(training);
    }

    let edge_base_url = resolved_edge_base_url(config)?;
    let snapshot = fetch_edge_snapshot(&edge_base_url).await?;
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
    training.training_lease = active_training_lease(config).await?;
    Ok(training)
}

#[cfg(feature = "wasm-peer")]
async fn active_training_lease(
    config: &DragonBrowserAppConfig,
) -> Result<Option<burn_p2p::WorkloadTrainingLease>> {
    let controller = BrowserAppController::connect_with(connect_config(config)?).await?;
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

fn session_identity_panel(
    session: Option<&BrowserSessionState>,
) -> Option<ContributionIdentityPanel> {
    let claims = session?.session.as_ref()?.claims.clone();
    let scoped_experiments = claims
        .granted_scopes
        .into_iter()
        .filter_map(|scope| match scope {
            ExperimentScope::Train { experiment_id }
            | ExperimentScope::Validate { experiment_id }
            | ExperimentScope::Archive { experiment_id } => Some(experiment_id),
            ExperimentScope::Connect
            | ExperimentScope::Discover
            | ExperimentScope::Admin { .. } => None,
        })
        .collect::<std::collections::BTreeSet<ExperimentId>>()
        .into_iter()
        .collect();
    Some(ContributionIdentityPanel {
        principal_id: claims.principal_id.as_str().into(),
        provider_label: auth_provider_label(&claims.provider),
        trust_badges: Vec::new(),
        scoped_experiments,
    })
}

fn runtime_capability_summary(view: &BrowserAppClientView) -> RuntimeCapabilitySummaryView {
    let backend_summary = if view.runtime_detail.is_empty() {
        view.capability_summary.clone()
    } else {
        format!("{} | {}", view.runtime_detail, view.capability_summary)
    };
    RuntimeCapabilitySummaryView {
        preferred_role: view.runtime_label.clone(),
        backend_summary,
        can_train: view.training.train_available || view.training.can_train,
    }
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

fn lifecycle_assignment_status(
    view: &BrowserAppClientView,
) -> Option<LifecycleAssignmentStatusView> {
    let experiment = view.selected_experiment.as_ref()?;
    let assignment_status = if view.training.active_assignment.is_some() {
        "active".into()
    } else if experiment.train_available {
        "train-ready".into()
    } else if experiment.validate_available {
        "validate-ready".into()
    } else {
        "viewer-only".into()
    };
    Some(LifecycleAssignmentStatusView {
        experiment_label: experiment.display_name.clone(),
        revision_label: experiment.revision_id.clone(),
        lifecycle_phase: view.default_surface.as_str().into(),
        assignment_status,
    })
}

fn latest_training_result_summary(
    view: &BrowserAppClientView,
) -> Option<TrainingResultSummaryView> {
    Some(TrainingResultSummaryView {
        artifact_id: view.training.last_artifact_id.clone()?,
        receipt_id: view.training.last_receipt_id.clone(),
        window_secs: view.training.last_window_secs.unwrap_or_default(),
    })
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

async fn ensure_required_session(
    config: &DragonBrowserAppConfig,
) -> Result<Option<BrowserSessionState>> {
    let session = load_browser_session(&resolved_edge_base_url(config)?).await?;
    if config.require_edge_auth {
        let _claims = session.session.as_ref().ok_or_else(|| {
            anyhow!("an authenticated browser session is required before joining this network")
        })?;
    }
    Ok(Some(session))
}

pub async fn connect_browser_app(config: &DragonBrowserAppConfig) -> Result<BrowserAppClientView> {
    let _ = ensure_required_session(config).await?;
    let controller = BrowserAppController::connect_with(connect_config(config)?).await?;
    Ok(controller.view())
}

pub async fn refresh_browser_app(config: &DragonBrowserAppConfig) -> Result<BrowserAppClientView> {
    let _ = ensure_required_session(config).await?;
    let mut controller = BrowserAppController::connect_with(connect_config(config)?).await?;
    controller.refresh().await.map_err(Into::into)
}

pub async fn resume_or_complete_browser_auth(
    config: &DragonBrowserAppConfig,
    release_manifest: Option<&ClientReleaseManifest>,
) -> Result<Option<BrowserSessionState>> {
    let edge_base_url = resolved_edge_base_url(config)?;
    if let Some(provider_code) = provider_code_from_window_location() {
        let release_manifest = resolve_browser_release_manifest(config, release_manifest).await?;
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
        return load_browser_session(&edge_base_url).await.map(Some);
    }
    Ok(None)
}

pub async fn start_browser_github_auth_with_scopes(
    config: &DragonBrowserAppConfig,
    release_manifest: Option<&ClientReleaseManifest>,
    requested_scopes: std::collections::BTreeSet<ExperimentScope>,
) -> Result<()> {
    let edge_base_url = resolved_edge_base_url(config)?;
    let release_manifest = resolve_browser_release_manifest(config, release_manifest).await?;
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
) -> Result<()> {
    start_browser_github_auth_with_scopes(config, release_manifest, config.requested_scopes.clone())
        .await
}

#[derive(Props, Clone, PartialEq)]
pub struct DragonBrowserAppProps {
    pub config: DragonBrowserAppConfig,
    pub release_manifest: Option<ClientReleaseManifest>,
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
    let mut show_connection_settings = use_signal(|| false);
    let mut show_live_details = use_signal(|| false);
    let mut show_admin_tools = use_signal(|| window_query_flag("admin"));
    let auth_bootstrap_started = use_signal(|| false);
    #[cfg(feature = "wasm-peer")]
    let local_training = use_signal(|| None::<DragonBrowserTrainingResult>);

    {
        let config = initial_config.clone();
        let release_manifest = props.release_manifest.clone();
        let mut session_state = session_state;
        let mut current_view = current_view;
        let mut status = status;
        let mut auth_bootstrap_started = auth_bootstrap_started;
        use_effect(move || {
            if *auth_bootstrap_started.read() {
                return;
            }
            auth_bootstrap_started.set(true);
            let config = config.clone();
            let release_manifest = release_manifest.clone();
            spawn(async move {
                match resume_or_complete_browser_auth(&config, release_manifest.as_ref()).await {
                    Ok(Some(session)) => {
                        session_state.set(Some(session));
                        if let Ok(view) = refresh_browser_app(&config).await {
                            current_view.set(Some(view));
                        }
                        if provider_code_from_window_location().is_some() {
                            status.set("signed in".into());
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
            let mut status = status;
            let mut current_view = current_view;
            let mut session_state = session_state;
            spawn(async move {
                status.set("Connecting…".into());
                match connect_browser_app(&next_config).await {
                    Ok(view) => {
                        current_view.set(Some(view));
                        let session = match resolved_edge_base_url(&next_config) {
                            Ok(edge_base_url) => load_browser_session(&edge_base_url).await.ok(),
                            Err(_) => None,
                        };
                        session_state.set(session);
                        status.set("Connected".into());
                    }
                    Err(error) => status.set(error.to_string()),
                }
            });
        }
    };

    let refresh_action = {
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
            spawn(async move {
                status.set("Refreshing…".into());
                match refresh_browser_app(&next_config).await {
                    Ok(view) => {
                        current_view.set(Some(view));
                        let session = match resolved_edge_base_url(&next_config) {
                            Ok(edge_base_url) => load_browser_session(&edge_base_url).await.ok(),
                            Err(_) => None,
                        };
                        session_state.set(session);
                        status.set("Refreshed".into());
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
            let mut status = status;
            spawn(async move {
                status.set("Starting sign-in…".into());
                if let Err(error) =
                    start_browser_github_auth(&next_config, release_manifest.as_ref()).await
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
            let mut admin_status = admin_status;
            spawn(async move {
                admin_status.set("Starting admin sign-in…".into());
                if let Err(error) = start_browser_github_auth_with_scopes(
                    &next_config,
                    release_manifest.as_ref(),
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
                match fetch_directory_entries(&edge_base_url).await {
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
                match fetch_directory_entries(&edge_base_url).await {
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
                        if let Ok(view) = refresh_browser_app(&next_config).await {
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
            let mut status = status;
            let mut session_state = session_state;
            spawn(async move {
                status.set("Completing sign-in callback…".into());
                match resume_or_complete_browser_auth(&next_config, release_manifest.as_ref()).await
                {
                    Ok(Some(session)) => {
                        session_state.set(Some(session));
                        status.set("Authenticated session ready".into());
                    }
                    Ok(None) => status.set("No callback code found in URL".into()),
                    Err(error) => status.set(error.to_string()),
                }
            });
        }
    };

    let view = current_view.read().clone();
    let session_panel = session_identity_panel(session_state.read().as_ref());
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
    let show_live_details_active = *show_live_details.read();
    let show_admin_tools_active = *show_admin_tools.read() || admin_scope_ready;
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
    let capability_footprint_label = browser_capability_decision
        .footprint
        .as_ref()
        .map(|footprint| format!("{} MiB", footprint.estimated_training_bytes / (1024 * 1024)))
        .unwrap_or_else(|| "n/a".into());
    let capability_tokens_per_second_label = browser_capability_decision
        .footprint
        .as_ref()
        .map(|footprint| format!("{:.1}", footprint.estimated_tokens_per_second))
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
    let selected_revision_label = view
        .as_ref()
        .and_then(|view| {
            view.selected_experiment
                .as_ref()
                .map(|experiment| experiment.revision_id.as_str().to_owned())
        })
        .or_else(|| props.config.selected_revision_id.clone())
        .unwrap_or_else(|| "nca-r1".into());
    let selected_experiment_label = view
        .as_ref()
        .and_then(|view| {
            view.selected_experiment
                .as_ref()
                .map(|experiment| experiment.display_name.clone())
        })
        .or_else(|| props.config.selected_experiment_id.clone())
        .unwrap_or_else(|| "nca-prepretraining".into());
    let active_head_label = view
        .as_ref()
        .and_then(|view| {
            view.training
                .latest_head_id
                .clone()
                .or_else(|| view.training.last_artifact_id.clone())
        })
        .unwrap_or_else(|| "awaiting checkpoint".into());
    let peer_summary = view
        .as_ref()
        .map(|view| {
            if view.network.estimated_network_size == 0 {
                "awaiting sync".to_owned()
            } else if view.network.direct_peers == 0 {
                format!("~{} visible", view.network.estimated_network_size)
            } else {
                format!(
                    "{} direct · ~{} visible",
                    view.network.direct_peers, view.network.estimated_network_size
                )
            }
        })
        .unwrap_or_else(|| "awaiting sync".into());
    let session_summary = session_state
        .read()
        .as_ref()
        .and_then(|session| session.session.as_ref())
        .map(|session| session.claims.principal_id.as_str().to_owned())
        .unwrap_or_else(|| {
            if auth_required {
                "sign-in required".into()
            } else {
                "guest mode".into()
            }
        });
    let runtime_label = view
        .as_ref()
        .map(|view| view.runtime_label.clone())
        .unwrap_or_else(|| {
            browser_runtime_role_label(&browser_capability_decision.capability.recommended_role)
                .replace('_', " ")
        });
    let contributor_mode_label = if auth_required {
        if session_state
            .read()
            .as_ref()
            .and_then(|session| session.session.as_ref())
            .is_some()
        {
            "signed in and ready".to_owned()
        } else {
            "sign in to contribute compute".to_owned()
        }
    } else {
        "guest access enabled".to_owned()
    };
    let hero_subtitle = format!(
        "{} · {} · browser peers help train the current head",
        selected_experiment_label, selected_revision_label
    );
    let landing_notice = if callback_available {
        Some((
            String::from("callback"),
            String::from("finishing github sign-in"),
            "accent",
        ))
    } else if session_state
        .read()
        .as_ref()
        .and_then(|session| session.session.as_ref())
        .is_some()
    {
        Some((
            String::from("ready"),
            String::from("this browser can now connect and contribute receipts"),
            "accent",
        ))
    } else {
        None
    };
    let status_message = status.read().clone();
    let edge_summary = edge_url.read().clone();

    #[cfg(feature = "wasm-peer")]
    let train_action = {
        let props = props.clone();
        move |_| {
            let mut next_config = props.config.clone();
            next_config = next_config.with_network_overrides(
                Some(edge_url.read().clone()),
                DragonPeerNetworkConfig::parse_seed_node_list(&seed_node_urls.read()),
            );
            let release_manifest = props.release_manifest.clone();
            let mut status = status;
            let mut current_view = current_view;
            let mut local_training = local_training;
            spawn(async move {
                let release_manifest =
                    match resolve_browser_release_manifest(&next_config, release_manifest.as_ref())
                        .await
                    {
                        Ok(release_manifest) => release_manifest,
                        Err(error) => {
                            status.set(error.to_string());
                            return;
                        }
                    };
                let training = match resolve_browser_training_config(&next_config).await {
                    Ok(training) => training,
                    Err(error) => {
                        status.set(error.to_string());
                        return;
                    }
                };
                status.set("Running browser training…".into());
                let edge_base_url = match resolved_edge_base_url(&next_config) {
                    Ok(edge_base_url) => edge_base_url,
                    Err(error) => {
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
                        if let Ok(view) = refresh_browser_app(&next_config).await {
                            current_view.set(Some(view));
                        }
                    }
                    Err(error) => {
                        status.set(error.to_string());
                        if let Ok(view) = refresh_browser_app(&next_config).await {
                            current_view.set(Some(view));
                        }
                    }
                }
            });
        }
    };

    #[cfg(feature = "wasm-peer")]
    let train_button = {
        let has_training_config =
            resolved_edge_base_url(&initial_config).is_ok() && browser_can_attempt_dynamic_training;
        rsx! {
            if has_training_config {
                button {
                    r#type: "button",
                    class: "action-button action-button-primary",
                    onclick: train_action,
                    "run browser training"
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
                        div { class: "eyebrow", "burn_dragon" }
                        h1 { class: "app-title", "train dragon together" }
                        p { class: "app-subtitle", "{hero_subtitle}" }
                        div { class: "badge-row",
                            StatusPill { label: contributor_mode_label, tone: "accent" }
                            StatusPill { label: selected_revision_label.clone(), tone: "neutral" }
                            StatusPill { label: runtime_label, tone: "neutral" }
                        }
                    }
                    div { class: "browser-quick-grid",
                        QuickCard { label: "head", value: active_head_label }
                        QuickCard { label: "peers", value: peer_summary }
                        QuickCard { label: "session", value: session_summary }
                    }
                }
                if let Some((label, detail, tone)) = landing_notice {
                    ActivityNotice { label: label, detail: detail, tone: tone }
                }
                if !status_message.is_empty() {
                    ActivityNotice {
                        label: String::from("status"),
                        detail: status_message,
                        tone: "accent",
                    }
                }
                if let Some(reason) = browser_capability_decision.downgrade_reason.clone() {
                    ActivityNotice {
                        label: String::from("capability policy"),
                        detail: reason,
                        tone: "neutral",
                    }
                }
                div { class: "browser-hero-bar",
                    div { class: "dragon-connection-editor",
                        div { class: "browser-action-row",
                            if auth_required {
                                ActionButton {
                                    label: "sign in with github",
                                    tone: "secondary",
                                    onclick: github_login_action,
                                }
                            }
                            ActionButton {
                                label: "connect browser peer",
                                tone: "primary",
                                onclick: connect_action,
                            }
                            ActionButton {
                                label: "refresh network",
                                tone: "secondary",
                                onclick: refresh_action,
                            }
                            {train_button}
                            if show_connection_settings_active {
                                button {
                                    r#type: "button",
                                    class: "action-button action-button-secondary",
                                    onclick: move |_| show_connection_settings.set(false),
                                    "hide connection settings"
                                }
                            } else {
                                button {
                                    r#type: "button",
                                    class: "action-button action-button-secondary",
                                    onclick: move |_| show_connection_settings.set(true),
                                    "advanced connection settings"
                                }
                            }
                            if show_live_details_active {
                                button {
                                    r#type: "button",
                                    class: "action-button action-button-secondary",
                                    onclick: move |_| show_live_details.set(false),
                                    "hide live details"
                                }
                            } else {
                                button {
                                    r#type: "button",
                                    class: "action-button action-button-secondary",
                                    onclick: move |_| show_live_details.set(true),
                                    "show live details"
                                }
                            }
                        }
                        if show_connection_settings_active {
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
                                if callback_available {
                                    button {
                                        r#type: "button",
                                        class: "action-button action-button-secondary",
                                        onclick: complete_callback_action,
                                        "retry callback"
                                    }
                                }
                            }
                        }
                    }
                    div { class: "browser-hero-actions",
                        div { class: "edge-summary",
                            span { class: "toolbar-meta-label", "edge" }
                            strong { class: "edge-summary-pill", "{edge_summary}" }
                        }
                    }
                }
            }
            div { class: "surface-layout browser-surface-layout",
                section { class: "panel primary-panel browser-focus-panel",
                    SectionHeader {
                        eyebrow: "landing",
                        title: "start here",
                        detail: "the default page is for contributors, not operators. sign in, connect, and let this browser help train the current head.",
                    }
                    p { class: "section-detail",
                        "burn_dragon_p2p lets browser peers join the trainer-only diffusion network, sync the latest checkpoint, and donate spare webgpu compute back into the project."
                    }
                    div { class: "dragon-landing-grid",
                        LandingCard {
                            eyebrow: "1",
                            title: "sign in",
                            detail: "use github so the edge can mint a browser session and scope your contribution receipts.",
                        }
                        LandingCard {
                            eyebrow: "2",
                            title: "connect",
                            detail: "the edge and seed nodes are preloaded. most contributors do not need to touch the advanced connection fields.",
                        }
                        LandingCard {
                            eyebrow: "3",
                            title: "contribute compute",
                            detail: "run browser training when your machine is idle and let webgpu windows publish signed progress back to the network.",
                        }
                    }
                    if props.config.training.is_some() {
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
                    if let Some(view) = view.clone() {
                        div { class: "dragon-panel-stack",
                            ActivityNotice {
                                label: String::from("connected"),
                                detail: String::from("this browser has a live view of the selected revision and can now inspect peer state or run a browser window."),
                                tone: "accent",
                            }
                            div { class: "keyvalue-list",
                                div { class: "keyvalue-row",
                                    span { "last loss" }
                                    strong { "{view.training.last_loss.clone().unwrap_or_else(|| \"n/a\".into())}" }
                                }
                                div { class: "keyvalue-row",
                                    span { "throughput" }
                                    strong { "{view.training.throughput_summary.clone().unwrap_or_else(|| \"n/a\".into())}" }
                                }
                                div { class: "keyvalue-row",
                                    span { "optimizer steps" }
                                    strong { "{view.training.optimizer_steps.map(|value| value.to_string()).unwrap_or_else(|| \"n/a\".into())}" }
                                }
                                div { class: "keyvalue-row",
                                    span { "accepted samples" }
                                    strong { "{view.training.accepted_samples.map(|value| value.to_string()).unwrap_or_else(|| \"n/a\".into())}" }
                                }
                            }
                        }
                    }
                }
                aside { class: "support-stack",
                    section { class: "panel compact-panel",
                        SectionHeader {
                            eyebrow: "identity",
                            title: "session",
                            detail: "who this browser is acting as on the edge.",
                        }
                        AuthSessionCard { session: session_panel }
                    }
                    section { class: "panel compact-panel",
                        SectionHeader {
                            eyebrow: "network",
                            title: "what this browser will do",
                            detail: "a short fit check before you donate compute.",
                        }
                        div { class: "keyvalue-list",
                            div { class: "keyvalue-row",
                                span { "role" }
                                strong { "{browser_runtime_role_label(&browser_capability_decision.capability.recommended_role).replace('_', \" \")}" }
                            }
                            div { class: "keyvalue-row",
                                span { "estimated footprint" }
                                strong { "{capability_footprint_label}" }
                            }
                            div { class: "keyvalue-row",
                                span { "window budget" }
                                strong { "{capability_window_label} secs" }
                            }
                            div { class: "keyvalue-row",
                                span { "tokens/sec" }
                                strong { "{capability_tokens_per_second_label}" }
                            }
                        }
                    }
                    if let Some(view) = view.clone() {
                        if let Some(status) = lifecycle_assignment_status(&view) {
                            section { class: "panel compact-panel",
                                SectionHeader {
                                    eyebrow: "assignment",
                                    title: "current role",
                                    detail: "what the selected revision thinks this browser can do right now.",
                                }
                                LifecycleAssignmentStatusCard { status: status }
                            }
                        }
                    }
                }
            }
            if show_live_details_active || view.is_some() {
                div { class: "surface-layout browser-surface-layout",
                    section { class: "panel primary-panel browser-focus-panel",
                        SectionHeader {
                            eyebrow: "details",
                            title: "live peer details",
                            detail: "runtime capability, network training, and the latest checkpoint state.",
                        }
                        if let Some(view) = view.clone() {
                            div { class: "dragon-panel-stack",
                                RuntimeCapabilityCard { summary: runtime_capability_summary(&view) }
                                TrainingResultPanel { result: latest_training_result_summary(&view) }
                            }
                        } else {
                            EmptyState {
                                title: "connect first",
                                detail: "the detailed runtime and training panels appear after the first successful connect or refresh.",
                            }
                        }
                    }
                    aside { class: "support-stack",
                        if let Some(view) = view.clone() {
                            section { class: "panel compact-panel",
                                SectionHeader {
                                    eyebrow: "transport",
                                    title: "network",
                                    detail: "edge connectivity, receipts, and peer visibility.",
                                }
                                TransportHealthPanel { network: view.network.clone() }
                            }
                            section { class: "panel compact-panel",
                                SectionHeader {
                                    eyebrow: "leaderboard",
                                    title: "top participants",
                                    detail: "current preview from the selected revision.",
                                }
                                if view.viewer.leaderboard_preview.is_empty() {
                                    EmptyState {
                                        title: "leaderboard pending",
                                        detail: "leaderboard rows appear after the first synced viewer snapshot.",
                                    }
                                } else {
                                    div { class: "keyvalue-list",
                                        for entry in view.viewer.leaderboard_preview {
                                            div { class: "keyvalue-row",
                                                span { "{entry.label}" }
                                                strong { "{entry.score} · {entry.receipts} receipts" }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        {local_training_section}
                    }
                }
            }
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
                                label: String::from("operator status"),
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
                                title: "hide tools",
                                detail: "the public landing page stays focused on contributors by default.",
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
            } else {
                section { class: "panel compact-panel dragon-admin-gate",
                    SectionHeader {
                        eyebrow: "admin",
                        title: "operator tools are hidden",
                        detail: "directory rollout stays off the landing page unless you explicitly open it.",
                    }
                    div { class: "browser-action-row",
                        button {
                            r#type: "button",
                            class: "action-button action-button-secondary",
                            onclick: move |_| show_admin_tools.set(true),
                            "open admin tools"
                        }
                    }
                }
            }
        }
    }
}

#[component]
fn StatusPill(label: String, tone: &'static str) -> Element {
    rsx! {
        span { class: "status-pill status-pill-{tone}", "{label}" }
    }
}

#[component]
fn QuickCard(label: &'static str, value: String) -> Element {
    rsx! {
        div { class: "browser-quick-card",
            span { "{label}" }
            strong { "{value}" }
        }
    }
}

#[component]
fn SectionHeader(eyebrow: &'static str, title: &'static str, detail: &'static str) -> Element {
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
    use super::normalized_browser_callback_url;

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
}
