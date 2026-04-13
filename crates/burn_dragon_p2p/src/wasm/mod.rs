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
        semver::Version::parse("0.21.0-pre.13").expect("valid burn_dragon version")
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
    #[cfg(feature = "wasm-peer")]
    let local_training = use_signal(|| None::<DragonBrowserTrainingResult>);

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
                button { onclick: train_action, "Run Browser Training" }
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
        let tokens_per_second_label = result
            .tokens_per_second
            .map(|value| format!("{value:.1}"))
            .unwrap_or_else(|| "n/a".into());
        let train_batches_label = result.train_batches.to_string();
        rsx! {
            section {
                h2 { "Local Browser Training" }
                p { "Experiment: ", {result.experiment_kind_label} }
                p { "Backend: ", {result.backend} }
                p { "Train loss: ", {format!("{:.4}", result.train_loss_mean)} }
                p { "Eval loss: ", {eval_loss_label} }
                p { "Train batches: ", {train_batches_label} }
                p { "Tokens/sec: ", {tokens_per_second_label} }
                if let Some(live) = result.live_participant {
                    p { "Receipt accepted: ", {live.receipt_submission_accepted.to_string()} }
                    p { "Accepted receipts: ", {live.accepted_receipt_ids.join(", ")} }
                    p { "Runtime state: ", {live.runtime_state.unwrap_or_else(|| "n/a".into())} }
                }
            }
        }
    } else {
        rsx! {}
    };
    #[cfg(not(feature = "wasm-peer"))]
    let local_training_section = rsx! {};

    rsx! {
        div {
            class: "burn-dragon-p2p-app",
            h1 { "burn_dragon p2p" }
            p { "Browser peer and operator shell for NCA and ClimbMix experiment networks." }
            div {
                label { "Edge URL" }
                input {
                    value: "{edge_url}",
                    oninput: move |event| edge_url.set(event.value()),
                }
            }
            div {
                label { "Seed Node URLs" }
                input {
                    value: "{seed_node_urls}",
                    oninput: move |event| seed_node_urls.set(event.value()),
                }
            }
            div {
                button { onclick: connect_action, "Connect" }
                button { onclick: refresh_action, "Refresh" }
                if auth_required {
                    button { onclick: github_login_action, "Sign In" }
                }
                if callback_available {
                    button { onclick: complete_callback_action, "Complete Callback" }
                }
                {train_button}
            }
            p { "{status}" }
            if let Some(reason) = browser_capability_decision.downgrade_reason.clone() {
                p { "Capability policy: {reason}" }
            }
            if props.config.training.is_some() {
                section {
                    h2 { "Local Trainer Capability" }
                    p { "Recommended role: {browser_runtime_role_label(&browser_capability_decision.capability.recommended_role)}" }
                    p { "Estimated training footprint: {capability_footprint_label}" }
                    p { "Trainer memory budget: {capability_budget_label}" }
                    p { "Estimated tokens/sec: {capability_tokens_per_second_label}" }
                    p { "Checkpoint budget: {capability_checkpoint_label}" }
                    p { "Shard budget: {capability_shard_label}" }
                    p { "Window budget secs: {capability_window_label}" }
                }
            }
            section {
                h2 { "Session" }
                AuthSessionCard { session: session_panel }
            }
            section {
                h2 { "Operator" }
                p { "Inspect and roll out live experiment-directory entries with an admin-scoped session." }
                AdminSessionCard { session: admin_session_card_view }
                div {
                    label { "Admin Study ID" }
                    input {
                        value: "{admin_study_id}",
                        oninput: move |event| admin_study_id.set(event.value()),
                    }
                }
                div {
                    label { "Experiment ID" }
                    input {
                        value: "{admin_experiment_id}",
                        oninput: move |event| admin_experiment_id.set(event.value()),
                    }
                }
                div {
                    button { onclick: admin_github_login_action, "Sign In (Admin)" }
                    button { onclick: admin_load_directory_action, "Load Directory" }
                    button { onclick: admin_load_selected_entry_action, "Load Selected Entry" }
                    button { onclick: admin_upsert_editor_entry_action, "Upsert Editor Entry" }
                    button { onclick: admin_rollout_directory_action, "Roll Out Directory" }
                }
                p { "Admin session for selected study: {admin_scope_label}" }
                if !admin_granted_studies.is_empty() {
                    p { "Granted admin studies: {admin_granted_studies_label}" }
                }
                p { "{admin_status}" }
                div {
                    label { "Directory JSON" }
                    textarea {
                        value: "{admin_directory_json}",
                        rows: "18",
                        oninput: move |event| admin_directory_json.set(event.value()),
                    }
                }
                div {
                    label { "Entry Editor JSON" }
                    textarea {
                        value: "{admin_entry_json}",
                        rows: "16",
                        oninput: move |event| admin_entry_json.set(event.value()),
                    }
                }
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
            {local_training_section}
            if let Some(view) = view {
                section {
                    h2 { "Runtime" }
                    RuntimeCapabilityCard { summary: runtime_capability_summary(&view) }
                }
                if let Some(status) = lifecycle_assignment_status(&view) {
                    section {
                        h2 { "Assignment" }
                        LifecycleAssignmentStatusCard { status: status }
                    }
                }
                section {
                    h2 { "Network Training" }
                    TrainingResultPanel { result: latest_training_result_summary(&view) }
                    p { "Last loss: {view.training.last_loss.clone().unwrap_or_else(|| \"n/a\".into())}" }
                    p { "Throughput: {view.training.throughput_summary.clone().unwrap_or_else(|| \"n/a\".into())}" }
                    p { "Optimizer steps: {view.training.optimizer_steps.map(|v| v.to_string()).unwrap_or_else(|| \"n/a\".into())}" }
                    p { "Accepted samples: {view.training.accepted_samples.map(|v| v.to_string()).unwrap_or_else(|| \"n/a\".into())}" }
                }
                section {
                    h2 { "Network" }
                    TransportHealthPanel { network: view.network.clone() }
                }
                section {
                    h2 { "Leaderboard" }
                    ul {
                        for entry in view.viewer.leaderboard_preview {
                            li { "{entry.label}: {entry.score} ({entry.receipts} receipts)" }
                        }
                    }
                }
            }
        }
    }
}
