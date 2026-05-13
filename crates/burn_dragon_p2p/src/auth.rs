use std::collections::BTreeSet;
#[cfg(feature = "native")]
use std::fs;
#[cfg(feature = "native")]
use std::path::{Path, PathBuf};

#[cfg(feature = "native")]
use anyhow::Context;
use anyhow::{Result, anyhow, bail};
#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
use burn_p2p::NetworkId;
#[cfg(feature = "native")]
use burn_p2p::create_peer_auth_envelope;
use burn_p2p::{
    AuthConfig, BrowserEdgeSnapshot, ClientReleaseManifest, ContentId, EdgeAuthClient,
    EdgeEnrollmentConfig, EdgePeerIdentity, ExperimentDirectoryEntry, ExperimentScope, LoginStart,
    PrincipalSession,
};
#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
use burn_p2p_browser::durability::{
    clear_durable_browser_storage, clear_durable_receipt_outbox, load_durable_browser_storage,
    persist_durable_browser_storage,
};
#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
use burn_p2p_browser::{
    BrowserEdgeClient, BrowserEnrollmentConfig, BrowserSessionState, BrowserUiBindings,
    BrowserWorkerIdentity,
};
use chrono::{DateTime, Duration, Utc};
#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
use gloo_timers::future::TimeoutFuture;
#[cfg(any(feature = "native", all(feature = "wasm-ui", target_arch = "wasm32")))]
use libp2p_identity::Keypair;
use serde::{Deserialize, Serialize};
#[cfg(feature = "native")]
use url::Url;
#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
use wasm_bindgen::JsCast;

use crate::config::DragonNativeAuthBundle;
#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
use crate::p2p_adapter::browser_enrollment_config_from_snapshot;

fn login_provider_for_snapshot(
    snapshot: &BrowserEdgeSnapshot,
) -> Result<&burn_p2p::BrowserLoginProvider> {
    snapshot
        .login_providers
        .first()
        .ok_or_else(|| anyhow!("edge snapshot does not advertise a browser login provider"))
}

pub fn login_provider_label(snapshot: &BrowserEdgeSnapshot) -> Option<&str> {
    snapshot
        .login_providers
        .first()
        .map(|provider| provider.label.as_str())
}

pub fn native_edge_enrollment_config(
    snapshot: &BrowserEdgeSnapshot,
    release_manifest: &ClientReleaseManifest,
    requested_scopes: BTreeSet<ExperimentScope>,
    session_ttl_secs: i64,
) -> Result<EdgeEnrollmentConfig> {
    let trust_bundle = snapshot
        .trust_bundle
        .as_ref()
        .ok_or_else(|| anyhow!("edge snapshot is missing a trust bundle"))?;
    let provider = login_provider_for_snapshot(snapshot)?;

    if !snapshot.allowed_target_artifact_hashes.is_empty()
        && !snapshot
            .allowed_target_artifact_hashes
            .contains(&release_manifest.target_artifact_hash)
    {
        bail!(
            "release target artifact {} is not approved by the edge",
            release_manifest.target_artifact_hash.as_str()
        );
    }
    release_manifest
        .validate_for_edge_snapshot(snapshot)
        .map_err(|error| anyhow!("release manifest is incompatible with edge snapshot: {error}"))?;

    Ok(EdgeEnrollmentConfig {
        network_id: snapshot.network_id.clone(),
        project_family_id: trust_bundle.project_family_id.clone(),
        protocol_major: release_manifest.protocol_major,
        app_semver: release_manifest.app_semver.clone(),
        release_train_hash: release_manifest.release_train_hash.clone(),
        target_artifact_id: release_manifest.target_artifact_id.clone(),
        target_artifact_hash: release_manifest.target_artifact_hash.clone(),
        login_path: provider.login_path.clone(),
        device_path: provider.device_path.clone(),
        callback_path: provider.callback_path.clone().unwrap_or_default(),
        trusted_callback_header: None,
        trusted_callback_token: None,
        enroll_path: snapshot.paths.enroll_path.clone(),
        trust_bundle_path: snapshot.paths.trust_bundle_path.clone(),
        requested_scopes,
        session_ttl_secs,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DragonPendingGitHubLogin {
    pub edge_base_url: String,
    pub enrollment: EdgeEnrollmentConfig,
    pub login: LoginStart,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeCliBridgeBootstrap {
    pub edge_base_url: String,
    pub site_base_url: String,
    pub target_artifact_id: String,
    pub app_semver: String,
    pub git_commit: String,
    pub enabled_features_label: String,
    pub requested_scopes: BTreeSet<ExperimentScope>,
    pub session_ttl_secs: i64,
    pub principal_hint: Option<String>,
    pub identity: EdgePeerIdentity,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NativeCliBridgeAuthResult {
    pub edge_base_url: String,
    pub enrollment: EdgeEnrollmentConfig,
    pub session: PrincipalSession,
    pub certificate: burn_p2p::NodeCertificate,
}

#[derive(Clone, Debug)]
pub struct DragonGitHubSession {
    pub auth: DragonNativeAuthBundle,
    pub session: PrincipalSession,
}

#[cfg(feature = "native")]
const NATIVE_AUTH_CACHE_RELATIVE_PATH: &str = "state/native-github-auth.json";
#[cfg(feature = "native")]
const NATIVE_AUTH_REFRESH_SKEW_SECS: i64 = 60;

#[cfg(feature = "native")]
fn auth_refresh_deadline() -> DateTime<Utc> {
    Utc::now() + Duration::seconds(NATIVE_AUTH_REFRESH_SKEW_SECS)
}

#[cfg(feature = "native")]
pub fn default_native_auth_bundle_path(storage_root: &Path) -> PathBuf {
    storage_root.join(NATIVE_AUTH_CACHE_RELATIVE_PATH)
}

#[cfg(feature = "native")]
pub fn load_cached_native_auth_bundle(
    storage_root: &Path,
) -> Result<Option<DragonNativeAuthBundle>> {
    let path = default_native_auth_bundle_path(storage_root);
    if !path.is_file() {
        return Ok(None);
    }
    let bytes = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map(Some)
        .with_context(|| format!("failed to decode {}", path.display()))
}

#[cfg(feature = "native")]
pub fn store_cached_native_auth_bundle(
    storage_root: &Path,
    bundle: &DragonNativeAuthBundle,
) -> Result<()> {
    let path = default_native_auth_bundle_path(storage_root);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let body = serde_json::to_vec_pretty(bundle)?;
    fs::write(&path, body).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

#[cfg(feature = "native")]
fn native_auth_refresh_deadline() -> DateTime<Utc> {
    auth_refresh_deadline()
}

#[cfg(feature = "native")]
pub fn native_auth_bundle_is_fresh(bundle: &DragonNativeAuthBundle) -> bool {
    if bundle.auth_config.local_peer_auth.is_none() {
        return false;
    }
    let Some(session) = bundle.session.as_ref() else {
        return false;
    };
    let Some(certificate_not_after) = bundle.certificate_not_after else {
        return false;
    };
    let deadline = native_auth_refresh_deadline();
    session.expires_at > deadline && certificate_not_after > deadline
}

#[cfg(feature = "native")]
pub async fn refresh_native_auth_bundle(
    storage_root: &Path,
    bundle: &DragonNativeAuthBundle,
    client_manifest_id: Option<ContentId>,
) -> Result<DragonNativeAuthBundle> {
    let edge_base_url = bundle
        .edge_base_url
        .as_deref()
        .ok_or_else(|| anyhow!("native auth bundle is missing edge_base_url"))?;
    let enrollment = bundle
        .enrollment
        .clone()
        .ok_or_else(|| anyhow!("native auth bundle is missing enrollment metadata"))?;
    let prior_session = bundle
        .session
        .as_ref()
        .ok_or_else(|| anyhow!("native auth bundle is missing session metadata"))?;
    let client = EdgeAuthClient::new(edge_base_url, enrollment.clone());
    let session = client.refresh_session(&prior_session.session_id).await?;
    let (_, identity) = edge_peer_identity_for_storage(storage_root, None)?;
    let certificate = client
        .enroll(&client.build_enrollment_request(&session, &identity))
        .await?;
    let refreshed = finalize_native_auth_session(
        storage_root,
        edge_base_url,
        &enrollment,
        session,
        certificate,
        client_manifest_id,
        "github-refresh",
    )?;
    store_cached_native_auth_bundle(storage_root, &refreshed.auth)?;
    Ok(refreshed.auth)
}

#[cfg(feature = "native")]
pub fn native_cli_bridge_url(
    pending: &DragonPendingGitHubLogin,
    callback_url: &str,
    nonce: &str,
) -> Result<String> {
    let authorize_url = pending
        .login
        .authorize_url
        .as_deref()
        .ok_or_else(|| anyhow!("edge did not return a browser authorize URL"))?;
    let authorize = Url::parse(authorize_url)
        .with_context(|| format!("failed to parse browser authorize URL {authorize_url}"))?;
    let redirect_uri = authorize
        .query_pairs()
        .find_map(|(key, value)| (key == "redirect_uri").then(|| value.into_owned()))
        .ok_or_else(|| anyhow!("browser authorize URL is missing redirect_uri"))?;
    let mut bridge = Url::parse(&redirect_uri)
        .with_context(|| format!("failed to parse browser redirect URI {redirect_uri}"))?;
    {
        let mut query = bridge.query_pairs_mut();
        query.append_pair("native_cli", "1");
        query.append_pair("native_callback", callback_url);
        query.append_pair("native_nonce", nonce);
        query.append_pair("native_authorize", authorize_url);
    }
    Ok(bridge.to_string())
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
fn resolve_native_bridge_target_artifact_hash(snapshot: &BrowserEdgeSnapshot) -> Result<ContentId> {
    let mut allowed = snapshot
        .allowed_target_artifact_hashes
        .iter()
        .cloned()
        .collect::<Vec<_>>();
    if allowed.is_empty()
        && let Some(trust_bundle) = snapshot.trust_bundle.as_ref()
    {
        allowed.extend(trust_bundle.allowed_target_artifact_hashes.iter().cloned());
    }
    if allowed.is_empty() {
        bail!("edge snapshot is missing allowed target artifact hashes")
    }
    if allowed.len() == 1 {
        return Ok(allowed.remove(0));
    }

    let nativeish = allowed
        .into_iter()
        .filter(|hash| {
            let label = hash.as_str().to_ascii_lowercase();
            label.contains("native") || !label.contains("browser")
        })
        .collect::<Vec<_>>();
    if nativeish.len() == 1 {
        return Ok(nativeish
            .into_iter()
            .next()
            .expect("native target hash exists"));
    }

    bail!("edge snapshot advertises multiple target artifact hashes for native auth bootstrap")
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
fn native_release_manifest_for_bridge(
    snapshot: &BrowserEdgeSnapshot,
    bootstrap: &NativeCliBridgeBootstrap,
) -> Result<ClientReleaseManifest> {
    let trust_bundle = snapshot
        .trust_bundle
        .as_ref()
        .ok_or_else(|| anyhow!("edge snapshot is missing a trust bundle"))?;
    let release_train_hash = snapshot
        .required_release_train_hash
        .clone()
        .unwrap_or_else(|| trust_bundle.required_release_train_hash.clone());
    let release_manifest = ClientReleaseManifest {
        project_family_id: trust_bundle.project_family_id.clone(),
        release_train_hash,
        target_artifact_id: bootstrap.target_artifact_id.clone(),
        target_artifact_hash: resolve_native_bridge_target_artifact_hash(snapshot)?,
        target_platform: burn_p2p::ClientPlatform::Native,
        app_semver: semver::Version::parse(&bootstrap.app_semver)
            .map_err(|error| anyhow!("invalid app semver {}: {error}", bootstrap.app_semver))?,
        git_commit: bootstrap.git_commit.clone(),
        cargo_lock_hash: ContentId::new("dragon-native-auth-lock"),
        burn_version_string: "0.21.0".into(),
        enabled_features_hash: ContentId::new(bootstrap.enabled_features_label.clone()),
        protocol_major: snapshot.protocol_major,
        supported_workloads: Vec::new(),
        built_at: Utc::now(),
    };
    release_manifest
        .validate_for_edge_snapshot(snapshot)
        .map_err(|error| {
            anyhow!("native bridge release manifest is incompatible with edge snapshot: {error}")
        })?;
    Ok(release_manifest)
}

#[cfg(feature = "native")]
pub fn native_cli_browser_auth_url(
    bootstrap: &NativeCliBridgeBootstrap,
    callback_url: &str,
    nonce: &str,
) -> Result<String> {
    let site_base_url = if bootstrap.site_base_url.ends_with('/') {
        bootstrap.site_base_url.clone()
    } else {
        format!("{}/", bootstrap.site_base_url)
    };
    let mut bridge = Url::parse(&site_base_url)
        .with_context(|| {
            format!(
                "failed to parse browser site base URL {}",
                bootstrap.site_base_url
            )
        })?
        .join("callback/github")
        .with_context(|| {
            format!(
                "failed to resolve browser callback against {}",
                bootstrap.site_base_url
            )
        })?;
    let bootstrap_json = serde_json::to_string(bootstrap)?;
    {
        let mut query = bridge.query_pairs_mut();
        query.append_pair("native_cli", "1");
        query.append_pair("native_callback", callback_url);
        query.append_pair("native_nonce", nonce);
        query.append_pair("native_auth_bootstrap", &bootstrap_json);
    }
    Ok(bridge.to_string())
}

#[cfg(feature = "native")]
pub fn finalize_native_auth_session_from_bridge_result(
    storage_root: &Path,
    result: &NativeCliBridgeAuthResult,
    client_manifest_id: Option<ContentId>,
) -> Result<DragonGitHubSession> {
    let authenticated = finalize_native_auth_session(
        storage_root,
        &result.edge_base_url,
        &result.enrollment,
        result.session.clone(),
        result.certificate.clone(),
        client_manifest_id,
        "github-auth-browser-bridge",
    )?;
    store_cached_native_auth_bundle(storage_root, &authenticated.auth)?;
    Ok(authenticated)
}

#[cfg(feature = "native")]
fn finalize_native_auth_session(
    storage_root: &Path,
    edge_base_url: &str,
    enrollment: &EdgeEnrollmentConfig,
    session: PrincipalSession,
    certificate: burn_p2p::NodeCertificate,
    client_manifest_id: Option<ContentId>,
    auth_event_label: &str,
) -> Result<DragonGitHubSession> {
    let (node_keypair, _) = edge_peer_identity_for_storage(storage_root, None)?;
    let trust_bundle_endpoint = format!(
        "{}{}",
        edge_base_url.trim_end_matches('/'),
        enrollment.trust_bundle_path
    );
    let certificate_not_after = certificate.claims().not_after;
    let peer_auth = create_peer_auth_envelope(
        &node_keypair,
        certificate,
        client_manifest_id,
        enrollment.requested_scopes.clone(),
        ContentId::new(format!(
            "{auth_event_label}-{}",
            Utc::now().timestamp_nanos_opt().unwrap_or_default()
        )),
        Utc::now(),
    )?;
    Ok(DragonGitHubSession {
        auth: DragonNativeAuthBundle {
            auth_config: AuthConfig::new()
                .with_local_peer_auth(peer_auth)
                .with_trust_bundle_endpoint(trust_bundle_endpoint.clone()),
            trust_bundle_endpoint,
            edge_base_url: Some(edge_base_url.trim_end_matches('/').to_owned()),
            session_id: Some(session.session_id.as_str().to_owned()),
            principal_id: Some(session.claims.principal_id.as_str().to_owned()),
            enrollment: Some(enrollment.clone()),
            session: Some(session.clone()),
            certificate_not_after: Some(certificate_not_after),
        },
        session,
    })
}

#[cfg(feature = "native")]
fn identity_key_path(storage_root: &Path) -> PathBuf {
    storage_root.join("state").join("identity.key")
}

#[cfg(feature = "native")]
fn load_or_generate_node_keypair(storage_root: &Path) -> Result<Keypair> {
    let path = identity_key_path(storage_root);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    if path.is_file() {
        let bytes =
            fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        return Keypair::from_protobuf_encoding(&bytes)
            .map_err(|error| anyhow!("failed to decode {}: {error}", path.display()));
    }
    let keypair = Keypair::generate_ed25519();
    let bytes = keypair
        .to_protobuf_encoding()
        .map_err(|error| anyhow!("failed to encode identity keypair: {error}"))?;
    fs::write(&path, bytes).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(keypair)
}

#[cfg(feature = "native")]
pub fn edge_peer_identity_for_storage(
    storage_root: &Path,
    client_policy_hash: Option<ContentId>,
) -> Result<(Keypair, EdgePeerIdentity)> {
    let keypair = load_or_generate_node_keypair(storage_root)?;
    let peer_id = burn_p2p::PeerId::new(
        libp2p_identity::PeerId::from_public_key(&keypair.public()).to_string(),
    );
    let public_key_hex = hex::encode(keypair.public().encode_protobuf());
    let identity = EdgePeerIdentity {
        peer_id,
        peer_public_key_hex: public_key_hex,
        serial: 1,
        client_policy_hash,
    };
    Ok((keypair, identity))
}

#[cfg(feature = "native")]
pub async fn fetch_edge_snapshot(edge_base_url: &str) -> Result<BrowserEdgeSnapshot> {
    reqwest::Client::new()
        .get(format!(
            "{}/portal/snapshot",
            edge_base_url.trim_end_matches('/')
        ))
        .send()
        .await?
        .error_for_status()?
        .json::<BrowserEdgeSnapshot>()
        .await
        .map_err(Into::into)
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
pub async fn fetch_edge_snapshot(edge_base_url: &str) -> Result<BrowserEdgeSnapshot> {
    let url = format!("{}/portal/snapshot", edge_base_url.trim_end_matches('/'));
    const EDGE_SNAPSHOT_FETCH_ATTEMPTS: usize = 3;
    const EDGE_SNAPSHOT_RETRY_DELAY_MILLIS: u32 = 300;

    let mut last_error = None;
    for attempt in 0..EDGE_SNAPSHOT_FETCH_ATTEMPTS {
        let response = match gloo_net::http::Request::get(&url).send().await {
            Ok(response) => response,
            Err(error) => {
                last_error = Some(anyhow!("failed to fetch edge snapshot: {error}"));
                if attempt + 1 < EDGE_SNAPSHOT_FETCH_ATTEMPTS {
                    TimeoutFuture::new(EDGE_SNAPSHOT_RETRY_DELAY_MILLIS * (attempt as u32 + 1))
                        .await;
                    continue;
                }
                break;
            }
        };
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        if !response.ok() {
            last_error = Some(anyhow!(
                "failed to fetch edge snapshot {}: http {} {}",
                url,
                status,
                trim_preview(&body)
            ));
        } else {
            match parse_edge_snapshot_body(&body) {
                Ok(snapshot) => return Ok(snapshot),
                Err(error) => last_error = Some(error),
            }
        }
        if attempt + 1 < EDGE_SNAPSHOT_FETCH_ATTEMPTS {
            TimeoutFuture::new(EDGE_SNAPSHOT_RETRY_DELAY_MILLIS * (attempt as u32 + 1)).await;
        }
    }
    Err(last_error.unwrap_or_else(|| anyhow!("failed to fetch edge snapshot: unknown error")))
}

pub async fn begin_native_github_login(
    edge_base_url: &str,
    release_manifest: &ClientReleaseManifest,
    requested_scopes: BTreeSet<ExperimentScope>,
    session_ttl_secs: i64,
    principal_hint: Option<String>,
    use_device_flow: bool,
) -> Result<DragonPendingGitHubLogin> {
    let snapshot = fetch_edge_snapshot(edge_base_url).await?;
    let enrollment = native_edge_enrollment_config(
        &snapshot,
        release_manifest,
        requested_scopes,
        session_ttl_secs,
    )?;
    let client = EdgeAuthClient::new(edge_base_url, enrollment.clone());
    let login = if use_device_flow {
        client.begin_device_login(principal_hint).await?
    } else {
        client.begin_login(principal_hint).await?
    };
    Ok(DragonPendingGitHubLogin {
        edge_base_url: edge_base_url.trim_end_matches('/').to_owned(),
        enrollment,
        login,
    })
}

#[cfg(feature = "native")]
pub async fn complete_native_github_login(
    storage_root: &Path,
    pending: &DragonPendingGitHubLogin,
    provider_code: &str,
    client_manifest_id: Option<ContentId>,
) -> Result<DragonGitHubSession> {
    let client = EdgeAuthClient::new(&pending.edge_base_url, pending.enrollment.clone());
    let session = client
        .complete_provider_login(&pending.login, provider_code.to_owned())
        .await?;
    let (_, identity) = edge_peer_identity_for_storage(storage_root, None)?;
    let certificate = client
        .enroll(&client.build_enrollment_request(&session, &identity))
        .await?;
    let authenticated = finalize_native_auth_session(
        storage_root,
        &pending.edge_base_url,
        &pending.enrollment,
        session,
        certificate,
        client_manifest_id,
        "github-auth",
    )?;
    store_cached_native_auth_bundle(storage_root, &authenticated.auth)?;
    Ok(authenticated)
}

#[cfg(feature = "native")]
#[allow(clippy::too_many_arguments)]
pub async fn enroll_native_static_principal(
    storage_root: &Path,
    edge_base_url: &str,
    release_manifest: &ClientReleaseManifest,
    requested_scopes: BTreeSet<ExperimentScope>,
    session_ttl_secs: i64,
    principal_hint: Option<String>,
    principal_id: burn_p2p::PrincipalId,
    trusted_callback_token: Option<String>,
    client_manifest_id: Option<ContentId>,
) -> Result<DragonGitHubSession> {
    let snapshot = fetch_edge_snapshot(edge_base_url).await?;
    let enrollment = native_edge_enrollment_config(
        &snapshot,
        release_manifest,
        requested_scopes,
        session_ttl_secs,
    )?;
    let mut client = EdgeAuthClient::new(edge_base_url, enrollment.clone());
    if let Some(token) = trusted_callback_token.filter(|value| !value.trim().is_empty()) {
        client = client.with_trusted_callback("x-burn-p2p-canary-token", token);
    }
    let (_, identity) = edge_peer_identity_for_storage(storage_root, None)?;
    let enrolled = client
        .enroll_static_principal(principal_hint, principal_id, &identity)
        .await?;
    let authenticated = finalize_native_auth_session(
        storage_root,
        edge_base_url,
        &enrollment,
        enrolled.session,
        enrolled.certificate,
        client_manifest_id,
        "static-auth",
    )?;
    store_cached_native_auth_bundle(storage_root, &authenticated.auth)?;
    Ok(authenticated)
}

pub fn compose_auth_config(
    base: Option<AuthConfig>,
    github_auth: Option<&DragonNativeAuthBundle>,
    experiment_directory: &[ExperimentDirectoryEntry],
) -> AuthConfig {
    let mut auth = base.unwrap_or_default();
    if let Some(github_auth) = github_auth {
        auth.local_peer_auth = github_auth.auth_config.local_peer_auth.clone();
        auth.trust_bundle_endpoints = github_auth.auth_config.trust_bundle_endpoints.clone();
    }
    auth.experiment_directory = experiment_directory.to_vec();
    auth
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
pub fn browser_github_enrollment_config(
    snapshot: &BrowserEdgeSnapshot,
    release_manifest: &ClientReleaseManifest,
    requested_scopes: BTreeSet<ExperimentScope>,
    session_ttl_secs: i64,
) -> Result<BrowserEnrollmentConfig> {
    browser_enrollment_config_from_snapshot(
        snapshot,
        release_manifest,
        requested_scopes,
        session_ttl_secs,
    )
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
const PENDING_GITHUB_LOGIN_KEY: &str = "burn-dragon-p2p.pending-github-login";
#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
const TRUSTED_CALLBACK_TOKEN_KEY: &str = "burn-dragon-p2p.canary-callback-token";
#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
const BROWSER_WORKER_IDENTITY_KEY_PREFIX: &str = "burn-dragon-p2p.browser-worker-identity.";
#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
const DURABLE_BROWSER_STORAGE_PREFIX: &str = "burn-p2p.browser.storage.";
#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
const DURABLE_BROWSER_RECEIPT_OUTBOX_PREFIX: &str = "burn-p2p.browser.receipt-outbox.";
#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
const PENDING_GITHUB_LOGIN_TTL_SECS: i64 = 15 * 60;
#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
const NATIVE_CALLBACK_BRIDGE_KEY: &str = "burn-dragon-p2p.native-cli-bridge";
#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
const NATIVE_CALLBACK_BRIDGE_AUTH_KEY: &str = "burn-dragon-p2p.native-cli-bridge-auth";

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
#[derive(Clone, Debug, Serialize, Deserialize)]
struct PendingBrowserGitHubLogin {
    edge_base_url: String,
    created_at: DateTime<Utc>,
    login: LoginStart,
    requested_scopes: BTreeSet<ExperimentScope>,
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
#[derive(Clone, Debug, Serialize, Deserialize)]
struct StoredBrowserWorkerIdentity {
    keypair_protobuf_hex: String,
    #[serde(default = "default_browser_worker_identity_serial")]
    serial: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    client_policy_hash: Option<ContentId>,
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
fn default_browser_worker_identity_serial() -> u64 {
    1
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct PendingNativeCliBridge {
    callback_url: String,
    nonce: String,
    authorize_url: Option<String>,
    auth_bootstrap: Option<NativeCliBridgeBootstrap>,
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
#[derive(Clone, Debug, Serialize, Deserialize)]
struct PendingNativeCliBridgeAuth {
    created_at: DateTime<Utc>,
    bootstrap: NativeCliBridgeBootstrap,
    login: LoginStart,
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum StoredPendingBrowserGitHubLogin {
    Current(PendingBrowserGitHubLogin),
    Legacy {
        login: LoginStart,
        requested_scopes: BTreeSet<ExperimentScope>,
    },
    LegacyLoginStart(LoginStart),
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
fn normalize_edge_base_url(edge_base_url: &str) -> String {
    edge_base_url.trim_end_matches('/').to_owned()
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
fn pending_browser_login_is_expired(created_at: DateTime<Utc>, now: DateTime<Utc>) -> bool {
    now.signed_duration_since(created_at) > Duration::seconds(PENDING_GITHUB_LOGIN_TTL_SECS)
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
fn pending_native_cli_bridge_auth_is_expired(
    created_at: DateTime<Utc>,
    login: &LoginStart,
    now: DateTime<Utc>,
) -> bool {
    pending_browser_login_is_expired(created_at, now) || login.expires_at <= now
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
fn parse_stored_pending_browser_login(
    raw: &str,
    edge_base_url: &str,
    requested_scopes: BTreeSet<ExperimentScope>,
    now: DateTime<Utc>,
) -> Result<PendingBrowserGitHubLogin> {
    Ok(
        match serde_json::from_str::<StoredPendingBrowserGitHubLogin>(raw)? {
            StoredPendingBrowserGitHubLogin::Current(pending) => pending,
            StoredPendingBrowserGitHubLogin::Legacy {
                login,
                requested_scopes,
            } => PendingBrowserGitHubLogin {
                edge_base_url: normalize_edge_base_url(edge_base_url),
                created_at: now,
                login,
                requested_scopes,
            },
            StoredPendingBrowserGitHubLogin::LegacyLoginStart(login) => PendingBrowserGitHubLogin {
                edge_base_url: normalize_edge_base_url(edge_base_url),
                created_at: now,
                login,
                requested_scopes,
            },
        },
    )
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
fn trim_preview(body: &str) -> String {
    const LIMIT: usize = 240;
    let trimmed = body.trim();
    let preview = trimmed.chars().take(LIMIT).collect::<String>();
    if preview.len() == trimmed.len() {
        preview
    } else {
        format!("{preview}...")
    }
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
fn parse_edge_snapshot_body(body: &str) -> Result<BrowserEdgeSnapshot> {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        bail!("failed to decode edge snapshot: empty response body");
    }
    serde_json::from_str(trimmed).map_err(|error| {
        anyhow!(
            "failed to decode edge snapshot: {} ({})",
            error,
            trim_preview(trimmed)
        )
    })
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
fn browser_storage() -> Result<web_sys::Storage> {
    web_sys::window()
        .ok_or_else(|| anyhow!("window unavailable"))?
        .local_storage()
        .map_err(|error| anyhow!("localStorage unavailable: {error:?}"))?
        .ok_or_else(|| anyhow!("localStorage unavailable"))
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
fn browser_worker_identity_storage_key(network_id: &NetworkId) -> String {
    format!(
        "{}{}",
        BROWSER_WORKER_IDENTITY_KEY_PREFIX,
        network_id.as_str()
    )
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
fn browser_worker_identity_from_keypair(
    keypair: &Keypair,
    serial: u64,
    client_policy_hash: Option<ContentId>,
) -> BrowserWorkerIdentity {
    BrowserWorkerIdentity {
        peer_id: burn_p2p::PeerId::new(
            libp2p_identity::PeerId::from_public_key(&keypair.public()).to_string(),
        ),
        peer_public_key_hex: hex::encode(keypair.public().encode_protobuf()),
        serial,
        client_policy_hash,
    }
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
fn load_or_generate_browser_worker_identity(
    network_id: &NetworkId,
) -> Result<BrowserWorkerIdentity> {
    let storage = browser_storage()?;
    let key = browser_worker_identity_storage_key(network_id);
    if let Some(raw) = storage
        .get_item(&key)
        .map_err(|error| anyhow!("failed to read browser worker identity: {error:?}"))?
    {
        let stored: StoredBrowserWorkerIdentity = serde_json::from_str(&raw)
            .map_err(|error| anyhow!("failed to decode browser worker identity: {error}"))?;
        let bytes = hex::decode(&stored.keypair_protobuf_hex).map_err(|error| {
            anyhow!("failed to decode browser worker identity keypair: {error}")
        })?;
        let keypair = Keypair::from_protobuf_encoding(&bytes).map_err(|error| {
            anyhow!("failed to restore browser worker identity keypair: {error}")
        })?;
        return Ok(browser_worker_identity_from_keypair(
            &keypair,
            stored.serial,
            stored.client_policy_hash,
        ));
    }

    let keypair = Keypair::generate_ed25519();
    let stored = StoredBrowserWorkerIdentity {
        keypair_protobuf_hex: hex::encode(
            keypair
                .to_protobuf_encoding()
                .map_err(|error| anyhow!("failed to encode browser worker identity: {error}"))?,
        ),
        serial: 1,
        client_policy_hash: None,
    };
    storage
        .set_item(&key, &serde_json::to_string(&stored)?)
        .map_err(|error| anyhow!("failed to persist browser worker identity: {error:?}"))?;
    Ok(browser_worker_identity_from_keypair(
        &keypair,
        stored.serial,
        stored.client_policy_hash,
    ))
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
fn browser_session_storage() -> Result<web_sys::Storage> {
    web_sys::window()
        .ok_or_else(|| anyhow!("window unavailable"))?
        .session_storage()
        .map_err(|error| anyhow!("sessionStorage unavailable: {error:?}"))?
        .ok_or_else(|| anyhow!("sessionStorage unavailable"))
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
fn validate_native_cli_loopback_callback(callback_url: &str) -> Result<()> {
    let callback = url::Url::parse(callback_url)
        .map_err(|error| anyhow!("failed to parse native callback URL {callback_url}: {error}"))?;
    if callback.scheme() != "http" {
        bail!("native CLI callback must use http scheme");
    }
    let Some(host) = callback.host() else {
        bail!("native CLI callback must include a host");
    };
    let loopback = match host {
        url::Host::Domain(domain) => domain.eq_ignore_ascii_case("localhost"),
        url::Host::Ipv4(ip) => ip.is_loopback(),
        url::Host::Ipv6(ip) => ip.is_loopback(),
    };
    if !loopback {
        bail!("native CLI callback host must be loopback");
    }
    if callback.port().is_none() {
        bail!("native CLI callback must include an explicit port");
    }
    Ok(())
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
fn parse_native_cli_bridge_launch(query: &str) -> Result<Option<PendingNativeCliBridge>> {
    let mut requested = false;
    let mut callback_url = None;
    let mut nonce = None;
    let mut authorize_url = None;
    let mut auth_bootstrap = None;
    for (key, value) in url::form_urlencoded::parse(query.trim_start_matches('?').as_bytes()) {
        match key.as_ref() {
            "native_cli" if value == "1" || value.eq_ignore_ascii_case("true") => {
                requested = true;
            }
            "native_callback" if !value.trim().is_empty() => {
                callback_url = Some(value.into_owned());
            }
            "native_nonce" if !value.trim().is_empty() => {
                nonce = Some(value.into_owned());
            }
            "native_authorize" if !value.trim().is_empty() => {
                authorize_url = Some(value.into_owned());
            }
            "native_auth_bootstrap" if !value.trim().is_empty() => {
                auth_bootstrap = Some(
                    serde_json::from_str::<NativeCliBridgeBootstrap>(value.as_ref()).map_err(
                        |error| anyhow!("failed to decode native auth bootstrap: {error}"),
                    )?,
                );
            }
            _ => {}
        }
    }
    if !requested {
        return Ok(None);
    }
    if auth_bootstrap.is_none() {
        bail!("native CLI bridge is missing native_auth_bootstrap");
    }
    let callback_url =
        callback_url.ok_or_else(|| anyhow!("native CLI bridge is missing native_callback"))?;
    validate_native_cli_loopback_callback(&callback_url)?;
    Ok(Some(PendingNativeCliBridge {
        callback_url,
        nonce: nonce.ok_or_else(|| anyhow!("native CLI bridge is missing native_nonce"))?,
        authorize_url,
        auth_bootstrap,
    }))
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
fn load_pending_native_cli_bridge() -> Result<Option<PendingNativeCliBridge>> {
    let Some(raw) = browser_session_storage()?
        .get_item(NATIVE_CALLBACK_BRIDGE_KEY)
        .map_err(|error| anyhow!("failed to read native CLI bridge state: {error:?}"))?
    else {
        return Ok(None);
    };
    serde_json::from_str(&raw)
        .map(Some)
        .map_err(|error| anyhow!("failed to decode native CLI bridge state: {error}"))
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
fn store_pending_native_cli_bridge(bridge: &PendingNativeCliBridge) -> Result<()> {
    browser_session_storage()?
        .set_item(NATIVE_CALLBACK_BRIDGE_KEY, &serde_json::to_string(bridge)?)
        .map_err(|error| anyhow!("failed to persist native CLI bridge state: {error:?}"))?;
    Ok(())
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
fn clear_pending_native_cli_bridge() -> Result<()> {
    browser_session_storage()?
        .remove_item(NATIVE_CALLBACK_BRIDGE_KEY)
        .map_err(|error| anyhow!("failed to clear native CLI bridge state: {error:?}"))?;
    Ok(())
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
fn load_pending_native_cli_bridge_auth() -> Result<Option<PendingNativeCliBridgeAuth>> {
    let Some(raw) = browser_session_storage()?
        .get_item(NATIVE_CALLBACK_BRIDGE_AUTH_KEY)
        .map_err(|error| anyhow!("failed to read native CLI auth bridge state: {error:?}"))?
    else {
        return Ok(None);
    };
    serde_json::from_str(&raw)
        .map(Some)
        .map_err(|error| anyhow!("failed to decode native CLI auth bridge state: {error}"))
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
fn store_pending_native_cli_bridge_auth(pending: &PendingNativeCliBridgeAuth) -> Result<()> {
    browser_session_storage()?
        .set_item(
            NATIVE_CALLBACK_BRIDGE_AUTH_KEY,
            &serde_json::to_string(pending)?,
        )
        .map_err(|error| anyhow!("failed to persist native CLI auth bridge state: {error:?}"))?;
    Ok(())
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
fn clear_pending_native_cli_bridge_auth() -> Result<()> {
    browser_session_storage()?
        .remove_item(NATIVE_CALLBACK_BRIDGE_AUTH_KEY)
        .map_err(|error| anyhow!("failed to clear native CLI auth bridge state: {error:?}"))?;
    Ok(())
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
fn browser_worker_identity_from_edge_identity(
    identity: &EdgePeerIdentity,
) -> BrowserWorkerIdentity {
    BrowserWorkerIdentity {
        peer_id: identity.peer_id.clone(),
        peer_public_key_hex: identity.peer_public_key_hex.clone(),
        serial: identity.serial,
        client_policy_hash: identity.client_policy_hash.clone(),
    }
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
fn native_cli_bridge_callback_url(
    bridge: &PendingNativeCliBridge,
    provider_code: &str,
    state: &str,
) -> Result<String> {
    validate_native_cli_loopback_callback(&bridge.callback_url)?;
    let mut url = url::Url::parse(&bridge.callback_url).map_err(|error| {
        anyhow!(
            "failed to parse native callback URL {}: {error}",
            bridge.callback_url
        )
    })?;
    {
        let mut query = url.query_pairs_mut();
        query.append_pair("native_nonce", &bridge.nonce);
        query.append_pair("provider_code", provider_code);
        query.append_pair("state", state);
    }
    Ok(url.to_string())
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
fn submit_native_cli_bridge_form(
    bridge: &PendingNativeCliBridge,
    fields: &[(&str, String)],
) -> Result<()> {
    let window = web_sys::window().ok_or_else(|| anyhow!("window unavailable"))?;
    let document = window
        .document()
        .ok_or_else(|| anyhow!("document unavailable"))?;
    let body = document
        .body()
        .ok_or_else(|| anyhow!("document body unavailable"))?;
    let form = document
        .create_element("form")
        .map_err(|error| anyhow!("failed to create callback form: {error:?}"))?
        .dyn_into::<web_sys::HtmlFormElement>()
        .map_err(|_| anyhow!("failed to cast callback form"))?;
    form.set_method("POST");
    form.set_action(&bridge.callback_url);
    form.set_attribute("style", "display:none")
        .map_err(|error| anyhow!("failed to style callback form: {error:?}"))?;
    for (name, value) in fields {
        let input = document
            .create_element("input")
            .map_err(|error| anyhow!("failed to create callback field: {error:?}"))?
            .dyn_into::<web_sys::HtmlInputElement>()
            .map_err(|_| anyhow!("failed to cast callback field"))?;
        input.set_type("hidden");
        input.set_name(name);
        input.set_value(value);
        form.append_child(&input)
            .map_err(|error| anyhow!("failed to append callback field: {error:?}"))?;
    }
    body.append_child(&form)
        .map_err(|error| anyhow!("failed to attach callback form: {error:?}"))?;
    form.submit()
        .map_err(|error| anyhow!("failed to submit native CLI callback form: {error:?}"))?;
    Ok(())
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
fn submit_native_cli_bridge_auth_result(
    bridge: &PendingNativeCliBridge,
    result: &NativeCliBridgeAuthResult,
) -> Result<()> {
    submit_native_cli_bridge_form(
        bridge,
        &[
            ("native_nonce", bridge.nonce.clone()),
            ("auth_result_json", serde_json::to_string(result)?),
        ],
    )
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
fn submit_native_cli_bridge_error(bridge: &PendingNativeCliBridge, message: &str) -> Result<()> {
    submit_native_cli_bridge_form(
        bridge,
        &[
            ("native_nonce", bridge.nonce.clone()),
            ("error_message", message.to_owned()),
        ],
    )
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
async fn begin_native_cli_browser_auth(bridge: &PendingNativeCliBridge) -> Result<String> {
    let bootstrap = bridge
        .auth_bootstrap
        .as_ref()
        .ok_or_else(|| anyhow!("native CLI bridge is missing browser auth bootstrap"))?;
    let snapshot = BrowserEdgeClient::new(
        BrowserUiBindings::new(&bootstrap.edge_base_url),
        BrowserEnrollmentConfig::for_runtime_sync(
            &fetch_edge_snapshot(&bootstrap.edge_base_url).await?,
        ),
    )
    .fetch_browser_edge_snapshot()
    .await?;
    let release_manifest = native_release_manifest_for_bridge(&snapshot, bootstrap)?;
    let enrollment = browser_github_enrollment_config(
        &snapshot,
        &release_manifest,
        bootstrap.requested_scopes.clone(),
        bootstrap.session_ttl_secs,
    )?;
    let client =
        BrowserEdgeClient::new(BrowserUiBindings::new(&bootstrap.edge_base_url), enrollment);
    let login = client.begin_login(bootstrap.principal_hint.clone()).await?;
    store_pending_native_cli_bridge_auth(&PendingNativeCliBridgeAuth {
        created_at: Utc::now(),
        bootstrap: bootstrap.clone(),
        login: login.clone(),
    })?;
    login
        .authorize_url
        .clone()
        .ok_or_else(|| anyhow!("edge did not return a browser authorize URL"))
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
async fn complete_native_cli_browser_auth(
    bridge: &PendingNativeCliBridge,
    provider_code: &str,
    state: &str,
) -> Result<()> {
    let pending = load_pending_native_cli_bridge_auth()?
        .ok_or_else(|| anyhow!("missing pending native CLI browser auth state"))?;
    if pending_native_cli_bridge_auth_is_expired(pending.created_at, &pending.login, Utc::now()) {
        let _ = clear_pending_native_cli_bridge_auth();
        bail!("pending native CLI auth state expired; restart CLI login");
    }
    if pending.login.state != state {
        let _ = clear_pending_native_cli_bridge_auth();
        bail!("browser auth callback state mismatch");
    }

    let snapshot = BrowserEdgeClient::new(
        BrowserUiBindings::new(&pending.bootstrap.edge_base_url),
        BrowserEnrollmentConfig::for_runtime_sync(
            &fetch_edge_snapshot(&pending.bootstrap.edge_base_url).await?,
        ),
    )
    .fetch_browser_edge_snapshot()
    .await?;
    let release_manifest = native_release_manifest_for_bridge(&snapshot, &pending.bootstrap)?;
    let browser_enrollment = browser_github_enrollment_config(
        &snapshot,
        &release_manifest,
        pending.bootstrap.requested_scopes.clone(),
        pending.bootstrap.session_ttl_secs,
    )?;
    let native_enrollment = native_edge_enrollment_config(
        &snapshot,
        &release_manifest,
        pending.bootstrap.requested_scopes.clone(),
        pending.bootstrap.session_ttl_secs,
    )?;
    let client = BrowserEdgeClient::new(
        BrowserUiBindings::new(&pending.bootstrap.edge_base_url),
        browser_enrollment,
    );
    let session = client
        .complete_provider_login(&pending.login, provider_code.to_owned())
        .await?;
    let identity = browser_worker_identity_from_edge_identity(&pending.bootstrap.identity);
    let certificate = client
        .enroll(&client.build_enrollment_request(&session, &identity))
        .await?;

    let auth_result = NativeCliBridgeAuthResult {
        edge_base_url: normalize_edge_base_url(&pending.bootstrap.edge_base_url),
        enrollment: native_enrollment,
        session,
        certificate,
    };
    let _ = clear_pending_native_cli_bridge_auth();
    let _ = clear_pending_native_cli_bridge();
    submit_native_cli_bridge_auth_result(bridge, &auth_result)
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
pub fn native_cli_bridge_mode_active() -> bool {
    native_cli_bridge_mode_active_result().unwrap_or(false)
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
fn native_cli_bridge_mode_active_result() -> Result<bool> {
    let window = web_sys::window().ok_or_else(|| anyhow!("window unavailable"))?;
    let location = window.location();
    let search = location
        .search()
        .map_err(|error| anyhow!("failed to inspect browser query params: {error:?}"))?;
    let query = search.strip_prefix('?').unwrap_or(&search);
    if parse_native_cli_bridge_launch(query)?.is_some() {
        return Ok(true);
    }
    let pathname = location
        .pathname()
        .map_err(|error| anyhow!("failed to inspect browser path: {error:?}"))?;
    Ok(pathname.contains("/callback/") && load_pending_native_cli_bridge()?.is_some())
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
pub async fn resume_or_complete_native_cli_bridge() -> Result<bool> {
    let window = web_sys::window().ok_or_else(|| anyhow!("window unavailable"))?;
    let location = window.location();
    let search = location
        .search()
        .map_err(|error| anyhow!("failed to inspect browser query params: {error:?}"))?;
    let query = search.strip_prefix('?').unwrap_or(&search);

    if let Some(bridge) = parse_native_cli_bridge_launch(query)? {
        store_pending_native_cli_bridge(&bridge)?;
        let authorize_url = if bridge.auth_bootstrap.is_some() {
            match begin_native_cli_browser_auth(&bridge).await {
                Ok(authorize_url) => authorize_url,
                Err(error) => {
                    let _ = clear_pending_native_cli_bridge_auth();
                    let _ = clear_pending_native_cli_bridge();
                    let _ = submit_native_cli_bridge_error(&bridge, &error.to_string());
                    return Err(error);
                }
            }
        } else {
            bridge
                .authorize_url
                .clone()
                .ok_or_else(|| anyhow!("native CLI bridge is missing native_authorize"))?
        };
        location
            .set_href(&authorize_url)
            .map_err(|error| anyhow!("failed to redirect browser auth bridge: {error:?}"))?;
        return Ok(true);
    }

    let Some(provider_code) = provider_code_from_window_location() else {
        return Ok(false);
    };
    let Some(bridge) = load_pending_native_cli_bridge()? else {
        return Ok(false);
    };
    let state = provider_state_from_window_location()
        .ok_or_else(|| anyhow!("browser auth callback is missing state for native CLI relay"))?;
    if bridge.auth_bootstrap.is_some() {
        match complete_native_cli_browser_auth(&bridge, &provider_code, &state).await {
            Ok(()) => Ok(true),
            Err(error) => {
                let _ = clear_pending_native_cli_bridge_auth();
                let _ = clear_pending_native_cli_bridge();
                let _ = submit_native_cli_bridge_error(&bridge, &error.to_string());
                Err(error)
            }
        }
    } else {
        let callback_url = native_cli_bridge_callback_url(&bridge, &provider_code, &state)?;
        location.set_href(&callback_url).map_err(|error| {
            anyhow!("failed to relay browser auth back to native CLI: {error:?}")
        })?;
        let _ = clear_pending_native_cli_bridge();
        Ok(true)
    }
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
fn load_trusted_callback_token() -> Result<Option<String>> {
    let Some(raw) = browser_session_storage()?
        .get_item(TRUSTED_CALLBACK_TOKEN_KEY)
        .map_err(|error| anyhow!("failed to read trusted callback token: {error:?}"))?
    else {
        return Ok(None);
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        Ok(None)
    } else {
        Ok(Some(trimmed.to_owned()))
    }
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
fn clear_trusted_callback_token() -> Result<()> {
    browser_session_storage()?
        .remove_item(TRUSTED_CALLBACK_TOKEN_KEY)
        .map_err(|error| anyhow!("failed to clear trusted callback token: {error:?}"))?;
    Ok(())
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
fn clear_pending_browser_login_state() -> Result<()> {
    browser_storage()?
        .remove_item(PENDING_GITHUB_LOGIN_KEY)
        .map_err(|error| anyhow!("failed to clear pending login: {error:?}"))?;
    Ok(())
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
fn clear_browser_local_storage_prefix(prefix: &str) -> Result<()> {
    let storage = browser_storage()?;
    let mut matching_keys = Vec::new();
    let storage_len = storage
        .length()
        .map_err(|error| anyhow!("failed to inspect browser storage length: {error:?}"))?;
    for index in 0..storage_len {
        if let Some(key) = storage
            .key(index)
            .map_err(|error| anyhow!("failed to inspect browser storage key: {error:?}"))?
            .filter(|key| key.starts_with(prefix))
        {
            matching_keys.push(key);
        }
    }
    for key in matching_keys {
        storage
            .remove_item(&key)
            .map_err(|error| anyhow!("failed to clear browser storage key {key}: {error:?}"))?;
    }
    Ok(())
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
pub async fn reset_browser_runtime_state(edge_base_url: &str) -> Result<()> {
    let _ = clear_pending_browser_login_state();
    let _ = clear_pending_native_cli_bridge();
    let _ = clear_trusted_callback_token();
    let _ = clear_browser_local_storage_prefix(DURABLE_BROWSER_STORAGE_PREFIX);
    let _ = clear_browser_local_storage_prefix(DURABLE_BROWSER_RECEIPT_OUTBOX_PREFIX);

    let snapshot = BrowserEdgeClient::new(
        BrowserUiBindings::new(edge_base_url),
        BrowserEnrollmentConfig::for_runtime_sync(&fetch_edge_snapshot(edge_base_url).await?),
    )
    .fetch_browser_edge_snapshot()
    .await?;
    clear_durable_receipt_outbox(&snapshot.network_id)
        .await
        .map_err(|error| anyhow!("failed to clear durable browser receipt outbox: {error}"))?;
    clear_durable_browser_storage(&snapshot.network_id)
        .await
        .map_err(|error| anyhow!("failed to clear durable browser storage: {error}"))?;
    Ok(())
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
pub async fn begin_browser_github_login(
    edge_base_url: &str,
    release_manifest: &ClientReleaseManifest,
    requested_scopes: BTreeSet<ExperimentScope>,
    session_ttl_secs: i64,
    principal_hint: Option<String>,
) -> Result<LoginStart> {
    let snapshot = BrowserEdgeClient::new(
        BrowserUiBindings::new(edge_base_url),
        BrowserEnrollmentConfig::for_runtime_sync(&fetch_edge_snapshot(edge_base_url).await?),
    )
    .fetch_browser_edge_snapshot()
    .await?;
    let enrollment = browser_github_enrollment_config(
        &snapshot,
        release_manifest,
        requested_scopes.clone(),
        session_ttl_secs,
    )?;
    let client = BrowserEdgeClient::new(BrowserUiBindings::new(edge_base_url), enrollment);
    let login = client.begin_login(principal_hint).await?;
    let pending = PendingBrowserGitHubLogin {
        edge_base_url: normalize_edge_base_url(edge_base_url),
        created_at: Utc::now(),
        login: login.clone(),
        requested_scopes,
    };
    browser_storage()?
        .set_item(PENDING_GITHUB_LOGIN_KEY, &serde_json::to_string(&pending)?)
        .map_err(|error| anyhow!("failed to persist pending login: {error:?}"))?;
    Ok(login)
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
pub async fn complete_browser_github_login(
    edge_base_url: &str,
    release_manifest: &ClientReleaseManifest,
    requested_scopes: BTreeSet<ExperimentScope>,
    session_ttl_secs: i64,
    provider_code: &str,
) -> Result<BrowserSessionState> {
    let storage = browser_storage()?;
    let pending = storage
        .get_item(PENDING_GITHUB_LOGIN_KEY)
        .map_err(|error| anyhow!("failed to read pending login: {error:?}"))?
        .ok_or_else(|| anyhow!("missing pending GitHub login state"))?;
    let now = Utc::now();
    let pending =
        match parse_stored_pending_browser_login(&pending, edge_base_url, requested_scopes, now) {
            Ok(pending) => pending,
            Err(error) => {
                let _ = storage.remove_item(PENDING_GITHUB_LOGIN_KEY);
                return Err(anyhow!("invalid pending GitHub login state: {error}"));
            }
        };
    if pending.edge_base_url != normalize_edge_base_url(edge_base_url) {
        let _ = storage.remove_item(PENDING_GITHUB_LOGIN_KEY);
        bail!(
            "pending GitHub login state belongs to a different edge; restart sign-in for {}",
            edge_base_url
        );
    }
    if pending_browser_login_is_expired(pending.created_at, now) {
        let _ = storage.remove_item(PENDING_GITHUB_LOGIN_KEY);
        bail!("pending GitHub login state expired; restart sign-in");
    }
    let snapshot = BrowserEdgeClient::new(
        BrowserUiBindings::new(edge_base_url),
        BrowserEnrollmentConfig::for_runtime_sync(&fetch_edge_snapshot(edge_base_url).await?),
    )
    .fetch_browser_edge_snapshot()
    .await?;
    let enrollment = browser_github_enrollment_config(
        &snapshot,
        release_manifest,
        pending.requested_scopes,
        session_ttl_secs,
    )?;
    let mut client = BrowserEdgeClient::new(BrowserUiBindings::new(edge_base_url), enrollment);
    if let Some(token) = load_trusted_callback_token()? {
        client = client.with_trusted_callback("x-burn-p2p-canary-token", token);
    }
    let session = client
        .complete_provider_login(&pending.login, provider_code.to_owned())
        .await?;
    let mut durable = load_durable_browser_storage(&snapshot.network_id)
        .await
        .map_err(|error| anyhow!("failed to load durable browser storage: {error}"))?;
    let enrolled_session = enroll_browser_session(
        &client,
        &snapshot.network_id,
        BrowserSessionState {
            session: Some(session),
            certificate: None,
            trust_bundle: client.fetch_trust_bundle().await.ok(),
            enrolled_at: Some(Utc::now()),
            reenrollment_required: false,
        },
    )
    .await?;
    durable.remember_session(enrolled_session.clone());
    persist_durable_browser_storage(&snapshot.network_id, &durable)
        .await
        .map_err(|error| anyhow!("failed to persist durable browser storage: {error}"))?;
    let _ = storage.remove_item(PENDING_GITHUB_LOGIN_KEY);
    let _ = clear_trusted_callback_token();
    Ok(enrolled_session)
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
async fn enroll_browser_session(
    client: &BrowserEdgeClient,
    network_id: &NetworkId,
    mut state: BrowserSessionState,
) -> Result<BrowserSessionState> {
    if state.certificate.is_some() && !state.reenrollment_required {
        return Ok(state);
    }
    let Some(session) = state.session.clone() else {
        return Ok(state);
    };
    let identity = load_or_generate_browser_worker_identity(network_id)?;
    let certificate = client
        .enroll(&client.build_enrollment_request(&session, &identity))
        .await?;
    state.session = Some(session);
    state.certificate = Some(certificate);
    state.trust_bundle = client
        .fetch_trust_bundle()
        .await
        .ok()
        .or(state.trust_bundle);
    state.enrolled_at = Some(Utc::now());
    state.refresh_reenrollment_requirement();
    Ok(state)
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
pub async fn load_or_enroll_browser_session(
    edge_base_url: &str,
    release_manifest: &ClientReleaseManifest,
    requested_scopes: BTreeSet<ExperimentScope>,
    session_ttl_secs: i64,
) -> Result<BrowserSessionState> {
    let snapshot = BrowserEdgeClient::new(
        BrowserUiBindings::new(edge_base_url),
        BrowserEnrollmentConfig::for_runtime_sync(&fetch_edge_snapshot(edge_base_url).await?),
    )
    .fetch_browser_edge_snapshot()
    .await?;
    let enrollment = browser_github_enrollment_config(
        &snapshot,
        release_manifest,
        requested_scopes,
        session_ttl_secs,
    )?;
    let client = BrowserEdgeClient::new(BrowserUiBindings::new(edge_base_url), enrollment);
    let mut durable = load_durable_browser_storage(&snapshot.network_id)
        .await
        .map_err(|error| anyhow!("failed to load durable browser storage: {error}"))?;
    let enrolled_session =
        enroll_browser_session(&client, &snapshot.network_id, durable.session.clone()).await?;
    durable.remember_session(enrolled_session.clone());
    persist_durable_browser_storage(&snapshot.network_id, &durable)
        .await
        .map_err(|error| anyhow!("failed to persist durable browser storage: {error}"))?;
    Ok(enrolled_session)
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
pub async fn load_browser_session(edge_base_url: &str) -> Result<BrowserSessionState> {
    let snapshot = BrowserEdgeClient::new(
        BrowserUiBindings::new(edge_base_url),
        BrowserEnrollmentConfig::for_runtime_sync(&fetch_edge_snapshot(edge_base_url).await?),
    )
    .fetch_browser_edge_snapshot()
    .await?;
    Ok(load_durable_browser_storage(&snapshot.network_id)
        .await
        .map_err(|error| anyhow!("failed to load durable browser storage: {error}"))?
        .session)
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
fn query_param_from_window_location(name: &str) -> Option<String> {
    let search = web_sys::window()?.location().search().ok()?;
    let query = search.strip_prefix('?').unwrap_or(&search);
    url::form_urlencoded::parse(query.as_bytes())
        .find_map(|(key, value)| (key == name).then(|| value.into_owned()))
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
pub fn provider_code_from_window_location() -> Option<String> {
    query_param_from_window_location("code")
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
pub fn provider_state_from_window_location() -> Option<String> {
    query_param_from_window_location("state")
}

#[cfg(all(test, feature = "wasm-ui", target_arch = "wasm32"))]
mod tests {
    use super::*;
    use burn_p2p_core::{
        BrowserDirectorySnapshot, BrowserEdgeMode, BrowserEdgePaths, BrowserLeaderboardSnapshot,
        BrowserTransportSurface,
    };

    #[test]
    fn pending_browser_login_parser_preserves_current_record() {
        let now = Utc::now();
        let raw = serde_json::to_string(&PendingBrowserGitHubLogin {
            edge_base_url: "https://edge.example".into(),
            created_at: now,
            login: LoginStart {
                login_id: burn_p2p::ContentId::new("login-1"),
                provider: burn_p2p::AuthProvider::GitHub,
                state: "state-1".into(),
                authorize_url: Some("https://github.example/auth".into()),
                expires_at: now + Duration::minutes(5),
            },
            requested_scopes: BTreeSet::from([ExperimentScope::Connect]),
        })
        .expect("serialize pending login");

        let parsed =
            parse_stored_pending_browser_login(&raw, "https://edge.example", BTreeSet::new(), now)
                .expect("parse current pending login");

        assert_eq!(parsed.edge_base_url, "https://edge.example");
        assert_eq!(parsed.login.state, "state-1");
        assert_eq!(
            parsed.requested_scopes,
            BTreeSet::from([ExperimentScope::Connect])
        );
    }

    #[test]
    fn pending_browser_login_expiration_is_enforced() {
        let now = Utc::now();

        assert!(pending_browser_login_is_expired(
            now - Duration::seconds(PENDING_GITHUB_LOGIN_TTL_SECS + 1),
            now
        ));
        assert!(!pending_browser_login_is_expired(
            now - Duration::seconds(PENDING_GITHUB_LOGIN_TTL_SECS),
            now
        ));
    }

    #[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
    #[test]
    fn parse_edge_snapshot_body_rejects_empty_payload() {
        let error = parse_edge_snapshot_body("").expect_err("empty snapshot should fail");
        assert!(error.to_string().contains("empty response body"));
    }

    #[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
    #[test]
    fn parse_edge_snapshot_body_trims_valid_json() {
        let raw = format!(
            "  \n{}\n ",
            serde_json::to_string(&sample_edge_snapshot()).unwrap()
        );
        let snapshot = parse_edge_snapshot_body(&raw).expect("valid snapshot");
        assert_eq!(snapshot.network_id.as_str(), "burn-dragon-mainnet");
    }

    #[test]
    fn native_cli_bridge_launch_parser_extracts_required_fields() {
        let bridge = parse_native_cli_bridge_launch(
            "native_cli=1&native_callback=http%3A%2F%2F127.0.0.1%3A43123%2Fcallback&native_nonce=nonce-1&native_authorize=https%3A%2F%2Fgithub.example%2Fauthorize",
        )
        .expect("parse bridge")
        .expect("bridge launch");
        assert_eq!(bridge.callback_url, "http://127.0.0.1:43123/callback");
        assert_eq!(bridge.nonce, "nonce-1");
        assert_eq!(
            bridge.authorize_url.as_deref(),
            Some("https://github.example/authorize")
        );
    }

    #[test]
    fn native_cli_bridge_launch_parser_rejects_non_loopback_callback() {
        let error = parse_native_cli_bridge_launch(
            "native_cli=1&native_callback=https%3A%2F%2Fattacker.example%2Fcollect&native_nonce=nonce-1&native_authorize=https%3A%2F%2Fgithub.example%2Fauthorize",
        )
        .expect_err("non-loopback callback should fail");
        assert!(
            error
                .to_string()
                .contains("native CLI callback host must be loopback")
        );
    }

    #[test]
    fn native_cli_bridge_callback_url_preserves_existing_query() {
        let bridge = PendingNativeCliBridge {
            callback_url: "http://127.0.0.1:43123/callback?source=dragon".into(),
            nonce: "nonce-1".into(),
            authorize_url: Some("https://github.example/authorize".into()),
            auth_bootstrap: None,
        };
        let callback = native_cli_bridge_callback_url(&bridge, "provider-code", "state-1")
            .expect("callback url");
        assert!(callback.contains("source=dragon"));
        assert!(callback.contains("native_nonce=nonce-1"));
        assert!(callback.contains("provider_code=provider-code"));
        assert!(callback.contains("state=state-1"));
    }

    #[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
    #[test]
    fn native_cli_bridge_mode_detection_handles_launch_and_callback_paths() {
        let launch = native_cli_bridge_mode_active_result_for_test(
            "/callback/github",
            "native_cli=1&native_callback=http%3A%2F%2F127.0.0.1%3A43123%2Fcallback&native_nonce=nonce-1&native_authorize=https%3A%2F%2Fgithub.example%2Fauthorize",
            false,
        )
        .expect("detect launch mode");
        assert!(launch);

        let callback = native_cli_bridge_mode_active_result_for_test(
            "/callback/github",
            "code=provider-code&state=state-1",
            true,
        )
        .expect("detect callback mode");
        assert!(callback);

        let normal = native_cli_bridge_mode_active_result_for_test("/", "", false)
            .expect("detect normal mode");
        assert!(!normal);
    }

    #[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
    fn native_cli_bridge_mode_active_result_for_test(
        pathname: &str,
        query: &str,
        pending_bridge: bool,
    ) -> Result<bool> {
        if parse_native_cli_bridge_launch(query)?.is_some() {
            return Ok(true);
        }
        Ok(pathname.contains("/callback/") && pending_bridge)
    }

    #[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
    fn sample_edge_snapshot() -> BrowserEdgeSnapshot {
        let now = Utc::now();
        BrowserEdgeSnapshot {
            network_id: burn_p2p::NetworkId::new("burn-dragon-mainnet"),
            protocol_major: 0,
            minimum_client_version: semver::Version::new(0, 0, 0),
            edge_mode: BrowserEdgeMode::Peer,
            browser_mode: burn_p2p::BrowserMode::Trainer,
            social_mode: burn_p2p::SocialMode::Public,
            profile_mode: burn_p2p::ProfileMode::Public,
            transports: BrowserTransportSurface {
                webrtc_direct: true,
                webtransport_gateway: false,
                wss_fallback: true,
            },
            paths: BrowserEdgePaths::default(),
            auth_enabled: false,
            login_providers: Vec::new(),
            required_release_train_hash: None,
            allowed_target_artifact_hashes: Default::default(),
            directory: BrowserDirectorySnapshot {
                network_id: burn_p2p::NetworkId::new("burn-dragon-mainnet"),
                generated_at: now,
                entries: Vec::new(),
            },
            heads: Vec::new(),
            leaderboard: BrowserLeaderboardSnapshot {
                network_id: burn_p2p::NetworkId::new("burn-dragon-mainnet"),
                score_version: "v1".into(),
                entries: Vec::new(),
                captured_at: now,
            },
            trust_bundle: None,
            captured_at: now,
        }
    }
}
