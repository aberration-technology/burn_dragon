use anyhow::{Result, anyhow};
use burn_p2p::ExperimentDirectoryEntry;
use serde::Serialize;

use crate::auth::fetch_edge_snapshot;

pub async fn fetch_directory_entries(edge_base_url: &str) -> Result<Vec<ExperimentDirectoryEntry>> {
    Ok(fetch_edge_snapshot(edge_base_url).await?.directory.entries)
}

pub fn upsert_directory_entry(
    entries: &mut Vec<ExperimentDirectoryEntry>,
    replacement: ExperimentDirectoryEntry,
) {
    if let Some(entry) = entries.iter_mut().find(|entry| {
        entry.study_id == replacement.study_id && entry.experiment_id == replacement.experiment_id
    }) {
        *entry = replacement;
    } else {
        entries.push(replacement);
    }
}

#[derive(Clone, Debug, Serialize)]
struct AuthPolicyRolloutRequest {
    minimum_revocation_epoch: Option<serde_json::Value>,
    directory_entries: Option<Vec<ExperimentDirectoryEntry>>,
    trusted_issuers: Option<serde_json::Value>,
    reenrollment: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Serialize)]
enum AdminActionRequest {
    RolloutAuthPolicy(AuthPolicyRolloutRequest),
}

#[cfg(not(target_arch = "wasm32"))]
async fn post_admin_action(
    edge_base_url: &str,
    session_id: &str,
    action: &AdminActionRequest,
) -> Result<serde_json::Value> {
    let response = reqwest::Client::new()
        .post(format!("{}/admin", edge_base_url.trim_end_matches('/')))
        .header("x-session-id", session_id)
        .json(action)
        .send()
        .await?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_else(|_| String::new());
        return Err(anyhow!(
            "admin rollout request failed with {status}: {}",
            body.trim()
        ));
    }
    response
        .json::<serde_json::Value>()
        .await
        .map_err(Into::into)
}

#[cfg(target_arch = "wasm32")]
async fn post_admin_action(
    edge_base_url: &str,
    session_id: &str,
    action: &AdminActionRequest,
) -> Result<serde_json::Value> {
    let response =
        gloo_net::http::Request::post(&format!("{}/admin", edge_base_url.trim_end_matches('/')))
            .header("x-session-id", session_id)
            .json(action)
            .map_err(|error| anyhow!("failed to encode admin rollout request: {error}"))?
            .send()
            .await
            .map_err(|error| anyhow!("failed to send admin rollout request: {error}"))?;
    let status = response.status();
    if !(200..300).contains(&status) {
        let body = response.text().await.unwrap_or_else(|_| String::new());
        return Err(anyhow!(
            "admin rollout request failed with {status}: {}",
            body.trim()
        ));
    }
    response
        .json::<serde_json::Value>()
        .await
        .map_err(|error| anyhow!("failed to decode admin rollout response: {error}"))
}

pub async fn rollout_directory_entries(
    edge_base_url: &str,
    session_id: &str,
    directory_entries: Vec<ExperimentDirectoryEntry>,
) -> Result<serde_json::Value> {
    post_admin_action(
        edge_base_url,
        session_id,
        &AdminActionRequest::RolloutAuthPolicy(AuthPolicyRolloutRequest {
            minimum_revocation_epoch: None,
            directory_entries: Some(directory_entries),
            trusted_issuers: None,
            reenrollment: None,
        }),
    )
    .await
}
