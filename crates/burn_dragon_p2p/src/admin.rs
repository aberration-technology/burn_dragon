use anyhow::{Result, anyhow};
use burn_p2p::{ContentId, ExperimentDirectoryEntry, HeadAnnouncement};
use burn_p2p_admin::{AdminClient, AdminClientConfig, AdminResult};

fn admin_client(edge_base_url: &str, session_id: Option<&str>) -> AdminClient {
    let config = session_id
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|session_id| {
            AdminClientConfig::new(edge_base_url).with_session_id(ContentId::new(session_id))
        })
        .unwrap_or_else(|| AdminClientConfig::new(edge_base_url));
    AdminClient::new(config)
}

pub async fn fetch_directory_entries(edge_base_url: &str) -> Result<Vec<ExperimentDirectoryEntry>> {
    admin_client(edge_base_url, None)
        .fetch_directory_entries()
        .await
        .map_err(|error| anyhow!("failed to fetch directory entries: {error}"))
}

pub fn upsert_directory_entry(
    entries: &mut Vec<ExperimentDirectoryEntry>,
    replacement: ExperimentDirectoryEntry,
) {
    AdminClient::upsert_directory_entry(entries, replacement);
}

pub async fn rollout_directory_entries(
    edge_base_url: &str,
    session_id: &str,
    directory_entries: Vec<ExperimentDirectoryEntry>,
) -> Result<AdminResult> {
    admin_client(edge_base_url, Some(session_id))
        .rollout_directory_entries(directory_entries)
        .await
        .map_err(|error| anyhow!("failed to roll out directory entries: {error}"))
}

pub async fn register_live_head(
    edge_base_url: &str,
    session_id: &str,
    announcement: HeadAnnouncement,
) -> Result<AdminResult> {
    admin_client(edge_base_url, Some(session_id))
        .register_live_head(announcement)
        .await
        .map_err(|error| anyhow!("failed to register live head on edge: {error}"))
}
