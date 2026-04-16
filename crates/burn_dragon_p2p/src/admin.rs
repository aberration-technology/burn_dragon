use anyhow::{Result, anyhow};
use burn_p2p::{ContentId, ExperimentDirectoryEntry, HeadAnnouncement, HeadId};
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

pub async fn fetch_signed_directory_entries(
    edge_base_url: &str,
    session_id: &str,
) -> Result<Vec<ExperimentDirectoryEntry>> {
    admin_client(edge_base_url, Some(session_id))
        .fetch_signed_directory()
        .await
        .map(|signed| signed.payload.payload.entries)
        .map_err(|error| anyhow!("failed to fetch signed directory entries: {error}"))
}

pub fn upsert_directory_entry(
    entries: &mut Vec<ExperimentDirectoryEntry>,
    replacement: ExperimentDirectoryEntry,
) {
    AdminClient::upsert_directory_entry(entries, replacement);
}

pub fn upsert_directory_entry_current_head(
    entries: &mut Vec<ExperimentDirectoryEntry>,
    template: &ExperimentDirectoryEntry,
    head_id: HeadId,
) -> bool {
    let mut replacement = entries
        .iter()
        .find(|entry| {
            entry.study_id == template.study_id
                && entry.experiment_id == template.experiment_id
                && entry.current_revision_id == template.current_revision_id
        })
        .cloned()
        .unwrap_or_else(|| template.clone());
    if replacement.current_head_id.as_ref() == Some(&head_id) {
        return false;
    }
    replacement.current_head_id = Some(head_id);
    upsert_directory_entry(entries, replacement);
    true
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    use burn_p2p::{
        ContentId, DatasetViewId, ExperimentId, ExperimentOptInPolicy,
        ExperimentResourceRequirements, ExperimentScope, ExperimentVisibility, NetworkId, PeerRole,
        PeerRoleSet, RevisionId, StudyId, WorkloadId,
    };

    fn sample_entry() -> ExperimentDirectoryEntry {
        ExperimentDirectoryEntry {
            network_id: NetworkId::new("burn-dragon-mainnet"),
            study_id: StudyId::new("burn-dragon-mainnet"),
            experiment_id: ExperimentId::new("nca-prepretraining"),
            workload_id: WorkloadId::new("dragon-nca-cpu"),
            display_name: "NCA".into(),
            model_schema_hash: ContentId::new("schema"),
            dataset_view_id: DatasetViewId::new("dataset"),
            resource_requirements: ExperimentResourceRequirements {
                minimum_roles: BTreeSet::from([PeerRole::TrainerGpu]),
                minimum_device_memory_bytes: None,
                minimum_system_memory_bytes: Some(1),
                estimated_download_bytes: 1,
                estimated_window_seconds: 30,
            },
            visibility: ExperimentVisibility::Public,
            opt_in_policy: ExperimentOptInPolicy::Open,
            current_revision_id: RevisionId::new("nca-r1"),
            current_head_id: None,
            allowed_roles: PeerRoleSet::new([PeerRole::TrainerGpu]),
            allowed_scopes: BTreeSet::from([ExperimentScope::Connect]),
            metadata: BTreeMap::from([("dragon_profile".into(), "{}".into())]),
        }
    }

    #[test]
    fn upsert_directory_entry_current_head_updates_existing_matching_entry() {
        let template = sample_entry();
        let mut entries = vec![template.clone()];

        let changed = upsert_directory_entry_current_head(
            &mut entries,
            &template,
            burn_p2p::HeadId::new("head-1"),
        );

        assert!(changed);
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0].current_head_id.as_ref().map(|id| id.as_str()),
            Some("head-1")
        );
    }

    #[test]
    fn upsert_directory_entry_current_head_inserts_template_when_missing() {
        let template = sample_entry();
        let mut entries = Vec::new();

        let changed = upsert_directory_entry_current_head(
            &mut entries,
            &template,
            burn_p2p::HeadId::new("head-1"),
        );

        assert!(changed);
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0].current_head_id.as_ref().map(|id| id.as_str()),
            Some("head-1")
        );
    }

    #[test]
    fn upsert_directory_entry_current_head_skips_when_head_matches() {
        let mut template = sample_entry();
        template.current_head_id = Some(burn_p2p::HeadId::new("head-1"));
        let mut entries = vec![template.clone()];

        let changed = upsert_directory_entry_current_head(
            &mut entries,
            &template,
            burn_p2p::HeadId::new("head-1"),
        );

        assert!(!changed);
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0].current_head_id.as_ref().map(|id| id.as_str()),
            Some("head-1")
        );
    }
}
