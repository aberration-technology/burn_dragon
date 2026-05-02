use anyhow::{Result, anyhow};
use burn_p2p::{ContentId, ExperimentDirectoryEntry, HeadAnnouncement, HeadDescriptor, HeadId};
use burn_p2p_admin::{AdminClient, AdminClientConfig, AdminResult};
#[cfg(feature = "native")]
use burn_p2p_publish::{PeerArtifactMirrorRequest, PeerArtifactMirrorResponse};

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

pub fn preserve_directory_entry_current_head(
    entries: &[ExperimentDirectoryEntry],
    replacement: &mut ExperimentDirectoryEntry,
) -> Option<HeadId> {
    if replacement.current_head_id.is_some() {
        return replacement.current_head_id.clone();
    }
    let current_head_id = entries
        .iter()
        .find(|entry| {
            entry.study_id == replacement.study_id
                && entry.experiment_id == replacement.experiment_id
                && entry.current_revision_id == replacement.current_revision_id
        })
        .and_then(|entry| entry.current_head_id.clone())?;
    replacement.current_head_id = Some(current_head_id.clone());
    Some(current_head_id)
}

pub fn recover_directory_current_head_from_visible_roots(
    entry: &ExperimentDirectoryEntry,
    heads: &[HeadDescriptor],
) -> Option<HeadId> {
    heads
        .iter()
        .filter(|head| {
            head.study_id == entry.study_id
                && head.experiment_id == entry.experiment_id
                && head.revision_id == entry.current_revision_id
                && head.parent_head_id.is_none()
        })
        .max_by(|left, right| {
            left.created_at
                .cmp(&right.created_at)
                .then_with(|| left.head_id.cmp(&right.head_id))
        })
        .map(|head| head.head_id.clone())
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

#[cfg(feature = "native")]
pub async fn mirror_peer_artifact(
    edge_base_url: &str,
    session_id: &str,
    request: PeerArtifactMirrorRequest,
) -> Result<PeerArtifactMirrorResponse> {
    let url = format!(
        "{}/admin/artifacts/mirror-peer",
        edge_base_url.trim_end_matches('/')
    );
    let response = reqwest::Client::new()
        .post(&url)
        .header("x-session-id", session_id)
        .json(&request)
        .send()
        .await
        .map_err(|error| anyhow!("failed to request peer artifact mirror: {error}"))?;
    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|error| anyhow!("failed to read peer artifact mirror response: {error}"))?;
    if !status.is_success() {
        return Err(anyhow!(
            "peer artifact mirror failed with status {status}: {body}"
        ));
    }
    serde_json::from_str(&body)
        .map_err(|error| anyhow!("failed to decode peer artifact mirror response: {error}: {body}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    use burn_p2p::{
        ArtifactId, ContentId, DatasetViewId, ExperimentId, ExperimentOptInPolicy,
        ExperimentResourceRequirements, ExperimentScope, ExperimentVisibility, HeadDescriptor,
        NetworkId, PeerRole, PeerRoleSet, RevisionId, StudyId, WorkloadId,
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

    #[test]
    fn preserve_directory_entry_current_head_keeps_existing_matching_revision() {
        let mut existing = sample_entry();
        existing.current_head_id = Some(HeadId::new("head-1"));
        let mut replacement = sample_entry();

        let preserved = preserve_directory_entry_current_head(&[existing], &mut replacement);

        assert_eq!(preserved.as_ref().map(|id| id.as_str()), Some("head-1"));
        assert_eq!(
            replacement.current_head_id.as_ref().map(|id| id.as_str()),
            Some("head-1")
        );
    }

    #[test]
    fn preserve_directory_entry_current_head_does_not_cross_revision() {
        let mut existing = sample_entry();
        existing.current_head_id = Some(HeadId::new("head-1"));
        existing.current_revision_id = RevisionId::new("old-revision");
        let mut replacement = sample_entry();

        let preserved = preserve_directory_entry_current_head(&[existing], &mut replacement);

        assert!(preserved.is_none());
        assert!(replacement.current_head_id.is_none());
    }

    fn sample_head(
        head_id: &str,
        global_step: u64,
        parent_head_id: Option<&str>,
    ) -> HeadDescriptor {
        HeadDescriptor {
            head_id: HeadId::new(head_id),
            study_id: StudyId::new("burn-dragon-mainnet"),
            experiment_id: ExperimentId::new("nca-prepretraining"),
            revision_id: RevisionId::new("nca-r1"),
            artifact_id: ArtifactId::new(format!("artifact-{head_id}")),
            parent_head_id: parent_head_id.map(HeadId::new),
            global_step,
            created_at: chrono::DateTime::from_timestamp(global_step as i64, 0)
                .expect("test timestamp should fit"),
            metrics: BTreeMap::new(),
        }
    }

    #[test]
    fn recover_directory_current_head_from_visible_roots_uses_latest_root_only() {
        let entry = sample_entry();
        let heads = vec![
            sample_head("older-root", 1, None),
            sample_head("newer-child", 3, Some("newer-root")),
            sample_head("newer-root", 2, None),
        ];

        let recovered = recover_directory_current_head_from_visible_roots(&entry, &heads);

        assert_eq!(recovered.as_ref().map(|id| id.as_str()), Some("newer-root"));
    }
}
