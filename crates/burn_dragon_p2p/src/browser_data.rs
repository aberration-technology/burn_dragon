use burn_p2p::WorkloadTrainingLease;
use burn_p2p_core::codec::multihash_sha256;

pub(crate) fn deterministic_sample_indices(
    sample_count: usize,
    max_samples: Option<usize>,
    selection_key: Option<&str>,
    training_lease: Option<&WorkloadTrainingLease>,
) -> Vec<usize> {
    let limit = max_samples.unwrap_or(sample_count).min(sample_count);
    let mut indices = (0..sample_count).collect::<Vec<_>>();
    let Some(material) = sample_selection_material(selection_key, training_lease) else {
        indices.truncate(limit);
        return indices;
    };

    indices.sort_by_key(|sample_index| {
        (
            sample_selection_rank(&material, *sample_index),
            *sample_index,
        )
    });
    indices.truncate(limit);
    indices
}

fn sample_selection_material(
    selection_key: Option<&str>,
    training_lease: Option<&WorkloadTrainingLease>,
) -> Option<String> {
    let has_selection_key = selection_key.is_some_and(|key| !key.trim().is_empty());
    if !has_selection_key && training_lease.is_none() {
        return None;
    }

    let mut material = selection_key
        .unwrap_or("browser-training")
        .trim()
        .to_owned();
    if let Some(lease) = training_lease {
        material.push_str("|lease=");
        material.push_str(lease.lease_id.as_str());
        material.push_str("|window=");
        material.push_str(&lease.window_id.0.to_string());
        material.push_str("|view=");
        material.push_str(lease.dataset_view_id.as_str());
        material.push_str("|assign=");
        material.push_str(lease.assignment_hash.as_str());
        material.push_str("|micro=");
        for microshard_id in &lease.microshards {
            material.push_str(microshard_id.as_str());
            material.push(',');
        }
    }
    Some(material)
}

fn sample_selection_rank(material: &str, sample_index: usize) -> u64 {
    let digest = multihash_sha256(format!("{material}\0{sample_index}").as_bytes());
    let bytes = digest.get(2..10).unwrap_or(&digest[..digest.len().min(8)]);
    let mut rank = [0_u8; 8];
    for (index, byte) in bytes.iter().enumerate() {
        rank[index] = *byte;
    }
    u64::from_be_bytes(rank)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_p2p::{ContentId, DatasetViewId, LeaseId, MicroShardId, WindowId};

    fn sample_lease(window_id: u64) -> WorkloadTrainingLease {
        WorkloadTrainingLease {
            lease_id: LeaseId::new(format!("lease-{window_id}")),
            window_id: WindowId(window_id),
            dataset_view_id: DatasetViewId::new("view"),
            assignment_hash: ContentId::new(format!("assignment-{window_id}")),
            microshards: vec![MicroShardId::new("micro-a"), MicroShardId::new("micro-b")],
        }
    }

    #[test]
    fn sample_selection_defaults_to_sequential_prefix_without_runtime_identity() {
        assert_eq!(
            deterministic_sample_indices(8, Some(3), None, None),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn sample_selection_uses_active_lease_to_rotate_browser_windows() {
        let first =
            deterministic_sample_indices(32, Some(8), Some("peer-a"), Some(&sample_lease(1)));
        let second =
            deterministic_sample_indices(32, Some(8), Some("peer-a"), Some(&sample_lease(2)));

        assert_eq!(first.len(), 8);
        assert_eq!(second.len(), 8);
        assert_ne!(first, second);
        assert!(first.iter().all(|index| *index < 32));
        assert!(second.iter().all(|index| *index < 32));
    }
}
