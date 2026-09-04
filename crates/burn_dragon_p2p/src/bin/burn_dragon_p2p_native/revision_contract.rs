//! Authority-signed revision-contract provisioning for a live Dragon revision.

use super::*;

use std::collections::BTreeSet;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

use burn_p2p::burn::BurnWorkloadAdapter;
use burn_p2p::{
    ArtifactKind, FsArtifactStore, GenesisArtifactLoadContext, MODEL_GENESIS_SIGNATURE_KEY_ID,
    ModelGenesisManifest, P2pWorkload, REVISION_CONTRACT_SIGNATURE_KEY_ID, RevisionContractBundle,
    RevisionId, SignedPayload, WorkloadId, sign_revision_contract_bundle,
    verify_revision_contract_with_trust_bundle,
};
use burn_p2p_core::{SchemaEnvelope, SignatureAlgorithm, SignatureMetadata};
use chrono::Utc;
use libp2p_identity::Keypair;

#[derive(Debug, Serialize)]
struct RevisionContractProvisionReport {
    experiment_id: String,
    revision_id: String,
    workload_id: String,
    contract_id: String,
    contract_path: PathBuf,
    signer: String,
    genesis_head_id: String,
    genesis_artifact_id: String,
    genesis_tensor_digest: String,
    retained_contracts: usize,
    reused_existing_contract: bool,
    rollout: Option<AdminResult>,
    edge_publication_verified: bool,
}

pub(super) fn admin_provision_revision_contract(
    args: AdminProvisionRevisionContractArgs,
) -> Result<()> {
    let requested_edge_url = args.edge_url.clone();
    let config = resolved_config(
        args.config.as_deref(),
        args.config_format,
        requested_edge_url.clone(),
        Vec::new(),
        None,
    )?;
    let experiment_kind = args.experiment_kind.into_config();
    let auth_bundle = resolve_or_login_native_auth_bundle(
        &config,
        experiment_kind,
        args.backend,
        NativeAuthResolutionOptions {
            auth_bundle_path: Some(args.auth_bundle.as_path()),
            auth_bundle_format: args.auth_bundle_format,
            principal_hint: None,
            session_ttl_secs: DEFAULT_SESSION_TTL_SECS,
            callback_timeout_secs: DEFAULT_AUTH_CALLBACK_TIMEOUT_SECS,
        },
    )?;
    let edge_base_url = requested_edge_url
        .or_else(|| auth_bundle.edge_base_url.clone())
        .or_else(|| config.effective_edge_base_url().map(ToOwned::to_owned))
        .ok_or_else(|| anyhow!("no edge base URL configured for revision-contract provisioning"))?;
    let session_id = auth_bundle
        .session_id
        .as_ref()
        .ok_or_else(|| anyhow!("auth bundle is missing a session_id for contract rollout"))?;

    let report = with_prepared_native_peer!(
        experiment_kind,
        args.backend,
        &config,
        Some(&auth_bundle),
        |prepared| provision_for_prepared_peer(
            prepared,
            &edge_base_url,
            session_id.as_str(),
            &args,
        )
    )?;
    write_output(None, args.output_format, &report)
}

fn provision_for_prepared_peer<B>(
    prepared: PreparedNativePeer<B>,
    edge_base_url: &str,
    session_id: &str,
    args: &AdminProvisionRevisionContractArgs,
) -> Result<RevisionContractProvisionReport>
where
    B: AutodiffBackend + Clone + 'static,
{
    anyhow::ensure!(args.authority_epoch > 0, "authority epoch must be positive");
    anyhow::ensure!(
        args.wait_timeout_secs > 0,
        "revision-contract wait timeout must be positive"
    );
    anyhow::ensure!(
        args.poll_interval_secs > 0,
        "revision-contract poll interval must be positive"
    );

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("build revision-contract provisioning runtime")?;
    let deadline = Instant::now() + Duration::from_secs(args.wait_timeout_secs);
    let store = FsArtifactStore::new(&prepared.storage_root);
    let (snapshot, genesis_head, genesis_artifact) = loop {
        let snapshot = runtime.block_on(fetch_edge_snapshot(edge_base_url))?;
        let revision = &prepared.manifests.revision_manifest;
        match active_genesis_head(
            &snapshot.directory.entries,
            &snapshot.heads,
            &revision.experiment_id,
            &revision.revision_id,
            &revision.workload_id,
        )
        .cloned()
        {
            Ok(genesis_head) => match store.load_manifest(&genesis_head.artifact_id) {
                Ok(descriptor) => break (snapshot, genesis_head, descriptor),
                Err(error) if Instant::now() < deadline => {
                    log::info!(
                        "waiting for local canonical genesis artifact {}: {error}",
                        genesis_head.artifact_id.as_str()
                    );
                }
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!(
                            "canonical genesis artifact {} was not materialized under {}",
                            genesis_head.artifact_id.as_str(),
                            prepared.storage_root.display()
                        )
                    });
                }
            },
            Err(error) if Instant::now() < deadline => {
                log::info!("waiting for active canonical genesis: {error:#}");
            }
            Err(error) => {
                return Err(error).context("active canonical genesis did not become ready");
            }
        }
        thread::sleep(Duration::from_secs(args.poll_interval_secs));
    };

    anyhow::ensure!(
        genesis_artifact.kind == ArtifactKind::FullHead
            && genesis_artifact.base_head_id.is_none()
            && genesis_artifact.head_id.as_ref() == Some(&genesis_head.head_id),
        "active genesis must be a complete full-head artifact bound to the root head"
    );

    if let Some(existing) = snapshot.revision_contracts.iter().find(|contract| {
        same_revision_scope(
            &contract.revision.experiment_id,
            &contract.revision.revision_id,
            &contract.revision.workload_id,
            &prepared.manifests.revision_manifest.experiment_id,
            &prepared.manifests.revision_manifest.revision_id,
            &prepared.manifests.revision_manifest.workload_id,
        )
    }) {
        anyhow::ensure!(
            existing.revision == prepared.manifests.revision_manifest
                && existing.training_contract_id == prepared.manifests.training_contract_id
                && existing.training == prepared.manifests.training_contract,
            "edge already publishes a semantically different contract for this revision"
        );
        anyhow::ensure!(
            existing.genesis.payload.payload.artifact == genesis_artifact,
            "edge revision contract genesis does not match the active canonical root artifact"
        );
        let trust_bundle = snapshot
            .trust_bundle
            .as_ref()
            .ok_or_else(|| anyhow!("edge snapshot has no authority trust bundle"))?;
        verify_revision_contract_with_trust_bundle(trust_bundle, existing)
            .context("verify existing edge revision contract")?;
        atomic_write_json(&args.contract_out, existing)?;
        return provision_report(
            existing,
            &genesis_head,
            args.contract_out.clone(),
            snapshot.revision_contracts.len(),
            true,
            None,
        );
    }

    let project = BurnWorkloadAdapter::try_new(
        prepared.project.clone(),
        prepared.manifests.workload_config.clone(),
    )
    .context("build Dragon workload adapter for genesis verification")?;
    let device = project.runtime_device();
    let initialized_model = project.init_model(&device);
    let canonical_model = project
        .load_genesis_artifact(
            initialized_model,
            GenesisArtifactLoadContext {
                descriptor: &genesis_artifact,
                training_contract_id: &prepared.manifests.training_contract_id,
                contract: &prepared.manifests.training_contract,
                materialization: &prepared.genesis_materialization,
                store: &store,
                device: &device,
            },
        )
        .context("decode canonical genesis artifact before authority signing")?;
    let tensor_digest = project
        .model_tensor_digest(&canonical_model)
        .context("compute canonical decoded-tensor digest")?;

    let authority = load_authority_key(&args.authority_key)?;
    let created_at = Utc::now();
    let trust_bundle = snapshot
        .trust_bundle
        .as_ref()
        .ok_or_else(|| anyhow!("edge snapshot has no authority trust bundle"))?;
    let genesis = ModelGenesisManifest {
        experiment_id: prepared.manifests.revision_manifest.experiment_id.clone(),
        revision_id: prepared.manifests.revision_manifest.revision_id.clone(),
        workload_id: prepared.manifests.revision_manifest.workload_id.clone(),
        training_contract_id: prepared.manifests.training_contract_id.clone(),
        artifact: genesis_artifact,
        tensor_digest,
        initialization_algorithm: args.initialization_algorithm.clone(),
        initialization_seed: Some(prepared.config.training.seed),
        materialization: prepared.genesis_materialization,
        authority_epoch: args
            .authority_epoch
            .max(trust_bundle.minimum_revocation_epoch.0),
        created_at,
    };
    let mut contract = RevisionContractBundle {
        revision: prepared.manifests.revision_manifest,
        training_contract_id: prepared.manifests.training_contract_id,
        training: prepared.manifests.training_contract,
        genesis: SignedPayload::new(
            SchemaEnvelope::new(
                "burn-p2p-model-genesis-v1",
                prepared.manifests.release_manifest.app_semver,
                genesis,
            ),
            unsigned_signature(MODEL_GENESIS_SIGNATURE_KEY_ID, created_at),
        )?,
        contract_signature: unsigned_signature(REVISION_CONTRACT_SIGNATURE_KEY_ID, created_at),
    };
    sign_revision_contract_bundle(&authority, &mut contract, created_at)
        .context("authority-sign Dragon revision contract")?;
    verify_revision_contract_with_trust_bundle(trust_bundle, &contract)
        .context("verify generated contract against the live edge trust bundle")?;
    atomic_write_json(&args.contract_out, &contract)?;

    let retained_contracts = snapshot
        .revision_contracts
        .iter()
        .filter(|existing| {
            !same_revision_scope(
                &existing.revision.experiment_id,
                &existing.revision.revision_id,
                &existing.revision.workload_id,
                &contract.revision.experiment_id,
                &contract.revision.revision_id,
                &contract.revision.workload_id,
            )
        })
        .cloned()
        .chain(std::iter::once(contract.clone()))
        .collect::<Vec<_>>();
    let rollout = runtime.block_on(rollout_revision_contracts(
        edge_base_url,
        session_id,
        retained_contracts.clone(),
        false,
    ))?;
    let published = runtime.block_on(fetch_edge_snapshot(edge_base_url))?;
    let edge_publication_verified = published
        .revision_contracts
        .iter()
        .any(|candidate| candidate == &contract);
    anyhow::ensure!(
        edge_publication_verified,
        "edge rollout returned success but the exact signed contract is absent from its snapshot"
    );
    provision_report(
        &contract,
        &genesis_head,
        args.contract_out.clone(),
        retained_contracts.len(),
        false,
        Some(rollout),
    )
}

fn same_revision_scope(
    left_experiment_id: &ExperimentId,
    left_revision_id: &RevisionId,
    left_workload_id: &WorkloadId,
    right_experiment_id: &ExperimentId,
    right_revision_id: &RevisionId,
    right_workload_id: &WorkloadId,
) -> bool {
    left_experiment_id == right_experiment_id
        && left_revision_id == right_revision_id
        && left_workload_id == right_workload_id
}

fn provision_report(
    contract: &RevisionContractBundle,
    genesis_head: &HeadDescriptor,
    contract_path: PathBuf,
    retained_contracts: usize,
    reused_existing_contract: bool,
    rollout: Option<AdminResult>,
) -> Result<RevisionContractProvisionReport> {
    let contract_id = ContentId::derive(contract)?;
    Ok(RevisionContractProvisionReport {
        experiment_id: contract.revision.experiment_id.as_str().to_owned(),
        revision_id: contract.revision.revision_id.as_str().to_owned(),
        workload_id: contract.revision.workload_id.as_str().to_owned(),
        contract_id: contract_id.as_str().to_owned(),
        contract_path,
        signer: contract.contract_signature.signer.as_str().to_owned(),
        genesis_head_id: genesis_head.head_id.as_str().to_owned(),
        genesis_artifact_id: contract
            .genesis
            .payload
            .payload
            .artifact
            .artifact_id
            .as_str()
            .to_owned(),
        genesis_tensor_digest: contract
            .genesis
            .payload
            .payload
            .tensor_digest
            .as_str()
            .to_owned(),
        retained_contracts,
        reused_existing_contract,
        rollout,
        edge_publication_verified: true,
    })
}

fn active_genesis_head<'a>(
    entries: &[ExperimentDirectoryEntry],
    heads: &'a [HeadDescriptor],
    experiment_id: &ExperimentId,
    revision_id: &RevisionId,
    workload_id: &WorkloadId,
) -> Result<&'a HeadDescriptor> {
    let entry = entries
        .iter()
        .find(|entry| {
            &entry.experiment_id == experiment_id
                && &entry.current_revision_id == revision_id
                && &entry.workload_id == workload_id
        })
        .ok_or_else(|| anyhow!("edge directory has no matching experiment revision"))?;
    let current_head_id = entry
        .current_head_id
        .as_ref()
        .ok_or_else(|| anyhow!("matching edge directory entry has no current head"))?;
    let mut cursor = heads
        .iter()
        .find(|head| &head.head_id == current_head_id)
        .ok_or_else(|| anyhow!("directory current head is not visible in the edge snapshot"))?;
    let mut visited = BTreeSet::new();
    loop {
        anyhow::ensure!(
            &cursor.experiment_id == experiment_id && &cursor.revision_id == revision_id,
            "active head lineage crosses the selected experiment revision"
        );
        anyhow::ensure!(
            visited.insert(cursor.head_id.clone()),
            "active head lineage contains a cycle"
        );
        let Some(parent_head_id) = cursor.parent_head_id.as_ref() else {
            anyhow::ensure!(
                cursor.global_step == 0,
                "active lineage root has nonzero global step {}",
                cursor.global_step
            );
            return Ok(cursor);
        };
        cursor = heads
            .iter()
            .find(|head| &head.head_id == parent_head_id)
            .ok_or_else(|| {
                anyhow!(
                    "active head lineage is incomplete at parent {}",
                    parent_head_id.as_str()
                )
            })?;
    }
}

fn unsigned_signature(key_id: &str, signed_at: chrono::DateTime<Utc>) -> SignatureMetadata {
    SignatureMetadata {
        signer: PeerId::new("unsigned"),
        key_id: key_id.into(),
        algorithm: SignatureAlgorithm::Ed25519,
        signed_at,
        signature_hex: "00".into(),
    }
}

fn load_authority_key(path: &Path) -> Result<Keypair> {
    ensure_secure_authority_key(path)?;
    let bytes = fs::read(path).with_context(|| format!("read authority key {}", path.display()))?;
    Keypair::from_protobuf_encoding(&bytes)
        .map_err(|error| anyhow!("decode authority key {}: {error}", path.display()))
}

#[cfg(unix)]
fn ensure_secure_authority_key(path: &Path) -> Result<()> {
    let metadata =
        fs::metadata(path).with_context(|| format!("inspect authority key {}", path.display()))?;
    anyhow::ensure!(
        metadata.permissions().mode() & 0o077 == 0,
        "authority key {} must not be group/world accessible",
        path.display()
    );
    Ok(())
}

#[cfg(not(unix))]
fn ensure_secure_authority_key(path: &Path) -> Result<()> {
    anyhow::ensure!(
        path.is_file(),
        "authority key does not exist: {}",
        path.display()
    );
    Ok(())
}

fn atomic_write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let temp = parent.join(format!(
        ".{}.tmp-{}",
        path.file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("revision-contract"),
        std::process::id()
    ));
    let bytes = serde_json::to_vec_pretty(value)?;
    fs::write(&temp, bytes).with_context(|| format!("write {}", temp.display()))?;
    fs::rename(&temp, path).with_context(|| format!("install {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use burn_p2p::{
        ArtifactId, DatasetViewId, ExperimentDirectoryEntry, ExperimentId, ExperimentOptInPolicy,
        ExperimentResourceRequirements, ExperimentScope, ExperimentVisibility, HeadId, MetricValue,
        NetworkId, PeerRoleSet, StudyId, TrainingProtocol,
    };

    fn head(id: &str, parent: Option<&str>, step: u64) -> HeadDescriptor {
        HeadDescriptor {
            head_id: HeadId::new(id),
            study_id: StudyId::new("study"),
            experiment_id: ExperimentId::new("experiment"),
            revision_id: RevisionId::new("revision"),
            artifact_id: ArtifactId::new(format!("artifact-{id}")),
            parent_head_id: parent.map(HeadId::new),
            global_step: step,
            created_at: Utc::now(),
            metrics: BTreeMap::<String, MetricValue>::new(),
        }
    }

    fn entry(current: &str) -> ExperimentDirectoryEntry {
        ExperimentDirectoryEntry {
            network_id: NetworkId::new("network"),
            study_id: StudyId::new("study"),
            experiment_id: ExperimentId::new("experiment"),
            workload_id: WorkloadId::new("workload"),
            display_name: "test".into(),
            model_schema_hash: ContentId::new("model"),
            dataset_view_id: DatasetViewId::new("dataset"),
            resource_requirements: ExperimentResourceRequirements {
                minimum_roles: BTreeSet::new(),
                minimum_device_memory_bytes: None,
                minimum_system_memory_bytes: None,
                estimated_download_bytes: 0,
                estimated_window_seconds: 0,
            },
            visibility: ExperimentVisibility::Public,
            opt_in_policy: ExperimentOptInPolicy::Open,
            current_revision_id: RevisionId::new("revision"),
            current_head_id: Some(HeadId::new(current)),
            allowed_roles: PeerRoleSet::default(),
            allowed_scopes: BTreeSet::<ExperimentScope>::new(),
            training_protocol: TrainingProtocol::default(),
            metadata: BTreeMap::new(),
        }
    }

    #[test]
    fn active_genesis_follows_the_directory_current_head_lineage() {
        let heads = vec![
            head("unrelated-root", None, 0),
            head("root", None, 0),
            head("middle", Some("root"), 1),
            head("current", Some("middle"), 2),
        ];
        let genesis = active_genesis_head(
            &[entry("current")],
            &heads,
            &ExperimentId::new("experiment"),
            &RevisionId::new("revision"),
            &WorkloadId::new("workload"),
        )
        .expect("active genesis");

        assert_eq!(genesis.head_id.as_str(), "root");
    }

    #[test]
    fn active_genesis_rejects_incomplete_lineage() {
        let heads = vec![head("current", Some("missing"), 2)];
        let error = active_genesis_head(
            &[entry("current")],
            &heads,
            &ExperimentId::new("experiment"),
            &RevisionId::new("revision"),
            &WorkloadId::new("workload"),
        )
        .expect_err("incomplete head lineage must be rejected");

        assert!(error.to_string().contains("incomplete at parent missing"));
    }

    #[test]
    fn revision_scope_requires_experiment_revision_and_workload_identity() {
        let experiment = ExperimentId::new("experiment");
        let revision = RevisionId::new("shared-revision-label");
        let workload = WorkloadId::new("workload");

        assert!(same_revision_scope(
            &experiment,
            &revision,
            &workload,
            &experiment,
            &revision,
            &workload,
        ));
        assert!(!same_revision_scope(
            &experiment,
            &revision,
            &workload,
            &ExperimentId::new("other-experiment"),
            &revision,
            &workload,
        ));
        assert!(!same_revision_scope(
            &experiment,
            &revision,
            &workload,
            &experiment,
            &revision,
            &WorkloadId::new("other-workload"),
        ));
    }
}
