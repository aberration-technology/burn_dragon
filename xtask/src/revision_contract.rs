use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use burn_p2p::{
    FsArtifactStore, MODEL_GENESIS_SIGNATURE_KEY_ID, ModelGenesisManifest, PeerId,
    REVISION_CONTRACT_SIGNATURE_KEY_ID, RevisionContractBundle, SignatureAlgorithm,
    SignatureMetadata, SignedPayload, TrustedIssuer, sign_revision_contract_bundle,
    verify_revision_contract_bundle, verify_revision_contract_with_trust_bundle,
};
use burn_p2p_core::{SchemaEnvelope, TrustBundleExport};
use chrono::{DateTime, Utc};
use clap::Args;
use libp2p_identity::Keypair;
use semver::Version;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize)]
pub struct RevisionContractBuildSpec {
    pub revision: burn_p2p::RevisionManifest,
    pub training: burn_p2p::TrainingContractManifest,
    pub genesis_artifact: burn_p2p::ArtifactDescriptor,
    pub tensor_digest: burn_p2p::ContentId,
    pub initialization_algorithm: String,
    #[serde(default)]
    pub initialization_seed: Option<u64>,
    /// Whether peers download the complete head or reconstruct immutable tensors.
    #[serde(default)]
    pub materialization: burn_p2p::GenesisMaterialization,
    pub authority_epoch: u64,
    pub created_at: DateTime<Utc>,
    pub protocol_version: Version,
}

#[derive(Clone, Debug, Args)]
pub struct BuildRevisionContractArgs {
    /// JSON build specification binding revision, training, and genesis.
    #[arg(long)]
    pub spec: PathBuf,
    /// Root of the artifact store containing the complete genesis artifact.
    #[arg(long)]
    pub artifact_store_root: PathBuf,
    /// Existing protobuf-encoded Ed25519 authority key. This command never creates it.
    #[arg(long)]
    pub authority_key: PathBuf,
    /// Atomic output path for the signed contract bundle.
    #[arg(long)]
    pub output: PathBuf,
}

#[derive(Clone, Debug, Args)]
pub struct VerifyRevisionContractArgs {
    #[arg(long)]
    pub bundle: PathBuf,
    #[arg(long)]
    pub trust_bundle: PathBuf,
    #[arg(long)]
    pub artifact_store_root: PathBuf,
}

#[derive(Clone, Debug, Args)]
pub struct VerifyEdgeRevisionContractArgs {
    #[arg(long)]
    pub edge_url: String,
    #[arg(long)]
    pub experiment_id: String,
    #[arg(long)]
    pub revision_id: String,
}

#[derive(Clone, Debug, Args)]
pub struct RotateRevisionContractsArgs {
    #[arg(long = "bundle", required = true)]
    pub bundles: Vec<PathBuf>,
    #[arg(long)]
    pub new_authority_key: PathBuf,
    /// Must not already exist; the complete directory is installed atomically.
    #[arg(long)]
    pub output_dir: PathBuf,
}

#[derive(Clone, Debug, Args)]
pub struct RolloutRevisionContractsArgs {
    #[arg(long)]
    pub edge_url: String,
    #[arg(long = "bundle", required = true)]
    pub bundles: Vec<PathBuf>,
    #[arg(long)]
    pub session_id: Option<String>,
    #[arg(long)]
    pub admin_token: Option<String>,
    #[arg(long, default_value_t = false)]
    pub allow_signature_rotation: bool,
}

#[derive(Clone, Debug, Serialize)]
struct RotationManifest {
    schema: &'static str,
    rotated_at: DateTime<Utc>,
    previous_signers: BTreeSet<PeerId>,
    new_signer: PeerId,
    contracts: BTreeMap<String, burn_p2p::ContentId>,
}

pub fn build_revision_contract(args: &BuildRevisionContractArgs) -> Result<()> {
    let spec: RevisionContractBuildSpec = read_json(&args.spec)?;
    verify_genesis_artifact(&args.artifact_store_root, &spec.genesis_artifact)?;
    let keypair = load_authority_key(&args.authority_key)?;
    let bundle = build_signed_bundle(spec, &keypair)?;
    atomic_write_json(&args.output, &bundle)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "revision_id": bundle.revision.revision_id,
            "training_contract_id": bundle.training_contract_id,
            "genesis_artifact_id": bundle.genesis.payload.payload.artifact.artifact_id,
            "signer": bundle.contract_signature.signer,
            "output": args.output,
        }))?
    );
    Ok(())
}

pub fn verify_revision_contract(args: &VerifyRevisionContractArgs) -> Result<()> {
    let bundle: RevisionContractBundle = read_json(&args.bundle)?;
    let trust: TrustBundleExport = read_json(&args.trust_bundle)?;
    verify_revision_contract_with_trust_bundle(&trust, &bundle)
        .context("verify revision contract authority signatures")?;
    verify_genesis_artifact(
        &args.artifact_store_root,
        &bundle.genesis.payload.payload.artifact,
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "verified": true,
            "revision_id": bundle.revision.revision_id,
            "training_contract_id": bundle.training_contract_id,
            "genesis_artifact_id": bundle.genesis.payload.payload.artifact.artifact_id,
            "signer": bundle.contract_signature.signer,
        }))?
    );
    Ok(())
}

pub fn verify_edge_revision_contract(args: &VerifyEdgeRevisionContractArgs) -> Result<()> {
    let url = format!("{}/portal/snapshot", args.edge_url.trim_end_matches('/'));
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .context("build browser edge verification client")?;
    let response = client
        .get(&url)
        .send()
        .with_context(|| format!("fetch browser edge snapshot from {url}"))?;
    let status = response.status();
    let bytes = response.bytes().context("read browser edge snapshot")?;
    ensure!(
        status.is_success(),
        "browser edge snapshot request failed with HTTP {}: {}",
        status,
        String::from_utf8_lossy(&bytes)
    );
    let snapshot: burn_p2p::BrowserEdgeSnapshot =
        serde_json::from_slice(&bytes).context("decode browser edge snapshot")?;
    let entry = snapshot
        .directory
        .entries
        .iter()
        .find(|entry| {
            entry.experiment_id.as_str() == args.experiment_id
                && entry.current_revision_id.as_str() == args.revision_id
        })
        .with_context(|| {
            format!(
                "edge directory has no experiment={} revision={}",
                args.experiment_id, args.revision_id
            )
        })?;
    let contract = snapshot
        .revision_contracts
        .iter()
        .find(|contract| {
            contract.revision.experiment_id == entry.experiment_id
                && contract.revision.revision_id == entry.current_revision_id
                && contract.revision.workload_id == entry.workload_id
        })
        .with_context(|| {
            format!(
                "edge has no authority-signed contract for experiment={} revision={} workload={}",
                entry.experiment_id.as_str(),
                entry.current_revision_id.as_str(),
                entry.workload_id.as_str()
            )
        })?;
    let trust = snapshot
        .trust_bundle
        .as_ref()
        .context("edge snapshot has no authority trust bundle")?;
    verify_revision_contract_with_trust_bundle(trust, contract)
        .context("verify edge revision contract authority signature")?;
    let contract_id = burn_p2p::ContentId::derive(contract)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "verified": true,
            "experiment_id": entry.experiment_id,
            "revision_id": entry.current_revision_id,
            "workload_id": entry.workload_id,
            "contract_id": contract_id,
            "signer": contract.contract_signature.signer,
            "genesis_artifact_id": contract.genesis.payload.payload.artifact.artifact_id,
        }))?
    );
    Ok(())
}

pub fn rotate_revision_contracts(args: &RotateRevisionContractsArgs) -> Result<()> {
    ensure!(
        !args.output_dir.exists(),
        "rotation output directory already exists: {}",
        args.output_dir.display()
    );
    let keypair = load_authority_key(&args.new_authority_key)?;
    let new_signer = peer_id_for_keypair(&keypair);
    let parent = args.output_dir.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let staging = parent.join(format!(
        ".{}.staging-{}",
        args.output_dir
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("revision-contracts"),
        std::process::id()
    ));
    ensure!(
        !staging.exists(),
        "rotation staging directory already exists: {}",
        staging.display()
    );
    fs::create_dir(&staging)?;

    let result = (|| -> Result<RotationManifest> {
        let mut previous_signers = BTreeSet::new();
        let mut contract_ids = BTreeMap::new();
        for path in &args.bundles {
            let mut bundle: RevisionContractBundle = read_json(path)?;
            let authority_payload = bundle.authority_payload();
            previous_signers.insert(bundle.contract_signature.signer.clone());
            sign_revision_contract_bundle(&keypair, &mut bundle, Utc::now())
                .with_context(|| format!("re-sign {}", path.display()))?;
            ensure!(
                bundle.authority_payload() == authority_payload,
                "signature rotation changed the authority payload for {}",
                bundle.revision.revision_id.as_str()
            );
            verify_with_keypair(&keypair, &bundle)?;
            let file_name = format!(
                "{}.revision-contract.json",
                bundle.revision.revision_id.as_str()
            );
            let output = staging.join(file_name);
            atomic_write_json(&output, &bundle)?;
            contract_ids.insert(
                bundle.revision.revision_id.as_str().to_owned(),
                burn_p2p::ContentId::derive(&bundle)?,
            );
        }
        Ok(RotationManifest {
            schema: "burn-dragon-revision-contract-rotation-v1",
            rotated_at: Utc::now(),
            previous_signers,
            new_signer,
            contracts: contract_ids,
        })
    })();

    match result {
        Ok(manifest) => {
            atomic_write_json(&staging.join("rotation-manifest.json"), &manifest)?;
            fs::rename(&staging, &args.output_dir).with_context(|| {
                format!(
                    "atomically install rotation directory {}",
                    args.output_dir.display()
                )
            })?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
            Ok(())
        }
        Err(error) => {
            let _ = fs::remove_dir_all(&staging);
            Err(error)
        }
    }
}

pub fn rollout_revision_contracts(args: &RolloutRevisionContractsArgs) -> Result<()> {
    ensure!(
        args.session_id.is_some() || args.admin_token.is_some(),
        "revision contract rollout requires --session-id or --admin-token"
    );
    let contracts = args
        .bundles
        .iter()
        .map(|path| read_json(path))
        .collect::<Result<Vec<RevisionContractBundle>>>()?;
    ensure!(!contracts.is_empty(), "at least one contract is required");
    for contract in &contracts {
        contract.validate()?;
    }
    let action = burn_p2p_admin::AdminAction::RolloutRevisionContracts {
        contracts,
        allow_signature_rotation: args.allow_signature_rotation,
    };
    let mut request = reqwest::blocking::Client::new()
        .post(format!("{}/admin", args.edge_url.trim_end_matches('/')))
        .json(&action);
    if let Some(session_id) = args.session_id.as_deref() {
        request = request.header("x-session-id", session_id);
    }
    if let Some(admin_token) = args.admin_token.as_deref() {
        request = request.header("x-admin-token", admin_token);
    }
    let response = request.send().context("post revision contract rollout")?;
    let status = response.status();
    let bytes = response.bytes().context("read rollout response")?;
    ensure!(
        status.is_success(),
        "revision contract rollout failed with HTTP {}: {}",
        status,
        String::from_utf8_lossy(&bytes)
    );
    let result: burn_p2p_admin::AdminResult =
        serde_json::from_slice(&bytes).context("decode revision contract rollout response")?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn build_signed_bundle(
    spec: RevisionContractBuildSpec,
    keypair: &Keypair,
) -> Result<RevisionContractBundle> {
    ensure!(
        spec.genesis_artifact.kind == burn_p2p::ArtifactKind::FullHead
            && spec.genesis_artifact.base_head_id.is_none(),
        "genesis must be a complete full-head artifact with no base head"
    );
    let training_contract_id = spec.training.contract_id()?;
    let genesis = ModelGenesisManifest {
        experiment_id: spec.revision.experiment_id.clone(),
        revision_id: spec.revision.revision_id.clone(),
        workload_id: spec.revision.workload_id.clone(),
        training_contract_id: training_contract_id.clone(),
        artifact: spec.genesis_artifact,
        tensor_digest: spec.tensor_digest,
        initialization_algorithm: spec.initialization_algorithm,
        initialization_seed: spec.initialization_seed,
        materialization: spec.materialization,
        authority_epoch: spec.authority_epoch,
        created_at: spec.created_at,
    };
    let placeholder = SignatureMetadata {
        signer: PeerId::new("unsigned"),
        key_id: MODEL_GENESIS_SIGNATURE_KEY_ID.into(),
        algorithm: SignatureAlgorithm::Ed25519,
        signed_at: spec.created_at,
        signature_hex: "00".into(),
    };
    let mut bundle = RevisionContractBundle {
        revision: spec.revision,
        training_contract_id,
        training: spec.training,
        genesis: SignedPayload::new(
            SchemaEnvelope::new("burn-p2p-model-genesis-v1", spec.protocol_version, genesis),
            placeholder,
        )?,
        contract_signature: SignatureMetadata {
            signer: PeerId::new("unsigned"),
            key_id: REVISION_CONTRACT_SIGNATURE_KEY_ID.into(),
            algorithm: SignatureAlgorithm::Ed25519,
            signed_at: spec.created_at,
            signature_hex: "00".into(),
        },
    };
    sign_revision_contract_bundle(keypair, &mut bundle, Utc::now())?;
    verify_with_keypair(keypair, &bundle)?;
    Ok(bundle)
}

fn verify_genesis_artifact(
    artifact_store_root: &Path,
    expected: &burn_p2p::ArtifactDescriptor,
) -> Result<()> {
    let store = FsArtifactStore::new(artifact_store_root);
    let stored = store
        .load_manifest(&expected.artifact_id)
        .with_context(|| format!("load genesis artifact {}", expected.artifact_id.as_str()))?;
    ensure!(
        &stored == expected,
        "stored genesis artifact descriptor does not match the contract specification"
    );
    let bytes = store
        .materialize_artifact_bytes(expected)
        .context("materialize and hash-check complete genesis artifact")?;
    ensure!(
        bytes.len() as u64 == expected.bytes_len,
        "materialized genesis artifact length does not match its descriptor"
    );
    Ok(())
}

fn verify_with_keypair(keypair: &Keypair, bundle: &RevisionContractBundle) -> Result<()> {
    let peer_id = peer_id_for_keypair(keypair);
    let trusted = BTreeMap::from([(
        peer_id.clone(),
        TrustedIssuer {
            issuer_peer_id: peer_id,
            issuer_public_key_hex: hex::encode(keypair.public().encode_protobuf()),
        },
    )]);
    verify_revision_contract_bundle(&trusted, bundle)?;
    Ok(())
}

fn peer_id_for_keypair(keypair: &Keypair) -> PeerId {
    PeerId::new(libp2p_identity::PeerId::from_public_key(&keypair.public()).to_string())
}

fn load_authority_key(path: &Path) -> Result<Keypair> {
    ensure_secure_key_permissions(path)?;
    let bytes = fs::read(path).with_context(|| format!("read authority key {}", path.display()))?;
    Keypair::from_protobuf_encoding(&bytes)
        .map_err(|error| anyhow::anyhow!("decode authority key {}: {error}", path.display()))
}

#[cfg(unix)]
fn ensure_secure_key_permissions(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let mode = fs::metadata(path)
        .with_context(|| format!("inspect authority key {}", path.display()))?
        .permissions()
        .mode();
    ensure!(
        mode & 0o077 == 0,
        "authority key {} must not be group/world accessible (mode {:o})",
        path.display(),
        mode & 0o777
    );
    Ok(())
}

#[cfg(not(unix))]
fn ensure_secure_key_permissions(path: &Path) -> Result<()> {
    ensure!(path.is_file(), "authority key does not exist");
    Ok(())
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("decode JSON {}", path.display()))
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
    if temp.exists() {
        fs::remove_file(&temp)?;
    }
    let bytes = serde_json::to_vec_pretty(value)?;
    fs::write(&temp, bytes).with_context(|| format!("write {}", temp.display()))?;
    fs::rename(&temp, path).with_context(|| format!("install {}", path.display()))
}
