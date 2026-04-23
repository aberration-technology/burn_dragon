#![cfg(feature = "native")]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::net::TcpListener;
use std::path::Path;
use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};

use burn_dragon_p2p::auth::{
    begin_native_github_login, complete_native_github_login, fetch_edge_snapshot,
};
use burn_dragon_p2p::config::{
    DragonCapabilityPolicy, DragonExistingShardDatasetConfig, DragonManifestSeed,
    DragonNativeAuthBundle, DragonNativePeerConfig, DragonNativeTarget, DragonPeerNetworkConfig,
    DragonShardExportConfig, TokenWindowRecord,
};
use burn_dragon_p2p::native::{
    ManagedRunningNativePeer, prepare_climbmix_native_cpu, prepare_nca_native_cpu,
    spawn_prepared_native_peer,
};
use burn_dragon_p2p::profile::{DragonBrowserProfileTokenSource, DragonExperimentProfile};
use burn_p2p::burn::{BurnShardedDataset, BurnShardedDatasetConfig, BurnWorkload};
use burn_p2p::{
    AuthConfig, AuthProvider, BrowserMode, CallbackPayload, ClientPlatform, ContentId,
    EdgePeerEnrollmentRequest, ExperimentDirectoryEntry, ExperimentDirectoryPolicyExt,
    ExperimentScope, HeadDescriptor, HeadPromotionMode, LeaseId, LoginRequest, MergeStrategy,
    MetricValue, MicroShardId, NodeCertificate, NodeCertificateClaims, PeerId, PeerRole,
    PeerRoleSet, PrincipalClaims, PrincipalId, PrincipalSession, ProjectFamilyId, RevocationEpoch,
    ShardCache, WindowCtx, WindowId, WorkloadInputSource, WorkloadTrainingLease,
};
use burn_p2p_browser::{
    BrowserConformanceHarness, BrowserDirectorySnapshot, BrowserEdgeClient, BrowserEdgeMode,
    BrowserEdgePaths, BrowserEdgeSnapshot, BrowserEnrollmentConfig, BrowserLeaderboardSnapshot,
    BrowserLoginProvider, BrowserReceiptSubmissionResponse, BrowserRuntimeConfig,
    BrowserRuntimeRole, BrowserSessionState, BrowserTrainingBudget, BrowserTrainingPlan,
    BrowserTransportSurface, BrowserUiBindings, BrowserValidationPlan, BrowserWorkerCommand,
    BrowserWorkerEvent, BrowserWorkerIdentity, TrustBundleExport,
    browser_conformance_capability_for_role, browser_conformance_directory,
    browser_conformance_session, browser_conformance_transport,
};
use burn_p2p_core::{SignatureAlgorithm, SignatureMetadata};
use chrono::Utc;
use semver::Version;
use tempfile::tempdir;
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

#[derive(Clone, Copy)]
struct SmokeModelSpec {
    n_layer: usize,
    n_embd: usize,
    n_head: usize,
    latent_total: usize,
    block_size: usize,
    batch_size: usize,
    max_iters: usize,
}

const SMALL_SPEC: SmokeModelSpec = SmokeModelSpec {
    n_layer: 2,
    n_embd: 32,
    n_head: 4,
    latent_total: 64,
    block_size: 64,
    batch_size: 2,
    max_iters: 8,
};

const MEDIUM_SPEC: SmokeModelSpec = SmokeModelSpec {
    n_layer: 4,
    n_embd: 64,
    n_head: 4,
    latent_total: 128,
    block_size: 128,
    batch_size: 4,
    max_iters: 24,
};

const LARGE_SPEC: SmokeModelSpec = SmokeModelSpec {
    n_layer: 6,
    n_embd: 96,
    n_head: 8,
    latent_total: 192,
    block_size: 128,
    batch_size: 4,
    max_iters: 32,
};

fn dummy_auth_bundle() -> DragonNativeAuthBundle {
    DragonNativeAuthBundle {
        auth_config: AuthConfig::new(),
        trust_bundle_endpoint: "https://edge.example/trust-bundle".into(),
        edge_base_url: None,
        session_id: None,
        principal_id: None,
    }
}

fn native_manifest_seed() -> DragonManifestSeed {
    DragonManifestSeed {
        project_family_id: "burn-dragon-language".into(),
        network_id: "dragon-p2p-testnet".into(),
        study_id: "dragon-p2p-study".into(),
        experiment_id: "dragon-p2p-exp".into(),
        revision_id: "r1".into(),
        display_name: "dragon p2p smoke".into(),
        description: "dragon p2p smoke network".into(),
        protocol_major: 0,
        authority_public_keys: Vec::new(),
        bootstrap_addrs: Vec::new(),
        ..DragonManifestSeed::default()
    }
}

fn write(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write config");
}

fn nca_corpus_config_toml(output_dir: &Path) -> String {
    format!(
        r#"
output_dir = "{}"
seed = 1337
name = "dragon-p2p-nca-smoke"
train_samples = 12
validation_samples = 6
chunk_token_capacity = 1024
"#,
        output_dir.display()
    )
}

fn nca_training_config_toml(
    cache_dir: &Path,
    nca_config_path: &Path,
    spec: SmokeModelSpec,
) -> String {
    format!(
        r#"
[dataset]
cache_dir = "{}"
train_split_ratio = 0.9
type = "universality_nca"
config = "{}"

[dataset.tokenizer]
type = "pretokenized"
vocab_size = 50257
eos_id = 50256

[model]
n_layer = {}
n_embd = {}
n_head = {}
latent_total = {}

[model.language_head]
type = "nca_factorized_patch"
state_count = 10
patch_size = 2
frame_special_tokens = true
eos_id = 50256

[training]
block_size = {}
batch_size = {}
max_iters = {}
checkpoint_interval_iters = 4
log_frequency = 1
seed = 1337

[training.continual_backprop]
enabled = true
target = "shared_lowrank_latents"
utility_decay = 0.99
replacement_rate = 0.0001
maturity_steps = 100
sample_interval_steps = 8
replace_interval_steps = 64
utility_epsilon = 0.000001
lr_coupling = "none"
lr_coupling_power = 1.0

[optimizer]
learning_rate = 0.001
weight_decay = 0.0

[generation]
prompt = "0 0 0"
"#,
        cache_dir.display(),
        nca_config_path.display(),
        spec.n_layer,
        spec.n_embd,
        spec.n_head,
        spec.latent_total,
        spec.block_size,
        spec.batch_size,
        spec.max_iters,
    )
}

fn climbmix_training_config_toml(cache_dir: &Path, spec: SmokeModelSpec) -> String {
    format!(
        r#"
[dataset]
cache_dir = "{}"
train_split_ratio = 0.9
type = "nemotron_climb_mix"
max_records = 64

[dataset.tokenizer]
type = "pretokenized"
vocab_size = 50257
eos_id = 50256

[model]
n_layer = {}
n_embd = {}
n_head = {}
latent_total = {}

[training]
block_size = {}
batch_size = {}
max_iters = {}
checkpoint_interval_iters = 4
log_frequency = 1
seed = 1337

[optimizer]
learning_rate = 0.003
weight_decay = 0.0

[generation]
prompt = "1 2 3"
"#,
        cache_dir.display(),
        spec.n_layer,
        spec.n_embd,
        spec.n_head,
        spec.latent_total,
        spec.block_size,
        spec.batch_size,
        spec.max_iters,
    )
}

fn simple_token_window_records(count: usize, block_size: usize) -> Vec<TokenWindowRecord> {
    (0..count)
        .map(|offset| {
            let base = ((offset * 7) % 1024) as i64;
            let inputs = (0..block_size)
                .map(|index| (base + index as i64) % 50256)
                .collect();
            let targets = (1..=block_size)
                .map(|index| (base + index as i64) % 50256)
                .collect();
            TokenWindowRecord {
                inputs,
                targets,
                reset_stream_state: offset % 4 == 0,
            }
        })
        .collect()
}

fn write_existing_climbmix_shards(root: &Path, count: usize, block_size: usize) {
    let records = simple_token_window_records(count, block_size);
    BurnShardedDataset::write_local(
        root,
        &records,
        BurnShardedDatasetConfig::new("dragon-climbmix-smoke")
            .with_microshards(4)
            .with_view_metadata_entry("experiment_kind", "climbmix-pretraining"),
    )
    .expect("write shard dataset");
}

fn metric_float(stats: &std::collections::BTreeMap<String, MetricValue>, key: &str) -> f64 {
    match stats.get(key).expect("metric") {
        MetricValue::Float(value) => *value,
        other => panic!("expected float metric for {key}, got {other:?}"),
    }
}

fn metric_integer(stats: &std::collections::BTreeMap<String, MetricValue>, key: &str) -> i64 {
    match stats.get(key).expect("metric") {
        MetricValue::Integer(value) => *value,
        other => panic!("expected integer metric for {key}, got {other:?}"),
    }
}

fn metric_float_any(stats: &std::collections::BTreeMap<String, MetricValue>, keys: &[&str]) -> f64 {
    for key in keys {
        match stats.get(*key) {
            Some(MetricValue::Float(value)) => return *value,
            Some(MetricValue::Integer(value)) => return *value as f64,
            Some(other) => panic!("expected numeric metric for {key}, got {other:?}"),
            None => continue,
        }
    }
    panic!("missing any metric in {:?}", keys);
}

fn wait_for(timeout: Duration, mut predicate: impl FnMut() -> bool, message: &str) {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if predicate() {
            return;
        }
        thread::sleep(Duration::from_millis(100));
    }
    panic!("{message}");
}

fn is_transient_diffusion_artifact_error(message: &str) -> bool {
    [
        "timed out waiting for artifact-chunk",
        "timed out waiting for artifact-manifest",
        "no connected peer provided chunk",
        "no connected peer provided artifact",
        "Failed to dial the requested peer",
    ]
    .iter()
    .any(|pattern| message.contains(pattern))
}

fn advance_diffusion_with_retry(
    label: &str,
    deadline: Instant,
    mut advance: impl FnMut() -> anyhow::Result<()>,
) {
    loop {
        match advance() {
            Ok(()) => return,
            Err(error)
                if Instant::now() < deadline
                    && is_transient_diffusion_artifact_error(&error.to_string()) =>
            {
                eprintln!("{label}: transient diffusion sync retry: {error}");
                thread::sleep(Duration::from_millis(50));
            }
            Err(error) => panic!("{label}: {error:#}"),
        }
    }
}

fn is_transient_head_sync_error(message: &str) -> bool {
    ["trailing characters", "EOF while parsing", "expected value"]
        .iter()
        .any(|pattern| message.contains(pattern))
}

fn sync_experiment_head_with_retry<B>(
    label: &str,
    peer: &ManagedRunningNativePeer<B>,
    experiment: &burn_p2p::ExperimentHandle,
    deadline: Instant,
) -> Option<HeadDescriptor>
where
    B: burn::tensor::backend::AutodiffBackend + Clone + 'static,
{
    loop {
        match peer.sync_experiment_head(experiment) {
            Ok(head) => return head,
            Err(error)
                if Instant::now() < deadline
                    && is_transient_head_sync_error(&error.to_string()) =>
            {
                eprintln!("{label}: transient head sync retry: {error}");
                thread::sleep(Duration::from_millis(25));
            }
            Err(error) => panic!("{label}: {error:#}"),
        }
    }
}

fn ensure_materialized_pinned_head<B>(
    label: &str,
    peer: &ManagedRunningNativePeer<B>,
    experiment: &burn_p2p::ExperimentHandle,
    head: &HeadDescriptor,
    provider_peer_ids: &[PeerId],
) where
    B: burn::tensor::backend::AutodiffBackend + Clone + 'static,
{
    ensure_materialized_artifact(
        label,
        peer,
        provider_peer_ids,
        &head.artifact_id,
        "pinned head",
        Duration::from_secs(30),
    );
    assert!(
        peer.adopt_known_head_if_present(experiment, head)
            .expect("adopt known pinned head"),
        "{label} should adopt the promoted pinned head locally once its artifact is present",
    );
}

fn ensure_materialized_artifact<B>(
    label: &str,
    peer: &ManagedRunningNativePeer<B>,
    provider_peer_ids: &[PeerId],
    artifact_id: &burn_p2p::ArtifactId,
    artifact_kind: &str,
    timeout: Duration,
) where
    B: burn::tensor::backend::AutodiffBackend + Clone + 'static,
{
    peer.wait_for_artifact_from_peers(provider_peer_ids, artifact_id, timeout)
        .unwrap_or_else(|error| {
            panic!(
                "{label} did not materialize {artifact_kind} artifact {}: {error:#}",
                artifact_id.as_str(),
            )
        });
    let store = peer.artifact_store().expect("artifact store");
    assert!(
        store
            .has_complete_artifact(artifact_id)
            .expect("check materialized artifact"),
        "{label} should have the {artifact_kind} artifact {} locally",
        artifact_id.as_str(),
    );
}

fn select_promoted_head_candidate(
    heads: [&Option<HeadDescriptor>; 3],
    base_head_id: &burn_p2p::HeadId,
    expected_global_step: u64,
) -> Option<HeadDescriptor> {
    heads
        .into_iter()
        .filter_map(|head| head.as_ref())
        .find(|head| {
            head.head_id != *base_head_id
                && head.parent_head_id.as_ref() == Some(base_head_id)
                && head.global_step == expected_global_step
        })
        .cloned()
}

fn peers_have_promoted_head(
    heads: [&Option<HeadDescriptor>; 3],
    promoted_head: &HeadDescriptor,
    base_head_id: &burn_p2p::HeadId,
    expected_global_step: u64,
) -> bool {
    heads.into_iter().all(|head| {
        head.as_ref().is_some_and(|head| {
            head.head_id == promoted_head.head_id
                && head.parent_head_id.as_ref() == Some(base_head_id)
                && head.global_step == expected_global_step
        })
    })
}

fn describe_head_state(head: &Option<HeadDescriptor>) -> String {
    match head {
        Some(head) => format!(
            "head={} parent={} step={}",
            head.head_id.as_str(),
            head.parent_head_id
                .as_ref()
                .map(|value| value.as_str())
                .unwrap_or("none"),
            head.global_step
        ),
        None => "none".into(),
    }
}

fn native_swarm_test_guard() -> std::sync::MutexGuard<'static, ()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn loopback_swarm_address() -> burn_p2p::SwarmAddress {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback listener");
    let port = listener.local_addr().expect("loopback addr").port();
    drop(listener);
    burn_p2p::SwarmAddress::new(format!("/ip4/127.0.0.1/tcp/{port}")).expect("swarm address")
}

fn log_loss_series(label: &str, losses: &[f64]) {
    let first = losses.first().copied().unwrap_or(f64::NAN);
    let final_loss = losses.last().copied().unwrap_or(f64::NAN);
    let best = losses.iter().copied().fold(f64::INFINITY, f64::min);
    eprintln!("{label}: losses={losses:?} first={first:.4} best={best:.4} final={final_loss:.4}");
}

fn shard_manifest_url(base_url: &str) -> String {
    format!("{}/fetch-manifest.json", base_url.trim_end_matches('/'))
}

#[derive(Clone)]
struct NativeWindowObservation {
    head: HeadDescriptor,
    loss: f64,
}

fn local_browser_training_and_verification_pair(
    entry: &burn_p2p::ExperimentDirectoryEntry,
    release_train_hash: burn_p2p::ContentId,
    target_artifact_hash: burn_p2p::ContentId,
    network_id: burn_p2p::NetworkId,
) -> (BrowserConformanceHarness, BrowserConformanceHarness) {
    let trainer_scopes = entry.allowed_scopes.clone();
    assert!(trainer_scopes.contains(&ExperimentScope::Train {
        experiment_id: entry.experiment_id.clone(),
    }));
    assert!(!trainer_scopes.contains(&ExperimentScope::Validate {
        experiment_id: entry.experiment_id.clone(),
    }));

    let trainer_session = browser_conformance_session(
        network_id.clone(),
        PrincipalId::new("browser-trainer-principal"),
        trainer_scopes,
    );
    let verifier_session = browser_conformance_session(
        network_id.clone(),
        PrincipalId::new("browser-local-verifier-principal"),
        local_mock_verifier_scopes(entry),
    );
    let trainer = BrowserConformanceHarness::start(
        BrowserRuntimeConfig {
            role: BrowserRuntimeRole::BrowserTrainerWgpu,
            site_seed_node_urls: vec!["/dns4/edge.example/tcp/443/wss".into()],
            ..BrowserRuntimeConfig::new(
                "https://edge.example",
                network_id.clone(),
                release_train_hash.clone(),
                "browser-wasm",
                target_artifact_hash.clone(),
            )
        },
        browser_conformance_capability_for_role(BrowserRuntimeRole::BrowserTrainerWgpu),
        browser_conformance_transport(),
        browser_conformance_directory(network_id.clone(), vec![entry.clone()]),
        trainer_session,
    );
    let verifier = BrowserConformanceHarness::start(
        BrowserRuntimeConfig {
            role: BrowserRuntimeRole::BrowserVerifier,
            site_seed_node_urls: vec!["/dns4/edge.example/tcp/443/wss".into()],
            ..BrowserRuntimeConfig::new(
                "https://edge.example",
                network_id.clone(),
                release_train_hash,
                "browser-wasm",
                target_artifact_hash,
            )
        },
        browser_conformance_capability_for_role(BrowserRuntimeRole::BrowserVerifier),
        browser_conformance_transport(),
        browser_conformance_directory(network_id, vec![entry.clone()]),
        verifier_session,
    );
    (trainer, verifier)
}

fn flush_and_ack_receipts(harness: &mut BrowserConformanceHarness) -> usize {
    let flush_events =
        harness
            .runtime
            .apply_command(BrowserWorkerCommand::FlushReceiptOutbox, None, None);
    let receipt_ids = flush_events
        .iter()
        .find_map(|event| match event {
            BrowserWorkerEvent::ReceiptOutboxReady { receipts, .. } => Some(
                receipts
                    .iter()
                    .map(|receipt| receipt.receipt_id.clone())
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        })
        .unwrap_or_default();
    if receipt_ids.is_empty() {
        assert!(
            harness.pending_receipts().is_empty(),
            "receipt flush emitted no receipts but the outbox is still non-empty"
        );
        return 0;
    }
    let ack_events = harness.runtime.apply_command(
        BrowserWorkerCommand::AcknowledgeSubmittedReceipts {
            receipt_ids: receipt_ids.clone(),
        },
        None,
        None,
    );
    assert!(ack_events.iter().any(|event| matches!(
        event,
        BrowserWorkerEvent::ReceiptsAcknowledged {
            receipt_ids: acknowledged,
            pending_receipts: 0,
        } if *acknowledged == receipt_ids
    )));
    assert!(
        harness.pending_receipts().is_empty(),
        "browser receipt outbox should be empty after acknowledgement"
    );
    receipt_ids.len()
}

fn run_training_windows_with_heads<B>(
    prepared: &burn_dragon_p2p::experiments::common::PreparedNativePeer<B>,
    windows: usize,
    head_prefix: &str,
) -> Vec<NativeWindowObservation>
where
    B: burn::tensor::backend::AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let project = &prepared.project;
    let device = project.runtime_device();
    let registration = project.dataset_registration().expect("registration");
    let microshard_plan = project
        .microshard_plan(&registration)
        .expect("microshard plan");
    let cache_root = tempdir().expect("cache root");
    let shard_cache = ShardCache::new(cache_root.path());
    let entry = &prepared.manifests.experiment_directory[0];

    let mut model = project.init_model(&device);
    let mut observations = Vec::new();
    let mut global_step = 0u64;
    let mut parent_head_id = None;

    for window_ordinal in 0..windows {
        let lease = burn_p2p::LeasePlanner::default()
            .plan_lease(
                prepared.manifests.network_manifest.network_id.clone(),
                entry.study_id.clone(),
                entry.experiment_id.clone(),
                entry.current_revision_id.clone(),
                &microshard_plan.dataset_view,
                PeerId::new(format!("{head_prefix}-peer-{}", window_ordinal + 1)),
                WindowId((window_ordinal + 1) as u64),
                Utc::now(),
                1,
                &microshard_plan.microshards,
            )
            .expect("lease")
            .lease;
        let cached = shard_cache
            .fetch_lease_microshards(&registration, &microshard_plan, &lease)
            .expect("cached microshards");
        let batches = project.load_batches(&lease, &cached).expect("load batches");
        let mut ctx = WindowCtx {
            device: device.clone(),
            model,
            lease,
            cached_microshards: cached,
            batches,
        };
        let report = project.train_window(&mut ctx).expect("train window");
        let train_steps = metric_integer(&report.stats, "train_steps");
        assert!(train_steps > 0);
        let loss = metric_float(&report.stats, "train_loss");
        assert!(loss.is_finite(), "train loss must be finite");
        global_step += train_steps as u64;
        let head = HeadDescriptor {
            head_id: burn_p2p::HeadId::new(format!("{head_prefix}-head-{}", window_ordinal + 1)),
            study_id: entry.study_id.clone(),
            experiment_id: entry.experiment_id.clone(),
            revision_id: entry.current_revision_id.clone(),
            artifact_id: burn_p2p::ArtifactId::new(format!(
                "{head_prefix}-artifact-{}",
                window_ordinal + 1
            )),
            parent_head_id: parent_head_id.clone(),
            global_step,
            created_at: Utc::now(),
            metrics: report.stats.clone(),
        };
        parent_head_id = Some(head.head_id.clone());
        observations.push(NativeWindowObservation { head, loss });
        model = ctx.model;
    }

    observations
}

fn run_training_windows<B>(
    prepared: &burn_dragon_p2p::experiments::common::PreparedNativePeer<B>,
    windows: usize,
) -> Vec<f64>
where
    B: burn::tensor::backend::AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let project = &prepared.project;
    let device = project.runtime_device();
    let registration = project.dataset_registration().expect("registration");
    let microshard_plan = project
        .microshard_plan(&registration)
        .expect("microshard plan");
    let cache_root = tempdir().expect("cache root");
    let shard_cache = ShardCache::new(cache_root.path());
    let entry = &prepared.manifests.experiment_directory[0];

    let mut model = project.init_model(&device);
    let mut losses = Vec::new();
    for window_ordinal in 0..windows {
        let lease = burn_p2p::LeasePlanner::default()
            .plan_lease(
                prepared.manifests.network_manifest.network_id.clone(),
                entry.study_id.clone(),
                entry.experiment_id.clone(),
                entry.current_revision_id.clone(),
                &microshard_plan.dataset_view,
                PeerId::new(format!("peer-{}", window_ordinal + 1)),
                WindowId((window_ordinal + 1) as u64),
                Utc::now(),
                1,
                &microshard_plan.microshards,
            )
            .expect("lease")
            .lease;
        let cached = shard_cache
            .fetch_lease_microshards(&registration, &microshard_plan, &lease)
            .expect("cached microshards");
        let batches = project.load_batches(&lease, &cached).expect("load batches");
        let mut ctx = WindowCtx {
            device: device.clone(),
            model,
            lease,
            cached_microshards: cached,
            batches,
        };
        let report = project.train_window(&mut ctx).expect("train window");
        assert!(metric_integer(&report.stats, "train_steps") > 0);
        let loss = metric_float(&report.stats, "train_loss");
        assert!(loss.is_finite(), "train loss must be finite");
        losses.push(loss);
        model = ctx.model;
    }
    losses
}

#[derive(Default)]
struct MockEdgeState {
    authorized_directory_fetches: usize,
    unauthorized_directory_fetches: usize,
    receipt_submission_batches: usize,
    submitted_receipt_ids: Vec<String>,
    enrolled_peer_ids: BTreeSet<String>,
    sessions: BTreeMap<String, PrincipalSession>,
    pending_logins: BTreeMap<String, PendingLogin>,
}

#[derive(Clone)]
struct PendingLogin {
    requested_scopes: BTreeSet<ExperimentScope>,
    state: String,
}

struct LocalEdgeMock {
    base_url: String,
    state: Arc<Mutex<MockEdgeState>>,
    stop: Arc<AtomicBool>,
    join: Option<thread::JoinHandle<()>>,
}

impl Drop for LocalEdgeMock {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(join) = self.join.take() {
            join.join().expect("edge server thread");
        }
    }
}

fn local_mock_verifier_scopes(entry: &ExperimentDirectoryEntry) -> BTreeSet<ExperimentScope> {
    let mut scopes = entry.allowed_scopes.clone();
    scopes.insert(ExperimentScope::Validate {
        experiment_id: entry.experiment_id.clone(),
    });
    scopes
}

fn edge_snapshot_for_manifests(
    manifests: &burn_dragon_p2p::config::DragonManifestBundle,
    browser_mode: BrowserMode,
) -> BrowserEdgeSnapshot {
    let paths = BrowserEdgePaths {
        login_path: "/login/github".into(),
        callback_path: "/callback/github".into(),
        ..BrowserEdgePaths::default()
    };

    BrowserEdgeSnapshot {
        network_id: manifests.network_manifest.network_id.clone(),
        edge_mode: BrowserEdgeMode::Peer,
        browser_mode,
        social_mode: burn_p2p::SocialMode::Disabled,
        profile_mode: burn_p2p::ProfileMode::Disabled,
        transports: BrowserTransportSurface {
            webrtc_direct: false,
            webtransport_gateway: true,
            wss_fallback: true,
        },
        paths,
        auth_enabled: true,
        login_providers: vec![BrowserLoginProvider {
            label: "GitHub".into(),
            login_path: "/login/github".into(),
            callback_path: Some("/callback/github".into()),
            device_path: None,
        }],
        required_release_train_hash: Some(manifests.release_manifest.release_train_hash.clone()),
        allowed_target_artifact_hashes: BTreeSet::from([manifests
            .release_manifest
            .target_artifact_hash
            .clone()]),
        directory: BrowserDirectorySnapshot {
            network_id: manifests.network_manifest.network_id.clone(),
            generated_at: Utc::now(),
            entries: manifests.experiment_directory.clone(),
        },
        heads: Vec::new(),
        leaderboard: BrowserLeaderboardSnapshot {
            network_id: manifests.network_manifest.network_id.clone(),
            score_version: "leaderboard_score_v1".into(),
            entries: Vec::new(),
            captured_at: Utc::now(),
        },
        trust_bundle: Some(TrustBundleExport {
            network_id: manifests.network_manifest.network_id.clone(),
            project_family_id: ProjectFamilyId::new(
                manifests.release_manifest.project_family_id.as_str(),
            ),
            required_release_train_hash: manifests.release_manifest.release_train_hash.clone(),
            allowed_target_artifact_hashes: BTreeSet::from([manifests
                .release_manifest
                .target_artifact_hash
                .clone()]),
            minimum_revocation_epoch: RevocationEpoch(0),
            active_issuer_peer_id: PeerId::new("dragon-edge-issuer"),
            issuers: Vec::new(),
            reenrollment: None,
        }),
        captured_at: Utc::now(),
    }
}

fn current_edge_head(entry: &ExperimentDirectoryEntry, label: &str) -> HeadDescriptor {
    HeadDescriptor {
        head_id: burn_p2p::HeadId::new(format!("{label}-edge-head")),
        study_id: entry.study_id.clone(),
        experiment_id: entry.experiment_id.clone(),
        revision_id: entry.current_revision_id.clone(),
        artifact_id: burn_p2p::ArtifactId::new(format!("{label}-edge-artifact")),
        parent_head_id: None,
        global_step: 1,
        created_at: Utc::now(),
        metrics: Default::default(),
    }
}

fn browser_worker_identity(label: &str) -> BrowserWorkerIdentity {
    BrowserWorkerIdentity {
        peer_id: PeerId::new(format!("{label}-browser-peer")),
        peer_public_key_hex: "deadbeef".into(),
        serial: 1,
        client_policy_hash: None,
    }
}

fn json_header() -> Header {
    Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).expect("json header")
}

fn respond_json<T: serde::Serialize>(request: Request, status: u16, value: &T) {
    let payload = serde_json::to_string(value).expect("serialize json response");
    request
        .respond(
            Response::from_string(payload)
                .with_status_code(StatusCode(status))
                .with_header(json_header()),
        )
        .expect("respond json");
}

fn respond_text(request: Request, status: u16, body: &str) {
    request
        .respond(
            Response::from_string(body.to_owned())
                .with_status_code(StatusCode(status))
                .with_header(json_header()),
        )
        .expect("respond text");
}

fn read_json<T: serde::de::DeserializeOwned>(request: &mut Request) -> T {
    let mut body = String::new();
    std::io::Read::read_to_string(request.as_reader(), &mut body).expect("request body");
    serde_json::from_str(&body).expect("decode request json")
}

fn header_value(request: &Request, name: &'static str) -> Option<String> {
    request
        .headers()
        .iter()
        .find(|header| header.field.equiv(name))
        .map(|header| header.value.as_str().to_owned())
}

fn principal_from_provider_code(provider_code: Option<String>) -> PrincipalId {
    let suffix = provider_code
        .as_deref()
        .filter(|value| !value.is_empty())
        .unwrap_or("github-user");
    PrincipalId::new(format!("github-{suffix}"))
}

fn node_certificate_for_session(
    snapshot: &BrowserEdgeSnapshot,
    session: &PrincipalSession,
    enrollment: &EdgePeerEnrollmentRequest,
) -> NodeCertificate {
    NodeCertificate::new(
        Version::new(0, 1, 0),
        NodeCertificateClaims {
            network_id: snapshot.network_id.clone(),
            project_family_id: snapshot
                .trust_bundle
                .as_ref()
                .expect("trust bundle")
                .project_family_id
                .clone(),
            release_train_hash: enrollment.release_train_hash.clone(),
            target_artifact_hash: enrollment.target_artifact_hash.clone(),
            peer_id: enrollment.peer_id.clone(),
            peer_public_key_hex: enrollment.peer_public_key_hex.clone(),
            principal_id: session.claims.principal_id.clone(),
            provider: session.claims.provider.clone(),
            granted_roles: PeerRoleSet::new([
                PeerRole::TrainerCpu,
                PeerRole::BrowserTrainerWgpu,
                PeerRole::BrowserVerifier,
                PeerRole::Viewer,
            ]),
            experiment_scopes: enrollment.requested_scopes.clone(),
            client_policy_hash: enrollment.client_policy_hash.clone(),
            auth_policy_snapshot: None,
            not_before: Utc::now(),
            not_after: Utc::now() + chrono::Duration::minutes(30),
            serial: enrollment.serial,
            revocation_epoch: RevocationEpoch(0),
        },
        SignatureMetadata {
            signer: PeerId::new("dragon-edge-issuer"),
            key_id: "dragon-edge-key".into(),
            algorithm: SignatureAlgorithm::Ed25519,
            signed_at: Utc::now(),
            signature_hex: "00".into(),
        },
    )
    .expect("node certificate")
}

fn spawn_local_edge(snapshot: BrowserEdgeSnapshot) -> LocalEdgeMock {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind edge");
    let addr = listener.local_addr().expect("edge local addr");
    let server = Server::from_listener(listener, None).expect("tiny_http server");
    let state = Arc::new(Mutex::new(MockEdgeState::default()));
    let stop = Arc::new(AtomicBool::new(false));
    let state_for_thread = Arc::clone(&state);
    let stop_for_thread = Arc::clone(&stop);
    let snapshot_for_thread = snapshot.clone();

    let join = thread::spawn(move || {
        while !stop_for_thread.load(Ordering::SeqCst) {
            let Some(mut request) = server
                .recv_timeout(Duration::from_millis(100))
                .expect("receive request")
            else {
                continue;
            };

            match (request.method(), request.url()) {
                (&Method::Get, "/portal/snapshot") => {
                    respond_json(request, 200, &snapshot_for_thread);
                }
                (&Method::Get, "/trust") => {
                    respond_json(
                        request,
                        200,
                        snapshot_for_thread
                            .trust_bundle
                            .as_ref()
                            .expect("trust bundle"),
                    );
                }
                (&Method::Get, "/directory") => {
                    let Some(session_id) = header_value(&request, "x-session-id") else {
                        state_for_thread
                            .lock()
                            .expect("state")
                            .unauthorized_directory_fetches += 1;
                        respond_text(request, 401, r#"{"error":"missing session"}"#);
                        continue;
                    };
                    let authorized = state_for_thread
                        .lock()
                        .expect("state")
                        .sessions
                        .contains_key(session_id.as_str());
                    if !authorized {
                        state_for_thread
                            .lock()
                            .expect("state")
                            .unauthorized_directory_fetches += 1;
                        respond_text(request, 401, r#"{"error":"unknown session"}"#);
                        continue;
                    }
                    state_for_thread
                        .lock()
                        .expect("state")
                        .authorized_directory_fetches += 1;
                    respond_json(request, 200, &snapshot_for_thread.directory.entries);
                }
                (&Method::Post, "/login/github") => {
                    let login: LoginRequest = read_json(&mut request);
                    let ordinal = state_for_thread.lock().expect("state").pending_logins.len() + 1;
                    let login_id = ContentId::new(format!("mock-login-{ordinal}"));
                    let state_token = format!("mock-state-{ordinal}");
                    state_for_thread
                        .lock()
                        .expect("state")
                        .pending_logins
                        .insert(
                            login_id.as_str().to_owned(),
                            PendingLogin {
                                requested_scopes: login.requested_scopes,
                                state: state_token.clone(),
                            },
                        );
                    respond_json(
                        request,
                        200,
                        &burn_p2p::LoginStart {
                            login_id,
                            provider: AuthProvider::GitHub,
                            state: state_token,
                            authorize_url: Some("https://github.example/authorize".into()),
                            expires_at: Utc::now() + chrono::Duration::minutes(5),
                        },
                    );
                }
                (&Method::Post, "/callback/github") => {
                    let callback: CallbackPayload = read_json(&mut request);
                    let pending = state_for_thread
                        .lock()
                        .expect("state")
                        .pending_logins
                        .get(callback.login_id.as_str())
                        .cloned()
                        .expect("pending login");
                    assert_eq!(
                        callback.state, pending.state,
                        "callback state must match login"
                    );
                    let principal_id = principal_from_provider_code(callback.provider_code.clone());
                    let session = PrincipalSession {
                        session_id: ContentId::new(format!(
                            "mock-session-{}",
                            callback.login_id.as_str()
                        )),
                        network_id: snapshot_for_thread.network_id.clone(),
                        claims: PrincipalClaims {
                            principal_id,
                            provider: AuthProvider::GitHub,
                            display_name: "dragon github principal".into(),
                            org_memberships: BTreeSet::from(["dragon".into()]),
                            group_memberships: BTreeSet::from(["trainers".into()]),
                            granted_roles: PeerRoleSet::new([
                                PeerRole::TrainerCpu,
                                PeerRole::BrowserTrainerWgpu,
                                PeerRole::BrowserVerifier,
                            ]),
                            granted_scopes: pending.requested_scopes,
                            custom_claims: BTreeMap::new(),
                            issued_at: Utc::now(),
                            expires_at: Utc::now() + chrono::Duration::minutes(30),
                        },
                        issued_at: Utc::now(),
                        expires_at: Utc::now() + chrono::Duration::minutes(30),
                    };
                    state_for_thread
                        .lock()
                        .expect("state")
                        .sessions
                        .insert(session.session_id.as_str().to_owned(), session.clone());
                    respond_json(request, 200, &session);
                }
                (&Method::Post, "/enroll") => {
                    let enrollment: EdgePeerEnrollmentRequest = read_json(&mut request);
                    let session = state_for_thread
                        .lock()
                        .expect("state")
                        .sessions
                        .get(enrollment.session_id.as_str())
                        .cloned()
                        .expect("session for enrollment");
                    let certificate =
                        node_certificate_for_session(&snapshot_for_thread, &session, &enrollment);
                    state_for_thread
                        .lock()
                        .expect("state")
                        .enrolled_peer_ids
                        .insert(enrollment.peer_id.as_str().to_owned());
                    respond_json(request, 200, &certificate);
                }
                (&Method::Post, "/receipts/browser") => {
                    let Some(session_id) = header_value(&request, "x-session-id") else {
                        respond_text(request, 401, r#"{"error":"missing session"}"#);
                        continue;
                    };
                    let mut state = state_for_thread.lock().expect("state");
                    if !state.sessions.contains_key(session_id.as_str()) {
                        respond_text(request, 401, r#"{"error":"unknown session"}"#);
                        continue;
                    }
                    let receipts: Vec<burn_p2p::ContributionReceipt> = read_json(&mut request);
                    state.receipt_submission_batches += 1;
                    state.submitted_receipt_ids.extend(
                        receipts
                            .iter()
                            .map(|receipt| receipt.receipt_id.as_str().to_owned()),
                    );
                    let response = BrowserReceiptSubmissionResponse {
                        accepted_receipt_ids: receipts
                            .iter()
                            .map(|receipt| receipt.receipt_id.clone())
                            .collect(),
                        pending_receipt_count: 0,
                    };
                    drop(state);
                    respond_json(request, 200, &response);
                }
                _ => {
                    respond_text(request, 404, r#"{"error":"not found"}"#);
                }
            }
        }
    });

    LocalEdgeMock {
        base_url: format!("http://{addr}"),
        state,
        stop,
        join: Some(join),
    }
}

fn acknowledge_browser_receipts(
    harness: &mut BrowserConformanceHarness,
    receipt_ids: Vec<burn_p2p::ContributionReceiptId>,
) {
    let ack_events = harness.runtime.apply_command(
        BrowserWorkerCommand::AcknowledgeSubmittedReceipts {
            receipt_ids: receipt_ids.clone(),
        },
        None,
        None,
    );
    assert!(ack_events.iter().any(|event| matches!(
        event,
        BrowserWorkerEvent::ReceiptsAcknowledged {
            receipt_ids: acknowledged,
            pending_receipts: 0,
        } if *acknowledged == receipt_ids
    )));
    assert!(
        harness.pending_receipts().is_empty(),
        "browser receipt outbox should be empty after edge acknowledgement"
    );
}

fn browser_runtime_for_edge(
    edge_base_url: &str,
    network_id: burn_p2p::NetworkId,
    release_train_hash: ContentId,
    target_artifact_hash: ContentId,
    role: BrowserRuntimeRole,
) -> BrowserRuntimeConfig {
    BrowserRuntimeConfig {
        role,
        site_seed_node_urls: vec!["/dns4/edge.example/tcp/443/wss".into()],
        ..BrowserRuntimeConfig::new(
            edge_base_url,
            network_id,
            release_train_hash,
            "browser-wasm",
            target_artifact_hash,
        )
    }
}

fn run_edge_drill_for_prepared<B>(
    prepared: &burn_dragon_p2p::experiments::common::PreparedNativePeer<B>,
    label: &str,
) where
    B: burn::tensor::backend::AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let entry = prepared.manifests.experiment_directory[0].clone();
    let snapshot = edge_snapshot_for_manifests(&prepared.manifests, BrowserMode::Trainer);
    let edge = spawn_local_edge(snapshot.clone());
    let trainer_requested_scopes = entry.allowed_scopes.clone();
    let local_verifier_requested_scopes = local_mock_verifier_scopes(&entry);
    let native_storage = tempdir().expect("native auth storage");

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    runtime.block_on(async {
        let fetched_snapshot = fetch_edge_snapshot(&edge.base_url)
            .await
            .expect("native fetch edge snapshot");
        assert_eq!(fetched_snapshot.network_id, snapshot.network_id);

        let pending = begin_native_github_login(
            &edge.base_url,
            &prepared.manifests.release_manifest,
            trainer_requested_scopes.clone(),
            1800,
            Some(format!("{label}-native")),
            false,
        )
        .await
        .expect("begin native github login");
        assert!(matches!(pending.login.provider, AuthProvider::GitHub));

        let native = complete_native_github_login(
            native_storage.path(),
            &pending,
            "native-provider-code",
            None,
        )
        .await
        .expect("complete native github login");
        assert!(matches!(
            native.session.claims.provider,
            AuthProvider::GitHub
        ));
        assert!(native.auth.auth_config.local_peer_auth.is_some());
        assert_eq!(
            native.auth.trust_bundle_endpoint,
            format!("{}/trust", edge.base_url)
        );

        let browser_boot_client = BrowserEdgeClient::new(
            BrowserUiBindings::new(&edge.base_url),
            BrowserEnrollmentConfig::for_runtime_sync(&snapshot),
        );
        let browser_snapshot = browser_boot_client
            .fetch_browser_edge_snapshot()
            .await
            .expect("browser fetch edge snapshot");
        assert_eq!(browser_snapshot.network_id, snapshot.network_id);

        let browser_client = BrowserEdgeClient::new(
            BrowserUiBindings::from_edge_snapshot(&edge.base_url, &browser_snapshot),
            BrowserEnrollmentConfig::from_edge_snapshot(
                &browser_snapshot,
                "browser-wasm",
                prepared
                    .manifests
                    .release_manifest
                    .target_artifact_hash
                    .clone(),
                trainer_requested_scopes,
                1800,
            )
            .expect("browser enrollment config"),
        );
        let browser_login = browser_client
            .begin_login(Some(format!("{label}-browser")))
            .await
            .expect("begin browser github login");
        assert!(matches!(browser_login.provider, AuthProvider::GitHub));

        let browser_session = browser_client
            .complete_provider_login(&browser_login, "browser-provider-code")
            .await
            .expect("complete browser github login");
        assert!(matches!(
            browser_session.claims.provider,
            AuthProvider::GitHub
        ));
        assert!(
            !browser_session
                .claims
                .granted_scopes
                .contains(&ExperimentScope::Validate {
                    experiment_id: entry.experiment_id.clone(),
                })
        );

        let trainer_worker_identity = browser_worker_identity(&format!("{label}-trainer"));
        let browser_certificate = browser_client
            .enroll(
                &browser_client
                    .build_enrollment_request(&browser_session, &trainer_worker_identity),
            )
            .await
            .expect("browser enroll");
        let trust_bundle = browser_client
            .fetch_trust_bundle()
            .await
            .expect("browser trust bundle");

        let verifier_client = BrowserEdgeClient::new(
            BrowserUiBindings::from_edge_snapshot(&edge.base_url, &browser_snapshot),
            BrowserEnrollmentConfig::from_edge_snapshot(
                &browser_snapshot,
                "browser-wasm-verifier",
                prepared
                    .manifests
                    .release_manifest
                    .target_artifact_hash
                    .clone(),
                local_verifier_requested_scopes,
                1800,
            )
            .expect("browser verifier enrollment config"),
        );
        let verifier_login = verifier_client
            .begin_login(Some(format!("{label}-browser-verifier")))
            .await
            .expect("begin browser verifier github login");
        let verifier_session = verifier_client
            .complete_provider_login(&verifier_login, "browser-verifier-provider-code")
            .await
            .expect("complete browser verifier github login");
        assert!(
            verifier_session
                .claims
                .granted_scopes
                .contains(&ExperimentScope::Validate {
                    experiment_id: entry.experiment_id.clone(),
                })
        );
        let verifier_worker_identity = browser_worker_identity(&format!("{label}-verifier"));
        let verifier_certificate = verifier_client
            .enroll(
                &verifier_client
                    .build_enrollment_request(&verifier_session, &verifier_worker_identity),
            )
            .await
            .expect("browser verifier enroll");
        let verifier_trust_bundle = verifier_client
            .fetch_trust_bundle()
            .await
            .expect("browser verifier trust bundle");

        assert!(
            browser_client.fetch_directory(None).await.is_err(),
            "directory fetch without session should be rejected"
        );
        let directory = browser_client
            .fetch_directory(Some(&browser_session.session_id))
            .await
            .expect("authorized directory fetch");
        assert_eq!(directory[0].experiment_id, entry.experiment_id);

        let head = current_edge_head(&entry, label);
        let browser_session_state = BrowserSessionState {
            session: Some(browser_session.clone()),
            certificate: Some(browser_certificate),
            trust_bundle: Some(trust_bundle),
            enrolled_at: Some(Utc::now()),
            reenrollment_required: false,
        };

        let mut trainer = BrowserConformanceHarness::start(
            browser_runtime_for_edge(
                &edge.base_url,
                prepared.manifests.network_manifest.network_id.clone(),
                prepared
                    .manifests
                    .release_manifest
                    .release_train_hash
                    .clone(),
                prepared
                    .manifests
                    .release_manifest
                    .target_artifact_hash
                    .clone(),
                BrowserRuntimeRole::BrowserTrainerWgpu,
            ),
            browser_conformance_capability_for_role(BrowserRuntimeRole::BrowserTrainerWgpu),
            browser_conformance_transport(),
            browser_conformance_directory(
                prepared.manifests.network_manifest.network_id.clone(),
                vec![entry.clone()],
            ),
            browser_session_state.clone(),
        );
        trainer.select_experiment(
            entry.experiment_id.clone(),
            Some(entry.current_revision_id.clone()),
        );
        trainer.apply_heads(std::slice::from_ref(&head));
        let training = trainer
            .run_training(BrowserTrainingPlan {
                study_id: entry.study_id.clone(),
                experiment_id: entry.experiment_id.clone(),
                revision_id: entry.current_revision_id.clone(),
                workload_id: entry.workload_id.clone(),
                budget: BrowserTrainingBudget::default(),
                lease: None,
            })
            .expect("browser training against edge-backed session");
        assert!(training.receipt_id.is_some());
        let pending_training_receipts = trainer.pending_receipts();
        assert!(
            !pending_training_receipts.is_empty(),
            "browser training should enqueue at least one receipt"
        );
        let training_submission = browser_client
            .submit_receipts(&browser_session.session_id, &pending_training_receipts)
            .await
            .expect("submit browser training receipts");
        assert_eq!(
            training_submission.accepted_receipt_ids.len(),
            pending_training_receipts.len()
        );
        acknowledge_browser_receipts(&mut trainer, training_submission.accepted_receipt_ids);

        let verifier_session_state = BrowserSessionState {
            session: Some(verifier_session.clone()),
            certificate: Some(verifier_certificate),
            trust_bundle: Some(verifier_trust_bundle),
            enrolled_at: Some(Utc::now()),
            reenrollment_required: false,
        };

        let mut verifier = BrowserConformanceHarness::start(
            browser_runtime_for_edge(
                &edge.base_url,
                prepared.manifests.network_manifest.network_id.clone(),
                prepared
                    .manifests
                    .release_manifest
                    .release_train_hash
                    .clone(),
                prepared
                    .manifests
                    .release_manifest
                    .target_artifact_hash
                    .clone(),
                BrowserRuntimeRole::BrowserVerifier,
            ),
            browser_conformance_capability_for_role(BrowserRuntimeRole::BrowserVerifier),
            browser_conformance_transport(),
            browser_conformance_directory(
                prepared.manifests.network_manifest.network_id.clone(),
                vec![entry.clone()],
            ),
            verifier_session_state,
        );
        verifier.select_experiment(
            entry.experiment_id.clone(),
            Some(entry.current_revision_id.clone()),
        );
        verifier.apply_heads(std::slice::from_ref(&head));
        let validation = verifier
            .run_validation(BrowserValidationPlan {
                head_id: head.head_id.clone(),
                max_checkpoint_bytes: 8 * 1024 * 1024,
                sample_budget: 4,
                emit_receipt: true,
            })
            .expect("browser validation against edge-backed session");
        assert!(validation.accepted);
        assert!(validation.emitted_receipt_id.is_some());
        let pending_validation_receipts = verifier.pending_receipts();
        assert!(
            !pending_validation_receipts.is_empty(),
            "browser validation should enqueue at least one receipt"
        );
        let validation_submission = browser_client
            .submit_receipts(&verifier_session.session_id, &pending_validation_receipts)
            .await
            .expect("submit browser validation receipts");
        assert_eq!(
            validation_submission.accepted_receipt_ids.len(),
            pending_validation_receipts.len()
        );
        acknowledge_browser_receipts(&mut verifier, validation_submission.accepted_receipt_ids);
    });

    let state = edge.state.lock().expect("edge state");
    assert_eq!(
        state.enrolled_peer_ids.len(),
        3,
        "native plus two distinct browser peers should enroll against the same edge"
    );
    assert!(
        state
            .enrolled_peer_ids
            .contains(&format!("{label}-trainer-browser-peer")),
        "trainer browser peer should be enrolled"
    );
    assert!(
        state
            .enrolled_peer_ids
            .contains(&format!("{label}-verifier-browser-peer")),
        "verifier browser peer should be enrolled"
    );
    assert_eq!(
        state.authorized_directory_fetches, 1,
        "browser directory fetch should succeed once with a session"
    );
    assert_eq!(
        state.unauthorized_directory_fetches, 1,
        "browser directory fetch without a session should be rejected"
    );
    assert!(
        state.receipt_submission_batches >= 2,
        "browser training and validation receipts should both submit to the edge"
    );
    assert!(
        state.submitted_receipt_ids.len() >= 2,
        "edge should record submitted browser receipts"
    );
}

#[test]
fn ci_native_smoke_suite() {
    nca_native_peer_exports_shards_and_executes_training_windows();
    nca_native_runtime_persists_and_publishes_artifacts();
    nca_bootstrap_only_topology_supports_trainer_only_diffusion_roles();
    browser_conformance_uses_native_dragon_manifests();
}

#[test]
fn nca_native_peer_exports_shards_and_executes_training_windows() {
    let _guard = native_swarm_test_guard();
    let root = tempdir().expect("root");
    let nca_config_path = root.path().join("nca.toml");
    let training_config_path = root.path().join("nca-train.toml");
    let shard_root = root.path().join("nca-shards");
    write(&nca_config_path, &nca_corpus_config_toml(root.path()));
    write(
        &training_config_path,
        &nca_training_config_toml(&root.path().join("nca-cache"), &nca_config_path, SMALL_SPEC),
    );

    let native = DragonNativePeerConfig {
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-native"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0-pre.22").expect("valid burn_dragon version"),
        git_commit: Some("smoke".into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export: Some(DragonShardExportConfig {
            root: shard_root.clone(),
            dataset_name: Some("dragon-nca-smoke".into()),
            microshards: Some(4),
            max_records: Some(32),
            http_upstream: None,
        }),
        existing_shard_dataset: None,
    };

    let prepared = prepare_nca_native_cpu(&native, Some(&dummy_auth_bundle())).expect("peer");
    assert_eq!(
        prepared.project.data_pipeline_kind(),
        burn_p2p::LeaseDataPipelineKind::ShardedStatic
    );
    match prepared
        .project
        .data_pipeline_descriptor()
        .input_source
        .as_ref()
    {
        Some(WorkloadInputSource::Generated { descriptor }) => {
            assert_eq!(descriptor.provider, "burn_dragon_universality_nca");
            assert_eq!(
                descriptor
                    .metadata
                    .get("experiment_kind")
                    .map(String::as_str),
                Some("nca-prepretraining")
            );
            assert_eq!(
                descriptor.metadata.get("config_path").map(String::as_str),
                Some(nca_config_path.to_string_lossy().as_ref())
            );
        }
        other => panic!("expected generated input source, got {other:?}"),
    }
    assert!(shard_root.join("fetch-manifest.json").is_file());
    assert!(shard_root.join("burn-sharded-dataset.json").is_file());

    let losses = run_training_windows(&prepared, 3);
    log_loss_series("nca_native_smoke", &losses);
    assert!(losses.last().copied().unwrap_or(f64::INFINITY) <= losses[0] + 0.5);
}

#[test]
fn nca_native_runtime_persists_and_publishes_artifacts() {
    let _guard = native_swarm_test_guard();
    let root = tempdir().expect("root");
    let nca_config_path = root.path().join("nca.toml");
    let training_config_path = root.path().join("nca-train.toml");
    let shard_root = root.path().join("nca-runtime-shards");
    write(&nca_config_path, &nca_corpus_config_toml(root.path()));
    write(
        &training_config_path,
        &nca_training_config_toml(&root.path().join("nca-cache"), &nca_config_path, SMALL_SPEC),
    );

    let native = DragonNativePeerConfig {
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-runtime-artifacts"),
        network: DragonPeerNetworkConfig::default()
            .with_listen_addresses(vec![loopback_swarm_address()]),
        target: Some(DragonNativeTarget::Trainer),
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0-pre.22").expect("valid burn_dragon version"),
        git_commit: Some("artifact-smoke".into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export: Some(DragonShardExportConfig {
            root: shard_root,
            dataset_name: Some("dragon-nca-runtime-artifacts".into()),
            microshards: Some(4),
            max_records: Some(32),
            http_upstream: None,
        }),
        existing_shard_dataset: None,
    };

    let prepared = prepare_nca_native_cpu(&native, Some(&dummy_auth_bundle())).expect("peer");
    let experiment_entry = prepared.manifests.experiment_directory[0].clone();
    let mut peer = spawn_prepared_native_peer(prepared).expect("spawn peer");
    let telemetry = peer.telemetry();
    wait_for(
        Duration::from_secs(10),
        || {
            let snapshot = telemetry.snapshot();
            snapshot.local_peer_id.is_some() && !snapshot.listen_addresses.is_empty()
        },
        "artifact runtime did not start",
    );

    let experiment = peer.mainnet().experiment(
        experiment_entry.study_id.clone(),
        experiment_entry.experiment_id.clone(),
        experiment_entry.current_revision_id.clone(),
    );
    let genesis_head = peer
        .initialize_local_head(&experiment)
        .expect("init local genesis head");
    let training = peer
        .train_window_once_with_pinned_head(&experiment, Some(&genesis_head))
        .expect("train one window");

    let loss = metric_float_any(&training.report.stats, &["loss", "train_loss"]);
    assert!(loss.is_finite(), "train loss must be finite");
    assert_eq!(
        training.head.parent_head_id,
        Some(genesis_head.head_id.clone())
    );
    assert!(
        training.artifact.bytes_len > 0,
        "artifact bytes should be non-zero"
    );
    assert!(
        !training.artifact.chunks.is_empty(),
        "artifact should contain at least one chunk"
    );

    let store = peer.artifact_store().expect("artifact store");
    assert!(
        store.has_manifest(&training.artifact.artifact_id),
        "runtime peer should persist the training update artifact manifest locally"
    );
    assert!(
        training
            .artifact
            .chunks
            .iter()
            .all(|chunk| store.has_chunk(&chunk.chunk_id)),
        "runtime peer should persist every training update artifact chunk locally"
    );
    assert!(
        store.has_manifest(&training.head.artifact_id),
        "runtime peer should persist the head artifact manifest locally"
    );

    peer.publish_head_provider(&experiment, &training.head)
        .expect("publish head provider");
    peer.publish_artifact_from_store(&training.artifact.artifact_id)
        .expect("publish delta artifact from local store");
    if training.head.artifact_id != training.artifact.artifact_id {
        peer.publish_artifact_from_store(&training.head.artifact_id)
            .expect("publish head artifact from local store");
    }

    shutdown_runtime_peer(peer, "artifact peer");
}

#[test]
#[ignore = "covered by the explicit nca-runtime-cluster validation rung"]
fn nca_native_runtime_cluster_smoke_converges_and_merges_heads() {
    let _guard = native_swarm_test_guard();
    let root = tempdir().expect("root");
    let bootstrap_storage = tempdir().expect("bootstrap storage");
    let nca_config_path = root.path().join("nca.toml");
    let training_config_path_seed = root.path().join("nca-train-seed.toml");
    let training_config_path_b = root.path().join("nca-train-b.toml");
    let training_config_path_c = root.path().join("nca-train-c.toml");
    write(&nca_config_path, &nca_corpus_config_toml(root.path()));
    write(
        &training_config_path_seed,
        &nca_training_config_toml(
            &root.path().join("nca-cache-seed"),
            &nca_config_path,
            SMALL_SPEC,
        ),
    );
    write(
        &training_config_path_b,
        &nca_training_config_toml(
            &root.path().join("nca-cache-b"),
            &nca_config_path,
            SMALL_SPEC,
        )
        .replace("seed = 1337", "seed = 1338")
        .replace("learning_rate = 0.001", "learning_rate = 0.0015"),
    );
    write(
        &training_config_path_c,
        &nca_training_config_toml(
            &root.path().join("nca-cache-c"),
            &nca_config_path,
            SMALL_SPEC,
        )
        .replace("seed = 1337", "seed = 1339")
        .replace("learning_rate = 0.001", "learning_rate = 0.002"),
    );

    let bootstrap_addr = loopback_swarm_address();
    let bootstrap_plan = burn_p2p_bootstrap::BootstrapSpec {
        preset: burn_p2p_bootstrap::BootstrapPreset::BootstrapOnly,
        genesis: burn_p2p_core::GenesisSpec {
            network_id: burn_p2p_core::NetworkId::new("dragon-p2p-testnet"),
            protocol_version: Version::new(0, 1, 0),
            display_name: "dragon runtime diffusion cluster smoke".into(),
            created_at: Utc::now(),
            metadata: BTreeMap::new(),
        },
        platform: ClientPlatform::Native,
        bootstrap_addresses: Vec::new(),
        listen_addresses: vec![bootstrap_addr.clone()],
        authority: None,
        archive: burn_p2p_bootstrap::ArchivePlan::default(),
        admin_api: burn_p2p_bootstrap::AdminApiPlan::default(),
    }
    .plan()
    .expect("bootstrap plan");
    let bootstrap = bootstrap_plan
        .spawn_bootstrap_peer_daemon(burn_p2p_bootstrap::BootstrapPeerDaemonConfig {
            node: burn_p2p::NodeConfig {
                identity: burn_p2p::IdentityConfig::Persistent,
                storage: Some(burn_p2p::StorageConfig::new(bootstrap_storage.path())),
                dataset: None,
                auth: None,
                network_manifest: None,
                client_release_manifest: None,
                selected_workload_id: None,
                metrics_retention: burn_p2p::MetricsRetentionConfig::default(),
                bootstrap_peers: Vec::new(),
                listen_addresses: vec![bootstrap_addr.clone()],
                external_addresses: Vec::new(),
            },
        })
        .expect("spawn bootstrap peer daemon");
    let bootstrap_telemetry = bootstrap.telemetry();
    wait_for(
        Duration::from_secs(10),
        || {
            let snapshot = bootstrap_telemetry.snapshot();
            snapshot.local_peer_id.is_some() && !snapshot.listen_addresses.is_empty()
        },
        "bootstrap-only peer daemon did not start",
    );
    assert!(
        !bootstrap_telemetry
            .snapshot()
            .configured_roles
            .contains(&PeerRole::Validator)
    );

    let make_trainer_config =
        |label: &str, training_config_path: &std::path::Path| DragonNativePeerConfig {
            training_config_paths: vec![training_config_path.to_path_buf()],
            storage_root: root.path().join(format!("storage-{label}")),
            network: Default::default(),
            target: Some(DragonNativeTarget::Trainer),
            identity: Default::default(),
            bootstrap_peers: vec![bootstrap_addr.clone()],
            manifest: native_manifest_seed(),
            app_semver: semver::Version::parse("0.21.0-pre.22").expect("valid burn_dragon version"),
            git_commit: Some(format!("runtime-cluster-{label}")),
            enabled_features_label: Some("native-cpu".into()),
            auth: None,
            capability_policy: Default::default(),
            shard_export: Some(DragonShardExportConfig {
                root: root.path().join(format!("shards-{label}")),
                dataset_name: Some(format!("dragon-nca-runtime-{label}")),
                microshards: Some(4),
                max_records: Some(32),
                http_upstream: None,
            }),
            existing_shard_dataset: None,
        };

    let seed_prepared = prepare_nca_native_cpu(
        &make_trainer_config("seed", &training_config_path_seed),
        Some(&dummy_auth_bundle()),
    )
    .expect("seed trainer");
    let experiment_entry = seed_prepared.manifests.experiment_directory[0].clone();
    let topology = experiment_entry
        .merge_topology_policy()
        .expect("diffusion merge topology");
    assert_eq!(topology.strategy, MergeStrategy::KRegularGossip);
    assert_eq!(
        topology.promotion_policy.mode,
        HeadPromotionMode::DiffusionSteadyState
    );
    assert!(
        experiment_entry
            .allowed_roles
            .contains(&PeerRole::TrainerCpu)
    );
    assert!(
        !experiment_entry
            .allowed_roles
            .contains(&PeerRole::Validator)
    );

    let trainer_b_prepared = prepare_nca_native_cpu(
        &make_trainer_config("trainer-b", &training_config_path_b),
        Some(&dummy_auth_bundle()),
    )
    .expect("trainer b");
    let trainer_c_prepared = prepare_nca_native_cpu(
        &make_trainer_config("trainer-c", &training_config_path_c),
        Some(&dummy_auth_bundle()),
    )
    .expect("trainer c");

    let mut seed = spawn_prepared_native_peer(seed_prepared).expect("spawn seed trainer");
    let mut trainer_b = spawn_prepared_native_peer(trainer_b_prepared).expect("spawn trainer b");
    let mut trainer_c = spawn_prepared_native_peer(trainer_c_prepared).expect("spawn trainer c");
    let seed_telemetry = seed.telemetry();
    let trainer_b_telemetry = trainer_b.telemetry();
    let trainer_c_telemetry = trainer_c.telemetry();

    wait_for(
        Duration::from_secs(30),
        || seed_telemetry.snapshot().connected_peers >= 1,
        "seed trainer did not connect",
    );
    wait_for(
        Duration::from_secs(30),
        || trainer_b_telemetry.snapshot().connected_peers >= 1,
        "trainer b did not connect",
    );
    wait_for(
        Duration::from_secs(30),
        || trainer_c_telemetry.snapshot().connected_peers >= 1,
        "trainer c did not connect",
    );

    let experiment = seed.mainnet().experiment(
        experiment_entry.study_id.clone(),
        experiment_entry.experiment_id.clone(),
        experiment_entry.current_revision_id.clone(),
    );
    let genesis_head = seed
        .initialize_local_head(&experiment)
        .expect("init diffusion genesis head");
    for trainer in [&trainer_b, &trainer_c] {
        wait_for(
            Duration::from_secs(45),
            || {
                trainer
                    .sync_experiment_head(&experiment)
                    .expect("sync trainer genesis head")
                    .is_some()
            },
            "trainer did not sync genesis head",
        );
    }

    let genesis_provider_peer_ids = [
        seed.snapshot().local_peer_id.expect("seed local peer id"),
        trainer_b
            .snapshot()
            .local_peer_id
            .expect("trainer b local peer id"),
        trainer_c
            .snapshot()
            .local_peer_id
            .expect("trainer c local peer id"),
    ];
    ensure_materialized_pinned_head(
        "seed",
        &seed,
        &experiment,
        &genesis_head,
        &genesis_provider_peer_ids,
    );
    ensure_materialized_pinned_head(
        "trainer-b",
        &trainer_b,
        &experiment,
        &genesis_head,
        &genesis_provider_peer_ids,
    );
    ensure_materialized_pinned_head(
        "trainer-c",
        &trainer_c,
        &experiment,
        &genesis_head,
        &genesis_provider_peer_ids,
    );

    let mut trainer_losses = Vec::new();
    let mut merged_losses = Vec::new();
    let mut canonical_head = genesis_head.clone();

    for round in 0..2 {
        let base_head_id = canonical_head.head_id.clone();
        let start_barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
        let experiment_for_seed = experiment.clone();
        let experiment_for_trainer_b = experiment.clone();
        let experiment_for_trainer_c = experiment.clone();
        let pinned_head_seed = canonical_head.clone();
        let pinned_head_b = canonical_head.clone();
        let pinned_head_c = canonical_head.clone();
        let seed_ref = &mut seed;
        let trainer_b_ref = &mut trainer_b;
        let trainer_c_ref = &mut trainer_c;
        let (seed_window, trainer_b_window, trainer_c_window) = thread::scope(|scope| {
            let seed = seed_ref;
            let seed_barrier = std::sync::Arc::clone(&start_barrier);
            let seed_run = scope.spawn(move || {
                seed_barrier.wait();
                seed.train_window_once_with_pinned_head(
                    &experiment_for_seed,
                    Some(&pinned_head_seed),
                )
            });
            let trainer_b = trainer_b_ref;
            let trainer_b_barrier = std::sync::Arc::clone(&start_barrier);
            let trainer_b_run = scope.spawn(move || {
                trainer_b_barrier.wait();
                trainer_b.train_window_once_with_pinned_head(
                    &experiment_for_trainer_b,
                    Some(&pinned_head_b),
                )
            });
            let trainer_c = trainer_c_ref;
            let trainer_c_barrier = std::sync::Arc::clone(&start_barrier);
            let trainer_c_run = scope.spawn(move || {
                trainer_c_barrier.wait();
                trainer_c.train_window_once_with_pinned_head(
                    &experiment_for_trainer_c,
                    Some(&pinned_head_c),
                )
            });
            let seed_window = seed_run
                .join()
                .map_err(|_| anyhow::anyhow!("runtime cluster seed train thread panicked"))??;
            let trainer_b_window = trainer_b_run.join().map_err(|_| {
                anyhow::anyhow!("runtime cluster trainer b train thread panicked")
            })??;
            let trainer_c_window = trainer_c_run.join().map_err(|_| {
                anyhow::anyhow!("runtime cluster trainer c train thread panicked")
            })??;
            anyhow::Ok((seed_window, trainer_b_window, trainer_c_window))
        })
        .expect("parallel runtime cluster windows");

        assert_eq!(
            seed_window.lease.window_id,
            trainer_b_window.lease.window_id
        );
        assert_eq!(
            seed_window.lease.window_id,
            trainer_c_window.lease.window_id
        );
        let window_id = seed_window.lease.window_id;

        let round_outcomes = [&seed_window, &trainer_b_window, &trainer_c_window];
        for outcome in round_outcomes {
            let loss = metric_float_any(&outcome.report.stats, &["loss", "train_loss"]);
            trainer_losses.push(loss);
            assert!(loss.is_finite());
            assert_eq!(outcome.head.parent_head_id, Some(base_head_id.clone()));
            assert_eq!(outcome.head.global_step, canonical_head.global_step + 1);
        }

        for (label, peer, outcome) in [
            ("seed", &seed, &seed_window),
            ("trainer-b", &trainer_b, &trainer_b_window),
            ("trainer-c", &trainer_c, &trainer_c_window),
        ] {
            let store = peer.artifact_store().expect("artifact store");
            assert!(
                store.has_manifest(&outcome.artifact.artifact_id),
                "{label} should persist its update artifact manifest locally"
            );
            assert!(
                outcome
                    .artifact
                    .chunks
                    .iter()
                    .all(|chunk| store.has_chunk(&chunk.chunk_id)),
                "{label} should persist all update artifact chunks locally"
            );
        }

        let provider_peer_ids = [
            seed.snapshot().local_peer_id.expect("seed local peer id"),
            trainer_b
                .snapshot()
                .local_peer_id
                .expect("trainer b local peer id"),
            trainer_c
                .snapshot()
                .local_peer_id
                .expect("trainer c local peer id"),
        ];
        for (label, peer) in [
            ("seed", &seed),
            ("trainer-b", &trainer_b),
            ("trainer-c", &trainer_c),
        ] {
            for (artifact_label, artifact_id) in [
                ("seed update", &seed_window.artifact.artifact_id),
                ("seed head", &seed_window.head.artifact_id),
                ("trainer-b update", &trainer_b_window.artifact.artifact_id),
                ("trainer-b head", &trainer_b_window.head.artifact_id),
                ("trainer-c update", &trainer_c_window.artifact.artifact_id),
                ("trainer-c head", &trainer_c_window.head.artifact_id),
            ] {
                ensure_materialized_artifact(
                    label,
                    peer,
                    &provider_peer_ids,
                    artifact_id,
                    artifact_label,
                    Duration::from_secs(45),
                );
            }
        }

        eprintln!(
            "nca_runtime_cluster_round_{round}_artifacts: seed_bytes={} seed_chunks={} trainer_b_bytes={} trainer_b_chunks={} trainer_c_bytes={} trainer_c_chunks={}",
            seed_window.artifact.bytes_len,
            seed_window.artifact.chunks.len(),
            trainer_b_window.artifact.bytes_len,
            trainer_b_window.artifact.chunks.len(),
            trainer_c_window.artifact.bytes_len,
            trainer_c_window.artifact.chunks.len(),
        );

        wait_for(
            Duration::from_secs(30),
            || {
                [
                    seed_telemetry.snapshot(),
                    trainer_b_telemetry.snapshot(),
                    trainer_c_telemetry.snapshot(),
                ]
                .into_iter()
                .all(|snapshot| {
                    snapshot
                        .control_plane
                        .update_announcements
                        .iter()
                        .filter(|announcement| {
                            announcement.update.study_id == experiment.study_id
                                && announcement.update.experiment_id == experiment.experiment_id
                                && announcement.update.revision_id == experiment.revision_id
                                && announcement.update.window_id == window_id
                                && announcement.update.base_head_id == base_head_id
                        })
                        .count()
                        >= 3
                        && snapshot
                            .control_plane
                            .reducer_assignment_announcements
                            .is_empty()
                        && snapshot
                            .control_plane
                            .aggregate_proposal_announcements
                            .is_empty()
                        && snapshot
                            .control_plane
                            .validation_quorum_announcements
                            .is_empty()
                })
            },
            "runtime diffusion cluster did not observe the trainer-only update frontier",
        );

        let convergence_deadline = Instant::now() + Duration::from_secs(120);
        let expected_promoted_global_step = canonical_head.global_step + 1;
        let promoted_head = loop {
            advance_diffusion_with_retry("advance seed diffusion", convergence_deadline, || {
                seed.advance_diffusion_steady_state(&experiment, None, None)
            });
            advance_diffusion_with_retry(
                "advance trainer b diffusion",
                convergence_deadline,
                || trainer_b.advance_diffusion_steady_state(&experiment, None, None),
            );
            advance_diffusion_with_retry(
                "advance trainer c diffusion",
                convergence_deadline,
                || trainer_c.advance_diffusion_steady_state(&experiment, None, None),
            );

            let seed_head = sync_experiment_head_with_retry(
                "sync runtime seed head",
                &seed,
                &experiment,
                convergence_deadline,
            );
            let trainer_b_head = sync_experiment_head_with_retry(
                "sync runtime trainer b head",
                &trainer_b,
                &experiment,
                convergence_deadline,
            );
            let trainer_c_head = sync_experiment_head_with_retry(
                "sync runtime trainer c head",
                &trainer_c,
                &experiment,
                convergence_deadline,
            );
            if let Some(candidate) = select_promoted_head_candidate(
                [&seed_head, &trainer_b_head, &trainer_c_head],
                &base_head_id,
                expected_promoted_global_step,
            ) {
                break candidate;
            }
            assert!(
                Instant::now() < convergence_deadline,
                "runtime diffusion cluster did not produce a valid promoted head; seed={} trainer-b={} trainer-c={}",
                describe_head_state(&seed_head),
                describe_head_state(&trainer_b_head),
                describe_head_state(&trainer_c_head),
            );
            thread::sleep(Duration::from_millis(25));
        };

        let propagation_deadline = Instant::now() + Duration::from_secs(60);
        loop {
            advance_diffusion_with_retry("propagate seed diffusion", propagation_deadline, || {
                seed.advance_diffusion_steady_state(&experiment, None, None)
            });
            advance_diffusion_with_retry(
                "propagate trainer b diffusion",
                propagation_deadline,
                || trainer_b.advance_diffusion_steady_state(&experiment, None, None),
            );
            advance_diffusion_with_retry(
                "propagate trainer c diffusion",
                propagation_deadline,
                || trainer_c.advance_diffusion_steady_state(&experiment, None, None),
            );

            let seed_head = sync_experiment_head_with_retry(
                "sync propagated runtime seed head",
                &seed,
                &experiment,
                propagation_deadline,
            );
            let trainer_b_head = sync_experiment_head_with_retry(
                "sync propagated runtime trainer b head",
                &trainer_b,
                &experiment,
                propagation_deadline,
            );
            let trainer_c_head = sync_experiment_head_with_retry(
                "sync propagated runtime trainer c head",
                &trainer_c,
                &experiment,
                propagation_deadline,
            );
            if peers_have_promoted_head(
                [&seed_head, &trainer_b_head, &trainer_c_head],
                &promoted_head,
                &base_head_id,
                expected_promoted_global_step,
            ) {
                break;
            }
            assert!(
                Instant::now() < propagation_deadline,
                "runtime diffusion cluster did not propagate promoted head {} across peers; seed={} trainer-b={} trainer-c={}",
                promoted_head.head_id.as_str(),
                describe_head_state(&seed_head),
                describe_head_state(&trainer_b_head),
                describe_head_state(&trainer_c_head),
            );
            thread::sleep(Duration::from_millis(25));
        }

        let merged_loss = metric_float_any(&promoted_head.metrics, &["loss", "train_loss"]);
        merged_losses.push(merged_loss);
        assert!(merged_loss.is_finite());
        assert_eq!(promoted_head.parent_head_id, Some(base_head_id.clone()));
        assert_eq!(promoted_head.global_step, expected_promoted_global_step);

        wait_for(
            Duration::from_secs(40),
            || {
                [
                    seed_telemetry.snapshot(),
                    trainer_b_telemetry.snapshot(),
                    trainer_c_telemetry.snapshot(),
                ]
                .into_iter()
                .all(|snapshot| {
                    snapshot
                        .control_plane
                        .diffusion_promotion_certificate_announcements
                        .iter()
                        .any(|announcement| {
                            announcement.certificate.window_id == window_id
                                && announcement.certificate.base_head_id == base_head_id
                                && announcement.certificate.merged_head_id == promoted_head.head_id
                                && announcement.certificate.promotion_mode
                                    == HeadPromotionMode::DiffusionSteadyState
                        })
                        && snapshot
                            .control_plane
                            .merge_announcements
                            .iter()
                            .any(|announcement| {
                                announcement.certificate.base_head_id == base_head_id
                                    && announcement.certificate.merged_head_id
                                        == promoted_head.head_id
                                    && announcement.certificate.promotion_mode
                                        == HeadPromotionMode::DiffusionSteadyState
                            })
                        && snapshot
                            .control_plane
                            .validation_quorum_announcements
                            .is_empty()
                })
            },
            "runtime diffusion promotion certificates did not propagate across the trainer swarm",
        );

        eprintln!(
            "nca_runtime_cluster_round_{round}: trainer_losses=({:.4}, {:.4}, {:.4}) merged_loss={:.4} global_step={}",
            metric_float_any(&seed_window.report.stats, &["loss", "train_loss"]),
            metric_float_any(&trainer_b_window.report.stats, &["loss", "train_loss"]),
            metric_float_any(&trainer_c_window.report.stats, &["loss", "train_loss"]),
            merged_loss,
            promoted_head.global_step,
        );

        ensure_materialized_pinned_head(
            "seed",
            &seed,
            &experiment,
            &promoted_head,
            &provider_peer_ids,
        );
        ensure_materialized_pinned_head(
            "trainer-b",
            &trainer_b,
            &experiment,
            &promoted_head,
            &provider_peer_ids,
        );
        ensure_materialized_pinned_head(
            "trainer-c",
            &trainer_c,
            &experiment,
            &promoted_head,
            &provider_peer_ids,
        );

        canonical_head = promoted_head;
    }

    log_loss_series("nca_runtime_cluster_trainers", &trainer_losses);
    log_loss_series("nca_runtime_cluster_merged", &merged_losses);
    assert!(trainer_losses.iter().all(|loss| loss.is_finite()));
    assert!(merged_losses.iter().all(|loss| loss.is_finite()));
    assert_eq!(canonical_head.global_step, 2);

    shutdown_runtime_peer(trainer_c, "runtime cluster trainer c");
    shutdown_runtime_peer(trainer_b, "runtime cluster trainer b");
    shutdown_runtime_peer(seed, "runtime cluster seed");
    bootstrap
        .shutdown()
        .expect("bootstrap-only peer daemon shutdown");
    bootstrap
        .await_termination()
        .expect("bootstrap-only peer daemon termination");
}

#[test]
fn nca_bootstrap_only_topology_supports_trainer_only_diffusion_roles() {
    let _guard = native_swarm_test_guard();
    let root = tempdir().expect("root");
    let nca_config_path = root.path().join("nca.toml");
    let training_config_path = root.path().join("nca-train.toml");
    write(&nca_config_path, &nca_corpus_config_toml(root.path()));
    write(
        &training_config_path,
        &nca_training_config_toml(&root.path().join("nca-cache"), &nca_config_path, SMALL_SPEC),
    );

    let bootstrap_roles = burn_p2p_bootstrap::BootstrapPreset::BootstrapOnly.roles();
    let bootstrap_services = burn_p2p_bootstrap::BootstrapPreset::BootstrapOnly.services();
    assert!(bootstrap_roles.contains(&PeerRole::Bootstrap));
    assert!(bootstrap_roles.contains(&PeerRole::RelayHelper));
    assert!(!bootstrap_roles.contains(&PeerRole::Validator));
    assert!(!bootstrap_services.contains(&burn_p2p_bootstrap::BootstrapService::Validator));

    let bootstrap_addr = loopback_swarm_address();
    let trainer_config = DragonNativePeerConfig {
        training_config_paths: vec![training_config_path.clone()],
        storage_root: root.path().join("storage-trainer-bootstrap-only"),
        network: Default::default(),
        target: Some(DragonNativeTarget::Trainer),
        identity: Default::default(),
        bootstrap_peers: vec![bootstrap_addr],
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0-pre.22").expect("valid burn_dragon version"),
        git_commit: Some("bootstrap-only-trainer".into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export: Some(DragonShardExportConfig {
            root: root.path().join("trainer-shards-bootstrap-only"),
            dataset_name: Some("dragon-nca-bootstrap-only-trainer".into()),
            microshards: Some(4),
            max_records: Some(32),
            http_upstream: None,
        }),
        existing_shard_dataset: None,
    };
    let trainer_prepared =
        prepare_nca_native_cpu(&trainer_config, Some(&dummy_auth_bundle())).expect("trainer");
    assert_eq!(
        trainer_prepared.target_decision.effective_target,
        DragonNativeTarget::Trainer
    );
    assert!(trainer_prepared.target_decision.can_train);
    let entry = &trainer_prepared.manifests.experiment_directory[0];
    assert!(!entry.allowed_roles.contains(&PeerRole::Validator));
    assert!(!entry.allowed_roles.contains(&PeerRole::BrowserVerifier));
    assert!(entry.allowed_roles.contains(&PeerRole::Archive));
    assert!(entry.allowed_scopes.contains(&ExperimentScope::Archive {
        experiment_id: entry.experiment_id.clone(),
    }));
    assert!(!entry.allowed_scopes.contains(&ExperimentScope::Validate {
        experiment_id: entry.experiment_id.clone(),
    }));
    let topology = entry
        .merge_topology_policy()
        .expect("trainer-only diffusion topology");
    assert_eq!(topology.strategy, MergeStrategy::KRegularGossip);
    assert_eq!(
        topology.promotion_policy.mode,
        HeadPromotionMode::DiffusionSteadyState
    );
}

#[test]
fn nca_bootstrap_only_topology_diffusion_converges_across_trainers() {
    let _guard = native_swarm_test_guard();
    let root = tempdir().expect("root");
    let bootstrap_storage = tempdir().expect("bootstrap storage");
    let nca_config_path = root.path().join("nca.toml");
    let training_config_path_seed = root.path().join("nca-train-seed.toml");
    let training_config_path_b = root.path().join("nca-train-b.toml");
    let training_config_path_c = root.path().join("nca-train-c.toml");
    write(&nca_config_path, &nca_corpus_config_toml(root.path()));
    write(
        &training_config_path_seed,
        &nca_training_config_toml(
            &root.path().join("nca-cache-seed"),
            &nca_config_path,
            SMALL_SPEC,
        ),
    );
    write(
        &training_config_path_b,
        &nca_training_config_toml(
            &root.path().join("nca-cache-b"),
            &nca_config_path,
            SMALL_SPEC,
        )
        .replace("seed = 1337", "seed = 1338")
        .replace("learning_rate = 0.001", "learning_rate = 0.0015"),
    );
    write(
        &training_config_path_c,
        &nca_training_config_toml(
            &root.path().join("nca-cache-c"),
            &nca_config_path,
            SMALL_SPEC,
        )
        .replace("seed = 1337", "seed = 1339")
        .replace("learning_rate = 0.001", "learning_rate = 0.002"),
    );

    let bootstrap_addr = loopback_swarm_address();
    let bootstrap_plan = burn_p2p_bootstrap::BootstrapSpec {
        preset: burn_p2p_bootstrap::BootstrapPreset::BootstrapOnly,
        genesis: burn_p2p_core::GenesisSpec {
            network_id: burn_p2p_core::NetworkId::new("dragon-p2p-testnet"),
            protocol_version: Version::new(0, 1, 0),
            display_name: "dragon bootstrap-only diffusion topology".into(),
            created_at: Utc::now(),
            metadata: BTreeMap::new(),
        },
        platform: ClientPlatform::Native,
        bootstrap_addresses: Vec::new(),
        listen_addresses: vec![bootstrap_addr.clone()],
        authority: None,
        archive: burn_p2p_bootstrap::ArchivePlan::default(),
        admin_api: burn_p2p_bootstrap::AdminApiPlan::default(),
    }
    .plan()
    .expect("bootstrap plan");
    let bootstrap = bootstrap_plan
        .spawn_bootstrap_peer_daemon(burn_p2p_bootstrap::BootstrapPeerDaemonConfig {
            node: burn_p2p::NodeConfig {
                identity: burn_p2p::IdentityConfig::Persistent,
                storage: Some(burn_p2p::StorageConfig::new(bootstrap_storage.path())),
                dataset: None,
                auth: None,
                network_manifest: None,
                client_release_manifest: None,
                selected_workload_id: None,
                metrics_retention: burn_p2p::MetricsRetentionConfig::default(),
                bootstrap_peers: Vec::new(),
                listen_addresses: vec![bootstrap_addr.clone()],
                external_addresses: Vec::new(),
            },
        })
        .expect("spawn bootstrap peer daemon");
    let bootstrap_telemetry = bootstrap.telemetry();
    wait_for(
        Duration::from_secs(10),
        || {
            let snapshot = bootstrap_telemetry.snapshot();
            snapshot.local_peer_id.is_some() && !snapshot.listen_addresses.is_empty()
        },
        "bootstrap-only peer daemon did not start",
    );
    assert!(
        !bootstrap_telemetry
            .snapshot()
            .configured_roles
            .contains(&PeerRole::Validator)
    );

    let make_trainer_config =
        |label: &str, training_config_path: &std::path::Path| DragonNativePeerConfig {
            training_config_paths: vec![training_config_path.to_path_buf()],
            storage_root: root.path().join(format!("storage-{label}")),
            network: Default::default(),
            target: Some(DragonNativeTarget::Trainer),
            identity: Default::default(),
            bootstrap_peers: vec![bootstrap_addr.clone()],
            manifest: native_manifest_seed(),
            app_semver: semver::Version::parse("0.21.0-pre.22").expect("valid burn_dragon version"),
            git_commit: Some(label.into()),
            enabled_features_label: Some("native-cpu".into()),
            auth: None,
            capability_policy: Default::default(),
            shard_export: Some(DragonShardExportConfig {
                root: root.path().join(format!("shards-{label}")),
                dataset_name: Some(format!("dragon-nca-{label}")),
                microshards: Some(4),
                max_records: Some(32),
                http_upstream: None,
            }),
            existing_shard_dataset: None,
        };

    let seed_prepared = prepare_nca_native_cpu(
        &make_trainer_config("bootstrap-diffusion-seed", &training_config_path_seed),
        Some(&dummy_auth_bundle()),
    )
    .expect("seed trainer");
    let experiment_entry = seed_prepared.manifests.experiment_directory[0].clone();
    let topology = experiment_entry
        .merge_topology_policy()
        .expect("diffusion merge topology");
    assert_eq!(topology.strategy, MergeStrategy::KRegularGossip);
    assert_eq!(
        topology.promotion_policy.mode,
        HeadPromotionMode::DiffusionSteadyState
    );
    assert!(
        experiment_entry
            .allowed_roles
            .contains(&PeerRole::TrainerCpu)
    );
    assert!(
        !experiment_entry
            .allowed_roles
            .contains(&PeerRole::Validator)
    );
    assert!(
        !experiment_entry
            .allowed_scopes
            .contains(&ExperimentScope::Validate {
                experiment_id: experiment_entry.experiment_id.clone(),
            })
    );

    let trainer_b_prepared = prepare_nca_native_cpu(
        &make_trainer_config("bootstrap-diffusion-b", &training_config_path_b),
        Some(&dummy_auth_bundle()),
    )
    .expect("trainer b");
    let trainer_c_prepared = prepare_nca_native_cpu(
        &make_trainer_config("bootstrap-diffusion-c", &training_config_path_c),
        Some(&dummy_auth_bundle()),
    )
    .expect("trainer c");

    let mut seed = spawn_prepared_native_peer(seed_prepared).expect("spawn seed trainer");
    let mut trainer_b = spawn_prepared_native_peer(trainer_b_prepared).expect("spawn trainer b");
    let mut trainer_c = spawn_prepared_native_peer(trainer_c_prepared).expect("spawn trainer c");
    let seed_telemetry = seed.telemetry();
    let trainer_b_telemetry = trainer_b.telemetry();
    let trainer_c_telemetry = trainer_c.telemetry();

    wait_for(
        Duration::from_secs(20),
        || seed_telemetry.snapshot().connected_peers >= 1,
        "seed trainer did not connect",
    );
    wait_for(
        Duration::from_secs(20),
        || trainer_b_telemetry.snapshot().connected_peers >= 1,
        "trainer b did not connect",
    );
    wait_for(
        Duration::from_secs(20),
        || trainer_c_telemetry.snapshot().connected_peers >= 1,
        "trainer c did not connect",
    );

    let experiment = seed.mainnet().experiment(
        experiment_entry.study_id.clone(),
        experiment_entry.experiment_id.clone(),
        experiment_entry.current_revision_id.clone(),
    );
    let genesis_head = seed
        .initialize_local_head(&experiment)
        .expect("init diffusion genesis head");

    for trainer in [&trainer_b, &trainer_c] {
        wait_for(
            Duration::from_secs(30),
            || {
                trainer
                    .sync_experiment_head(&experiment)
                    .expect("sync trainer genesis head")
                    .is_some()
            },
            "trainer did not sync genesis head",
        );
    }

    let start_barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
    let experiment_for_seed = experiment.clone();
    let experiment_for_trainer_b = experiment.clone();
    let experiment_for_trainer_c = experiment.clone();
    let seed_ref = &mut seed;
    let trainer_b_ref = &mut trainer_b;
    let trainer_c_ref = &mut trainer_c;
    let (seed_window, trainer_b_window, trainer_c_window) = thread::scope(|scope| {
        let seed = seed_ref;
        let seed_barrier = std::sync::Arc::clone(&start_barrier);
        let seed_run = scope.spawn(move || {
            seed_barrier.wait();
            seed.train_window_once(&experiment_for_seed)
        });
        let trainer_b = trainer_b_ref;
        let trainer_b_barrier = std::sync::Arc::clone(&start_barrier);
        let trainer_b_run = scope.spawn(move || {
            trainer_b_barrier.wait();
            trainer_b.train_window_once(&experiment_for_trainer_b)
        });
        let trainer_c = trainer_c_ref;
        let trainer_c_barrier = std::sync::Arc::clone(&start_barrier);
        let trainer_c_run = scope.spawn(move || {
            trainer_c_barrier.wait();
            trainer_c.train_window_once(&experiment_for_trainer_c)
        });
        let seed_window = seed_run
            .join()
            .map_err(|_| anyhow::anyhow!("diffusion seed train thread panicked"))??;
        let trainer_b_window = trainer_b_run
            .join()
            .map_err(|_| anyhow::anyhow!("diffusion trainer b train thread panicked"))??;
        let trainer_c_window = trainer_c_run
            .join()
            .map_err(|_| anyhow::anyhow!("diffusion trainer c train thread panicked"))??;
        anyhow::Ok((seed_window, trainer_b_window, trainer_c_window))
    })
    .expect("parallel diffusion windows");
    for outcome in [&seed_window, &trainer_b_window, &trainer_c_window] {
        assert_eq!(
            outcome.head.parent_head_id,
            Some(genesis_head.head_id.clone())
        );
        assert_eq!(outcome.head.global_step, 1);
    }

    wait_for(
        Duration::from_secs(20),
        || {
            [
                seed_telemetry.snapshot(),
                trainer_b_telemetry.snapshot(),
                trainer_c_telemetry.snapshot(),
            ]
            .into_iter()
            .all(|snapshot| {
                let updates = snapshot
                    .control_plane
                    .update_announcements
                    .iter()
                    .filter(|announcement| {
                        announcement.update.study_id == experiment.study_id
                            && announcement.update.experiment_id == experiment.experiment_id
                            && announcement.update.revision_id == experiment.revision_id
                            && announcement.update.window_id == WindowId(1)
                            && announcement.update.base_head_id == genesis_head.head_id
                    })
                    .count();
                updates >= 3
                    && snapshot
                        .control_plane
                        .reducer_assignment_announcements
                        .is_empty()
                    && snapshot
                        .control_plane
                        .aggregate_proposal_announcements
                        .is_empty()
                    && snapshot
                        .control_plane
                        .validation_quorum_announcements
                        .is_empty()
            })
        },
        "diffusion trainers did not observe the trainer-only update frontier",
    );

    let convergence_deadline = Instant::now() + Duration::from_secs(20);
    let expected_promoted_global_step = genesis_head.global_step + 1;
    let promoted_head = loop {
        advance_diffusion_with_retry("advance seed diffusion", convergence_deadline, || {
            seed.advance_diffusion_steady_state(&experiment, None, None)
        });
        advance_diffusion_with_retry("advance trainer b diffusion", convergence_deadline, || {
            trainer_b.advance_diffusion_steady_state(&experiment, None, None)
        });
        advance_diffusion_with_retry("advance trainer c diffusion", convergence_deadline, || {
            trainer_c.advance_diffusion_steady_state(&experiment, None, None)
        });

        let seed_head = sync_experiment_head_with_retry(
            "sync diffusion seed head",
            &seed,
            &experiment,
            convergence_deadline,
        );
        let trainer_b_head = sync_experiment_head_with_retry(
            "sync diffusion trainer b head",
            &trainer_b,
            &experiment,
            convergence_deadline,
        );
        let trainer_c_head = sync_experiment_head_with_retry(
            "sync diffusion trainer c head",
            &trainer_c,
            &experiment,
            convergence_deadline,
        );
        if let Some(candidate) = select_promoted_head_candidate(
            [&seed_head, &trainer_b_head, &trainer_c_head],
            &genesis_head.head_id,
            expected_promoted_global_step,
        ) {
            break candidate;
        }
        assert!(
            Instant::now() < convergence_deadline,
            "diffusion trainers did not produce a valid promoted head; seed={} trainer-b={} trainer-c={}",
            describe_head_state(&seed_head),
            describe_head_state(&trainer_b_head),
            describe_head_state(&trainer_c_head),
        );
        thread::sleep(Duration::from_millis(25));
    };

    let propagation_deadline = Instant::now() + Duration::from_secs(20);
    loop {
        advance_diffusion_with_retry("propagate seed diffusion", propagation_deadline, || {
            seed.advance_diffusion_steady_state(&experiment, None, None)
        });
        advance_diffusion_with_retry(
            "propagate trainer b diffusion",
            propagation_deadline,
            || trainer_b.advance_diffusion_steady_state(&experiment, None, None),
        );
        advance_diffusion_with_retry(
            "propagate trainer c diffusion",
            propagation_deadline,
            || trainer_c.advance_diffusion_steady_state(&experiment, None, None),
        );

        let seed_head = sync_experiment_head_with_retry(
            "sync propagated diffusion seed head",
            &seed,
            &experiment,
            propagation_deadline,
        );
        let trainer_b_head = sync_experiment_head_with_retry(
            "sync propagated diffusion trainer b head",
            &trainer_b,
            &experiment,
            propagation_deadline,
        );
        let trainer_c_head = sync_experiment_head_with_retry(
            "sync propagated diffusion trainer c head",
            &trainer_c,
            &experiment,
            propagation_deadline,
        );
        if peers_have_promoted_head(
            [&seed_head, &trainer_b_head, &trainer_c_head],
            &promoted_head,
            &genesis_head.head_id,
            expected_promoted_global_step,
        ) {
            break;
        }
        assert!(
            Instant::now() < propagation_deadline,
            "diffusion trainers did not propagate promoted head {} across peers; seed={} trainer-b={} trainer-c={}",
            promoted_head.head_id.as_str(),
            describe_head_state(&seed_head),
            describe_head_state(&trainer_b_head),
            describe_head_state(&trainer_c_head),
        );
        thread::sleep(Duration::from_millis(25));
    }

    assert_eq!(promoted_head.global_step, expected_promoted_global_step);
    assert_eq!(
        promoted_head.parent_head_id,
        Some(genesis_head.head_id.clone())
    );
    wait_for(
        Duration::from_secs(10),
        || {
            [
                seed_telemetry.snapshot(),
                trainer_b_telemetry.snapshot(),
                trainer_c_telemetry.snapshot(),
            ]
            .into_iter()
            .all(|snapshot| {
                !snapshot
                    .control_plane
                    .diffusion_promotion_certificate_announcements
                    .is_empty()
                    && !snapshot.control_plane.merge_announcements.is_empty()
                    && snapshot
                        .control_plane
                        .validation_quorum_announcements
                        .is_empty()
            })
        },
        "diffusion promotion certificates did not propagate across the trainer swarm",
    );

    shutdown_runtime_peer(trainer_c, "bootstrap diffusion trainer c");
    shutdown_runtime_peer(trainer_b, "bootstrap diffusion trainer b");
    shutdown_runtime_peer(seed, "bootstrap diffusion seed");
    bootstrap
        .shutdown()
        .expect("bootstrap-only peer daemon shutdown");
    bootstrap
        .await_termination()
        .expect("bootstrap-only peer daemon termination");
}

fn shutdown_runtime_peer<B>(peer: ManagedRunningNativePeer<B>, label: &str)
where
    B: burn::tensor::backend::AutodiffBackend + Clone + 'static,
{
    peer.shutdown()
        .unwrap_or_else(|error| panic!("{label} shutdown: {error:#}"));
    match peer.await_termination_timeout(Duration::from_secs(10)) {
        Ok(_prepared) => {}
        Err(error) if error.to_string().contains("runtime thread panicked") => {
            eprintln!(
                "{label} termination hit known upstream libp2p runtime panic during shutdown: {error:#}"
            );
        }
        Err(error) => panic!("{label} termination: {error:#}"),
    }
}

#[test]
fn nca_native_auto_target_holds_trainer_role_under_tight_budget() {
    let _guard = native_swarm_test_guard();
    let root = tempdir().expect("root");
    let nca_config_path = root.path().join("nca.toml");
    let training_config_path = root.path().join("nca-train.toml");
    write(&nca_config_path, &nca_corpus_config_toml(root.path()));
    write(
        &training_config_path,
        &nca_training_config_toml(&root.path().join("nca-cache"), &nca_config_path, SMALL_SPEC),
    );

    let native = DragonNativePeerConfig {
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-downgrade"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0-pre.22").expect("valid burn_dragon version"),
        git_commit: Some("downgrade".into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: DragonCapabilityPolicy {
            native_cpu_memory_budget_bytes: Some(1),
            ..Default::default()
        },
        shard_export: None,
        existing_shard_dataset: None,
    };

    let prepared = prepare_nca_native_cpu(&native, Some(&dummy_auth_bundle())).expect("peer");
    assert_eq!(
        prepared.target_decision.requested_target,
        DragonNativeTarget::Auto
    );
    assert_eq!(
        prepared.target_decision.effective_target,
        DragonNativeTarget::Trainer
    );
    assert!(!prepared.target_decision.can_train);
    assert!(prepared.target_decision.downgrade_reason.is_none());
    assert_eq!(
        prepared.manifests.experiment_directory[0]
            .resource_requirements
            .minimum_device_memory_bytes,
        None
    );
    assert_eq!(
        prepared.manifests.experiment_directory[0]
            .resource_requirements
            .minimum_system_memory_bytes,
        Some(
            prepared
                .footprint
                .estimated_training_bytes
                .max(512 * 1024 * 1024)
        )
    );
    let expected_training_bytes = prepared.footprint.estimated_training_bytes.to_string();
    assert_eq!(
        prepared.manifests.experiment_directory[0]
            .metadata
            .get("estimated_training_bytes")
            .map(String::as_str),
        Some(expected_training_bytes.as_str())
    );
}

#[test]
fn nca_native_persisted_runtime_failure_holds_trainer_role_on_reprepare() {
    let _guard = native_swarm_test_guard();
    let root = tempdir().expect("root");
    let nca_config_path = root.path().join("nca.toml");
    let training_config_path = root.path().join("nca-train.toml");
    write(&nca_config_path, &nca_corpus_config_toml(root.path()));
    write(
        &training_config_path,
        &nca_training_config_toml(&root.path().join("nca-cache"), &nca_config_path, SMALL_SPEC),
    );

    let native = DragonNativePeerConfig {
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-downgrade-persisted"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0-pre.22").expect("valid burn_dragon version"),
        git_commit: Some("downgrade-persisted".into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export: None,
        existing_shard_dataset: None,
    };

    let prepared = prepare_nca_native_cpu(&native, Some(&dummy_auth_bundle())).expect("trainer");
    assert!(prepared.target_decision.can_train);
    assert_eq!(
        prepared.target_decision.effective_target,
        DragonNativeTarget::Trainer
    );

    prepared
        .record_runtime_training_failure("out of memory allocating optimizer state")
        .expect("persist runtime downgrade");

    let downgraded =
        prepare_nca_native_cpu(&native, Some(&dummy_auth_bundle())).expect("reprepare");
    assert_eq!(
        downgraded.target_decision.effective_target,
        DragonNativeTarget::Trainer
    );
    assert!(!downgraded.target_decision.can_train);
    assert!(
        downgraded
            .target_decision
            .downgrade_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("persisted trainer failure")
                && reason.contains("holding trainer role"))
    );

    downgraded
        .clear_runtime_downgrade()
        .expect("clear persisted downgrade");

    let recovered = prepare_nca_native_cpu(&native, Some(&dummy_auth_bundle())).expect("recovered");
    assert!(recovered.target_decision.can_train);
    assert_eq!(
        recovered.target_decision.effective_target,
        DragonNativeTarget::Trainer
    );
}

#[test]
fn climbmix_native_existing_shards_supports_multi_peer_windows() {
    let _guard = native_swarm_test_guard();
    let root = tempdir().expect("root");
    let shard_root = root.path().join("climbmix-shards");
    fs::create_dir_all(&shard_root).expect("mkdir shards");
    write_existing_climbmix_shards(&shard_root, 16, 8);
    let training_config_path = root.path().join("climbmix-train.toml");
    write(
        &training_config_path,
        &climbmix_training_config_toml(&root.path().join("climbmix-cache"), SMALL_SPEC),
    );

    let base_native = DragonNativePeerConfig {
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-peer-a"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0-pre.22").expect("valid burn_dragon version"),
        git_commit: Some("smoke".into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export: None,
        existing_shard_dataset: Some(DragonExistingShardDatasetConfig {
            root: shard_root.clone(),
            http_upstream: None,
        }),
    };
    let peer_a =
        prepare_climbmix_native_cpu(&base_native, Some(&dummy_auth_bundle())).expect("peer a");
    let mut peer_b_config = base_native.clone();
    peer_b_config.storage_root = root.path().join("storage-peer-b");
    let peer_b =
        prepare_climbmix_native_cpu(&peer_b_config, Some(&dummy_auth_bundle())).expect("peer b");

    assert_eq!(
        peer_a.manifests.network_manifest.network_id,
        peer_b.manifests.network_manifest.network_id
    );
    assert_eq!(
        peer_a.manifests.supported_workload.workload_id,
        peer_b.manifests.supported_workload.workload_id
    );
    assert_eq!(
        peer_a.manifests.experiment_directory[0].dataset_view_id,
        peer_b.manifests.experiment_directory[0].dataset_view_id
    );

    let losses_a = run_training_windows(&peer_a, 3);
    let losses_b = run_training_windows(&peer_b, 3);
    log_loss_series("climbmix_native_smoke_peer_a", &losses_a);
    log_loss_series("climbmix_native_smoke_peer_b", &losses_b);
    assert!(losses_a.iter().all(|loss| loss.is_finite()));
    assert!(losses_b.iter().all(|loss| loss.is_finite()));
    assert!(losses_a.iter().copied().fold(f64::INFINITY, f64::min) <= losses_a[0] + 0.5);
    assert!(losses_b.iter().copied().fold(f64::INFINITY, f64::min) <= losses_b[0] + 0.5);
}

#[test]
fn browser_conformance_uses_native_dragon_manifests() {
    let _guard = native_swarm_test_guard();
    let root = tempdir().expect("root");
    let nca_config_path = root.path().join("nca.toml");
    let training_config_path = root.path().join("nca-train.toml");
    let shard_root = root.path().join("nca-shards");
    write(&nca_config_path, &nca_corpus_config_toml(root.path()));
    write(
        &training_config_path,
        &nca_training_config_toml(&root.path().join("nca-cache"), &nca_config_path, SMALL_SPEC),
    );

    let native = DragonNativePeerConfig {
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-browser-compat"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0-pre.22").expect("valid burn_dragon version"),
        git_commit: Some("smoke".into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export: Some(DragonShardExportConfig {
            root: shard_root,
            dataset_name: Some("dragon-browser-net".into()),
            microshards: Some(2),
            max_records: Some(16),
            http_upstream: None,
        }),
        existing_shard_dataset: None,
    };
    let prepared = prepare_nca_native_cpu(&native, Some(&dummy_auth_bundle())).expect("peer");
    match prepared
        .project
        .data_pipeline_descriptor()
        .input_source
        .as_ref()
    {
        Some(WorkloadInputSource::Generated { descriptor }) => {
            assert_eq!(descriptor.provider, "burn_dragon_universality_nca");
        }
        other => panic!("expected generated input source, got {other:?}"),
    }
    let entry = prepared.manifests.experiment_directory[0].clone();
    let network_id = prepared.manifests.network_manifest.network_id.clone();
    let trainer_session = browser_conformance_session(
        network_id.clone(),
        PrincipalId::new("browser-trainer-principal"),
        entry.allowed_scopes.clone(),
    );
    assert!(
        !trainer_session
            .session
            .as_ref()
            .expect("trainer session")
            .claims
            .granted_scopes
            .contains(&ExperimentScope::Validate {
                experiment_id: entry.experiment_id.clone(),
            })
    );
    let verifier_session = browser_conformance_session(
        network_id.clone(),
        PrincipalId::new("browser-local-verifier-principal"),
        local_mock_verifier_scopes(&entry),
    );
    let mut harness = BrowserConformanceHarness::start(
        browser_runtime_for_edge(
            "https://edge.example",
            network_id.clone(),
            prepared
                .manifests
                .release_manifest
                .release_train_hash
                .clone(),
            prepared
                .manifests
                .release_manifest
                .target_artifact_hash
                .clone(),
            BrowserRuntimeRole::BrowserTrainerWgpu,
        ),
        browser_conformance_capability_for_role(BrowserRuntimeRole::BrowserTrainerWgpu),
        browser_conformance_transport(),
        browser_conformance_directory(network_id.clone(), vec![entry.clone()]),
        trainer_session,
    );
    harness.select_experiment(
        entry.experiment_id.clone(),
        Some(entry.current_revision_id.clone()),
    );
    harness.apply_heads(&[HeadDescriptor {
        head_id: burn_p2p::HeadId::new("dragon-head"),
        study_id: entry.study_id.clone(),
        experiment_id: entry.experiment_id.clone(),
        revision_id: entry.current_revision_id.clone(),
        artifact_id: burn_p2p::ArtifactId::new("dragon-artifact"),
        parent_head_id: None,
        global_step: 1,
        created_at: Utc::now(),
        metrics: Default::default(),
    }]);
    let training_lease = WorkloadTrainingLease {
        lease_id: LeaseId::new("dragon-browser-lease"),
        window_id: WindowId(1),
        dataset_view_id: entry.dataset_view_id.clone(),
        assignment_hash: ContentId::new("dragon-browser-assignment"),
        microshards: vec![MicroShardId::new("dragon-browser-shard-a")],
    };

    let training = harness
        .run_training(BrowserTrainingPlan {
            study_id: entry.study_id.clone(),
            experiment_id: entry.experiment_id.clone(),
            revision_id: entry.current_revision_id.clone(),
            workload_id: entry.workload_id.clone(),
            budget: BrowserTrainingBudget::default(),
            lease: Some(training_lease.clone()),
        })
        .expect("training");
    assert_eq!(harness.active_training_lease(), Some(&training_lease));
    let mut verifier = BrowserConformanceHarness::start(
        browser_runtime_for_edge(
            "https://edge.example",
            network_id.clone(),
            prepared
                .manifests
                .release_manifest
                .release_train_hash
                .clone(),
            prepared
                .manifests
                .release_manifest
                .target_artifact_hash
                .clone(),
            BrowserRuntimeRole::BrowserVerifier,
        ),
        browser_conformance_capability_for_role(BrowserRuntimeRole::BrowserVerifier),
        browser_conformance_transport(),
        browser_conformance_directory(network_id, vec![entry.clone()]),
        verifier_session.clone(),
    );
    assert!(
        verifier_session
            .session
            .as_ref()
            .expect("verifier session")
            .claims
            .granted_scopes
            .contains(&ExperimentScope::Validate {
                experiment_id: entry.experiment_id.clone(),
            })
    );
    verifier.select_experiment(
        entry.experiment_id.clone(),
        Some(entry.current_revision_id.clone()),
    );
    verifier.apply_heads(&[HeadDescriptor {
        head_id: burn_p2p::HeadId::new("dragon-head"),
        study_id: entry.study_id,
        experiment_id: entry.experiment_id.clone(),
        revision_id: entry.current_revision_id.clone(),
        artifact_id: burn_p2p::ArtifactId::new("dragon-artifact"),
        parent_head_id: None,
        global_step: 1,
        created_at: Utc::now(),
        metrics: Default::default(),
    }]);

    let validation = verifier
        .run_validation(BrowserValidationPlan {
            head_id: burn_p2p::HeadId::new("dragon-head"),
            max_checkpoint_bytes: 8 * 1024 * 1024,
            sample_budget: 4,
            emit_receipt: true,
        })
        .expect("validation");

    eprintln!(
        "browser_conformance: window_secs={} training_receipt={:?} validation_receipt={:?}",
        training.window_secs, training.receipt_id, validation.emitted_receipt_id
    );
    assert_eq!(training.window_secs, 30);
    assert!(training.receipt_id.is_some());
    assert!(validation.accepted);
}

#[test]
fn climbmix_http_shards_publish_http_input_source_descriptor() {
    let _guard = native_swarm_test_guard();
    let root = tempdir().expect("root");
    let shard_root = root.path().join("climbmix-http-shards");
    fs::create_dir_all(&shard_root).expect("mkdir shards");
    write_existing_climbmix_shards(&shard_root, 16, 8);
    let training_config_path = root.path().join("climbmix-train.toml");
    write(
        &training_config_path,
        &climbmix_training_config_toml(&root.path().join("climbmix-cache"), SMALL_SPEC),
    );

    let http_upstream = "https://datasets.example/climbmix";
    let native = DragonNativePeerConfig {
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-http-climbmix"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0-pre.22").expect("valid burn_dragon version"),
        git_commit: Some("http".into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export: None,
        existing_shard_dataset: Some(DragonExistingShardDatasetConfig {
            root: shard_root,
            http_upstream: Some(http_upstream.into()),
        }),
    };

    let prepared = prepare_climbmix_native_cpu(&native, Some(&dummy_auth_bundle())).expect("peer");
    let profile = DragonExperimentProfile::from_entry_metadata(
        prepared
            .manifests
            .experiment_directory
            .first()
            .expect("directory entry"),
    )
    .expect("profile decode")
    .expect("profile");
    match prepared
        .project
        .data_pipeline_descriptor()
        .input_source
        .as_ref()
    {
        Some(WorkloadInputSource::ShardManifestHttp {
            manifest_url,
            shard_count,
        }) => {
            assert_eq!(manifest_url, &shard_manifest_url(http_upstream));
            assert_eq!(*shard_count, Some(4));
        }
        other => panic!("expected shard-manifest http input source, got {other:?}"),
    }
    match profile.browser.expect("browser profile").train_source {
        DragonBrowserProfileTokenSource::ShardManifestHttp {
            manifest_url,
            selection,
            max_shards_per_window,
        } => {
            assert_eq!(
                manifest_url,
                "/dragon-datasets/climbmix-pretraining/r1/fetch-manifest.json"
            );
            assert_eq!(
                selection,
                burn_dragon_p2p::config::DragonBrowserShardSelectionPolicy::DeterministicPeer
            );
            assert_eq!(max_shards_per_window, Some(4));
        }
        other => panic!("expected browser shard-manifest source, got {other:?}"),
    }
}

#[test]
fn nca_mixed_fleet_browser_and_native_same_net_progresses() {
    let _guard = native_swarm_test_guard();
    let root = tempdir().expect("root");
    let nca_config_path = root.path().join("nca.toml");
    let training_config_path = root.path().join("nca-train.toml");
    let shard_root = root.path().join("nca-shards-mixed");
    write(&nca_config_path, &nca_corpus_config_toml(root.path()));
    write(
        &training_config_path,
        &nca_training_config_toml(&root.path().join("nca-cache"), &nca_config_path, SMALL_SPEC),
    );

    let native = DragonNativePeerConfig {
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-mixed-native"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0-pre.22").expect("valid burn_dragon version"),
        git_commit: Some("mixed".into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export: Some(DragonShardExportConfig {
            root: shard_root,
            dataset_name: Some("dragon-nca-mixed".into()),
            microshards: Some(4),
            max_records: Some(32),
            http_upstream: None,
        }),
        existing_shard_dataset: None,
    };
    let prepared = prepare_nca_native_cpu(&native, Some(&dummy_auth_bundle())).expect("peer");
    let entry = prepared.manifests.experiment_directory[0].clone();
    let (mut trainer, mut verifier) = local_browser_training_and_verification_pair(
        &entry,
        prepared
            .manifests
            .release_manifest
            .release_train_hash
            .clone(),
        prepared
            .manifests
            .release_manifest
            .target_artifact_hash
            .clone(),
        prepared.manifests.network_manifest.network_id.clone(),
    );
    trainer.select_experiment(
        entry.experiment_id.clone(),
        Some(entry.current_revision_id.clone()),
    );
    verifier.select_experiment(
        entry.experiment_id.clone(),
        Some(entry.current_revision_id.clone()),
    );

    let native_obs = run_training_windows_with_heads(&prepared, 3, "nca-mixed");
    let native_losses = native_obs.iter().map(|obs| obs.loss).collect::<Vec<_>>();
    let mut train_receipts = 0usize;
    let mut verify_receipts = 0usize;

    for obs in &native_obs {
        trainer.apply_heads(std::slice::from_ref(&obs.head));
        verifier.apply_heads(std::slice::from_ref(&obs.head));

        let training = trainer
            .run_training(BrowserTrainingPlan {
                study_id: entry.study_id.clone(),
                experiment_id: entry.experiment_id.clone(),
                revision_id: entry.current_revision_id.clone(),
                workload_id: entry.workload_id.clone(),
                budget: BrowserTrainingBudget::default(),
                lease: None,
            })
            .expect("browser training");
        assert_eq!(training.window_secs, 30);
        assert!(training.receipt_id.is_some());
        train_receipts += flush_and_ack_receipts(&mut trainer);

        let validation = verifier
            .run_validation(BrowserValidationPlan {
                head_id: obs.head.head_id.clone(),
                max_checkpoint_bytes: 8 * 1024 * 1024,
                sample_budget: 4,
                emit_receipt: true,
            })
            .expect("browser validation");
        assert!(validation.accepted);
        assert_eq!(validation.checked_chunks, 4);
        assert!(validation.emitted_receipt_id.is_some());
        verify_receipts += flush_and_ack_receipts(&mut verifier);
    }

    log_loss_series("nca_mixed_fleet_native", &native_losses);
    assert!(native_losses.iter().all(|loss| loss.is_finite()));
    assert!(native_losses.iter().copied().fold(f64::INFINITY, f64::min) <= native_losses[0] + 0.5);
    assert!(
        (1..=native_obs.len()).contains(&train_receipts),
        "browser training receipts should flush at least once and at most once per window"
    );
    assert!(
        (1..=native_obs.len()).contains(&verify_receipts),
        "browser validation receipts should flush at least once and at most once per window"
    );
}

#[test]
fn climbmix_mixed_fleet_browser_and_native_same_net_progresses() {
    let _guard = native_swarm_test_guard();
    let root = tempdir().expect("root");
    let shard_root = root.path().join("climbmix-shards-mixed");
    fs::create_dir_all(&shard_root).expect("mkdir shards");
    write_existing_climbmix_shards(&shard_root, 24, 8);
    let training_config_path = root.path().join("climbmix-train.toml");
    write(
        &training_config_path,
        &climbmix_training_config_toml(&root.path().join("climbmix-cache"), SMALL_SPEC),
    );

    let base_native = DragonNativePeerConfig {
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-peer-a"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0-pre.22").expect("valid burn_dragon version"),
        git_commit: Some("mixed".into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export: None,
        existing_shard_dataset: Some(DragonExistingShardDatasetConfig {
            root: shard_root.clone(),
            http_upstream: None,
        }),
    };
    let peer_a =
        prepare_climbmix_native_cpu(&base_native, Some(&dummy_auth_bundle())).expect("peer a");
    let mut peer_b_config = base_native.clone();
    peer_b_config.storage_root = root.path().join("storage-peer-b");
    let peer_b =
        prepare_climbmix_native_cpu(&peer_b_config, Some(&dummy_auth_bundle())).expect("peer b");
    let entry = peer_a.manifests.experiment_directory[0].clone();
    let (mut trainer, mut verifier) = local_browser_training_and_verification_pair(
        &entry,
        peer_a.manifests.release_manifest.release_train_hash.clone(),
        peer_a
            .manifests
            .release_manifest
            .target_artifact_hash
            .clone(),
        peer_a.manifests.network_manifest.network_id.clone(),
    );
    trainer.select_experiment(
        entry.experiment_id.clone(),
        Some(entry.current_revision_id.clone()),
    );
    verifier.select_experiment(
        entry.experiment_id.clone(),
        Some(entry.current_revision_id.clone()),
    );

    let obs_a = run_training_windows_with_heads(&peer_a, 2, "climbmix-peer-a");
    let obs_b = run_training_windows_with_heads(&peer_b, 2, "climbmix-peer-b");
    let ordered = [obs_a.as_slice(), obs_b.as_slice()]
        .into_iter()
        .flat_map(|slice| slice.iter())
        .cloned()
        .collect::<Vec<_>>();
    let losses_a = obs_a.iter().map(|obs| obs.loss).collect::<Vec<_>>();
    let losses_b = obs_b.iter().map(|obs| obs.loss).collect::<Vec<_>>();
    let mut train_receipts = 0usize;
    let mut verify_receipts = 0usize;

    for obs in &ordered {
        trainer.apply_heads(std::slice::from_ref(&obs.head));
        verifier.apply_heads(std::slice::from_ref(&obs.head));

        let training = trainer
            .run_training(BrowserTrainingPlan {
                study_id: entry.study_id.clone(),
                experiment_id: entry.experiment_id.clone(),
                revision_id: entry.current_revision_id.clone(),
                workload_id: entry.workload_id.clone(),
                budget: BrowserTrainingBudget::default(),
                lease: None,
            })
            .expect("browser training");
        assert_eq!(training.window_secs, 30);
        assert!(training.receipt_id.is_some());
        train_receipts += flush_and_ack_receipts(&mut trainer);

        let validation = verifier
            .run_validation(BrowserValidationPlan {
                head_id: obs.head.head_id.clone(),
                max_checkpoint_bytes: 8 * 1024 * 1024,
                sample_budget: 4,
                emit_receipt: true,
            })
            .expect("browser validation");
        assert!(validation.accepted);
        assert!(validation.emitted_receipt_id.is_some());
        verify_receipts += flush_and_ack_receipts(&mut verifier);
    }

    log_loss_series("climbmix_mixed_fleet_peer_a", &losses_a);
    log_loss_series("climbmix_mixed_fleet_peer_b", &losses_b);
    assert!(losses_a.iter().all(|loss| loss.is_finite()));
    assert!(losses_b.iter().all(|loss| loss.is_finite()));
    assert!(losses_a.iter().copied().fold(f64::INFINITY, f64::min) <= losses_a[0] + 0.5);
    assert!(losses_b.iter().copied().fold(f64::INFINITY, f64::min) <= losses_b[0] + 0.5);
    assert!(
        (1..=ordered.len()).contains(&train_receipts),
        "browser training receipts should flush at least once and at most once per window"
    );
    assert!(
        (1..=ordered.len()).contains(&verify_receipts),
        "browser validation receipts should flush at least once and at most once per window"
    );
}

#[test]
#[ignore = "covered by the explicit mixed-fleet medium validation rung"]
fn nca_mixed_fleet_browser_and_native_same_net_medium() {
    let _guard = native_swarm_test_guard();
    let root = tempdir().expect("root");
    let nca_config_path = root.path().join("nca.toml");
    let training_config_path = root.path().join("nca-train.toml");
    let shard_root = root.path().join("nca-shards-mixed-medium");
    write(&nca_config_path, &nca_corpus_config_toml(root.path()));
    write(
        &training_config_path,
        &nca_training_config_toml(
            &root.path().join("nca-cache"),
            &nca_config_path,
            MEDIUM_SPEC,
        ),
    );

    let native = DragonNativePeerConfig {
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-mixed-medium"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0-pre.22").expect("valid burn_dragon version"),
        git_commit: Some("mixed-medium".into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export: Some(DragonShardExportConfig {
            root: shard_root,
            dataset_name: Some("dragon-nca-mixed-medium".into()),
            microshards: Some(8),
            max_records: Some(96),
            http_upstream: None,
        }),
        existing_shard_dataset: None,
    };
    let prepared = prepare_nca_native_cpu(&native, Some(&dummy_auth_bundle())).expect("peer");
    let entry = prepared.manifests.experiment_directory[0].clone();
    let (mut trainer, mut verifier) = local_browser_training_and_verification_pair(
        &entry,
        prepared
            .manifests
            .release_manifest
            .release_train_hash
            .clone(),
        prepared
            .manifests
            .release_manifest
            .target_artifact_hash
            .clone(),
        prepared.manifests.network_manifest.network_id.clone(),
    );
    trainer.select_experiment(
        entry.experiment_id.clone(),
        Some(entry.current_revision_id.clone()),
    );
    verifier.select_experiment(
        entry.experiment_id.clone(),
        Some(entry.current_revision_id.clone()),
    );

    let native_obs = run_training_windows_with_heads(&prepared, 5, "nca-mixed-medium");
    let native_losses = native_obs.iter().map(|obs| obs.loss).collect::<Vec<_>>();
    let mut train_receipts = 0usize;
    let mut verify_receipts = 0usize;

    for obs in &native_obs {
        trainer.apply_heads(std::slice::from_ref(&obs.head));
        verifier.apply_heads(std::slice::from_ref(&obs.head));
        let training = trainer
            .run_training(BrowserTrainingPlan {
                study_id: entry.study_id.clone(),
                experiment_id: entry.experiment_id.clone(),
                revision_id: entry.current_revision_id.clone(),
                workload_id: entry.workload_id.clone(),
                budget: BrowserTrainingBudget::default(),
                lease: None,
            })
            .expect("browser training");
        assert!(training.receipt_id.is_some());
        train_receipts += flush_and_ack_receipts(&mut trainer);

        let validation = verifier
            .run_validation(BrowserValidationPlan {
                head_id: obs.head.head_id.clone(),
                max_checkpoint_bytes: 8 * 1024 * 1024,
                sample_budget: 6,
                emit_receipt: true,
            })
            .expect("browser validation");
        assert!(validation.accepted);
        assert!(validation.emitted_receipt_id.is_some());
        verify_receipts += flush_and_ack_receipts(&mut verifier);
    }

    log_loss_series("nca_mixed_fleet_medium_native", &native_losses);
    assert!(native_losses.iter().all(|loss| loss.is_finite()));
    assert!(
        native_losses.iter().copied().fold(f64::INFINITY, f64::min) <= native_losses[0] - 0.5,
        "mixed-fleet medium NCA should show a material best-window improvement"
    );
    assert!(
        (1..=native_obs.len()).contains(&train_receipts),
        "browser training receipts should flush at least once and at most once per window"
    );
    assert!(
        (1..=native_obs.len()).contains(&verify_receipts),
        "browser validation receipts should flush at least once and at most once per window"
    );
}

#[test]
#[ignore = "covered by the explicit mixed-fleet medium validation rung"]
fn climbmix_mixed_fleet_browser_and_native_three_peers_medium() {
    let _guard = native_swarm_test_guard();
    let root = tempdir().expect("root");
    let shard_root = root.path().join("climbmix-shards-mixed-medium");
    fs::create_dir_all(&shard_root).expect("mkdir shards");
    write_existing_climbmix_shards(&shard_root, 48, 16);
    let training_config_path = root.path().join("climbmix-train.toml");
    write(
        &training_config_path,
        &climbmix_training_config_toml(&root.path().join("climbmix-cache"), MEDIUM_SPEC),
    );

    let base_native = DragonNativePeerConfig {
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-peer-a"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0-pre.22").expect("valid burn_dragon version"),
        git_commit: Some("mixed-medium".into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export: None,
        existing_shard_dataset: Some(DragonExistingShardDatasetConfig {
            root: shard_root.clone(),
            http_upstream: None,
        }),
    };
    let peer_a =
        prepare_climbmix_native_cpu(&base_native, Some(&dummy_auth_bundle())).expect("peer a");
    let mut peer_b_config = base_native.clone();
    peer_b_config.storage_root = root.path().join("storage-peer-b");
    let peer_b =
        prepare_climbmix_native_cpu(&peer_b_config, Some(&dummy_auth_bundle())).expect("peer b");
    let mut peer_c_config = base_native.clone();
    peer_c_config.storage_root = root.path().join("storage-peer-c");
    let peer_c =
        prepare_climbmix_native_cpu(&peer_c_config, Some(&dummy_auth_bundle())).expect("peer c");

    let entry = peer_a.manifests.experiment_directory[0].clone();
    let (mut trainer, mut verifier) = local_browser_training_and_verification_pair(
        &entry,
        peer_a.manifests.release_manifest.release_train_hash.clone(),
        peer_a
            .manifests
            .release_manifest
            .target_artifact_hash
            .clone(),
        peer_a.manifests.network_manifest.network_id.clone(),
    );
    trainer.select_experiment(
        entry.experiment_id.clone(),
        Some(entry.current_revision_id.clone()),
    );
    verifier.select_experiment(
        entry.experiment_id.clone(),
        Some(entry.current_revision_id.clone()),
    );

    let obs_a = run_training_windows_with_heads(&peer_a, 2, "climbmix-medium-peer-a");
    let obs_b = run_training_windows_with_heads(&peer_b, 2, "climbmix-medium-peer-b");
    let obs_c = run_training_windows_with_heads(&peer_c, 2, "climbmix-medium-peer-c");
    let ordered = [obs_a.as_slice(), obs_b.as_slice(), obs_c.as_slice()]
        .into_iter()
        .flat_map(|slice| slice.iter())
        .cloned()
        .collect::<Vec<_>>();
    let losses_a = obs_a.iter().map(|obs| obs.loss).collect::<Vec<_>>();
    let losses_b = obs_b.iter().map(|obs| obs.loss).collect::<Vec<_>>();
    let losses_c = obs_c.iter().map(|obs| obs.loss).collect::<Vec<_>>();
    let mut train_receipts = 0usize;
    let mut verify_receipts = 0usize;

    for obs in &ordered {
        trainer.apply_heads(std::slice::from_ref(&obs.head));
        verifier.apply_heads(std::slice::from_ref(&obs.head));

        let training = trainer
            .run_training(BrowserTrainingPlan {
                study_id: entry.study_id.clone(),
                experiment_id: entry.experiment_id.clone(),
                revision_id: entry.current_revision_id.clone(),
                workload_id: entry.workload_id.clone(),
                budget: BrowserTrainingBudget::default(),
                lease: None,
            })
            .expect("browser training");
        assert!(training.receipt_id.is_some());
        train_receipts += flush_and_ack_receipts(&mut trainer);

        let validation = verifier
            .run_validation(BrowserValidationPlan {
                head_id: obs.head.head_id.clone(),
                max_checkpoint_bytes: 8 * 1024 * 1024,
                sample_budget: 6,
                emit_receipt: true,
            })
            .expect("browser validation");
        assert!(validation.accepted);
        assert!(validation.emitted_receipt_id.is_some());
        verify_receipts += flush_and_ack_receipts(&mut verifier);
    }

    log_loss_series("climbmix_mixed_fleet_medium_peer_a", &losses_a);
    log_loss_series("climbmix_mixed_fleet_medium_peer_b", &losses_b);
    log_loss_series("climbmix_mixed_fleet_medium_peer_c", &losses_c);
    assert!(losses_a.iter().all(|loss| loss.is_finite()));
    assert!(losses_b.iter().all(|loss| loss.is_finite()));
    assert!(losses_c.iter().all(|loss| loss.is_finite()));
    assert!(losses_a.iter().copied().fold(f64::INFINITY, f64::min) <= losses_a[0] + 0.5);
    assert!(losses_b.iter().copied().fold(f64::INFINITY, f64::min) <= losses_b[0] + 0.5);
    assert!(losses_c.iter().copied().fold(f64::INFINITY, f64::min) <= losses_c[0] + 0.5);
    assert!(
        (1..=ordered.len()).contains(&train_receipts),
        "browser training receipts should flush at least once and at most once per window"
    );
    assert!(
        (1..=ordered.len()).contains(&verify_receipts),
        "browser validation receipts should flush at least once and at most once per window"
    );
}

#[test]
#[ignore = "covered by the explicit native-scale validation rung"]
fn nca_native_peer_medium_model_converges_over_more_windows() {
    let _guard = native_swarm_test_guard();
    let root = tempdir().expect("root");
    let nca_config_path = root.path().join("nca.toml");
    let training_config_path = root.path().join("nca-train.toml");
    let shard_root = root.path().join("nca-shards-medium");
    write(&nca_config_path, &nca_corpus_config_toml(root.path()));
    write(
        &training_config_path,
        &nca_training_config_toml(
            &root.path().join("nca-cache"),
            &nca_config_path,
            MEDIUM_SPEC,
        ),
    );

    let native = DragonNativePeerConfig {
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-medium"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0-pre.22").expect("valid burn_dragon version"),
        git_commit: Some("scale".into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export: Some(DragonShardExportConfig {
            root: shard_root,
            dataset_name: Some("dragon-nca-medium".into()),
            microshards: Some(8),
            max_records: Some(96),
            http_upstream: None,
        }),
        existing_shard_dataset: None,
    };

    let prepared = prepare_nca_native_cpu(&native, Some(&dummy_auth_bundle())).expect("peer");
    let losses = run_training_windows(&prepared, 6);
    log_loss_series("nca_native_scale", &losses);
    assert!(losses.iter().all(|loss| loss.is_finite()));
    assert!(
        losses.iter().copied().fold(f64::INFINITY, f64::min) <= losses[0] - 0.5,
        "medium NCA rung should show a material best-window improvement"
    );
}

#[test]
#[ignore = "covered by the explicit native-scale validation rung"]
fn climbmix_native_three_peers_medium_model_stays_consistent() {
    let _guard = native_swarm_test_guard();
    let root = tempdir().expect("root");
    let shard_root = root.path().join("climbmix-shards-medium");
    fs::create_dir_all(&shard_root).expect("mkdir shards");
    write_existing_climbmix_shards(&shard_root, 48, 16);
    let training_config_path = root.path().join("climbmix-train.toml");
    write(
        &training_config_path,
        &climbmix_training_config_toml(&root.path().join("climbmix-cache"), MEDIUM_SPEC),
    );

    let base_native = DragonNativePeerConfig {
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-peer-a"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0-pre.22").expect("valid burn_dragon version"),
        git_commit: Some("scale".into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export: None,
        existing_shard_dataset: Some(DragonExistingShardDatasetConfig {
            root: shard_root.clone(),
            http_upstream: None,
        }),
    };
    let peer_a =
        prepare_climbmix_native_cpu(&base_native, Some(&dummy_auth_bundle())).expect("peer a");
    let mut peer_b_config = base_native.clone();
    peer_b_config.storage_root = root.path().join("storage-peer-b");
    let peer_b =
        prepare_climbmix_native_cpu(&peer_b_config, Some(&dummy_auth_bundle())).expect("peer b");
    let mut peer_c_config = base_native.clone();
    peer_c_config.storage_root = root.path().join("storage-peer-c");
    let peer_c =
        prepare_climbmix_native_cpu(&peer_c_config, Some(&dummy_auth_bundle())).expect("peer c");

    let losses_a = run_training_windows(&peer_a, 4);
    let losses_b = run_training_windows(&peer_b, 4);
    let losses_c = run_training_windows(&peer_c, 4);
    log_loss_series("climbmix_native_scale_peer_a", &losses_a);
    log_loss_series("climbmix_native_scale_peer_b", &losses_b);
    log_loss_series("climbmix_native_scale_peer_c", &losses_c);
    assert!(losses_a.iter().all(|loss| loss.is_finite()));
    assert!(losses_b.iter().all(|loss| loss.is_finite()));
    assert!(losses_c.iter().all(|loss| loss.is_finite()));
    assert!(losses_a.iter().copied().fold(f64::INFINITY, f64::min) <= losses_a[0] + 0.5);
    assert!(losses_b.iter().copied().fold(f64::INFINITY, f64::min) <= losses_b[0] + 0.5);
    assert!(losses_c.iter().copied().fold(f64::INFINITY, f64::min) <= losses_c[0] + 0.5);
}

#[test]
#[ignore = "covered by the explicit native-large validation rung"]
fn nca_native_peer_large_model_converges_over_more_windows() {
    let _guard = native_swarm_test_guard();
    let root = tempdir().expect("root");
    let nca_config_path = root.path().join("nca.toml");
    let training_config_path = root.path().join("nca-train.toml");
    let shard_root = root.path().join("nca-shards-large");
    write(&nca_config_path, &nca_corpus_config_toml(root.path()));
    write(
        &training_config_path,
        &nca_training_config_toml(&root.path().join("nca-cache"), &nca_config_path, LARGE_SPEC),
    );

    let native = DragonNativePeerConfig {
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-large"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0-pre.22").expect("valid burn_dragon version"),
        git_commit: Some("large".into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export: Some(DragonShardExportConfig {
            root: shard_root,
            dataset_name: Some("dragon-nca-large".into()),
            microshards: Some(8),
            max_records: Some(128),
            http_upstream: None,
        }),
        existing_shard_dataset: None,
    };

    let prepared = prepare_nca_native_cpu(&native, Some(&dummy_auth_bundle())).expect("peer");
    let losses = run_training_windows(&prepared, 8);
    log_loss_series("nca_native_large", &losses);
    assert!(losses.iter().all(|loss| loss.is_finite()));
    assert!(
        losses.iter().copied().fold(f64::INFINITY, f64::min) <= losses[0] - 0.5,
        "large NCA rung should show a material improvement over the initial window"
    );
}

#[test]
#[ignore = "covered by the explicit native-large validation rung"]
fn climbmix_native_three_peers_large_model_stays_consistent() {
    let _guard = native_swarm_test_guard();
    let root = tempdir().expect("root");
    let shard_root = root.path().join("climbmix-shards-large");
    fs::create_dir_all(&shard_root).expect("mkdir shards");
    write_existing_climbmix_shards(&shard_root, 64, 24);
    let training_config_path = root.path().join("climbmix-train.toml");
    write(
        &training_config_path,
        &climbmix_training_config_toml(&root.path().join("climbmix-cache"), LARGE_SPEC),
    );

    let base_native = DragonNativePeerConfig {
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-peer-a"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0-pre.22").expect("valid burn_dragon version"),
        git_commit: Some("large".into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export: None,
        existing_shard_dataset: Some(DragonExistingShardDatasetConfig {
            root: shard_root.clone(),
            http_upstream: None,
        }),
    };
    let peer_a =
        prepare_climbmix_native_cpu(&base_native, Some(&dummy_auth_bundle())).expect("peer a");
    let mut peer_b_config = base_native.clone();
    peer_b_config.storage_root = root.path().join("storage-peer-b");
    let peer_b =
        prepare_climbmix_native_cpu(&peer_b_config, Some(&dummy_auth_bundle())).expect("peer b");
    let mut peer_c_config = base_native.clone();
    peer_c_config.storage_root = root.path().join("storage-peer-c");
    let peer_c =
        prepare_climbmix_native_cpu(&peer_c_config, Some(&dummy_auth_bundle())).expect("peer c");

    let losses_a = run_training_windows(&peer_a, 5);
    let losses_b = run_training_windows(&peer_b, 5);
    let losses_c = run_training_windows(&peer_c, 5);
    log_loss_series("climbmix_native_large_peer_a", &losses_a);
    log_loss_series("climbmix_native_large_peer_b", &losses_b);
    log_loss_series("climbmix_native_large_peer_c", &losses_c);
    assert!(losses_a.iter().all(|loss| loss.is_finite()));
    assert!(losses_b.iter().all(|loss| loss.is_finite()));
    assert!(losses_c.iter().all(|loss| loss.is_finite()));
    assert!(losses_a.iter().copied().fold(f64::INFINITY, f64::min) <= losses_a[0] + 0.5);
    assert!(losses_b.iter().copied().fold(f64::INFINITY, f64::min) <= losses_b[0] + 0.5);
    assert!(losses_c.iter().copied().fold(f64::INFINITY, f64::min) <= losses_c[0] + 0.5);
}

#[test]
#[ignore = "covered by the explicit edge-drill validation rung"]
fn nca_edge_drill_native_and_browser_github_auth_and_receipts() {
    let _guard = native_swarm_test_guard();
    let root = tempdir().expect("root");
    let nca_config_path = root.path().join("nca.toml");
    let training_config_path = root.path().join("nca-train.toml");
    let shard_root = root.path().join("nca-shards-edge");
    write(&nca_config_path, &nca_corpus_config_toml(root.path()));
    write(
        &training_config_path,
        &nca_training_config_toml(&root.path().join("nca-cache"), &nca_config_path, SMALL_SPEC),
    );

    let native = DragonNativePeerConfig {
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-edge-native"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0-pre.22").expect("valid burn_dragon version"),
        git_commit: Some("edge-drill".into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export: Some(DragonShardExportConfig {
            root: shard_root,
            dataset_name: Some("dragon-nca-edge".into()),
            microshards: Some(4),
            max_records: Some(32),
            http_upstream: None,
        }),
        existing_shard_dataset: None,
    };
    let prepared = prepare_nca_native_cpu(&native, Some(&dummy_auth_bundle())).expect("peer");
    run_edge_drill_for_prepared(&prepared, "nca-edge");
}

#[test]
#[ignore = "covered by the explicit edge-drill validation rung"]
fn climbmix_edge_drill_native_and_browser_github_auth_and_receipts() {
    let _guard = native_swarm_test_guard();
    let root = tempdir().expect("root");
    let shard_root = root.path().join("climbmix-shards-edge");
    fs::create_dir_all(&shard_root).expect("mkdir shards");
    write_existing_climbmix_shards(&shard_root, 24, 8);
    let training_config_path = root.path().join("climbmix-train.toml");
    write(
        &training_config_path,
        &climbmix_training_config_toml(&root.path().join("climbmix-cache"), SMALL_SPEC),
    );

    let native = DragonNativePeerConfig {
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-edge-peer-a"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0-pre.22").expect("valid burn_dragon version"),
        git_commit: Some("edge-drill".into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export: None,
        existing_shard_dataset: Some(DragonExistingShardDatasetConfig {
            root: shard_root,
            http_upstream: None,
        }),
    };
    let prepared = prepare_climbmix_native_cpu(&native, Some(&dummy_auth_bundle())).expect("peer");
    run_edge_drill_for_prepared(&prepared, "climbmix-edge");
}
