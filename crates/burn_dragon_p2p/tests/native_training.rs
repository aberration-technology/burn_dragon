#![cfg(feature = "native")]
#![recursion_limit = "256"]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::net::TcpListener;
use std::panic::resume_unwind;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};

use serde::Deserialize;
use sha2::{Digest, Sha256};

use burn_dragon_p2p::auth::{
    begin_native_github_login, complete_native_github_login, fetch_edge_snapshot,
    load_cached_native_auth_bundle, native_auth_bundle_is_fresh, native_cli_bridge_url,
    refresh_native_auth_bundle, store_cached_native_auth_bundle,
};
use burn_dragon_p2p::capability::DragonCapabilityClass;
use burn_dragon_p2p::config::{
    DragonAggregationConfig, DragonCapabilityPolicy, DragonExistingShardDatasetConfig,
    DragonManifestSeed, DragonNativeAuthBundle, DragonNativePeerConfig, DragonNativeTarget,
    DragonPeerNetworkConfig, DragonShardExportConfig, TokenWindowRecord,
};
use burn_dragon_p2p::experiments::common::DragonBurnProject;
use burn_dragon_p2p::experiments::common::PreparedNativePeer;
#[cfg(feature = "cuda")]
use burn_dragon_p2p::native::prepare_nca_native_cuda;
use burn_dragon_p2p::native::{
    ManagedRunningNativePeer, NativeCpuBackend, prepare_climbmix_native_cpu,
    prepare_nca_native_cpu, spawn_prepared_native_peer,
};
use burn_dragon_p2p::profile::{DragonBrowserProfileTokenSource, DragonExperimentProfile};
use burn_p2p::burn::{
    BurnShardedDataset, BurnShardedDatasetConfig, BurnWorkload, BurnWorkloadAdapter,
};
use burn_p2p::{
    ArtifactKind, AssignmentLease, AuthConfig, AuthProvider, BaseCheckpointId, BrowserMode,
    CallbackPayload, ClientPlatform, ContentId, DatasetRegistration, DiLoCoPolicy,
    DiLoCoReferenceCoordinator, DiLoCoReferencePeer, DiLoCoWorkload, EdgePeerEnrollmentRequest,
    ExperimentDirectoryEntry, ExperimentDirectoryPolicyExt, ExperimentHandle, ExperimentScope,
    FsArtifactStore, GradientCodec, HeadDescriptor, HeadPromotionMode, LeaseId, LoginRequest,
    MODEL_GENESIS_SIGNATURE_KEY_ID, MergeModelCandidate, MergePolicy, MergeStrategy, MetricValue,
    MicroShardId, MicroShardPlan, ModelGenesisManifest, NodeCertificate, NodeCertificateClaims,
    OuterOptimizerPolicy, P2pWorkload, PeerId, PeerRole, PeerRoleSet, PrincipalClaims, PrincipalId,
    PrincipalSession, ProjectFamilyId, REVISION_CONTRACT_SIGNATURE_KEY_ID, RequestFailureOperation,
    RequestFailureReason, RevisionContractBundle, RevocationEpoch, RoundCursor, ShardCache,
    SignedPayload, TrainingProtocol, TrustedIssuer, WindowCtx, WindowId, WorkloadInputSource,
    WorkloadTrainingLease, sign_revision_contract_bundle, verify_revision_contract_bundle,
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
use burn_p2p_core::{SchemaEnvelope, SignatureAlgorithm, SignatureMetadata};
use chrono::Utc;
use libp2p_identity::Keypair;
use semver::Version;
use tempfile::tempdir;
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

use burn_dragon_universality::{
    RuliadCorpusConfig, RuliadSerializationConfig, RuliadSourceSelectionConfig,
    RuliadTokenizationConfig, compact_ruliad_families,
};

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

const MATCHED_512_SMALL_SPEC: SmokeModelSpec = SmokeModelSpec {
    n_layer: 2,
    n_embd: 64,
    n_head: 4,
    latent_total: 128,
    block_size: 512,
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

const RULIAD_PARITY_1M_SPEC: SmokeModelSpec = SmokeModelSpec {
    n_layer: 2,
    n_embd: 256,
    n_head: 4,
    latent_total: 1024,
    block_size: 64,
    batch_size: 4,
    max_iters: 108,
};

const TEST_WEBRTC_DIRECT_SEED: &str = "/dns4/edge.example/udp/443/webrtc-direct/certhash/uEiAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
const RULIAD_CONVERGENCE_WINDOWS_ENV: &str = "BURN_DRAGON_RULIAD_CONVERGENCE_WINDOWS";
const RULIAD_CONVERGENCE_MAX_SECONDS_ENV: &str = "BURN_DRAGON_RULIAD_CONVERGENCE_MAX_SECONDS";
const RULIAD_CONVERGENCE_MAX_ITERS_ENV: &str = "BURN_DRAGON_RULIAD_CONVERGENCE_MAX_ITERS";
const RULIAD_CONVERGENCE_ROOT_ENV: &str = "BURN_DRAGON_RULIAD_CONVERGENCE_ROOT";
const P2P_PARITY_SEED_ENV: &str = "BURN_DRAGON_P2P_PARITY_SEED";
const P2P_PARITY_ROUNDS_ENV: &str = "BURN_DRAGON_P2P_PARITY_ROUNDS";
const P2P_PARITY_REPORT_ROOT_ENV: &str = "BURN_DRAGON_P2P_PARITY_REPORT_ROOT";
const P2P_PARITY_REPLAY_ENV: &str = "BURN_DRAGON_P2P_PARITY_REPLAY";
const P2P_PARITY_SEQUENTIAL_ENV: &str = "BURN_DRAGON_P2P_PARITY_SEQUENTIAL";
const P2P_PARITY_SYNCHRONIZED_ENV: &str = "BURN_DRAGON_P2P_PARITY_SYNCHRONIZED";
const P2P_PARITY_ROOT_EMA_BASIS_POINTS_ENV: &str = "BURN_DRAGON_P2P_PARITY_ROOT_EMA_BASIS_POINTS";
const P2P_PARITY_LOCAL_STEPS_ENV: &str = "BURN_DRAGON_P2P_PARITY_LOCAL_STEPS";
const P2P_PARITY_SIGNED_CONTRACT_ENV: &str = "BURN_DRAGON_P2P_PARITY_SIGNED_CONTRACT";
const P2P_PARITY_RESTART_AFTER_ROUND_ENV: &str = "BURN_DRAGON_P2P_PARITY_RESTART_AFTER_ROUND";
const P2P_PARITY_MIN_SYNC_PROGRESS_RATIO_ENV: &str =
    "BURN_DRAGON_P2P_PARITY_MIN_SYNC_PROGRESS_RATIO";
const P2P_PARITY_REQUIRE_CONVERGENCE_ENV: &str = "BURN_DRAGON_P2P_PARITY_REQUIRE_CONVERGENCE";
const P2P_DILOCO_CODEC_ENV: &str = "BURN_DRAGON_P2P_DILOCO_CODEC";
const P2P_DILOCO_OUTER_LR_MICROS_ENV: &str = "BURN_DRAGON_P2P_DILOCO_OUTER_LR_MICROS";
const P2P_DILOCO_MOMENTUM_MICROS_ENV: &str = "BURN_DRAGON_P2P_DILOCO_MOMENTUM_MICROS";
const P2P_DILOCO_NESTEROV_ENV: &str = "BURN_DRAGON_P2P_DILOCO_NESTEROV";
const P2P_DILOCO_WEIGHT_DECAY_MICROS_ENV: &str = "BURN_DRAGON_P2P_DILOCO_WEIGHT_DECAY_MICROS";
const P2P_DILOCO_REPORT_ROOT_ENV: &str = "BURN_DRAGON_P2P_DILOCO_REPORT_ROOT";
const P2P_DILOCO_MATCHMAKING_TIMEOUT_MS_ENV: &str = "BURN_DRAGON_P2P_DILOCO_MATCHMAKING_TIMEOUT_MS";

fn run_with_large_stack(name: &'static str, test: impl FnOnce() + Send + 'static) {
    let handle = thread::Builder::new()
        .name(name.into())
        .stack_size(64 * 1024 * 1024)
        .spawn(test)
        .expect("spawn large-stack test thread");
    if let Err(payload) = handle.join() {
        resume_unwind(payload);
    }
}

fn positive_env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn positive_env_duration(name: &str) -> Option<Duration> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .map(Duration::from_secs)
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default)
}

fn env_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .and_then(|value| match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        })
        .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(default)
}

fn convergence_parity_passes(
    progress_ratio: Option<f64>,
    reference_loss_reduction: Option<f64>,
    minimum_progress_ratio: f64,
) -> bool {
    progress_ratio
        .zip(reference_loss_reduction)
        .is_some_and(|(ratio, reduction)| reduction > 0.0 && ratio >= minimum_progress_ratio)
}

fn is_diloco_request_operation(operation: &RequestFailureOperation) -> bool {
    matches!(
        operation,
        RequestFailureOperation::DiLoCoStateFetch
            | RequestFailureOperation::DiLoCoParameterStateFetch
            | RequestFailureOperation::DiLoCoGradientManifestFetch
            | RequestFailureOperation::DiLoCoGradientChunkFetch
            | RequestFailureOperation::DiLoCoAggregateChunkFetch
            | RequestFailureOperation::DiLoCoRoundRequest
    )
}

fn is_hard_request_failure_reason(reason: &RequestFailureReason) -> bool {
    !matches!(reason, RequestFailureReason::NotFound)
}

fn install_deterministic_test_identity(storage_root: &Path, seed: u64, role: &str) -> PeerId {
    let mut hasher = Sha256::new();
    hasher.update(b"burn-dragon-p2p-parity-identity-v1");
    hasher.update(seed.to_le_bytes());
    hasher.update(role.as_bytes());
    let secret: [u8; 32] = hasher.finalize().into();
    let keypair = Keypair::ed25519_from_bytes(secret).expect("derive deterministic test identity");
    let encoded = keypair
        .to_protobuf_encoding()
        .expect("encode deterministic test identity");
    let state_dir = storage_root.join("state");
    fs::create_dir_all(&state_dir).expect("create deterministic identity state directory");
    fs::write(state_dir.join("identity.key"), encoded)
        .expect("write deterministic persistent test identity");
    PeerId::new(libp2p_identity::PeerId::from_public_key(&keypair.public()).to_string())
}

fn diloco_policy_from_env(num_inner_steps: usize) -> DiLoCoPolicy {
    let codec = match std::env::var(P2P_DILOCO_CODEC_ENV)
        .unwrap_or_else(|_| "fp32".into())
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "fp32" => GradientCodec::Fp32,
        "fp16" => GradientCodec::Fp16,
        "int8" | "blockwise-int8" => GradientCodec::BlockwiseInt8 { block_size: 256 },
        value => panic!("unsupported {P2P_DILOCO_CODEC_ENV} value {value:?}"),
    };
    let momentum_micros = env_u64(P2P_DILOCO_MOMENTUM_MICROS_ENV, 0);
    let weight_decay_micros = env_u64(P2P_DILOCO_WEIGHT_DECAY_MICROS_ENV, 0);
    let policy = DiLoCoPolicy {
        num_inner_steps: u32::try_from(num_inner_steps).expect("inner steps fit in u32"),
        target_group_size: 3,
        minimum_group_size: 3,
        matchmaking_timeout_ms: u32::try_from(env_u64(
            P2P_DILOCO_MATCHMAKING_TIMEOUT_MS_ENV,
            60_000,
        ))
        .expect("DiLoCo matchmaking timeout fits in u32"),
        aggregation_timeout_ms: 60_000,
        checkpoint_interval_rounds: 1,
        codec,
        outer_optimizer_policy: OuterOptimizerPolicy::Sgd {
            learning_rate_micros: env_u64(P2P_DILOCO_OUTER_LR_MICROS_ENV, 1_000_000),
            momentum_micros: (momentum_micros > 0).then_some(momentum_micros),
            nesterov: env_bool(P2P_DILOCO_NESTEROV_ENV, false),
            weight_decay_micros: (weight_decay_micros > 0).then_some(weight_decay_micros),
        },
        ..DiLoCoPolicy::default()
    };
    policy.validate().expect("valid DiLoCo experiment policy");
    policy
}

fn diloco_policy_slug(policy: &DiLoCoPolicy) -> String {
    let codec = match &policy.codec {
        GradientCodec::Fp32 => "fp32".to_owned(),
        GradientCodec::Fp16 => "fp16".to_owned(),
        GradientCodec::BlockwiseInt8 { block_size } => format!("int8b{block_size}"),
        GradientCodec::Qsgd { bits, stochastic } => {
            format!(
                "qsgd{bits}-{}",
                if *stochastic { "stochastic" } else { "nearest" }
            )
        }
        GradientCodec::LowRank { rank } => format!("lowrank{rank}"),
        GradientCodec::SignSgd { error_feedback } => format!(
            "signsgd-{}",
            if *error_feedback { "feedback" } else { "plain" }
        ),
    };
    match &policy.outer_optimizer_policy {
        OuterOptimizerPolicy::Sgd {
            learning_rate_micros,
            momentum_micros,
            nesterov,
            weight_decay_micros,
        } => format!(
            "{codec}-lr{learning_rate_micros}-m{}-n{}-wd{}",
            momentum_micros.unwrap_or_default(),
            u8::from(*nesterov),
            weight_decay_micros.unwrap_or_default(),
        ),
    }
}

#[test]
fn convergence_parity_gate_requires_positive_reference_progress_and_threshold() {
    assert!(convergence_parity_passes(Some(0.90), Some(1.0), 0.90));
    assert!(!convergence_parity_passes(Some(0.89), Some(1.0), 0.90));
    assert!(!convergence_parity_passes(Some(1.0), Some(0.0), 0.90));
    assert!(!convergence_parity_passes(None, Some(1.0), 0.90));
}

#[test]
fn diloco_transport_gate_separates_poll_misses_from_hard_failures() {
    assert!(is_diloco_request_operation(
        &RequestFailureOperation::DiLoCoGradientChunkFetch
    ));
    assert!(!is_diloco_request_operation(
        &RequestFailureOperation::ArtifactChunkFetch
    ));
    assert!(!is_hard_request_failure_reason(
        &RequestFailureReason::NotFound
    ));
    assert!(is_hard_request_failure_reason(
        &RequestFailureReason::Timeout
    ));
    assert!(is_hard_request_failure_reason(
        &RequestFailureReason::UnexpectedResponse
    ));
}

#[test]
fn deterministic_parity_identities_are_stable_and_role_separated() {
    let first = tempdir().expect("first identity root");
    let second = tempdir().expect("second identity root");
    let third = tempdir().expect("third identity root");
    let seed_a = install_deterministic_test_identity(first.path(), 1337, "seed");
    let seed_b = install_deterministic_test_identity(second.path(), 1337, "seed");
    let trainer = install_deterministic_test_identity(third.path(), 1337, "trainer-b");

    assert_eq!(seed_a, seed_b);
    assert_ne!(seed_a, trainer);
    assert!(first.path().join("state/identity.key").is_file());
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn ruliad_convergence_root() -> (PathBuf, Option<tempfile::TempDir>) {
    if let Some(path) = env_path(RULIAD_CONVERGENCE_ROOT_ENV) {
        fs::create_dir_all(&path).expect("create ruliad convergence root");
        return (path, None);
    }
    let root = tempdir().expect("root");
    (root.path().to_path_buf(), Some(root))
}

fn dummy_auth_bundle() -> DragonNativeAuthBundle {
    DragonNativeAuthBundle {
        auth_config: AuthConfig::new(),
        trust_bundle_endpoint: "https://edge.example/trust-bundle".into(),
        edge_base_url: None,
        session_id: None,
        principal_id: None,
        enrollment: None,
        session: None,
        certificate_not_after: None,
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

[training.dynamics]
enabled = true
hard_collapse_rollback_attempts = 2
minimum_continual_backprop_scale = 0.25
soft_recovery_lr_scale = 0.5
validation_recovery_lr_scale = 0.5
hard_recovery_lr_scale = 0.25
soft_recovery_continual_backprop_scale = 2.0
validation_recovery_continual_backprop_scale = 1.25
hard_recovery_continual_backprop_scale = 1.25
stable_source_difficulty_pressure = 1.0
recovery_source_difficulty_pressure = 0.75
difficulty_advance_source_pressure = 1.5
stable_hash_noise_max_probability = 0.01
recovery_hash_noise_max_probability = 0.0

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

fn ruliad_corpus_config_toml(output_dir: &Path) -> String {
    let config = RuliadCorpusConfig {
        output_dir: output_dir.into(),
        seed: 1337,
        name: "dragon-p2p-ruliad-smoke".into(),
        train_samples: 32,
        validation_samples: 8,
        chunk_token_capacity: 4096,
        serialization: RuliadSerializationConfig {
            document_tokens: 513,
            preview_samples: 2,
            ..RuliadSerializationConfig::default()
        },
        tokenization: RuliadTokenizationConfig::default(),
        source_selection: RuliadSourceSelectionConfig {
            enabled: true,
            ..RuliadSourceSelectionConfig::default()
        },
        families: compact_ruliad_families(),
        proof_tasks: None,
        lean_task_limit: None,
    };
    toml::to_string_pretty(&config).expect("ruliad smoke corpus config")
}

fn ruliad_training_config_toml(
    cache_dir: &Path,
    ruliad_config_path: &Path,
    spec: SmokeModelSpec,
) -> String {
    format!(
        r#"
[dataset]
cache_dir = "{}"
train_split_ratio = 0.9
type = "universality_ruliad"
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
type = "standard_token_classification"

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
prompt = "[R2"
"#,
        cache_dir.display(),
        ruliad_config_path.display(),
        spec.n_layer,
        spec.n_embd,
        spec.n_head,
        spec.latent_total,
        spec.block_size,
        spec.batch_size,
        spec.max_iters,
    )
}

fn ruliad_parity_corpus_config_toml(output_dir: &Path, seed: u64) -> String {
    let config = RuliadCorpusConfig {
        output_dir: output_dir.into(),
        seed,
        name: format!("dragon-p2p-ruliad-parity-{seed}"),
        train_samples: 48,
        validation_samples: 12,
        chunk_token_capacity: 16 * 1024,
        serialization: RuliadSerializationConfig {
            document_tokens: 513,
            preview_samples: 2,
            ..RuliadSerializationConfig::default()
        },
        tokenization: RuliadTokenizationConfig::StructuredSymbolic {
            vocab_size: 272,
            eos_id: Some(271),
        },
        source_selection: RuliadSourceSelectionConfig {
            enabled: true,
            ..RuliadSourceSelectionConfig::default()
        },
        families: compact_ruliad_families(),
        proof_tasks: None,
        lean_task_limit: None,
    };
    toml::to_string_pretty(&config).expect("ruliad parity corpus config")
}

fn ruliad_parity_training_config_toml(
    cache_dir: &Path,
    ruliad_config_path: &Path,
    seed: u64,
    max_iters: usize,
    gradient_accumulation_steps: usize,
) -> String {
    let spec = RULIAD_PARITY_1M_SPEC;
    format!(
        r#"
[dataset]
cache_dir = "{}"
train_split_ratio = 0.8
type = "universality_ruliad"
config = "{}"

[dataset.tokenizer]
type = "pretokenized"
vocab_size = 272
eos_id = 271

[model]
n_layer = {}
n_embd = {}
n_head = {}
latent_total = {}
dropout = 0.0

[model.language_head]
type = "standard_token_classification"

[training]
block_size = {}
batch_size = {}
max_iters = {}
gradient_accumulation_steps = {}
checkpoint_interval_iters = 144
log_frequency = 1
seed = {}
tbptt_persist_across_steps = false

[training.ruliad_supervision]
mode = "full_document"

[training.continual_backprop]
enabled = false

[training.dynamics]
enabled = false

[optimizer]
name = "adamw"
learning_rate = 0.002
weight_decay = 0.0

[generation]
prompt = ""
"#,
        cache_dir.display(),
        ruliad_config_path.display(),
        spec.n_layer,
        spec.n_embd,
        spec.n_head,
        spec.latent_total,
        spec.block_size,
        spec.batch_size,
        max_iters,
        gradient_accumulation_steps,
        seed,
    )
}

fn write_ruliad_parity_training_config(root: &Path, seed: u64, max_iters: usize) -> PathBuf {
    let ruliad_config_path = root.join("ruliad-parity.toml");
    let training_config_path = root.join("ruliad-parity-training.toml");
    write(
        &ruliad_config_path,
        &ruliad_parity_corpus_config_toml(root, seed),
    );
    write(
        &training_config_path,
        &ruliad_parity_training_config_toml(
            &root.join("ruliad-parity-cache"),
            &ruliad_config_path,
            seed,
            max_iters,
            1,
        ),
    );
    training_config_path
}

fn write_ruliad_synchronized_reference_config(
    root: &Path,
    seed: u64,
    max_iters: usize,
    gradient_accumulation_steps: usize,
) -> PathBuf {
    let ruliad_config_path = root.join("ruliad-parity.toml");
    let training_config_path = root.join("ruliad-synchronized-reference-training.toml");
    write(
        &training_config_path,
        &ruliad_parity_training_config_toml(
            &root.join("ruliad-synchronized-reference-cache"),
            &ruliad_config_path,
            seed,
            max_iters,
            gradient_accumulation_steps,
        ),
    );
    training_config_path
}

fn write_nca_smoke_training_config(root: &Path, spec: SmokeModelSpec) -> PathBuf {
    let nca_config_path = root.join("nca.toml");
    let training_config_path = root.join("nca-train.toml");
    write(&nca_config_path, &nca_corpus_config_toml(root));
    write(
        &training_config_path,
        &nca_training_config_toml(&root.join("nca-cache"), &nca_config_path, spec),
    );
    training_config_path
}

fn write_ruliad_smoke_training_config(root: &Path, spec: SmokeModelSpec) -> PathBuf {
    let ruliad_config_path = root.join("ruliad.toml");
    let training_config_path = root.join("ruliad-train.toml");
    write(&ruliad_config_path, &ruliad_corpus_config_toml(root));
    write(
        &training_config_path,
        &ruliad_training_config_toml(&root.join("ruliad-cache"), &ruliad_config_path, spec),
    );
    training_config_path
}

fn native_smoke_peer_config(
    root: &Path,
    training_config_path: PathBuf,
    storage_name: &str,
    git_commit: &str,
    shard_export: Option<DragonShardExportConfig>,
) -> DragonNativePeerConfig {
    DragonNativePeerConfig {
        training_overrides: Default::default(),
        training_config_paths: vec![training_config_path],
        storage_root: root.join(storage_name),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0").expect("valid burn_dragon version"),
        git_commit: Some(git_commit.into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export,
        existing_shard_dataset: None,
    }
}

fn smoke_shard_export(
    root: &Path,
    shard_dir_name: &str,
    dataset_name: &str,
    microshards: u32,
    max_records: usize,
) -> DragonShardExportConfig {
    DragonShardExportConfig {
        root: root.join(shard_dir_name),
        dataset_name: Some(dataset_name.into()),
        microshards: Some(microshards),
        max_records: Some(max_records),
        http_upstream: None,
    }
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
                ..TokenWindowRecord::default()
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

fn optional_metric_float(
    stats: &std::collections::BTreeMap<String, MetricValue>,
    key: &str,
) -> Option<f64> {
    match stats.get(key) {
        Some(MetricValue::Float(value)) => Some(*value),
        Some(MetricValue::Integer(value)) => Some(*value as f64),
        Some(other) => panic!("expected numeric metric for {key}, got {other:?}"),
        None => None,
    }
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
    elapsed: Duration,
}

fn assert_ruliad_source_selection_metrics(
    label: &str,
    stats: &std::collections::BTreeMap<String, MetricValue>,
) {
    let entropy = metric_float(stats, "ruliad_source_selection_entropy_bits");
    let hash_noise = metric_float(stats, "ruliad_source_selection_hash_noise_probability");
    let mean_loss = metric_float(stats, "ruliad_source_selection_mean_loss");
    let progress = metric_float(stats, "ruliad_source_selection_mean_learning_progress");
    let verifier_failures = metric_integer(stats, "ruliad_source_selection_verifier_failures");
    assert!(entropy.is_finite(), "{label} entropy must be finite");
    assert!(
        (0.0..=1.0).contains(&hash_noise),
        "{label} hash-noise probability must be in [0, 1], got {hash_noise}"
    );
    assert!(
        mean_loss.is_finite(),
        "{label} source mean loss must be finite"
    );
    assert!(
        progress.is_finite(),
        "{label} source learning progress must be finite"
    );
    assert_eq!(
        verifier_failures, 0,
        "{label} source selection should not record verifier failures"
    );
}

fn best_loss(losses: &[f64]) -> f64 {
    losses.iter().copied().fold(f64::INFINITY, f64::min)
}

fn assert_material_best_improvement(label: &str, losses: &[f64]) {
    assert!(losses.len() >= 2, "{label} needs at least two windows");
    assert!(losses.iter().all(|loss| loss.is_finite()));
    let first = losses[0];
    let best = best_loss(losses);
    let absolute = first - best;
    let relative = if first.abs() <= f64::EPSILON {
        0.0
    } else {
        absolute / first.abs()
    };
    assert!(
        relative >= 0.05 || absolute >= 0.1,
        "{label} should improve by at least 5% or 0.1 absolute loss (first={first:.4}, best={best:.4}, relative={relative:.4})"
    );
}

fn observation_report_json(
    label: &str,
    spec: SmokeModelSpec,
    observations: &[NativeWindowObservation],
) -> serde_json::Value {
    let windows = observations
        .iter()
        .enumerate()
        .map(|(index, obs)| {
            let train_steps = metric_integer(&obs.head.metrics, "train_steps");
            let elapsed_secs = obs.elapsed.as_secs_f64();
            let tokens = train_steps.max(0) as f64 * spec.batch_size as f64 * spec.block_size as f64;
            serde_json::json!({
                "window": index + 1,
                "loss": obs.loss,
                "train_steps": train_steps,
                "elapsed_secs": elapsed_secs,
                "tokens_per_sec": if elapsed_secs > 0.0 { tokens / elapsed_secs } else { 0.0 },
                "source_selection_entropy_bits": optional_metric_float(&obs.head.metrics, "ruliad_source_selection_entropy_bits"),
                "source_selection_hash_noise_probability": optional_metric_float(&obs.head.metrics, "ruliad_source_selection_hash_noise_probability"),
                "source_selection_mean_loss": optional_metric_float(&obs.head.metrics, "ruliad_source_selection_mean_loss"),
                "source_selection_mean_learning_progress": optional_metric_float(&obs.head.metrics, "ruliad_source_selection_mean_learning_progress"),
                "source_selection_verifier_failures": optional_metric_float(&obs.head.metrics, "ruliad_source_selection_verifier_failures"),
            })
        })
        .collect::<Vec<_>>();
    let losses = observations.iter().map(|obs| obs.loss).collect::<Vec<_>>();
    let first = losses.first().copied().unwrap_or(f64::NAN);
    let best = best_loss(&losses);
    serde_json::json!({
        "label": label,
        "first_loss": first,
        "best_loss": best,
        "absolute_improvement": first - best,
        "relative_improvement": if first.is_finite() && first.abs() > f64::EPSILON {
            (first - best) / first.abs()
        } else {
            0.0
        },
        "windows": windows,
    })
}

fn local_browser_training_and_verification_pair(
    entry: &burn_p2p::ExperimentDirectoryEntry,
    release_train_hash: burn_p2p::ContentId,
    target_artifact_hash: burn_p2p::ContentId,
    network_id: burn_p2p::NetworkId,
) -> (BrowserConformanceHarness, BrowserConformanceHarness) {
    let trainer_scopes = local_mock_trainer_scopes(entry);
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
            site_seed_node_urls: vec![TEST_WEBRTC_DIRECT_SEED.into()],
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
            site_seed_node_urls: vec![TEST_WEBRTC_DIRECT_SEED.into()],
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

fn apply_canonical_browser_head(harness: &mut BrowserConformanceHarness, head: &HeadDescriptor) {
    let mut directory = harness.directory.clone();
    let mut updated = false;
    for entry in &mut directory.entries {
        if entry.study_id == head.study_id
            && entry.experiment_id == head.experiment_id
            && entry.current_revision_id == head.revision_id
        {
            entry.current_head_id = Some(head.head_id.clone());
            updated = true;
        }
    }
    assert!(
        updated,
        "canonical browser head must match a directory entry"
    );
    harness.update_directory(directory);
    assert_eq!(
        harness.apply_heads(std::slice::from_ref(head)),
        Some(head.head_id.clone())
    );
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
    run_training_windows_with_heads_until(prepared, windows, None, head_prefix, None)
}

fn run_training_windows_with_heads_until<B>(
    prepared: &burn_dragon_p2p::experiments::common::PreparedNativePeer<B>,
    max_windows: usize,
    max_elapsed: Option<Duration>,
    head_prefix: &str,
    progress_label: Option<&str>,
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
    let run_start = Instant::now();

    for window_ordinal in 0..max_windows {
        if window_ordinal > 0 && max_elapsed.is_some_and(|duration| run_start.elapsed() >= duration)
        {
            break;
        }
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
        let train_start = Instant::now();
        let report = project.train_window(&mut ctx).expect("train window");
        let elapsed = train_start.elapsed();
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
        observations.push(NativeWindowObservation {
            head,
            loss,
            elapsed,
        });
        if let Some(label) = progress_label {
            let obs = observations.last().expect("observation");
            let train_steps = metric_integer(&obs.head.metrics, "train_steps");
            let elapsed_secs = obs.elapsed.as_secs_f64();
            let tokens = train_steps.max(0) as f64
                * MATCHED_512_SMALL_SPEC.batch_size as f64
                * MATCHED_512_SMALL_SPEC.block_size as f64;
            eprintln!(
                "{label}_window_report={}",
                serde_json::to_string(&serde_json::json!({
                    "window": window_ordinal + 1,
                    "loss": obs.loss,
                    "train_steps": train_steps,
                    "elapsed_secs": elapsed_secs,
                    "tokens_per_sec": if elapsed_secs > 0.0 { tokens / elapsed_secs } else { 0.0 },
                    "run_elapsed_secs": run_start.elapsed().as_secs_f64(),
                    "source_selection_entropy_bits": optional_metric_float(&obs.head.metrics, "ruliad_source_selection_entropy_bits"),
                    "source_selection_hash_noise_probability": optional_metric_float(&obs.head.metrics, "ruliad_source_selection_hash_noise_probability"),
                    "source_selection_mean_loss": optional_metric_float(&obs.head.metrics, "ruliad_source_selection_mean_loss"),
                    "source_selection_mean_learning_progress": optional_metric_float(&obs.head.metrics, "ruliad_source_selection_mean_learning_progress"),
                    "source_selection_verifier_failures": optional_metric_float(&obs.head.metrics, "ruliad_source_selection_verifier_failures"),
                }))
                .expect("window report json")
            );
        }
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

type CpuParityProject = BurnWorkloadAdapter<DragonBurnProject<NativeCpuBackend>>;
type CpuParityModel = <CpuParityProject as P2pWorkload>::Model;
type CpuParityBatch = <CpuParityProject as P2pWorkload>::Batch;

struct ReferenceWindow {
    model: CpuParityModel,
    stats: BTreeMap<String, MetricValue>,
}

struct OracleCandidate {
    peer_id: PeerId,
    head_id: burn_p2p::HeadId,
    artifact_id: burn_p2p::ArtifactId,
    model: CpuParityModel,
    sample_weight: f64,
    quality_weight: f64,
    announced_at: chrono::DateTime<Utc>,
}

fn load_reference_lease_batches(
    project: &CpuParityProject,
    lease: &AssignmentLease,
    registration: &DatasetRegistration,
    microshard_plan: &MicroShardPlan,
    shard_cache: &ShardCache,
) -> Vec<CpuParityBatch> {
    let cached = shard_cache
        .fetch_lease_microshards(registration, microshard_plan, lease)
        .expect("cache reference lease");
    project
        .load_batches(lease, &cached)
        .expect("load reference lease batches")
}

fn train_reference_lease(
    project: &CpuParityProject,
    model: CpuParityModel,
    lease: &AssignmentLease,
    registration: &DatasetRegistration,
    microshard_plan: &MicroShardPlan,
    shard_cache: &ShardCache,
) -> ReferenceWindow {
    let cached = shard_cache
        .fetch_lease_microshards(registration, microshard_plan, lease)
        .expect("cache reference lease");
    let batches = project
        .load_batches(lease, &cached)
        .expect("load reference lease batches");
    let mut ctx = WindowCtx {
        device: project.runtime_device(),
        model,
        lease: lease.clone(),
        cached_microshards: cached,
        batches,
    };
    let report = project
        .train_window(&mut ctx)
        .expect("train reference lease");
    ReferenceWindow {
        model: ctx.model,
        stats: report.stats,
    }
}

#[allow(clippy::too_many_arguments)]
fn train_synchronized_reference_round(
    batch_project: &CpuParityProject,
    synchronized_project: &CpuParityProject,
    model: CpuParityModel,
    leases: &[&AssignmentLease],
    registration: &DatasetRegistration,
    microshard_plan: &MicroShardPlan,
    shard_cache: &ShardCache,
    expected_batches_per_peer: usize,
) -> ReferenceWindow {
    let peer_batches = leases
        .iter()
        .map(|lease| {
            load_reference_lease_batches(
                batch_project,
                lease,
                registration,
                microshard_plan,
                shard_cache,
            )
        })
        .collect::<Vec<_>>();
    assert!(
        peer_batches
            .iter()
            .all(|batches| batches.len() == expected_batches_per_peer),
        "synchronized reference requires the exact bounded batch count from every peer"
    );

    let mut batches = Vec::with_capacity(expected_batches_per_peer * leases.len());
    for step in 0..expected_batches_per_peer {
        for peer in &peer_batches {
            batches.push(peer[step].clone());
        }
    }

    let microshard_ids = leases
        .iter()
        .flat_map(|lease| lease.microshards.iter().cloned())
        .collect::<BTreeSet<_>>();
    let selected_microshards = microshard_plan
        .microshards
        .iter()
        .filter(|microshard| microshard_ids.contains(&microshard.microshard_id))
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(
        selected_microshards.len(),
        microshard_ids.len(),
        "synchronized reference must resolve every peer microshard"
    );
    let first = leases.first().expect("at least one synchronized lease");
    let mut planner = burn_p2p::LeasePlanner::default();
    planner.config.max_microshards_per_lease = selected_microshards.len().max(1);
    let synchronized_lease = planner
        .plan_lease(
            first.network_id.clone(),
            first.study_id.clone(),
            first.experiment_id.clone(),
            first.revision_id.clone(),
            &registration.view,
            PeerId::new("central-synchronized-reference"),
            first.window_id,
            Utc::now(),
            leases
                .iter()
                .map(|lease| lease.budget_work_units)
                .sum::<u64>()
                .max(1),
            &selected_microshards,
        )
        .expect("plan synchronized reference lease")
        .lease;
    let cached = shard_cache
        .fetch_lease_microshards(registration, microshard_plan, &synchronized_lease)
        .expect("cache synchronized reference lease");
    let mut ctx = WindowCtx {
        device: synchronized_project.runtime_device(),
        model,
        lease: synchronized_lease,
        cached_microshards: cached,
        batches,
    };
    let report = synchronized_project
        .train_window(&mut ctx)
        .expect("train synchronized reference round");
    ReferenceWindow {
        model: ctx.model,
        stats: report.stats,
    }
}

fn round_trip_reference_model(
    project: &CpuParityProject,
    model: &CpuParityModel,
    store: &FsArtifactStore,
    label: &str,
    parent_head_id: Option<burn_p2p::HeadId>,
) -> CpuParityModel {
    let descriptor = project
        .materialize_model_artifact(
            model,
            ArtifactKind::FullHead,
            burn_p2p::HeadId::new(label),
            parent_head_id,
            store,
        )
        .expect("materialize reference model");
    project
        .load_model_artifact(
            project.init_model(&project.runtime_device()),
            &descriptor,
            store,
            &project.runtime_device(),
        )
        .expect("reload reference model")
}

fn model_digest(project: &CpuParityProject, model: &CpuParityModel) -> ContentId {
    project
        .model_tensor_digest(model)
        .expect("compute model tensor digest")
}

fn signed_revision_contract_fixture(
    prepared: &PreparedNativePeer<NativeCpuBackend>,
    authority: &Keypair,
    initialization_seed: u64,
) -> (RevisionContractBundle, String) {
    let project = BurnWorkloadAdapter::try_new(
        prepared.project.clone(),
        prepared.manifests.workload_config.clone(),
    )
    .expect("build signed-contract workload adapter");
    let device = project.runtime_device();
    let initialized_model = project.init_model(&device);
    let store = FsArtifactStore::new(&prepared.storage_root);
    let head_id = burn_p2p::HeadId::new(format!(
        "{}-{}-genesis",
        prepared.manifests.revision_manifest.experiment_id.as_str(),
        prepared.manifests.revision_manifest.revision_id.as_str(),
    ));
    let artifact = project
        .materialize_model_artifact(
            &initialized_model,
            ArtifactKind::FullHead,
            head_id,
            None,
            &store,
        )
        .expect("materialize signed genesis");
    let canonical_model = project
        .load_model_artifact(initialized_model, &artifact, &store, &device)
        .expect("reload signed genesis");
    let tensor_digest = project
        .model_tensor_digest(&canonical_model)
        .expect("digest signed genesis");
    let created_at = Utc::now();
    let genesis = ModelGenesisManifest {
        experiment_id: prepared.manifests.revision_manifest.experiment_id.clone(),
        revision_id: prepared.manifests.revision_manifest.revision_id.clone(),
        workload_id: prepared.manifests.revision_manifest.workload_id.clone(),
        training_contract_id: prepared.manifests.training_contract_id.clone(),
        artifact,
        tensor_digest,
        initialization_algorithm: "burn-dragon-deterministic-init-v1".into(),
        initialization_seed: Some(initialization_seed),
        authority_epoch: 1,
        created_at,
    };
    let mut bundle = RevisionContractBundle {
        revision: prepared.manifests.revision_manifest.clone(),
        training_contract_id: prepared.manifests.training_contract_id.clone(),
        training: prepared.manifests.training_contract.clone(),
        genesis: SignedPayload::new(
            SchemaEnvelope::new("burn-p2p-model-genesis-v1", Version::new(0, 21, 0), genesis),
            SignatureMetadata {
                signer: PeerId::new("unsigned"),
                key_id: MODEL_GENESIS_SIGNATURE_KEY_ID.into(),
                algorithm: SignatureAlgorithm::Ed25519,
                signed_at: created_at,
                signature_hex: "00".into(),
            },
        )
        .expect("placeholder signed genesis"),
        contract_signature: SignatureMetadata {
            signer: PeerId::new("unsigned"),
            key_id: REVISION_CONTRACT_SIGNATURE_KEY_ID.into(),
            algorithm: SignatureAlgorithm::Ed25519,
            signed_at: created_at,
            signature_hex: "00".into(),
        },
    };
    sign_revision_contract_bundle(authority, &mut bundle, created_at)
        .expect("sign Dragon revision contract");

    let signer =
        PeerId::new(libp2p_identity::PeerId::from_public_key(&authority.public()).to_string());
    let trusted_issuers = BTreeMap::from([(
        signer.clone(),
        TrustedIssuer {
            issuer_peer_id: signer,
            issuer_public_key_hex: hex::encode(authority.public().encode_protobuf()),
        },
    )]);
    verify_revision_contract_bundle(&trusted_issuers, &bundle)
        .expect("verify Dragon revision contract");
    (bundle, hex::encode(authority.public().encode_protobuf()))
}

fn validation_loss(project: &CpuParityProject, model: &CpuParityModel) -> f64 {
    metric_float_any(
        &project
            .evaluate(model, burn_p2p::EvalSplit::Validation)
            .metrics,
        &["loss", "validation_loss"],
    )
}

fn update_quality_weight(stats: &BTreeMap<String, MetricValue>) -> f64 {
    let quality_metric = stats
        .get("loss")
        .or_else(|| stats.get("train_loss"))
        .and_then(|value| match value {
            MetricValue::Float(value) => Some(*value),
            MetricValue::Integer(value) => Some(*value as f64),
            _ => None,
        })
        .unwrap_or_default();
    (1.0 / (1.0 + quality_metric.abs())).max(0.01)
}

fn run_three_peer_diloco_round<B>(
    experiment: &ExperimentHandle,
    seed: &mut ManagedRunningNativePeer<B>,
    trainer_b: &mut ManagedRunningNativePeer<B>,
    trainer_c: &mut ManagedRunningNativePeer<B>,
) -> ([burn_p2p::DiLoCoRoundOutcome; 3], Duration, Duration)
where
    B: burn::tensor::backend::AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let preparation_started = Instant::now();
    let seed_prepared = seed
        .prepare_diloco_round(experiment)
        .expect("prepare seed DiLoCo round");
    wait_for_lease_visibility("seed lease at trainer b", trainer_b, seed_prepared.lease());
    wait_for_lease_visibility("seed lease at trainer c", trainer_c, seed_prepared.lease());
    let trainer_b_prepared = trainer_b
        .prepare_diloco_round(experiment)
        .expect("prepare trainer b DiLoCo round");
    wait_for_lease_visibility("trainer b lease at seed", seed, trainer_b_prepared.lease());
    wait_for_lease_visibility(
        "trainer b lease at trainer c",
        trainer_c,
        trainer_b_prepared.lease(),
    );
    let trainer_c_prepared = trainer_c
        .prepare_diloco_round(experiment)
        .expect("prepare trainer c DiLoCo round");
    assert!(
        [&seed_prepared, &trainer_b_prepared, &trainer_c_prepared]
            .iter()
            .all(|prepared| prepared.batch_count() > 0),
        "every DiLoCo participant must materialize local batches"
    );
    let preparation_elapsed = preparation_started.elapsed();

    let start_barrier = Arc::new(std::sync::Barrier::new(3));
    let collective_started = Instant::now();
    let experiment_for_seed = experiment.clone();
    let experiment_for_b = experiment.clone();
    let experiment_for_c = experiment.clone();
    let seed_ref = &mut *seed;
    let trainer_b_ref = &mut *trainer_b;
    let trainer_c_ref = &mut *trainer_c;
    thread::scope(|scope| {
        let seed_barrier = Arc::clone(&start_barrier);
        let seed_run = scope.spawn(move || {
            seed_barrier.wait();
            assert_eq!(seed_prepared.experiment(), &experiment_for_seed);
            seed_ref
                .execute_prepared_diloco_round(seed_prepared)
                .map_err(|error| format_diloco_round_error("seed", error, seed_ref.snapshot()))
        });
        let trainer_b_barrier = Arc::clone(&start_barrier);
        let trainer_b_run = scope.spawn(move || {
            trainer_b_barrier.wait();
            assert_eq!(trainer_b_prepared.experiment(), &experiment_for_b);
            trainer_b_ref
                .execute_prepared_diloco_round(trainer_b_prepared)
                .map_err(|error| {
                    format_diloco_round_error("trainer-b", error, trainer_b_ref.snapshot())
                })
        });
        let trainer_c_barrier = Arc::clone(&start_barrier);
        let trainer_c_run = scope.spawn(move || {
            trainer_c_barrier.wait();
            assert_eq!(trainer_c_prepared.experiment(), &experiment_for_c);
            trainer_c_ref
                .execute_prepared_diloco_round(trainer_c_prepared)
                .map_err(|error| {
                    format_diloco_round_error("trainer-c", error, trainer_c_ref.snapshot())
                })
        });
        let results = [
            (
                "seed",
                seed_run
                    .join()
                    .map_err(|_| "round thread panicked".to_owned())
                    .and_then(|result| result),
            ),
            (
                "trainer-b",
                trainer_b_run
                    .join()
                    .map_err(|_| "round thread panicked".to_owned())
                    .and_then(|result| result),
            ),
            (
                "trainer-c",
                trainer_c_run
                    .join()
                    .map_err(|_| "round thread panicked".to_owned())
                    .and_then(|result| result),
            ),
        ];
        let errors = results
            .iter()
            .filter_map(|(label, result)| {
                result
                    .as_ref()
                    .err()
                    .map(|error| format!("{label}: {error}"))
            })
            .collect::<Vec<_>>();
        assert!(
            errors.is_empty(),
            "three-peer DiLoCo round failed:\n{}",
            errors.join("\n")
        );
        let mut outcomes = results
            .into_iter()
            .map(|(_, result)| result.expect("checked successful DiLoCo result"));
        let outcomes = [
            outcomes.next().expect("seed outcome"),
            outcomes.next().expect("trainer b outcome"),
            outcomes.next().expect("trainer c outcome"),
        ];
        (outcomes, preparation_elapsed, collective_started.elapsed())
    })
}

fn format_diloco_round_error(
    label: &str,
    error: anyhow::Error,
    snapshot: burn_p2p::NodeTelemetrySnapshot,
) -> String {
    let recent_events = snapshot
        .recent_events
        .iter()
        .rev()
        .filter(|event| {
            matches!(
                event,
                burn_p2p::LiveControlPlaneEvent::RequestFailure { .. }
                    | burn_p2p::LiveControlPlaneEvent::ResponseSendFailure { .. }
                    | burn_p2p::LiveControlPlaneEvent::ConnectionClosed { .. }
                    | burn_p2p::LiveControlPlaneEvent::OutgoingConnectionError { .. }
            ) || matches!(
                event,
                burn_p2p::LiveControlPlaneEvent::Other { kind }
                    if kind.contains("DiLoCo")
                        || kind.contains("connection-closed-detail")
                        || kind.contains("connection-established-detail")
            )
        })
        .take(16)
        .collect::<Vec<_>>();
    format!(
        "{error:#}; label={label}; local_peer={:?}; connected={:?}; request_failures={:?}; recent_events={:?}",
        snapshot.local_peer_id.as_ref().map(PeerId::as_str),
        snapshot
            .connected_peer_ids
            .iter()
            .map(PeerId::as_str)
            .collect::<Vec<_>>(),
        snapshot.request_failures,
        recent_events,
    )
}

fn wait_for_lease_visibility<B>(
    label: &str,
    peer: &ManagedRunningNativePeer<B>,
    lease: &AssignmentLease,
) where
    B: burn::tensor::backend::AutodiffBackend + Clone + 'static,
{
    wait_for(
        Duration::from_secs(30),
        || {
            peer.snapshot()
                .control_plane
                .lease_announcements
                .iter()
                .any(|announcement| announcement.lease.lease_id == lease.lease_id)
        },
        &format!("{label} was not visible before collective execution"),
    );
}

fn ensure_three_peer_full_mesh<B>(
    seed: &ManagedRunningNativePeer<B>,
    trainer_b: &ManagedRunningNativePeer<B>,
    trainer_c: &ManagedRunningNativePeer<B>,
) where
    B: burn::tensor::backend::AutodiffBackend + Clone + 'static,
{
    let peers = [seed, trainer_b, trainer_c];
    let addresses = peers
        .iter()
        .map(|peer| {
            peer.snapshot()
                .listen_addresses
                .into_iter()
                .next()
                .expect("native trainer must expose a listen address")
        })
        .collect::<Vec<_>>();
    let peer_ids = peers
        .iter()
        .map(|peer| {
            peer.snapshot()
                .local_peer_id
                .expect("native trainer peer id")
        })
        .collect::<Vec<_>>();
    for (left, address) in addresses.iter().enumerate() {
        for peer in peers.iter().skip(left + 1) {
            if !peer.snapshot().connected_peer_ids.contains(&peer_ids[left]) {
                peer.control_handle()
                    .dial_address(address.clone())
                    .expect("request deterministic trainer-mesh dial");
            }
        }
    }
    for (index, (label, peer)) in [
        ("seed", seed),
        ("trainer-b", trainer_b),
        ("trainer-c", trainer_c),
    ]
    .into_iter()
    .enumerate()
    {
        let expected_trainer_peers = peer_ids
            .iter()
            .enumerate()
            .filter(|(peer_index, _)| *peer_index != index)
            .map(|(_, peer_id)| peer_id.clone())
            .collect::<BTreeSet<_>>();
        wait_for(
            Duration::from_secs(30),
            || {
                let connected = peer.snapshot().connected_peer_ids;
                expected_trainer_peers.is_subset(&connected)
            },
            &format!("DiLoCo {label} did not establish a full trainer mesh"),
        );
        let snapshot = peer.snapshot();
        assert_eq!(
            snapshot
                .runtime_boundary
                .as_ref()
                .and_then(|boundary| boundary.transport_policy.max_established_per_peer),
            Some(1),
            "DiLoCo {label} must constrain each trainer pair to one request route"
        );
        if std::env::var_os("BURN_P2P_DILOCO_TRACE").is_some() {
            let route_events = snapshot
                .recent_events
                .iter()
                .filter(|event| {
                    matches!(
                        event,
                        burn_p2p::LiveControlPlaneEvent::Other { kind }
                            if kind.contains("connection-established-detail")
                                || kind.contains("connection-closed-detail")
                    )
                })
                .collect::<Vec<_>>();
            eprintln!("diloco-route-state label={label} events={route_events:?}");
        }
    }
}

fn converge_three_peer_diffusion_round<B>(
    experiment: &burn_p2p::ExperimentHandle,
    seed: &mut ManagedRunningNativePeer<B>,
    trainer_b: &mut ManagedRunningNativePeer<B>,
    trainer_c: &mut ManagedRunningNativePeer<B>,
    base_head: &HeadDescriptor,
) -> HeadDescriptor
where
    B: burn::tensor::backend::AutodiffBackend + Clone + 'static,
{
    let base_head_id = &base_head.head_id;
    let expected_global_step = base_head.global_step + 1;
    let convergence_deadline = Instant::now() + Duration::from_secs(180);
    let promoted_head = loop {
        advance_diffusion_with_retry(
            "advance parity seed diffusion",
            convergence_deadline,
            || seed.advance_diffusion_steady_state(experiment, None, None),
        );
        advance_diffusion_with_retry(
            "advance parity trainer b diffusion",
            convergence_deadline,
            || trainer_b.advance_diffusion_steady_state(experiment, None, None),
        );
        advance_diffusion_with_retry(
            "advance parity trainer c diffusion",
            convergence_deadline,
            || trainer_c.advance_diffusion_steady_state(experiment, None, None),
        );

        let seed_head = sync_experiment_head_with_retry(
            "sync parity seed head",
            seed,
            experiment,
            convergence_deadline,
        );
        let trainer_b_head = sync_experiment_head_with_retry(
            "sync parity trainer b head",
            trainer_b,
            experiment,
            convergence_deadline,
        );
        let trainer_c_head = sync_experiment_head_with_retry(
            "sync parity trainer c head",
            trainer_c,
            experiment,
            convergence_deadline,
        );
        if let Some(candidate) = select_promoted_head_candidate(
            [&seed_head, &trainer_b_head, &trainer_c_head],
            base_head_id,
            expected_global_step,
        ) {
            break candidate;
        }
        assert!(
            Instant::now() < convergence_deadline,
            "parity swarm did not promote a head; seed={} trainer-b={} trainer-c={}",
            describe_head_state(&seed_head),
            describe_head_state(&trainer_b_head),
            describe_head_state(&trainer_c_head),
        );
        thread::sleep(Duration::from_millis(25));
    };

    let propagation_deadline = Instant::now() + Duration::from_secs(90);
    loop {
        advance_diffusion_with_retry("propagate parity seed", propagation_deadline, || {
            seed.advance_diffusion_steady_state(experiment, None, None)
        });
        advance_diffusion_with_retry("propagate parity trainer b", propagation_deadline, || {
            trainer_b.advance_diffusion_steady_state(experiment, None, None)
        });
        advance_diffusion_with_retry("propagate parity trainer c", propagation_deadline, || {
            trainer_c.advance_diffusion_steady_state(experiment, None, None)
        });

        let seed_head = sync_experiment_head_with_retry(
            "sync propagated parity seed",
            seed,
            experiment,
            propagation_deadline,
        );
        let trainer_b_head = sync_experiment_head_with_retry(
            "sync propagated parity trainer b",
            trainer_b,
            experiment,
            propagation_deadline,
        );
        let trainer_c_head = sync_experiment_head_with_retry(
            "sync propagated parity trainer c",
            trainer_c,
            experiment,
            propagation_deadline,
        );
        if peers_have_promoted_head(
            [&seed_head, &trainer_b_head, &trainer_c_head],
            &promoted_head,
            base_head_id,
            expected_global_step,
        ) {
            return promoted_head;
        }
        assert!(
            Instant::now() < propagation_deadline,
            "parity swarm did not converge on promoted head {}; seed={} trainer-b={} trainer-c={}",
            promoted_head.head_id.as_str(),
            describe_head_state(&seed_head),
            describe_head_state(&trainer_b_head),
            describe_head_state(&trainer_c_head),
        );
        thread::sleep(Duration::from_millis(25));
    }
}

#[derive(Default)]
struct MockEdgeState {
    authorized_directory_fetches: usize,
    unauthorized_directory_fetches: usize,
    receipt_submission_batches: usize,
    refresh_requests: usize,
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
    entry
        .allowed_scopes
        .iter()
        .filter(|scope| !matches!(scope, ExperimentScope::Train { .. }))
        .cloned()
        .collect()
}

fn local_mock_trainer_scopes(entry: &ExperimentDirectoryEntry) -> BTreeSet<ExperimentScope> {
    entry
        .allowed_scopes
        .iter()
        .filter(|scope| !matches!(scope, ExperimentScope::Validate { .. }))
        .cloned()
        .collect()
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
        protocol_major: manifests.network_manifest.protocol_major,
        minimum_client_version: semver::Version::new(0, 0, 0),
        edge_mode: BrowserEdgeMode::Peer,
        browser_mode,
        social_mode: burn_p2p::SocialMode::Disabled,
        profile_mode: burn_p2p::ProfileMode::Disabled,
        transports: BrowserTransportSurface {
            webrtc_direct: true,
            webtransport_gateway: false,
            wss_fallback: false,
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
        revision_contracts: Vec::new(),
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
            protocol_major: manifests.network_manifest.protocol_major,
            minimum_client_version: semver::Version::new(0, 0, 0),
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
                            authorize_url: Some("https://github.example/authorize?redirect_uri=https%3A%2F%2Fdragon.example%2Fcallback%2Fgithub".into()),
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
                (&Method::Post, "/refresh") => {
                    #[derive(Deserialize)]
                    struct RefreshRequest {
                        session_id: ContentId,
                    }

                    let refresh: RefreshRequest = read_json(&mut request);
                    let mut state = state_for_thread.lock().expect("state");
                    let Some(session) = state.sessions.get(refresh.session_id.as_str()).cloned()
                    else {
                        respond_text(request, 401, r#"{"error":"unknown session"}"#);
                        continue;
                    };
                    let refreshed = PrincipalSession {
                        session_id: ContentId::new(format!(
                            "refreshed-{}-{}",
                            session.session_id.as_str(),
                            state.refresh_requests + 1
                        )),
                        network_id: session.network_id.clone(),
                        claims: PrincipalClaims {
                            expires_at: Utc::now() + chrono::Duration::minutes(30),
                            issued_at: Utc::now(),
                            ..session.claims.clone()
                        },
                        issued_at: Utc::now(),
                        expires_at: Utc::now() + chrono::Duration::minutes(30),
                    };
                    state.refresh_requests += 1;
                    state
                        .sessions
                        .insert(refreshed.session_id.as_str().to_owned(), refreshed.clone());
                    drop(state);
                    respond_json(request, 200, &refreshed);
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
        site_seed_node_urls: vec![TEST_WEBRTC_DIRECT_SEED.into()],
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
    let mut entry = prepared.manifests.experiment_directory[0].clone();
    let head = current_edge_head(&entry, label);
    entry.current_head_id = Some(head.head_id.clone());
    let mut snapshot = edge_snapshot_for_manifests(&prepared.manifests, BrowserMode::Trainer);
    snapshot.directory.entries = vec![entry.clone()];
    snapshot.heads = vec![head.clone()];
    let edge = spawn_local_edge(snapshot.clone());
    let trainer_requested_scopes = local_mock_trainer_scopes(&entry);
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
            BrowserEnrollmentConfig::from_edge_snapshot_with_app_version(
                &browser_snapshot,
                "browser-wasm",
                prepared
                    .manifests
                    .release_manifest
                    .target_artifact_hash
                    .clone(),
                prepared.manifests.release_manifest.app_semver.clone(),
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
            BrowserEnrollmentConfig::from_edge_snapshot_with_app_version(
                &browser_snapshot,
                "browser-wasm-verifier",
                prepared
                    .manifests
                    .release_manifest
                    .target_artifact_hash
                    .clone(),
                prepared.manifests.release_manifest.app_semver.clone(),
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
        apply_canonical_browser_head(&mut trainer, &head);
        let training = trainer
            .run_training(BrowserTrainingPlan {
                study_id: entry.study_id.clone(),
                experiment_id: entry.experiment_id.clone(),
                revision_id: entry.current_revision_id.clone(),
                workload_id: entry.workload_id.clone(),
                budget: BrowserTrainingBudget::default(),
                lease: None,
                contribution: None,
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
        apply_canonical_browser_head(&mut verifier, &head);
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
    run_with_large_stack("ci-native-smoke-suite", || {
        nca_native_peer_exports_shards_and_executes_training_windows_impl();
        nca_native_runtime_persists_and_publishes_artifacts_impl();
        nca_bootstrap_only_topology_supports_diffusion_and_read_only_browser_roles();
        browser_conformance_uses_native_dragon_manifests();
    });
}

#[test]
fn nca_native_peer_exports_shards_and_executes_training_windows() {
    run_with_large_stack(
        "nca-native-shard-training",
        nca_native_peer_exports_shards_and_executes_training_windows_impl,
    );
}

fn nca_native_peer_exports_shards_and_executes_training_windows_impl() {
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
        training_overrides: Default::default(),
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-native"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0").expect("valid burn_dragon version"),
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
fn ruliad_native_peer_executes_live_source_training_window() {
    run_with_large_stack("ruliad-live-source-smoke", || {
        let _guard = native_swarm_test_guard();
        let root = tempdir().expect("root");
        let training_config_path =
            write_ruliad_smoke_training_config(root.path(), MATCHED_512_SMALL_SPEC);
        let native = native_smoke_peer_config(
            root.path(),
            training_config_path,
            "storage-ruliad-live",
            "ruliad-live",
            None,
        );

        let prepared = prepare_nca_native_cpu(&native, Some(&dummy_auth_bundle())).expect("peer");
        assert_eq!(
            prepared.project.data_pipeline_kind(),
            burn_p2p::LeaseDataPipelineKind::IndexedDataset
        );
        let observations = run_training_windows_with_heads(&prepared, 1, "ruliad-live");
        let losses = observations.iter().map(|obs| obs.loss).collect::<Vec<_>>();
        log_loss_series("ruliad_native_live_source_smoke", &losses);
        assert!(losses.iter().all(|loss| loss.is_finite()));
        assert!(losses.iter().copied().fold(f64::INFINITY, f64::min) <= losses[0] + 0.5);
        for (index, observation) in observations.iter().enumerate() {
            assert_ruliad_source_selection_metrics(
                &format!("ruliad live source window {}", index + 1),
                &observation.head.metrics,
            );
        }
    });
}

#[test]
fn nca_native_runtime_persists_and_publishes_artifacts() {
    run_with_large_stack(
        "nca-native-runtime-artifacts",
        nca_native_runtime_persists_and_publishes_artifacts_impl,
    );
}

fn nca_native_runtime_persists_and_publishes_artifacts_impl() {
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
        training_overrides: Default::default(),
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-runtime-artifacts"),
        network: DragonPeerNetworkConfig::default()
            .with_listen_addresses(vec![loopback_swarm_address()]),
        target: Some(DragonNativeTarget::Trainer),
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0").expect("valid burn_dragon version"),
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

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA GPU and NVIDIA driver devices"]
fn nca_native_cuda_runtime_trains_window() {
    if !Path::new("/dev/nvidiactl").exists() || !Path::new("/dev/nvidia0").exists() {
        eprintln!(
            "skipping CUDA runtime smoke because /dev/nvidiactl and /dev/nvidia0 are not visible"
        );
        return;
    }

    let _guard = native_swarm_test_guard();
    let root = tempdir().expect("root");
    let nca_config_path = root.path().join("nca.toml");
    let training_config_path = root.path().join("nca-train-cuda.toml");
    let shard_root = root.path().join("nca-cuda-shards");
    write(&nca_config_path, &nca_corpus_config_toml(root.path()));
    write(
        &training_config_path,
        &nca_training_config_toml(
            &root.path().join("nca-cache-cuda"),
            &nca_config_path,
            SMALL_SPEC,
        ),
    );

    let native = DragonNativePeerConfig {
        training_overrides: Default::default(),
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-cuda-runtime"),
        network: DragonPeerNetworkConfig::default()
            .with_listen_addresses(vec![loopback_swarm_address()]),
        target: Some(DragonNativeTarget::Trainer),
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0").expect("valid burn_dragon version"),
        git_commit: Some("cuda-runtime-smoke".into()),
        enabled_features_label: Some("native,cuda".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export: Some(DragonShardExportConfig {
            root: shard_root,
            dataset_name: Some("dragon-nca-cuda-runtime".into()),
            microshards: Some(4),
            max_records: Some(32),
            http_upstream: None,
        }),
        existing_shard_dataset: None,
    };

    let prepared = prepare_nca_native_cuda(&native, Some(&dummy_auth_bundle())).expect("cuda peer");
    assert_eq!(prepared.backend_label, "cuda");
    let experiment_entry = prepared.manifests.experiment_directory[0].clone();
    let mut peer = spawn_prepared_native_peer(prepared).expect("spawn cuda peer");
    let telemetry = peer.telemetry();
    wait_for(
        Duration::from_secs(10),
        || {
            let snapshot = telemetry.snapshot();
            snapshot.local_peer_id.is_some() && !snapshot.listen_addresses.is_empty()
        },
        "cuda runtime peer did not start",
    );

    let experiment = peer.mainnet().experiment(
        experiment_entry.study_id,
        experiment_entry.experiment_id,
        experiment_entry.current_revision_id,
    );
    let genesis_head = peer
        .initialize_local_head(&experiment)
        .expect("init cuda genesis head");
    let training = peer
        .train_window_once_with_pinned_head(&experiment, Some(&genesis_head))
        .expect("train one cuda window");
    let loss = metric_float_any(&training.report.stats, &["loss", "train_loss"]);
    assert!(loss.is_finite(), "cuda train loss must be finite");
    assert_eq!(
        training.head.parent_head_id,
        Some(genesis_head.head_id.clone())
    );
    assert_eq!(training.head.global_step, genesis_head.global_step + 1);

    shutdown_runtime_peer(peer, "cuda runtime peer");
}

#[test]
#[ignore = "covered by the explicit nca-runtime-cluster validation rung"]
fn nca_native_runtime_cluster_smoke_converges_and_merges_heads() {
    run_with_large_stack(
        "nca-runtime-cluster",
        nca_native_runtime_cluster_smoke_converges_and_merges_heads_impl,
    );
}

fn nca_native_runtime_cluster_smoke_converges_and_merges_heads_impl() {
    let _guard = native_swarm_test_guard();
    let root = tempdir().expect("root");
    let bootstrap_storage = tempdir().expect("bootstrap storage");
    let nca_config_path = root.path().join("nca.toml");
    let training_config_path = root.path().join("nca-train.toml");
    let shared_shard_root = root.path().join("shared-shards");
    write(&nca_config_path, &nca_corpus_config_toml(root.path()));
    write(
        &training_config_path,
        &nca_training_config_toml(&root.path().join("nca-cache"), &nca_config_path, SMALL_SPEC),
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
                transport_policy: None,
                metrics_retention: burn_p2p::MetricsRetentionConfig::default(),
                bootstrap_peers: Vec::new(),
                listen_addresses: vec![bootstrap_addr.clone()],
                external_addresses: Vec::new(),
            },
            head_artifact_mirror_source_roots: Vec::new(),
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

    let make_trainer_config = |label: &str, export_shared_shards: bool| DragonNativePeerConfig {
        training_overrides: Default::default(),
        training_config_paths: vec![training_config_path.clone()],
        storage_root: root.path().join(format!("storage-{label}")),
        network: Default::default(),
        target: Some(DragonNativeTarget::Trainer),
        identity: Default::default(),
        bootstrap_peers: vec![bootstrap_addr.clone()],
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0").expect("valid burn_dragon version"),
        git_commit: Some(format!("runtime-cluster-{label}")),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export: export_shared_shards.then(|| DragonShardExportConfig {
            root: shared_shard_root.clone(),
            dataset_name: Some(format!("dragon-nca-runtime-{label}")),
            microshards: Some(4),
            max_records: Some(32),
            http_upstream: None,
        }),
        existing_shard_dataset: (!export_shared_shards).then(|| DragonExistingShardDatasetConfig {
            root: shared_shard_root.clone(),
            http_upstream: None,
        }),
    };

    let seed_prepared = prepare_nca_native_cpu(
        &make_trainer_config("seed", true),
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
        &make_trainer_config("trainer-b", false),
        Some(&dummy_auth_bundle()),
    )
    .expect("trainer b");
    let trainer_c_prepared = prepare_nca_native_cpu(
        &make_trainer_config("trainer-c", false),
        Some(&dummy_auth_bundle()),
    )
    .expect("trainer c");
    assert_eq!(
        seed_prepared.manifests.training_contract_id,
        trainer_b_prepared.manifests.training_contract_id
    );
    assert_eq!(
        seed_prepared.manifests.training_contract_id,
        trainer_c_prepared.manifests.training_contract_id
    );
    assert_eq!(
        seed_prepared.manifests.experiment_directory[0].dataset_view_id,
        trainer_b_prepared.manifests.experiment_directory[0].dataset_view_id
    );
    assert_eq!(
        seed_prepared.manifests.experiment_directory[0].dataset_view_id,
        trainer_c_prepared.manifests.experiment_directory[0].dataset_view_id
    );

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
#[ignore = "release-only multi-peer convergence parity gate"]
fn ruliad_native_runtime_1m_convergence_matches_federated_oracle() {
    run_with_large_stack("ruliad-p2p-1m-parity", || {
        let _guard = native_swarm_test_guard();
        let seed_value = env_u64(P2P_PARITY_SEED_ENV, 1337);
        let rounds = positive_env_usize(P2P_PARITY_ROUNDS_ENV, 2);
        let replay_candidates = env_bool(P2P_PARITY_REPLAY_ENV, true);
        let run_sequential_reference = env_bool(P2P_PARITY_SEQUENTIAL_ENV, true);
        let run_synchronized_reference = env_bool(P2P_PARITY_SYNCHRONIZED_ENV, true);
        let use_signed_revision_contract = env_bool(P2P_PARITY_SIGNED_CONTRACT_ENV, false);
        let minimum_synchronized_progress_ratio =
            env_f64(P2P_PARITY_MIN_SYNC_PROGRESS_RATIO_ENV, 0.90);
        let require_convergence_parity = env_bool(P2P_PARITY_REQUIRE_CONVERGENCE_ENV, false);
        assert!(
            (0.0..=1.0).contains(&minimum_synchronized_progress_ratio),
            "minimum synchronized progress ratio must be in [0, 1]"
        );
        assert!(
            !require_convergence_parity || run_synchronized_reference,
            "hard convergence parity requires the synchronized reference"
        );
        let root_ema_update_basis_points =
            u16::try_from(env_u64(P2P_PARITY_ROOT_EMA_BASIS_POINTS_ENV, 10_000))
                .expect("root EMA basis points must fit in u16");
        let peer_local_steps = positive_env_usize(P2P_PARITY_LOCAL_STEPS_ENV, 9);
        let restart_after_round = usize::try_from(env_u64(P2P_PARITY_RESTART_AFTER_ROUND_ENV, 0))
            .expect("restart round must fit in usize");
        assert!(
            restart_after_round == 0 || restart_after_round < rounds,
            "restart-after-round must be zero or precede the final round"
        );
        assert!(
            peer_local_steps <= RULIAD_PARITY_1M_SPEC.max_iters,
            "peer-local steps must not exceed the configured per-window max_iters"
        );
        let records_per_round = peer_local_steps
            .checked_mul(RULIAD_PARITY_1M_SPEC.batch_size)
            .and_then(|records| records.checked_mul(3))
            .expect("parity records per round");
        let exported_records = records_per_round
            .checked_mul(rounds)
            .expect("parity exported record count");
        let root = tempdir().expect("parity root");
        let bootstrap_storage = tempdir().expect("parity bootstrap storage");
        let training_config_path =
            write_ruliad_parity_training_config(root.path(), seed_value, peer_local_steps);
        let synchronized_training_config_path = run_synchronized_reference.then(|| {
            write_ruliad_synchronized_reference_config(
                root.path(),
                seed_value,
                peer_local_steps
                    .checked_mul(3)
                    .expect("synchronized reference batch count"),
                3,
            )
        });
        let shared_shard_root = root.path().join("shared-ruliad-shards");

        let bootstrap_addr = loopback_swarm_address();
        let bootstrap_plan = burn_p2p_bootstrap::BootstrapSpec {
            preset: burn_p2p_bootstrap::BootstrapPreset::BootstrapOnly,
            genesis: burn_p2p_core::GenesisSpec {
                network_id: burn_p2p_core::NetworkId::new("dragon-p2p-parity-testnet"),
                protocol_version: Version::new(0, 1, 0),
                display_name: "dragon 1m ruliad parity".into(),
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
        .expect("parity bootstrap plan");
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
                    transport_policy: None,
                    metrics_retention: burn_p2p::MetricsRetentionConfig::default(),
                    bootstrap_peers: Vec::new(),
                    listen_addresses: vec![bootstrap_addr.clone()],
                    external_addresses: Vec::new(),
                },
                head_artifact_mirror_source_roots: Vec::new(),
            })
            .expect("spawn parity bootstrap");
        let bootstrap_telemetry = bootstrap.telemetry();
        wait_for(
            Duration::from_secs(10),
            || {
                let snapshot = bootstrap_telemetry.snapshot();
                snapshot.local_peer_id.is_some() && !snapshot.listen_addresses.is_empty()
            },
            "parity bootstrap did not start",
        );

        let make_trainer_config =
            |label: &str, export_shared_shards: bool| DragonNativePeerConfig {
                training_overrides: burn_dragon_p2p::config::DragonNativeTrainingOverrides {
                    max_eval_batches: Some(4),
                    ..Default::default()
                },
                training_config_paths: vec![training_config_path.clone()],
                storage_root: root.path().join(format!("storage-{label}")),
                network: Default::default(),
                target: Some(DragonNativeTarget::Trainer),
                identity: burn_p2p::IdentityConfig::Persistent,
                bootstrap_peers: vec![bootstrap_addr.clone()],
                manifest: DragonManifestSeed {
                    network_id: "dragon-p2p-parity-testnet".into(),
                    display_name: "dragon 1m ruliad parity".into(),
                    description: "three-peer 1m-class convergence parity gate".into(),
                    aggregation: DragonAggregationConfig {
                        root_ema_update_basis_points,
                    },
                    ..native_manifest_seed()
                },
                app_semver: semver::Version::parse("0.21.0").expect("valid burn_dragon version"),
                git_commit: Some("p2p-1m-parity".into()),
                enabled_features_label: Some("native-cpu".into()),
                auth: None,
                capability_policy: Default::default(),
                shard_export: export_shared_shards.then(|| DragonShardExportConfig {
                    root: shared_shard_root.clone(),
                    dataset_name: Some("dragon-ruliad-parity".into()),
                    microshards: Some(3),
                    max_records: Some(exported_records),
                    http_upstream: None,
                }),
                existing_shard_dataset: (!export_shared_shards).then(|| {
                    DragonExistingShardDatasetConfig {
                        root: shared_shard_root.clone(),
                        http_upstream: None,
                    }
                }),
            };

        let unsigned_seed_prepared = prepare_nca_native_cpu(
            &make_trainer_config("seed", true),
            Some(&dummy_auth_bundle()),
        )
        .expect("prepare unsigned parity seed");
        let signed_setup = use_signed_revision_contract.then(|| {
            let authority = Keypair::generate_ed25519();
            let (bundle, authority_public_key) =
                signed_revision_contract_fixture(&unsigned_seed_prepared, &authority, seed_value);
            let path = root.path().join("signed-revision-contract.json");
            fs::write(
                &path,
                serde_json::to_vec_pretty(&bundle).expect("serialize signed revision contract"),
            )
            .expect("write signed revision contract");
            (path, authority_public_key, bundle)
        });
        let apply_signed_setup = |mut config: DragonNativePeerConfig| {
            if let Some((path, authority_public_key, _)) = signed_setup.as_ref() {
                config.manifest.revision_contract_path = Some(path.clone());
                config.manifest.require_signed_revision_contracts = true;
                config.manifest.authority_public_keys = vec![authority_public_key.clone()];
            }
            config
        };
        let seed_prepared = if signed_setup.is_some() {
            prepare_nca_native_cpu(
                &apply_signed_setup(make_trainer_config("seed", false)),
                Some(&dummy_auth_bundle()),
            )
            .expect("prepare strict signed parity seed")
        } else {
            unsigned_seed_prepared
        };
        let reference_project = BurnWorkloadAdapter::try_new(
            seed_prepared.project.clone(),
            seed_prepared.manifests.workload_config.clone(),
        )
        .expect("build parity reference adapter");
        let synchronized_project =
            synchronized_training_config_path
                .as_ref()
                .map(|training_config_path| {
                    let mut config = make_trainer_config("synchronized-reference", false);
                    config.training_config_paths = vec![training_config_path.clone()];
                    let prepared = prepare_nca_native_cpu(&config, Some(&dummy_auth_bundle()))
                        .expect("prepare synchronized reference");
                    BurnWorkloadAdapter::try_new(
                        prepared.project,
                        prepared.manifests.workload_config,
                    )
                    .expect("build synchronized reference adapter")
                });
        let experiment_entry = seed_prepared.manifests.experiment_directory[0].clone();
        let trainer_b_prepared = prepare_nca_native_cpu(
            &apply_signed_setup(make_trainer_config("trainer-b", false)),
            Some(&dummy_auth_bundle()),
        )
        .expect("prepare parity trainer b");
        let trainer_c_prepared = prepare_nca_native_cpu(
            &apply_signed_setup(make_trainer_config("trainer-c", false)),
            Some(&dummy_auth_bundle()),
        )
        .expect("prepare parity trainer c");

        for prepared in [&trainer_b_prepared, &trainer_c_prepared] {
            assert_eq!(
                seed_prepared.manifests.training_contract_id,
                prepared.manifests.training_contract_id
            );
            assert_eq!(
                experiment_entry.dataset_view_id,
                prepared.manifests.experiment_directory[0].dataset_view_id
            );
            assert_eq!(
                seed_prepared.manifests.release_manifest.release_train_hash,
                prepared.manifests.release_manifest.release_train_hash
            );
            assert_eq!(
                seed_prepared
                    .manifests
                    .supported_workload
                    .model_program_hash,
                prepared.manifests.supported_workload.model_program_hash
            );
            assert_eq!(
                seed_prepared.manifests.revision_manifest,
                prepared.manifests.revision_manifest
            );
        }
        let training_contract_id = seed_prepared.manifests.training_contract_id.clone();
        let optimizer_state_policy = seed_prepared
            .manifests
            .training_contract
            .optimizer_state_policy
            .clone();
        let scheduler_state_policy = seed_prepared
            .manifests
            .training_contract
            .scheduler_state_policy
            .clone();

        let reference_device = reference_project.runtime_device();
        let initial_reference_model = reference_project.init_model(&reference_device);
        let inventory =
            burn_p2p::burn::inspect_module::<NativeCpuBackend, _>(&initial_reference_model);
        assert!(
            (800_000..=1_200_000).contains(&inventory.total_scalar_parameters),
            "parity model should be 1m-class, got {} parameters",
            inventory.total_scalar_parameters
        );
        let reference_registration = reference_project
            .dataset_registration()
            .expect("reference registration");
        assert_eq!(
            reference_registration
                .view
                .metadata
                .get("partitioning")
                .map(String::as_str),
            Some("dragon-bounded-stream-segment-balanced-v2")
        );
        let reference_microshard_plan = reference_project
            .microshard_plan(&reference_registration)
            .expect("reference microshard plan");
        assert_eq!(reference_microshard_plan.microshards.len(), 3);
        if let Some(synchronized_project) = synchronized_project.as_ref() {
            assert_eq!(
                synchronized_project.model_schema_hash(),
                reference_project.model_schema_hash()
            );
            assert_eq!(
                synchronized_project
                    .dataset_registration()
                    .expect("synchronized reference registration")
                    .view,
                reference_registration.view
            );
            // Both references are initialized from the serialized network
            // genesis below. Independent learner construction is not itself a
            // canonical initialization boundary.
        }
        let reference_cache_root = tempdir().expect("reference shard cache");
        let reference_shard_cache = ShardCache::new(reference_cache_root.path());
        let reference_artifact_root = tempdir().expect("reference artifact root");
        let reference_artifact_store = FsArtifactStore::new(reference_artifact_root.path());
        let mut oracle_model = round_trip_reference_model(
            &reference_project,
            &initial_reference_model,
            &reference_artifact_store,
            "oracle-genesis",
            None,
        );

        let mut seed = spawn_prepared_native_peer(seed_prepared).expect("spawn parity seed");
        let mut trainer_b =
            spawn_prepared_native_peer(trainer_b_prepared).expect("spawn parity trainer b");
        let mut trainer_c =
            spawn_prepared_native_peer(trainer_c_prepared).expect("spawn parity trainer c");
        let seed_telemetry = seed.telemetry();
        let trainer_b_telemetry = trainer_b.telemetry();
        let mut trainer_c_telemetry = trainer_c.telemetry();

        for (label, telemetry) in [
            ("seed", &seed_telemetry),
            ("trainer-b", &trainer_b_telemetry),
            ("trainer-c", &trainer_c_telemetry),
        ] {
            wait_for(
                Duration::from_secs(30),
                || telemetry.snapshot().connected_peers >= 1,
                &format!("parity {label} did not connect"),
            );
        }

        let experiment = seed.mainnet().experiment(
            experiment_entry.study_id.clone(),
            experiment_entry.experiment_id.clone(),
            experiment_entry.current_revision_id.clone(),
        );
        let genesis_head = seed
            .initialize_local_head(&experiment)
            .expect("initialize parity genesis");
        for trainer in [&trainer_b, &trainer_c] {
            wait_for(
                Duration::from_secs(45),
                || {
                    trainer
                        .sync_experiment_head(&experiment)
                        .expect("sync parity genesis")
                        .is_some()
                },
                "parity trainer did not sync genesis",
            );
        }

        let provider_peer_ids = [
            seed.snapshot().local_peer_id.expect("seed peer id"),
            trainer_b
                .snapshot()
                .local_peer_id
                .expect("trainer b peer id"),
            trainer_c
                .snapshot()
                .local_peer_id
                .expect("trainer c peer id"),
        ];
        ensure_materialized_pinned_head(
            "parity-seed",
            &seed,
            &experiment,
            &genesis_head,
            &provider_peer_ids,
        );
        ensure_materialized_pinned_head(
            "parity-trainer-b",
            &trainer_b,
            &experiment,
            &genesis_head,
            &provider_peer_ids,
        );
        ensure_materialized_pinned_head(
            "parity-trainer-c",
            &trainer_c,
            &experiment,
            &genesis_head,
            &provider_peer_ids,
        );

        let oracle_genesis_digest = model_digest(&reference_project, &oracle_model);
        let prepared_genesis_digest = model_digest(&reference_project, &initial_reference_model);
        let genesis_digests = [
            seed.materialized_head_tensor_digest(&genesis_head)
                .expect("seed genesis digest"),
            trainer_b
                .materialized_head_tensor_digest(&genesis_head)
                .expect("trainer b genesis digest"),
            trainer_c
                .materialized_head_tensor_digest(&genesis_head)
                .expect("trainer c genesis digest"),
        ];
        eprintln!(
            "p2p_1m_genesis_digests: prepared={} prepared_roundtrip={} seed={} trainer_b={} trainer_c={}",
            prepared_genesis_digest.as_str(),
            oracle_genesis_digest.as_str(),
            genesis_digests[0].as_str(),
            genesis_digests[1].as_str(),
            genesis_digests[2].as_str(),
        );
        let seed_artifact_store = seed.artifact_store().expect("seed artifact store");
        let canonical_genesis_descriptor = seed_artifact_store
            .load_manifest(&genesis_head.artifact_id)
            .expect("load canonical genesis descriptor");
        let canonical_genesis_model = reference_project
            .load_model_artifact(
                reference_project.init_model(&reference_device),
                &canonical_genesis_descriptor,
                &seed_artifact_store,
                &reference_device,
            )
            .expect("load canonical genesis through reference adapter");
        let canonical_genesis_digest = model_digest(&reference_project, &canonical_genesis_model);
        assert!(
            genesis_digests
                .iter()
                .all(|digest| digest == &canonical_genesis_digest),
            "all peers and the independent adapter must decode the exact canonical genesis tensor set"
        );
        if let Some((_, _, bundle)) = signed_setup.as_ref() {
            assert_eq!(
                genesis_head.artifact_id,
                bundle.genesis.payload.payload.artifact.artifact_id
            );
            assert_eq!(
                canonical_genesis_digest,
                bundle.genesis.payload.payload.tensor_digest
            );
        }
        let prepared_genesis_matches_canonical = oracle_genesis_digest == canonical_genesis_digest;
        oracle_model = canonical_genesis_model;
        let mut sequential_model = oracle_model.clone();
        let mut synchronized_model = synchronized_project.as_ref().map(|_| oracle_model.clone());
        let genesis_losses = [
            metric_float_any(
                &seed
                    .evaluate_materialized_head(&genesis_head, burn_p2p::EvalSplit::Validation)
                    .expect("seed genesis evaluation")
                    .metrics,
                &["loss", "validation_loss"],
            ),
            metric_float_any(
                &trainer_b
                    .evaluate_materialized_head(&genesis_head, burn_p2p::EvalSplit::Validation)
                    .expect("trainer b genesis evaluation")
                    .metrics,
                &["loss", "validation_loss"],
            ),
            metric_float_any(
                &trainer_c
                    .evaluate_materialized_head(&genesis_head, burn_p2p::EvalSplit::Validation)
                    .expect("trainer c genesis evaluation")
                    .metrics,
                &["loss", "validation_loss"],
            ),
        ];
        assert!(
            genesis_losses
                .iter()
                .all(|loss| (*loss - genesis_losses[0]).abs() <= 1.0e-8)
        );
        let genesis_loss = genesis_losses[0];
        assert!(
            (validation_loss(&reference_project, &oracle_model) - genesis_loss).abs() <= 1.0e-8
        );

        let mut canonical_head = genesis_head;
        let mut p2p_validation_losses = vec![genesis_loss];
        let mut oracle_validation_losses = vec![genesis_loss];
        let mut sequential_validation_losses = if run_sequential_reference {
            vec![genesis_loss]
        } else {
            Vec::new()
        };
        let mut synchronized_validation_losses = if run_synchronized_reference {
            vec![genesis_loss]
        } else {
            Vec::new()
        };
        let mut round_reports = Vec::new();
        let mut total_training_secs = 0.0;
        let mut total_promotion_secs = 0.0;
        let mut restart_report = None;

        for round in 0..rounds {
            let base_head = canonical_head.clone();
            let base_head_id = base_head.head_id.clone();
            let start_barrier = Arc::new(std::sync::Barrier::new(3));
            let experiment_for_seed = experiment.clone();
            let experiment_for_b = experiment.clone();
            let experiment_for_c = experiment.clone();
            let pinned_seed = base_head.clone();
            let pinned_b = base_head.clone();
            let pinned_c = base_head.clone();
            let seed_ref = &mut seed;
            let trainer_b_ref = &mut trainer_b;
            let trainer_c_ref = &mut trainer_c;
            let training_started = Instant::now();
            let (seed_window, trainer_b_window, trainer_c_window) = thread::scope(|scope| {
                let seed_barrier = Arc::clone(&start_barrier);
                let seed_run = scope.spawn(move || {
                    seed_barrier.wait();
                    seed_ref.train_window_once_with_pinned_head(
                        &experiment_for_seed,
                        Some(&pinned_seed),
                    )
                });
                let trainer_b_barrier = Arc::clone(&start_barrier);
                let trainer_b_run = scope.spawn(move || {
                    trainer_b_barrier.wait();
                    trainer_b_ref
                        .train_window_once_with_pinned_head(&experiment_for_b, Some(&pinned_b))
                });
                let trainer_c_barrier = Arc::clone(&start_barrier);
                let trainer_c_run = scope.spawn(move || {
                    trainer_c_barrier.wait();
                    trainer_c_ref
                        .train_window_once_with_pinned_head(&experiment_for_c, Some(&pinned_c))
                });
                let seed_window = seed_run
                    .join()
                    .map_err(|_| anyhow::anyhow!("parity seed train thread panicked"))??;
                let trainer_b_window = trainer_b_run
                    .join()
                    .map_err(|_| anyhow::anyhow!("parity trainer b train thread panicked"))??;
                let trainer_c_window = trainer_c_run
                    .join()
                    .map_err(|_| anyhow::anyhow!("parity trainer c train thread panicked"))??;
                anyhow::Ok((seed_window, trainer_b_window, trainer_c_window))
            })
            .expect("parallel parity windows");
            let training_elapsed = training_started.elapsed();
            total_training_secs += training_elapsed.as_secs_f64();
            let outcomes = [&seed_window, &trainer_b_window, &trainer_c_window];

            assert!(
                outcomes
                    .iter()
                    .all(|outcome| outcome.lease.window_id == seed_window.lease.window_id)
            );
            assert!(
                outcomes
                    .iter()
                    .all(|outcome| outcome.lease.dataset_view_id
                        == seed_window.lease.dataset_view_id)
            );
            assert_eq!(
                seed_window.lease.dataset_view_id,
                experiment_entry.dataset_view_id
            );
            let lease_sets = outcomes
                .iter()
                .map(|outcome| {
                    outcome
                        .lease
                        .microshards
                        .iter()
                        .cloned()
                        .collect::<BTreeSet<_>>()
                })
                .collect::<Vec<_>>();
            for left in 0..lease_sets.len() {
                for right in (left + 1)..lease_sets.len() {
                    assert!(
                        lease_sets[left].is_disjoint(&lease_sets[right]),
                        "round {} peer leases overlap: {:?} vs {:?}",
                        round + 1,
                        lease_sets[left],
                        lease_sets[right]
                    );
                }
            }
            assert!(
                lease_sets.iter().all(|set| !set.is_empty()),
                "every peer must receive at least one microshard"
            );

            for outcome in outcomes {
                let mean = metric_float(&outcome.report.stats, "train_loss_mean");
                let last = metric_float(&outcome.report.stats, "train_loss_last");
                let steps = metric_integer(&outcome.report.stats, "train_steps");
                assert!(mean.is_finite() && last.is_finite() && steps > 0);
                assert_eq!(
                    steps, peer_local_steps as i64,
                    "every peer must execute the configured bounded micro-epoch"
                );
                assert_eq!(outcome.head.parent_head_id, Some(base_head_id.clone()));
                assert_eq!(outcome.head.global_step, base_head.global_step + 1);
            }

            for (label, peer) in [
                ("parity-seed", &seed),
                ("parity-trainer-b", &trainer_b),
                ("parity-trainer-c", &trainer_c),
            ] {
                for (artifact_label, artifact_id, provider_peer_id) in [
                    (
                        "seed update",
                        &seed_window.artifact.artifact_id,
                        &provider_peer_ids[0],
                    ),
                    (
                        "seed head",
                        &seed_window.head.artifact_id,
                        &provider_peer_ids[0],
                    ),
                    (
                        "trainer-b update",
                        &trainer_b_window.artifact.artifact_id,
                        &provider_peer_ids[1],
                    ),
                    (
                        "trainer-b head",
                        &trainer_b_window.head.artifact_id,
                        &provider_peer_ids[1],
                    ),
                    (
                        "trainer-c update",
                        &trainer_c_window.artifact.artifact_id,
                        &provider_peer_ids[2],
                    ),
                    (
                        "trainer-c head",
                        &trainer_c_window.head.artifact_id,
                        &provider_peer_ids[2],
                    ),
                ] {
                    ensure_materialized_artifact(
                        label,
                        peer,
                        std::slice::from_ref(provider_peer_id),
                        artifact_id,
                        artifact_label,
                        Duration::from_secs(60),
                    );
                }
            }

            let window_id = seed_window.lease.window_id;
            wait_for(
                Duration::from_secs(45),
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
                            .map(|announcement| announcement.update.peer_id.clone())
                            .collect::<BTreeSet<_>>()
                            .len()
                            == 3
                    })
                },
                "parity peers did not all observe three update announcements",
            );
            let visible_updates = seed_telemetry
                .snapshot()
                .control_plane
                .update_announcements
                .into_iter()
                .filter(|announcement| {
                    announcement.update.study_id == experiment.study_id
                        && announcement.update.experiment_id == experiment.experiment_id
                        && announcement.update.revision_id == experiment.revision_id
                        && announcement.update.window_id == window_id
                        && announcement.update.base_head_id == base_head_id
                })
                .map(|announcement| announcement.update)
                .collect::<Vec<_>>();
            assert_eq!(
                visible_updates
                    .iter()
                    .map(|update| update.peer_id.clone())
                    .collect::<BTreeSet<_>>()
                    .len(),
                3
            );

            let runtime_candidate_digests = [
                seed.materialized_head_tensor_digest(&seed_window.head)
                    .expect("seed candidate digest"),
                trainer_b
                    .materialized_head_tensor_digest(&trainer_b_window.head)
                    .expect("trainer b candidate digest"),
                trainer_c
                    .materialized_head_tensor_digest(&trainer_c_window.head)
                    .expect("trainer c candidate digest"),
            ];
            let mut oracle_candidates = Vec::new();
            for (index, outcome) in outcomes.iter().enumerate() {
                let model = if replay_candidates {
                    let reference = train_reference_lease(
                        &reference_project,
                        oracle_model.clone(),
                        &outcome.lease,
                        &reference_registration,
                        &reference_microshard_plan,
                        &reference_shard_cache,
                    );
                    let runtime_loss = metric_float(&outcome.report.stats, "train_loss_mean");
                    let reference_loss = metric_float(&reference.stats, "train_loss_mean");
                    assert!(
                        (runtime_loss - reference_loss).abs() <= 1.0e-6,
                        "round {} peer {} train loss diverged from reference: runtime={} reference={}",
                        round + 1,
                        index,
                        runtime_loss,
                        reference_loss
                    );
                    round_trip_reference_model(
                        &reference_project,
                        &reference.model,
                        &reference_artifact_store,
                        &format!("oracle-candidate-{}-{index}", round + 1),
                        Some(base_head_id.clone()),
                    )
                } else {
                    let descriptor = seed_artifact_store
                        .load_manifest(&outcome.head.artifact_id)
                        .expect("load replicated candidate descriptor");
                    reference_project
                        .load_model_artifact(
                            reference_project.init_model(&reference_device),
                            &descriptor,
                            &seed_artifact_store,
                            &reference_device,
                        )
                        .expect("decode replicated candidate through oracle adapter")
                };
                let digest = model_digest(&reference_project, &model);
                assert_eq!(
                    digest,
                    runtime_candidate_digests[index],
                    "round {} peer {} candidate tensors diverged from local replay",
                    round + 1,
                    index
                );
                let update = visible_updates
                    .iter()
                    .find(|update| update.peer_id == provider_peer_ids[index])
                    .expect("visible peer update");
                assert!(
                    (update.sample_weight - outcome.contribution.accepted_weight).abs() <= 1.0e-8
                );
                assert!(
                    (update.quality_weight - update_quality_weight(&outcome.report.stats)).abs()
                        <= 1.0e-8
                );
                oracle_candidates.push(OracleCandidate {
                    peer_id: update.peer_id.clone(),
                    head_id: outcome.head.head_id.clone(),
                    artifact_id: outcome.head.artifact_id.clone(),
                    model,
                    sample_weight: update.sample_weight,
                    quality_weight: update.quality_weight,
                    announced_at: update.announced_at,
                });
            }
            oracle_candidates.sort_by(|left, right| {
                (right.sample_weight * right.quality_weight)
                    .total_cmp(&(left.sample_weight * left.quality_weight))
                    .then(right.announced_at.cmp(&left.announced_at))
                    .then(left.peer_id.cmp(&right.peer_id))
                    .then(left.artifact_id.cmp(&right.artifact_id))
            });
            let merge_inputs = oracle_candidates
                .iter()
                .map(|candidate| MergeModelCandidate {
                    peer_id: &candidate.peer_id,
                    head_id: &candidate.head_id,
                    artifact_id: &candidate.artifact_id,
                    model: &candidate.model,
                    sample_weight: candidate.sample_weight,
                    quality_weight: candidate.quality_weight,
                })
                .collect::<Vec<_>>();
            let merged = reference_project
                .merge_candidate_models(
                    &oracle_model,
                    &merge_inputs,
                    MergePolicy::QualityWeightedEma,
                )
                .expect("merge oracle candidates")
                .expect("Dragon parity workload must support model merge");
            let merged = reference_project
                .apply_single_root_ema(&oracle_model, merged, MergePolicy::QualityWeightedEma)
                .expect("apply oracle root EMA");
            oracle_model = round_trip_reference_model(
                &reference_project,
                &merged,
                &reference_artifact_store,
                &format!("oracle-merged-{}", round + 1),
                Some(base_head_id.clone()),
            );
            let oracle_digest = model_digest(&reference_project, &oracle_model);
            let oracle_loss = validation_loss(&reference_project, &oracle_model);

            let sequential_loss = if run_sequential_reference {
                let mut sequential_outcomes = outcomes;
                sequential_outcomes
                    .sort_by(|left, right| left.lease.microshards.cmp(&right.lease.microshards));
                for outcome in sequential_outcomes {
                    let reference = train_reference_lease(
                        &reference_project,
                        sequential_model,
                        &outcome.lease,
                        &reference_registration,
                        &reference_microshard_plan,
                        &reference_shard_cache,
                    );
                    sequential_model = reference.model;
                }
                sequential_model = round_trip_reference_model(
                    &reference_project,
                    &sequential_model,
                    &reference_artifact_store,
                    &format!("sequential-{}", round + 1),
                    None,
                );
                Some(validation_loss(&reference_project, &sequential_model))
            } else {
                None
            };
            let synchronized_reference =
                synchronized_project.as_ref().map(|synchronized_project| {
                    let model = synchronized_model
                        .take()
                        .expect("synchronized reference model");
                    let leases = outcomes
                        .iter()
                        .map(|outcome| &outcome.lease)
                        .collect::<Vec<_>>();
                    let reference = train_synchronized_reference_round(
                        &reference_project,
                        synchronized_project,
                        model,
                        &leases,
                        &reference_registration,
                        &reference_microshard_plan,
                        &reference_shard_cache,
                        peer_local_steps,
                    );
                    let train_steps = metric_integer(&reference.stats, "train_steps");
                    assert_eq!(
                        train_steps,
                        peer_local_steps.saturating_mul(3) as i64,
                        "synchronized reference must consume every peer batch exactly once"
                    );
                    let model = round_trip_reference_model(
                        synchronized_project,
                        &reference.model,
                        &reference_artifact_store,
                        &format!("synchronized-{}", round + 1),
                        None,
                    );
                    let loss = validation_loss(synchronized_project, &model);
                    synchronized_model = Some(model);
                    (loss, train_steps)
                });
            let synchronized_loss =
                synchronized_reference.map(|(validation_loss, _)| validation_loss);

            let promotion_started = Instant::now();
            let promoted_head = converge_three_peer_diffusion_round(
                &experiment,
                &mut seed,
                &mut trainer_b,
                &mut trainer_c,
                &base_head,
            );
            let promotion_elapsed = promotion_started.elapsed();
            total_promotion_secs += promotion_elapsed.as_secs_f64();
            wait_for(
                Duration::from_secs(45),
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
                                    && announcement.certificate.merged_head_id
                                        == promoted_head.head_id
                                    && announcement.certificate.promotion_mode
                                        == HeadPromotionMode::DiffusionSteadyState
                                    && announcement.certificate.attester_count == 3
                            })
                            && snapshot.control_plane.merge_announcements.iter().any(
                                |announcement| {
                                    announcement.certificate.base_head_id == base_head_id
                                        && announcement.certificate.merged_head_id
                                            == promoted_head.head_id
                                        && announcement.certificate.contribution_receipts.len() == 3
                                },
                            )
                    })
                },
                "three-way parity promotion certificate did not propagate",
            );
            for (label, peer) in [
                ("parity-seed", &seed),
                ("parity-trainer-b", &trainer_b),
                ("parity-trainer-c", &trainer_c),
            ] {
                ensure_materialized_artifact(
                    label,
                    peer,
                    &provider_peer_ids,
                    &promoted_head.artifact_id,
                    "promoted head",
                    Duration::from_secs(60),
                );
            }

            let p2p_digests = [
                seed.materialized_head_tensor_digest(&promoted_head)
                    .expect("seed promoted digest"),
                trainer_b
                    .materialized_head_tensor_digest(&promoted_head)
                    .expect("trainer b promoted digest"),
                trainer_c
                    .materialized_head_tensor_digest(&promoted_head)
                    .expect("trainer c promoted digest"),
            ];
            assert!(
                p2p_digests.iter().all(|digest| digest == &oracle_digest),
                "round {} promoted tensors diverged from federated oracle",
                round + 1
            );
            let p2p_losses = [
                metric_float_any(
                    &seed
                        .evaluate_materialized_head(&promoted_head, burn_p2p::EvalSplit::Validation)
                        .expect("seed promoted evaluation")
                        .metrics,
                    &["loss", "validation_loss"],
                ),
                metric_float_any(
                    &trainer_b
                        .evaluate_materialized_head(&promoted_head, burn_p2p::EvalSplit::Validation)
                        .expect("trainer b promoted evaluation")
                        .metrics,
                    &["loss", "validation_loss"],
                ),
                metric_float_any(
                    &trainer_c
                        .evaluate_materialized_head(&promoted_head, burn_p2p::EvalSplit::Validation)
                        .expect("trainer c promoted evaluation")
                        .metrics,
                    &["loss", "validation_loss"],
                ),
            ];
            assert!(
                p2p_losses
                    .iter()
                    .all(|loss| (*loss - oracle_loss).abs() <= 1.0e-8),
                "round {} promoted validation diverged from federated oracle",
                round + 1
            );
            let promoted_metric_loss =
                metric_float_any(&promoted_head.metrics, &["loss", "validation_loss"]);
            assert!((promoted_metric_loss - oracle_loss).abs() <= 1.0e-8);
            let promoted_descriptor = seed
                .artifact_store()
                .expect("seed artifact store")
                .load_manifest(&promoted_head.artifact_id)
                .expect("load promoted descriptor");

            p2p_validation_losses.push(p2p_losses[0]);
            oracle_validation_losses.push(oracle_loss);
            if let Some(sequential_loss) = sequential_loss {
                sequential_validation_losses.push(sequential_loss);
            }
            if let Some(synchronized_loss) = synchronized_loss {
                synchronized_validation_losses.push(synchronized_loss);
            }
            let mut sequential_lease_order = outcomes
                .iter()
                .map(|outcome| {
                    outcome
                        .lease
                        .microshards
                        .iter()
                        .map(|id| id.as_str())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            sequential_lease_order.sort();
            let aggregate_peer_steps_per_training_second =
                (peer_local_steps * outcomes.len()) as f64 / training_elapsed.as_secs_f64();
            round_reports.push(serde_json::json!({
                "round": round + 1,
                "window_id": window_id.0,
                "train_loss_mean": outcomes.iter().map(|outcome| metric_float(&outcome.report.stats, "train_loss_mean")).collect::<Vec<_>>(),
                "train_loss_last": outcomes.iter().map(|outcome| metric_float(&outcome.report.stats, "train_loss_last")).collect::<Vec<_>>(),
                "train_steps": outcomes.iter().map(|outcome| metric_integer(&outcome.report.stats, "train_steps")).collect::<Vec<_>>(),
                "lease_microshards": outcomes.iter().map(|outcome| outcome.lease.microshards.iter().map(|id| id.as_str()).collect::<Vec<_>>()).collect::<Vec<_>>(),
                "sequential_lease_order": sequential_lease_order,
                "candidate_artifact_bytes": outcomes.iter().map(|outcome| outcome.artifact.bytes_len).collect::<Vec<_>>(),
                "training_elapsed_secs": training_elapsed.as_secs_f64(),
                "promotion_elapsed_secs": promotion_elapsed.as_secs_f64(),
                "aggregate_peer_steps_per_training_second": aggregate_peer_steps_per_training_second,
                "promoted_artifact_bytes": promoted_descriptor.bytes_len,
                "p2p_validation_loss": p2p_losses[0],
                "oracle_validation_loss": oracle_loss,
                "sequential_validation_loss": sequential_loss,
                "synchronized_validation_loss": synchronized_loss,
                "synchronized_train_batches": synchronized_reference.map(|(_, train_steps)| train_steps),
                "synchronized_optimizer_updates": synchronized_reference
                    .map(|(_, train_steps)| train_steps / 3),
                "canonical_tensor_digest": oracle_digest.as_str(),
                "connected_peers": [
                    seed_telemetry.snapshot().connected_peers,
                    trainer_b_telemetry.snapshot().connected_peers,
                    trainer_c_telemetry.snapshot().connected_peers,
                ],
            }));
            eprintln!(
                "p2p_1m_parity_round={} train={:?} p2p_val={:.6} oracle_val={:.6} synchronized_val={:?} sequential_val={:?} train_secs={:.3} promotion_secs={:.3}",
                round + 1,
                outcomes
                    .iter()
                    .map(|outcome| metric_float(&outcome.report.stats, "train_loss_mean"))
                    .collect::<Vec<_>>(),
                p2p_losses[0],
                oracle_loss,
                synchronized_loss,
                sequential_loss,
                training_elapsed.as_secs_f64(),
                promotion_elapsed.as_secs_f64(),
            );
            canonical_head = promoted_head;

            if restart_after_round == round + 1 {
                let restart_started = Instant::now();
                let peer_id_before = trainer_c
                    .snapshot()
                    .local_peer_id
                    .expect("trainer c peer id before restart");
                trainer_c
                    .shutdown()
                    .expect("request parity trainer c restart shutdown");
                let prepared = trainer_c
                    .await_termination_timeout(Duration::from_secs(15))
                    .expect("stop parity trainer c for restart");
                thread::sleep(Duration::from_secs(2));
                trainer_c = spawn_prepared_native_peer(prepared).expect("restart parity trainer c");
                trainer_c_telemetry = trainer_c.telemetry();
                wait_for(
                    Duration::from_secs(30),
                    || trainer_c_telemetry.snapshot().connected_peers >= 1,
                    "restarted parity trainer c did not reconnect",
                );
                let peer_id_after = trainer_c_telemetry
                    .snapshot()
                    .local_peer_id
                    .expect("trainer c peer id after restart");
                assert_eq!(
                    peer_id_after, peer_id_before,
                    "persistent trainer identity changed across restart"
                );
                let recovered_head = sync_experiment_head_with_retry(
                    "restarted parity trainer c",
                    &trainer_c,
                    &experiment,
                    Instant::now() + Duration::from_secs(45),
                )
                .expect("restarted trainer should recover canonical head");
                assert_eq!(recovered_head.head_id, canonical_head.head_id);
                ensure_materialized_pinned_head(
                    "restarted-parity-trainer-c",
                    &trainer_c,
                    &experiment,
                    &canonical_head,
                    &provider_peer_ids,
                );
                let recovered_digest = trainer_c
                    .materialized_head_tensor_digest(&canonical_head)
                    .expect("restarted trainer canonical digest");
                assert_eq!(recovered_digest, oracle_digest);
                let recovered_loss = metric_float_any(
                    &trainer_c
                        .evaluate_materialized_head(
                            &canonical_head,
                            burn_p2p::EvalSplit::Validation,
                        )
                        .expect("restarted trainer canonical evaluation")
                        .metrics,
                    &["loss", "validation_loss"],
                );
                assert!((recovered_loss - oracle_loss).abs() <= 1.0e-8);
                restart_report = Some(serde_json::json!({
                    "kind": "trainer_process_restart_with_connectivity_outage",
                    "after_round": round + 1,
                    "offline_secs": 2.0,
                    "recovery_elapsed_secs": restart_started.elapsed().as_secs_f64(),
                    "persistent_peer_id": peer_id_after.as_str(),
                    "recovered_head_id": recovered_head.head_id.as_str(),
                    "recovered_tensor_digest": recovered_digest.as_str(),
                    "recovered_validation_loss": recovered_loss,
                }));
            }
        }

        let best_p2p_loss = best_loss(&p2p_validation_losses);
        let final_p2p_loss = *p2p_validation_losses.last().expect("final p2p loss");
        let material_improvement = best_p2p_loss <= genesis_loss - 0.01;
        let no_final_regression = final_p2p_loss <= genesis_loss + 0.01;
        let p2p_loss_reduction = genesis_loss - final_p2p_loss;
        let sequential_final_loss = sequential_validation_losses.last().copied();
        let sequential_loss_reduction = sequential_final_loss.map(|loss| genesis_loss - loss);
        let p2p_to_sequential_progress_ratio = sequential_loss_reduction
            .filter(|reduction| reduction.abs() > f64::EPSILON)
            .map(|reduction| p2p_loss_reduction / reduction);
        let synchronized_final_loss = synchronized_validation_losses.last().copied();
        let synchronized_loss_reduction = synchronized_final_loss.map(|loss| genesis_loss - loss);
        let p2p_to_synchronized_progress_ratio = synchronized_loss_reduction
            .filter(|reduction| reduction.abs() > f64::EPSILON)
            .map(|reduction| p2p_loss_reduction / reduction);
        let synchronized_reference_complete = synchronized_validation_losses.len() == rounds + 1;
        let synchronized_convergence_parity = convergence_parity_passes(
            p2p_to_synchronized_progress_ratio,
            synchronized_loss_reduction,
            minimum_synchronized_progress_ratio,
        );
        let report = serde_json::json!({
            "schema_version": 6,
            "seed": seed_value,
            "backend": "ndarray-cpu-release",
            "peer_count": 3,
            "round_count": rounds,
            "candidate_replay_enabled": replay_candidates,
            "sequential_reference_enabled": run_sequential_reference,
            "synchronized_reference_enabled": run_synchronized_reference,
            "signed_revision_contract_exercised": use_signed_revision_contract,
            "signed_revision_contract": signed_setup.as_ref().map(|(_, _, bundle)| serde_json::json!({
                "signer": bundle.contract_signature.signer.as_str(),
                "training_contract_id": bundle.training_contract_id.as_str(),
                "genesis_artifact_id": bundle.genesis.payload.payload.artifact.artifact_id.as_str(),
                "genesis_tensor_digest": bundle.genesis.payload.payload.tensor_digest.as_str(),
            })),
            "recovery_drill": restart_report.clone(),
            "aggregation": {
                "root_ema_update_basis_points": root_ema_update_basis_points,
                "root_ema_update_weight": f64::from(root_ema_update_basis_points) / 10_000.0,
            },
            "model": {
                "n_layer": RULIAD_PARITY_1M_SPEC.n_layer,
                "n_embd": RULIAD_PARITY_1M_SPEC.n_embd,
                "n_head": RULIAD_PARITY_1M_SPEC.n_head,
                "latent_total": RULIAD_PARITY_1M_SPEC.latent_total,
                "block_size": RULIAD_PARITY_1M_SPEC.block_size,
                "batch_size": RULIAD_PARITY_1M_SPEC.batch_size,
                "parameter_count": inventory.total_scalar_parameters,
                "parameter_tensor_count": inventory.parameter_count,
                "parameter_bytes": inventory.total_scalar_parameters.saturating_mul(4),
            },
            "work": {
                "peer_local_steps_per_round": peer_local_steps,
                "records_per_round": records_per_round,
                "exported_records": exported_records,
                "micro_epoch_selection": "window-rotating-bounded-stream-segments-v2",
                "aggregate_peer_local_steps": peer_local_steps
                    .saturating_mul(3)
                    .saturating_mul(rounds),
                "federated_optimizer_updates_per_peer": peer_local_steps.saturating_mul(rounds),
                "synchronized_optimizer_updates": peer_local_steps.saturating_mul(rounds),
                "sequential_optimizer_updates": peer_local_steps
                    .saturating_mul(3)
                    .saturating_mul(rounds),
            },
            "state_contract": {
                "optimizer": optimizer_state_policy,
                "scheduler": scheduler_state_policy,
            },
            "timing": {
                "training_elapsed_secs": total_training_secs,
                "promotion_elapsed_secs": total_promotion_secs,
                "protocol_cycle_duty_fraction": total_training_secs
                    / (total_training_secs + total_promotion_secs),
                "aggregate_peer_steps_per_training_second":
                    (peer_local_steps * 3 * rounds) as f64 / total_training_secs,
            },
            "training_contract_id": training_contract_id.as_str(),
            "dataset_view_id": experiment_entry.dataset_view_id.as_str(),
            "genesis_validation_loss": genesis_loss,
            "prepared_genesis_matches_canonical": prepared_genesis_matches_canonical,
            "p2p_validation_losses": p2p_validation_losses,
            "oracle_validation_losses": oracle_validation_losses,
            "sequential_validation_losses": sequential_validation_losses,
            "synchronized_validation_losses": synchronized_validation_losses,
            "best_p2p_validation_loss": best_p2p_loss,
            "final_p2p_validation_loss": final_p2p_loss,
            "convergence_comparison": {
                "p2p_loss_reduction": p2p_loss_reduction,
                "promotion_gate": {
                    "minimum_synchronized_progress_ratio": minimum_synchronized_progress_ratio,
                    "hard_assertion_enabled": require_convergence_parity,
                    "passed": run_synchronized_reference
                        .then_some(synchronized_convergence_parity),
                },
                "synchronized_reference": {
                    "kind": "centralized_same_examples_same_optimizer_update_count_gradient_accumulation_3",
                    "final_loss": synchronized_final_loss,
                    "loss_reduction": synchronized_loss_reduction,
                    "p2p_to_reference_progress_ratio": p2p_to_synchronized_progress_ratio,
                    "p2p_final_loss_minus_reference": synchronized_final_loss
                        .map(|loss| final_p2p_loss - loss),
                },
                "sequential_upper_bound": {
                    "kind": "sequential_same_examples_three_times_optimizer_updates_reset_per_lease",
                    "final_loss": sequential_final_loss,
                    "loss_reduction": sequential_loss_reduction,
                    "p2p_to_reference_progress_ratio": p2p_to_sequential_progress_ratio,
                    "p2p_final_loss_minus_reference": sequential_final_loss
                        .map(|loss| final_p2p_loss - loss),
                },
            },
            "rounds": round_reports,
            "gates": {
                "shared_training_contract": true,
                "shared_dataset_view": true,
                "disjoint_nonempty_leases": true,
                "bounded_window_rotating_micro_epochs": true,
                "bounded_stream_segments_balanced_across_shards": true,
                "all_updates_and_artifacts_propagated": true,
                "three_receipt_merge_certificates": true,
                "candidate_tensor_parity": true,
                "canonical_tensor_parity": true,
                "validation_parity": true,
                "synchronized_reference_complete": run_synchronized_reference
                    .then_some(synchronized_reference_complete),
                "synchronized_convergence_parity": run_synchronized_reference
                    .then_some(synchronized_convergence_parity),
                "strict_signed_revision_contract": use_signed_revision_contract.then_some(true),
                "restart_recovery": (restart_after_round != 0).then_some(restart_report.is_some()),
                "material_validation_improvement": material_improvement,
                "no_final_validation_regression": no_final_regression,
            },
        });
        let report_root = env_path(P2P_PARITY_REPORT_ROOT_ENV)
            .unwrap_or_else(|| PathBuf::from("target/test-artifacts/p2p-convergence-parity"));
        fs::create_dir_all(&report_root).expect("create parity report root");
        let report_path = report_root.join(format!("seed-{seed_value}.json"));
        fs::write(
            &report_path,
            serde_json::to_vec_pretty(&report).expect("serialize parity report"),
        )
        .expect("write parity report");
        eprintln!("p2p_1m_parity_report={}", report_path.display());

        assert!(
            material_improvement,
            "p2p validation did not materially improve: genesis={genesis_loss:.6} best={best_p2p_loss:.6}"
        );
        assert!(
            no_final_regression,
            "p2p validation regressed: genesis={genesis_loss:.6} final={final_p2p_loss:.6}"
        );
        if require_convergence_parity {
            assert!(
                synchronized_convergence_parity,
                "p2p convergence parity failed: progress_ratio={:?} required={minimum_synchronized_progress_ratio:.3}",
                p2p_to_synchronized_progress_ratio
            );
        }
        assert_eq!(canonical_head.global_step, rounds as u64);

        shutdown_runtime_peer(trainer_c, "parity trainer c");
        shutdown_runtime_peer(trainer_b, "parity trainer b");
        shutdown_runtime_peer(seed, "parity seed");
        bootstrap.shutdown().expect("parity bootstrap shutdown");
        bootstrap
            .await_termination()
            .expect("parity bootstrap termination");
    });
}

#[test]
#[ignore = "release-only three-peer DiLoCo convergence and transport gate"]
fn ruliad_native_runtime_1m_diloco_matches_protocol_oracle() {
    run_with_large_stack("ruliad-p2p-1m-diloco", || {
        let _guard = native_swarm_test_guard();
        let seed_value = env_u64(P2P_PARITY_SEED_ENV, 1337);
        let rounds = positive_env_usize(P2P_PARITY_ROUNDS_ENV, 2);
        let peer_local_steps = positive_env_usize(P2P_PARITY_LOCAL_STEPS_ENV, 9);
        let minimum_synchronized_progress_ratio =
            env_f64(P2P_PARITY_MIN_SYNC_PROGRESS_RATIO_ENV, 0.90);
        let require_convergence_parity = env_bool(P2P_PARITY_REQUIRE_CONVERGENCE_ENV, false);
        assert!(
            peer_local_steps <= RULIAD_PARITY_1M_SPEC.max_iters,
            "peer-local steps must not exceed the configured per-window max_iters"
        );
        assert!(
            (0.0..=1.0).contains(&minimum_synchronized_progress_ratio),
            "minimum synchronized progress ratio must be in [0, 1]"
        );
        let policy = diloco_policy_from_env(peer_local_steps);
        let policy_slug = diloco_policy_slug(&policy);
        let records_per_round = peer_local_steps
            .checked_mul(RULIAD_PARITY_1M_SPEC.batch_size)
            .and_then(|records| records.checked_mul(3))
            .expect("DiLoCo records per round");
        let exported_records = records_per_round
            .checked_mul(rounds)
            .expect("DiLoCo exported record count");

        let root = tempdir().expect("DiLoCo root");
        let bootstrap_storage = tempdir().expect("DiLoCo bootstrap storage");
        let training_config_path =
            write_ruliad_parity_training_config(root.path(), seed_value, peer_local_steps);
        let synchronized_training_config_path = write_ruliad_synchronized_reference_config(
            root.path(),
            seed_value,
            peer_local_steps
                .checked_mul(3)
                .expect("synchronized reference batch count"),
            3,
        );
        let shared_shard_root = root.path().join("shared-ruliad-shards");
        let deterministic_peer_ids = ["seed", "trainer-b", "trainer-c"].map(|role| {
            install_deterministic_test_identity(
                &root.path().join(format!("storage-{role}")),
                seed_value,
                role,
            )
        });

        let bootstrap_addr = loopback_swarm_address();
        let bootstrap_plan = burn_p2p_bootstrap::BootstrapSpec {
            preset: burn_p2p_bootstrap::BootstrapPreset::BootstrapOnly,
            genesis: burn_p2p_core::GenesisSpec {
                network_id: burn_p2p_core::NetworkId::new("dragon-p2p-diloco-testnet"),
                protocol_version: Version::new(0, 1, 0),
                display_name: "dragon 1m ruliad DiLoCo".into(),
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
        .expect("DiLoCo bootstrap plan");
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
                    transport_policy: None,
                    metrics_retention: burn_p2p::MetricsRetentionConfig::default(),
                    bootstrap_peers: Vec::new(),
                    listen_addresses: vec![bootstrap_addr.clone()],
                    external_addresses: Vec::new(),
                },
                head_artifact_mirror_source_roots: Vec::new(),
            })
            .expect("spawn DiLoCo bootstrap");
        let bootstrap_telemetry = bootstrap.telemetry();
        wait_for(
            Duration::from_secs(10),
            || {
                let snapshot = bootstrap_telemetry.snapshot();
                snapshot.local_peer_id.is_some() && !snapshot.listen_addresses.is_empty()
            },
            "DiLoCo bootstrap did not start",
        );

        let make_trainer_config =
            |label: &str, export_shared_shards: bool| DragonNativePeerConfig {
                training_overrides: burn_dragon_p2p::config::DragonNativeTrainingOverrides {
                    max_eval_batches: Some(4),
                    ..Default::default()
                },
                training_config_paths: vec![training_config_path.clone()],
                storage_root: root.path().join(format!("storage-{label}")),
                network: Default::default(),
                target: Some(DragonNativeTarget::Trainer),
                identity: burn_p2p::IdentityConfig::Persistent,
                bootstrap_peers: vec![bootstrap_addr.clone()],
                manifest: DragonManifestSeed {
                    network_id: "dragon-p2p-diloco-testnet".into(),
                    display_name: "dragon 1m ruliad DiLoCo".into(),
                    description: "three-peer 1m-class DiLoCo convergence gate".into(),
                    training_protocol: TrainingProtocol::DiLoCo(policy.clone()),
                    ..native_manifest_seed()
                },
                app_semver: semver::Version::parse("0.21.0").expect("valid burn_dragon version"),
                git_commit: Some("p2p-1m-diloco".into()),
                enabled_features_label: Some("native-cpu".into()),
                auth: None,
                capability_policy: Default::default(),
                shard_export: export_shared_shards.then(|| DragonShardExportConfig {
                    root: shared_shard_root.clone(),
                    dataset_name: Some("dragon-ruliad-diloco".into()),
                    microshards: Some(3),
                    max_records: Some(exported_records),
                    http_upstream: None,
                }),
                existing_shard_dataset: (!export_shared_shards).then(|| {
                    DragonExistingShardDatasetConfig {
                        root: shared_shard_root.clone(),
                        http_upstream: None,
                    }
                }),
            };

        let seed_prepared = prepare_nca_native_cpu(
            &make_trainer_config("seed", true),
            Some(&dummy_auth_bundle()),
        )
        .expect("prepare DiLoCo seed");
        let reference_project = BurnWorkloadAdapter::try_new(
            seed_prepared.project.clone(),
            seed_prepared.manifests.workload_config.clone(),
        )
        .expect("build DiLoCo reference adapter");
        let synchronized_project = {
            let mut config = make_trainer_config("synchronized-reference", false);
            config.training_config_paths = vec![synchronized_training_config_path];
            let prepared = prepare_nca_native_cpu(&config, Some(&dummy_auth_bundle()))
                .expect("prepare DiLoCo synchronized reference");
            BurnWorkloadAdapter::try_new(prepared.project, prepared.manifests.workload_config)
                .expect("build DiLoCo synchronized reference adapter")
        };
        let experiment_entry = seed_prepared.manifests.experiment_directory[0].clone();
        assert_eq!(
            experiment_entry.training_protocol(),
            TrainingProtocol::DiLoCo(policy.clone())
        );
        assert_eq!(
            seed_prepared.manifests.revision_manifest.training_protocol,
            TrainingProtocol::DiLoCo(policy.clone())
        );
        let trainer_b_prepared = prepare_nca_native_cpu(
            &make_trainer_config("trainer-b", false),
            Some(&dummy_auth_bundle()),
        )
        .expect("prepare DiLoCo trainer b");
        let trainer_c_prepared = prepare_nca_native_cpu(
            &make_trainer_config("trainer-c", false),
            Some(&dummy_auth_bundle()),
        )
        .expect("prepare DiLoCo trainer c");
        for prepared in [&trainer_b_prepared, &trainer_c_prepared] {
            assert_eq!(
                seed_prepared.manifests.training_contract_id,
                prepared.manifests.training_contract_id
            );
            assert_eq!(
                seed_prepared.manifests.revision_manifest,
                prepared.manifests.revision_manifest
            );
            assert_eq!(
                experiment_entry.dataset_view_id,
                prepared.manifests.experiment_directory[0].dataset_view_id
            );
        }

        let reference_device = reference_project.runtime_device();
        let initial_reference_model = reference_project.init_model(&reference_device);
        let inventory =
            burn_p2p::burn::inspect_module::<NativeCpuBackend, _>(&initial_reference_model);
        assert!(
            (800_000..=1_200_000).contains(&inventory.total_scalar_parameters),
            "DiLoCo model should be 1m-class, got {} parameters",
            inventory.total_scalar_parameters
        );
        let reference_registration = reference_project
            .dataset_registration()
            .expect("DiLoCo reference registration");
        let reference_microshard_plan = reference_project
            .microshard_plan(&reference_registration)
            .expect("DiLoCo reference microshard plan");
        assert_eq!(reference_microshard_plan.microshards.len(), 3);
        assert_eq!(
            synchronized_project.model_schema_hash(),
            reference_project.model_schema_hash()
        );
        let reference_cache_root = tempdir().expect("DiLoCo reference shard cache");
        let reference_shard_cache = ShardCache::new(reference_cache_root.path());

        let mut seed = spawn_prepared_native_peer(seed_prepared).expect("spawn DiLoCo seed");
        let mut trainer_b =
            spawn_prepared_native_peer(trainer_b_prepared).expect("spawn DiLoCo trainer b");
        let mut trainer_c =
            spawn_prepared_native_peer(trainer_c_prepared).expect("spawn DiLoCo trainer c");
        let seed_telemetry = seed.telemetry();
        let trainer_b_telemetry = trainer_b.telemetry();
        let trainer_c_telemetry = trainer_c.telemetry();
        for (label, telemetry) in [
            ("seed", &seed_telemetry),
            ("trainer-b", &trainer_b_telemetry),
            ("trainer-c", &trainer_c_telemetry),
        ] {
            wait_for(
                Duration::from_secs(30),
                || telemetry.snapshot().connected_peers >= 1,
                &format!("DiLoCo {label} did not connect"),
            );
        }
        ensure_three_peer_full_mesh(&seed, &trainer_b, &trainer_c);

        let experiment = seed.mainnet().experiment(
            experiment_entry.study_id.clone(),
            experiment_entry.experiment_id.clone(),
            experiment_entry.current_revision_id.clone(),
        );
        let genesis_head = seed
            .initialize_local_head(&experiment)
            .expect("initialize DiLoCo genesis");
        for trainer in [&trainer_b, &trainer_c] {
            wait_for(
                Duration::from_secs(45),
                || {
                    trainer
                        .sync_experiment_head(&experiment)
                        .expect("sync DiLoCo genesis")
                        .is_some()
                },
                "DiLoCo trainer did not sync genesis",
            );
        }
        let provider_peer_ids = [
            seed.snapshot().local_peer_id.expect("DiLoCo seed peer id"),
            trainer_b
                .snapshot()
                .local_peer_id
                .expect("DiLoCo trainer b peer id"),
            trainer_c
                .snapshot()
                .local_peer_id
                .expect("DiLoCo trainer c peer id"),
        ];
        assert_eq!(
            provider_peer_ids, deterministic_peer_ids,
            "runtime peer identities must match the seeded convergence fixture"
        );
        for (label, peer) in [
            ("diloco-seed", &seed),
            ("diloco-trainer-b", &trainer_b),
            ("diloco-trainer-c", &trainer_c),
        ] {
            ensure_materialized_pinned_head(
                label,
                peer,
                &experiment,
                &genesis_head,
                &provider_peer_ids,
            );
        }

        let seed_store = seed.artifact_store().expect("DiLoCo seed artifact store");
        let genesis_descriptor = seed_store
            .load_manifest(&genesis_head.artifact_id)
            .expect("load DiLoCo genesis descriptor");
        let canonical_genesis_model = reference_project
            .load_model_artifact(
                initial_reference_model,
                &genesis_descriptor,
                &seed_store,
                &reference_device,
            )
            .expect("load canonical DiLoCo genesis");
        let genesis_loss = validation_loss(&reference_project, &canonical_genesis_model);
        let outer_optimizer_state = reference_project
            .initialize_outer_optimizer_state(
                &canonical_genesis_model,
                &policy.outer_optimizer_policy,
            )
            .expect("initialize DiLoCo oracle outer optimizer");
        let initial_cursor = RoundCursor::new(
            BaseCheckpointId::from(genesis_head.head_id.clone()),
            policy.num_inner_steps,
        );
        let mut oracle_peers = provider_peer_ids
            .iter()
            .cloned()
            .map(|peer_id| DiLoCoReferencePeer {
                peer_id,
                model: canonical_genesis_model.clone(),
                outer_optimizer_state: outer_optimizer_state.clone(),
                inner_optimizer_state: None,
                round_cursor: initial_cursor.clone(),
                checkpoint_head_id: Some(genesis_head.head_id.clone()),
            })
            .collect::<Vec<_>>();
        let coordinator = DiLoCoReferenceCoordinator::new(
            experiment.experiment_id.clone(),
            experiment.revision_id.clone(),
            policy.clone(),
        )
        .with_chunk_size_bytes(1024 * 1024);
        let mut synchronized_model = canonical_genesis_model;
        let mut p2p_validation_losses = vec![genesis_loss];
        let mut synchronized_validation_losses = vec![genesis_loss];
        let mut round_reports = Vec::new();
        let mut total_network_round_secs = 0.0;
        let mut total_local_gradient_payload_bytes = 0_u64;
        let mut total_aggregate_payload_bytes = 0_u64;
        let mut total_estimated_wire_payload_bytes = 0_u64;
        let mut all_protocol_oracle_exact = true;

        for round in 0..rounds {
            let (outcomes, preparation_elapsed, network_elapsed) =
                run_three_peer_diloco_round(&experiment, &mut seed, &mut trainer_b, &mut trainer_c);
            total_network_round_secs += network_elapsed.as_secs_f64();

            let expected_round = u64::try_from(round).expect("round fits in u64");
            assert!(
                outcomes.iter().all(|outcome| {
                    outcome.completed_round.round_id.as_u64() == expected_round
                        && outcome.next_round_cursor.round_id.as_u64() == expected_round + 1
                }),
                "all peers must complete the same DiLoCo round"
            );
            assert!(
                outcomes
                    .iter()
                    .all(|outcome| outcome.group_id == outcomes[0].group_id),
                "all peers must resolve the same DiLoCo cohort"
            );
            assert!(
                outcomes
                    .iter()
                    .all(|outcome| outcome.participant_peer_ids.len() == 3),
                "all peers must commit to the same three-participant cohort"
            );
            let reducer_peer_id = outcomes[0].reducer_peer_id.clone();
            let expected_reducer_index = usize::try_from(
                outcomes[0].completed_round.round_id.as_u64()
                    % outcomes[0].participant_peer_ids.len() as u64,
            )
            .expect("reducer index fits usize");
            assert_eq!(
                reducer_peer_id, outcomes[0].participant_peer_ids[expected_reducer_index],
                "the reducer must rotate deterministically in canonical peer order"
            );
            assert!(
                outcomes
                    .iter()
                    .all(|outcome| outcome.reducer_peer_id == reducer_peer_id),
                "all peers must resolve the same round reducer"
            );
            let contribution_manifest_ids = outcomes[0].contribution_manifest_ids.clone();
            assert_eq!(contribution_manifest_ids.len(), 3);
            assert_eq!(
                contribution_manifest_ids
                    .iter()
                    .collect::<BTreeSet<_>>()
                    .len(),
                3,
                "the reducer must commit to three unique local gradients"
            );
            assert!(outcomes.iter().all(|outcome| {
                outcome.contribution_manifest_ids == contribution_manifest_ids
                    && outcome.aggregate_manifest.manifest_id
                        == outcomes[0].aggregate_manifest.manifest_id
                    && contribution_manifest_ids
                        .contains(&outcome.local_gradient_manifest.manifest_id)
                    && if outcome.local_gradient_manifest.peer_id == reducer_peer_id {
                        outcome.contributions.len() == 3
                    } else {
                        outcome.contributions.len() == 1
                    }
            }));
            assert!(
                outcomes
                    .iter()
                    .all(|outcome| outcome.current_parameters == outcomes[0].current_parameters),
                "all peers must apply the exact same outer update"
            );
            assert!(
                outcomes
                    .iter()
                    .all(|outcome| outcome.local_inner_report.steps_completed
                        == policy.num_inner_steps),
                "all peers must complete the configured inner loop"
            );

            let leases = outcomes
                .iter()
                .map(|outcome| {
                    outcome
                        .training_lease
                        .clone()
                        .expect("automatic DiLoCo round must report its data lease")
                })
                .collect::<Vec<_>>();
            assert!(
                leases
                    .iter()
                    .all(|lease| lease.dataset_view_id == experiment_entry.dataset_view_id)
            );
            assert!(
                leases
                    .iter()
                    .all(|lease| lease.window_id == leases[0].window_id)
            );
            let lease_sets = leases
                .iter()
                .map(|lease| lease.microshards.iter().cloned().collect::<BTreeSet<_>>())
                .collect::<Vec<_>>();
            assert!(lease_sets.iter().all(|set| !set.is_empty()));
            for left in 0..lease_sets.len() {
                for right in (left + 1)..lease_sets.len() {
                    assert!(
                        lease_sets[left].is_disjoint(&lease_sets[right]),
                        "DiLoCo round {} leases overlap",
                        round + 1
                    );
                }
            }

            let local_train_losses = outcomes
                .iter()
                .map(|outcome| {
                    metric_float_any(
                        &outcome.local_inner_report.metrics,
                        &["train_loss_mean", "train_loss", "loss"],
                    )
                })
                .collect::<Vec<_>>();
            assert!(local_train_losses.iter().all(|loss| loss.is_finite()));

            let peer_batches = provider_peer_ids
                .iter()
                .zip(&leases)
                .map(|(peer_id, lease)| {
                    (
                        peer_id.clone(),
                        load_reference_lease_batches(
                            &reference_project,
                            lease,
                            &reference_registration,
                            &reference_microshard_plan,
                            &reference_shard_cache,
                        ),
                    )
                })
                .collect::<BTreeMap<_, _>>();
            assert!(
                peer_batches
                    .values()
                    .all(|batches| batches.len() == peer_local_steps)
            );
            let oracle_started = Instant::now();
            let oracle_outcome = coordinator
                .run_round(&reference_project, &mut oracle_peers, &peer_batches)
                .expect("run DiLoCo protocol oracle");
            let oracle_elapsed = oracle_started.elapsed();
            let oracle_pack = reference_project
                .export_parameter_pack(&oracle_peers[0].model)
                .expect("export DiLoCo oracle parameters");
            let protocol_oracle_exact = outcomes[0].current_parameters == oracle_pack
                && outcomes[0].aggregate == oracle_outcome.aggregate;
            all_protocol_oracle_exact &= protocol_oracle_exact;
            assert!(
                protocol_oracle_exact,
                "network DiLoCo round {} diverged from the deterministic protocol oracle",
                round + 1
            );

            let p2p_model = reference_project
                .import_parameter_pack(&reference_device, &outcomes[0].current_parameters)
                .expect("import network DiLoCo parameters");
            let p2p_loss = validation_loss(&reference_project, &p2p_model);
            let checkpoint_heads = outcomes
                .iter()
                .map(|outcome| {
                    outcome
                        .published_checkpoint
                        .as_ref()
                        .expect("checkpoint interval one must publish every round")
                })
                .collect::<Vec<_>>();
            assert!(
                checkpoint_heads
                    .iter()
                    .all(|head| head.head_id == checkpoint_heads[0].head_id)
            );
            assert!(checkpoint_heads.iter().all(|head| {
                (metric_float_any(&head.metrics, &["loss", "validation_loss"]) - p2p_loss).abs()
                    <= 1.0e-8
            }));

            let lease_refs = leases.iter().collect::<Vec<_>>();
            let synchronized_started = Instant::now();
            let synchronized = train_synchronized_reference_round(
                &reference_project,
                &synchronized_project,
                synchronized_model,
                &lease_refs,
                &reference_registration,
                &reference_microshard_plan,
                &reference_shard_cache,
                peer_local_steps,
            );
            let synchronized_elapsed = synchronized_started.elapsed();
            assert_eq!(
                metric_integer(&synchronized.stats, "train_steps"),
                peer_local_steps.saturating_mul(3) as i64
            );
            synchronized_model = synchronized.model;
            let synchronized_loss = validation_loss(&synchronized_project, &synchronized_model);

            let unique_payload_bytes = outcomes
                .iter()
                .map(|outcome| outcome.local_gradient_manifest.total_encoded_bytes)
                .sum::<u64>();
            total_local_gradient_payload_bytes =
                total_local_gradient_payload_bytes.saturating_add(unique_payload_bytes);
            let reducer_pull_bytes = outcomes
                .iter()
                .filter(|outcome| outcome.local_gradient_manifest.peer_id != reducer_peer_id)
                .map(|outcome| outcome.local_gradient_manifest.total_encoded_bytes)
                .sum::<u64>();
            let aggregate_broadcast_bytes = outcomes[0]
                .aggregate_manifest
                .total_encoded_bytes
                .saturating_mul((outcomes.len() - 1) as u64);
            let estimated_wire_payload_bytes =
                reducer_pull_bytes.saturating_add(aggregate_broadcast_bytes);
            total_aggregate_payload_bytes = total_aggregate_payload_bytes
                .saturating_add(outcomes[0].aggregate_manifest.total_encoded_bytes);
            total_estimated_wire_payload_bytes =
                total_estimated_wire_payload_bytes.saturating_add(estimated_wire_payload_bytes);
            let fp32_payload_baseline_bytes = u64::try_from(
                inventory
                    .total_scalar_parameters
                    .saturating_mul(4)
                    .saturating_mul(outcomes.len()),
            )
            .expect("FP32 payload baseline fits u64");
            let max_inner_ms = outcomes
                .iter()
                .map(|outcome| outcome.timing.local_inner_loop_ms)
                .max()
                .unwrap_or_default();
            let compute_duty_fraction =
                max_inner_ms as f64 / network_elapsed.as_secs_f64().mul_add(1000.0, 0.0).max(1.0);

            p2p_validation_losses.push(p2p_loss);
            synchronized_validation_losses.push(synchronized_loss);
            round_reports.push(serde_json::json!({
                "round": round + 1,
                "round_id": outcomes[0].completed_round.round_id.as_u64(),
                "window_id": leases[0].window_id.0,
                "group_id": outcomes[0].group_id.as_str(),
                "reducer_peer_id": reducer_peer_id.as_str(),
                "participant_peer_ids": outcomes[0]
                    .participant_peer_ids
                    .iter()
                    .map(PeerId::as_str)
                    .collect::<Vec<_>>(),
                "contribution_manifest_ids": contribution_manifest_ids
                    .iter()
                    .map(ContentId::as_str)
                    .collect::<Vec<_>>(),
                "aggregate_manifest_id": outcomes[0].aggregate_manifest.manifest_id.as_str(),
                "aggregate_checksum": outcomes[0].aggregate_manifest.checksum.as_str(),
                "lease_microshards": leases
                    .iter()
                    .map(|lease| lease.microshards.iter().map(MicroShardId::as_str).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
                "local_train_losses": local_train_losses,
                "inner_steps_completed": outcomes
                    .iter()
                    .map(|outcome| outcome.local_inner_report.steps_completed)
                    .collect::<Vec<_>>(),
                "p2p_validation_loss": p2p_loss,
                "synchronized_validation_loss": synchronized_loss,
                "network_protocol_oracle_exact": protocol_oracle_exact,
                "network_parameter_checksum": outcomes[0].current_parameters.checksum().expect("network checksum").as_str(),
                "oracle_parameter_checksum": oracle_pack.checksum().expect("oracle checksum").as_str(),
                "transport": {
                    "codec": &policy.codec,
                    "unique_local_gradient_payload_bytes": unique_payload_bytes,
                    "reducer_pull_bytes_excluding_protocol_overhead": reducer_pull_bytes,
                    "aggregate_broadcast_bytes_excluding_protocol_overhead": aggregate_broadcast_bytes,
                    "estimated_wire_payload_bytes_excluding_protocol_overhead": estimated_wire_payload_bytes,
                    "fp32_unique_payload_baseline_bytes": fp32_payload_baseline_bytes,
                    "encoded_to_fp32_payload_ratio": unique_payload_bytes as f64 / fp32_payload_baseline_bytes as f64,
                    "local_gradient_chunk_counts": outcomes.iter().map(|outcome| outcome.local_gradient_manifest.chunk_count).collect::<Vec<_>>(),
                    "aggregate_chunk_count": outcomes[0].aggregate_manifest.chunk_count,
                },
                "timing": {
                    "preparation_wall_secs": preparation_elapsed.as_secs_f64(),
                    "network_round_wall_secs": network_elapsed.as_secs_f64(),
                    "protocol_oracle_secs": oracle_elapsed.as_secs_f64(),
                    "synchronized_reference_secs": synchronized_elapsed.as_secs_f64(),
                    "compute_duty_fraction": compute_duty_fraction,
                    "peers": outcomes.iter().map(|outcome| serde_json::json!({
                        "state_sync_ms": outcome.timing.state_sync_ms,
                        "matchmaking_ms": outcome.timing.matchmaking_ms,
                        "local_inner_loop_ms": outcome.timing.local_inner_loop_ms,
                        "gradient_exchange_ms": outcome.timing.gradient_exchange_ms,
                        "gradient_publish_ms": outcome.timing.gradient_publish_ms,
                        "gradient_collection_ms": outcome.timing.gradient_collection_ms,
                        "outer_apply_ms": outcome.timing.outer_apply_ms,
                        "checkpoint_publish_ms": outcome.timing.checkpoint_publish_ms,
                        "total_ms": outcome.timing.total_ms,
                    })).collect::<Vec<_>>(),
                },
            }));
            eprintln!(
                "p2p_1m_diloco_round={} policy={} train={:?} p2p_val={:.6} synchronized_val={:.6} bytes={} prepare_secs={:.3} wall_secs={:.3} duty={:.3}",
                round + 1,
                policy_slug,
                local_train_losses,
                p2p_loss,
                synchronized_loss,
                unique_payload_bytes,
                preparation_elapsed.as_secs_f64(),
                network_elapsed.as_secs_f64(),
                compute_duty_fraction,
            );
        }

        let final_p2p_loss = *p2p_validation_losses.last().expect("final DiLoCo loss");
        let synchronized_final_loss = *synchronized_validation_losses
            .last()
            .expect("final synchronized loss");
        let p2p_loss_reduction = genesis_loss - final_p2p_loss;
        let synchronized_loss_reduction = genesis_loss - synchronized_final_loss;
        let progress_ratio = (synchronized_loss_reduction > f64::EPSILON)
            .then_some(p2p_loss_reduction / synchronized_loss_reduction);
        let convergence_parity = convergence_parity_passes(
            progress_ratio,
            Some(synchronized_loss_reduction),
            minimum_synchronized_progress_ratio,
        );
        let material_improvement = best_loss(&p2p_validation_losses) <= genesis_loss - 0.01;
        let no_final_regression = final_p2p_loss <= genesis_loss + 0.01;
        let mut hard_diloco_request_failure_count = 0_u64;
        let diloco_request_failures = [
            ("seed", &seed),
            ("trainer-b", &trainer_b),
            ("trainer-c", &trainer_c),
        ]
        .into_iter()
        .map(|(label, peer)| {
            let snapshot = peer.snapshot();
            let counters = snapshot
                .request_failures
                .into_iter()
                .filter(|counter| is_diloco_request_operation(&counter.kind.operation))
                .collect::<Vec<_>>();
            let hard_failure_count = counters
                .iter()
                .filter(|counter| is_hard_request_failure_reason(&counter.kind.reason))
                .map(|counter| counter.count)
                .sum::<u64>();
            hard_diloco_request_failure_count =
                hard_diloco_request_failure_count.saturating_add(hard_failure_count);
            serde_json::json!({
                "label": label,
                "peer_id": snapshot.local_peer_id.as_ref().map(PeerId::as_str),
                "hard_failure_count": hard_failure_count,
                "counters": counters,
            })
        })
        .collect::<Vec<_>>();
        let no_hard_diloco_request_failures = hard_diloco_request_failure_count == 0;
        let build_profile = if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        };
        let report = serde_json::json!({
            "schema_version": 3,
            "experiment": "dragon_ruliad_1m_three_peer_diloco",
            "seed": seed_value,
            "backend": "ndarray-cpu",
            "build_profile": build_profile,
            "identity_fixture": "sha256-seed-role-ed25519-v1",
            "peer_count": 3,
            "round_count": rounds,
            "policy_slug": policy_slug,
            "training_protocol": TrainingProtocol::DiLoCo(policy.clone()),
            "model": {
                "n_layer": RULIAD_PARITY_1M_SPEC.n_layer,
                "n_embd": RULIAD_PARITY_1M_SPEC.n_embd,
                "n_head": RULIAD_PARITY_1M_SPEC.n_head,
                "latent_total": RULIAD_PARITY_1M_SPEC.latent_total,
                "block_size": RULIAD_PARITY_1M_SPEC.block_size,
                "batch_size": RULIAD_PARITY_1M_SPEC.batch_size,
                "parameter_count": inventory.total_scalar_parameters,
                "parameter_tensor_count": inventory.parameter_count,
                "parameter_bytes": inventory.total_scalar_parameters.saturating_mul(4),
            },
            "work": {
                "peer_local_steps_per_round": peer_local_steps,
                "records_per_round": records_per_round,
                "exported_records": exported_records,
                "aggregate_peer_local_steps": peer_local_steps.saturating_mul(3).saturating_mul(rounds),
                "peer_inner_optimizer_updates_per_peer": peer_local_steps.saturating_mul(rounds),
                "aggregate_peer_inner_optimizer_updates": peer_local_steps.saturating_mul(3).saturating_mul(rounds),
                "diloco_outer_updates": rounds,
                "synchronized_optimizer_updates": peer_local_steps.saturating_mul(rounds),
                "synchronized_microbatches_per_optimizer_update": 3,
                "synchronized_reference_semantics": "AdamW on gradients accumulated over the same three peer microbatches at each local-step index",
                "inner_optimizer_state_semantics": "burn_adapter_resets_optimizer_each_round; persistence hook is present but current learner returns the input opaque state",
            },
            "timing": {
                "network_round_wall_secs": total_network_round_secs,
                "aggregate_peer_inner_steps_per_network_second":
                    (peer_local_steps * 3 * rounds) as f64 / total_network_round_secs,
            },
            "transport": {
                "total_local_gradient_payload_bytes": total_local_gradient_payload_bytes,
                "total_aggregate_payload_bytes": total_aggregate_payload_bytes,
                "total_estimated_wire_payload_bytes_excluding_protocol_overhead": total_estimated_wire_payload_bytes,
                "control_plane_and_request_overhead_included": false,
                "multiplexer": "libp2p-yamux-current-auto-tuned",
                "configured_max_established_per_peer": 1,
                "temporary_route_reconciliation_slack": 1,
                "diloco_request_failures": diloco_request_failures,
                "hard_diloco_request_failure_count": hard_diloco_request_failure_count,
            },
            "genesis_validation_loss": genesis_loss,
            "p2p_validation_losses": p2p_validation_losses,
            "synchronized_validation_losses": synchronized_validation_losses,
            "final_p2p_validation_loss": final_p2p_loss,
            "final_synchronized_validation_loss": synchronized_final_loss,
            "convergence": {
                "p2p_loss_reduction": p2p_loss_reduction,
                "synchronized_loss_reduction": synchronized_loss_reduction,
                "p2p_to_synchronized_progress_ratio": progress_ratio,
                "minimum_progress_ratio": minimum_synchronized_progress_ratio,
            },
            "rounds": round_reports,
            "gates": {
                "deployable_manifest_protocol_bound": true,
                "automatic_disjoint_nonempty_leases": true,
                "deterministic_seeded_peer_identities": true,
                "full_trainer_mesh_before_rounds": true,
                "deterministic_single_route_reconciliation": true,
                "two_phase_cohort_state_barrier": true,
                "rotating_reducer_cohort_commitment": true,
                "uniform_transport_decode_for_lossy_codecs": true,
                "all_three_contributions_reduced": true,
                "aggregate_broadcast_exact": true,
                "all_peer_parameter_packs_exact": true,
                "network_protocol_oracle_exact": all_protocol_oracle_exact,
                "no_hard_diloco_request_failures": no_hard_diloco_request_failures,
                "material_validation_improvement": material_improvement,
                "no_final_validation_regression": no_final_regression,
                "synchronized_convergence_parity": convergence_parity,
                "hard_convergence_assertion_enabled": require_convergence_parity,
            },
        });
        let report_root = env_path(P2P_DILOCO_REPORT_ROOT_ENV)
            .unwrap_or_else(|| PathBuf::from("target/test-artifacts/p2p-diloco-convergence"));
        fs::create_dir_all(&report_root).expect("create DiLoCo report root");
        let report_path = report_root.join(format!("seed-{seed_value}-{policy_slug}.json"));
        fs::write(
            &report_path,
            serde_json::to_vec_pretty(&report).expect("serialize DiLoCo report"),
        )
        .expect("write DiLoCo report");
        eprintln!("p2p_1m_diloco_report={}", report_path.display());

        assert!(
            material_improvement,
            "DiLoCo validation did not materially improve: genesis={genesis_loss:.6} final={final_p2p_loss:.6}"
        );
        assert!(
            no_final_regression,
            "DiLoCo validation regressed: genesis={genesis_loss:.6} final={final_p2p_loss:.6}"
        );
        assert!(
            no_hard_diloco_request_failures,
            "DiLoCo transport recorded {hard_diloco_request_failure_count} hard request failure(s)"
        );
        if require_convergence_parity {
            assert!(
                convergence_parity,
                "DiLoCo convergence parity failed: progress_ratio={progress_ratio:?} required={minimum_synchronized_progress_ratio:.3}"
            );
        }

        shutdown_runtime_peer(trainer_c, "DiLoCo trainer c");
        shutdown_runtime_peer(trainer_b, "DiLoCo trainer b");
        shutdown_runtime_peer(seed, "DiLoCo seed");
        bootstrap.shutdown().expect("DiLoCo bootstrap shutdown");
        bootstrap
            .await_termination()
            .expect("DiLoCo bootstrap termination");
    });
}

#[test]
fn nca_bootstrap_only_topology_supports_diffusion_and_read_only_browser_roles() {
    run_with_large_stack("bootstrap-only-role-contract", || {
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
            training_overrides: Default::default(),
            training_config_paths: vec![training_config_path.clone()],
            storage_root: root.path().join("storage-trainer-bootstrap-only"),
            network: Default::default(),
            target: Some(DragonNativeTarget::Trainer),
            identity: Default::default(),
            bootstrap_peers: vec![bootstrap_addr],
            manifest: native_manifest_seed(),
            app_semver: semver::Version::parse("0.21.0").expect("valid burn_dragon version"),
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
        assert!(entry.allowed_roles.contains(&PeerRole::BrowserObserver));
        assert!(entry.allowed_roles.contains(&PeerRole::BrowserVerifier));
        assert!(entry.allowed_roles.contains(&PeerRole::Archive));
        assert!(entry.allowed_scopes.contains(&ExperimentScope::Archive {
            experiment_id: entry.experiment_id.clone(),
        }));
        assert!(entry.allowed_scopes.contains(&ExperimentScope::Validate {
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
    });
}

#[test]
fn nca_bootstrap_only_topology_diffusion_converges_across_trainers() {
    run_with_large_stack("bootstrap-only-diffusion", || {
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
                    transport_policy: None,
                    metrics_retention: burn_p2p::MetricsRetentionConfig::default(),
                    bootstrap_peers: Vec::new(),
                    listen_addresses: vec![bootstrap_addr.clone()],
                    external_addresses: Vec::new(),
                },
                head_artifact_mirror_source_roots: Vec::new(),
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
                training_overrides: Default::default(),
                training_config_paths: vec![training_config_path.to_path_buf()],
                storage_root: root.path().join(format!("storage-{label}")),
                network: Default::default(),
                target: Some(DragonNativeTarget::Trainer),
                identity: Default::default(),
                bootstrap_peers: vec![bootstrap_addr.clone()],
                manifest: native_manifest_seed(),
                app_semver: semver::Version::parse("0.21.0").expect("valid burn_dragon version"),
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
            experiment_entry
                .allowed_roles
                .contains(&PeerRole::BrowserVerifier)
        );
        assert!(
            experiment_entry
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
        let mut trainer_b =
            spawn_prepared_native_peer(trainer_b_prepared).expect("spawn trainer b");
        let mut trainer_c =
            spawn_prepared_native_peer(trainer_c_prepared).expect("spawn trainer c");
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
    });
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
    run_with_large_stack(
        "nca-native-auto-target",
        nca_native_auto_target_holds_trainer_role_under_tight_budget_impl,
    );
}

fn nca_native_auto_target_holds_trainer_role_under_tight_budget_impl() {
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
        training_overrides: Default::default(),
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-downgrade"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0").expect("valid burn_dragon version"),
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
fn nca_native_persisted_runtime_failure_reprepares_as_read_only_observer() {
    run_with_large_stack(
        "nca-native-runtime-downgrade",
        nca_native_persisted_runtime_failure_reprepares_as_read_only_observer_impl,
    );
}

fn nca_native_persisted_runtime_failure_reprepares_as_read_only_observer_impl() {
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
        training_overrides: Default::default(),
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-downgrade-persisted"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0").expect("valid burn_dragon version"),
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
    assert_eq!(
        downgraded
            .target_decision
            .burn_target(DragonCapabilityClass::NativeCpu)
            .roles(),
        PeerRoleSet::new([PeerRole::Viewer])
    );
    assert!(
        downgraded
            .target_decision
            .downgrade_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("persisted trainer failure")
                && reason.contains("holding observer role"))
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
    run_with_large_stack(
        "climbmix-native-existing-shards",
        climbmix_native_existing_shards_supports_multi_peer_windows_impl,
    );
}

fn climbmix_native_existing_shards_supports_multi_peer_windows_impl() {
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
        training_overrides: Default::default(),
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-peer-a"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0").expect("valid burn_dragon version"),
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
    run_with_large_stack("browser-native-manifest-conformance", || {
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
            training_overrides: Default::default(),
            training_config_paths: vec![training_config_path],
            storage_root: root.path().join("storage-browser-compat"),
            network: Default::default(),
            target: None,
            identity: Default::default(),
            bootstrap_peers: Vec::new(),
            manifest: native_manifest_seed(),
            app_semver: semver::Version::parse("0.21.0").expect("valid burn_dragon version"),
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
            local_mock_trainer_scopes(&entry),
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
        let browser_head = HeadDescriptor {
            head_id: burn_p2p::HeadId::new("dragon-head"),
            study_id: entry.study_id.clone(),
            experiment_id: entry.experiment_id.clone(),
            revision_id: entry.current_revision_id.clone(),
            artifact_id: burn_p2p::ArtifactId::new("dragon-artifact"),
            parent_head_id: None,
            global_step: 1,
            created_at: Utc::now(),
            metrics: Default::default(),
        };
        apply_canonical_browser_head(&mut harness, &browser_head);
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
                contribution: None,
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
        assert!(
            !verifier_session
                .session
                .as_ref()
                .expect("verifier session")
                .claims
                .granted_scopes
                .contains(&ExperimentScope::Train {
                    experiment_id: entry.experiment_id.clone(),
                })
        );
        verifier.select_experiment(
            entry.experiment_id.clone(),
            Some(entry.current_revision_id.clone()),
        );
        apply_canonical_browser_head(&mut verifier, &browser_head);

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
    });
}

#[test]
fn climbmix_http_shards_publish_http_input_source_descriptor() {
    run_with_large_stack("climbmix-http-shard-descriptor", || {
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
            training_overrides: Default::default(),
            training_config_paths: vec![training_config_path],
            storage_root: root.path().join("storage-http-climbmix"),
            network: Default::default(),
            target: None,
            identity: Default::default(),
            bootstrap_peers: Vec::new(),
            manifest: native_manifest_seed(),
            app_semver: semver::Version::parse("0.21.0").expect("valid burn_dragon version"),
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

        let prepared =
            prepare_climbmix_native_cpu(&native, Some(&dummy_auth_bundle())).expect("peer");
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
    });
}

#[test]
fn nca_mixed_fleet_browser_and_native_same_net_progresses() {
    run_with_large_stack(
        "nca-mixed-fleet-progress",
        nca_mixed_fleet_browser_and_native_same_net_progresses_impl,
    );
}

fn nca_mixed_fleet_browser_and_native_same_net_progresses_impl() {
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
        training_overrides: Default::default(),
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-mixed-native"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0").expect("valid burn_dragon version"),
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
        apply_canonical_browser_head(&mut trainer, &obs.head);
        apply_canonical_browser_head(&mut verifier, &obs.head);

        let training = trainer
            .run_training(BrowserTrainingPlan {
                study_id: entry.study_id.clone(),
                experiment_id: entry.experiment_id.clone(),
                revision_id: entry.current_revision_id.clone(),
                workload_id: entry.workload_id.clone(),
                budget: BrowserTrainingBudget::default(),
                lease: None,
                contribution: None,
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
fn local_browser_training_e2e() {
    run_with_large_stack(
        "local-browser-training-e2e",
        local_browser_training_e2e_impl,
    );
}

fn local_browser_training_e2e_impl() {
    let _guard = native_swarm_test_guard();
    let root = tempdir().expect("root");
    let nca_config_path = root.path().join("nca.toml");
    let training_config_path = root.path().join("nca-train.toml");
    let shard_root = root.path().join("nca-shards-local-browser-e2e");
    write(&nca_config_path, &nca_corpus_config_toml(root.path()));
    write(
        &training_config_path,
        &nca_training_config_toml(&root.path().join("nca-cache"), &nca_config_path, SMALL_SPEC),
    );

    let native = DragonNativePeerConfig {
        training_overrides: Default::default(),
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-local-browser-e2e"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0").expect("valid burn_dragon version"),
        git_commit: Some("local-browser-e2e".into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export: Some(DragonShardExportConfig {
            root: shard_root,
            dataset_name: Some("dragon-nca-local-browser-e2e".into()),
            microshards: Some(4),
            max_records: Some(32),
            http_upstream: None,
        }),
        existing_shard_dataset: None,
    };
    let prepared = prepare_nca_native_cpu(&native, Some(&dummy_auth_bundle())).expect("peer");
    run_edge_drill_for_prepared(&prepared, "local-browser-e2e");
}

#[test]
fn climbmix_mixed_fleet_browser_and_native_same_net_progresses() {
    run_with_large_stack(
        "climbmix-mixed-fleet-progress",
        climbmix_mixed_fleet_browser_and_native_same_net_progresses_impl,
    );
}

fn climbmix_mixed_fleet_browser_and_native_same_net_progresses_impl() {
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
        training_overrides: Default::default(),
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-peer-a"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0").expect("valid burn_dragon version"),
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
        apply_canonical_browser_head(&mut trainer, &obs.head);
        apply_canonical_browser_head(&mut verifier, &obs.head);

        let training = trainer
            .run_training(BrowserTrainingPlan {
                study_id: entry.study_id.clone(),
                experiment_id: entry.experiment_id.clone(),
                revision_id: entry.current_revision_id.clone(),
                workload_id: entry.workload_id.clone(),
                budget: BrowserTrainingBudget::default(),
                lease: None,
                contribution: None,
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
        training_overrides: Default::default(),
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-mixed-medium"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0").expect("valid burn_dragon version"),
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
        apply_canonical_browser_head(&mut trainer, &obs.head);
        apply_canonical_browser_head(&mut verifier, &obs.head);
        let training = trainer
            .run_training(BrowserTrainingPlan {
                study_id: entry.study_id.clone(),
                experiment_id: entry.experiment_id.clone(),
                revision_id: entry.current_revision_id.clone(),
                workload_id: entry.workload_id.clone(),
                budget: BrowserTrainingBudget::default(),
                lease: None,
                contribution: None,
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
        training_overrides: Default::default(),
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-peer-a"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0").expect("valid burn_dragon version"),
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
        apply_canonical_browser_head(&mut trainer, &obs.head);
        apply_canonical_browser_head(&mut verifier, &obs.head);

        let training = trainer
            .run_training(BrowserTrainingPlan {
                study_id: entry.study_id.clone(),
                experiment_id: entry.experiment_id.clone(),
                revision_id: entry.current_revision_id.clone(),
                workload_id: entry.workload_id.clone(),
                budget: BrowserTrainingBudget::default(),
                lease: None,
                contribution: None,
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
        training_overrides: Default::default(),
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-medium"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0").expect("valid burn_dragon version"),
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
#[ignore = "covered by the explicit ruliad convergence validation rung"]
fn ruliad_native_peer_small_model_converges_over_more_windows() {
    run_with_large_stack("ruliad-small-convergence", || {
        let _guard = native_swarm_test_guard();
        let (root, _temp_root) = ruliad_convergence_root();
        let spec = SmokeModelSpec {
            max_iters: positive_env_usize(
                RULIAD_CONVERGENCE_MAX_ITERS_ENV,
                MATCHED_512_SMALL_SPEC.max_iters,
            ),
            ..MATCHED_512_SMALL_SPEC
        };
        let training_config_path = write_ruliad_smoke_training_config(&root, spec);
        let native = native_smoke_peer_config(
            &root,
            training_config_path,
            "storage-ruliad-small",
            "ruliad-small-scale",
            None,
        );

        let prepared = prepare_nca_native_cpu(&native, Some(&dummy_auth_bundle())).expect("peer");
        let max_elapsed = positive_env_duration(RULIAD_CONVERGENCE_MAX_SECONDS_ENV);
        let windows = positive_env_usize(RULIAD_CONVERGENCE_WINDOWS_ENV, 1);
        let observations = run_training_windows_with_heads_until(
            &prepared,
            windows,
            max_elapsed,
            "ruliad-small",
            Some("ruliad_native_small_scale"),
        );
        assert!(
            !observations.is_empty(),
            "ruliad convergence run produced no windows"
        );
        for (index, observation) in observations.iter().enumerate() {
            assert_ruliad_source_selection_metrics(
                &format!("ruliad convergence window {}", index + 1),
                &observation.head.metrics,
            );
        }
        let losses = observations.iter().map(|obs| obs.loss).collect::<Vec<_>>();
        log_loss_series("ruliad_native_small_scale", &losses);
        let report = serde_json::json!({
            "comparison": "ruliad_native_peer_small_model_converges_over_more_windows",
            "requested_windows": windows,
            "completed_windows": observations.len(),
            "max_elapsed_secs": max_elapsed.map(|duration| duration.as_secs()),
            "root": root.display().to_string(),
            "matched_spec": {
                "n_layer": spec.n_layer,
                "n_embd": spec.n_embd,
                "n_head": spec.n_head,
                "latent_total": spec.latent_total,
                "block_size": spec.block_size,
                "batch_size": spec.batch_size,
                "max_iters": spec.max_iters,
            },
            "ruliad": observation_report_json("ruliad", spec, &observations),
        });
        eprintln!(
            "ruliad_native_small_model_convergence_report={}",
            serde_json::to_string_pretty(&report).expect("report json")
        );
        if losses.len() >= 2 {
            assert_material_best_improvement("small ruliad rung", &losses);
        } else {
            assert!(losses.iter().all(|loss| loss.is_finite()));
        }
    });
}

#[test]
#[ignore = "covered by the explicit ruliad convergence validation rung"]
fn nca_vs_ruliad_small_model_convergence_report() {
    run_with_large_stack("nca-vs-ruliad-report", || {
        let _guard = native_swarm_test_guard();
        let root = tempdir().expect("root");

        let nca_training_config_path =
            write_nca_smoke_training_config(root.path(), MATCHED_512_SMALL_SPEC);
        let nca_native = native_smoke_peer_config(
            root.path(),
            nca_training_config_path,
            "storage-nca-report",
            "nca-ruliad-report",
            Some(smoke_shard_export(
                root.path(),
                "nca-report-shards",
                "dragon-nca-report",
                4,
                64,
            )),
        );

        let ruliad_training_config_path =
            write_ruliad_smoke_training_config(root.path(), MATCHED_512_SMALL_SPEC);
        let ruliad_native = native_smoke_peer_config(
            root.path(),
            ruliad_training_config_path,
            "storage-ruliad-report",
            "nca-ruliad-report",
            Some(smoke_shard_export(
                root.path(),
                "ruliad-report-shards",
                "dragon-ruliad-report",
                4,
                64,
            )),
        );

        let nca =
            prepare_nca_native_cpu(&nca_native, Some(&dummy_auth_bundle())).expect("nca peer");
        let ruliad = prepare_nca_native_cpu(&ruliad_native, Some(&dummy_auth_bundle()))
            .expect("ruliad peer");
        let nca_observations = run_training_windows_with_heads(&nca, 4, "nca-report");
        let ruliad_observations = run_training_windows_with_heads(&ruliad, 4, "ruliad-report");
        let nca_losses = nca_observations
            .iter()
            .map(|obs| obs.loss)
            .collect::<Vec<_>>();
        let ruliad_losses = ruliad_observations
            .iter()
            .map(|obs| obs.loss)
            .collect::<Vec<_>>();
        log_loss_series("nca_vs_ruliad_report_nca", &nca_losses);
        log_loss_series("nca_vs_ruliad_report_ruliad", &ruliad_losses);
        assert!(nca_losses.iter().all(|loss| loss.is_finite()));
        assert!(ruliad_losses.iter().all(|loss| loss.is_finite()));

        let report = serde_json::json!({
            "comparison": "nca_vs_ruliad_small_model_convergence_report",
            "matched_spec": {
                "n_layer": MATCHED_512_SMALL_SPEC.n_layer,
                "n_embd": MATCHED_512_SMALL_SPEC.n_embd,
                "n_head": MATCHED_512_SMALL_SPEC.n_head,
                "latent_total": MATCHED_512_SMALL_SPEC.latent_total,
                "block_size": MATCHED_512_SMALL_SPEC.block_size,
                "batch_size": MATCHED_512_SMALL_SPEC.batch_size,
                "max_iters": MATCHED_512_SMALL_SPEC.max_iters,
            },
            "nca": observation_report_json("nca", MATCHED_512_SMALL_SPEC, &nca_observations),
            "ruliad": observation_report_json("ruliad", MATCHED_512_SMALL_SPEC, &ruliad_observations),
        });
        eprintln!(
            "nca_vs_ruliad_small_model_convergence_report={}",
            serde_json::to_string_pretty(&report).expect("report json")
        );
    });
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
        training_overrides: Default::default(),
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-peer-a"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0").expect("valid burn_dragon version"),
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
        training_overrides: Default::default(),
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-large"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0").expect("valid burn_dragon version"),
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
        training_overrides: Default::default(),
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-peer-a"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0").expect("valid burn_dragon version"),
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
fn native_auth_refresh_reenrolls_and_updates_cached_bundle() {
    run_with_large_stack(
        "native-auth-refresh",
        native_auth_refresh_reenrolls_and_updates_cached_bundle_impl,
    );
}

fn native_auth_refresh_reenrolls_and_updates_cached_bundle_impl() {
    let _guard = native_swarm_test_guard();
    let root = tempdir().expect("root");
    let nca_config_path = root.path().join("nca-refresh.toml");
    let training_config_path = root.path().join("nca-refresh-train.toml");
    let shard_root = root.path().join("nca-refresh-shards");
    fs::create_dir_all(&shard_root).expect("mkdir shards");
    write(&nca_config_path, &nca_corpus_config_toml(root.path()));
    write(
        &training_config_path,
        &nca_training_config_toml(
            &root.path().join("nca-refresh-cache"),
            &nca_config_path,
            SMALL_SPEC,
        ),
    );

    let native = DragonNativePeerConfig {
        training_overrides: Default::default(),
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-refresh-native"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0").expect("valid burn_dragon version"),
        git_commit: Some("auth-refresh".into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export: Some(DragonShardExportConfig {
            root: shard_root,
            dataset_name: Some("dragon-nca-refresh".into()),
            microshards: Some(4),
            max_records: Some(32),
            http_upstream: None,
        }),
        existing_shard_dataset: None,
    };
    let prepared = prepare_nca_native_cpu(&native, Some(&dummy_auth_bundle())).expect("peer");
    let edge = spawn_local_edge(edge_snapshot_for_manifests(
        &prepared.manifests,
        BrowserMode::Trainer,
    ));
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    runtime.block_on(async {
        let requested_scopes = prepared.manifests.experiment_directory[0]
            .allowed_scopes
            .clone();
        let pending = begin_native_github_login(
            &edge.base_url,
            &prepared.manifests.release_manifest,
            requested_scopes,
            1800,
            Some("refresh-native".into()),
            false,
        )
        .await
        .expect("begin native github login");

        let bridge_url =
            native_cli_bridge_url(&pending, "http://127.0.0.1:43123/callback", "nonce-refresh")
                .expect("native bridge url");
        assert!(bridge_url.starts_with("https://dragon.example/callback/github"));

        let native_session = complete_native_github_login(
            native.storage_root.as_path(),
            &pending,
            "native-provider-code",
            None,
        )
        .await
        .expect("complete native github login");
        assert!(
            native_session.auth.enrollment.is_some(),
            "completed auth should persist enrollment metadata"
        );
        assert!(
            native_session.auth.session.is_some(),
            "completed auth should persist session metadata"
        );
        assert!(
            load_cached_native_auth_bundle(native.storage_root.as_path())
                .expect("load cached auth")
                .is_some(),
            "completed auth should write the default cache file"
        );

        let mut stale = native_session.auth.clone();
        stale.certificate_not_after = Some(Utc::now() - chrono::Duration::seconds(5));
        if let Some(session) = stale.session.as_mut() {
            session.expires_at = Utc::now() - chrono::Duration::seconds(5);
        }
        assert!(
            !native_auth_bundle_is_fresh(&stale),
            "expired session metadata should force refresh"
        );
        store_cached_native_auth_bundle(native.storage_root.as_path(), &stale)
            .expect("store stale auth cache");

        let refreshed = refresh_native_auth_bundle(native.storage_root.as_path(), &stale, None)
            .await
            .expect("refresh native auth");
        assert!(
            native_auth_bundle_is_fresh(&refreshed),
            "refreshed bundle should be reusable"
        );
        assert_ne!(refreshed.session_id, stale.session_id);
        assert_eq!(
            edge.state.lock().expect("state").refresh_requests,
            1,
            "refresh endpoint should be exercised once"
        );
        let cached = load_cached_native_auth_bundle(native.storage_root.as_path())
            .expect("load refreshed cache")
            .expect("cached bundle");
        assert_eq!(cached.session_id, refreshed.session_id);
    });
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
        training_overrides: Default::default(),
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-edge-native"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0").expect("valid burn_dragon version"),
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
        training_overrides: Default::default(),
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-edge-peer-a"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::parse("0.21.0").expect("valid burn_dragon version"),
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
