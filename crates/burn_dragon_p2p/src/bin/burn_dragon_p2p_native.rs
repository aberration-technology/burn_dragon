use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::io::{BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc,
};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use burn::tensor::backend::AutodiffBackend;
use burn_dragon_language::load_training_config;
use burn_dragon_p2p::admin::{
    fetch_directory_entries, fetch_signed_directory_entries, mirror_peer_artifact,
    preserve_directory_entry_current_head, recover_directory_current_head_from_visible_roots,
    register_live_head, rollout_directory_entries, upsert_directory_entry,
    upsert_directory_entry_current_head,
};
use burn_dragon_p2p::auth::{
    DragonPendingGitHubLogin, NativeCliBridgeAuthResult, NativeCliBridgeBootstrap,
    begin_native_github_login, complete_native_github_login, default_native_auth_bundle_path,
    edge_peer_identity_for_storage, enroll_native_static_principal, fetch_edge_snapshot,
    finalize_native_auth_session_from_bridge_result, load_cached_native_auth_bundle,
    native_auth_bundle_is_fresh, native_cli_browser_auth_url, refresh_native_auth_bundle,
};
use burn_dragon_p2p::build_info;
use burn_dragon_p2p::capability_state::{
    NativeDowngradeObservation, NativeDowngradeScope, clear_native_downgrade,
    persist_native_downgrade,
};
use burn_dragon_p2p::config::{
    DragonCapabilityPolicy, DragonExperimentKind, DragonManifestBundle, DragonManifestSeed,
    DragonNativeAuthBundle, DragonNativePeerConfig, DragonNativeTarget, DragonPeerNetworkConfig,
};
use burn_dragon_p2p::deployment::{
    DeploymentDiagnosticsOptions, assert_deployment_ready, collect_deployment_diagnostics,
};
use burn_dragon_p2p::experiments::common::PreparedNativePeer;
use burn_dragon_p2p::native::{
    ManagedRunningNativePeer, assess_native_peer, prepare_climbmix_native_cpu,
    prepare_nca_native_cpu, spawn_prepared_native_peer,
};
#[cfg(feature = "cuda")]
use burn_dragon_p2p::native::{prepare_climbmix_native_cuda, prepare_nca_native_cuda};
#[cfg(feature = "rocm")]
use burn_dragon_p2p::native::{prepare_climbmix_native_rocm, prepare_nca_native_rocm};
#[cfg(feature = "wgpu")]
use burn_dragon_p2p::native::{prepare_climbmix_native_wgpu, prepare_nca_native_wgpu};
use burn_dragon_p2p::profile::DragonExperimentProfile;
use burn_dragon_p2p::profile::build_profile_from_local_config;
use burn_p2p::{
    AuthConfig, ClientPlatform, ClientReleaseManifest, ContentId, ControlPlaneSnapshot,
    ExperimentDirectoryEntry, ExperimentDirectoryPolicyExt, ExperimentHandle, ExperimentId,
    ExperimentScope, HeadAnnouncement, HeadDescriptor, HeadId, HeadPromotionMode,
    LiveControlPlaneEvent, MetricValue, NativeControlPlaneShell, NetworkId, PeerId, PeerRoleSet,
    PrincipalId, ProtocolSet, RuntimeStatus, RuntimeTransportPolicy, SwarmAddress,
};
use burn_p2p_admin::AdminResult;
use burn_p2p_core::operator_visible_last_error;
use clap::{ArgAction, CommandFactory, FromArgMatches, Parser, Subcommand, ValueEnum};
use rand::{RngCore, rngs::OsRng};
use serde::{Serialize, de::DeserializeOwned};
use url::Url;

const MIB: u64 = 1024 * 1024;
const DEFAULT_SESSION_TTL_SECS: i64 = 1800;
const DEFAULT_AUTH_CALLBACK_TIMEOUT_SECS: u64 = 300;
const DEFAULT_STATUS_INTERVAL_SECS: u64 = 30;
const DEFAULT_VALIDATION_INTERVAL_MILLIS: u64 = 250;
const DEFAULT_HEAD_SYNC_INTERVAL_SECS: u64 = 15;
const EDGE_HEAD_ARTIFACT_MIRROR_TIMEOUT_MILLIS: u64 = 10 * 60 * 1000;
const NATIVE_AUTH_CALLBACK_READ_TIMEOUT: Duration = Duration::from_secs(10);
const NATIVE_AUTH_CALLBACK_MAX_REQUEST_LINE_BYTES: usize = 8 * 1024;
const NATIVE_AUTH_CALLBACK_MAX_HEADER_LINE_BYTES: usize = 16 * 1024;
const NATIVE_AUTH_CALLBACK_MAX_HEADER_BYTES: usize = 64 * 1024;
const NATIVE_AUTH_CALLBACK_MAX_BODY_BYTES: usize = 512 * 1024;
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(15);
const STATUS_POLL_INTERVAL: Duration = Duration::from_millis(500);
const RUNTIME_READY_TIMEOUT: Duration = Duration::from_secs(10);
const TRAIN_WINDOW_P2P_CONNECTIVITY_TIMEOUT: Duration = Duration::from_secs(60);
const TRAIN_WINDOW_P2P_REDIAL_INTERVAL: Duration = Duration::from_secs(2);
const DEFAULT_TRAIN_WINDOW_HEAD_SYNC_TIMEOUT_SECS: u64 = 300;
const NATIVE_BROWSER_APP_BASE_URL_ENV: &str = "BURN_DRAGON_P2P_BROWSER_APP_BASE_URL";
const NATIVE_STORAGE_ROOT_ENV: &str = "BURN_DRAGON_P2P_NATIVE_STORAGE_ROOT";
const DEFAULT_MAINNET_EDGE_BASE_URL: &str = "https://edge.dragon.aberration.technology";
const DEFAULT_MAINNET_PROJECT_FAMILY_ID: &str = "burn-dragon-language";
const DEFAULT_MAINNET_NETWORK_ID: &str = "burn-dragon-mainnet";
const DEFAULT_MAINNET_STUDY_ID: &str = "burn-dragon-mainnet";
const DEFAULT_MAINNET_EXPERIMENT_ID: &str = "nca-prepretraining";
const DEFAULT_MAINNET_REVISION_ID: &str = "nca-r1";
const DEFAULT_MAINNET_SEED_NODE_URLS: &[&str] = &[
    "/dns4/edge.dragon.aberration.technology/tcp/4001",
    "/dns4/edge.dragon.aberration.technology/udp/4001/quic-v1",
];

#[derive(Debug, Parser)]
#[command(author, version, about = "burn_dragon native peer operator")]
struct Cli {
    #[command(subcommand)]
    command: CommandKind,
}

#[derive(Debug, Subcommand)]
enum CommandKind {
    ResolveConfig(ResolveConfigArgs),
    AssessCapability(AssessCapabilityArgs),
    DeploymentDiagnostics(DeploymentDiagnosticsArgs),
    Doctor(DoctorArgs),
    ProbeSwarm(ProbeSwarmArgs),
    BuildProfile(BuildProfileArgs),
    AdminExportDirectory(AdminExportDirectoryArgs),
    AdminRolloutProfile(AdminRolloutProfileArgs),
    #[command(alias = "github-login")]
    Login(LoginArgs),
    #[command(alias = "begin-login")]
    BeginGithubLogin(BeginGithubLoginArgs),
    #[command(alias = "complete-login")]
    CompleteGithubLogin(CompleteGithubLoginArgs),
    EnrollStaticPrincipal(EnrollStaticPrincipalArgs),
    TrainWindowOnce(TrainWindowOnceArgs),
    RunPeer(RunPeerArgs),
    RunHeadMirror(RunHeadMirrorArgs),
    RunValidatorDaemon(RunValidatorDaemonArgs),
    MarkRuntimeFailure(MarkRuntimeFailureArgs),
    ClearDowngrade(ClearDowngradeArgs),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum ConfigFormat {
    Auto,
    Toml,
    Json,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum OutputFormat {
    Toml,
    Json,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum ExperimentKindArg {
    Nca,
    Climbmix,
}

impl ExperimentKindArg {
    fn into_config(self) -> DragonExperimentKind {
        match self {
            Self::Nca => DragonExperimentKind::NcaPrepretraining,
            Self::Climbmix => DragonExperimentKind::ClimbMixPretraining,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum BackendArg {
    Cpu,
    #[value(alias = "webgpu")]
    Wgpu,
    Cuda,
    Rocm,
}

impl BackendArg {
    fn as_label(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Wgpu => "wgpu",
            Self::Cuda => "cuda",
            Self::Rocm => "rocm",
        }
    }

    fn default_enabled_features_label(self) -> &'static str {
        match self {
            Self::Cpu => "native",
            Self::Wgpu => "native,wgpu",
            Self::Cuda => "native,cuda",
            Self::Rocm => "native,rocm",
        }
    }
}

macro_rules! with_prepared_native_peer {
    ($experiment_kind:expr, $backend:expr, $config:expr, $auth_bundle:expr, |$prepared:ident| $body:expr) => {
        match ($experiment_kind, $backend) {
            (DragonExperimentKind::NcaPrepretraining, BackendArg::Cpu) => {
                let $prepared = prepare_nca_native_cpu($config, $auth_bundle)?;
                $body
            }
            (DragonExperimentKind::ClimbMixPretraining, BackendArg::Cpu) => {
                let $prepared = prepare_climbmix_native_cpu($config, $auth_bundle)?;
                $body
            }
            #[cfg(feature = "wgpu")]
            (DragonExperimentKind::NcaPrepretraining, BackendArg::Wgpu) => {
                let $prepared = prepare_nca_native_wgpu($config, $auth_bundle)?;
                $body
            }
            #[cfg(feature = "wgpu")]
            (DragonExperimentKind::ClimbMixPretraining, BackendArg::Wgpu) => {
                let $prepared = prepare_climbmix_native_wgpu($config, $auth_bundle)?;
                $body
            }
            #[cfg(feature = "cuda")]
            (DragonExperimentKind::NcaPrepretraining, BackendArg::Cuda) => {
                let $prepared = prepare_nca_native_cuda($config, $auth_bundle)?;
                $body
            }
            #[cfg(feature = "cuda")]
            (DragonExperimentKind::ClimbMixPretraining, BackendArg::Cuda) => {
                let $prepared = prepare_climbmix_native_cuda($config, $auth_bundle)?;
                $body
            }
            #[cfg(feature = "rocm")]
            (DragonExperimentKind::NcaPrepretraining, BackendArg::Rocm) => {
                let $prepared = prepare_nca_native_rocm($config, $auth_bundle)?;
                $body
            }
            #[cfg(feature = "rocm")]
            (DragonExperimentKind::ClimbMixPretraining, BackendArg::Rocm) => {
                let $prepared = prepare_climbmix_native_rocm($config, $auth_bundle)?;
                $body
            }
            #[cfg(not(feature = "wgpu"))]
            (_, BackendArg::Wgpu) => bail!("this binary was built without the `wgpu` feature"),
            #[cfg(not(feature = "cuda"))]
            (_, BackendArg::Cuda) => bail!("this binary was built without the `cuda` feature"),
            #[cfg(not(feature = "rocm"))]
            (_, BackendArg::Rocm) => bail!("this binary was built without the `rocm` feature"),
        }
    };
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum ManagedPrincipalKindArg {
    Trainer,
    Validator,
}

#[derive(Debug, Parser, Clone, Default)]
struct CapabilityPolicyArgs {
    #[arg(long)]
    native_cpu_memory_budget_mib: Option<u64>,
    #[arg(long)]
    native_wgpu_memory_budget_mib: Option<u64>,
    #[arg(long)]
    native_cuda_memory_budget_mib: Option<u64>,
    #[arg(long)]
    native_rocm_memory_budget_mib: Option<u64>,
    #[arg(long)]
    browser_wgpu_memory_budget_mib: Option<u64>,
    #[arg(long)]
    no_native_validator_fallback: bool,
    #[arg(long)]
    no_browser_verifier_fallback: bool,
}

impl CapabilityPolicyArgs {
    fn apply_to(self, mut policy: DragonCapabilityPolicy) -> DragonCapabilityPolicy {
        if let Some(value) = self.native_cpu_memory_budget_mib {
            policy.native_cpu_memory_budget_bytes = Some(value.saturating_mul(MIB));
        }
        if let Some(value) = self.native_wgpu_memory_budget_mib {
            policy.native_wgpu_memory_budget_bytes = Some(value.saturating_mul(MIB));
        }
        if let Some(value) = self.native_cuda_memory_budget_mib {
            policy.native_cuda_memory_budget_bytes = Some(value.saturating_mul(MIB));
        }
        if let Some(value) = self.native_rocm_memory_budget_mib {
            policy.native_rocm_memory_budget_bytes = Some(value.saturating_mul(MIB));
        }
        if let Some(value) = self.browser_wgpu_memory_budget_mib {
            policy.browser_wgpu_memory_budget_bytes = Some(value.saturating_mul(MIB));
        }
        if self.no_native_validator_fallback {
            policy.allow_native_validator_fallback = false;
        }
        if self.no_browser_verifier_fallback {
            policy.allow_browser_verifier_fallback = false;
        }
        policy
    }
}

#[derive(Clone, Copy, Debug, Default, Parser)]
struct NativeTrainingOverrideArgs {
    #[arg(long = "training-batch-size", value_name = "BATCH_SIZE")]
    batch_size: Option<usize>,
    #[arg(long = "training-max-iters", value_name = "ITERS")]
    max_iters: Option<usize>,
    #[arg(long = "evaluation-max-batches", value_name = "BATCHES")]
    max_eval_batches: Option<usize>,
}

impl NativeTrainingOverrideArgs {
    fn apply_to(self, config: &mut DragonNativePeerConfig) {
        if let Some(batch_size) = self.batch_size {
            config.training_overrides.batch_size = Some(batch_size);
        }
        if let Some(max_iters) = self.max_iters {
            config.training_overrides.max_iters = Some(max_iters);
        }
        if let Some(max_eval_batches) = self.max_eval_batches {
            config.training_overrides.max_eval_batches = Some(max_eval_batches);
        }
    }
}

#[derive(Debug, Parser)]
struct ResolveConfigArgs {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "auto")]
    config_format: ConfigFormat,
    #[arg(long)]
    edge_url: Option<String>,
    #[arg(long = "seed-node-url", alias = "seed", value_delimiter = ',')]
    seed_node_urls: Vec<String>,
    #[arg(long, value_enum, default_value = "toml")]
    output_format: OutputFormat,
    #[command(flatten)]
    capability_policy: CapabilityPolicyArgs,
}

#[derive(Debug, Parser)]
struct AssessCapabilityArgs {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "auto")]
    config_format: ConfigFormat,
    #[arg(long, value_enum, default_value = "nca")]
    experiment_kind: ExperimentKindArg,
    #[arg(long, value_enum, default_value = "wgpu")]
    backend: BackendArg,
    #[arg(long, value_enum, default_value = "toml")]
    output_format: OutputFormat,
    #[command(flatten)]
    capability_policy: CapabilityPolicyArgs,
}

#[derive(Debug, Parser)]
struct DeploymentDiagnosticsArgs {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "auto")]
    config_format: ConfigFormat,
    #[arg(long, value_enum, default_value = "nca")]
    experiment_kind: ExperimentKindArg,
    #[arg(long, value_enum, default_value = "wgpu")]
    backend: BackendArg,
    #[arg(long)]
    edge_url: Option<String>,
    #[arg(long = "seed-node-url", alias = "seed", value_delimiter = ',')]
    seed_node_urls: Vec<String>,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "json")]
    output_format: OutputFormat,
    #[arg(long, default_value_t = false)]
    check_metrics_catchup: bool,
    #[arg(long, default_value_t = false)]
    check_auth_authorize: bool,
    #[arg(long, default_value_t = false)]
    check_artifact_head_view: bool,
    #[arg(long, default_value_t = false)]
    require_head_published: bool,
    #[arg(long, default_value_t = false)]
    require_head_advanced: bool,
    #[arg(long, default_value_t = false)]
    require_directory_entry_published: bool,
    #[arg(long, default_value_t = false)]
    require_metrics_catchup: bool,
    #[arg(long, default_value_t = false)]
    require_auth_authorize: bool,
    #[arg(long, default_value_t = false)]
    require_artifact_head_view: bool,
    #[arg(long, default_value_t = false)]
    assert_ready: bool,
}

#[derive(Debug, Parser)]
struct DoctorArgs {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "auto")]
    config_format: ConfigFormat,
    #[arg(long, value_enum, default_value = "nca")]
    experiment_kind: ExperimentKindArg,
    #[arg(long, value_enum, default_value = "wgpu")]
    backend: BackendArg,
    #[arg(long)]
    edge_url: Option<String>,
    #[arg(long = "seed-node-url", alias = "seed", value_delimiter = ',')]
    seed_node_urls: Vec<String>,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "json")]
    output_format: OutputFormat,
    #[arg(long, default_value_t = false)]
    assert_ready: bool,
    #[command(flatten)]
    capability_policy: CapabilityPolicyArgs,
}

#[derive(Debug, Parser)]
struct ProbeSwarmArgs {
    #[arg(long, default_value = "burn-dragon-mainnet")]
    network_id: String,
    #[arg(long)]
    address: String,
    #[arg(long, default_value_t = 15)]
    timeout_secs: u64,
    #[arg(long, default_value_t = 64)]
    max_events: usize,
    #[arg(long, default_value_t = false)]
    fetch_snapshot: bool,
    #[arg(long, default_value_t = 5)]
    snapshot_timeout_secs: u64,
    #[arg(long, value_enum, default_value = "json")]
    output_format: OutputFormat,
}

#[derive(Debug, Parser)]
struct BuildProfileArgs {
    #[arg(long = "training-config", required = true)]
    training_config_paths: Vec<PathBuf>,
    #[arg(long, value_enum)]
    experiment_kind: ExperimentKindArg,
    #[arg(long)]
    revision_id: Option<String>,
    #[arg(long)]
    browser_climbmix_manifest_url: Option<String>,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "json")]
    output_format: OutputFormat,
}

#[derive(Debug, Parser)]
struct BeginGithubLoginArgs {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "auto")]
    config_format: ConfigFormat,
    #[arg(long, value_enum, default_value = "nca")]
    experiment_kind: ExperimentKindArg,
    #[arg(long, value_enum, default_value = "wgpu")]
    backend: BackendArg,
    #[arg(long)]
    edge_url: Option<String>,
    #[arg(long = "seed-node-url", alias = "seed", value_delimiter = ',')]
    seed_node_urls: Vec<String>,
    #[arg(long)]
    principal_hint: Option<String>,
    #[arg(long)]
    device_flow: bool,
    #[arg(long, default_value_t = DEFAULT_SESSION_TTL_SECS)]
    session_ttl_secs: i64,
    #[arg(long)]
    pending_out: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "json")]
    output_format: OutputFormat,
}

#[derive(Debug, Parser)]
struct AdminExportDirectoryArgs {
    #[arg(long)]
    edge_url: String,
    #[arg(long, value_enum, default_value = "json")]
    output_format: OutputFormat,
}

#[derive(Debug, Parser)]
struct AdminRolloutProfileArgs {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "auto")]
    config_format: ConfigFormat,
    #[arg(long, value_enum)]
    experiment_kind: ExperimentKindArg,
    #[arg(long, value_enum)]
    backend: BackendArg,
    #[arg(long)]
    auth_bundle: PathBuf,
    #[arg(long, value_enum, default_value = "auto")]
    auth_bundle_format: ConfigFormat,
    #[arg(long)]
    edge_url: Option<String>,
    #[arg(long, action = ArgAction::SetTrue)]
    recover_current_head_from_visible_root: bool,
    #[arg(long, action = ArgAction::SetTrue)]
    reset_current_head_to_visible_root: bool,
    #[arg(long, value_enum, default_value = "json")]
    output_format: OutputFormat,
}

#[derive(Debug, Parser)]
struct LoginArgs {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "auto")]
    config_format: ConfigFormat,
    #[arg(long, value_enum, default_value = "nca")]
    experiment_kind: ExperimentKindArg,
    #[arg(long, value_enum, default_value = "wgpu")]
    backend: BackendArg,
    #[arg(long)]
    edge_url: Option<String>,
    #[arg(long = "seed-node-url", alias = "seed", value_delimiter = ',')]
    seed_node_urls: Vec<String>,
    #[arg(long)]
    principal_hint: Option<String>,
    #[arg(long, default_value_t = DEFAULT_SESSION_TTL_SECS)]
    session_ttl_secs: i64,
    #[arg(long, default_value_t = DEFAULT_AUTH_CALLBACK_TIMEOUT_SECS)]
    callback_timeout_secs: u64,
    #[arg(long)]
    auth_bundle_out: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "json")]
    output_format: OutputFormat,
}

#[derive(Debug, Parser)]
struct CompleteGithubLoginArgs {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "auto")]
    config_format: ConfigFormat,
    #[arg(long)]
    pending: PathBuf,
    #[arg(long, value_enum, default_value = "auto")]
    pending_format: ConfigFormat,
    #[arg(long)]
    provider_code: String,
    #[arg(long)]
    auth_bundle_out: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "json")]
    output_format: OutputFormat,
}

#[derive(Debug, Parser)]
struct EnrollStaticPrincipalArgs {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "auto")]
    config_format: ConfigFormat,
    #[arg(long, value_enum)]
    experiment_kind: ExperimentKindArg,
    #[arg(long, value_enum)]
    backend: BackendArg,
    #[arg(long)]
    edge_url: Option<String>,
    #[arg(long = "seed-node-url", alias = "seed", value_delimiter = ',')]
    seed_node_urls: Vec<String>,
    #[arg(long)]
    principal_id: String,
    #[arg(long)]
    principal_hint: Option<String>,
    #[arg(long)]
    trusted_callback_token: Option<String>,
    #[arg(long, value_enum, default_value = "trainer")]
    principal_kind: ManagedPrincipalKindArg,
    #[arg(long)]
    target_artifact_hash: Option<String>,
    #[arg(long, default_value_t = DEFAULT_SESSION_TTL_SECS)]
    session_ttl_secs: i64,
    #[arg(long)]
    auth_bundle_out: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "json")]
    output_format: OutputFormat,
}

#[derive(Debug, Parser)]
struct RunPeerArgs {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "auto")]
    config_format: ConfigFormat,
    #[arg(long, value_enum, default_value = "nca")]
    experiment_kind: ExperimentKindArg,
    #[arg(long, value_enum, default_value = "wgpu")]
    backend: BackendArg,
    #[arg(long)]
    edge_url: Option<String>,
    #[arg(long = "seed-node-url", alias = "seed", value_delimiter = ',')]
    seed_node_urls: Vec<String>,
    #[arg(long)]
    auth_bundle: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "auto")]
    auth_bundle_format: ConfigFormat,
    #[arg(long, default_value_t = DEFAULT_STATUS_INTERVAL_SECS)]
    status_interval_secs: u64,
    #[arg(long, default_value_t = false)]
    initialize_head_on_start: bool,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    restore_head_on_start: bool,
    #[arg(long, default_value_t = DEFAULT_HEAD_SYNC_INTERVAL_SECS)]
    head_sync_interval_secs: u64,
    #[command(flatten)]
    capability_policy: CapabilityPolicyArgs,
}

#[derive(Debug, Parser)]
struct TrainWindowOnceArgs {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "auto")]
    config_format: ConfigFormat,
    #[arg(long, value_enum, default_value = "nca")]
    experiment_kind: ExperimentKindArg,
    #[arg(long, value_enum, default_value = "wgpu")]
    backend: BackendArg,
    #[arg(long)]
    edge_url: Option<String>,
    #[arg(long = "seed-node-url", alias = "seed", value_delimiter = ',')]
    seed_node_urls: Vec<String>,
    #[arg(long)]
    auth_bundle: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "auto")]
    auth_bundle_format: ConfigFormat,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    initialize_head_on_start: bool,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    restore_head_on_start: bool,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "json")]
    output_format: OutputFormat,
    #[arg(long, default_value_t = false)]
    require_head_advanced: bool,
    #[arg(long, default_value_t = DEFAULT_TRAIN_WINDOW_HEAD_SYNC_TIMEOUT_SECS)]
    head_sync_timeout_secs: u64,
    #[arg(long, default_value_t = false)]
    settle_diffusion: bool,
    #[arg(long, default_value_t = 3)]
    diffusion_settle_passes: u32,
    #[arg(long, default_value_t = 0)]
    serve_after_publish_secs: u64,
    #[arg(long, default_value_t = false)]
    mirror_live_head_to_edge: bool,
    #[command(flatten)]
    training_overrides: NativeTrainingOverrideArgs,
    #[command(flatten)]
    capability_policy: CapabilityPolicyArgs,
}

#[derive(Debug, Parser)]
struct RunHeadMirrorArgs {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "auto")]
    config_format: ConfigFormat,
    #[arg(long, value_enum)]
    experiment_kind: ExperimentKindArg,
    #[arg(long, value_enum, default_value = "cpu")]
    backend: BackendArg,
    #[arg(long)]
    edge_url: Option<String>,
    #[arg(long = "seed-node-url", alias = "seed", value_delimiter = ',')]
    seed_node_urls: Vec<String>,
    #[arg(long)]
    auth_bundle: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "auto")]
    auth_bundle_format: ConfigFormat,
    #[arg(long, default_value_t = DEFAULT_STATUS_INTERVAL_SECS)]
    status_interval_secs: u64,
    #[arg(long, default_value_t = DEFAULT_HEAD_SYNC_INTERVAL_SECS)]
    head_sync_interval_secs: u64,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    initialize_head_on_start: bool,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    restore_head_on_start: bool,
    #[command(flatten)]
    capability_policy: CapabilityPolicyArgs,
}

#[derive(Debug, Parser)]
struct RunValidatorDaemonArgs {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "auto")]
    config_format: ConfigFormat,
    #[arg(long, value_enum)]
    experiment_kind: ExperimentKindArg,
    #[arg(long, value_enum, default_value = "cpu")]
    backend: BackendArg,
    #[arg(long)]
    edge_url: Option<String>,
    #[arg(long = "seed-node-url", alias = "seed", value_delimiter = ',')]
    seed_node_urls: Vec<String>,
    #[arg(long)]
    auth_bundle: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "auto")]
    auth_bundle_format: ConfigFormat,
    #[arg(long, default_value_t = DEFAULT_STATUS_INTERVAL_SECS)]
    status_interval_secs: u64,
    #[arg(long, default_value_t = DEFAULT_VALIDATION_INTERVAL_MILLIS)]
    validation_interval_millis: u64,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    initialize_head_on_start: bool,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    restore_head_on_start: bool,
    #[command(flatten)]
    training_overrides: NativeTrainingOverrideArgs,
    #[command(flatten)]
    capability_policy: CapabilityPolicyArgs,
}

#[derive(Debug, Parser)]
struct MarkRuntimeFailureArgs {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "auto")]
    config_format: ConfigFormat,
    #[arg(long, value_enum)]
    experiment_kind: ExperimentKindArg,
    #[arg(long, value_enum)]
    backend: BackendArg,
    #[arg(long)]
    reason: String,
    #[arg(long, default_value = "runtime")]
    source: String,
    #[command(flatten)]
    capability_policy: CapabilityPolicyArgs,
}

#[derive(Debug, Parser)]
struct ClearDowngradeArgs {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "auto")]
    config_format: ConfigFormat,
    #[arg(long, value_enum)]
    experiment_kind: ExperimentKindArg,
    #[arg(long, value_enum)]
    backend: BackendArg,
}

#[derive(Debug, Serialize)]
struct CapabilityAssessmentReport {
    config_path: Option<PathBuf>,
    experiment_kind: DragonExperimentKind,
    backend: String,
    assessment: burn_dragon_p2p::capability::DragonNativeCapabilityAssessment,
}

#[derive(Debug, Serialize)]
struct AdminDirectoryEntryReport {
    entry: ExperimentDirectoryEntry,
    dragon_profile: Option<DragonExperimentProfile>,
}

#[derive(Debug, Serialize)]
struct AdminRolloutReport {
    edge_base_url: String,
    experiment_id: String,
    revision_id: String,
    current_head_id: Option<String>,
    preserved_current_head_id: Option<String>,
    recovered_current_head_id: Option<String>,
    reset_current_head_id: Option<String>,
    directory_entries: usize,
    result: AdminResult,
}

#[derive(Debug, Serialize)]
struct TrainWindowOnceTimingReport {
    data_fetch_time_ms: u64,
    publish_latency_ms: u64,
}

#[derive(Debug, Serialize)]
struct DiffusionSettlementReport {
    enabled: bool,
    passes_requested: u32,
    passes_completed: u32,
    served_after_publish_secs: u64,
    merge_windows: usize,
    updates: usize,
    attestations: usize,
    certificates: usize,
    merges: usize,
}

#[derive(Debug, Serialize)]
struct TrainWindowOnceReport {
    experiment_kind: DragonExperimentKind,
    backend: String,
    edge_base_url: Option<String>,
    seed_node_count: usize,
    effective_target: String,
    can_train: bool,
    downgrade_reason: Option<String>,
    local_peer_id: String,
    base_head_id: String,
    base_global_step: u64,
    published_head_id: String,
    published_global_step: u64,
    artifact_id: String,
    contribution_receipt_id: String,
    lease_window_id: String,
    lease_microshard_count: usize,
    timing: TrainWindowOnceTimingReport,
    diffusion_settlement: Option<DiffusionSettlementReport>,
    metrics: BTreeMap<String, MetricValue>,
}

#[derive(Debug, Serialize)]
struct DoctorCheck {
    name: String,
    ok: bool,
    message: String,
}

#[derive(Debug, Serialize)]
struct DoctorEdgeSnapshotReport {
    network_id: String,
    protocol_major: u16,
    minimum_client_version: String,
    auth_enabled: bool,
    directory_entries: usize,
    browser_mode: String,
}

#[derive(Debug, Serialize)]
struct DoctorReport {
    config_path: Option<PathBuf>,
    experiment_kind: DragonExperimentKind,
    backend: String,
    storage_root: PathBuf,
    edge_base_url: Option<String>,
    seed_node_count: usize,
    install_features: String,
    capability: burn_dragon_p2p::capability::DragonNativeCapabilityAssessment,
    edge_snapshot: Option<DoctorEdgeSnapshotReport>,
    checks: Vec<DoctorCheck>,
    ready: bool,
}

#[derive(Clone, Copy)]
struct TrainWindowOnceRunOptions<'a> {
    initialize_head_on_start: bool,
    restore_head_on_start: bool,
    output: Option<&'a Path>,
    output_format: OutputFormat,
    require_head_advanced: bool,
    head_sync_timeout_secs: u64,
    settle_diffusion: bool,
    diffusion_settle_passes: u32,
    serve_after_publish_secs: u64,
    mirror_live_head_to_edge: bool,
}

fn main() -> Result<()> {
    let cli = parse_cli();
    burn_dragon_p2p::logging::init_native_logging();
    log::info!(
        "burn_dragon_p2p_native starting command={}",
        command_label(&cli.command)
    );
    match cli.command {
        CommandKind::ResolveConfig(args) => resolve_config(args),
        CommandKind::AssessCapability(args) => assess_capability(args),
        CommandKind::DeploymentDiagnostics(args) => deployment_diagnostics(args),
        CommandKind::Doctor(args) => doctor(args),
        CommandKind::ProbeSwarm(args) => probe_swarm(args),
        CommandKind::BuildProfile(args) => build_profile(args),
        CommandKind::AdminExportDirectory(args) => admin_export_directory(args),
        CommandKind::AdminRolloutProfile(args) => admin_rollout_profile(args),
        CommandKind::Login(args) => login(args),
        CommandKind::BeginGithubLogin(args) => begin_github_login(args),
        CommandKind::CompleteGithubLogin(args) => complete_github_login(args),
        CommandKind::EnrollStaticPrincipal(args) => enroll_static_principal(args),
        CommandKind::TrainWindowOnce(args) => train_window_once(args),
        CommandKind::RunPeer(args) => run_peer(args),
        CommandKind::RunHeadMirror(args) => run_head_mirror(args),
        CommandKind::RunValidatorDaemon(args) => run_validator_daemon(args),
        CommandKind::MarkRuntimeFailure(args) => mark_runtime_failure(args),
        CommandKind::ClearDowngrade(args) => clear_downgrade(args),
    }
}

fn command_label(command: &CommandKind) -> &'static str {
    match command {
        CommandKind::ResolveConfig(_) => "resolve-config",
        CommandKind::AssessCapability(_) => "assess-capability",
        CommandKind::DeploymentDiagnostics(_) => "deployment-diagnostics",
        CommandKind::Doctor(_) => "doctor",
        CommandKind::ProbeSwarm(_) => "probe-swarm",
        CommandKind::BuildProfile(_) => "build-profile",
        CommandKind::AdminExportDirectory(_) => "admin-export-directory",
        CommandKind::AdminRolloutProfile(_) => "admin-rollout-profile",
        CommandKind::Login(_) => "login",
        CommandKind::BeginGithubLogin(_) => "begin-github-login",
        CommandKind::CompleteGithubLogin(_) => "complete-github-login",
        CommandKind::EnrollStaticPrincipal(_) => "enroll-static-principal",
        CommandKind::TrainWindowOnce(_) => "train-window-once",
        CommandKind::RunPeer(_) => "run-peer",
        CommandKind::RunHeadMirror(_) => "run-head-mirror",
        CommandKind::RunValidatorDaemon(_) => "run-validator-daemon",
        CommandKind::MarkRuntimeFailure(_) => "mark-runtime-failure",
        CommandKind::ClearDowngrade(_) => "clear-downgrade",
    }
}

fn parse_cli() -> Cli {
    let long_version: &'static str = Box::leak(build_info::cli_long_version().into_boxed_str());
    let matches = Cli::command().long_version(long_version).get_matches();
    Cli::from_arg_matches(&matches).unwrap_or_else(|error| error.exit())
}

#[derive(Debug, Serialize)]
struct ProbeSwarmReport {
    network_id: String,
    address: String,
    local_peer_id: String,
    connected: bool,
    connected_peer_id: Option<String>,
    elapsed_millis: u64,
    events: Vec<LiveControlPlaneEvent>,
    snapshot: Option<ProbeSwarmSnapshotSummary>,
    snapshot_error: Option<String>,
    last_error: Option<String>,
}

#[derive(Debug, Serialize)]
struct ProbeSwarmSnapshotSummary {
    head_announcements: usize,
    directory_announcements: usize,
    peer_directory_announcements: usize,
    merge_announcements: usize,
    merge_window_announcements: usize,
    update_announcements: usize,
    aggregate_proposal_announcements: usize,
    reduction_certificate_announcements: usize,
    validation_quorum_announcements: usize,
    trainer_promotion_attestation_announcements: usize,
    diffusion_promotion_certificate_announcements: usize,
    heads: Vec<ProbeSwarmHeadSummary>,
    directory_entries: Vec<ProbeSwarmDirectoryEntrySummary>,
}

#[derive(Debug, Serialize)]
struct ProbeSwarmHeadSummary {
    provider_peer_id: Option<String>,
    study_id: String,
    experiment_id: String,
    revision_id: String,
    head_id: String,
    parent_head_id: Option<String>,
    artifact_id: String,
    global_step: u64,
}

#[derive(Debug, Serialize)]
struct ProbeSwarmDirectoryEntrySummary {
    study_id: String,
    experiment_id: String,
    revision_id: String,
    current_head_id: Option<String>,
}

fn probe_swarm(args: ProbeSwarmArgs) -> Result<()> {
    let timeout = Duration::from_secs(args.timeout_secs);
    let started = Instant::now();
    let network_id = NetworkId::new(args.network_id.clone());
    let protocols = ProtocolSet::for_network(&network_id)
        .with_context(|| format!("failed to build protocol set for {}", args.network_id))?;
    let transport_policy =
        RuntimeTransportPolicy::native_for_roles(&PeerRoleSet::default_trainer());
    let mut shell = NativeControlPlaneShell::new(protocols.control, transport_policy)
        .context("failed to initialize native control-plane swarm")?;
    let local_peer_id = shell.local_peer_id().to_string();
    let address = SwarmAddress::new(args.address.clone())
        .with_context(|| format!("invalid swarm address {}", args.address))?;
    if let Some(listen_address) = probe_swarm_listen_address_for_target(address.as_str()) {
        shell
            .listen_on(SwarmAddress::new(listen_address)?)
            .with_context(|| {
                format!(
                    "failed to open required local listener before probing {}",
                    args.address
                )
            })?;
    }
    shell
        .dial(address)
        .with_context(|| format!("failed to enqueue swarm dial to {}", args.address))?;

    let deadline = Instant::now() + timeout;
    let mut connected_peer_id = None;
    let mut events = Vec::new();
    let mut last_error = None;

    while connected_peer_id.is_none() && events.len() < args.max_events {
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        let wait_for = deadline.duration_since(now).min(Duration::from_millis(500));
        let Some(event) = shell.wait_event(wait_for) else {
            continue;
        };
        match &event {
            LiveControlPlaneEvent::ConnectionEstablished { peer_id } => {
                connected_peer_id = Some(peer_id.clone());
            }
            LiveControlPlaneEvent::OutgoingConnectionError { message, .. }
            | LiveControlPlaneEvent::IncomingConnectionError { message }
            | LiveControlPlaneEvent::InboundFailure { message, .. }
            | LiveControlPlaneEvent::ResponseSendFailure { message, .. }
            | LiveControlPlaneEvent::RequestFailure { message, .. } => {
                last_error = Some(message.clone());
            }
            _ => {}
        }
        events.push(event);
    }

    let connected = connected_peer_id.is_some();
    let (snapshot, snapshot_error) = if args.fetch_snapshot {
        match connected_peer_id.as_deref() {
            Some(peer_id) => match shell.fetch_snapshot(
                peer_id,
                Duration::from_secs(args.snapshot_timeout_secs.max(1)),
            ) {
                Ok(snapshot) => (Some(probe_swarm_snapshot_summary(&snapshot)), None),
                Err(error) => (None, Some(error.to_string())),
            },
            None => (None, Some("not connected".into())),
        }
    } else {
        (None, None)
    };
    let elapsed_millis = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
    let report = ProbeSwarmReport {
        network_id: args.network_id,
        address: args.address,
        local_peer_id,
        connected,
        connected_peer_id,
        elapsed_millis,
        events,
        snapshot,
        snapshot_error,
        last_error,
    };
    write_output(None, args.output_format, &report)?;
    if !connected {
        bail!(
            "swarm probe did not establish a connection within {:?}",
            timeout
        );
    }
    Ok(())
}

fn probe_swarm_listen_address_for_target(address: &str) -> Option<&'static str> {
    if address.split('/').any(|segment| segment == "webrtc-direct") {
        Some("/ip4/0.0.0.0/udp/0/webrtc-direct")
    } else {
        None
    }
}

fn probe_swarm_snapshot_summary(snapshot: &ControlPlaneSnapshot) -> ProbeSwarmSnapshotSummary {
    let heads = snapshot
        .head_announcements
        .iter()
        .map(|announcement| ProbeSwarmHeadSummary {
            provider_peer_id: announcement
                .provider_peer_id
                .as_ref()
                .map(|peer_id| peer_id.as_str().to_owned()),
            study_id: announcement.head.study_id.as_str().to_owned(),
            experiment_id: announcement.head.experiment_id.as_str().to_owned(),
            revision_id: announcement.head.revision_id.as_str().to_owned(),
            head_id: announcement.head.head_id.as_str().to_owned(),
            parent_head_id: announcement
                .head
                .parent_head_id
                .as_ref()
                .map(|head_id| head_id.as_str().to_owned()),
            artifact_id: announcement.head.artifact_id.as_str().to_owned(),
            global_step: announcement.head.global_step,
        })
        .collect();
    let directory_entries = snapshot
        .directory_announcements
        .iter()
        .flat_map(|announcement| announcement.entries.iter())
        .map(|entry| ProbeSwarmDirectoryEntrySummary {
            study_id: entry.study_id.as_str().to_owned(),
            experiment_id: entry.experiment_id.as_str().to_owned(),
            revision_id: entry.current_revision_id.as_str().to_owned(),
            current_head_id: entry
                .current_head_id
                .as_ref()
                .map(|head_id| head_id.as_str().to_owned()),
        })
        .collect();
    ProbeSwarmSnapshotSummary {
        head_announcements: snapshot.head_announcements.len(),
        directory_announcements: snapshot.directory_announcements.len(),
        peer_directory_announcements: snapshot.peer_directory_announcements.len(),
        merge_announcements: snapshot.merge_announcements.len(),
        merge_window_announcements: snapshot.merge_window_announcements.len(),
        update_announcements: snapshot.update_announcements.len(),
        aggregate_proposal_announcements: snapshot.aggregate_proposal_announcements.len(),
        reduction_certificate_announcements: snapshot.reduction_certificate_announcements.len(),
        validation_quorum_announcements: snapshot.validation_quorum_announcements.len(),
        trainer_promotion_attestation_announcements: snapshot
            .trainer_promotion_attestation_announcements
            .len(),
        diffusion_promotion_certificate_announcements: snapshot
            .diffusion_promotion_certificate_announcements
            .len(),
        heads,
        directory_entries,
    }
}

fn build_profile(args: BuildProfileArgs) -> Result<()> {
    let config = load_training_config(&args.training_config_paths)?;
    let profile = build_profile_from_local_config(
        &config,
        args.experiment_kind.into_config(),
        args.revision_id.as_deref(),
        args.browser_climbmix_manifest_url.as_deref(),
    )?;
    write_output(args.output.as_deref(), args.output_format, &profile)
}

fn resolve_config(args: ResolveConfigArgs) -> Result<()> {
    let config = resolved_config(
        args.config.as_deref(),
        args.config_format,
        args.edge_url,
        args.seed_node_urls,
        Some(args.capability_policy),
    )?;
    write_output(None, args.output_format, &config)
}

fn assess_capability(args: AssessCapabilityArgs) -> Result<()> {
    let config = resolved_config(
        args.config.as_deref(),
        args.config_format,
        None,
        Vec::new(),
        Some(args.capability_policy),
    )?;
    let report = CapabilityAssessmentReport {
        config_path: args.config.clone(),
        experiment_kind: args.experiment_kind.into_config(),
        backend: args.backend.as_label().into(),
        assessment: assess_native_peer(
            &config,
            args.experiment_kind.into_config(),
            args.backend.as_label(),
        )?,
    };
    write_output(None, args.output_format, &report)
}

fn deployment_diagnostics(args: DeploymentDiagnosticsArgs) -> Result<()> {
    let config = resolved_config(
        args.config.as_deref(),
        args.config_format,
        args.edge_url,
        args.seed_node_urls,
        None,
    )?;
    let report = collect_deployment_diagnostics(
        &config,
        args.experiment_kind.into_config(),
        args.backend.as_label(),
        DeploymentDiagnosticsOptions {
            check_metrics_catchup: args.check_metrics_catchup,
            check_auth_authorize: args.check_auth_authorize,
            check_artifact_head_view: args.check_artifact_head_view,
            require_head_published: args.require_head_published,
            require_head_advanced: args.require_head_advanced,
            require_directory_entry_published: args.require_directory_entry_published,
            require_metrics_catchup: args.require_metrics_catchup,
            require_auth_authorize: args.require_auth_authorize,
            require_artifact_head_view: args.require_artifact_head_view,
        },
    );
    write_output(args.output.as_deref(), args.output_format, &report)?;
    if args.assert_ready {
        assert_deployment_ready(&report)?;
    }
    Ok(())
}

fn doctor(args: DoctorArgs) -> Result<()> {
    let config = resolved_config(
        args.config.as_deref(),
        args.config_format,
        args.edge_url,
        args.seed_node_urls,
        Some(args.capability_policy),
    )?;
    fs::create_dir_all(&config.storage_root).with_context(|| {
        format!(
            "failed to create native storage root {}",
            config.storage_root.display()
        )
    })?;
    let experiment_kind = args.experiment_kind.into_config();
    let backend = args.backend.as_label().to_owned();
    let capability = assess_native_peer(&config, experiment_kind, &backend)?;
    let mut checks = Vec::new();
    checks.push(DoctorCheck {
        name: "storage_root".into(),
        ok: true,
        message: config.storage_root.display().to_string(),
    });
    checks.push(DoctorCheck {
        name: "capability".into(),
        ok: capability.target_decision.can_train,
        message: capability
            .target_decision
            .downgrade_reason
            .clone()
            .unwrap_or_else(|| "native trainer capability accepted".into()),
    });
    let edge_base_url = config.effective_edge_base_url().map(ToOwned::to_owned);
    let mut edge_snapshot = None;
    if let Some(edge_url) = edge_base_url.as_deref() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("failed to build async runtime for doctor edge snapshot")?;
        match runtime.block_on(fetch_edge_snapshot(edge_url)) {
            Ok(snapshot) => {
                checks.push(DoctorCheck {
                    name: "edge_snapshot".into(),
                    ok: true,
                    message: format!(
                        "{} entries from {}",
                        snapshot.directory.entries.len(),
                        edge_url
                    ),
                });
                edge_snapshot = Some(DoctorEdgeSnapshotReport {
                    network_id: snapshot.network_id.as_str().to_owned(),
                    protocol_major: snapshot.protocol_major,
                    minimum_client_version: snapshot.minimum_client_version.to_string(),
                    auth_enabled: snapshot.auth_enabled,
                    directory_entries: snapshot.directory.entries.len(),
                    browser_mode: format!("{:?}", snapshot.browser_mode),
                });
            }
            Err(error) => checks.push(DoctorCheck {
                name: "edge_snapshot".into(),
                ok: false,
                message: error.to_string(),
            }),
        }
    } else {
        checks.push(DoctorCheck {
            name: "edge_snapshot".into(),
            ok: false,
            message: "no edge_base_url configured".into(),
        });
    }
    checks.push(DoctorCheck {
        name: "seed_nodes".into(),
        ok: !config.effective_seed_node_urls().is_empty(),
        message: format!(
            "{} configured seed(s)",
            config.effective_seed_node_urls().len()
        ),
    });
    let ready = checks.iter().all(|check| check.ok);
    let report = DoctorReport {
        config_path: args.config,
        experiment_kind,
        backend,
        storage_root: config.storage_root.clone(),
        edge_base_url,
        seed_node_count: config.effective_seed_node_urls().len(),
        install_features: args.backend.default_enabled_features_label().into(),
        capability,
        edge_snapshot,
        checks,
        ready,
    };
    write_output(args.output.as_deref(), args.output_format, &report)?;
    if args.assert_ready && !ready {
        bail!("native peer doctor checks did not pass");
    }
    Ok(())
}

fn admin_export_directory(args: AdminExportDirectoryArgs) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to build async runtime for directory export")?;
    let entries = runtime.block_on(fetch_directory_entries(&args.edge_url))?;
    let report = entries
        .into_iter()
        .map(|entry| AdminDirectoryEntryReport {
            dragon_profile: DragonExperimentProfile::from_entry_metadata(&entry)
                .ok()
                .flatten(),
            entry,
        })
        .collect::<Vec<_>>();
    write_output(None, args.output_format, &report)
}

fn admin_rollout_profile(args: AdminRolloutProfileArgs) -> Result<()> {
    let requested_edge_url = args.edge_url.clone();
    let config = resolved_config(
        args.config.as_deref(),
        args.config_format,
        requested_edge_url.clone(),
        Vec::new(),
        None,
    )?;
    let auth_bundle = resolve_or_login_native_auth_bundle(
        &config,
        args.experiment_kind.into_config(),
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
        .ok_or_else(|| anyhow!("no edge base URL configured for admin rollout"))?;
    let session_id = auth_bundle
        .session_id
        .clone()
        .ok_or_else(|| anyhow!("auth bundle is missing a session_id for admin rollout"))?;

    let local_config = config.clone().with_network_overrides(None, None);
    let manifests = prepared_manifests(
        &local_config,
        args.experiment_kind.into_config(),
        args.backend,
    )?;
    let mut replacement = manifests
        .experiment_directory
        .first()
        .cloned()
        .ok_or_else(|| anyhow!("manifest bundle is missing an experiment directory entry"))?;

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to build async runtime for admin rollout")?;
    let mut directory_entries =
        runtime.block_on(fetch_signed_directory_entries(&edge_base_url, &session_id))?;
    let preserved_current_head_id = if args.reset_current_head_to_visible_root {
        None
    } else {
        preserve_directory_entry_current_head(&directory_entries, &mut replacement)
    };
    let mut recovered_current_head_id = None;
    let mut reset_current_head_id = None;
    if args.reset_current_head_to_visible_root
        || (replacement.current_head_id.is_none() && args.recover_current_head_from_visible_root)
    {
        let snapshot = runtime.block_on(fetch_edge_snapshot(&edge_base_url))?;
        let recovered =
            recover_directory_current_head_from_visible_roots(&replacement, &snapshot.heads);
        if args.reset_current_head_to_visible_root && recovered.is_none() {
            bail!(
                "cannot reset current head for experiment={} revision={} because no visible root head was available",
                replacement.experiment_id.as_str(),
                replacement.current_revision_id.as_str()
            );
        }
        if let Some(head_id) = recovered.as_ref() {
            replacement.current_head_id = Some(head_id.clone());
        }
        if args.reset_current_head_to_visible_root {
            reset_current_head_id = recovered;
        } else {
            recovered_current_head_id = recovered;
        }
    }
    upsert_directory_entry(&mut directory_entries, replacement.clone());
    let result = runtime.block_on(rollout_directory_entries(
        &edge_base_url,
        &session_id,
        directory_entries.clone(),
    ))?;

    write_output(
        None,
        args.output_format,
        &AdminRolloutReport {
            edge_base_url,
            experiment_id: replacement.experiment_id.as_str().to_owned(),
            revision_id: replacement.current_revision_id.as_str().to_owned(),
            current_head_id: replacement
                .current_head_id
                .as_ref()
                .map(|head_id| head_id.as_str().to_owned()),
            preserved_current_head_id: preserved_current_head_id
                .as_ref()
                .map(|head_id| head_id.as_str().to_owned()),
            recovered_current_head_id: recovered_current_head_id
                .as_ref()
                .map(|head_id| head_id.as_str().to_owned()),
            reset_current_head_id: reset_current_head_id
                .as_ref()
                .map(|head_id| head_id.as_str().to_owned()),
            directory_entries: directory_entries.len(),
            result,
        },
    )
}

#[derive(Debug)]
enum NativeBrowserAuthCallback {
    ProviderCode {
        provider_code: String,
        state: String,
    },
    AuthResult(Box<NativeCliBridgeAuthResult>),
}

struct NativeBrowserAuthListener {
    callback_url: String,
    nonce: String,
    receiver: mpsc::Receiver<Result<NativeBrowserAuthCallback>>,
    stop: Arc<AtomicBool>,
    join: Option<thread::JoinHandle<()>>,
}

impl NativeBrowserAuthListener {
    fn wait(mut self, timeout: Duration) -> Result<NativeBrowserAuthCallback> {
        let callback = match self.receiver.recv_timeout(timeout) {
            Ok(result) => result,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                bail!(
                    "timed out waiting for browser auth callback after {:?}",
                    timeout
                )
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                bail!("browser auth listener terminated before delivering a callback")
            }
        }?;
        self.stop.store(true, Ordering::SeqCst);
        if let Some(join) = self.join.take() {
            join.join().expect("browser auth listener thread");
        }
        Ok(callback)
    }
}

impl Drop for NativeBrowserAuthListener {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

fn browser_auth_response_html(title: &str, message: &str) -> String {
    format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>{title}</title></head><body style=\"font-family: ui-monospace, monospace; background: #111; color: #f3f3f3; padding: 2rem;\"><h1 style=\"font-size: 1.1rem; margin-bottom: 1rem;\">{title}</h1><p>{message}</p><script>setTimeout(() => window.close(), 250);</script></body></html>"
    )
}

fn write_browser_auth_response(stream: &mut TcpStream, status: &str, body: &str) -> Result<()> {
    write!(
        stream,
        concat!(
            "HTTP/1.1 {}\r\n",
            "Content-Type: text/html; charset=utf-8\r\n",
            "Cache-Control: no-store\r\n",
            "Content-Length: {}\r\n",
            "Connection: close\r\n",
            "X-Content-Type-Options: nosniff\r\n",
            "Referrer-Policy: no-referrer\r\n",
            "\r\n{}"
        ),
        status,
        body.len(),
        body,
    )?;
    stream.flush()?;
    Ok(())
}

fn parse_native_browser_auth_callback(
    stream: &mut TcpStream,
    expected_nonce: &str,
) -> Result<NativeBrowserAuthCallback> {
    stream.set_read_timeout(Some(NATIVE_AUTH_CALLBACK_READ_TIMEOUT))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let request_line =
        read_bounded_browser_auth_line(&mut reader, NATIVE_AUTH_CALLBACK_MAX_REQUEST_LINE_BYTES)?
            .ok_or_else(|| anyhow!("browser auth callback closed before request line"))?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or_default();
    let target = parts.next().unwrap_or_default();
    if !matches!(method, "GET" | "POST") {
        let _ = write_browser_auth_response(
            stream,
            "405 Method Not Allowed",
            &browser_auth_response_html(
                "login failed",
                "the local auth callback only accepts GET or POST requests.",
            ),
        );
        bail!("browser auth callback used unsupported method {method}");
    }
    let url = Url::parse(&format!("http://127.0.0.1{target}"))
        .with_context(|| format!("failed to parse browser auth callback target {target}"))?;
    if url.path() != "/callback" {
        let _ = write_browser_auth_response(
            stream,
            "404 Not Found",
            &browser_auth_response_html("login failed", "unexpected local callback path."),
        );
        bail!("browser auth callback used unexpected path {}", url.path());
    }

    let mut content_length = 0usize;
    let mut header_bytes = 0usize;
    loop {
        let Some(header) = read_bounded_browser_auth_line(
            &mut reader,
            NATIVE_AUTH_CALLBACK_MAX_HEADER_LINE_BYTES,
        )?
        else {
            break;
        };
        header_bytes = header_bytes
            .checked_add(header.len())
            .ok_or_else(|| anyhow!("browser auth callback headers exceeded maximum size"))?;
        if header_bytes > NATIVE_AUTH_CALLBACK_MAX_HEADER_BYTES {
            bail!(
                "browser auth callback headers exceeded {} bytes",
                NATIVE_AUTH_CALLBACK_MAX_HEADER_BYTES
            );
        }
        if header == "\r\n" || header.is_empty() {
            break;
        }
        if let Some(value) = header.split_once(':')
            && value.0.eq_ignore_ascii_case("content-length")
        {
            content_length = value
                .1
                .trim()
                .parse::<usize>()
                .context("invalid browser auth callback content-length")?;
            if content_length > NATIVE_AUTH_CALLBACK_MAX_BODY_BYTES {
                bail!(
                    "browser auth callback body exceeded {} bytes",
                    NATIVE_AUTH_CALLBACK_MAX_BODY_BYTES
                );
            }
        }
    }

    let mut nonce = None;
    let mut provider_code = None;
    let mut state = None;
    let mut auth_result_json = None;
    let mut error_message = None;
    for (key, value) in url.query_pairs() {
        match key.as_ref() {
            "native_nonce" => nonce = Some(value.into_owned()),
            "provider_code" => provider_code = Some(value.into_owned()),
            "state" => state = Some(value.into_owned()),
            "auth_result_json" => auth_result_json = Some(value.into_owned()),
            "error_message" => error_message = Some(value.into_owned()),
            _ => {}
        }
    }
    if method == "POST" && content_length > 0 {
        let mut body = vec![0_u8; content_length];
        reader.read_exact(&mut body)?;
        for (key, value) in url::form_urlencoded::parse(&body) {
            match key.as_ref() {
                "native_nonce" => nonce = Some(value.into_owned()),
                "provider_code" => provider_code = Some(value.into_owned()),
                "state" => state = Some(value.into_owned()),
                "auth_result_json" => auth_result_json = Some(value.into_owned()),
                "error_message" => error_message = Some(value.into_owned()),
                _ => {}
            }
        }
    }

    if nonce.as_deref() != Some(expected_nonce) {
        let _ = write_browser_auth_response(
            stream,
            "400 Bad Request",
            &browser_auth_response_html("login failed", "the local auth nonce did not match."),
        );
        bail!("browser auth callback nonce mismatch");
    }

    if let Some(message) = error_message.filter(|value| !value.trim().is_empty()) {
        let _ = write_browser_auth_response(
            stream,
            "200 OK",
            &browser_auth_response_html("login failed", &message),
        );
        bail!("browser auth bridge failed: {message}");
    }

    if let Some(auth_result_json) = auth_result_json.filter(|value| !value.trim().is_empty()) {
        let auth_result = serde_json::from_str::<NativeCliBridgeAuthResult>(&auth_result_json)
            .context("failed to decode native auth bridge result")?;
        write_browser_auth_response(
            stream,
            "200 OK",
            &browser_auth_response_html(
                "login complete",
                "GitHub login completed. You can return to the CLI.",
            ),
        )?;
        return Ok(NativeBrowserAuthCallback::AuthResult(Box::new(auth_result)));
    }

    let provider_code = provider_code
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| anyhow!("browser auth callback is missing provider_code"))?;
    let state = state
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| anyhow!("browser auth callback is missing state"))?;
    write_browser_auth_response(
        stream,
        "200 OK",
        &browser_auth_response_html(
            "login complete",
            "GitHub login completed. You can return to the CLI.",
        ),
    )?;
    Ok(NativeBrowserAuthCallback::ProviderCode {
        provider_code,
        state,
    })
}

fn read_bounded_browser_auth_line(
    reader: &mut BufReader<TcpStream>,
    max_len: usize,
) -> Result<Option<String>> {
    let mut bytes = Vec::new();
    loop {
        let mut byte = [0_u8; 1];
        match reader.read(&mut byte)? {
            0 => {
                if bytes.is_empty() {
                    return Ok(None);
                }
                break;
            }
            _ => {
                bytes.push(byte[0]);
                if bytes.len() > max_len {
                    bail!("browser auth callback line exceeded {max_len} bytes");
                }
                if byte[0] == b'\n' {
                    break;
                }
            }
        }
    }
    String::from_utf8(bytes)
        .map(Some)
        .context("browser auth callback line was not utf-8")
}

fn random_browser_auth_nonce() -> String {
    let mut bytes = [0_u8; 16];
    OsRng.fill_bytes(&mut bytes);
    hex::encode(bytes)
}

fn start_native_browser_auth_listener() -> Result<NativeBrowserAuthListener> {
    let listener = TcpListener::bind(("127.0.0.1", 0))
        .context("failed to bind browser auth callback listener")?;
    listener
        .set_nonblocking(true)
        .context("failed to configure browser auth callback listener")?;
    let callback_url = format!(
        "http://127.0.0.1:{}/callback",
        listener.local_addr()?.port()
    );
    let nonce = random_browser_auth_nonce();
    let expected_nonce = nonce.clone();
    let (sender, receiver) = mpsc::channel();
    let stop = Arc::new(AtomicBool::new(false));
    let stop_for_thread = Arc::clone(&stop);
    let join = thread::spawn(move || {
        loop {
            if stop_for_thread.load(Ordering::SeqCst) {
                return;
            }
            match listener.accept() {
                Ok((mut stream, _)) => {
                    let result = parse_native_browser_auth_callback(&mut stream, &expected_nonce);
                    let _ = sender.send(result);
                    stop_for_thread.store(true, Ordering::SeqCst);
                    return;
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(50));
                }
                Err(error) => {
                    let _ = sender.send(Err(anyhow!(
                        "failed to accept browser auth callback: {error}"
                    )));
                    stop_for_thread.store(true, Ordering::SeqCst);
                    return;
                }
            }
        }
    });
    Ok(NativeBrowserAuthListener {
        callback_url,
        nonce,
        receiver,
        stop,
        join: Some(join),
    })
}

fn open_url_in_system_browser(url: &str) -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        let status = Command::new("open").arg(url).status()?;
        if status.success() {
            return Ok(());
        }
        bail!("open exited with status {status}");
    }

    #[cfg(target_os = "windows")]
    {
        let status = Command::new("cmd")
            .args(["/C", "start", "", url])
            .status()?;
        if status.success() {
            return Ok(());
        }
        bail!("start exited with status {status}");
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        for (program, args) in [("xdg-open", vec![url]), ("gio", vec!["open", url])] {
            match Command::new(program).args(args).status() {
                Ok(status) if status.success() => return Ok(()),
                Ok(_) | Err(_) => continue,
            }
        }
        bail!("failed to launch a system browser via xdg-open or gio open");
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows", unix)))]
    {
        bail!("automatic browser launch is not implemented on this platform");
    }
}

fn auth_bundle_output_format(path: &Path, format: ConfigFormat) -> Result<OutputFormat> {
    let format = match format {
        ConfigFormat::Auto => infer_format(path)?,
        explicit => explicit,
    };
    match format {
        ConfigFormat::Toml => Ok(OutputFormat::Toml),
        ConfigFormat::Json => Ok(OutputFormat::Json),
        ConfigFormat::Auto => unreachable!(),
    }
}

fn write_auth_bundle(
    path: &Path,
    format: ConfigFormat,
    value: &DragonNativeAuthBundle,
) -> Result<()> {
    write_output(Some(path), auth_bundle_output_format(path, format)?, value)
}

fn resolve_browser_site_base_url(
    edge_base_url: &str,
    override_base_url: Option<&str>,
) -> Result<String> {
    if let Some(base_url) = override_base_url
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        return Ok(base_url.trim_end_matches('/').to_owned());
    }
    let mut url = Url::parse(edge_base_url)
        .with_context(|| format!("failed to parse edge base URL {edge_base_url}"))?;
    let host = url
        .host_str()
        .ok_or_else(|| anyhow!("edge base URL {edge_base_url} is missing a host"))?
        .to_owned();
    let site_host = host.strip_prefix("edge.").unwrap_or(&host).to_owned();
    url.set_host(Some(&site_host)).map_err(|error| {
        anyhow!("failed to derive browser site host from {edge_base_url}: {error}")
    })?;
    Ok(url.to_string().trim_end_matches('/').to_owned())
}

fn infer_browser_site_base_url(edge_base_url: &str) -> Result<String> {
    let override_base_url = env::var(NATIVE_BROWSER_APP_BASE_URL_ENV).ok();
    resolve_browser_site_base_url(edge_base_url, override_base_url.as_deref())
}

fn build_native_cli_browser_auth_bootstrap(
    config: &DragonNativePeerConfig,
    _experiment_kind: DragonExperimentKind,
    backend: BackendArg,
    principal_hint: Option<String>,
    session_ttl_secs: i64,
) -> Result<NativeCliBridgeBootstrap> {
    let edge_base_url = config
        .effective_edge_base_url()
        .ok_or_else(|| anyhow!("no edge base URL configured"))?
        .to_owned();
    let site_base_url = infer_browser_site_base_url(&edge_base_url)?;
    let requested_scopes = requested_scopes_for_config(config);
    let (_, identity) = edge_peer_identity_for_storage(config.storage_root.as_path(), None)?;
    Ok(NativeCliBridgeBootstrap {
        edge_base_url: edge_base_url.trim_end_matches('/').to_owned(),
        site_base_url,
        target_artifact_id: native_target_artifact_id(backend).into(),
        app_semver: config.app_semver.to_string(),
        git_commit: config
            .git_commit
            .clone()
            .or_else(build_info::embedded_git_commit_owned)
            .unwrap_or_else(|| "unknown".into()),
        enabled_features_label: config
            .enabled_features_label
            .clone()
            .unwrap_or_else(|| backend.default_enabled_features_label().into()),
        requested_scopes,
        session_ttl_secs,
        principal_hint,
        identity,
    })
}

fn build_pending_native_login(
    config: &DragonNativePeerConfig,
    _experiment_kind: DragonExperimentKind,
    backend: BackendArg,
    principal_hint: Option<String>,
    session_ttl_secs: i64,
    use_device_flow: bool,
) -> Result<(tokio::runtime::Runtime, DragonPendingGitHubLogin)> {
    let edge_base_url = config
        .effective_edge_base_url()
        .ok_or_else(|| anyhow!("no edge base URL configured"))?
        .to_owned();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to build async runtime for GitHub login")?;
    let snapshot = runtime.block_on(fetch_edge_snapshot(&edge_base_url))?;
    let release_manifest = native_release_manifest_for_snapshot(config, &snapshot, backend, None)?;
    let requested_scopes = requested_scopes_for_config(config);
    let pending = runtime.block_on(begin_native_github_login(
        &edge_base_url,
        &release_manifest,
        requested_scopes,
        session_ttl_secs,
        principal_hint,
        use_device_flow,
    ))?;
    Ok((runtime, pending))
}

fn perform_interactive_native_login(
    config: &DragonNativePeerConfig,
    experiment_kind: DragonExperimentKind,
    backend: BackendArg,
    principal_hint: Option<String>,
    session_ttl_secs: i64,
    callback_timeout_secs: u64,
) -> Result<DragonNativeAuthBundle> {
    let bootstrap = build_native_cli_browser_auth_bootstrap(
        config,
        experiment_kind,
        backend,
        principal_hint.clone(),
        session_ttl_secs,
    )?;
    let listener = start_native_browser_auth_listener()?;
    let bridge_url =
        native_cli_browser_auth_url(&bootstrap, &listener.callback_url, &listener.nonce)?;
    eprintln!("Open this URL if the browser did not open automatically:\n{bridge_url}");
    match open_url_in_system_browser(&bridge_url) {
        Ok(()) => eprintln!("launched browser for GitHub login"),
        Err(error) => {
            eprintln!("automatic browser launch failed: {error}");
        }
    }
    let callback = listener.wait(Duration::from_secs(callback_timeout_secs))?;
    match callback {
        NativeBrowserAuthCallback::AuthResult(result) => {
            let session = finalize_native_auth_session_from_bridge_result(
                &config.storage_root,
                &result,
                None,
            )?;
            Ok(session.auth)
        }
        NativeBrowserAuthCallback::ProviderCode {
            provider_code,
            state,
        } => {
            eprintln!(
                "browser returned provider code only; falling back to native edge completion"
            );
            let (runtime, pending) = build_pending_native_login(
                config,
                experiment_kind,
                backend,
                principal_hint,
                session_ttl_secs,
                false,
            )?;
            if state != pending.login.state {
                bail!("browser auth callback state mismatch");
            }
            let session = runtime.block_on(complete_native_github_login(
                &config.storage_root,
                &pending,
                &provider_code,
                None,
            ))?;
            Ok(session.auth)
        }
    }
}

#[derive(Clone, Debug)]
struct NativeAuthResolutionOptions<'a> {
    auth_bundle_path: Option<&'a Path>,
    auth_bundle_format: ConfigFormat,
    principal_hint: Option<String>,
    session_ttl_secs: i64,
    callback_timeout_secs: u64,
}

fn resolve_or_login_native_auth_bundle(
    config: &DragonNativePeerConfig,
    experiment_kind: DragonExperimentKind,
    backend: BackendArg,
    options: NativeAuthResolutionOptions<'_>,
) -> Result<DragonNativeAuthBundle> {
    let mut loaded = if let Some(path) = options.auth_bundle_path {
        if path.is_file() {
            Some(load_typed::<DragonNativeAuthBundle>(
                path,
                options.auth_bundle_format,
            )?)
        } else {
            None
        }
    } else {
        load_cached_native_auth_bundle(&config.storage_root)?
    };

    if let Some(bundle) = loaded.take() {
        if native_auth_bundle_is_fresh(&bundle) {
            if let Some(path) = options.auth_bundle_path {
                write_auth_bundle(path, options.auth_bundle_format, &bundle)?;
            }
            return Ok(bundle);
        }
        match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("failed to build async runtime for auth refresh")?
            .block_on(refresh_native_auth_bundle(
                &config.storage_root,
                &bundle,
                None,
            )) {
            Ok(refreshed) => {
                if let Some(path) = options.auth_bundle_path {
                    write_auth_bundle(path, options.auth_bundle_format, &refreshed)?;
                }
                return Ok(refreshed);
            }
            Err(error) => {
                eprintln!("native auth refresh failed: {error}");
                eprintln!("falling back to interactive browser login");
            }
        }
    }

    let authenticated = perform_interactive_native_login(
        config,
        experiment_kind,
        backend,
        options.principal_hint,
        options.session_ttl_secs,
        options.callback_timeout_secs,
    )?;
    if let Some(path) = options.auth_bundle_path {
        write_auth_bundle(path, options.auth_bundle_format, &authenticated)?;
    }
    Ok(authenticated)
}

fn login(args: LoginArgs) -> Result<()> {
    let config = resolved_config(
        args.config.as_deref(),
        args.config_format,
        args.edge_url,
        args.seed_node_urls,
        None,
    )?;
    let auth = perform_interactive_native_login(
        &config,
        args.experiment_kind.into_config(),
        args.backend,
        args.principal_hint,
        args.session_ttl_secs,
        args.callback_timeout_secs,
    )?;
    eprintln!(
        "native auth cache updated: {}",
        default_native_auth_bundle_path(&config.storage_root).display()
    );
    if let Some(path) = args.auth_bundle_out.as_deref() {
        write_auth_bundle(path, ConfigFormat::Auto, &auth)?;
    }
    write_output(None, args.output_format, &auth)
}

fn begin_github_login(args: BeginGithubLoginArgs) -> Result<()> {
    let config = resolved_config(
        args.config.as_deref(),
        args.config_format,
        args.edge_url,
        args.seed_node_urls,
        None,
    )?;
    let (_runtime, pending) = build_pending_native_login(
        &config,
        args.experiment_kind.into_config(),
        args.backend,
        args.principal_hint,
        args.session_ttl_secs,
        args.device_flow,
    )?;
    if let Some(authorize_url) = pending.login.authorize_url.as_deref() {
        eprintln!("Open this URL to continue GitHub login:\n{authorize_url}");
    }
    write_output(args.pending_out.as_deref(), args.output_format, &pending)
}

fn complete_github_login(args: CompleteGithubLoginArgs) -> Result<()> {
    let config = resolved_config(
        args.config.as_deref(),
        args.config_format,
        None,
        Vec::new(),
        None,
    )?;
    let pending: DragonPendingGitHubLogin = load_typed(&args.pending, args.pending_format)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to build async runtime for GitHub login completion")?;
    let session = runtime.block_on(complete_native_github_login(
        &config.storage_root,
        &pending,
        &args.provider_code,
        None,
    ))?;
    write_output(
        args.auth_bundle_out.as_deref(),
        args.output_format,
        &session.auth,
    )
}

fn enroll_static_principal(args: EnrollStaticPrincipalArgs) -> Result<()> {
    let config = resolved_config(
        args.config.as_deref(),
        args.config_format,
        args.edge_url,
        args.seed_node_urls,
        None,
    )?;
    let edge_base_url = config
        .effective_edge_base_url()
        .ok_or_else(|| anyhow!("no edge base URL configured"))?
        .to_owned();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to build async runtime for static principal enrollment")?;
    let snapshot = runtime.block_on(fetch_edge_snapshot(&edge_base_url))?;
    let release_manifest = native_release_manifest_for_snapshot(
        &config,
        &snapshot,
        args.backend,
        args.target_artifact_hash,
    )?;
    let experiment_id = ExperimentId::new(config.manifest.experiment_id.clone());
    let requested_scopes = match args.principal_kind {
        ManagedPrincipalKindArg::Trainer => managed_trainer_scopes(&experiment_id),
        ManagedPrincipalKindArg::Validator => managed_validator_scopes(&experiment_id),
    };
    let session = runtime.block_on(enroll_native_static_principal(
        &config.storage_root,
        &edge_base_url,
        &release_manifest,
        requested_scopes,
        args.session_ttl_secs,
        args.principal_hint,
        PrincipalId::new(args.principal_id),
        args.trusted_callback_token,
        None,
    ))?;
    write_output(
        args.auth_bundle_out.as_deref(),
        args.output_format,
        &session.auth,
    )
}

fn train_window_once(args: TrainWindowOnceArgs) -> Result<()> {
    let mut config = resolved_config(
        args.config.as_deref(),
        args.config_format,
        args.edge_url,
        args.seed_node_urls,
        Some(args.capability_policy),
    )?;
    args.training_overrides.apply_to(&mut config);
    ensure_training_backend_runtime_accessible(args.backend)?;
    let auth_bundle = resolve_or_login_native_auth_bundle(
        &config,
        args.experiment_kind.into_config(),
        args.backend,
        NativeAuthResolutionOptions {
            auth_bundle_path: args.auth_bundle.as_deref(),
            auth_bundle_format: args.auth_bundle_format,
            principal_hint: None,
            session_ttl_secs: DEFAULT_SESSION_TTL_SECS,
            callback_timeout_secs: DEFAULT_AUTH_CALLBACK_TIMEOUT_SECS,
        },
    )?;
    let run_options = TrainWindowOnceRunOptions {
        initialize_head_on_start: args.initialize_head_on_start,
        restore_head_on_start: args.restore_head_on_start,
        output: args.output.as_deref(),
        output_format: args.output_format,
        require_head_advanced: args.require_head_advanced,
        head_sync_timeout_secs: args.head_sync_timeout_secs,
        settle_diffusion: args.settle_diffusion,
        diffusion_settle_passes: args.diffusion_settle_passes,
        serve_after_publish_secs: args.serve_after_publish_secs,
        mirror_live_head_to_edge: args.mirror_live_head_to_edge,
    };

    with_prepared_native_peer!(
        args.experiment_kind.into_config(),
        args.backend,
        &config,
        Some(&auth_bundle),
        |prepared| {
            run_prepared_train_window_once(
                prepared,
                &config,
                Some(&auth_bundle),
                args.backend,
                run_options,
            )
        }
    )
}

fn native_target_artifact_id(backend: BackendArg) -> &'static str {
    match backend {
        BackendArg::Cpu => "native-cpu",
        BackendArg::Wgpu => "native-wgpu",
        BackendArg::Cuda => "native-cuda",
        BackendArg::Rocm => "native-rocm",
    }
}

fn resolve_native_target_artifact_hash(
    snapshot: &burn_p2p::BrowserEdgeSnapshot,
    override_hash: Option<String>,
) -> Result<ContentId> {
    if let Some(target_artifact_hash) = override_hash.as_deref().map(str::trim)
        && !target_artifact_hash.is_empty()
    {
        return Ok(ContentId::new(target_artifact_hash));
    }

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
        bail!(
            "edge snapshot is missing allowed target artifact hashes; pass --target-artifact-hash explicitly"
        )
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
        return Ok(nativeish.into_iter().next().expect("nativeish hash exists"));
    }

    bail!(
        "edge snapshot advertises multiple target artifact hashes; pass --target-artifact-hash explicitly"
    )
}

fn native_release_manifest_for_snapshot(
    config: &DragonNativePeerConfig,
    snapshot: &burn_p2p::BrowserEdgeSnapshot,
    backend: BackendArg,
    target_artifact_hash: Option<String>,
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
        target_artifact_id: native_target_artifact_id(backend).into(),
        target_artifact_hash: resolve_native_target_artifact_hash(snapshot, target_artifact_hash)?,
        target_platform: ClientPlatform::Native,
        app_semver: config.app_semver.clone(),
        git_commit: config
            .git_commit
            .clone()
            .or_else(build_info::embedded_git_commit_owned)
            .unwrap_or_else(|| "unknown".into()),
        cargo_lock_hash: ContentId::new("dragon-native-auth-lock"),
        burn_version_string: "0.21.0".into(),
        enabled_features_hash: ContentId::new(
            config
                .enabled_features_label
                .clone()
                .unwrap_or_else(|| backend.default_enabled_features_label().into()),
        ),
        protocol_major: snapshot.protocol_major,
        supported_workloads: Vec::new(),
        built_at: chrono::Utc::now(),
    };
    release_manifest
        .validate_for_edge_snapshot(snapshot)
        .map_err(|error| {
            anyhow!("native release manifest is incompatible with edge snapshot: {error}")
        })?;
    Ok(release_manifest)
}

fn run_peer(args: RunPeerArgs) -> Result<()> {
    let config = resolved_config(
        args.config.as_deref(),
        args.config_format,
        args.edge_url,
        args.seed_node_urls,
        Some(args.capability_policy),
    )?;
    ensure_training_backend_runtime_accessible(args.backend)?;
    let auth_bundle = Some(resolve_or_login_native_auth_bundle(
        &config,
        args.experiment_kind.into_config(),
        args.backend,
        NativeAuthResolutionOptions {
            auth_bundle_path: args.auth_bundle.as_deref(),
            auth_bundle_format: args.auth_bundle_format,
            principal_hint: None,
            session_ttl_secs: DEFAULT_SESSION_TTL_SECS,
            callback_timeout_secs: DEFAULT_AUTH_CALLBACK_TIMEOUT_SECS,
        },
    )?);

    with_prepared_native_peer!(
        args.experiment_kind.into_config(),
        args.backend,
        &config,
        auth_bundle.as_ref(),
        |prepared| run_prepared_peer(
            prepared,
            &config,
            args.backend,
            args.status_interval_secs,
            args.initialize_head_on_start,
            args.restore_head_on_start,
            args.head_sync_interval_secs,
        )
    )
}

fn run_head_mirror(args: RunHeadMirrorArgs) -> Result<()> {
    let config = resolved_config(
        args.config.as_deref(),
        args.config_format,
        args.edge_url,
        args.seed_node_urls,
        Some(args.capability_policy),
    )?;
    ensure_training_backend_runtime_accessible(args.backend)?;
    let auth_bundle = Some(resolve_or_login_native_auth_bundle(
        &config,
        args.experiment_kind.into_config(),
        args.backend,
        NativeAuthResolutionOptions {
            auth_bundle_path: args.auth_bundle.as_deref(),
            auth_bundle_format: args.auth_bundle_format,
            principal_hint: None,
            session_ttl_secs: DEFAULT_SESSION_TTL_SECS,
            callback_timeout_secs: DEFAULT_AUTH_CALLBACK_TIMEOUT_SECS,
        },
    )?);

    with_prepared_native_peer!(
        args.experiment_kind.into_config(),
        args.backend,
        &config,
        auth_bundle.as_ref(),
        |prepared| run_prepared_head_mirror(
            prepared,
            &config,
            auth_bundle.as_ref(),
            args.backend,
            args.status_interval_secs,
            args.head_sync_interval_secs,
            args.initialize_head_on_start,
            args.restore_head_on_start,
        )
    )
}

fn run_validator_daemon(args: RunValidatorDaemonArgs) -> Result<()> {
    let mut config = resolved_config(
        args.config.as_deref(),
        args.config_format,
        args.edge_url,
        args.seed_node_urls,
        Some(args.capability_policy),
    )?;
    args.training_overrides.apply_to(&mut config);
    config.target = Some(DragonNativeTarget::Validator);
    ensure_training_backend_runtime_accessible(args.backend)?;
    let auth_bundle = Some(resolve_or_login_native_auth_bundle(
        &config,
        args.experiment_kind.into_config(),
        args.backend,
        NativeAuthResolutionOptions {
            auth_bundle_path: args.auth_bundle.as_deref(),
            auth_bundle_format: args.auth_bundle_format,
            principal_hint: None,
            session_ttl_secs: DEFAULT_SESSION_TTL_SECS,
            callback_timeout_secs: DEFAULT_AUTH_CALLBACK_TIMEOUT_SECS,
        },
    )?);

    with_prepared_native_peer!(
        args.experiment_kind.into_config(),
        args.backend,
        &config,
        auth_bundle.as_ref(),
        |prepared| run_prepared_validator_daemon(
            prepared,
            &config,
            args.backend,
            args.status_interval_secs,
            args.validation_interval_millis,
            args.initialize_head_on_start,
            args.restore_head_on_start,
        )
    )
}

fn mark_runtime_failure(args: MarkRuntimeFailureArgs) -> Result<()> {
    let config = resolved_config(
        args.config.as_deref(),
        args.config_format,
        None,
        Vec::new(),
        Some(args.capability_policy),
    )?;
    let assessment = assess_native_peer(
        &config,
        args.experiment_kind.into_config(),
        args.backend.as_label(),
    )?;
    let record = persist_native_downgrade(
        NativeDowngradeScope {
            storage_root: &config.storage_root,
            experiment_kind: args.experiment_kind.into_config(),
            backend_label: args.backend.as_label(),
            model_config: &assessment.model_config,
            batch_size: assessment.batch_size,
            block_size: assessment.block_size,
        },
        NativeDowngradeObservation {
            footprint: &assessment.footprint,
            trainer_budget_bytes: assessment.target_decision.trainer_memory_budget_bytes,
            downgrade_to: "trainer",
            reason: &args.reason,
            source: &args.source,
        },
    )?;
    write_output(None, OutputFormat::Json, &record)
}

fn clear_downgrade(args: ClearDowngradeArgs) -> Result<()> {
    let config = resolved_config(
        args.config.as_deref(),
        args.config_format,
        None,
        Vec::new(),
        None,
    )?;
    let assessment = assess_native_peer(
        &config,
        args.experiment_kind.into_config(),
        args.backend.as_label(),
    )?;
    clear_native_downgrade(NativeDowngradeScope {
        storage_root: &config.storage_root,
        experiment_kind: args.experiment_kind.into_config(),
        backend_label: args.backend.as_label(),
        model_config: &assessment.model_config,
        batch_size: assessment.batch_size,
        block_size: assessment.block_size,
    })?;
    Ok(())
}

fn resolved_config(
    path: Option<&Path>,
    format: ConfigFormat,
    edge_url: Option<String>,
    seed_node_urls: Vec<String>,
    capability_policy: Option<CapabilityPolicyArgs>,
) -> Result<DragonNativePeerConfig> {
    let mut config = if let Some(path) = path {
        load_native_config(path, format)?
    } else {
        default_mainnet_native_config()
    };
    config = config.with_network_overrides(
        edge_url,
        (!seed_node_urls.is_empty()).then_some(seed_node_urls),
    );
    if let Some(capability_policy) = capability_policy {
        config.capability_policy = capability_policy.apply_to(config.capability_policy.clone());
    }
    let _ = config.effective_bootstrap_peers()?;
    Ok(config)
}

fn default_mainnet_storage_root() -> PathBuf {
    if let Some(root) = env::var_os(NATIVE_STORAGE_ROOT_ENV) {
        return PathBuf::from(root);
    }
    if let Some(root) = env::var_os("XDG_DATA_HOME") {
        return PathBuf::from(root)
            .join("burn_dragon_p2p")
            .join("mainnet-native");
    }
    if let Some(home) = env::var_os("HOME") {
        return PathBuf::from(home)
            .join(".local")
            .join("share")
            .join("burn_dragon_p2p")
            .join("mainnet-native");
    }
    PathBuf::from("runs/p2p/mainnet-native")
}

fn default_mainnet_native_config() -> DragonNativePeerConfig {
    DragonNativePeerConfig {
        training_overrides: Default::default(),
        training_config_paths: Vec::new(),
        storage_root: default_mainnet_storage_root(),
        network: DragonPeerNetworkConfig::default()
            .with_edge_base_url(Some(DEFAULT_MAINNET_EDGE_BASE_URL.to_owned()))
            .with_seed_node_urls(Some(
                DEFAULT_MAINNET_SEED_NODE_URLS
                    .iter()
                    .map(|seed| (*seed).to_owned())
                    .collect(),
            )),
        target: Some(DragonNativeTarget::Trainer),
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: DragonManifestSeed {
            project_family_id: DEFAULT_MAINNET_PROJECT_FAMILY_ID.into(),
            network_id: DEFAULT_MAINNET_NETWORK_ID.into(),
            study_id: DEFAULT_MAINNET_STUDY_ID.into(),
            experiment_id: DEFAULT_MAINNET_EXPERIMENT_ID.into(),
            revision_id: DEFAULT_MAINNET_REVISION_ID.into(),
            display_name: "burn_dragon mainnet NCA pre-pre-training".into(),
            description: "burn_dragon mainnet native trainer".into(),
            ..DragonManifestSeed::default()
        },
        app_semver: semver::Version::parse(env!("CARGO_PKG_VERSION"))
            .expect("valid burn_dragon version"),
        git_commit: build_info::embedded_git_commit_owned(),
        enabled_features_label: None,
        auth: None,
        capability_policy: DragonCapabilityPolicy::default(),
        shard_export: None,
        existing_shard_dataset: None,
    }
}

fn prepared_manifests(
    config: &DragonNativePeerConfig,
    experiment_kind: DragonExperimentKind,
    backend: BackendArg,
) -> Result<DragonManifestBundle> {
    let placeholder_auth = DragonNativeAuthBundle {
        auth_config: AuthConfig::new(),
        trust_bundle_endpoint: "https://edge.invalid/trust-bundle".into(),
        edge_base_url: None,
        session_id: None,
        principal_id: None,
        enrollment: None,
        session: None,
        certificate_not_after: None,
    };
    with_prepared_native_peer!(
        experiment_kind,
        backend,
        config,
        Some(&placeholder_auth),
        |prepared| Ok(prepared.manifests)
    )
}

fn requested_scopes_for_config(config: &DragonNativePeerConfig) -> BTreeSet<ExperimentScope> {
    let experiment_id = ExperimentId::new(config.manifest.experiment_id.clone());
    match config.target_or_default() {
        DragonNativeTarget::Validator => managed_validator_scopes(&experiment_id),
        DragonNativeTarget::Auto | DragonNativeTarget::Trainer | DragonNativeTarget::Reducer => {
            standard_experiment_scopes(&experiment_id)
        }
    }
}

fn standard_experiment_scopes(experiment_id: &ExperimentId) -> BTreeSet<ExperimentScope> {
    BTreeSet::from([
        ExperimentScope::Connect,
        ExperimentScope::Discover,
        ExperimentScope::Train {
            experiment_id: experiment_id.clone(),
        },
        ExperimentScope::Archive {
            experiment_id: experiment_id.clone(),
        },
    ])
}

fn managed_trainer_scopes(experiment_id: &ExperimentId) -> BTreeSet<ExperimentScope> {
    standard_experiment_scopes(experiment_id)
}

fn managed_validator_scopes(experiment_id: &ExperimentId) -> BTreeSet<ExperimentScope> {
    BTreeSet::from([
        ExperimentScope::Connect,
        ExperimentScope::Discover,
        ExperimentScope::Validate {
            experiment_id: experiment_id.clone(),
        },
        ExperimentScope::Archive {
            experiment_id: experiment_id.clone(),
        },
    ])
}

fn ensure_training_backend_runtime_accessible(backend: BackendArg) -> Result<()> {
    #[cfg(target_os = "linux")]
    {
        match backend {
            BackendArg::Cuda => {
                if !Path::new("/dev/nvidiactl").exists() || !Path::new("/dev/nvidia0").exists() {
                    bail!(
                        "cuda backend requested but NVIDIA device nodes are not visible; expected /dev/nvidiactl and /dev/nvidia0. Run on a CUDA host/container with NVIDIA driver devices mounted, or use `login --backend cuda` separately to refresh auth without starting training."
                    );
                }
            }
            BackendArg::Rocm => {
                if !Path::new("/dev/kfd").exists() || !Path::new("/dev/dri").exists() {
                    bail!(
                        "rocm backend requested but ROCm device nodes are not visible; expected /dev/kfd and /dev/dri. Run on a ROCm host/container with AMD GPU devices mounted, or use `login --backend rocm` separately to refresh auth without starting training."
                    );
                }
            }
            BackendArg::Cpu | BackendArg::Wgpu => {}
        }
    }
    Ok(())
}

fn run_prepared_peer<B>(
    prepared: PreparedNativePeer<B>,
    config: &DragonNativePeerConfig,
    backend: BackendArg,
    status_interval_secs: u64,
    initialize_head_on_start: bool,
    restore_head_on_start: bool,
    head_sync_interval_secs: u64,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
{
    eprintln!(
        "starting burn_dragon native peer: experiment={} backend={} target={:?} can_train={} edge={} seeds={} storage_root={}",
        prepared.experiment_kind.workload_slug(),
        backend.as_label(),
        prepared.target_decision.effective_target,
        prepared.target_decision.can_train,
        config.effective_edge_base_url().unwrap_or("<none>"),
        config.effective_seed_node_urls().len(),
        config.storage_root.display(),
    );
    if let Some(reason) = prepared.target_decision.downgrade_reason.as_deref() {
        eprintln!("capability decision: {reason}");
    }

    let experiment_entry = prepared
        .manifests
        .experiment_directory
        .first()
        .cloned()
        .ok_or_else(|| anyhow!("prepared native peer is missing an experiment"))?;
    let mut running = spawn_prepared_native_peer(prepared)?;
    if initialize_head_on_start || restore_head_on_start || head_sync_interval_secs > 0 {
        wait_for_runtime_ready(&running, RUNTIME_READY_TIMEOUT)?;
        let experiment = running.mainnet().experiment(
            experiment_entry.study_id.clone(),
            experiment_entry.experiment_id.clone(),
            experiment_entry.current_revision_id.clone(),
        );
        let mut served_head_id = None;
        let _ = sync_or_initialize_latest_head_provider(
            &mut running,
            &experiment,
            initialize_head_on_start,
            restore_head_on_start,
            &mut served_head_id,
            HeadProviderSyncMode::DirectoryCurrent,
            "peer",
        )?;
    }
    let shutdown_requested = Arc::new(AtomicBool::new(false));
    let shutdown_requested_for_handler = Arc::clone(&shutdown_requested);
    let control = running.control_handle();
    ctrlc::set_handler(move || {
        if !shutdown_requested_for_handler.swap(true, Ordering::SeqCst) {
            let _ = control.shutdown();
        }
    })
    .context("failed to install ctrl-c handler")?;

    let status_interval = Duration::from_secs(status_interval_secs.max(1));
    let mut last_status = Instant::now()
        .checked_sub(status_interval)
        .unwrap_or_else(Instant::now);
    let experiment =
        if initialize_head_on_start || restore_head_on_start || head_sync_interval_secs > 0 {
            Some(running.mainnet().experiment(
                experiment_entry.study_id,
                experiment_entry.experiment_id,
                experiment_entry.current_revision_id,
            ))
        } else {
            None
        };
    let head_sync_interval = Duration::from_secs(head_sync_interval_secs.max(1));
    let mut served_head_id = None;
    let mut last_head_sync = Instant::now()
        .checked_sub(head_sync_interval)
        .unwrap_or_else(Instant::now);

    loop {
        if head_sync_interval_secs > 0
            && let Some(experiment) = experiment.as_ref()
            && last_head_sync.elapsed() >= head_sync_interval
        {
            let _ = sync_or_initialize_latest_head_provider(
                &mut running,
                experiment,
                initialize_head_on_start,
                restore_head_on_start,
                &mut served_head_id,
                HeadProviderSyncMode::DirectoryCurrent,
                "peer",
            )?;
            last_head_sync = Instant::now();
        }

        let snapshot = running.snapshot();
        if status_interval_secs > 0 && last_status.elapsed() >= status_interval {
            eprintln!(
                "peer-status status={:?} node_state={:?} connected_peers={} last_error={}",
                snapshot.status,
                snapshot.node_state,
                snapshot.connected_peers,
                operator_visible_last_error(snapshot.last_error.as_deref())
                    .as_deref()
                    .unwrap_or("-"),
            );
            last_status = Instant::now();
        }

        match snapshot.status {
            RuntimeStatus::Failed => {
                let reason = snapshot
                    .last_error
                    .unwrap_or_else(|| "peer runtime failed".into());
                let _ = running.shutdown();
                let _ = running.await_termination_timeout(SHUTDOWN_TIMEOUT);
                bail!("peer runtime failed: {reason}");
            }
            RuntimeStatus::Stopped => {
                let _prepared = running.await_termination_timeout(SHUTDOWN_TIMEOUT)?;
                eprintln!("peer stopped cleanly");
                return Ok(());
            }
            _ => {}
        }

        thread::sleep(STATUS_POLL_INTERVAL);
    }
}

fn run_prepared_train_window_once<B>(
    prepared: PreparedNativePeer<B>,
    config: &DragonNativePeerConfig,
    auth_bundle: Option<&DragonNativeAuthBundle>,
    backend: BackendArg,
    options: TrainWindowOnceRunOptions<'_>,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
{
    let experiment_entry = prepared
        .manifests
        .experiment_directory
        .first()
        .cloned()
        .ok_or_else(|| anyhow!("prepared native peer is missing an experiment"))?;
    eprintln!(
        "starting burn_dragon train-window-once: experiment={} backend={} target={:?} can_train={} edge={} seeds={} storage_root={}",
        prepared.experiment_kind.workload_slug(),
        backend.as_label(),
        prepared.target_decision.effective_target,
        prepared.target_decision.can_train,
        config.effective_edge_base_url().unwrap_or("<none>"),
        config.effective_seed_node_urls().len(),
        config.storage_root.display(),
    );
    if let Some(reason) = prepared.target_decision.downgrade_reason.as_deref() {
        eprintln!("capability decision: {reason}");
    }
    if !prepared.target_decision.can_train
        || !matches!(
            prepared.target_decision.effective_target,
            DragonNativeTarget::Auto | DragonNativeTarget::Trainer
        )
    {
        bail!(
            "train-window-once requires a trainer-capable target; resolved {:?}",
            prepared.target_decision.effective_target
        );
    }

    let started = Instant::now();
    eprintln!("train-window-once progress: spawning native peer runtime");
    let mut running = spawn_prepared_native_peer(prepared)?;
    let edge_registration = if options.mirror_live_head_to_edge {
        train_window_edge_registration(config, auth_bundle)?
    } else {
        None
    };
    let report_result = (|| -> Result<TrainWindowOnceReport> {
        eprintln!("train-window-once progress: waiting for runtime readiness");
        wait_for_runtime_ready(&running, RUNTIME_READY_TIMEOUT)?;
        let local_peer_id = running
            .snapshot()
            .local_peer_id
            .ok_or_else(|| anyhow!("peer runtime did not report a local peer id"))?;
        eprintln!(
            "train-window-once progress: runtime ready peer={} elapsed_ms={}",
            local_peer_id,
            started.elapsed().as_millis()
        );
        ensure_p2p_publication_connectivity(
            &running,
            config,
            "before canonical head sync",
            TRAIN_WINDOW_P2P_CONNECTIVITY_TIMEOUT,
        )?;
        let experiment = running.mainnet().experiment(
            experiment_entry.study_id.clone(),
            experiment_entry.experiment_id.clone(),
            experiment_entry.current_revision_id.clone(),
        );
        let mut served_head_id = None;
        eprintln!("train-window-once progress: resolving active canonical head");
        let base_head = wait_for_head_provider(
            &mut running,
            &experiment,
            options.initialize_head_on_start,
            options.restore_head_on_start,
            &mut served_head_id,
            "trainer",
            Duration::from_secs(options.head_sync_timeout_secs.max(1)),
        )?;
        eprintln!(
            "train-window-once progress: active head ready head={} step={} served_head={:?} elapsed_ms={}",
            base_head.head_id,
            base_head.global_step,
            served_head_id,
            started.elapsed().as_millis()
        );
        eprintln!("train-window-once progress: preparing pinned trainer state");
        eprintln!(
            "train-window-once progress: trainer ready; running one training window elapsed_ms={}",
            started.elapsed().as_millis()
        );
        let outcome = running.train_window_once_with_pinned_head(&experiment, Some(&base_head))?;
        let train_loss = outcome
            .report
            .stats
            .get("train_loss")
            .or_else(|| outcome.report.stats.get("loss"));
        eprintln!(
            "train-window-once progress: window published head={} step={} artifact={} train_loss={:?} data_fetch_ms={} publish_ms={} elapsed_ms={}",
            outcome.head.head_id,
            outcome.head.global_step,
            outcome.artifact.artifact_id,
            train_loss,
            outcome.timing.data_fetch_time_ms,
            outcome.timing.publish_latency_ms,
            started.elapsed().as_millis()
        );
        ensure_p2p_publication_connectivity(
            &running,
            config,
            "after local training before diffusion publication",
            TRAIN_WINDOW_P2P_CONNECTIVITY_TIMEOUT,
        )?;
        publish_train_window_head(
            &running,
            &experiment,
            &local_peer_id,
            &outcome.head,
            "after local training",
        )?;
        let mut diffusion_settlement = None;
        if options.settle_diffusion || options.serve_after_publish_secs > 0 {
            if directory_entry_promotes_with_diffusion(&experiment_entry) {
                let passes_requested = if options.settle_diffusion {
                    options.diffusion_settle_passes.max(1)
                } else {
                    0
                };
                let mut passes_completed = 0_u32;
                if options.settle_diffusion {
                    for pass in 1..=passes_requested {
                        eprintln!(
                            "train-window-once progress: diffusion settle pass={} starting elapsed_ms={}",
                            pass,
                            started.elapsed().as_millis(),
                        );
                        ensure_p2p_publication_connectivity(
                            &running,
                            config,
                            "before diffusion settle pass",
                            TRAIN_WINDOW_P2P_CONNECTIVITY_TIMEOUT,
                        )?;
                        publish_train_window_head(
                            &running,
                            &experiment,
                            &local_peer_id,
                            &outcome.head,
                            "before diffusion settle pass",
                        )?;
                        running.advance_diffusion_steady_state(
                            &experiment,
                            Some(outcome.lease.window_id),
                            Some(&base_head.head_id),
                        )?;
                        passes_completed = pass;
                        let snapshot = running.snapshot();
                        eprintln!(
                            "train-window-once progress: diffusion settle pass={} connected_peers={} merge_windows={} updates={} attestations={} certificates={} merges={} elapsed_ms={}",
                            pass,
                            snapshot.connected_peers,
                            snapshot.control_plane.merge_window_announcements.len(),
                            snapshot.control_plane.update_announcements.len(),
                            snapshot
                                .control_plane
                                .trainer_promotion_attestation_announcements
                                .len(),
                            snapshot
                                .control_plane
                                .diffusion_promotion_certificate_announcements
                                .len(),
                            snapshot.control_plane.merge_announcements.len(),
                            started.elapsed().as_millis(),
                        );
                        thread::sleep(Duration::from_millis(250));
                    }
                }
                if options.serve_after_publish_secs > 0 {
                    let serve_for = Duration::from_secs(options.serve_after_publish_secs);
                    let serve_deadline = Instant::now() + serve_for;
                    let status_interval = Duration::from_secs(5);
                    let mut last_status = Instant::now()
                        .checked_sub(status_interval)
                        .unwrap_or_else(Instant::now);
                    let mut last_head_announcement = last_status;
                    eprintln!(
                        "train-window-once progress: serving published artifact for {}s elapsed_ms={}",
                        options.serve_after_publish_secs,
                        started.elapsed().as_millis()
                    );
                    ensure_p2p_publication_connectivity(
                        &running,
                        config,
                        "before serving published artifact",
                        TRAIN_WINDOW_P2P_CONNECTIVITY_TIMEOUT,
                    )?;
                    while Instant::now() < serve_deadline {
                        let mut connected_peers = running.snapshot().connected_peers;
                        if connected_peers == 0 {
                            connected_peers = ensure_p2p_publication_connectivity(
                                &running,
                                config,
                                "while serving published artifact",
                                TRAIN_WINDOW_P2P_CONNECTIVITY_TIMEOUT,
                            )?;
                        }
                        if last_head_announcement.elapsed() >= status_interval {
                            publish_train_window_head(
                                &running,
                                &experiment,
                                &local_peer_id,
                                &outcome.head,
                                "while serving published artifact",
                            )?;
                            last_head_announcement = Instant::now();
                        }
                        let snapshot = running.snapshot();
                        if last_status.elapsed() >= status_interval {
                            eprintln!(
                                "train-window-once progress: serving status connected_peers={} merge_windows={} updates={} attestations={} certificates={} merges={} last_error={} elapsed_ms={}",
                                connected_peers,
                                snapshot.control_plane.merge_window_announcements.len(),
                                snapshot.control_plane.update_announcements.len(),
                                snapshot
                                    .control_plane
                                    .trainer_promotion_attestation_announcements
                                    .len(),
                                snapshot
                                    .control_plane
                                    .diffusion_promotion_certificate_announcements
                                    .len(),
                                snapshot.control_plane.merge_announcements.len(),
                                operator_visible_last_error(snapshot.last_error.as_deref())
                                    .as_deref()
                                    .unwrap_or("-"),
                                started.elapsed().as_millis(),
                            );
                            last_status = Instant::now();
                        }
                        match snapshot.status {
                            RuntimeStatus::Failed => {
                                let reason = snapshot
                                    .last_error
                                    .unwrap_or_else(|| "peer runtime failed".into());
                                bail!("train-window-once runtime failed while serving: {reason}");
                            }
                            RuntimeStatus::Stopped => {
                                bail!("train-window-once runtime stopped while serving");
                            }
                            _ => {}
                        }
                        thread::sleep(STATUS_POLL_INTERVAL);
                    }
                }
                let snapshot = running.snapshot();
                diffusion_settlement = Some(diffusion_settlement_report(
                    &snapshot.control_plane,
                    true,
                    passes_requested,
                    passes_completed,
                    options.serve_after_publish_secs,
                ));
            } else {
                eprintln!(
                    "train-window-once progress: diffusion settlement requested but experiment promotion mode is not diffusion-steady-state"
                );
                let snapshot = running.snapshot();
                diffusion_settlement = Some(diffusion_settlement_report(
                    &snapshot.control_plane,
                    false,
                    0,
                    0,
                    options.serve_after_publish_secs,
                ));
            }
        }
        if let Some((registration_runtime, edge_base_url, session_id)) = edge_registration.as_ref()
        {
            let announcement = HeadAnnouncement {
                overlay: experiment.overlay_set()?.heads,
                provider_peer_id: Some(local_peer_id.clone()),
                head: outcome.head.clone(),
                announced_at: chrono::Utc::now(),
            };
            mirror_live_head_with_edge(
                registration_runtime,
                edge_base_url,
                session_id,
                &experiment_entry,
                &announcement,
            )
            .with_context(|| {
                format!(
                    "failed to mirror published head {} artifact {} to edge",
                    announcement.head.head_id.as_str(),
                    announcement.head.artifact_id.as_str()
                )
            })?;
        }
        Ok(TrainWindowOnceReport {
            experiment_kind: running.prepared().experiment_kind,
            backend: backend.as_label().into(),
            edge_base_url: config.effective_edge_base_url().map(ToOwned::to_owned),
            seed_node_count: config.effective_seed_node_urls().len(),
            effective_target: format!("{:?}", running.prepared().target_decision.effective_target),
            can_train: running.prepared().target_decision.can_train,
            downgrade_reason: running.prepared().target_decision.downgrade_reason.clone(),
            local_peer_id: local_peer_id.as_str().to_owned(),
            base_head_id: base_head.head_id.as_str().to_owned(),
            base_global_step: base_head.global_step,
            published_head_id: outcome.head.head_id.as_str().to_owned(),
            published_global_step: outcome.head.global_step,
            artifact_id: outcome.artifact.artifact_id.as_str().to_owned(),
            contribution_receipt_id: outcome.contribution.receipt_id.as_str().to_owned(),
            lease_window_id: outcome.lease.window_id.0.to_string(),
            lease_microshard_count: outcome.lease.microshards.len(),
            timing: TrainWindowOnceTimingReport {
                data_fetch_time_ms: outcome.timing.data_fetch_time_ms,
                publish_latency_ms: outcome.timing.publish_latency_ms,
            },
            diffusion_settlement,
            metrics: outcome.report.stats,
        })
    })();

    let shutdown_result = running.shutdown();
    let termination_result = running.await_termination_timeout(SHUTDOWN_TIMEOUT);

    if let Err(error) = shutdown_result {
        eprintln!("train-window-once shutdown error: {error}");
    }
    if let Err(error) = termination_result {
        match &report_result {
            Ok(_) => return Err(error),
            Err(_) => eprintln!("train-window-once termination error: {error}"),
        }
    }

    let report = report_result?;
    if options.require_head_advanced && report.published_global_step <= report.base_global_step {
        bail!(
            "train-window-once did not advance the experiment head: base step {} published step {}",
            report.base_global_step,
            report.published_global_step
        );
    }
    write_output(options.output, options.output_format, &report)
}

fn train_window_edge_registration(
    config: &DragonNativePeerConfig,
    auth_bundle: Option<&DragonNativeAuthBundle>,
) -> Result<Option<(tokio::runtime::Runtime, String, String)>> {
    let Some((edge_base_url, session_id)) = auth_bundle.and_then(|auth| {
        auth.session_id.as_ref().and_then(|session_id| {
            let edge_base_url = auth
                .edge_base_url
                .clone()
                .or_else(|| config.effective_edge_base_url().map(ToOwned::to_owned));
            edge_base_url.map(|edge_base_url| (edge_base_url, session_id.clone()))
        })
    }) else {
        return Ok(None);
    };
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to build async runtime for train-window edge registration")?;
    Ok(Some((runtime, edge_base_url, session_id)))
}

fn wait_for_head_provider<B>(
    running: &mut ManagedRunningNativePeer<B>,
    experiment: &burn_p2p::ExperimentHandle,
    initialize_head_on_start: bool,
    restore_head_on_start: bool,
    served_head_id: &mut Option<burn_p2p::HeadId>,
    log_prefix: &str,
    timeout: Duration,
) -> Result<burn_p2p::HeadDescriptor>
where
    B: AutodiffBackend + Clone + 'static,
{
    let deadline = Instant::now() + timeout;
    let started = Instant::now();
    let mut attempts = 0_u64;
    let mut last_error = None;
    loop {
        attempts += 1;
        match sync_or_initialize_latest_head_provider(
            running,
            experiment,
            initialize_head_on_start,
            restore_head_on_start,
            served_head_id,
            HeadProviderSyncMode::DirectoryCurrent,
            log_prefix,
        ) {
            Ok(Some(head)) => return Ok(head),
            Ok(None) => {}
            Err(error) => last_error = Some(error.to_string()),
        }

        if Instant::now() >= deadline {
            let detail = last_error
                .map(|error| format!("; last error: {error}"))
                .unwrap_or_default();
            bail!(
                "no experiment head became available within {:?}; rerun with --initialize-head-on-start true or seed a head first{}",
                timeout,
                detail
            );
        }

        if attempts == 1 || attempts.is_multiple_of(10) {
            let snapshot = running.snapshot();
            let last_snapshot_heads = snapshot
                .last_snapshot
                .as_ref()
                .map(|snapshot| snapshot.head_announcements.len())
                .unwrap_or(0);
            eprintln!(
                "{log_prefix}-head-waiting elapsed_ms={} attempts={} connected_peers={} local_heads={} last_snapshot_peer={} last_snapshot_heads={} node_state={:?} last_error={}",
                started.elapsed().as_millis(),
                attempts,
                snapshot.connected_peers,
                snapshot.control_plane.head_announcements.len(),
                snapshot
                    .last_snapshot_peer_id
                    .as_ref()
                    .map(|peer_id| peer_id.as_str())
                    .unwrap_or("-"),
                last_snapshot_heads,
                snapshot.node_state,
                operator_visible_last_error(snapshot.last_error.as_deref())
                    .as_deref()
                    .unwrap_or("-"),
            );
        }
        thread::sleep(STATUS_POLL_INTERVAL);
    }
}

#[derive(Clone, Copy)]
enum HeadProviderSyncMode {
    DirectoryCurrent,
    LatestPromoted,
}

fn sync_or_initialize_latest_head_provider<B>(
    running: &mut ManagedRunningNativePeer<B>,
    experiment: &burn_p2p::ExperimentHandle,
    initialize_head_on_start: bool,
    restore_head_on_start: bool,
    served_head_id: &mut Option<burn_p2p::HeadId>,
    sync_mode: HeadProviderSyncMode,
    log_prefix: &str,
) -> Result<Option<burn_p2p::HeadDescriptor>>
where
    B: AutodiffBackend + Clone + 'static,
{
    let restored = if restore_head_on_start {
        eprintln!("{log_prefix}-head-restore-start");
        match running.restore_experiment_head(experiment) {
            Ok(head) => {
                if let Some(head) = head.as_ref() {
                    eprintln!(
                        "{log_prefix}-head-restored id={} global_step={}",
                        head.head_id.as_str(),
                        head.global_step,
                    );
                }
                head
            }
            Err(error) if initialize_head_on_start => {
                eprintln!(
                    "{log_prefix}-head-restore-failed error={error}; continuing with sync/initialize"
                );
                None
            }
            Err(error) => return Err(error),
        }
    } else {
        None
    };

    let synced_result = match sync_mode {
        HeadProviderSyncMode::DirectoryCurrent => running.sync_experiment_head(experiment),
        HeadProviderSyncMode::LatestPromoted => {
            running.sync_latest_promoted_experiment_head(experiment)
        }
    };
    let synced = match synced_result {
        Ok(Some(head)) => {
            eprintln!(
                "{log_prefix}-head-synced id={} global_step={}",
                head.head_id.as_str(),
                head.global_step,
            );
            Some(head)
        }
        Ok(None) => None,
        Err(error) if restored.is_some() => {
            eprintln!(
                "{log_prefix}-head-sync-failed error={error}; keeping restored head candidate"
            );
            None
        }
        Err(error) if initialize_head_on_start => {
            eprintln!(
                "{log_prefix}-head-sync-failed error={error}; falling back to local genesis initialization if no restored head is available"
            );
            None
        }
        Err(error) => return Err(error),
    };

    let (head, source) = match select_latest_head_candidate(restored, synced) {
        Some(candidate) => candidate,
        None if initialize_head_on_start => {
            eprintln!("{log_prefix}-initializing local genesis head");
            let head = running.initialize_local_head(experiment)?;
            eprintln!(
                "{log_prefix}-initialized genesis head id={} global_step={}",
                head.head_id.as_str(),
                head.global_step,
            );
            (head, "initialized")
        }
        None => return Ok(None),
    };

    if source == "restored" && !running.adopt_known_head_if_present(experiment, &head)? {
        bail!(
            "{log_prefix}-head-restored id={} artifact={} could not be re-adopted",
            head.head_id.as_str(),
            head.artifact_id.as_str()
        );
    }
    eprintln!(
        "{log_prefix}-head-selected source={} id={} global_step={}",
        source,
        head.head_id.as_str(),
        head.global_step,
    );
    serve_head_provider(running, experiment, head, served_head_id, log_prefix).map(Some)
}

fn select_latest_head_candidate(
    restored: Option<burn_p2p::HeadDescriptor>,
    synced: Option<burn_p2p::HeadDescriptor>,
) -> Option<(burn_p2p::HeadDescriptor, &'static str)> {
    match (restored, synced) {
        (Some(restored), Some(synced)) if restored.global_step > synced.global_step => {
            Some((restored, "restored"))
        }
        (Some(_), Some(synced)) => Some((synced, "synced")),
        (Some(restored), None) => Some((restored, "restored")),
        (None, Some(synced)) => Some((synced, "synced")),
        (None, None) => None,
    }
}

fn serve_head_provider<B>(
    running: &mut ManagedRunningNativePeer<B>,
    experiment: &burn_p2p::ExperimentHandle,
    head: burn_p2p::HeadDescriptor,
    served_head_id: &mut Option<burn_p2p::HeadId>,
    log_prefix: &str,
) -> Result<burn_p2p::HeadDescriptor>
where
    B: AutodiffBackend + Clone + 'static,
{
    // Re-announce the locally materialized head on every sync pass so late
    // browser peers can always discover at least one live provider.
    running.publish_head_provider(experiment, &head)?;

    if served_head_id.as_ref() != Some(&head.head_id) {
        eprintln!(
            "{log_prefix}-serving head id={} global_step={}",
            head.head_id.as_str(),
            head.global_step,
        );
        *served_head_id = Some(head.head_id.clone());
    }

    Ok(head)
}

fn directory_entry_promotes_with_diffusion(entry: &ExperimentDirectoryEntry) -> bool {
    entry.merge_topology_policy().is_some_and(|policy| {
        matches!(
            policy.promotion_policy.mode,
            HeadPromotionMode::DiffusionSteadyState
        )
    })
}

fn diffusion_settlement_report(
    snapshot: &ControlPlaneSnapshot,
    enabled: bool,
    passes_requested: u32,
    passes_completed: u32,
    served_after_publish_secs: u64,
) -> DiffusionSettlementReport {
    DiffusionSettlementReport {
        enabled,
        passes_requested,
        passes_completed,
        served_after_publish_secs,
        merge_windows: snapshot.merge_window_announcements.len(),
        updates: snapshot.update_announcements.len(),
        attestations: snapshot.trainer_promotion_attestation_announcements.len(),
        certificates: snapshot.diffusion_promotion_certificate_announcements.len(),
        merges: snapshot.merge_announcements.len(),
    }
}

fn mirror_live_head_with_edge(
    runtime: &tokio::runtime::Runtime,
    edge_base_url: &str,
    session_id: &str,
    directory_template: &ExperimentDirectoryEntry,
    announcement: &HeadAnnouncement,
) -> Result<()> {
    register_live_head_with_edge_options(
        runtime,
        edge_base_url,
        session_id,
        Some(directory_template),
        announcement,
    )
}

fn register_live_head_with_edge_options(
    runtime: &tokio::runtime::Runtime,
    edge_base_url: &str,
    session_id: &str,
    directory_template: Option<&ExperimentDirectoryEntry>,
    announcement: &HeadAnnouncement,
) -> Result<()> {
    let provider_peer_id = announcement
        .provider_peer_id
        .as_ref()
        .ok_or_else(|| anyhow!("live head registration requires a provider peer id"))?;
    let mirror = runtime
        .block_on(mirror_peer_artifact(
            edge_base_url,
            session_id,
            burn_p2p_publish::PeerArtifactMirrorRequest {
                artifact_id: announcement.head.artifact_id.clone(),
                provider_peer_ids: vec![provider_peer_id.clone()],
                timeout_ms: Some(EDGE_HEAD_ARTIFACT_MIRROR_TIMEOUT_MILLIS),
            },
        ))
        .with_context(|| {
            format!(
                "failed to mirror head artifact {} from provider {} before live head registration",
                announcement.head.artifact_id.as_str(),
                provider_peer_id.as_str()
            )
        })?;
    let mirrored_provider_peer_id = mirror.mirrored_provider_peer_id.clone().ok_or_else(|| {
        anyhow!(
            "edge mirror response for artifact {} did not include a mirrored provider peer id",
            announcement.head.artifact_id.as_str()
        )
    })?;
    eprintln!(
        "head-mirror-edge-artifact-mirrored artifact_id={} source_provider={} edge_provider={} bytes={} chunks={}",
        mirror.artifact_id.as_str(),
        mirror.mirrored_from.as_str(),
        mirrored_provider_peer_id.as_str(),
        mirror.bytes_len,
        mirror.chunk_count,
    );

    let edge_announcement =
        mirrored_edge_head_announcement(announcement, mirrored_provider_peer_id.clone());
    register_edge_head_and_directory(
        runtime,
        edge_base_url,
        session_id,
        directory_template,
        edge_announcement,
        Some(provider_peer_id),
    )
}

fn register_edge_head_and_directory(
    runtime: &tokio::runtime::Runtime,
    edge_base_url: &str,
    session_id: &str,
    directory_template: Option<&ExperimentDirectoryEntry>,
    edge_announcement: HeadAnnouncement,
    source_provider_peer_id: Option<&PeerId>,
) -> Result<()> {
    let _ = runtime.block_on(register_live_head(
        edge_base_url,
        session_id,
        edge_announcement.clone(),
    ))?;
    eprintln!(
        "head-mirror-edge-head-registered head_id={} provider={} source_provider={}",
        edge_announcement.head.head_id.as_str(),
        edge_announcement
            .provider_peer_id
            .as_ref()
            .map(|peer_id| peer_id.as_str())
            .unwrap_or("-"),
        source_provider_peer_id
            .map(|peer_id| peer_id.as_str())
            .unwrap_or("-"),
    );
    if let Some(directory_template) = directory_template {
        let mut directory_entries =
            runtime.block_on(fetch_signed_directory_entries(edge_base_url, session_id))?;
        if upsert_directory_entry_current_head(
            &mut directory_entries,
            directory_template,
            edge_announcement.head.head_id.clone(),
        ) {
            let _ = runtime.block_on(rollout_directory_entries(
                edge_base_url,
                session_id,
                directory_entries,
            ))?;
            eprintln!(
                "head-mirror-edge-directory-updated head_id={}",
                edge_announcement.head.head_id.as_str(),
            );
        }
    }
    Ok(())
}

fn mirrored_edge_head_announcement(
    announcement: &HeadAnnouncement,
    mirrored_provider_peer_id: PeerId,
) -> HeadAnnouncement {
    let mut edge_announcement = announcement.clone();
    edge_announcement.provider_peer_id = Some(mirrored_provider_peer_id);
    edge_announcement
}

#[allow(clippy::too_many_arguments)]
fn run_prepared_head_mirror<B>(
    prepared: PreparedNativePeer<B>,
    config: &DragonNativePeerConfig,
    auth_bundle: Option<&DragonNativeAuthBundle>,
    backend: BackendArg,
    status_interval_secs: u64,
    head_sync_interval_secs: u64,
    initialize_head_on_start: bool,
    restore_head_on_start: bool,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
{
    let experiment_entry = prepared
        .manifests
        .experiment_directory
        .first()
        .cloned()
        .ok_or_else(|| anyhow!("prepared head mirror is missing an experiment"))?;
    eprintln!(
        "starting burn_dragon head mirror: experiment={} backend={} target={:?} can_train={} edge={} seeds={} storage_root={}",
        prepared.experiment_kind.workload_slug(),
        backend.as_label(),
        prepared.target_decision.effective_target,
        prepared.target_decision.can_train,
        config.effective_edge_base_url().unwrap_or("<none>"),
        config.effective_seed_node_urls().len(),
        config.storage_root.display(),
    );
    if let Some(reason) = prepared.target_decision.downgrade_reason.as_deref() {
        eprintln!("capability decision: {reason}");
    }
    if !prepared.target_decision.can_train {
        eprintln!(
            "head mirror continuing with estimated training footprint above the configured budget; target={:?}",
            prepared.target_decision.effective_target,
        );
    }

    let mut running = spawn_prepared_native_peer(prepared)?;
    wait_for_runtime_ready(&running, RUNTIME_READY_TIMEOUT)?;
    let shutdown_requested = Arc::new(AtomicBool::new(false));
    let shutdown_requested_for_handler = Arc::clone(&shutdown_requested);
    let control = running.control_handle();
    ctrlc::set_handler(move || {
        if !shutdown_requested_for_handler.swap(true, Ordering::SeqCst) {
            let _ = control.shutdown();
        }
    })
    .context("failed to install ctrl-c handler")?;

    let experiment = running.mainnet().experiment(
        experiment_entry.study_id.clone(),
        experiment_entry.experiment_id.clone(),
        experiment_entry.current_revision_id.clone(),
    );
    let edge_registration = auth_bundle
        .and_then(|auth| {
            auth.session_id.as_ref().and_then(|session_id| {
                let edge_base_url = auth
                    .edge_base_url
                    .clone()
                    .or_else(|| config.effective_edge_base_url().map(ToOwned::to_owned));
                edge_base_url.map(|edge_base_url| (edge_base_url, session_id.clone()))
            })
        })
        .map(|(edge_base_url, session_id)| {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .context("failed to build async runtime for head mirror edge registration")?;
            Ok::<_, anyhow::Error>((runtime, edge_base_url, session_id))
        })
        .transpose()?;
    let status_interval = Duration::from_secs(status_interval_secs.max(1));
    let head_sync_interval = Duration::from_secs(head_sync_interval_secs.max(1));
    let mut last_status = Instant::now()
        .checked_sub(status_interval)
        .unwrap_or_else(Instant::now);
    let mut last_head_sync = Instant::now()
        .checked_sub(head_sync_interval)
        .unwrap_or_else(Instant::now);
    let mut served_head_id = None;
    let mut edge_registered_head_id = None;

    loop {
        if last_head_sync.elapsed() >= head_sync_interval {
            let head = sync_or_initialize_latest_head_provider(
                &mut running,
                &experiment,
                initialize_head_on_start,
                restore_head_on_start,
                &mut served_head_id,
                HeadProviderSyncMode::LatestPromoted,
                "head-mirror",
            )?;
            let snapshot = running.snapshot();
            let visible_promoted = latest_visible_promoted_head_announcement(
                &snapshot.control_plane,
                &experiment,
                head.as_ref(),
            );
            if let (Some(announcement), Some((registration_runtime, edge_base_url, session_id))) =
                (visible_promoted.as_ref(), edge_registration.as_ref())
            {
                if edge_registered_head_id.as_ref() != Some(&announcement.head.head_id) {
                    match register_live_head_with_edge_options(
                        registration_runtime,
                        edge_base_url,
                        session_id,
                        Some(&experiment_entry),
                        announcement,
                    ) {
                        Ok(()) => {
                            eprintln!(
                                "head-mirror-edge-visible-head-registered head_id={} provider={}",
                                announcement.head.head_id.as_str(),
                                announcement
                                    .provider_peer_id
                                    .as_ref()
                                    .map(|peer_id| peer_id.as_str())
                                    .unwrap_or("-"),
                            );
                            edge_registered_head_id = Some(announcement.head.head_id.clone());
                        }
                        Err(error) => {
                            eprintln!(
                                "head-mirror-edge-visible-head-registration-failed head_id={} provider={} error={error}",
                                announcement.head.head_id.as_str(),
                                announcement
                                    .provider_peer_id
                                    .as_ref()
                                    .map(|peer_id| peer_id.as_str())
                                    .unwrap_or("-"),
                            );
                            if let (Some(head), Some(local_peer_id)) =
                                (head.as_ref(), snapshot.local_peer_id.clone())
                            {
                                if should_register_edge_local_fallback(
                                    &announcement.head,
                                    head,
                                    edge_registered_head_id.as_ref(),
                                ) {
                                    let local_announcement = edge_local_head_announcement(
                                        head,
                                        &experiment,
                                        local_peer_id.clone(),
                                    )?;
                                    match register_live_head_with_edge_options(
                                        registration_runtime,
                                        edge_base_url,
                                        session_id,
                                        Some(&experiment_entry),
                                        &local_announcement,
                                    ) {
                                        Ok(()) => {
                                            eprintln!(
                                                "head-mirror-edge-local-fallback-registered head_id={} provider={} superseded_head={}",
                                                local_announcement.head.head_id.as_str(),
                                                local_peer_id.as_str(),
                                                announcement.head.head_id.as_str(),
                                            );
                                            edge_registered_head_id =
                                                Some(local_announcement.head.head_id.clone());
                                        }
                                        Err(fallback_error) => {
                                            eprintln!(
                                                "head-mirror-edge-local-fallback-registration-failed head_id={} provider={} superseded_head={} error={fallback_error}",
                                                local_announcement.head.head_id.as_str(),
                                                local_peer_id.as_str(),
                                                announcement.head.head_id.as_str(),
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else if let (Some(head), Some((registration_runtime, edge_base_url, session_id))) =
                (head.as_ref(), edge_registration.as_ref())
            {
                let snapshot = running.snapshot();
                if let Some(local_peer_id) = snapshot.local_peer_id {
                    if edge_registered_head_id.as_ref() != Some(&head.head_id) {
                        let announcement =
                            edge_local_head_announcement(head, &experiment, local_peer_id.clone())?;
                        if let Err(error) = register_live_head_with_edge_options(
                            registration_runtime,
                            edge_base_url,
                            session_id,
                            Some(&experiment_entry),
                            &announcement,
                        ) {
                            eprintln!(
                                "head-mirror-edge-local-registration-failed head_id={} provider={} error={error}",
                                head.head_id.as_str(),
                                local_peer_id.as_str(),
                            );
                        } else {
                            eprintln!(
                                "head-mirror-edge-local-registered head_id={} provider={}",
                                head.head_id.as_str(),
                                local_peer_id.as_str(),
                            );
                            edge_registered_head_id = Some(head.head_id.clone());
                        }
                    }
                }
            }
            last_head_sync = Instant::now();
        }

        let snapshot = running.snapshot();
        if status_interval_secs > 0 && last_status.elapsed() >= status_interval {
            eprintln!(
                "head-mirror-status status={:?} node_state={:?} connected_peers={} served_head={} edge_registered_head={} last_error={}",
                snapshot.status,
                snapshot.node_state,
                snapshot.connected_peers,
                served_head_id
                    .as_ref()
                    .map(|head_id| head_id.as_str())
                    .unwrap_or("-"),
                edge_registered_head_id
                    .as_ref()
                    .map(|head_id| head_id.as_str())
                    .unwrap_or("-"),
                operator_visible_last_error(snapshot.last_error.as_deref())
                    .as_deref()
                    .unwrap_or("-"),
            );
            last_status = Instant::now();
        }

        match snapshot.status {
            RuntimeStatus::Failed => {
                let reason = snapshot
                    .last_error
                    .unwrap_or_else(|| "peer runtime failed".into());
                let _ = running.shutdown();
                let _ = running.await_termination_timeout(SHUTDOWN_TIMEOUT);
                bail!("head mirror failed: {reason}");
            }
            RuntimeStatus::Stopped => {
                let _prepared = running.await_termination_timeout(SHUTDOWN_TIMEOUT)?;
                eprintln!("head mirror stopped cleanly");
                return Ok(());
            }
            _ => {}
        }

        thread::sleep(STATUS_POLL_INTERVAL);
    }
}

fn edge_local_head_announcement(
    head: &HeadDescriptor,
    experiment: &ExperimentHandle,
    local_peer_id: PeerId,
) -> Result<HeadAnnouncement> {
    Ok(HeadAnnouncement {
        overlay: experiment.overlay_set()?.heads,
        provider_peer_id: Some(local_peer_id),
        head: head.clone(),
        announced_at: chrono::Utc::now(),
    })
}

fn should_register_edge_local_fallback(
    failed_visible_head: &HeadDescriptor,
    local_head: &HeadDescriptor,
    edge_registered_head_id: Option<&HeadId>,
) -> bool {
    failed_visible_head.head_id != local_head.head_id
        && edge_registered_head_id != Some(&local_head.head_id)
}

fn latest_visible_promoted_head_announcement(
    snapshot: &ControlPlaneSnapshot,
    experiment: &ExperimentHandle,
    baseline: Option<&HeadDescriptor>,
) -> Option<HeadAnnouncement> {
    snapshot
        .head_announcements
        .iter()
        .filter(|announcement| announcement.provider_peer_id.is_some())
        .filter(|announcement| head_matches_experiment(&announcement.head, experiment))
        .filter(|announcement| {
            baseline.is_none_or(|baseline| head_is_newer_than(&announcement.head, baseline))
        })
        .max_by(|left, right| {
            left.head
                .global_step
                .cmp(&right.head.global_step)
                .then(left.head.created_at.cmp(&right.head.created_at))
                .then(left.announced_at.cmp(&right.announced_at))
        })
        .cloned()
}

fn head_matches_experiment(head: &HeadDescriptor, experiment: &ExperimentHandle) -> bool {
    head.study_id == experiment.study_id
        && head.experiment_id == experiment.experiment_id
        && head.revision_id == experiment.revision_id
}

fn head_is_newer_than(candidate: &HeadDescriptor, baseline: &HeadDescriptor) -> bool {
    candidate.global_step > baseline.global_step
        || (candidate.global_step == baseline.global_step
            && candidate.created_at > baseline.created_at
            && candidate.head_id != baseline.head_id)
}

fn run_prepared_validator_daemon<B>(
    prepared: PreparedNativePeer<B>,
    config: &DragonNativePeerConfig,
    backend: BackendArg,
    status_interval_secs: u64,
    validation_interval_millis: u64,
    initialize_head_on_start: bool,
    restore_head_on_start: bool,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
{
    let experiment_entry = prepared
        .manifests
        .experiment_directory
        .first()
        .cloned()
        .ok_or_else(|| anyhow!("prepared validator manifest bundle is missing an experiment"))?;
    let diffusion_promotion = directory_entry_promotes_with_diffusion(&experiment_entry);
    eprintln!(
        "starting burn_dragon validator daemon: experiment={} backend={} target={:?} can_train={} promotion={} edge={} seeds={} storage_root={}",
        prepared.experiment_kind.workload_slug(),
        backend.as_label(),
        prepared.target_decision.effective_target,
        prepared.target_decision.can_train,
        if diffusion_promotion {
            "diffusion-steady-state"
        } else {
            "validator-quorum"
        },
        config.effective_edge_base_url().unwrap_or("<none>"),
        config.effective_seed_node_urls().len(),
        config.storage_root.display(),
    );
    if let Some(reason) = prepared.target_decision.downgrade_reason.as_deref() {
        eprintln!("capability decision: {reason}");
    }
    if prepared.target_decision.effective_target
        != burn_dragon_p2p::config::DragonNativeTarget::Validator
    {
        bail!(
            "validator daemon requires effective validator target; resolved {:?}",
            prepared.target_decision.effective_target
        );
    }

    let mut running = spawn_prepared_native_peer(prepared)?;
    wait_for_runtime_ready(&running, RUNTIME_READY_TIMEOUT)?;
    let shutdown_requested = Arc::new(AtomicBool::new(false));
    let shutdown_requested_for_handler = Arc::clone(&shutdown_requested);
    let control = running.control_handle();
    ctrlc::set_handler(move || {
        if !shutdown_requested_for_handler.swap(true, Ordering::SeqCst) {
            let _ = control.shutdown();
        }
    })
    .context("failed to install ctrl-c handler")?;

    let experiment = running.mainnet().experiment(
        experiment_entry.study_id,
        experiment_entry.experiment_id,
        experiment_entry.current_revision_id,
    );
    let mut served_head_id = None;

    let status_interval = Duration::from_secs(status_interval_secs.max(1));
    let validation_interval = Duration::from_millis(validation_interval_millis.max(25));
    let head_sync_interval = Duration::from_secs(status_interval_secs.clamp(1, 5));
    let mut last_status = Instant::now()
        .checked_sub(status_interval)
        .unwrap_or_else(Instant::now);
    let mut last_validation = Instant::now()
        .checked_sub(validation_interval)
        .unwrap_or_else(Instant::now);
    let mut last_head_sync = Instant::now()
        .checked_sub(head_sync_interval)
        .unwrap_or_else(Instant::now);
    let mut head_sync_attempts = 0_u64;

    loop {
        if last_head_sync.elapsed() >= head_sync_interval {
            head_sync_attempts = head_sync_attempts.saturating_add(1);
            match sync_or_initialize_latest_head_provider(
                &mut running,
                &experiment,
                initialize_head_on_start,
                restore_head_on_start,
                &mut served_head_id,
                HeadProviderSyncMode::DirectoryCurrent,
                "validator",
            ) {
                Ok(Some(_)) => {}
                Ok(None) => {
                    if head_sync_attempts == 1 || head_sync_attempts.is_multiple_of(12) {
                        let snapshot = running.snapshot();
                        eprintln!(
                            "validator-head-sync-waiting attempts={} connected_peers={} local_heads={} node_state={:?} last_error={}",
                            head_sync_attempts,
                            snapshot.connected_peers,
                            snapshot.control_plane.head_announcements.len(),
                            snapshot.node_state,
                            operator_visible_last_error(snapshot.last_error.as_deref())
                                .as_deref()
                                .unwrap_or("-"),
                        );
                    }
                }
                Err(error) => {
                    eprintln!("validator-head-sync-error: {error}");
                }
            }
            last_head_sync = Instant::now();
        }

        let snapshot = running.snapshot();
        if status_interval_secs > 0 && last_status.elapsed() >= status_interval {
            eprintln!(
                "validator-status status={:?} node_state={:?} connected_peers={} served_head={} last_error={}",
                snapshot.status,
                snapshot.node_state,
                snapshot.connected_peers,
                served_head_id
                    .as_ref()
                    .map(|head_id| head_id.as_str())
                    .unwrap_or("-"),
                operator_visible_last_error(snapshot.last_error.as_deref())
                    .as_deref()
                    .unwrap_or("-"),
            );
            last_status = Instant::now();
        }

        match snapshot.status {
            RuntimeStatus::Failed => {
                let reason = snapshot
                    .last_error
                    .unwrap_or_else(|| "validator runtime failed".into());
                let _ = running.shutdown();
                let _ = running.await_termination_timeout(SHUTDOWN_TIMEOUT);
                bail!("validator runtime failed: {reason}");
            }
            RuntimeStatus::Stopped => {
                let _prepared = running.await_termination_timeout(SHUTDOWN_TIMEOUT)?;
                eprintln!("validator stopped cleanly");
                return Ok(());
            }
            _ => {}
        }

        if last_validation.elapsed() >= validation_interval {
            if served_head_id.is_none() {
                last_validation = Instant::now();
                thread::sleep(STATUS_POLL_INTERVAL);
                continue;
            } else if diffusion_promotion {
                if let Err(error) = running.advance_diffusion_steady_state(&experiment, None, None)
                {
                    eprintln!("validator-diffusion-pass-error: {error}");
                }
            } else {
                match running.validate_candidates_once(&experiment) {
                    Ok(Some(outcome)) => {
                        eprintln!(
                            "validator-promoted merged_head_id={} global_step={}",
                            outcome.merged_head.head_id.as_str(),
                            outcome.merged_head.global_step,
                        );
                    }
                    Ok(None) => {}
                    Err(error) => {
                        eprintln!("validator-validation-pass-error: {error}");
                    }
                }
            }
            last_validation = Instant::now();
        }

        thread::sleep(STATUS_POLL_INTERVAL);
    }
}

fn wait_for_runtime_ready<B>(running: &ManagedRunningNativePeer<B>, timeout: Duration) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
{
    let deadline = Instant::now() + timeout;
    loop {
        let snapshot = running.snapshot();
        if snapshot.local_peer_id.is_some() && !snapshot.listen_addresses.is_empty() {
            return Ok(());
        }
        if snapshot.status == RuntimeStatus::Failed {
            bail!(
                "peer runtime failed before becoming ready: {}",
                snapshot.last_error.as_deref().unwrap_or("unknown error"),
            );
        }
        if snapshot.status == RuntimeStatus::Stopped {
            bail!("peer runtime stopped before becoming ready");
        }
        if Instant::now() >= deadline {
            bail!("peer runtime did not become ready within {:?}", timeout);
        }
        thread::sleep(STATUS_POLL_INTERVAL);
    }
}

fn ensure_p2p_publication_connectivity<B>(
    running: &ManagedRunningNativePeer<B>,
    config: &DragonNativePeerConfig,
    context: &str,
    timeout: Duration,
) -> Result<usize>
where
    B: AutodiffBackend + Clone + 'static,
{
    let bootstrap_peers = config.effective_bootstrap_peers()?;
    if bootstrap_peers.is_empty() {
        let connected_peers = running.snapshot().connected_peers;
        eprintln!(
            "train-window-once progress: p2p connectivity check skipped context={context:?} reason=no-bootstrap-peers connected_peers={connected_peers}"
        );
        return Ok(connected_peers);
    }

    let control = running.control_handle();
    let deadline = Instant::now() + timeout;
    let mut last_dial = Instant::now()
        .checked_sub(TRAIN_WINDOW_P2P_REDIAL_INTERVAL)
        .unwrap_or_else(Instant::now);
    let mut last_dial_errors = Vec::new();

    loop {
        let snapshot = running.snapshot();
        if snapshot.connected_peers > 0 {
            eprintln!(
                "train-window-once progress: p2p connectivity ready context={context:?} connected_peers={} seeds={}",
                snapshot.connected_peers,
                bootstrap_peers.len(),
            );
            return Ok(snapshot.connected_peers);
        }
        match snapshot.status {
            RuntimeStatus::Failed => {
                bail!(
                    "train-window-once runtime failed while waiting for p2p connectivity {context:?}: {}",
                    snapshot.last_error.as_deref().unwrap_or("unknown error"),
                );
            }
            RuntimeStatus::Stopped => {
                bail!(
                    "train-window-once runtime stopped while waiting for p2p connectivity {context:?}"
                );
            }
            _ => {}
        }
        if Instant::now() >= deadline {
            let last_error = operator_visible_last_error(snapshot.last_error.as_deref())
                .unwrap_or_else(|| "-".into());
            let seed_preview = bootstrap_peers
                .iter()
                .take(4)
                .map(|address| address.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            let dial_errors = if last_dial_errors.is_empty() {
                "-".to_owned()
            } else {
                last_dial_errors.join("; ")
            };
            bail!(
                "train-window-once p2p connectivity unavailable {context:?} after {}s; connected_peers=0 seeds={} seed_preview=[{}] last_error={} dial_errors={}",
                timeout.as_secs(),
                bootstrap_peers.len(),
                seed_preview,
                last_error,
                dial_errors,
            );
        }

        if last_dial.elapsed() >= TRAIN_WINDOW_P2P_REDIAL_INTERVAL {
            last_dial_errors.clear();
            for address in &bootstrap_peers {
                if let Err(error) = control.dial_address(address.clone()) {
                    last_dial_errors.push(format!("{}: {error}", address.as_str()));
                }
            }
            let last_error = operator_visible_last_error(snapshot.last_error.as_deref())
                .unwrap_or_else(|| "-".into());
            eprintln!(
                "train-window-once progress: waiting for p2p connectivity context={context:?} connected_peers=0 seeds={} dial_errors={} last_error={}",
                bootstrap_peers.len(),
                last_dial_errors.len(),
                last_error,
            );
            last_dial = Instant::now();
        }

        thread::sleep(STATUS_POLL_INTERVAL);
    }
}

fn publish_train_window_head<B>(
    running: &ManagedRunningNativePeer<B>,
    experiment: &ExperimentHandle,
    local_peer_id: &PeerId,
    head: &HeadDescriptor,
    context: &str,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
{
    running
        .control_handle()
        .publish_head(HeadAnnouncement {
            overlay: experiment.overlay_set()?.heads,
            provider_peer_id: Some(local_peer_id.clone()),
            head: head.clone(),
            announced_at: chrono::Utc::now(),
        })
        .with_context(|| {
            format!(
                "failed to announce train-window-once head {} {context}",
                head.head_id.as_str()
            )
        })?;
    eprintln!(
        "train-window-once progress: announced published head context={context:?} head={} step={}",
        head.head_id.as_str(),
        head.global_step,
    );
    Ok(())
}

fn load_native_config(path: &Path, format: ConfigFormat) -> Result<DragonNativePeerConfig> {
    load_typed(path, format)
}

fn load_typed<T>(path: &Path, format: ConfigFormat) -> Result<T>
where
    T: DeserializeOwned,
{
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let format = match format {
        ConfigFormat::Auto => infer_format(path)?,
        explicit => explicit,
    };
    match format {
        ConfigFormat::Toml => toml::from_str(
            std::str::from_utf8(&bytes)
                .with_context(|| format!("TOML document is not valid UTF-8: {}", path.display()))?,
        )
        .with_context(|| format!("failed to parse TOML {}", path.display())),
        ConfigFormat::Json => serde_json::from_slice(&bytes)
            .with_context(|| format!("failed to parse JSON {}", path.display())),
        ConfigFormat::Auto => unreachable!(),
    }
}

fn write_output<T>(path: Option<&Path>, format: OutputFormat, value: &T) -> Result<()>
where
    T: Serialize,
{
    let body = match format {
        OutputFormat::Toml => toml::to_string_pretty(value)?,
        OutputFormat::Json => serde_json::to_string_pretty(value)?,
    };
    if let Some(path) = path {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        fs::write(path, body).with_context(|| format!("failed to write {}", path.display()))?;
    } else {
        println!("{body}");
    }
    Ok(())
}

fn infer_format(path: &Path) -> Result<ConfigFormat> {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("toml") => Ok(ConfigFormat::Toml),
        Some("json") => Ok(ConfigFormat::Json),
        _ => bail!(
            "could not infer config format for {}; pass --config-format",
            path.display()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    use burn_p2p::{
        AuthProvider, DatasetViewId, EdgeEnrollmentConfig, ExperimentId, ExperimentOptInPolicy,
        ExperimentResourceRequirements, ExperimentVisibility, NodeCertificate,
        NodeCertificateClaims, OverlayTopic, PeerId, PeerRole, PrincipalClaims, PrincipalSession,
        ProjectFamilyId, RevisionId, RevocationEpoch, StudyId, WorkloadId,
    };
    use burn_p2p_core::{SignatureAlgorithm, SignatureMetadata};
    use chrono::Utc;
    use semver::Version;
    use tempfile::tempdir;

    fn test_enrollment(requested_scopes: BTreeSet<ExperimentScope>) -> EdgeEnrollmentConfig {
        EdgeEnrollmentConfig {
            network_id: NetworkId::new("dragon-native-auth-testnet"),
            project_family_id: ProjectFamilyId::new("burn-dragon-language"),
            protocol_major: 0,
            app_semver: semver::Version::parse(env!("CARGO_PKG_VERSION"))
                .expect("valid burn_dragon version"),
            release_train_hash: ContentId::new("dragon-native-auth-release"),
            target_artifact_id: "native-cpu".into(),
            target_artifact_hash: ContentId::new("burn-dragon-native"),
            login_path: "/login/github".into(),
            device_path: None,
            callback_path: "/callback/github".into(),
            trusted_callback_header: None,
            trusted_callback_token: None,
            enroll_path: "/enroll".into(),
            trust_bundle_path: "/trust".into(),
            requested_scopes,
            session_ttl_secs: 1800,
        }
    }

    fn test_session(enrollment: &EdgeEnrollmentConfig) -> PrincipalSession {
        let now = Utc::now();
        PrincipalSession {
            session_id: ContentId::new("dragon-native-auth-session"),
            network_id: enrollment.network_id.clone(),
            claims: PrincipalClaims {
                principal_id: PrincipalId::new("github-native-cli"),
                provider: AuthProvider::GitHub,
                display_name: "native cli".into(),
                org_memberships: BTreeSet::new(),
                group_memberships: BTreeSet::new(),
                granted_roles: PeerRoleSet::new([PeerRole::TrainerCpu, PeerRole::Archive]),
                granted_scopes: enrollment.requested_scopes.clone(),
                custom_claims: BTreeMap::new(),
                issued_at: now,
                expires_at: now + chrono::Duration::minutes(30),
            },
            issued_at: now,
            expires_at: now + chrono::Duration::minutes(30),
        }
    }

    fn test_certificate(
        enrollment: &EdgeEnrollmentConfig,
        session: &PrincipalSession,
        identity: &burn_p2p::EdgePeerIdentity,
    ) -> NodeCertificate {
        let now = Utc::now();
        NodeCertificate::new(
            Version::new(0, 1, 0),
            NodeCertificateClaims {
                network_id: enrollment.network_id.clone(),
                project_family_id: enrollment.project_family_id.clone(),
                release_train_hash: enrollment.release_train_hash.clone(),
                target_artifact_hash: enrollment.target_artifact_hash.clone(),
                peer_id: identity.peer_id.clone(),
                peer_public_key_hex: identity.peer_public_key_hex.clone(),
                principal_id: session.claims.principal_id.clone(),
                provider: session.claims.provider.clone(),
                granted_roles: session.claims.granted_roles.clone(),
                experiment_scopes: enrollment.requested_scopes.clone(),
                client_policy_hash: identity.client_policy_hash.clone(),
                auth_policy_snapshot: None,
                not_before: now,
                not_after: now + chrono::Duration::minutes(30),
                serial: identity.serial,
                revocation_epoch: RevocationEpoch(0),
            },
            SignatureMetadata {
                signer: PeerId::new("dragon-native-auth-issuer"),
                key_id: "dragon-native-auth-key".into(),
                algorithm: SignatureAlgorithm::Ed25519,
                signed_at: now,
                signature_hex: "00".into(),
            },
        )
        .expect("test certificate")
    }

    fn post_form(callback_url: &str, fields: &[(&str, String)]) -> Result<String> {
        let url = Url::parse(callback_url)?;
        let host = url
            .host_str()
            .ok_or_else(|| anyhow!("callback url missing host"))?;
        let port = url
            .port()
            .ok_or_else(|| anyhow!("callback url missing port"))?;
        let mut body = url::form_urlencoded::Serializer::new(String::new());
        for (key, value) in fields {
            body.append_pair(key, value);
        }
        let body = body.finish();
        let target = match url.query() {
            Some(query) => format!("{}?{query}", url.path()),
            None => url.path().to_owned(),
        };
        let mut stream = TcpStream::connect((host, port))?;
        write!(
            stream,
            "POST {target} HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Type: application/x-www-form-urlencoded\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )?;
        stream.flush()?;
        let mut response = String::new();
        stream.read_to_string(&mut response)?;
        Ok(response)
    }

    fn test_experiment_entry() -> ExperimentDirectoryEntry {
        ExperimentDirectoryEntry {
            network_id: NetworkId::new("burn-dragon-mainnet"),
            study_id: StudyId::new("burn-dragon-mainnet"),
            experiment_id: ExperimentId::new("nca-prepretraining"),
            workload_id: WorkloadId::new("dragon-nca-prepretraining-cpu"),
            display_name: "NCA".into(),
            model_schema_hash: ContentId::new("schema"),
            dataset_view_id: DatasetViewId::new("dataset"),
            resource_requirements: ExperimentResourceRequirements {
                minimum_roles: BTreeSet::new(),
                minimum_device_memory_bytes: None,
                minimum_system_memory_bytes: Some(1),
                estimated_download_bytes: 1,
                estimated_window_seconds: 30,
            },
            visibility: ExperimentVisibility::Public,
            opt_in_policy: ExperimentOptInPolicy::Open,
            current_revision_id: RevisionId::new("nca-r1"),
            current_head_id: None,
            allowed_roles: PeerRoleSet::new([PeerRole::TrainerCpu]),
            allowed_scopes: BTreeSet::from([ExperimentScope::Connect]),
            training_protocol: Default::default(),
            metadata: BTreeMap::new(),
        }
    }

    #[test]
    fn directory_entry_promotes_with_diffusion_reads_merge_topology_metadata() {
        let mut entry = test_experiment_entry();
        assert!(!directory_entry_promotes_with_diffusion(&entry));

        let policy = burn_p2p::MergeTopologyPolicy {
            promotion_policy: burn_p2p::HeadPromotionPolicy {
                mode: HeadPromotionMode::DiffusionSteadyState,
                diffusion: Some(burn_p2p::DiffusionSteadyStatePolicy::default()),
                ..burn_p2p::HeadPromotionPolicy::default()
            },
            ..burn_p2p::MergeTopologyPolicy::default()
        };
        entry.metadata.insert(
            "burn_p2p.revision.merge_topology.policy_json".into(),
            serde_json::to_string(&policy).expect("serialize merge topology policy"),
        );

        assert!(directory_entry_promotes_with_diffusion(&entry));
    }

    fn test_head_announcement(provider_peer_id: Option<PeerId>) -> HeadAnnouncement {
        HeadAnnouncement {
            overlay: OverlayTopic::control(NetworkId::new("burn-dragon-mainnet")),
            provider_peer_id,
            head: burn_p2p::HeadDescriptor {
                head_id: burn_p2p::HeadId::new("head-1"),
                study_id: StudyId::new("burn-dragon-mainnet"),
                experiment_id: ExperimentId::new("nca-prepretraining"),
                revision_id: RevisionId::new("nca-r1"),
                artifact_id: burn_p2p::ArtifactId::new("artifact-1"),
                parent_head_id: None,
                global_step: 0,
                created_at: Utc::now(),
                metrics: BTreeMap::new(),
            },
            announced_at: Utc::now(),
        }
    }

    fn test_head_descriptor(head_id: &str, global_step: u64) -> burn_p2p::HeadDescriptor {
        burn_p2p::HeadDescriptor {
            head_id: burn_p2p::HeadId::new(head_id),
            study_id: StudyId::new("burn-dragon-mainnet"),
            experiment_id: ExperimentId::new("nca-prepretraining"),
            revision_id: RevisionId::new("nca-r1"),
            artifact_id: burn_p2p::ArtifactId::new(format!("artifact-{head_id}")),
            parent_head_id: None,
            global_step,
            created_at: Utc::now(),
            metrics: BTreeMap::new(),
        }
    }

    fn test_experiment_handle() -> ExperimentHandle {
        ExperimentHandle {
            network_id: NetworkId::new("burn-dragon-mainnet"),
            study_id: StudyId::new("burn-dragon-mainnet"),
            experiment_id: ExperimentId::new("nca-prepretraining"),
            revision_id: RevisionId::new("nca-r1"),
        }
    }

    fn test_head_announcement_for(
        head: burn_p2p::HeadDescriptor,
        provider: &str,
    ) -> HeadAnnouncement {
        HeadAnnouncement {
            overlay: OverlayTopic::control(NetworkId::new("burn-dragon-mainnet")),
            provider_peer_id: (!provider.is_empty()).then(|| PeerId::new(provider)),
            head,
            announced_at: Utc::now(),
        }
    }

    #[test]
    fn latest_head_candidate_keeps_restored_head_when_network_is_stale() {
        let restored = test_head_descriptor("head-window-2", 2);
        let synced = test_head_descriptor("head-genesis", 0);

        let (selected, source) =
            select_latest_head_candidate(Some(restored), Some(synced)).expect("selected head");

        assert_eq!(source, "restored");
        assert_eq!(selected.head_id.as_str(), "head-window-2");
        assert_eq!(selected.global_step, 2);
    }

    #[test]
    fn latest_head_candidate_prefers_synced_head_when_it_is_current() {
        let restored = test_head_descriptor("head-window-1", 1);
        let synced = test_head_descriptor("head-window-2", 2);

        let (selected, source) =
            select_latest_head_candidate(Some(restored), Some(synced)).expect("selected head");

        assert_eq!(source, "synced");
        assert_eq!(selected.head_id.as_str(), "head-window-2");
        assert_eq!(selected.global_step, 2);
    }

    #[test]
    fn visible_promoted_head_candidate_prefers_provider_backed_newer_head() {
        let experiment = test_experiment_handle();
        let mut served = test_head_descriptor("head-window-2", 2);
        let mut stale = test_head_descriptor("head-window-1", 1);
        let mut promoted = test_head_descriptor("head-window-3", 3);
        let mut providerless = test_head_descriptor("head-window-4", 4);
        let base_time = Utc::now();
        served.created_at = base_time;
        stale.created_at = base_time - chrono::Duration::seconds(1);
        promoted.created_at = base_time + chrono::Duration::seconds(1);
        providerless.created_at = base_time + chrono::Duration::seconds(2);

        let snapshot = ControlPlaneSnapshot {
            head_announcements: vec![
                test_head_announcement_for(providerless, ""),
                test_head_announcement_for(stale, "provider-stale"),
                test_head_announcement_for(promoted, "provider-promoted"),
            ],
            ..ControlPlaneSnapshot::default()
        };

        let selected =
            latest_visible_promoted_head_announcement(&snapshot, &experiment, Some(&served))
                .expect("newer provider-backed head");

        assert_eq!(selected.head.head_id.as_str(), "head-window-3");
        assert_eq!(
            selected.provider_peer_id.as_ref().map(|peer| peer.as_str()),
            Some("provider-promoted"),
        );
    }

    #[test]
    fn edge_local_head_announcement_uses_local_provider() {
        let experiment = test_experiment_handle();
        let head = test_head_descriptor("head-window-2", 2);
        let local_peer_id = PeerId::new("local-head-mirror");

        let announcement = edge_local_head_announcement(&head, &experiment, local_peer_id.clone())
            .expect("local head announcement");

        assert_eq!(announcement.head, head);
        assert_eq!(announcement.provider_peer_id, Some(local_peer_id));
        assert_eq!(
            announcement.overlay,
            experiment.overlay_set().unwrap().heads
        );
    }

    #[test]
    fn edge_local_fallback_is_selected_after_unreachable_newer_head() {
        let visible = test_head_descriptor("head-window-4", 4);
        let local = test_head_descriptor("head-window-3", 3);

        assert!(should_register_edge_local_fallback(&visible, &local, None));
        assert!(!should_register_edge_local_fallback(
            &visible,
            &local,
            Some(&local.head_id),
        ));
        assert!(!should_register_edge_local_fallback(
            &visible, &visible, None,
        ));
    }

    fn spawn_single_response_server(
        status: &'static str,
        body: &'static str,
    ) -> (String, Arc<Mutex<Vec<String>>>, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let address = listener.local_addr().expect("test server address");
        let requests = Arc::new(Mutex::new(Vec::new()));
        let requests_for_thread = Arc::clone(&requests);
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept request");
            let mut buffer = [0; 4096];
            let read = stream.read(&mut buffer).expect("read request");
            let request = String::from_utf8_lossy(&buffer[..read]);
            let path = request
                .lines()
                .next()
                .and_then(|line| line.split_whitespace().nth(1))
                .unwrap_or("<missing>")
                .to_owned();
            requests_for_thread
                .lock()
                .expect("requests lock")
                .push(path);
            write!(
                stream,
                "HTTP/1.1 {status}\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .expect("write response");
            stream.flush().expect("flush response");
        });
        (format!("http://{address}"), requests, handle)
    }

    #[test]
    fn default_mainnet_native_config_targets_public_nca_profile() {
        let config = default_mainnet_native_config();
        let expected_seeds = DEFAULT_MAINNET_SEED_NODE_URLS
            .iter()
            .map(|seed| (*seed).to_owned())
            .collect::<Vec<_>>();

        assert!(config.training_config_paths.is_empty());
        assert_eq!(config.target, Some(DragonNativeTarget::Trainer));
        assert_eq!(
            config.effective_edge_base_url(),
            Some(DEFAULT_MAINNET_EDGE_BASE_URL)
        );
        assert_eq!(config.effective_seed_node_urls(), expected_seeds);
        assert_eq!(
            config.manifest.project_family_id,
            DEFAULT_MAINNET_PROJECT_FAMILY_ID
        );
        assert_eq!(config.manifest.network_id, DEFAULT_MAINNET_NETWORK_ID);
        assert_eq!(config.manifest.study_id, DEFAULT_MAINNET_STUDY_ID);
        assert_eq!(config.manifest.experiment_id, DEFAULT_MAINNET_EXPERIMENT_ID);
        assert_eq!(config.manifest.revision_id, DEFAULT_MAINNET_REVISION_ID);
    }

    #[test]
    fn native_join_commands_default_to_mainnet_wgpu_and_head_sync() {
        let run_peer = Cli::try_parse_from(["burn_dragon_p2p_native", "run-peer"])
            .expect("parse run-peer defaults");
        let CommandKind::RunPeer(run_peer) = run_peer.command else {
            panic!("expected run-peer command");
        };
        assert!(run_peer.config.is_none());
        assert_eq!(run_peer.experiment_kind, ExperimentKindArg::Nca);
        assert_eq!(run_peer.backend, BackendArg::Wgpu);
        assert!(run_peer.restore_head_on_start);
        assert_eq!(
            run_peer.head_sync_interval_secs,
            DEFAULT_HEAD_SYNC_INTERVAL_SECS
        );

        let doctor = Cli::try_parse_from(["burn_dragon_p2p_native", "doctor"])
            .expect("parse doctor defaults");
        let CommandKind::Doctor(doctor) = doctor.command else {
            panic!("expected doctor command");
        };
        assert!(doctor.config.is_none());
        assert_eq!(doctor.experiment_kind, ExperimentKindArg::Nca);
        assert_eq!(doctor.backend, BackendArg::Wgpu);

        let train_once = Cli::try_parse_from([
            "burn_dragon_p2p_native",
            "train-window-once",
            "--backend",
            "webgpu",
        ])
        .expect("parse train-window-once defaults");
        let CommandKind::TrainWindowOnce(train_once) = train_once.command else {
            panic!("expected train-window-once command");
        };
        assert!(train_once.config.is_none());
        assert_eq!(train_once.experiment_kind, ExperimentKindArg::Nca);
        assert_eq!(train_once.backend, BackendArg::Wgpu);
        assert!(train_once.initialize_head_on_start);
        assert!(train_once.restore_head_on_start);
        assert!(!train_once.settle_diffusion);
        assert_eq!(train_once.diffusion_settle_passes, 3);
        assert_eq!(train_once.serve_after_publish_secs, 0);
        assert!(!train_once.mirror_live_head_to_edge);
        assert_eq!(train_once.training_overrides.batch_size, None);
        assert_eq!(train_once.training_overrides.max_iters, None);
        assert_eq!(train_once.training_overrides.max_eval_batches, None);

        let no_restore = Cli::try_parse_from([
            "burn_dragon_p2p_native",
            "train-window-once",
            "--initialize-head-on-start",
            "false",
            "--restore-head-on-start",
            "false",
            "--training-batch-size",
            "1",
            "--training-max-iters",
            "4",
            "--evaluation-max-batches",
            "1",
            "--settle-diffusion",
            "--diffusion-settle-passes",
            "7",
            "--serve-after-publish-secs",
            "30",
            "--mirror-live-head-to-edge",
        ])
        .expect("parse explicit head flags");
        let CommandKind::TrainWindowOnce(no_restore) = no_restore.command else {
            panic!("expected train-window-once command");
        };
        assert!(!no_restore.initialize_head_on_start);
        assert!(!no_restore.restore_head_on_start);
        assert!(no_restore.settle_diffusion);
        assert_eq!(no_restore.diffusion_settle_passes, 7);
        assert_eq!(no_restore.serve_after_publish_secs, 30);
        assert!(no_restore.mirror_live_head_to_edge);
        assert_eq!(no_restore.training_overrides.batch_size, Some(1));
        assert_eq!(no_restore.training_overrides.max_iters, Some(4));
        assert_eq!(no_restore.training_overrides.max_eval_batches, Some(1));

        let admin_rollout = Cli::try_parse_from([
            "burn_dragon_p2p_native",
            "admin-rollout-profile",
            "--experiment-kind",
            "nca",
            "--backend",
            "cpu",
            "--auth-bundle",
            "/tmp/auth.json",
            "--reset-current-head-to-visible-root",
        ])
        .expect("parse admin rollout repair flags");
        let CommandKind::AdminRolloutProfile(admin_rollout) = admin_rollout.command else {
            panic!("expected admin-rollout-profile command");
        };
        assert!(admin_rollout.config.is_none());
        assert!(admin_rollout.reset_current_head_to_visible_root);
    }

    #[test]
    fn validator_config_requests_validate_scopes() {
        let mut config = default_mainnet_native_config();
        config.target = Some(DragonNativeTarget::Validator);
        let scopes = requested_scopes_for_config(&config);
        let experiment_id = ExperimentId::new(DEFAULT_MAINNET_EXPERIMENT_ID);
        assert!(scopes.contains(&ExperimentScope::Connect));
        assert!(scopes.contains(&ExperimentScope::Discover));
        assert!(scopes.contains(&ExperimentScope::Validate {
            experiment_id: experiment_id.clone()
        }));
        assert!(scopes.contains(&ExperimentScope::Archive { experiment_id }));
        assert!(!scopes.iter().any(|scope| {
            matches!(
                scope,
                ExperimentScope::Train {
                    experiment_id
                } if experiment_id.as_str() == DEFAULT_MAINNET_EXPERIMENT_ID
            )
        }));
    }

    #[test]
    fn head_mirror_registration_requires_artifact_mirror_before_live_head() {
        let (edge_base_url, requests, server) =
            spawn_single_response_server("502 Bad Gateway", "mirror unavailable");
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        let announcement = test_head_announcement(Some(PeerId::new(
            "12D3KooWCPbD9DgsaDHtPC6cC6DsvLNL64rtfo8UsQCVMBuazuuP",
        )));

        let error = register_live_head_with_edge_options(
            &runtime,
            &edge_base_url,
            "session-1",
            Some(&test_experiment_entry()),
            &announcement,
        )
        .expect_err("mirror failure should block live head registration");

        assert!(
            error
                .to_string()
                .contains("failed to mirror head artifact artifact-1"),
            "{error:#}"
        );
        server.join().expect("server thread");
        assert_eq!(
            *requests.lock().expect("requests lock"),
            vec!["/admin/artifacts/mirror-peer".to_owned()]
        );
    }

    #[test]
    fn head_mirror_registration_uses_edge_provider_after_mirror() {
        let source_provider = PeerId::new("12D3KooWCPbD9DgsaDHtPC6cC6DsvLNL64rtfo8UsQCVMBuazuuP");
        let edge_provider = PeerId::new("12D3KooWJLKDYyWyB26bcJwV3u2ASqXvewHdKWRLkTe8xH7gb63");
        let announcement = test_head_announcement(Some(source_provider));
        let edge_announcement =
            mirrored_edge_head_announcement(&announcement, edge_provider.clone());

        assert_eq!(edge_announcement.provider_peer_id, Some(edge_provider));
        assert_eq!(edge_announcement.head, announcement.head);
        assert_eq!(edge_announcement.overlay, announcement.overlay);
    }

    #[test]
    fn native_backend_labels_match_install_features() {
        assert_eq!(BackendArg::Cpu.default_enabled_features_label(), "native");
        assert_eq!(
            BackendArg::Wgpu.default_enabled_features_label(),
            "native,wgpu"
        );
        assert_eq!(
            BackendArg::Cuda.default_enabled_features_label(),
            "native,cuda"
        );
        assert_eq!(
            BackendArg::Rocm.default_enabled_features_label(),
            "native,rocm"
        );
        assert_eq!(native_target_artifact_id(BackendArg::Rocm), "native-rocm");
    }

    #[test]
    fn native_cli_browser_auth_url_targets_pages_callback() {
        let storage = tempdir().expect("storage");
        let (_, identity) = edge_peer_identity_for_storage(storage.path(), None).expect("identity");
        let bootstrap = NativeCliBridgeBootstrap {
            edge_base_url: "https://edge.dragon.example".into(),
            site_base_url: "https://dragon.example".into(),
            target_artifact_id: "native-cpu".into(),
            app_semver: "0.21.0".into(),
            git_commit: "test".into(),
            enabled_features_label: "native".into(),
            requested_scopes: BTreeSet::from([ExperimentScope::Connect]),
            session_ttl_secs: 1800,
            principal_hint: Some("alice".into()),
            identity,
        };

        let url =
            native_cli_browser_auth_url(&bootstrap, "http://127.0.0.1:43123/callback", "nonce-1")
                .expect("bridge url");
        let parsed = Url::parse(&url).expect("parse bridge url");
        assert_eq!(parsed.scheme(), "https");
        assert_eq!(parsed.host_str(), Some("dragon.example"));
        assert_eq!(parsed.path(), "/callback/github");
        let query = parsed.query_pairs().collect::<BTreeMap<_, _>>();
        assert_eq!(
            query.get("native_cli").map(|value| value.as_ref()),
            Some("1")
        );
        assert!(query.contains_key("native_auth_bootstrap"));
        assert!(!query.contains_key("native_authorize"));
    }

    #[test]
    fn browser_site_base_url_override_avoids_edge_hostname_guessing() {
        assert_eq!(
            resolve_browser_site_base_url(
                "https://edge-staging.dragon.example",
                Some("https://staging.dragon.example/"),
            )
            .expect("browser site base url"),
            "https://staging.dragon.example"
        );
        assert_eq!(
            resolve_browser_site_base_url("https://edge.dragon.example", None)
                .expect("inferred browser site base url"),
            "https://dragon.example"
        );
    }

    #[test]
    fn probe_swarm_opens_listener_for_webrtc_direct_targets() {
        assert_eq!(
            probe_swarm_listen_address_for_target(
                "/dns4/edge.dragon.example/udp/443/webrtc-direct/certhash/uEiabc"
            ),
            Some("/ip4/0.0.0.0/udp/0/webrtc-direct")
        );
        assert_eq!(
            probe_swarm_listen_address_for_target("/dns4/edge.dragon.example/tcp/4001"),
            None
        );
    }

    #[test]
    fn native_browser_auth_listener_accepts_bridge_auth_result_and_updates_cache() {
        let storage = tempdir().expect("storage");
        let (_, identity) = edge_peer_identity_for_storage(storage.path(), None).expect("identity");
        let requested_scopes = BTreeSet::from([
            ExperimentScope::Connect,
            ExperimentScope::Train {
                experiment_id: ExperimentId::new("nca-prepretraining"),
            },
        ]);
        let enrollment = test_enrollment(requested_scopes);
        let session = test_session(&enrollment);
        let certificate = test_certificate(&enrollment, &session, &identity);
        let auth_result = NativeCliBridgeAuthResult {
            edge_base_url: "https://edge.dragon.example".into(),
            enrollment,
            session,
            certificate,
        };
        let listener = start_native_browser_auth_listener().expect("listener");
        let callback_url = listener.callback_url.clone();
        let nonce = listener.nonce.clone();
        let response = post_form(
            &callback_url,
            &[
                ("native_nonce", nonce),
                (
                    "auth_result_json",
                    serde_json::to_string(&auth_result).expect("auth result json"),
                ),
            ],
        )
        .expect("post callback form");
        assert!(response.starts_with("HTTP/1.1 200 OK"));

        let callback = listener
            .wait(Duration::from_secs(2))
            .expect("auth callback");
        let NativeBrowserAuthCallback::AuthResult(result) = callback else {
            panic!("expected bridge auth result");
        };
        assert_eq!(result.session.session_id, auth_result.session.session_id);

        let authenticated =
            finalize_native_auth_session_from_bridge_result(storage.path(), &result, None)
                .expect("finalize native auth");
        assert!(native_auth_bundle_is_fresh(&authenticated.auth));
        assert!(authenticated.auth.auth_config.local_peer_auth.is_some());
        let cached = load_cached_native_auth_bundle(storage.path())
            .expect("load cached auth")
            .expect("cached auth");
        assert_eq!(cached.session_id, authenticated.auth.session_id);
    }
}
