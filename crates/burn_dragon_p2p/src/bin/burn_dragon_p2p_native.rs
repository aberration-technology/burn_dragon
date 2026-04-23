use std::collections::BTreeSet;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
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
    fetch_directory_entries, fetch_signed_directory_entries, register_live_head,
    rollout_directory_entries, upsert_directory_entry, upsert_directory_entry_current_head,
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
    DragonCapabilityPolicy, DragonExperimentKind, DragonManifestBundle, DragonNativeAuthBundle,
    DragonNativePeerConfig, DragonNativeTarget,
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
#[cfg(feature = "wgpu")]
use burn_dragon_p2p::native::{prepare_climbmix_native_wgpu, prepare_nca_native_wgpu};
use burn_dragon_p2p::profile::DragonExperimentProfile;
use burn_dragon_p2p::profile::build_profile_from_local_config;
use burn_p2p::{
    AuthConfig, ClientPlatform, ClientReleaseManifest, ContentId, ExperimentDirectoryEntry,
    ExperimentId, ExperimentScope, HeadAnnouncement, LiveControlPlaneEvent,
    NativeControlPlaneShell, NetworkId, PeerRoleSet, PrincipalId, ProtocolSet, RuntimeStatus,
    RuntimeTransportPolicy, SwarmAddress,
};
use burn_p2p_admin::AdminResult;
use burn_p2p_core::operator_visible_last_error;
use clap::{CommandFactory, FromArgMatches, Parser, Subcommand, ValueEnum};
use rand::{RngCore, rngs::OsRng};
use serde::{Serialize, de::DeserializeOwned};
use url::Url;

const MIB: u64 = 1024 * 1024;
const DEFAULT_SESSION_TTL_SECS: i64 = 1800;
const DEFAULT_AUTH_CALLBACK_TIMEOUT_SECS: u64 = 300;
const DEFAULT_STATUS_INTERVAL_SECS: u64 = 30;
const DEFAULT_VALIDATION_INTERVAL_MILLIS: u64 = 250;
const DEFAULT_HEAD_SYNC_INTERVAL_SECS: u64 = 15;
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(15);
const STATUS_POLL_INTERVAL: Duration = Duration::from_millis(500);
const RUNTIME_READY_TIMEOUT: Duration = Duration::from_secs(10);

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
    Wgpu,
    Cuda,
}

impl BackendArg {
    fn as_label(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Wgpu => "wgpu",
            Self::Cuda => "cuda",
        }
    }
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

#[derive(Debug, Parser)]
struct ResolveConfigArgs {
    #[arg(long)]
    config: PathBuf,
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
    config: PathBuf,
    #[arg(long, value_enum, default_value = "auto")]
    config_format: ConfigFormat,
    #[arg(long, value_enum)]
    experiment_kind: ExperimentKindArg,
    #[arg(long, value_enum)]
    backend: BackendArg,
    #[arg(long, value_enum, default_value = "toml")]
    output_format: OutputFormat,
    #[command(flatten)]
    capability_policy: CapabilityPolicyArgs,
}

#[derive(Debug, Parser)]
struct DeploymentDiagnosticsArgs {
    #[arg(long)]
    config: PathBuf,
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
struct ProbeSwarmArgs {
    #[arg(long, default_value = "burn-dragon-mainnet")]
    network_id: String,
    #[arg(long)]
    address: String,
    #[arg(long, default_value_t = 15)]
    timeout_secs: u64,
    #[arg(long, default_value_t = 64)]
    max_events: usize,
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
    config: PathBuf,
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
    config: PathBuf,
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
    #[arg(long, value_enum, default_value = "json")]
    output_format: OutputFormat,
}

#[derive(Debug, Parser)]
struct LoginArgs {
    #[arg(long)]
    config: PathBuf,
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
    config: PathBuf,
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
    config: PathBuf,
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
    config: PathBuf,
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
    auth_bundle: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "auto")]
    auth_bundle_format: ConfigFormat,
    #[arg(long, default_value_t = DEFAULT_STATUS_INTERVAL_SECS)]
    status_interval_secs: u64,
    #[arg(long, default_value_t = false)]
    initialize_head_on_start: bool,
    #[arg(long, default_value_t = false)]
    restore_head_on_start: bool,
    #[arg(long, default_value_t = 0)]
    head_sync_interval_secs: u64,
    #[command(flatten)]
    capability_policy: CapabilityPolicyArgs,
}

#[derive(Debug, Parser)]
struct TrainWindowOnceArgs {
    #[arg(long)]
    config: PathBuf,
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
    auth_bundle: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "auto")]
    auth_bundle_format: ConfigFormat,
    #[arg(long, default_value_t = true)]
    initialize_head_on_start: bool,
    #[arg(long, default_value_t = true)]
    restore_head_on_start: bool,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "json")]
    output_format: OutputFormat,
    #[command(flatten)]
    capability_policy: CapabilityPolicyArgs,
}

#[derive(Debug, Parser)]
struct RunHeadMirrorArgs {
    #[arg(long)]
    config: PathBuf,
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
    #[arg(long, default_value_t = true)]
    initialize_head_on_start: bool,
    #[arg(long, default_value_t = true)]
    restore_head_on_start: bool,
    #[command(flatten)]
    capability_policy: CapabilityPolicyArgs,
}

#[derive(Debug, Parser)]
struct RunValidatorDaemonArgs {
    #[arg(long)]
    config: PathBuf,
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
    #[arg(long, default_value_t = true)]
    initialize_head_on_start: bool,
    #[arg(long, default_value_t = true)]
    restore_head_on_start: bool,
    #[command(flatten)]
    capability_policy: CapabilityPolicyArgs,
}

#[derive(Debug, Parser)]
struct MarkRuntimeFailureArgs {
    #[arg(long)]
    config: PathBuf,
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
    config: PathBuf,
    #[arg(long, value_enum, default_value = "auto")]
    config_format: ConfigFormat,
    #[arg(long, value_enum)]
    experiment_kind: ExperimentKindArg,
    #[arg(long, value_enum)]
    backend: BackendArg,
}

#[derive(Debug, Serialize)]
struct CapabilityAssessmentReport {
    config_path: PathBuf,
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
    session_id: String,
    experiment_id: String,
    revision_id: String,
    directory_entries: usize,
    result: AdminResult,
}

#[derive(Debug, Serialize)]
struct TrainWindowOnceTimingReport {
    data_fetch_time_ms: u64,
    publish_latency_ms: u64,
}

#[derive(Debug, Serialize)]
struct TrainWindowOnceReport {
    experiment_kind: DragonExperimentKind,
    backend: String,
    local_peer_id: String,
    base_head_id: String,
    published_head_id: String,
    published_global_step: u64,
    artifact_id: String,
    contribution_receipt_id: String,
    lease_window_id: String,
    lease_microshard_count: usize,
    timing: TrainWindowOnceTimingReport,
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
    last_error: Option<String>,
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

    let elapsed_millis = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
    let connected = connected_peer_id.is_some();
    let report = ProbeSwarmReport {
        network_id: args.network_id,
        address: args.address,
        local_peer_id,
        connected,
        connected_peer_id,
        elapsed_millis,
        events,
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
        &args.config,
        args.config_format,
        args.edge_url,
        args.seed_node_urls,
        Some(args.capability_policy),
    )?;
    write_output(None, args.output_format, &config)
}

fn assess_capability(args: AssessCapabilityArgs) -> Result<()> {
    let config = resolved_config(
        &args.config,
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
        &args.config,
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
    let config = load_native_config(&args.config, args.config_format)?;
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
    let edge_base_url = args
        .edge_url
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
    let replacement = manifests
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
            session_id,
            experiment_id: replacement.experiment_id.as_str().to_owned(),
            revision_id: replacement.current_revision_id.as_str().to_owned(),
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
        "HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\nCache-Control: no-store\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
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
    let mut request_line = String::new();
    let mut reader = BufReader::new(stream.try_clone()?);
    reader.read_line(&mut request_line)?;
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
    loop {
        let mut header = String::new();
        reader.read_line(&mut header)?;
        if header == "\r\n" || header.is_empty() {
            break;
        }
        if let Some(value) = header.split_once(':')
            && value.0.eq_ignore_ascii_case("content-length")
        {
            content_length = value.1.trim().parse::<usize>().unwrap_or_default();
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

fn infer_browser_site_base_url(edge_base_url: &str) -> Result<String> {
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

fn build_native_cli_browser_auth_bootstrap(
    config: &DragonNativePeerConfig,
    experiment_kind: DragonExperimentKind,
    backend: BackendArg,
    principal_hint: Option<String>,
    session_ttl_secs: i64,
) -> Result<NativeCliBridgeBootstrap> {
    let edge_base_url = config
        .effective_edge_base_url()
        .ok_or_else(|| anyhow!("no edge base URL configured"))?
        .to_owned();
    let site_base_url = infer_browser_site_base_url(&edge_base_url)?;
    let manifests = prepared_manifests(config, experiment_kind, backend)?;
    let requested_scopes = requested_scopes(
        manifests
            .experiment_directory
            .first()
            .ok_or_else(|| anyhow!("manifest bundle is missing an experiment directory entry"))?,
    );
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
            .unwrap_or_else(|| backend.as_label().into()),
        requested_scopes,
        session_ttl_secs,
        principal_hint,
        identity,
    })
}

fn build_pending_native_login(
    config: &DragonNativePeerConfig,
    experiment_kind: DragonExperimentKind,
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
    let manifests = prepared_manifests(config, experiment_kind, backend)?;
    let requested_scopes = requested_scopes(
        manifests
            .experiment_directory
            .first()
            .ok_or_else(|| anyhow!("manifest bundle is missing an experiment directory entry"))?,
    );
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
    match open_url_in_system_browser(&bridge_url) {
        Ok(()) => eprintln!("launched browser for GitHub login"),
        Err(error) => {
            eprintln!("automatic browser launch failed: {error}");
            eprintln!(
                "Open this URL to continue GitHub login:
{bridge_url}"
            );
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
        &args.config,
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
        &args.config,
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
    let config = load_native_config(&args.config, args.config_format)?;
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
        &args.config,
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
    let config = resolved_config(
        &args.config,
        args.config_format,
        args.edge_url,
        args.seed_node_urls,
        Some(args.capability_policy),
    )?;
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

    match (args.experiment_kind.into_config(), args.backend) {
        (DragonExperimentKind::NcaPrepretraining, BackendArg::Cpu) => {
            run_prepared_train_window_once(
                prepare_nca_native_cpu(&config, Some(&auth_bundle))?,
                &config,
                args.backend,
                args.initialize_head_on_start,
                args.restore_head_on_start,
                args.output.as_deref(),
                args.output_format,
            )
        }
        (DragonExperimentKind::ClimbMixPretraining, BackendArg::Cpu) => {
            run_prepared_train_window_once(
                prepare_climbmix_native_cpu(&config, Some(&auth_bundle))?,
                &config,
                args.backend,
                args.initialize_head_on_start,
                args.restore_head_on_start,
                args.output.as_deref(),
                args.output_format,
            )
        }
        #[cfg(feature = "wgpu")]
        (DragonExperimentKind::NcaPrepretraining, BackendArg::Wgpu) => {
            run_prepared_train_window_once(
                prepare_nca_native_wgpu(&config, Some(&auth_bundle))?,
                &config,
                args.backend,
                args.initialize_head_on_start,
                args.restore_head_on_start,
                args.output.as_deref(),
                args.output_format,
            )
        }
        #[cfg(feature = "wgpu")]
        (DragonExperimentKind::ClimbMixPretraining, BackendArg::Wgpu) => {
            run_prepared_train_window_once(
                prepare_climbmix_native_wgpu(&config, Some(&auth_bundle))?,
                &config,
                args.backend,
                args.initialize_head_on_start,
                args.restore_head_on_start,
                args.output.as_deref(),
                args.output_format,
            )
        }
        #[cfg(feature = "cuda")]
        (DragonExperimentKind::NcaPrepretraining, BackendArg::Cuda) => {
            run_prepared_train_window_once(
                prepare_nca_native_cuda(&config, Some(&auth_bundle))?,
                &config,
                args.backend,
                args.initialize_head_on_start,
                args.restore_head_on_start,
                args.output.as_deref(),
                args.output_format,
            )
        }
        #[cfg(feature = "cuda")]
        (DragonExperimentKind::ClimbMixPretraining, BackendArg::Cuda) => {
            run_prepared_train_window_once(
                prepare_climbmix_native_cuda(&config, Some(&auth_bundle))?,
                &config,
                args.backend,
                args.initialize_head_on_start,
                args.restore_head_on_start,
                args.output.as_deref(),
                args.output_format,
            )
        }
        #[cfg(not(feature = "wgpu"))]
        (_, BackendArg::Wgpu) => bail!("this binary was built without the `wgpu` feature"),
        #[cfg(not(feature = "cuda"))]
        (_, BackendArg::Cuda) => bail!("this binary was built without the `cuda` feature"),
    }
}

fn native_target_artifact_id(backend: BackendArg) -> &'static str {
    match backend {
        BackendArg::Cpu => "native-cpu",
        BackendArg::Wgpu => "native-wgpu",
        BackendArg::Cuda => "native-cuda",
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
    Ok(ClientReleaseManifest {
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
        burn_version_string: "0.21.0-pre.3".into(),
        enabled_features_hash: ContentId::new(
            config
                .enabled_features_label
                .clone()
                .unwrap_or_else(|| backend.as_label().into()),
        ),
        protocol_major: 0,
        supported_workloads: Vec::new(),
        built_at: chrono::Utc::now(),
    })
}

fn run_peer(args: RunPeerArgs) -> Result<()> {
    let config = resolved_config(
        &args.config,
        args.config_format,
        args.edge_url,
        args.seed_node_urls,
        Some(args.capability_policy),
    )?;
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

    match (args.experiment_kind.into_config(), args.backend) {
        (DragonExperimentKind::NcaPrepretraining, BackendArg::Cpu) => run_prepared_peer(
            prepare_nca_native_cpu(&config, auth_bundle.as_ref())?,
            &config,
            args.backend,
            args.status_interval_secs,
            args.initialize_head_on_start,
            args.restore_head_on_start,
            args.head_sync_interval_secs,
        ),
        (DragonExperimentKind::ClimbMixPretraining, BackendArg::Cpu) => run_prepared_peer(
            prepare_climbmix_native_cpu(&config, auth_bundle.as_ref())?,
            &config,
            args.backend,
            args.status_interval_secs,
            args.initialize_head_on_start,
            args.restore_head_on_start,
            args.head_sync_interval_secs,
        ),
        #[cfg(feature = "wgpu")]
        (DragonExperimentKind::NcaPrepretraining, BackendArg::Wgpu) => run_prepared_peer(
            prepare_nca_native_wgpu(&config, auth_bundle.as_ref())?,
            &config,
            args.backend,
            args.status_interval_secs,
            args.initialize_head_on_start,
            args.restore_head_on_start,
            args.head_sync_interval_secs,
        ),
        #[cfg(feature = "wgpu")]
        (DragonExperimentKind::ClimbMixPretraining, BackendArg::Wgpu) => run_prepared_peer(
            prepare_climbmix_native_wgpu(&config, auth_bundle.as_ref())?,
            &config,
            args.backend,
            args.status_interval_secs,
            args.initialize_head_on_start,
            args.restore_head_on_start,
            args.head_sync_interval_secs,
        ),
        #[cfg(feature = "cuda")]
        (DragonExperimentKind::NcaPrepretraining, BackendArg::Cuda) => run_prepared_peer(
            prepare_nca_native_cuda(&config, auth_bundle.as_ref())?,
            &config,
            args.backend,
            args.status_interval_secs,
            args.initialize_head_on_start,
            args.restore_head_on_start,
            args.head_sync_interval_secs,
        ),
        #[cfg(feature = "cuda")]
        (DragonExperimentKind::ClimbMixPretraining, BackendArg::Cuda) => run_prepared_peer(
            prepare_climbmix_native_cuda(&config, auth_bundle.as_ref())?,
            &config,
            args.backend,
            args.status_interval_secs,
            args.initialize_head_on_start,
            args.restore_head_on_start,
            args.head_sync_interval_secs,
        ),
        #[cfg(not(feature = "wgpu"))]
        (_, BackendArg::Wgpu) => bail!("this binary was built without the `wgpu` feature"),
        #[cfg(not(feature = "cuda"))]
        (_, BackendArg::Cuda) => bail!("this binary was built without the `cuda` feature"),
    }
}

fn run_head_mirror(args: RunHeadMirrorArgs) -> Result<()> {
    let config = resolved_config(
        &args.config,
        args.config_format,
        args.edge_url,
        args.seed_node_urls,
        Some(args.capability_policy),
    )?;
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

    match (args.experiment_kind.into_config(), args.backend) {
        (DragonExperimentKind::NcaPrepretraining, BackendArg::Cpu) => run_prepared_head_mirror(
            prepare_nca_native_cpu(&config, auth_bundle.as_ref())?,
            &config,
            auth_bundle.as_ref(),
            args.backend,
            args.status_interval_secs,
            args.head_sync_interval_secs,
            args.initialize_head_on_start,
            args.restore_head_on_start,
        ),
        (DragonExperimentKind::ClimbMixPretraining, BackendArg::Cpu) => run_prepared_head_mirror(
            prepare_climbmix_native_cpu(&config, auth_bundle.as_ref())?,
            &config,
            auth_bundle.as_ref(),
            args.backend,
            args.status_interval_secs,
            args.head_sync_interval_secs,
            args.initialize_head_on_start,
            args.restore_head_on_start,
        ),
        #[cfg(feature = "wgpu")]
        (DragonExperimentKind::NcaPrepretraining, BackendArg::Wgpu) => run_prepared_head_mirror(
            prepare_nca_native_wgpu(&config, auth_bundle.as_ref())?,
            &config,
            auth_bundle.as_ref(),
            args.backend,
            args.status_interval_secs,
            args.head_sync_interval_secs,
            args.initialize_head_on_start,
            args.restore_head_on_start,
        ),
        #[cfg(feature = "wgpu")]
        (DragonExperimentKind::ClimbMixPretraining, BackendArg::Wgpu) => run_prepared_head_mirror(
            prepare_climbmix_native_wgpu(&config, auth_bundle.as_ref())?,
            &config,
            auth_bundle.as_ref(),
            args.backend,
            args.status_interval_secs,
            args.head_sync_interval_secs,
            args.initialize_head_on_start,
            args.restore_head_on_start,
        ),
        #[cfg(feature = "cuda")]
        (DragonExperimentKind::NcaPrepretraining, BackendArg::Cuda) => run_prepared_head_mirror(
            prepare_nca_native_cuda(&config, auth_bundle.as_ref())?,
            &config,
            auth_bundle.as_ref(),
            args.backend,
            args.status_interval_secs,
            args.head_sync_interval_secs,
            args.initialize_head_on_start,
            args.restore_head_on_start,
        ),
        #[cfg(feature = "cuda")]
        (DragonExperimentKind::ClimbMixPretraining, BackendArg::Cuda) => run_prepared_head_mirror(
            prepare_climbmix_native_cuda(&config, auth_bundle.as_ref())?,
            &config,
            auth_bundle.as_ref(),
            args.backend,
            args.status_interval_secs,
            args.head_sync_interval_secs,
            args.initialize_head_on_start,
            args.restore_head_on_start,
        ),
        #[cfg(not(feature = "wgpu"))]
        (_, BackendArg::Wgpu) => bail!("this binary was built without the `wgpu` feature"),
        #[cfg(not(feature = "cuda"))]
        (_, BackendArg::Cuda) => bail!("this binary was built without the `cuda` feature"),
    }
}

fn run_validator_daemon(args: RunValidatorDaemonArgs) -> Result<()> {
    let config = resolved_config(
        &args.config,
        args.config_format,
        args.edge_url,
        args.seed_node_urls,
        Some(args.capability_policy),
    )?;
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

    match (args.experiment_kind.into_config(), args.backend) {
        (DragonExperimentKind::NcaPrepretraining, BackendArg::Cpu) => {
            run_prepared_validator_daemon(
                prepare_nca_native_cpu(&config, auth_bundle.as_ref())?,
                &config,
                args.backend,
                args.status_interval_secs,
                args.validation_interval_millis,
                args.initialize_head_on_start,
                args.restore_head_on_start,
            )
        }
        (DragonExperimentKind::ClimbMixPretraining, BackendArg::Cpu) => {
            run_prepared_validator_daemon(
                prepare_climbmix_native_cpu(&config, auth_bundle.as_ref())?,
                &config,
                args.backend,
                args.status_interval_secs,
                args.validation_interval_millis,
                args.initialize_head_on_start,
                args.restore_head_on_start,
            )
        }
        #[cfg(feature = "wgpu")]
        (DragonExperimentKind::NcaPrepretraining, BackendArg::Wgpu) => {
            run_prepared_validator_daemon(
                prepare_nca_native_wgpu(&config, auth_bundle.as_ref())?,
                &config,
                args.backend,
                args.status_interval_secs,
                args.validation_interval_millis,
                args.initialize_head_on_start,
                args.restore_head_on_start,
            )
        }
        #[cfg(feature = "wgpu")]
        (DragonExperimentKind::ClimbMixPretraining, BackendArg::Wgpu) => {
            run_prepared_validator_daemon(
                prepare_climbmix_native_wgpu(&config, auth_bundle.as_ref())?,
                &config,
                args.backend,
                args.status_interval_secs,
                args.validation_interval_millis,
                args.initialize_head_on_start,
                args.restore_head_on_start,
            )
        }
        #[cfg(feature = "cuda")]
        (DragonExperimentKind::NcaPrepretraining, BackendArg::Cuda) => {
            run_prepared_validator_daemon(
                prepare_nca_native_cuda(&config, auth_bundle.as_ref())?,
                &config,
                args.backend,
                args.status_interval_secs,
                args.validation_interval_millis,
                args.initialize_head_on_start,
                args.restore_head_on_start,
            )
        }
        #[cfg(feature = "cuda")]
        (DragonExperimentKind::ClimbMixPretraining, BackendArg::Cuda) => {
            run_prepared_validator_daemon(
                prepare_climbmix_native_cuda(&config, auth_bundle.as_ref())?,
                &config,
                args.backend,
                args.status_interval_secs,
                args.validation_interval_millis,
                args.initialize_head_on_start,
                args.restore_head_on_start,
            )
        }
        #[cfg(not(feature = "wgpu"))]
        (_, BackendArg::Wgpu) => bail!("this binary was built without the `wgpu` feature"),
        #[cfg(not(feature = "cuda"))]
        (_, BackendArg::Cuda) => bail!("this binary was built without the `cuda` feature"),
    }
}

fn mark_runtime_failure(args: MarkRuntimeFailureArgs) -> Result<()> {
    let config = resolved_config(
        &args.config,
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
    let config = load_native_config(&args.config, args.config_format)?;
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
    path: &Path,
    format: ConfigFormat,
    edge_url: Option<String>,
    seed_node_urls: Vec<String>,
    capability_policy: Option<CapabilityPolicyArgs>,
) -> Result<DragonNativePeerConfig> {
    let mut config = load_native_config(path, format)?;
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
    match (experiment_kind, backend) {
        (DragonExperimentKind::NcaPrepretraining, BackendArg::Cpu) => {
            Ok(prepare_nca_native_cpu(config, Some(&placeholder_auth))?.manifests)
        }
        (DragonExperimentKind::ClimbMixPretraining, BackendArg::Cpu) => {
            Ok(prepare_climbmix_native_cpu(config, Some(&placeholder_auth))?.manifests)
        }
        #[cfg(feature = "wgpu")]
        (DragonExperimentKind::NcaPrepretraining, BackendArg::Wgpu) => {
            Ok(prepare_nca_native_wgpu(config, Some(&placeholder_auth))?.manifests)
        }
        #[cfg(feature = "wgpu")]
        (DragonExperimentKind::ClimbMixPretraining, BackendArg::Wgpu) => {
            Ok(prepare_climbmix_native_wgpu(config, Some(&placeholder_auth))?.manifests)
        }
        #[cfg(feature = "cuda")]
        (DragonExperimentKind::NcaPrepretraining, BackendArg::Cuda) => {
            Ok(prepare_nca_native_cuda(config, Some(&placeholder_auth))?.manifests)
        }
        #[cfg(feature = "cuda")]
        (DragonExperimentKind::ClimbMixPretraining, BackendArg::Cuda) => {
            Ok(prepare_climbmix_native_cuda(config, Some(&placeholder_auth))?.manifests)
        }
        #[cfg(not(feature = "wgpu"))]
        (_, BackendArg::Wgpu) => bail!("this binary was built without the `wgpu` feature"),
        #[cfg(not(feature = "cuda"))]
        (_, BackendArg::Cuda) => bail!("this binary was built without the `cuda` feature"),
    }
}

fn requested_scopes(entry: &ExperimentDirectoryEntry) -> BTreeSet<ExperimentScope> {
    standard_experiment_scopes(&entry.experiment_id)
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
        let mut mirrored_head_id = None;
        let _ = sync_or_initialize_head_provider(
            &mut running,
            &experiment,
            initialize_head_on_start,
            restore_head_on_start,
            &mut mirrored_head_id,
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
    let mut mirrored_head_id = None;
    let mut last_head_sync = Instant::now()
        .checked_sub(head_sync_interval)
        .unwrap_or_else(Instant::now);

    loop {
        if head_sync_interval_secs > 0
            && let Some(experiment) = experiment.as_ref()
            && last_head_sync.elapsed() >= head_sync_interval
        {
            let _ = sync_or_initialize_head_provider(
                &mut running,
                experiment,
                initialize_head_on_start,
                restore_head_on_start,
                &mut mirrored_head_id,
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
    backend: BackendArg,
    initialize_head_on_start: bool,
    restore_head_on_start: bool,
    output: Option<&Path>,
    output_format: OutputFormat,
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

    let mut running = spawn_prepared_native_peer(prepared)?;
    let report_result = (|| -> Result<TrainWindowOnceReport> {
        wait_for_runtime_ready(&running, RUNTIME_READY_TIMEOUT)?;
        let local_peer_id = running
            .snapshot()
            .local_peer_id
            .ok_or_else(|| anyhow!("peer runtime did not report a local peer id"))?;
        let experiment = running.mainnet().experiment(
            experiment_entry.study_id,
            experiment_entry.experiment_id,
            experiment_entry.current_revision_id,
        );
        let mut mirrored_head_id = None;
        let base_head = sync_or_initialize_head_provider(
            &mut running,
            &experiment,
            initialize_head_on_start,
            restore_head_on_start,
            &mut mirrored_head_id,
            "trainer",
        )?
        .ok_or_else(|| {
            anyhow!(
                "no experiment head is available; rerun with --initialize-head-on-start or seed a head first"
            )
        })?;
        let mut trainer = running.continuous_trainer(&experiment)?;
        let outcome = trainer.train_next_window()?;
        Ok(TrainWindowOnceReport {
            experiment_kind: running.prepared().experiment_kind,
            backend: backend.as_label().into(),
            local_peer_id: local_peer_id.as_str().to_owned(),
            base_head_id: base_head.head_id.as_str().to_owned(),
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
    write_output(output, output_format, &report)
}

fn sync_or_initialize_head_provider<B>(
    running: &mut ManagedRunningNativePeer<B>,
    experiment: &burn_p2p::ExperimentHandle,
    initialize_head_on_start: bool,
    restore_head_on_start: bool,
    mirrored_head_id: &mut Option<burn_p2p::HeadId>,
    log_prefix: &str,
) -> Result<Option<burn_p2p::HeadDescriptor>>
where
    B: AutodiffBackend + Clone + 'static,
{
    let restored = if restore_head_on_start {
        running.restore_experiment_head(experiment)?
    } else {
        None
    };
    let synced = running.sync_experiment_head(experiment)?;
    let head = if let Some(head) = synced.or(restored) {
        head
    } else if initialize_head_on_start {
        let head = running.initialize_local_head(experiment)?;
        eprintln!(
            "{log_prefix}-initialized genesis head id={} global_step={}",
            head.head_id.as_str(),
            head.global_step,
        );
        head
    } else {
        return Ok(None);
    };

    // Re-announce the locally materialized head on every sync pass so late
    // browser peers can always discover at least one live provider.
    running.publish_head_provider(experiment, &head)?;

    if mirrored_head_id.as_ref() != Some(&head.head_id) {
        eprintln!(
            "{log_prefix}-mirroring head id={} global_step={}",
            head.head_id.as_str(),
            head.global_step,
        );
        *mirrored_head_id = Some(head.head_id.clone());
    }

    Ok(Some(head))
}

fn register_live_head_with_edge(
    runtime: &tokio::runtime::Runtime,
    edge_base_url: &str,
    session_id: &str,
    directory_template: &ExperimentDirectoryEntry,
    announcement: &HeadAnnouncement,
) -> Result<()> {
    let _ = runtime.block_on(register_live_head(
        edge_base_url,
        session_id,
        announcement.clone(),
    ))?;
    let mut directory_entries =
        runtime.block_on(fetch_signed_directory_entries(edge_base_url, session_id))?;
    if upsert_directory_entry_current_head(
        &mut directory_entries,
        directory_template,
        announcement.head.head_id.clone(),
    ) {
        let _ = runtime.block_on(rollout_directory_entries(
            edge_base_url,
            session_id,
            directory_entries,
        ))?;
    }
    Ok(())
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
    let mut mirrored_head_id = None;

    loop {
        if last_head_sync.elapsed() >= head_sync_interval {
            let head = sync_or_initialize_head_provider(
                &mut running,
                &experiment,
                initialize_head_on_start,
                restore_head_on_start,
                &mut mirrored_head_id,
                "head-mirror",
            )?;
            if let (Some(head), Some((registration_runtime, edge_base_url, session_id))) =
                (head.as_ref(), edge_registration.as_ref())
            {
                let snapshot = running.snapshot();
                if let Some(local_peer_id) = snapshot.local_peer_id {
                    let announcement = HeadAnnouncement {
                        overlay: experiment.overlay_set()?.heads,
                        provider_peer_id: Some(local_peer_id),
                        head: head.clone(),
                        announced_at: chrono::Utc::now(),
                    };
                    if let Err(error) = register_live_head_with_edge(
                        registration_runtime,
                        edge_base_url,
                        session_id,
                        &experiment_entry,
                        &announcement,
                    ) {
                        eprintln!("head-mirror-edge-registration-failed: {error}");
                    }
                }
            }
            last_head_sync = Instant::now();
        }

        let snapshot = running.snapshot();
        if status_interval_secs > 0 && last_status.elapsed() >= status_interval {
            eprintln!(
                "head-mirror-status status={:?} node_state={:?} connected_peers={} mirrored_head={} last_error={}",
                snapshot.status,
                snapshot.node_state,
                snapshot.connected_peers,
                mirrored_head_id
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
    eprintln!(
        "starting burn_dragon validator daemon: experiment={} backend={} target={:?} can_train={} edge={} seeds={} storage_root={}",
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
    let restored = if restore_head_on_start {
        running.restore_experiment_head(&experiment)?
    } else {
        None
    };
    let synced = running.sync_experiment_head(&experiment)?;
    if restored.is_none() && synced.is_none() && initialize_head_on_start {
        let head = running.initialize_local_head(&experiment)?;
        eprintln!(
            "validator-daemon initialized genesis head id={} global_step={}",
            head.head_id.as_str(),
            head.global_step,
        );
    }

    let status_interval = Duration::from_secs(status_interval_secs.max(1));
    let validation_interval = Duration::from_millis(validation_interval_millis.max(25));
    let mut last_status = Instant::now()
        .checked_sub(status_interval)
        .unwrap_or_else(Instant::now);
    let mut last_validation = Instant::now()
        .checked_sub(validation_interval)
        .unwrap_or_else(Instant::now);

    loop {
        let snapshot = running.snapshot();
        if status_interval_secs > 0 && last_status.elapsed() >= status_interval {
            eprintln!(
                "validator-status status={:?} node_state={:?} connected_peers={} last_error={}",
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
