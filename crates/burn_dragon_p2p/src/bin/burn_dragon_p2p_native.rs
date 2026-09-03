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
use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
use burn_autodiff::Autodiff;
use burn_dragon_language::{TrainingConfig, load_training_config, train};
use burn_dragon_p2p::admin::{
    fetch_directory_entries, fetch_signed_directory_entries, mirror_peer_artifact,
    preserve_directory_entry_current_head, recover_directory_current_head_from_visible_roots,
    register_live_head, rollout_directory_entries, rollout_revision_contracts,
    upsert_directory_entry, upsert_directory_entry_current_head,
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
    prepare_nca_native_cpu, prepare_ruliad_native_cpu, spawn_prepared_native_peer,
};
#[cfg(feature = "cuda")]
use burn_dragon_p2p::native::{
    prepare_climbmix_native_cuda, prepare_nca_native_cuda, prepare_ruliad_native_cuda,
};
#[cfg(feature = "rocm")]
use burn_dragon_p2p::native::{
    prepare_climbmix_native_rocm, prepare_nca_native_rocm, prepare_ruliad_native_rocm,
};
#[cfg(feature = "wgpu")]
use burn_dragon_p2p::native::{
    prepare_climbmix_native_wgpu, prepare_nca_native_wgpu, prepare_ruliad_native_wgpu,
};
use burn_dragon_p2p::profile::DragonExperimentProfile;
use burn_dragon_p2p::profile::build_profile_from_local_config;
use burn_ndarray::NdArray;
use burn_p2p::{
    AuthConfig, ClientPlatform, ClientReleaseManifest, ContentId, ControlPlaneSnapshot,
    ExperimentDirectoryEntry, ExperimentDirectoryPolicyExt, ExperimentHandle, ExperimentId,
    ExperimentScope, HeadAnnouncement, HeadDescriptor, HeadId, HeadPromotionMode,
    LiveControlPlaneEvent, MetricValue, NativeControlPlaneShell, NetworkId, PeerId, PeerRole,
    PeerRoleSet, PrincipalId, ProtocolSet, RuntimeStatus, RuntimeTransportPolicy, SwarmAddress,
    TrainingProtocolStepOutcome, directory_revision_contract_matches,
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
const DEFAULT_TRAIN_WINDOW_HEAD_SYNC_TIMEOUT_SECS: u64 = 600;
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
    AdminProvisionRevisionContract(AdminProvisionRevisionContractArgs),
    #[command(alias = "github-login")]
    Login(LoginArgs),
    #[command(alias = "begin-login")]
    BeginGithubLogin(BeginGithubLoginArgs),
    #[command(alias = "complete-login")]
    CompleteGithubLogin(CompleteGithubLoginArgs),
    EnrollStaticPrincipal(EnrollStaticPrincipalArgs),
    TrainLocal(TrainLocalArgs),
    MonitorRun(MonitorRunArgs),
    TrainWindowOnce(TrainWindowOnceArgs),
    RunPeer(RunPeerArgs),
    RunTrainerDaemon(RunTrainerDaemonArgs),
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
    Ruliad,
    Climbmix,
}

impl ExperimentKindArg {
    fn into_config(self) -> DragonExperimentKind {
        match self {
            Self::Nca => DragonExperimentKind::NcaPrepretraining,
            Self::Ruliad => DragonExperimentKind::RuliadPretraining,
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
            (DragonExperimentKind::RuliadPretraining, BackendArg::Cpu) => {
                let $prepared = prepare_ruliad_native_cpu($config, $auth_bundle)?;
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
            (DragonExperimentKind::RuliadPretraining, BackendArg::Wgpu) => {
                let $prepared = prepare_ruliad_native_wgpu($config, $auth_bundle)?;
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
            (DragonExperimentKind::RuliadPretraining, BackendArg::Cuda) => {
                let $prepared = prepare_ruliad_native_cuda($config, $auth_bundle)?;
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
            (DragonExperimentKind::RuliadPretraining, BackendArg::Rocm) => {
                let $prepared = prepare_ruliad_native_rocm($config, $auth_bundle)?;
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum TrainingProgressRendererArg {
    Quiet,
    Default,
}

impl TrainingProgressRendererArg {
    fn as_env(self) -> &'static str {
        match self {
            Self::Quiet => "quiet",
            Self::Default => "default",
        }
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

#[derive(Clone, Copy, Debug, Default, Parser)]
struct LocalTrainingOverrideArgs {
    #[arg(long = "n-layer", value_name = "LAYERS")]
    n_layer: Option<usize>,
    #[arg(long = "n-embd", value_name = "DIM")]
    n_embd: Option<usize>,
    #[arg(long = "n-head", value_name = "HEADS")]
    n_head: Option<usize>,
    #[arg(long = "latent-total", value_name = "LATENTS")]
    latent_total: Option<usize>,
    #[arg(long = "training-block-size", value_name = "TOKENS")]
    block_size: Option<usize>,
    #[arg(long = "training-batch-size", value_name = "BATCH_SIZE")]
    batch_size: Option<usize>,
    #[arg(long = "training-max-iters", value_name = "ITERS")]
    max_iters: Option<usize>,
    #[arg(long = "checkpoint-interval-iters", value_name = "ITERS")]
    checkpoint_interval_iters: Option<usize>,
}

impl LocalTrainingOverrideArgs {
    fn apply_to(self, config: &mut TrainingConfig) -> Result<()> {
        if let Some(n_layer) = self.n_layer {
            config.model.n_layer = Some(n_layer);
        }
        if let Some(n_embd) = self.n_embd {
            config.model.n_embd = Some(n_embd);
        }
        if let Some(n_head) = self.n_head {
            config.model.n_head = Some(n_head);
        }
        if let Some(latent_total) = self.latent_total {
            config.model.latent_total = Some(latent_total);
            if let Some(n_embd) = self.n_embd.or(config.model.n_embd) {
                if latent_total % n_embd != 0 {
                    bail!(
                        "--latent-total must be divisible by the resolved --n-embd/model.n_embd (got latent_total={latent_total} n_embd={n_embd})"
                    );
                }
                config.model.mlp_internal_dim_multiplier = Some(latent_total / n_embd);
            }
        }
        if let Some(block_size) = self.block_size {
            config.training.block_size = block_size;
            config.model.block_size = Some(block_size);
        }
        if let Some(batch_size) = self.batch_size {
            config.training.batch_size = batch_size;
        }
        if let Some(max_iters) = self.max_iters {
            config.training.max_iters = max_iters;
        }
        if let Some(checkpoint_interval_iters) = self.checkpoint_interval_iters {
            config.training.checkpoint_interval_iters = checkpoint_interval_iters;
        }
        config.validate()
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
    require_revision_contract: bool,
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
struct AdminProvisionRevisionContractArgs {
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
    authority_key: PathBuf,
    #[arg(long)]
    contract_out: PathBuf,
    #[arg(long)]
    edge_url: Option<String>,
    #[arg(long, default_value_t = 1)]
    authority_epoch: u64,
    #[arg(long, default_value = "burn-dragon-deterministic-init-v1")]
    initialization_algorithm: String,
    #[arg(long, default_value_t = 600)]
    wait_timeout_secs: u64,
    #[arg(long, default_value_t = 5)]
    poll_interval_secs: u64,
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
struct RunTrainerDaemonArgs {
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
    #[arg(long, default_value_t = false, action = ArgAction::Set)]
    initialize_head_on_start: bool,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    restore_head_on_start: bool,
    #[arg(long, default_value_t = DEFAULT_HEAD_SYNC_INTERVAL_SECS)]
    head_sync_interval_secs: u64,
    #[arg(long, default_value_t = 1)]
    minimum_step_interval_secs: u64,
    #[arg(long, default_value_t = 5)]
    failure_backoff_secs: u64,
    #[arg(long, default_value_t = 12)]
    max_consecutive_failures: u32,
    /// Stops cleanly after this many successful protocol steps; zero runs indefinitely.
    #[arg(long, default_value_t = 0)]
    max_protocol_steps: u64,
    #[command(flatten)]
    training_overrides: NativeTrainingOverrideArgs,
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
struct TrainLocalArgs {
    #[arg(long = "training-config", alias = "config", required = true)]
    training_config_paths: Vec<PathBuf>,
    #[arg(long, value_enum, default_value = "cpu")]
    backend: BackendArg,
    #[arg(long, value_enum, default_value = "quiet")]
    progress: TrainingProgressRendererArg,
    #[arg(long)]
    run_root: Option<PathBuf>,
    #[arg(long)]
    run_dir: Option<PathBuf>,
    #[arg(long)]
    run_name: Option<String>,
    #[command(flatten)]
    training_overrides: LocalTrainingOverrideArgs,
}

#[derive(Debug, Parser)]
struct MonitorRunArgs {
    #[arg(long)]
    run_dir: PathBuf,
    #[arg(long)]
    run_name: Option<String>,
    #[arg(long, default_value_t = false)]
    follow: bool,
    #[arg(long, default_value_t = 5)]
    poll_interval_secs: u64,
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
    #[arg(long, default_value_t = false, action = ArgAction::Set)]
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
    revision_contract_changed: bool,
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

#[derive(Clone, Copy)]
struct TrainerDaemonPolicy {
    minimum_step_interval: Duration,
    failure_backoff: Duration,
    max_consecutive_failures: u32,
    max_protocol_steps: Option<u64>,
}

#[derive(Clone, Copy)]
struct NativePeerServiceOptions {
    backend: BackendArg,
    status_interval_secs: u64,
    initialize_head_on_start: bool,
    restore_head_on_start: bool,
    head_sync_interval_secs: u64,
    trainer_daemon: Option<TrainerDaemonPolicy>,
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
        CommandKind::AdminProvisionRevisionContract(args) => {
            admin_provision_revision_contract(args)
        }
        CommandKind::Login(args) => login(args),
        CommandKind::BeginGithubLogin(args) => begin_github_login(args),
        CommandKind::CompleteGithubLogin(args) => complete_github_login(args),
        CommandKind::EnrollStaticPrincipal(args) => enroll_static_principal(args),
        CommandKind::TrainLocal(args) => train_local(args),
        CommandKind::MonitorRun(args) => monitor_run(args),
        CommandKind::TrainWindowOnce(args) => train_window_once(args),
        CommandKind::RunPeer(args) => run_peer(args),
        CommandKind::RunTrainerDaemon(args) => run_trainer_daemon(args),
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
        CommandKind::AdminProvisionRevisionContract(_) => "admin-provision-revision-contract",
        CommandKind::Login(_) => "login",
        CommandKind::BeginGithubLogin(_) => "begin-github-login",
        CommandKind::CompleteGithubLogin(_) => "complete-github-login",
        CommandKind::EnrollStaticPrincipal(_) => "enroll-static-principal",
        CommandKind::TrainLocal(_) => "train-local",
        CommandKind::MonitorRun(_) => "monitor-run",
        CommandKind::TrainWindowOnce(_) => "train-window-once",
        CommandKind::RunPeer(_) => "run-peer",
        CommandKind::RunTrainerDaemon(_) => "run-trainer-daemon",
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

#[path = "burn_dragon_p2p_native/auth.rs"]
mod auth;
#[path = "burn_dragon_p2p_native/diagnostics.rs"]
mod diagnostics;
#[path = "burn_dragon_p2p_native/io.rs"]
mod io;
#[path = "burn_dragon_p2p_native/local_training.rs"]
mod local_training;
#[path = "burn_dragon_p2p_native/revision_contract.rs"]
mod revision_contract;
#[path = "burn_dragon_p2p_native/runtime.rs"]
mod runtime;
#[path = "burn_dragon_p2p_native/services.rs"]
mod services;
#[path = "burn_dragon_p2p_native/train_window.rs"]
mod train_window;

use auth::*;
use diagnostics::*;
use io::*;
use local_training::*;
use revision_contract::*;
use runtime::*;
use services::*;
use train_window::*;

#[cfg(test)]
#[path = "burn_dragon_p2p_native/tests.rs"]
mod tests;
