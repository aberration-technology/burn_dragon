use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use burn::tensor::backend::AutodiffBackend;
use burn_dragon_language::load_training_config;
use burn_dragon_p2p::admin::{
    fetch_directory_entries, rollout_directory_entries, upsert_directory_entry,
};
use burn_dragon_p2p::auth::{
    DragonPendingGitHubLogin, begin_native_github_login, complete_native_github_login,
    enroll_native_static_principal, fetch_edge_snapshot,
};
use burn_dragon_p2p::capability_state::{clear_native_downgrade, persist_native_downgrade};
use burn_dragon_p2p::config::{
    DragonCapabilityPolicy, DragonExperimentKind, DragonManifestBundle, DragonNativeAuthBundle,
    DragonNativePeerConfig,
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
    ExperimentId, ExperimentScope, PrincipalId, RuntimeStatus,
};
use burn_p2p_admin::AdminResult;
use clap::{Parser, Subcommand, ValueEnum};
use serde::{Serialize, de::DeserializeOwned};

const MIB: u64 = 1024 * 1024;
const DEFAULT_SESSION_TTL_SECS: i64 = 1800;
const DEFAULT_STATUS_INTERVAL_SECS: u64 = 30;
const DEFAULT_VALIDATION_INTERVAL_MILLIS: u64 = 250;
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
    BuildProfile(BuildProfileArgs),
    AdminExportDirectory(AdminExportDirectoryArgs),
    AdminRolloutProfile(AdminRolloutProfileArgs),
    #[command(alias = "begin-login")]
    BeginGithubLogin(BeginGithubLoginArgs),
    #[command(alias = "complete-login")]
    CompleteGithubLogin(CompleteGithubLoginArgs),
    EnrollStaticPrincipal(EnrollStaticPrincipalArgs),
    RunPeer(RunPeerArgs),
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

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        CommandKind::ResolveConfig(args) => resolve_config(args),
        CommandKind::AssessCapability(args) => assess_capability(args),
        CommandKind::BuildProfile(args) => build_profile(args),
        CommandKind::AdminExportDirectory(args) => admin_export_directory(args),
        CommandKind::AdminRolloutProfile(args) => admin_rollout_profile(args),
        CommandKind::BeginGithubLogin(args) => begin_github_login(args),
        CommandKind::CompleteGithubLogin(args) => complete_github_login(args),
        CommandKind::EnrollStaticPrincipal(args) => enroll_static_principal(args),
        CommandKind::RunPeer(args) => run_peer(args),
        CommandKind::RunValidatorDaemon(args) => run_validator_daemon(args),
        CommandKind::MarkRuntimeFailure(args) => mark_runtime_failure(args),
        CommandKind::ClearDowngrade(args) => clear_downgrade(args),
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
    let auth_bundle: DragonNativeAuthBundle =
        load_typed(&args.auth_bundle, args.auth_bundle_format)?;
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
    let mut directory_entries = runtime.block_on(fetch_directory_entries(&edge_base_url))?;
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

fn begin_github_login(args: BeginGithubLoginArgs) -> Result<()> {
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
    let manifests = prepared_manifests(&config, args.experiment_kind.into_config(), args.backend)?;
    let requested_scopes = requested_scopes(
        manifests
            .experiment_directory
            .first()
            .ok_or_else(|| anyhow!("manifest bundle is missing an experiment directory entry"))?,
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to build async runtime for GitHub login")?;
    let pending = runtime.block_on(begin_native_github_login(
        &edge_base_url,
        &manifests.release_manifest,
        requested_scopes,
        args.session_ttl_secs,
        args.principal_hint,
        args.device_flow,
    ))?;
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
        None,
    ))?;
    write_output(
        args.auth_bundle_out.as_deref(),
        args.output_format,
        &session.auth,
    )
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
            .unwrap_or_else(|| "native-static-enroll".into()),
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
    let auth_bundle = match args.auth_bundle {
        Some(path) => Some(load_typed::<DragonNativeAuthBundle>(
            &path,
            args.auth_bundle_format,
        )?),
        None => None,
    };

    match (args.experiment_kind.into_config(), args.backend) {
        (DragonExperimentKind::NcaPrepretraining, BackendArg::Cpu) => run_prepared_peer(
            prepare_nca_native_cpu(&config, auth_bundle.as_ref())?,
            &config,
            args.backend,
            args.status_interval_secs,
        ),
        (DragonExperimentKind::ClimbMixPretraining, BackendArg::Cpu) => run_prepared_peer(
            prepare_climbmix_native_cpu(&config, auth_bundle.as_ref())?,
            &config,
            args.backend,
            args.status_interval_secs,
        ),
        #[cfg(feature = "wgpu")]
        (DragonExperimentKind::NcaPrepretraining, BackendArg::Wgpu) => run_prepared_peer(
            prepare_nca_native_wgpu(&config, auth_bundle.as_ref())?,
            &config,
            args.backend,
            args.status_interval_secs,
        ),
        #[cfg(feature = "wgpu")]
        (DragonExperimentKind::ClimbMixPretraining, BackendArg::Wgpu) => run_prepared_peer(
            prepare_climbmix_native_wgpu(&config, auth_bundle.as_ref())?,
            &config,
            args.backend,
            args.status_interval_secs,
        ),
        #[cfg(feature = "cuda")]
        (DragonExperimentKind::NcaPrepretraining, BackendArg::Cuda) => run_prepared_peer(
            prepare_nca_native_cuda(&config, auth_bundle.as_ref())?,
            &config,
            args.backend,
            args.status_interval_secs,
        ),
        #[cfg(feature = "cuda")]
        (DragonExperimentKind::ClimbMixPretraining, BackendArg::Cuda) => run_prepared_peer(
            prepare_climbmix_native_cuda(&config, auth_bundle.as_ref())?,
            &config,
            args.backend,
            args.status_interval_secs,
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
    let auth_bundle = match args.auth_bundle {
        Some(path) => Some(load_typed::<DragonNativeAuthBundle>(
            &path,
            args.auth_bundle_format,
        )?),
        None => None,
    };

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
        &config.storage_root,
        args.experiment_kind.into_config(),
        args.backend.as_label(),
        &assessment.model_config,
        assessment.batch_size,
        assessment.block_size,
        &assessment.footprint,
        assessment.target_decision.trainer_memory_budget_bytes,
        "trainer",
        &args.reason,
        &args.source,
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
    clear_native_downgrade(
        &config.storage_root,
        args.experiment_kind.into_config(),
        args.backend.as_label(),
        &assessment.model_config,
        assessment.batch_size,
        assessment.block_size,
    )?;
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

    let running = spawn_prepared_native_peer(prepared)?;
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

    loop {
        let snapshot = running.snapshot();
        if status_interval_secs > 0 && last_status.elapsed() >= status_interval {
            eprintln!(
                "peer-status status={:?} node_state={:?} connected_peers={} last_error={}",
                snapshot.status,
                snapshot.node_state,
                snapshot.connected_peers,
                snapshot.last_error.as_deref().unwrap_or("-"),
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
                snapshot.last_error.as_deref().unwrap_or("-"),
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
