use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail, ensure};
use clap::{Args, ValueEnum};

use crate::{browser_site, workflow_tools};

const LOCAL_BROWSER_E2E_ARTIFACT_ROOT: &str = "target/test-artifacts/browser-peer-e2e";
const LOCAL_BROWSER_E2E_SITE_DIR: &str = "target/browser-site";
const LOCAL_BROWSER_E2E_SYNTHETIC_SITE_BASE_URL: &str = "http://127.0.0.1:17777";
const CI_BURN_P2P_REF: &str = "900a8fbc988edd7db503b1fb1ee2eed29dcc99bc";
const DEFAULT_CI_TARGET_DIR: &str = "target/local-browser-e2e-ci-sibling";
const DEFAULT_CARGO: &str =
    "/home/mosure/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin/cargo";
const DEFAULT_RUSTC: &str =
    "/home/mosure/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin/rustc";

#[derive(Clone, Debug, Args)]
pub struct LocalBrowserE2eArgs {
    #[arg(long, value_enum, default_value_t = LocalBrowserE2eLane::All)]
    lane: LocalBrowserE2eLane,
    #[arg(long, default_value = LOCAL_BROWSER_E2E_ARTIFACT_ROOT)]
    artifact_root: PathBuf,
    #[arg(long, default_value = LOCAL_BROWSER_E2E_SITE_DIR)]
    site_dir: PathBuf,
    #[arg(long, default_value_t = false)]
    build_site: bool,
    #[arg(long)]
    edge_base_url: Option<String>,
    #[arg(long)]
    site_base_url: Option<String>,
    #[arg(long)]
    principal_id: Option<String>,
    #[arg(long)]
    callback_token: Option<String>,
    #[arg(long)]
    experiment_id: Option<String>,
    #[arg(long, default_value_t = 240_000)]
    connect_timeout_ms: u64,
    #[arg(long, default_value_t = 300_000)]
    train_timeout_ms: u64,
    #[arg(long, default_value_t = false)]
    headed: bool,
}

#[derive(Clone, Debug, Args)]
pub struct LocalBrowserE2eCiSiblingArgs {
    #[command(flatten)]
    e2e: LocalBrowserE2eArgs,
    #[arg(long, default_value = "../burn_p2p")]
    burn_p2p_repo: PathBuf,
    #[arg(long, default_value = CI_BURN_P2P_REF)]
    burn_p2p_ref: String,
    #[arg(long, default_value = DEFAULT_CI_TARGET_DIR)]
    target_dir: PathBuf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum LocalBrowserE2eLane {
    All,
    Contracts,
    NativeReceipts,
    WasmTraining,
    Site,
    CanaryAll,
    CanaryAutoConnect,
    CanaryWebrtcDirectConnect,
    CanaryWebrtcDirectCheckpoint,
    CanaryWebrtcDirectTraining,
    CanaryProductionProfileTraining,
    CanaryFirefoxWebrtcDirectConnect,
}

#[derive(Clone, Copy, Debug)]
struct BrowserCanaryLaneConfig {
    lane: &'static str,
    browser: &'static str,
    transport_mode: &'static str,
    expect_training: bool,
    expect_checkpoint_sync: bool,
    min_accepted_receipts: u64,
    use_production_training_profile: bool,
}

#[derive(Clone, Copy)]
pub struct LocalBrowserE2eRunner {
    pub deployment_script_checks: fn() -> Result<()>,
    pub cargo_native_test: fn(Option<&str>, bool) -> Result<()>,
    pub wasm_training_smoke: fn() -> Result<()>,
}

pub fn run(
    args: LocalBrowserE2eArgs,
    force_build_site: bool,
    runner: LocalBrowserE2eRunner,
) -> Result<()> {
    if args.lane.runs_contracts() {
        (runner.deployment_script_checks)()?;
    }
    if args.lane.runs_site_build() || force_build_site || args.build_site {
        build_local_browser_site(&args)?;
    }
    if args.lane.runs_native_receipts() {
        (runner.cargo_native_test)(Some("local_browser_training_e2e"), false)?;
    }
    if args.lane.runs_wasm_training() {
        (runner.wasm_training_smoke)()?;
    }
    for lane in args.lane.canary_lanes() {
        run_local_browser_canary_lane(&args, lane)?;
    }
    Ok(())
}

pub fn run_ci_sibling(args: &LocalBrowserE2eCiSiblingArgs) -> Result<()> {
    let dragon_root = workspace_root();
    let burn_p2p_repo = absolute_under_workspace(&dragon_root, &args.burn_p2p_repo);
    ensure!(
        burn_p2p_repo.join(".git").exists(),
        "burn_p2p repository not found at {}; pass --burn-p2p-repo",
        burn_p2p_repo.display()
    );

    let temp = tempfile::Builder::new()
        .prefix("burn-dragon-browser-e2e.")
        .tempdir()
        .context("failed to create temporary local browser e2e worktree root")?;
    let temp_root = temp.path();
    let dragon_worktree_path = temp_root.join("burn_dragon");
    let burn_p2p_worktree_path = temp_root.join("burn_p2p");

    let dragon_worktree_arg = path_arg(&dragon_worktree_path)?;
    run_git(
        &dragon_root,
        &[
            "worktree",
            "add",
            "--detach",
            dragon_worktree_arg.as_str(),
            "HEAD",
        ],
    )?;
    let dragon_worktree = GitWorktreeGuard::new(dragon_root.clone(), dragon_worktree_path.clone());

    apply_current_dragon_diff(&dragon_root, &dragon_worktree_path)?;
    copy_untracked_dragon_files(&dragon_root, &dragon_worktree_path)?;

    let burn_p2p_worktree = if git_commit_exists(&burn_p2p_repo, &args.burn_p2p_ref)? {
        let burn_p2p_worktree_arg = path_arg(&burn_p2p_worktree_path)?;
        run_git(
            &burn_p2p_repo,
            &[
                "worktree",
                "add",
                "--detach",
                burn_p2p_worktree_arg.as_str(),
                args.burn_p2p_ref.as_str(),
            ],
        )?;
        Some(GitWorktreeGuard::new(
            burn_p2p_repo.clone(),
            burn_p2p_worktree_path.clone(),
        ))
    } else {
        let burn_p2p_worktree_arg = path_arg(&burn_p2p_worktree_path)?;
        run_command(
            "git",
            &[
                "clone",
                "https://github.com/aberration-technology/burn_p2p",
                burn_p2p_worktree_arg.as_str(),
            ],
            None,
        )?;
        run_git(
            &burn_p2p_worktree_path,
            &["checkout", args.burn_p2p_ref.as_str()],
        )?;
        None
    };

    let cargo = std::env::var("CARGO").unwrap_or_else(|_| DEFAULT_CARGO.to_owned());
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| DEFAULT_RUSTC.to_owned());
    let target_dir = absolute_under_workspace(&dragon_root, &args.target_dir);
    let e2e_args = args.e2e.to_xtask_args(Some(&dragon_root));
    let mut command = Command::new(&cargo);
    command
        .current_dir(&dragon_worktree_path)
        .arg("run")
        .arg("-p")
        .arg("xtask")
        .arg("--")
        .arg("local-browser-e2e")
        .args(&e2e_args)
        .env("CARGO", &cargo)
        .env("RUSTC", &rustc)
        .env("CARGO_TARGET_DIR", &target_dir)
        .stdin(Stdio::null());
    eprintln!(
        "+ {} run -p xtask -- local-browser-e2e {}",
        cargo,
        e2e_args.join(" ")
    );
    let status = command
        .status()
        .with_context(|| format!("failed to start `{cargo}` for local-browser-e2e-ci-sibling"))?;

    drop(burn_p2p_worktree);
    drop(dragon_worktree);
    if !status.success() {
        bail!("local-browser-e2e-ci-sibling failed");
    }
    Ok(())
}

impl LocalBrowserE2eArgs {
    fn to_xtask_args(&self, path_base: Option<&Path>) -> Vec<String> {
        let mut args = vec![
            "--lane".to_owned(),
            self.lane.as_cli_value().to_owned(),
            "--artifact-root".to_owned(),
            child_path_arg(&self.artifact_root, path_base),
            "--site-dir".to_owned(),
            child_path_arg(&self.site_dir, path_base),
            "--connect-timeout-ms".to_owned(),
            self.connect_timeout_ms.to_string(),
            "--train-timeout-ms".to_owned(),
            self.train_timeout_ms.to_string(),
        ];
        if self.build_site {
            args.push("--build-site".to_owned());
        }
        push_optional_arg(&mut args, "--edge-base-url", self.edge_base_url.as_deref());
        push_optional_arg(&mut args, "--site-base-url", self.site_base_url.as_deref());
        push_optional_arg(&mut args, "--principal-id", self.principal_id.as_deref());
        push_optional_arg(
            &mut args,
            "--callback-token",
            self.callback_token.as_deref(),
        );
        push_optional_arg(&mut args, "--experiment-id", self.experiment_id.as_deref());
        if self.headed {
            args.push("--headed".to_owned());
        }
        args
    }
}

impl LocalBrowserE2eLane {
    fn as_cli_value(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Contracts => "contracts",
            Self::NativeReceipts => "native-receipts",
            Self::WasmTraining => "wasm-training",
            Self::Site => "site",
            Self::CanaryAll => "canary-all",
            Self::CanaryAutoConnect => "canary-auto-connect",
            Self::CanaryWebrtcDirectConnect => "canary-webrtc-direct-connect",
            Self::CanaryWebrtcDirectCheckpoint => "canary-webrtc-direct-checkpoint",
            Self::CanaryWebrtcDirectTraining => "canary-webrtc-direct-training",
            Self::CanaryProductionProfileTraining => "canary-production-profile-training",
            Self::CanaryFirefoxWebrtcDirectConnect => "canary-firefox-webrtc-direct-connect",
        }
    }

    fn runs_contracts(self) -> bool {
        matches!(self, Self::All | Self::Contracts)
    }

    fn runs_native_receipts(self) -> bool {
        matches!(self, Self::All | Self::NativeReceipts)
    }

    fn runs_wasm_training(self) -> bool {
        matches!(self, Self::All | Self::WasmTraining)
    }

    fn runs_site_build(self) -> bool {
        matches!(self, Self::Site)
    }

    fn canary_lanes(self) -> Vec<BrowserCanaryLaneConfig> {
        match self {
            Self::CanaryAll => vec![
                BrowserCanaryLaneConfig::auto_connect(),
                BrowserCanaryLaneConfig::webrtc_direct_connect(),
                BrowserCanaryLaneConfig::webrtc_direct_checkpoint(),
                BrowserCanaryLaneConfig::webrtc_direct_training(),
            ],
            Self::CanaryAutoConnect => vec![BrowserCanaryLaneConfig::auto_connect()],
            Self::CanaryWebrtcDirectConnect => {
                vec![BrowserCanaryLaneConfig::webrtc_direct_connect()]
            }
            Self::CanaryWebrtcDirectCheckpoint => {
                vec![BrowserCanaryLaneConfig::webrtc_direct_checkpoint()]
            }
            Self::CanaryWebrtcDirectTraining => {
                vec![BrowserCanaryLaneConfig::webrtc_direct_training()]
            }
            Self::CanaryProductionProfileTraining => {
                vec![BrowserCanaryLaneConfig::production_profile_training()]
            }
            Self::CanaryFirefoxWebrtcDirectConnect => {
                vec![BrowserCanaryLaneConfig::firefox_webrtc_direct_connect()]
            }
            _ => Vec::new(),
        }
    }
}

impl BrowserCanaryLaneConfig {
    fn auto_connect() -> Self {
        Self {
            lane: "chromium-auto-connect",
            browser: "chromium",
            transport_mode: "auto",
            expect_training: false,
            expect_checkpoint_sync: false,
            min_accepted_receipts: 0,
            use_production_training_profile: false,
        }
    }

    fn webrtc_direct_connect() -> Self {
        Self {
            lane: "chromium-webrtc-direct-connect",
            browser: "chromium",
            transport_mode: "webrtc-direct",
            expect_training: false,
            expect_checkpoint_sync: false,
            min_accepted_receipts: 0,
            use_production_training_profile: false,
        }
    }

    fn webrtc_direct_checkpoint() -> Self {
        Self {
            lane: "chromium-webrtc-direct-checkpoint",
            browser: "chromium",
            transport_mode: "webrtc-direct",
            expect_training: false,
            expect_checkpoint_sync: true,
            min_accepted_receipts: 0,
            use_production_training_profile: true,
        }
    }

    fn webrtc_direct_training() -> Self {
        Self {
            lane: "chromium-webrtc-direct-training",
            browser: "chromium",
            transport_mode: "webrtc-direct",
            expect_training: true,
            expect_checkpoint_sync: false,
            min_accepted_receipts: 2,
            use_production_training_profile: false,
        }
    }

    fn production_profile_training() -> Self {
        Self {
            lane: "chromium-production-profile-training",
            browser: "chromium",
            transport_mode: "webrtc-direct",
            expect_training: true,
            expect_checkpoint_sync: false,
            min_accepted_receipts: 2,
            use_production_training_profile: true,
        }
    }

    fn firefox_webrtc_direct_connect() -> Self {
        Self {
            lane: "firefox-webrtc-direct-connect",
            browser: "firefox",
            transport_mode: "webrtc-direct",
            expect_training: false,
            expect_checkpoint_sync: false,
            min_accepted_receipts: 0,
            use_production_training_profile: false,
        }
    }
}

fn build_local_browser_site(args: &LocalBrowserE2eArgs) -> Result<()> {
    browser_site::build_browser_site(&browser_site::BuildBrowserSiteArgs {
        out_dir: args.site_dir.clone(),
        edge_url: args.edge_base_url.clone(),
        seed_node_urls: Vec::new(),
        selected_experiment_id: args.experiment_id.clone(),
        selected_revision_id: None,
        require_edge_auth: true,
    })
}

fn run_local_browser_canary_lane(
    args: &LocalBrowserE2eArgs,
    lane: BrowserCanaryLaneConfig,
) -> Result<()> {
    ensure_browser_site_artifact(&args.site_dir)?;
    ensure_canary_required_value(
        "BURN_DRAGON_BROWSER_CANARY_EDGE_BASE_URL",
        args.edge_base_url.as_deref(),
    )?;
    ensure_canary_required_value(
        "BURN_DRAGON_BROWSER_CANARY_PRINCIPAL_ID",
        args.principal_id.as_deref(),
    )?;
    ensure_canary_required_value(
        "BURN_DRAGON_BROWSER_CANARY_CALLBACK_TOKEN",
        args.callback_token.as_deref(),
    )?;
    workflow_tools::install_playwright_chromium()?;

    let artifact_dir = local_browser_canary_artifact_dir(&args.artifact_root, lane.lane);
    fs::create_dir_all(&artifact_dir).with_context(|| {
        format!(
            "failed to create local browser canary artifact directory {}",
            artifact_dir.display()
        )
    })?;
    let output_json = artifact_dir.join("canary-summary.json");
    let mut overrides = vec![
        ("BURN_DRAGON_BROWSER_CANARY_LANE", lane.lane.to_owned()),
        (
            "BURN_DRAGON_BROWSER_CANARY_BROWSER",
            lane.browser.to_owned(),
        ),
        (
            "BURN_DRAGON_BROWSER_CANARY_TRANSPORT_MODE",
            lane.transport_mode.to_owned(),
        ),
        (
            "BURN_DRAGON_BROWSER_CANARY_EXPECT_TRAINING",
            bool_env(lane.expect_training),
        ),
        (
            "BURN_DRAGON_BROWSER_CANARY_EXPECT_CHECKPOINT_SYNC",
            bool_env(lane.expect_checkpoint_sync),
        ),
        (
            "BURN_DRAGON_BROWSER_CANARY_USE_PRODUCTION_TRAINING_PROFILE",
            bool_env(lane.use_production_training_profile),
        ),
        (
            "BURN_DRAGON_BROWSER_CANARY_MIN_ACCEPTED_RECEIPTS",
            lane.min_accepted_receipts.to_string(),
        ),
        (
            "BURN_DRAGON_BROWSER_CANARY_CONNECT_TIMEOUT_MS",
            args.connect_timeout_ms.to_string(),
        ),
        (
            "BURN_DRAGON_BROWSER_CANARY_TRAIN_TIMEOUT_MS",
            args.train_timeout_ms.to_string(),
        ),
        (
            "BURN_DRAGON_BROWSER_CANARY_SITE_BASE_URL",
            args.site_base_url.clone().unwrap_or_else(|| {
                std::env::var("BURN_DRAGON_BROWSER_CANARY_SITE_BASE_URL")
                    .unwrap_or_else(|_| LOCAL_BROWSER_E2E_SYNTHETIC_SITE_BASE_URL.to_owned())
            }),
        ),
        (
            "BURN_DRAGON_BROWSER_CANARY_SITE_OVERRIDE_DIR",
            args.site_dir.display().to_string(),
        ),
        (
            "BURN_DRAGON_BROWSER_CANARY_ARTIFACT_DIR",
            artifact_dir.display().to_string(),
        ),
        (
            "BURN_DRAGON_BROWSER_CANARY_OUTPUT_JSON",
            output_json.display().to_string(),
        ),
        ("PLAYWRIGHT_BROWSERS", lane.browser.to_owned()),
    ];
    push_optional_canary_override(
        &mut overrides,
        "BURN_DRAGON_BROWSER_CANARY_EDGE_BASE_URL",
        args.edge_base_url.as_deref(),
    );
    push_optional_canary_override(
        &mut overrides,
        "BURN_DRAGON_BROWSER_CANARY_PRINCIPAL_ID",
        args.principal_id.as_deref(),
    );
    push_optional_canary_override(
        &mut overrides,
        "BURN_DRAGON_BROWSER_CANARY_CALLBACK_TOKEN",
        args.callback_token.as_deref(),
    );
    push_optional_canary_override(
        &mut overrides,
        "BURN_DRAGON_BROWSER_CANARY_EXPERIMENT_ID",
        args.experiment_id.as_deref(),
    );
    if args.headed {
        overrides.push(("BURN_DRAGON_BROWSER_CANARY_HEADED", "1".to_owned()));
    }
    workflow_tools::run_live_browser_canary_with_overrides(&overrides)
}

fn ensure_browser_site_artifact(site_dir: &Path) -> Result<()> {
    for relative_path in [
        "index.html",
        "browser-app-config.json",
        "burn_dragon_p2p_browser.js",
        "burn_dragon_p2p_browser_bg.wasm",
    ] {
        let path = site_dir.join(relative_path);
        ensure!(
            path.is_file(),
            "local browser canary requires a built browser site; missing {}. Run `cargo run -p xtask -- local-browser-e2e --lane site` or pass `--build-site`.",
            path.display()
        );
    }
    Ok(())
}

fn apply_current_dragon_diff(dragon_root: &Path, dragon_worktree_path: &Path) -> Result<()> {
    let output = Command::new("git")
        .current_dir(dragon_root)
        .arg("diff")
        .arg("--binary")
        .arg("HEAD")
        .output()
        .context("failed to collect current Dragon diff")?;
    if !output.status.success() {
        bail!("git diff --binary HEAD failed");
    }
    if output.stdout.is_empty() {
        return Ok(());
    }
    let patch_path = dragon_worktree_path
        .parent()
        .context("Dragon worktree path had no parent")?
        .join("burn_dragon.patch");
    fs::write(&patch_path, output.stdout)
        .with_context(|| format!("failed to write {}", patch_path.display()))?;
    let patch_arg = path_arg(&patch_path)?;
    run_git(dragon_worktree_path, &["apply", patch_arg.as_str()])
}

fn copy_untracked_dragon_files(dragon_root: &Path, dragon_worktree_path: &Path) -> Result<()> {
    let output = Command::new("git")
        .current_dir(dragon_root)
        .arg("ls-files")
        .arg("--others")
        .arg("--exclude-standard")
        .arg("-z")
        .output()
        .context("failed to list untracked Dragon files")?;
    if !output.status.success() {
        bail!("git ls-files --others failed");
    }
    for raw in output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|raw| !raw.is_empty())
    {
        let relative = std::str::from_utf8(raw).context("untracked path was not utf-8")?;
        let source = dragon_root.join(relative);
        if !source.is_file() {
            continue;
        }
        let destination = dragon_worktree_path.join(relative);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        fs::copy(&source, &destination).with_context(|| {
            format!(
                "failed to copy untracked file {} to {}",
                source.display(),
                destination.display()
            )
        })?;
    }
    Ok(())
}

fn ensure_canary_required_value(env_name: &str, explicit: Option<&str>) -> Result<()> {
    if explicit.is_some_and(|value| !value.trim().is_empty())
        || std::env::var(env_name)
            .map(|value| !value.trim().is_empty())
            .unwrap_or(false)
    {
        return Ok(());
    }
    bail!("local browser canary lane requires `{env_name}` or the matching command-line option");
}

fn push_optional_canary_override(
    overrides: &mut Vec<(&'static str, String)>,
    key: &'static str,
    value: Option<&str>,
) {
    if let Some(value) = value.filter(|value| !value.trim().is_empty()) {
        overrides.push((key, value.to_owned()));
    }
}

fn push_optional_arg(args: &mut Vec<String>, key: &str, value: Option<&str>) {
    if let Some(value) = value.filter(|value| !value.trim().is_empty()) {
        args.push(key.to_owned());
        args.push(value.to_owned());
    }
}

fn local_browser_canary_artifact_dir(root: &Path, lane: &str) -> PathBuf {
    let timestamp = chrono::Utc::now().format("%Y%m%dT%H%M%SZ");
    root.join(format!("{lane}-{timestamp}"))
}

fn bool_env(value: bool) -> String {
    (if value { "1" } else { "0" }).to_owned()
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask should live under workspace root")
        .to_path_buf()
}

fn absolute_under_workspace(workspace_root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        workspace_root.join(path)
    }
}

fn child_path_arg(path: &Path, path_base: Option<&Path>) -> String {
    path_base
        .map(|base| absolute_under_workspace(base, path))
        .unwrap_or_else(|| path.to_path_buf())
        .display()
        .to_string()
}

fn git_commit_exists(repo: &Path, commit: &str) -> Result<bool> {
    let status = Command::new("git")
        .current_dir(repo)
        .arg("cat-file")
        .arg("-e")
        .arg(format!("{commit}^{{commit}}"))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .with_context(|| format!("failed to query git commit {commit} in {}", repo.display()))?;
    Ok(status.success())
}

fn run_git(cwd: &Path, args: &[&str]) -> Result<()> {
    run_command("git", args, Some(cwd))
}

fn run_command(program: &str, args: &[&str], cwd: Option<&Path>) -> Result<()> {
    let mut command = Command::new(program);
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }
    command.args(args).stdin(Stdio::null());
    eprintln!("+ {} {}", program, args.join(" "));
    let status = command
        .status()
        .with_context(|| format!("failed to start `{program}`"))?;
    if !status.success() {
        bail!("command failed: {} {}", program, args.join(" "));
    }
    Ok(())
}

fn path_arg(path: &Path) -> Result<String> {
    path.to_str()
        .map(ToOwned::to_owned)
        .with_context(|| format!("path was not valid utf-8: {}", path.display()))
}

struct GitWorktreeGuard {
    repo: PathBuf,
    path: PathBuf,
}

impl GitWorktreeGuard {
    fn new(repo: PathBuf, path: PathBuf) -> Self {
        Self { repo, path }
    }
}

impl Drop for GitWorktreeGuard {
    fn drop(&mut self) {
        let _ = Command::new("git")
            .current_dir(&self.repo)
            .arg("worktree")
            .arg("remove")
            .arg("--force")
            .arg(&self.path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}
