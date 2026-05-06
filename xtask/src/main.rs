mod agent_task;
mod bootstrap_runtime;
mod bootstrap_settings;
mod browser_site;
mod deploy_settings;
mod native_canary;
mod workflow_contracts;
mod workflow_tools;

use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{bail, ensure, Context, Result};
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(author, version, about = "burn_dragon p2p task runner")]
struct Cli {
    #[command(subcommand)]
    command: CommandKind,
}

#[derive(Debug, Subcommand)]
enum CommandKind {
    ArtifactCheck,
    BuildNative,
    BuildNativeWgpu,
    BuildNativeCuda,
    BuildNativeRocm,
    BuildBrowserCpu,
    BuildBrowser,
    BuildBrowserSite(browser_site::BuildBrowserSiteArgs),
    ResolvePagesDeploySettings(deploy_settings::ResolvePagesDeploySettingsArgs),
    ResolvePagesDeploySettingsOutputs(deploy_settings::ResolvePagesDeploySettingsArgs),
    InstallPlaywrightChromium,
    RunLiveBrowserCanary,
    RunPagesPredeployCanary,
    SummarizeLiveBrowserCanary(workflow_tools::SummarizeLiveBrowserCanaryArgs),
    SummarizeLiveNativeTrainingCanary(workflow_tools::SummarizeLiveNativeTrainingCanaryArgs),
    SummarizeDeploymentDiagnostics(workflow_tools::SummarizeDeploymentDiagnosticsArgs),
    SummarizeInspection(workflow_tools::SummarizeInspectionArgs),
    ExtractDeploymentDiagnostics(workflow_tools::ExtractDeploymentDiagnosticsArgs),
    WriteInspectionSummary(workflow_tools::WriteInspectionSummaryArgs),
    WriteBootstrapInspectParams {
        output_path: PathBuf,
    },
    RenderHeadMirrorSeedRepairCommands,
    CheckDeploymentGuardrails,
    DeploymentGuardrailReport,
    ResolveBootstrapStackSettings(workflow_tools::BootstrapStackSettingsArgs),
    SyncBootstrapRuntimeConfig,
    RunLiveNativeTrainingCanary,
    PublishCrates(workflow_tools::PublishCratesArgs),
    AgentTask {
        #[command(subcommand)]
        command: agent_task::AgentTaskCommand,
    },
    DispatchPagesDeployAndWait,
    DispatchNativeTrainingCanaryAndWait,
    BuildMatrix,
    NativeSmoke,
    NativeIntegration,
    NativeScale,
    NativeLarge,
    DowngradeSmoke,
    MixedFleet,
    EdgeDrill,
    LocalProdE2e,
    LocalBrowserE2e,
    WasmTrainingSmoke,
    WasmSmoke,
    CudaCheck,
    Smoke,
    DeployCheck,
    DeploymentScriptChecks,
    All,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        CommandKind::BuildNative => build_native(),
        CommandKind::BuildNativeWgpu => build_native_wgpu(),
        CommandKind::BuildNativeCuda => build_native_cuda(),
        CommandKind::BuildNativeRocm => build_native_rocm(),
        CommandKind::BuildBrowserCpu => build_browser_cpu(),
        CommandKind::BuildBrowser => build_browser(),
        CommandKind::BuildBrowserSite(args) => browser_site::build_browser_site(&args),
        CommandKind::ResolvePagesDeploySettings(args) => {
            deploy_settings::resolve_pages_deploy_settings(&args)
        }
        CommandKind::ResolvePagesDeploySettingsOutputs(args) => {
            workflow_tools::resolve_pages_deploy_settings_outputs(&args)
        }
        CommandKind::InstallPlaywrightChromium => workflow_tools::install_playwright_chromium(),
        CommandKind::RunLiveBrowserCanary => workflow_tools::run_live_browser_canary(),
        CommandKind::RunPagesPredeployCanary => workflow_tools::run_pages_predeploy_canary(),
        CommandKind::SummarizeLiveBrowserCanary(args) => {
            workflow_tools::summarize_live_browser_canary(&args)
        }
        CommandKind::SummarizeLiveNativeTrainingCanary(args) => {
            workflow_tools::summarize_live_native_training_canary(&args)
        }
        CommandKind::SummarizeDeploymentDiagnostics(args) => {
            workflow_tools::summarize_deployment_diagnostics(&args)
        }
        CommandKind::SummarizeInspection(args) => workflow_tools::summarize_inspection(&args),
        CommandKind::ExtractDeploymentDiagnostics(args) => {
            workflow_tools::extract_deployment_diagnostics(&args)
        }
        CommandKind::WriteInspectionSummary(args) => {
            workflow_tools::write_inspection_summary(&args)
        }
        CommandKind::WriteBootstrapInspectParams { output_path } => {
            workflow_tools::write_bootstrap_inspect_params(&output_path)
        }
        CommandKind::RenderHeadMirrorSeedRepairCommands => {
            workflow_tools::render_head_mirror_seed_repair_commands()
        }
        CommandKind::CheckDeploymentGuardrails => workflow_tools::check_deployment_guardrails(),
        CommandKind::DeploymentGuardrailReport => workflow_tools::deployment_guardrail_report(),
        CommandKind::ResolveBootstrapStackSettings(args) => {
            workflow_tools::resolve_bootstrap_stack_settings(&args)
        }
        CommandKind::SyncBootstrapRuntimeConfig => workflow_tools::sync_bootstrap_runtime_config(),
        CommandKind::RunLiveNativeTrainingCanary => {
            workflow_tools::run_live_native_training_canary()
        }
        CommandKind::PublishCrates(args) => workflow_tools::publish_crates(&args),
        CommandKind::AgentTask { command } => agent_task::run(command),
        CommandKind::DispatchPagesDeployAndWait => agent_task::dispatch_pages_deploy_and_wait(),
        CommandKind::DispatchNativeTrainingCanaryAndWait => {
            agent_task::dispatch_native_training_canary_and_wait()
        }
        CommandKind::BuildMatrix => build_matrix(),
        CommandKind::ArtifactCheck => artifact_check(),
        CommandKind::NativeSmoke => native_smoke(),
        CommandKind::NativeIntegration => native_integration(),
        CommandKind::NativeScale => native_scale(),
        CommandKind::NativeLarge => native_large(),
        CommandKind::DowngradeSmoke => downgrade_smoke(),
        CommandKind::MixedFleet => mixed_fleet(),
        CommandKind::EdgeDrill => edge_drill(),
        CommandKind::LocalProdE2e => local_prod_e2e(),
        CommandKind::LocalBrowserE2e => local_browser_e2e(),
        CommandKind::WasmTrainingSmoke => wasm_training_smoke(),
        CommandKind::WasmSmoke => wasm_smoke(),
        CommandKind::CudaCheck => cuda_check(),
        CommandKind::Smoke => smoke(),
        CommandKind::DeployCheck => deploy_check(),
        CommandKind::DeploymentScriptChecks => deployment_script_checks(),
        CommandKind::All => all(),
    }
}

const P2P_PACKAGE: &str = "burn_dragon_p2p";
const NATIVE_TEST: &str = "native_training";

#[derive(Clone, Copy)]
enum NativeBuildTarget {
    Cpu,
    Wgpu,
    Cuda,
    Rocm,
}

impl NativeBuildTarget {
    fn features(self) -> &'static str {
        match self {
            Self::Cpu => "native",
            Self::Wgpu => "native,wgpu",
            Self::Cuda => "native,cuda",
            Self::Rocm => "native,rocm",
        }
    }
}

#[derive(Clone, Copy)]
enum BrowserBuildTarget {
    Cpu,
    Wgpu,
}

impl BrowserBuildTarget {
    fn features(self) -> &'static str {
        match self {
            Self::Cpu => "wasm-ui,wasm-peer",
            Self::Wgpu => "wasm-ui,wasm-peer,wgpu",
        }
    }
}

fn smoke() -> Result<()> {
    native_smoke()?;
    wasm_smoke()?;
    if cuda_toolchain_available() {
        build_native_cuda()?;
    } else {
        eprintln!("+ skipping CUDA smoke: `nvcc` was not found");
    }
    Ok(())
}

fn artifact_check() -> Result<()> {
    run(
        "cargo",
        &[
            "test",
            "-p",
            P2P_PACKAGE,
            "--features",
            NativeBuildTarget::Wgpu.features(),
            "--test",
            NATIVE_TEST,
            "nca_native_runtime_persists_and_publishes_artifacts",
            "--",
            "--exact",
            "--nocapture",
        ],
    )
}

fn all() -> Result<()> {
    build_matrix()?;
    deploy_check()
}

fn deploy_check() -> Result<()> {
    deployment_script_checks()?;
    browser_site::build_browser_site_default()?;
    smoke()?;
    artifact_check()?;
    downgrade_smoke()?;
    native_scale()?;
    mixed_fleet()?;
    native_large()?;
    edge_drill()?;
    Ok(())
}

fn build_native() -> Result<()> {
    build_native_target(NativeBuildTarget::Cpu)
}

fn build_native_wgpu() -> Result<()> {
    build_native_target(NativeBuildTarget::Wgpu)
}

fn build_native_cuda() -> Result<()> {
    build_native_target(NativeBuildTarget::Cuda)
}

fn build_native_rocm() -> Result<()> {
    build_native_target(NativeBuildTarget::Rocm)
}

fn build_browser_cpu() -> Result<()> {
    build_browser_target(BrowserBuildTarget::Cpu)
}

fn build_browser() -> Result<()> {
    build_browser_target(BrowserBuildTarget::Wgpu)
}

fn build_matrix() -> Result<()> {
    build_native()?;
    build_native_wgpu()?;
    build_native_cuda()?;
    build_native_rocm()?;
    build_browser_cpu()?;
    build_browser()?;
    browser_site::build_browser_site_default()?;
    Ok(())
}

fn native_smoke() -> Result<()> {
    cargo_native_test(Some("ci_native_smoke_suite"), false)
}

fn native_integration() -> Result<()> {
    cargo_native_test(None, false)
}

fn native_scale() -> Result<()> {
    cargo_native_test(Some("medium_model"), true)
}

fn native_large() -> Result<()> {
    cargo_native_test(Some("large_model"), true)
}

fn mixed_fleet() -> Result<()> {
    cargo_native_test(Some("mixed_fleet"), false)?;
    cargo_native_test(Some("mixed_fleet"), true)
}

fn downgrade_smoke() -> Result<()> {
    cargo_native_test(Some("downgrades_to_validator"), false)?;
    wasm_smoke()
}

fn edge_drill() -> Result<()> {
    cargo_native_test(Some("edge_drill"), true)
}

fn local_browser_e2e() -> Result<()> {
    local_browser_contract_e2e(false)
}

fn local_prod_e2e() -> Result<()> {
    local_browser_contract_e2e(true)
}

fn local_browser_contract_e2e(build_site: bool) -> Result<()> {
    deployment_script_checks()?;
    if build_site {
        browser_site::build_browser_site_default()?;
    }
    cargo_native_test(Some("local_browser_training_e2e"), false)?;
    wasm_training_smoke()
}

fn wasm_training_smoke() -> Result<()> {
    wasm_browser_test(Some("browser_training_smoke_generated_nca"))
}

fn wasm_smoke() -> Result<()> {
    wasm_browser_test(Some("browser_training_smoke"))
}

fn wasm_browser_test(filter: Option<&str>) -> Result<()> {
    let chrome = resolve_chrome_path()
        .context("could not find Google Chrome; install it or set BURN_DRAGON_PLAYWRIGHT_CHROME")?;
    let chromedriver = ensure_chromedriver(&chrome)?;
    let wasm_bindgen_test_runner = ensure_wasm_bindgen_test_runner()?;
    let webdriver_json = write_webdriver_config(&chrome)?;
    let mut envs = vec![
        (
            OsString::from("CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER"),
            wasm_bindgen_test_runner.into_os_string(),
        ),
        (
            OsString::from("CHROMEDRIVER"),
            chromedriver.into_os_string(),
        ),
        (
            OsString::from("WASM_BINDGEN_TEST_ONLY_WEB"),
            OsString::from("1"),
        ),
        (
            OsString::from("WASM_BINDGEN_TEST_DRIVER_TIMEOUT"),
            OsString::from("60"),
        ),
        (
            OsString::from("WASM_BINDGEN_TEST_TIMEOUT"),
            OsString::from("180"),
        ),
        (
            OsString::from("WASM_BINDGEN_TEST_WEBDRIVER_JSON"),
            webdriver_json.into_os_string(),
        ),
    ];
    if let Some(existing) = std::env::var_os("PATH") {
        envs.push((OsString::from("PATH"), existing));
    }
    let mut args = vec![
        "test",
        "-p",
        P2P_PACKAGE,
        "--target",
        "wasm32-unknown-unknown",
        "--no-default-features",
        "--features",
        BrowserBuildTarget::Wgpu.features(),
        "--lib",
    ];
    if let Some(filter) = filter {
        args.push(filter);
    }
    args.push("--");
    args.push("--nocapture");
    run_with_env("cargo", &args, &envs)
}

fn cuda_check() -> Result<()> {
    build_native_cuda()
}

fn cargo_native_test(filter: Option<&str>, ignored: bool) -> Result<()> {
    let mut args = vec![
        "test",
        "-p",
        P2P_PACKAGE,
        "--features",
        NativeBuildTarget::Wgpu.features(),
        "--test",
        NATIVE_TEST,
    ];
    if let Some(filter) = filter {
        args.push(filter);
    }
    args.push("--");
    if ignored {
        args.push("--ignored");
    }
    args.push("--test-threads=1");
    args.push("--nocapture");
    run("cargo", &args)
}

fn build_native_target(target: NativeBuildTarget) -> Result<()> {
    cargo_p2p_check(&["--no-default-features", "--features", target.features()])?;
    cargo_p2p_native_bin_check(target.features())
}

fn build_browser_target(target: BrowserBuildTarget) -> Result<()> {
    cargo_p2p_wasm_check(target.features())
}

fn cargo_p2p_check(extra_args: &[&str]) -> Result<()> {
    let mut args = vec!["check", "-p", P2P_PACKAGE];
    args.extend_from_slice(extra_args);
    run("cargo", &args)
}

fn cargo_p2p_native_bin_check(features: &str) -> Result<()> {
    run(
        "cargo",
        &[
            "check",
            "-p",
            P2P_PACKAGE,
            "--no-default-features",
            "--features",
            features,
            "--bin",
            "burn_dragon_p2p_native",
        ],
    )
}

fn cargo_p2p_wasm_check(features: &str) -> Result<()> {
    run(
        "cargo",
        &[
            "check",
            "-p",
            P2P_PACKAGE,
            "--target",
            "wasm32-unknown-unknown",
            "--no-default-features",
            "--features",
            features,
        ],
    )
}

fn run(program: &str, args: &[&str]) -> Result<()> {
    let resolved_program;
    let program = if program == "cargo" {
        resolved_program = cargo_bin();
        resolved_program.as_str()
    } else {
        program
    };
    run_with_env(program, args, &[])
}

fn deployment_script_checks() -> Result<()> {
    workflow_contracts::run()
}

fn run_with_env(program: &str, args: &[&str], envs: &[(OsString, OsString)]) -> Result<()> {
    let resolved_program;
    let program = if program == "cargo" {
        resolved_program = cargo_bin();
        resolved_program.as_str()
    } else {
        program
    };
    eprintln!("+ {} {}", program, args.join(" "));
    let mut command = Command::new(program);
    command
        .current_dir(workspace_root())
        .args(args)
        .stdin(Stdio::null());
    for (key, value) in envs {
        command.env(key, value);
    }
    let status = command
        .status()
        .with_context(|| format!("failed to start `{program}`"))?;
    if !status.success() {
        bail!("command failed: {} {}", program, args.join(" "));
    }
    Ok(())
}

fn ensure_wasm_bindgen_test_runner() -> Result<PathBuf> {
    if let Some(explicit) = std::env::var_os("BURN_DRAGON_WASM_BINDGEN_TEST_RUNNER")
        .map(PathBuf::from)
        .filter(|path| path.is_file())
    {
        return Ok(explicit);
    }
    if let Some(found) = find_in_path(binary_name("wasm-bindgen-test-runner")) {
        return Ok(found);
    }

    let version = lockfile_package_version("wasm-bindgen")?;
    let install_root = workspace_root()
        .join("target")
        .join("xtask")
        .join("wasm-bindgen-cli")
        .join(&version);
    let runner = install_root
        .join("bin")
        .join(binary_name("wasm-bindgen-test-runner"));
    if runner.is_file() {
        return Ok(runner);
    }

    fs::create_dir_all(&install_root).with_context(|| {
        format!(
            "failed to create wasm-bindgen-cli cache {}",
            install_root.display()
        )
    })?;

    let cargo = cargo_bin();
    run(
        &cargo,
        &[
            "install",
            "--locked",
            "--root",
            install_root
                .to_str()
                .context("wasm-bindgen install root was not valid utf-8")?,
            "wasm-bindgen-cli",
            "--version",
            &version,
        ],
    )
    .with_context(|| format!("failed to install wasm-bindgen-cli {version}"))?;

    ensure!(
        runner.is_file(),
        "missing wasm-bindgen test runner at {} after install",
        runner.display()
    );
    Ok(runner)
}

fn lockfile_package_version(package_name: &str) -> Result<String> {
    let lockfile = workspace_root().join("Cargo.lock");
    let text = fs::read_to_string(&lockfile)
        .with_context(|| format!("failed to read {}", lockfile.display()))?;
    for block in text.split("[[package]]") {
        let mut saw_name = false;
        let mut version = None;
        for line in block.lines().map(str::trim) {
            if line == format!("name = \"{package_name}\"") {
                saw_name = true;
            } else if let Some(raw_version) = line.strip_prefix("version = ") {
                version = Some(raw_version.trim_matches('"').to_owned());
            }
        }
        if saw_name {
            if let Some(version) = version {
                return Ok(version);
            }
            bail!("lockfile entry for {package_name} was missing a version");
        }
    }
    bail!("package {package_name} was not found in Cargo.lock")
}

fn binary_name(base: &str) -> String {
    if cfg!(windows) {
        format!("{base}.exe")
    } else {
        base.to_owned()
    }
}

fn cargo_bin() -> String {
    std::env::var("CARGO").unwrap_or_else(|_| "cargo".into())
}

fn ensure_chromedriver(chrome: &Path) -> Result<PathBuf> {
    if let Some(explicit) = std::env::var_os("CHROMEDRIVER")
        .map(PathBuf::from)
        .filter(|path| path.exists())
    {
        return Ok(explicit);
    }
    if let Some(found) = find_in_path("chromedriver") {
        return Ok(found);
    }

    let version = chrome_version(chrome)?;
    let cache_root = workspace_root()
        .join("target")
        .join("xtask")
        .join("chromedriver")
        .join(&version);
    let binary = cache_root.join("chromedriver-linux64").join("chromedriver");
    if binary.is_file() {
        return Ok(binary);
    }

    fs::create_dir_all(&cache_root).with_context(|| {
        format!(
            "failed to create chromedriver cache {}",
            cache_root.display()
        )
    })?;
    let zip_path = cache_root.join("chromedriver-linux64.zip");
    let url = format!(
        "https://storage.googleapis.com/chrome-for-testing-public/{version}/linux64/chromedriver-linux64.zip"
    );

    run(
        "curl",
        &[
            "-fsSL",
            "--connect-timeout",
            "20",
            "--max-time",
            "180",
            "--retry",
            "3",
            "--retry-all-errors",
            &url,
            "-o",
            zip_path.to_str().context("zip path was not valid utf-8")?,
        ],
    )
    .with_context(|| format!("failed to download chromedriver for Chrome {version}"))?;
    run(
        "unzip",
        &[
            "-oq",
            zip_path.to_str().context("zip path was not valid utf-8")?,
            "-d",
            cache_root
                .to_str()
                .context("cache root path was not valid utf-8")?,
        ],
    )
    .context("failed to unzip chromedriver bundle")?;

    if !binary.is_file() {
        bail!(
            "downloaded chromedriver bundle did not contain expected binary at {}",
            binary.display()
        );
    }
    Ok(binary)
}

fn chrome_version(chrome: &Path) -> Result<String> {
    let output = Command::new(chrome)
        .arg("--version")
        .output()
        .with_context(|| format!("failed to run `{}` --version", chrome.display()))?;
    if !output.status.success() {
        bail!("`{}` --version failed", chrome.display());
    }
    let stdout = String::from_utf8(output.stdout).context("chrome version output was not utf-8")?;
    stdout
        .split_whitespace()
        .find(|part| part.chars().next().is_some_and(|ch| ch.is_ascii_digit()))
        .map(str::to_owned)
        .context("failed to parse Chrome version")
}

fn write_webdriver_config(chrome: &Path) -> Result<PathBuf> {
    let config_path = workspace_root()
        .join("target")
        .join("xtask")
        .join("dragon-webdriver.json");
    if let Some(parent) = config_path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create webdriver config directory {}",
                parent.display()
            )
        })?;
    }
    let payload = format!(
        concat!(
            "{{\n",
            "  \"goog:chromeOptions\": {{\n",
            "    \"binary\": \"{}\",\n",
            "    \"args\": [\n",
            "      \"--enable-unsafe-webgpu\",\n",
            "      \"--use-angle=swiftshader\"\n",
            "    ]\n",
            "  }}\n",
            "}}\n"
        ),
        chrome.display()
    );
    fs::write(&config_path, payload).with_context(|| {
        format!(
            "failed to write webdriver config to {}",
            config_path.display()
        )
    })?;
    Ok(config_path)
}

fn resolve_chrome_path() -> Option<PathBuf> {
    resolve_binary(
        "BURN_DRAGON_PLAYWRIGHT_CHROME",
        &[
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        ],
    )
    .or_else(|| {
        resolve_binary(
            "BURN_P2P_PLAYWRIGHT_CHROME",
            &[
                "/usr/bin/google-chrome",
                "/usr/bin/google-chrome-stable",
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            ],
        )
    })
}

fn resolve_binary(env_var: &str, candidates: &[&str]) -> Option<PathBuf> {
    std::env::var_os(env_var)
        .map(PathBuf::from)
        .filter(|path| path.exists())
        .or_else(|| {
            candidates
                .iter()
                .map(PathBuf::from)
                .find(|path| path.exists())
        })
}

fn cuda_toolchain_available() -> bool {
    resolve_binary(
        "CUDA_HOME",
        &[
            "/usr/local/cuda",
            "/opt/cuda",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
        ],
    )
    .into_iter()
    .map(|root| root.join("bin").join(binary_name("nvcc")))
    .find(|path| path.is_file())
    .or_else(|| {
        ["CUDA_PATH", "CUDA_ROOT", "CUDA_TOOLKIT_ROOT_DIR"]
            .into_iter()
            .filter_map(std::env::var_os)
            .map(PathBuf::from)
            .map(|root| root.join("bin").join(binary_name("nvcc")))
            .find(|path| path.is_file())
    })
    .or_else(|| find_in_path(binary_name("nvcc")))
    .is_some()
}

fn find_in_path(binary: impl AsRef<Path>) -> Option<PathBuf> {
    let binary = binary.as_ref();
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(binary);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask should live under workspace root")
        .to_path_buf()
}
