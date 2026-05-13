use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail, ensure};
use clap::{Args, ValueEnum};
use serde::Serialize;
use serde_json::{Value, json};

use crate::deploy_settings::{ResolvePagesDeploySettingsArgs, resolve_pages_deploy_settings_inner};

const MONTHLY_HOURS: f64 = 730.0;
const PUBLIC_IPV4_HOURLY_USD: f64 = 0.005;
const GP3_STORAGE_PER_GIB_MONTH_USD: f64 = 0.08;
const CLOUDWATCH_DASHBOARD_MONTHLY_USD: f64 = 3.00;
const CLOUDWATCH_STANDARD_ALARM_MONTHLY_USD: f64 = 0.10;
const ROUTE53_HEALTH_CHECK_MONTHLY_USD: f64 = 0.50;
const MODEST_S3_STORAGE_RESERVE_MONTHLY_USD: f64 = 1.00;
const FIXED_MONTHLY_COST_CAP_USD: f64 = 100.00;

#[derive(Debug, Clone, Args)]
pub struct SummarizeLiveBrowserCanaryArgs {
    pub report_path: PathBuf,
}

#[derive(Debug, Clone, Args)]
pub struct SummarizeLiveNativeTrainingCanaryArgs {
    pub report_path: PathBuf,
}

#[derive(Debug, Clone, Args)]
pub struct SummarizeDeploymentDiagnosticsArgs {
    pub report_path: PathBuf,
}

#[derive(Debug, Clone, Args)]
pub struct SummarizeInspectionArgs {
    pub summary_path: PathBuf,
}

#[derive(Debug, Clone, Args)]
pub struct ExtractDeploymentDiagnosticsArgs {
    pub ssm_output_path: PathBuf,
    pub output_path: PathBuf,
}

#[derive(Debug, Clone, Args)]
pub struct WriteInspectionSummaryArgs {
    pub artifact_dir: PathBuf,
}

#[derive(Debug, Clone, Args)]
pub struct BootstrapStackSettingsArgs {
    #[arg(value_enum)]
    pub mode: BootstrapStackSettingsMode,
}

#[derive(Debug, Clone, Args)]
pub struct PublishCratesArgs {
    #[arg(long, default_value = "true")]
    pub dry_run: String,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum BootstrapStackSettingsMode {
    Deploy,
    Restore,
}

const PUBLISH_CRATES: &[&str] = &[
    "burn_dragon_tokenizer",
    "burn_dragon_universality",
    "burn_dragon_time",
    "burn_dragon_kernel",
    "burn_dragon_core",
    "burn_dragon_checkpoint",
    "burn_dragon_train",
    "burn_dragon_language",
    "burn_dragon_p2p",
];

#[derive(Debug, Serialize)]
struct GuardrailReport {
    environment: String,
    operation: String,
    workspace: String,
    fixed_monthly_cost_usd: f64,
    cost_breakdown: BTreeMap<String, f64>,
    errors: Vec<String>,
    warnings: Vec<String>,
}

pub fn resolve_pages_deploy_settings_outputs(args: &ResolvePagesDeploySettingsArgs) -> Result<()> {
    let settings = serde_json::to_value(resolve_pages_deploy_settings_inner(args)?)?;
    write_pages_settings_outputs(&settings)
}

fn write_pages_settings_outputs(settings: &Value) -> Result<()> {
    let seed_node_urls = settings
        .get("seed_node_urls")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join(",")
        })
        .unwrap_or_default();
    let env_lines = [
        format!("EDGE_BASE_URL={}", json_string(settings, "edge_base_url")),
        format!("SEED_NODE_URLS={seed_node_urls}"),
        format!(
            "SELECTED_EXPERIMENT_ID={}",
            json_string(settings, "selected_experiment_id")
        ),
        format!(
            "SELECTED_REVISION_ID={}",
            json_string(settings, "selected_revision_id")
        ),
        format!(
            "BROWSER_CANARY_PRINCIPAL_ID={}",
            json_string(settings, "canary_principal_id")
        ),
        format!("SITE_HOST={}", json_string(settings, "site_host")),
    ];
    let output_lines = [
        format!("edge_base_url={}", json_string(settings, "edge_base_url")),
        format!(
            "browser_app_base_url={}",
            json_string(settings, "browser_app_base_url")
        ),
        format!(
            "selected_experiment_id={}",
            json_string(settings, "selected_experiment_id")
        ),
        format!(
            "selected_revision_id={}",
            json_string(settings, "selected_revision_id")
        ),
        format!(
            "canary_principal_id={}",
            json_string(settings, "canary_principal_id")
        ),
    ];
    append_env_lines("GITHUB_ENV", &env_lines)?;
    append_env_lines("GITHUB_OUTPUT", &output_lines)?;
    Ok(())
}

pub fn install_playwright_chromium() -> Result<()> {
    run("npx", &["--yes", "playwright", "--version"], &[])?;
    run(
        "npx",
        &[
            "--yes",
            "playwright",
            "install",
            "--with-deps",
            &std::env::var("PLAYWRIGHT_BROWSERS").unwrap_or_else(|_| "chromium".to_owned()),
        ],
        &[],
    )
}

pub fn run_live_browser_canary() -> Result<()> {
    run_live_browser_canary_with_env(&[])
}

pub fn run_pages_predeploy_canary() -> Result<()> {
    for name in [
        "BURN_DRAGON_BROWSER_CANARY_SITE_BASE_URL",
        "BURN_DRAGON_BROWSER_CANARY_EDGE_BASE_URL",
        "BURN_DRAGON_BROWSER_CANARY_PRINCIPAL_ID",
        "BURN_DRAGON_BROWSER_CANARY_CALLBACK_TOKEN",
        "BURN_DRAGON_BROWSER_CANARY_EXPERIMENT_ID",
    ] {
        required_env(name)?;
    }

    let site_dir = std::env::var("BURN_DRAGON_PAGES_PREDEPLOY_SITE_DIR")
        .unwrap_or_else(|_| "target/browser-site".to_owned());
    for path in [
        Path::new(&site_dir).join("browser-app-config.json"),
        Path::new(&site_dir).join("burn_dragon_p2p_browser_bg.wasm"),
    ] {
        ensure!(
            path.is_file(),
            "predeploy browser canary requires a built browser site; missing {}",
            path.display()
        );
    }

    install_playwright_chromium()?;
    let artifact_dir =
        std::env::var("BURN_DRAGON_BROWSER_CANARY_ARTIFACT_DIR").unwrap_or_else(|_| {
            format!(
                "{}/burn-dragon-pages-predeploy-canary",
                std::env::var("RUNNER_TEMP").unwrap_or_else(|_| "/tmp".to_owned())
            )
        });
    let output_json = std::env::var("BURN_DRAGON_BROWSER_CANARY_OUTPUT_JSON")
        .unwrap_or_else(|_| format!("{artifact_dir}/canary-summary.json"));
    let overrides = [
        ("PLAYWRIGHT_BROWSERS", "chromium".to_owned()),
        ("BURN_DRAGON_BROWSER_CANARY_BROWSER", "chromium".to_owned()),
        (
            "BURN_DRAGON_BROWSER_CANARY_TRANSPORT_MODE",
            "webrtc-direct".to_owned(),
        ),
        ("BURN_DRAGON_BROWSER_CANARY_EXPECT_TRAINING", "1".to_owned()),
        (
            "BURN_DRAGON_BROWSER_CANARY_EXPECT_CHECKPOINT_SYNC",
            "0".to_owned(),
        ),
        (
            "BURN_DRAGON_BROWSER_CANARY_MIN_ACCEPTED_RECEIPTS",
            "2".to_owned(),
        ),
        ("BURN_DRAGON_BROWSER_CANARY_SITE_OVERRIDE_DIR", site_dir),
        ("BURN_DRAGON_BROWSER_CANARY_ARTIFACT_DIR", artifact_dir),
        ("BURN_DRAGON_BROWSER_CANARY_OUTPUT_JSON", output_json),
    ];
    run_live_browser_canary_with_env(&overrides)
}

fn run_live_browser_canary_with_env(overrides: &[(&str, String)]) -> Result<()> {
    let artifact_dir =
        env_override_or_current(overrides, "BURN_DRAGON_BROWSER_CANARY_ARTIFACT_DIR")?;
    let output_json = env_override_or_current(overrides, "BURN_DRAGON_BROWSER_CANARY_OUTPUT_JSON")?;
    fs::create_dir_all(&artifact_dir)
        .with_context(|| format!("failed to create browser canary artifact dir {artifact_dir}"))?;
    append_env_lines(
        "GITHUB_OUTPUT",
        &[
            format!("artifact_dir={artifact_dir}"),
            format!("report_path={output_json}"),
        ],
    )?;
    run("node", &["xtask/assets/live-browser-canary.mjs"], overrides)
}

pub fn summarize_live_browser_canary(args: &SummarizeLiveBrowserCanaryArgs) -> Result<()> {
    if !args.report_path.is_file() {
        return Ok(());
    }
    let report = read_json(&args.report_path)?;
    let receipt = object_field(&report, "receipt_submission");
    let durable_receipt = object_field(&report, "durable_receipt_snapshot");
    let e2e_contract = object_field(&report, "e2e_contract");
    let machine_state = object_field(&report, "browser_machine_state");
    let webrtc_markers = object_field(&report, "webrtc_direct_console_markers");
    let control_requests = array_field(&report, "quiet_window_control_plane_requests");
    let artifact_fallback = array_field(&report, "artifact_http_fallback_requests");
    let contract_invariants = e2e_contract
        .get("invariants")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let missing_contract = contract_invariants
        .iter()
        .filter(|item| {
            item.get("required")
                .and_then(Value::as_bool)
                .unwrap_or(false)
                && !item.get("passed").and_then(Value::as_bool).unwrap_or(false)
        })
        .filter_map(|item| item.get("name").and_then(Value::as_str))
        .collect::<Vec<_>>();
    println!("## live browser canary\n");
    println!("- Success: `{}`", bool_field(&report, "success"));
    println!(
        "- Principal id: `{}`",
        json_string_or(&report, "principal_id", "n/a")
    );
    println!(
        "- Browser: `{}`",
        json_string_or(&report, "browser_name", "n/a")
    );
    println!(
        "- Transport mode: `{}`",
        json_string_or(&report, "transport_mode", "n/a")
    );
    println!(
        "- Expected connected transport: `{}`",
        json_string_or(&report, "expected_connected_transport", "n/a")
    );
    println!(
        "- Expected minimum direct peers: `{}`",
        json_value_display(&report, "expected_min_direct_peers", "n/a")
    );
    println!(
        "- Expect training: `{}`",
        json_value_display(&report, "expect_training", "n/a")
    );
    println!(
        "- Minimum accepted receipts: `{}`",
        json_value_display(&report, "min_accepted_receipts", "n/a")
    );
    println!(
        "- Live status: `{}`",
        json_string_or(&report, "live_status_label", "n/a")
    );
    println!(
        "- Transport signal: `{}`",
        json_string_or(&report, "transport_summary", "n/a")
    );
    println!(
        "- Machine connected transport: `{}`",
        json_string_or(&machine_state, "connected_transport", "n/a")
    );
    println!(
        "- Machine direct peers: `{}`",
        json_value_display(&machine_state, "direct_peers", "n/a")
    );
    println!(
        "- Machine last error: `{}`",
        json_string_or(&machine_state, "last_error", "none")
    );
    println!(
        "- WebRTC-direct phase evidence: `{}/{}`",
        array_field(&webrtc_markers, "observed").len(),
        array_field(&webrtc_markers, "required").len()
    );
    println!(
        "- Missing WebRTC-direct phases: `{}`",
        json_string_array(&webrtc_markers, "missing")
            .join(", ")
            .if_empty("none")
    );
    println!(
        "- Signed seed transports: `{}`",
        json_string_array(&report, "signed_seed_transport_preference")
            .join(", ")
            .if_empty("none")
    );
    println!(
        "- Connect clicked: `{}`",
        bool_field(&report, "connect_clicked")
    );
    println!(
        "- Training button visible: `{}`",
        bool_field(&report, "training_button_visible")
    );
    println!(
        "- Training P2P checkpoint ready: `{}`",
        json_value_display(&report, "training_p2p_checkpoint_ready", "n/a")
    );
    println!(
        "- Quiet-window control-plane requests: `{}`",
        control_requests.len()
    );
    println!(
        "- Edge artifact fallback requests: `{}`",
        artifact_fallback.len()
    );
    println!(
        "- Receipt submissions: `{}`",
        array_field(&receipt, "submissions").len()
    );
    println!(
        "- Accepted receipt count: `{}`",
        json_value_display(&receipt, "accepted_receipt_count", "n/a")
    );
    println!(
        "- Accepted receipt ids: `{}`",
        json_string_array(&receipt, "accepted_receipt_ids")
            .join(", ")
            .if_empty("none")
    );
    println!(
        "- Durable receipts: `{}` (baseline `{}`)",
        json_value_display(&durable_receipt, "observed_accepted_receipts", "n/a"),
        json_value_display(&report, "accepted_receipts_before_training", "n/a")
    );
    println!(
        "- E2E contract: `{}`",
        json_value_display(&e2e_contract, "passed", "n/a")
    );
    println!(
        "- Failed required invariants: `{}`",
        missing_contract.join(", ").if_empty("none")
    );
    Ok(())
}

pub fn summarize_live_native_training_canary(
    args: &SummarizeLiveNativeTrainingCanaryArgs,
) -> Result<()> {
    if !args.report_path.is_file() {
        return Ok(());
    }
    let report = read_json(&args.report_path)?;
    let windows = array_field(&report, "windows");

    println!("## live native training canary\n");
    println!("- Success: `{}`", bool_field(&report, "success"));
    println!(
        "- Edge: `{}`",
        json_string_or(&report, "edge_base_url", "n/a")
    );
    println!(
        "- Experiment: `{}`",
        json_string_or(&report, "experiment_id", "n/a")
    );
    println!("- Backend: `{}`", json_string_or(&report, "backend", "n/a"));
    println!(
        "- Head sync timeout: `{}s`",
        json_value_display(&report, "head_sync_timeout_secs", "n/a")
    );
    println!(
        "- Settle diffusion: `{}`",
        json_value_display(&report, "settle_diffusion", "n/a")
    );
    println!(
        "- Diffusion settle passes: `{}`",
        json_value_display(&report, "diffusion_settle_passes", "n/a")
    );
    println!(
        "- Serve after publish: `{}s`",
        json_value_display(&report, "serve_after_publish_secs", "n/a")
    );
    println!(
        "- Start local validator: `{}`",
        json_value_display(&report, "start_local_validator", "n/a")
    );
    println!(
        "- Training batch size: `{}`",
        json_value_display(&report, "training_batch_size", "n/a")
    );
    println!(
        "- Training max iters: `{}`",
        json_value_display(&report, "training_max_iters", "n/a")
    );
    println!(
        "- Evaluation max batches: `{}`",
        json_value_display(&report, "evaluation_max_batches", "n/a")
    );
    println!(
        "- Head before: `{}`",
        object_field(&report, "head_before")
            .get("head_id")
            .and_then(Value::as_str)
            .unwrap_or("n/a")
    );
    println!(
        "- Head after: `{}`",
        object_field(&report, "head_after")
            .get("head_id")
            .and_then(Value::as_str)
            .unwrap_or("n/a")
    );
    println!("- Windows: `{}`", windows.len());
    for window in windows {
        let train = object_field(&window, "train_report");
        let settlement = object_field(&train, "diffusion_settlement");
        let train_signal = object_field(&window, "train_signal");
        let canonical_signal = object_field(&window, "canonical_signal");
        let published_p2p_signal = object_field(&window, "published_head_p2p_signal");
        let p2p_signal = object_field(&window, "p2p_signal");
        println!(
            "  - window `{}`: `{}` -> `{}`, published `{}`, train loss `{}`, canonical loss `{}`, canonical improved `{}`, published p2p wait `{}s`, canonical wait `{}s`, settlement passes `{}`, settlement certs `{}`, canonical p2p wait `{}s`, published p2p heads `{}`, p2p updates `{}`, p2p attestations `{}`, p2p diffusion certs `{}`",
            json_value_display(&window, "window_index", "n/a"),
            object_field(&window, "head_before")
                .get("head_id")
                .and_then(Value::as_str)
                .unwrap_or("n/a"),
            object_field(&window, "head_after")
                .get("head_id")
                .and_then(Value::as_str)
                .unwrap_or("n/a"),
            json_string_or(&train, "published_head_id", "n/a"),
            json_value_display(&train_signal, "train_loss", "n/a"),
            json_value_display(&canonical_signal, "canonical_loss_after", "n/a"),
            json_value_display(&canonical_signal, "canonical_loss_improved", "n/a"),
            json_value_display(&window, "published_head_p2p_wait_secs", "n/a"),
            json_value_display(&window, "canonical_wait_secs", "n/a"),
            json_value_display(&settlement, "passes_completed", "n/a"),
            json_value_display(&settlement, "certificates", "n/a"),
            json_value_display(&window, "p2p_wait_secs", "n/a"),
            json_value_display(&published_p2p_signal, "head_announcements", "n/a"),
            json_value_display(&p2p_signal, "update_announcements", "n/a"),
            json_value_display(
                &p2p_signal,
                "trainer_promotion_attestation_announcements",
                "n/a"
            ),
            json_value_display(
                &p2p_signal,
                "diffusion_promotion_certificate_announcements",
                "n/a"
            ),
        );
    }
    Ok(())
}

pub fn summarize_deployment_diagnostics(args: &SummarizeDeploymentDiagnosticsArgs) -> Result<()> {
    if !args.report_path.is_file() {
        return Ok(());
    }
    let report = read_json(&args.report_path)?;
    let readiness = object_field(&report, "readiness");
    let profile_resolution = object_field(&object_field(&report, "profile_resolution"), "value");
    let edge_snapshot = object_field(&object_field(&report, "edge_snapshot"), "value");
    let artifact_head_check = object_field(&report, "artifact_head_view");
    let artifact_head_view = object_field(&artifact_head_check, "value");
    let blockers = json_string_array(&readiness, "blocking_issues");
    let warnings = json_string_array(&readiness, "observed_warnings");

    println!("## deployment diagnostics\n");
    println!("- Ready: `{}`", bool_field(&readiness, "ready"));
    println!(
        "- Profile source: `{}`",
        json_string_or(&profile_resolution, "source", "unknown")
    );
    println!(
        "- Matching head present: `{}`",
        bool_field(&edge_snapshot, "matching_head_present")
    );
    println!(
        "- Matching head id: `{}`",
        json_string_or(&edge_snapshot, "matching_head_id", "none")
    );
    println!(
        "- Directory current head id: `{}`",
        json_string_or(&edge_snapshot, "matching_directory_current_head_id", "none")
    );
    println!(
        "- Directory current head visible: `{}`",
        bool_field(&edge_snapshot, "matching_directory_current_head_visible")
    );
    println!(
        "- Matching head global step: `{}`",
        json_value_display(&edge_snapshot, "matching_head_global_step", "n/a")
    );
    println!(
        "- Matching directory entry present: `{}`",
        bool_field(&edge_snapshot, "matching_directory_entry_present")
    );
    println!(
        "- Artifact head probe ok: `{}`",
        bool_field(&artifact_head_check, "ok")
    );
    println!(
        "- Artifact connected providers: `{}`",
        json_string_array(&artifact_head_view, "connected_provider_peer_ids")
            .join(", ")
            .if_empty("none")
    );
    println!("- Blockers: `{}`", blockers.join(", ").if_empty("none"));
    println!("- Warnings: `{}`", warnings.join(", ").if_empty("none"));
    Ok(())
}

pub fn summarize_inspection(args: &SummarizeInspectionArgs) -> Result<()> {
    if !args.summary_path.is_file() {
        return Ok(());
    }
    let summary = read_json(&args.summary_path)?;
    let blockers = json_string_array(&summary, "blocking_issues");
    let warnings = json_string_array(&summary, "observed_warnings");

    println!("## inspection summary\n");
    println!(
        "- Instance id: `{}`",
        json_string_or(&summary, "instance_id", "unknown")
    );
    println!(
        "- SSM status: `{}`",
        json_string_or(&summary, "ssm_status", "unknown")
    );
    println!(
        "- Deployment ready: `{}`",
        json_value_display(&summary, "deployment_ready", "n/a")
    );
    println!(
        "- Profile source: `{}`",
        json_string_or(&summary, "profile_source", "unknown")
    );
    println!(
        "- Matching head present: `{}`",
        json_value_display(&summary, "matching_head_present", "n/a")
    );
    println!(
        "- Matching head id: `{}`",
        json_string_or(&summary, "matching_head_id", "none")
    );
    println!("- Blockers: `{}`", blockers.join(", ").if_empty("none"));
    println!("- Warnings: `{}`", warnings.join(", ").if_empty("none"));
    Ok(())
}

pub fn extract_deployment_diagnostics(args: &ExtractDeploymentDiagnosticsArgs) -> Result<()> {
    if !args.ssm_output_path.exists() {
        return Ok(());
    }
    let payload = read_json(&args.ssm_output_path)?;
    let stdout = payload
        .get("StandardOutputContent")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let begin = "--- deployment diagnostics json begin ---";
    let end = "--- deployment diagnostics json end ---";
    let Some(start) = stdout.find(begin).map(|index| index + begin.len()) else {
        return Ok(());
    };
    let Some(finish) = stdout[start..].find(end).map(|index| start + index) else {
        return Ok(());
    };
    let body = stdout[start..finish].trim();
    if body.is_empty() {
        return Ok(());
    }
    serde_json::from_str::<Value>(body).context("deployment diagnostics block is not JSON")?;
    fs::write(&args.output_path, format!("{body}\n"))
        .with_context(|| format!("failed to write {}", args.output_path.display()))?;
    Ok(())
}

pub fn write_inspection_summary(args: &WriteInspectionSummaryArgs) -> Result<()> {
    fs::create_dir_all(&args.artifact_dir)
        .with_context(|| format!("failed to create {}", args.artifact_dir.display()))?;
    let mut summary = json!({
        "instance_id": null,
        "ssm_status": null,
        "deployment_ready": null,
        "profile_source": null,
        "matching_head_present": null,
        "matching_head_id": null,
        "blocking_issues": [],
        "observed_warnings": [],
    });
    let describe_path = args.artifact_dir.join("instance-describe.json");
    if describe_path.exists() {
        let describe = read_json(&describe_path)?;
        if let Some(instance) = describe
            .get("Reservations")
            .and_then(Value::as_array)
            .and_then(|reservations| reservations.first())
            .and_then(|reservation| reservation.get("Instances"))
            .and_then(Value::as_array)
            .and_then(|instances| instances.first())
        {
            summary["instance_id"] = instance.get("InstanceId").cloned().unwrap_or(Value::Null);
        }
    }
    let diagnostics_path = args
        .artifact_dir
        .join("bootstrap-deployment-diagnostics.json");
    if diagnostics_path.exists() {
        let diagnostics = read_json(&diagnostics_path)?;
        let readiness = object_field(&diagnostics, "readiness");
        let edge_snapshot = object_field(&object_field(&diagnostics, "edge_snapshot"), "value");
        let profile_resolution =
            object_field(&object_field(&diagnostics, "profile_resolution"), "value");
        summary["deployment_ready"] = readiness.get("ready").cloned().unwrap_or(Value::Null);
        summary["profile_source"] = profile_resolution
            .get("source")
            .cloned()
            .unwrap_or(Value::Null);
        summary["matching_head_present"] = edge_snapshot
            .get("matching_head_present")
            .cloned()
            .unwrap_or(Value::Null);
        summary["matching_head_id"] = edge_snapshot
            .get("matching_head_id")
            .cloned()
            .unwrap_or(Value::Null);
        summary["blocking_issues"] = readiness
            .get("blocking_issues")
            .cloned()
            .unwrap_or_else(|| json!([]));
        summary["observed_warnings"] = readiness
            .get("observed_warnings")
            .cloned()
            .unwrap_or_else(|| json!([]));
    }
    let ssm_status_path = args.artifact_dir.join("ssm-status.txt");
    if ssm_status_path.exists() {
        summary["ssm_status"] = json!(fs::read_to_string(&ssm_status_path)?.trim());
    }
    fs::write(
        args.artifact_dir.join("inspection-summary.json"),
        serde_json::to_string_pretty(&summary)? + "\n",
    )?;
    Ok(())
}

pub fn write_bootstrap_inspect_params(output_path: &Path) -> Result<()> {
    let run_bootstrap_start_probe =
        env_or("BURN_DRAGON_INSPECT_RUN_BOOTSTRAP_START_PROBE", "false");
    let mut commands = vec![format!(
        "export BURN_DRAGON_INSPECT_RUN_BOOTSTRAP_START_PROBE={run_bootstrap_start_probe:?}"
    )];
    commands.extend(
        include_str!("../assets/bootstrap-inspect-commands.txt")
            .lines()
            .map(str::trim_end)
            .filter(|line| !line.trim().is_empty())
            .map(str::to_owned),
    );
    fs::write(
        output_path,
        serde_json::to_string(&json!({ "commands": commands }))?,
    )
    .with_context(|| format!("failed to write {}", output_path.display()))?;
    Ok(())
}

pub fn render_head_mirror_seed_repair_commands() -> Result<()> {
    let private_ip = required_env("BOOTSTRAP_PRIVATE_IP")?;
    let replacement = format!("seed_node_urls = [\\n  \"/ip4/{private_ip}/tcp/4001\",\\n]\\n");
    let rewrite = format!(
        "python3 - <<'PY2'\nfrom pathlib import Path\nimport re\np = Path('/etc/burn_dragon_p2p/bootstrap-head-mirror.toml')\ntext = p.read_text()\nreplacement = {replacement:?}\ntext, count = re.subn(r'seed_node_urls\\s*=\\s*\\[(?:.|\\n)*?\\]\\n', replacement, text, count=1, flags=re.MULTILINE)\nif count != 1:\n    raise SystemExit('failed to rewrite seed_node_urls in bootstrap-head-mirror.toml')\np.write_text(text)\nprint(text)\nPY2"
    );
    let commands = vec![
        "set -eu".to_owned(),
        rewrite,
        "/usr/local/bin/burn-dragon-p2p-fetch-head-mirror-auth-bundle || true".to_owned(),
        "systemctl reset-failed burn-dragon-p2p-head-mirror || true".to_owned(),
        "systemctl restart burn-dragon-p2p-head-mirror".to_owned(),
        "systemctl status burn-dragon-p2p-head-mirror --no-pager || true".to_owned(),
        "journalctl -u burn-dragon-p2p-head-mirror --no-pager -n 120 || true".to_owned(),
        "curl -fsS https://127.0.0.1/heads || true".to_owned(),
        "curl -fsS https://127.0.0.1/portal/snapshot || true".to_owned(),
    ];
    println!(
        "{}",
        serde_json::to_string(&json!({ "commands": commands }))?
    );
    Ok(())
}

pub fn check_deployment_guardrails() -> Result<()> {
    let report = build_guardrail_report()?;
    write_guardrail_outputs(&report)?;
    println!(
        "[guardrails] {} {}/{}: estimated fixed monthly AWS cost ${:.2}",
        report.operation, report.environment, report.workspace, report.fixed_monthly_cost_usd
    );
    for (key, value) in &report.cost_breakdown {
        println!("[guardrails]   {key}: ${value:.2}");
    }
    for warning in &report.warnings {
        println!("[guardrails] warning: {warning}");
    }
    if !report.errors.is_empty() {
        for error in &report.errors {
            eprintln!("[guardrails] error: {error}");
        }
        bail!("deployment guardrails failed");
    }
    Ok(())
}

pub fn deployment_guardrail_report() -> Result<()> {
    let report = build_guardrail_report()?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

pub fn resolve_bootstrap_stack_settings(args: &BootstrapStackSettingsArgs) -> Result<()> {
    crate::bootstrap_settings::resolve(args.mode)
}

pub fn sync_bootstrap_runtime_config() -> Result<()> {
    crate::bootstrap_runtime::sync_bootstrap_runtime_config()
}

pub fn run_live_native_training_canary() -> Result<()> {
    crate::native_canary::run()
}

pub fn publish_crates(args: &PublishCratesArgs) -> Result<()> {
    let dry_run = parse_bool(&args.dry_run)?;
    let version = workspace_version()?;
    if !dry_run {
        required_env("CARGO_REGISTRY_TOKEN")?;
    }

    for krate in PUBLISH_CRATES {
        if dry_run {
            verify_publish_crate(krate)?;
            continue;
        }
        run(
            "cargo",
            &[
                "publish",
                "--manifest-path",
                "Cargo.toml",
                "-p",
                krate,
                "--locked",
            ],
            &[],
        )?;
        wait_for_crate_version(krate, &version)?;
    }
    Ok(())
}

fn build_guardrail_report() -> Result<GuardrailReport> {
    let environment = env_or("DEPLOY_ENVIRONMENT", "production");
    let operation = env_or("DEPLOYMENT_OPERATION", "deploy");
    let workspace = env_or("TF_WORKSPACE_NAME", "");
    let bootstrap_install_source = env_first(
        &[
            "BOOTSTRAP_INSTALL_SOURCE",
            "TF_VAR_bootstrap_install_source",
        ],
        "bootstrap_install_source",
    )?;
    let instance_type = env_first(&["TF_VAR_instance_type"], "instance_type")?;
    let root_volume_size_gib = parse_i64(&env_first(
        &["TF_VAR_root_volume_size_gib"],
        "root_volume_size_gib",
    )?)?;
    let use_retained_bootstrap_data_volume = parse_bool(&env_first(
        &["TF_VAR_use_retained_bootstrap_data_volume"],
        "use_retained_bootstrap_data_volume",
    )?)?;
    let data_volume_size_gib = parse_i64(&env_first(
        &["TF_VAR_data_volume_size_gib"],
        "data_volume_size_gib",
    )?)?;
    let enable_bootstrap_status_alarms = parse_bool(&env_first(
        &["TF_VAR_enable_bootstrap_status_alarms"],
        "enable_bootstrap_status_alarms",
    )?)?;
    let enable_control_plane_operational_alarms = parse_bool(&env_first(
        &["TF_VAR_enable_control_plane_operational_alarms"],
        "enable_control_plane_operational_alarms",
    )?)?;
    let enable_control_plane_dashboard = parse_bool(&env_first(
        &["TF_VAR_enable_control_plane_dashboard"],
        "enable_control_plane_dashboard",
    )?)?;
    let enable_managed_control_plane_redis = parse_bool(&env_first(
        &["TF_VAR_enable_managed_control_plane_redis"],
        "enable_managed_control_plane_redis",
    )?)?;
    let managed_trainer_desired_capacity = parse_i64(&env_first(
        &["TF_VAR_managed_trainer_desired_capacity"],
        "managed_trainer_desired_capacity",
    )?)?;
    let managed_trainer_instance_type = env_first(
        &["TF_VAR_managed_trainer_instance_type"],
        "managed_trainer_instance_type",
    )?;
    let managed_trainer_root_volume_size_gib = parse_i64(&env_first(
        &["TF_VAR_managed_trainer_root_volume_size_gib"],
        "managed_trainer_root_volume_size_gib",
    )?)?;
    let disaster_recovery_region = env_first(
        &["TF_VAR_disaster_recovery_region"],
        "disaster_recovery_region",
    )?;
    let enable_data_volume_snapshots = parse_bool(&env_first(
        &["TF_VAR_enable_data_volume_snapshots"],
        "enable_data_volume_snapshots",
    )?)?;
    let enable_disaster_recovery_snapshot_copies = parse_bool(&env_first(
        &["TF_VAR_enable_disaster_recovery_snapshot_copies"],
        "enable_disaster_recovery_snapshot_copies",
    )?)?;
    let alarm_sns_topic_arn = env_first(
        &[
            "TF_VAR_alarm_sns_topic_arn",
            "BURN_DRAGON_P2P_ALARM_SNS_TOPIC_ARN",
        ],
        "alarm_sns_topic_arn",
    )?;

    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    let mut breakdown = BTreeMap::new();
    if let Some(rate) = instance_hourly_usd(&instance_type) {
        breakdown.insert("bootstrap_ec2".to_owned(), monthly_cost_from_hourly(rate));
    } else {
        errors.push(format!(
            "unknown bootstrap instance type `{instance_type}` for cost guardrails; extend xtask workflow_tools before deploying"
        ));
    }
    breakdown.insert(
        "bootstrap_root_gp3".to_owned(),
        root_volume_size_gib as f64 * GP3_STORAGE_PER_GIB_MONTH_USD,
    );
    breakdown.insert(
        "bootstrap_public_ipv4".to_owned(),
        monthly_cost_from_hourly(PUBLIC_IPV4_HOURLY_USD),
    );
    breakdown.insert(
        "route53_health_check".to_owned(),
        ROUTE53_HEALTH_CHECK_MONTHLY_USD,
    );
    breakdown.insert(
        "modest_s3_storage_reserve".to_owned(),
        MODEST_S3_STORAGE_RESERVE_MONTHLY_USD,
    );
    if use_retained_bootstrap_data_volume {
        breakdown.insert(
            "bootstrap_retained_data_gp3".to_owned(),
            data_volume_size_gib as f64 * GP3_STORAGE_PER_GIB_MONTH_USD,
        );
        if enable_data_volume_snapshots {
            warnings.push("bootstrap data snapshots are enabled; snapshot storage is usage-driven and is not fully modeled in the fixed monthly estimate".to_owned());
        }
    }
    let mut alarm_count = 0;
    if enable_bootstrap_status_alarms {
        alarm_count += 2;
    }
    if enable_control_plane_operational_alarms {
        alarm_count += 2;
        if enable_managed_control_plane_redis {
            alarm_count += 2;
        }
        if managed_trainer_desired_capacity > 0 {
            alarm_count += 1;
        }
    }
    if alarm_count > 0 {
        breakdown.insert(
            "cloudwatch_alarms".to_owned(),
            alarm_count as f64 * CLOUDWATCH_STANDARD_ALARM_MONTHLY_USD,
        );
    }
    if enable_control_plane_dashboard {
        breakdown.insert(
            "cloudwatch_dashboard".to_owned(),
            CLOUDWATCH_DASHBOARD_MONTHLY_USD,
        );
    }
    if enable_managed_control_plane_redis {
        breakdown.insert(
            "control_plane_redis".to_owned(),
            monthly_cost_from_hourly(0.0320),
        );
    }
    if managed_trainer_desired_capacity > 0 {
        if let Some(rate) = instance_hourly_usd(&managed_trainer_instance_type) {
            let per_trainer = monthly_cost_from_hourly(rate)
                + managed_trainer_root_volume_size_gib as f64 * GP3_STORAGE_PER_GIB_MONTH_USD;
            breakdown.insert(
                "managed_trainer_pool".to_owned(),
                managed_trainer_desired_capacity as f64 * per_trainer,
            );
        } else {
            errors.push(format!(
                "unknown managed trainer instance type `{managed_trainer_instance_type}` for cost guardrails; extend xtask workflow_tools before deploying"
            ));
        }
    }
    if !disaster_recovery_region.is_empty() {
        warnings.push(format!(
            "warm DR is enabled for `{disaster_recovery_region}`; replicated storage and snapshot-copy costs are usage-driven and not fully modeled in the fixed monthly estimate"
        ));
        if enable_disaster_recovery_snapshot_copies {
            warnings.push("cross-region snapshot copies are enabled; snapshot copy/storage costs are not included in the fixed monthly estimate".to_owned());
        }
    }
    let fixed_monthly_cost_usd = (breakdown.values().sum::<f64>() * 100.0).round() / 100.0;
    let alarms_expected = enable_bootstrap_status_alarms || enable_control_plane_operational_alarms;
    if environment == "production" {
        if workspace != "mainnet" {
            errors.push(format!(
                "production deployment must use terraform workspace `mainnet`, got `{}`",
                if workspace.is_empty() {
                    "unset"
                } else {
                    &workspace
                }
            ));
        }
        if bootstrap_install_source != "crate" {
            errors.push(format!(
                "production deployment must use bootstrap_install_source=crate, got `{bootstrap_install_source}`"
            ));
        }
        if !enable_bootstrap_status_alarms {
            errors
                .push("production deployment must keep bootstrap status alarms enabled".to_owned());
        }
        if !enable_control_plane_operational_alarms {
            errors.push(
                "production deployment must keep control-plane operational alarms enabled"
                    .to_owned(),
            );
        }
        if !enable_control_plane_dashboard {
            errors.push(
                "production deployment must keep the control-plane dashboard enabled".to_owned(),
            );
        }
        if alarms_expected && alarm_sns_topic_arn.is_empty() {
            errors.push("production deployment must set BURN_DRAGON_P2P_ALARM_SNS_TOPIC_ARN so CloudWatch alarms route somewhere actionable".to_owned());
        }
    }
    if fixed_monthly_cost_usd > FIXED_MONTHLY_COST_CAP_USD {
        errors.push(format!(
            "estimated fixed monthly AWS cost ${fixed_monthly_cost_usd:.2} exceeds the hard cap of ${FIXED_MONTHLY_COST_CAP_USD:.2}"
        ));
    }
    Ok(GuardrailReport {
        environment,
        operation,
        workspace,
        fixed_monthly_cost_usd,
        cost_breakdown: breakdown,
        errors,
        warnings,
    })
}

fn write_guardrail_outputs(report: &GuardrailReport) -> Result<()> {
    append_env_lines(
        "GITHUB_OUTPUT",
        &[
            format!(
                "fixed_monthly_cost_usd={:.2}",
                report.fixed_monthly_cost_usd
            ),
            format!(
                "fixed_monthly_cost_breakdown_json={}",
                serde_json::to_string(&report.cost_breakdown)?
            ),
            format!("warning_count={}", report.warnings.len()),
        ],
    )
}

fn env_first(names: &[&str], terraform_variable: &str) -> Result<String> {
    for name in names {
        if let Ok(value) = std::env::var(name)
            && !value.is_empty()
        {
            return Ok(value.trim().to_owned());
        }
    }
    terraform_default(terraform_variable)
}

fn terraform_default(variable_name: &str) -> Result<String> {
    let path = workspace_root().join("crates/burn_dragon_p2p/deploy/terraform/aws/variables.tf");
    let text =
        fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let marker = format!("variable \"{variable_name}\"");
    let start = text
        .find(&marker)
        .ok_or_else(|| anyhow!("missing terraform variable {variable_name}"))?;
    let rest = &text[start..];
    let default = rest
        .lines()
        .map(str::trim)
        .find_map(|line| {
            line.strip_prefix("default")
                .and_then(|value| value.split_once('=').map(|(_, raw)| raw.trim()))
        })
        .ok_or_else(|| anyhow!("missing terraform default for {variable_name}"))?;
    Ok(default.trim_matches('"').to_owned())
}

fn instance_hourly_usd(instance_type: &str) -> Option<f64> {
    match instance_type {
        "t3a.small" => Some(0.0188),
        "t3a.medium" => Some(0.0376),
        "t3a.large" => Some(0.0752),
        "m7i.large" => Some(0.1008),
        _ => None,
    }
}

fn monthly_cost_from_hourly(rate: f64) -> f64 {
    rate * MONTHLY_HOURS
}

fn parse_bool(value: &str) -> Result<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" | "" => Ok(false),
        _ => bail!("invalid boolean value: {value:?}"),
    }
}

fn parse_i64(value: &str) -> Result<i64> {
    value
        .trim()
        .parse()
        .with_context(|| format!("invalid integer value: {value:?}"))
}

fn verify_publish_crate(krate: &str) -> Result<()> {
    run(
        "cargo",
        &[
            "package",
            "--manifest-path",
            "Cargo.toml",
            "-p",
            krate,
            "--locked",
            "--list",
        ],
        &[],
    )?;
    run(
        "cargo",
        &[
            "check",
            "--manifest-path",
            "Cargo.toml",
            "-p",
            krate,
            "--locked",
        ],
        &[],
    )
}

fn wait_for_crate_version(krate: &str, version: &str) -> Result<()> {
    let url = format!("https://crates.io/api/v1/crates/{krate}");
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(20))
        .build()
        .context("build crates.io client")?;
    for attempt in 1..=36 {
        let found = client
            .get(&url)
            .send()
            .and_then(|response| response.error_for_status())
            .and_then(|response| response.json::<Value>())
            .ok()
            .and_then(|payload| payload.get("versions").and_then(Value::as_array).cloned())
            .is_some_and(|versions| {
                versions
                    .iter()
                    .any(|item| item.get("num").and_then(Value::as_str) == Some(version))
            });
        if found {
            return Ok(());
        }
        if attempt < 36 {
            thread::sleep(Duration::from_secs(10));
        }
    }
    bail!("{krate} {version} did not appear on crates.io before timeout")
}

fn workspace_version() -> Result<String> {
    let manifest = read_json_like_toml_value("Cargo.toml", "workspace.package.version")?;
    ensure!(!manifest.is_empty(), "workspace package version is empty");
    Ok(manifest)
}

fn read_json_like_toml_value(path: &str, dotted_key: &str) -> Result<String> {
    let text = fs::read_to_string(workspace_root().join(path))
        .with_context(|| format!("failed to read {path}"))?;
    let mut in_workspace_package = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            in_workspace_package = trimmed == "[workspace.package]";
            continue;
        }
        if in_workspace_package
            && let Some(raw) = trimmed
                .strip_prefix(dotted_key.rsplit('.').next().unwrap_or(dotted_key))
                .and_then(|value| value.split_once('=').map(|(_, raw)| raw.trim()))
        {
            return Ok(raw.trim_matches('"').to_owned());
        }
    }
    bail!("missing {dotted_key} in {path}")
}

fn read_json(path: &Path) -> Result<Value> {
    serde_json::from_slice(
        &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to decode {}", path.display()))
}

fn run(program: &str, args: &[&str], envs: &[(&str, String)]) -> Result<()> {
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

fn cargo_bin() -> String {
    if let Ok(path) = std::env::var("CARGO")
        && !path.is_empty()
    {
        return path;
    }
    let direct =
        Path::new("/home/mosure/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin/cargo");
    if direct.exists() {
        return direct.display().to_string();
    }
    "cargo".to_owned()
}

fn append_env_lines(env_name: &str, lines: &[String]) -> Result<()> {
    let Ok(path) = std::env::var(env_name) else {
        return Ok(());
    };
    if path.is_empty() {
        return Ok(());
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("failed to open {env_name} file {path}"))?;
    for line in lines {
        writeln!(file, "{line}")?;
    }
    Ok(())
}

fn required_env(name: &str) -> Result<String> {
    let value = std::env::var(name).unwrap_or_default();
    ensure!(!value.is_empty(), "{name} must be set");
    Ok(value)
}

fn env_override_or_current(overrides: &[(&str, String)], key: &str) -> Result<String> {
    if let Some((_, value)) = overrides.iter().find(|(name, _)| *name == key) {
        return Ok(value.clone());
    }
    required_env(key)
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name)
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| default.to_owned())
        .trim()
        .to_owned()
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask manifest has workspace parent")
        .to_path_buf()
}

fn json_string(value: &Value, field: &str) -> String {
    value
        .get(field)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

fn json_string_or(value: &Value, field: &str, default: &str) -> String {
    value
        .get(field)
        .and_then(Value::as_str)
        .unwrap_or(default)
        .to_owned()
}

fn json_value_display(value: &Value, field: &str, default: &str) -> String {
    match value.get(field) {
        Some(Value::Null) | None => default.to_owned(),
        Some(Value::String(value)) => value.clone(),
        Some(value) => value.to_string(),
    }
}

fn bool_field(value: &Value, field: &str) -> bool {
    value.get(field).and_then(Value::as_bool).unwrap_or(false)
}

fn object_field(value: &Value, field: &str) -> Value {
    value.get(field).cloned().unwrap_or_else(|| json!({}))
}

fn array_field(value: &Value, field: &str) -> Vec<Value> {
    value
        .get(field)
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

fn json_string_array(value: &Value, field: &str) -> Vec<String> {
    array_field(value, field)
        .into_iter()
        .filter_map(|value| value.as_str().map(ToOwned::to_owned))
        .collect()
}

trait EmptyDefault {
    fn if_empty(self, default: &str) -> String;
}

impl EmptyDefault for String {
    fn if_empty(self, default: &str) -> String {
        if self.is_empty() {
            default.to_owned()
        } else {
            self
        }
    }
}
