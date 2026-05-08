use std::fmt::Write as _;
use std::fs::{self, File};
use std::net::{SocketAddr, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};

pub fn run() -> Result<()> {
    let config = NativeCanaryConfig::from_env()?;
    fs::create_dir_all(&config.artifact_dir)?;

    let repair_storage = config.artifact_dir.join("repair-storage");
    let trainer_storage = config.artifact_dir.join("trainer-enroll-storage");
    let validator_storage = config.artifact_dir.join("validator-storage");
    let probe_storage = config.artifact_dir.join("probe-storage");
    let repair_bundle = config.artifact_dir.join("repair-auth-bundle.json");
    let trainer_bundle = config.artifact_dir.join("trainer-auth-bundle.json");
    let validator_bundle = config.artifact_dir.join("validator-auth-bundle.json");

    let repair_report = if config.repair_current_head_to_visible_root {
        enroll_static_principal(
            &config,
            &config.principal_id,
            "trainer",
            &repair_bundle,
            &repair_storage,
            &config.artifact_dir.join("enroll-repair.log"),
            &config.backend,
        )?;
        Some(repair_current_head_to_visible_root(
            &config,
            &repair_bundle,
            &repair_storage,
        )?)
    } else {
        None
    };

    let head_before = current_directory_head(&config.edge_base_url, &config.experiment_id)?;
    let p2p_before = p2p_probe_summary(&probe_p2p_snapshot(
        &config.binary,
        &config.edge_base_url,
        &probe_storage,
        &config.artifact_dir.join("p2p-probe-before.log"),
        60,
    )?);
    let head_provider_before = if head_before.get("head_id").and_then(Value::as_str).is_some() {
        Some(assert_head_provider_signal(
            &head_before,
            &p2p_before,
            config.require_edge_head_provider,
        )?)
    } else {
        None
    };
    let initialize_head_on_start = head_before.get("head_id").and_then(Value::as_str).is_none();

    enroll_static_principal(
        &config,
        &config.principal_id,
        "trainer",
        &trainer_bundle,
        &trainer_storage,
        &config.artifact_dir.join("enroll-trainer.log"),
        &config.backend,
    )?;

    let mut validator = None;
    if config.start_local_validator {
        enroll_static_principal(
            &config,
            &config.validator_principal_id,
            "validator",
            &validator_bundle,
            &validator_storage,
            &config.artifact_dir.join("enroll-validator.log"),
            "cpu",
        )?;
        validator = Some(start_validator(
            &config,
            &validator_bundle,
            &validator_storage,
            &config.artifact_dir.join("validator.log"),
            initialize_head_on_start,
        )?);
    }

    let result = run_windows(
        &config,
        &trainer_bundle,
        &probe_storage,
        &head_before,
        initialize_head_on_start,
    );
    if let Some(mut child) = validator {
        stop_child(&mut child);
    }
    let window_reports = result?;

    let summary = json!({
        "success": true,
        "edge_base_url": config.edge_base_url,
        "experiment_kind": config.experiment_kind,
        "experiment_id": config.experiment_id,
        "backend": config.backend,
        "head_sync_timeout_secs": config.head_sync_timeout_secs,
        "settle_diffusion": config.settle_diffusion,
        "diffusion_settle_passes": config.diffusion_settle_passes,
        "serve_after_publish_secs": config.serve_after_publish_secs,
        "start_local_validator": config.start_local_validator,
        "mirror_live_head_to_edge": config.mirror_live_head_to_edge,
        "require_edge_head_provider": config.require_edge_head_provider,
        "repair_current_head_to_visible_root": config.repair_current_head_to_visible_root,
        "require_canonical_loss_non_regression": config.require_canonical_loss_non_regression,
        "training_batch_size": config.training_batch_size,
        "training_max_iters": config.training_max_iters,
        "evaluation_max_batches": config.evaluation_max_batches,
        "p2p_timeout_secs": config.p2p_timeout_secs,
        "initialize_head_on_start": initialize_head_on_start,
        "principal_id": config.principal_id,
        "validator_principal_id": config.validator_principal_id,
        "head_before": head_before,
        "p2p_before": p2p_before,
        "head_provider_before": head_provider_before,
        "repair_report": repair_report,
        "windows": window_reports,
        "head_after": current_directory_head(&config.edge_base_url, &config.experiment_id)?,
        "catchup": fetch_json(&format!("{}/metrics/catchup/{}", config.edge_base_url, config.experiment_id), 30)?,
        "live_latest": fetch_json(&format!("{}/metrics/live/latest", config.edge_base_url), 30)?,
        "leaderboard": fetch_json(&format!("{}/leaderboard/signed", config.edge_base_url), 30)?,
    });
    if let Some(parent) = config.output_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&config.output_json, serde_json::to_string_pretty(&summary)?)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn run_windows(
    config: &NativeCanaryConfig,
    trainer_bundle: &Path,
    probe_storage: &Path,
    head_before: &Value,
    initialize_head_on_start: bool,
) -> Result<Vec<Value>> {
    let mut previous_head = head_before.clone();
    let mut reports = Vec::new();
    for window_index in 0..config.windows {
        let report_path = config
            .artifact_dir
            .join(format!("train-window-{}.json", window_index + 1));
        let storage = config
            .artifact_dir
            .join(format!("trainer-window-{}-storage", window_index + 1));
        let mut command = vec![
            config.binary.clone(),
            "train-window-once".to_owned(),
            "--experiment-kind".to_owned(),
            config.experiment_kind.clone(),
            "--backend".to_owned(),
            config.backend.clone(),
            "--edge-url".to_owned(),
            config.edge_base_url.clone(),
            "--auth-bundle".to_owned(),
            trainer_bundle.display().to_string(),
            "--initialize-head-on-start".to_owned(),
            initialize_head_on_start.to_string(),
            "--restore-head-on-start".to_owned(),
            "true".to_owned(),
            "--head-sync-timeout-secs".to_owned(),
            config.head_sync_timeout_secs.to_string(),
            "--serve-after-publish-secs".to_owned(),
            config.serve_after_publish_secs.to_string(),
            "--require-head-advanced".to_owned(),
            "--output".to_owned(),
            report_path.display().to_string(),
            "--output-format".to_owned(),
            "json".to_owned(),
        ];
        if config.mirror_live_head_to_edge {
            command.push("--mirror-live-head-to-edge".to_owned());
        }
        append_training_overrides(config, &mut command);
        if config.settle_diffusion {
            command.extend([
                "--settle-diffusion".to_owned(),
                "--diffusion-settle-passes".to_owned(),
                config.diffusion_settle_passes.to_string(),
            ]);
        }
        run_native(
            command,
            &storage,
            config.command_timeout_secs,
            &config
                .artifact_dir
                .join(format!("train-window-{}.log", window_index + 1)),
        )?;
        let train_report: Value = serde_json::from_slice(&fs::read(&report_path)?)?;
        let train_signal = assert_train_report(&train_report)?;
        let published_head_id = train_report
            .get("published_head_id")
            .and_then(Value::as_str)
            .filter(|head_id| !head_id.is_empty())
            .with_context(|| {
                format!("native trainer did not report published_head_id: {train_report}")
            })?;
        let (published_head_p2p_signal, published_head_p2p_wait_secs) = wait_for_p2p_head(
            &config.binary,
            &config.edge_base_url,
            published_head_id,
            probe_storage,
            &config.artifact_dir.join(format!(
                "p2p-window-{}-published-head",
                window_index + 1
            )),
            config.p2p_timeout_secs,
        )
        .with_context(|| {
            format!(
                "published native training head {published_head_id} did not become visible through the p2p bootstrap before edge canonical promotion"
            )
        })?;
        let previous_head_id = previous_head
            .get("head_id")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let (advanced_head, wait_secs) = wait_for_head_advance(
            &config.edge_base_url,
            &config.experiment_id,
            previous_head_id.as_deref(),
            config.canonical_timeout_secs,
        )?;
        let canonical_signal = assert_canonical_signal(
            &previous_head,
            &advanced_head,
            config.require_canonical_loss_non_regression,
        )?;
        let (p2p_signal, p2p_wait_secs) = wait_for_p2p_head(
            &config.binary,
            &config.edge_base_url,
            advanced_head
                .get("head_id")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            probe_storage,
            &config
                .artifact_dir
                .join(format!("p2p-window-{}", window_index + 1)),
            config.p2p_timeout_secs,
        )?;
        let head_provider_signal = assert_head_provider_signal(
            &advanced_head,
            &p2p_signal,
            config.require_edge_head_provider,
        )?;
        reports.push(json!({
            "window_index": window_index + 1,
            "head_before": previous_head,
            "train_report": train_report,
            "train_signal": train_signal,
            "head_after": advanced_head,
            "canonical_wait_secs": wait_secs,
            "canonical_signal": canonical_signal,
            "published_head_p2p_wait_secs": published_head_p2p_wait_secs,
            "published_head_p2p_signal": published_head_p2p_signal,
            "p2p_wait_secs": p2p_wait_secs,
            "p2p_signal": p2p_signal,
            "head_provider_signal": head_provider_signal,
        }));
        previous_head = reports
            .last()
            .and_then(|report| report.get("head_after"))
            .cloned()
            .unwrap_or(Value::Null);
    }
    Ok(reports)
}

struct NativeCanaryConfig {
    binary: String,
    edge_base_url: String,
    experiment_kind: String,
    experiment_id: String,
    backend: String,
    principal_id: String,
    validator_principal_id: String,
    trusted_callback_token: String,
    windows: usize,
    head_sync_timeout_secs: u64,
    settle_diffusion: bool,
    diffusion_settle_passes: u64,
    serve_after_publish_secs: u64,
    start_local_validator: bool,
    mirror_live_head_to_edge: bool,
    require_edge_head_provider: bool,
    repair_current_head_to_visible_root: bool,
    require_canonical_loss_non_regression: bool,
    training_batch_size: Option<u64>,
    training_max_iters: Option<u64>,
    evaluation_max_batches: Option<u64>,
    command_timeout_secs: u64,
    canonical_timeout_secs: u64,
    p2p_timeout_secs: u64,
    artifact_dir: PathBuf,
    output_json: PathBuf,
}

impl NativeCanaryConfig {
    fn from_env() -> Result<Self> {
        let artifact_dir = PathBuf::from(env_or(
            "BURN_DRAGON_NATIVE_CANARY_ARTIFACT_DIR",
            "/tmp/burn-dragon-native-canary",
        ));
        Ok(Self {
            binary: env_or(
                "BURN_DRAGON_NATIVE_CANARY_BINARY",
                "target/debug/burn_dragon_p2p_native",
            ),
            edge_base_url: env_or(
                "BURN_DRAGON_NATIVE_CANARY_EDGE_BASE_URL",
                "https://edge.dragon.aberration.technology",
            )
            .trim_end_matches('/')
            .to_owned(),
            experiment_kind: env_or("BURN_DRAGON_NATIVE_CANARY_EXPERIMENT_KIND", "nca"),
            experiment_id: env_or(
                "BURN_DRAGON_NATIVE_CANARY_EXPERIMENT_ID",
                "nca-prepretraining",
            ),
            backend: env_or("BURN_DRAGON_NATIVE_CANARY_BACKEND", "cpu"),
            principal_id: env_or(
                "BURN_DRAGON_NATIVE_CANARY_PRINCIPAL_ID",
                "native-canary-mainnet-nca",
            ),
            validator_principal_id: env_or(
                "BURN_DRAGON_NATIVE_CANARY_VALIDATOR_PRINCIPAL_ID",
                &format!(
                    "{}-validator",
                    env_or(
                        "BURN_DRAGON_NATIVE_CANARY_PRINCIPAL_ID",
                        "native-canary-mainnet-nca"
                    )
                ),
            ),
            trusted_callback_token: required_env("BURN_DRAGON_NATIVE_CANARY_CALLBACK_TOKEN")?,
            windows: parse_env("BURN_DRAGON_NATIVE_CANARY_WINDOWS", 1)?,
            head_sync_timeout_secs: parse_env(
                "BURN_DRAGON_NATIVE_CANARY_HEAD_SYNC_TIMEOUT_SECS",
                300,
            )?,
            settle_diffusion: env_bool("BURN_DRAGON_NATIVE_CANARY_SETTLE_DIFFUSION", true)?,
            diffusion_settle_passes: parse_env(
                "BURN_DRAGON_NATIVE_CANARY_DIFFUSION_SETTLE_PASSES",
                2,
            )?,
            serve_after_publish_secs: parse_env(
                "BURN_DRAGON_NATIVE_CANARY_SERVE_AFTER_PUBLISH_SECS",
                45,
            )?,
            start_local_validator: env_bool("BURN_DRAGON_NATIVE_CANARY_START_VALIDATOR", false)?,
            mirror_live_head_to_edge: env_bool(
                "BURN_DRAGON_NATIVE_CANARY_MIRROR_LIVE_HEAD_TO_EDGE",
                true,
            )?,
            require_edge_head_provider: env_bool(
                "BURN_DRAGON_NATIVE_CANARY_REQUIRE_EDGE_HEAD_PROVIDER",
                true,
            )?,
            repair_current_head_to_visible_root: env_bool(
                "BURN_DRAGON_NATIVE_CANARY_REPAIR_CURRENT_HEAD_TO_VISIBLE_ROOT",
                false,
            )?,
            require_canonical_loss_non_regression: env_bool(
                "BURN_DRAGON_NATIVE_CANARY_REQUIRE_CANONICAL_LOSS_NON_REGRESSION",
                false,
            )?,
            training_batch_size: optional_positive_env(
                "BURN_DRAGON_NATIVE_CANARY_TRAINING_BATCH_SIZE",
            )?,
            training_max_iters: optional_positive_env(
                "BURN_DRAGON_NATIVE_CANARY_TRAINING_MAX_ITERS",
            )?,
            evaluation_max_batches: optional_positive_env(
                "BURN_DRAGON_NATIVE_CANARY_EVALUATION_MAX_BATCHES",
            )?,
            command_timeout_secs: parse_env("BURN_DRAGON_NATIVE_CANARY_COMMAND_TIMEOUT_SECS", 900)?,
            canonical_timeout_secs: parse_env(
                "BURN_DRAGON_NATIVE_CANARY_CANONICAL_TIMEOUT_SECS",
                480,
            )?,
            p2p_timeout_secs: parse_env("BURN_DRAGON_NATIVE_CANARY_P2P_TIMEOUT_SECS", 300)?,
            output_json: PathBuf::from(env_or(
                "BURN_DRAGON_NATIVE_CANARY_OUTPUT_JSON",
                &artifact_dir
                    .join("native-canary-summary.json")
                    .display()
                    .to_string(),
            )),
            artifact_dir,
        })
    }
}

fn repair_current_head_to_visible_root(
    config: &NativeCanaryConfig,
    trainer_bundle: &Path,
    storage_root: &Path,
) -> Result<Value> {
    let log_path = config.artifact_dir.join("repair-current-head.log");
    run_native(
        vec![
            config.binary.clone(),
            "admin-rollout-profile".to_owned(),
            "--experiment-kind".to_owned(),
            config.experiment_kind.clone(),
            "--backend".to_owned(),
            config.backend.clone(),
            "--edge-url".to_owned(),
            config.edge_base_url.clone(),
            "--auth-bundle".to_owned(),
            trainer_bundle.display().to_string(),
            "--reset-current-head-to-visible-root".to_owned(),
            "--output-format".to_owned(),
            "json".to_owned(),
        ],
        storage_root,
        config.command_timeout_secs,
        &log_path,
    )?;
    let output = fs::read_to_string(&log_path)?;
    serde_json::from_str(output.trim())
        .or_else(|_| {
            let start = output.find('{').unwrap_or(0);
            let end = output
                .rfind('}')
                .map(|index| index + 1)
                .unwrap_or(output.len());
            serde_json::from_str(&output[start..end])
        })
        .context("failed to parse repair current head report")
}

fn enroll_static_principal(
    config: &NativeCanaryConfig,
    principal_id: &str,
    principal_kind: &str,
    auth_bundle: &Path,
    storage_root: &Path,
    log_path: &Path,
    backend: &str,
) -> Result<()> {
    run_native(
        vec![
            config.binary.clone(),
            "enroll-static-principal".to_owned(),
            "--experiment-kind".to_owned(),
            config.experiment_kind.clone(),
            "--backend".to_owned(),
            backend.to_owned(),
            "--edge-url".to_owned(),
            config.edge_base_url.clone(),
            "--principal-id".to_owned(),
            principal_id.to_owned(),
            "--principal-hint".to_owned(),
            principal_id.to_owned(),
            "--principal-kind".to_owned(),
            principal_kind.to_owned(),
            "--trusted-callback-token".to_owned(),
            config.trusted_callback_token.clone(),
            "--auth-bundle-out".to_owned(),
            auth_bundle.display().to_string(),
            "--output-format".to_owned(),
            "json".to_owned(),
        ],
        storage_root,
        config.command_timeout_secs,
        log_path,
    )
}

fn start_validator(
    config: &NativeCanaryConfig,
    auth_bundle: &Path,
    storage_root: &Path,
    log_path: &Path,
    initialize_head_on_start: bool,
) -> Result<Child> {
    let mut command = vec![
        config.binary.clone(),
        "run-validator-daemon".to_owned(),
        "--experiment-kind".to_owned(),
        config.experiment_kind.clone(),
        "--backend".to_owned(),
        "cpu".to_owned(),
        "--edge-url".to_owned(),
        config.edge_base_url.clone(),
        "--auth-bundle".to_owned(),
        auth_bundle.display().to_string(),
        "--status-interval-secs".to_owned(),
        "10".to_owned(),
        "--validation-interval-millis".to_owned(),
        "500".to_owned(),
        "--initialize-head-on-start".to_owned(),
        initialize_head_on_start.to_string(),
        "--restore-head-on-start".to_owned(),
        "true".to_owned(),
    ];
    append_training_overrides(config, &mut command);
    if let Some(parent) = log_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let log = File::create(log_path)?;
    let stderr = log.try_clone()?;
    let mut child = Command::new(&command[0])
        .args(&command[1..])
        .env("BURN_DRAGON_P2P_NATIVE_STORAGE_ROOT", storage_root)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(stderr))
        .spawn()
        .with_context(|| format!("failed to start {}", command.join(" ")))?;
    thread::sleep(Duration::from_secs(5));
    if let Some(status) = child.try_wait()? {
        let tail = tail(log_path, 6000);
        bail!("validator exited early with {status}\n{tail}");
    }
    Ok(child)
}

fn run_native(
    command: Vec<String>,
    storage_root: &Path,
    timeout_secs: u64,
    stdout_path: &Path,
) -> Result<()> {
    if let Some(parent) = stdout_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let log = File::create(stdout_path)?;
    let stderr = log.try_clone()?;
    eprintln!("+ {}", command.join(" "));
    let mut child = Command::new(&command[0])
        .args(&command[1..])
        .env("BURN_DRAGON_P2P_NATIVE_STORAGE_ROOT", storage_root)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(stderr))
        .spawn()
        .with_context(|| format!("failed to start {}", command.join(" ")))?;
    let started = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            if status.success() {
                return Ok(());
            }
            bail!(
                "command failed with exit {status}: {}\n{}",
                command.join(" "),
                tail(stdout_path, 6000)
            );
        }
        if started.elapsed() >= Duration::from_secs(timeout_secs) {
            let _ = child.kill();
            let _ = child.wait();
            bail!(
                "command timed out after {timeout_secs}s: {}\n{}",
                command.join(" "),
                tail(stdout_path, 6000)
            );
        }
        thread::sleep(Duration::from_millis(500));
    }
}

fn append_training_overrides(config: &NativeCanaryConfig, command: &mut Vec<String>) {
    if let Some(value) = config.training_batch_size {
        command.extend(["--training-batch-size".to_owned(), value.to_string()]);
    }
    if let Some(value) = config.training_max_iters {
        command.extend(["--training-max-iters".to_owned(), value.to_string()]);
    }
    if let Some(value) = config.evaluation_max_batches {
        command.extend(["--evaluation-max-batches".to_owned(), value.to_string()]);
    }
}

fn current_directory_head(edge_base_url: &str, experiment_id: &str) -> Result<Value> {
    let signed = fetch_json(&format!("{edge_base_url}/directory/signed"), 30)?;
    let entries = signed
        .pointer("/payload/payload/entries")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    for entry in entries {
        if entry.get("experiment_id").and_then(Value::as_str) == Some(experiment_id) {
            let head_id = entry.get("current_head_id").and_then(Value::as_str);
            let artifact = if let Some(head_id) = head_id {
                fetch_head_artifact(edge_base_url, head_id)?
            } else {
                json!({})
            };
            return Ok(json!({
                "head_id": head_id,
                "revision_id": entry.get("current_revision_id").cloned().unwrap_or(Value::Null),
                "workload_id": entry.get("workload_id").cloned().unwrap_or(Value::Null),
                "generated_at": signed.pointer("/payload/payload/generated_at").cloned().unwrap_or(Value::Null),
                "global_step": artifact.get("global_step").cloned().unwrap_or(Value::Null),
                "artifact_id": artifact.get("artifact_id").cloned().unwrap_or(Value::Null),
                "parent_head_id": artifact.get("parent_head_id").cloned().unwrap_or(Value::Null),
                "metrics": artifact.get("metrics").cloned().unwrap_or_else(|| json!({})),
                "provider_peer_ids": artifact.get("provider_peer_ids").cloned().unwrap_or_else(|| json!([])),
                "connected_provider_peer_ids": artifact.get("connected_provider_peer_ids").cloned().unwrap_or_else(|| json!([])),
                "available_profiles": artifact.get("available_profiles").cloned().unwrap_or_else(|| json!([])),
                "published_artifacts": artifact.get("published_artifacts").cloned().unwrap_or_else(|| json!([])),
            }));
        }
    }
    bail!("directory has no entry for experiment {experiment_id:?}")
}

fn fetch_head_artifact(edge_base_url: &str, head_id: &str) -> Result<Value> {
    let artifact = fetch_json(
        &format!(
            "{}/artifacts/heads/{}",
            edge_base_url,
            percent_encode(head_id)
        ),
        30,
    )?;
    let head = artifact.get("head").cloned().unwrap_or_else(|| json!({}));
    ensure!(
        head.get("head_id").and_then(Value::as_str) == Some(head_id),
        "head artifact mismatch: expected {head_id:?}, got {:?}",
        head.get("head_id")
    );
    Ok(json!({
        "head_id": head.get("head_id").cloned().unwrap_or(Value::Null),
        "parent_head_id": head.get("parent_head_id").cloned().unwrap_or(Value::Null),
        "artifact_id": head.get("artifact_id").cloned().unwrap_or(Value::Null),
        "global_step": head.get("global_step").cloned().unwrap_or(Value::Null),
        "metrics": head.get("metrics").cloned().unwrap_or_else(|| json!({})),
        "provider_peer_ids": artifact.get("provider_peer_ids").cloned().unwrap_or_else(|| json!([])),
        "connected_provider_peer_ids": artifact.get("connected_provider_peer_ids").cloned().unwrap_or_else(|| json!([])),
        "available_profiles": artifact.get("available_profiles").cloned().unwrap_or_else(|| json!([])),
        "published_artifacts": artifact.get("published_artifacts").cloned().unwrap_or_else(|| json!([])),
    }))
}

fn fetch_json(url: &str, timeout_secs: u64) -> Result<Value> {
    let attempts: u64 = parse_env("BURN_DRAGON_NATIVE_CANARY_HTTP_ATTEMPTS", 5)?;
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .build()?;
    let mut last_error = None;
    for attempt in 1..=attempts {
        match client
            .get(url)
            .send()
            .and_then(|response| response.error_for_status())
            .and_then(|response| response.json::<Value>())
        {
            Ok(value) => return Ok(value),
            Err(error) => {
                last_error = Some(error);
                if attempt < attempts {
                    thread::sleep(Duration::from_secs((2 * attempt).min(10)));
                }
            }
        }
    }
    bail!(
        "failed to fetch {url} after {attempts} attempts: {}",
        last_error
            .map(|error| error.to_string())
            .unwrap_or_else(|| "unknown error".to_owned())
    )
}

fn p2p_bootstrap_addresses(edge_base_url: &str) -> Result<Vec<String>> {
    let override_addrs = env_or("BURN_DRAGON_NATIVE_CANARY_P2P_BOOTSTRAP_ADDRS", "");
    if !override_addrs.is_empty() {
        return Ok(override_addrs
            .split(',')
            .map(str::trim)
            .filter(|address| !address.is_empty())
            .map(str::to_owned)
            .collect());
    }
    let url = reqwest::Url::parse(edge_base_url)?;
    let host = url
        .host_str()
        .ok_or_else(|| anyhow::anyhow!("edge URL has no hostname: {edge_base_url:?}"))?;
    let mut addresses = vec![
        format!("/dns4/{host}/tcp/4001"),
        format!("/dns4/{host}/udp/4001/quic-v1"),
    ];
    let socket_iter = (host, 4001).to_socket_addrs().unwrap_or_default();
    for socket in socket_iter {
        if let SocketAddr::V4(v4) = socket {
            addresses.push(format!("/ip4/{}/tcp/4001", v4.ip()));
            addresses.push(format!("/ip4/{}/udp/4001/quic-v1", v4.ip()));
        }
    }
    addresses.dedup();
    Ok(addresses)
}

fn probe_p2p_snapshot(
    binary: &str,
    edge_base_url: &str,
    storage_root: &Path,
    log_path: &Path,
    timeout_secs: u64,
) -> Result<Value> {
    let addresses = p2p_bootstrap_addresses(edge_base_url)?;
    let mut errors = Vec::new();
    for (index, address) in addresses.iter().enumerate() {
        let candidate_log_path = if addresses.len() > 1 {
            log_path.with_file_name(format!(
                "{}-addr{}{}",
                log_path
                    .file_stem()
                    .and_then(|value| value.to_str())
                    .unwrap_or("p2p-probe"),
                index + 1,
                log_path
                    .extension()
                    .and_then(|value| value.to_str())
                    .map(|ext| format!(".{ext}"))
                    .unwrap_or_default()
            ))
        } else {
            log_path.to_path_buf()
        };
        match probe_p2p_snapshot_address(
            binary,
            address,
            storage_root,
            &candidate_log_path,
            timeout_secs,
        ) {
            Ok(value) => return Ok(value),
            Err(error) => errors.push(format!("{address}: {error:#}")),
        }
    }
    bail!(
        "p2p bootstrap probes failed across {} addresses:\n{}",
        addresses.len(),
        errors
            .iter()
            .rev()
            .take(4)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    )
}

fn probe_p2p_snapshot_address(
    binary: &str,
    address: &str,
    storage_root: &Path,
    log_path: &Path,
    timeout_secs: u64,
) -> Result<Value> {
    run_native(
        vec![
            binary.to_owned(),
            "probe-swarm".to_owned(),
            "--address".to_owned(),
            address.to_owned(),
            "--timeout-secs".to_owned(),
            "30".to_owned(),
            "--max-events".to_owned(),
            "96".to_owned(),
            "--fetch-snapshot".to_owned(),
            "--snapshot-timeout-secs".to_owned(),
            "15".to_owned(),
            "--output-format".to_owned(),
            "json".to_owned(),
        ],
        storage_root,
        timeout_secs,
        log_path,
    )?;
    let probe = parse_json_stdout(log_path)?;
    ensure!(
        probe
            .get("connected")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        "p2p bootstrap probe did not connect: {:?}",
        p2p_probe_summary(&probe)
    );
    ensure!(
        probe.get("snapshot_error").is_none_or(Value::is_null),
        "p2p bootstrap snapshot fetch failed: {:?}",
        p2p_probe_summary(&probe)
    );
    ensure!(
        probe.get("snapshot").is_some(),
        "p2p bootstrap probe did not return a snapshot: {probe}"
    );
    Ok(probe)
}

fn wait_for_p2p_head(
    binary: &str,
    edge_base_url: &str,
    head_id: &str,
    storage_root: &Path,
    log_dir: &Path,
    timeout_secs: u64,
) -> Result<(Value, f64)> {
    fs::create_dir_all(log_dir)?;
    let started = Instant::now();
    let mut attempt = 1;
    let mut last_summary = None;
    let mut last_error = None;
    while started.elapsed() < Duration::from_secs(timeout_secs) {
        match probe_p2p_snapshot(
            binary,
            edge_base_url,
            storage_root,
            &log_dir.join(format!("p2p-probe-{attempt}.log")),
            60,
        ) {
            Ok(probe) => {
                let summary = p2p_probe_summary(&probe);
                let head_ids = summary
                    .get("head_ids")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                if head_ids.iter().any(|value| value.as_str() == Some(head_id)) {
                    return Ok((summary, started.elapsed().as_secs_f64()));
                }
                last_summary = Some(summary);
                last_error = None;
            }
            Err(error) => last_error = Some(error.to_string()),
        }
        attempt += 1;
        thread::sleep(Duration::from_secs(5));
    }
    bail!(
        "p2p bootstrap snapshot did not advertise canonical head {head_id} within {timeout_secs}s; last={last_summary:?}; last_error={last_error:?}"
    )
}

fn p2p_probe_summary(probe: &Value) -> Value {
    let snapshot = probe.get("snapshot").cloned().unwrap_or_else(|| json!({}));
    let heads = snapshot
        .get("heads")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    json!({
        "connected": probe.get("connected").cloned().unwrap_or(Value::Null),
        "connected_peer_id": probe.get("connected_peer_id").cloned().unwrap_or(Value::Null),
        "address": probe.get("address").cloned().unwrap_or(Value::Null),
        "elapsed_millis": probe.get("elapsed_millis").cloned().unwrap_or(Value::Null),
        "snapshot_error": probe.get("snapshot_error").cloned().unwrap_or(Value::Null),
        "head_announcements": snapshot.get("head_announcements").cloned().unwrap_or(Value::Null),
        "directory_announcements": snapshot.get("directory_announcements").cloned().unwrap_or(Value::Null),
        "peer_directory_announcements": snapshot.get("peer_directory_announcements").cloned().unwrap_or(Value::Null),
        "merge_announcements": snapshot.get("merge_announcements").cloned().unwrap_or(Value::Null),
        "merge_window_announcements": snapshot.get("merge_window_announcements").cloned().unwrap_or(Value::Null),
        "update_announcements": snapshot.get("update_announcements").cloned().unwrap_or(Value::Null),
        "aggregate_proposal_announcements": snapshot.get("aggregate_proposal_announcements").cloned().unwrap_or(Value::Null),
        "reduction_certificate_announcements": snapshot.get("reduction_certificate_announcements").cloned().unwrap_or(Value::Null),
        "validation_quorum_announcements": snapshot.get("validation_quorum_announcements").cloned().unwrap_or(Value::Null),
        "trainer_promotion_attestation_announcements": snapshot.get("trainer_promotion_attestation_announcements").cloned().unwrap_or(Value::Null),
        "diffusion_promotion_certificate_announcements": snapshot.get("diffusion_promotion_certificate_announcements").cloned().unwrap_or(Value::Null),
        "head_ids": heads.iter().filter_map(|head| head.get("head_id").and_then(Value::as_str)).collect::<Vec<_>>(),
    })
}

fn assert_head_provider_signal(
    head: &Value,
    p2p_signal: &Value,
    require_edge_provider: bool,
) -> Result<Value> {
    let head_id = head
        .get("head_id")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let providers = head
        .get("provider_peer_ids")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|value| value.as_str().map(str::to_owned))
        .collect::<Vec<_>>();
    ensure!(
        !providers.is_empty(),
        "head {head_id} has no artifact provider peers: {head}"
    );
    let edge_peer_id = p2p_signal
        .get("connected_peer_id")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let edge_provider = !edge_peer_id.is_empty() && providers.iter().any(|id| id == edge_peer_id);
    if require_edge_provider && !edge_provider {
        bail!(
            "canonical head is visible over p2p but is not edge-backed for fresh restores: head={head_id} edge_peer_id={edge_peer_id} providers={providers:?}"
        );
    }
    Ok(json!({
        "head_id": head_id,
        "edge_peer_id": edge_peer_id,
        "provider_peer_ids": providers,
        "connected_provider_peer_ids": head.get("connected_provider_peer_ids").cloned().unwrap_or_else(|| json!([])),
        "available_profiles": head.get("available_profiles").cloned().unwrap_or_else(|| json!([])),
        "published_artifacts": head.get("published_artifacts").cloned().unwrap_or_else(|| json!([])),
        "edge_provider": edge_provider,
    }))
}

fn wait_for_head_advance(
    edge_base_url: &str,
    experiment_id: &str,
    previous_head_id: Option<&str>,
    timeout_secs: u64,
) -> Result<(Value, f64)> {
    let started = Instant::now();
    let mut last_head = None;
    let mut last_error = None;
    while started.elapsed() < Duration::from_secs(timeout_secs) {
        match current_directory_head(edge_base_url, experiment_id) {
            Ok(head) => {
                let head_id = head.get("head_id").and_then(Value::as_str);
                if head_id.is_some() && head_id != previous_head_id {
                    return Ok((head, started.elapsed().as_secs_f64()));
                }
                last_head = Some(head);
                last_error = None;
            }
            Err(error) => last_error = Some(error.to_string()),
        }
        thread::sleep(Duration::from_secs(5));
    }
    bail!(
        "canonical head did not advance from {previous_head_id:?} within {timeout_secs}s; last={last_head:?}; last_error={last_error:?}"
    )
}

fn assert_train_report(report: &Value) -> Result<Value> {
    ensure!(
        report
            .get("can_train")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        "native trainer was not train-capable: {report}"
    );
    let base_global_step = number(report, "base_global_step").unwrap_or(0.0) as i64;
    let published_global_step = number(report, "published_global_step").unwrap_or(0.0) as i64;
    ensure!(
        published_global_step > base_global_step,
        "native trainer did not advance its local head: base={base_global_step} published={published_global_step}"
    );
    let metrics = report.get("metrics").cloned().unwrap_or_else(|| json!({}));
    let train_loss = require_metric_number(&metrics, &["train_loss", "loss"])?;
    let train_steps = require_metric_number(&metrics, &["train_steps", "batch_count"])?;
    let batch_count = metric_number(&metrics, &["batch_count"]).unwrap_or(train_steps);
    ensure!(
        train_steps > 0.0 && batch_count > 0.0,
        "native trainer reported no work: metrics={metrics}"
    );
    if let Some(settlement) = report.get("diffusion_settlement") {
        ensure!(
            settlement
                .get("enabled")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            "diffusion settlement was requested but not enabled: {settlement}"
        );
        ensure!(
            number(settlement, "passes_completed").unwrap_or(0.0) > 0.0,
            "diffusion settlement did not run: {settlement}"
        );
        ensure!(
            metric_number(settlement, &["update_announcements", "updates"]).unwrap_or(0.0) > 0.0,
            "diffusion settlement saw no trainer updates: {settlement}"
        );
    }
    Ok(json!({
        "train_loss": train_loss,
        "train_steps": train_steps,
        "batch_count": batch_count,
    }))
}

fn assert_canonical_signal(
    before: &Value,
    after: &Value,
    require_loss_non_regression: bool,
) -> Result<Value> {
    let before_step = number(before, "global_step").unwrap_or(0.0) as i64;
    let after_step = number(after, "global_step").unwrap_or(0.0) as i64;
    ensure!(
        after_step > before_step,
        "canonical head did not advance global step: before={before_step} after={after_step}"
    );
    let before_metrics = before.get("metrics").cloned().unwrap_or_else(|| json!({}));
    let after_metrics = after.get("metrics").cloned().unwrap_or_else(|| json!({}));
    let before_loss = metric_number(&before_metrics, &["train_loss", "loss"]);
    let after_loss = require_metric_number(&after_metrics, &["train_loss", "loss"])?;
    let comparable = comparable_loss_signal(&before_metrics, &after_metrics);
    if let Some((metric, before_value, after_value)) = comparable.as_ref()
        && require_loss_non_regression
        && *after_value > *before_value + 1e-6
    {
        bail!(
            "canonical loss regressed after native training window: metric={metric} before={before_value} after={after_value}"
        );
    }
    Ok(json!({
        "canonical_loss_before": before_loss,
        "canonical_loss_after": after_loss,
        "canonical_loss_delta": before_loss.map(|loss| after_loss - loss),
        "canonical_loss_improved": comparable.as_ref().map(|(_, before, after)| after <= before),
        "canonical_loss_metric": comparable.as_ref().map(|(metric, _, _)| metric),
        "comparable_loss_before": comparable.as_ref().map(|(_, before, _)| before),
        "comparable_loss_after": comparable.as_ref().map(|(_, _, after)| after),
    }))
}

fn comparable_loss_signal(before: &Value, after: &Value) -> Option<(String, f64, f64)> {
    for key in ["train_loss", "loss"] {
        if let (Some(before_value), Some(after_value)) = (number(before, key), number(after, key)) {
            return Some((key.to_owned(), before_value, after_value));
        }
    }
    None
}

fn require_metric_number(metrics: &Value, keys: &[&str]) -> Result<f64> {
    metric_number(metrics, keys)
        .with_context(|| format!("missing finite metric; expected one of {}", keys.join(", ")))
}

fn metric_number(metrics: &Value, keys: &[&str]) -> Option<f64> {
    keys.iter().find_map(|key| number(metrics, key))
}

fn number(value: &Value, key: &str) -> Option<f64> {
    let number = value.get(key)?.as_f64()?;
    number.is_finite().then_some(number)
}

fn parse_json_stdout(path: &Path) -> Result<Value> {
    let text = fs::read_to_string(path)?;
    let start = text.find('{').ok_or_else(|| {
        anyhow::anyhow!(
            "command output did not contain a JSON object: {}",
            tail(path, 2000)
        )
    })?;
    let end = text.rfind('}').ok_or_else(|| {
        anyhow::anyhow!(
            "command output did not contain a JSON object: {}",
            tail(path, 2000)
        )
    })?;
    serde_json::from_str(&text[start..=end]).context("failed to parse command JSON output")
}

fn stop_child(child: &mut Child) {
    if matches!(child.try_wait(), Ok(Some(_))) {
        return;
    }
    let _ = child.kill();
    let _ = child.wait();
}

fn tail(path: &Path, bytes: usize) -> String {
    fs::read(path)
        .ok()
        .map(|mut data| {
            if data.len() > bytes {
                data = data.split_off(data.len() - bytes);
            }
            String::from_utf8_lossy(&data).to_string()
        })
        .unwrap_or_default()
}

fn percent_encode(value: &str) -> String {
    value
        .bytes()
        .flat_map(|byte| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                vec![byte as char]
            }
            _ => {
                let mut encoded = String::new();
                let _ = write!(&mut encoded, "%{byte:02X}");
                encoded.chars().collect()
            }
        })
        .collect()
}

fn required_env(name: &str) -> Result<String> {
    let value = std::env::var(name)
        .with_context(|| format!("missing required environment variable {name}"))?;
    ensure!(
        !value.is_empty(),
        "missing required environment variable {name}"
    );
    Ok(value)
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name)
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| default.to_owned())
}

fn env_bool(name: &str, default: bool) -> Result<bool> {
    let raw = env_or(name, if default { "true" } else { "false" });
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => bail!("environment variable {name} must be boolean, got {raw:?}"),
    }
}

fn parse_env<T>(name: &str, default: T) -> Result<T>
where
    T: std::str::FromStr + ToString,
    T::Err: std::fmt::Display,
{
    env_or(name, &default.to_string())
        .parse()
        .map_err(|error| anyhow::anyhow!("invalid {name}: {error}"))
}

fn optional_positive_env(name: &str) -> Result<Option<u64>> {
    let value = env_or(name, "");
    if value.trim().is_empty() {
        return Ok(None);
    }
    let parsed: u64 = value.parse().with_context(|| format!("invalid {name}"))?;
    ensure!(
        parsed > 0,
        "environment variable {name} must be > 0, got {value:?}"
    );
    Ok(Some(parsed))
}
