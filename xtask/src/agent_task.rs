use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, ensure};
use chrono::{DateTime, Utc};
use clap::{Args, Subcommand};
use serde_json::{Map, Value, json};

const DEFAULT_INTERVAL_SECS: u64 = 180;
const DEFAULT_DISCOVER_TIMEOUT_SECS: u64 = 150;
const DEFAULT_FAILED_LOG_LINES: usize = 40;
const DEFAULT_TAIL_LINES: usize = 40;

#[derive(Debug, Subcommand)]
pub enum AgentTaskCommand {
    /// Run a local command and emit only a final actionable summary.
    Run(RunArgs),
    /// Dispatch a GitHub workflow and optionally wait quietly for completion.
    GhDispatch(GhDispatchArgs),
    /// Wait quietly for an existing GitHub Actions run.
    GhWait(GhWaitArgs),
    /// List recent local agent tasks.
    Status(StatusArgs),
    /// Print a saved task summary.
    Summarize(SummarizeArgs),
    #[command(hide = true)]
    RunWorker(WorkerArgs),
    #[command(hide = true)]
    GhWaitWorker(WorkerArgs),
}

#[derive(Debug, Clone, Args)]
pub struct CommonTaskArgs {
    #[arg(long)]
    pub state_root: Option<PathBuf>,
    #[arg(long)]
    pub task_id: Option<String>,
    #[arg(long)]
    pub cwd: Option<PathBuf>,
}

#[derive(Debug, Clone, Args)]
pub struct RunArgs {
    #[command(flatten)]
    pub common: CommonTaskArgs,
    #[arg(long)]
    pub label: Option<String>,
    #[arg(long, default_value_t = 0)]
    pub timeout_secs: u64,
    #[arg(long, default_value_t = 0)]
    pub stale_secs: u64,
    #[arg(long, default_value_t = DEFAULT_TAIL_LINES)]
    pub tail_lines: usize,
    #[arg(long)]
    pub detach: bool,
    #[arg(long)]
    pub wait: bool,
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    pub command: Vec<String>,
}

#[derive(Debug, Clone, Args)]
pub struct WaitOptions {
    #[arg(long, default_value_t = DEFAULT_INTERVAL_SECS)]
    pub interval_secs: u64,
    #[arg(long, default_value_t = 0)]
    pub timeout_secs: u64,
    #[arg(long, default_value_t = 0)]
    pub stale_secs: u64,
    #[arg(long, default_value_t = DEFAULT_FAILED_LOG_LINES)]
    pub failed_log_lines: usize,
    #[arg(long)]
    pub exit_status: bool,
    #[arg(long)]
    pub detach: bool,
    #[arg(long)]
    pub wait: bool,
    #[arg(long, hide = true)]
    pub watch: bool,
}

#[derive(Debug, Clone, Args)]
pub struct GhDispatchArgs {
    #[command(flatten)]
    pub common: CommonTaskArgs,
    #[command(flatten)]
    pub wait_options: WaitOptions,
    #[arg(long)]
    pub repo: String,
    #[arg(long)]
    pub workflow: String,
    #[arg(long = "ref")]
    pub ref_name: String,
    #[arg(long)]
    pub label: Option<String>,
    #[arg(long = "input")]
    pub inputs: Vec<String>,
    #[arg(long, default_value = "agent_task_id")]
    pub agent_task_input: String,
    #[arg(long, default_value_t = DEFAULT_DISCOVER_TIMEOUT_SECS)]
    pub discover_timeout_secs: u64,
    #[arg(long, default_value_t = DEFAULT_TAIL_LINES)]
    pub tail_lines: usize,
}

#[derive(Debug, Clone, Args)]
pub struct GhWaitArgs {
    #[command(flatten)]
    pub common: CommonTaskArgs,
    #[command(flatten)]
    pub wait_options: WaitOptions,
    #[arg(long)]
    pub repo: String,
    #[arg(long)]
    pub run_id: String,
    #[arg(long)]
    pub label: Option<String>,
}

#[derive(Debug, Clone, Args)]
pub struct StatusArgs {
    #[command(flatten)]
    pub common: CommonTaskArgs,
    #[arg(long, default_value_t = 20)]
    pub limit: usize,
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Clone, Args)]
pub struct SummarizeArgs {
    #[command(flatten)]
    pub common: CommonTaskArgs,
    pub task_id: Option<String>,
}

#[derive(Debug, Clone, Args)]
pub struct WorkerArgs {
    #[arg(long)]
    pub task_dir: PathBuf,
}

#[derive(Debug, Clone)]
struct RunSummary {
    workflow_name: String,
    display_title: String,
    status: String,
    conclusion: String,
    url: String,
    active_job: String,
    active_step: String,
    failed_job: String,
    failed_step: String,
    failed_log_tail: Vec<String>,
}

impl RunSummary {
    fn signature(&self) -> (&str, &str, &str, &str, &str, &str) {
        (
            &self.status,
            &self.conclusion,
            &self.active_job,
            &self.active_step,
            &self.failed_job,
            &self.failed_step,
        )
    }
}

pub fn run(command: AgentTaskCommand) -> Result<()> {
    let code = match command {
        AgentTaskCommand::Run(args) => command_run(args)?,
        AgentTaskCommand::GhDispatch(args) => command_gh_dispatch(args)?,
        AgentTaskCommand::GhWait(args) => command_gh_wait(args)?,
        AgentTaskCommand::Status(args) => command_status(args)?,
        AgentTaskCommand::Summarize(args) => command_summarize(args)?,
        AgentTaskCommand::RunWorker(args) => run_local_worker(&args.task_dir)?,
        AgentTaskCommand::GhWaitWorker(args) => wait_github_worker(&args.task_dir)?,
    };
    if code == 0 {
        Ok(())
    } else {
        std::process::exit(code);
    }
}

pub fn dispatch_pages_deploy_and_wait() -> Result<()> {
    required_env("GH_TOKEN")?;
    let repo = required_env("GITHUB_REPOSITORY")?;
    let ref_name = required_env("GITHUB_REF_NAME")?;
    let args = GhDispatchArgs {
        common: CommonTaskArgs::default(),
        wait_options: WaitOptions {
            interval_secs: env_u64("BURN_DRAGON_DEPLOY_PAGES_WATCH_INTERVAL_SECS", 180),
            exit_status: true,
            wait: true,
            ..WaitOptions::default()
        },
        repo,
        workflow: ".github/workflows/deploy-pages.yml".into(),
        ref_name,
        label: Some("deploy-pages".into()),
        inputs: vec![
            input_env("environment", "BURN_DRAGON_DEPLOY_PAGES_ENVIRONMENT")?,
            input_env("edge_base_url", "BURN_DRAGON_DEPLOY_PAGES_EDGE_BASE_URL")?,
            input_env(
                "selected_experiment_id",
                "BURN_DRAGON_DEPLOY_PAGES_EXPERIMENT_ID",
            )?,
            input_env(
                "selected_revision_id",
                "BURN_DRAGON_DEPLOY_PAGES_REVISION_ID",
            )?,
            input_env(
                "require_edge_auth",
                "BURN_DRAGON_DEPLOY_PAGES_REQUIRE_EDGE_AUTH",
            )?,
        ],
        agent_task_input: "agent_task_id".into(),
        discover_timeout_secs: DEFAULT_DISCOVER_TIMEOUT_SECS,
        tail_lines: DEFAULT_TAIL_LINES,
    };
    let code = command_gh_dispatch(args)?;
    if code == 0 {
        Ok(())
    } else {
        std::process::exit(code);
    }
}

pub fn dispatch_native_training_canary_and_wait() -> Result<()> {
    required_env("GH_TOKEN")?;
    let repo = required_env("GITHUB_REPOSITORY")?;
    let ref_name = required_env("GITHUB_REF_NAME")?;
    let args = GhDispatchArgs {
        common: CommonTaskArgs::default(),
        wait_options: WaitOptions {
            interval_secs: env_u64("BURN_DRAGON_NATIVE_CANARY_WATCH_INTERVAL_SECS", 60),
            exit_status: true,
            wait: true,
            ..WaitOptions::default()
        },
        repo,
        workflow: ".github/workflows/live-native-training-canary.yml".into(),
        ref_name,
        label: Some("live-native-training-canary".into()),
        inputs: vec![
            input_env("environment", "BURN_DRAGON_NATIVE_CANARY_ENVIRONMENT")?,
            input_env("edge_base_url", "BURN_DRAGON_NATIVE_CANARY_EDGE_BASE_URL")?,
            input_env(
                "experiment_kind",
                "BURN_DRAGON_NATIVE_CANARY_EXPERIMENT_KIND",
            )?,
            input_env("experiment_id", "BURN_DRAGON_NATIVE_CANARY_EXPERIMENT_ID")?,
            input_env("backend", "BURN_DRAGON_NATIVE_CANARY_BACKEND")?,
            input_env("principal_id", "BURN_DRAGON_NATIVE_CANARY_PRINCIPAL_ID")?,
            input_env("windows", "BURN_DRAGON_NATIVE_CANARY_WINDOWS")?,
            input_env_default(
                "settle_diffusion",
                "BURN_DRAGON_NATIVE_CANARY_SETTLE_DIFFUSION",
                "true",
            ),
            input_env_default(
                "diffusion_settle_passes",
                "BURN_DRAGON_NATIVE_CANARY_DIFFUSION_SETTLE_PASSES",
                "2",
            ),
            input_env_default(
                "serve_after_publish_secs",
                "BURN_DRAGON_NATIVE_CANARY_SERVE_AFTER_PUBLISH_SECS",
                "45",
            ),
            input_env_default(
                "command_timeout_secs",
                "BURN_DRAGON_NATIVE_CANARY_COMMAND_TIMEOUT_SECS",
                "900",
            ),
            input_env_default(
                "start_validator",
                "BURN_DRAGON_NATIVE_CANARY_START_VALIDATOR",
                "false",
            ),
            input_env_default(
                "training_batch_size",
                "BURN_DRAGON_NATIVE_CANARY_TRAINING_BATCH_SIZE",
                "1",
            ),
            input_env_default(
                "training_max_iters",
                "BURN_DRAGON_NATIVE_CANARY_TRAINING_MAX_ITERS",
                "2",
            ),
            input_env_default(
                "evaluation_max_batches",
                "BURN_DRAGON_NATIVE_CANARY_EVALUATION_MAX_BATCHES",
                "1",
            ),
            input_env_default(
                "require_canonical_loss_non_regression",
                "BURN_DRAGON_NATIVE_CANARY_REQUIRE_CANONICAL_LOSS_NON_REGRESSION",
                "false",
            ),
        ],
        agent_task_input: "agent_task_id".into(),
        discover_timeout_secs: DEFAULT_DISCOVER_TIMEOUT_SECS,
        tail_lines: DEFAULT_TAIL_LINES,
    };
    let code = command_gh_dispatch(args)?;
    if code == 0 {
        Ok(())
    } else {
        std::process::exit(code);
    }
}

fn command_run(args: RunArgs) -> Result<i32> {
    let cwd = resolve_cwd(&args.common)?;
    let command = parse_command(args.command)?;
    let label = args.label.clone().unwrap_or_else(|| {
        Path::new(&command[0])
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into()
    });
    let (task_dir, mut task) = create_task(
        &args.common,
        "local",
        &label,
        &cwd,
        json!({
            "command": command,
            "timeout_secs": args.timeout_secs,
            "stale_secs": args.stale_secs,
            "tail_lines": args.tail_lines,
        }),
    )?;
    println!("agent task {} started: {}", task_id(&task), label);
    println!("task dir: {}", string_field(&task, "task_dir"));
    task.insert("status".into(), json!("queued"));
    write_task(&task_dir, &task)?;
    if args.detach {
        return spawn_worker(&task_dir, "run-worker");
    }
    run_local_worker(&task_dir)
}

fn command_gh_dispatch(args: GhDispatchArgs) -> Result<i32> {
    let cwd = resolve_cwd(&args.common)?;
    let label = args.label.clone().unwrap_or_else(|| {
        format!(
            "gh-{}",
            slugify(
                Path::new(&args.workflow)
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .as_ref()
            )
        )
    });
    let (task_dir, mut task) = create_task(
        &args.common,
        "github-workflow",
        &label,
        &cwd,
        json!({
            "repo": args.repo,
            "workflow": args.workflow,
            "ref": args.ref_name,
            "interval_secs": normalized_interval(args.wait_options.interval_secs),
            "timeout_secs": args.wait_options.timeout_secs,
            "stale_secs": args.wait_options.stale_secs,
            "failed_log_lines": args.wait_options.failed_log_lines,
            "exit_status": args.wait_options.exit_status,
        }),
    )?;
    let task_id = task_id(&task);
    let mut inputs = parse_inputs(&args.inputs)?;
    if !args.agent_task_input.is_empty()
        && !inputs
            .iter()
            .any(|(key, _)| key.as_str() == args.agent_task_input.as_str())
    {
        inputs.insert(args.agent_task_input.clone(), task_id.clone());
    }
    let dispatch_started_at = Utc::now();
    let mut command = vec![
        "workflow".to_owned(),
        "run".to_owned(),
        args.workflow.clone(),
        "--repo".to_owned(),
        args.repo.clone(),
        "--ref".to_owned(),
        args.ref_name.clone(),
    ];
    for (key, value) in &inputs {
        command.push("-f".into());
        command.push(format!("{key}={value}"));
    }
    task.insert("inputs".into(), json!(inputs));
    task.insert("dispatch_command".into(), json!(command));
    write_task(&task_dir, &task)?;
    println!("agent task {task_id} dispatching: {}", args.workflow);
    let dispatch = run_capture("gh", string_vec(&task, "dispatch_command")?, Some(&cwd))?;
    fs::write(string_field(&task, "stdout_path"), &dispatch.stdout)?;
    fs::write(string_field(&task, "stderr_path"), &dispatch.stderr)?;
    if !dispatch.status.success() {
        task.insert("status".into(), json!("completed"));
        task.insert("conclusion".into(), json!("failure"));
        task.insert(
            "exit_code".into(),
            json!(dispatch.status.code().unwrap_or(1)),
        );
        task.insert(
            "stderr_tail".into(),
            json!(tail_lines(
                &PathBuf::from(string_field(&task, "stderr_path")),
                args.tail_lines
            )),
        );
        return finalize_task(&task_dir, task);
    }
    let run = discover_github_run(
        &args.repo,
        &args.workflow,
        &task_id,
        &args.ref_name,
        dispatch_started_at,
        args.discover_timeout_secs,
    )?
    .with_context(|| {
        format!(
            "failed to discover {} run dispatched for ref {}",
            args.workflow, args.ref_name
        )
    })?;
    let run_id = run
        .get("databaseId")
        .and_then(Value::as_i64)
        .unwrap_or_default()
        .to_string();
    task.insert("run_id".into(), json!(run_id));
    task.insert(
        "run_url".into(),
        json!(run.get("url").and_then(Value::as_str).unwrap_or_default()),
    );
    task.insert(
        "github_status".into(),
        json!(
            run.get("status")
                .and_then(Value::as_str)
                .unwrap_or_default()
        ),
    );
    write_task(&task_dir, &task)?;
    write_github_output(&task)?;
    println!(
        "github run {} dispatched for task {}",
        string_field(&task, "run_id"),
        task_id
    );
    if args.wait_options.detach {
        return spawn_worker(&task_dir, "gh-wait-worker");
    }
    if args.wait_options.wait || args.wait_options.watch {
        return wait_github_worker(&task_dir);
    }
    fs::write(
        string_field(&task, "summary_path"),
        render_task_summary(&task),
    )?;
    Ok(0)
}

fn command_gh_wait(args: GhWaitArgs) -> Result<i32> {
    let cwd = resolve_cwd(&args.common)?;
    let label = args
        .label
        .clone()
        .unwrap_or_else(|| format!("gh-run-{}", args.run_id));
    let (task_dir, task) = create_task(
        &args.common,
        "github-run",
        &label,
        &cwd,
        json!({
            "repo": args.repo,
            "run_id": args.run_id,
            "interval_secs": normalized_interval(args.wait_options.interval_secs),
            "timeout_secs": args.wait_options.timeout_secs,
            "stale_secs": args.wait_options.stale_secs,
            "failed_log_lines": args.wait_options.failed_log_lines,
            "exit_status": args.wait_options.exit_status,
        }),
    )?;
    println!(
        "agent task {} waiting for github run {}",
        task_id(&task),
        string_field(&task, "run_id")
    );
    if args.wait_options.detach {
        return spawn_worker(&task_dir, "gh-wait-worker");
    }
    wait_github_worker(&task_dir)
}

fn command_status(args: StatusArgs) -> Result<i32> {
    let cwd = resolve_cwd(&args.common)?;
    let state_root = state_root(&args.common, &cwd)?;
    let mut task_paths = fs::read_dir(&state_root)
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.join("task.json").exists())
        .collect::<Vec<_>>();
    task_paths.sort_by_key(|path| {
        fs::metadata(path.join("task.json"))
            .and_then(|meta| meta.modified())
            .unwrap_or(UNIX_EPOCH)
    });
    task_paths.reverse();
    let tasks = task_paths
        .into_iter()
        .take(args.limit)
        .map(|path| read_task(&path))
        .collect::<Result<Vec<_>>>()?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&tasks)?);
    } else {
        for task in tasks {
            let detail = task
                .get("run_id")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
                .or_else(|| {
                    task.get("command").and_then(Value::as_array).map(|parts| {
                        parts
                            .iter()
                            .filter_map(Value::as_str)
                            .collect::<Vec<_>>()
                            .join(" ")
                    })
                })
                .unwrap_or_default();
            println!(
                "{}\t{}\t{}\t{}",
                task_id(&task),
                string_field(&task, "status"),
                string_field_or(&task, "conclusion", "n/a"),
                detail
            );
        }
    }
    Ok(0)
}

fn command_summarize(args: SummarizeArgs) -> Result<i32> {
    let cwd = resolve_cwd(&args.common)?;
    let state_root = state_root(&args.common, &cwd)?;
    let task_dir = resolve_task_dir(&state_root, args.task_id.as_deref())?;
    let task = read_task(&task_dir)?;
    let summary_path = PathBuf::from(string_field(&task, "summary_path"));
    if !summary_path.exists() {
        fs::write(&summary_path, render_task_summary(&task))?;
    }
    println!("{}", fs::read_to_string(summary_path)?.trim_end());
    Ok(0)
}

fn run_local_worker(task_dir: &Path) -> Result<i32> {
    let mut task = read_task(task_dir)?;
    let stdout_path = PathBuf::from(string_field(&task, "stdout_path"));
    let stderr_path = PathBuf::from(string_field(&task, "stderr_path"));
    let cwd = PathBuf::from(string_field(&task, "cwd"));
    let command = string_vec(&task, "command")?;
    let timeout_secs = u64_field(&task, "timeout_secs");
    let stale_secs = u64_field(&task, "stale_secs");
    let tail_limit = usize_field(&task, "tail_lines", DEFAULT_TAIL_LINES);
    let started = Instant::now();
    let mut last_activity = SystemTime::now();

    task.insert("status".into(), json!("running"));
    task.insert("started_at".into(), json!(utc_now()));
    write_task(task_dir, &task)?;

    let stdout = File::create(&stdout_path)?;
    let stderr = File::create(&stderr_path)?;
    let mut child = Command::new(&command[0])
        .args(&command[1..])
        .current_dir(&cwd)
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .spawn()
        .with_context(|| format!("failed to spawn {}", command[0]))?;
    task.insert("pid".into(), json!(child.id()));
    write_task(task_dir, &task)?;

    let mut forced_conclusion: Option<(&'static str, i32)> = None;
    loop {
        if child.try_wait()?.is_some() {
            break;
        }
        if let Some(activity) = newest_activity(&[stdout_path.as_path(), stderr_path.as_path()]) {
            if activity > last_activity {
                last_activity = activity;
            }
        }
        if timeout_secs > 0 && started.elapsed() > Duration::from_secs(timeout_secs) {
            let _ = child.kill();
            forced_conclusion = Some(("timeout", 124));
            break;
        }
        if stale_secs > 0
            && SystemTime::now()
                .duration_since(last_activity)
                .unwrap_or_default()
                > Duration::from_secs(stale_secs)
        {
            let _ = child.kill();
            forced_conclusion = Some(("stale", 124));
            break;
        }
        thread::sleep(Duration::from_millis(500));
    }

    let status = child.wait()?;
    let (conclusion, exit_code) = forced_conclusion.unwrap_or_else(|| {
        let code = status.code().unwrap_or(1);
        if code == 0 {
            ("success", 0)
        } else {
            ("failure", code)
        }
    });
    task.insert("status".into(), json!("completed"));
    task.insert("conclusion".into(), json!(conclusion));
    task.insert("exit_code".into(), json!(exit_code));
    task.insert("completed_at".into(), json!(utc_now()));
    task.insert(
        "duration_secs".into(),
        json!(started.elapsed().as_secs_f64()),
    );
    task.insert(
        "stdout_tail".into(),
        json!(tail_lines(&stdout_path, tail_limit)),
    );
    task.insert(
        "stderr_tail".into(),
        json!(tail_lines(&stderr_path, tail_limit)),
    );
    finalize_task(task_dir, task)
}

fn wait_github_worker(task_dir: &Path) -> Result<i32> {
    let mut task = read_task(task_dir)?;
    let repo = string_field(&task, "repo");
    let run_id = string_field(&task, "run_id");
    let interval_secs =
        normalized_interval(u64_field_or(&task, "interval_secs", DEFAULT_INTERVAL_SECS));
    let timeout_secs = u64_field(&task, "timeout_secs");
    let stale_secs = u64_field(&task, "stale_secs");
    let failed_log_lines = usize_field(&task, "failed_log_lines", DEFAULT_FAILED_LOG_LINES);
    let started = Instant::now();
    let mut last_signature_change = Instant::now();
    let mut last_signature: Option<(String, String, String, String, String, String)> = None;

    task.insert("status".into(), json!("running"));
    task.insert("started_at".into(), json!(utc_now()));
    write_task(task_dir, &task)?;

    loop {
        let summary = summarize_github_run(&repo, &run_id, failed_log_lines)?;
        let signature = summary.signature();
        let owned_signature = (
            signature.0.to_owned(),
            signature.1.to_owned(),
            signature.2.to_owned(),
            signature.3.to_owned(),
            signature.4.to_owned(),
            signature.5.to_owned(),
        );
        if last_signature.as_ref() != Some(&owned_signature) {
            last_signature = Some(owned_signature);
            last_signature_change = Instant::now();
        }
        task.insert("github_status".into(), json!(summary.status));
        task.insert("github_conclusion".into(), json!(summary.conclusion));
        task.insert("workflow_name".into(), json!(summary.workflow_name));
        task.insert("display_title".into(), json!(summary.display_title));
        task.insert("run_url".into(), json!(summary.url));
        task.insert("active_job".into(), json!(summary.active_job));
        task.insert("active_step".into(), json!(summary.active_step));
        task.insert("failed_job".into(), json!(summary.failed_job));
        task.insert("failed_step".into(), json!(summary.failed_step));
        task.insert("failed_log_tail".into(), json!(summary.failed_log_tail));
        write_task(task_dir, &task)?;

        if string_field(&task, "github_status") == "completed" {
            let conclusion = string_field_or(&task, "github_conclusion", "failure");
            task.insert("status".into(), json!("completed"));
            task.insert("conclusion".into(), json!(conclusion));
            task.insert(
                "exit_code".into(),
                json!(if string_field(&task, "github_conclusion") == "success" {
                    0
                } else {
                    1
                }),
            );
            task.insert("completed_at".into(), json!(utc_now()));
            task.insert(
                "duration_secs".into(),
                json!(started.elapsed().as_secs_f64()),
            );
            let code = finalize_task(task_dir, task)?;
            let refreshed = read_task(task_dir)?;
            if !bool_field(&refreshed, "exit_status")
                && string_field(&refreshed, "conclusion") != "success"
            {
                return Ok(0);
            }
            return Ok(code);
        }
        if timeout_secs > 0 && started.elapsed() > Duration::from_secs(timeout_secs) {
            task.insert("status".into(), json!("completed"));
            task.insert("conclusion".into(), json!("timeout"));
            task.insert("exit_code".into(), json!(124));
            task.insert("completed_at".into(), json!(utc_now()));
            task.insert(
                "duration_secs".into(),
                json!(started.elapsed().as_secs_f64()),
            );
            return finalize_task(task_dir, task);
        }
        if stale_secs > 0 && last_signature_change.elapsed() > Duration::from_secs(stale_secs) {
            task.insert("status".into(), json!("completed"));
            task.insert("conclusion".into(), json!("stale"));
            task.insert("exit_code".into(), json!(124));
            task.insert("completed_at".into(), json!(utc_now()));
            task.insert(
                "duration_secs".into(),
                json!(started.elapsed().as_secs_f64()),
            );
            return finalize_task(task_dir, task);
        }
        thread::sleep(Duration::from_secs(interval_secs));
    }
}

fn summarize_github_run(repo: &str, run_id: &str, failed_log_lines: usize) -> Result<RunSummary> {
    let run = gh_json(&[
        "run",
        "view",
        run_id,
        "--repo",
        repo,
        "--json",
        "conclusion,displayTitle,jobs,status,url,workflowName",
    ])?;
    let jobs = run
        .get("jobs")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let active = jobs
        .iter()
        .find(|job| {
            matches!(
                job.get("status").and_then(Value::as_str),
                Some("queued" | "in_progress")
            )
        })
        .cloned()
        .unwrap_or_else(|| json!({}));
    let failed = jobs
        .iter()
        .find(|job| {
            !matches!(
                job.get("conclusion").and_then(Value::as_str),
                None | Some("") | Some("success") | Some("skipped")
            )
        })
        .cloned()
        .unwrap_or_else(|| json!({}));
    let failed_log_tail = if failed.as_object().is_some_and(|object| !object.is_empty()) {
        interesting_log_lines(
            &gh_text(&["run", "view", run_id, "--repo", repo, "--log-failed"])?,
            failed_log_lines,
        )
    } else {
        Vec::new()
    };
    Ok(RunSummary {
        workflow_name: json_string(&run, "workflowName", "unknown"),
        display_title: json_string(&run, "displayTitle", ""),
        status: json_string(&run, "status", "unknown"),
        conclusion: json_string(&run, "conclusion", ""),
        url: json_string(&run, "url", ""),
        active_job: json_string(&active, "name", ""),
        active_step: active_step(&active),
        failed_job: json_string(&failed, "name", ""),
        failed_step: failed_step(&failed),
        failed_log_tail,
    })
}

fn discover_github_run(
    repo: &str,
    workflow: &str,
    task_id: &str,
    branch: &str,
    dispatch_started_at: DateTime<Utc>,
    timeout_secs: u64,
) -> Result<Option<Value>> {
    let deadline = Instant::now() + Duration::from_secs(timeout_secs);
    while Instant::now() < deadline {
        let runs = gh_json(&[
            "run",
            "list",
            "--repo",
            repo,
            "--workflow",
            workflow,
            "--limit",
            "20",
            "--json",
            "databaseId,createdAt,displayTitle,headBranch,status,url",
        ])?;
        let run_values = runs.as_array().cloned().unwrap_or_default();
        if let Some(run) = newest_matching_run(&run_values, |run| {
            run.get("displayTitle")
                .and_then(Value::as_str)
                .is_some_and(|title| title.contains(task_id))
        }) {
            return Ok(Some(run));
        }
        if let Some(run) = newest_matching_run(&run_values, |run| {
            if run.get("headBranch").and_then(Value::as_str) != Some(branch) {
                return false;
            }
            run.get("createdAt")
                .and_then(Value::as_str)
                .and_then(|value| value.parse::<DateTime<Utc>>().ok())
                .is_some_and(|created_at| created_at >= dispatch_started_at)
        }) {
            return Ok(Some(run));
        }
        thread::sleep(Duration::from_secs(5));
    }
    Ok(None)
}

fn newest_matching_run<F>(runs: &[Value], predicate: F) -> Option<Value>
where
    F: Fn(&Value) -> bool,
{
    let mut matches = runs
        .iter()
        .filter(|run| predicate(run))
        .cloned()
        .collect::<Vec<_>>();
    matches.sort_by_key(|run| {
        run.get("databaseId")
            .and_then(Value::as_i64)
            .unwrap_or_default()
    });
    matches.pop()
}

fn gh_json(args: &[&str]) -> Result<Value> {
    let mut last_error = None;
    for attempt in 0..3 {
        match run_capture("gh", args.iter().copied(), None) {
            Ok(output) if output.status.success() => {
                return serde_json::from_slice(&output.stdout)
                    .with_context(|| format!("failed to parse gh json for args {args:?}"));
            }
            Ok(output) => {
                last_error = Some(anyhow::anyhow!(
                    "gh {:?} failed: {}",
                    args,
                    String::from_utf8_lossy(&output.stderr)
                ));
            }
            Err(error) => last_error = Some(error),
        }
        if attempt < 2 {
            thread::sleep(Duration::from_secs(2 * (attempt + 1)));
        }
    }
    Err(last_error.unwrap_or_else(|| anyhow::anyhow!("gh command failed")))
}

fn gh_text(args: &[&str]) -> Result<String> {
    let output = run_capture("gh", args.iter().copied(), None)?;
    Ok(String::from_utf8_lossy(&output.stdout).into_owned()
        + &String::from_utf8_lossy(&output.stderr))
}

fn run_capture<I, S>(program: &str, args: I, cwd: Option<&Path>) -> Result<std::process::Output>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let mut command = Command::new(program);
    command.args(args);
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }
    command
        .output()
        .with_context(|| format!("failed to run {program}"))
}

fn create_task(
    common: &CommonTaskArgs,
    kind: &str,
    label: &str,
    cwd: &Path,
    extra: Value,
) -> Result<(PathBuf, Map<String, Value>)> {
    let state_root = state_root(common, cwd)?;
    let task_id = common.task_id.clone().unwrap_or_else(|| new_task_id(label));
    let task_dir = state_root.join(&task_id);
    fs::create_dir_all(&task_dir)?;
    let mut task = Map::new();
    task.insert("id".into(), json!(task_id));
    task.insert("kind".into(), json!(kind));
    task.insert("label".into(), json!(label));
    task.insert("status".into(), json!("queued"));
    task.insert("conclusion".into(), json!(""));
    task.insert("created_at".into(), json!(utc_now()));
    task.insert("updated_at".into(), json!(utc_now()));
    task.insert("cwd".into(), json!(abs(cwd)));
    task.insert("task_dir".into(), json!(abs(&task_dir)));
    task.insert(
        "summary_path".into(),
        json!(abs(&task_dir.join("summary.md"))),
    );
    task.insert(
        "stdout_path".into(),
        json!(abs(&task_dir.join("stdout.log"))),
    );
    task.insert(
        "stderr_path".into(),
        json!(abs(&task_dir.join("stderr.log"))),
    );
    if let Some(extra) = extra.as_object() {
        for (key, value) in extra {
            task.insert(key.clone(), value.clone());
        }
    }
    write_task(&task_dir, &task)?;
    Ok((task_dir, task))
}

fn finalize_task(task_dir: &Path, mut task: Map<String, Value>) -> Result<i32> {
    let summary = render_task_summary(&task);
    fs::write(string_field(&task, "summary_path"), &summary)?;
    task.insert("updated_at".into(), json!(utc_now()));
    write_task(task_dir, &task)?;
    append_step_summary(&summary)?;
    println!("{}", summary.trim_end());
    Ok(match string_field(&task, "conclusion").as_str() {
        "success" => 0,
        "timeout" | "stale" => 124,
        _ => i64_field(&task, "exit_code", 1) as i32,
    })
}

fn render_task_summary(task: &Map<String, Value>) -> String {
    let mut lines = vec![
        format!("# agent task {}", task_id(task)),
        String::new(),
        format!("- Label: `{}`", string_field(task, "label")),
        format!("- Kind: `{}`", string_field(task, "kind")),
        format!("- Status: `{}`", string_field(task, "status")),
        format!(
            "- Conclusion: `{}`",
            string_field_or(task, "conclusion", "n/a")
        ),
        format!("- Task dir: `{}`", string_field(task, "task_dir")),
    ];
    if let Some(value) = task.get("duration_secs").and_then(Value::as_f64) {
        lines.push(format!("- Duration: `{value:.1}s`"));
    }
    if task.get("exit_code").is_some() {
        lines.push(format!(
            "- Exit code: `{}`",
            i64_field(task, "exit_code", 0)
        ));
    }
    for (field, label) in [
        ("run_id", "GitHub run id"),
        ("run_url", "GitHub run URL"),
        ("repo", "Repository"),
        ("workflow", "Workflow"),
    ] {
        let value = string_field(task, field);
        if !value.is_empty() {
            lines.push(format!("- {label}: `{value}`"));
        }
    }
    if let Some(parts) = task.get("command").and_then(Value::as_array) {
        let command = parts
            .iter()
            .filter_map(Value::as_str)
            .map(shell_quote)
            .collect::<Vec<_>>()
            .join(" ");
        if !command.is_empty() {
            lines.push(format!("- Command: `{command}`"));
        }
    }
    let failed_job = string_field(task, "failed_job");
    if !failed_job.is_empty() {
        let failed_step = string_field(task, "failed_step");
        let failed = if failed_step.is_empty() {
            failed_job
        } else {
            format!("{failed_job} / {failed_step}")
        };
        lines.push(format!("- Failed: `{failed}`"));
    }
    let failed_log_tail = string_array(task, "failed_log_tail");
    let stderr_tail = string_array(task, "stderr_tail");
    if !failed_log_tail.is_empty() {
        lines.extend(["".into(), "```text".into()]);
        lines.extend(
            failed_log_tail
                .into_iter()
                .rev()
                .take(DEFAULT_TAIL_LINES)
                .collect::<Vec<_>>()
                .into_iter()
                .rev(),
        );
        lines.push("```".into());
    } else if !stderr_tail.is_empty() && string_field(task, "conclusion") != "success" {
        lines.extend(["".into(), "```text".into()]);
        lines.extend(
            stderr_tail
                .into_iter()
                .rev()
                .take(DEFAULT_TAIL_LINES)
                .collect::<Vec<_>>()
                .into_iter()
                .rev(),
        );
        lines.push("```".into());
    }
    lines.push(String::new());
    lines.join("\n")
}

fn spawn_worker(task_dir: &Path, worker: &str) -> Result<i32> {
    let mut task = read_task(task_dir)?;
    let broker_log = task_dir.join("broker.log");
    let log = File::create(&broker_log)?;
    let log_err = log.try_clone()?;
    let mut command = Command::new(std::env::current_exe()?);
    command
        .args(["agent-task", worker, "--task-dir"])
        .arg(task_dir)
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(log_err));
    start_new_session(&mut command);
    let child = command
        .spawn()
        .context("failed to spawn agent task worker")?;
    task.insert("broker_pid".into(), json!(child.id()));
    task.insert("broker_log_path".into(), json!(abs(&broker_log)));
    write_task(task_dir, &task)?;
    println!("agent task {} detached", task_id(&task));
    println!("summary: {}", string_field(&task, "summary_path"));
    Ok(0)
}

#[cfg(unix)]
fn start_new_session(command: &mut Command) {
    use std::os::unix::process::CommandExt;

    unsafe {
        command.pre_exec(|| {
            if setsid() == -1 {
                Err(std::io::Error::last_os_error())
            } else {
                Ok(())
            }
        });
    }
}

#[cfg(not(unix))]
fn start_new_session(_command: &mut Command) {}

#[cfg(unix)]
unsafe extern "C" {
    fn setsid() -> i32;
}

fn write_task(task_dir: &Path, task: &Map<String, Value>) -> Result<()> {
    let mut task = task.clone();
    task.insert("updated_at".into(), json!(utc_now()));
    let path = task_dir.join("task.json");
    let tmp = task_dir.join("task.json.tmp");
    fs::write(&tmp, serde_json::to_vec_pretty(&Value::Object(task))?)?;
    fs::rename(tmp, path)?;
    Ok(())
}

fn read_task(task_dir: &Path) -> Result<Map<String, Value>> {
    let value: Value =
        serde_json::from_slice(&fs::read(task_dir.join("task.json")).with_context(|| {
            format!("failed to read {}", task_dir.join("task.json").display())
        })?)?;
    value
        .as_object()
        .cloned()
        .context("task json is not an object")
}

fn resolve_task_dir(state_root: &Path, task_id: Option<&str>) -> Result<PathBuf> {
    if let Some(task_id) = task_id {
        let path = PathBuf::from(task_id);
        if path.join("task.json").exists() {
            return Ok(path);
        }
        return Ok(state_root.join(task_id));
    }
    let mut tasks = fs::read_dir(state_root)
        .with_context(|| format!("failed to list {}", state_root.display()))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.join("task.json").exists())
        .collect::<Vec<_>>();
    tasks.sort_by_key(|path| {
        fs::metadata(path.join("task.json"))
            .and_then(|meta| meta.modified())
            .unwrap_or(UNIX_EPOCH)
    });
    tasks.pop().context("no agent tasks found")
}

fn state_root(common: &CommonTaskArgs, cwd: &Path) -> Result<PathBuf> {
    Ok(common.state_root.clone().unwrap_or_else(|| {
        repo_root_for(cwd)
            .unwrap_or_else(|| cwd.to_path_buf())
            .join("target/agent-tasks")
    }))
}

fn repo_root_for(cwd: &Path) -> Option<PathBuf> {
    let output = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .current_dir(cwd)
        .output()
        .ok()?;
    if output.status.success() {
        let value = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        if !value.is_empty() {
            return Some(PathBuf::from(value));
        }
    }
    None
}

fn resolve_cwd(common: &CommonTaskArgs) -> Result<PathBuf> {
    Ok(common
        .cwd
        .clone()
        .unwrap_or(std::env::current_dir()?)
        .canonicalize()?)
}

fn parse_command(mut command: Vec<String>) -> Result<Vec<String>> {
    if command.first().is_some_and(|value| value == "--") {
        command.remove(0);
    }
    ensure!(!command.is_empty(), "missing command after --");
    Ok(command)
}

fn parse_inputs(values: &[String]) -> Result<BTreeMap<String, String>> {
    let mut parsed = BTreeMap::new();
    for value in values {
        let (key, raw) = value
            .split_once('=')
            .with_context(|| format!("--input must use key=value form: {value}"))?;
        ensure!(!key.is_empty(), "--input has an empty key: {value}");
        parsed.insert(key.to_owned(), raw.to_owned());
    }
    Ok(parsed)
}

fn newest_activity(paths: &[&Path]) -> Option<SystemTime> {
    paths
        .iter()
        .filter_map(|path| fs::metadata(path).and_then(|meta| meta.modified()).ok())
        .max()
}

fn tail_lines(path: &Path, limit: usize) -> Vec<String> {
    if limit == 0 {
        return Vec::new();
    }
    fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .rev()
        .take(limit)
        .map(str::to_owned)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
}

fn interesting_log_lines(text: &str, limit: usize) -> Vec<String> {
    let markers = [
        "error",
        "failed",
        "failure",
        "timed out",
        "timeout",
        "panic",
        "exception",
        "##[error]",
    ];
    let lines = text
        .lines()
        .map(str::trim_end)
        .filter(|line| !line.trim().is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let interesting = lines
        .iter()
        .filter(|line| {
            let lower = line.to_ascii_lowercase();
            markers.iter().any(|marker| lower.contains(marker))
        })
        .cloned()
        .collect::<Vec<_>>();
    let selected = if interesting.is_empty() {
        lines
    } else {
        interesting
    };
    selected
        .into_iter()
        .rev()
        .take(limit)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
}

fn active_step(job: &Value) -> String {
    job.get("steps")
        .and_then(Value::as_array)
        .and_then(|steps| {
            steps.iter().find(|step| {
                matches!(
                    step.get("status").and_then(Value::as_str),
                    Some("queued" | "in_progress")
                )
            })
        })
        .map(step_name)
        .unwrap_or_default()
}

fn failed_step(job: &Value) -> String {
    job.get("steps")
        .and_then(Value::as_array)
        .and_then(|steps| {
            steps.iter().find(|step| {
                !matches!(
                    step.get("conclusion").and_then(Value::as_str),
                    None | Some("") | Some("success") | Some("skipped")
                )
            })
        })
        .map(step_name)
        .unwrap_or_default()
}

fn step_name(step: &Value) -> String {
    step.get("name")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .or_else(|| step.get("number").map(ToString::to_string))
        .unwrap_or_else(|| "unknown".into())
}

fn write_github_output(task: &Map<String, Value>) -> Result<()> {
    if let Ok(path) = std::env::var("GITHUB_OUTPUT") {
        let mut file = File::options().append(true).create(true).open(path)?;
        if !string_field(task, "run_id").is_empty() {
            writeln!(file, "run_id={}", string_field(task, "run_id"))?;
        }
        writeln!(file, "agent_task_id={}", task_id(task))?;
    }
    Ok(())
}

fn append_step_summary(summary: &str) -> Result<()> {
    if let Ok(path) = std::env::var("GITHUB_STEP_SUMMARY") {
        let mut file = File::options().append(true).create(true).open(path)?;
        writeln!(file, "{summary}")?;
    }
    Ok(())
}

fn required_env(name: &str) -> Result<String> {
    std::env::var(name).with_context(|| format!("{name} must be set"))
}

fn input_env(input: &str, env: &str) -> Result<String> {
    Ok(format!("{input}={}", required_env(env)?))
}

fn input_env_default(input: &str, env: &str, default: &str) -> String {
    format!(
        "{input}={}",
        std::env::var(env).unwrap_or_else(|_| default.into())
    )
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn new_task_id(label: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{}-{nanos:x}-{}", slugify(label), std::process::id())
}

fn slugify(value: &str) -> String {
    let mut slug = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '-') {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>();
    while slug.contains("--") {
        slug = slug.replace("--", "-");
    }
    slug.trim_matches('-').chars().take(48).collect::<String>()
}

fn utc_now() -> String {
    Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

fn abs(path: &Path) -> String {
    path.canonicalize()
        .unwrap_or_else(|_| path.to_path_buf())
        .display()
        .to_string()
}

fn normalized_interval(value: u64) -> u64 {
    value.max(1)
}

fn task_id(task: &Map<String, Value>) -> String {
    string_field(task, "id")
}

fn string_field(task: &Map<String, Value>, field: &str) -> String {
    string_field_or(task, field, "")
}

fn string_field_or(task: &Map<String, Value>, field: &str, default: &str) -> String {
    task.get(field)
        .and_then(Value::as_str)
        .unwrap_or(default)
        .to_owned()
}

fn json_string(value: &Value, field: &str, default: &str) -> String {
    value
        .get(field)
        .and_then(Value::as_str)
        .unwrap_or(default)
        .to_owned()
}

fn bool_field(task: &Map<String, Value>, field: &str) -> bool {
    task.get(field).and_then(Value::as_bool).unwrap_or(false)
}

fn i64_field(task: &Map<String, Value>, field: &str, default: i64) -> i64 {
    task.get(field).and_then(Value::as_i64).unwrap_or(default)
}

fn u64_field(task: &Map<String, Value>, field: &str) -> u64 {
    u64_field_or(task, field, 0)
}

fn u64_field_or(task: &Map<String, Value>, field: &str, default: u64) -> u64 {
    task.get(field).and_then(Value::as_u64).unwrap_or(default)
}

fn usize_field(task: &Map<String, Value>, field: &str, default: usize) -> usize {
    task.get(field)
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .unwrap_or(default)
}

fn string_vec(task: &Map<String, Value>, field: &str) -> Result<Vec<String>> {
    task.get(field)
        .and_then(Value::as_array)
        .context("expected array field")?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(ToOwned::to_owned)
                .context("expected string array item")
        })
        .collect()
}

fn string_array(task: &Map<String, Value>, field: &str) -> Vec<String> {
    task.get(field)
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(ToOwned::to_owned)
        .collect()
}

fn shell_quote(value: &str) -> String {
    if value
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || "-_./:=+".contains(ch))
    {
        value.to_owned()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}

impl Default for CommonTaskArgs {
    fn default() -> Self {
        Self {
            state_root: None,
            task_id: None,
            cwd: None,
        }
    }
}

impl Default for WaitOptions {
    fn default() -> Self {
        Self {
            interval_secs: DEFAULT_INTERVAL_SECS,
            timeout_secs: 0,
            stale_secs: 0,
            failed_log_lines: DEFAULT_FAILED_LOG_LINES,
            exit_status: false,
            detach: false,
            wait: false,
            watch: false,
        }
    }
}
