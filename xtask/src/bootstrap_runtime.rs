use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use base64::Engine;
use serde_json::{Value, json};
use tempfile::TempDir;

pub fn sync_bootstrap_runtime_config() -> Result<()> {
    let aws_region = required_env("AWS_REGION")?;
    let instance_id = required_env("BOOTSTRAP_INSTANCE_ID")?;
    let artifact_bucket_name = required_env("ARTIFACT_BUCKET_NAME")?;
    let runtime_config_prefix = env_or(
        "RUNTIME_CONFIG_PREFIX",
        &format!(
            "runtime-config/bootstrap/{}-{}",
            env_or("GITHUB_RUN_ID", "manual"),
            env_or("GITHUB_RUN_ATTEMPT", "0")
        ),
    );

    let bootstrap_install_source = env_or("BOOTSTRAP_INSTALL_SOURCE", "crate");
    let bootstrap_crate_version = env_or("BOOTSTRAP_CRATE_VERSION", "0.21.5");
    let bootstrap_git_repository = env_or(
        "BOOTSTRAP_GIT_REPOSITORY",
        "https://github.com/aberration-technology/burn_p2p.git",
    );
    let bootstrap_git_ref = env_or("BOOTSTRAP_GIT_REF", "");
    let bootstrap_binary_path = env_or("BOOTSTRAP_BINARY_PATH", "");
    let bootstrap_binary_sha256 = env_or("BOOTSTRAP_BINARY_SHA256", "");
    let auth_connector_kind = env_or("AUTH_CONNECTOR_KIND", "github");
    let bootstrap_auth_feature = env_or(
        "BOOTSTRAP_AUTH_FEATURE",
        auth_feature_for_connector(&auth_connector_kind),
    );
    let bootstrap_features = env_or(
        "BOOTSTRAP_FEATURES",
        &format!(
            "admin-http,metrics,metrics-indexer,artifact-publish,artifact-download,artifact-fs,artifact-s3,browser-edge,browser-join,{bootstrap_auth_feature},rbac,social"
        ),
    );
    let bootstrap_reinstall = env_or("BOOTSTRAP_REINSTALL", "false");
    let dragon_git_repository = env_or(
        "DRAGON_GIT_REPOSITORY",
        "https://github.com/aberration-technology/burn_dragon.git",
    );
    let dragon_git_ref = env_or("DRAGON_GIT_REF", "main");
    let head_mirror_binary_path = env_or("HEAD_MIRROR_BINARY_PATH", "");
    let head_mirror_binary_sha256 = env_or("HEAD_MIRROR_BINARY_SHA256", "");
    let head_mirror_reinstall = env_or("HEAD_MIRROR_REINSTALL", "true");
    let edge_base_url = env_or("EDGE_BASE_URL", "");

    let tmpdir = TempDir::new().context("create runtime config tempdir")?;
    let bootstrap_config_path = tmpdir.path().join("bootstrap.json");
    write_base64_env("BOOTSTRAP_CONFIG_JSON_B64", &bootstrap_config_path)?;
    write_base64_env("CADDYFILE_B64", &tmpdir.path().join("Caddyfile"))?;
    write_base64_env(
        "BOOTSTRAP_SERVICE_UNIT_B64",
        &tmpdir.path().join("burn-p2p-bootstrap.service"),
    )?;
    write_base64_env(
        "BOOTSTRAP_HEAD_MIRROR_CONFIG_B64",
        &tmpdir.path().join("bootstrap-head-mirror.toml"),
    )?;
    write_base64_env(
        "BOOTSTRAP_HEAD_MIRROR_AUTH_SCRIPT_B64",
        &tmpdir
            .path()
            .join("burn-dragon-p2p-fetch-head-mirror-auth-bundle"),
    )?;
    write_base64_env(
        "BOOTSTRAP_HEAD_MIRROR_SERVICE_UNIT_B64",
        &tmpdir.path().join("burn-dragon-p2p-head-mirror.service"),
    )?;

    if !edge_base_url.is_empty()
        && let Err(error) = preserve_bootstrap_current_heads(
            &bootstrap_config_path,
            &format!("{}/portal/snapshot", edge_base_url.trim_end_matches('/')),
            true,
        )
    {
        eprintln!(
            "warning: failed to preserve bootstrap current heads from {edge_base_url}: {error:#}"
        );
    }

    let objects = RuntimeObjects::new(&artifact_bucket_name, &runtime_config_prefix);
    let cleanup = RuntimeCleanup::new(
        objects.clone(),
        bootstrap_binary_path.clone(),
        head_mirror_binary_path.clone(),
    );

    aws_s3_cp(&bootstrap_config_path, &objects.bootstrap)?;
    aws_s3_cp(&tmpdir.path().join("Caddyfile"), &objects.caddy)?;
    aws_s3_cp(
        &tmpdir.path().join("burn-p2p-bootstrap.service"),
        &objects.bootstrap_service,
    )?;
    aws_s3_cp(
        &tmpdir.path().join("bootstrap-head-mirror.toml"),
        &objects.head_mirror_config,
    )?;
    aws_s3_cp(
        &tmpdir
            .path()
            .join("burn-dragon-p2p-fetch-head-mirror-auth-bundle"),
        &objects.head_mirror_auth_script,
    )?;
    aws_s3_cp(
        &tmpdir.path().join("burn-dragon-p2p-head-mirror.service"),
        &objects.head_mirror_service,
    )?;
    if !bootstrap_binary_path.is_empty() {
        aws_s3_cp_path(&bootstrap_binary_path, &objects.bootstrap_binary)?;
    }
    if !head_mirror_binary_path.is_empty() {
        aws_s3_cp_path(&head_mirror_binary_path, &objects.head_mirror_binary)?;
    }

    wait_for_ssm_online(&aws_region, &instance_id)?;
    let commands = render_bootstrap_runtime_sync_commands(&RuntimeCommandEnv {
        bootstrap_object_uri: objects.bootstrap.clone(),
        caddy_object_uri: objects.caddy.clone(),
        bootstrap_service_unit_object_uri: objects.bootstrap_service.clone(),
        head_mirror_config_object_uri: objects.head_mirror_config.clone(),
        head_mirror_auth_script_object_uri: objects.head_mirror_auth_script.clone(),
        head_mirror_service_object_uri: objects.head_mirror_service.clone(),
        bootstrap_install_source,
        bootstrap_crate_version,
        bootstrap_git_repository,
        bootstrap_git_ref,
        bootstrap_binary_object_uri: if bootstrap_binary_path.is_empty() {
            String::new()
        } else {
            objects.bootstrap_binary.clone()
        },
        bootstrap_binary_sha256,
        bootstrap_features,
        bootstrap_reinstall,
        dragon_git_repository,
        dragon_git_ref,
        head_mirror_binary_object_uri: if head_mirror_binary_path.is_empty() {
            String::new()
        } else {
            objects.head_mirror_binary.clone()
        },
        head_mirror_binary_sha256,
        head_mirror_reinstall,
    })?;
    let params_json = serde_json::to_string(&json!({ "commands": commands }))?;
    let command_id = aws_output(&[
        "ssm",
        "send-command",
        "--region",
        &aws_region,
        "--instance-ids",
        &instance_id,
        "--document-name",
        "AWS-RunShellScript",
        "--comment",
        "sync burn_dragon bootstrap runtime config",
        "--parameters",
        &params_json,
        "--query",
        "Command.CommandId",
        "--output",
        "text",
    ])?;
    let command_id = command_id.trim().to_owned();
    wait_for_runtime_sync(&aws_region, &instance_id, &command_id)?;
    let invocation = aws_output(&[
        "ssm",
        "get-command-invocation",
        "--region",
        &aws_region,
        "--command-id",
        &command_id,
        "--instance-id",
        &instance_id,
        "--output",
        "json",
    ])?;
    println!("{invocation}");
    drop(cleanup);
    Ok(())
}

pub fn render_bootstrap_runtime_sync_commands(env: &RuntimeCommandEnv) -> Result<Vec<String>> {
    let mut commands = vec![
        "set -eu".to_owned(),
        "export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH".to_owned(),
        wait_for_runtime_sync_prereqs_command(),
        ensure_aws_cli_command(),
        "install -d -m 0755 /etc/burn-dragon-p2p /etc/caddy /etc/burn_dragon_p2p /usr/local/bin"
            .to_owned(),
        "install -d -m 0700 /var/lib/burn_dragon_p2p".to_owned(),
    ];

    let mut head_mirror_setup = Vec::new();
    if !env.head_mirror_binary_object_uri.is_empty() {
        head_mirror_setup.push(format!(
            "aws s3 cp '{}' /usr/local/bin/burn_dragon_p2p_native",
            env.head_mirror_binary_object_uri
        ));
        head_mirror_setup.push("chmod 0755 /usr/local/bin/burn_dragon_p2p_native".to_owned());
        if !env.head_mirror_binary_sha256.trim().is_empty() {
            head_mirror_setup.push(format!(
                "remote_sha=$(sha256sum /usr/local/bin/burn_dragon_p2p_native | awk '{{print $1}}'); if [ \"$remote_sha\" != '{}' ]; then echo \"head mirror binary checksum mismatch: expected {} got $remote_sha\" >&2; exit 1; fi",
                env.head_mirror_binary_sha256, env.head_mirror_binary_sha256
            ));
        }
    } else if !truthy(&env.head_mirror_reinstall) {
        bail!("HEAD_MIRROR_BINARY_OBJECT_URI is required when HEAD_MIRROR_REINSTALL is false");
    } else if truthy(&env.head_mirror_reinstall) {
        head_mirror_setup.push(head_mirror_install_command(env)?);
        head_mirror_setup.push("if [ -x /root/.cargo/bin/burn_dragon_p2p_native ]; then ln -sf /root/.cargo/bin/burn_dragon_p2p_native /usr/local/bin/burn_dragon_p2p_native; fi".to_owned());
    }

    let mut bootstrap_setup = Vec::new();
    if !env.bootstrap_binary_object_uri.is_empty() {
        bootstrap_setup.push(format!(
            "aws s3 cp '{}' /usr/local/bin/burn-p2p-bootstrap",
            env.bootstrap_binary_object_uri
        ));
        bootstrap_setup.push("chmod 0755 /usr/local/bin/burn-p2p-bootstrap".to_owned());
        if !env.bootstrap_binary_sha256.trim().is_empty() {
            bootstrap_setup.push(format!(
                "remote_sha=$(sha256sum /usr/local/bin/burn-p2p-bootstrap | awk '{{print $1}}'); if [ \"$remote_sha\" != '{}' ]; then echo \"bootstrap binary checksum mismatch: expected {} got $remote_sha\" >&2; exit 1; fi",
                env.bootstrap_binary_sha256, env.bootstrap_binary_sha256
            ));
        }
    } else if env.bootstrap_install_source == "git" && !truthy(&env.bootstrap_reinstall) {
        bail!(
            "BOOTSTRAP_BINARY_OBJECT_URI is required for git bootstrap sync when BOOTSTRAP_REINSTALL is false"
        );
    } else {
        bootstrap_setup.push(format!(
            "if [ ! -x /usr/local/bin/burn-p2p-bootstrap ] && [ ! -x /root/.cargo/bin/burn-p2p-bootstrap ]; then {}; fi",
            bootstrap_install_command(env)?
        ));
        bootstrap_setup.push("if [ -x /root/.cargo/bin/burn-p2p-bootstrap ]; then ln -sf /root/.cargo/bin/burn-p2p-bootstrap /usr/local/bin/burn-p2p-bootstrap; fi".to_owned());
    }

    commands.extend(head_mirror_setup);
    commands.extend(bootstrap_setup);
    commands.extend([
        "systemctl reset-failed burn-p2p-bootstrap || true".to_owned(),
        "systemctl reset-failed burn-dragon-p2p-head-mirror || true".to_owned(),
        format!(
            "aws s3 cp '{}' /etc/burn-dragon-p2p/bootstrap.json",
            env.bootstrap_object_uri
        ),
        format!("aws s3 cp '{}' /etc/caddy/Caddyfile", env.caddy_object_uri),
        format!(
            "aws s3 cp '{}' /etc/systemd/system/burn-p2p-bootstrap.service",
            env.bootstrap_service_unit_object_uri
        ),
        format!(
            "aws s3 cp '{}' /etc/burn_dragon_p2p/bootstrap-head-mirror.toml",
            env.head_mirror_config_object_uri
        ),
        format!(
            "aws s3 cp '{}' /usr/local/bin/burn-dragon-p2p-fetch-head-mirror-auth-bundle",
            env.head_mirror_auth_script_object_uri
        ),
        format!(
            "aws s3 cp '{}' /etc/systemd/system/burn-dragon-p2p-head-mirror.service",
            env.head_mirror_service_object_uri
        ),
    ]);
    commands.extend(bootstrap_public_ip_rewrite_commands());
    commands.extend([
        "chmod 0644 /etc/burn-dragon-p2p/bootstrap.json /etc/caddy/Caddyfile /etc/systemd/system/burn-p2p-bootstrap.service /etc/burn_dragon_p2p/bootstrap-head-mirror.toml /etc/systemd/system/burn-dragon-p2p-head-mirror.service".to_owned(),
        "chmod 0755 /usr/local/bin/burn-dragon-p2p-fetch-head-mirror-auth-bundle".to_owned(),
        "/usr/local/bin/burn-dragon-p2p-sync-secrets".to_owned(),
        "systemctl stop burn-dragon-p2p-head-mirror || true".to_owned(),
        "systemctl stop burn-p2p-bootstrap || true".to_owned(),
        quarantine_corrupt_bootstrap_state_command(),
        "systemctl daemon-reload".to_owned(),
        "systemctl restart caddy".to_owned(),
        "systemctl enable burn-dragon-p2p-head-mirror".to_owned(),
        "systemctl enable burn-p2p-bootstrap".to_owned(),
        "systemctl restart burn-p2p-bootstrap".to_owned(),
        wait_for_systemd_service_active_command("caddy"),
        wait_for_systemd_service_active_command("burn-p2p-bootstrap"),
        "journalctl -u caddy -u burn-p2p-bootstrap -u burn-dragon-p2p-head-mirror --no-pager -n 60 || true".to_owned(),
    ]);
    Ok(commands)
}

#[derive(Debug)]
pub struct RuntimeCommandEnv {
    pub bootstrap_object_uri: String,
    pub caddy_object_uri: String,
    pub bootstrap_service_unit_object_uri: String,
    pub head_mirror_config_object_uri: String,
    pub head_mirror_auth_script_object_uri: String,
    pub head_mirror_service_object_uri: String,
    pub bootstrap_install_source: String,
    pub bootstrap_crate_version: String,
    pub bootstrap_git_repository: String,
    pub bootstrap_git_ref: String,
    pub bootstrap_binary_object_uri: String,
    pub bootstrap_binary_sha256: String,
    pub bootstrap_features: String,
    pub bootstrap_reinstall: String,
    pub dragon_git_repository: String,
    pub dragon_git_ref: String,
    pub head_mirror_binary_object_uri: String,
    pub head_mirror_binary_sha256: String,
    pub head_mirror_reinstall: String,
}

#[derive(Clone)]
struct RuntimeObjects {
    bootstrap: String,
    caddy: String,
    bootstrap_service: String,
    head_mirror_config: String,
    head_mirror_auth_script: String,
    head_mirror_service: String,
    bootstrap_binary: String,
    head_mirror_binary: String,
}

impl RuntimeObjects {
    fn new(bucket: &str, prefix: &str) -> Self {
        let uri = |name: &str| format!("s3://{bucket}/{prefix}/{name}");
        Self {
            bootstrap: uri("bootstrap.json"),
            caddy: uri("Caddyfile"),
            bootstrap_service: uri("burn-p2p-bootstrap.service"),
            head_mirror_config: uri("bootstrap-head-mirror.toml"),
            head_mirror_auth_script: uri("burn-dragon-p2p-fetch-head-mirror-auth-bundle"),
            head_mirror_service: uri("burn-dragon-p2p-head-mirror.service"),
            bootstrap_binary: uri("burn-p2p-bootstrap"),
            head_mirror_binary: uri("burn_dragon_p2p_native"),
        }
    }
}

struct RuntimeCleanup {
    objects: RuntimeObjects,
    bootstrap_binary_path: String,
    head_mirror_binary_path: String,
}

impl RuntimeCleanup {
    fn new(
        objects: RuntimeObjects,
        bootstrap_binary_path: String,
        head_mirror_binary_path: String,
    ) -> Self {
        Self {
            objects,
            bootstrap_binary_path,
            head_mirror_binary_path,
        }
    }
}

impl Drop for RuntimeCleanup {
    fn drop(&mut self) {
        for uri in [
            &self.objects.bootstrap,
            &self.objects.caddy,
            &self.objects.bootstrap_service,
            &self.objects.head_mirror_config,
            &self.objects.head_mirror_auth_script,
            &self.objects.head_mirror_service,
        ] {
            let _ = aws_status(&["s3", "rm", uri]);
        }
        if !self.bootstrap_binary_path.is_empty() {
            let _ = aws_status(&["s3", "rm", &self.objects.bootstrap_binary]);
        }
        if !self.head_mirror_binary_path.is_empty() {
            let _ = aws_status(&["s3", "rm", &self.objects.head_mirror_binary]);
        }
    }
}

pub fn preserve_bootstrap_current_heads(
    config_path: &Path,
    snapshot_url: &str,
    recover_roots: bool,
) -> Result<Value> {
    let mut config: Value = serde_json::from_slice(&fs::read(config_path)?)?;
    let snapshot: Value = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(8))
        .build()?
        .get(snapshot_url)
        .send()?
        .error_for_status()?
        .json()?;
    let report = preserve_current_heads(&mut config, &snapshot, recover_roots);
    fs::write(config_path, serde_json::to_string(&config)?)?;
    eprintln!("{}", serde_json::to_string(&report)?);
    Ok(report)
}

pub fn preserve_current_heads(config: &mut Value, snapshot: &Value, recover_roots: bool) -> Value {
    let Some(entries) = config
        .get_mut("auth")
        .and_then(|auth| auth.get_mut("directory_entries"))
        .and_then(Value::as_array_mut)
    else {
        return json!({ "preserved": 0, "recovered": 0 });
    };
    let mut preserved = 0;
    let mut recovered = 0;
    for entry in entries {
        if entry
            .get("current_head_id")
            .and_then(Value::as_str)
            .is_some()
        {
            continue;
        }
        if let Some(head_id) = live_current_head_for_entry(snapshot, entry) {
            entry["current_head_id"] = json!(head_id);
            preserved += 1;
        } else if recover_roots && let Some(head_id) = recover_visible_root(snapshot, entry) {
            entry["current_head_id"] = json!(head_id);
            recovered += 1;
        }
    }
    json!({ "preserved": preserved, "recovered": recovered })
}

fn live_current_head_for_entry(snapshot: &Value, target: &Value) -> Option<String> {
    snapshot
        .get("directory")
        .and_then(|directory| directory.get("entries"))
        .and_then(Value::as_array)?
        .iter()
        .find(|entry| entry_key(entry) == entry_key(target))
        .and_then(|entry| entry.get("current_head_id"))
        .and_then(Value::as_str)
        .map(str::to_owned)
}

fn recover_visible_root(snapshot: &Value, target: &Value) -> Option<String> {
    let target_key = entry_key(target);
    snapshot
        .get("heads")
        .and_then(Value::as_array)?
        .iter()
        .filter(|head| {
            (
                string_field(head, "study_id"),
                string_field(head, "experiment_id"),
                string_field(head, "revision_id"),
            ) == target_key
                && head.get("parent_head_id").is_none_or(Value::is_null)
                && head.get("head_id").and_then(Value::as_str).is_some()
        })
        .max_by_key(|head| {
            (
                string_field(head, "created_at"),
                string_field(head, "head_id"),
            )
        })
        .and_then(|head| head.get("head_id"))
        .and_then(Value::as_str)
        .map(str::to_owned)
}

fn entry_key(entry: &Value) -> (String, String, String) {
    (
        string_field(entry, "study_id"),
        string_field(entry, "experiment_id"),
        string_field(entry, "current_revision_id").if_empty(string_field(entry, "revision_id")),
    )
}

fn bootstrap_install_command(env: &RuntimeCommandEnv) -> Result<String> {
    let mut command = "export HOME=/root CARGO_HOME=/root/.cargo RUSTUP_HOME=/root/.rustup; if [ ! -x /root/.cargo/bin/cargo ]; then curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal; fi; . /root/.cargo/env; ".to_owned();
    match env.bootstrap_install_source.as_str() {
        "crate" => {
            command.push_str(&format!(
                "cargo install --locked burn_p2p_bootstrap --version '{}' --bin burn-p2p-bootstrap --no-default-features --features '{}'",
                env.bootstrap_crate_version, env.bootstrap_features
            ));
        }
        "git" => {
            if env.bootstrap_git_ref.is_empty() {
                bail!("BOOTSTRAP_GIT_REF is required for git bootstrap install");
            }
            command.push_str(&format!(
                "cargo install --locked --git '{}' --rev '{}' burn_p2p_bootstrap --bin burn-p2p-bootstrap --no-default-features --features '{}'",
                env.bootstrap_git_repository, env.bootstrap_git_ref, env.bootstrap_features
            ));
        }
        other => bail!("unsupported bootstrap install source: {other}"),
    }
    Ok(command)
}

fn head_mirror_install_command(env: &RuntimeCommandEnv) -> Result<String> {
    if env.dragon_git_ref.is_empty() {
        bail!("DRAGON_GIT_REF is required for head mirror install");
    }
    Ok(format!(
        "if [ ! -x /usr/local/bin/burn_dragon_p2p_native ] && [ ! -x /root/.cargo/bin/burn_dragon_p2p_native ]; then export HOME=/root CARGO_HOME=/root/.cargo RUSTUP_HOME=/root/.rustup; if [ ! -x /root/.cargo/bin/cargo ]; then curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal; fi; . /root/.cargo/env; cargo install --locked --git '{}' --rev '{}' burn_dragon_p2p --bin burn_dragon_p2p_native --no-default-features --features native; fi",
        env.dragon_git_repository, env.dragon_git_ref
    ))
}

fn wait_for_runtime_sync_prereqs_command() -> String {
    "runtime_sync_ready=; for attempt in $(seq 1 60); do if [ -x /usr/local/bin/burn-dragon-p2p-sync-secrets ] && command -v aws >/dev/null 2>&1; then runtime_sync_ready=1; break; fi; sleep 5; done; if [ -z \"$runtime_sync_ready\" ]; then echo 'timed out waiting for bootstrap runtime sync prerequisites' >&2; exit 1; fi".to_owned()
}

fn ensure_aws_cli_command() -> String {
    "if ! command -v aws >/dev/null 2>&1; then tmpdir=$(mktemp -d); curl -fsSL 'https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip' -o \"$tmpdir/awscliv2.zip\"; unzip -q \"$tmpdir/awscliv2.zip\" -d \"$tmpdir\"; \"$tmpdir/aws/install\" --bin-dir /usr/local/bin --install-dir /usr/local/aws-cli --update; rm -rf \"$tmpdir\"; fi".to_owned()
}

fn bootstrap_public_ip_rewrite_commands() -> Vec<String> {
    vec![
        "IMDS_TOKEN=$(curl -fsSL -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 21600' || true)".to_owned(),
        "BOOTSTRAP_PUBLIC_IPV4=$(if [ -n \"$IMDS_TOKEN\" ]; then curl -fsSL -H \"X-aws-ec2-metadata-token: $IMDS_TOKEN\" http://169.254.169.254/latest/meta-data/public-ipv4; else curl -fsSL http://169.254.169.254/latest/meta-data/public-ipv4; fi || true)".to_owned(),
        r#"if [ -n "$BOOTSTRAP_PUBLIC_IPV4" ]; then python3 - "$BOOTSTRAP_PUBLIC_IPV4" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path("/etc/burn-dragon-p2p/bootstrap.json")
config = json.loads(config_path.read_text())
external_addresses = (
    config.get("bootstrap_peer", {})
    .get("node", {})
    .get("external_addresses", [])
)
config["bootstrap_peer"]["node"]["external_addresses"] = [
    address.replace("PUBLIC_IP", sys.argv[1]) for address in external_addresses
]
config_path.write_text(json.dumps(config, indent=2) + "\n")
PY
fi"#
        .to_owned(),
    ]
}

fn wait_for_systemd_service_active_command(service_name: &str) -> String {
    format!(
        "service_state=''; for attempt in $(seq 1 30); do service_state=$(systemctl is-active {service_name} 2>/dev/null || true); if [ \"$service_state\" = 'active' ]; then break; fi; if [ \"$service_state\" = 'failed' ]; then systemctl status {service_name} --no-pager || true; journalctl -u {service_name} --no-pager -n 200 || true; echo '{service_name} failed to reach active state' >&2; exit 1; fi; sleep 2; done; if [ \"$service_state\" != 'active' ]; then systemctl status {service_name} --no-pager || true; journalctl -u {service_name} --no-pager -n 200 || true; echo \"timed out waiting for {service_name} to reach active state (last state: $service_state)\" >&2; exit 1; fi"
    )
}

fn quarantine_corrupt_bootstrap_state_command() -> String {
    r#"python3 - <<'PY'
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

bootstrap_root = Path("/var/lib/burn-p2p/bootstrap-peer")
candidate_paths = [
    bootstrap_root / "state" / "known-peers.json",
    bootstrap_root / "state" / "slot-assignment-primary.json",
    bootstrap_root / "state" / "slot-assignments.json",
    bootstrap_root / "state" / "security-state.json",
    bootstrap_root / "state" / "control-plane-state.json",
]
candidate_paths.extend(
    sorted((bootstrap_root / "state" / "transfers").glob("*.json"))
)
candidate_paths.extend(sorted((bootstrap_root / "leases").glob("*.json")))


def quarantine(path: Path, reason: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = path.with_name(f"{path.name}.corrupt-{stamp}")
    suffix = 0
    while destination.exists():
        suffix += 1
        destination = path.with_name(f"{path.name}.corrupt-{stamp}-{suffix}")
    print(
        f"quarantining corrupt persisted bootstrap state {path} -> {destination}: {reason}",
        flush=True,
    )
    path.rename(destination)


for path in candidate_paths:
    if not path.exists():
        continue
    try:
        json.loads(path.read_text())
    except Exception as error:
        quarantine(path, str(error))
PY"#
    .to_owned()
}

fn wait_for_ssm_online(region: &str, instance_id: &str) -> Result<()> {
    for _ in 1..=60 {
        let status = aws_output_allow_failure(&[
            "ssm",
            "describe-instance-information",
            "--region",
            region,
            "--filters",
            &format!("Key=InstanceIds,Values={instance_id}"),
            "--query",
            "InstanceInformationList[0].PingStatus",
            "--output",
            "text",
        ])?;
        if status.trim() == "Online" {
            return Ok(());
        }
        thread::sleep(Duration::from_secs(10));
    }
    bail!("bootstrap instance did not reach SSM Online status; cannot sync runtime config")
}

fn wait_for_runtime_sync(region: &str, instance_id: &str, command_id: &str) -> Result<()> {
    for _ in 1..=60 {
        let status = aws_output_allow_failure(&[
            "ssm",
            "get-command-invocation",
            "--region",
            region,
            "--command-id",
            command_id,
            "--instance-id",
            instance_id,
            "--query",
            "Status",
            "--output",
            "text",
        ])?;
        match status.trim() {
            "Success" => return Ok(()),
            "Cancelled" | "TimedOut" | "Failed" | "Cancelling" => {
                eprintln!(
                    "bootstrap runtime config sync failed with status {}",
                    status.trim()
                );
                let _ = aws_status(&[
                    "ssm",
                    "get-command-invocation",
                    "--region",
                    region,
                    "--command-id",
                    command_id,
                    "--instance-id",
                    instance_id,
                    "--output",
                    "json",
                ]);
                bail!("bootstrap runtime config sync failed");
            }
            _ => {}
        }
        thread::sleep(Duration::from_secs(5));
    }
    let _ = aws_status(&[
        "ssm",
        "get-command-invocation",
        "--region",
        region,
        "--command-id",
        command_id,
        "--instance-id",
        instance_id,
        "--output",
        "json",
    ]);
    bail!("timed out waiting for bootstrap runtime config sync to finish")
}

fn write_base64_env(name: &str, path: &Path) -> Result<()> {
    let value = required_env(name)?;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(value.trim())
        .with_context(|| format!("failed to decode {name}"))?;
    fs::write(path, bytes).with_context(|| format!("failed to write {}", path.display()))
}

fn aws_s3_cp(local: &Path, uri: &str) -> Result<()> {
    let local = local
        .to_str()
        .ok_or_else(|| anyhow!("path is not valid utf-8: {}", local.display()))?;
    aws_status_checked(&["s3", "cp", local, uri])
}

fn aws_s3_cp_path(local: &str, uri: &str) -> Result<()> {
    aws_status_checked(&["s3", "cp", local, uri])
}

fn aws_output(args: &[&str]) -> Result<String> {
    let output = Command::new("aws")
        .args(args)
        .stdin(Stdio::null())
        .output()
        .with_context(|| format!("failed to start aws {}", args.join(" ")))?;
    if !output.status.success() {
        bail!(
            "aws {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn aws_output_allow_failure(args: &[&str]) -> Result<String> {
    let output = Command::new("aws")
        .args(args)
        .stdin(Stdio::null())
        .output()
        .with_context(|| format!("failed to start aws {}", args.join(" ")))?;
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn aws_status_checked(args: &[&str]) -> Result<()> {
    let status = aws_status(args)?;
    if !status {
        bail!("aws {} failed", args.join(" "));
    }
    Ok(())
}

fn aws_status(args: &[&str]) -> Result<bool> {
    let status = Command::new("aws")
        .args(args)
        .stdin(Stdio::null())
        .status()
        .with_context(|| format!("failed to start aws {}", args.join(" ")))?;
    Ok(status.success())
}

fn required_env(name: &str) -> Result<String> {
    let value = std::env::var(name).with_context(|| format!("{name} is required"))?;
    if value.is_empty() {
        bail!("{name} is required");
    }
    Ok(value)
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name)
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| default.to_owned())
}

fn auth_feature_for_connector(kind: &str) -> &'static str {
    match kind.trim().to_ascii_lowercase().as_str() {
        "github" => "auth-github",
        "oidc" => "auth-oidc",
        "oauth" => "auth-oauth",
        "external" => "auth-external",
        _ => "auth-static",
    }
}

fn truthy(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn string_field(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

trait IfEmpty {
    fn if_empty(self, fallback: String) -> String;
}

impl IfEmpty for String {
    fn if_empty(self, fallback: String) -> String {
        if self.is_empty() { fallback } else { self }
    }
}
