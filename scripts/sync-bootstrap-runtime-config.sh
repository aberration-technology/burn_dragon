#!/usr/bin/env bash

set -euo pipefail

aws_region="${AWS_REGION:?AWS_REGION is required}"
instance_id="${BOOTSTRAP_INSTANCE_ID:?BOOTSTRAP_INSTANCE_ID is required}"
artifact_bucket_name="${ARTIFACT_BUCKET_NAME:?ARTIFACT_BUCKET_NAME is required}"
bootstrap_config_json_b64="${BOOTSTRAP_CONFIG_JSON_B64:?BOOTSTRAP_CONFIG_JSON_B64 is required}"
caddyfile_b64="${CADDYFILE_B64:?CADDYFILE_B64 is required}"
bootstrap_head_mirror_config_b64="${BOOTSTRAP_HEAD_MIRROR_CONFIG_B64:?BOOTSTRAP_HEAD_MIRROR_CONFIG_B64 is required}"
bootstrap_head_mirror_auth_script_b64="${BOOTSTRAP_HEAD_MIRROR_AUTH_SCRIPT_B64:?BOOTSTRAP_HEAD_MIRROR_AUTH_SCRIPT_B64 is required}"
bootstrap_head_mirror_service_unit_b64="${BOOTSTRAP_HEAD_MIRROR_SERVICE_UNIT_B64:?BOOTSTRAP_HEAD_MIRROR_SERVICE_UNIT_B64 is required}"
runtime_config_prefix="${RUNTIME_CONFIG_PREFIX:-runtime-config/bootstrap/${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-0}}"
bootstrap_install_source="${BOOTSTRAP_INSTALL_SOURCE:-crate}"
bootstrap_crate_version="${BOOTSTRAP_CRATE_VERSION:-0.21.0-pre.15}"
bootstrap_git_repository="${BOOTSTRAP_GIT_REPOSITORY:-https://github.com/aberration-technology/burn_p2p.git}"
bootstrap_git_ref="${BOOTSTRAP_GIT_REF:-}"
auth_connector_kind="${AUTH_CONNECTOR_KIND:-github}"
bootstrap_auth_feature="${BOOTSTRAP_AUTH_FEATURE:-}"
bootstrap_features="${BOOTSTRAP_FEATURES:-}"
bootstrap_reinstall="${BOOTSTRAP_REINSTALL:-false}"
dragon_git_repository="${DRAGON_GIT_REPOSITORY:-https://github.com/aberration-technology/burn_dragon.git}"
dragon_git_ref="${DRAGON_GIT_REF:-main}"
head_mirror_reinstall="${HEAD_MIRROR_REINSTALL:-true}"

if [ -z "$bootstrap_auth_feature" ]; then
  case "$auth_connector_kind" in
    github) bootstrap_auth_feature="auth-github" ;;
    oidc) bootstrap_auth_feature="auth-oidc" ;;
    oauth) bootstrap_auth_feature="auth-oauth" ;;
    external) bootstrap_auth_feature="auth-external" ;;
    *) bootstrap_auth_feature="auth-static" ;;
  esac
fi

if [ -z "$bootstrap_features" ]; then
  bootstrap_features="admin-http,metrics,metrics-indexer,artifact-publish,artifact-download,artifact-fs,artifact-s3,browser-edge,browser-join,${bootstrap_auth_feature},rbac,social"
fi

tmpdir="$(mktemp -d)"
bootstrap_object_uri="s3://${artifact_bucket_name}/${runtime_config_prefix}/bootstrap.json"
caddy_object_uri="s3://${artifact_bucket_name}/${runtime_config_prefix}/Caddyfile"
head_mirror_config_object_uri="s3://${artifact_bucket_name}/${runtime_config_prefix}/bootstrap-head-mirror.toml"
head_mirror_auth_script_object_uri="s3://${artifact_bucket_name}/${runtime_config_prefix}/burn-dragon-p2p-fetch-head-mirror-auth-bundle"
head_mirror_service_object_uri="s3://${artifact_bucket_name}/${runtime_config_prefix}/burn-dragon-p2p-head-mirror.service"

cleanup() {
  aws s3 rm "$bootstrap_object_uri" >/dev/null 2>&1 || true
  aws s3 rm "$caddy_object_uri" >/dev/null 2>&1 || true
  aws s3 rm "$head_mirror_config_object_uri" >/dev/null 2>&1 || true
  aws s3 rm "$head_mirror_auth_script_object_uri" >/dev/null 2>&1 || true
  aws s3 rm "$head_mirror_service_object_uri" >/dev/null 2>&1 || true
  rm -rf "$tmpdir"
}
trap cleanup EXIT

printf '%s' "$bootstrap_config_json_b64" | base64 -d >"$tmpdir/bootstrap.json"
printf '%s' "$caddyfile_b64" | base64 -d >"$tmpdir/Caddyfile"
printf '%s' "$bootstrap_head_mirror_config_b64" | base64 -d >"$tmpdir/bootstrap-head-mirror.toml"
printf '%s' "$bootstrap_head_mirror_auth_script_b64" | base64 -d >"$tmpdir/burn-dragon-p2p-fetch-head-mirror-auth-bundle"
printf '%s' "$bootstrap_head_mirror_service_unit_b64" | base64 -d >"$tmpdir/burn-dragon-p2p-head-mirror.service"

aws s3 cp "$tmpdir/bootstrap.json" "$bootstrap_object_uri" >/dev/null
aws s3 cp "$tmpdir/Caddyfile" "$caddy_object_uri" >/dev/null
aws s3 cp "$tmpdir/bootstrap-head-mirror.toml" "$head_mirror_config_object_uri" >/dev/null
aws s3 cp "$tmpdir/burn-dragon-p2p-fetch-head-mirror-auth-bundle" "$head_mirror_auth_script_object_uri" >/dev/null
aws s3 cp "$tmpdir/burn-dragon-p2p-head-mirror.service" "$head_mirror_service_object_uri" >/dev/null

ssm_status=""
for attempt in $(seq 1 60); do
  ssm_status="$(aws ssm describe-instance-information \
    --region "$aws_region" \
    --filters "Key=InstanceIds,Values=$instance_id" \
    --query 'InstanceInformationList[0].PingStatus' \
    --output text 2>/dev/null || true)"
  if [ "${ssm_status:-}" = "Online" ]; then
    break
  fi
  sleep 10
done

if [ "${ssm_status:-}" != "Online" ]; then
  echo "bootstrap instance did not reach SSM Online status; cannot sync runtime config" >&2
  exit 1
fi

params_json="$(BOOTSTRAP_OBJECT_URI="$bootstrap_object_uri" \
  CADDY_OBJECT_URI="$caddy_object_uri" \
  HEAD_MIRROR_CONFIG_OBJECT_URI="$head_mirror_config_object_uri" \
  HEAD_MIRROR_AUTH_SCRIPT_OBJECT_URI="$head_mirror_auth_script_object_uri" \
  HEAD_MIRROR_SERVICE_OBJECT_URI="$head_mirror_service_object_uri" \
  BOOTSTRAP_INSTALL_SOURCE="$bootstrap_install_source" \
  BOOTSTRAP_CRATE_VERSION="$bootstrap_crate_version" \
  BOOTSTRAP_GIT_REPOSITORY="$bootstrap_git_repository" \
  BOOTSTRAP_GIT_REF="$bootstrap_git_ref" \
  BOOTSTRAP_FEATURES="$bootstrap_features" \
  BOOTSTRAP_REINSTALL="$bootstrap_reinstall" \
  DRAGON_GIT_REPOSITORY="$dragon_git_repository" \
  DRAGON_GIT_REF="$dragon_git_ref" \
  HEAD_MIRROR_REINSTALL="$head_mirror_reinstall" \
  python3 - <<'PY'
import json
import os

bootstrap_install_source = os.environ["BOOTSTRAP_INSTALL_SOURCE"]
bootstrap_crate_version = os.environ["BOOTSTRAP_CRATE_VERSION"]
bootstrap_git_repository = os.environ["BOOTSTRAP_GIT_REPOSITORY"]
bootstrap_git_ref = os.environ["BOOTSTRAP_GIT_REF"]
bootstrap_features = os.environ["BOOTSTRAP_FEATURES"]
bootstrap_reinstall = os.environ["BOOTSTRAP_REINSTALL"].lower() in {
    "1",
    "true",
    "yes",
    "on",
}
dragon_git_repository = os.environ["DRAGON_GIT_REPOSITORY"]
dragon_git_ref = os.environ["DRAGON_GIT_REF"]
head_mirror_reinstall = os.environ["HEAD_MIRROR_REINSTALL"].lower() in {
    "1",
    "true",
    "yes",
    "on",
}

commands = [
    "set -eu",
    "cloud-init status --wait || true",
    "systemctl reset-failed burn-p2p-bootstrap || true",
    "systemctl reset-failed burn-dragon-p2p-head-mirror || true",
    "aws s3 cp '{}' /etc/burn-dragon-p2p/bootstrap.json".format(os.environ["BOOTSTRAP_OBJECT_URI"]),
    "aws s3 cp '{}' /etc/caddy/Caddyfile".format(os.environ["CADDY_OBJECT_URI"]),
    "aws s3 cp '{}' /etc/burn_dragon_p2p/bootstrap-head-mirror.toml".format(os.environ["HEAD_MIRROR_CONFIG_OBJECT_URI"]),
    "aws s3 cp '{}' /usr/local/bin/burn-dragon-p2p-fetch-head-mirror-auth-bundle".format(os.environ["HEAD_MIRROR_AUTH_SCRIPT_OBJECT_URI"]),
    "aws s3 cp '{}' /etc/systemd/system/burn-dragon-p2p-head-mirror.service".format(os.environ["HEAD_MIRROR_SERVICE_OBJECT_URI"]),
    "chmod 0644 /etc/burn-dragon-p2p/bootstrap.json /etc/caddy/Caddyfile /etc/burn_dragon_p2p/bootstrap-head-mirror.toml /etc/systemd/system/burn-dragon-p2p-head-mirror.service",
    "chmod 0755 /usr/local/bin/burn-dragon-p2p-fetch-head-mirror-auth-bundle",
    "/usr/local/bin/burn-dragon-p2p-sync-secrets",
    "systemctl restart caddy",
    "systemctl restart burn-p2p-bootstrap",
    "systemctl daemon-reload",
    "systemctl enable burn-dragon-p2p-head-mirror",
    "systemctl restart burn-dragon-p2p-head-mirror || true",
    "systemctl is-active caddy",
    "systemctl is-active burn-p2p-bootstrap",
    "journalctl -u caddy -u burn-p2p-bootstrap -u burn-dragon-p2p-head-mirror --no-pager -n 60 || true",
]
if bootstrap_reinstall:
    install_command = (
        "export HOME=/root CARGO_HOME=/root/.cargo RUSTUP_HOME=/root/.rustup; "
        ". /root/.cargo/env; "
    )
    if bootstrap_install_source == "crate":
        install_command += (
            "cargo install --locked burn_p2p_bootstrap "
            f"--version '{bootstrap_crate_version}' "
            "--bin burn-p2p-bootstrap --no-default-features "
            f"--features '{bootstrap_features}'"
        )
    else:
        if not bootstrap_git_ref:
            raise SystemExit("BOOTSTRAP_GIT_REF is required when BOOTSTRAP_INSTALL_SOURCE=git")
        install_command += (
            "cargo install --locked "
            f"--git '{bootstrap_git_repository}' "
            f"--rev '{bootstrap_git_ref}' "
            "burn_p2p_bootstrap "
            "--bin burn-p2p-bootstrap --no-default-features "
            f"--features '{bootstrap_features}'"
        )
    commands[1:1] = [
        "ready=0; for attempt in $(seq 1 180); do if [ -x /usr/local/bin/burn-p2p-bootstrap ]; then ready=1; break; fi; sleep 5; done; if [ \"$ready\" -ne 1 ]; then echo 'burn-p2p-bootstrap executable was not ready before runtime sync' >&2; exit 1; fi",
        install_command,
        "ln -sf /root/.cargo/bin/burn-p2p-bootstrap /usr/local/bin/burn-p2p-bootstrap",
    ]
if head_mirror_reinstall:
    commands[1:1] = [
        "export HOME=/root CARGO_HOME=/root/.cargo RUSTUP_HOME=/root/.rustup; . /root/.cargo/env; cargo install --locked --git '{}' --rev '{}' burn_dragon_p2p --bin burn_dragon_p2p_native --no-default-features --features native".format(
            dragon_git_repository,
            dragon_git_ref,
        ),
        "ln -sf /root/.cargo/bin/burn_dragon_p2p_native /usr/local/bin/burn_dragon_p2p_native",
    ]
print(json.dumps({"commands": commands}))
PY
)"

command_id="$(aws ssm send-command \
  --region "$aws_region" \
  --instance-ids "$instance_id" \
  --document-name AWS-RunShellScript \
  --comment "sync burn_dragon bootstrap runtime config" \
  --parameters "$params_json" \
  --query 'Command.CommandId' \
  --output text)"

for attempt in $(seq 1 30); do
  invocation_status="$(aws ssm get-command-invocation \
    --region "$aws_region" \
    --command-id "$command_id" \
    --instance-id "$instance_id" \
    --query 'Status' \
    --output text 2>/dev/null || true)"
  case "$invocation_status" in
    Success)
      break
      ;;
    Cancelled|TimedOut|Failed|Cancelling)
      echo "bootstrap runtime config sync failed with status ${invocation_status}" >&2
      aws ssm get-command-invocation \
        --region "$aws_region" \
        --command-id "$command_id" \
        --instance-id "$instance_id" \
        --output json || true
      exit 1
      ;;
  esac
  sleep 5
done

aws ssm get-command-invocation \
  --region "$aws_region" \
  --command-id "$command_id" \
  --instance-id "$instance_id" \
  --output json
