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
bootstrap_binary_path="${BOOTSTRAP_BINARY_PATH:-}"
auth_connector_kind="${AUTH_CONNECTOR_KIND:-github}"
bootstrap_auth_feature="${BOOTSTRAP_AUTH_FEATURE:-}"
bootstrap_features="${BOOTSTRAP_FEATURES:-}"
bootstrap_reinstall="${BOOTSTRAP_REINSTALL:-false}"
dragon_git_repository="${DRAGON_GIT_REPOSITORY:-https://github.com/aberration-technology/burn_dragon.git}"
dragon_git_ref="${DRAGON_GIT_REF:-main}"
head_mirror_binary_path="${HEAD_MIRROR_BINARY_PATH:-}"
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
bootstrap_binary_object_uri="s3://${artifact_bucket_name}/${runtime_config_prefix}/burn-p2p-bootstrap"
head_mirror_binary_object_uri="s3://${artifact_bucket_name}/${runtime_config_prefix}/burn_dragon_p2p_native"

if [ -z "$bootstrap_binary_path" ]; then
  bootstrap_binary_object_uri=""
fi
if [ -z "$head_mirror_binary_path" ]; then
  head_mirror_binary_object_uri=""
fi

cleanup() {
  aws s3 rm "$bootstrap_object_uri" >/dev/null 2>&1 || true
  aws s3 rm "$caddy_object_uri" >/dev/null 2>&1 || true
  aws s3 rm "$head_mirror_config_object_uri" >/dev/null 2>&1 || true
  aws s3 rm "$head_mirror_auth_script_object_uri" >/dev/null 2>&1 || true
  aws s3 rm "$head_mirror_service_object_uri" >/dev/null 2>&1 || true
  if [ -n "$bootstrap_binary_path" ]; then
    aws s3 rm "$bootstrap_binary_object_uri" >/dev/null 2>&1 || true
  fi
  if [ -n "$head_mirror_binary_path" ]; then
    aws s3 rm "$head_mirror_binary_object_uri" >/dev/null 2>&1 || true
  fi
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
if [ -n "$bootstrap_binary_path" ]; then
  aws s3 cp "$bootstrap_binary_path" "$bootstrap_binary_object_uri" >/dev/null
fi
if [ -n "$head_mirror_binary_path" ]; then
  aws s3 cp "$head_mirror_binary_path" "$head_mirror_binary_object_uri" >/dev/null
fi

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
  BOOTSTRAP_BINARY_OBJECT_URI="$bootstrap_binary_object_uri" \
  BOOTSTRAP_FEATURES="$bootstrap_features" \
  BOOTSTRAP_REINSTALL="$bootstrap_reinstall" \
  DRAGON_GIT_REPOSITORY="$dragon_git_repository" \
  DRAGON_GIT_REF="$dragon_git_ref" \
  HEAD_MIRROR_BINARY_OBJECT_URI="$head_mirror_binary_object_uri" \
  HEAD_MIRROR_REINSTALL="$head_mirror_reinstall" \
  python3 "$(dirname "$0")/render_bootstrap_runtime_sync_commands.py"
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
