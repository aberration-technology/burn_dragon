#!/usr/bin/env bash

set -euo pipefail

aws_region="${AWS_REGION:?AWS_REGION is required}"
instance_id="${BOOTSTRAP_INSTANCE_ID:?BOOTSTRAP_INSTANCE_ID is required}"
artifact_bucket_name="${ARTIFACT_BUCKET_NAME:?ARTIFACT_BUCKET_NAME is required}"
bootstrap_config_json_b64="${BOOTSTRAP_CONFIG_JSON_B64:?BOOTSTRAP_CONFIG_JSON_B64 is required}"
caddyfile_b64="${CADDYFILE_B64:?CADDYFILE_B64 is required}"
runtime_config_prefix="${RUNTIME_CONFIG_PREFIX:-runtime-config/bootstrap/${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-0}}"

tmpdir="$(mktemp -d)"
bootstrap_object_uri="s3://${artifact_bucket_name}/${runtime_config_prefix}/bootstrap.json"
caddy_object_uri="s3://${artifact_bucket_name}/${runtime_config_prefix}/Caddyfile"

cleanup() {
  aws s3 rm "$bootstrap_object_uri" >/dev/null 2>&1 || true
  aws s3 rm "$caddy_object_uri" >/dev/null 2>&1 || true
  rm -rf "$tmpdir"
}
trap cleanup EXIT

printf '%s' "$bootstrap_config_json_b64" | base64 -d >"$tmpdir/bootstrap.json"
printf '%s' "$caddyfile_b64" | base64 -d >"$tmpdir/Caddyfile"

aws s3 cp "$tmpdir/bootstrap.json" "$bootstrap_object_uri" >/dev/null
aws s3 cp "$tmpdir/Caddyfile" "$caddy_object_uri" >/dev/null

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

params_json="$(BOOTSTRAP_OBJECT_URI="$bootstrap_object_uri" CADDY_OBJECT_URI="$caddy_object_uri" python3 - <<'PY'
import json
import os

commands = [
    "set -eu",
    "aws s3 cp '{}' /etc/burn-dragon-p2p/bootstrap.json".format(os.environ["BOOTSTRAP_OBJECT_URI"]),
    "aws s3 cp '{}' /etc/caddy/Caddyfile".format(os.environ["CADDY_OBJECT_URI"]),
    "chmod 0644 /etc/burn-dragon-p2p/bootstrap.json /etc/caddy/Caddyfile",
    "/usr/local/bin/burn-dragon-p2p-sync-secrets",
    "systemctl restart caddy",
    "systemctl restart burn-p2p-bootstrap",
    "systemctl is-active caddy",
    "systemctl is-active burn-p2p-bootstrap",
    "journalctl -u caddy -u burn-p2p-bootstrap --no-pager -n 60 || true",
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
