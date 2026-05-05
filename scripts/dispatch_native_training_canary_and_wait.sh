#!/usr/bin/env bash
set -euo pipefail

: "${GH_TOKEN:?GH_TOKEN must be set}"
: "${BURN_DRAGON_NATIVE_CANARY_ENVIRONMENT:?BURN_DRAGON_NATIVE_CANARY_ENVIRONMENT must be set}"
: "${BURN_DRAGON_NATIVE_CANARY_EDGE_BASE_URL:?BURN_DRAGON_NATIVE_CANARY_EDGE_BASE_URL must be set}"
: "${BURN_DRAGON_NATIVE_CANARY_EXPERIMENT_KIND:?BURN_DRAGON_NATIVE_CANARY_EXPERIMENT_KIND must be set}"
: "${BURN_DRAGON_NATIVE_CANARY_EXPERIMENT_ID:?BURN_DRAGON_NATIVE_CANARY_EXPERIMENT_ID must be set}"
: "${BURN_DRAGON_NATIVE_CANARY_BACKEND:?BURN_DRAGON_NATIVE_CANARY_BACKEND must be set}"
: "${BURN_DRAGON_NATIVE_CANARY_PRINCIPAL_ID:?BURN_DRAGON_NATIVE_CANARY_PRINCIPAL_ID must be set}"
: "${BURN_DRAGON_NATIVE_CANARY_WINDOWS:?BURN_DRAGON_NATIVE_CANARY_WINDOWS must be set}"
: "${BURN_DRAGON_NATIVE_CANARY_SETTLE_DIFFUSION:=true}"
: "${BURN_DRAGON_NATIVE_CANARY_DIFFUSION_SETTLE_PASSES:=3}"
: "${BURN_DRAGON_NATIVE_CANARY_SERVE_AFTER_PUBLISH_SECS:=120}"
: "${BURN_DRAGON_NATIVE_CANARY_COMMAND_TIMEOUT_SECS:=1800}"

dispatch_started_at="$(date -u +%s)"

gh workflow run .github/workflows/live-native-training-canary.yml \
  --repo "${GITHUB_REPOSITORY}" \
  --ref "${GITHUB_REF_NAME}" \
  -f environment="${BURN_DRAGON_NATIVE_CANARY_ENVIRONMENT}" \
  -f edge_base_url="${BURN_DRAGON_NATIVE_CANARY_EDGE_BASE_URL}" \
  -f experiment_kind="${BURN_DRAGON_NATIVE_CANARY_EXPERIMENT_KIND}" \
  -f experiment_id="${BURN_DRAGON_NATIVE_CANARY_EXPERIMENT_ID}" \
  -f backend="${BURN_DRAGON_NATIVE_CANARY_BACKEND}" \
  -f principal_id="${BURN_DRAGON_NATIVE_CANARY_PRINCIPAL_ID}" \
  -f windows="${BURN_DRAGON_NATIVE_CANARY_WINDOWS}" \
  -f settle_diffusion="${BURN_DRAGON_NATIVE_CANARY_SETTLE_DIFFUSION}" \
  -f diffusion_settle_passes="${BURN_DRAGON_NATIVE_CANARY_DIFFUSION_SETTLE_PASSES}" \
  -f serve_after_publish_secs="${BURN_DRAGON_NATIVE_CANARY_SERVE_AFTER_PUBLISH_SECS}" \
  -f command_timeout_secs="${BURN_DRAGON_NATIVE_CANARY_COMMAND_TIMEOUT_SECS}"

native_canary_run_id=""
for _ in $(seq 1 30); do
  native_canary_run_id="$(gh run list \
    --repo "${GITHUB_REPOSITORY}" \
    --workflow .github/workflows/live-native-training-canary.yml \
    --limit 10 \
    --json databaseId,createdAt,headBranch \
    | DISPATCH_STARTED_AT="$dispatch_started_at" GITHUB_REF_NAME="$GITHUB_REF_NAME" python3 -c 'import datetime, json, os, sys; runs = json.load(sys.stdin); after = int(os.environ["DISPATCH_STARTED_AT"]); branch = os.environ["GITHUB_REF_NAME"]; matches = [run for run in runs if run.get("headBranch") == branch and int(datetime.datetime.fromisoformat(run["createdAt"].replace("Z", "+00:00")).timestamp()) >= after]; matches.sort(key=lambda run: run.get("databaseId", 0), reverse=True); print(matches[0]["databaseId"] if matches else "")')"
  if [ -n "$native_canary_run_id" ]; then
    break
  fi
  sleep 5
done

if [ -z "$native_canary_run_id" ]; then
  echo "failed to discover live-native-training-canary run dispatched for branch $GITHUB_REF_NAME" >&2
  exit 1
fi

if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "run_id=$native_canary_run_id" >>"$GITHUB_OUTPUT"
fi

gh run watch "$native_canary_run_id" \
  --repo "${GITHUB_REPOSITORY}" \
  --interval 30 \
  --exit-status
