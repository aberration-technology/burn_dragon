#!/usr/bin/env bash
set -euo pipefail

: "${GH_TOKEN:?GH_TOKEN must be set}"
: "${BURN_DRAGON_DEPLOY_PAGES_ENVIRONMENT:?BURN_DRAGON_DEPLOY_PAGES_ENVIRONMENT must be set}"
: "${BURN_DRAGON_DEPLOY_PAGES_EDGE_BASE_URL:?BURN_DRAGON_DEPLOY_PAGES_EDGE_BASE_URL must be set}"
: "${BURN_DRAGON_DEPLOY_PAGES_EXPERIMENT_ID:?BURN_DRAGON_DEPLOY_PAGES_EXPERIMENT_ID must be set}"
: "${BURN_DRAGON_DEPLOY_PAGES_REVISION_ID:?BURN_DRAGON_DEPLOY_PAGES_REVISION_ID must be set}"
: "${BURN_DRAGON_DEPLOY_PAGES_REQUIRE_EDGE_AUTH:?BURN_DRAGON_DEPLOY_PAGES_REQUIRE_EDGE_AUTH must be set}"

dispatch_started_at="$(date -u +%s)"

gh workflow run .github/workflows/deploy-pages.yml \
  --repo "${GITHUB_REPOSITORY}" \
  --ref "${GITHUB_REF_NAME}" \
  -f environment="${BURN_DRAGON_DEPLOY_PAGES_ENVIRONMENT}" \
  -f edge_base_url="${BURN_DRAGON_DEPLOY_PAGES_EDGE_BASE_URL}" \
  -f selected_experiment_id="${BURN_DRAGON_DEPLOY_PAGES_EXPERIMENT_ID}" \
  -f selected_revision_id="${BURN_DRAGON_DEPLOY_PAGES_REVISION_ID}" \
  -f require_edge_auth="${BURN_DRAGON_DEPLOY_PAGES_REQUIRE_EDGE_AUTH}"

pages_run_id=""
for _ in $(seq 1 30); do
  pages_run_id="$(gh run list \
    --repo "${GITHUB_REPOSITORY}" \
    --workflow .github/workflows/deploy-pages.yml \
    --limit 10 \
    --json databaseId,createdAt,headBranch \
    | DISPATCH_STARTED_AT="$dispatch_started_at" GITHUB_REF_NAME="$GITHUB_REF_NAME" python3 -c 'import datetime, json, os, sys; runs = json.load(sys.stdin); after = int(os.environ["DISPATCH_STARTED_AT"]); branch = os.environ["GITHUB_REF_NAME"]; matches = [run for run in runs if run.get("headBranch") == branch and int(datetime.datetime.fromisoformat(run["createdAt"].replace("Z", "+00:00")).timestamp()) >= after]; matches.sort(key=lambda run: run.get("databaseId", 0), reverse=True); print(matches[0]["databaseId"] if matches else "")')"
  if [ -n "$pages_run_id" ]; then
    break
  fi
  sleep 5
done

if [ -z "$pages_run_id" ]; then
  echo "failed to discover deploy-pages run dispatched for branch $GITHUB_REF_NAME" >&2
  exit 1
fi

if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "run_id=$pages_run_id" >>"$GITHUB_OUTPUT"
fi

python3 scripts/summarize_github_run.py \
  --repo "${GITHUB_REPOSITORY}" \
  --run-id "$pages_run_id" \
  --watch \
  --interval-secs "${BURN_DRAGON_DEPLOY_PAGES_WATCH_INTERVAL_SECS:-180}" \
  --exit-status
