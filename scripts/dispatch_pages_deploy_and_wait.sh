#!/usr/bin/env bash
set -euo pipefail

: "${GH_TOKEN:?GH_TOKEN must be set}"
: "${BURN_DRAGON_DEPLOY_PAGES_ENVIRONMENT:?BURN_DRAGON_DEPLOY_PAGES_ENVIRONMENT must be set}"
: "${BURN_DRAGON_DEPLOY_PAGES_EDGE_BASE_URL:?BURN_DRAGON_DEPLOY_PAGES_EDGE_BASE_URL must be set}"
: "${BURN_DRAGON_DEPLOY_PAGES_EXPERIMENT_ID:?BURN_DRAGON_DEPLOY_PAGES_EXPERIMENT_ID must be set}"
: "${BURN_DRAGON_DEPLOY_PAGES_REVISION_ID:?BURN_DRAGON_DEPLOY_PAGES_REVISION_ID must be set}"
: "${BURN_DRAGON_DEPLOY_PAGES_REQUIRE_EDGE_AUTH:?BURN_DRAGON_DEPLOY_PAGES_REQUIRE_EDGE_AUTH must be set}"

python3 scripts/agent_task.py gh-dispatch \
  --repo "${GITHUB_REPOSITORY}" \
  --workflow .github/workflows/deploy-pages.yml \
  --ref "${GITHUB_REF_NAME}" \
  --label deploy-pages \
  --input environment="${BURN_DRAGON_DEPLOY_PAGES_ENVIRONMENT}" \
  --input edge_base_url="${BURN_DRAGON_DEPLOY_PAGES_EDGE_BASE_URL}" \
  --input selected_experiment_id="${BURN_DRAGON_DEPLOY_PAGES_EXPERIMENT_ID}" \
  --input selected_revision_id="${BURN_DRAGON_DEPLOY_PAGES_REVISION_ID}" \
  --input require_edge_auth="${BURN_DRAGON_DEPLOY_PAGES_REQUIRE_EDGE_AUTH}" \
  --wait \
  --interval-secs "${BURN_DRAGON_DEPLOY_PAGES_WATCH_INTERVAL_SECS:-180}" \
  --exit-status
