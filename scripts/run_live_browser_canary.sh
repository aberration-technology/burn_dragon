#!/usr/bin/env bash
set -euo pipefail

: "${BURN_DRAGON_BROWSER_CANARY_ARTIFACT_DIR:?BURN_DRAGON_BROWSER_CANARY_ARTIFACT_DIR must be set}"
: "${BURN_DRAGON_BROWSER_CANARY_OUTPUT_JSON:?BURN_DRAGON_BROWSER_CANARY_OUTPUT_JSON must be set}"

mkdir -p "$BURN_DRAGON_BROWSER_CANARY_ARTIFACT_DIR"

if [ -n "${GITHUB_OUTPUT:-}" ]; then
  {
    echo "artifact_dir=$BURN_DRAGON_BROWSER_CANARY_ARTIFACT_DIR"
    echo "report_path=$BURN_DRAGON_BROWSER_CANARY_OUTPUT_JSON"
  } >>"$GITHUB_OUTPUT"
fi

node scripts/live-browser-canary.mjs
