#!/usr/bin/env bash
set -euo pipefail

npx --yes playwright --version
npx --yes playwright install --with-deps chromium
