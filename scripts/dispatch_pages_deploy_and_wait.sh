#!/usr/bin/env bash
set -euo pipefail

exec "${CARGO:-cargo}" run -p xtask -- dispatch-pages-deploy-and-wait
