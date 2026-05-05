#!/usr/bin/env bash
set -euo pipefail

exec "${CARGO:-cargo}" run -p xtask -- dispatch-native-training-canary-and-wait
