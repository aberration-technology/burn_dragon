#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE_DIR="$ROOT_DIR/crates/burn_dragon_p2p/deploy/profiles"
ARMS_CSV="block512_reset,block512_carry,chunk128_reset,chunk128_carry"
SEEDS_CSV="1337,7331,4242"
MAX_ITERS=512
BATCH_SIZE=32
LOG_FREQUENCY=32
CHECKPOINT_INTERVAL_ITERS=512
BACKEND="cuda"
FEATURES="train,cuda"
BUILD_RELEASE=1
ALLOW_STALE_BINARY=0
DRY_RUN=0
TIMEOUT_SECONDS=3600
SAMPLE_INTERVAL_SECONDS=2
GPU_TELEMETRY_SECONDS=1
PROBE_ITEMS=""
PROBE_TOKENS=""
PROBE_MIN_BATCH_ROWS=""
PROBE_MAX_PROMPT_SPAN=""
PROBE_DEVICE_BUFFER_TOKENS=""
PROBE_MAX_IN_FLIGHT_ROWS=""
MAX_SYSTEM_MEMORY_FRACTION="${BURN_DRAGON_STATEFUL_TBPTT_MAX_SYSTEM_MEMORY_FRACTION:-0.80}"
MIN_AVAILABLE_MB="${BURN_DRAGON_STATEFUL_TBPTT_MIN_AVAILABLE_MB:-24576}"
OUT_DIR="$ROOT_DIR/target/experiments/stateful-tbptt/$(date +%Y%m%d-%H%M%S)"

usage() {
  cat <<'USAGE'
Usage: scripts/stateful_tbptt_matrix.sh [options]

  --arms <csv>              block512_reset, block512_carry, chunk128_reset,
                            chunk128_carry, and/or chunk64_carry.
  --seeds <csv>             Deterministic training seeds. Default: 1337,7331,4242.
  --max-iters <n>           Updates per trial. Default: 512.
  --batch-size <n>          Fixed micro-batch for every arm. Default: 32.
  --log-frequency <n>       Training metric cadence. Default: 32.
  --checkpoint-interval <n> Checkpoint cadence. Default: max-iters.
  --backend <cuda|cpu>      Default: cuda.
  --features <csv>          Cargo features. Default: train,cuda.
  --out-dir <path>          Matrix artifact directory.
  --timeout-seconds <n>     Per-trial wall timeout. Default: 3600.
  --probe-items <n>         Override free-run verifier items; zero disables it.
  --probe-tokens <n>        Override the base free-run generation budget.
  --probe-min-batch-rows <n> Override the buffered cohort minimum.
  --probe-max-prompt-span <n> Override the ragged cohort prompt-length span.
  --probe-device-buffer <n> Override accelerator-resident greedy steps.
  --probe-max-in-flight <n> Override evaluator rows resident at once.
  --no-build                Reuse a current release binary.
  --allow-stale-binary      Permit --no-build with newer Rust sources.
  --dry-run                 Materialize overlays/manifests without training.

Safety environment variables:
  BURN_DRAGON_STATEFUL_TBPTT_MAX_SYSTEM_MEMORY_FRACTION  Default: 0.80
  BURN_DRAGON_STATEFUL_TBPTT_MIN_AVAILABLE_MB            Default: 24576
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arms) ARMS_CSV="$2"; shift 2 ;;
    --seeds) SEEDS_CSV="$2"; shift 2 ;;
    --max-iters) MAX_ITERS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --log-frequency) LOG_FREQUENCY="$2"; shift 2 ;;
    --checkpoint-interval) CHECKPOINT_INTERVAL_ITERS="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --features) FEATURES="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --timeout-seconds) TIMEOUT_SECONDS="$2"; shift 2 ;;
    --probe-items) PROBE_ITEMS="$2"; shift 2 ;;
    --probe-tokens) PROBE_TOKENS="$2"; shift 2 ;;
    --probe-min-batch-rows) PROBE_MIN_BATCH_ROWS="$2"; shift 2 ;;
    --probe-max-prompt-span) PROBE_MAX_PROMPT_SPAN="$2"; shift 2 ;;
    --probe-device-buffer) PROBE_DEVICE_BUFFER_TOKENS="$2"; shift 2 ;;
    --probe-max-in-flight) PROBE_MAX_IN_FLIGHT_ROWS="$2"; shift 2 ;;
    --no-build) BUILD_RELEASE=0; shift ;;
    --allow-stale-binary) ALLOW_STALE_BINARY=1; shift ;;
    --dry-run) DRY_RUN=1; BUILD_RELEASE=0; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$BACKEND" == "cpu" && "$FEATURES" == "train,cuda" ]]; then
  FEATURES="train"
fi
for value in "$MAX_ITERS" "$BATCH_SIZE" "$LOG_FREQUENCY" "$CHECKPOINT_INTERVAL_ITERS"; do
  if (( value <= 0 )); then
    echo "integer matrix settings must be > 0" >&2
    exit 2
  fi
done
for value in "$PROBE_ITEMS" "$PROBE_TOKENS" "$PROBE_MIN_BATCH_ROWS" \
  "$PROBE_MAX_PROMPT_SPAN" "$PROBE_DEVICE_BUFFER_TOKENS" "$PROBE_MAX_IN_FLIGHT_ROWS"; do
  if [[ -n "$value" && ! "$value" =~ ^[0-9]+$ ]]; then
    echo "probe overrides must be non-negative integers" >&2
    exit 2
  fi
done
for value in "$PROBE_TOKENS" "$PROBE_MIN_BATCH_ROWS" "$PROBE_MAX_PROMPT_SPAN" \
  "$PROBE_DEVICE_BUFFER_TOKENS" "$PROBE_MAX_IN_FLIGHT_ROWS"; do
  if [[ -n "$value" ]] && (( value <= 0 )); then
    echo "enabled probe generation overrides must be > 0" >&2
    exit 2
  fi
done

profile_for_arm() {
  case "$1" in
    block512_reset) echo "$PROFILE_DIR/ruliad-r3.stateful-tbptt-block512-reset.toml" ;;
    block512_carry) echo "$PROFILE_DIR/ruliad-r3.stateful-tbptt-block512-carry.toml" ;;
    chunk128_reset) echo "$PROFILE_DIR/ruliad-r3.stateful-tbptt-chunk128-reset.toml" ;;
    chunk128_carry) echo "$PROFILE_DIR/ruliad-r3.stateful-tbptt-chunk128-carry.toml" ;;
    chunk64_carry) echo "$PROFILE_DIR/ruliad-r3.stateful-tbptt-chunk64-carry.toml" ;;
    *) echo "unknown stateful TBPTT arm: $1" >&2; return 2 ;;
  esac
}

mem_total_kb() {
  awk '/^MemTotal:/ {print $2}' /proc/meminfo
}

mem_available_kb() {
  awk '/^MemAvailable:/ {print $2}' /proc/meminfo
}

fraction_to_bps() {
  awk -v value="$1" 'BEGIN { printf "%d", value * 10000 }'
}

latest_run_dir() {
  find "$1" -mindepth 1 -maxdepth 4 -type f -name dashboard.md -printf '%T@ %h\n' 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

kill_trial() {
  local pid="$1"
  kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  sleep 3
  kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
}

MONITOR_STATUS="not_started"
MONITOR_ELAPSED_SECONDS=0
MONITOR_PEAK_USED_MB=0
MONITOR_MIN_AVAILABLE_MB=0

monitor_trial() {
  local pid="$1"
  local log_path="$2"
  local total_kb available_kb used_kb peak_used_kb min_seen_kb
  local started now elapsed exit_code status max_fraction_bps min_available_kb
  total_kb="$(mem_total_kb)"
  available_kb="$(mem_available_kb)"
  peak_used_kb=$((total_kb - available_kb))
  min_seen_kb="$available_kb"
  max_fraction_bps="$(fraction_to_bps "$MAX_SYSTEM_MEMORY_FRACTION")"
  min_available_kb=$((MIN_AVAILABLE_MB * 1024))
  started="$(date +%s)"
  status="ok"

  while kill -0 "$pid" 2>/dev/null; do
    available_kb="$(mem_available_kb)"
    used_kb=$((total_kb - available_kb))
    (( used_kb > peak_used_kb )) && peak_used_kb="$used_kb"
    (( available_kb < min_seen_kb )) && min_seen_kb="$available_kb"
    if (( used_kb * 10000 > total_kb * max_fraction_bps )); then
      status="killed_ram_fraction"
      echo "RAM guard: used=${used_kb}KiB limit_fraction=$MAX_SYSTEM_MEMORY_FRACTION" >> "$log_path"
      kill_trial "$pid"
      break
    fi
    if (( available_kb < min_available_kb )); then
      status="killed_low_available_ram"
      echo "RAM guard: available=${available_kb}KiB floor=${min_available_kb}KiB" >> "$log_path"
      kill_trial "$pid"
      break
    fi
    now="$(date +%s)"
    elapsed=$((now - started))
    if (( TIMEOUT_SECONDS > 0 && elapsed > TIMEOUT_SECONDS )); then
      status="killed_timeout"
      echo "timeout guard: elapsed=${elapsed}s limit=${TIMEOUT_SECONDS}s" >> "$log_path"
      kill_trial "$pid"
      break
    fi
    sleep "$SAMPLE_INTERVAL_SECONDS"
  done

  set +e
  wait "$pid"
  exit_code=$?
  set -e
  if (( exit_code != 0 )) && [[ "$status" == "ok" ]]; then
    status="failed_exit_${exit_code}"
  fi
  now="$(date +%s)"
  MONITOR_STATUS="$status"
  MONITOR_ELAPSED_SECONDS=$((now - started))
  MONITOR_PEAK_USED_MB=$((peak_used_kb / 1024))
  MONITOR_MIN_AVAILABLE_MB=$((min_seen_kb / 1024))
}

write_overlay() {
  local path="$1"
  local seed="$2"
  local checkpoint_interval="$CHECKPOINT_INTERVAL_ITERS"
  (( checkpoint_interval > MAX_ITERS )) && checkpoint_interval="$MAX_ITERS"
  cat > "$path" <<EOF
[training]
seed = $seed
batch_size = $BATCH_SIZE
max_iters = $MAX_ITERS
checkpoint_interval_iters = $checkpoint_interval
log_frequency = $LOG_FREQUENCY

[training.events]
flush_every_steps = $LOG_FREQUENCY
EOF
  if [[ -n "$PROBE_ITEMS" ]]; then
    printf 'ruliad_correctness_probe_items = %s\n' "$PROBE_ITEMS" >> "$path"
  fi
  if [[ -n "$PROBE_TOKENS" ]]; then
    printf 'ruliad_correctness_probe_tokens = %s\n' "$PROBE_TOKENS" >> "$path"
  fi
  if [[ -n "$PROBE_MIN_BATCH_ROWS$PROBE_MAX_PROMPT_SPAN$PROBE_DEVICE_BUFFER_TOKENS$PROBE_MAX_IN_FLIGHT_ROWS" ]]; then
    printf '\n[training.ruliad_probe_generation]\n' >> "$path"
    if [[ -n "$PROBE_MIN_BATCH_ROWS" ]]; then
      printf 'minimum_batch_rows = %s\n' "$PROBE_MIN_BATCH_ROWS" >> "$path"
    fi
    if [[ -n "$PROBE_MAX_PROMPT_SPAN" ]]; then
      printf 'maximum_prompt_position_span = %s\n' "$PROBE_MAX_PROMPT_SPAN" >> "$path"
    fi
    if [[ -n "$PROBE_DEVICE_BUFFER_TOKENS" ]]; then
      printf 'device_buffer_tokens = %s\n' "$PROBE_DEVICE_BUFFER_TOKENS" >> "$path"
    fi
    if [[ -n "$PROBE_MAX_IN_FLIGHT_ROWS" ]]; then
      printf 'max_in_flight_rows = %s\n' "$PROBE_MAX_IN_FLIGHT_ROWS" >> "$path"
    fi
  fi
}

write_manifest() {
  local path="$1" arm="$2" seed="$3" profile="$4" overlay="$5"
  local run_root="$6" run_dir="$7" log_path="$8" gpu_log="$9" time_log="${10}"
  MANIFEST_PATH="$path" ARM="$arm" SEED="$seed" PROFILE="$profile" OVERLAY="$overlay" \
  RUN_ROOT="$run_root" RUN_DIR="$run_dir" LOG_PATH="$log_path" GPU_LOG="$gpu_log" \
  TIME_LOG="$time_log" STATUS="$MONITOR_STATUS" ELAPSED="$MONITOR_ELAPSED_SECONDS" \
  PEAK_RAM="$MONITOR_PEAK_USED_MB" MIN_AVAILABLE="$MONITOR_MIN_AVAILABLE_MB" \
  MAX_ITERS_VALUE="$MAX_ITERS" BATCH_SIZE_VALUE="$BATCH_SIZE" BACKEND_VALUE="$BACKEND" \
  ROOT_DIR_VALUE="$ROOT_DIR" python3 - <<'PY'
import json
import os
import pathlib
import subprocess

root = os.environ["ROOT_DIR_VALUE"]
def git(*args):
    return subprocess.run(
        ["git", "-C", root, *args], text=True, capture_output=True, check=False
    ).stdout.strip()

payload = {
    "schema_version": 1,
    "arm": os.environ["ARM"],
    "seed": int(os.environ["SEED"]),
    "profile": os.environ["PROFILE"],
    "overlay": os.environ["OVERLAY"],
    "run_root": os.environ["RUN_ROOT"],
    "run_dir": os.environ["RUN_DIR"],
    "log_path": os.environ["LOG_PATH"],
    "gpu_log_path": os.environ["GPU_LOG"],
    "time_log_path": os.environ["TIME_LOG"],
    "status": os.environ["STATUS"],
    "elapsed_seconds": int(os.environ["ELAPSED"]),
    "peak_used_mb": int(os.environ["PEAK_RAM"]),
    "min_available_mb": int(os.environ["MIN_AVAILABLE"]),
    "max_iters": int(os.environ["MAX_ITERS_VALUE"]),
    "batch_size": int(os.environ["BATCH_SIZE_VALUE"]),
    "backend": os.environ["BACKEND_VALUE"],
    "git_sha": git("rev-parse", "HEAD"),
    "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    "git_dirty": bool(git("status", "--porcelain")),
}
path = pathlib.Path(os.environ["MANIFEST_PATH"])
path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

RUSTUP_CARGO="$(rustup which cargo)"
RUSTUP_RUSTC="$(rustup which rustc)"
TRAIN_BINARY="$ROOT_DIR/target/release/examples/train_language"
mkdir -p "$OUT_DIR/overlays" "$OUT_DIR/logs" "$OUT_DIR/manifests" "$OUT_DIR/run_roots"
RUN_INDEX="$OUT_DIR/run-index.tsv"
printf 'arm\tseed\tstatus\telapsed_seconds\tpeak_used_mb\tmin_available_mb\trun_dir\tmanifest\n' > "$RUN_INDEX"
ARMS_CSV_VALUE="$ARMS_CSV" SEEDS_CSV_VALUE="$SEEDS_CSV" \
MAX_ITERS_VALUE="$MAX_ITERS" BATCH_SIZE_VALUE="$BATCH_SIZE" BACKEND_VALUE="$BACKEND" \
OUT_DIR_VALUE="$OUT_DIR" python3 - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "schema_version": 1,
    "requested_arms": [value for value in os.environ["ARMS_CSV_VALUE"].split(",") if value],
    "requested_seeds": [int(value) for value in os.environ["SEEDS_CSV_VALUE"].split(",") if value],
    "max_iters": int(os.environ["MAX_ITERS_VALUE"]),
    "batch_size": int(os.environ["BATCH_SIZE_VALUE"]),
    "backend": os.environ["BACKEND_VALUE"],
}
Path(os.environ["OUT_DIR_VALUE"], "matrix-config.json").write_text(
    json.dumps(payload, indent=2) + "\n"
)
PY

if (( BUILD_RELEASE == 1 )); then
  (
    cd "$ROOT_DIR"
    export CARGO="$RUSTUP_CARGO"
    export RUSTC="$RUSTUP_RUSTC"
    "$RUSTUP_CARGO" build --release -p burn_dragon_language --example train_language --features "$FEATURES"
  )
elif (( DRY_RUN == 0 )); then
  [[ -x "$TRAIN_BINARY" ]] || { echo "missing release binary: $TRAIN_BINARY" >&2; exit 2; }
  newer_source="$(find "$ROOT_DIR/crates" -type f \( -name '*.rs' -o -name Cargo.toml \) -newer "$TRAIN_BINARY" -print -quit 2>/dev/null || true)"
  if [[ -n "$newer_source" && "$ALLOW_STALE_BINARY" != "1" ]]; then
    echo "source is newer than release binary: $newer_source" >&2
    exit 2
  fi
fi

IFS=',' read -r -a ARMS <<< "$ARMS_CSV"
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
echo "stateful TBPTT matrix: out=$OUT_DIR arms=${ARMS[*]} seeds=${SEEDS[*]} max_iters=$MAX_ITERS batch=$BATCH_SIZE backend=$BACKEND"
echo "RAM guards: max_fraction=$MAX_SYSTEM_MEMORY_FRACTION min_available_mb=$MIN_AVAILABLE_MB"

for seed in "${SEEDS[@]}"; do
  for arm in "${ARMS[@]}"; do
    profile="$(profile_for_arm "$arm")"
    [[ -f "$profile" ]] || { echo "missing profile: $profile" >&2; exit 2; }
    trial="${arm}-seed${seed}-i${MAX_ITERS}-b${BATCH_SIZE}-${BACKEND}"
    overlay="$OUT_DIR/overlays/${trial}.toml"
    log_path="$OUT_DIR/logs/${trial}.log"
    gpu_log="$OUT_DIR/logs/${trial}.gpu.csv"
    time_log="$OUT_DIR/logs/${trial}.time.txt"
    manifest="$OUT_DIR/manifests/${trial}.json"
    run_root="$OUT_DIR/run_roots/${trial}"
    mkdir -p "$run_root"
    write_overlay "$overlay" "$seed"

    MONITOR_STATUS="dry_run"
    MONITOR_ELAPSED_SECONDS=0
    MONITOR_PEAK_USED_MB=0
    MONITOR_MIN_AVAILABLE_MB=$(( $(mem_available_kb) / 1024 ))
    run_dir=""
    echo "==> arm=$arm seed=$seed profile=$(basename "$profile")" | tee "$log_path"
    if (( DRY_RUN == 0 )); then
      gpu_pid=""
      if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=timestamp,index,pstate,utilization.gpu,utilization.memory,power.draw,clocks.current.sm,temperature.gpu,memory.used,memory.total --format=csv -l "$GPU_TELEMETRY_SECONDS" > "$gpu_log" 2>/dev/null &
        gpu_pid=$!
      fi
      (
        cd "$ROOT_DIR"
        export CARGO="$RUSTUP_CARGO"
        export RUSTC="$RUSTUP_RUSTC"
        export BURN_DRAGON_RUN_ROOT="$run_root"
        export DragonModel_STAGE_PROFILE=1
        exec setsid /usr/bin/time -v -o "$time_log" "$TRAIN_BINARY" \
          --backend "$BACKEND" --config "$profile" --config "$overlay"
      ) >> "$log_path" 2>&1 &
      pid=$!
      monitor_trial "$pid" "$log_path"
      if [[ -n "$gpu_pid" ]]; then
        kill "$gpu_pid" 2>/dev/null || true
        wait "$gpu_pid" 2>/dev/null || true
      fi
      run_dir="$(latest_run_dir "$run_root")"
    fi
    write_manifest "$manifest" "$arm" "$seed" "$profile" "$overlay" "$run_root" "$run_dir" "$log_path" "$gpu_log" "$time_log"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$arm" "$seed" "$MONITOR_STATUS" "$MONITOR_ELAPSED_SECONDS" "$MONITOR_PEAK_USED_MB" "$MONITOR_MIN_AVAILABLE_MB" "$run_dir" "$manifest" >> "$RUN_INDEX"
    if [[ "$MONITOR_STATUS" != "ok" && "$MONITOR_STATUS" != "dry_run" ]]; then
      echo "stopping after failed trial: arm=$arm seed=$seed status=$MONITOR_STATUS" >&2
      exit 1
    fi
  done
done

if (( DRY_RUN == 0 )); then
  python3 "$ROOT_DIR/scripts/stateful_tbptt_analyze.py" "$OUT_DIR"
fi
echo "matrix artifacts: $OUT_DIR"
