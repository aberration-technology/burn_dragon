#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BACKEND="${BURN_DRAGON_HDRAGON_ABLATION_BACKEND:-cuda}"
FEATURES="${BURN_DRAGON_HDRAGON_ABLATION_FEATURES:-train,cuda}"
OUT_DIR="${BURN_DRAGON_HDRAGON_ABLATION_OUT_DIR:-$ROOT_DIR/target/hierarchical-dragon-ablation/$(date -u +%Y%m%dT%H%M%SZ)}"
PROFILES_CSV="${BURN_DRAGON_HDRAGON_ABLATION_PROFILES:-crates/burn_dragon_p2p/deploy/profiles/ruliad-r1.jepa-nextlat-decoupled-delayed1024-sparse16-probe128-fixed-ablation.toml,crates/burn_dragon_p2p/deploy/profiles/ruliad-r1.hdragon-shared-rho-shared-weights-probe128-fixed-ablation.toml,crates/burn_dragon_p2p/deploy/profiles/ruliad-r1.hdragon-split-rho-shared-weights-probe128-fixed-ablation.toml,crates/burn_dragon_p2p/deploy/profiles/ruliad-r1.hdragon-split-rho-split-weights-probe128-fixed-ablation.toml}"
STEPS_CSV="${BURN_DRAGON_HDRAGON_ABLATION_MAX_STEPS:-1,4,8}"
EVAL_STEPS_CSV="${BURN_DRAGON_HDRAGON_ABLATION_EVAL_STEPS:-1,2,4,8,16}"
SEEDS_CSV="${BURN_DRAGON_HDRAGON_ABLATION_SEEDS:-20260623}"
MAX_ITERS="${BURN_DRAGON_HDRAGON_ABLATION_MAX_ITERS:-256}"
BATCH_SIZE="${BURN_DRAGON_HDRAGON_ABLATION_BATCH_SIZE:-4}"
LOG_FREQUENCY="${BURN_DRAGON_HDRAGON_ABLATION_LOG_FREQUENCY:-8}"
CHECKPOINT_INTERVAL_ITERS="${BURN_DRAGON_HDRAGON_ABLATION_CHECKPOINT_INTERVAL_ITERS:-64}"
RULIAD_PROBE_ITEMS="${BURN_DRAGON_HDRAGON_ABLATION_RULIAD_PROBE_ITEMS:-128}"
RULIAD_PROBE_TOKENS="${BURN_DRAGON_HDRAGON_ABLATION_RULIAD_PROBE_TOKENS:-64}"
TIMEOUT_SECONDS="${BURN_DRAGON_HDRAGON_ABLATION_TIMEOUT_SECONDS:-1800}"
MAX_SYSTEM_MEMORY_FRACTION="${BURN_DRAGON_HDRAGON_ABLATION_MAX_SYSTEM_MEMORY_FRACTION:-0.80}"
MIN_AVAILABLE_MB="${BURN_DRAGON_HDRAGON_ABLATION_MIN_AVAILABLE_MB:-24576}"
SAMPLE_INTERVAL_SECONDS="${BURN_DRAGON_HDRAGON_ABLATION_SAMPLE_INTERVAL_SECONDS:-2}"
GPU_TELEMETRY_SECONDS="${BURN_DRAGON_HDRAGON_ABLATION_GPU_TELEMETRY_SECONDS:-1}"
BUILD_RELEASE="${BURN_DRAGON_HDRAGON_ABLATION_BUILD_RELEASE:-1}"
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage:
  scripts/hierarchical_dragon_ablation.sh [options]

Options:
  --profiles <csv>          Profile paths. Default: JEPA/NextLat baseline plus three hdragon arms.
  --steps <csv>             Fixed latent max_steps values. Default: 1,4,8.
  --eval-steps <csv>        Validation-only eval step sweep. Default: 1,2,4,8,16.
  --seeds <csv>             Training seed list. Default: 20260623.
  --max-iters <n>           Iterations per trial. Default: 256.
  --batch-size <n>          Batch size override. Default: 4.
  --log-frequency <n>       Metric log frequency. Default: 8.
  --checkpoint-interval <n> Checkpoint/validation cadence. Default: 64.
  --probe-items <n>         Ruliad correctness probe items. Default: 128.
  --probe-tokens <n>        Ruliad correctness generation tokens. Default: 64.
  --timeout-seconds <n>     Per-trial timeout. Default: 1800.
  --backend <cuda|cpu>      Backend. Default: cuda.
  --features <features>     Cargo features. Default: train,cuda.
  --out-dir <path>          Output directory.
  --dry-run                 Write overlays/manifests only.
  --no-build                Skip release build.

Safety:
  BURN_DRAGON_HDRAGON_ABLATION_MAX_SYSTEM_MEMORY_FRACTION  Default: 0.80
  BURN_DRAGON_HDRAGON_ABLATION_MIN_AVAILABLE_MB            Default: 24576
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profiles) PROFILES_CSV="$2"; shift 2 ;;
    --steps) STEPS_CSV="$2"; shift 2 ;;
    --eval-steps) EVAL_STEPS_CSV="$2"; shift 2 ;;
    --seeds) SEEDS_CSV="$2"; shift 2 ;;
    --max-iters) MAX_ITERS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --log-frequency) LOG_FREQUENCY="$2"; shift 2 ;;
    --checkpoint-interval) CHECKPOINT_INTERVAL_ITERS="$2"; shift 2 ;;
    --probe-items) RULIAD_PROBE_ITEMS="$2"; shift 2 ;;
    --probe-tokens) RULIAD_PROBE_TOKENS="$2"; shift 2 ;;
    --timeout-seconds) TIMEOUT_SECONDS="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --features) FEATURES="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --no-build) BUILD_RELEASE=0; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$BACKEND" == "cpu" && "$FEATURES" == "train,cuda" ]]; then
  FEATURES="train"
fi
if (( DRY_RUN == 1 )); then
  BUILD_RELEASE=0
fi

RUSTUP_CARGO="$(rustup which cargo)"
RUSTUP_RUSTC="$(rustup which rustc)"
TRAIN_BINARY="$ROOT_DIR/target/release/examples/train_language"

mkdir -p "$OUT_DIR/overlays" "$OUT_DIR/logs" "$OUT_DIR/manifests" "$OUT_DIR/run_roots"
RUN_INDEX="$OUT_DIR/run-index.tsv"
if [[ ! -f "$RUN_INDEX" ]]; then
  printf "trial_key\tprofile\tmax_steps\tseed\tmax_iters\tbatch_size\tstatus\telapsed_seconds\tpeak_used_mb\tmin_available_mb\trun_dir\tmanifest\tlog\tgpu_log\n" > "$RUN_INDEX"
fi

if (( BUILD_RELEASE == 1 )); then
  (
    cd "$ROOT_DIR"
    export CARGO="$RUSTUP_CARGO"
    export RUSTC="$RUSTUP_RUSTC"
    "$RUSTUP_CARGO" build --release -p burn_dragon_language --example train_language --features "$FEATURES"
  )
fi

mem_total_kb() {
  awk '/^MemTotal:/ {print $2}' /proc/meminfo
}

mem_available_kb() {
  awk '/^MemAvailable:/ {print $2}' /proc/meminfo
}

fraction_to_bps() {
  awk -v value="$1" 'BEGIN { printf "%d", value * 10000 }'
}

json_escape() {
  python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"
}

csv_to_toml_array() {
  python3 -c 'import sys; print("[" + ", ".join(str(int(x.strip())) for x in sys.argv[1].split(",") if x.strip()) + "]")' "$1"
}

slugify() {
  basename "$1" .toml \
    | sed -E 's/\.training$//; s/[^A-Za-z0-9_.-]+/-/g'
}

kill_training_process() {
  local pid="$1"
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    sleep 5
  fi
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  fi
}

MONITOR_STATUS="not_started"
MONITOR_PEAK_USED_MB=0
MONITOR_MIN_AVAILABLE_MB=0
MONITOR_ELAPSED_SECONDS=0

monitor_process() {
  local pid="$1"
  local log_path="$2"
  local max_fraction_bps
  local total_kb
  local min_available_kb
  local peak_used_kb=0
  local min_seen_available_kb=0
  local started
  local now
  local elapsed
  local status="ok"

  max_fraction_bps="$(fraction_to_bps "$MAX_SYSTEM_MEMORY_FRACTION")"
  total_kb="$(mem_total_kb)"
  min_available_kb=$((MIN_AVAILABLE_MB * 1024))
  min_seen_available_kb="$(mem_available_kb)"
  started="$(date +%s)"

  while kill -0 "$pid" 2>/dev/null; do
    local available_kb
    local used_kb
    available_kb="$(mem_available_kb)"
    used_kb=$((total_kb - available_kb))
    if (( used_kb > peak_used_kb )); then
      peak_used_kb="$used_kb"
    fi
    if (( available_kb < min_seen_available_kb )); then
      min_seen_available_kb="$available_kb"
    fi
    if (( used_kb * 10000 > total_kb * max_fraction_bps )); then
      status="killed_ram_fraction"
      echo "RAM guard tripped: used=${used_kb}KiB total=${total_kb}KiB limit=${MAX_SYSTEM_MEMORY_FRACTION}" >> "$log_path"
      kill_training_process "$pid"
      break
    fi
    if (( available_kb < min_available_kb )); then
      status="killed_low_available_ram"
      echo "RAM guard tripped: available=${available_kb}KiB floor=${min_available_kb}KiB" >> "$log_path"
      kill_training_process "$pid"
      break
    fi
    now="$(date +%s)"
    elapsed=$((now - started))
    if (( TIMEOUT_SECONDS > 0 && elapsed > TIMEOUT_SECONDS )); then
      status="killed_timeout"
      echo "timeout guard tripped: elapsed=${elapsed}s timeout=${TIMEOUT_SECONDS}s" >> "$log_path"
      kill_training_process "$pid"
      break
    fi
    sleep "$SAMPLE_INTERVAL_SECONDS"
  done

  set +e
  wait "$pid"
  local exit_code=$?
  set -e
  if [[ "$exit_code" != "0" && "$status" == "ok" ]]; then
    status="failed_exit_${exit_code}"
  fi

  now="$(date +%s)"
  MONITOR_STATUS="$status"
  MONITOR_PEAK_USED_MB="$((peak_used_kb / 1024))"
  MONITOR_MIN_AVAILABLE_MB="$((min_seen_available_kb / 1024))"
  MONITOR_ELAPSED_SECONDS="$((now - started))"
}

write_overlay() {
  local path="$1"
  local seed="$2"
  local steps="$3"
  cat > "$path" <<EOF
[model.latent_reasoning]
enabled = true
max_steps = $steps
min_steps = $steps
adaptive_halting = false

[training]
seed = $seed
max_iters = $MAX_ITERS
batch_size = $BATCH_SIZE
log_frequency = $LOG_FREQUENCY
checkpoint_interval_iters = $CHECKPOINT_INTERVAL_ITERS

[training.auto_batch_size]
enabled = false

[training.events]
ruliad_correctness_probe_items = $RULIAD_PROBE_ITEMS
ruliad_correctness_probe_tokens = $RULIAD_PROBE_TOKENS

[training.latent_reasoning]
eval_step_sweep = $(csv_to_toml_array "$EVAL_STEPS_CSV")
EOF
}

latest_run_dir() {
  local run_root="$1"
  find "$run_root" -mindepth 1 -maxdepth 3 -type d -name checkpoint -printf '%T@ %h\n' 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

write_manifest() {
  local manifest="$1"
  local trial_key="$2"
  local profile="$3"
  local steps="$4"
  local seed="$5"
  local overlay="$6"
  local run_root="$7"
  local run_dir="$8"
  local log_path="$9"
  local gpu_log="${10}"
  local status="${11}"
  local git_sha
  local git_branch
  local dirty

  git_sha="$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || true)"
  git_branch="$(git -C "$ROOT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  if [[ -z "$(git -C "$ROOT_DIR" status --porcelain 2>/dev/null)" ]]; then
    dirty=false
  else
    dirty=true
  fi

  cat > "$manifest" <<EOF
{
  "trial_key": $(json_escape "$trial_key"),
  "profile": $(json_escape "$profile"),
  "max_steps": $steps,
  "eval_step_sweep": $(json_escape "$EVAL_STEPS_CSV"),
  "seed": $seed,
  "max_iters": $MAX_ITERS,
  "batch_size": $BATCH_SIZE,
  "ruliad_probe_items": $RULIAD_PROBE_ITEMS,
  "ruliad_probe_tokens": $RULIAD_PROBE_TOKENS,
  "backend": $(json_escape "$BACKEND"),
  "features": $(json_escape "$FEATURES"),
  "overlay": $(json_escape "$overlay"),
  "run_root": $(json_escape "$run_root"),
  "run_dir": $(json_escape "$run_dir"),
  "log_path": $(json_escape "$log_path"),
  "gpu_log_path": $(json_escape "$gpu_log"),
  "status": $(json_escape "$status"),
  "elapsed_seconds": $MONITOR_ELAPSED_SECONDS,
  "peak_used_mb": $MONITOR_PEAK_USED_MB,
  "min_available_mb": $MONITOR_MIN_AVAILABLE_MB,
  "max_system_memory_fraction": $MAX_SYSTEM_MEMORY_FRACTION,
  "min_available_guard_mb": $MIN_AVAILABLE_MB,
  "git_sha": $(json_escape "$git_sha"),
  "git_branch": $(json_escape "$git_branch"),
  "git_dirty": $dirty
}
EOF
}

run_trial() {
  local profile="$1"
  local steps="$2"
  local seed="$3"
  local profile_slug
  local trial_key
  local overlay
  local log_path
  local gpu_log
  local manifest
  local run_root
  local run_dir=""

  profile_slug="$(slugify "$profile")"
  trial_key="hdragon-${profile_slug}-ms${steps}-seed${seed}-i${MAX_ITERS}-b${BATCH_SIZE}-${BACKEND}"
  overlay="$OUT_DIR/overlays/${trial_key}.toml"
  log_path="$OUT_DIR/logs/${trial_key}.log"
  gpu_log="$OUT_DIR/logs/${trial_key}.gpu.csv"
  manifest="$OUT_DIR/manifests/${trial_key}.json"
  run_root="$OUT_DIR/run_roots/${trial_key}"
  mkdir -p "$run_root"
  write_overlay "$overlay" "$seed" "$steps"

  local cmd=(
    "$TRAIN_BINARY"
    --backend "$BACKEND"
    --config "$profile"
    --config "$overlay"
  )

  echo "==> $trial_key" | tee "$log_path"
  printf "command:" >> "$log_path"
  printf " %q" "${cmd[@]}" >> "$log_path"
  printf "\n" >> "$log_path"

  if (( DRY_RUN == 1 )); then
    MONITOR_STATUS="dry_run"
    MONITOR_ELAPSED_SECONDS=0
    MONITOR_PEAK_USED_MB=0
    MONITOR_MIN_AVAILABLE_MB=0
    write_manifest "$manifest" "$trial_key" "$profile" "$steps" "$seed" "$overlay" "$run_root" "" "$log_path" "$gpu_log" "$MONITOR_STATUS"
  else
    local gpu_pid=""
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi \
        --query-gpu=timestamp,index,utilization.gpu,power.draw,memory.used,memory.total \
        --format=csv \
        -l "$GPU_TELEMETRY_SECONDS" > "$gpu_log" 2>/dev/null &
      gpu_pid="$!"
    fi
    (
      cd "$ROOT_DIR"
      export CARGO="$RUSTUP_CARGO"
      export RUSTC="$RUSTUP_RUSTC"
      export BURN_DRAGON_RUN_ROOT="$run_root"
      export DragonModel_STAGE_PROFILE=1
      exec "${cmd[@]}"
    ) >> "$log_path" 2>&1 &
    local pid=$!
    monitor_process "$pid" "$log_path"
    if [[ -n "$gpu_pid" ]]; then
      kill "$gpu_pid" 2>/dev/null || true
      wait "$gpu_pid" 2>/dev/null || true
    fi
    run_dir="$(latest_run_dir "$run_root")"
    write_manifest "$manifest" "$trial_key" "$profile" "$steps" "$seed" "$overlay" "$run_root" "$run_dir" "$log_path" "$gpu_log" "$MONITOR_STATUS"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$trial_key" "$profile" "$steps" "$seed" "$MAX_ITERS" "$BATCH_SIZE" "$MONITOR_STATUS" "$MONITOR_ELAPSED_SECONDS" "$MONITOR_PEAK_USED_MB" "$MONITOR_MIN_AVAILABLE_MB" "$run_dir" "$manifest" "$log_path" "$gpu_log" \
    | tee -a "$RUN_INDEX"

  [[ "$MONITOR_STATUS" == "ok" || "$MONITOR_STATUS" == "dry_run" ]]
}

IFS=',' read -r -a PROFILES <<< "$PROFILES_CSV"
IFS=',' read -r -a STEPS <<< "$STEPS_CSV"
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"

echo "hierarchical Dragon ablation output: $OUT_DIR"
echo "RAM guards: max_system_memory_fraction=$MAX_SYSTEM_MEMORY_FRACTION min_available_mb=$MIN_AVAILABLE_MB"
echo "profiles=${#PROFILES[@]} steps=${#STEPS[@]} seeds=${#SEEDS[@]} max_iters=$MAX_ITERS batch_size=$BATCH_SIZE probe_items=$RULIAD_PROBE_ITEMS probe_tokens=$RULIAD_PROBE_TOKENS backend=$BACKEND"
echo "latent eval step sweep=$EVAL_STEPS_CSV"

for seed in "${SEEDS[@]}"; do
  for profile in "${PROFILES[@]}"; do
    for steps in "${STEPS[@]}"; do
      run_trial "$profile" "$steps" "$seed" || {
        echo "stopping ablation after failed/guarded trial: profile=$profile max_steps=$steps seed=$seed status=$MONITOR_STATUS" >&2
        exit 1
      }
    done
  done
done

echo "hierarchical Dragon ablation complete: $RUN_INDEX"
