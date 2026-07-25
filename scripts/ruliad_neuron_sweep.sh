#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="${BURN_DRAGON_SWEEP_BACKEND:-cuda}"
FEATURES="${BURN_DRAGON_SWEEP_FEATURES:-train,cuda}"
MAX_ITERS="${BURN_DRAGON_SWEEP_MAX_ITERS:-4}"
BATCH_SIZES_CSV="${BURN_DRAGON_SWEEP_BATCH_SIZES:-1}"
BLOCK_SIZE="${BURN_DRAGON_SWEEP_BLOCK_SIZE:-}"
TIMEOUT_SECONDS="${BURN_DRAGON_SWEEP_TIMEOUT_SECONDS:-1200}"
CHECKPOINT_INTERVAL_ITERS="${BURN_DRAGON_SWEEP_CHECKPOINT_INTERVAL_ITERS:-1000000}"
MAX_SYSTEM_MEMORY_FRACTION="${BURN_DRAGON_SWEEP_MAX_SYSTEM_MEMORY_FRACTION:-0.70}"
MIN_AVAILABLE_MB="${BURN_DRAGON_SWEEP_MIN_AVAILABLE_MB:-49152}"
SAMPLE_INTERVAL_SECONDS="${BURN_DRAGON_SWEEP_SAMPLE_INTERVAL_SECONDS:-1}"
OUT_DIR="${BURN_DRAGON_SWEEP_OUT_DIR:-$ROOT_DIR/runs/ruliad-neuron-sweep}"
PROFILES_CSV="${BURN_DRAGON_SWEEP_PROFILES:-crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-16k.jepa.training.toml,crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-32k.jepa.training.toml,crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-64k.jepa.training.toml}"
N_LAYER="${BURN_DRAGON_SWEEP_N_LAYER:-}"
N_EMBD="${BURN_DRAGON_SWEEP_N_EMBD:-}"
N_HEAD="${BURN_DRAGON_SWEEP_N_HEAD:-}"
LATENT_TOTAL="${BURN_DRAGON_SWEEP_LATENT_TOTAL:-}"

usage() {
  cat <<'USAGE'
Usage:
  scripts/ruliad_neuron_sweep.sh [options]

Options:
  --backend <cuda|cpu>          Training backend. Default: cuda.
  --features <features>         Cargo features. Default: train,cuda.
  --profiles <csv>              Comma-separated training profile paths.
  --batch-sizes <csv>           Comma-separated batch sizes. Default: 1.
  --block-size <n>              Override training block size.
  --max-iters <n>               Train iterations per probe. Default: 4.
  --shape L,E,H,Z               Override n_layer,n_embd,n_head,latent_total.
  --timeout-seconds <n>         Wall-clock timeout per probe. Default: 1200.
  --out-dir <path>              Logs and sweep report directory.

Environment guards:
  BURN_DRAGON_SWEEP_MAX_SYSTEM_MEMORY_FRACTION  Default: 0.70
  BURN_DRAGON_SWEEP_MIN_AVAILABLE_MB            Default: 49152

The script runs guarded short smokes only. Increase --max-iters after a candidate
has proven safe at the requested batch size.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    --features)
      FEATURES="$2"
      shift 2
      ;;
    --profiles)
      PROFILES_CSV="$2"
      shift 2
      ;;
    --batch-sizes)
      BATCH_SIZES_CSV="$2"
      shift 2
      ;;
    --block-size)
      BLOCK_SIZE="$2"
      shift 2
      ;;
    --max-iters)
      MAX_ITERS="$2"
      shift 2
      ;;
    --shape)
      IFS=',' read -r N_LAYER N_EMBD N_HEAD LATENT_TOTAL <<< "$2"
      shift 2
      ;;
    --timeout-seconds)
      TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$BACKEND" == "cpu" && "$FEATURES" == "train,cuda" ]]; then
  FEATURES="train"
fi

RUSTUP_CARGO="$(rustup which cargo)"
RUSTUP_RUSTC="$(rustup which rustc)"
mkdir -p "$OUT_DIR"

REPORT="$OUT_DIR/sweep-report.tsv"
if [[ ! -f "$REPORT" ]]; then
  printf "profile\tbatch_size\tstatus\tpeak_used_mb\tmin_available_mb\telapsed_seconds\tlog\n" > "$REPORT"
fi

MONITOR_STATUS="not_started"
MONITOR_PEAK_USED_MB=0
MONITOR_MIN_AVAILABLE_MB=0
MONITOR_ELAPSED_SECONDS=0

mem_total_kb() {
  awk '/^MemTotal:/ {print $2}' /proc/meminfo
}

mem_available_kb() {
  awk '/^MemAvailable:/ {print $2}' /proc/meminfo
}

preflight_memory_guard() {
  local total_kb
  local available_kb
  local used_kb
  local max_fraction_bps
  local min_available_kb

  total_kb="$(mem_total_kb)"
  available_kb="$(mem_available_kb)"
  used_kb=$((total_kb - available_kb))
  max_fraction_bps="$(fraction_to_bps "$MAX_SYSTEM_MEMORY_FRACTION")"
  min_available_kb=$((MIN_AVAILABLE_MB * 1024))
  if (( used_kb * 10000 > total_kb * max_fraction_bps )); then
    echo "preflight RAM guard tripped: used=${used_kb}KiB total=${total_kb}KiB fraction_limit=${MAX_SYSTEM_MEMORY_FRACTION}" >&2
    return 1
  fi
  if (( available_kb < min_available_kb )); then
    echo "preflight RAM guard tripped: available=${available_kb}KiB floor=${min_available_kb}KiB" >&2
    return 1
  fi
}

fraction_to_bps() {
  awk -v value="$1" 'BEGIN { printf "%d", value * 10000 }'
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
      echo "RAM guard tripped: used=${used_kb}KiB total=${total_kb}KiB fraction_limit=${MAX_SYSTEM_MEMORY_FRACTION}" >> "$log_path"
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
    if (( elapsed > TIMEOUT_SECONDS )); then
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
  elapsed=$((now - started))
  MONITOR_STATUS="$status"
  MONITOR_PEAK_USED_MB="$((peak_used_kb / 1024))"
  MONITOR_MIN_AVAILABLE_MB="$((min_seen_available_kb / 1024))"
  MONITOR_ELAPSED_SECONDS="$elapsed"
}

run_probe() {
  local profile="$1"
  local batch_size="$2"
  local profile_name
  local shape_suffix=""
  local log_path
  local status
  local peak_used_mb
  local min_available_mb
  local elapsed_seconds

  profile_name="$(basename "$profile" .training.toml)"
  if [[ -n "$LATENT_TOTAL" ]]; then
    shape_suffix="-z${LATENT_TOTAL}"
  fi
  log_path="$OUT_DIR/${profile_name}${shape_suffix}-bs${batch_size}-iters${MAX_ITERS}.log"

  echo "==> profile=$profile batch_size=$batch_size block_size=${BLOCK_SIZE:-profile} max_iters=$MAX_ITERS backend=$BACKEND shape=${N_LAYER:-profile},${N_EMBD:-profile},${N_HEAD:-profile},${LATENT_TOTAL:-profile}" | tee "$log_path"
  if ! preflight_memory_guard 2>&1 | tee -a "$log_path"; then
    local total_kb
    local available_kb
    local used_kb
    total_kb="$(mem_total_kb)"
    available_kb="$(mem_available_kb)"
    used_kb=$((total_kb - available_kb))
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$profile" "$batch_size" "preflight_ram_guard" "$((used_kb / 1024))" "$((available_kb / 1024))" "0" "$log_path" \
      | tee -a "$REPORT"
    return 1
  fi
  (
    cd "$ROOT_DIR"
    export RUSTC="$RUSTUP_RUSTC"
    export CARGO="$RUSTUP_CARGO"
    export BURN_DRAGON_RUN_ROOT="$OUT_DIR/runs"
    export DragonModel_STAGE_PROFILE=1
    override_args=()
    if [[ -n "$N_LAYER" ]]; then override_args+=(--n-layer "$N_LAYER"); fi
    if [[ -n "$N_EMBD" ]]; then override_args+=(--n-embd "$N_EMBD"); fi
    if [[ -n "$N_HEAD" ]]; then override_args+=(--n-head "$N_HEAD"); fi
    if [[ -n "$LATENT_TOTAL" ]]; then override_args+=(--latent-total "$LATENT_TOTAL"); fi
    if [[ -n "$BLOCK_SIZE" ]]; then override_args+=(--block-size "$BLOCK_SIZE"); fi
    exec "$RUSTUP_CARGO" run --release -p burn_dragon_language --example train_language \
      --features "$FEATURES" -- \
      --backend "$BACKEND" \
      --config "$profile" \
      "${override_args[@]}" \
      --batch-size "$batch_size" \
      --max-iters "$MAX_ITERS" \
      --checkpoint-interval-iters "$CHECKPOINT_INTERVAL_ITERS"
  ) >> "$log_path" 2>&1 &
  local pid=$!

  monitor_process "$pid" "$log_path"
  status="$MONITOR_STATUS"
  peak_used_mb="$MONITOR_PEAK_USED_MB"
  min_available_mb="$MONITOR_MIN_AVAILABLE_MB"
  elapsed_seconds="$MONITOR_ELAPSED_SECONDS"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$profile" "$batch_size" "$status" "$peak_used_mb" "$min_available_mb" "$elapsed_seconds" "$log_path" \
    | tee -a "$REPORT"

  [[ "$status" == "ok" ]]
}

IFS=',' read -r -a PROFILES <<< "$PROFILES_CSV"
IFS=',' read -r -a BATCH_SIZES <<< "$BATCH_SIZES_CSV"

echo "sweep output: $OUT_DIR"
echo "RAM guards: max_system_memory_fraction=$MAX_SYSTEM_MEMORY_FRACTION min_available_mb=$MIN_AVAILABLE_MB"

for profile in "${PROFILES[@]}"; do
  for batch_size in "${BATCH_SIZES[@]}"; do
    run_probe "$profile" "$batch_size" || {
      echo "stopping sweep after failed/guarded probe: profile=$profile batch_size=$batch_size" >&2
      exit 1
    }
  done
done

echo "sweep complete: $REPORT"
