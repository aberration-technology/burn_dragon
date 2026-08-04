#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BACKEND="${BURN_DRAGON_PC_PAPER_BACKEND:-cuda}"
FEATURES="${BURN_DRAGON_PC_PAPER_FEATURES:-train,cuda}"
PROFILE="${BURN_DRAGON_PC_PAPER_PROFILE:-}"
OUT_DIR="${BURN_DRAGON_PC_PAPER_OUT_DIR:-$ROOT_DIR/target/pc-paper/$(date -u +%Y%m%dT%H%M%SZ)}"
MATRIX="${BURN_DRAGON_PC_PAPER_MATRIX:-smoke}"
BATCH_SIZE="${BURN_DRAGON_PC_PAPER_BATCH_SIZE:-}"
TBPTT_CHUNK_SIZE="${BURN_DRAGON_PC_PAPER_TBPTT_CHUNK_SIZE:-64}"
CHECKPOINT_INTERVAL_ITERS="${BURN_DRAGON_PC_PAPER_CHECKPOINT_INTERVAL_ITERS:-512}"
LOG_FREQUENCY="${BURN_DRAGON_PC_PAPER_LOG_FREQUENCY:-16}"
SOURCE_SELECTION_EVERY_STEPS="${BURN_DRAGON_PC_PAPER_SOURCE_SELECTION_EVERY_STEPS:-16}"
SOURCE_WEIGHTED_VALIDATION_BATCHES="${BURN_DRAGON_PC_PAPER_SOURCE_WEIGHTED_VALIDATION_BATCHES:-1}"
DEGENERACY_PROBE_EVERY_EPOCHS="${BURN_DRAGON_PC_PAPER_DEGENERACY_PROBE_EVERY_EPOCHS:-1}"
RULIAD_CORRECTNESS_PROBE_EVERY_EPOCHS="${BURN_DRAGON_PC_PAPER_RULIAD_CORRECTNESS_PROBE_EVERY_EPOCHS:-1}"
RULIAD_CORRECTNESS_PROBE_ITEMS="${BURN_DRAGON_PC_PAPER_RULIAD_CORRECTNESS_PROBE_ITEMS:-32}"
PC_AMORTIZATION_TOLERANCE="${BURN_DRAGON_PC_PAPER_AMORTIZATION_TOLERANCE:-0.05}"
TIMEOUT_SECONDS="${BURN_DRAGON_PC_PAPER_TIMEOUT_SECONDS:-0}"
WALL_CLOCK_SECONDS="${BURN_DRAGON_PC_PAPER_WALL_CLOCK_SECONDS:-0}"
MAX_SYSTEM_MEMORY_FRACTION="${BURN_DRAGON_PC_PAPER_MAX_SYSTEM_MEMORY_FRACTION:-0.90}"
MIN_AVAILABLE_MB="${BURN_DRAGON_PC_PAPER_MIN_AVAILABLE_MB:-12288}"
SAMPLE_INTERVAL_SECONDS="${BURN_DRAGON_PC_PAPER_SAMPLE_INTERVAL_SECONDS:-2}"
BUILD_RELEASE="${BURN_DRAGON_PC_PAPER_BUILD_RELEASE:-1}"
DRY_RUN=0
SEEDS_CSV="${BURN_DRAGON_PC_PAPER_SEEDS:-}"
ITERS_CSV="${BURN_DRAGON_PC_PAPER_ITERS:-}"
ARMS_CSV="${BURN_DRAGON_PC_PAPER_ARMS:-}"

usage() {
  cat <<'USAGE'
Usage:
  scripts/pc_paper_experiments.sh [options]

Options:
  --matrix <name>              smoke | main-fixed-token | controls | wall-clock | stability |
                               local-factor | hparam | nextlat-tbptt
  --profile <path>             Base training TOML. Default: ruliad-1m JEPA profile.
  --backend <cuda|cpu>         Backend. Default: cuda.
  --features <features>        Cargo features. Default: train,cuda.
  --out-dir <path>             Output directory for overlays, logs, manifests, and run roots.
  --seeds <csv>                Override matrix seeds.
  --iters <csv>                Override matrix iteration counts.
  --arms <csv>                 Override matrix arms.
  --batch-size <n>             Override matrix batch size.
  --timeout-seconds <n>        Hard wall timeout per trial. 0 disables.
  --wall-clock-seconds <n>     Treat timeout as successful fixed-wall-clock completion.
  --dry-run                    Write overlays/manifests and print commands without launching training.
  --no-build                   Reuse the existing release executable without invoking Cargo.

Safety guards:
  BURN_DRAGON_PC_PAPER_MAX_SYSTEM_MEMORY_FRACTION  Default: 0.90
  BURN_DRAGON_PC_PAPER_MIN_AVAILABLE_MB            Default: 12288

The runner isolates every trial under its own BURN_DRAGON_RUN_ROOT and writes
one JSON manifest per trial. Raw checkpoints and metric events remain under
the generated Dragon run directory.

Set BURN_DRAGON_PC_PAPER_TBPTT_CHUNK_SIZE to preserve a production chunk
geometry in systems matrices. The default remains 64 for historical parity.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --matrix)
      MATRIX="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    --features)
      FEATURES="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --seeds)
      SEEDS_CSV="$2"
      shift 2
      ;;
    --iters)
      ITERS_CSV="$2"
      shift 2
      ;;
    --arms)
      ARMS_CSV="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --timeout-seconds)
      TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --wall-clock-seconds)
      WALL_CLOCK_SECONDS="$2"
      TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --no-build)
      BUILD_RELEASE=0
      shift
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
TRAIN_BINARY="$ROOT_DIR/target/release/examples/train_language"

matrix_defaults() {
  case "$MATRIX" in
    smoke)
      : "${PROFILE:=crates/burn_dragon_p2p/deploy/profiles/ruliad-1m.jepa.training.toml}"
      : "${SEEDS_CSV:=20260621}"
      : "${ITERS_CSV:=4}"
      : "${ARMS_CSV:=adamw}"
      : "${BATCH_SIZE:=8}"
      if [[ "$TIMEOUT_SECONDS" == "0" ]]; then
        TIMEOUT_SECONDS=600
      fi
      ;;
    main-fixed-token)
      : "${PROFILE:=crates/burn_dragon_p2p/deploy/profiles/ruliad-1m.jepa.training.toml}"
      : "${SEEDS_CSV:=20260621,20260622,20260623,20260624,20260625}"
      : "${ITERS_CSV:=2048,8192}"
      : "${ARMS_CSV:=adamw,adamwpc,adamwpc_every4}"
      : "${BATCH_SIZE:=64}"
      ;;
    controls)
      : "${PROFILE:=crates/burn_dragon_p2p/deploy/profiles/ruliad-1m.jepa.training.toml}"
      : "${SEEDS_CSV:=20260621,20260622,20260623}"
      : "${ITERS_CSV:=512,2048}"
      : "${ARMS_CSV:=pconly}"
      : "${BATCH_SIZE:=64}"
      ;;
    wall-clock)
      : "${PROFILE:=crates/burn_dragon_p2p/deploy/profiles/ruliad-1m.jepa.training.toml}"
      : "${SEEDS_CSV:=20260621,20260622,20260623}"
      : "${ITERS_CSV:=100000000}"
      : "${ARMS_CSV:=adamw,adamwpc_every4}"
      : "${BATCH_SIZE:=64}"
      if [[ "$WALL_CLOCK_SECONDS" == "0" ]]; then
        WALL_CLOCK_SECONDS=3600
      fi
      if [[ "$TIMEOUT_SECONDS" == "0" ]]; then
        TIMEOUT_SECONDS="$WALL_CLOCK_SECONDS"
      fi
      ;;
    stability)
      : "${PROFILE:=crates/burn_dragon_p2p/deploy/profiles/ruliad-1m.jepa.training.toml}"
      : "${SEEDS_CSV:=20260621,20260622}"
      : "${ITERS_CSV:=100000000}"
      : "${ARMS_CSV:=adamw,adamwpc_every4}"
      : "${BATCH_SIZE:=64}"
      if [[ "$WALL_CLOCK_SECONDS" == "0" ]]; then
        WALL_CLOCK_SECONDS=21600
      fi
      if [[ "$TIMEOUT_SECONDS" == "0" ]]; then
        TIMEOUT_SECONDS="$WALL_CLOCK_SECONDS"
      fi
      ;;
    local-factor)
      : "${PROFILE:=config/language/experiments/predictive_coding/local-pc-1m.toml}"
      : "${SEEDS_CSV:=20260804,20260805,20260806}"
      : "${ITERS_CSV:=128}"
      : "${ARMS_CSV:=local_backprop,local_pc_steps1,local_pc_steps2,local_pc_steps4}"
      : "${BATCH_SIZE:=32}"
      if [[ -z "${BURN_DRAGON_PC_PAPER_TBPTT_CHUNK_SIZE:-}" ]]; then
        TBPTT_CHUNK_SIZE=0
      fi
      if [[ -z "${BURN_DRAGON_PC_PAPER_CHECKPOINT_INTERVAL_ITERS:-}" ]]; then
        CHECKPOINT_INTERVAL_ITERS=128
      fi
      if [[ "$TIMEOUT_SECONDS" == "0" ]]; then
        TIMEOUT_SECONDS=1800
      fi
      ;;
    hparam)
      : "${PROFILE:=crates/burn_dragon_p2p/deploy/profiles/ruliad-1m.jepa.training.toml}"
      : "${SEEDS_CSV:=20260621,20260622,20260623}"
      : "${ITERS_CSV:=512}"
      : "${ARMS_CSV:=adamwpc,adamwpc_step003,adamwpc_step03,adamwpc_steps2,adamwpc_allstate,adamwpc_oracle_block_negative_control}"
      : "${BATCH_SIZE:=64}"
      ;;
    nextlat-tbptt)
      : "${PROFILE:=crates/burn_dragon_p2p/deploy/profiles/ruliad-r1.jepa-nextlat-decoupled-delayed1024-sparse16-tbptt256-probe128-fixed-ablation.toml}"
      : "${SEEDS_CSV:=20260621}"
      : "${ITERS_CSV:=2048,4096}"
      : "${ARMS_CSV:=adamw,adamwpc,adamwpc_warm1024,adamwpc_step003,adamwpc_warm1024_step003}"
      : "${BATCH_SIZE:=8}"
      if [[ "$TIMEOUT_SECONDS" == "0" ]]; then
        TIMEOUT_SECONDS=1800
      fi
      ;;
    *)
      echo "unknown matrix: $MATRIX" >&2
      usage >&2
      exit 2
      ;;
  esac
}

matrix_defaults

if (( DRY_RUN == 1 && BUILD_RELEASE == 1 )); then
  BUILD_RELEASE=0
fi

mkdir -p "$OUT_DIR/overlays" "$OUT_DIR/logs" "$OUT_DIR/manifests" "$OUT_DIR/run_roots" "$OUT_DIR/gpu"
RUN_INDEX="$OUT_DIR/run-index.tsv"
if [[ ! -f "$RUN_INDEX" ]]; then
  printf "trial_key\tmatrix\titers\tarm\tseed\tbatch_size\tstatus\telapsed_seconds\tpeak_used_mb\tmin_available_mb\trun_dir\tmanifest\tlog\n" > "$RUN_INDEX"
fi

if (( BUILD_RELEASE == 1 )); then
  echo "building release train_language example"
  (
    cd "$ROOT_DIR"
    export RUSTC="$RUSTUP_RUSTC"
    export CARGO="$RUSTUP_CARGO"
    "$RUSTUP_CARGO" build --release -p burn_dragon_language --example train_language --features "$FEATURES"
  )
fi
if (( DRY_RUN == 0 )) && [[ ! -x "$TRAIN_BINARY" ]]; then
  echo "release train_language executable is missing: $TRAIN_BINARY" >&2
  echo "rerun without --no-build" >&2
  exit 2
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
  local gpu_path="$3"
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
    if [[ -n "$gpu_path" ]]; then
      nvidia-smi \
        --query-gpu=timestamp,index,utilization.gpu,utilization.memory,power.draw,power.limit,clocks.current.graphics,clocks.current.memory,temperature.gpu \
        --format=csv,noheader,nounits >> "$gpu_path" 2>/dev/null || true
    fi
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
      if (( WALL_CLOCK_SECONDS > 0 )); then
        status="wall_clock_complete"
      else
        status="killed_timeout"
      fi
      echo "timeout guard tripped: elapsed=${elapsed}s timeout=${TIMEOUT_SECONDS}s status=${status}" >> "$log_path"
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

write_pc_block() {
  local path="$1"
  local enabled="$2"
  local state_scope="$3"
  local backward_mode="$4"
  local parameter_update="$5"
  local steps="$6"
  local step_size="$7"
  local apply_every_chunks="$8"
  local warmup_steps="${9:-0}"
  local observation_contract="${10:-observed_prefix}"
  local gradient_norm_scope="${11:-per_sample}"
  local sync_diagnostics="${12:-false}"
  {
    echo "[training.predictive_coding]"
    echo "enabled = $enabled"
    echo "mode = \"recurrent_state\""
    echo "state_scope = \"$state_scope\""
    echo "backward_mode = \"$backward_mode\""
    echo "parameter_update = \"$parameter_update\""
    echo "observation_contract = \"$observation_contract\""
    if [[ "$observation_contract" == "oracle_next_token_negative_control" ]]; then
      echo "allow_oracle_target_leak = true"
    fi
    echo "steps = $steps"
    echo "step_size = $step_size"
    echo "latent_decay = 0.0"
    echo "max_grad_norm = 1.0"
    echo "gradient_norm_scope = \"$gradient_norm_scope\""
    echo "eps = 1.0e-8"
    echo "apply_every_chunks = $apply_every_chunks"
    echo "amortization_tolerance = $PC_AMORTIZATION_TOLERANCE"
    echo "amortization_max_state_slots = 128"
    echo "warmup_steps = $warmup_steps"
    echo "sync_diagnostics = $sync_diagnostics"
  } >> "$path"
}

write_overlay() {
  local path="$1"
  local arm="$2"
  local seed="$3"
  local iters="$4"
  local batch_size="$5"
  local algorithm_line=""
  local tbptt_line=""

  case "$arm" in
    local_backprop)
      algorithm_line='algorithm = "backpropagation"'
      ;;
    local_pc*)
      algorithm_line='algorithm = "predictive_coding"'
      ;;
  esac
  if (( TBPTT_CHUNK_SIZE > 0 )); then
    tbptt_line="tbptt_chunk_size = $TBPTT_CHUNK_SIZE"
  fi

  cat > "$path" <<EOF
[training]
${algorithm_line}
batch_size = $batch_size
max_iters = $iters
checkpoint_interval_iters = $CHECKPOINT_INTERVAL_ITERS
log_frequency = $LOG_FREQUENCY
seed = $seed
${tbptt_line}
launch_mode = "fresh"

[training.events]
flush_every_steps = 8
source_selection_every_steps = $SOURCE_SELECTION_EVERY_STEPS
source_weighted_validation_batches = $SOURCE_WEIGHTED_VALIDATION_BATCHES
degeneracy_probe_every_epochs = $DEGENERACY_PROBE_EVERY_EPOCHS
degeneracy_probe_tokens = 64
ruliad_correctness_probe_every_epochs = $RULIAD_CORRECTNESS_PROBE_EVERY_EPOCHS
ruliad_correctness_probe_items = $RULIAD_CORRECTNESS_PROBE_ITEMS
ruliad_correctness_probe_tokens = 64

[training.continual_backprop]
enabled = false

[training.neuron_scaling]
enabled = false

[training.dynamics]
enabled = false

EOF

  case "$arm" in
    local_backprop)
      cat >> "$path" <<EOF
[optimizer]
name = "adamw"
learning_rate = 0.001
weight_decay = 0.01

EOF
      ;;
    local_pc_steps1|local_pc_steps2|local_pc_steps4|local_pc_steps8|local_pc_steps16|local_pc_steps32|local_pc_steps64|local_pc_steps128)
      local local_steps="${arm#local_pc_steps}"
      cat >> "$path" <<EOF
[optimizer]
name = "adamw"
learning_rate = 0.001
weight_decay = 0.01

[training.local_predictive_coding.inference]
steps = $local_steps

EOF
      ;;
    local_pc_steps4_eta05_lr003|local_pc_steps4_eta05_lr01)
      local local_lr="0.003"
      if [[ "$arm" == *_lr01 ]]; then
        local_lr="0.01"
      fi
      cat >> "$path" <<EOF
[optimizer]
name = "adamw"
learning_rate = $local_lr
weight_decay = 0.01

[training.local_predictive_coding.inference]
steps = 4
step_size = 0.5

EOF
      ;;
    local_pc_steps*_eta*)
      if [[ ! "$arm" =~ ^local_pc_steps(1|2|4|8|16|32|64|128)_eta(01|02|05|10)$ ]]; then
        echo "unknown local PC arm: $arm" >&2
        return 2
      fi
      local local_steps="${BASH_REMATCH[1]}"
      local eta_code="${BASH_REMATCH[2]}"
      local local_eta=""
      case "$eta_code" in
        01) local_eta="0.1" ;;
        02) local_eta="0.2" ;;
        05) local_eta="0.5" ;;
        10) local_eta="1.0" ;;
      esac
      cat >> "$path" <<EOF
[optimizer]
name = "adamw"
learning_rate = 0.001
weight_decay = 0.01

[training.local_predictive_coding.inference]
steps = $local_steps
step_size = $local_eta

EOF
      ;;
    adamw)
      cat >> "$path" <<EOF
[optimizer]
name = "adamw"
learning_rate = 0.001
weight_decay = 0.01

EOF
      write_pc_block "$path" false core chunked optimizer 1 0.01 2
      ;;
    adamwpc)
      cat >> "$path" <<EOF
[optimizer]
name = "adamw"
learning_rate = 0.001
weight_decay = 0.01

EOF
      write_pc_block "$path" true core chunked optimizer 1 0.01 2
      ;;
    adamwpc_oracle_negative_control)
      cat >> "$path" <<EOF
[optimizer]
name = "adamw"
learning_rate = 0.001
weight_decay = 0.01

EOF
      write_pc_block "$path" true core chunked optimizer 1 0.01 2 0 oracle_next_token_negative_control
      ;;
    adamwpc_every_chunk)
      cat >> "$path" <<EOF
[optimizer]
name = "adamw"
learning_rate = 0.001
weight_decay = 0.01

EOF
      write_pc_block "$path" true core chunked optimizer 1 0.01 1
      ;;
    adamwpc_every4|adamwpc_every8)
      local apply_every="${arm#adamwpc_every}"
      cat >> "$path" <<EOF
[optimizer]
name = "adamw"
learning_rate = 0.001
weight_decay = 0.01

EOF
      write_pc_block "$path" true core chunked optimizer 1 0.01 "$apply_every"
      ;;
    adamwpc_every4_global)
      cat >> "$path" <<EOF
[optimizer]
name = "adamw"
learning_rate = 0.001
weight_decay = 0.01

EOF
      write_pc_block "$path" true core chunked optimizer 1 0.01 4 0 observed_prefix global
      ;;
    adamwpc_every4_diagnostics|adamwpc_every4_step*_diagnostics)
      local pc_step_size="0.01"
      if [[ "$arm" =~ ^adamwpc_every4_step([0-9]+)_diagnostics$ ]]; then
        pc_step_size="${BASH_REMATCH[1]}.0"
      fi
      cat >> "$path" <<EOF
[optimizer]
name = "adamw"
learning_rate = 0.001
weight_decay = 0.01

EOF
      write_pc_block "$path" true core chunked optimizer 1 "$pc_step_size" 4 0 observed_prefix per_sample true
      ;;
    adamwpc_every4_step*)
      if [[ ! "$arm" =~ ^adamwpc_every4_step([0-9]+)$ ]]; then
        echo "unknown arm: $arm" >&2
        return 2
      fi
      local pc_step_size="${BASH_REMATCH[1]}.0"
      cat >> "$path" <<EOF
[optimizer]
name = "adamw"
learning_rate = 0.001
weight_decay = 0.01

EOF
      write_pc_block "$path" true core chunked optimizer 1 "$pc_step_size" 4
      ;;
    adamwpc_warm1024)
      cat >> "$path" <<EOF
[optimizer]
name = "adamw"
learning_rate = 0.001
weight_decay = 0.01

EOF
      write_pc_block "$path" true core chunked optimizer 1 0.01 2 1024
      ;;
    pconly)
      cat >> "$path" <<EOF
[optimizer]
name = "adamw"
learning_rate = 0.001
weight_decay = 0.01

EOF
      write_pc_block "$path" true core chunked state_only_control 1 0.01 2
      ;;
    adamwpc_step003)
      cat >> "$path" <<EOF
[optimizer]
name = "adamw"
learning_rate = 0.001
weight_decay = 0.01

EOF
      write_pc_block "$path" true core chunked optimizer 1 0.003 2
      ;;
    adamwpc_warm1024_step003)
      cat >> "$path" <<EOF
[optimizer]
name = "adamw"
learning_rate = 0.001
weight_decay = 0.01

EOF
      write_pc_block "$path" true core chunked optimizer 1 0.003 2 1024
      ;;
    adamwpc_step03)
      cat >> "$path" <<EOF
[optimizer]
name = "adamw"
learning_rate = 0.001
weight_decay = 0.01

EOF
      write_pc_block "$path" true core chunked optimizer 1 0.03 2
      ;;
    adamwpc_steps2)
      cat >> "$path" <<EOF
[optimizer]
name = "adamw"
learning_rate = 0.001
weight_decay = 0.01

EOF
      write_pc_block "$path" true core chunked optimizer 2 0.01 2
      ;;
    adamwpc_allstate)
      cat >> "$path" <<EOF
[optimizer]
name = "adamw"
learning_rate = 0.001
weight_decay = 0.01

EOF
      write_pc_block "$path" true all chunked optimizer 1 0.01 2
      ;;
    adamwpc_block|adamwpc_oracle_block_negative_control)
      cat >> "$path" <<EOF
[optimizer]
name = "adamw"
learning_rate = 0.001
weight_decay = 0.01

EOF
      write_pc_block "$path" true core block optimizer 1 0.01 2 0 oracle_next_token_negative_control
      ;;
    *)
      echo "unknown arm: $arm" >&2
      return 2
      ;;
  esac
}

write_manifest() {
  local manifest="$1"
  local trial_key="$2"
  local arm="$3"
  local seed="$4"
  local iters="$5"
  local batch_size="$6"
  local overlay="$7"
  local run_root="$8"
  local run_dir="$9"
  local log_path="${10}"
  local status="${11}"
  local elapsed="${12}"
  local peak_used_mb="${13}"
  local min_available_mb="${14}"
  local exit_note="${15:-}"
  local gpu_path="${16:-}"
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
  "matrix": $(json_escape "$MATRIX"),
  "arm": $(json_escape "$arm"),
  "seed": $seed,
  "iters": $iters,
  "batch_size": $batch_size,
  "backend": $(json_escape "$BACKEND"),
  "features": $(json_escape "$FEATURES"),
  "profile": $(json_escape "$PROFILE"),
  "overlay": $(json_escape "$overlay"),
  "run_root": $(json_escape "$run_root"),
  "run_dir": $(json_escape "$run_dir"),
  "log_path": $(json_escape "$log_path"),
  "gpu_path": $(json_escape "$gpu_path"),
  "status": $(json_escape "$status"),
  "elapsed_seconds": $elapsed,
  "peak_used_mb": $peak_used_mb,
  "min_available_mb": $min_available_mb,
  "max_system_memory_fraction": $MAX_SYSTEM_MEMORY_FRACTION,
  "min_available_guard_mb": $MIN_AVAILABLE_MB,
  "wall_clock_seconds": $WALL_CLOCK_SECONDS,
  "git_sha": $(json_escape "$git_sha"),
  "git_branch": $(json_escape "$git_branch"),
  "git_dirty": $dirty,
  "note": $(json_escape "$exit_note")
}
EOF
}

latest_run_dir_for_root() {
  local run_root="$1"
  local latest="$run_root/latest"
  if [[ -f "$latest" ]]; then
    local name
    name="$(tr -d '\n\r' < "$latest")"
    if [[ -n "$name" && -d "$run_root/$name" ]]; then
      printf "%s\n" "$run_root/$name"
      return 0
    fi
  fi
  find "$run_root" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | awk 'NR==1 {print $2}'
}

run_trial() {
  local arm="$1"
  local seed="$2"
  local iters="$3"
  local trial_key="pc-${MATRIX}-${iters}-${arm}-seed${seed}-b${BATCH_SIZE}-${BACKEND}"
  local overlay="$OUT_DIR/overlays/${trial_key}.toml"
  local log_path="$OUT_DIR/logs/${trial_key}.log"
  local manifest="$OUT_DIR/manifests/${trial_key}.json"
  local run_root="$OUT_DIR/run_roots/${trial_key}"
  local run_dir=""
  local status="not_started"
  local gpu_path=""

  mkdir -p "$run_root"
  write_overlay "$overlay" "$arm" "$seed" "$iters" "$BATCH_SIZE"

  local cmd=(
    "$TRAIN_BINARY"
    --backend "$BACKEND"
    --config "$PROFILE"
    --config "$overlay"
  )

  echo "==> $trial_key" | tee "$log_path"
  printf "command:" >> "$log_path"
  printf " %q" "${cmd[@]}" >> "$log_path"
  printf "\n" >> "$log_path"

  if (( DRY_RUN == 1 )); then
    status="dry_run"
    write_manifest "$manifest" "$trial_key" "$arm" "$seed" "$iters" "$BATCH_SIZE" "$overlay" "$run_root" "" "$log_path" "$status" 0 0 0 "not launched" ""
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$trial_key" "$MATRIX" "$iters" "$arm" "$seed" "$BATCH_SIZE" "$status" 0 0 0 "" "$manifest" "$log_path" \
      | tee -a "$RUN_INDEX"
    return 0
  fi

  if [[ "$BACKEND" == "cuda" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    gpu_path="$OUT_DIR/gpu/${trial_key}.gpu.csv"
    printf "timestamp,index,utilization_gpu,utilization_memory,power_w,power_limit_w,graphics_clock_mhz,memory_clock_mhz,temperature_c\n" > "$gpu_path"
  fi

  (
    cd "$ROOT_DIR"
    export BURN_DRAGON_RUN_ROOT="$run_root"
    export DragonModel_STAGE_PROFILE=1
    exec "${cmd[@]}"
  ) >> "$log_path" 2>&1 &
  local pid=$!

  monitor_process "$pid" "$log_path" "$gpu_path"
  status="$MONITOR_STATUS"
  run_dir="$(latest_run_dir_for_root "$run_root" || true)"
  write_manifest "$manifest" "$trial_key" "$arm" "$seed" "$iters" "$BATCH_SIZE" "$overlay" "$run_root" "$run_dir" "$log_path" "$status" "$MONITOR_ELAPSED_SECONDS" "$MONITOR_PEAK_USED_MB" "$MONITOR_MIN_AVAILABLE_MB" "" "$gpu_path"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$trial_key" "$MATRIX" "$iters" "$arm" "$seed" "$BATCH_SIZE" "$status" "$MONITOR_ELAPSED_SECONDS" "$MONITOR_PEAK_USED_MB" "$MONITOR_MIN_AVAILABLE_MB" "$run_dir" "$manifest" "$log_path" \
    | tee -a "$RUN_INDEX"

  [[ "$status" == "ok" || "$status" == "wall_clock_complete" ]]
}

IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
IFS=',' read -r -a ITERS <<< "$ITERS_CSV"
IFS=',' read -r -a ARMS <<< "$ARMS_CSV"

echo "pc paper matrix: matrix=$MATRIX backend=$BACKEND profile=$PROFILE batch_size=$BATCH_SIZE out_dir=$OUT_DIR"
echo "seeds=$SEEDS_CSV iters=$ITERS_CSV arms=$ARMS_CSV"
echo "guards: max_system_memory_fraction=$MAX_SYSTEM_MEMORY_FRACTION min_available_mb=$MIN_AVAILABLE_MB timeout_seconds=$TIMEOUT_SECONDS"

matrix_status=0
for iters in "${ITERS[@]}"; do
  for arm in "${ARMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      if ! run_trial "$arm" "$seed" "$iters"; then
        matrix_status=1
        echo "trial failed; continuing matrix: arm=$arm seed=$seed iters=$iters" >&2
      fi
    done
  done
done

echo "matrix complete: $RUN_INDEX"
exit "$matrix_status"
