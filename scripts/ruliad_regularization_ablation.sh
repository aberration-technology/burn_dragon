#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BACKEND="${BURN_DRAGON_REG_ABLATION_BACKEND:-cuda}"
FEATURES="${BURN_DRAGON_REG_ABLATION_FEATURES:-train,cuda}"
BASE_PROFILE="${BURN_DRAGON_REG_ABLATION_BASE_PROFILE:-crates/burn_dragon_p2p/deploy/profiles/ruliad-r1.jepa.training.toml}"
FIXED_PROFILE="${BURN_DRAGON_REG_ABLATION_FIXED_PROFILE:-crates/burn_dragon_p2p/deploy/profiles/ruliad-r1.adamw-fixed-ablation.toml}"
OUT_DIR="${BURN_DRAGON_REG_ABLATION_OUT_DIR:-$ROOT_DIR/target/ruliad-regularization/$(date -u +%Y%m%dT%H%M%SZ)}"
ARMS_CSV="${BURN_DRAGON_REG_ABLATION_ARMS:-ce_only,anti_collapse,hidden_sigreg,rho_sigreg,hidden_rho_sigreg}"
SEEDS_CSV="${BURN_DRAGON_REG_ABLATION_SEEDS:-20260622}"
ITERS_CSV="${BURN_DRAGON_REG_ABLATION_ITERS:-64}"
BATCH_SIZE="${BURN_DRAGON_REG_ABLATION_BATCH_SIZE:-8}"
LOG_FREQUENCY="${BURN_DRAGON_REG_ABLATION_LOG_FREQUENCY:-8}"
CHECKPOINT_INTERVAL_ITERS="${BURN_DRAGON_REG_ABLATION_CHECKPOINT_INTERVAL_ITERS:-}"
SOURCE_SELECTION_EVERY_STEPS="${BURN_DRAGON_REG_ABLATION_SOURCE_SELECTION_EVERY_STEPS:-8}"
DEGENERACY_PROBE_EVERY_EPOCHS="${BURN_DRAGON_REG_ABLATION_DEGENERACY_PROBE_EVERY_EPOCHS:-1}"
RULIAD_CORRECTNESS_PROBE_EVERY_EPOCHS="${BURN_DRAGON_REG_ABLATION_RULIAD_CORRECTNESS_PROBE_EVERY_EPOCHS:-1}"
SIGREG_SCALE="${BURN_DRAGON_REG_ABLATION_SIGREG_SCALE:-0.003}"
ANTI_COLLAPSE_WARMUP_STEPS="${BURN_DRAGON_REG_ABLATION_ANTI_COLLAPSE_WARMUP_STEPS:-}"
ANTI_COLLAPSE_RAMP_STEPS="${BURN_DRAGON_REG_ABLATION_ANTI_COLLAPSE_RAMP_STEPS:-}"
OPTIMIZER_LR="${BURN_DRAGON_REG_ABLATION_OPTIMIZER_LR:-}"
OPTIMIZER_WEIGHT_DECAY="${BURN_DRAGON_REG_ABLATION_OPTIMIZER_WEIGHT_DECAY:-}"
MAX_SYSTEM_MEMORY_FRACTION="${BURN_DRAGON_REG_ABLATION_MAX_SYSTEM_MEMORY_FRACTION:-0.80}"
MIN_AVAILABLE_MB="${BURN_DRAGON_REG_ABLATION_MIN_AVAILABLE_MB:-24576}"
TIMEOUT_SECONDS="${BURN_DRAGON_REG_ABLATION_TIMEOUT_SECONDS:-900}"
SAMPLE_INTERVAL_SECONDS="${BURN_DRAGON_REG_ABLATION_SAMPLE_INTERVAL_SECONDS:-2}"
GPU_TELEMETRY_SECONDS="${BURN_DRAGON_REG_ABLATION_GPU_TELEMETRY_SECONDS:-1}"
BUILD_RELEASE="${BURN_DRAGON_REG_ABLATION_BUILD_RELEASE:-1}"
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage:
  scripts/ruliad_regularization_ablation.sh [options]

Options:
  --arms <csv>                ce_only, anti_collapse, hidden_sigreg, rho_sigreg, hidden_rho_sigreg,
                              rho_sigreg_001, rho_sigreg_003, rho_sigreg_01, hidden_rho_sigreg_003,
                              anti_collapse_strong, hidden_rho_sigreg_strong_003,
                              anti_collapse_rollout, hidden_rho_sigreg_rollout_003,
                              anti_collapse_self_recovery, hidden_rho_sigreg_self_recovery_003
  --seeds <csv>               Seed list. Default: 20260622.
  --iters <csv>               Iteration counts. Default: 64.
  --batch-size <n>            Batch size. Default: 8.
  --out-dir <path>            Output directory.
  --backend <cuda|cpu>        Backend. Default: cuda.
  --features <features>       Cargo features. Default: train,cuda.
  --timeout-seconds <n>       Per-trial timeout. Default: 900.
  --dry-run                   Write overlays and manifests without training.
  --no-build                  Skip release build.

Safety:
  BURN_DRAGON_REG_ABLATION_MAX_SYSTEM_MEMORY_FRACTION  Default: 0.80
  BURN_DRAGON_REG_ABLATION_MIN_AVAILABLE_MB            Default: 24576
  BURN_DRAGON_REG_ABLATION_ANTI_COLLAPSE_WARMUP_STEPS Optional override for short ablations.
  BURN_DRAGON_REG_ABLATION_ANTI_COLLAPSE_RAMP_STEPS   Optional override for short ablations.
  BURN_DRAGON_REG_ABLATION_OPTIMIZER_LR               Optional explicit optimizer LR override.
  BURN_DRAGON_REG_ABLATION_OPTIMIZER_WEIGHT_DECAY     Optional explicit weight decay override.
  BURN_DRAGON_REG_ABLATION_CHECKPOINT_INTERVAL_ITERS   Optional validation/checkpoint cadence.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arms) ARMS_CSV="$2"; shift 2 ;;
    --seeds) SEEDS_CSV="$2"; shift 2 ;;
    --iters) ITERS_CSV="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --features) FEATURES="$2"; shift 2 ;;
    --timeout-seconds) TIMEOUT_SECONDS="$2"; shift 2 ;;
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
  printf "trial_key\titers\tarm\tseed\tbatch_size\tstatus\telapsed_seconds\tpeak_used_mb\tmin_available_mb\trun_dir\tmanifest\tlog\tgpu_log\n" > "$RUN_INDEX"
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

sigreg_scale_for_arm() {
  case "$1" in
    *_001) printf "0.001\n" ;;
    *_003) printf "0.003\n" ;;
    *_01) printf "0.01\n" ;;
    *) printf "%s\n" "$SIGREG_SCALE" ;;
  esac
}

sigreg_target_for_arm() {
  case "$1" in
    hidden_sigreg*) printf "hidden\n" ;;
    rho_sigreg*) printf "rho_memory_slots\n" ;;
    hidden_rho_sigreg*) printf "hidden_and_rho_memory_slots\n" ;;
    *) printf "hidden\n" ;;
  esac
}

write_regularizer_off_blocks() {
  local path="$1"
  cat >> "$path" <<'EOF'
[training.input_corruption]
enabled = false

[training.logit_entropy_floor]
enabled = false

[training.repeat_unlikelihood]
enabled = false

[training.greedy_rollout_unlikelihood]
enabled = false

[training.latent_reasoning]
enabled = false

[training.dynamics_anchor]
enabled = false

[training.predictive_coding]
enabled = false

EOF
}

write_sigreg_block() {
  local path="$1"
  local target="$2"
  local scale="$3"
  cat >> "$path" <<EOF
[training.latent_reasoning]
enabled = true
every_steps = 1
jepa_future_offsets = [9999]
target_encoder = "detached_student"
teacher_update_rate = 0.01
negative_source = "in_batch_and_corrupt_answer"

[training.latent_reasoning.sigreg]
enabled = true
mode = "weak_covariance"
target = "$target"
target_variance = 1.0
min_variance = 0.2
mean_tolerance = 0.05

[training.latent_reasoning.constraint_balancer]
enabled = true
normalized_aux_scale = $scale
warmup_steps = 0
stop_target_mean_steps = 2.0
stop_tolerance_steps = 0.5

EOF
}

write_anti_collapse_schedule_overrides() {
  local path="$1"
  if [[ -z "$ANTI_COLLAPSE_WARMUP_STEPS" && -z "$ANTI_COLLAPSE_RAMP_STEPS" ]]; then
    return
  fi
  local warmup="${ANTI_COLLAPSE_WARMUP_STEPS:-0}"
  local ramp="${ANTI_COLLAPSE_RAMP_STEPS:-1}"
  cat >> "$path" <<EOF
[training.input_corruption]
warmup_steps = $warmup
ramp_steps = $ramp

[training.logit_entropy_floor]
warmup_steps = $warmup
ramp_steps = $ramp

[training.repeat_unlikelihood]
warmup_steps = $warmup
ramp_steps = $ramp

[training.greedy_rollout_unlikelihood]
warmup_steps = $warmup
ramp_steps = $ramp

[training.dynamics_anchor]
warmup_steps = $warmup
ramp_steps = $ramp

EOF
}

write_optimizer_override() {
  local path="$1"
  if [[ -z "$OPTIMIZER_LR" && -z "$OPTIMIZER_WEIGHT_DECAY" ]]; then
    return
  fi
  cat >> "$path" <<EOF
[optimizer]
name = "adamw"
EOF
  if [[ -n "$OPTIMIZER_LR" ]]; then
    printf "learning_rate = %s\n" "$OPTIMIZER_LR" >> "$path"
  fi
  if [[ -n "$OPTIMIZER_WEIGHT_DECAY" ]]; then
    printf "weight_decay = %s\n" "$OPTIMIZER_WEIGHT_DECAY" >> "$path"
  fi
  printf "\n" >> "$path"
}

write_strong_anti_collapse_overrides() {
  local path="$1"
  local warmup="${ANTI_COLLAPSE_WARMUP_STEPS:-32}"
  local ramp="${ANTI_COLLAPSE_RAMP_STEPS:-128}"
  cat >> "$path" <<EOF
[training.input_corruption]
enabled = true
probability = 0.02
warmup_steps = $warmup
ramp_steps = $ramp

[training.logit_entropy_floor]
enabled = true
weight = 0.015
target_entropy_bits = 2.25
marginal_weight = 0.02
target_marginal_entropy_bits = 5.5
target_coverage_weight = 0.015
target_coverage_epsilon = 0.000001
warmup_steps = $warmup
ramp_steps = $ramp
every_steps = 1

[training.repeat_unlikelihood]
enabled = true
weight = 0.012
cycle_weight = 0.024
cycle_margin_weight = 0.006
cycle_margin = 0.35
cycle_min_lag = 2
cycle_max_lag = 64
cycle_lags_per_step = 4
warmup_steps = $warmup
ramp_steps = $ramp
every_steps = 4
history_lags = [2, 3, 4, 8, 16, 32, 64]
epsilon = 0.0001

[training.greedy_rollout_unlikelihood]
enabled = true
recovery_only = false
weight = 0.03
margin_weight = 0.008
margin = 0.35
recovery_weight = 0.04
entropy_floor_weight = 0.012
target_entropy_bits = 2.25
cycle_weight = 0.03
cycle_margin_weight = 0.008
cycle_min_lag = 2
cycle_max_lag = 64
warmup_steps = $warmup
ramp_steps = $ramp
every_steps = 64
prompt_tokens = 32
rollout_tokens = 24
history_tokens = 32
batch_prompts = 2
epsilon = 0.0001

[training.dynamics_anchor]
enabled = true
weight = 0.03
teacher_update_rate = 0.005
kl = "jensen_shannon"
mask = "all_tokens"
warmup_steps = 0
ramp_steps = 128
every_steps = 1

[training.gates]
degeneracy_entropy_min_bits = 1.25
degeneracy_max_probability_max = 0.82
degeneracy_distinct_2_min_fraction = 0.35
degeneracy_repetition_max_fraction = 0.45
degeneracy_period_2_max_fraction = 0.35
degeneracy_period_3_max_fraction = 0.45
degeneracy_period_2_to_16_max_fraction = 0.55
degeneracy_period_2_to_64_max_fraction = 0.55

EOF
}

write_rollout_recovery_overrides() {
  local path="$1"
  local warmup="${ANTI_COLLAPSE_WARMUP_STEPS:-16}"
  local ramp="${ANTI_COLLAPSE_RAMP_STEPS:-96}"
  cat >> "$path" <<EOF
[training.input_corruption]
enabled = true
probability = 0.05
warmup_steps = $warmup
ramp_steps = $ramp

[training.logit_entropy_floor]
enabled = true
weight = 0.02
target_entropy_bits = 3.0
marginal_weight = 0.03
target_marginal_entropy_bits = 6.0
target_coverage_weight = 0.02
target_coverage_epsilon = 0.000001
warmup_steps = $warmup
ramp_steps = $ramp
every_steps = 1

[training.repeat_unlikelihood]
enabled = true
weight = 0.015
cycle_weight = 0.03
cycle_margin_weight = 0.008
cycle_margin = 0.35
cycle_min_lag = 2
cycle_max_lag = 64
cycle_lags_per_step = 6
warmup_steps = $warmup
ramp_steps = $ramp
every_steps = 2
history_lags = [2, 3, 4, 8, 16, 32, 64]
epsilon = 0.0001

[training.greedy_rollout_unlikelihood]
enabled = true
recovery_only = false
weight = 0.04
margin_weight = 0.01
margin = 0.35
recovery_weight = 0.25
entropy_floor_weight = 0.02
target_entropy_bits = 3.0
cycle_weight = 0.04
cycle_margin_weight = 0.01
cycle_min_lag = 2
cycle_max_lag = 64
warmup_steps = $warmup
ramp_steps = $ramp
every_steps = 16
prompt_tokens = 32
rollout_tokens = 32
history_tokens = 40
batch_prompts = 2
epsilon = 0.0001

[training.dynamics_anchor]
enabled = true
weight = 0.025
teacher_update_rate = 0.005
kl = "jensen_shannon"
mask = "all_tokens"
warmup_steps = 0
ramp_steps = 128
every_steps = 1

[training.gates]
degeneracy_entropy_min_bits = 1.5
degeneracy_max_probability_max = 0.78
degeneracy_distinct_2_min_fraction = 0.40
degeneracy_repetition_max_fraction = 0.40
degeneracy_period_2_max_fraction = 0.35
degeneracy_period_3_max_fraction = 0.40
degeneracy_period_2_to_16_max_fraction = 0.50
degeneracy_period_2_to_64_max_fraction = 0.50

EOF
}

write_self_recovery_overrides() {
  local path="$1"
  local warmup="${ANTI_COLLAPSE_WARMUP_STEPS:-16}"
  local ramp="${ANTI_COLLAPSE_RAMP_STEPS:-96}"
  cat >> "$path" <<EOF
[training.input_corruption]
enabled = true
probability = 0.025
warmup_steps = $warmup
ramp_steps = $ramp

[training.logit_entropy_floor]
enabled = true
weight = 0.008
target_entropy_bits = 1.5
marginal_weight = 0.012
target_marginal_entropy_bits = 4.5
target_coverage_weight = 0.006
target_coverage_epsilon = 0.000001
warmup_steps = $warmup
ramp_steps = $ramp
every_steps = 1

[training.repeat_unlikelihood]
enabled = true
weight = 0.006
cycle_weight = 0.012
cycle_margin_weight = 0.003
cycle_margin = 0.35
cycle_min_lag = 2
cycle_max_lag = 64
cycle_lags_per_step = 4
warmup_steps = $warmup
ramp_steps = $ramp
every_steps = 4
history_lags = [2, 3, 4, 8, 16, 32, 64]
epsilon = 0.0001

[training.greedy_rollout_unlikelihood]
enabled = true
recovery_only = false
weight = 0.0
margin_weight = 0.0
margin = 0.35
recovery_weight = 0.0
sequence_recovery_weight = 1.0
entropy_floor_weight = 0.0
target_entropy_bits = 1.5
cycle_weight = 0.0
cycle_margin_weight = 0.0
cycle_min_lag = 2
cycle_max_lag = 64
warmup_steps = $warmup
ramp_steps = $ramp
every_steps = 4
prompt_tokens = 64
rollout_tokens = 64
history_tokens = 64
batch_prompts = 4
epsilon = 0.0001

[training.dynamics_anchor]
enabled = true
weight = 0.035
teacher_update_rate = 0.005
kl = "jensen_shannon"
mask = "all_tokens"
warmup_steps = 0
ramp_steps = 128
every_steps = 1

[training.gates]
degeneracy_entropy_min_bits = 1.35
degeneracy_max_probability_max = 0.82
degeneracy_distinct_2_min_fraction = 0.35
degeneracy_repetition_max_fraction = 0.45
degeneracy_period_2_max_fraction = 0.35
degeneracy_period_3_max_fraction = 0.40
degeneracy_period_2_to_16_max_fraction = 0.50
degeneracy_period_2_to_64_max_fraction = 0.50

EOF
}

write_overlay() {
  local path="$1"
  local arm="$2"
  local seed="$3"
  local iters="$4"
  local checkpoint_interval="${CHECKPOINT_INTERVAL_ITERS:-$iters}"

  cat > "$path" <<EOF
[training]
batch_size = $BATCH_SIZE
max_iters = $iters
checkpoint_interval_iters = $checkpoint_interval
log_frequency = $LOG_FREQUENCY
seed = $seed
launch_mode = "fresh"

[training.events]
flush_every_steps = 8
source_selection_every_steps = $SOURCE_SELECTION_EVERY_STEPS
source_weighted_validation_batches = 1
degeneracy_probe_every_epochs = $DEGENERACY_PROBE_EVERY_EPOCHS
degeneracy_probe_tokens = 64
ruliad_correctness_probe_every_epochs = $RULIAD_CORRECTNESS_PROBE_EVERY_EPOCHS
ruliad_correctness_probe_items = 32
ruliad_correctness_probe_tokens = 64

[training.auto_batch_size]
enabled = false

[training.continual_backprop]
enabled = false

[training.neuron_scaling]
enabled = false

[training.dynamics]
enabled = false

EOF

  write_optimizer_override "$path"

  case "$arm" in
    ce_only)
      write_regularizer_off_blocks "$path"
      ;;
    anti_collapse|anti_collapse_strong|anti_collapse_rollout|anti_collapse_self_recovery)
      cat >> "$path" <<'EOF'
[training.latent_reasoning]
enabled = false

[training.predictive_coding]
enabled = false

EOF
      if [[ "$arm" == *self_recovery* ]]; then
        write_self_recovery_overrides "$path"
      elif [[ "$arm" == *rollout* ]]; then
        write_rollout_recovery_overrides "$path"
      elif [[ "$arm" == *strong* ]]; then
        write_strong_anti_collapse_overrides "$path"
      else
        write_anti_collapse_schedule_overrides "$path"
      fi
      ;;
    hidden_sigreg*|rho_sigreg*|hidden_rho_sigreg*)
      local target
      local scale
      target="$(sigreg_target_for_arm "$arm")"
      scale="$(sigreg_scale_for_arm "$arm")"
      write_sigreg_block "$path" "$target" "$scale"
      if [[ "$arm" == *self_recovery* ]]; then
        write_self_recovery_overrides "$path"
      elif [[ "$arm" == *rollout* ]]; then
        write_rollout_recovery_overrides "$path"
      elif [[ "$arm" == *strong* ]]; then
        write_strong_anti_collapse_overrides "$path"
      else
        write_anti_collapse_schedule_overrides "$path"
      fi
      ;;
    *)
      echo "unknown arm: $arm" >&2
      return 2
      ;;
  esac
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

write_manifest() {
  local manifest="$1"
  local trial_key="$2"
  local arm="$3"
  local seed="$4"
  local iters="$5"
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
  "arm": $(json_escape "$arm"),
  "seed": $seed,
  "iters": $iters,
  "batch_size": $BATCH_SIZE,
  "backend": $(json_escape "$BACKEND"),
  "features": $(json_escape "$FEATURES"),
  "base_profile": $(json_escape "$BASE_PROFILE"),
  "fixed_profile": $(json_escape "$FIXED_PROFILE"),
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
  "sigreg_scale": $(json_escape "$(sigreg_scale_for_arm "$arm")"),
  "git_sha": $(json_escape "$git_sha"),
  "git_branch": $(json_escape "$git_branch"),
  "git_dirty": $dirty
}
EOF
}

run_trial() {
  local arm="$1"
  local seed="$2"
  local iters="$3"
  local trial_key="reg-${iters}-${arm}-seed${seed}-b${BATCH_SIZE}-${BACKEND}"
  local overlay="$OUT_DIR/overlays/${trial_key}.toml"
  local log_path="$OUT_DIR/logs/${trial_key}.log"
  local gpu_log="$OUT_DIR/logs/${trial_key}.gpu.csv"
  local manifest="$OUT_DIR/manifests/${trial_key}.json"
  local run_root="$OUT_DIR/run_roots/${trial_key}"
  local run_dir=""

  mkdir -p "$run_root"
  write_overlay "$overlay" "$arm" "$seed" "$iters"

  local cmd=(
    "$TRAIN_BINARY"
    --backend "$BACKEND"
    --config "$BASE_PROFILE"
    --config "$FIXED_PROFILE"
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
    write_manifest "$manifest" "$trial_key" "$arm" "$seed" "$iters" "$overlay" "$run_root" "" "$log_path" "$gpu_log" "$MONITOR_STATUS"
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
      exec "${cmd[@]}"
    ) >> "$log_path" 2>&1 &
    local pid=$!
    monitor_process "$pid" "$log_path"
    if [[ -n "$gpu_pid" ]]; then
      kill "$gpu_pid" 2>/dev/null || true
      wait "$gpu_pid" 2>/dev/null || true
    fi
    run_dir="$(latest_run_dir_for_root "$run_root" || true)"
    write_manifest "$manifest" "$trial_key" "$arm" "$seed" "$iters" "$overlay" "$run_root" "$run_dir" "$log_path" "$gpu_log" "$MONITOR_STATUS"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$trial_key" "$iters" "$arm" "$seed" "$BATCH_SIZE" "$MONITOR_STATUS" "$MONITOR_ELAPSED_SECONDS" "$MONITOR_PEAK_USED_MB" "$MONITOR_MIN_AVAILABLE_MB" "$run_dir" "$manifest" "$log_path" "$gpu_log" \
    | tee -a "$RUN_INDEX"

  [[ "$MONITOR_STATUS" == "ok" || "$MONITOR_STATUS" == "dry_run" ]]
}

IFS=',' read -r -a ARMS <<< "$ARMS_CSV"
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
IFS=',' read -r -a ITERS <<< "$ITERS_CSV"

echo "ruliad regularization ablation: backend=$BACKEND batch_size=$BATCH_SIZE out_dir=$OUT_DIR"
echo "profiles: base=$BASE_PROFILE fixed=$FIXED_PROFILE"
echo "arms=$ARMS_CSV seeds=$SEEDS_CSV iters=$ITERS_CSV sigreg_scale=$SIGREG_SCALE"
echo "guards: max_system_memory_fraction=$MAX_SYSTEM_MEMORY_FRACTION min_available_mb=$MIN_AVAILABLE_MB timeout_seconds=$TIMEOUT_SECONDS"

for iters in "${ITERS[@]}"; do
  for arm in "${ARMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_trial "$arm" "$seed" "$iters"
    done
  done
done

echo "matrix complete: $RUN_INDEX"
