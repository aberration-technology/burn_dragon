#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BACKEND="${BURN_DRAGON_CAP_FEEDBACK_BACKEND:-cuda}"
FEATURES="${BURN_DRAGON_CAP_FEEDBACK_FEATURES:-train,cuda}"
BASE_PROFILE="${BURN_DRAGON_CAP_FEEDBACK_BASE_PROFILE:-crates/burn_dragon_p2p/deploy/profiles/ruliad-r1.jepa-nextlat-decoupled-delayed1024-sparse16-probe128-fixed-ablation.toml}"
OUT_DIR="${BURN_DRAGON_CAP_FEEDBACK_OUT_DIR:-$ROOT_DIR/target/ruliad-capability-feedback/$(date -u +%Y%m%dT%H%M%SZ)}"
ARMS_CSV="${BURN_DRAGON_CAP_FEEDBACK_ARMS:-baseline,feedback,gated_aux,full_policy}"
SEEDS_CSV="${BURN_DRAGON_CAP_FEEDBACK_SEEDS:-20260624,20260625,20260626}"
MAX_ITERS="${BURN_DRAGON_CAP_FEEDBACK_MAX_ITERS:-2048}"
BATCH_SIZE="${BURN_DRAGON_CAP_FEEDBACK_BATCH_SIZE:-4}"
BLOCK_SIZE="${BURN_DRAGON_CAP_FEEDBACK_BLOCK_SIZE:-256}"
MAX_STEPS="${BURN_DRAGON_CAP_FEEDBACK_MAX_STEPS:-1}"
EVAL_STEPS_CSV="${BURN_DRAGON_CAP_FEEDBACK_EVAL_STEPS:-1,2,4,8}"
N_LAYER="${BURN_DRAGON_CAP_FEEDBACK_N_LAYER:-4}"
N_EMBD="${BURN_DRAGON_CAP_FEEDBACK_N_EMBD:-256}"
N_HEAD="${BURN_DRAGON_CAP_FEEDBACK_N_HEAD:-4}"
LATENT_TOTAL="${BURN_DRAGON_CAP_FEEDBACK_LATENT_TOTAL:-12288}"
NEXTLAT_START_AFTER="${BURN_DRAGON_CAP_FEEDBACK_NEXTLAT_START_AFTER:-128}"
NEXTLAT_EVERY_STEPS="${BURN_DRAGON_CAP_FEEDBACK_NEXTLAT_EVERY_STEPS:-16}"
JEPA_EVERY_STEPS="${BURN_DRAGON_CAP_FEEDBACK_JEPA_EVERY_STEPS:-8}"
LOG_FREQUENCY="${BURN_DRAGON_CAP_FEEDBACK_LOG_FREQUENCY:-16}"
CHECKPOINT_INTERVAL_ITERS="${BURN_DRAGON_CAP_FEEDBACK_CHECKPOINT_INTERVAL_ITERS:-128}"
RULIAD_PROBE_ITEMS="${BURN_DRAGON_CAP_FEEDBACK_RULIAD_PROBE_ITEMS:-128}"
RULIAD_PROBE_TOKENS="${BURN_DRAGON_CAP_FEEDBACK_RULIAD_PROBE_TOKENS:-64}"
TIMEOUT_SECONDS="${BURN_DRAGON_CAP_FEEDBACK_TIMEOUT_SECONDS:-2400}"
MAX_SYSTEM_MEMORY_FRACTION="${BURN_DRAGON_CAP_FEEDBACK_MAX_SYSTEM_MEMORY_FRACTION:-0.80}"
MIN_AVAILABLE_MB="${BURN_DRAGON_CAP_FEEDBACK_MIN_AVAILABLE_MB:-24576}"
BUILD_RELEASE="${BURN_DRAGON_CAP_FEEDBACK_BUILD_RELEASE:-1}"
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage:
  scripts/ruliad_capability_feedback_ablation.sh [options]

Options:
  --arms <csv>              baseline,feedback,gated_aux,full_policy.
  --seeds <csv>             Seed list. Default: 20260624,20260625,20260626.
  --max-iters <n>           Iterations per trial. Default: 2048.
  --batch-size <n>          Batch size. Default: 4.
  --block-size <n>          Block size. Default: 256.
  --max-steps <n>           Fixed latent reasoning steps. Default: 1.
  --eval-steps <csv>        Validation-only eval step sweep. Default: 1,2,4,8.
  --shape L,E,H,Z           Model shape n_layer,n_embd,n_head,latent_total.
  --base-profile <path>     Base profile.
  --out-dir <path>          Output directory.
  --timeout-seconds <n>     Per-trial timeout. Default: 2400.
  --probe-items <n>         Ruliad correctness probe items. Default: 128.
  --probe-tokens <n>        Ruliad correctness generation tokens. Default: 64.
  --backend <cuda|cpu>      Backend. Default: cuda.
  --features <features>     Cargo features. Default: train,cuda.
  --dry-run                 Write overlays/manifests only.
  --no-build                Skip release build.
  --help                    Show this message.

Safety:
  BURN_DRAGON_CAP_FEEDBACK_MAX_SYSTEM_MEMORY_FRACTION  Default: 0.80
  BURN_DRAGON_CAP_FEEDBACK_MIN_AVAILABLE_MB            Default: 24576

Arms:
  baseline    capability probes on, source-selection capability feedback off, passive gates.
  feedback    capability probes on, source-selection capability feedback on, passive gates.
  gated_aux   feedback on, passive gates, NextLat starts only after capability gate opens.
  full_policy feedback on, active capability gate policy, NextLat starts after capability gate opens.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arms) ARMS_CSV="$2"; shift 2 ;;
    --seeds) SEEDS_CSV="$2"; shift 2 ;;
    --max-iters) MAX_ITERS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --block-size) BLOCK_SIZE="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --eval-steps) EVAL_STEPS_CSV="$2"; shift 2 ;;
    --shape)
      IFS=',' read -r N_LAYER N_EMBD N_HEAD LATENT_TOTAL <<< "$2"
      shift 2
      ;;
    --base-profile) BASE_PROFILE="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --timeout-seconds) TIMEOUT_SECONDS="$2"; shift 2 ;;
    --probe-items) RULIAD_PROBE_ITEMS="$2"; shift 2 ;;
    --probe-tokens) RULIAD_PROBE_TOKENS="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --features) FEATURES="$2"; shift 2 ;;
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

BASE_PROFILE_ABS="$(realpath "$ROOT_DIR/$BASE_PROFILE" 2>/dev/null || realpath "$BASE_PROFILE")"
mkdir -p "$OUT_DIR/profiles"

write_arm_profile() {
  local arm="$1"
  local profile="$2"
  local feedback="true"
  local gates_enabled="true"
  local passive_grace="9999"
  local passive_patience="9999"
  local nextlat_start_policy="fixed_step"

  case "$arm" in
    baseline)
      feedback="false"
      ;;
    feedback)
      ;;
    gated_aux)
      nextlat_start_policy="capability_gate"
      ;;
    full_policy)
      nextlat_start_policy="capability_gate"
      passive_grace="3"
      passive_patience="2"
      ;;
    *)
      echo "unknown arm: $arm" >&2
      exit 2
      ;;
  esac

  cat > "$profile" <<EOF
extends = ["$BASE_PROFILE_ABS"]

[training.events]
source_selection_capability_feedback = $feedback

[training.gates]
enabled = $gates_enabled
fatal_stop = false
capability_grace_epochs = $passive_grace
capability_regression_patience_epochs = $passive_patience
capability_required_after_first_pass = true

[training.latent_reasoning.next_latent]
start_policy = "$nextlat_start_policy"
EOF
}

IFS=',' read -r -a ARMS <<< "$ARMS_CSV"

echo "ruliad capability feedback ablation output: $OUT_DIR"
echo "base_profile=$BASE_PROFILE_ABS"
echo "arms=${ARMS_CSV} seeds=${SEEDS_CSV} max_iters=${MAX_ITERS} max_steps=${MAX_STEPS} backend=${BACKEND}"
echo "shape: n_layer=$N_LAYER n_embd=$N_EMBD n_head=$N_HEAD latent_total=$LATENT_TOTAL block_size=$BLOCK_SIZE batch_size=$BATCH_SIZE"
echo "RAM guards: max_system_memory_fraction=$MAX_SYSTEM_MEMORY_FRACTION min_available_mb=$MIN_AVAILABLE_MB"

first_arm=1
for arm in "${ARMS[@]}"; do
  arm_profile="$OUT_DIR/profiles/${arm}.toml"
  arm_out="$OUT_DIR/$arm"
  write_arm_profile "$arm" "$arm_profile"
  args=(
    --base-profile "$arm_profile"
    --steps "$MAX_STEPS"
    --eval-steps "$EVAL_STEPS_CSV"
    --seeds "$SEEDS_CSV"
    --max-iters "$MAX_ITERS"
    --batch-size "$BATCH_SIZE"
    --block-size "$BLOCK_SIZE"
    --shape "$N_LAYER,$N_EMBD,$N_HEAD,$LATENT_TOTAL"
    --nextlat-start "$NEXTLAT_START_AFTER"
    --out-dir "$arm_out"
    --timeout-seconds "$TIMEOUT_SECONDS"
    --probe-items "$RULIAD_PROBE_ITEMS"
    --probe-tokens "$RULIAD_PROBE_TOKENS"
    --backend "$BACKEND"
    --features "$FEATURES"
  )
  if (( DRY_RUN == 1 )); then
    args+=(--dry-run)
  elif (( BUILD_RELEASE == 0 || first_arm == 0 )); then
    args+=(--no-build --allow-stale-binary)
  fi

  echo "==> arm=$arm profile=$arm_profile"
  BURN_DRAGON_LR_STEPS_NEXTLAT_EVERY_STEPS="$NEXTLAT_EVERY_STEPS" \
    BURN_DRAGON_LR_STEPS_JEPA_EVERY_STEPS="$JEPA_EVERY_STEPS" \
    BURN_DRAGON_LR_STEPS_LOG_FREQUENCY="$LOG_FREQUENCY" \
    BURN_DRAGON_LR_STEPS_CHECKPOINT_INTERVAL_ITERS="$CHECKPOINT_INTERVAL_ITERS" \
    BURN_DRAGON_LR_STEPS_MAX_SYSTEM_MEMORY_FRACTION="$MAX_SYSTEM_MEMORY_FRACTION" \
    BURN_DRAGON_LR_STEPS_MIN_AVAILABLE_MB="$MIN_AVAILABLE_MB" \
    "$ROOT_DIR/scripts/latent_reasoning_steps_ablation.sh" "${args[@]}"

  if (( DRY_RUN == 0 )); then
    python3 "$ROOT_DIR/scripts/latent_reasoning_steps_analyze.py" "$arm_out" --out-dir "$arm_out/analysis"
  fi
  first_arm=0
done

if (( DRY_RUN == 0 )); then
  python3 "$ROOT_DIR/scripts/ruliad_capability_feedback_analyze.py" "$OUT_DIR"
fi

echo "ruliad capability feedback ablation complete: $OUT_DIR"
