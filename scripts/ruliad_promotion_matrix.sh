#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BACKEND="${BURN_DRAGON_PROMOTION_BACKEND:-cuda}"
FEATURES="${BURN_DRAGON_PROMOTION_FEATURES:-train,cuda}"
OUT_DIR="${BURN_DRAGON_PROMOTION_OUT_DIR:-$ROOT_DIR/target/ruliad-promotion-matrix/$(date -u +%Y%m%dT%H%M%SZ)}"
ARMS_CSV="${BURN_DRAGON_PROMOTION_ARMS:-jepa,jepa_nextlat,jepa_nextlat_pc_warm,cap_feedback}"
BASELINE_ARM="${BURN_DRAGON_PROMOTION_BASELINE_ARM:-jepa}"
SEEDS_CSV="${BURN_DRAGON_PROMOTION_SEEDS:-20260624,20260625,20260626}"
MAX_ITERS="${BURN_DRAGON_PROMOTION_MAX_ITERS:-2048}"
BATCH_SIZE="${BURN_DRAGON_PROMOTION_BATCH_SIZE:-4}"
BLOCK_SIZE="${BURN_DRAGON_PROMOTION_BLOCK_SIZE:-256}"
MAX_STEPS="${BURN_DRAGON_PROMOTION_MAX_STEPS:-1}"
EVAL_STEPS_CSV="${BURN_DRAGON_PROMOTION_EVAL_STEPS:-1,2,4,8}"
N_LAYER="${BURN_DRAGON_PROMOTION_N_LAYER:-4}"
N_EMBD="${BURN_DRAGON_PROMOTION_N_EMBD:-256}"
N_HEAD="${BURN_DRAGON_PROMOTION_N_HEAD:-4}"
LATENT_TOTAL="${BURN_DRAGON_PROMOTION_LATENT_TOTAL:-12288}"
NEXTLAT_START_AFTER="${BURN_DRAGON_PROMOTION_NEXTLAT_START_AFTER:-128}"
NEXTLAT_EVERY_STEPS="${BURN_DRAGON_PROMOTION_NEXTLAT_EVERY_STEPS:-16}"
JEPA_EVERY_STEPS="${BURN_DRAGON_PROMOTION_JEPA_EVERY_STEPS:-8}"
LOG_FREQUENCY="${BURN_DRAGON_PROMOTION_LOG_FREQUENCY:-16}"
CHECKPOINT_INTERVAL_ITERS="${BURN_DRAGON_PROMOTION_CHECKPOINT_INTERVAL_ITERS:-128}"
RULIAD_PROBE_ITEMS="${BURN_DRAGON_PROMOTION_RULIAD_PROBE_ITEMS:-128}"
RULIAD_PROBE_TOKENS="${BURN_DRAGON_PROMOTION_RULIAD_PROBE_TOKENS:-64}"
TIMEOUT_SECONDS="${BURN_DRAGON_PROMOTION_TIMEOUT_SECONDS:-2400}"
MAX_SYSTEM_MEMORY_FRACTION="${BURN_DRAGON_PROMOTION_MAX_SYSTEM_MEMORY_FRACTION:-0.80}"
MIN_AVAILABLE_MB="${BURN_DRAGON_PROMOTION_MIN_AVAILABLE_MB:-24576}"
BUILD_RELEASE="${BURN_DRAGON_PROMOTION_BUILD_RELEASE:-1}"
ALLOW_STALE_BINARY="${BURN_DRAGON_PROMOTION_ALLOW_STALE_BINARY:-0}"
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage:
  scripts/ruliad_promotion_matrix.sh [options]

Options:
  --arms <csv>              jepa,jepa_nextlat,jepa_nextlat_pc,jepa_nextlat_pc_warm,cap_feedback,
                            ruliad_1m_la16k_answer_window,ruliad_1m_la16k_answer_completion,
                            ruliad_1m_la16k_answer_completion_ranking,
                            ruliad_1m_la16k_answer_completion_denoising,
                            ruliad_1m_la16k_answer_completion_ranking_denoising,
                            ruliad_1m_la16k_verifier_reward,
                            ruliad_1m_la16k_verifier_vpo,
                            ruliad_1m_la16k_mixed.
  --baseline-arm <name>     Analyzer control arm. Default: jepa.
  --seeds <csv>             Seed list. Default: 20260624,20260625,20260626.
  --max-iters <n>           Iterations per trial. Default: 2048.
  --batch-size <n>          Batch size. Default: 4.
  --block-size <n>          Block size. Default: 256.
  --max-steps <n>           Fixed latent reasoning steps. Default: 1.
  --eval-steps <csv>        Validation-only eval step sweep. Default: 1,2,4,8.
  --shape L,E,H,Z           Model shape n_layer,n_embd,n_head,latent_total.
  --out-dir <path>          Output directory.
  --timeout-seconds <n>     Per-trial timeout. Default: 2400.
  --probe-items <n>         Ruliad correctness probe items. Default: 128.
  --probe-tokens <n>        Ruliad correctness generation tokens. Default: 64.
  --backend <cuda|cpu>      Backend. Default: cuda.
  --features <features>     Cargo features. Default: train,cuda.
  --dry-run                 Write overlays/manifests only.
  --no-build                Skip release build.
  --allow-stale-binary      Permit --no-build when sources are newer than release binary.
  --help                    Show this message.

Safety:
  BURN_DRAGON_PROMOTION_MAX_SYSTEM_MEMORY_FRACTION  Default: 0.80
  BURN_DRAGON_PROMOTION_MIN_AVAILABLE_MB            Default: 24576

The final analyzer applies promotion gates relative to the jepa arm by default.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arms) ARMS_CSV="$2"; shift 2 ;;
    --baseline-arm) BASELINE_ARM="$2"; shift 2 ;;
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
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --timeout-seconds) TIMEOUT_SECONDS="$2"; shift 2 ;;
    --probe-items) RULIAD_PROBE_ITEMS="$2"; shift 2 ;;
    --probe-tokens) RULIAD_PROBE_TOKENS="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --features) FEATURES="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --no-build) BUILD_RELEASE=0; shift ;;
    --allow-stale-binary) ALLOW_STALE_BINARY=1; shift ;;
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

profile_for_arm() {
  case "$1" in
    jepa)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-r1.jepa-probe128-fixed-ablation.toml"
      ;;
    jepa_nextlat)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-r1.jepa-nextlat-decoupled-delayed1024-sparse16-probe128-fixed-ablation.toml"
      ;;
    jepa_nextlat_pc)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-r1.jepa-nextlat-decoupled-delayed1024-sparse16-pc-probe128-fixed-ablation.toml"
      ;;
    jepa_nextlat_pc_warm)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-r1.jepa-nextlat-decoupled-delayed1024-sparse16-pc-warm1024-probe128-fixed-ablation.toml"
      ;;
    cap_feedback)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-r1.jepa-nextlat-decoupled-delayed1024-sparse16-probe128-fixed-ablation.toml"
      ;;
    ruliad_1m_la16k_answer_window)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-16k.self-recovery.training.toml"
      ;;
    ruliad_1m_la16k_answer_completion)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-16k.answer-completion.self-recovery.training.toml"
      ;;
    ruliad_1m_la16k_answer_completion_ranking)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-16k.answer-completion-ranking.self-recovery.training.toml"
      ;;
    ruliad_1m_la16k_answer_completion_denoising)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-16k.answer-completion-denoising.self-recovery.training.toml"
      ;;
    ruliad_1m_la16k_answer_completion_ranking_denoising)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-16k.answer-completion-ranking-denoising.self-recovery.training.toml"
      ;;
    ruliad_1m_la16k_verifier_reward)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-16k.verifier-reward.training.toml"
      ;;
    ruliad_1m_la16k_verifier_vpo)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-16k.verifier-vpo.training.toml"
      ;;
    ruliad_1m_la16k_mixed)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-16k.mixed.self-recovery.training.toml"
      ;;
    *)
      echo "unknown promotion arm: $1" >&2
      exit 2
      ;;
  esac
}

source_feedback_for_arm() {
  case "$1" in
    cap_feedback) printf 'true\n' ;;
    *) printf 'false\n' ;;
  esac
}

write_arm_profile() {
  local arm="$1"
  local profile="$2"
  local base
  local feedback

  base="$(profile_for_arm "$arm")"
  base="$(realpath "$ROOT_DIR/$base" 2>/dev/null || realpath "$base")"
  feedback="$(source_feedback_for_arm "$arm")"
  cat > "$profile" <<EOF
extends = ["$base"]

[training.events]
source_selection_capability_feedback = $feedback
EOF
}

IFS=',' read -r -a ARMS <<< "$ARMS_CSV"
mkdir -p "$OUT_DIR/profiles"

echo "ruliad promotion matrix output: $OUT_DIR"
echo "arms=${ARMS_CSV} baseline=${BASELINE_ARM} seeds=${SEEDS_CSV} max_iters=${MAX_ITERS} max_steps=${MAX_STEPS} backend=${BACKEND}"
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
    args+=(--no-build)
    if (( ALLOW_STALE_BINARY == 1 || first_arm == 0 )); then
      args+=(--allow-stale-binary)
    fi
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
  python3 "$ROOT_DIR/scripts/ruliad_promotion_matrix_analyze.py" "$OUT_DIR" --baseline-arm "$BASELINE_ARM"
fi

echo "ruliad promotion matrix complete: $OUT_DIR"
