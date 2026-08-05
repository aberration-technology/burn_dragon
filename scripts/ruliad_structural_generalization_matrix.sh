#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BACKEND="${BURN_DRAGON_STRUCTURAL_BACKEND:-cuda}"
FEATURES="${BURN_DRAGON_STRUCTURAL_FEATURES:-train,cuda}"
OUT_DIR="${BURN_DRAGON_STRUCTURAL_OUT_DIR:-$ROOT_DIR/target/ruliad-structural-generalization/$(date -u +%Y%m%dT%H%M%SZ)}"
ARMS_CSV="${BURN_DRAGON_STRUCTURAL_ARMS:-seed_ce,structural_ce,structural_dagger025}"
CANDIDATE_ARM="${BURN_DRAGON_STRUCTURAL_CANDIDATE_ARM:-structural_dagger025}"
COMPARISON_ARM="${BURN_DRAGON_STRUCTURAL_COMPARISON_ARM:-structural_ce}"
SEEDS_CSV="${BURN_DRAGON_STRUCTURAL_SEEDS:-1337,2027,9001}"
MAX_ITERS="${BURN_DRAGON_STRUCTURAL_MAX_ITERS:-1024}"
EPOCHS="${BURN_DRAGON_STRUCTURAL_EPOCHS:-4}"
BATCH_SIZE="${BURN_DRAGON_STRUCTURAL_BATCH_SIZE:-}"
BLOCK_SIZE="${BURN_DRAGON_STRUCTURAL_BLOCK_SIZE:-512}"
SHAPE="${BURN_DRAGON_STRUCTURAL_SHAPE:-2,64,4,256}"
PROBE_ITEMS="${BURN_DRAGON_STRUCTURAL_PROBE_ITEMS:-128}"
PROBE_TOKENS="${BURN_DRAGON_STRUCTURAL_PROBE_TOKENS:-32}"
POLICY_PROBE_SYMMETRY="${BURN_DRAGON_STRUCTURAL_POLICY_PROBE_SYMMETRY:-cyclic_orbit_average}"
TIMEOUT_SECONDS="${BURN_DRAGON_STRUCTURAL_TIMEOUT_SECONDS:-1800}"
MIN_PROMOTION_ITERS="${BURN_DRAGON_STRUCTURAL_MIN_PROMOTION_ITERS:-1024}"
MAX_SYSTEM_MEMORY_FRACTION="${BURN_DRAGON_STRUCTURAL_MAX_SYSTEM_MEMORY_FRACTION:-0.80}"
MIN_AVAILABLE_MB="${BURN_DRAGON_STRUCTURAL_MIN_AVAILABLE_MB:-24576}"
BUILD_RELEASE=1
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage:
  scripts/ruliad_structural_generalization_matrix.sh [options]

Options:
  --arms <csv>              seed_ce,structural_ce,structural_energy_static025,structural_energy_head_only025,structural_energy_head_only_fullrate100,structural_energy_fullrate100,structural_semantic_ce,structural_semantic_value_binding,structural_energy_value_binding025,structural_semantic_static025,structural_semantic_language_head_only025,structural_semantic_static_dense025,structural_semantic_static_prefix025,structural_semantic_static_marginal025,structural_values,structural_value_balanced,structural_static025,structural_static_marginal025,structural_static_orbit_marginal025,structural_static_orbit_worst_marginal025,structural_dagger025,structural_dagger_marginal025,structural_bc_paired_dagger_marginal025,structural_bc_paired_dagger_orbit_marginal025.
  --candidate-arm <name>    Candidate arm under promotion gates.
  --comparison-arm <name>   Matched causal baseline. Default: structural_ce.
  --seeds <csv>             Matched model seeds. Default: 1337,2027,9001.
  --max-iters <n>           Optimizer updates per trial. Default: 1024.
  --epochs <n>              Validation checkpoints per trial. Default: 4.
  --batch-size <n>          Fixed batch size. Default: 64 for the tiny CUDA shape,
                            48 for calibrated CUDA shape 4,256,8,4096, otherwise 4.
  --block-size <n>          Fixed token block size. Default: 512.
  --shape L,E,H,Z           n_layer,n_embd,n_head,latent_total. Default: 2,64,4,256.
  --probe-items <n>         Validation correctness items. Default: 128.
  --probe-tokens <n>        Completion token budget. Default: 32.
  --policy-probe-symmetry <inherit|canonical|balanced_rotation|cyclic_orbit_average>
                            Apply one evaluator contract to every matrix arm.
                            Default: cyclic_orbit_average.
  --out-dir <path>          Artifact root.
  --timeout-seconds <n>     Per-trial timeout. Default: 1800.
  --backend <cuda|cpu>      Training backend. Default: cuda.
  --features <features>     Cargo features. Default: train,cuda.
  --no-build                Reuse a current release binary.
  --dry-run                 Materialize trial configs only.

The seed-only arm is a diagnostic leakage control, not a promotion candidate.
Promotion compares --candidate-arm against structural_ce and requires at least
three matched seeds and 1,024 updates per seed.
Every arm defaults to the same exact cyclic-orbit evaluator so aggregate scores
cannot hide a weak canonical or worst candidate presentation.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arms) ARMS_CSV="$2"; shift 2 ;;
    --candidate-arm) CANDIDATE_ARM="$2"; shift 2 ;;
    --comparison-arm) COMPARISON_ARM="$2"; shift 2 ;;
    --seeds) SEEDS_CSV="$2"; shift 2 ;;
    --max-iters) MAX_ITERS="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --block-size) BLOCK_SIZE="$2"; shift 2 ;;
    --shape) SHAPE="$2"; shift 2 ;;
    --probe-items) PROBE_ITEMS="$2"; shift 2 ;;
    --probe-tokens) PROBE_TOKENS="$2"; shift 2 ;;
    --policy-probe-symmetry) POLICY_PROBE_SYMMETRY="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --timeout-seconds) TIMEOUT_SECONDS="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --features) FEATURES="$2"; shift 2 ;;
    --no-build) BUILD_RELEASE=0; shift ;;
    --dry-run) DRY_RUN=1; BUILD_RELEASE=0; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
case "$POLICY_PROBE_SYMMETRY" in
  inherit|canonical|balanced_rotation|cyclic_orbit_average) ;;
  *)
    echo "--policy-probe-symmetry must be inherit, canonical, balanced_rotation, or cyclic_orbit_average" >&2
    exit 2
    ;;
esac

if [[ "$BACKEND" == "cpu" && "$FEATURES" == "train,cuda" ]]; then
  FEATURES="train"
fi
if [[ -z "$BATCH_SIZE" ]]; then
  if [[ "$BACKEND" == "cuda" && "$SHAPE" == "2,64,4,256" ]]; then
    # Equal-token calibration on GB10: batch 64 delivered the highest useful
    # throughput. Never carry this default into an uncalibrated larger shape.
    BATCH_SIZE=64
  elif [[ "$BACKEND" == "cuda" && "$SHAPE" == "4,256,8,4096" ]]; then
    # The 512-row equal-token CE sweep plateaued from 32 through 56, with batch
    # 48 the measured winner. The full policy objective peaked at 46.0 GiB host
    # use, leaving ample room beneath the launcher's independent memory guard.
    BATCH_SIZE=48
  else
    BATCH_SIZE=4
  fi
fi
for numeric in MAX_ITERS EPOCHS BATCH_SIZE BLOCK_SIZE PROBE_ITEMS PROBE_TOKENS; do
  value="${!numeric}"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$numeric must be a positive integer (got $value)" >&2
    exit 2
  fi
done

profile_for_arm() {
  case "$1" in
    seed_ce)
      echo "crates/burn_dragon_p2p/deploy/profiles/ruliad-r3.action-policy-seed-control-fixed-ablation.toml"
      ;;
    structural_ce)
      echo "crates/burn_dragon_p2p/deploy/profiles/ruliad-r3.action-policy-fixed-ablation.toml"
      ;;
    structural_energy_static025)
      echo "crates/burn_dragon_p2p/deploy/profiles/ruliad-r3.action-policy-semantic-energy-fixed-ablation.toml"
      ;;
    structural_energy_head_only025)
      echo "crates/burn_dragon_p2p/deploy/profiles/ruliad-r3.action-policy-semantic-energy-head-only-fixed-ablation.toml"
      ;;
    structural_energy_head_only_fullrate100)
      echo "crates/burn_dragon_p2p/deploy/profiles/ruliad-r3.action-policy-semantic-energy-head-only-fullrate-ablation.toml"
      ;;
    structural_energy_fullrate100)
      echo "crates/burn_dragon_p2p/deploy/profiles/ruliad-r3.action-policy-semantic-energy-fullrate-ablation.toml"
      ;;
    structural_semantic_value_binding)
      echo "crates/burn_dragon_p2p/deploy/profiles/ruliad-r3.semantic-action-value-binding-fixed-ablation.toml"
      ;;
    structural_energy_value_binding025)
      echo "crates/burn_dragon_p2p/deploy/profiles/ruliad-r3.action-policy-semantic-energy-value-binding-fixed-ablation.toml"
      ;;
    structural_semantic_ce)
      echo "crates/burn_dragon_p2p/deploy/profiles/ruliad-r3.semantic-action-fixed-ablation.toml"
      ;;
    structural_semantic_static025)
      echo "crates/burn_dragon_p2p/deploy/profiles/ruliad-r3.semantic-action-static-fixed-ablation.toml"
      ;;
    structural_semantic_language_head_only025)
      echo "crates/burn_dragon_p2p/deploy/profiles/ruliad-r3.semantic-action-language-head-only-fixed-ablation.toml"
      ;;
    structural_semantic_static_dense025)
      echo "crates/burn_dragon_p2p/deploy/profiles/ruliad-r3.semantic-action-static-every-two-steps-fixed-ablation.toml"
      ;;
    structural_semantic_static_prefix025)
      echo "crates/burn_dragon_p2p/deploy/profiles/ruliad-r3.semantic-action-static-prefix-fixed-ablation.toml"
      ;;
    structural_semantic_static_marginal025)
      echo "crates/burn_dragon_p2p/deploy/profiles/ruliad-r3.semantic-action-static-marginal-fixed-ablation.toml"
      ;;
    structural_values)
      echo "crates/burn_dragon_p2p/deploy/profiles/ruliad-r3.action-policy-values-fixed-ablation.toml"
      ;;
    structural_value_balanced)
      echo "crates/burn_dragon_p2p/deploy/profiles/ruliad-r3.action-policy-value-balanced-fixed-ablation.toml"
      ;;
    structural_static025)
      echo "crates/burn_dragon_p2p/deploy/profiles/ruliad-r3.action-policy-static-fixed-ablation.toml"
      ;;
    structural_static_marginal025)
      echo "crates/burn_dragon_p2p/deploy/profiles/ruliad-r3.action-policy-static-marginal-fixed-ablation.toml"
      ;;
    structural_static_orbit_marginal025)
      echo "crates/burn_dragon_p2p/deploy/profiles/ruliad-r3.action-policy-static-orbit-marginal-fixed-ablation.toml"
      ;;
    structural_static_orbit_worst_marginal025)
      echo "crates/burn_dragon_p2p/deploy/profiles/ruliad-r3.action-policy-static-orbit-worst-marginal-fixed-ablation.toml"
      ;;
    structural_dagger025)
      echo "crates/burn_dragon_p2p/deploy/profiles/ruliad-r3.action-policy-dagger-fixed-ablation.toml"
      ;;
    structural_dagger_marginal025)
      echo "crates/burn_dragon_p2p/deploy/profiles/ruliad-r3.action-policy-dagger-marginal-fixed-ablation.toml"
      ;;
    structural_bc_paired_dagger_marginal025)
      echo "crates/burn_dragon_p2p/deploy/profiles/ruliad-r3.action-policy-bc-paired-dagger-marginal-fixed-ablation.toml"
      ;;
    structural_bc_paired_dagger_orbit_marginal025)
      echo "crates/burn_dragon_p2p/deploy/profiles/ruliad-r3.action-policy-bc-paired-dagger-orbit-marginal-fixed-ablation.toml"
      ;;
    *)
      echo "unknown structural-generalization arm: $1" >&2
      exit 2
      ;;
  esac
}

IFS=',' read -r -a ARMS <<< "$ARMS_CSV"
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
candidate_present=0
comparison_present=0
for arm in "${ARMS[@]}"; do
  if [[ "$arm" == "$CANDIDATE_ARM" ]]; then
    candidate_present=1
  fi
  if [[ "$arm" == "$COMPARISON_ARM" ]]; then
    comparison_present=1
  fi
done
if (( candidate_present == 0 )); then
  echo "candidate arm $CANDIDATE_ARM is not present in --arms=$ARMS_CSV" >&2
  exit 2
fi
if (( comparison_present == 0 )); then
  echo "comparison arm $COMPARISON_ARM is not present in --arms=$ARMS_CSV" >&2
  exit 2
fi
mkdir -p "$OUT_DIR"

cat > "$OUT_DIR/matrix-contract.json" <<EOF
{
  "schema_version": 1,
  "arms": "$ARMS_CSV",
  "matched_seeds": "$SEEDS_CSV",
  "max_iters": $MAX_ITERS,
  "epochs": $EPOCHS,
  "batch_size": $BATCH_SIZE,
  "block_size": $BLOCK_SIZE,
  "shape": "$SHAPE",
  "backend": "$BACKEND",
  "baseline_arm": "$COMPARISON_ARM",
  "candidate_arm": "$CANDIDATE_ARM",
  "policy_probe_candidate_symmetry": "$POLICY_PROBE_SYMMETRY",
  "minimum_promotion_iters": $MIN_PROMOTION_ITERS
}
EOF

echo "ruliad structural-generalization matrix: $OUT_DIR"
echo "arms=$ARMS_CSV seeds=$SEEDS_CSV max_iters=$MAX_ITERS epochs=$EPOCHS backend=$BACKEND"
echo "candidate_arm=$CANDIDATE_ARM"
echo "comparison_arm=$COMPARISON_ARM"
echo "shape=$SHAPE batch=$BATCH_SIZE block=$BLOCK_SIZE policy_probe_symmetry=$POLICY_PROBE_SYMMETRY"

first_arm=1
for arm in "${ARMS[@]}"; do
  profile="$(profile_for_arm "$arm")"
  arm_out="$OUT_DIR/$arm"
  args=(
    --base-profile "$profile"
    --steps 1
    # The trained arm is already fixed to one latent step. An empty sweep avoids
    # issuing an identical second correctness-generation pass at every epoch.
    --eval-steps ""
    --seeds "$SEEDS_CSV"
    --max-iters "$MAX_ITERS"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --block-size "$BLOCK_SIZE"
    --shape "$SHAPE"
    --probe-items "$PROBE_ITEMS"
    --probe-tokens "$PROBE_TOKENS"
    --policy-probe-symmetry "$POLICY_PROBE_SYMMETRY"
    --timeout-seconds "$TIMEOUT_SECONDS"
    --out-dir "$arm_out"
    --backend "$BACKEND"
    --features "$FEATURES"
  )
  if (( DRY_RUN == 1 )); then
    args+=(--dry-run)
  elif (( BUILD_RELEASE == 0 || first_arm == 0 )); then
    args+=(--no-build --allow-stale-binary)
  fi

  echo "==> arm=$arm profile=$profile"
  BURN_DRAGON_LR_STEPS_MAX_SYSTEM_MEMORY_FRACTION="$MAX_SYSTEM_MEMORY_FRACTION" \
    BURN_DRAGON_LR_STEPS_MIN_AVAILABLE_MB="$MIN_AVAILABLE_MB" \
    BURN_DRAGON_LR_STEPS_DYNAMICS_ENABLED=false \
    BURN_DRAGON_LR_STEPS_RULIAD_PROBE_EVERY_EPOCHS=2 \
    "$ROOT_DIR/scripts/latent_reasoning_steps_ablation.sh" "${args[@]}"

  if (( DRY_RUN == 0 )); then
    python3 "$ROOT_DIR/scripts/latent_reasoning_steps_analyze.py" \
      "$arm_out" --out-dir "$arm_out/analysis"
  fi
  first_arm=0
done

if (( DRY_RUN == 0 )); then
  python3 "$ROOT_DIR/scripts/ruliad_structural_generalization_analyze.py" \
    "$OUT_DIR" \
    --candidate-arm "$CANDIDATE_ARM" \
    --comparison-arm "$COMPARISON_ARM" \
    --expected-seeds "$SEEDS_CSV" \
    --minimum-promotion-iters "$MIN_PROMOTION_ITERS"
fi

echo "ruliad structural-generalization matrix complete: $OUT_DIR"
