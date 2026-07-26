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
EPOCHS="${BURN_DRAGON_PROMOTION_EPOCHS:-inherit}"
BATCH_SIZE="${BURN_DRAGON_PROMOTION_BATCH_SIZE:-4}"
BLOCK_SIZE="${BURN_DRAGON_PROMOTION_BLOCK_SIZE:-256}"
MAX_STEPS="${BURN_DRAGON_PROMOTION_MAX_STEPS:-1}"
EVAL_STEPS_CSV="${BURN_DRAGON_PROMOTION_EVAL_STEPS:-1,2,4,8}"
MAX_STEPS_OVERRIDDEN=0
N_LAYER="${BURN_DRAGON_PROMOTION_N_LAYER:-4}"
N_EMBD="${BURN_DRAGON_PROMOTION_N_EMBD:-256}"
N_HEAD="${BURN_DRAGON_PROMOTION_N_HEAD:-4}"
LATENT_TOTAL="${BURN_DRAGON_PROMOTION_LATENT_TOTAL:-12288}"
SHAPE_OVERRIDDEN=0
BATCH_SIZE_OVERRIDDEN=0
NEXTLAT_START_AFTER="${BURN_DRAGON_PROMOTION_NEXTLAT_START_AFTER:-128}"
NEXTLAT_EVERY_STEPS="${BURN_DRAGON_PROMOTION_NEXTLAT_EVERY_STEPS:-16}"
JEPA_EVERY_STEPS="${BURN_DRAGON_PROMOTION_JEPA_EVERY_STEPS:-8}"
VERIFIER_POLICY_START_AFTER="${BURN_DRAGON_PROMOTION_VERIFIER_POLICY_START_AFTER:-0}"
VPO_CORRECTNESS_MASS_FLOOR="${BURN_DRAGON_PROMOTION_VPO_CORRECTNESS_MASS_FLOOR:-inherit}"
VPO_SCHEMA_QUALITY_MASS_FLOOR="${BURN_DRAGON_PROMOTION_VPO_SCHEMA_QUALITY_MASS_FLOOR:-inherit}"
VPO_COMPLETION_HEALTH_MASS_FLOOR="${BURN_DRAGON_PROMOTION_VPO_COMPLETION_HEALTH_MASS_FLOOR:-inherit}"
RULIAD_ANSWER_CLOSE_MARKER_STRIDE="${BURN_DRAGON_PROMOTION_RULIAD_ANSWER_CLOSE_MARKER_STRIDE:-1}"
RULIAD_ANSWER_CLOSE_MARKER_WEIGHT="${BURN_DRAGON_PROMOTION_RULIAD_ANSWER_CLOSE_MARKER_WEIGHT:-1}"
RULIAD_ANSWER_SCHEMA_WEIGHT="${BURN_DRAGON_PROMOTION_RULIAD_ANSWER_SCHEMA_WEIGHT:-1}"
RULIAD_ANSWER_SCHEMA_START_WEIGHT="${BURN_DRAGON_PROMOTION_RULIAD_ANSWER_SCHEMA_START_WEIGHT:-1}"
RULIAD_ANSWER_VALUE_WEIGHT="${BURN_DRAGON_PROMOTION_RULIAD_ANSWER_VALUE_WEIGHT:-1}"
LOG_FREQUENCY="${BURN_DRAGON_PROMOTION_LOG_FREQUENCY:-16}"
CHECKPOINT_INTERVAL_ITERS="${BURN_DRAGON_PROMOTION_CHECKPOINT_INTERVAL_ITERS:-128}"
RULIAD_PROBE_ITEMS="${BURN_DRAGON_PROMOTION_RULIAD_PROBE_ITEMS:-128}"
RULIAD_PROBE_TOKENS="${BURN_DRAGON_PROMOTION_RULIAD_PROBE_TOKENS:-64}"
FIELD_BINDING_CONTRAST_WEIGHT="${BURN_DRAGON_PROMOTION_FIELD_BINDING_CONTRAST_WEIGHT:-inherit}"
FIELD_BINDING_CONTRAST_EVERY_STEPS="${BURN_DRAGON_PROMOTION_FIELD_BINDING_CONTRAST_EVERY_STEPS:-inherit}"
FIELD_BINDING_CONTRAST_MARGIN="${BURN_DRAGON_PROMOTION_FIELD_BINDING_CONTRAST_MARGIN:-inherit}"
FIELD_BINDING_CONTRAST_PAIR_WEIGHT="${BURN_DRAGON_PROMOTION_FIELD_BINDING_CONTRAST_PAIR_WEIGHT:-inherit}"
FIELD_BINDING_CONTRAST_MAX_PAIRS="${BURN_DRAGON_PROMOTION_FIELD_BINDING_CONTRAST_MAX_PAIRS:-inherit}"
FIELD_BINDING_CONTRAST_REPLAY_CAPACITY="${BURN_DRAGON_PROMOTION_FIELD_BINDING_CONTRAST_REPLAY_CAPACITY:-inherit}"
GENERATED_ATTRACTOR_REPLAY_CAPACITY="${BURN_DRAGON_PROMOTION_GENERATED_ATTRACTOR_REPLAY_CAPACITY:-inherit}"
GENERATED_ATTRACTOR_REPLAY_MIN_COUNT="${BURN_DRAGON_PROMOTION_GENERATED_ATTRACTOR_REPLAY_MIN_COUNT:-inherit}"
GENERATED_ATTRACTOR_REPLAY_MAX_CANDIDATES="${BURN_DRAGON_PROMOTION_GENERATED_ATTRACTOR_REPLAY_MAX_CANDIDATES:-inherit}"
GENERATED_ATTRACTOR_REPLAY_MIN_DISTINCT="${BURN_DRAGON_PROMOTION_GENERATED_ATTRACTOR_REPLAY_MIN_DISTINCT:-inherit}"
GENERATED_ATTRACTOR_REPLAY_MAX_DOMINANT="${BURN_DRAGON_PROMOTION_GENERATED_ATTRACTOR_REPLAY_MAX_DOMINANT:-inherit}"
VERIFIER_ROLLOUT_IMITATION_WEIGHT="${BURN_DRAGON_PROMOTION_VERIFIER_ROLLOUT_IMITATION_WEIGHT:-inherit}"
VERIFIER_ROLLOUT_RECOVERY_WEIGHT="${BURN_DRAGON_PROMOTION_VERIFIER_ROLLOUT_RECOVERY_WEIGHT:-inherit}"
VERIFIER_ROLLOUT_EVERY_STEPS="${BURN_DRAGON_PROMOTION_VERIFIER_ROLLOUT_EVERY_STEPS:-inherit}"
VERIFIER_ROLLOUT_START_AFTER="${BURN_DRAGON_PROMOTION_VERIFIER_ROLLOUT_START_AFTER:-inherit}"
VERIFIER_ROLLOUT_MIN_PARTIAL_PROGRESS_PPM="${BURN_DRAGON_PROMOTION_VERIFIER_ROLLOUT_MIN_PARTIAL_PROGRESS_PPM:-inherit}"
VERIFIER_ROLLOUT_MIN_COMPLETION_QUALITY_PPM="${BURN_DRAGON_PROMOTION_VERIFIER_ROLLOUT_MIN_COMPLETION_QUALITY_PPM:-inherit}"
VERIFIER_ROLLOUT_MAX_ROWS_PER_STEP="${BURN_DRAGON_PROMOTION_VERIFIER_ROLLOUT_MAX_ROWS_PER_STEP:-inherit}"
STRUCTURED_RECOVERY_WEIGHT="${BURN_DRAGON_PROMOTION_STRUCTURED_RECOVERY_WEIGHT:-inherit}"
STRUCTURED_RECOVERY_EVERY_STEPS="${BURN_DRAGON_PROMOTION_STRUCTURED_RECOVERY_EVERY_STEPS:-inherit}"
STRUCTURED_RECOVERY_START_AFTER="${BURN_DRAGON_PROMOTION_STRUCTURED_RECOVERY_START_AFTER:-inherit}"
STRUCTURED_RECOVERY_MAX_COMPLETION_TOKENS="${BURN_DRAGON_PROMOTION_STRUCTURED_RECOVERY_MAX_COMPLETION_TOKENS:-inherit}"
STRUCTURED_RECOVERY_NEGATIVE_COUNT="${BURN_DRAGON_PROMOTION_STRUCTURED_RECOVERY_NEGATIVE_COUNT:-inherit}"
STRUCTURED_RECOVERY_TEMPLATE_NEGATIVE_COUNT="${BURN_DRAGON_PROMOTION_STRUCTURED_RECOVERY_TEMPLATE_NEGATIVE_COUNT:-inherit}"
STRUCTURED_RECOVERY_SCHEMA_NEGATIVE_COUNT="${BURN_DRAGON_PROMOTION_STRUCTURED_RECOVERY_SCHEMA_NEGATIVE_COUNT:-inherit}"
MIN_MATURE_ITERS="${BURN_DRAGON_PROMOTION_MIN_MATURE_ITERS:-1024}"
DYNAMICS_ENABLED="${BURN_DRAGON_PROMOTION_DYNAMICS_ENABLED:-true}"
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
                            ruliad_1m_la16k_answer_completion_recovery_denoising,
                            ruliad_1m_la16k_answer_completion_denoising,
                            ruliad_1m_la16k_answer_completion_ranking_denoising,
                            ruliad_1m_la16k_answer_completion_rollout,
                            ruliad_1m_la16k_structured_contrast,
                            ruliad_1m_la16k_field_binding_contrast,
                            ruliad_1m_la16k_field_binding_recovery,
                            ruliad_1m_la16k_field_binding_contrast_rollout,
                            ruliad_smoke_answer_window,
                            ruliad_smoke_field_binding_contrast,
                            ruliad_1m_la64k_answer_window,
                            ruliad_1m_la64k_jepa,
                            ruliad_1m_la64k_answer_completion_stable,
                            ruliad_1m_la64k_answer_contract,
                            ruliad_1m_la64k_answer_contract_schema,
                            ruliad_1m_la64k_answer_contract_schema_start,
                            ruliad_1m_la64k_answer_contract_schema_trace_answer,
                            ruliad_1m_la64k_answer_contract_schema_mixed_trace,
                            ruliad_1m_la64k_answer_contract_schema_field_binding,
                            ruliad_1m_la64k_answer_contract_value_binding,
                            ruliad_1m_la64k_answer_contract_values,
                            ruliad_1m_la64k_answer_completion_recovery,
                            ruliad_1m_la64k_field_binding_contrast,
                            ruliad_1m_la16k_verifier_rollout_imitation,
                            ruliad_1m_la16k_verifier_reward,
                            ruliad_1m_la16k_verifier_vpo,
                            ruliad_1m_la16k_verifier_vpo_oracle,
                            ruliad_1m_la16k_verifier_vpo_oracle_structured,
                            ruliad_1m_la16k_verifier_vpo_oracle_structured_contrast,
                            ruliad_1m_la16k_verifier_vpo_oracle_structured_contrast_field_binding,
                            ruliad_1m_la16k_verifier_vpo_oracle_structured_contrast_field_binding_permissive_attractor,
                            ruliad_1m_la16k_verifier_vpo_oracle_structured_contrast_field_binding_no_attractor,
                            ruliad_1m_la16k_mixed.
  --baseline-arm <name>     Analyzer control arm. Default: jepa.
  --seeds <csv>             Seed list. Default: 20260624,20260625,20260626.
  --max-iters <n>           Iterations per trial. Default: 2048.
  --epochs <n|inherit>      Split max_iters into this many bounded logical epochs. Default: inherit.
  --batch-size <n>          Batch size. Default: 4.
  --block-size <n>          Block size. Default: 256.
  --max-steps <n>           Fixed latent reasoning steps. Default: 1.
  --eval-steps <csv>        Validation-only eval step sweep. Default: 1,2,4,8.
  --shape L,E,H,Z           Model shape n_layer,n_embd,n_head,latent_total.
  --out-dir <path>          Output directory.
  --timeout-seconds <n>     Per-trial timeout. Default: 2400.
  --probe-items <n>         Ruliad correctness probe items. Default: 128.
  --probe-tokens <n>        Ruliad correctness generation tokens. Default: 64.
  --min-mature-iters <n>    Minimum iterations before promotion gates are mature. Default: 1024.
  --verifier-policy-start <n>
                            Override verifier policy start steps for verifier arms. Default: 0.
  --vpo-floors c,s,h        Override VPO correctness/schema-quality/completion-health mass floors
                            for verifier arms. Use "inherit" for a field to keep the profile value.
  BURN_DRAGON_PROMOTION_GENERATED_ATTRACTOR_REPLAY_CAPACITY
                            Override generated-attractor replay capacity globally; the
                            *_field_binding_no_attractor arm always forces capacity 0.
                            *_field_binding_permissive_attractor arm forces min_distinct=1 and
                            max_dominant=1.0 unless overridden globally.
  BURN_DRAGON_PROMOTION_GENERATED_ATTRACTOR_REPLAY_MIN_DISTINCT
                            Override minimum distinct wrong answers before generated-attractor replay.
  BURN_DRAGON_PROMOTION_GENERATED_ATTRACTOR_REPLAY_MAX_DOMINANT
                            Override maximum dominant wrong-answer fraction before generated-attractor replay.
  --dynamics <true|false|inherit>
                            Override training.dynamics.enabled for generated arms. Default: true.
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
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; BATCH_SIZE_OVERRIDDEN=1; shift 2 ;;
    --block-size) BLOCK_SIZE="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; MAX_STEPS_OVERRIDDEN=1; shift 2 ;;
    --eval-steps) EVAL_STEPS_CSV="$2"; shift 2 ;;
    --shape)
      IFS=',' read -r N_LAYER N_EMBD N_HEAD LATENT_TOTAL <<< "$2"
      SHAPE_OVERRIDDEN=1
      shift 2
      ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --timeout-seconds) TIMEOUT_SECONDS="$2"; shift 2 ;;
    --probe-items) RULIAD_PROBE_ITEMS="$2"; shift 2 ;;
    --probe-tokens) RULIAD_PROBE_TOKENS="$2"; shift 2 ;;
    --min-mature-iters) MIN_MATURE_ITERS="$2"; shift 2 ;;
    --verifier-policy-start) VERIFIER_POLICY_START_AFTER="$2"; shift 2 ;;
    --vpo-floors)
      IFS=',' read -r VPO_CORRECTNESS_MASS_FLOOR VPO_SCHEMA_QUALITY_MASS_FLOOR VPO_COMPLETION_HEALTH_MASS_FLOOR <<< "$2"
      shift 2
      ;;
    --dynamics) DYNAMICS_ENABLED="$2"; shift 2 ;;
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
if ! [[ "$MIN_MATURE_ITERS" =~ ^[0-9]+$ ]]; then
  echo "--min-mature-iters must be a non-negative integer" >&2
  exit 2
fi
if [[ "$EPOCHS" != "inherit" ]] && ! [[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--epochs must be inherit or a positive integer" >&2
  exit 2
fi
case "$DYNAMICS_ENABLED" in
  true|false|inherit) ;;
  *) echo "--dynamics must be true, false, or inherit" >&2; exit 2 ;;
esac

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
    ruliad_1m_la16k_answer_completion_recovery_denoising)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-16k.answer-completion-recovery-denoising.self-recovery.training.toml"
      ;;
    ruliad_1m_la16k_answer_completion_denoising)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-16k.answer-completion-denoising.self-recovery.training.toml"
      ;;
    ruliad_1m_la16k_answer_completion_ranking_denoising)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-16k.answer-completion-ranking-denoising.self-recovery.training.toml"
      ;;
    ruliad_1m_la16k_answer_completion_rollout)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-16k.answer-completion.self-recovery.training.toml"
      ;;
    ruliad_1m_la16k_structured_contrast)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-16k.structured-contrast.training.toml"
      ;;
    ruliad_1m_la16k_field_binding_contrast)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-16k.field-binding-contrast.training.toml"
      ;;
    ruliad_1m_la16k_field_binding_recovery)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-16k.field-binding-recovery.training.toml"
      ;;
    ruliad_1m_la16k_field_binding_contrast_rollout)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-16k.field-binding-contrast.training.toml"
      ;;
    ruliad_smoke_answer_window)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-16k.training.toml"
      ;;
    ruliad_smoke_field_binding_contrast)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-16k.field-binding-contrast.training.toml"
      ;;
    ruliad_1m_la64k_answer_window)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-64k.training.toml"
      ;;
    ruliad_1m_la64k_jepa)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-64k.jepa.training.toml"
      ;;
    ruliad_1m_la64k_answer_completion_stable)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-64k.answer-completion-stable.training.toml"
      ;;
    ruliad_1m_la64k_answer_contract)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-64k.answer-contract.training.toml"
      ;;
    ruliad_1m_la64k_answer_contract_schema)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-64k.answer-contract-schema.training.toml"
      ;;
    ruliad_1m_la64k_answer_contract_schema_start)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-64k.answer-contract-schema-start.training.toml"
      ;;
    ruliad_1m_la64k_answer_contract_schema_trace_answer)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-64k.answer-contract-schema-trace-answer.training.toml"
      ;;
    ruliad_1m_la64k_answer_contract_schema_mixed_trace)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-64k.answer-contract-schema-mixed-trace.training.toml"
      ;;
    ruliad_1m_la64k_answer_contract_schema_field_binding)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-64k.answer-contract-schema-field-binding.training.toml"
      ;;
    ruliad_1m_la64k_answer_contract_value_binding)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-64k.answer-contract-value-binding.training.toml"
      ;;
    ruliad_1m_la64k_answer_contract_values)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-64k.answer-contract-values.training.toml"
      ;;
    ruliad_1m_la64k_answer_completion_recovery)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-64k.answer-completion-recovery.training.toml"
      ;;
    ruliad_1m_la64k_field_binding_contrast)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-64k.field-binding-contrast.training.toml"
      ;;
    ruliad_1m_la16k_verifier_rollout_imitation)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-16k.verifier-rollout-imitation.training.toml"
      ;;
    ruliad_1m_la16k_verifier_reward)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-16k.verifier-reward.training.toml"
      ;;
    ruliad_1m_la16k_verifier_vpo)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-16k.verifier-vpo.training.toml"
      ;;
    ruliad_1m_la16k_verifier_vpo_oracle)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-16k.verifier-vpo-oracle.training.toml"
      ;;
    ruliad_1m_la16k_verifier_vpo_oracle_structured)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-16k.verifier-vpo-oracle-structured.training.toml"
      ;;
    ruliad_1m_la16k_verifier_vpo_oracle_structured_contrast)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-16k.verifier-vpo-oracle-structured-contrast.training.toml"
      ;;
    ruliad_1m_la16k_verifier_vpo_oracle_structured_contrast_field_binding)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-16k.verifier-vpo-oracle-structured-contrast-field-binding.training.toml"
      ;;
    ruliad_1m_la16k_verifier_vpo_oracle_structured_contrast_field_binding_permissive_attractor)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-16k.verifier-vpo-oracle-structured-contrast-field-binding.training.toml"
      ;;
    ruliad_1m_la16k_verifier_vpo_oracle_structured_contrast_field_binding_no_attractor)
      printf '%s\n' "crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-16k.verifier-vpo-oracle-structured-contrast-field-binding.training.toml"
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
    cap_feedback|ruliad_*) printf 'true\n' ;;
    *) printf 'false\n' ;;
  esac
}

answer_ranking_for_arm() {
  case "$1" in
    ruliad_1m_la16k_answer_completion_ranking|ruliad_1m_la16k_answer_completion_ranking_denoising)
      printf 'true\n'
      ;;
    *)
      printf '%s\n' "${BURN_DRAGON_PROMOTION_ANSWER_RANKING:-inherit}"
      ;;
  esac
}

answer_denoising_for_arm() {
  case "$1" in
    ruliad_1m_la16k_answer_completion_denoising|ruliad_1m_la16k_answer_completion_ranking_denoising)
      printf 'true\n'
      ;;
    *)
      printf '%s\n' "${BURN_DRAGON_PROMOTION_ANSWER_DENOISING:-inherit}"
      ;;
  esac
}

ruliad_mask_high_entropy_for_arm() {
  case "$1" in
    jepa|jepa_nextlat|jepa_nextlat_pc|jepa_nextlat_pc_warm|cap_feedback|ruliad_*)
      printf '%s\n' "${BURN_DRAGON_PROMOTION_RULIAD_MASK_HIGH_ENTROPY:-true}"
      ;;
    *)
      printf '%s\n' "${BURN_DRAGON_PROMOTION_RULIAD_MASK_HIGH_ENTROPY:-false}"
      ;;
  esac
}

rollout_unlikelihood_for_arm() {
  case "$1" in
    ruliad_1m_la16k_answer_completion_rollout|ruliad_1m_la16k_field_binding_contrast_rollout)
      printf 'true\n'
      ;;
    *)
      printf '%s\n' "${BURN_DRAGON_PROMOTION_ROLLOUT_UNLIKELIHOOD:-false}"
      ;;
  esac
}

rollout_value_for_arm() {
  local arm="$1"
  local env_name="$2"
  local rollout_default="$3"
  local default_value="$4"
  if [[ "$arm" == "ruliad_1m_la16k_answer_completion_rollout" || "$arm" == "ruliad_1m_la16k_field_binding_contrast_rollout" ]]; then
    printf '%s\n' "${!env_name:-$rollout_default}"
  else
    printf '%s\n' "${!env_name:-$default_value}"
  fi
}

latent_total_for_arm() {
  case "$1" in
    ruliad_1m_la64k_*)
      printf '%s\n' "${BURN_DRAGON_PROMOTION_LATENT_TOTAL:-65536}"
      ;;
    ruliad_smoke_*)
      if (( SHAPE_OVERRIDDEN == 0 )) && [[ -z "${BURN_DRAGON_PROMOTION_LATENT_TOTAL+x}" ]]; then
        printf '1024\n'
      else
        printf '%s\n' "$LATENT_TOTAL"
      fi
      ;;
    *)
      printf '%s\n' "$LATENT_TOTAL"
      ;;
  esac
}

n_layer_for_arm() {
  if [[ "$1" == ruliad_smoke_* ]] && (( SHAPE_OVERRIDDEN == 0 )) && [[ -z "${BURN_DRAGON_PROMOTION_N_LAYER+x}" ]]; then
    printf '2\n'
  else
    printf '%s\n' "$N_LAYER"
  fi
}

n_embd_for_arm() {
  if [[ "$1" == ruliad_smoke_* ]] && (( SHAPE_OVERRIDDEN == 0 )) && [[ -z "${BURN_DRAGON_PROMOTION_N_EMBD+x}" ]]; then
    printf '128\n'
  else
    printf '%s\n' "$N_EMBD"
  fi
}

n_head_for_arm() {
  if [[ "$1" == ruliad_smoke_* ]] && (( SHAPE_OVERRIDDEN == 0 )) && [[ -z "${BURN_DRAGON_PROMOTION_N_HEAD+x}" ]]; then
    printf '4\n'
  else
    printf '%s\n' "$N_HEAD"
  fi
}

batch_size_for_arm() {
  case "$1" in
    ruliad_1m_la64k_*)
      if (( BATCH_SIZE_OVERRIDDEN == 0 )) && [[ -z "${BURN_DRAGON_PROMOTION_BATCH_SIZE+x}" ]]; then
        printf '1\n'
      else
        printf '%s\n' "$BATCH_SIZE"
      fi
      ;;
    ruliad_smoke_*)
      if (( BATCH_SIZE_OVERRIDDEN == 0 )) && [[ -z "${BURN_DRAGON_PROMOTION_BATCH_SIZE+x}" ]]; then
        printf '1\n'
      else
        printf '%s\n' "$BATCH_SIZE"
      fi
      ;;
    *)
      printf '%s\n' "$BATCH_SIZE"
      ;;
  esac
}

max_steps_for_arm() {
  case "$1" in
    ruliad_1m_la64k_answer_completion_stable|ruliad_1m_la64k_answer_contract|ruliad_1m_la64k_answer_contract_schema|ruliad_1m_la64k_answer_contract_schema_start|ruliad_1m_la64k_answer_contract_schema_trace_answer|ruliad_1m_la64k_answer_contract_schema_mixed_trace|ruliad_1m_la64k_answer_contract_schema_field_binding|ruliad_1m_la64k_answer_contract_value_binding|ruliad_1m_la64k_answer_contract_values|ruliad_1m_la64k_answer_completion_recovery|ruliad_1m_la64k_field_binding_contrast)
      if (( MAX_STEPS_OVERRIDDEN == 0 )) && [[ -z "${BURN_DRAGON_PROMOTION_MAX_STEPS+x}" ]]; then
        printf '2\n'
      else
        printf '%s\n' "$MAX_STEPS"
      fi
      ;;
    *)
      printf '%s\n' "$MAX_STEPS"
      ;;
  esac
}

residual_gate_for_arm() {
  case "$1" in
    ruliad_1m_la64k_answer_completion_stable|ruliad_1m_la64k_answer_contract|ruliad_1m_la64k_answer_contract_schema|ruliad_1m_la64k_answer_contract_schema_start|ruliad_1m_la64k_answer_contract_schema_trace_answer|ruliad_1m_la64k_answer_contract_schema_mixed_trace|ruliad_1m_la64k_answer_contract_schema_field_binding|ruliad_1m_la64k_answer_contract_value_binding|ruliad_1m_la64k_answer_contract_values|ruliad_1m_la64k_answer_completion_recovery|ruliad_1m_la64k_field_binding_contrast)
      printf '%s\n' "${BURN_DRAGON_PROMOTION_RESIDUAL_GATE:-true}"
      ;;
    *)
      printf '%s\n' "${BURN_DRAGON_PROMOTION_RESIDUAL_GATE:-false}"
      ;;
  esac
}

residual_gate_init_for_arm() {
  case "$1" in
    ruliad_1m_la64k_answer_completion_stable|ruliad_1m_la64k_answer_contract|ruliad_1m_la64k_answer_contract_schema|ruliad_1m_la64k_answer_contract_schema_start|ruliad_1m_la64k_answer_contract_schema_trace_answer|ruliad_1m_la64k_answer_contract_schema_mixed_trace|ruliad_1m_la64k_answer_contract_schema_field_binding|ruliad_1m_la64k_answer_contract_value_binding|ruliad_1m_la64k_answer_contract_values|ruliad_1m_la64k_answer_completion_recovery|ruliad_1m_la64k_field_binding_contrast)
      printf '%s\n' "${BURN_DRAGON_PROMOTION_RESIDUAL_GATE_INIT:-0.05}"
      ;;
    *)
      printf '%s\n' "${BURN_DRAGON_PROMOTION_RESIDUAL_GATE_INIT:-0.25}"
      ;;
  esac
}

normalize_steps_for_arm() {
  case "$1" in
    ruliad_1m_la64k_answer_completion_stable|ruliad_1m_la64k_answer_contract|ruliad_1m_la64k_answer_contract_schema|ruliad_1m_la64k_answer_contract_schema_start|ruliad_1m_la64k_answer_contract_schema_trace_answer|ruliad_1m_la64k_answer_contract_schema_mixed_trace|ruliad_1m_la64k_answer_contract_schema_field_binding|ruliad_1m_la64k_answer_contract_value_binding|ruliad_1m_la64k_answer_contract_values|ruliad_1m_la64k_answer_completion_recovery|ruliad_1m_la64k_field_binding_contrast)
      printf '%s\n' "${BURN_DRAGON_PROMOTION_NORMALIZE_STEPS:-true}"
      ;;
    *)
      printf '%s\n' "${BURN_DRAGON_PROMOTION_NORMALIZE_STEPS:-false}"
      ;;
  esac
}

structured_recovery_value_for_arm() {
  local arm="$1"
  local value="$2"
  case "$arm" in
    ruliad_1m_la16k_answer_completion_recovery_denoising|ruliad_1m_la16k_field_binding_recovery)
      printf '%s\n' "$value"
      ;;
    *)
      printf 'inherit\n'
      ;;
  esac
}

verifier_policy_for_arm() {
  case "$1" in
    ruliad_1m_la16k_verifier_*) printf 'true\n' ;;
    *) printf 'false\n' ;;
  esac
}

generated_attractor_capacity_for_arm() {
  case "$1" in
    ruliad_1m_la16k_verifier_vpo_oracle_structured_contrast_field_binding_no_attractor)
      printf '0\n'
      ;;
    *)
      printf '%s\n' "$GENERATED_ATTRACTOR_REPLAY_CAPACITY"
      ;;
  esac
}

generated_attractor_min_distinct_for_arm() {
  case "$1" in
    ruliad_1m_la16k_verifier_vpo_oracle_structured_contrast_field_binding_permissive_attractor)
      printf '%s\n' "${BURN_DRAGON_PROMOTION_GENERATED_ATTRACTOR_REPLAY_MIN_DISTINCT:-1}"
      ;;
    *)
      printf '%s\n' "$GENERATED_ATTRACTOR_REPLAY_MIN_DISTINCT"
      ;;
  esac
}

generated_attractor_max_dominant_for_arm() {
  case "$1" in
    ruliad_1m_la16k_verifier_vpo_oracle_structured_contrast_field_binding_permissive_attractor)
      printf '%s\n' "${BURN_DRAGON_PROMOTION_GENERATED_ATTRACTOR_REPLAY_MAX_DOMINANT:-1.0}"
      ;;
    *)
      printf '%s\n' "$GENERATED_ATTRACTOR_REPLAY_MAX_DOMINANT"
      ;;
  esac
}

ruliad_answer_close_stride_for_arm() {
  case "$1" in
    ruliad_1m_la64k_answer_contract_schema|ruliad_1m_la64k_answer_contract_schema_start|ruliad_1m_la64k_answer_contract_schema_trace_answer|ruliad_1m_la64k_answer_contract_schema_mixed_trace|ruliad_1m_la64k_answer_contract_schema_field_binding|ruliad_1m_la64k_answer_contract_value_binding)
      printf '%s\n' "${BURN_DRAGON_PROMOTION_RULIAD_ANSWER_CONTRACT_SCHEMA_CLOSE_MARKER_STRIDE:-4}"
      ;;
    ruliad_1m_la64k_answer_contract_values)
      printf '%s\n' "${BURN_DRAGON_PROMOTION_RULIAD_ANSWER_CONTRACT_VALUES_CLOSE_MARKER_STRIDE:-1}"
      ;;
    *)
      printf '%s\n' "$RULIAD_ANSWER_CLOSE_MARKER_STRIDE"
      ;;
  esac
}

ruliad_answer_close_weight_for_arm() {
  case "$1" in
    ruliad_1m_la64k_answer_contract_schema|ruliad_1m_la64k_answer_contract_schema_start|ruliad_1m_la64k_answer_contract_schema_trace_answer|ruliad_1m_la64k_answer_contract_schema_mixed_trace|ruliad_1m_la64k_answer_contract_schema_field_binding|ruliad_1m_la64k_answer_contract_value_binding)
      printf '%s\n' "${BURN_DRAGON_PROMOTION_RULIAD_ANSWER_CONTRACT_SCHEMA_CLOSE_MARKER_WEIGHT:-$RULIAD_ANSWER_CLOSE_MARKER_WEIGHT}"
      ;;
    ruliad_1m_la64k_answer_contract_values)
      printf '%s\n' "${BURN_DRAGON_PROMOTION_RULIAD_ANSWER_CONTRACT_VALUES_CLOSE_MARKER_WEIGHT:-2}"
      ;;
    *)
      printf '%s\n' "$RULIAD_ANSWER_CLOSE_MARKER_WEIGHT"
      ;;
  esac
}

ruliad_answer_schema_weight_for_arm() {
  case "$1" in
    ruliad_1m_la64k_answer_contract_schema|ruliad_1m_la64k_answer_contract_schema_start|ruliad_1m_la64k_answer_contract_schema_trace_answer|ruliad_1m_la64k_answer_contract_schema_mixed_trace|ruliad_1m_la64k_answer_contract_schema_field_binding|ruliad_1m_la64k_answer_contract_value_binding)
      printf '%s\n' "${BURN_DRAGON_PROMOTION_RULIAD_ANSWER_CONTRACT_SCHEMA_SCHEMA_WEIGHT:-4}"
      ;;
    ruliad_1m_la64k_answer_contract_values)
      printf '%s\n' "${BURN_DRAGON_PROMOTION_RULIAD_ANSWER_CONTRACT_VALUES_SCHEMA_WEIGHT:-1}"
      ;;
    *)
      printf '%s\n' "$RULIAD_ANSWER_SCHEMA_WEIGHT"
      ;;
  esac
}

ruliad_answer_schema_start_weight_for_arm() {
  case "$1" in
    ruliad_1m_la64k_answer_contract_schema_start)
      printf '%s\n' "${BURN_DRAGON_PROMOTION_RULIAD_ANSWER_CONTRACT_SCHEMA_START_WEIGHT:-12}"
      ;;
    *)
      printf '%s\n' "$RULIAD_ANSWER_SCHEMA_START_WEIGHT"
      ;;
  esac
}

ruliad_answer_value_weight_for_arm() {
  case "$1" in
    ruliad_1m_la64k_answer_contract_schema|ruliad_1m_la64k_answer_contract_schema_start|ruliad_1m_la64k_answer_contract_schema_trace_answer|ruliad_1m_la64k_answer_contract_schema_mixed_trace|ruliad_1m_la64k_answer_contract_schema_field_binding|ruliad_1m_la64k_answer_contract_value_binding)
      printf '%s\n' "${BURN_DRAGON_PROMOTION_RULIAD_ANSWER_CONTRACT_SCHEMA_VALUE_WEIGHT:-$RULIAD_ANSWER_VALUE_WEIGHT}"
      ;;
    ruliad_1m_la64k_answer_contract_values)
      printf '%s\n' "${BURN_DRAGON_PROMOTION_RULIAD_ANSWER_CONTRACT_VALUES_VALUE_WEIGHT:-6}"
      ;;
    *)
      printf '%s\n' "$RULIAD_ANSWER_VALUE_WEIGHT"
      ;;
  esac
}

write_arm_profile() {
  local arm="$1"
  local profile="$2"
  local base
  local feedback
  local verifier_policy

  base="$(profile_for_arm "$arm")"
  base="$(realpath "$ROOT_DIR/$base" 2>/dev/null || realpath "$base")"
  feedback="$(source_feedback_for_arm "$arm")"
  verifier_policy="$(verifier_policy_for_arm "$arm")"
  cat > "$profile" <<EOF
extends = ["$base"]

[training.events]
source_selection_capability_feedback = $feedback
EOF
  if [[ "$DYNAMICS_ENABLED" != "inherit" ]]; then
    cat >> "$profile" <<EOF

[training.dynamics]
enabled = $DYNAMICS_ENABLED
EOF
  fi
  if [[ "$verifier_policy" == "true" ]]; then
    cat >> "$profile" <<EOF

[training.ruliad_supervision.verifier_reward]
start_after_steps = $VERIFIER_POLICY_START_AFTER
EOF
    if [[ "$VPO_CORRECTNESS_MASS_FLOOR" != "inherit" ]]; then
      echo "vpo_correctness_mass_floor = $VPO_CORRECTNESS_MASS_FLOOR" >> "$profile"
    fi
    if [[ "$VPO_SCHEMA_QUALITY_MASS_FLOOR" != "inherit" ]]; then
      echo "vpo_schema_quality_mass_floor = $VPO_SCHEMA_QUALITY_MASS_FLOOR" >> "$profile"
    fi
    if [[ "$VPO_COMPLETION_HEALTH_MASS_FLOOR" != "inherit" ]]; then
      echo "vpo_completion_health_mass_floor = $VPO_COMPLETION_HEALTH_MASS_FLOOR" >> "$profile"
    fi
  fi
}

IFS=',' read -r -a ARMS <<< "$ARMS_CSV"
mkdir -p "$OUT_DIR/profiles"

echo "ruliad promotion matrix output: $OUT_DIR"
echo "arms=${ARMS_CSV} baseline=${BASELINE_ARM} seeds=${SEEDS_CSV} max_iters=${MAX_ITERS} epochs=${EPOCHS} max_steps=${MAX_STEPS} backend=${BACKEND}"
echo "shape: n_layer=$N_LAYER n_embd=$N_EMBD n_head=$N_HEAD latent_total=$LATENT_TOTAL block_size=$BLOCK_SIZE batch_size=$BATCH_SIZE"
echo "verifier policy start override for verifier arms: $VERIFIER_POLICY_START_AFTER"
echo "VPO floor overrides for verifier arms: correctness=$VPO_CORRECTNESS_MASS_FLOOR schema=$VPO_SCHEMA_QUALITY_MASS_FLOOR health=$VPO_COMPLETION_HEALTH_MASS_FLOOR"
echo "answer weighting: close_stride=$RULIAD_ANSWER_CLOSE_MARKER_STRIDE close=$RULIAD_ANSWER_CLOSE_MARKER_WEIGHT schema=$RULIAD_ANSWER_SCHEMA_WEIGHT schema_start=$RULIAD_ANSWER_SCHEMA_START_WEIGHT value=$RULIAD_ANSWER_VALUE_WEIGHT"
echo "structured recovery overrides: weight=$STRUCTURED_RECOVERY_WEIGHT every=$STRUCTURED_RECOVERY_EVERY_STEPS start=$STRUCTURED_RECOVERY_START_AFTER tokens=$STRUCTURED_RECOVERY_MAX_COMPLETION_TOKENS field=$STRUCTURED_RECOVERY_NEGATIVE_COUNT template=$STRUCTURED_RECOVERY_TEMPLATE_NEGATIVE_COUNT schema=$STRUCTURED_RECOVERY_SCHEMA_NEGATIVE_COUNT"
echo "generated attractor replay overrides: capacity=$GENERATED_ATTRACTOR_REPLAY_CAPACITY min_count=$GENERATED_ATTRACTOR_REPLAY_MIN_COUNT max_candidates=$GENERATED_ATTRACTOR_REPLAY_MAX_CANDIDATES min_distinct=$GENERATED_ATTRACTOR_REPLAY_MIN_DISTINCT max_dominant=$GENERATED_ATTRACTOR_REPLAY_MAX_DOMINANT"
echo "verifier rollout overrides: imitation_weight=$VERIFIER_ROLLOUT_IMITATION_WEIGHT recovery_weight=$VERIFIER_ROLLOUT_RECOVERY_WEIGHT every=$VERIFIER_ROLLOUT_EVERY_STEPS start=$VERIFIER_ROLLOUT_START_AFTER min_partial=$VERIFIER_ROLLOUT_MIN_PARTIAL_PROGRESS_PPM min_quality=$VERIFIER_ROLLOUT_MIN_COMPLETION_QUALITY_PPM max_rows=$VERIFIER_ROLLOUT_MAX_ROWS_PER_STEP"
echo "RAM guards: max_system_memory_fraction=$MAX_SYSTEM_MEMORY_FRACTION min_available_mb=$MIN_AVAILABLE_MB dynamics=$DYNAMICS_ENABLED"

first_arm=1
for arm in "${ARMS[@]}"; do
  arm_profile="$OUT_DIR/profiles/${arm}.toml"
  arm_out="$OUT_DIR/$arm"
  arm_n_layer="$(n_layer_for_arm "$arm")"
  arm_n_embd="$(n_embd_for_arm "$arm")"
  arm_n_head="$(n_head_for_arm "$arm")"
  arm_latent_total="$(latent_total_for_arm "$arm")"
  arm_batch_size="$(batch_size_for_arm "$arm")"
  arm_max_steps="$(max_steps_for_arm "$arm")"
  arm_residual_gate="$(residual_gate_for_arm "$arm")"
  arm_residual_gate_init="$(residual_gate_init_for_arm "$arm")"
  arm_normalize_steps="$(normalize_steps_for_arm "$arm")"
  arm_answer_ranking="$(answer_ranking_for_arm "$arm")"
  arm_answer_denoising="$(answer_denoising_for_arm "$arm")"
  arm_mask_high_entropy="$(ruliad_mask_high_entropy_for_arm "$arm")"
  arm_rollout_unlikelihood="$(rollout_unlikelihood_for_arm "$arm")"
  arm_rollout_weight="$(rollout_value_for_arm "$arm" BURN_DRAGON_PROMOTION_ROLLOUT_UNLIKELIHOOD_WEIGHT 0.02 0.0)"
  arm_rollout_margin_weight="$(rollout_value_for_arm "$arm" BURN_DRAGON_PROMOTION_ROLLOUT_UNLIKELIHOOD_MARGIN_WEIGHT 0.0 0.0)"
  arm_rollout_margin="$(rollout_value_for_arm "$arm" BURN_DRAGON_PROMOTION_ROLLOUT_UNLIKELIHOOD_MARGIN 0.0 0.0)"
  arm_rollout_recovery_weight="$(rollout_value_for_arm "$arm" BURN_DRAGON_PROMOTION_ROLLOUT_UNLIKELIHOOD_RECOVERY_WEIGHT 0.0 0.0)"
  arm_rollout_sequence_recovery_weight="$(rollout_value_for_arm "$arm" BURN_DRAGON_PROMOTION_ROLLOUT_UNLIKELIHOOD_SEQUENCE_RECOVERY_WEIGHT 0.0 0.0)"
  arm_rollout_entropy_weight="$(rollout_value_for_arm "$arm" BURN_DRAGON_PROMOTION_ROLLOUT_UNLIKELIHOOD_ENTROPY_WEIGHT 0.005 0.0)"
  arm_rollout_target_entropy_bits="$(rollout_value_for_arm "$arm" BURN_DRAGON_PROMOTION_ROLLOUT_UNLIKELIHOOD_TARGET_ENTROPY_BITS 2.0 0.0)"
  arm_rollout_cycle_weight="$(rollout_value_for_arm "$arm" BURN_DRAGON_PROMOTION_ROLLOUT_UNLIKELIHOOD_CYCLE_WEIGHT 0.05 0.0)"
  arm_rollout_cycle_margin_weight="$(rollout_value_for_arm "$arm" BURN_DRAGON_PROMOTION_ROLLOUT_UNLIKELIHOOD_CYCLE_MARGIN_WEIGHT 0.0 0.0)"
  arm_rollout_cycle_min_lag="$(rollout_value_for_arm "$arm" BURN_DRAGON_PROMOTION_ROLLOUT_UNLIKELIHOOD_CYCLE_MIN_LAG 2 2)"
  arm_rollout_cycle_max_lag="$(rollout_value_for_arm "$arm" BURN_DRAGON_PROMOTION_ROLLOUT_UNLIKELIHOOD_CYCLE_MAX_LAG 64 64)"
  arm_rollout_every_steps="$(rollout_value_for_arm "$arm" BURN_DRAGON_PROMOTION_ROLLOUT_UNLIKELIHOOD_EVERY_STEPS 32 64)"
  arm_rollout_prompt_tokens="$(rollout_value_for_arm "$arm" BURN_DRAGON_PROMOTION_ROLLOUT_UNLIKELIHOOD_PROMPT_TOKENS 32 32)"
  arm_rollout_rollout_tokens="$(rollout_value_for_arm "$arm" BURN_DRAGON_PROMOTION_ROLLOUT_UNLIKELIHOOD_ROLLOUT_TOKENS 16 8)"
  arm_rollout_history_tokens="$(rollout_value_for_arm "$arm" BURN_DRAGON_PROMOTION_ROLLOUT_UNLIKELIHOOD_HISTORY_TOKENS 16 8)"
  arm_rollout_batch_prompts="$(rollout_value_for_arm "$arm" BURN_DRAGON_PROMOTION_ROLLOUT_UNLIKELIHOOD_BATCH_PROMPTS 1 1)"
  arm_rollout_warmup_steps="$(rollout_value_for_arm "$arm" BURN_DRAGON_PROMOTION_ROLLOUT_UNLIKELIHOOD_WARMUP_STEPS 64 0)"
  arm_rollout_ramp_steps="$(rollout_value_for_arm "$arm" BURN_DRAGON_PROMOTION_ROLLOUT_UNLIKELIHOOD_RAMP_STEPS 128 0)"
  arm_rollout_recovery_only="$(rollout_value_for_arm "$arm" BURN_DRAGON_PROMOTION_ROLLOUT_UNLIKELIHOOD_RECOVERY_ONLY false false)"
  arm_structured_recovery_weight="$(structured_recovery_value_for_arm "$arm" "$STRUCTURED_RECOVERY_WEIGHT")"
  arm_structured_recovery_every_steps="$(structured_recovery_value_for_arm "$arm" "$STRUCTURED_RECOVERY_EVERY_STEPS")"
  arm_structured_recovery_start_after="$(structured_recovery_value_for_arm "$arm" "$STRUCTURED_RECOVERY_START_AFTER")"
  arm_structured_recovery_max_completion_tokens="$(structured_recovery_value_for_arm "$arm" "$STRUCTURED_RECOVERY_MAX_COMPLETION_TOKENS")"
  arm_structured_recovery_negative_count="$(structured_recovery_value_for_arm "$arm" "$STRUCTURED_RECOVERY_NEGATIVE_COUNT")"
  arm_structured_recovery_template_negative_count="$(structured_recovery_value_for_arm "$arm" "$STRUCTURED_RECOVERY_TEMPLATE_NEGATIVE_COUNT")"
  arm_structured_recovery_schema_negative_count="$(structured_recovery_value_for_arm "$arm" "$STRUCTURED_RECOVERY_SCHEMA_NEGATIVE_COUNT")"
  arm_generated_attractor_replay_capacity="$(generated_attractor_capacity_for_arm "$arm")"
  arm_generated_attractor_replay_min_distinct="$(generated_attractor_min_distinct_for_arm "$arm")"
  arm_generated_attractor_replay_max_dominant="$(generated_attractor_max_dominant_for_arm "$arm")"
  arm_ruliad_answer_close_marker_stride="$(ruliad_answer_close_stride_for_arm "$arm")"
  arm_ruliad_answer_close_marker_weight="$(ruliad_answer_close_weight_for_arm "$arm")"
  arm_ruliad_answer_schema_weight="$(ruliad_answer_schema_weight_for_arm "$arm")"
  arm_ruliad_answer_schema_start_weight="$(ruliad_answer_schema_start_weight_for_arm "$arm")"
  arm_ruliad_answer_value_weight="$(ruliad_answer_value_weight_for_arm "$arm")"
  write_arm_profile "$arm" "$arm_profile"

  args=(
    --base-profile "$arm_profile"
    --steps "$arm_max_steps"
    --eval-steps "$EVAL_STEPS_CSV"
    --seeds "$SEEDS_CSV"
    --max-iters "$MAX_ITERS"
    --epochs "$EPOCHS"
    --batch-size "$arm_batch_size"
    --block-size "$BLOCK_SIZE"
    --shape "$arm_n_layer,$arm_n_embd,$arm_n_head,$arm_latent_total"
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

  echo "==> arm=$arm profile=$arm_profile shape=${arm_n_layer},${arm_n_embd},${arm_n_head},${arm_latent_total} batch_size=$arm_batch_size max_steps=$arm_max_steps residual_gate=$arm_residual_gate residual_gate_init=$arm_residual_gate_init normalize_steps=$arm_normalize_steps answer_close_stride=$arm_ruliad_answer_close_marker_stride answer_close_weight=$arm_ruliad_answer_close_marker_weight answer_schema_weight=$arm_ruliad_answer_schema_weight answer_schema_start_weight=$arm_ruliad_answer_schema_start_weight answer_value_weight=$arm_ruliad_answer_value_weight"
  BURN_DRAGON_LR_STEPS_NEXTLAT_EVERY_STEPS="$NEXTLAT_EVERY_STEPS" \
    BURN_DRAGON_LR_STEPS_JEPA_EVERY_STEPS="$JEPA_EVERY_STEPS" \
    BURN_DRAGON_LR_STEPS_RESIDUAL_GATE="$arm_residual_gate" \
    BURN_DRAGON_LR_STEPS_RESIDUAL_GATE_INIT="$arm_residual_gate_init" \
    BURN_DRAGON_LR_STEPS_NORMALIZE_STEPS="$arm_normalize_steps" \
    BURN_DRAGON_LR_STEPS_LOG_FREQUENCY="$LOG_FREQUENCY" \
    BURN_DRAGON_LR_STEPS_CHECKPOINT_INTERVAL_ITERS="$CHECKPOINT_INTERVAL_ITERS" \
    BURN_DRAGON_LR_STEPS_EPOCHS="$EPOCHS" \
    BURN_DRAGON_LR_STEPS_MAX_SYSTEM_MEMORY_FRACTION="$MAX_SYSTEM_MEMORY_FRACTION" \
    BURN_DRAGON_LR_STEPS_MIN_AVAILABLE_MB="$MIN_AVAILABLE_MB" \
    BURN_DRAGON_LR_STEPS_DYNAMICS_ENABLED="$DYNAMICS_ENABLED" \
    BURN_DRAGON_LR_STEPS_RULIAD_MASK_HIGH_ENTROPY="$arm_mask_high_entropy" \
    BURN_DRAGON_LR_STEPS_RULIAD_ANSWER_CLOSE_MARKER_STRIDE="$arm_ruliad_answer_close_marker_stride" \
    BURN_DRAGON_LR_STEPS_RULIAD_ANSWER_CLOSE_MARKER_WEIGHT="$arm_ruliad_answer_close_marker_weight" \
    BURN_DRAGON_LR_STEPS_RULIAD_ANSWER_SCHEMA_WEIGHT="$arm_ruliad_answer_schema_weight" \
    BURN_DRAGON_LR_STEPS_RULIAD_ANSWER_SCHEMA_START_WEIGHT="$arm_ruliad_answer_schema_start_weight" \
    BURN_DRAGON_LR_STEPS_RULIAD_ANSWER_VALUE_WEIGHT="$arm_ruliad_answer_value_weight" \
    BURN_DRAGON_LR_STEPS_ANSWER_RANKING="$arm_answer_ranking" \
    BURN_DRAGON_LR_STEPS_ANSWER_DENOISING="$arm_answer_denoising" \
    BURN_DRAGON_LR_STEPS_STRUCTURED_RECOVERY_WEIGHT="$arm_structured_recovery_weight" \
    BURN_DRAGON_LR_STEPS_STRUCTURED_RECOVERY_EVERY_STEPS="$arm_structured_recovery_every_steps" \
    BURN_DRAGON_LR_STEPS_STRUCTURED_RECOVERY_START_AFTER="$arm_structured_recovery_start_after" \
    BURN_DRAGON_LR_STEPS_STRUCTURED_RECOVERY_MAX_COMPLETION_TOKENS="$arm_structured_recovery_max_completion_tokens" \
    BURN_DRAGON_LR_STEPS_STRUCTURED_RECOVERY_NEGATIVE_COUNT="$arm_structured_recovery_negative_count" \
    BURN_DRAGON_LR_STEPS_STRUCTURED_RECOVERY_TEMPLATE_NEGATIVE_COUNT="$arm_structured_recovery_template_negative_count" \
    BURN_DRAGON_LR_STEPS_STRUCTURED_RECOVERY_SCHEMA_NEGATIVE_COUNT="$arm_structured_recovery_schema_negative_count" \
    BURN_DRAGON_LR_STEPS_FIELD_BINDING_CONTRAST_WEIGHT="$FIELD_BINDING_CONTRAST_WEIGHT" \
    BURN_DRAGON_LR_STEPS_FIELD_BINDING_CONTRAST_EVERY_STEPS="$FIELD_BINDING_CONTRAST_EVERY_STEPS" \
    BURN_DRAGON_LR_STEPS_FIELD_BINDING_CONTRAST_MARGIN="$FIELD_BINDING_CONTRAST_MARGIN" \
    BURN_DRAGON_LR_STEPS_FIELD_BINDING_CONTRAST_PAIR_WEIGHT="$FIELD_BINDING_CONTRAST_PAIR_WEIGHT" \
    BURN_DRAGON_LR_STEPS_FIELD_BINDING_CONTRAST_MAX_PAIRS="$FIELD_BINDING_CONTRAST_MAX_PAIRS" \
    BURN_DRAGON_LR_STEPS_FIELD_BINDING_CONTRAST_REPLAY_CAPACITY="$FIELD_BINDING_CONTRAST_REPLAY_CAPACITY" \
    BURN_DRAGON_LR_STEPS_GENERATED_ATTRACTOR_REPLAY_CAPACITY="$arm_generated_attractor_replay_capacity" \
    BURN_DRAGON_LR_STEPS_GENERATED_ATTRACTOR_REPLAY_MIN_COUNT="$GENERATED_ATTRACTOR_REPLAY_MIN_COUNT" \
    BURN_DRAGON_LR_STEPS_GENERATED_ATTRACTOR_REPLAY_MAX_CANDIDATES="$GENERATED_ATTRACTOR_REPLAY_MAX_CANDIDATES" \
    BURN_DRAGON_LR_STEPS_GENERATED_ATTRACTOR_REPLAY_MIN_DISTINCT="$arm_generated_attractor_replay_min_distinct" \
    BURN_DRAGON_LR_STEPS_GENERATED_ATTRACTOR_REPLAY_MAX_DOMINANT="$arm_generated_attractor_replay_max_dominant" \
    BURN_DRAGON_LR_STEPS_VERIFIER_ROLLOUT_IMITATION_WEIGHT="$VERIFIER_ROLLOUT_IMITATION_WEIGHT" \
    BURN_DRAGON_LR_STEPS_VERIFIER_ROLLOUT_RECOVERY_WEIGHT="$VERIFIER_ROLLOUT_RECOVERY_WEIGHT" \
    BURN_DRAGON_LR_STEPS_VERIFIER_ROLLOUT_EVERY_STEPS="$VERIFIER_ROLLOUT_EVERY_STEPS" \
    BURN_DRAGON_LR_STEPS_VERIFIER_ROLLOUT_START_AFTER="$VERIFIER_ROLLOUT_START_AFTER" \
    BURN_DRAGON_LR_STEPS_VERIFIER_ROLLOUT_MIN_PARTIAL_PROGRESS_PPM="$VERIFIER_ROLLOUT_MIN_PARTIAL_PROGRESS_PPM" \
    BURN_DRAGON_LR_STEPS_VERIFIER_ROLLOUT_MIN_COMPLETION_QUALITY_PPM="$VERIFIER_ROLLOUT_MIN_COMPLETION_QUALITY_PPM" \
    BURN_DRAGON_LR_STEPS_VERIFIER_ROLLOUT_MAX_ROWS_PER_STEP="$VERIFIER_ROLLOUT_MAX_ROWS_PER_STEP" \
    BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD="$arm_rollout_unlikelihood" \
    BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_WEIGHT="$arm_rollout_weight" \
    BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_MARGIN_WEIGHT="$arm_rollout_margin_weight" \
    BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_MARGIN="$arm_rollout_margin" \
    BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_RECOVERY_WEIGHT="$arm_rollout_recovery_weight" \
    BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_SEQUENCE_RECOVERY_WEIGHT="$arm_rollout_sequence_recovery_weight" \
    BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_ENTROPY_WEIGHT="$arm_rollout_entropy_weight" \
    BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_TARGET_ENTROPY_BITS="$arm_rollout_target_entropy_bits" \
    BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_CYCLE_WEIGHT="$arm_rollout_cycle_weight" \
    BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_CYCLE_MARGIN_WEIGHT="$arm_rollout_cycle_margin_weight" \
    BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_CYCLE_MIN_LAG="$arm_rollout_cycle_min_lag" \
    BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_CYCLE_MAX_LAG="$arm_rollout_cycle_max_lag" \
    BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_EVERY_STEPS="$arm_rollout_every_steps" \
    BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_PROMPT_TOKENS="$arm_rollout_prompt_tokens" \
    BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_ROLLOUT_TOKENS="$arm_rollout_rollout_tokens" \
    BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_HISTORY_TOKENS="$arm_rollout_history_tokens" \
    BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_BATCH_PROMPTS="$arm_rollout_batch_prompts" \
    BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_WARMUP_STEPS="$arm_rollout_warmup_steps" \
    BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_RAMP_STEPS="$arm_rollout_ramp_steps" \
    BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_RECOVERY_ONLY="$arm_rollout_recovery_only" \
    "$ROOT_DIR/scripts/latent_reasoning_steps_ablation.sh" "${args[@]}"

  if (( DRY_RUN == 0 )); then
    python3 "$ROOT_DIR/scripts/latent_reasoning_steps_analyze.py" "$arm_out" --out-dir "$arm_out/analysis"
  fi
  first_arm=0
done

if (( DRY_RUN == 0 )); then
  python3 "$ROOT_DIR/scripts/ruliad_promotion_matrix_analyze.py" "$OUT_DIR" \
    --baseline-arm "$BASELINE_ARM" \
    --min-mature-iters "$MIN_MATURE_ITERS"
fi

echo "ruliad promotion matrix complete: $OUT_DIR"
