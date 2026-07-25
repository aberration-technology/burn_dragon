#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BACKEND="${BURN_DRAGON_LR_STEPS_BACKEND:-cuda}"
FEATURES="${BURN_DRAGON_LR_STEPS_FEATURES:-train,cuda}"
BASE_PROFILE="${BURN_DRAGON_LR_STEPS_BASE_PROFILE:-crates/burn_dragon_p2p/deploy/profiles/ruliad-r1.jepa-nextlat-decoupled-delayed1024-sparse16-probe128-fixed-ablation.toml}"
OUT_DIR="${BURN_DRAGON_LR_STEPS_OUT_DIR:-$ROOT_DIR/target/latent-reasoning-steps/$(date -u +%Y%m%dT%H%M%SZ)}"
STEPS_CSV="${BURN_DRAGON_LR_STEPS_MAX_STEPS:-1,2,4,8}"
EVAL_STEPS_CSV="${BURN_DRAGON_LR_STEPS_EVAL_STEPS:-1,2,4,8,16}"
SEEDS_CSV="${BURN_DRAGON_LR_STEPS_SEEDS:-20260624}"
MAX_ITERS="${BURN_DRAGON_LR_STEPS_MAX_ITERS:-512}"
EPOCHS="${BURN_DRAGON_LR_STEPS_EPOCHS:-inherit}"
BATCH_SIZE="${BURN_DRAGON_LR_STEPS_BATCH_SIZE:-4}"
AUTO_BATCH_SIZE="${BURN_DRAGON_LR_STEPS_AUTO_BATCH_SIZE:-false}"
AUTO_BATCH_MIN_BATCH_SIZE="${BURN_DRAGON_LR_STEPS_AUTO_BATCH_MIN_BATCH_SIZE:-1}"
AUTO_BATCH_MAX_BATCH_SIZE="${BURN_DRAGON_LR_STEPS_AUTO_BATCH_MAX_BATCH_SIZE:-$BATCH_SIZE}"
AUTO_BATCH_MAX_PROBE_BATCH_SIZE="${BURN_DRAGON_LR_STEPS_AUTO_BATCH_MAX_PROBE_BATCH_SIZE:-$BATCH_SIZE}"
AUTO_BATCH_TARGET_DEVICE_MEMORY_MB="${BURN_DRAGON_LR_STEPS_AUTO_BATCH_TARGET_DEVICE_MEMORY_MB:-0}"
AUTO_BATCH_PROBE_STEPS="${BURN_DRAGON_LR_STEPS_AUTO_BATCH_PROBE_STEPS:-1}"
AUTO_BATCH_MAX_SYSTEM_MEMORY_FRACTION="${BURN_DRAGON_LR_STEPS_AUTO_BATCH_MAX_SYSTEM_MEMORY_FRACTION:-0.90}"
AUTO_BATCH_PROBE_SAFETY_MARGIN="${BURN_DRAGON_LR_STEPS_AUTO_BATCH_PROBE_SAFETY_MARGIN:-1.15}"
BLOCK_SIZE="${BURN_DRAGON_LR_STEPS_BLOCK_SIZE:-256}"
N_LAYER="${BURN_DRAGON_LR_STEPS_N_LAYER:-4}"
N_EMBD="${BURN_DRAGON_LR_STEPS_N_EMBD:-256}"
N_HEAD="${BURN_DRAGON_LR_STEPS_N_HEAD:-4}"
LATENT_TOTAL="${BURN_DRAGON_LR_STEPS_LATENT_TOTAL:-12288}"
ENERGY_HEAD="${BURN_DRAGON_LR_STEPS_ENERGY_HEAD:-false}"
RESIDUAL_GATE="${BURN_DRAGON_LR_STEPS_RESIDUAL_GATE:-false}"
RESIDUAL_GATE_INIT="${BURN_DRAGON_LR_STEPS_RESIDUAL_GATE_INIT:-0.25}"
NORMALIZE_STEPS="${BURN_DRAGON_LR_STEPS_NORMALIZE_STEPS:-false}"
STEP_CONDITIONED_DECODER="${BURN_DRAGON_LR_STEPS_STEP_CONDITIONED_DECODER:-false}"
STEP_CONDITIONED_DECODER_SCALE="${BURN_DRAGON_LR_STEPS_STEP_CONDITIONED_DECODER_SCALE:-1.0}"
ENERGY_MODEL="${BURN_DRAGON_LR_STEPS_ENERGY_MODEL:-inherit}"
ENERGY_START_AFTER="${BURN_DRAGON_LR_STEPS_ENERGY_START_AFTER:-}"
ENERGY_EVERY_STEPS="${BURN_DRAGON_LR_STEPS_ENERGY_EVERY_STEPS:-}"
STEP_CONTRACT="${BURN_DRAGON_LR_STEPS_STEP_CONTRACT:-inherit}"
STEP_CONTRACT_START_AFTER="${BURN_DRAGON_LR_STEPS_STEP_CONTRACT_START_AFTER:-}"
STEP_CONTRACT_EVERY_STEPS="${BURN_DRAGON_LR_STEPS_STEP_CONTRACT_EVERY_STEPS:-}"
STEP_CONTRACT_CE_WEIGHT="${BURN_DRAGON_LR_STEPS_STEP_CONTRACT_CE_WEIGHT:-}"
STEP_CONTRACT_TOKEN_KL_WEIGHT="${BURN_DRAGON_LR_STEPS_STEP_CONTRACT_TOKEN_KL_WEIGHT:-}"
STEP_CONTRACT_MONOTONIC_CE_WEIGHT="${BURN_DRAGON_LR_STEPS_STEP_CONTRACT_MONOTONIC_CE_WEIGHT:-}"
STEP_CONTRACT_CONTRACTIVE_WEIGHT="${BURN_DRAGON_LR_STEPS_STEP_CONTRACT_CONTRACTIVE_WEIGHT:-}"
STEP_CONTRACT_CE_TOLERANCE="${BURN_DRAGON_LR_STEPS_STEP_CONTRACT_CE_TOLERANCE:-}"
STEP_CONTRACT_TRUST_RADIUS="${BURN_DRAGON_LR_STEPS_STEP_CONTRACT_TRUST_RADIUS:-}"
RULIAD_SUPERVISION_MODE="${BURN_DRAGON_LR_STEPS_RULIAD_MODE:-inherit}"
RULIAD_MASK_HIGH_ENTROPY_SPANS="${BURN_DRAGON_LR_STEPS_RULIAD_MASK_HIGH_ENTROPY:-false}"
RULIAD_ANSWER_CLOSE_MARKER_STRIDE="${BURN_DRAGON_LR_STEPS_RULIAD_ANSWER_CLOSE_MARKER_STRIDE:-1}"
RULIAD_ANSWER_CLOSE_MARKER_WEIGHT="${BURN_DRAGON_LR_STEPS_RULIAD_ANSWER_CLOSE_MARKER_WEIGHT:-1}"
RULIAD_ANSWER_SCHEMA_WEIGHT="${BURN_DRAGON_LR_STEPS_RULIAD_ANSWER_SCHEMA_WEIGHT:-1}"
RULIAD_ANSWER_SCHEMA_START_WEIGHT="${BURN_DRAGON_LR_STEPS_RULIAD_ANSWER_SCHEMA_START_WEIGHT:-1}"
RULIAD_ANSWER_VALUE_WEIGHT="${BURN_DRAGON_LR_STEPS_RULIAD_ANSWER_VALUE_WEIGHT:-1}"
ANSWER_RANKING="${BURN_DRAGON_LR_STEPS_ANSWER_RANKING:-inherit}"
ANSWER_RANKING_WEIGHT="${BURN_DRAGON_LR_STEPS_ANSWER_RANKING_WEIGHT:-0.25}"
ANSWER_RANKING_MARGIN="${BURN_DRAGON_LR_STEPS_ANSWER_RANKING_MARGIN:-0.5}"
ANSWER_RANKING_CORRUPT_OFFSET="${BURN_DRAGON_LR_STEPS_ANSWER_RANKING_CORRUPT_OFFSET:-1}"
ANSWER_DENOISING="${BURN_DRAGON_LR_STEPS_ANSWER_DENOISING:-inherit}"
ANSWER_DENOISING_WEIGHT="${BURN_DRAGON_LR_STEPS_ANSWER_DENOISING_WEIGHT:-0.5}"
ANSWER_DENOISING_PROBABILITY="${BURN_DRAGON_LR_STEPS_ANSWER_DENOISING_PROBABILITY:-1.0}"
ANSWER_DENOISING_CORRUPT_OFFSET="${BURN_DRAGON_LR_STEPS_ANSWER_DENOISING_CORRUPT_OFFSET:-1}"
STRUCTURED_RECOVERY_WEIGHT="${BURN_DRAGON_LR_STEPS_STRUCTURED_RECOVERY_WEIGHT:-inherit}"
STRUCTURED_RECOVERY_EVERY_STEPS="${BURN_DRAGON_LR_STEPS_STRUCTURED_RECOVERY_EVERY_STEPS:-inherit}"
STRUCTURED_RECOVERY_START_AFTER="${BURN_DRAGON_LR_STEPS_STRUCTURED_RECOVERY_START_AFTER:-inherit}"
STRUCTURED_RECOVERY_MAX_COMPLETION_TOKENS="${BURN_DRAGON_LR_STEPS_STRUCTURED_RECOVERY_MAX_COMPLETION_TOKENS:-inherit}"
STRUCTURED_RECOVERY_NEGATIVE_COUNT="${BURN_DRAGON_LR_STEPS_STRUCTURED_RECOVERY_NEGATIVE_COUNT:-inherit}"
STRUCTURED_RECOVERY_TEMPLATE_NEGATIVE_COUNT="${BURN_DRAGON_LR_STEPS_STRUCTURED_RECOVERY_TEMPLATE_NEGATIVE_COUNT:-inherit}"
STRUCTURED_RECOVERY_SCHEMA_NEGATIVE_COUNT="${BURN_DRAGON_LR_STEPS_STRUCTURED_RECOVERY_SCHEMA_NEGATIVE_COUNT:-inherit}"
FIELD_BINDING_CONTRAST_WEIGHT="${BURN_DRAGON_LR_STEPS_FIELD_BINDING_CONTRAST_WEIGHT:-inherit}"
FIELD_BINDING_CONTRAST_EVERY_STEPS="${BURN_DRAGON_LR_STEPS_FIELD_BINDING_CONTRAST_EVERY_STEPS:-inherit}"
FIELD_BINDING_CONTRAST_MARGIN="${BURN_DRAGON_LR_STEPS_FIELD_BINDING_CONTRAST_MARGIN:-inherit}"
FIELD_BINDING_CONTRAST_PAIR_WEIGHT="${BURN_DRAGON_LR_STEPS_FIELD_BINDING_CONTRAST_PAIR_WEIGHT:-inherit}"
FIELD_BINDING_CONTRAST_MAX_PAIRS="${BURN_DRAGON_LR_STEPS_FIELD_BINDING_CONTRAST_MAX_PAIRS:-inherit}"
FIELD_BINDING_CONTRAST_REPLAY_CAPACITY="${BURN_DRAGON_LR_STEPS_FIELD_BINDING_CONTRAST_REPLAY_CAPACITY:-inherit}"
GENERATED_ATTRACTOR_REPLAY_CAPACITY="${BURN_DRAGON_LR_STEPS_GENERATED_ATTRACTOR_REPLAY_CAPACITY:-inherit}"
GENERATED_ATTRACTOR_REPLAY_MIN_COUNT="${BURN_DRAGON_LR_STEPS_GENERATED_ATTRACTOR_REPLAY_MIN_COUNT:-inherit}"
GENERATED_ATTRACTOR_REPLAY_MAX_CANDIDATES="${BURN_DRAGON_LR_STEPS_GENERATED_ATTRACTOR_REPLAY_MAX_CANDIDATES:-inherit}"
GENERATED_ATTRACTOR_REPLAY_MIN_DISTINCT="${BURN_DRAGON_LR_STEPS_GENERATED_ATTRACTOR_REPLAY_MIN_DISTINCT:-inherit}"
GENERATED_ATTRACTOR_REPLAY_MAX_DOMINANT="${BURN_DRAGON_LR_STEPS_GENERATED_ATTRACTOR_REPLAY_MAX_DOMINANT:-inherit}"
VERIFIER_ROLLOUT_IMITATION_WEIGHT="${BURN_DRAGON_LR_STEPS_VERIFIER_ROLLOUT_IMITATION_WEIGHT:-inherit}"
VERIFIER_ROLLOUT_RECOVERY_WEIGHT="${BURN_DRAGON_LR_STEPS_VERIFIER_ROLLOUT_RECOVERY_WEIGHT:-inherit}"
VERIFIER_ROLLOUT_EVERY_STEPS="${BURN_DRAGON_LR_STEPS_VERIFIER_ROLLOUT_EVERY_STEPS:-inherit}"
VERIFIER_ROLLOUT_START_AFTER="${BURN_DRAGON_LR_STEPS_VERIFIER_ROLLOUT_START_AFTER:-inherit}"
VERIFIER_ROLLOUT_MIN_PARTIAL_PROGRESS_PPM="${BURN_DRAGON_LR_STEPS_VERIFIER_ROLLOUT_MIN_PARTIAL_PROGRESS_PPM:-inherit}"
VERIFIER_ROLLOUT_MIN_COMPLETION_QUALITY_PPM="${BURN_DRAGON_LR_STEPS_VERIFIER_ROLLOUT_MIN_COMPLETION_QUALITY_PPM:-inherit}"
VERIFIER_ROLLOUT_MAX_ROWS_PER_STEP="${BURN_DRAGON_LR_STEPS_VERIFIER_ROLLOUT_MAX_ROWS_PER_STEP:-inherit}"
ROLLOUT_UNLIKELIHOOD="${BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD:-false}"
ROLLOUT_UNLIKELIHOOD_WEIGHT="${BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_WEIGHT:-0.0}"
ROLLOUT_UNLIKELIHOOD_MARGIN_WEIGHT="${BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_MARGIN_WEIGHT:-0.0}"
ROLLOUT_UNLIKELIHOOD_MARGIN="${BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_MARGIN:-0.0}"
ROLLOUT_UNLIKELIHOOD_RECOVERY_WEIGHT="${BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_RECOVERY_WEIGHT:-0.0}"
ROLLOUT_UNLIKELIHOOD_SEQUENCE_RECOVERY_WEIGHT="${BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_SEQUENCE_RECOVERY_WEIGHT:-0.0}"
ROLLOUT_UNLIKELIHOOD_ENTROPY_WEIGHT="${BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_ENTROPY_WEIGHT:-0.0}"
ROLLOUT_UNLIKELIHOOD_TARGET_ENTROPY_BITS="${BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_TARGET_ENTROPY_BITS:-0.0}"
ROLLOUT_UNLIKELIHOOD_CYCLE_WEIGHT="${BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_CYCLE_WEIGHT:-0.0}"
ROLLOUT_UNLIKELIHOOD_CYCLE_MARGIN_WEIGHT="${BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_CYCLE_MARGIN_WEIGHT:-0.0}"
ROLLOUT_UNLIKELIHOOD_CYCLE_MIN_LAG="${BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_CYCLE_MIN_LAG:-2}"
ROLLOUT_UNLIKELIHOOD_CYCLE_MAX_LAG="${BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_CYCLE_MAX_LAG:-64}"
ROLLOUT_UNLIKELIHOOD_EVERY_STEPS="${BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_EVERY_STEPS:-64}"
ROLLOUT_UNLIKELIHOOD_PROMPT_TOKENS="${BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_PROMPT_TOKENS:-32}"
ROLLOUT_UNLIKELIHOOD_ROLLOUT_TOKENS="${BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_ROLLOUT_TOKENS:-8}"
ROLLOUT_UNLIKELIHOOD_HISTORY_TOKENS="${BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_HISTORY_TOKENS:-8}"
ROLLOUT_UNLIKELIHOOD_BATCH_PROMPTS="${BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_BATCH_PROMPTS:-1}"
ROLLOUT_UNLIKELIHOOD_WARMUP_STEPS="${BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_WARMUP_STEPS:-0}"
ROLLOUT_UNLIKELIHOOD_RAMP_STEPS="${BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_RAMP_STEPS:-0}"
ROLLOUT_UNLIKELIHOOD_RECOVERY_ONLY="${BURN_DRAGON_LR_STEPS_ROLLOUT_UNLIKELIHOOD_RECOVERY_ONLY:-false}"
NEXTLAT_START_AFTER="${BURN_DRAGON_LR_STEPS_NEXTLAT_START_AFTER:-128}"
NEXTLAT_EVERY_STEPS="${BURN_DRAGON_LR_STEPS_NEXTLAT_EVERY_STEPS:-16}"
JEPA_EVERY_STEPS="${BURN_DRAGON_LR_STEPS_JEPA_EVERY_STEPS:-8}"
LOG_FREQUENCY="${BURN_DRAGON_LR_STEPS_LOG_FREQUENCY:-16}"
CHECKPOINT_INTERVAL_ITERS="${BURN_DRAGON_LR_STEPS_CHECKPOINT_INTERVAL_ITERS:-128}"
RULIAD_PROBE_ITEMS="${BURN_DRAGON_LR_STEPS_RULIAD_PROBE_ITEMS:-128}"
RULIAD_PROBE_TOKENS="${BURN_DRAGON_LR_STEPS_RULIAD_PROBE_TOKENS:-64}"
TIMEOUT_SECONDS="${BURN_DRAGON_LR_STEPS_TIMEOUT_SECONDS:-2400}"
MAX_SYSTEM_MEMORY_FRACTION="${BURN_DRAGON_LR_STEPS_MAX_SYSTEM_MEMORY_FRACTION:-0.80}"
MIN_AVAILABLE_MB="${BURN_DRAGON_LR_STEPS_MIN_AVAILABLE_MB:-24576}"
SAMPLE_INTERVAL_SECONDS="${BURN_DRAGON_LR_STEPS_SAMPLE_INTERVAL_SECONDS:-2}"
GPU_TELEMETRY_SECONDS="${BURN_DRAGON_LR_STEPS_GPU_TELEMETRY_SECONDS:-1}"
DYNAMICS_ENABLED="${BURN_DRAGON_LR_STEPS_DYNAMICS_ENABLED:-inherit}"
BUILD_RELEASE="${BURN_DRAGON_LR_STEPS_BUILD_RELEASE:-1}"
ALLOW_STALE_BINARY="${BURN_DRAGON_LR_STEPS_ALLOW_STALE_BINARY:-0}"
STALE_BINARY_RISK=false
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage:
  scripts/latent_reasoning_steps_ablation.sh [options]

Options:
  --base-profile <path>       Base JEPA/NextLat profile.
  --steps <csv>               Fixed latent max_steps values. Default: 1,2,4,8.
  --eval-steps <csv>          Validation-only eval step sweep. Default: 1,2,4,8,16.
  --seeds <csv>               Training seeds. Default: 20260624.
  --max-iters <n>             Iterations per trial. Default: 512.
  --epochs <n|inherit>        Split max_iters into this many bounded logical epochs by deriving checkpoint_interval_iters. Default: inherit.
  --batch-size <n>            Batch size override. Default: 4.
  Auto-batch can be enabled with BURN_DRAGON_LR_STEPS_AUTO_BATCH_SIZE=true and bounded with
                            BURN_DRAGON_LR_STEPS_AUTO_BATCH_{MIN,MAX,MAX_PROBE}_BATCH_SIZE.
  --block-size <n>            Block size override. Default: 256.
  --shape L,E,H,Z             Model shape n_layer,n_embd,n_head,latent_total.
  --energy-head <true|false>  Override model.latent_reasoning.energy_head. Default: false.
  --residual-gate <true|false> Enable residual-gated latent refinement. Default: false.
  --residual-gate-init <x>    Initial residual gate multiplier in (0, 1). Default: 0.25.
  --normalize-steps <true|false>
                            Normalize latent refinement steps. Default: false.
  --step-decoder <true|false> Enable step-conditioned latent decoder. Default: false.
  --step-decoder-scale <x>    Scale for step-conditioned decoder embedding. Default: 1.0.
  --energy-model <mode>       Override energy aux enabled state: inherit,true,false. Default: inherit.
  --energy-start <n>          Override energy_model.start_after_steps.
  --energy-every <n>          Override energy_model.every_steps.
  --step-contract <mode>      Override step contract enabled state: inherit,true,false. Default: inherit.
  --step-contract-start <n>   Override step_contract.start_after_steps.
  --step-contract-every <n>   Override step_contract.every_steps.
  --step-contract-ce <x>      Override step_contract.ce_weight.
  --step-contract-token-kl <x> Override step_contract.token_kl_weight.
  --step-contract-mono <x>    Override step_contract.monotonic_ce_weight.
  --step-contract-contract <x> Override step_contract.contractive_weight.
  --step-contract-tolerance <x> Override step_contract.ce_tolerance.
  --step-contract-trust <x>   Override step_contract.trust_radius.
  --ruliad-mode <mode>        Override training.ruliad_supervision.mode, or inherit. Default: inherit.
  --ruliad-mask-high-entropy <bool> Mask high-entropy ruliad targets such as hashes. Default: false.
  --ruliad-answer-close-stride <n> Supervise one answer close marker per n deterministic answer spans; 0 disables close targets. Default: 1.
  --ruliad-answer-close-weight <n> Integer loss-mask weight for answer close marker tokens. Default: 1.
  --ruliad-answer-schema-weight <n> Integer loss-mask weight for deterministic answer schema/key tokens. Default: 1.
  --ruliad-answer-value-weight <n> Integer loss-mask weight for deterministic answer values. Default: 1.
  --answer-ranking <bool>     Enable answer-token oracle-vs-corrupt ranking. Default: false.
  --answer-ranking-weight <x> Ranking loss weight. Default: 0.25.
  --answer-ranking-margin <x> Ranking margin. Default: 0.5.
  --answer-ranking-offset <n> Positive corrupt token offset. Default: 1.
  --answer-denoising <bool>  Enable answer-prefix denoising auxiliary. Default: false.
  --answer-denoising-weight <x> Denoising CE loss weight. Default: 0.5.
  --answer-denoising-prob <x> Prefix corruption probability in [0,1]. Default: 1.0.
  --answer-denoising-offset <n> Positive corrupt token offset. Default: 1.
  --structured-recovery-weight <x|inherit>
                            Override answer_denoising.structured_recovery_weight.
  --structured-recovery-every <n|inherit>
                            Override answer_denoising.structured_recovery_every_steps.
  --structured-recovery-start <n|inherit>
                            Override answer_denoising.structured_recovery_start_after_steps.
  --structured-recovery-tokens <n|inherit>
                            Override answer_denoising.structured_recovery_max_completion_tokens.
  --structured-recovery-negatives <n|inherit>
                            Override answer_denoising.structured_recovery_negative_count.
  --structured-recovery-template-negatives <n|inherit>
                            Override answer_denoising.structured_recovery_template_negative_count.
  --structured-recovery-schema-negatives <n|inherit>
                            Override answer_denoising.structured_recovery_schema_negative_count.
  --generated-attractor-capacity <n|inherit>
                            Override verifier_reward.generated_attractor_replay_capacity.
  --generated-attractor-min-count <n|inherit>
                            Override verifier_reward.generated_attractor_replay_min_count.
  --generated-attractor-max-candidates <n|inherit>
                            Override verifier_reward.generated_attractor_replay_max_candidates.
  --generated-attractor-min-distinct <n|inherit>
                            Override verifier_reward.generated_attractor_replay_min_distinct_answers.
  --generated-attractor-max-dominant <x|inherit>
                            Override verifier_reward.generated_attractor_replay_max_dominant_fraction.
  --rollout-unlikelihood <bool> Enable greedy free-run unlikelihood auxiliary. Default: false.
  --rollout-weight <x>      Penalize generated tokens that repeat recent history. Default: 0.0.
  --rollout-margin-weight <x> Add margin penalty for repeated generated tokens. Default: 0.0.
  --rollout-margin <x>      Repeat margin. Default: 0.0.
  --rollout-recovery-weight <x> Penalize degenerate repeated rollout steps. Default: 0.0.
  --rollout-sequence-recovery-weight <x> Penalize whole degenerate rollout sequences. Default: 0.0.
  --rollout-entropy-weight <x> Entropy floor weight during rollout. Default: 0.0.
  --rollout-target-entropy-bits <x> Rollout entropy target. Default: 0.0.
  --rollout-cycle-weight <x> Penalize generated periodic cycles. Default: 0.0.
  --rollout-cycle-margin-weight <x> Add margin penalty for periodic cycles. Default: 0.0.
  --rollout-cycle-lags <min,max> Periodic cycle lag range. Default: 2,64.
  --rollout-every <n>       Rollout auxiliary cadence. Default: 64.
  --rollout-prompt-tokens <n> Prompt tokens per rollout. Default: 32.
  --rollout-tokens <n>      Generated tokens per rollout auxiliary. Default: 8.
  --rollout-history-tokens <n> Recent-token history length. Default: 8.
  --rollout-batch-prompts <n> Prompts per batch used for rollout. Default: 1.
  --rollout-warmup <n>      Rollout auxiliary warmup steps. Default: 0.
  --rollout-ramp <n>        Rollout auxiliary ramp steps. Default: 0.
  --rollout-recovery-only <bool> Run rollout auxiliary only in recovery mode. Default: false.
  --nextlat-start <n>         Override NextLat start_after_steps. Default: 128.
  --out-dir <path>            Output directory.
  --timeout-seconds <n>       Per-trial timeout. Default: 2400.
  --probe-items <n>           Ruliad correctness probe items. Default: 128.
  --probe-tokens <n>          Ruliad correctness generation tokens. Default: 64.
  --dynamics <inherit|true|false>
                            Override training.dynamics.enabled. Default: inherit.
  --backend <cuda|cpu>        Backend. Default: cuda.
  --features <features>       Cargo features. Default: train,cuda.
  --dry-run                   Write overlays/manifests only.
  --no-build                  Skip release build.
  --allow-stale-binary        Permit --no-build when sources are newer than the release binary.

Safety:
  BURN_DRAGON_LR_STEPS_MAX_SYSTEM_MEMORY_FRACTION  Default: 0.80
  BURN_DRAGON_LR_STEPS_MIN_AVAILABLE_MB            Default: 24576

This sweep is fixed-step only. Adaptive halting currently lacks a trained halt
objective in the JEPA/NextLat path, so it is intentionally excluded.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-profile) BASE_PROFILE="$2"; shift 2 ;;
    --steps) STEPS_CSV="$2"; shift 2 ;;
    --eval-steps) EVAL_STEPS_CSV="$2"; shift 2 ;;
    --seeds) SEEDS_CSV="$2"; shift 2 ;;
    --max-iters) MAX_ITERS="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --block-size) BLOCK_SIZE="$2"; shift 2 ;;
    --shape)
      IFS=',' read -r N_LAYER N_EMBD N_HEAD LATENT_TOTAL <<< "$2"
      shift 2
      ;;
    --energy-head) ENERGY_HEAD="$2"; shift 2 ;;
    --residual-gate) RESIDUAL_GATE="$2"; shift 2 ;;
    --residual-gate-init) RESIDUAL_GATE_INIT="$2"; shift 2 ;;
    --normalize-steps) NORMALIZE_STEPS="$2"; shift 2 ;;
    --step-decoder) STEP_CONDITIONED_DECODER="$2"; shift 2 ;;
    --step-decoder-scale) STEP_CONDITIONED_DECODER_SCALE="$2"; shift 2 ;;
    --energy-model) ENERGY_MODEL="$2"; shift 2 ;;
    --energy-start) ENERGY_START_AFTER="$2"; shift 2 ;;
    --energy-every) ENERGY_EVERY_STEPS="$2"; shift 2 ;;
    --step-contract) STEP_CONTRACT="$2"; shift 2 ;;
    --step-contract-start) STEP_CONTRACT_START_AFTER="$2"; shift 2 ;;
    --step-contract-every) STEP_CONTRACT_EVERY_STEPS="$2"; shift 2 ;;
    --step-contract-ce) STEP_CONTRACT_CE_WEIGHT="$2"; shift 2 ;;
    --step-contract-token-kl) STEP_CONTRACT_TOKEN_KL_WEIGHT="$2"; shift 2 ;;
    --step-contract-mono) STEP_CONTRACT_MONOTONIC_CE_WEIGHT="$2"; shift 2 ;;
    --step-contract-contract) STEP_CONTRACT_CONTRACTIVE_WEIGHT="$2"; shift 2 ;;
    --step-contract-tolerance) STEP_CONTRACT_CE_TOLERANCE="$2"; shift 2 ;;
    --step-contract-trust) STEP_CONTRACT_TRUST_RADIUS="$2"; shift 2 ;;
    --ruliad-mode) RULIAD_SUPERVISION_MODE="$2"; shift 2 ;;
    --ruliad-mask-high-entropy) RULIAD_MASK_HIGH_ENTROPY_SPANS="$2"; shift 2 ;;
    --ruliad-answer-close-stride) RULIAD_ANSWER_CLOSE_MARKER_STRIDE="$2"; shift 2 ;;
    --ruliad-answer-close-weight) RULIAD_ANSWER_CLOSE_MARKER_WEIGHT="$2"; shift 2 ;;
    --ruliad-answer-schema-weight) RULIAD_ANSWER_SCHEMA_WEIGHT="$2"; shift 2 ;;
    --ruliad-answer-schema-start-weight) RULIAD_ANSWER_SCHEMA_START_WEIGHT="$2"; shift 2 ;;
    --ruliad-answer-value-weight) RULIAD_ANSWER_VALUE_WEIGHT="$2"; shift 2 ;;
    --answer-ranking) ANSWER_RANKING="$2"; shift 2 ;;
    --answer-ranking-weight) ANSWER_RANKING_WEIGHT="$2"; shift 2 ;;
    --answer-ranking-margin) ANSWER_RANKING_MARGIN="$2"; shift 2 ;;
    --answer-ranking-offset) ANSWER_RANKING_CORRUPT_OFFSET="$2"; shift 2 ;;
    --answer-denoising) ANSWER_DENOISING="$2"; shift 2 ;;
    --answer-denoising-weight) ANSWER_DENOISING_WEIGHT="$2"; shift 2 ;;
    --answer-denoising-prob) ANSWER_DENOISING_PROBABILITY="$2"; shift 2 ;;
    --answer-denoising-offset) ANSWER_DENOISING_CORRUPT_OFFSET="$2"; shift 2 ;;
    --structured-recovery-weight) STRUCTURED_RECOVERY_WEIGHT="$2"; shift 2 ;;
    --structured-recovery-every) STRUCTURED_RECOVERY_EVERY_STEPS="$2"; shift 2 ;;
    --structured-recovery-start) STRUCTURED_RECOVERY_START_AFTER="$2"; shift 2 ;;
    --structured-recovery-tokens) STRUCTURED_RECOVERY_MAX_COMPLETION_TOKENS="$2"; shift 2 ;;
    --structured-recovery-negatives) STRUCTURED_RECOVERY_NEGATIVE_COUNT="$2"; shift 2 ;;
    --structured-recovery-template-negatives) STRUCTURED_RECOVERY_TEMPLATE_NEGATIVE_COUNT="$2"; shift 2 ;;
    --structured-recovery-schema-negatives) STRUCTURED_RECOVERY_SCHEMA_NEGATIVE_COUNT="$2"; shift 2 ;;
    --generated-attractor-capacity) GENERATED_ATTRACTOR_REPLAY_CAPACITY="$2"; shift 2 ;;
    --generated-attractor-min-count) GENERATED_ATTRACTOR_REPLAY_MIN_COUNT="$2"; shift 2 ;;
    --generated-attractor-max-candidates) GENERATED_ATTRACTOR_REPLAY_MAX_CANDIDATES="$2"; shift 2 ;;
    --generated-attractor-min-distinct) GENERATED_ATTRACTOR_REPLAY_MIN_DISTINCT="$2"; shift 2 ;;
    --generated-attractor-max-dominant) GENERATED_ATTRACTOR_REPLAY_MAX_DOMINANT="$2"; shift 2 ;;
    --verifier-rollout-imitation-weight) VERIFIER_ROLLOUT_IMITATION_WEIGHT="$2"; shift 2 ;;
    --verifier-rollout-recovery-weight) VERIFIER_ROLLOUT_RECOVERY_WEIGHT="$2"; shift 2 ;;
    --verifier-rollout-every) VERIFIER_ROLLOUT_EVERY_STEPS="$2"; shift 2 ;;
    --verifier-rollout-start) VERIFIER_ROLLOUT_START_AFTER="$2"; shift 2 ;;
    --verifier-rollout-min-partial) VERIFIER_ROLLOUT_MIN_PARTIAL_PROGRESS_PPM="$2"; shift 2 ;;
    --verifier-rollout-min-quality) VERIFIER_ROLLOUT_MIN_COMPLETION_QUALITY_PPM="$2"; shift 2 ;;
    --verifier-rollout-max-rows) VERIFIER_ROLLOUT_MAX_ROWS_PER_STEP="$2"; shift 2 ;;
    --rollout-unlikelihood) ROLLOUT_UNLIKELIHOOD="$2"; shift 2 ;;
    --rollout-weight) ROLLOUT_UNLIKELIHOOD_WEIGHT="$2"; shift 2 ;;
    --rollout-margin-weight) ROLLOUT_UNLIKELIHOOD_MARGIN_WEIGHT="$2"; shift 2 ;;
    --rollout-margin) ROLLOUT_UNLIKELIHOOD_MARGIN="$2"; shift 2 ;;
    --rollout-recovery-weight) ROLLOUT_UNLIKELIHOOD_RECOVERY_WEIGHT="$2"; shift 2 ;;
    --rollout-sequence-recovery-weight) ROLLOUT_UNLIKELIHOOD_SEQUENCE_RECOVERY_WEIGHT="$2"; shift 2 ;;
    --rollout-entropy-weight) ROLLOUT_UNLIKELIHOOD_ENTROPY_WEIGHT="$2"; shift 2 ;;
    --rollout-target-entropy-bits) ROLLOUT_UNLIKELIHOOD_TARGET_ENTROPY_BITS="$2"; shift 2 ;;
    --rollout-cycle-weight) ROLLOUT_UNLIKELIHOOD_CYCLE_WEIGHT="$2"; shift 2 ;;
    --rollout-cycle-margin-weight) ROLLOUT_UNLIKELIHOOD_CYCLE_MARGIN_WEIGHT="$2"; shift 2 ;;
    --rollout-cycle-lags)
      IFS=',' read -r ROLLOUT_UNLIKELIHOOD_CYCLE_MIN_LAG ROLLOUT_UNLIKELIHOOD_CYCLE_MAX_LAG <<< "$2"
      shift 2
      ;;
    --rollout-every) ROLLOUT_UNLIKELIHOOD_EVERY_STEPS="$2"; shift 2 ;;
    --rollout-prompt-tokens) ROLLOUT_UNLIKELIHOOD_PROMPT_TOKENS="$2"; shift 2 ;;
    --rollout-tokens) ROLLOUT_UNLIKELIHOOD_ROLLOUT_TOKENS="$2"; shift 2 ;;
    --rollout-history-tokens) ROLLOUT_UNLIKELIHOOD_HISTORY_TOKENS="$2"; shift 2 ;;
    --rollout-batch-prompts) ROLLOUT_UNLIKELIHOOD_BATCH_PROMPTS="$2"; shift 2 ;;
    --rollout-warmup) ROLLOUT_UNLIKELIHOOD_WARMUP_STEPS="$2"; shift 2 ;;
    --rollout-ramp) ROLLOUT_UNLIKELIHOOD_RAMP_STEPS="$2"; shift 2 ;;
    --rollout-recovery-only) ROLLOUT_UNLIKELIHOOD_RECOVERY_ONLY="$2"; shift 2 ;;
    --nextlat-start) NEXTLAT_START_AFTER="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --timeout-seconds) TIMEOUT_SECONDS="$2"; shift 2 ;;
    --probe-items) RULIAD_PROBE_ITEMS="$2"; shift 2 ;;
    --probe-tokens) RULIAD_PROBE_TOKENS="$2"; shift 2 ;;
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
case "$ENERGY_MODEL" in
  inherit|true|false) ;;
  *) echo "--energy-model must be inherit, true, or false" >&2; exit 2 ;;
esac
case "$STEP_CONTRACT" in
  inherit|true|false) ;;
  *) echo "--step-contract must be inherit, true, or false" >&2; exit 2 ;;
esac
case "$RULIAD_SUPERVISION_MODE" in
  inherit|full_document|answer_window|answer_completion|mixed) ;;
  *) echo "--ruliad-mode must be inherit, full_document, answer_window, answer_completion, or mixed" >&2; exit 2 ;;
esac
case "$ANSWER_RANKING" in
  inherit|true|false) ;;
  *) echo "--answer-ranking must be inherit, true, or false" >&2; exit 2 ;;
esac
case "$RULIAD_MASK_HIGH_ENTROPY_SPANS" in
  true|false) ;;
  *) echo "--ruliad-mask-high-entropy must be true or false" >&2; exit 2 ;;
esac
if (( RULIAD_ANSWER_CLOSE_MARKER_STRIDE < 0 )); then
  echo "--ruliad-answer-close-stride must be >= 0" >&2
  exit 2
fi
if (( RULIAD_ANSWER_CLOSE_MARKER_WEIGHT < 1 || RULIAD_ANSWER_CLOSE_MARKER_WEIGHT > 16 )); then
  echo "--ruliad-answer-close-weight must be in [1, 16]" >&2
  exit 2
fi
if (( RULIAD_ANSWER_SCHEMA_WEIGHT < 1 || RULIAD_ANSWER_SCHEMA_WEIGHT > 16 )); then
  echo "--ruliad-answer-schema-weight must be in [1, 16]" >&2
  exit 2
fi
if (( RULIAD_ANSWER_SCHEMA_START_WEIGHT < 1 || RULIAD_ANSWER_SCHEMA_START_WEIGHT > 16 )); then
  echo "--ruliad-answer-schema-start-weight must be in [1, 16]" >&2
  exit 2
fi
if (( RULIAD_ANSWER_VALUE_WEIGHT < 1 || RULIAD_ANSWER_VALUE_WEIGHT > 16 )); then
  echo "--ruliad-answer-value-weight must be in [1, 16]" >&2
  exit 2
fi
case "$ANSWER_DENOISING" in
  inherit|true|false) ;;
  *) echo "--answer-denoising must be inherit, true, or false" >&2; exit 2 ;;
esac
case "$ROLLOUT_UNLIKELIHOOD" in
  true|false) ;;
  *) echo "--rollout-unlikelihood must be true or false" >&2; exit 2 ;;
esac
case "$ROLLOUT_UNLIKELIHOOD_RECOVERY_ONLY" in
  true|false) ;;
  *) echo "--rollout-recovery-only must be true or false" >&2; exit 2 ;;
esac
case "$DYNAMICS_ENABLED" in
  inherit|true|false) ;;
  *) echo "--dynamics must be inherit, true, or false" >&2; exit 2 ;;
esac
if [[ "$EPOCHS" != "inherit" ]] && ! [[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--epochs must be inherit or a positive integer" >&2
  exit 2
fi
if [[ "$EPOCHS" != "inherit" ]]; then
  CHECKPOINT_INTERVAL_ITERS=$(( (MAX_ITERS + EPOCHS - 1) / EPOCHS ))
fi
if ! [[ "$CHECKPOINT_INTERVAL_ITERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "checkpoint_interval_iters must be a positive integer" >&2
  exit 2
fi
if (( ROLLOUT_UNLIKELIHOOD_EVERY_STEPS <= 0 )); then
  echo "--rollout-every must be > 0" >&2
  exit 2
fi
if (( ROLLOUT_UNLIKELIHOOD_PROMPT_TOKENS <= 0 )); then
  echo "--rollout-prompt-tokens must be > 0" >&2
  exit 2
fi
if (( ROLLOUT_UNLIKELIHOOD_ROLLOUT_TOKENS <= 0 )); then
  echo "--rollout-tokens must be > 0" >&2
  exit 2
fi
if (( ROLLOUT_UNLIKELIHOOD_HISTORY_TOKENS <= 0 )); then
  echo "--rollout-history-tokens must be > 0" >&2
  exit 2
fi
if (( ROLLOUT_UNLIKELIHOOD_BATCH_PROMPTS <= 0 )); then
  echo "--rollout-batch-prompts must be > 0" >&2
  exit 2
fi
if (( ROLLOUT_UNLIKELIHOOD_CYCLE_MIN_LAG <= 0 )); then
  echo "--rollout-cycle-lags min must be > 0" >&2
  exit 2
fi
if (( ROLLOUT_UNLIKELIHOOD_CYCLE_MAX_LAG < ROLLOUT_UNLIKELIHOOD_CYCLE_MIN_LAG )); then
  echo "--rollout-cycle-lags max must be >= min" >&2
  exit 2
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
  printf "trial_key\tmax_steps\tseed\tmax_iters\tbatch_size\tblock_size\tn_layer\tn_embd\tn_head\tlatent_total\tstatus\telapsed_seconds\tpeak_used_mb\tmin_available_mb\trun_dir\tmanifest\tlog\tgpu_log\n" > "$RUN_INDEX"
fi

if (( BUILD_RELEASE == 1 )); then
  (
    cd "$ROOT_DIR"
    export CARGO="$RUSTUP_CARGO"
    export RUSTC="$RUSTUP_RUSTC"
    "$RUSTUP_CARGO" build --release -p burn_dragon_language --example train_language --features "$FEATURES"
  )
elif (( DRY_RUN == 0 )); then
  if [[ ! -x "$TRAIN_BINARY" ]]; then
    echo "release train binary is missing or not executable: $TRAIN_BINARY" >&2
    echo "run without --no-build, or build it explicitly first" >&2
    exit 2
  fi
  newer_source="$(
    {
      find "$ROOT_DIR/crates" -type f \
        \( -name '*.rs' -o -name 'Cargo.toml' \) \
        -newer "$TRAIN_BINARY" -print -quit 2>/dev/null || true
      for path in "$ROOT_DIR/Cargo.toml" "$ROOT_DIR/Cargo.lock"; do
        if [[ -f "$path" && "$path" -nt "$TRAIN_BINARY" ]]; then
          printf '%s\n' "$path"
          break
        fi
      done
    } | head -n 1
  )"
  if [[ -n "$newer_source" ]]; then
    STALE_BINARY_RISK=true
    if [[ "$ALLOW_STALE_BINARY" != "1" ]]; then
      echo "source is newer than release train binary: $newer_source" >&2
      echo "rerun without --no-build, or pass --allow-stale-binary to acknowledge the risk" >&2
      exit 2
    fi
    echo "warning: source is newer than release train binary: $newer_source" >&2
  fi
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

json_bool_or_string() {
  case "$1" in
    true|false) printf '%s\n' "$1" ;;
    *) json_escape "$1" ;;
  esac
}

short_digest() {
  local value="$1"
  if command -v sha1sum >/dev/null 2>&1; then
    printf '%s' "$value" | sha1sum | cut -c1-12
  else
    python3 -c 'import hashlib,sys; print(hashlib.sha1(sys.stdin.read().encode()).hexdigest()[:12])' <<<"$value"
  fi
}

csv_to_toml_array() {
  python3 -c 'import sys; print("[" + ", ".join(str(int(x.strip())) for x in sys.argv[1].split(",") if x.strip()) + "]")' "$1"
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
[model]
n_layer = $N_LAYER
n_embd = $N_EMBD
n_head = $N_HEAD
latent_total = $LATENT_TOTAL

[model.latent_reasoning]
enabled = true
max_steps = $steps
min_steps = $steps
adaptive_halting = false
halt_threshold = 0.55
refiner_hidden_multiplier = 1
normalize_steps = $NORMALIZE_STEPS
residual_refinement_gate = $RESIDUAL_GATE
residual_refinement_gate_init = $RESIDUAL_GATE_INIT
energy_head = $ENERGY_HEAD
step_conditioned_decoder = $STEP_CONDITIONED_DECODER
step_conditioned_decoder_scale = $STEP_CONDITIONED_DECODER_SCALE
stop_bias_init = -2.0
energy_margin = 1.0

[training]
seed = $seed
max_iters = $MAX_ITERS
batch_size = $BATCH_SIZE
block_size = $BLOCK_SIZE
log_frequency = $LOG_FREQUENCY
checkpoint_interval_iters = $CHECKPOINT_INTERVAL_ITERS

[training.auto_batch_size]
enabled = $AUTO_BATCH_SIZE
min_batch_size = $AUTO_BATCH_MIN_BATCH_SIZE
max_batch_size = $AUTO_BATCH_MAX_BATCH_SIZE
max_probe_batch_size = $AUTO_BATCH_MAX_PROBE_BATCH_SIZE
target_device_memory_mb = $AUTO_BATCH_TARGET_DEVICE_MEMORY_MB
probe_steps = $AUTO_BATCH_PROBE_STEPS
binary_search = true
recompute_on_neuron_scale = true
max_system_memory_fraction = $AUTO_BATCH_MAX_SYSTEM_MEMORY_FRACTION
probe_safety_margin = $AUTO_BATCH_PROBE_SAFETY_MARGIN
EOF
  if [[ "$DYNAMICS_ENABLED" != "inherit" ]]; then
    {
      echo
      echo "[training.dynamics]"
      echo "enabled = $DYNAMICS_ENABLED"
    } >> "$path"
  fi
  cat >> "$path" <<EOF

[training.events]
flush_every_steps = 1
source_selection_every_steps = 16
ruliad_correctness_probe_every_epochs = 1
ruliad_correctness_probe_items = $RULIAD_PROBE_ITEMS
ruliad_correctness_probe_tokens = $RULIAD_PROBE_TOKENS

[training.latent_reasoning]
eval_step_sweep = $(csv_to_toml_array "$EVAL_STEPS_CSV")
jepa_every_steps = $JEPA_EVERY_STEPS
jepa_start_after_steps = 0

[training.latent_reasoning.next_latent]
every_steps = $NEXTLAT_EVERY_STEPS
start_after_steps = $NEXTLAT_START_AFTER
EOF
  if [[ "$ENERGY_MODEL" != "inherit" || -n "$ENERGY_START_AFTER" || -n "$ENERGY_EVERY_STEPS" ]]; then
    {
      echo
      echo "[training.latent_reasoning.energy_model]"
      if [[ "$ENERGY_MODEL" != "inherit" ]]; then
        echo "enabled = $ENERGY_MODEL"
      fi
      if [[ -n "$ENERGY_START_AFTER" ]]; then
        echo "start_after_steps = $ENERGY_START_AFTER"
      fi
      if [[ -n "$ENERGY_EVERY_STEPS" ]]; then
        echo "every_steps = $ENERGY_EVERY_STEPS"
      fi
    } >> "$path"
  fi
  if [[ "$STEP_CONTRACT" != "inherit" \
    || -n "$STEP_CONTRACT_START_AFTER" \
    || -n "$STEP_CONTRACT_EVERY_STEPS" \
    || -n "$STEP_CONTRACT_CE_WEIGHT" \
    || -n "$STEP_CONTRACT_TOKEN_KL_WEIGHT" \
    || -n "$STEP_CONTRACT_MONOTONIC_CE_WEIGHT" \
    || -n "$STEP_CONTRACT_CONTRACTIVE_WEIGHT" \
    || -n "$STEP_CONTRACT_CE_TOLERANCE" \
    || -n "$STEP_CONTRACT_TRUST_RADIUS" ]]; then
    {
      echo
      echo "[training.latent_reasoning.step_contract]"
      if [[ "$STEP_CONTRACT" != "inherit" ]]; then
        echo "enabled = $STEP_CONTRACT"
      fi
      if [[ -n "$STEP_CONTRACT_START_AFTER" ]]; then
        echo "start_after_steps = $STEP_CONTRACT_START_AFTER"
      fi
      if [[ -n "$STEP_CONTRACT_EVERY_STEPS" ]]; then
        echo "every_steps = $STEP_CONTRACT_EVERY_STEPS"
      fi
      if [[ -n "$STEP_CONTRACT_CE_WEIGHT" ]]; then
        echo "ce_weight = $STEP_CONTRACT_CE_WEIGHT"
      fi
      if [[ -n "$STEP_CONTRACT_TOKEN_KL_WEIGHT" ]]; then
        echo "token_kl_weight = $STEP_CONTRACT_TOKEN_KL_WEIGHT"
      fi
      if [[ -n "$STEP_CONTRACT_MONOTONIC_CE_WEIGHT" ]]; then
        echo "monotonic_ce_weight = $STEP_CONTRACT_MONOTONIC_CE_WEIGHT"
      fi
      if [[ -n "$STEP_CONTRACT_CONTRACTIVE_WEIGHT" ]]; then
        echo "contractive_weight = $STEP_CONTRACT_CONTRACTIVE_WEIGHT"
      fi
      if [[ -n "$STEP_CONTRACT_CE_TOLERANCE" ]]; then
        echo "ce_tolerance = $STEP_CONTRACT_CE_TOLERANCE"
      fi
      if [[ -n "$STEP_CONTRACT_TRUST_RADIUS" ]]; then
        echo "trust_radius = $STEP_CONTRACT_TRUST_RADIUS"
      fi
    } >> "$path"
  fi
  if [[ "$RULIAD_SUPERVISION_MODE" != "inherit" || "$RULIAD_MASK_HIGH_ENTROPY_SPANS" == "true" || "$RULIAD_ANSWER_CLOSE_MARKER_STRIDE" != "1" || "$RULIAD_ANSWER_CLOSE_MARKER_WEIGHT" != "1" || "$RULIAD_ANSWER_SCHEMA_WEIGHT" != "1" || "$RULIAD_ANSWER_SCHEMA_START_WEIGHT" != "1" || "$RULIAD_ANSWER_VALUE_WEIGHT" != "1" || "$ANSWER_RANKING" != "inherit" || "$ANSWER_DENOISING" != "inherit" || "$STRUCTURED_RECOVERY_WEIGHT" != "inherit" || "$STRUCTURED_RECOVERY_EVERY_STEPS" != "inherit" || "$STRUCTURED_RECOVERY_START_AFTER" != "inherit" || "$STRUCTURED_RECOVERY_MAX_COMPLETION_TOKENS" != "inherit" || "$STRUCTURED_RECOVERY_NEGATIVE_COUNT" != "inherit" || "$STRUCTURED_RECOVERY_TEMPLATE_NEGATIVE_COUNT" != "inherit" || "$STRUCTURED_RECOVERY_SCHEMA_NEGATIVE_COUNT" != "inherit" ]]; then
    {
      echo
      echo "[training.ruliad_supervision]"
      if [[ "$RULIAD_SUPERVISION_MODE" != "inherit" ]]; then
        echo "mode = \"$RULIAD_SUPERVISION_MODE\""
      fi
      echo "mask_high_entropy_spans = $RULIAD_MASK_HIGH_ENTROPY_SPANS"
      echo "answer_close_marker_stride = $RULIAD_ANSWER_CLOSE_MARKER_STRIDE"
      echo "answer_close_marker_weight = $RULIAD_ANSWER_CLOSE_MARKER_WEIGHT"
      echo "answer_schema_token_weight = $RULIAD_ANSWER_SCHEMA_WEIGHT"
      echo "answer_schema_start_token_weight = $RULIAD_ANSWER_SCHEMA_START_WEIGHT"
      echo "answer_value_token_weight = $RULIAD_ANSWER_VALUE_WEIGHT"
      if [[ "$ANSWER_RANKING" != "inherit" ]]; then
        echo
        echo "[training.ruliad_supervision.answer_ranking]"
        echo "enabled = $ANSWER_RANKING"
        echo "weight = $ANSWER_RANKING_WEIGHT"
        echo "margin = $ANSWER_RANKING_MARGIN"
        echo "corrupt_offset = $ANSWER_RANKING_CORRUPT_OFFSET"
      fi
      if [[ "$ANSWER_DENOISING" != "inherit" || "$STRUCTURED_RECOVERY_WEIGHT" != "inherit" || "$STRUCTURED_RECOVERY_EVERY_STEPS" != "inherit" || "$STRUCTURED_RECOVERY_START_AFTER" != "inherit" || "$STRUCTURED_RECOVERY_MAX_COMPLETION_TOKENS" != "inherit" || "$STRUCTURED_RECOVERY_NEGATIVE_COUNT" != "inherit" || "$STRUCTURED_RECOVERY_TEMPLATE_NEGATIVE_COUNT" != "inherit" || "$STRUCTURED_RECOVERY_SCHEMA_NEGATIVE_COUNT" != "inherit" ]]; then
        echo
        echo "[training.ruliad_supervision.answer_denoising]"
        if [[ "$ANSWER_DENOISING" != "inherit" ]]; then
          echo "enabled = $ANSWER_DENOISING"
          echo "weight = $ANSWER_DENOISING_WEIGHT"
          echo "probability = $ANSWER_DENOISING_PROBABILITY"
          echo "corrupt_offset = $ANSWER_DENOISING_CORRUPT_OFFSET"
        fi
        if [[ "$STRUCTURED_RECOVERY_WEIGHT" != "inherit" ]]; then
          echo "structured_recovery_weight = $STRUCTURED_RECOVERY_WEIGHT"
        fi
        if [[ "$STRUCTURED_RECOVERY_EVERY_STEPS" != "inherit" ]]; then
          echo "structured_recovery_every_steps = $STRUCTURED_RECOVERY_EVERY_STEPS"
        fi
        if [[ "$STRUCTURED_RECOVERY_START_AFTER" != "inherit" ]]; then
          echo "structured_recovery_start_after_steps = $STRUCTURED_RECOVERY_START_AFTER"
        fi
        if [[ "$STRUCTURED_RECOVERY_MAX_COMPLETION_TOKENS" != "inherit" ]]; then
          echo "structured_recovery_max_completion_tokens = $STRUCTURED_RECOVERY_MAX_COMPLETION_TOKENS"
        fi
        if [[ "$STRUCTURED_RECOVERY_NEGATIVE_COUNT" != "inherit" ]]; then
          echo "structured_recovery_negative_count = $STRUCTURED_RECOVERY_NEGATIVE_COUNT"
        fi
        if [[ "$STRUCTURED_RECOVERY_TEMPLATE_NEGATIVE_COUNT" != "inherit" ]]; then
          echo "structured_recovery_template_negative_count = $STRUCTURED_RECOVERY_TEMPLATE_NEGATIVE_COUNT"
        fi
        if [[ "$STRUCTURED_RECOVERY_SCHEMA_NEGATIVE_COUNT" != "inherit" ]]; then
          echo "structured_recovery_schema_negative_count = $STRUCTURED_RECOVERY_SCHEMA_NEGATIVE_COUNT"
        fi
      fi
    } >> "$path"
  fi
  if [[ "$ROLLOUT_UNLIKELIHOOD" == "true" ]]; then
    {
      echo
      echo "[training.greedy_rollout_unlikelihood]"
      echo "enabled = $ROLLOUT_UNLIKELIHOOD"
      echo "recovery_only = $ROLLOUT_UNLIKELIHOOD_RECOVERY_ONLY"
      echo "weight = $ROLLOUT_UNLIKELIHOOD_WEIGHT"
      echo "margin_weight = $ROLLOUT_UNLIKELIHOOD_MARGIN_WEIGHT"
      echo "margin = $ROLLOUT_UNLIKELIHOOD_MARGIN"
      echo "recovery_weight = $ROLLOUT_UNLIKELIHOOD_RECOVERY_WEIGHT"
      echo "sequence_recovery_weight = $ROLLOUT_UNLIKELIHOOD_SEQUENCE_RECOVERY_WEIGHT"
      echo "entropy_floor_weight = $ROLLOUT_UNLIKELIHOOD_ENTROPY_WEIGHT"
      echo "target_entropy_bits = $ROLLOUT_UNLIKELIHOOD_TARGET_ENTROPY_BITS"
      echo "cycle_weight = $ROLLOUT_UNLIKELIHOOD_CYCLE_WEIGHT"
      echo "cycle_margin_weight = $ROLLOUT_UNLIKELIHOOD_CYCLE_MARGIN_WEIGHT"
      echo "cycle_min_lag = $ROLLOUT_UNLIKELIHOOD_CYCLE_MIN_LAG"
      echo "cycle_max_lag = $ROLLOUT_UNLIKELIHOOD_CYCLE_MAX_LAG"
      echo "warmup_steps = $ROLLOUT_UNLIKELIHOOD_WARMUP_STEPS"
      echo "ramp_steps = $ROLLOUT_UNLIKELIHOOD_RAMP_STEPS"
      echo "every_steps = $ROLLOUT_UNLIKELIHOOD_EVERY_STEPS"
      echo "prompt_tokens = $ROLLOUT_UNLIKELIHOOD_PROMPT_TOKENS"
      echo "rollout_tokens = $ROLLOUT_UNLIKELIHOOD_ROLLOUT_TOKENS"
      echo "history_tokens = $ROLLOUT_UNLIKELIHOOD_HISTORY_TOKENS"
      echo "batch_prompts = $ROLLOUT_UNLIKELIHOOD_BATCH_PROMPTS"
    } >> "$path"
  fi
  if [[ "$FIELD_BINDING_CONTRAST_WEIGHT" != "inherit" || "$FIELD_BINDING_CONTRAST_EVERY_STEPS" != "inherit" || "$FIELD_BINDING_CONTRAST_MARGIN" != "inherit" || "$FIELD_BINDING_CONTRAST_PAIR_WEIGHT" != "inherit" || "$FIELD_BINDING_CONTRAST_MAX_PAIRS" != "inherit" || "$FIELD_BINDING_CONTRAST_REPLAY_CAPACITY" != "inherit" || "$GENERATED_ATTRACTOR_REPLAY_CAPACITY" != "inherit" || "$GENERATED_ATTRACTOR_REPLAY_MIN_COUNT" != "inherit" || "$GENERATED_ATTRACTOR_REPLAY_MAX_CANDIDATES" != "inherit" || "$GENERATED_ATTRACTOR_REPLAY_MIN_DISTINCT" != "inherit" || "$GENERATED_ATTRACTOR_REPLAY_MAX_DOMINANT" != "inherit" || "$VERIFIER_ROLLOUT_IMITATION_WEIGHT" != "inherit" || "$VERIFIER_ROLLOUT_RECOVERY_WEIGHT" != "inherit" || "$VERIFIER_ROLLOUT_EVERY_STEPS" != "inherit" || "$VERIFIER_ROLLOUT_START_AFTER" != "inherit" || "$VERIFIER_ROLLOUT_MIN_PARTIAL_PROGRESS_PPM" != "inherit" || "$VERIFIER_ROLLOUT_MIN_COMPLETION_QUALITY_PPM" != "inherit" || "$VERIFIER_ROLLOUT_MAX_ROWS_PER_STEP" != "inherit" ]]; then
    {
      echo
      echo "[training.ruliad_supervision.verifier_reward]"
      if [[ "$FIELD_BINDING_CONTRAST_WEIGHT" != "inherit" ]]; then
        echo "field_binding_contrast_weight = $FIELD_BINDING_CONTRAST_WEIGHT"
      fi
      if [[ "$FIELD_BINDING_CONTRAST_EVERY_STEPS" != "inherit" ]]; then
        echo "field_binding_contrast_every_steps = $FIELD_BINDING_CONTRAST_EVERY_STEPS"
      fi
      if [[ "$FIELD_BINDING_CONTRAST_MARGIN" != "inherit" ]]; then
        echo "field_binding_contrast_margin = $FIELD_BINDING_CONTRAST_MARGIN"
      fi
      if [[ "$FIELD_BINDING_CONTRAST_PAIR_WEIGHT" != "inherit" ]]; then
        echo "field_binding_contrast_pair_weight = $FIELD_BINDING_CONTRAST_PAIR_WEIGHT"
      fi
      if [[ "$FIELD_BINDING_CONTRAST_MAX_PAIRS" != "inherit" ]]; then
        echo "field_binding_contrast_max_pairs = $FIELD_BINDING_CONTRAST_MAX_PAIRS"
      fi
      if [[ "$FIELD_BINDING_CONTRAST_REPLAY_CAPACITY" != "inherit" ]]; then
        echo "field_binding_contrast_replay_capacity = $FIELD_BINDING_CONTRAST_REPLAY_CAPACITY"
      fi
      if [[ "$GENERATED_ATTRACTOR_REPLAY_CAPACITY" != "inherit" ]]; then
        echo "generated_attractor_replay_capacity = $GENERATED_ATTRACTOR_REPLAY_CAPACITY"
      fi
      if [[ "$GENERATED_ATTRACTOR_REPLAY_MIN_COUNT" != "inherit" ]]; then
        echo "generated_attractor_replay_min_count = $GENERATED_ATTRACTOR_REPLAY_MIN_COUNT"
      fi
      if [[ "$GENERATED_ATTRACTOR_REPLAY_MAX_CANDIDATES" != "inherit" ]]; then
        echo "generated_attractor_replay_max_candidates = $GENERATED_ATTRACTOR_REPLAY_MAX_CANDIDATES"
      fi
      if [[ "$GENERATED_ATTRACTOR_REPLAY_MIN_DISTINCT" != "inherit" ]]; then
        echo "generated_attractor_replay_min_distinct_answers = $GENERATED_ATTRACTOR_REPLAY_MIN_DISTINCT"
      fi
      if [[ "$GENERATED_ATTRACTOR_REPLAY_MAX_DOMINANT" != "inherit" ]]; then
        echo "generated_attractor_replay_max_dominant_fraction = $GENERATED_ATTRACTOR_REPLAY_MAX_DOMINANT"
      fi
      if [[ "$VERIFIER_ROLLOUT_IMITATION_WEIGHT" != "inherit" ]]; then
        echo "rollout_imitation_weight = $VERIFIER_ROLLOUT_IMITATION_WEIGHT"
      fi
      if [[ "$VERIFIER_ROLLOUT_RECOVERY_WEIGHT" != "inherit" ]]; then
        echo "rollout_recovery_weight = $VERIFIER_ROLLOUT_RECOVERY_WEIGHT"
      fi
      if [[ "$VERIFIER_ROLLOUT_EVERY_STEPS" != "inherit" ]]; then
        echo "rollout_imitation_every_steps = $VERIFIER_ROLLOUT_EVERY_STEPS"
      fi
      if [[ "$VERIFIER_ROLLOUT_START_AFTER" != "inherit" ]]; then
        echo "rollout_imitation_start_after_steps = $VERIFIER_ROLLOUT_START_AFTER"
      fi
      if [[ "$VERIFIER_ROLLOUT_MIN_PARTIAL_PROGRESS_PPM" != "inherit" ]]; then
        echo "rollout_imitation_min_partial_progress_ppm = $VERIFIER_ROLLOUT_MIN_PARTIAL_PROGRESS_PPM"
      fi
      if [[ "$VERIFIER_ROLLOUT_MIN_COMPLETION_QUALITY_PPM" != "inherit" ]]; then
        echo "rollout_imitation_min_completion_quality_ppm = $VERIFIER_ROLLOUT_MIN_COMPLETION_QUALITY_PPM"
      fi
      if [[ "$VERIFIER_ROLLOUT_MAX_ROWS_PER_STEP" != "inherit" ]]; then
        echo "rollout_imitation_max_rows_per_step = $VERIFIER_ROLLOUT_MAX_ROWS_PER_STEP"
      fi
    } >> "$path"
  fi
}

latest_run_dir() {
  local run_root="$1"
  find "$run_root" -mindepth 1 -maxdepth 4 -type f -name dashboard.md -printf '%T@ %h\n' 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

write_manifest() {
  local manifest="$1"
  local trial_key="$2"
  local steps="$3"
  local seed="$4"
  local overlay="$5"
  local run_root="$6"
  local run_dir="$7"
  local log_path="$8"
  local gpu_log="$9"
  local status="${10}"
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
  "base_profile": $(json_escape "$BASE_PROFILE"),
  "max_steps": $steps,
  "adaptive_halting": false,
  "eval_step_sweep": $(json_escape "$EVAL_STEPS_CSV"),
  "seed": $seed,
  "max_iters": $MAX_ITERS,
  "logical_epochs": $(json_bool_or_string "$EPOCHS"),
  "checkpoint_interval_iters": $CHECKPOINT_INTERVAL_ITERS,
  "batch_size": $BATCH_SIZE,
  "block_size": $BLOCK_SIZE,
  "n_layer": $N_LAYER,
  "n_embd": $N_EMBD,
  "n_head": $N_HEAD,
  "latent_total": $LATENT_TOTAL,
  "energy_head": $ENERGY_HEAD,
  "residual_refinement_gate": $RESIDUAL_GATE,
  "residual_refinement_gate_init": $RESIDUAL_GATE_INIT,
  "normalize_steps": $NORMALIZE_STEPS,
  "step_conditioned_decoder": $STEP_CONDITIONED_DECODER,
  "step_conditioned_decoder_scale": $STEP_CONDITIONED_DECODER_SCALE,
  "energy_model": $(json_escape "$ENERGY_MODEL"),
  "energy_start_after_steps": $(if [[ -n "$ENERGY_START_AFTER" ]]; then echo "$ENERGY_START_AFTER"; else echo "null"; fi),
  "energy_every_steps": $(if [[ -n "$ENERGY_EVERY_STEPS" ]]; then echo "$ENERGY_EVERY_STEPS"; else echo "null"; fi),
  "step_contract": $(json_escape "$STEP_CONTRACT"),
  "step_contract_start_after_steps": $(if [[ -n "$STEP_CONTRACT_START_AFTER" ]]; then echo "$STEP_CONTRACT_START_AFTER"; else echo "null"; fi),
  "step_contract_every_steps": $(if [[ -n "$STEP_CONTRACT_EVERY_STEPS" ]]; then echo "$STEP_CONTRACT_EVERY_STEPS"; else echo "null"; fi),
  "step_contract_ce_weight": $(if [[ -n "$STEP_CONTRACT_CE_WEIGHT" ]]; then echo "$STEP_CONTRACT_CE_WEIGHT"; else echo "null"; fi),
  "step_contract_token_kl_weight": $(if [[ -n "$STEP_CONTRACT_TOKEN_KL_WEIGHT" ]]; then echo "$STEP_CONTRACT_TOKEN_KL_WEIGHT"; else echo "null"; fi),
  "step_contract_monotonic_ce_weight": $(if [[ -n "$STEP_CONTRACT_MONOTONIC_CE_WEIGHT" ]]; then echo "$STEP_CONTRACT_MONOTONIC_CE_WEIGHT"; else echo "null"; fi),
  "step_contract_contractive_weight": $(if [[ -n "$STEP_CONTRACT_CONTRACTIVE_WEIGHT" ]]; then echo "$STEP_CONTRACT_CONTRACTIVE_WEIGHT"; else echo "null"; fi),
  "step_contract_ce_tolerance": $(if [[ -n "$STEP_CONTRACT_CE_TOLERANCE" ]]; then echo "$STEP_CONTRACT_CE_TOLERANCE"; else echo "null"; fi),
  "step_contract_trust_radius": $(if [[ -n "$STEP_CONTRACT_TRUST_RADIUS" ]]; then echo "$STEP_CONTRACT_TRUST_RADIUS"; else echo "null"; fi),
  "ruliad_supervision_mode": $(json_escape "$RULIAD_SUPERVISION_MODE"),
  "ruliad_mask_high_entropy_spans": $RULIAD_MASK_HIGH_ENTROPY_SPANS,
  "ruliad_answer_close_marker_stride": $RULIAD_ANSWER_CLOSE_MARKER_STRIDE,
  "ruliad_answer_close_marker_weight": $RULIAD_ANSWER_CLOSE_MARKER_WEIGHT,
  "ruliad_answer_schema_weight": $RULIAD_ANSWER_SCHEMA_WEIGHT,
  "ruliad_answer_schema_start_weight": $RULIAD_ANSWER_SCHEMA_START_WEIGHT,
  "ruliad_answer_value_weight": $RULIAD_ANSWER_VALUE_WEIGHT,
  "answer_ranking": $(json_bool_or_string "$ANSWER_RANKING"),
  "answer_ranking_weight": $ANSWER_RANKING_WEIGHT,
  "answer_ranking_margin": $ANSWER_RANKING_MARGIN,
  "answer_ranking_corrupt_offset": $ANSWER_RANKING_CORRUPT_OFFSET,
  "answer_denoising": $(json_bool_or_string "$ANSWER_DENOISING"),
  "answer_denoising_weight": $ANSWER_DENOISING_WEIGHT,
  "answer_denoising_probability": $ANSWER_DENOISING_PROBABILITY,
  "answer_denoising_corrupt_offset": $ANSWER_DENOISING_CORRUPT_OFFSET,
  "structured_recovery_weight": $(json_bool_or_string "$STRUCTURED_RECOVERY_WEIGHT"),
  "structured_recovery_every_steps": $(json_bool_or_string "$STRUCTURED_RECOVERY_EVERY_STEPS"),
  "structured_recovery_start_after_steps": $(json_bool_or_string "$STRUCTURED_RECOVERY_START_AFTER"),
  "structured_recovery_max_completion_tokens": $(json_bool_or_string "$STRUCTURED_RECOVERY_MAX_COMPLETION_TOKENS"),
  "structured_recovery_negative_count": $(json_bool_or_string "$STRUCTURED_RECOVERY_NEGATIVE_COUNT"),
  "structured_recovery_template_negative_count": $(json_bool_or_string "$STRUCTURED_RECOVERY_TEMPLATE_NEGATIVE_COUNT"),
  "structured_recovery_schema_negative_count": $(json_bool_or_string "$STRUCTURED_RECOVERY_SCHEMA_NEGATIVE_COUNT"),
  "field_binding_contrast_weight": $(json_bool_or_string "$FIELD_BINDING_CONTRAST_WEIGHT"),
  "field_binding_contrast_every_steps": $(json_bool_or_string "$FIELD_BINDING_CONTRAST_EVERY_STEPS"),
  "field_binding_contrast_margin": $(json_bool_or_string "$FIELD_BINDING_CONTRAST_MARGIN"),
  "field_binding_contrast_pair_weight": $(json_bool_or_string "$FIELD_BINDING_CONTRAST_PAIR_WEIGHT"),
  "field_binding_contrast_max_pairs": $(json_bool_or_string "$FIELD_BINDING_CONTRAST_MAX_PAIRS"),
  "field_binding_contrast_replay_capacity": $(json_bool_or_string "$FIELD_BINDING_CONTRAST_REPLAY_CAPACITY"),
  "generated_attractor_replay_capacity": $(json_bool_or_string "$GENERATED_ATTRACTOR_REPLAY_CAPACITY"),
  "generated_attractor_replay_min_count": $(json_bool_or_string "$GENERATED_ATTRACTOR_REPLAY_MIN_COUNT"),
  "generated_attractor_replay_max_candidates": $(json_bool_or_string "$GENERATED_ATTRACTOR_REPLAY_MAX_CANDIDATES"),
  "generated_attractor_replay_min_distinct": $(json_bool_or_string "$GENERATED_ATTRACTOR_REPLAY_MIN_DISTINCT"),
  "generated_attractor_replay_max_dominant": $(json_bool_or_string "$GENERATED_ATTRACTOR_REPLAY_MAX_DOMINANT"),
  "verifier_rollout_imitation_weight": $(json_bool_or_string "$VERIFIER_ROLLOUT_IMITATION_WEIGHT"),
  "verifier_rollout_recovery_weight": $(json_bool_or_string "$VERIFIER_ROLLOUT_RECOVERY_WEIGHT"),
  "verifier_rollout_every_steps": $(json_bool_or_string "$VERIFIER_ROLLOUT_EVERY_STEPS"),
  "verifier_rollout_start_after_steps": $(json_bool_or_string "$VERIFIER_ROLLOUT_START_AFTER"),
  "verifier_rollout_min_partial_progress_ppm": $(json_bool_or_string "$VERIFIER_ROLLOUT_MIN_PARTIAL_PROGRESS_PPM"),
  "verifier_rollout_min_completion_quality_ppm": $(json_bool_or_string "$VERIFIER_ROLLOUT_MIN_COMPLETION_QUALITY_PPM"),
  "verifier_rollout_max_rows_per_step": $(json_bool_or_string "$VERIFIER_ROLLOUT_MAX_ROWS_PER_STEP"),
  "rollout_unlikelihood": $ROLLOUT_UNLIKELIHOOD,
  "rollout_unlikelihood_weight": $ROLLOUT_UNLIKELIHOOD_WEIGHT,
  "rollout_unlikelihood_margin_weight": $ROLLOUT_UNLIKELIHOOD_MARGIN_WEIGHT,
  "rollout_unlikelihood_margin": $ROLLOUT_UNLIKELIHOOD_MARGIN,
  "rollout_unlikelihood_recovery_weight": $ROLLOUT_UNLIKELIHOOD_RECOVERY_WEIGHT,
  "rollout_unlikelihood_sequence_recovery_weight": $ROLLOUT_UNLIKELIHOOD_SEQUENCE_RECOVERY_WEIGHT,
  "rollout_unlikelihood_entropy_floor_weight": $ROLLOUT_UNLIKELIHOOD_ENTROPY_WEIGHT,
  "rollout_unlikelihood_target_entropy_bits": $ROLLOUT_UNLIKELIHOOD_TARGET_ENTROPY_BITS,
  "rollout_unlikelihood_cycle_weight": $ROLLOUT_UNLIKELIHOOD_CYCLE_WEIGHT,
  "rollout_unlikelihood_cycle_margin_weight": $ROLLOUT_UNLIKELIHOOD_CYCLE_MARGIN_WEIGHT,
  "rollout_unlikelihood_cycle_min_lag": $ROLLOUT_UNLIKELIHOOD_CYCLE_MIN_LAG,
  "rollout_unlikelihood_cycle_max_lag": $ROLLOUT_UNLIKELIHOOD_CYCLE_MAX_LAG,
  "rollout_unlikelihood_every_steps": $ROLLOUT_UNLIKELIHOOD_EVERY_STEPS,
  "rollout_unlikelihood_prompt_tokens": $ROLLOUT_UNLIKELIHOOD_PROMPT_TOKENS,
  "rollout_unlikelihood_rollout_tokens": $ROLLOUT_UNLIKELIHOOD_ROLLOUT_TOKENS,
  "rollout_unlikelihood_history_tokens": $ROLLOUT_UNLIKELIHOOD_HISTORY_TOKENS,
  "rollout_unlikelihood_batch_prompts": $ROLLOUT_UNLIKELIHOOD_BATCH_PROMPTS,
  "rollout_unlikelihood_warmup_steps": $ROLLOUT_UNLIKELIHOOD_WARMUP_STEPS,
  "rollout_unlikelihood_ramp_steps": $ROLLOUT_UNLIKELIHOOD_RAMP_STEPS,
  "rollout_unlikelihood_recovery_only": $ROLLOUT_UNLIKELIHOOD_RECOVERY_ONLY,
  "nextlat_start_after_steps": $NEXTLAT_START_AFTER,
  "ruliad_probe_items": $RULIAD_PROBE_ITEMS,
  "ruliad_probe_tokens": $RULIAD_PROBE_TOKENS,
  "dynamics_enabled": $(json_bool_or_string "$DYNAMICS_ENABLED"),
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
  "git_dirty": $dirty,
  "stale_binary_risk": $STALE_BINARY_RISK
}
EOF
}

run_trial() {
  local steps="$1"
  local seed="$2"
  local trial_key
  local overlay
  local log_path
  local gpu_log
  local manifest
  local run_root
  local run_dir=""
  local answer_rank_key
  local answer_denoise_weight_key
  local answer_denoise_prob_key
  local answer_close_key
  local answer_value_key
  local rollout_key
  local full_trial_key
  local trial_digest

  answer_rank_key="${ANSWER_RANKING_WEIGHT//./p}"
  answer_denoise_weight_key="${ANSWER_DENOISING_WEIGHT//./p}"
  answer_denoise_prob_key="${ANSWER_DENOISING_PROBABILITY//./p}"
  local structured_recovery_key="srw${STRUCTURED_RECOVERY_WEIGHT//./p}e${STRUCTURED_RECOVERY_EVERY_STEPS}s${STRUCTURED_RECOVERY_START_AFTER}t${STRUCTURED_RECOVERY_MAX_COMPLETION_TOKENS}n${STRUCTURED_RECOVERY_NEGATIVE_COUNT}u${STRUCTURED_RECOVERY_TEMPLATE_NEGATIVE_COUNT}c${STRUCTURED_RECOVERY_SCHEMA_NEGATIVE_COUNT}"
  local field_binding_key="fbw${FIELD_BINDING_CONTRAST_WEIGHT//./p}p${FIELD_BINDING_CONTRAST_PAIR_WEIGHT//./p}e${FIELD_BINDING_CONTRAST_EVERY_STEPS}m${FIELD_BINDING_CONTRAST_MAX_PAIRS}"
  local generated_attractor_key="gac${GENERATED_ATTRACTOR_REPLAY_CAPACITY}n${GENERATED_ATTRACTOR_REPLAY_MIN_COUNT}m${GENERATED_ATTRACTOR_REPLAY_MAX_CANDIDATES}d${GENERATED_ATTRACTOR_REPLAY_MIN_DISTINCT}q${GENERATED_ATTRACTOR_REPLAY_MAX_DOMINANT}"
  local verifier_rollout_imitation_key="${VERIFIER_ROLLOUT_IMITATION_WEIGHT//./p}"
  local verifier_rollout_recovery_key="${VERIFIER_ROLLOUT_RECOVERY_WEIGHT//./p}"
  local verifier_rollout_key="vri${verifier_rollout_imitation_key}vrr${verifier_rollout_recovery_key}e${VERIFIER_ROLLOUT_EVERY_STEPS}s${VERIFIER_ROLLOUT_START_AFTER}q${VERIFIER_ROLLOUT_MIN_COMPLETION_QUALITY_PPM}r${VERIFIER_ROLLOUT_MAX_ROWS_PER_STEP}"
  answer_close_key="c${RULIAD_ANSWER_CLOSE_MARKER_STRIDE}w${RULIAD_ANSWER_CLOSE_MARKER_WEIGHT}"
  answer_value_key="s${RULIAD_ANSWER_SCHEMA_WEIGHT}k${RULIAD_ANSWER_SCHEMA_START_WEIGHT}v${RULIAD_ANSWER_VALUE_WEIGHT}"
  rollout_key="off"
  if [[ "$ROLLOUT_UNLIKELIHOOD" == "true" ]]; then
    local rollout_weight_key="${ROLLOUT_UNLIKELIHOOD_WEIGHT//./p}"
    local rollout_cycle_key="${ROLLOUT_UNLIKELIHOOD_CYCLE_WEIGHT//./p}"
    local rollout_sequence_key="${ROLLOUT_UNLIKELIHOOD_SEQUENCE_RECOVERY_WEIGHT//./p}"
    local rollout_entropy_key="${ROLLOUT_UNLIKELIHOOD_ENTROPY_WEIGHT//./p}"
    local rollout_target_entropy_key="${ROLLOUT_UNLIKELIHOOD_TARGET_ENTROPY_BITS//./p}"
    rollout_key="onw${rollout_weight_key}c${rollout_cycle_key}s${rollout_sequence_key}h${rollout_entropy_key}b${rollout_target_entropy_key}e${ROLLOUT_UNLIKELIHOOD_EVERY_STEPS}t${ROLLOUT_UNLIKELIHOOD_ROLLOUT_TOKENS}"
  fi
  full_trial_key="latent-steps-ms${steps}-seed${seed}-i${MAX_ITERS}-b${BATCH_SIZE}-bs${BLOCK_SIZE}-l${N_LAYER}-e${N_EMBD}-h${N_HEAD}-z${LATENT_TOTAL}-ckpt${CHECKPOINT_INTERVAL_ITERS}-rs${RULIAD_SUPERVISION_MODE}m${RULIAD_MASK_HIGH_ENTROPY_SPANS}${answer_close_key}${answer_value_key}-ar${ANSWER_RANKING}w${answer_rank_key}-ad${ANSWER_DENOISING}w${answer_denoise_weight_key}p${answer_denoise_prob_key}-${structured_recovery_key}-${field_binding_key}-${generated_attractor_key}-${verifier_rollout_key}-ru${rollout_key}-${BACKEND}"
  trial_digest="$(short_digest "$full_trial_key")"
  trial_key="latent-steps-ms${steps}-seed${seed}-i${MAX_ITERS}-b${BATCH_SIZE}-bs${BLOCK_SIZE}-l${N_LAYER}-e${N_EMBD}-h${N_HEAD}-z${LATENT_TOTAL}-cfg${trial_digest}-${BACKEND}"
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
    --config "$BASE_PROFILE"
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
    write_manifest "$manifest" "$trial_key" "$steps" "$seed" "$overlay" "$run_root" "" "$log_path" "$gpu_log" "$MONITOR_STATUS"
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
    write_manifest "$manifest" "$trial_key" "$steps" "$seed" "$overlay" "$run_root" "$run_dir" "$log_path" "$gpu_log" "$MONITOR_STATUS"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$trial_key" "$steps" "$seed" "$MAX_ITERS" "$BATCH_SIZE" "$BLOCK_SIZE" "$N_LAYER" "$N_EMBD" "$N_HEAD" "$LATENT_TOTAL" "$MONITOR_STATUS" "$MONITOR_ELAPSED_SECONDS" "$MONITOR_PEAK_USED_MB" "$MONITOR_MIN_AVAILABLE_MB" "$run_dir" "$manifest" "$log_path" "$gpu_log" \
    | tee -a "$RUN_INDEX"

  [[ "$MONITOR_STATUS" == "ok" || "$MONITOR_STATUS" == "dry_run" ]]
}

IFS=',' read -r -a STEPS <<< "$STEPS_CSV"
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"

echo "latent reasoning max_steps ablation output: $OUT_DIR"
echo "base_profile=$BASE_PROFILE"
echo "shape: n_layer=$N_LAYER n_embd=$N_EMBD n_head=$N_HEAD latent_total=$LATENT_TOTAL block_size=$BLOCK_SIZE batch_size=$BATCH_SIZE energy_head=$ENERGY_HEAD residual_gate=$RESIDUAL_GATE residual_gate_init=$RESIDUAL_GATE_INIT normalize_steps=$NORMALIZE_STEPS step_decoder=$STEP_CONDITIONED_DECODER energy_model=$ENERGY_MODEL step_contract=$STEP_CONTRACT ruliad_mode=$RULIAD_SUPERVISION_MODE ruliad_mask_high_entropy=$RULIAD_MASK_HIGH_ENTROPY_SPANS ruliad_answer_close_stride=$RULIAD_ANSWER_CLOSE_MARKER_STRIDE ruliad_answer_close_weight=$RULIAD_ANSWER_CLOSE_MARKER_WEIGHT ruliad_answer_schema_weight=$RULIAD_ANSWER_SCHEMA_WEIGHT ruliad_answer_schema_start_weight=$RULIAD_ANSWER_SCHEMA_START_WEIGHT ruliad_answer_value_weight=$RULIAD_ANSWER_VALUE_WEIGHT answer_ranking=$ANSWER_RANKING answer_ranking_weight=$ANSWER_RANKING_WEIGHT answer_denoising=$ANSWER_DENOISING answer_denoising_weight=$ANSWER_DENOISING_WEIGHT rollout_unlikelihood=$ROLLOUT_UNLIKELIHOOD rollout_weight=$ROLLOUT_UNLIKELIHOOD_WEIGHT rollout_cycle_weight=$ROLLOUT_UNLIKELIHOOD_CYCLE_WEIGHT"
echo "schedules: max_iters=$MAX_ITERS epochs=$EPOCHS checkpoint_interval=$CHECKPOINT_INTERVAL_ITERS jepa_every=$JEPA_EVERY_STEPS nextlat_every=$NEXTLAT_EVERY_STEPS nextlat_start=$NEXTLAT_START_AFTER energy_every=${ENERGY_EVERY_STEPS:-profile} energy_start=${ENERGY_START_AFTER:-profile} step_contract_every=${STEP_CONTRACT_EVERY_STEPS:-profile} step_contract_start=${STEP_CONTRACT_START_AFTER:-profile} step_contract_ce=${STEP_CONTRACT_CE_WEIGHT:-profile} step_contract_mono=${STEP_CONTRACT_MONOTONIC_CE_WEIGHT:-profile} step_contract_contract=${STEP_CONTRACT_CONTRACTIVE_WEIGHT:-profile} probe_items=$RULIAD_PROBE_ITEMS probe_tokens=$RULIAD_PROBE_TOKENS"
echo "rollout schedule: every=$ROLLOUT_UNLIKELIHOOD_EVERY_STEPS prompt_tokens=$ROLLOUT_UNLIKELIHOOD_PROMPT_TOKENS rollout_tokens=$ROLLOUT_UNLIKELIHOOD_ROLLOUT_TOKENS history_tokens=$ROLLOUT_UNLIKELIHOOD_HISTORY_TOKENS batch_prompts=$ROLLOUT_UNLIKELIHOOD_BATCH_PROMPTS recovery_only=$ROLLOUT_UNLIKELIHOOD_RECOVERY_ONLY"
echo "latent eval step sweep=$EVAL_STEPS_CSV"
echo "RAM guards: max_system_memory_fraction=$MAX_SYSTEM_MEMORY_FRACTION min_available_mb=$MIN_AVAILABLE_MB dynamics=$DYNAMICS_ENABLED"
echo "max_step_values=${#STEPS[@]} seeds=${#SEEDS[@]} backend=$BACKEND"

for seed in "${SEEDS[@]}"; do
  for steps in "${STEPS[@]}"; do
    run_trial "$steps" "$seed" || {
      echo "stopping ablation after failed/guarded trial: max_steps=$steps seed=$seed status=$MONITOR_STATUS" >&2
      exit 1
    }
  done
done

echo "latent reasoning max_steps ablation complete: $RUN_INDEX"
