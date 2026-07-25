#!/usr/bin/env python3
"""Apply promotion gates to ruliad continual-learning candidate arms."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


TRIAL_SUMMARY = "latent_reasoning_steps_summary.csv"

METRIC_COLUMNS = [
    "max_iters",
    "stage_model_tokens_per_sec",
    "gpu_util_mean",
    "elapsed_seconds",
    "peak_used_mb",
    "valid_teacher_ce_last",
    "latent_eval_final_ce_delta_last",
    "latent_eval_final_ce_violation_last",
    "latent_eval_final_entropy_bits_last",
    "latent_eval_final_delta_rms_last",
    "latent_extra_eval_max_ce_delta_last",
    "latent_extra_eval_max_ce_violation_last",
    "latent_extra_eval_min_entropy_bits_last",
    "latent_extra_eval_max_delta_rms_last",
    "source_mean_difficulty_last",
    "source_active_max_difficulty_last",
    "source_mastered_max_difficulty_last",
    "source_max_difficulty_level_last",
    "source_materialized_frontier_edge_last",
    "source_max_difficulty_probability_last",
    "source_normalized_difficulty_score_last",
    "source_target_difficulty_score_last",
    "source_entropy_bits_last",
    "source_active_candidate_count_last",
    "source_active_max_entropy_bits_last",
    "source_normalized_entropy_last",
    "source_hash_noise_probability_last",
    "source_capability_lagging_probability_last",
    "ruliad_verifier_last",
    "ruliad_semantic_last",
    "ruliad_partial_last",
    "ruliad_schema_wrong_last",
    "ruliad_malformed_last",
    "ruliad_answer_field_accuracy_last",
    "ruliad_answer_field_coverage_last",
    "ruliad_answer_termination_rate_last",
    "ruliad_completion_quality_last",
    "ruliad_expected_answer_distinct_last",
    "ruliad_answer_distinct_last",
    "completion_health_last",
    "completion_distinct_2_last",
    "completion_period_2_to_64_last",
    "completion_repetition_last",
    "contract_probe_verifier_last",
    "contract_probe_semantic_last",
    "contract_probe_partial_last",
    "contract_probe_schema_wrong_last",
    "contract_probe_malformed_last",
    "contract_probe_answer_field_accuracy_last",
    "contract_probe_answer_field_coverage_last",
    "contract_probe_completion_health_last",
    "contract_probe_answer_distinct_last",
    "contract_probe_verifier_delta",
    "contract_probe_completion_delta",
    "contract_probe_answer_field_delta",
    "contract_probe_answer_distinct_delta",
    "capability_score_auc",
    "capability_verifier_auc",
    "capability_score_drop_from_best",
    "capability_verifier_drop_from_best",
    "capability_completion_drop_from_best",
    "capability_bucket_lagging_count",
    "capability_contract_lagging_count",
    "recovery_control_fraction",
    "source_capability_recovery_control_count",
    "source_capability_recovery_control_fraction",
    "capability_quality_recovery_count",
    "capability_gate_failed_count",
    "best_eval_steps",
    "best_eval_verifier",
    "best_eval_semantic",
    "best_eval_partial",
    "best_eval_schema_wrong",
    "best_eval_completion",
    "best_eval_verifier_delta",
    "best_eval_schema_delta",
    "best_eval_completion_delta",
    "extra_eval_step_count",
    "extra_eval_worst_steps",
    "extra_eval_min_verifier",
    "extra_eval_min_verifier_delta",
    "extra_eval_min_completion",
    "extra_eval_min_completion_delta",
    "extra_eval_max_schema_delta",
    "extra_eval_max_malformed_delta",
    "extra_eval_max_period_2_to_64",
    "output_entropy_bits_last",
    "output_distinct_2_last",
    "output_repetition_last",
    "output_period_2_to_64_last",
    "fatal_gate_count",
    "rank_score",
]

POLICY_TELEMETRY_FILE = "events/ruliad_verifier_policy.jsonl"
POLICY_METRIC_COLUMNS = [
    "policy_sample_groups",
    "policy_completion_rows",
    "policy_oracle_sample_groups",
    "policy_oracle_completion_rows",
    "policy_oracle_truncated_completion_rows",
    "policy_structured_negative_completion_rows",
    "policy_generated_attractor_completion_rows",
    "policy_gated_sample_groups",
    "policy_gated_completion_rows",
    "policy_scalarization_count",
    "policy_reward_mean",
    "policy_reward_std",
    "policy_reward_min",
    "policy_reward_max",
    "policy_advantage_mean",
    "policy_advantage_std",
    "policy_advantage_clip_fraction",
    "policy_update_applied_fraction",
    "policy_update_skipped_count",
    "policy_vector_verifier_match_mean",
    "policy_vector_semantic_match_mean",
    "policy_vector_partial_progress_mean",
    "policy_vector_field_accuracy_mean",
    "policy_vector_certificate_prefix_mean",
    "policy_vector_compactness_mean",
    "policy_vector_schema_quality_mean",
    "policy_vector_hash_safety_mean",
    "policy_vector_answer_termination_mean",
    "policy_vector_completion_health_mean",
    "policy_vpo_dominant_compactness",
    "policy_vpo_dominant_schema_quality",
    "policy_vpo_dominant_completion_health",
]

POLICY_CONFIG_COLUMNS = [
    "policy_config_enabled",
    "policy_config_weight",
    "policy_config_start_after_steps",
    "policy_config_every_steps",
    "policy_config_expected_update_steps",
]

STRUCTURED_CONTRAST_TELEMETRY_FILE = "events/ruliad_structured_contrast.jsonl"
STRUCTURED_CONTRAST_METRIC_COLUMNS = [
    "contrast_sample_groups",
    "contrast_oracle_completion_rows",
    "contrast_field_negative_completion_rows",
    "contrast_template_negative_completion_rows",
    "contrast_generated_attractor_negative_completion_rows",
    "contrast_pairs",
    "contrast_discriminative_tokens",
    "contrast_weight",
    "contrast_margin",
]

STRUCTURED_CONTRAST_CONFIG_COLUMNS = [
    "contrast_config_weight",
    "contrast_config_start_after_steps",
    "contrast_config_every_steps",
    "contrast_config_expected_update_steps",
]

FIELD_BINDING_CONTRAST_TELEMETRY_FILE = "events/ruliad_field_binding_contrast.jsonl"
FIELD_BINDING_CONTRAST_METRIC_COLUMNS = [
    "field_binding_sample_groups",
    "field_binding_prompt_pairs",
    "field_binding_contrast_pairs",
    "field_binding_candidate_pairs",
    "field_binding_discriminative_tokens",
    "field_binding_negative_pool_size",
    "field_binding_replay_pool_size",
    "field_binding_replay_contrast_pairs",
    "field_binding_generated_attractor_pool_size",
    "field_binding_generated_attractor_negative_pool_size",
    "field_binding_generated_attractor_contrast_pairs",
    "field_binding_rank_metric_pairs",
    "field_binding_rank_metric_tokens",
    "field_binding_logit_margin_mean",
    "field_binding_positive_token_fraction",
    "field_binding_margin_satisfied_token_fraction",
    "field_binding_exact_pair_rank_fraction",
    "field_binding_exact_pair_margin_fraction",
    "field_binding_weight",
    "field_binding_margin",
]

FIELD_BINDING_CONTRAST_CONFIG_COLUMNS = [
    "field_binding_config_weight",
    "field_binding_config_start_after_steps",
    "field_binding_config_every_steps",
    "field_binding_config_rank_metric_every_steps",
    "field_binding_config_pair_weight",
    "field_binding_config_replay_capacity",
    "field_binding_config_expected_update_steps",
    "field_binding_config_expected_rank_metric_steps",
]

GENERATED_ATTRACTOR_TELEMETRY_FILE = "events/ruliad_generated_attractor_replay.jsonl"
GENERATED_ATTRACTOR_METRIC_COLUMNS = [
    "generated_attractor_observed_rows",
    "generated_attractor_recorded_rows",
    "generated_attractor_selected_candidate_rows",
    "generated_attractor_selected_field_binding_pairs",
    "generated_attractor_replay_pool_size",
    "generated_attractor_active_count",
    "generated_attractor_active_observation_count",
    "generated_attractor_distinct_answer_count",
    "generated_attractor_dominant_answer_count",
    "generated_attractor_dominant_answer_fraction",
]

GENERATED_ATTRACTOR_CONFIG_COLUMNS = [
    "generated_attractor_config_capacity",
    "generated_attractor_config_min_count",
    "generated_attractor_config_max_candidates",
    "generated_attractor_config_min_distinct_answers",
    "generated_attractor_config_max_dominant_fraction",
]

STRUCTURED_RECOVERY_TELEMETRY_FILE = "events/ruliad_structured_recovery.jsonl"
STRUCTURED_RECOVERY_METRIC_COLUMNS = [
    "recovery_sample_groups",
    "recovery_rows",
    "recovery_field_negative_rows",
    "recovery_template_negative_rows",
    "recovery_schema_negative_rows",
    "recovery_policy_batch_present_fraction",
    "recovery_missing_policy_batch_count",
    "recovery_weight",
    "recovery_max_completion_tokens",
]

STRUCTURED_RECOVERY_CONFIG_COLUMNS = [
    "recovery_config_weight",
    "recovery_config_start_after_steps",
    "recovery_config_every_steps",
    "recovery_config_negative_count",
    "recovery_config_template_negative_count",
    "recovery_config_schema_negative_count",
    "recovery_config_max_completion_tokens",
    "recovery_config_expected_update_steps",
]

ANSWER_CONTRACT_TELEMETRY_FILE = "events/ruliad_answer_contract.jsonl"
ANSWER_CONTRACT_METRIC_COLUMNS = [
    "answer_contract_sample_groups",
    "answer_contract_prompt_schema_sample_groups",
    "answer_contract_oracle_rows",
    "answer_contract_prompt_schema_rows",
    "answer_contract_tokens",
    "answer_contract_prompt_schema_value_tokens",
    "answer_contract_schema_tokens",
    "answer_contract_schema_start_tokens",
    "answer_contract_value_tokens",
    "answer_contract_other_tokens",
    "answer_contract_premature_close_tokens",
    "answer_contract_policy_batch_present_fraction",
    "answer_contract_missing_policy_batch_count",
    "answer_contract_weight",
    "answer_contract_premature_close_unlikelihood_weight",
    "answer_contract_max_completion_tokens",
    "answer_contract_max_rows_per_step",
    "answer_contract_prompt_schema_max_rows_per_step",
]

ANSWER_CONTRACT_CONFIG_COLUMNS = [
    "answer_contract_config_weight",
    "answer_contract_config_premature_close_unlikelihood_weight",
    "answer_contract_config_schema_start_token_weight",
    "answer_contract_config_prompt_schema_value_weight",
    "answer_contract_config_start_after_steps",
    "answer_contract_config_every_steps",
    "answer_contract_config_max_completion_tokens",
    "answer_contract_config_max_rows_per_step",
    "answer_contract_config_prompt_schema_max_rows_per_step",
    "answer_contract_config_expected_update_steps",
]

VERIFIER_ROLLOUT_TELEMETRY_FILE = "events/ruliad_verifier_rollout_imitation.jsonl"
VERIFIER_ROLLOUT_METRIC_COLUMNS = [
    "rollout_imitation_sample_groups",
    "rollout_imitation_generated_rows",
    "rollout_imitation_candidate_rows",
    "rollout_imitation_accepted_rows",
    "rollout_imitation_accepted_imitation_rows",
    "rollout_imitation_accepted_recovery_rows",
    "rollout_imitation_health_gate_passed_fraction",
    "rollout_imitation_verifier_rate",
    "rollout_imitation_schema_wrong_rate",
    "rollout_imitation_malformed_rate",
    "rollout_imitation_verifier_rows",
    "rollout_imitation_semantic_rows",
    "rollout_imitation_partial_rows",
    "rollout_imitation_schema_wrong_rows",
    "rollout_imitation_malformed_rows",
    "rollout_imitation_missing_rows",
    "rollout_imitation_field_accuracy_mean",
    "rollout_imitation_partial_progress_mean",
    "rollout_imitation_completion_quality_mean",
    "rollout_imitation_weight",
    "rollout_recovery_weight",
    "rollout_imitation_max_completion_tokens",
]

RAW_COMPLETION_FILE = "events/ruliad_completion_samples.jsonl"
RAW_COMPLETION_METRIC_COLUMNS = [
    "raw_completion_rows",
    "raw_completion_family_count",
    "raw_completion_min_family_rows",
    "raw_completion_verifier_rate",
    "raw_completion_semantic_rate",
    "raw_completion_partial_rate",
    "raw_completion_schema_wrong_rate",
    "raw_completion_malformed_rate",
    "raw_completion_missing_rate",
    "raw_completion_field_accuracy_mean",
    "raw_completion_termination_rate",
    "raw_completion_quality_mean",
    "raw_completion_generated_tokens_mean",
    "raw_completion_hash_canary_rate",
    "raw_completion_expected_answer_distinct_fraction",
    "raw_completion_actual_answer_distinct_fraction",
    "raw_completion_actual_answer_dominant_fraction",
    "raw_completion_expected_field_value_distinct_fraction",
    "raw_completion_actual_field_value_distinct_fraction",
    "raw_completion_field_value_distinct_ratio",
    "raw_completion_actual_field_value_dominant_fraction",
    "raw_completion_actual_field_value_entropy_bits",
    "raw_completion_status_entropy_bits",
    "raw_completion_dominant_status_fraction",
    "raw_completion_worst_family_verifier_rate",
    "raw_completion_worst_family_partial_rate",
    "raw_completion_worst_family_field_accuracy",
    "raw_completion_worst_family_completion_quality",
    "raw_completion_max_family_schema_wrong_rate",
    "raw_completion_max_family_malformed_rate",
    "raw_completion_max_family_schema_key_mismatch_rate",
    "raw_completion_max_family_answer_dominant_fraction",
]
PROMPT_SCHEMA_COMPLETION_METRIC_COLUMNS = [
    "prompt_schema_completion_rows",
    "prompt_schema_completion_verifier_rate",
    "prompt_schema_completion_semantic_rate",
    "prompt_schema_completion_partial_rate",
    "prompt_schema_completion_schema_wrong_rate",
    "prompt_schema_completion_malformed_rate",
    "prompt_schema_completion_missing_rate",
    "prompt_schema_completion_field_accuracy_mean",
    "prompt_schema_completion_termination_rate",
    "prompt_schema_completion_quality_mean",
    "prompt_schema_completion_generated_tokens_mean",
    "prompt_schema_completion_hash_canary_rate",
    "prompt_schema_completion_expected_answer_distinct_fraction",
    "prompt_schema_completion_actual_answer_distinct_fraction",
    "prompt_schema_completion_actual_answer_dominant_fraction",
    "prompt_schema_completion_expected_field_value_distinct_fraction",
    "prompt_schema_completion_actual_field_value_distinct_fraction",
    "prompt_schema_completion_field_value_distinct_ratio",
    "prompt_schema_completion_actual_field_value_dominant_fraction",
    "prompt_schema_completion_actual_field_value_entropy_bits",
    "prompt_schema_completion_status_entropy_bits",
    "prompt_schema_completion_dominant_status_fraction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Output directory from ruliad_promotion_matrix.sh")
    parser.add_argument("--out-dir", default=None, help="Analysis output directory")
    parser.add_argument("--baseline-arm", default="jepa", help="Control arm name")
    parser.add_argument("--max-valid-ce-delta", type=float, default=0.15)
    parser.add_argument("--max-source-difficulty-delta", type=float, default=0.75)
    parser.add_argument("--max-verifier-regression", type=float, default=0.03125)
    parser.add_argument("--max-schema-wrong-delta", type=float, default=0.10)
    parser.add_argument("--max-malformed-delta", type=float, default=0.05)
    parser.add_argument("--max-completion-regression", type=float, default=0.10)
    parser.add_argument("--max-answer-field-regression", type=float, default=0.05)
    parser.add_argument("--max-answer-termination-regression", type=float, default=0.10)
    parser.add_argument(
        "--min-mature-verifier-rate",
        type=float,
        default=0.03125,
        help=(
            "Minimum verifier rate required once a run is mature. This is a "
            "floor, not a promotion target; it rejects verifier-zero dynamics."
        ),
    )
    parser.add_argument(
        "--min-mature-semantic-rate",
        type=float,
        default=0.03125,
        help="Minimum semantic rate required once a run is mature.",
    )
    parser.add_argument(
        "--min-mature-partial-rate",
        type=float,
        default=0.05,
        help="Minimum partial-credit rate required once a run is mature.",
    )
    parser.add_argument(
        "--min-mature-answer-field-rate",
        type=float,
        default=0.10,
        help="Minimum answer-field accuracy required once a run is mature.",
    )
    parser.add_argument("--min-completion-distinct-2", type=float, default=0.20)
    parser.add_argument("--max-completion-period", type=float, default=0.70)
    parser.add_argument("--max-completion-repetition", type=float, default=0.70)
    parser.add_argument("--min-output-entropy", type=float, default=0.25)
    parser.add_argument("--min-output-distinct-2", type=float, default=0.10)
    parser.add_argument("--min-throughput-ratio", type=float, default=0.85)
    parser.add_argument(
        "--max-peak-memory-ratio",
        type=float,
        default=1.50,
        help=(
            "Reject mature candidates whose peak memory exceeds this multiple of "
            "the baseline unless raw verifier improves enough to justify it."
        ),
    )
    parser.add_argument(
        "--min-raw-verifier-gain-for-memory-regression",
        type=float,
        default=0.03125,
        help="Minimum raw verifier gain required to tolerate a peak-memory regression.",
    )
    parser.add_argument("--max-capability-score-drop", type=float, default=1.0)
    parser.add_argument("--max-verifier-drop-from-best", type=float, default=0.125)
    parser.add_argument("--max-completion-drop-from-best", type=float, default=0.30)
    parser.add_argument("--max-extra-step-verifier-drop", type=float, default=0.125)
    parser.add_argument("--max-extra-step-completion-drop", type=float, default=0.25)
    parser.add_argument("--max-extra-step-malformed-increase", type=float, default=0.25)
    parser.add_argument("--max-latent-eval-ce-delta", type=float, default=1.0)
    parser.add_argument("--max-latent-eval-ce-violation", type=float, default=0.25)
    parser.add_argument("--min-latent-eval-entropy", type=float, default=0.10)
    parser.add_argument("--max-latent-eval-delta-rms", type=float, default=16.0)
    parser.add_argument("--max-latent-extra-eval-ce-delta", type=float, default=1.0)
    parser.add_argument("--max-latent-extra-eval-ce-violation", type=float, default=0.25)
    parser.add_argument("--min-latent-extra-eval-entropy", type=float, default=0.10)
    parser.add_argument("--max-latent-extra-eval-delta-rms", type=float, default=16.0)
    parser.add_argument("--max-capability-lagging-buckets", type=float, default=8.0)
    parser.add_argument("--max-recovery-control-fraction", type=float, default=0.50)
    parser.add_argument("--max-policy-advantage-clip-fraction", type=float, default=0.95)
    parser.add_argument("--min-raw-completion-quality", type=float, default=0.20)
    parser.add_argument("--min-raw-completion-rows", type=float, default=16.0)
    parser.add_argument("--min-raw-completion-verifier-rate", type=float, default=0.03125)
    parser.add_argument("--min-raw-completion-semantic-rate", type=float, default=0.03125)
    parser.add_argument("--min-raw-completion-partial-rate", type=float, default=0.05)
    parser.add_argument("--max-raw-completion-schema-wrong-rate", type=float, default=0.75)
    parser.add_argument("--max-raw-completion-malformed-rate", type=float, default=0.25)
    parser.add_argument("--max-raw-completion-missing-rate", type=float, default=0.25)
    parser.add_argument("--min-raw-completion-answer-distinct", type=float, default=0.20)
    parser.add_argument("--min-raw-completion-field-value-distinct-ratio", type=float, default=0.35)
    parser.add_argument("--max-raw-completion-field-value-dominance", type=float, default=0.85)
    parser.add_argument("--min-raw-completion-family-count", type=float, default=4.0)
    parser.add_argument("--min-raw-completion-family-rows", type=float, default=2.0)
    parser.add_argument("--min-raw-completion-family-verifier-rate", type=float, default=0.03125)
    parser.add_argument("--min-raw-completion-family-partial-rate", type=float, default=0.05)
    parser.add_argument("--min-raw-completion-family-field-rate", type=float, default=0.05)
    parser.add_argument("--min-raw-completion-family-quality", type=float, default=0.20)
    parser.add_argument("--max-raw-completion-family-schema-wrong-rate", type=float, default=0.90)
    parser.add_argument("--max-raw-completion-family-malformed-rate", type=float, default=0.75)
    parser.add_argument("--max-raw-completion-family-schema-key-mismatch", type=float, default=0.50)
    parser.add_argument("--max-raw-completion-family-answer-dominance", type=float, default=0.85)
    parser.add_argument("--min-prompt-schema-completion-quality", type=float, default=0.20)
    parser.add_argument("--min-prompt-schema-completion-rows", type=float, default=16.0)
    parser.add_argument(
        "--min-prompt-schema-completion-verifier-rate", type=float, default=0.03125
    )
    parser.add_argument(
        "--min-prompt-schema-completion-semantic-rate", type=float, default=0.03125
    )
    parser.add_argument("--min-prompt-schema-completion-partial-rate", type=float, default=0.05)
    parser.add_argument(
        "--max-prompt-schema-completion-schema-wrong-rate", type=float, default=0.75
    )
    parser.add_argument(
        "--max-prompt-schema-completion-malformed-rate", type=float, default=0.25
    )
    parser.add_argument(
        "--max-prompt-schema-completion-missing-rate", type=float, default=0.25
    )
    parser.add_argument(
        "--min-prompt-schema-completion-answer-distinct", type=float, default=0.20
    )
    parser.add_argument(
        "--min-prompt-schema-completion-field-value-distinct-ratio",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--max-prompt-schema-completion-field-value-dominance",
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--max-free-run-contract-verifier-gap",
        type=float,
        default=0.125,
        help=(
            "Reject mature arms whose fixed-contract verifier score exceeds the raw "
            "free-run verifier score by more than this amount."
        ),
    )
    parser.add_argument("--min-field-binding-positive-token-fraction", type=float, default=0.55)
    parser.add_argument("--min-field-binding-exact-pair-rank-fraction", type=float, default=0.35)
    parser.add_argument(
        "--min-mature-iters",
        type=int,
        default=1024,
        help=(
            "Minimum per-trial max_iters required before maturity-sensitive gates "
            "can reject or promote an arm. Shorter runs are marked hold."
        ),
    )
    return parser.parse_args()


def finite(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.upper() in {"[N/A]", "N/A", "NA", "NONE"}:
            return None
        value = stripped.split()[0].rstrip("%,")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def fmt(value: Any) -> str:
    number = finite(value)
    if number is None:
        return "" if value is None else str(value)
    if abs(number) >= 1000.0:
        return f"{number:.3f}"
    if abs(number) >= 10.0:
        return f"{number:.4f}"
    return f"{number:.6f}"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def read_policy_telemetry(run_dir: str | None) -> dict[str, float | int | None]:
    if not run_dir:
        return {column: None for column in POLICY_METRIC_COLUMNS}
    path = Path(run_dir) / POLICY_TELEMETRY_FILE
    if not path.exists():
        return {column: None for column in POLICY_METRIC_COLUMNS}
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not records:
        return {column: None for column in POLICY_METRIC_COLUMNS}

    def last_sum(key: str) -> float | None:
        values = [finite(record.get(key)) for record in records]
        clean = [value for value in values if value is not None]
        return sum(clean) if clean else None

    def weighted_mean(key: str, weight_key: str = "completion_rows") -> float | None:
        total = 0.0
        weight_sum = 0.0
        for record in records:
            value = finite(record.get(key))
            weight = finite(record.get(weight_key))
            if value is None or weight is None or weight <= 0.0:
                continue
            total += value * weight
            weight_sum += weight
        return total / weight_sum if weight_sum > 0.0 else None

    def min_value(key: str) -> float | None:
        clean = [value for value in (finite(record.get(key)) for record in records) if value is not None]
        return min(clean) if clean else None

    def max_value(key: str) -> float | None:
        clean = [value for value in (finite(record.get(key)) for record in records) if value is not None]
        return max(clean) if clean else None

    return {
        "policy_sample_groups": last_sum("sample_groups"),
        "policy_completion_rows": last_sum("completion_rows"),
        "policy_oracle_sample_groups": last_sum("oracle_sample_groups"),
        "policy_oracle_completion_rows": last_sum("oracle_completion_rows"),
        "policy_oracle_truncated_completion_rows": last_sum("oracle_truncated_completion_rows"),
        "policy_structured_negative_completion_rows": last_sum("structured_negative_completion_rows"),
        "policy_generated_attractor_completion_rows": last_sum("generated_attractor_completion_rows"),
        "policy_gated_sample_groups": last_sum("gated_sample_groups"),
        "policy_gated_completion_rows": last_sum("gated_completion_rows"),
        "policy_scalarization_count": last_sum("scalarization_count"),
        "policy_reward_mean": weighted_mean("reward_mean"),
        "policy_reward_std": weighted_mean("reward_std"),
        "policy_reward_min": min_value("reward_min"),
        "policy_reward_max": max_value("reward_max"),
        "policy_advantage_mean": weighted_mean("advantage_mean"),
        "policy_advantage_std": weighted_mean("advantage_std"),
        "policy_advantage_clip_fraction": weighted_mean("advantage_clip_fraction"),
        "policy_update_applied_fraction": mean([
            1.0 if record.get("policy_update_applied", True) else 0.0
            for record in records
        ]),
        "policy_update_skipped_count": sum(
            0 if record.get("policy_update_applied", True) else 1
            for record in records
        ),
        "policy_vector_verifier_match_mean": weighted_mean("vector_verifier_match_mean", "vector_sample_count"),
        "policy_vector_semantic_match_mean": weighted_mean("vector_semantic_match_mean", "vector_sample_count"),
        "policy_vector_partial_progress_mean": weighted_mean("vector_partial_progress_mean", "vector_sample_count"),
        "policy_vector_field_accuracy_mean": weighted_mean("vector_field_accuracy_mean", "vector_sample_count"),
        "policy_vector_certificate_prefix_mean": weighted_mean("vector_certificate_prefix_mean", "vector_sample_count"),
        "policy_vector_compactness_mean": weighted_mean("vector_compactness_mean", "vector_sample_count"),
        "policy_vector_schema_quality_mean": weighted_mean("vector_schema_quality_mean", "vector_sample_count"),
        "policy_vector_hash_safety_mean": weighted_mean("vector_hash_safety_mean", "vector_sample_count"),
        "policy_vector_answer_termination_mean": weighted_mean("vector_answer_termination_mean", "vector_sample_count"),
        "policy_vector_completion_health_mean": weighted_mean("vector_completion_health_mean", "vector_sample_count"),
        "policy_vpo_dominant_compactness": last_sum("vpo_scalarization_dominant_compactness"),
        "policy_vpo_dominant_schema_quality": last_sum("vpo_scalarization_dominant_schema_quality"),
        "policy_vpo_dominant_completion_health": last_sum("vpo_scalarization_dominant_completion_health"),
    }


def read_structured_contrast_telemetry(run_dir: str | None) -> dict[str, float | int | None]:
    if not run_dir:
        return {column: None for column in STRUCTURED_CONTRAST_METRIC_COLUMNS}
    path = Path(run_dir) / STRUCTURED_CONTRAST_TELEMETRY_FILE
    if not path.exists():
        return {column: None for column in STRUCTURED_CONTRAST_METRIC_COLUMNS}
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not records:
        return {column: None for column in STRUCTURED_CONTRAST_METRIC_COLUMNS}

    def sum_value(key: str) -> float | None:
        clean = [
            value
            for value in (finite(record.get(key)) for record in records)
            if value is not None
        ]
        return sum(clean) if clean else None

    def last_value(key: str) -> float | None:
        for record in reversed(records):
            value = finite(record.get(key))
            if value is not None:
                return value
        return None

    return {
        "contrast_sample_groups": sum_value("sample_groups"),
        "contrast_oracle_completion_rows": sum_value("oracle_completion_rows"),
        "contrast_field_negative_completion_rows": sum_value("field_negative_completion_rows"),
        "contrast_template_negative_completion_rows": sum_value("template_negative_completion_rows"),
        "contrast_generated_attractor_negative_completion_rows": sum_value(
            "generated_attractor_negative_completion_rows"
        ),
        "contrast_pairs": sum_value("contrast_pairs"),
        "contrast_discriminative_tokens": sum_value("contrast_discriminative_tokens"),
        "contrast_weight": last_value("structured_contrast_weight"),
        "contrast_margin": last_value("structured_contrast_margin"),
    }


def read_structured_contrast_config(run_dir: str | None) -> dict[str, float | int | None]:
    if not run_dir:
        return {column: None for column in STRUCTURED_CONTRAST_CONFIG_COLUMNS}
    path = Path(run_dir) / "training_config.json"
    if not path.exists():
        return {column: None for column in STRUCTURED_CONTRAST_CONFIG_COLUMNS}
    try:
        config = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {column: None for column in STRUCTURED_CONTRAST_CONFIG_COLUMNS}
    training = config.get("training")
    if not isinstance(training, dict):
        return {column: None for column in STRUCTURED_CONTRAST_CONFIG_COLUMNS}
    supervision = training.get("ruliad_supervision")
    if not isinstance(supervision, dict):
        return {column: None for column in STRUCTURED_CONTRAST_CONFIG_COLUMNS}
    verifier = supervision.get("verifier_reward")
    if not isinstance(verifier, dict):
        return {column: None for column in STRUCTURED_CONTRAST_CONFIG_COLUMNS}
    enabled = bool(verifier.get("enabled", False))
    weight = finite(verifier.get("structured_contrast_weight")) or 0.0
    every_steps = int(finite(verifier.get("structured_contrast_every_steps")) or 0)
    start_after_steps = int(finite(verifier.get("structured_contrast_start_after_steps")) or 0)
    max_iters = int(finite(training.get("max_iters")) or 0)
    expected_updates = 0
    if enabled and weight > 0.0 and every_steps > 0 and max_iters > start_after_steps:
        expected_updates = ((max_iters - 1 - start_after_steps) // every_steps) + 1
    return {
        "contrast_config_weight": weight,
        "contrast_config_start_after_steps": start_after_steps,
        "contrast_config_every_steps": every_steps,
        "contrast_config_expected_update_steps": expected_updates,
    }


def read_field_binding_contrast_telemetry(run_dir: str | None) -> dict[str, float | int | None]:
    if not run_dir:
        return {column: None for column in FIELD_BINDING_CONTRAST_METRIC_COLUMNS}
    path = Path(run_dir) / FIELD_BINDING_CONTRAST_TELEMETRY_FILE
    if not path.exists():
        return {column: None for column in FIELD_BINDING_CONTRAST_METRIC_COLUMNS}
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not records:
        return {column: None for column in FIELD_BINDING_CONTRAST_METRIC_COLUMNS}

    def sum_value(key: str) -> float | None:
        clean = [
            value
            for value in (finite(record.get(key)) for record in records)
            if value is not None
        ]
        return sum(clean) if clean else None

    def last_value(key: str) -> float | None:
        for record in reversed(records):
            value = finite(record.get(key))
            if value is not None:
                return value
        return None

    def weighted_mean(value_key: str, weight_key: str) -> float | None:
        total = 0.0
        weight_total = 0.0
        for record in records:
            value = finite(record.get(value_key))
            weight = finite(record.get(weight_key))
            if value is None or weight is None or weight <= 0.0:
                continue
            total += value * weight
            weight_total += weight
        return total / weight_total if weight_total > 0.0 else None

    return {
        "field_binding_sample_groups": sum_value("sample_groups"),
        "field_binding_prompt_pairs": sum_value("prompt_pairs"),
        "field_binding_contrast_pairs": sum_value("contrast_pairs"),
        "field_binding_candidate_pairs": sum_value("candidate_pairs"),
        "field_binding_discriminative_tokens": sum_value("contrast_discriminative_tokens"),
        "field_binding_negative_pool_size": last_value("negative_pool_size"),
        "field_binding_replay_pool_size": last_value("replay_pool_size"),
        "field_binding_replay_contrast_pairs": sum_value("replay_contrast_pairs"),
        "field_binding_generated_attractor_pool_size": last_value("generated_attractor_pool_size"),
        "field_binding_generated_attractor_negative_pool_size": last_value(
            "generated_attractor_negative_pool_size"
        ),
        "field_binding_generated_attractor_contrast_pairs": sum_value(
            "generated_attractor_contrast_pairs"
        ),
        "field_binding_rank_metric_pairs": sum_value("rank_metric_pairs"),
        "field_binding_rank_metric_tokens": sum_value("rank_metric_tokens"),
        "field_binding_logit_margin_mean": weighted_mean(
            "logit_margin_mean", "rank_metric_tokens"
        ),
        "field_binding_positive_token_fraction": weighted_mean(
            "positive_token_fraction", "rank_metric_tokens"
        ),
        "field_binding_margin_satisfied_token_fraction": weighted_mean(
            "margin_satisfied_token_fraction", "rank_metric_tokens"
        ),
        "field_binding_exact_pair_rank_fraction": weighted_mean(
            "exact_pair_rank_fraction", "rank_metric_pairs"
        ),
        "field_binding_exact_pair_margin_fraction": weighted_mean(
            "exact_pair_margin_fraction", "rank_metric_pairs"
        ),
        "field_binding_weight": last_value("field_binding_contrast_weight"),
        "field_binding_margin": last_value("field_binding_contrast_margin"),
    }


def read_field_binding_contrast_config(run_dir: str | None) -> dict[str, float | int | None]:
    if not run_dir:
        return {column: None for column in FIELD_BINDING_CONTRAST_CONFIG_COLUMNS}
    path = Path(run_dir) / "training_config.json"
    if not path.exists():
        return {column: None for column in FIELD_BINDING_CONTRAST_CONFIG_COLUMNS}
    try:
        config = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {column: None for column in FIELD_BINDING_CONTRAST_CONFIG_COLUMNS}
    training = config.get("training")
    if not isinstance(training, dict):
        return {column: None for column in FIELD_BINDING_CONTRAST_CONFIG_COLUMNS}
    supervision = training.get("ruliad_supervision")
    if not isinstance(supervision, dict):
        return {column: None for column in FIELD_BINDING_CONTRAST_CONFIG_COLUMNS}
    verifier = supervision.get("verifier_reward")
    if not isinstance(verifier, dict):
        return {column: None for column in FIELD_BINDING_CONTRAST_CONFIG_COLUMNS}
    enabled = bool(verifier.get("enabled", False))
    weight = finite(verifier.get("field_binding_contrast_weight")) or 0.0
    every_steps = int(finite(verifier.get("field_binding_contrast_every_steps")) or 0)
    start_after_steps = int(finite(verifier.get("field_binding_contrast_start_after_steps")) or 0)
    rank_metric_every_steps = int(
        finite(verifier.get("field_binding_contrast_rank_metric_every_steps")) or 0
    )
    pair_weight = finite(verifier.get("field_binding_contrast_pair_weight")) or 0.0
    replay_capacity = int(finite(verifier.get("field_binding_contrast_replay_capacity")) or 0)
    max_iters = int(finite(training.get("max_iters")) or 0)
    expected_updates = 0
    if enabled and weight > 0.0 and every_steps > 0 and max_iters > start_after_steps:
        expected_updates = ((max_iters - 1 - start_after_steps) // every_steps) + 1
    expected_rank_metrics = 0
    if (
        enabled
        and weight > 0.0
        and every_steps > 0
        and rank_metric_every_steps > 0
        and max_iters > start_after_steps
    ):
        for step in range(start_after_steps, max_iters, every_steps):
            if step % rank_metric_every_steps == 0:
                expected_rank_metrics += 1
    return {
        "field_binding_config_weight": weight,
        "field_binding_config_start_after_steps": start_after_steps,
        "field_binding_config_every_steps": every_steps,
        "field_binding_config_rank_metric_every_steps": rank_metric_every_steps,
        "field_binding_config_pair_weight": pair_weight,
        "field_binding_config_replay_capacity": replay_capacity,
        "field_binding_config_expected_update_steps": expected_updates,
        "field_binding_config_expected_rank_metric_steps": expected_rank_metrics,
    }


def read_generated_attractor_config(run_dir: str | None) -> dict[str, float | int | None]:
    if not run_dir:
        return {column: None for column in GENERATED_ATTRACTOR_CONFIG_COLUMNS}
    path = Path(run_dir) / "training_config.json"
    if not path.exists():
        return {column: None for column in GENERATED_ATTRACTOR_CONFIG_COLUMNS}
    try:
        config = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {column: None for column in GENERATED_ATTRACTOR_CONFIG_COLUMNS}
    training = config.get("training")
    if not isinstance(training, dict):
        return {column: None for column in GENERATED_ATTRACTOR_CONFIG_COLUMNS}
    supervision = training.get("ruliad_supervision")
    if not isinstance(supervision, dict):
        return {column: None for column in GENERATED_ATTRACTOR_CONFIG_COLUMNS}
    verifier = supervision.get("verifier_reward")
    if not isinstance(verifier, dict):
        return {column: None for column in GENERATED_ATTRACTOR_CONFIG_COLUMNS}
    return {
        "generated_attractor_config_capacity": int(
            finite(verifier.get("generated_attractor_replay_capacity")) or 0
        ),
        "generated_attractor_config_min_count": int(
            finite(verifier.get("generated_attractor_replay_min_count")) or 0
        ),
        "generated_attractor_config_max_candidates": int(
            finite(verifier.get("generated_attractor_replay_max_candidates")) or 0
        ),
        "generated_attractor_config_min_distinct_answers": int(
            finite(verifier.get("generated_attractor_replay_min_distinct_answers")) or 0
        ),
        "generated_attractor_config_max_dominant_fraction": finite(
            verifier.get("generated_attractor_replay_max_dominant_fraction")
        ),
    }


def read_generated_attractor_telemetry(run_dir: str | None) -> dict[str, float | int | None]:
    if not run_dir:
        return {column: None for column in GENERATED_ATTRACTOR_METRIC_COLUMNS}
    path = Path(run_dir) / GENERATED_ATTRACTOR_TELEMETRY_FILE
    if not path.exists():
        return {column: None for column in GENERATED_ATTRACTOR_METRIC_COLUMNS}
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                records.append(row)
    if not records:
        return {column: None for column in GENERATED_ATTRACTOR_METRIC_COLUMNS}

    def sum_value(key: str) -> float | None:
        clean = [
            value
            for value in (finite(record.get(key)) for record in records)
            if value is not None
        ]
        return sum(clean) if clean else None

    def last_value(key: str) -> float | None:
        for record in reversed(records):
            value = finite(record.get(key))
            if value is not None:
                return value
        return None

    return {
        "generated_attractor_observed_rows": sum_value("observed_completion_rows"),
        "generated_attractor_recorded_rows": sum_value("recorded_attractor_rows"),
        "generated_attractor_selected_candidate_rows": sum_value("selected_candidate_rows"),
        "generated_attractor_selected_field_binding_pairs": sum_value(
            "selected_field_binding_pairs"
        ),
        "generated_attractor_replay_pool_size": last_value("replay_pool_size"),
        "generated_attractor_active_count": last_value("active_attractor_count"),
        "generated_attractor_active_observation_count": last_value(
            "active_observation_count"
        ),
        "generated_attractor_distinct_answer_count": last_value("distinct_answer_count"),
        "generated_attractor_dominant_answer_count": last_value("dominant_answer_count"),
        "generated_attractor_dominant_answer_fraction": last_value(
            "dominant_answer_fraction"
        ),
    }


def read_structured_recovery_config(run_dir: str | None) -> dict[str, float | int | None]:
    if not run_dir:
        return {column: None for column in STRUCTURED_RECOVERY_CONFIG_COLUMNS}
    path = Path(run_dir) / "training_config.json"
    if not path.exists():
        return {column: None for column in STRUCTURED_RECOVERY_CONFIG_COLUMNS}
    try:
        config = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {column: None for column in STRUCTURED_RECOVERY_CONFIG_COLUMNS}
    training = config.get("training")
    if not isinstance(training, dict):
        return {column: None for column in STRUCTURED_RECOVERY_CONFIG_COLUMNS}
    supervision = training.get("ruliad_supervision")
    if not isinstance(supervision, dict):
        return {column: None for column in STRUCTURED_RECOVERY_CONFIG_COLUMNS}
    denoising = supervision.get("answer_denoising")
    if not isinstance(denoising, dict):
        return {column: None for column in STRUCTURED_RECOVERY_CONFIG_COLUMNS}

    enabled = bool(denoising.get("enabled", False))
    weight = finite(denoising.get("structured_recovery_weight")) or 0.0
    every_steps = int(finite(denoising.get("structured_recovery_every_steps")) or 0)
    start_after_steps = int(finite(denoising.get("structured_recovery_start_after_steps")) or 0)
    negative_count = int(finite(denoising.get("structured_recovery_negative_count")) or 0)
    template_negative_count = int(
        finite(denoising.get("structured_recovery_template_negative_count")) or 0
    )
    schema_negative_count = int(
        finite(denoising.get("structured_recovery_schema_negative_count")) or 0
    )
    max_completion_tokens = int(
        finite(denoising.get("structured_recovery_max_completion_tokens")) or 0
    )
    max_iters = int(finite(training.get("max_iters")) or 0)
    expected_updates = 0
    if enabled and weight > 0.0 and every_steps > 0 and max_iters > start_after_steps:
        expected_updates = ((max_iters - 1 - start_after_steps) // every_steps) + 1
    return {
        "recovery_config_weight": weight,
        "recovery_config_start_after_steps": start_after_steps,
        "recovery_config_every_steps": every_steps,
        "recovery_config_negative_count": negative_count,
        "recovery_config_template_negative_count": template_negative_count,
        "recovery_config_schema_negative_count": schema_negative_count,
        "recovery_config_max_completion_tokens": max_completion_tokens,
        "recovery_config_expected_update_steps": expected_updates,
    }


def read_structured_recovery_telemetry(run_dir: str | None) -> dict[str, float | int | None]:
    if not run_dir:
        return {column: None for column in STRUCTURED_RECOVERY_METRIC_COLUMNS}
    path = Path(run_dir) / STRUCTURED_RECOVERY_TELEMETRY_FILE
    if not path.exists():
        return {column: None for column in STRUCTURED_RECOVERY_METRIC_COLUMNS}
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                records.append(row)
    if not records:
        return {column: None for column in STRUCTURED_RECOVERY_METRIC_COLUMNS}

    def sum_value(key: str) -> float | None:
        clean = [
            value
            for value in (finite(record.get(key)) for record in records)
            if value is not None
        ]
        return sum(clean) if clean else None

    def last_value(key: str) -> float | None:
        for record in reversed(records):
            value = finite(record.get(key))
            if value is not None:
                return value
        return None

    policy_present = [
        1.0 if record.get("policy_batch_present") else 0.0
        for record in records
        if isinstance(record.get("policy_batch_present"), bool)
    ]
    missing_policy_count = sum(
        1 for record in records if record.get("skip_reason") == "missing_policy_batch"
    )

    return {
        "recovery_sample_groups": sum_value("sample_groups"),
        "recovery_rows": sum_value("recovery_rows"),
        "recovery_field_negative_rows": sum_value("field_negative_recovery_rows"),
        "recovery_template_negative_rows": sum_value("template_negative_recovery_rows"),
        "recovery_schema_negative_rows": sum_value("schema_negative_recovery_rows"),
        "recovery_policy_batch_present_fraction": mean(policy_present),
        "recovery_missing_policy_batch_count": missing_policy_count,
        "recovery_weight": last_value("structured_recovery_weight"),
        "recovery_max_completion_tokens": last_value("structured_recovery_max_completion_tokens"),
    }


def read_answer_contract_config(run_dir: str | None) -> dict[str, float | int | None]:
    if not run_dir:
        return {column: None for column in ANSWER_CONTRACT_CONFIG_COLUMNS}
    path = Path(run_dir) / "training_config.json"
    if not path.exists():
        return {column: None for column in ANSWER_CONTRACT_CONFIG_COLUMNS}
    try:
        config = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {column: None for column in ANSWER_CONTRACT_CONFIG_COLUMNS}
    training = config.get("training")
    if not isinstance(training, dict):
        return {column: None for column in ANSWER_CONTRACT_CONFIG_COLUMNS}
    supervision = training.get("ruliad_supervision")
    if not isinstance(supervision, dict):
        return {column: None for column in ANSWER_CONTRACT_CONFIG_COLUMNS}
    contract = supervision.get("answer_contract")
    if not isinstance(contract, dict):
        return {column: None for column in ANSWER_CONTRACT_CONFIG_COLUMNS}
    enabled = bool(contract.get("enabled", False))
    weight = finite(contract.get("weight")) or 0.0
    premature_close_weight = (
        finite(contract.get("premature_close_unlikelihood_weight")) or 0.0
    )
    every_steps = int(finite(contract.get("every_steps")) or 0)
    start_after_steps = int(finite(contract.get("start_after_steps")) or 0)
    max_completion_tokens = int(finite(contract.get("max_completion_tokens")) or 0)
    max_rows_per_step = int(finite(contract.get("max_rows_per_step")) or 0)
    prompt_schema_max_rows_per_step = int(
        finite(contract.get("prompt_schema_max_rows_per_step")) or 0
    )
    if prompt_schema_max_rows_per_step == 0:
        prompt_schema_max_rows_per_step = max_rows_per_step
    schema_start_weight = finite(contract.get("schema_start_token_weight")) or 0.0
    prompt_schema_value_weight = (
        finite(contract.get("prompt_schema_value_weight")) or 0.0
    )
    max_iters = int(finite(training.get("max_iters")) or 0)
    expected_updates = 0
    if enabled and weight > 0.0 and every_steps > 0 and max_iters > start_after_steps:
        expected_updates = ((max_iters - 1 - start_after_steps) // every_steps) + 1
    return {
        "answer_contract_config_weight": weight,
        "answer_contract_config_premature_close_unlikelihood_weight": premature_close_weight,
        "answer_contract_config_schema_start_token_weight": schema_start_weight,
        "answer_contract_config_prompt_schema_value_weight": prompt_schema_value_weight,
        "answer_contract_config_start_after_steps": start_after_steps,
        "answer_contract_config_every_steps": every_steps,
        "answer_contract_config_max_completion_tokens": max_completion_tokens,
        "answer_contract_config_max_rows_per_step": max_rows_per_step,
        "answer_contract_config_prompt_schema_max_rows_per_step": prompt_schema_max_rows_per_step,
        "answer_contract_config_expected_update_steps": expected_updates,
    }


def read_answer_contract_telemetry(run_dir: str | None) -> dict[str, float | int | None]:
    if not run_dir:
        return {column: None for column in ANSWER_CONTRACT_METRIC_COLUMNS}
    path = Path(run_dir) / ANSWER_CONTRACT_TELEMETRY_FILE
    if not path.exists():
        return {column: None for column in ANSWER_CONTRACT_METRIC_COLUMNS}
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                records.append(row)
    if not records:
        return {column: None for column in ANSWER_CONTRACT_METRIC_COLUMNS}

    def sum_value(key: str) -> float | None:
        clean = [
            value
            for value in (finite(record.get(key)) for record in records)
            if value is not None
        ]
        return sum(clean) if clean else None

    def last_value(key: str) -> float | None:
        for record in reversed(records):
            value = finite(record.get(key))
            if value is not None:
                return value
        return None

    policy_present = [
        1.0 if record.get("policy_batch_present") else 0.0
        for record in records
        if isinstance(record.get("policy_batch_present"), bool)
    ]
    missing_policy_count = sum(
        1 for record in records if record.get("skip_reason") == "missing_policy_batch"
    )

    return {
        "answer_contract_sample_groups": sum_value("sample_groups"),
        "answer_contract_prompt_schema_sample_groups": sum_value(
            "prompt_schema_sample_groups"
        ),
        "answer_contract_oracle_rows": sum_value("oracle_rows"),
        "answer_contract_prompt_schema_rows": sum_value("prompt_schema_rows"),
        "answer_contract_tokens": sum_value("contract_tokens"),
        "answer_contract_prompt_schema_value_tokens": sum_value(
            "prompt_schema_value_tokens"
        ),
        "answer_contract_schema_tokens": sum_value("schema_tokens"),
        "answer_contract_schema_start_tokens": sum_value("schema_start_tokens"),
        "answer_contract_value_tokens": sum_value("value_tokens"),
        "answer_contract_other_tokens": sum_value("other_tokens"),
        "answer_contract_premature_close_tokens": sum_value("premature_close_tokens"),
        "answer_contract_policy_batch_present_fraction": mean(policy_present),
        "answer_contract_missing_policy_batch_count": missing_policy_count,
        "answer_contract_weight": last_value("answer_contract_weight"),
        "answer_contract_premature_close_unlikelihood_weight": last_value(
            "premature_close_unlikelihood_weight"
        ),
        "answer_contract_max_completion_tokens": last_value("max_completion_tokens"),
        "answer_contract_max_rows_per_step": last_value("max_rows_per_step"),
        "answer_contract_prompt_schema_max_rows_per_step": last_value(
            "prompt_schema_max_rows_per_step"
        ),
    }


def read_verifier_rollout_telemetry(run_dir: str | None) -> dict[str, float | int | None]:
    if not run_dir:
        return {column: None for column in VERIFIER_ROLLOUT_METRIC_COLUMNS}
    path = Path(run_dir) / VERIFIER_ROLLOUT_TELEMETRY_FILE
    if not path.exists():
        return {column: None for column in VERIFIER_ROLLOUT_METRIC_COLUMNS}
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not records:
        return {column: None for column in VERIFIER_ROLLOUT_METRIC_COLUMNS}

    def sum_value(key: str) -> float | None:
        clean = [
            value
            for value in (finite(record.get(key)) for record in records)
            if value is not None
        ]
        return sum(clean) if clean else None

    def last_value(key: str) -> float | None:
        for record in reversed(records):
            value = finite(record.get(key))
            if value is not None:
                return value
        return None

    def weighted_mean(key: str, weight_key: str = "generated_completion_rows") -> float | None:
        total = 0.0
        weight_sum = 0.0
        for record in records:
            value = finite(record.get(key))
            weight = finite(record.get(weight_key))
            if value is None or weight is None or weight <= 0.0:
                continue
            total += value * weight
            weight_sum += weight
        return total / weight_sum if weight_sum > 0.0 else None

    def bool_mean(key: str) -> float | None:
        values: list[float] = []
        for record in records:
            value = record.get(key)
            if isinstance(value, bool):
                values.append(1.0 if value else 0.0)
        return sum(values) / len(values) if values else None

    def weighted_ppm_rate(key: str) -> float | None:
        value = weighted_mean(key)
        return value / 1_000_000.0 if value is not None else None

    return {
        "rollout_imitation_sample_groups": sum_value("sample_groups"),
        "rollout_imitation_generated_rows": sum_value("generated_completion_rows"),
        "rollout_imitation_candidate_rows": sum_value("candidate_completion_rows"),
        "rollout_imitation_accepted_rows": sum_value("accepted_completion_rows"),
        "rollout_imitation_accepted_imitation_rows": sum_value("accepted_imitation_rows"),
        "rollout_imitation_accepted_recovery_rows": sum_value("accepted_recovery_rows"),
        "rollout_imitation_health_gate_passed_fraction": bool_mean("health_gate_passed"),
        "rollout_imitation_verifier_rate": weighted_ppm_rate("verifier_rate_ppm"),
        "rollout_imitation_schema_wrong_rate": weighted_ppm_rate("schema_wrong_rate_ppm"),
        "rollout_imitation_malformed_rate": weighted_ppm_rate("malformed_rate_ppm"),
        "rollout_imitation_verifier_rows": sum_value("verifier_match_rows"),
        "rollout_imitation_semantic_rows": sum_value("semantic_match_rows"),
        "rollout_imitation_partial_rows": sum_value("partial_rows"),
        "rollout_imitation_schema_wrong_rows": sum_value("schema_wrong_rows"),
        "rollout_imitation_malformed_rows": sum_value("malformed_rows"),
        "rollout_imitation_missing_rows": sum_value("missing_rows"),
        "rollout_imitation_field_accuracy_mean": weighted_mean("field_accuracy_mean"),
        "rollout_imitation_partial_progress_mean": weighted_mean("partial_progress_mean"),
        "rollout_imitation_completion_quality_mean": weighted_mean("completion_quality_mean"),
        "rollout_imitation_weight": last_value("rollout_imitation_weight"),
        "rollout_recovery_weight": last_value("rollout_recovery_weight"),
        "rollout_imitation_max_completion_tokens": last_value("max_completion_tokens"),
    }


def read_completion_samples_for_probe(
    run_dir: str | None,
    *,
    probe_name: str,
    prefix: str,
    columns: list[str],
    fallback_to_any_probe: bool = False,
    include_family_metrics: bool = False,
) -> dict[str, float | int | None]:
    empty = {column: None for column in columns}
    if not run_dir:
        return empty
    path = Path(run_dir) / RAW_COMPLETION_FILE
    if not path.exists():
        return empty

    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            records.append(row)
    if not records:
        return empty

    base_records = [row for row in records if row.get("probe_name") == probe_name]
    selected = latest_probe_records(base_records or records if fallback_to_any_probe else base_records)
    if not selected:
        return empty

    def bool_rate_for(rows: list[dict[str, Any]], key: str) -> float | None:
        values = [row.get(key) for row in rows if isinstance(row.get(key), bool)]
        return mean([1.0 if value else 0.0 for value in values])

    def numeric_mean_for(
        rows: list[dict[str, Any]], key: str, scale: float = 1.0
    ) -> float | None:
        clean = [value for value in (finite(row.get(key)) for row in rows) if value is not None]
        return mean([value / scale for value in clean])

    def status_rate_for(rows: list[dict[str, Any]], status: str) -> float:
        return sum(1 for row in rows if row.get("status") == status) / len(rows)

    def field_score(row: dict[str, Any]) -> float | None:
        correct = finite(row.get("correct_field_count"))
        expected = finite(row.get("expected_field_count"))
        if correct is None or expected is None or expected <= 0.0:
            return None
        return max(0.0, min(correct / expected, 1.0))

    def field_score_mean(rows: list[dict[str, Any]]) -> float | None:
        return mean([score for score in (field_score(row) for row in rows) if score is not None])

    def schema_key_mismatch(row: dict[str, Any]) -> float | None:
        expected_keys = set(parse_answer_fields(row.get("expected_answer")))
        if not expected_keys:
            return None
        actual_keys = set(parse_answer_fields(row.get("actual_answer")))
        return 0.0 if actual_keys == expected_keys else 1.0

    def schema_key_mismatch_rate(rows: list[dict[str, Any]]) -> float | None:
        values = [
            value
            for value in (schema_key_mismatch(row) for row in rows)
            if value is not None
        ]
        return mean(values)

    def family_name(row: dict[str, Any]) -> str:
        family = str(row.get("family") or "").strip()
        return family or "<unknown>"

    answers = [
        str(row.get("actual_answer"))
        for row in selected
        if row.get("actual_answer") is not None
    ]
    expected_answers = [
        str(row.get("expected_answer"))
        for row in selected
        if row.get("expected_answer") is not None
    ]
    expected_field_values: list[str] = []
    actual_field_values: list[str] = []
    for row in selected:
        expected_fields = parse_answer_fields(row.get("expected_answer"))
        actual_fields = parse_answer_fields(row.get("actual_answer"))
        if not expected_fields:
            continue
        for key, expected_value in expected_fields.items():
            expected_field_values.append(f"{key}={expected_value}")
            actual_field_values.append(f"{key}={actual_fields.get(key, '<missing>')}")
    expected_field_distinct = (
        len(set(expected_field_values)) / len(expected_field_values)
        if expected_field_values
        else None
    )
    actual_field_distinct = (
        len(set(actual_field_values)) / len(actual_field_values)
        if actual_field_values
        else None
    )
    statuses = [str(row.get("status") or "") for row in selected]

    result: dict[str, float | int | None] = dict(empty)
    result.update(
        {
            f"{prefix}_rows": len(selected),
            f"{prefix}_verifier_rate": bool_rate_for(selected, "verifier_match"),
            f"{prefix}_semantic_rate": bool_rate_for(selected, "semantic_match"),
            f"{prefix}_partial_rate": bool_rate_for(selected, "partial_credit"),
            f"{prefix}_schema_wrong_rate": status_rate_for(selected, "SchemaValidWrong"),
            f"{prefix}_malformed_rate": status_rate_for(selected, "Malformed"),
            f"{prefix}_missing_rate": status_rate_for(selected, "Missing"),
            f"{prefix}_field_accuracy_mean": field_score_mean(selected),
            f"{prefix}_termination_rate": bool_rate_for(selected, "answer_terminated"),
            f"{prefix}_quality_mean": numeric_mean_for(
            selected, "completion_quality_ppm", 1_000_000.0
            ),
            f"{prefix}_generated_tokens_mean": numeric_mean_for(
                selected, "generated_token_count"
            ),
            f"{prefix}_hash_canary_rate": bool_rate_for(selected, "hash_canary"),
            f"{prefix}_expected_answer_distinct_fraction": (
                len(set(expected_answers)) / len(expected_answers) if expected_answers else None
            ),
            f"{prefix}_actual_answer_distinct_fraction": (
                len(set(answers)) / len(answers) if answers else None
            ),
            f"{prefix}_actual_answer_dominant_fraction": dominant_fraction(answers),
            f"{prefix}_expected_field_value_distinct_fraction": expected_field_distinct,
            f"{prefix}_actual_field_value_distinct_fraction": actual_field_distinct,
            f"{prefix}_field_value_distinct_ratio": (
                actual_field_distinct / expected_field_distinct
                if actual_field_distinct is not None
                and expected_field_distinct is not None
                and expected_field_distinct > 0.0
                else None
            ),
            f"{prefix}_actual_field_value_dominant_fraction": dominant_fraction(
                actual_field_values
            ),
            f"{prefix}_actual_field_value_entropy_bits": entropy_bits(actual_field_values),
            f"{prefix}_status_entropy_bits": entropy_bits(statuses),
            f"{prefix}_dominant_status_fraction": dominant_fraction(statuses),
        }
    )

    if include_family_metrics:
        families: dict[str, list[dict[str, Any]]] = {}
        for row in selected:
            families.setdefault(family_name(row), []).append(row)
        family_values = list(families.values())
        family_verifier_rates = [
            value
            for value in (bool_rate_for(rows, "verifier_match") for rows in family_values)
            if value is not None
        ]
        family_partial_rates = [
            value
            for value in (bool_rate_for(rows, "partial_credit") for rows in family_values)
            if value is not None
        ]
        family_field_scores = [
            value
            for value in (field_score_mean(rows) for rows in family_values)
            if value is not None
        ]
        family_completion_quality = [
            value
            for value in (
                numeric_mean_for(rows, "completion_quality_ppm", 1_000_000.0)
                for rows in family_values
            )
            if value is not None
        ]
        family_schema_wrong_rates = [
            status_rate_for(rows, "SchemaValidWrong") for rows in family_values
        ]
        family_malformed_rates = [
            status_rate_for(rows, "Malformed") for rows in family_values
        ]
        family_schema_key_mismatch_rates = [
            value
            for value in (schema_key_mismatch_rate(rows) for rows in family_values)
            if value is not None
        ]
        family_answer_dominance = [
            value
            for value in (
                dominant_fraction(
                    [
                        str(row.get("actual_answer"))
                        for row in rows
                        if row.get("actual_answer") is not None
                    ]
                )
                for rows in family_values
            )
            if value is not None
        ]
        result.update(
            {
                f"{prefix}_family_count": len(families),
                f"{prefix}_min_family_rows": min(
                    (len(rows) for rows in family_values), default=None
                ),
                f"{prefix}_worst_family_verifier_rate": (
                    min(family_verifier_rates) if family_verifier_rates else None
                ),
                f"{prefix}_worst_family_partial_rate": (
                    min(family_partial_rates) if family_partial_rates else None
                ),
                f"{prefix}_worst_family_field_accuracy": (
                    min(family_field_scores) if family_field_scores else None
                ),
                f"{prefix}_worst_family_completion_quality": (
                    min(family_completion_quality) if family_completion_quality else None
                ),
                f"{prefix}_max_family_schema_wrong_rate": (
                    max(family_schema_wrong_rates) if family_schema_wrong_rates else None
                ),
                f"{prefix}_max_family_malformed_rate": (
                    max(family_malformed_rates) if family_malformed_rates else None
                ),
                f"{prefix}_max_family_schema_key_mismatch_rate": (
                    max(family_schema_key_mismatch_rates)
                    if family_schema_key_mismatch_rates
                    else None
                ),
                f"{prefix}_max_family_answer_dominant_fraction": (
                    max(family_answer_dominance) if family_answer_dominance else None
                ),
            }
        )

    return result


def read_raw_completion_samples(run_dir: str | None) -> dict[str, float | int | None]:
    return read_completion_samples_for_probe(
        run_dir,
        probe_name="ruliad_correctness",
        prefix="raw_completion",
        columns=RAW_COMPLETION_METRIC_COLUMNS,
        fallback_to_any_probe=True,
        include_family_metrics=True,
    )


def read_prompt_schema_completion_samples(run_dir: str | None) -> dict[str, float | int | None]:
    return read_completion_samples_for_probe(
        run_dir,
        probe_name="ruliad_correctness_prompt_schema",
        prefix="prompt_schema_completion",
        columns=PROMPT_SCHEMA_COMPLETION_METRIC_COLUMNS,
    )


def parse_answer_fields(answer: Any) -> dict[str, str]:
    if answer is None:
        return {}
    fields: dict[str, str] = {}
    for part in str(answer).strip().split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        if not key:
            continue
        fields[key] = value.strip()
    return fields


def latest_probe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []

    def group_key(row: dict[str, Any]) -> tuple[float, float, str]:
        epoch = finite(row.get("epoch")) or 0.0
        step = finite(row.get("absolute_step")) or 0.0
        probe = str(row.get("probe_name") or "")
        return epoch, step, probe

    latest_key = max(group_key(row) for row in records)
    return [row for row in records if group_key(row) == latest_key]


def entropy_bits(values: list[str]) -> float | None:
    if not values:
        return None
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    total = len(values)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def dominant_fraction(values: list[str]) -> float | None:
    if not values:
        return None
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts.values()) / len(values)


def read_trial_manifest(arm_dir: Path, trial_key: str | None) -> dict[str, Any]:
    if not trial_key:
        return {}
    path = arm_dir / "manifests" / f"{trial_key}.json"
    if not path.exists():
        return {}
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        "max_iters": manifest.get("max_iters"),
        "batch_size": manifest.get("batch_size"),
        "block_size": manifest.get("block_size"),
        "latent_total": manifest.get("latent_total"),
    }


def read_policy_config(run_dir: str | None) -> dict[str, float | int | None]:
    if not run_dir:
        return {column: None for column in POLICY_CONFIG_COLUMNS}
    path = Path(run_dir) / "training_config.json"
    if not path.exists():
        return {column: None for column in POLICY_CONFIG_COLUMNS}
    try:
        config = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {column: None for column in POLICY_CONFIG_COLUMNS}
    training = config.get("training")
    if not isinstance(training, dict):
        return {column: None for column in POLICY_CONFIG_COLUMNS}
    supervision = training.get("ruliad_supervision")
    if not isinstance(supervision, dict):
        return {column: None for column in POLICY_CONFIG_COLUMNS}
    verifier = supervision.get("verifier_reward")
    if not isinstance(verifier, dict):
        return {column: None for column in POLICY_CONFIG_COLUMNS}
    enabled = bool(verifier.get("enabled", False))
    weight = finite(verifier.get("weight")) or 0.0
    every_steps = int(finite(verifier.get("every_steps")) or 0)
    start_after_steps = int(finite(verifier.get("start_after_steps")) or 0)
    max_iters = int(finite(training.get("max_iters")) or 0)
    expected_updates = 0
    if enabled and weight > 0.0 and every_steps > 0 and max_iters > start_after_steps:
        expected_updates = ((max_iters - 1 - start_after_steps) // every_steps) + 1
    return {
        "policy_config_enabled": 1.0 if enabled else 0.0,
        "policy_config_weight": weight,
        "policy_config_start_after_steps": start_after_steps,
        "policy_config_every_steps": every_steps,
        "policy_config_expected_update_steps": expected_updates,
    }


def collect_trials(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in sorted(root.glob(f"*/analysis/{TRIAL_SUMMARY}")):
        arm = summary.parents[1].name
        arm_dir = summary.parents[1]
        for row in read_csv(summary):
            out: dict[str, Any] = {"arm": arm}
            out.update(row)
            for key, value in read_trial_manifest(arm_dir, row.get("trial_key")).items():
                out.setdefault(key, value)
            out.update(read_policy_config(row.get("run_dir")))
            out.update(read_policy_telemetry(row.get("run_dir")))
            out.update(read_structured_recovery_config(row.get("run_dir")))
            out.update(read_structured_recovery_telemetry(row.get("run_dir")))
            out.update(read_answer_contract_config(row.get("run_dir")))
            out.update(read_answer_contract_telemetry(row.get("run_dir")))
            out.update(read_structured_contrast_config(row.get("run_dir")))
            out.update(read_structured_contrast_telemetry(row.get("run_dir")))
            out.update(read_field_binding_contrast_config(row.get("run_dir")))
            out.update(read_field_binding_contrast_telemetry(row.get("run_dir")))
            out.update(read_generated_attractor_config(row.get("run_dir")))
            out.update(read_generated_attractor_telemetry(row.get("run_dir")))
            out.update(read_verifier_rollout_telemetry(row.get("run_dir")))
            out.update(read_raw_completion_samples(row.get("run_dir")))
            out.update(read_prompt_schema_completion_samples(row.get("run_dir")))
            rows.append(out)
    return rows


def summarize_by_arm(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for arm in sorted({str(row.get("arm") or "") for row in rows}):
        arm_rows = [row for row in rows if row.get("arm") == arm]
        ok_rows = [row for row in arm_rows if row.get("status") == "ok"]
        healthy_values = [
            1.0 if str(row.get("healthy") or "").lower() == "true" else 0.0
            for row in ok_rows
        ]
        summary: dict[str, Any] = {
            "arm": arm,
            "trials": len(arm_rows),
            "ok_trials": len(ok_rows),
            "healthy_trial_fraction": mean(healthy_values),
        }
        for column in (
            METRIC_COLUMNS
            + POLICY_METRIC_COLUMNS
            + POLICY_CONFIG_COLUMNS
            + ANSWER_CONTRACT_CONFIG_COLUMNS
            + ANSWER_CONTRACT_METRIC_COLUMNS
            + STRUCTURED_RECOVERY_CONFIG_COLUMNS
            + STRUCTURED_RECOVERY_METRIC_COLUMNS
            + STRUCTURED_CONTRAST_CONFIG_COLUMNS
            + STRUCTURED_CONTRAST_METRIC_COLUMNS
            + FIELD_BINDING_CONTRAST_CONFIG_COLUMNS
            + FIELD_BINDING_CONTRAST_METRIC_COLUMNS
            + GENERATED_ATTRACTOR_CONFIG_COLUMNS
            + GENERATED_ATTRACTOR_METRIC_COLUMNS
            + VERIFIER_ROLLOUT_METRIC_COLUMNS
            + RAW_COMPLETION_METRIC_COLUMNS
            + PROMPT_SCHEMA_COMPLETION_METRIC_COLUMNS
        ):
            clean = [value for value in (finite(row.get(column)) for row in ok_rows) if value is not None]
            summary[f"{column}_mean"] = mean(clean)
        summaries.append(summary)
    return summaries


def value(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    parsed = finite(row.get(key))
    return default if parsed is None else parsed


def add_gate_decisions(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    baseline = next((row for row in rows if row.get("arm") == args.baseline_arm), None)
    if baseline is None:
        raise SystemExit(f"baseline arm not found: {args.baseline_arm}")

    base_model_tps = value(baseline, "stage_model_tokens_per_sec_mean", 0.0)
    base_peak_memory = value(baseline, "peak_used_mb_mean", 0.0)
    base_valid = value(baseline, "valid_teacher_ce_last_mean", 0.0)
    base_source = value(baseline, "source_mean_difficulty_last_mean", 0.0)
    base_verifier = value(baseline, "ruliad_verifier_last_mean", 0.0)
    base_semantic = value(baseline, "ruliad_semantic_last_mean", 0.0)
    base_partial = value(baseline, "ruliad_partial_last_mean", 0.0)
    base_schema = value(baseline, "ruliad_schema_wrong_last_mean", 0.0)
    base_malformed = value(baseline, "ruliad_malformed_last_mean", 0.0)
    base_answer_field = value(baseline, "ruliad_answer_field_accuracy_last_mean", 0.0)
    base_answer_coverage = value(baseline, "ruliad_answer_field_coverage_last_mean", 0.0)
    base_answer_termination = value(baseline, "ruliad_answer_termination_rate_last_mean", 0.0)
    base_completion = value(baseline, "completion_health_last_mean", 0.0)
    base_entropy = value(baseline, "output_entropy_bits_last_mean", 0.0)
    base_distinct2 = value(baseline, "output_distinct_2_last_mean", 0.0)
    base_best_eval_verifier = finite(baseline.get("best_eval_verifier_mean"))
    base_best_eval_completion = finite(baseline.get("best_eval_completion_mean"))

    out: list[dict[str, Any]] = []
    for row in rows:
        arm = str(row.get("arm") or "")
        model_tps = value(row, "stage_model_tokens_per_sec_mean", 0.0)
        peak_memory = value(row, "peak_used_mb_mean", 0.0)
        valid = value(row, "valid_teacher_ce_last_mean", 0.0)
        latent_eval_ce_delta = finite(row.get("latent_eval_final_ce_delta_last_mean"))
        latent_eval_ce_violation = finite(row.get("latent_eval_final_ce_violation_last_mean"))
        latent_eval_entropy = finite(row.get("latent_eval_final_entropy_bits_last_mean"))
        latent_eval_delta_rms = finite(row.get("latent_eval_final_delta_rms_last_mean"))
        latent_extra_eval_ce_delta = finite(row.get("latent_extra_eval_max_ce_delta_last_mean"))
        latent_extra_eval_ce_violation = finite(
            row.get("latent_extra_eval_max_ce_violation_last_mean")
        )
        latent_extra_eval_entropy = finite(
            row.get("latent_extra_eval_min_entropy_bits_last_mean")
        )
        latent_extra_eval_delta_rms = finite(
            row.get("latent_extra_eval_max_delta_rms_last_mean")
        )
        source = value(row, "source_mean_difficulty_last_mean", 0.0)
        verifier = value(row, "ruliad_verifier_last_mean", 0.0)
        semantic = value(row, "ruliad_semantic_last_mean", 0.0)
        partial = value(row, "ruliad_partial_last_mean", 0.0)
        schema = value(row, "ruliad_schema_wrong_last_mean", 0.0)
        malformed = value(row, "ruliad_malformed_last_mean", 0.0)
        answer_field = value(row, "ruliad_answer_field_accuracy_last_mean", 0.0)
        answer_coverage = value(row, "ruliad_answer_field_coverage_last_mean", 0.0)
        answer_termination = value(row, "ruliad_answer_termination_rate_last_mean", 0.0)
        completion = value(row, "completion_health_last_mean", 0.0)
        completion_distinct2 = value(row, "completion_distinct_2_last_mean", 1.0)
        completion_period = value(row, "completion_period_2_to_64_last_mean", 0.0)
        completion_repetition = value(row, "completion_repetition_last_mean", 0.0)
        capability_score_drop = value(row, "capability_score_drop_from_best_mean", 0.0)
        capability_lagging_buckets = value(row, "capability_bucket_lagging_count_mean", 0.0)
        capability_lagging_contracts = value(
            row, "capability_contract_lagging_count_mean", 0.0
        )
        verifier_drop_from_best = value(
            row, "capability_verifier_drop_from_best_mean", 0.0
        )
        completion_drop_from_best = value(
            row, "capability_completion_drop_from_best_mean", 0.0
        )
        best_eval_steps = finite(row.get("best_eval_steps_mean"))
        best_eval_verifier_raw = finite(row.get("best_eval_verifier_mean"))
        best_eval_completion_raw = finite(row.get("best_eval_completion_mean"))
        best_eval_verifier = verifier if best_eval_verifier_raw is None else best_eval_verifier_raw
        best_eval_completion = (
            completion if best_eval_completion_raw is None else best_eval_completion_raw
        )
        best_eval_verifier_step_delta = value(row, "best_eval_verifier_delta_mean", 0.0)
        best_eval_completion_step_delta = value(
            row, "best_eval_completion_delta_mean", 0.0
        )
        extra_eval_step_count = value(row, "extra_eval_step_count_mean", 0.0)
        extra_eval_min_verifier_delta = value(
            row, "extra_eval_min_verifier_delta_mean", 0.0
        )
        extra_eval_min_completion_delta = value(
            row, "extra_eval_min_completion_delta_mean", 0.0
        )
        extra_eval_max_malformed_delta = value(
            row, "extra_eval_max_malformed_delta_mean", 0.0
        )
        best_eval_verifier_baseline_delta = (
            0.0
            if base_best_eval_verifier is None or best_eval_verifier_raw is None
            else best_eval_verifier - base_best_eval_verifier
        )
        best_eval_completion_baseline_delta = (
            0.0
            if base_best_eval_completion is None or best_eval_completion_raw is None
            else best_eval_completion - base_best_eval_completion
        )
        recovery_control_fraction = value(row, "recovery_control_fraction_mean", 0.0)
        raw_completion_quality = value(row, "raw_completion_quality_mean_mean", 1.0)
        raw_completion_rows = value(row, "raw_completion_rows_mean", 0.0)
        raw_completion_verifier = value(row, "raw_completion_verifier_rate_mean", 0.0)
        raw_completion_semantic = value(row, "raw_completion_semantic_rate_mean", 0.0)
        raw_completion_partial = value(row, "raw_completion_partial_rate_mean", 0.0)
        raw_completion_schema_wrong = value(row, "raw_completion_schema_wrong_rate_mean", 0.0)
        raw_completion_malformed = value(row, "raw_completion_malformed_rate_mean", 0.0)
        raw_completion_missing = value(row, "raw_completion_missing_rate_mean", 0.0)
        raw_completion_expected_distinct = value(
            row,
            "raw_completion_expected_answer_distinct_fraction_mean",
            1.0,
        )
        raw_completion_answer_distinct = value(
            row,
            "raw_completion_actual_answer_distinct_fraction_mean",
            1.0,
        )
        raw_completion_field_value_distinct_ratio = value(
            row,
            "raw_completion_field_value_distinct_ratio_mean",
            1.0,
        )
        raw_completion_field_value_dominance = value(
            row,
            "raw_completion_actual_field_value_dominant_fraction_mean",
            0.0,
        )
        raw_completion_family_count = value(row, "raw_completion_family_count_mean", 0.0)
        raw_completion_min_family_rows = value(row, "raw_completion_min_family_rows_mean", 0.0)
        raw_completion_worst_family_verifier = value(
            row,
            "raw_completion_worst_family_verifier_rate_mean",
            raw_completion_verifier,
        )
        raw_completion_worst_family_partial = value(
            row,
            "raw_completion_worst_family_partial_rate_mean",
            raw_completion_partial,
        )
        raw_completion_worst_family_field = value(
            row,
            "raw_completion_worst_family_field_accuracy_mean",
            raw_completion_field_value_distinct_ratio,
        )
        raw_completion_worst_family_quality = value(
            row,
            "raw_completion_worst_family_completion_quality_mean",
            raw_completion_quality,
        )
        raw_completion_max_family_schema_wrong = value(
            row,
            "raw_completion_max_family_schema_wrong_rate_mean",
            raw_completion_schema_wrong,
        )
        raw_completion_max_family_malformed = value(
            row,
            "raw_completion_max_family_malformed_rate_mean",
            raw_completion_malformed,
        )
        raw_completion_max_family_schema_key_mismatch = value(
            row,
            "raw_completion_max_family_schema_key_mismatch_rate_mean",
            0.0,
        )
        raw_completion_max_family_answer_dominance = value(
            row,
            "raw_completion_max_family_answer_dominant_fraction_mean",
            raw_completion_field_value_dominance,
        )
        prompt_schema_completion_rows = value(
            row, "prompt_schema_completion_rows_mean", 0.0
        )
        prompt_schema_completion_quality = value(
            row, "prompt_schema_completion_quality_mean_mean", 1.0
        )
        prompt_schema_completion_verifier = value(
            row, "prompt_schema_completion_verifier_rate_mean", 0.0
        )
        prompt_schema_completion_semantic = value(
            row, "prompt_schema_completion_semantic_rate_mean", 0.0
        )
        prompt_schema_completion_partial = value(
            row, "prompt_schema_completion_partial_rate_mean", 0.0
        )
        prompt_schema_completion_schema_wrong = value(
            row, "prompt_schema_completion_schema_wrong_rate_mean", 0.0
        )
        prompt_schema_completion_malformed = value(
            row, "prompt_schema_completion_malformed_rate_mean", 0.0
        )
        prompt_schema_completion_missing = value(
            row, "prompt_schema_completion_missing_rate_mean", 0.0
        )
        prompt_schema_completion_expected_distinct = value(
            row,
            "prompt_schema_completion_expected_answer_distinct_fraction_mean",
            1.0,
        )
        prompt_schema_completion_answer_distinct = value(
            row,
            "prompt_schema_completion_actual_answer_distinct_fraction_mean",
            1.0,
        )
        prompt_schema_completion_field_value_distinct_ratio = value(
            row,
            "prompt_schema_completion_field_value_distinct_ratio_mean",
            1.0,
        )
        prompt_schema_completion_field_value_dominance = value(
            row,
            "prompt_schema_completion_actual_field_value_dominant_fraction_mean",
            0.0,
        )
        contract_probe_present = finite(row.get("contract_probe_verifier_last_mean")) is not None
        contract_verifier = value(row, "contract_probe_verifier_last_mean", verifier)
        contract_field = value(row, "contract_probe_answer_field_accuracy_last_mean", answer_field)
        contract_completion = value(row, "contract_probe_completion_health_last_mean", completion)
        contract_verifier_delta = value(row, "contract_probe_verifier_delta_mean", 0.0)
        contract_field_delta = value(row, "contract_probe_answer_field_delta_mean", 0.0)
        contract_completion_delta = value(row, "contract_probe_completion_delta_mean", 0.0)
        answer_contract_config_weight = value(row, "answer_contract_config_weight_mean", 0.0)
        answer_contract_prompt_schema_value_weight = value(
            row, "answer_contract_config_prompt_schema_value_weight_mean", 0.0
        )
        answer_contract_expected_updates = value(
            row, "answer_contract_config_expected_update_steps_mean", 0.0
        )
        answer_contract_rows = value(row, "answer_contract_oracle_rows_mean", 0.0)
        answer_contract_prompt_schema_rows = value(
            row, "answer_contract_prompt_schema_rows_mean", 0.0
        )
        answer_contract_tokens = value(row, "answer_contract_tokens_mean", 0.0)
        answer_contract_prompt_schema_value_tokens = value(
            row, "answer_contract_prompt_schema_value_tokens_mean", 0.0
        )
        answer_contract_missing_policy = value(
            row, "answer_contract_missing_policy_batch_count_mean", 0.0
        )
        answer_contract_policy_present = value(
            row, "answer_contract_policy_batch_present_fraction_mean", 1.0
        )
        policy_completion_rows = value(row, "policy_completion_rows_mean", 0.0)
        policy_clip_fraction = value(row, "policy_advantage_clip_fraction_mean", 0.0)
        policy_skipped = value(row, "policy_update_skipped_count_mean", 0.0)
        policy_config_weight = value(row, "policy_config_weight_mean", 0.0)
        policy_expected_updates = value(row, "policy_config_expected_update_steps_mean", 0.0)
        recovery_config_weight = value(row, "recovery_config_weight_mean", 0.0)
        recovery_config_every_steps = value(row, "recovery_config_every_steps_mean", 0.0)
        recovery_expected_updates = value(
            row, "recovery_config_expected_update_steps_mean", 0.0
        )
        recovery_rows = value(row, "recovery_rows_mean", 0.0)
        recovery_weight = value(row, "recovery_weight_mean", 0.0)
        recovery_effective_weight = max(recovery_weight, recovery_config_weight)
        recovery_missing_policy = value(row, "recovery_missing_policy_batch_count_mean", 0.0)
        recovery_policy_present = value(row, "recovery_policy_batch_present_fraction_mean", 1.0)
        contrast_weight = value(row, "contrast_config_weight_mean", 0.0)
        contrast_expected_updates = value(row, "contrast_config_expected_update_steps_mean", 0.0)
        contrast_pairs = value(row, "contrast_pairs_mean", 0.0)
        field_binding_weight = value(row, "field_binding_config_weight_mean", 0.0)
        field_binding_expected_updates = value(
            row, "field_binding_config_expected_update_steps_mean", 0.0
        )
        field_binding_expected_rank_metrics = value(
            row, "field_binding_config_expected_rank_metric_steps_mean", 0.0
        )
        field_binding_pairs = value(row, "field_binding_contrast_pairs_mean", 0.0)
        field_binding_rank_tokens = value(row, "field_binding_rank_metric_tokens_mean", 0.0)
        field_binding_positive_fraction = value(
            row, "field_binding_positive_token_fraction_mean", 1.0
        )
        field_binding_exact_pair_fraction = value(
            row, "field_binding_exact_pair_rank_fraction_mean", 1.0
        )
        generated_attractor_capacity = value(
            row, "generated_attractor_config_capacity_mean", 0.0
        )
        generated_attractor_min_distinct = value(
            row, "generated_attractor_config_min_distinct_answers_mean", 1.0
        )
        generated_attractor_max_dominant_fraction = value(
            row, "generated_attractor_config_max_dominant_fraction_mean", 1.0
        )
        generated_attractor_observed = value(
            row, "generated_attractor_observed_rows_mean", 0.0
        )
        generated_attractor_recorded = value(
            row, "generated_attractor_recorded_rows_mean", 0.0
        )
        generated_attractor_active = value(
            row, "generated_attractor_active_count_mean", 0.0
        )
        generated_attractor_distinct = value(
            row, "generated_attractor_distinct_answer_count_mean", 0.0
        )
        generated_attractor_dominant_fraction = value(
            row, "generated_attractor_dominant_answer_fraction_mean", 0.0
        )
        generated_attractor_selected = value(
            row, "generated_attractor_selected_candidate_rows_mean", 0.0
        )
        generated_attractor_policy_rows = value(
            row, "policy_generated_attractor_completion_rows_mean", 0.0
        )
        generated_attractor_contrast_rows = value(
            row, "contrast_generated_attractor_negative_completion_rows_mean", 0.0
        )
        generated_attractor_field_pairs = value(
            row, "field_binding_generated_attractor_contrast_pairs_mean", 0.0
        )
        rollout_imitation_weight = value(row, "rollout_imitation_weight_mean", 0.0)
        rollout_imitation_generated = value(row, "rollout_imitation_generated_rows_mean", 0.0)
        rollout_imitation_candidate = value(row, "rollout_imitation_candidate_rows_mean", 0.0)
        rollout_imitation_accepted = value(row, "rollout_imitation_accepted_rows_mean", 0.0)
        rollout_imitation_gate = value(
            row, "rollout_imitation_health_gate_passed_fraction_mean", 1.0
        )
        max_iters = value(row, "max_iters_mean", 0.0)
        mature_enough = args.min_mature_iters <= 0 or max_iters >= args.min_mature_iters
        entropy = value(row, "output_entropy_bits_last_mean", 0.0)
        distinct2 = value(row, "output_distinct_2_last_mean", 0.0)
        healthy_fraction = value(row, "healthy_trial_fraction", 1.0)
        throughput_ratio = model_tps / base_model_tps if base_model_tps > 0.0 else 0.0
        peak_memory_ratio = (
            peak_memory / base_peak_memory if base_peak_memory > 0.0 else 1.0
        )
        valid_delta = valid - base_valid
        source_delta = source - base_source
        verifier_delta = verifier - base_verifier
        semantic_delta = semantic - base_semantic
        partial_delta = partial - base_partial
        schema_delta = schema - base_schema
        malformed_delta = malformed - base_malformed
        answer_field_delta = answer_field - base_answer_field
        answer_coverage_delta = answer_coverage - base_answer_coverage
        answer_termination_delta = answer_termination - base_answer_termination
        completion_delta = completion - base_completion
        entropy_delta = entropy - base_entropy
        distinct2_delta = distinct2 - base_distinct2
        fatal_gate_count = value(row, "fatal_gate_count_mean", 0.0)

        reasons: list[str] = []
        if not mature_enough:
            reasons.append("insufficient_mature_iters")
        if int(row.get("ok_trials") or 0) < int(row.get("trials") or 0):
            reasons.append("failed_trials")
        if mature_enough and healthy_fraction < 1.0:
            reasons.append("unhealthy_trials")
        if fatal_gate_count > 0.0:
            reasons.append("fatal_gates")
        if throughput_ratio < args.min_throughput_ratio:
            reasons.append("slow")
        if (
            arm != args.baseline_arm
            and peak_memory_ratio > args.max_peak_memory_ratio
            and verifier_delta < args.min_raw_verifier_gain_for_memory_regression
        ):
            reasons.append("memory_regression_without_raw_gain")
        if mature_enough:
            if valid_delta > args.max_valid_ce_delta:
                reasons.append("valid_ce_regression")
            if (
                latent_eval_ce_delta is not None
                and latent_eval_ce_delta > args.max_latent_eval_ce_delta
            ):
                reasons.append("latent_eval_ce_explosion")
            if (
                latent_eval_ce_violation is not None
                and latent_eval_ce_violation > args.max_latent_eval_ce_violation
            ):
                reasons.append("latent_eval_monotonic_violation")
            if (
                latent_eval_entropy is not None
                and latent_eval_entropy < args.min_latent_eval_entropy
            ):
                reasons.append("latent_eval_entropy_collapse")
            if (
                latent_eval_delta_rms is not None
                and latent_eval_delta_rms > args.max_latent_eval_delta_rms
            ):
                reasons.append("latent_eval_delta_explosion")
            if (
                latent_extra_eval_ce_delta is not None
                and latent_extra_eval_ce_delta > args.max_latent_extra_eval_ce_delta
            ):
                reasons.append("latent_extra_eval_ce_explosion")
            if (
                latent_extra_eval_ce_violation is not None
                and latent_extra_eval_ce_violation
                > args.max_latent_extra_eval_ce_violation
            ):
                reasons.append("latent_extra_eval_monotonic_violation")
            if (
                latent_extra_eval_entropy is not None
                and latent_extra_eval_entropy < args.min_latent_extra_eval_entropy
            ):
                reasons.append("latent_extra_eval_entropy_collapse")
            if (
                latent_extra_eval_delta_rms is not None
                and latent_extra_eval_delta_rms > args.max_latent_extra_eval_delta_rms
            ):
                reasons.append("latent_extra_eval_delta_explosion")
            if source_delta > args.max_source_difficulty_delta:
                reasons.append("source_difficulty_overshoot")
            if verifier_delta < -args.max_verifier_regression:
                reasons.append("verifier_regression")
            if schema_delta > args.max_schema_wrong_delta:
                reasons.append("schema_regression")
            if malformed_delta > args.max_malformed_delta:
                reasons.append("malformed_regression")
            if completion_delta < -args.max_completion_regression:
                reasons.append("completion_regression")
            if answer_field_delta < -args.max_answer_field_regression:
                reasons.append("answer_field_regression")
            if answer_termination_delta < -args.max_answer_termination_regression:
                reasons.append("answer_termination_regression")
            if verifier < args.min_mature_verifier_rate:
                reasons.append("verifier_floor")
            if semantic < args.min_mature_semantic_rate:
                reasons.append("semantic_floor")
            if partial < args.min_mature_partial_rate:
                reasons.append("partial_floor")
            if completion_distinct2 < args.min_completion_distinct_2:
                reasons.append("completion_distinct2_collapse")
            if completion_period > args.max_completion_period:
                reasons.append("completion_period_collapse")
            if completion_repetition > args.max_completion_repetition:
                reasons.append("completion_repetition_collapse")
            if contract_probe_present and contract_verifier < args.min_mature_verifier_rate:
                reasons.append("contract_probe_verifier_floor")
            if contract_probe_present and contract_field < args.min_mature_answer_field_rate:
                reasons.append("contract_probe_field_floor")
            if (
                contract_probe_present
                and contract_verifier_delta > args.max_free_run_contract_verifier_gap
                and verifier < args.min_mature_verifier_rate
            ):
                reasons.append("free_run_contract_gap")
            if (
                contract_probe_present
                and contract_verifier < args.min_mature_verifier_rate
                and verifier < args.min_mature_verifier_rate
            ):
                reasons.append("contract_value_failure")
            if (
                best_eval_steps is not None
                and verifier < args.min_mature_verifier_rate
                and best_eval_verifier >= args.min_mature_verifier_rate
            ):
                reasons.append("latent_step_selector_needed")
            if capability_score_drop > args.max_capability_score_drop:
                reasons.append("capability_score_drop")
            if capability_lagging_buckets > args.max_capability_lagging_buckets:
                reasons.append("capability_lagging_buckets")
            if capability_lagging_contracts > args.max_capability_lagging_buckets:
                reasons.append("capability_lagging_contracts")
            if verifier_drop_from_best > args.max_verifier_drop_from_best:
                reasons.append("verifier_drop_from_best")
            if completion_drop_from_best > args.max_completion_drop_from_best:
                reasons.append("completion_drop_from_best")
            if (
                extra_eval_step_count > 0.0
                and extra_eval_min_verifier_delta < -args.max_extra_step_verifier_drop
            ):
                reasons.append("extra_step_verifier_collapse")
            if (
                extra_eval_step_count > 0.0
                and extra_eval_min_completion_delta < -args.max_extra_step_completion_drop
            ):
                reasons.append("extra_step_completion_collapse")
            if (
                extra_eval_step_count > 0.0
                and extra_eval_max_malformed_delta > args.max_extra_step_malformed_increase
            ):
                reasons.append("extra_step_malformed_collapse")
            if recovery_control_fraction > args.max_recovery_control_fraction:
                reasons.append("recovery_thrash")
            if raw_completion_rows < args.min_raw_completion_rows:
                reasons.append("raw_completion_probe_too_small")
            if raw_completion_verifier < args.min_raw_completion_verifier_rate:
                reasons.append("raw_completion_verifier_floor")
            if raw_completion_semantic < args.min_raw_completion_semantic_rate:
                reasons.append("raw_completion_semantic_floor")
            if raw_completion_partial < args.min_raw_completion_partial_rate:
                reasons.append("raw_completion_partial_floor")
            if raw_completion_schema_wrong > args.max_raw_completion_schema_wrong_rate:
                reasons.append("raw_completion_schema_wrong_high")
            if raw_completion_malformed > args.max_raw_completion_malformed_rate:
                reasons.append("raw_completion_malformed_high")
            if raw_completion_missing > args.max_raw_completion_missing_rate:
                reasons.append("raw_completion_missing_high")
            if raw_completion_quality < args.min_raw_completion_quality:
                reasons.append("raw_completion_quality_collapse")
            if (
                raw_completion_expected_distinct >= args.min_raw_completion_answer_distinct
                and raw_completion_answer_distinct < args.min_raw_completion_answer_distinct
            ):
                reasons.append("raw_completion_answer_collapse")
            if (
                raw_completion_field_value_distinct_ratio
                < args.min_raw_completion_field_value_distinct_ratio
            ):
                reasons.append("raw_completion_field_value_collapse")
            if (
                raw_completion_field_value_dominance
                > args.max_raw_completion_field_value_dominance
            ):
                reasons.append("raw_completion_field_value_dominance")
            if raw_completion_family_count < args.min_raw_completion_family_count:
                reasons.append("raw_completion_family_probe_too_narrow")
            if raw_completion_min_family_rows < args.min_raw_completion_family_rows:
                reasons.append("raw_completion_family_probe_too_thin")
            if (
                raw_completion_worst_family_verifier
                < args.min_raw_completion_family_verifier_rate
            ):
                reasons.append("raw_completion_family_verifier_floor")
            if (
                raw_completion_worst_family_partial
                < args.min_raw_completion_family_partial_rate
            ):
                reasons.append("raw_completion_family_partial_floor")
            if (
                raw_completion_worst_family_field
                < args.min_raw_completion_family_field_rate
            ):
                reasons.append("raw_completion_family_field_floor")
            if (
                raw_completion_worst_family_quality
                < args.min_raw_completion_family_quality
            ):
                reasons.append("raw_completion_family_quality_collapse")
            if (
                raw_completion_max_family_schema_wrong
                > args.max_raw_completion_family_schema_wrong_rate
            ):
                reasons.append("raw_completion_family_schema_wrong_high")
            if (
                raw_completion_max_family_malformed
                > args.max_raw_completion_family_malformed_rate
            ):
                reasons.append("raw_completion_family_malformed_high")
            if (
                raw_completion_max_family_schema_key_mismatch
                > args.max_raw_completion_family_schema_key_mismatch
            ):
                reasons.append("raw_completion_schema_key_leakage")
            if (
                raw_completion_max_family_answer_dominance
                > args.max_raw_completion_family_answer_dominance
            ):
                reasons.append("raw_completion_family_answer_attractor")
            if prompt_schema_completion_rows > 0.0:
                if prompt_schema_completion_rows < args.min_prompt_schema_completion_rows:
                    reasons.append("prompt_schema_completion_probe_too_small")
                if (
                    prompt_schema_completion_verifier
                    < args.min_prompt_schema_completion_verifier_rate
                ):
                    reasons.append("prompt_schema_completion_verifier_floor")
                if (
                    prompt_schema_completion_semantic
                    < args.min_prompt_schema_completion_semantic_rate
                ):
                    reasons.append("prompt_schema_completion_semantic_floor")
                if (
                    prompt_schema_completion_partial
                    < args.min_prompt_schema_completion_partial_rate
                ):
                    reasons.append("prompt_schema_completion_partial_floor")
                if (
                    prompt_schema_completion_schema_wrong
                    > args.max_prompt_schema_completion_schema_wrong_rate
                ):
                    reasons.append("prompt_schema_completion_schema_wrong_high")
                if (
                    prompt_schema_completion_malformed
                    > args.max_prompt_schema_completion_malformed_rate
                ):
                    reasons.append("prompt_schema_completion_malformed_high")
                if (
                    prompt_schema_completion_missing
                    > args.max_prompt_schema_completion_missing_rate
                ):
                    reasons.append("prompt_schema_completion_missing_high")
                if (
                    prompt_schema_completion_quality
                    < args.min_prompt_schema_completion_quality
                ):
                    reasons.append("prompt_schema_completion_quality_collapse")
                if (
                    prompt_schema_completion_expected_distinct
                    >= args.min_prompt_schema_completion_answer_distinct
                    and prompt_schema_completion_answer_distinct
                    < args.min_prompt_schema_completion_answer_distinct
                ):
                    reasons.append("prompt_schema_completion_answer_collapse")
                if (
                    prompt_schema_completion_field_value_distinct_ratio
                    < args.min_prompt_schema_completion_field_value_distinct_ratio
                ):
                    reasons.append("prompt_schema_completion_field_value_collapse")
                if (
                    prompt_schema_completion_field_value_dominance
                    > args.max_prompt_schema_completion_field_value_dominance
                ):
                    reasons.append("prompt_schema_completion_field_value_dominance")
        if policy_completion_rows > 0.0 and policy_clip_fraction > args.max_policy_advantage_clip_fraction:
            reasons.append("policy_advantage_clip_saturation")
        if policy_config_weight > 0.0 and policy_expected_updates > 0.0 and policy_completion_rows <= 0.0:
            reasons.append("policy_objective_inactive")
        if policy_skipped > 0.0:
            reasons.append("policy_update_skipped")
        if (
            recovery_config_weight > 0.0
            and recovery_expected_updates <= 0.0
        ):
            reasons.append("recovery_objective_unscheduled")
        if (
            recovery_effective_weight > 0.0
            and (recovery_expected_updates > 0.0 or recovery_config_weight <= 0.0)
            and recovery_rows <= 0.0
        ):
            reasons.append("recovery_objective_inactive")
        if recovery_effective_weight > 0.0 and recovery_missing_policy > 0.0:
            reasons.append("recovery_policy_batch_missing")
        if recovery_effective_weight > 0.0 and recovery_policy_present < 1.0:
            reasons.append("recovery_policy_batch_partial")
        if (
            answer_contract_config_weight > 0.0
            and answer_contract_expected_updates > 0.0
            and answer_contract_rows <= 0.0
        ):
            reasons.append("answer_contract_objective_inactive")
        if (
            answer_contract_config_weight > 0.0
            and answer_contract_expected_updates > 0.0
            and answer_contract_tokens <= 0.0
        ):
            reasons.append("answer_contract_objective_no_tokens")
        if (
            answer_contract_config_weight > 0.0
            and answer_contract_prompt_schema_value_weight > 0.0
            and answer_contract_expected_updates > 0.0
            and answer_contract_prompt_schema_rows <= 0.0
        ):
            reasons.append("answer_contract_prompt_schema_objective_inactive")
        if (
            answer_contract_config_weight > 0.0
            and answer_contract_prompt_schema_value_weight > 0.0
            and answer_contract_expected_updates > 0.0
            and answer_contract_prompt_schema_value_tokens <= 0.0
        ):
            reasons.append("answer_contract_prompt_schema_no_tokens")
        if answer_contract_config_weight > 0.0 and answer_contract_missing_policy > 0.0:
            reasons.append("answer_contract_policy_batch_missing")
        if answer_contract_config_weight > 0.0 and answer_contract_policy_present < 1.0:
            reasons.append("answer_contract_policy_batch_partial")
        if contrast_weight > 0.0 and contrast_expected_updates > 0.0 and contrast_pairs <= 0.0:
            reasons.append("contrast_objective_inactive")
        if (
            field_binding_weight > 0.0
            and field_binding_expected_updates > 0.0
            and field_binding_pairs <= 0.0
        ):
            reasons.append("field_binding_objective_inactive")
        if (
            field_binding_weight > 0.0
            and field_binding_expected_rank_metrics > 0.0
            and field_binding_rank_tokens <= 0.0
        ):
            reasons.append("field_binding_rank_metrics_missing")
        if (
            field_binding_weight > 0.0
            and field_binding_rank_tokens > 0.0
            and field_binding_positive_fraction
            < args.min_field_binding_positive_token_fraction
        ):
            reasons.append("field_binding_positive_rank_weak")
        if (
            field_binding_weight > 0.0
            and field_binding_rank_tokens > 0.0
            and field_binding_exact_pair_fraction
            < args.min_field_binding_exact_pair_rank_fraction
        ):
            reasons.append("field_binding_pair_rank_weak")
        if mature_enough and generated_attractor_capacity > 0.0:
            if generated_attractor_observed <= 0.0:
                reasons.append("generated_attractor_generation_inactive")
            elif generated_attractor_recorded > 0.0:
                if generated_attractor_active <= 0.0:
                    reasons.append("generated_attractor_replay_inactive")
                elif generated_attractor_distinct < generated_attractor_min_distinct:
                    reasons.append("generated_attractor_answer_diversity_low")
                elif (
                    generated_attractor_dominant_fraction
                    > generated_attractor_max_dominant_fraction
                    + 1.0e-9
                ):
                    reasons.append("generated_attractor_dominance_high")
                elif generated_attractor_selected <= 0.0:
                    reasons.append("generated_attractor_replay_unselected")
                elif (
                    generated_attractor_policy_rows
                    + generated_attractor_contrast_rows
                    + generated_attractor_field_pairs
                    <= 0.0
                ):
                    reasons.append("generated_attractor_replay_unconsumed")
        if rollout_imitation_weight > 0.0 and rollout_imitation_generated <= 0.0:
            reasons.append("rollout_imitation_no_generations")
        if (
            rollout_imitation_weight > 0.0
            and rollout_imitation_candidate > 0.0
            and rollout_imitation_accepted <= 0.0
            and rollout_imitation_gate <= 0.0
        ):
            reasons.append("rollout_imitation_health_gate_blocked")
        if rollout_imitation_weight > 0.0 and rollout_imitation_accepted <= 0.0:
            reasons.append("rollout_imitation_inactive")
        if mature_enough and entropy < args.min_output_entropy:
            reasons.append("entropy_collapse")
        if mature_enough and distinct2 < args.min_output_distinct_2:
            reasons.append("output_distinct2_collapse")

        score_delta = (
            verifier_delta * 6.0
            + semantic_delta * 3.0
            + partial_delta * 2.0
            - schema_delta * 2.0
            + completion_delta * 1.5
            + answer_field_delta
            + answer_termination_delta * 0.5
            + max(0.0, contract_verifier_delta) * 0.5
            + max(0.0, contract_field_delta) * 0.25
            + max(0.0, contract_completion_delta) * 0.25
            + max(0.0, best_eval_verifier_baseline_delta) * 1.0
            + max(0.0, best_eval_completion_baseline_delta) * 0.5
            + max(0.0, best_eval_verifier_step_delta) * 0.5
            + max(0.0, best_eval_completion_step_delta) * 0.25
            - max(0.0, -best_eval_verifier_step_delta) * 0.25
            - max(0.0, -best_eval_completion_step_delta) * 0.10
            - valid_delta
            - max(source_delta, 0.0) * 0.20
            + (throughput_ratio - 1.0) * 0.50
            + entropy_delta * 0.05
            + distinct2_delta * 0.5
            - (1.0 - healthy_fraction) * 2.0
            - capability_score_drop
            - max(0.0, capability_lagging_buckets - args.max_capability_lagging_buckets) * 0.10
            - max(0.0, capability_lagging_contracts - args.max_capability_lagging_buckets) * 0.20
            - verifier_drop_from_best * 4.0
            - completion_drop_from_best * 2.0
            - max(
                0.0,
                args.min_raw_completion_family_verifier_rate
                - raw_completion_worst_family_verifier,
            )
            * 4.0
            - max(
                0.0,
                raw_completion_max_family_schema_key_mismatch
                - args.max_raw_completion_family_schema_key_mismatch,
            )
            * 2.0
            - max(
                0.0,
                args.min_prompt_schema_completion_verifier_rate
                - prompt_schema_completion_verifier,
            )
            * 3.0
            - max(
                0.0,
                args.min_prompt_schema_completion_field_value_distinct_ratio
                - prompt_schema_completion_field_value_distinct_ratio,
            )
            * 2.0
            - max(
                0.0,
                prompt_schema_completion_field_value_dominance
                - args.max_prompt_schema_completion_field_value_dominance,
            )
            - recovery_control_fraction
        )
        if arm == args.baseline_arm:
            decision = "control"
            reasons_text = ",".join(reasons)
            score_delta = 0.0
        elif not mature_enough:
            decision = "hold"
            reasons_text = ",".join(reasons)
        elif reasons:
            decision = "reject"
            reasons_text = ",".join(reasons)
        elif score_delta >= 0.0:
            decision = "promote"
            reasons_text = ""
        else:
            decision = "hold"
            reasons_text = "passes_gates_negative_score"

        item = dict(row)
        item.update(
            {
                "decision": decision,
                "fail_reasons": reasons_text,
                "mature_enough": 1.0 if mature_enough else 0.0,
                "promotion_score_delta": score_delta,
                "throughput_ratio": throughput_ratio,
                "peak_memory_ratio": peak_memory_ratio,
                "valid_ce_delta": valid_delta,
                "source_difficulty_delta": source_delta,
                "verifier_delta": verifier_delta,
                "semantic_delta": semantic_delta,
                "partial_delta": partial_delta,
                "schema_wrong_delta": schema_delta,
                "malformed_delta": malformed_delta,
                "completion_delta": completion_delta,
                "answer_field_delta": answer_field_delta,
                "answer_field_coverage_delta": answer_coverage_delta,
                "answer_termination_delta": answer_termination_delta,
                "output_entropy_delta": entropy_delta,
                "output_distinct_2_delta": distinct2_delta,
                "best_eval_verifier_baseline_delta": best_eval_verifier_baseline_delta,
                "best_eval_completion_baseline_delta": best_eval_completion_baseline_delta,
            }
        )
        out.append(item)
    return sorted(
        out,
        key=lambda row: (
            {"promote": 3, "control": 2, "hold": 1, "reject": 0}.get(str(row.get("decision")), 0),
            finite(row.get("promotion_score_delta")) or -1e9,
        ),
        reverse=True,
    )


def validation_summary(rows: list[dict[str, Any]], baseline_arm: str) -> dict[str, Any]:
    promoted = [
        str(row.get("arm") or "")
        for row in rows
        if row.get("decision") == "promote" and not row.get("fail_reasons")
    ]
    healthy = [
        str(row.get("arm") or "")
        for row in rows
        if value(row, "healthy_trial_fraction", 0.0) >= 1.0
            and value(row, "mature_enough", 1.0) >= 1.0
            and int(row.get("ok_trials") or 0) == int(row.get("trials") or 0)
    ]
    unhealthy_controls = [
        str(row.get("arm") or "")
        for row in rows
        if row.get("decision") == "control"
        and value(row, "mature_enough", 1.0) >= 1.0
        and row.get("fail_reasons")
    ]
    mature_count = sum(1 for row in rows if value(row, "mature_enough", 1.0) >= 1.0)
    if promoted:
        status = "validated_candidate"
    elif mature_count == 0:
        status = "insufficient_mature_evidence"
    else:
        status = "no_validated_candidate"
    return {
        "status": status,
        "baseline_arm": baseline_arm,
        "arm_count": len(rows),
        "mature_arm_count": mature_count,
        "healthy_arm_count": len(healthy),
        "promoted_arm_count": len(promoted),
        "rejected_arm_count": sum(1 for row in rows if row.get("decision") == "reject"),
        "hold_arm_count": sum(1 for row in rows if row.get("decision") == "hold"),
        "unhealthy_control_count": len(unhealthy_controls),
        "healthy_arms": healthy,
        "promoted_arms": promoted,
        "unhealthy_control_arms": unhealthy_controls,
    }


def write_markdown(rows: list[dict[str, Any]], out_dir: Path, baseline_arm: str) -> None:
    path = out_dir / "ruliad_promotion_matrix_summary.md"
    summary = validation_summary(rows, baseline_arm)
    with path.open("w") as handle:
        handle.write("# Ruliad Promotion Matrix\n\n")
        handle.write(f"Baseline arm: `{baseline_arm}`\n\n")
        handle.write(f"Validation status: `{summary['status']}`\n\n")
        handle.write(
            "Validated/promoted arms: "
            + (", ".join(f"`{arm}`" for arm in summary["promoted_arms"]) or "none")
            + "\n\n"
        )
        if summary["unhealthy_control_arms"]:
            handle.write(
                "Unhealthy controls: "
                + ", ".join(f"`{arm}`" for arm in summary["unhealthy_control_arms"])
                + "\n\n"
            )
        handle.write(
            "Mature arms are rejected on absolute verifier/semantic/partial floors, "
            "raw completion probe coverage, raw verifier/semantic/partial floors, "
            "per-family raw verifier floors, schema-key leakage, and raw answer/field "
            "diversity collapse in addition to relative baseline gates.\n\n"
        )
        header = "| arm | decision | mature | iters | ok/trials | healthy | seconds | peak MB | model tok/s | tput | valid CE | dCE | source diff | active d | mastered d | source lag | verifier | dver | partial | schema | dschema | field | dfield | coverage | dcov | term | dterm | completion | dcomp | best step | best v | best dbase | best dstep | best comp | best comp dbase | best comp dstep | extra dver | extra dcomp | extra dmal | contract v | contract field | contract comp | contract dver | contract dfield | cap drop | lag buckets | lag contracts | rec frac | src rec | q rec | raw q | raw exp distinct | raw actual distinct | raw ans dom | raw field ratio | raw field dom | raw field ent | ps v | ps p | ps schema | ps mal | ps field ratio | ps field dom | raw families | raw min family rows | raw worst fam v | raw worst fam partial | raw worst fam field | raw fam schema leak | raw fam ans dom | comp d2 | comp period | out d2 | answer contract expected | answer contract rows | answer contract ps rows | answer contract toks | answer contract ps toks | answer contract close | answer contract policy | recovery expected | recovery rows | recovery groups | recovery missing | recovery policy | policy expected | policy rstd | policy clip | policy applied | policy skipped | policy gated | oracle rows | oracle trunc | struct neg | gen attract policy | contrast expected | contrast pairs | contrast hard neg | contrast gen attract | bind expected | bind pairs | bind toks | bind rank toks | bind margin | bind pos | bind pair | bind gen attract | attract cap | attract obs | attract record | attract active obs | attract distinct | attract select | attract bind | attract dom | roll gen | roll cand | roll acc | roll imitate | roll recover | roll gate | roll field | policy comp | policy schema | policy health | vpo compact | vpo schema | score d | reasons |\n"
        handle.write(header)
        column_count = header.count("|") - 1
        handle.write(
            "| " + " | ".join(["---", "---"] + ["---:"] * (column_count - 2)) + " |\n"
        )
        for row in rows:
            handle.write(
                "| "
                + " | ".join(
                    [
                        str(row.get("arm") or ""),
                        str(row.get("decision") or ""),
                        fmt(row.get("mature_enough")),
                        fmt(row.get("max_iters_mean")),
                        f"{row.get('ok_trials', 0)}/{row.get('trials', 0)}",
                        fmt(row.get("healthy_trial_fraction")),
                        fmt(row.get("elapsed_seconds_mean")),
                        fmt(row.get("peak_used_mb_mean")),
                        fmt(row.get("stage_model_tokens_per_sec_mean")),
                        fmt(row.get("throughput_ratio")),
                        fmt(row.get("valid_teacher_ce_last_mean")),
                        fmt(row.get("valid_ce_delta")),
                        fmt(row.get("source_mean_difficulty_last_mean")),
                        fmt(row.get("source_active_max_difficulty_last_mean")),
                        fmt(row.get("source_mastered_max_difficulty_last_mean")),
                        fmt(row.get("source_capability_lagging_probability_last_mean")),
                        fmt(row.get("ruliad_verifier_last_mean")),
                        fmt(row.get("verifier_delta")),
                        fmt(row.get("ruliad_partial_last_mean")),
                        fmt(row.get("ruliad_schema_wrong_last_mean")),
                        fmt(row.get("schema_wrong_delta")),
                        fmt(row.get("ruliad_answer_field_accuracy_last_mean")),
                        fmt(row.get("answer_field_delta")),
                        fmt(row.get("ruliad_answer_field_coverage_last_mean")),
                        fmt(row.get("answer_field_coverage_delta")),
                        fmt(row.get("ruliad_answer_termination_rate_last_mean")),
                        fmt(row.get("answer_termination_delta")),
                        fmt(row.get("completion_health_last_mean")),
                        fmt(row.get("completion_delta")),
                        fmt(row.get("best_eval_steps_mean")),
                        fmt(row.get("best_eval_verifier_mean")),
                        fmt(row.get("best_eval_verifier_baseline_delta")),
                        fmt(row.get("best_eval_verifier_delta_mean")),
                        fmt(row.get("best_eval_completion_mean")),
                        fmt(row.get("best_eval_completion_baseline_delta")),
                        fmt(row.get("best_eval_completion_delta_mean")),
                        fmt(row.get("extra_eval_min_verifier_delta_mean")),
                        fmt(row.get("extra_eval_min_completion_delta_mean")),
                        fmt(row.get("extra_eval_max_malformed_delta_mean")),
                        fmt(row.get("contract_probe_verifier_last_mean")),
                        fmt(row.get("contract_probe_answer_field_accuracy_last_mean")),
                        fmt(row.get("contract_probe_completion_health_last_mean")),
                        fmt(row.get("contract_probe_verifier_delta_mean")),
                        fmt(row.get("contract_probe_answer_field_delta_mean")),
                        fmt(row.get("capability_score_drop_from_best_mean")),
                        fmt(row.get("capability_bucket_lagging_count_mean")),
                        fmt(row.get("capability_contract_lagging_count_mean")),
                        fmt(row.get("recovery_control_fraction_mean")),
                        fmt(row.get("source_capability_recovery_control_count_mean")),
                        fmt(row.get("capability_quality_recovery_count_mean")),
                        fmt(row.get("raw_completion_quality_mean_mean")),
                        fmt(row.get("raw_completion_expected_answer_distinct_fraction_mean")),
                        fmt(row.get("raw_completion_actual_answer_distinct_fraction_mean")),
                        fmt(row.get("raw_completion_actual_answer_dominant_fraction_mean")),
                        fmt(row.get("raw_completion_field_value_distinct_ratio_mean")),
                        fmt(row.get("raw_completion_actual_field_value_dominant_fraction_mean")),
                        fmt(row.get("raw_completion_actual_field_value_entropy_bits_mean")),
                        fmt(row.get("prompt_schema_completion_verifier_rate_mean")),
                        fmt(row.get("prompt_schema_completion_partial_rate_mean")),
                        fmt(row.get("prompt_schema_completion_schema_wrong_rate_mean")),
                        fmt(row.get("prompt_schema_completion_malformed_rate_mean")),
                        fmt(row.get("prompt_schema_completion_field_value_distinct_ratio_mean")),
                        fmt(row.get("prompt_schema_completion_actual_field_value_dominant_fraction_mean")),
                        fmt(row.get("raw_completion_family_count_mean")),
                        fmt(row.get("raw_completion_min_family_rows_mean")),
                        fmt(row.get("raw_completion_worst_family_verifier_rate_mean")),
                        fmt(row.get("raw_completion_worst_family_partial_rate_mean")),
                        fmt(row.get("raw_completion_worst_family_field_accuracy_mean")),
                        fmt(row.get("raw_completion_max_family_schema_key_mismatch_rate_mean")),
                        fmt(row.get("raw_completion_max_family_answer_dominant_fraction_mean")),
                        fmt(row.get("completion_distinct_2_last_mean")),
                        fmt(row.get("completion_period_2_to_64_last_mean")),
                        fmt(row.get("output_distinct_2_last_mean")),
                        fmt(row.get("answer_contract_config_expected_update_steps_mean")),
                        fmt(row.get("answer_contract_oracle_rows_mean")),
                        fmt(row.get("answer_contract_prompt_schema_rows_mean")),
                        fmt(row.get("answer_contract_tokens_mean")),
                        fmt(row.get("answer_contract_prompt_schema_value_tokens_mean")),
                        fmt(row.get("answer_contract_premature_close_tokens_mean")),
                        fmt(row.get("answer_contract_policy_batch_present_fraction_mean")),
                        fmt(row.get("recovery_config_expected_update_steps_mean")),
                        fmt(row.get("recovery_rows_mean")),
                        fmt(row.get("recovery_sample_groups_mean")),
                        fmt(row.get("recovery_missing_policy_batch_count_mean")),
                        fmt(row.get("recovery_policy_batch_present_fraction_mean")),
                        fmt(row.get("policy_config_expected_update_steps_mean")),
                        fmt(row.get("policy_reward_std_mean")),
                        fmt(row.get("policy_advantage_clip_fraction_mean")),
                        fmt(row.get("policy_update_applied_fraction_mean")),
                        fmt(row.get("policy_update_skipped_count_mean")),
                        fmt(row.get("policy_gated_sample_groups_mean")),
                        fmt(row.get("policy_oracle_completion_rows_mean")),
                        fmt(row.get("policy_oracle_truncated_completion_rows_mean")),
                        fmt(row.get("policy_structured_negative_completion_rows_mean")),
                        fmt(row.get("policy_generated_attractor_completion_rows_mean")),
                        fmt(row.get("contrast_config_expected_update_steps_mean")),
                        fmt(row.get("contrast_pairs_mean")),
                        fmt(row.get("contrast_template_negative_completion_rows_mean")),
                        fmt(row.get("contrast_generated_attractor_negative_completion_rows_mean")),
                        fmt(row.get("field_binding_config_expected_update_steps_mean")),
                        fmt(row.get("field_binding_contrast_pairs_mean")),
                        fmt(row.get("field_binding_discriminative_tokens_mean")),
                        fmt(row.get("field_binding_rank_metric_tokens_mean")),
                        fmt(row.get("field_binding_logit_margin_mean_mean")),
                        fmt(row.get("field_binding_positive_token_fraction_mean")),
                        fmt(row.get("field_binding_exact_pair_rank_fraction_mean")),
                        fmt(row.get("field_binding_generated_attractor_contrast_pairs_mean")),
                        fmt(row.get("generated_attractor_config_capacity_mean")),
                        fmt(row.get("generated_attractor_observed_rows_mean")),
                        fmt(row.get("generated_attractor_recorded_rows_mean")),
                        fmt(row.get("generated_attractor_active_observation_count_mean")),
                        fmt(row.get("generated_attractor_distinct_answer_count_mean")),
                        fmt(row.get("generated_attractor_selected_candidate_rows_mean")),
                        fmt(row.get("generated_attractor_selected_field_binding_pairs_mean")),
                        fmt(row.get("generated_attractor_dominant_answer_fraction_mean")),
                        fmt(row.get("rollout_imitation_generated_rows_mean")),
                        fmt(row.get("rollout_imitation_candidate_rows_mean")),
                        fmt(row.get("rollout_imitation_accepted_rows_mean")),
                        fmt(row.get("rollout_imitation_accepted_imitation_rows_mean")),
                        fmt(row.get("rollout_imitation_accepted_recovery_rows_mean")),
                        fmt(row.get("rollout_imitation_health_gate_passed_fraction_mean")),
                        fmt(row.get("rollout_imitation_field_accuracy_mean_mean")),
                        fmt(row.get("policy_vector_compactness_mean_mean")),
                        fmt(row.get("policy_vector_schema_quality_mean_mean")),
                        fmt(row.get("policy_vector_completion_health_mean_mean")),
                        fmt(row.get("policy_vpo_dominant_compactness_mean")),
                        fmt(row.get("policy_vpo_dominant_schema_quality_mean")),
                        fmt(row.get("promotion_score_delta")),
                        str(row.get("fail_reasons") or ""),
                    ]
                )
                + " |\n"
            )
    print(path)


def write_validation_summary(rows: list[dict[str, Any]], out_dir: Path, baseline_arm: str) -> None:
    path = out_dir / "ruliad_promotion_matrix_validation.json"
    path.write_text(json.dumps(validation_summary(rows, baseline_arm), indent=2) + "\n")
    print(path)


def main() -> None:
    args = parse_args()
    root = Path(args.input)
    out_dir = Path(args.out_dir) if args.out_dir else root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    trials = collect_trials(root)
    if not trials:
        raise SystemExit(f"no arm summaries found under {root}")
    arms = summarize_by_arm(trials)
    gated = add_gate_decisions(arms, args)
    arm_fields = (
        [
            "arm",
            "decision",
            "fail_reasons",
            "mature_enough",
            "trials",
            "ok_trials",
            "healthy_trial_fraction",
        ]
        + [
            f"{column}_mean"
            for column in (
                METRIC_COLUMNS
                + POLICY_METRIC_COLUMNS
                + POLICY_CONFIG_COLUMNS
                + ANSWER_CONTRACT_CONFIG_COLUMNS
                + ANSWER_CONTRACT_METRIC_COLUMNS
                + STRUCTURED_RECOVERY_CONFIG_COLUMNS
                + STRUCTURED_RECOVERY_METRIC_COLUMNS
                + STRUCTURED_CONTRAST_CONFIG_COLUMNS
                + STRUCTURED_CONTRAST_METRIC_COLUMNS
                + FIELD_BINDING_CONTRAST_CONFIG_COLUMNS
                + FIELD_BINDING_CONTRAST_METRIC_COLUMNS
                + GENERATED_ATTRACTOR_CONFIG_COLUMNS
                + GENERATED_ATTRACTOR_METRIC_COLUMNS
                + VERIFIER_ROLLOUT_METRIC_COLUMNS
                + RAW_COMPLETION_METRIC_COLUMNS
                + PROMPT_SCHEMA_COMPLETION_METRIC_COLUMNS
            )
        ]
        + [
            "promotion_score_delta",
            "throughput_ratio",
            "peak_memory_ratio",
            "valid_ce_delta",
            "source_difficulty_delta",
            "verifier_delta",
            "semantic_delta",
            "partial_delta",
            "schema_wrong_delta",
            "malformed_delta",
            "completion_delta",
            "answer_field_delta",
            "answer_field_coverage_delta",
            "answer_termination_delta",
            "best_eval_verifier_baseline_delta",
            "best_eval_completion_baseline_delta",
            "output_entropy_delta",
            "output_distinct_2_delta",
        ]
    )
    trial_fields = ["arm"] + [field for field in trials[0].keys() if field != "arm"]
    write_csv(out_dir / "ruliad_promotion_matrix_trials.csv", trials, trial_fields)
    write_csv(out_dir / "ruliad_promotion_matrix_arm_summary.csv", gated, arm_fields)
    write_markdown(gated, out_dir, args.baseline_arm)
    write_validation_summary(gated, out_dir, args.baseline_arm)
    print(out_dir / "ruliad_promotion_matrix_arm_summary.csv")
    print(out_dir / "ruliad_promotion_matrix_trials.csv")


if __name__ == "__main__":
    main()
