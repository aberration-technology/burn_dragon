#!/usr/bin/env python3
"""Aggregate Dragon predictive-coding paper runs.

The analyzer intentionally uses only the Python standard library so it can run
on benchmark hosts without a notebook or pandas environment. It accepts raw
summary CSVs from older runs plus Dragon ECS event JSONL directories from newer
runs, then emits small CSV/Markdown tables suitable for the paper draft.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

LOSS_SERIES_PREFIXES = ("fixed_holdout", "source_weighted", "stream_warm")
LOSS_SERIES_EVENT_SUFFIXES = (
    "checkpoints",
    "first_checkpoint",
    "best",
    "best_epoch",
    "final_minus_best",
    "regression_fraction",
    "slope_per_checkpoint",
)
LOSS_SERIES_ANALYSIS_SUFFIXES = tuple(
    suffix
    for suffix in LOSS_SERIES_EVENT_SUFFIXES
    if suffix not in {"checkpoints", "best_epoch"}
)


SUMMARY_COLUMNS = [
    "run",
    "matrix",
    "backend",
    "profile",
    "iters",
    "arm",
    "seed",
    "batch_size",
    "checkpoint_interval_iters",
    "wall_clock_seconds",
    "ruliad_policy_probe_every_epochs",
    "source_selection_feedback_updates_enabled",
    "proof_policy_mode",
    "wall_s",
    "tok_s",
    "model_tok_s",
    "model_duty_fraction",
    "train_first",
    "train_last",
    "valid_last",
    "valid_mean",
    "validation_objective",
    "validation_sampling",
    "ruliad_panel_base_difficulty_levels",
    "validation_loss_first_checkpoint",
    "validation_loss_best",
    "validation_loss_final_minus_best",
    "validation_loss_regression_fraction",
    "validation_loss_slope_per_checkpoint",
    *(
        f"{prefix}_loss_{suffix}"
        for prefix in LOSS_SERIES_PREFIXES
        for suffix in LOSS_SERIES_ANALYSIS_SUFFIXES
    ),
    "valid_supervised_tokens",
    "valid_supervised_batches",
    "valid_empty_supervision_batches",
    "stream_warm_loss",
    "validation_objective_loss",
    "stream_paired_warm_loss",
    "stream_paired_cold_loss",
    "stream_carry_nll_gain",
    "stream_carry_relative_gain",
    "lr_last",
    "pc_ms_mean",
    "pc_corrected_fraction",
    "source_loss",
    "source_loss_cadence_mean",
    "source_mean_difficulty",
    "source_active_max_difficulty",
    "source_curriculum_released_max_difficulty",
    "source_active_max_difficulty_probability",
    "source_norm_difficulty",
    "verifier_failures",
    "ruliad_verifier_accuracy",
    "ruliad_verifier_accuracy_best",
    "ruliad_verifier_accuracy_final_minus_best",
    "ruliad_constrained_equivalent_top1",
    "ruliad_constrained_free_accuracy_gap",
    "ruliad_constrained_equivalent_nll",
    "ruliad_constrained_valid_invalid_margin",
    "ruliad_context_swap_top1_change_rate",
    "ruliad_context_swap_equivalent_probability_drop",
    "ruliad_context_swap_js_divergence",
    "ruliad_counterfactual_target_equivalent_top1",
    "ruliad_counterfactual_target_top1_change_rate",
    "ruliad_counterfactual_target_equivalent_probability_gain",
    "ruliad_counterfactual_target_js_divergence",
    "ruliad_partial_progress",
    "ruliad_policy_rollout_items",
    "ruliad_policy_solve_rate",
    "ruliad_policy_solve_rate_best",
    "ruliad_policy_solve_rate_final_minus_best",
    "ruliad_policy_solve_rate_promoted",
    "ruliad_policy_goal_completion_rate",
    "ruliad_policy_goal_completion_rate_best",
    "ruliad_policy_goal_completion_rate_final_minus_best",
    "ruliad_policy_goal_completion_rate_promoted",
    "ruliad_policy_valid_action_rate",
    "ruliad_policy_top1_expert_rate",
    "ruliad_policy_top1_expert_rate_promoted",
    "ruliad_policy_promotion_gate_passed",
    "ruliad_deployment_capability_gate_passed",
    "checkpoint_promoted_count",
    "checkpoint_last_epoch",
    "checkpoint_last_absolute_step",
    "checkpoint_last_promoted_epoch",
    "checkpoint_promotion_ineligible_count",
    "capability_statistical_regression_count",
    "capability_quality_regression_count",
    "dynamics_control_count",
    "rollback_recovery_count",
    "source_capability_recovery_count",
    "validation_recovery_count",
    "capability_allowed_max_difficulty",
    "output_entropy_bits",
    "output_distinct_2_fraction",
]

# Keep aggregation, paired comparisons, and CSV serialization on one schema.
# A metric added only to the event parser is otherwise silently omitted from
# the paper artifacts, which is especially dangerous for stability evidence.
ANALYSIS_METRICS = [
    "wall_s",
    "tok_s",
    "model_tok_s",
    "model_duty_fraction",
    "train_last",
    "valid_last",
    "valid_mean",
    "validation_loss_first_checkpoint",
    "validation_loss_best",
    "validation_loss_final_minus_best",
    "validation_loss_regression_fraction",
    "validation_loss_slope_per_checkpoint",
    *(
        f"{prefix}_loss_{suffix}"
        for prefix in LOSS_SERIES_PREFIXES
        for suffix in LOSS_SERIES_ANALYSIS_SUFFIXES
    ),
    "valid_supervised_tokens",
    "valid_supervised_batches",
    "valid_empty_supervision_batches",
    "stream_warm_loss",
    "validation_objective_loss",
    "stream_paired_warm_loss",
    "stream_paired_cold_loss",
    "stream_carry_nll_gain",
    "stream_carry_relative_gain",
    "lr_last",
    "source_loss",
    "source_loss_cadence_mean",
    "source_mean_difficulty",
    "source_active_max_difficulty",
    "source_curriculum_released_max_difficulty",
    "source_active_max_difficulty_probability",
    "source_norm_difficulty",
    "ruliad_verifier_accuracy",
    "ruliad_verifier_accuracy_best",
    "ruliad_verifier_accuracy_final_minus_best",
    "ruliad_constrained_equivalent_top1",
    "ruliad_constrained_free_accuracy_gap",
    "ruliad_constrained_equivalent_nll",
    "ruliad_constrained_valid_invalid_margin",
    "ruliad_context_swap_top1_change_rate",
    "ruliad_context_swap_equivalent_probability_drop",
    "ruliad_context_swap_js_divergence",
    "ruliad_counterfactual_target_equivalent_top1",
    "ruliad_counterfactual_target_top1_change_rate",
    "ruliad_counterfactual_target_equivalent_probability_gain",
    "ruliad_counterfactual_target_js_divergence",
    "ruliad_partial_progress",
    "ruliad_policy_rollout_items",
    "ruliad_policy_solve_rate",
    "ruliad_policy_solve_rate_best",
    "ruliad_policy_solve_rate_final_minus_best",
    "ruliad_policy_solve_rate_promoted",
    "ruliad_policy_goal_completion_rate",
    "ruliad_policy_goal_completion_rate_best",
    "ruliad_policy_goal_completion_rate_final_minus_best",
    "ruliad_policy_goal_completion_rate_promoted",
    "ruliad_policy_valid_action_rate",
    "ruliad_policy_top1_expert_rate",
    "ruliad_policy_top1_expert_rate_promoted",
    "ruliad_policy_promotion_gate_passed",
    "ruliad_deployment_capability_gate_passed",
    "checkpoint_promoted_count",
    "checkpoint_last_epoch",
    "checkpoint_last_absolute_step",
    "checkpoint_last_promoted_epoch",
    "checkpoint_promotion_ineligible_count",
    "capability_statistical_regression_count",
    "capability_quality_regression_count",
    "dynamics_control_count",
    "rollback_recovery_count",
    "source_capability_recovery_count",
    "validation_recovery_count",
    "capability_allowed_max_difficulty",
    "output_entropy_bits",
    "output_distinct_2_fraction",
    "pc_ms_mean",
]

EVENT_SUMMARY_COLUMNS = [
    "run",
    "run_dir",
    "trial_key",
    "matrix",
    "iters",
    "arm",
    "seed",
    "status",
    "elapsed_seconds",
    "batch_size",
    "checkpoint_interval_iters",
    "wall_clock_seconds",
    "ruliad_policy_probe_every_epochs",
    "backend",
    "profile",
    "source_selection_feedback_updates_enabled",
    "proof_policy_mode",
    "training_wall_seconds",
    "train_tokens",
    "source_batches",
    "source_tokens",
    "train_steps",
    "structured_terminal_steps",
    "structured_terminal_rows",
    "structured_terminal_padded_tokens",
    "stream_advance_ns",
    "wall_tokens_per_second",
    "model_tokens_per_second",
    "model_duty_fraction",
    "train_compute_fraction",
    "optimizer_fraction",
    "dataloader_foreground_wait_fraction",
    "host_sync_points",
    "train_loss_first",
    "train_loss_last",
    "lr_last",
    "valid_loss_last",
    "valid_loss_mean",
    "validation_checkpoints",
    "validation_loss_first_checkpoint",
    "validation_loss_best",
    "validation_loss_best_epoch",
    "validation_loss_final_minus_best",
    "validation_loss_regression_fraction",
    "validation_loss_slope_per_checkpoint",
    "validation_objective_kind_last",
    *(
        f"{prefix}_loss_{suffix}"
        for prefix in LOSS_SERIES_PREFIXES
        for suffix in LOSS_SERIES_EVENT_SUFFIXES
    ),
    "valid_supervised_tokens_last",
    "valid_supervised_batches_last",
    "valid_empty_supervision_batches_last",
    "stream_warm_loss_mean",
    "validation_objective_loss_last",
    "stream_paired_warm_loss_last",
    "stream_paired_cold_loss_last",
    "stream_carry_nll_gain_last",
    "stream_carry_relative_gain_last",
    "source_loss_last",
    "source_loss_cadence_mean",
    "source_loss_observations",
    "source_entropy_bits_last",
    "source_mean_difficulty_last",
    "source_active_max_difficulty_last",
    "source_curriculum_released_max_difficulty_last",
    "source_active_max_difficulty_probability_last",
    "source_norm_difficulty_last",
    "source_mastered_probability_last",
    "source_capability_allowed_max_difficulty_last",
    "source_verifier_failures_last",
    "ruliad_verifier_accuracy_last",
    "ruliad_verifier_accuracy_best",
    "ruliad_verifier_accuracy_final_minus_best",
    "ruliad_constrained_items_last",
    "ruliad_constrained_equivalent_top1_last",
    "ruliad_constrained_preferred_top1_last",
    "ruliad_constrained_equivalent_nll_last",
    "ruliad_constrained_valid_invalid_margin_last",
    "ruliad_constrained_worst_presentation_top1_last",
    "ruliad_constrained_orbit_js_last",
    "ruliad_context_swap_items_last",
    "ruliad_context_swap_top1_change_rate_last",
    "ruliad_context_swap_equivalent_probability_drop_last",
    "ruliad_context_swap_js_divergence_last",
    "ruliad_counterfactual_target_items_last",
    "ruliad_counterfactual_target_equivalent_top1_last",
    "ruliad_counterfactual_target_top1_change_rate_last",
    "ruliad_counterfactual_target_equivalent_probability_gain_last",
    "ruliad_counterfactual_target_js_divergence_last",
    "ruliad_partial_progress_last",
    "ruliad_policy_rollout_items_last",
    "ruliad_policy_solve_rate_last",
    "ruliad_policy_solve_rate_best",
    "ruliad_policy_solve_rate_final_minus_best",
    "ruliad_policy_solve_rate_promoted",
    "ruliad_policy_goal_completion_rate_last",
    "ruliad_policy_goal_completion_rate_best",
    "ruliad_policy_goal_completion_rate_final_minus_best",
    "ruliad_policy_goal_completion_rate_promoted",
    "ruliad_policy_valid_action_rate_last",
    "ruliad_policy_top1_expert_rate_last",
    "ruliad_policy_top1_expert_rate_promoted",
    "ruliad_policy_promotion_gate_passed_last",
    "ruliad_deployment_capability_gate_passed_last",
    "checkpoint_count",
    "checkpoint_last_epoch",
    "checkpoint_last_absolute_step",
    "checkpoint_promoted_count",
    "checkpoint_last_promoted_epoch",
    "checkpoint_promotion_ineligible_count",
    "capability_statistical_regression_count",
    "capability_quality_regression_count",
    "dynamics_control_count",
    "rollback_recovery_count",
    "source_capability_recovery_count",
    "validation_recovery_count",
    "output_entropy_bits_last",
    "output_mean_max_probability_last",
    "output_distinct_2_fraction_last",
    "output_repetition_fraction_last",
    "output_period_2_to_64_fraction_last",
    "sequence_state_rho_rms_last",
    "sequence_state_rho_slot_redundancy_last",
    "sequence_state_rho_slot_variance_ratio_last",
    "gate_count",
    "fatal_gate_count",
    "capacity_scale_count",
    "pc_event_count",
    "pc_ms_mean",
    "pc_learning_contract_last",
    "pc_execution_contract_version_last",
    "pc_activity_derivative_contract_last",
    "pc_parameter_derivative_contract_last",
    "pc_global_autodiff_graph_last",
    "pc_global_backward_calls_total",
    "pc_local_vjp_calls_total",
    "pc_temporal_state_vjp_calls_total",
    "pc_fused_temporal_vjp_calls_total",
    "pc_temporal_credit_mode_last",
    "pc_temporal_window_chunks_last",
    "pc_direct_forward_updates_total",
    "pc_feedback_parameter_updates_total",
    "pc_adjoint_teacher_updates_total",
    "pc_adjoint_local_updates_total",
    "pc_adjoint_calibration_samples_total",
    "pc_adjoint_calibration_loss_last",
    "pc_adjoint_cosine_alignment_last",
    "pc_adjoint_prediction_teacher_norm_ratio_last",
    "pc_adjoint_update_rms_last",
    "pc_local_parameter_update_intents_total",
    "pc_parameter_updates_total",
    "pc_structured_terminal_steps_total",
    "pc_structured_terminal_skipped_steps_total",
    "pc_structured_terminal_groups_total",
    "pc_structured_terminal_rows_total",
    "pc_factors_last",
    "pc_gradient_tensors_last",
    "pc_clip_fraction_mean_last",
    "pc_constraint_rms_last",
    "pc_dual_rms_last",
    "pc_composite_signal_rms_last",
    "pc_observation_contract_last",
    "pc_deployment_aligned_last",
    "pc_amortization_components_last",
    "pc_amortization_loss_last",
]

CAPABILITY_COVERAGE_COLUMNS = [
    "run",
    "absolute_step",
    "difficulty_level",
    "candidate_coverage",
    "family_coverage",
    "task_coverage",
    "contract_coverage",
    "observed_items",
    "mastered",
]

CAPABILITY_GROUP_COLUMNS = [
    "run",
    "epoch",
    "absolute_step",
    "probe_name",
    "kind",
    "label",
    "item_count",
    "exact_rate",
    "semantic_rate",
    "verifier_rate",
    "partial_credit_rate",
    "schema_valid_wrong_rate",
    "malformed_rate",
    "missing_rate",
    "mean_partial_progress",
    "answer_field_accuracy",
    "answer_field_coverage",
    "answer_termination_rate",
]

BUCKET_COLUMNS = [
    "run",
    "absolute_step",
    "kind",
    "label",
    "family",
    "task_kind",
    "difficulty_level",
    "candidate_count",
    "probability",
    "loss_ema",
    "mean_loss",
    "learning_progress",
    "mastered",
    "mastered_probability",
    "mean_difficulty_level",
    "capability_feedback_probability",
    "capability_verifier_ema",
    "capability_completion_health_ema",
    "capability_schema_wrong_ema",
    "capability_malformed_ema",
    "capability_missing_ema",
    "capability_lagging_probability",
]

GPU_COLUMNS = [
    "file",
    "trial_key",
    "arm",
    "seed",
    "iters",
    "samples",
    "util_mean",
    "util_p10",
    "util_p50",
    "util_p90",
    "power_mean",
    "power_p10",
    "power_p50",
    "power_p90",
]

MANIFEST_COLUMNS = [
    "trial_key",
    "matrix",
    "iters",
    "arm",
    "behavior_arm",
    "seed",
    "batch_size",
    "checkpoint_interval_iters",
    "wall_clock_seconds",
    "defer_expensive_ruliad_probes",
    "block_size",
    "local_learning_rate",
    "learning_rate_schedule",
    "cosine_min_lr",
    "cosine_warmup_steps",
    "tbptt_chunk_size",
    "tbptt_credit_window_chunks",
    "model_sequence_executor",
    "verifier_every_steps",
    "proof_policy_scoring",
    "proof_policy_mode",
    "proof_policy_gradient_scope",
    "proof_policy_normalization",
    "proof_policy_candidate_symmetry",
    "proof_policy_presentation_risk",
    "policy_probe_normalization",
    "policy_probe_candidate_symmetry",
    "proof_policy_counterfactual_targets",
    "proof_policy_semantic_refresh_every",
    "proof_policy_semantic_refresh_counterfactual_targets",
    "tbptt_persist_across_steps",
    "sequence_batching",
    "sequence_state_probe",
    "sequence_state_probe_paired_batches",
    "source_selection_feedback_updates_enabled",
    "validation_objective",
    "validation_sampling",
    "ruliad_panel_base_difficulty_levels",
    "ruliad_policy_probe_every_epochs",
    "ruliad_policy_probe_closed_loop_every_epochs",
    "backend",
    "features",
    "profile",
    "overlay",
    "run_root",
    "run_dir",
    "log_path",
    "gpu_path",
    "status",
    "elapsed_seconds",
    "peak_used_mb",
    "min_available_mb",
    "git_sha",
    "git_branch",
    "git_dirty",
    "clean_git_required",
    "train_binary_sha256",
    "runner_sha256",
]


@dataclass
class MetricStats:
    count: int
    mean: float
    std: float
    ci95: float

    def fmt(self) -> str:
        if self.count == 0 or not math.isfinite(self.mean):
            return ""
        if self.count == 1:
            return f"{self.mean:.4g}"
        return f"{self.mean:.4g} +/- {self.ci95:.3g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        default=["target/pc-paper"],
        help="CSV files, event JSONL files, or directories to scan.",
    )
    parser.add_argument(
        "--out-dir",
        default="target/pc-paper/analysis",
        help="Directory for generated CSV/Markdown summaries.",
    )
    parser.add_argument(
        "--baseline",
        default="adamw",
        help="Baseline arm for paired deltas.",
    )
    parser.add_argument(
        "--compare",
        default="adamwpc",
        help="Comparison arm for paired deltas.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a small analyzer self-test and exit.",
    )
    return parser.parse_args()


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "[n/a]", "n/a", "none"}:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def as_int(value: Any) -> int | None:
    parsed = as_float(value)
    if parsed is None:
        return None
    return int(parsed)


EXPERIMENT_CONTEXT_FIELDS = (
    "matrix",
    "backend",
    "profile",
    "iters",
    "batch_size",
    "checkpoint_interval_iters",
    "wall_clock_seconds",
    "ruliad_policy_probe_every_epochs",
    "source_selection_feedback_updates_enabled",
    "proof_policy_mode",
)


def experiment_context(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in EXPERIMENT_CONTEXT_FIELDS)


def experiment_context_row(context: tuple[Any, ...]) -> dict[str, Any]:
    return dict(zip(EXPERIMENT_CONTEXT_FIELDS, context, strict=True))


def experiment_context_sort_key(context: tuple[Any, ...]) -> tuple[Any, ...]:
    (
        matrix,
        backend,
        profile,
        iters,
        batch_size,
        checkpoint_interval,
        wall_clock_seconds,
        policy_probe_cadence,
        source_feedback,
        policy_mode,
    ) = context
    return (
        str(matrix or ""),
        str(backend or ""),
        str(profile or ""),
        as_int(iters) or -1,
        as_int(batch_size) or -1,
        as_int(checkpoint_interval) or -1,
        as_int(wall_clock_seconds) or 0,
        as_int(policy_probe_cadence) or -1,
        str(source_feedback or ""),
        str(policy_mode or ""),
    )


def stats(values: Iterable[float | None]) -> MetricStats:
    clean = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not clean:
        return MetricStats(0, math.nan, math.nan, math.nan)
    if len(clean) == 1:
        return MetricStats(1, clean[0], 0.0, 0.0)
    std = statistics.stdev(clean)
    critical = student_t_critical_95(len(clean) - 1)
    return MetricStats(
        len(clean),
        statistics.mean(clean),
        std,
        critical * std / math.sqrt(len(clean)),
    )


def linear_slope(values: list[float]) -> float:
    """Least-squares slope over equally spaced observations."""
    if len(values) < 2:
        return 0.0
    center = (len(values) - 1) / 2.0
    denominator = sum((index - center) ** 2 for index in range(len(values)))
    if denominator == 0.0:
        return 0.0
    mean = statistics.mean(values)
    return sum(
        (index - center) * (value - mean) for index, value in enumerate(values)
    ) / denominator


def summarize_loss_series(
    summary: dict[str, Any], by_epoch: dict[int, float], prefix: str
) -> None:
    points = sorted(by_epoch.items())
    if not points:
        return
    values = [value for _, value in points]
    best_index = min(range(len(values)), key=values.__getitem__)
    best_loss = values[best_index]
    final_loss = values[-1]
    checkpoint_key = (
        "validation_checkpoints" if prefix == "validation" else f"{prefix}_loss_checkpoints"
    )
    summary[checkpoint_key] = len(values)
    summary[f"{prefix}_loss_first_checkpoint"] = values[0]
    summary[f"{prefix}_loss_best"] = best_loss
    summary[f"{prefix}_loss_best_epoch"] = points[best_index][0]
    summary[f"{prefix}_loss_final_minus_best"] = final_loss - best_loss
    summary[f"{prefix}_loss_regression_fraction"] = (
        (final_loss - best_loss) / best_loss if best_loss > 0.0 else ""
    )
    summary[f"{prefix}_loss_slope_per_checkpoint"] = linear_slope(values)


def student_t_critical_95(degrees_of_freedom: int) -> float:
    """Two-sided 95% Student-t critical value for small experiment matrices."""
    table = (
        12.706,
        4.303,
        3.182,
        2.776,
        2.571,
        2.447,
        2.365,
        2.306,
        2.262,
        2.228,
        2.201,
        2.179,
        2.160,
        2.145,
        2.131,
        2.120,
        2.110,
        2.101,
        2.093,
        2.086,
        2.080,
        2.074,
        2.069,
        2.064,
        2.060,
        2.056,
        2.052,
        2.048,
        2.045,
        2.042,
    )
    if degrees_of_freedom <= 0:
        return math.nan
    if degrees_of_freedom <= len(table):
        return table[degrees_of_freedom - 1]
    return 1.96


def percentile(values: list[float], q: float) -> float | None:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return clean[lo]
    frac = pos - lo
    return clean[lo] * (1.0 - frac) + clean[hi] * frac


def discover_inputs(paths: Iterable[str]) -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    summary_csvs: list[Path] = []
    gpu_csvs: list[Path] = []
    event_jsonls: list[Path] = []
    manifest_jsons: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            continue
        candidates = [path]
        if path.is_dir():
            candidates = sorted(path.rglob("*"))
        for candidate in candidates:
            if not candidate.is_file():
                continue
            if is_generated_analysis_file(candidate):
                continue
            name = candidate.name
            if name.endswith(".csv") and "gpu" in name:
                gpu_csvs.append(candidate)
            elif name.endswith(".csv") and "summary" in name:
                summary_csvs.append(candidate)
            elif name in {"training_events.jsonl", "source_selection.jsonl", "capacity_scaling.jsonl"}:
                event_jsonls.append(candidate)
            elif name.endswith(".json") and candidate.parent.name == "manifests":
                manifest_jsons.append(candidate)
    return (
        sorted(set(summary_csvs)),
        sorted(set(gpu_csvs)),
        sorted(set(event_jsonls)),
        sorted(set(manifest_jsons)),
    )


def is_generated_analysis_file(path: Path) -> bool:
    generated_names = {
        "normalized_summary.csv",
        "summary_by_arm.csv",
        "paired_deltas.csv",
        "event_run_summary.csv",
        "source_bucket_summary.csv",
        "source_capability_coverage.csv",
        "gpu_summary.csv",
        "manifest_summary.csv",
    }
    if path.name in generated_names:
        return True
    return any(part == "analysis" for part in path.parts)


def normalize_summary_row(row: dict[str, str]) -> dict[str, Any]:
    normalized: dict[str, Any] = {key: "" for key in SUMMARY_COLUMNS}
    normalized["run"] = row.get("run", "")
    source_feedback = row.get("source_selection_feedback_updates_enabled", "")
    normalized["matrix"] = canonical_matrix(row.get("matrix", ""), source_feedback)
    normalized["backend"] = row.get("backend", "")
    normalized["profile"] = row.get("profile", "")
    normalized["iters"] = as_int(row.get("iters")) or infer_iters_from_run(normalized["run"])
    normalized["arm"] = row.get("arm", "")
    normalized["seed"] = as_int(row.get("seed"))
    normalized["batch_size"] = as_int(row.get("batch_size"))
    normalized["checkpoint_interval_iters"] = as_int(
        row.get("checkpoint_interval_iters")
    )
    normalized["ruliad_policy_probe_every_epochs"] = as_int(
        row.get("ruliad_policy_probe_every_epochs")
    )
    normalized["source_selection_feedback_updates_enabled"] = source_feedback
    normalized["proof_policy_mode"] = row.get("proof_policy_mode", "") or infer_proof_policy_mode(
        normalized["arm"]
    )
    aliases = {
        "wall_s": ["wall_s", "wall"],
        "tok_s": ["tok_s"],
        "train_first": ["train_first"],
        "train_last": ["train_last"],
        "valid_last": ["valid_last"],
        "valid_mean": ["valid_mean", "valid_last"],
        "stream_warm_loss": ["stream_warm_loss"],
        "validation_objective_loss": ["validation_objective_loss"],
        "stream_paired_warm_loss": ["stream_paired_warm_loss"],
        "stream_paired_cold_loss": ["stream_paired_cold_loss"],
        "stream_carry_nll_gain": ["stream_carry_nll_gain"],
        "stream_carry_relative_gain": ["stream_carry_relative_gain"],
        "lr_last": ["lr_last"],
        "pc_ms_mean": ["pc_ms_mean"],
        "pc_corrected_fraction": ["pc_corrected_fraction", "pc_events"],
        "source_loss": ["source_loss", "src_loss"],
        "source_mean_difficulty": ["source_mean_difficulty", "src_mean_diff"],
        "source_norm_difficulty": ["source_norm_difficulty", "src_norm_diff"],
        "verifier_failures": ["verifier_failures", "src_verifier_failures"],
        "ruliad_verifier_accuracy": ["ruliad_verifier_accuracy"],
        "ruliad_partial_progress": ["ruliad_partial_progress"],
        "capability_allowed_max_difficulty": [
            "capability_allowed_max_difficulty"
        ],
        "output_entropy_bits": ["output_entropy_bits"],
        "output_distinct_2_fraction": ["output_distinct_2_fraction"],
    }
    for out_key, in_keys in aliases.items():
        for in_key in in_keys:
            if in_key in row and row[in_key] != "":
                normalized[out_key] = as_float(row[in_key])
                break
    return normalized


def infer_iters_from_run(run: str) -> int | None:
    for part in run.split("-"):
        if part.isdigit():
            value = int(part)
            if value in {512, 2048, 8192} or value > 0:
                return value
    return None


def infer_proof_policy_mode(arm: str) -> str:
    """Recover the policy contract from pre-field experiment arm names."""
    if "paired_dagger" in arm:
        return "static_then_paired_dagger"
    if "verifier_dagger" in arm:
        return "dagger"
    if "verifier_static" in arm:
        return "static_expert"
    return ""


def canonical_matrix(matrix: str, source_feedback: Any) -> str:
    """Correct the pre-taxonomy label for feedback-frozen closed-loop runs."""
    feedback_is_false = source_feedback is False or str(source_feedback).strip().lower() in {
        "0",
        "false",
    }
    if matrix == "local-verifier-closed-loop" and feedback_is_false:
        return "local-verifier-source-frozen"
    return matrix


def read_summary_csvs(paths: Iterable[Path]) -> list[dict[str, Any]]:
    keyed: dict[tuple[Any, ...], dict[str, Any]] = {}
    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                normalized = normalize_summary_row(row)
                key = (
                    *experiment_context(normalized),
                    normalized.get("arm"),
                    normalized.get("seed"),
                    normalized.get("run"),
                )
                existing = keyed.get(key)
                if existing is None or populated_count(normalized) > populated_count(existing):
                    keyed[key] = normalized
    return sorted(
        keyed.values(),
        key=lambda row: (
            *experiment_context_sort_key(experiment_context(row)),
            str(row.get("arm") or ""),
            row.get("seed") or -1,
            str(row.get("run") or ""),
        ),
    )


def populated_count(row: dict[str, Any]) -> int:
    return sum(1 for value in row.values() if value not in ("", None))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                print(f"warning: skipping malformed JSON in {path}: {error}", file=sys.stderr)


def update_metric(summary: dict[str, Any], event: dict[str, Any]) -> None:
    split = event.get("split")
    name = event.get("name")
    value = as_float(event.get("value"))
    running_value = as_float(event.get("running_value"))
    if value is None:
        return
    epoch = as_int(event.get("epoch"))
    if split == "train" and name in {"Loss", "Stream Warm Loss"}:
        if summary.get("train_loss_first", "") == "":
            summary["train_loss_first"] = value
        summary["train_loss_last"] = value
    elif split == "train" and name == "Learning Rate":
        summary["lr_last"] = value
    elif split == "valid" and name == "Loss":
        summary["valid_loss_last"] = value
        summary["valid_loss_mean"] = running_value if running_value is not None else value
        if epoch is not None:
            summary["_fixed_holdout_loss_by_epoch"][epoch] = (
                running_value if running_value is not None else value
            )
    elif split == "valid" and name == "Random Cold Loss":
        summary["valid_loss_mean"] = value
        if epoch is not None:
            summary["_fixed_holdout_loss_by_epoch"][epoch] = value
    elif split == "valid" and name == "Validation Supervised Tokens":
        summary["valid_supervised_tokens_last"] = value
    elif split == "valid" and name == "Validation Supervised Batches":
        summary["valid_supervised_batches_last"] = value
    elif split == "valid" and name == "Validation Empty Supervision Batches":
        summary["valid_empty_supervision_batches_last"] = value
    elif split == "valid" and name == "Stream Warm Loss":
        summary["stream_warm_loss_mean"] = (
            running_value if running_value is not None else value
        )
        if epoch is not None:
            summary["_stream_warm_loss_by_epoch"][epoch] = (
                running_value if running_value is not None else value
            )
    elif split == "valid" and name == "Source Weighted Loss":
        if epoch is not None:
            summary["_source_weighted_loss_by_epoch"][epoch] = (
                running_value if running_value is not None else value
            )
    elif split == "valid" and name == "Stream Paired Warm Loss":
        summary["stream_paired_warm_loss_last"] = value
    elif split == "valid" and name == "Stream Paired Cold Loss":
        summary["stream_paired_cold_loss_last"] = value
    elif split == "valid" and name == "Stream Carry NLL Gain":
        summary["stream_carry_nll_gain_last"] = value
    elif split == "valid" and name == "Stream Carry Relative Gain":
        summary["stream_carry_relative_gain_last"] = value
    elif split == "valid" and name == "Ruliad Verifier Accuracy":
        summary["ruliad_verifier_accuracy_last"] = value
        epoch = as_int(event.get("epoch"))
        if epoch is not None:
            summary["_ruliad_verifier_by_epoch"][epoch] = value
    elif split == "valid" and name == "Ruliad Policy Rollout Items":
        summary["ruliad_policy_rollout_items_last"] = value
    elif split == "valid" and name == "Ruliad Policy Rollout Solve Rate":
        summary["ruliad_policy_solve_rate_last"] = value
        if epoch is not None:
            summary["_ruliad_policy_solve_by_epoch"][epoch] = value
    elif split == "valid" and name == "Ruliad Policy Rollout Goal Completion Rate":
        summary["ruliad_policy_goal_completion_rate_last"] = value
        if epoch is not None:
            summary["_ruliad_policy_goal_by_epoch"][epoch] = value
    elif split == "valid" and name == "Ruliad Policy Rollout Valid Action Rate":
        summary["ruliad_policy_valid_action_rate_last"] = value
        if epoch is not None:
            summary["_ruliad_policy_valid_action_by_epoch"][epoch] = value
    elif split == "valid" and name == "Ruliad Policy Model Top-1 Expert Rate":
        summary["ruliad_policy_top1_expert_rate_last"] = value
        if epoch is not None:
            summary["_ruliad_policy_top1_by_epoch"][epoch] = value
    elif split == "valid" and name == "Ruliad Policy Promotion Gate Passed":
        summary["ruliad_policy_promotion_gate_passed_last"] = value
    elif split == "valid" and name == "Ruliad Deployment Capability Gate Passed":
        summary["ruliad_deployment_capability_gate_passed_last"] = value
    elif split == "valid" and name == "Ruliad Correctness Constrained Items":
        summary["ruliad_constrained_items_last"] = value
    elif (
        split == "valid"
        and name == "Ruliad Correctness Constrained Equivalent Top-1 Rate"
    ):
        summary["ruliad_constrained_equivalent_top1_last"] = value
    elif (
        split == "valid"
        and name == "Ruliad Correctness Constrained Preferred Top-1 Rate"
    ):
        summary["ruliad_constrained_preferred_top1_last"] = value
    elif split == "valid" and name == "Ruliad Correctness Constrained Equivalent NLL":
        summary["ruliad_constrained_equivalent_nll_last"] = value
    elif (
        split == "valid"
        and name == "Ruliad Correctness Constrained Valid-Invalid Margin"
    ):
        summary["ruliad_constrained_valid_invalid_margin_last"] = value
    elif (
        split == "valid"
        and name
        == "Ruliad Correctness Constrained Worst-Presentation Equivalent Top-1 Rate"
    ):
        summary["ruliad_constrained_worst_presentation_top1_last"] = value
    elif (
        split == "valid"
        and name == "Ruliad Correctness Constrained Orbit JS Divergence"
    ):
        summary["ruliad_constrained_orbit_js_last"] = value
    elif split == "valid" and name == "Ruliad Correctness Constrained Context-Swap Items":
        summary["ruliad_context_swap_items_last"] = value
    elif (
        split == "valid"
        and name == "Ruliad Correctness Constrained Context-Swap Top-1 Change Rate"
    ):
        summary["ruliad_context_swap_top1_change_rate_last"] = value
    elif (
        split == "valid"
        and name
        == "Ruliad Correctness Constrained Context-Swap Equivalent Probability Drop"
    ):
        summary["ruliad_context_swap_equivalent_probability_drop_last"] = value
    elif (
        split == "valid"
        and name == "Ruliad Correctness Constrained Context-Swap JS Divergence"
    ):
        summary["ruliad_context_swap_js_divergence_last"] = value
    elif (
        split == "valid"
        and name == "Ruliad Correctness Constrained Counterfactual-Target Items"
    ):
        summary["ruliad_counterfactual_target_items_last"] = value
    elif (
        split == "valid"
        and name
        == "Ruliad Correctness Constrained Counterfactual-Target Equivalent Top-1 Rate"
    ):
        summary["ruliad_counterfactual_target_equivalent_top1_last"] = value
    elif (
        split == "valid"
        and name
        == "Ruliad Correctness Constrained Counterfactual-Target Top-1 Change Rate"
    ):
        summary["ruliad_counterfactual_target_top1_change_rate_last"] = value
    elif (
        split == "valid"
        and name
        == "Ruliad Correctness Constrained Counterfactual-Target Equivalent Probability Gain"
    ):
        summary["ruliad_counterfactual_target_equivalent_probability_gain_last"] = value
    elif (
        split == "valid"
        and name
        == "Ruliad Correctness Constrained Counterfactual-Target JS Divergence"
    ):
        summary["ruliad_counterfactual_target_js_divergence_last"] = value
    elif split == "valid" and name in {
        "Ruliad Partial Progress",
        "Ruliad Mean Partial Progress",
    }:
        summary["ruliad_partial_progress_last"] = value
    elif split == "valid" and name == "Output Entropy Bits":
        summary["output_entropy_bits_last"] = value
    elif split == "valid" and name == "Output Mean Max Probability":
        summary["output_mean_max_probability_last"] = value
    elif split == "valid" and name == "Output Distinct-2 Fraction":
        summary["output_distinct_2_fraction_last"] = value
    elif split == "valid" and name == "Output Repetition Fraction":
        summary["output_repetition_fraction_last"] = value
    elif split == "valid" and name == "Output Max Period-2..64 Fraction":
        summary["output_period_2_to_64_fraction_last"] = value
    elif split == "valid" and name == "Sequence State Rho RMS":
        summary["sequence_state_rho_rms_last"] = value
    elif split == "valid" and name == "Sequence State Rho Slot Redundancy":
        summary["sequence_state_rho_slot_redundancy_last"] = value
    elif split == "valid" and name == "Sequence State Rho Slot Variance Ratio":
        summary["sequence_state_rho_slot_variance_ratio_last"] = value


def update_source(summary: dict[str, Any], event: dict[str, Any]) -> None:
    loss = as_float(event.get("loss"))
    if loss is not None:
        absolute_step = as_int(event.get("absolute_step"))
        if absolute_step is None:
            summary["_source_loss_unkeyed"].append(loss)
        else:
            summary["_source_loss_by_step"][absolute_step] = loss
    fields = {
        "source_loss_last": "loss",
        "source_entropy_bits_last": "entropy_bits",
        "source_mean_difficulty_last": "mean_difficulty_level",
        "source_active_max_difficulty_last": "active_max_difficulty_level",
        "source_curriculum_released_max_difficulty_last": (
            "curriculum_released_max_difficulty_level"
        ),
        "source_active_max_difficulty_probability_last": (
            "active_max_difficulty_probability"
        ),
        "source_norm_difficulty_last": "normalized_difficulty_score",
        "source_mastered_probability_last": "mastered_probability",
        "source_capability_allowed_max_difficulty_last": (
            "capability_frontier_allowed_max_difficulty"
        ),
        "source_verifier_failures_last": "verifier_failures",
    }
    for summary_key, event_key in fields.items():
        value = as_float(event.get(event_key))
        if value is not None:
            summary[summary_key] = value


def default_event_summary(run: str, run_dir: Path) -> dict[str, Any]:
    summary = {key: "" for key in EVENT_SUMMARY_COLUMNS}
    summary["run"] = run
    summary["run_dir"] = str(run_dir)
    summary["gate_count"] = 0
    summary["fatal_gate_count"] = 0
    summary["capacity_scale_count"] = 0
    summary["checkpoint_count"] = 0
    summary["checkpoint_promoted_count"] = 0
    summary["checkpoint_promotion_ineligible_count"] = 0
    summary["capability_statistical_regression_count"] = 0
    summary["capability_quality_regression_count"] = 0
    summary["dynamics_control_count"] = 0
    summary["rollback_recovery_count"] = 0
    summary["source_capability_recovery_count"] = 0
    summary["validation_recovery_count"] = 0
    summary["pc_event_count"] = 0
    summary["pc_global_backward_calls_total"] = 0
    summary["pc_local_vjp_calls_total"] = 0
    summary["pc_temporal_state_vjp_calls_total"] = 0
    summary["pc_fused_temporal_vjp_calls_total"] = 0
    summary["pc_direct_forward_updates_total"] = 0
    summary["pc_feedback_parameter_updates_total"] = 0
    summary["pc_adjoint_teacher_updates_total"] = 0
    summary["pc_adjoint_local_updates_total"] = 0
    summary["pc_adjoint_calibration_samples_total"] = 0
    summary["pc_local_parameter_update_intents_total"] = 0
    summary["pc_parameter_updates_total"] = 0
    summary["pc_structured_terminal_steps_total"] = 0
    summary["pc_structured_terminal_skipped_steps_total"] = 0
    summary["pc_structured_terminal_groups_total"] = 0
    summary["pc_structured_terminal_rows_total"] = 0
    summary["_pc_ms_values"] = []
    summary["_source_loss_by_step"] = {}
    summary["_source_loss_unkeyed"] = []
    summary["_validation_loss_by_epoch"] = {}
    summary["_fixed_holdout_loss_by_epoch"] = {}
    summary["_source_weighted_loss_by_epoch"] = {}
    summary["_stream_warm_loss_by_epoch"] = {}
    summary["_ruliad_verifier_by_epoch"] = {}
    summary["_ruliad_policy_solve_by_epoch"] = {}
    summary["_ruliad_policy_goal_by_epoch"] = {}
    summary["_ruliad_policy_valid_action_by_epoch"] = {}
    summary["_ruliad_policy_top1_by_epoch"] = {}
    return summary


def read_stage_profile(log_path: Any) -> dict[str, float]:
    path = Path(str(log_path or ""))
    if not path.is_file():
        return {}
    last_profile = ""
    try:
        with path.open(errors="replace") as handle:
            for line in handle:
                if "[stage-profile][training]" in line:
                    last_profile = line
    except OSError:
        return {}
    if not last_profile:
        return {}
    parsed: dict[str, float] = {}
    for key, raw_value in re.findall(r"([a-zA-Z0-9_]+)=([^\s]+)", last_profile):
        value = as_float(raw_value)
        if value is not None:
            parsed[key] = value
    return parsed


def latest_experiment_planned_max_iters(run_dir: Path) -> int | None:
    """Read the authoritative stopping horizon after exact-resume extensions."""
    path = run_dir / "experiment_manifest.json"
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    launches = manifest.get("launches") or []
    if not launches:
        return None
    latest = launches[-1]
    planned = as_int(latest.get("planned_max_iters"))
    if planned is not None:
        return planned
    snapshot = latest.get("config_snapshot")
    if not isinstance(snapshot, str) or not snapshot:
        return None
    try:
        config = json.loads((run_dir / snapshot).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return as_int(((config.get("training") or {}).get("training") or {}).get("max_iters"))


def collect_event_summaries(
    paths: Iterable[Path], manifests: list[dict[str, Any]]
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    summaries: dict[str, dict[str, Any]] = {}
    latest_source_by_run: dict[str, dict[str, Any]] = {}
    capability_groups: dict[tuple[Any, ...], dict[str, Any]] = {}

    for path in paths:
        run_dir = path.parent.parent if path.parent.name == "events" else path.parent
        for event in read_jsonl(path):
            run = event.get("run_id") or run_dir.name
            summary = summaries.setdefault(run, default_event_summary(run, run_dir))
            event_type = event.get("type")
            if path.name == "source_selection.jsonl" or event_type == "source_selection":
                update_source(summary, event)
                if any(
                    event.get(key)
                    for key in (
                        "capability_frontier_coverage",
                        "difficulty_buckets",
                        "family_buckets",
                        "task_buckets",
                        "contract_buckets",
                        "top_buckets",
                    )
                ):
                    latest_source_by_run[run] = event
            elif event_type == "metric":
                update_metric(summary, event)
            elif event_type == "validation_finished":
                validation_loss = as_float(event.get("loss"))
                summary["validation_objective_loss_last"] = validation_loss
                objective = event.get("objective")
                if isinstance(objective, str) and objective:
                    summary["validation_objective_kind_last"] = objective
                epoch = as_int(event.get("epoch"))
                if validation_loss is not None and epoch is not None:
                    summary["_validation_loss_by_epoch"][epoch] = validation_loss
            elif event_type == "capability_probe":
                for group in event.get("group_buckets") or []:
                    raw_label = str(group.get("label") or "")
                    kind, separator, label = raw_label.partition(":")
                    if not separator:
                        kind, label = "unknown", raw_label
                    row = {key: "" for key in CAPABILITY_GROUP_COLUMNS}
                    row.update(
                        {
                            "run": run,
                            "epoch": event.get("epoch", ""),
                            "absolute_step": event.get("absolute_step", ""),
                            "probe_name": event.get("probe_name", ""),
                            "kind": kind,
                            "label": label,
                            **{
                                key: group.get(key, "")
                                for key in CAPABILITY_GROUP_COLUMNS
                                if key
                                not in {
                                    "run",
                                    "epoch",
                                    "absolute_step",
                                    "probe_name",
                                    "kind",
                                    "label",
                                }
                            },
                        }
                    )
                    key = (
                        run,
                        row["epoch"],
                        row["absolute_step"],
                        row["probe_name"],
                        kind,
                        label,
                    )
                    capability_groups[key] = row
            elif event_type == "output_degeneracy":
                summary["output_entropy_bits_last"] = as_float(event.get("entropy_bits"))
                summary["output_mean_max_probability_last"] = as_float(event.get("mean_max_probability"))
                summary["output_distinct_2_fraction_last"] = as_float(event.get("distinct_2_fraction"))
                summary["output_repetition_fraction_last"] = as_float(event.get("repetition_fraction"))
                summary["output_period_2_to_64_fraction_last"] = as_float(
                    event.get("max_period_2_to_64_fraction")
                )
            elif event_type == "gate":
                summary["gate_count"] += 1
                if event.get("severity") == "fatal":
                    summary["fatal_gate_count"] += 1
                gate = event.get("gate")
                if gate == "continual_learning_checkpoint_promotion_ineligible":
                    summary["checkpoint_promotion_ineligible_count"] += 1
                elif gate == "continual_learning_ruliad_capability_regression":
                    summary["capability_statistical_regression_count"] += 1
                elif gate == "continual_learning_capability_quality_regression":
                    summary["capability_quality_regression_count"] += 1
            elif event_type == "dynamics_control":
                summary["dynamics_control_count"] += 1
                mode = event.get("mode")
                if mode == "rollback_recovery":
                    summary["rollback_recovery_count"] += 1
                elif mode == "source_capability_recovery":
                    summary["source_capability_recovery_count"] += 1
                elif mode == "validation_recovery":
                    summary["validation_recovery_count"] += 1
            elif event_type == "checkpoint":
                summary["checkpoint_count"] += 1
                summary["checkpoint_last_epoch"] = event.get("epoch", "")
                summary["checkpoint_last_absolute_step"] = event.get(
                    "absolute_step", ""
                )
                if event.get("promoted") is True:
                    summary["checkpoint_promoted_count"] += 1
                    summary["checkpoint_last_promoted_epoch"] = event.get("epoch", "")
            elif event_type == "model_scale_applied":
                summary["capacity_scale_count"] += 1
            elif event_type == "predictive_coding":
                summary["pc_event_count"] += 1
                summary["pc_learning_contract_last"] = event.get(
                    "learning_contract", ""
                )
                summary["pc_execution_contract_version_last"] = event.get(
                    "execution_contract_version", ""
                )
                summary["pc_activity_derivative_contract_last"] = event.get(
                    "activity_derivative_contract", ""
                )
                summary["pc_parameter_derivative_contract_last"] = event.get(
                    "parameter_derivative_contract", ""
                )
                summary["pc_global_autodiff_graph_last"] = event.get(
                    "global_autodiff_graph", ""
                )
                summary["pc_global_backward_calls_total"] += as_int(
                    event.get("global_backward_calls")
                ) or 0
                summary["pc_local_vjp_calls_total"] += as_int(
                    event.get("local_vjp_calls")
                ) or 0
                summary["pc_temporal_state_vjp_calls_total"] += as_int(
                    event.get("temporal_state_vjp_calls")
                ) or 0
                summary["pc_fused_temporal_vjp_calls_total"] += as_int(
                    event.get("fused_temporal_vjp_calls")
                ) or 0
                summary["pc_temporal_credit_mode_last"] = event.get(
                    "temporal_credit_mode", ""
                )
                summary["pc_temporal_window_chunks_last"] = event.get(
                    "temporal_window_chunks", ""
                )
                summary["pc_direct_forward_updates_total"] += as_int(
                    event.get("direct_forward_updates")
                ) or 0
                summary["pc_feedback_parameter_updates_total"] += as_int(
                    event.get("feedback_parameter_updates")
                ) or 0
                summary["pc_adjoint_teacher_updates_total"] += as_int(
                    event.get("adjoint_teacher_updates")
                ) or 0
                summary["pc_adjoint_local_updates_total"] += as_int(
                    event.get("adjoint_local_updates")
                ) or 0
                summary["pc_adjoint_calibration_samples_total"] += as_int(
                    event.get("adjoint_calibration_samples")
                ) or 0
                summary["pc_adjoint_calibration_loss_last"] = as_float(
                    event.get("adjoint_calibration_loss")
                )
                summary["pc_adjoint_cosine_alignment_last"] = as_float(
                    event.get("adjoint_cosine_alignment")
                )
                summary["pc_adjoint_prediction_teacher_norm_ratio_last"] = as_float(
                    event.get("adjoint_prediction_teacher_norm_ratio")
                )
                summary["pc_adjoint_update_rms_last"] = as_float(
                    event.get("adjoint_update_rms")
                )
                summary["pc_local_parameter_update_intents_total"] += as_int(
                    event.get("local_parameter_update_intents")
                ) or 0
                summary["pc_parameter_updates_total"] += as_int(
                    event.get("parameter_updates")
                ) or 0
                summary["pc_structured_terminal_steps_total"] += as_int(
                    event.get("structured_terminal_steps")
                ) or 0
                summary["pc_structured_terminal_skipped_steps_total"] += as_int(
                    event.get("structured_terminal_skipped_steps")
                ) or 0
                summary["pc_structured_terminal_groups_total"] += as_int(
                    event.get("structured_terminal_groups")
                ) or 0
                summary["pc_structured_terminal_rows_total"] += as_int(
                    event.get("structured_terminal_rows")
                ) or 0
                summary["pc_factors_last"] = event.get("factors", "")
                summary["pc_gradient_tensors_last"] = event.get(
                    "gradient_tensors", ""
                )
                summary["pc_clip_fraction_mean_last"] = as_float(
                    event.get("clip_fraction_mean")
                )
                summary["pc_constraint_rms_last"] = as_float(
                    event.get("constraint_rms")
                )
                summary["pc_dual_rms_last"] = as_float(event.get("dual_rms"))
                summary["pc_composite_signal_rms_last"] = as_float(
                    event.get("composite_signal_rms")
                )
                summary["pc_observation_contract_last"] = event.get(
                    "observation_contract", ""
                )
                summary["pc_deployment_aligned_last"] = event.get(
                    "deployment_aligned", ""
                )
                summary["pc_amortization_components_last"] = event.get(
                    "amortization_components", ""
                )
                summary["pc_amortization_loss_last"] = event.get(
                    "amortization_loss", ""
                )
                pc_ms = as_float(event.get("elapsed_ms") or event.get("pc_ms"))
                if pc_ms is not None:
                    summary["_pc_ms_values"].append(pc_ms)

    rows: list[dict[str, Any]] = []
    manifest_by_run = manifests_by_run_name(manifests)
    for summary in summaries.values():
        pc_values = summary.pop("_pc_ms_values", [])
        summary["pc_ms_mean"] = stats(pc_values).mean if pc_values else ""
        source_loss_values = list(summary.pop("_source_loss_by_step", {}).values())
        source_loss_values.extend(summary.pop("_source_loss_unkeyed", []))
        summary["source_loss_cadence_mean"] = (
            stats(source_loss_values).mean if source_loss_values else ""
        )
        summary["source_loss_observations"] = len(source_loss_values)
        validation_by_epoch = summary.pop("_validation_loss_by_epoch", {})
        summarize_loss_series(summary, validation_by_epoch, "validation")
        for prefix in LOSS_SERIES_PREFIXES:
            summarize_loss_series(
                summary,
                summary.pop(f"_{prefix}_loss_by_epoch", {}),
                prefix,
            )
        verifier_by_epoch = summary.pop("_ruliad_verifier_by_epoch", {})
        verifier_points = sorted(verifier_by_epoch.items())
        if verifier_points:
            verifier_values = [value for _, value in verifier_points]
            verifier_best = max(verifier_values)
            summary["ruliad_verifier_accuracy_best"] = verifier_best
            summary["ruliad_verifier_accuracy_final_minus_best"] = (
                verifier_values[-1] - verifier_best
            )
        policy_series = {
            "ruliad_policy_solve_rate": summary.pop(
                "_ruliad_policy_solve_by_epoch", {}
            ),
            "ruliad_policy_goal_completion_rate": summary.pop(
                "_ruliad_policy_goal_by_epoch", {}
            ),
            "ruliad_policy_valid_action_rate": summary.pop(
                "_ruliad_policy_valid_action_by_epoch", {}
            ),
            "ruliad_policy_top1_expert_rate": summary.pop(
                "_ruliad_policy_top1_by_epoch", {}
            ),
        }
        promoted_epoch = as_int(summary.get("checkpoint_last_promoted_epoch"))
        for prefix, values_by_epoch in policy_series.items():
            points = sorted(values_by_epoch.items())
            if points:
                values = [value for _, value in points]
                best = max(values)
                summary[f"{prefix}_best"] = best
                summary[f"{prefix}_final_minus_best"] = values[-1] - best
            if promoted_epoch is not None and promoted_epoch in values_by_epoch:
                summary[f"{prefix}_promoted"] = values_by_epoch[promoted_epoch]
        manifest = manifest_by_run.get(summary["run"])
        if manifest:
            for key in (
                "trial_key",
                "matrix",
                "iters",
                "arm",
                "seed",
                "status",
                "elapsed_seconds",
                "batch_size",
                "checkpoint_interval_iters",
                "wall_clock_seconds",
                "ruliad_policy_probe_every_epochs",
                "backend",
                "profile",
                "validation_objective",
                "validation_sampling",
                "ruliad_panel_base_difficulty_levels",
                "source_selection_feedback_updates_enabled",
                "proof_policy_mode",
            ):
                summary[key] = manifest.get(key, "")
            if not summary.get("proof_policy_mode"):
                summary["proof_policy_mode"] = infer_proof_policy_mode(
                    str(summary.get("arm") or "")
                )
            summary["matrix"] = canonical_matrix(
                str(summary.get("matrix") or ""),
                summary.get("source_selection_feedback_updates_enabled"),
            )
            extended_horizon = latest_experiment_planned_max_iters(Path(summary["run_dir"]))
            if extended_horizon is not None:
                summary["iters"] = extended_horizon
            profile = read_stage_profile(manifest.get("log_path"))
            if "total_ns" in profile:
                summary["training_wall_seconds"] = profile["total_ns"] / 1_000_000_000.0
            for key in (
                "train_tokens",
                "source_batches",
                "source_tokens",
                "train_steps",
                "structured_terminal_steps",
                "structured_terminal_rows",
                "structured_terminal_padded_tokens",
                "stream_advance_ns",
                "wall_tokens_per_second",
                "model_tokens_per_second",
                "model_duty_fraction",
                "train_compute_fraction",
                "optimizer_fraction",
                "dataloader_foreground_wait_fraction",
                "host_sync_points",
            ):
                if key in profile:
                    summary[key] = profile[key]
        rows.append(summary)

    bucket_rows = collect_bucket_rows(latest_source_by_run)
    coverage_rows = collect_capability_coverage_rows(latest_source_by_run)
    capability_group_rows = sorted(
        capability_groups.values(),
        key=lambda row: (
            str(row["run"]),
            as_int(row["epoch"]) or -1,
            str(row["probe_name"]),
            str(row["kind"]),
            str(row["label"]),
        ),
    )
    return (
        sorted(rows, key=lambda row: row["run"]),
        bucket_rows,
        coverage_rows,
        capability_group_rows,
    )


def normalize_event_summaries(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    for event in rows:
        if not event.get("arm"):
            continue
        row = {key: "" for key in SUMMARY_COLUMNS}
        row.update(
            {
                "run": event.get("run", ""),
                "matrix": canonical_matrix(
                    str(event.get("matrix") or ""),
                    event.get("source_selection_feedback_updates_enabled", ""),
                ),
                "backend": event.get("backend", ""),
                "profile": event.get("profile", ""),
                "iters": as_int(event.get("iters")),
                "arm": event.get("arm", ""),
                "seed": as_int(event.get("seed")),
                "batch_size": as_int(event.get("batch_size")),
                "checkpoint_interval_iters": as_int(
                    event.get("checkpoint_interval_iters")
                ),
                "wall_clock_seconds": as_int(event.get("wall_clock_seconds")),
                "ruliad_policy_probe_every_epochs": as_int(
                    event.get("ruliad_policy_probe_every_epochs")
                ),
                "source_selection_feedback_updates_enabled": event.get(
                    "source_selection_feedback_updates_enabled", ""
                ),
                "proof_policy_mode": event.get("proof_policy_mode", "")
                or infer_proof_policy_mode(str(event.get("arm") or "")),
                "wall_s": as_float(event.get("training_wall_seconds"))
                or as_float(event.get("elapsed_seconds")),
                "tok_s": as_float(event.get("wall_tokens_per_second")),
                "model_tok_s": as_float(event.get("model_tokens_per_second")),
                "model_duty_fraction": as_float(event.get("model_duty_fraction")),
                "train_first": as_float(event.get("train_loss_first")),
                "train_last": as_float(event.get("train_loss_last")),
                "lr_last": as_float(event.get("lr_last")),
                "valid_last": as_float(event.get("valid_loss_last")),
                "valid_mean": as_float(event.get("valid_loss_mean")),
                "validation_objective": event.get(
                    "validation_objective_kind_last", ""
                )
                or event.get("validation_objective", ""),
                "validation_sampling": event.get("validation_sampling", ""),
                "ruliad_panel_base_difficulty_levels": as_int(
                    event.get("ruliad_panel_base_difficulty_levels")
                ),
                "validation_loss_first_checkpoint": as_float(
                    event.get("validation_loss_first_checkpoint")
                ),
                "validation_loss_best": as_float(
                    event.get("validation_loss_best")
                ),
                "validation_loss_final_minus_best": as_float(
                    event.get("validation_loss_final_minus_best")
                ),
                "validation_loss_regression_fraction": as_float(
                    event.get("validation_loss_regression_fraction")
                ),
                "validation_loss_slope_per_checkpoint": as_float(
                    event.get("validation_loss_slope_per_checkpoint")
                ),
                "valid_supervised_tokens": as_float(
                    event.get("valid_supervised_tokens_last")
                ),
                "valid_supervised_batches": as_float(
                    event.get("valid_supervised_batches_last")
                ),
                "valid_empty_supervision_batches": as_float(
                    event.get("valid_empty_supervision_batches_last")
                ),
                "stream_warm_loss": as_float(event.get("stream_warm_loss_mean")),
                "validation_objective_loss": as_float(
                    event.get("validation_objective_loss_last")
                ),
                "stream_paired_warm_loss": as_float(
                    event.get("stream_paired_warm_loss_last")
                ),
                "stream_paired_cold_loss": as_float(
                    event.get("stream_paired_cold_loss_last")
                ),
                "stream_carry_nll_gain": as_float(
                    event.get("stream_carry_nll_gain_last")
                ),
                "stream_carry_relative_gain": as_float(
                    event.get("stream_carry_relative_gain_last")
                ),
                "pc_ms_mean": as_float(event.get("pc_ms_mean")),
                "source_loss": as_float(event.get("source_loss_last")),
                "source_loss_cadence_mean": as_float(
                    event.get("source_loss_cadence_mean")
                ),
                "source_mean_difficulty": as_float(
                    event.get("source_mean_difficulty_last")
                ),
                "source_active_max_difficulty": as_float(
                    event.get("source_active_max_difficulty_last")
                ),
                "source_curriculum_released_max_difficulty": as_float(
                    event.get("source_curriculum_released_max_difficulty_last")
                ),
                "source_active_max_difficulty_probability": as_float(
                    event.get("source_active_max_difficulty_probability_last")
                ),
                "source_norm_difficulty": as_float(
                    event.get("source_norm_difficulty_last")
                ),
                "verifier_failures": as_float(
                    event.get("source_verifier_failures_last")
                ),
                "ruliad_verifier_accuracy": as_float(
                    event.get("ruliad_verifier_accuracy_last")
                ),
                "ruliad_verifier_accuracy_best": as_float(
                    event.get("ruliad_verifier_accuracy_best")
                ),
                "ruliad_verifier_accuracy_final_minus_best": as_float(
                    event.get("ruliad_verifier_accuracy_final_minus_best")
                ),
                "ruliad_constrained_equivalent_top1": as_float(
                    event.get("ruliad_constrained_equivalent_top1_last")
                ),
                "ruliad_constrained_equivalent_nll": as_float(
                    event.get("ruliad_constrained_equivalent_nll_last")
                ),
                "ruliad_constrained_valid_invalid_margin": as_float(
                    event.get("ruliad_constrained_valid_invalid_margin_last")
                ),
                "ruliad_context_swap_top1_change_rate": as_float(
                    event.get("ruliad_context_swap_top1_change_rate_last")
                ),
                "ruliad_context_swap_equivalent_probability_drop": as_float(
                    event.get("ruliad_context_swap_equivalent_probability_drop_last")
                ),
                "ruliad_context_swap_js_divergence": as_float(
                    event.get("ruliad_context_swap_js_divergence_last")
                ),
                "ruliad_counterfactual_target_equivalent_top1": as_float(
                    event.get("ruliad_counterfactual_target_equivalent_top1_last")
                ),
                "ruliad_counterfactual_target_top1_change_rate": as_float(
                    event.get("ruliad_counterfactual_target_top1_change_rate_last")
                ),
                "ruliad_counterfactual_target_equivalent_probability_gain": as_float(
                    event.get(
                        "ruliad_counterfactual_target_equivalent_probability_gain_last"
                    )
                ),
                "ruliad_counterfactual_target_js_divergence": as_float(
                    event.get("ruliad_counterfactual_target_js_divergence_last")
                ),
                "ruliad_partial_progress": as_float(
                    event.get("ruliad_partial_progress_last")
                ),
                "ruliad_policy_rollout_items": as_float(
                    event.get("ruliad_policy_rollout_items_last")
                ),
                "ruliad_policy_solve_rate": as_float(
                    event.get("ruliad_policy_solve_rate_last")
                ),
                "ruliad_policy_solve_rate_best": as_float(
                    event.get("ruliad_policy_solve_rate_best")
                ),
                "ruliad_policy_solve_rate_final_minus_best": as_float(
                    event.get("ruliad_policy_solve_rate_final_minus_best")
                ),
                "ruliad_policy_solve_rate_promoted": as_float(
                    event.get("ruliad_policy_solve_rate_promoted")
                ),
                "ruliad_policy_goal_completion_rate": as_float(
                    event.get("ruliad_policy_goal_completion_rate_last")
                ),
                "ruliad_policy_goal_completion_rate_best": as_float(
                    event.get("ruliad_policy_goal_completion_rate_best")
                ),
                "ruliad_policy_goal_completion_rate_final_minus_best": as_float(
                    event.get("ruliad_policy_goal_completion_rate_final_minus_best")
                ),
                "ruliad_policy_goal_completion_rate_promoted": as_float(
                    event.get("ruliad_policy_goal_completion_rate_promoted")
                ),
                "ruliad_policy_valid_action_rate": as_float(
                    event.get("ruliad_policy_valid_action_rate_last")
                ),
                "ruliad_policy_top1_expert_rate": as_float(
                    event.get("ruliad_policy_top1_expert_rate_last")
                ),
                "ruliad_policy_top1_expert_rate_promoted": as_float(
                    event.get("ruliad_policy_top1_expert_rate_promoted")
                ),
                "ruliad_policy_promotion_gate_passed": as_float(
                    event.get("ruliad_policy_promotion_gate_passed_last")
                ),
                "ruliad_deployment_capability_gate_passed": as_float(
                    event.get("ruliad_deployment_capability_gate_passed_last")
                ),
                "checkpoint_promoted_count": as_float(
                    event.get("checkpoint_promoted_count")
                ),
                "checkpoint_last_epoch": as_float(event.get("checkpoint_last_epoch")),
                "checkpoint_last_absolute_step": as_float(
                    event.get("checkpoint_last_absolute_step")
                ),
                "checkpoint_last_promoted_epoch": as_float(
                    event.get("checkpoint_last_promoted_epoch")
                ),
                "checkpoint_promotion_ineligible_count": as_float(
                    event.get("checkpoint_promotion_ineligible_count")
                ),
                "capability_statistical_regression_count": as_float(
                    event.get("capability_statistical_regression_count")
                ),
                "capability_quality_regression_count": as_float(
                    event.get("capability_quality_regression_count")
                ),
                "dynamics_control_count": as_float(
                    event.get("dynamics_control_count")
                ),
                "rollback_recovery_count": as_float(
                    event.get("rollback_recovery_count")
                ),
                "source_capability_recovery_count": as_float(
                    event.get("source_capability_recovery_count")
                ),
                "validation_recovery_count": as_float(
                    event.get("validation_recovery_count")
                ),
                "capability_allowed_max_difficulty": as_float(
                    event.get("source_capability_allowed_max_difficulty_last")
                ),
                "output_entropy_bits": as_float(
                    event.get("output_entropy_bits_last")
                ),
                "output_distinct_2_fraction": as_float(
                    event.get("output_distinct_2_fraction_last")
                ),
            }
        )
        for prefix in LOSS_SERIES_PREFIXES:
            for suffix in LOSS_SERIES_ANALYSIS_SUFFIXES:
                key = f"{prefix}_loss_{suffix}"
                row[key] = as_float(event.get(key))
        constrained = as_float(row.get("ruliad_constrained_equivalent_top1"))
        free = as_float(row.get("ruliad_verifier_accuracy"))
        if constrained is not None and free is not None:
            row["ruliad_constrained_free_accuracy_gap"] = constrained - free
        normalized_rows.append(row)
    return normalized_rows


def merge_summary_rows(
    legacy_rows: Iterable[dict[str, Any]], event_rows: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in [*legacy_rows, *event_rows]:
        key = (
            *experiment_context(row),
            row.get("arm"),
            row.get("seed"),
            row.get("run"),
        )
        existing = merged.get(key)
        if existing is None:
            merged[key] = dict(row)
            continue
        for field, value in row.items():
            if value not in ("", None):
                existing[field] = value
    return sorted(
        merged.values(),
        key=lambda row: (
            *experiment_context_sort_key(experiment_context(row)),
            str(row.get("arm") or ""),
            row.get("seed") or -1,
            str(row.get("run") or ""),
        ),
    )


def read_manifests(paths: Iterable[Path]) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for path in paths:
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            print(f"warning: skipping manifest {path}: {error}", file=sys.stderr)
            continue
        if not isinstance(data, dict) or "trial_key" not in data:
            continue
        row = {key: data.get(key, "") for key in MANIFEST_COLUMNS}
        row["matrix"] = canonical_matrix(
            str(row.get("matrix") or ""),
            row.get("source_selection_feedback_updates_enabled"),
        )
        if not row.get("proof_policy_mode"):
            row["proof_policy_mode"] = infer_proof_policy_mode(str(row.get("arm") or ""))
        manifests.append(row)
    return sorted(manifests, key=lambda row: str(row.get("trial_key", "")))


def manifests_by_run_name(manifests: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_run: dict[str, dict[str, Any]] = {}
    for manifest in manifests:
        run_dir = str(manifest.get("run_dir") or "")
        if not run_dir:
            continue
        run_name = Path(run_dir).name
        if run_name:
            by_run[run_name] = manifest
    return by_run


def collect_bucket_rows(latest_source_by_run: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run, event in latest_source_by_run.items():
        step = event.get("absolute_step", "")
        for kind, buckets in (
            ("top", event.get("top_buckets") or []),
            ("difficulty", event.get("difficulty_buckets") or []),
            ("family", event.get("family_buckets") or []),
            ("task", event.get("task_buckets") or []),
        ):
            for bucket in buckets:
                row = {key: "" for key in BUCKET_COLUMNS}
                row.update(
                    {
                        "run": run,
                        "absolute_step": step,
                        "kind": kind,
                        "label": bucket.get("label", ""),
                        "family": bucket.get("family", ""),
                        "task_kind": bucket.get("task_kind", ""),
                        "difficulty_level": bucket.get("difficulty_level", ""),
                        "candidate_count": bucket.get("candidate_count", ""),
                        "probability": bucket.get("probability", ""),
                        "loss_ema": bucket.get("loss_ema", ""),
                        "mean_loss": bucket.get("mean_loss", ""),
                        "learning_progress": bucket.get("learning_progress", ""),
                        "mastered": bucket.get("mastered", ""),
                        "mastered_probability": bucket.get("mastered_probability", ""),
                        "mean_difficulty_level": bucket.get("mean_difficulty_level", ""),
                        "capability_feedback_probability": bucket.get(
                            "capability_feedback_probability", ""
                        ),
                        "capability_verifier_ema": bucket.get(
                            "capability_verifier_ema", ""
                        ),
                        "capability_completion_health_ema": bucket.get(
                            "capability_completion_health_ema", ""
                        ),
                        "capability_schema_wrong_ema": bucket.get(
                            "capability_schema_wrong_ema", ""
                        ),
                        "capability_malformed_ema": bucket.get(
                            "capability_malformed_ema", ""
                        ),
                        "capability_missing_ema": bucket.get(
                            "capability_missing_ema", ""
                        ),
                        "capability_lagging_probability": bucket.get(
                            "capability_lagging_probability", ""
                        ),
                    }
                )
                rows.append(row)
    return rows


def collect_capability_coverage_rows(
    latest_source_by_run: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run, event in latest_source_by_run.items():
        for coverage in event.get("capability_frontier_coverage") or []:
            row = {key: "" for key in CAPABILITY_COVERAGE_COLUMNS}
            row.update(
                {
                    "run": run,
                    "absolute_step": event.get("absolute_step", ""),
                    "difficulty_level": coverage.get("difficulty_level", ""),
                    "candidate_coverage": coverage.get("candidate_coverage", ""),
                    "family_coverage": coverage.get("family_coverage", ""),
                    "task_coverage": coverage.get("task_coverage", ""),
                    "contract_coverage": coverage.get("contract_coverage", ""),
                    "observed_items": coverage.get("observed_items", ""),
                    "mastered": coverage.get("mastered", ""),
                }
            )
            rows.append(row)
    return rows


def read_gpu_csvs(
    paths: Iterable[Path], manifests: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    manifest_by_gpu_path = {
        str(Path(str(manifest.get("gpu_path"))).resolve()): manifest
        for manifest in manifests
        if manifest.get("gpu_path")
    }
    rows: list[dict[str, Any]] = []
    for path in paths:
        util: list[float] = []
        power: list[float] = []
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                util_value = as_float(row.get("utilization_gpu"))
                power_value = as_float(row.get("power_w"))
                if util_value is not None:
                    util.append(util_value)
                if power_value is not None:
                    power.append(power_value)
        row = {key: "" for key in GPU_COLUMNS}
        manifest = manifest_by_gpu_path.get(str(path.resolve()))
        row.update(
            {
                "file": str(path),
                "trial_key": manifest.get("trial_key", "") if manifest else "",
                "arm": manifest.get("arm", "") if manifest else "",
                "seed": manifest.get("seed", "") if manifest else "",
                "iters": manifest.get("iters", "") if manifest else "",
                "samples": max(len(util), len(power)),
                "util_mean": stats(util).mean,
                "util_p10": percentile(util, 0.10),
                "util_p50": percentile(util, 0.50),
                "util_p90": percentile(util, 0.90),
                "power_mean": stats(power).mean,
                "power_p10": percentile(power, 0.10),
                "power_p50": percentile(power, 0.50),
                "power_p90": percentile(power, 0.90),
            }
        )
        rows.append(row)
    return rows


def write_csv(path: Path, columns: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def grouped_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[tuple[Any, ...], Any], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(experiment_context(row), row.get("arm"))].append(row)
    out: list[dict[str, Any]] = []
    for (context, arm), group in sorted(
        groups.items(),
        key=lambda item: (*experiment_context_sort_key(item[0][0]), str(item[0][1])),
    ):
        row: dict[str, Any] = {
            **experiment_context_row(context),
            "arm": arm,
            "runs": len(group),
            "seeds": len({g.get("seed") for g in group}),
        }
        for metric in ANALYSIS_METRICS:
            metric_stats = stats(as_float(g.get(metric)) for g in group)
            row[f"{metric}_mean"] = metric_stats.mean
            row[f"{metric}_ci95"] = metric_stats.ci95
        out.append(row)
    return out


def paired_deltas(
    rows: list[dict[str, Any]], baseline: str, compare: str
) -> list[dict[str, Any]]:
    by_key: dict[tuple[tuple[Any, ...], Any], dict[str, dict[str, Any]]] = defaultdict(
        dict
    )
    for row in rows:
        by_key[(experiment_context(row), row.get("seed"))][row.get("arm")] = row

    deltas: dict[tuple[tuple[Any, ...], str], list[float]] = defaultdict(list)
    for (context, _seed), arms in by_key.items():
        if baseline not in arms or compare not in arms:
            continue
        for metric in ANALYSIS_METRICS:
            base = as_float(arms[baseline].get(metric))
            comp = as_float(arms[compare].get(metric))
            if base is None or comp is None:
                continue
            deltas[(context, metric)].append(comp - base)

    rows_out: list[dict[str, Any]] = []
    for (context, metric), values in sorted(
        deltas.items(),
        key=lambda item: (*experiment_context_sort_key(item[0][0]), item[0][1]),
    ):
        metric_stats = stats(values)
        rows_out.append(
            {
                **experiment_context_row(context),
                "comparison": f"{compare}-{baseline}",
                "metric": metric,
                "pairs": metric_stats.count,
                "delta_mean": metric_stats.mean,
                "delta_ci95": metric_stats.ci95,
            }
        )
    return rows_out


def write_markdown(
    path: Path,
    summary_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    gpu_rows: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append("# Predictive Coding Paper Tables")
    lines.append("")
    lines.append("Generated by `scripts/pc_paper_analyze.py`.")
    lines.append("")

    lines.append("## Run Summary")
    lines.append("")
    lines.append(
        "| Matrix | Backend | Profile | Iters | Wall budget | Completed updates | Batch | Ckpt | Policy probe | Feedback | Policy mode | Arm | Runs | Seeds | Cold valid | Cold supervised tokens | Stream warm | Validation objective | Best objective | Final-best | Regression | Val slope | Source cadence | Free verifier acc | Action top-1 | Decode gap | Action NLL | Context swap | Counterfactual gain | Policy solve | Goal completion | Valid action | Partial progress | Wall tok/s | Model tok/s | Duty | PC ms |"
    )
    lines.append(
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for row in summary_rows:
        lines.append(
            "| {matrix} | {backend} | {profile} | {iters} | {wall_clock_seconds} | {completed_updates} | {batch_size} | {checkpoint_interval} | {policy_probe_cadence} | {source_feedback} | {policy_mode} | {arm} | {runs} | {seeds} | {valid} | {cold_tokens} | {warm} | {objective} | {best_objective} | {final_minus_best} | {regression} | {slope} | {source} | {verifier} | {action_top1} | {decode_gap} | {action_nll} | {context_swap} | {counterfactual_gain} | {policy_solve} | {goal_completion} | {valid_action} | {partial} | {tok} | {model_tok} | {duty} | {pc} |".format(
                matrix=row.get("matrix", ""),
                backend=row.get("backend", ""),
                profile=Path(str(row.get("profile") or "")).name,
                iters=row.get("iters", ""),
                wall_clock_seconds=row.get("wall_clock_seconds", ""),
                completed_updates=fmt_mean_ci(
                    row, "checkpoint_last_absolute_step"
                ),
                batch_size=row.get("batch_size", ""),
                checkpoint_interval=row.get("checkpoint_interval_iters", ""),
                policy_probe_cadence=row.get(
                    "ruliad_policy_probe_every_epochs", ""
                ),
                source_feedback=row.get(
                    "source_selection_feedback_updates_enabled", ""
                ),
                policy_mode=row.get("proof_policy_mode", ""),
                arm=row.get("arm", ""),
                runs=row.get("runs", ""),
                seeds=row.get("seeds", ""),
                valid=fmt_mean_ci(row, "valid_mean"),
                cold_tokens=fmt_mean_ci(row, "valid_supervised_tokens"),
                warm=fmt_mean_ci(row, "stream_warm_loss"),
                objective=fmt_mean_ci(row, "validation_objective_loss"),
                best_objective=fmt_mean_ci(row, "validation_loss_best"),
                final_minus_best=fmt_mean_ci(
                    row, "validation_loss_final_minus_best"
                ),
                regression=fmt_mean_ci(
                    row, "validation_loss_regression_fraction"
                ),
                slope=fmt_mean_ci(row, "validation_loss_slope_per_checkpoint"),
                source=fmt_mean_ci(row, "source_loss_cadence_mean"),
                verifier=fmt_mean_ci(row, "ruliad_verifier_accuracy"),
                action_top1=fmt_mean_ci(row, "ruliad_constrained_equivalent_top1"),
                decode_gap=fmt_mean_ci(
                    row, "ruliad_constrained_free_accuracy_gap"
                ),
                action_nll=fmt_mean_ci(row, "ruliad_constrained_equivalent_nll"),
                context_swap=fmt_mean_ci(
                    row, "ruliad_context_swap_top1_change_rate"
                ),
                counterfactual_gain=fmt_mean_ci(
                    row,
                    "ruliad_counterfactual_target_equivalent_probability_gain",
                ),
                policy_solve=fmt_mean_ci(row, "ruliad_policy_solve_rate"),
                goal_completion=fmt_mean_ci(
                    row, "ruliad_policy_goal_completion_rate"
                ),
                valid_action=fmt_mean_ci(row, "ruliad_policy_valid_action_rate"),
                partial=fmt_mean_ci(row, "ruliad_partial_progress"),
                tok=fmt_mean_ci(row, "tok_s"),
                model_tok=fmt_mean_ci(row, "model_tok_s"),
                duty=fmt_mean_ci(row, "model_duty_fraction"),
                pc=fmt_mean_ci(row, "pc_ms_mean"),
            )
        )
    lines.append("")

    lines.append("## Paired Deltas")
    lines.append("")
    lines.append(
        "| Matrix | Backend | Profile | Iters | Batch | Ckpt | Policy probe | Feedback | Policy mode | Comparison | Metric | Pairs | Delta |"
    )
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: | ---: |")
    for row in paired_rows:
        lines.append(
            "| {matrix} | {backend} | {profile} | {iters} | {batch_size} | {checkpoint_interval} | {policy_probe_cadence} | {source_feedback} | {policy_mode} | {comparison} | {metric} | {pairs} | {delta} |".format(
                matrix=row.get("matrix", ""),
                backend=row.get("backend", ""),
                profile=Path(str(row.get("profile") or "")).name,
                iters=row.get("iters", ""),
                batch_size=row.get("batch_size", ""),
                checkpoint_interval=row.get("checkpoint_interval_iters", ""),
                policy_probe_cadence=row.get(
                    "ruliad_policy_probe_every_epochs", ""
                ),
                source_feedback=row.get(
                    "source_selection_feedback_updates_enabled", ""
                ),
                policy_mode=row.get("proof_policy_mode", ""),
                comparison=row.get("comparison", ""),
                metric=row.get("metric", ""),
                pairs=row.get("pairs", ""),
                delta=fmt_value_ci(row.get("delta_mean"), row.get("delta_ci95")),
            )
        )
    lines.append("")

    if event_rows:
        lines.append("## Event-Stream Diagnostics")
        lines.append("")

        pc_rows = [row for row in event_rows if as_int(row.get("pc_event_count"))]
        if pc_rows:
            lines.append("## Local Learning Contract")
            lines.append("")
            lines.append(
                "| Run | Contract | Temporal mode/window | Derivatives | Global graph | Global backwards | Local/temporal/fused VJPs | Direct updates | Feedback updates | Adjoint teacher/local | Adjoint fit n/loss/cos/norm/update | Update intents/optimizer apps | Structured steps/skips | Structured groups/rows | Factors | Gradient tensors | ALM clip/constraint/dual/signal |"
            )
            lines.append("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |")
            for row in pc_rows[:40]:
                lines.append(
                    "| {run} | {contract} | {temporal_mode}/{temporal_window} | v{version} activity={activity}, params={parameters} | {graph} | {backwards} | {vjps}/{temporal_vjps}/{fused_temporal_vjps} | {direct} | {feedback} | {adjoint_teacher}/{adjoint_local} | {adjoint_samples}/{adjoint_loss}/{adjoint_cosine}/{adjoint_norm_ratio}/{adjoint_update_rms} | {update_intents}/{updates} | {structured_steps}/{structured_skips} | {structured_groups}/{structured_rows} | {factors} | {grads} | {clip}/{constraint}/{dual}/{signal} |".format(
                        run=row.get("run", ""),
                        contract=row.get("pc_learning_contract_last", ""),
                        temporal_mode=row.get("pc_temporal_credit_mode_last", ""),
                        temporal_window=row.get("pc_temporal_window_chunks_last", ""),
                        version=row.get("pc_execution_contract_version_last", ""),
                        activity=row.get("pc_activity_derivative_contract_last", ""),
                        parameters=row.get("pc_parameter_derivative_contract_last", ""),
                        graph=row.get("pc_global_autodiff_graph_last", ""),
                        backwards=row.get("pc_global_backward_calls_total", ""),
                        vjps=row.get("pc_local_vjp_calls_total", ""),
                        temporal_vjps=row.get(
                            "pc_temporal_state_vjp_calls_total", ""
                        ),
                        fused_temporal_vjps=row.get(
                            "pc_fused_temporal_vjp_calls_total", ""
                        ),
                        direct=row.get("pc_direct_forward_updates_total", ""),
                        feedback=row.get("pc_feedback_parameter_updates_total", ""),
                        adjoint_teacher=row.get("pc_adjoint_teacher_updates_total", ""),
                        adjoint_local=row.get("pc_adjoint_local_updates_total", ""),
                        adjoint_samples=row.get(
                            "pc_adjoint_calibration_samples_total", ""
                        ),
                        adjoint_loss=fmt_scalar(
                            row.get("pc_adjoint_calibration_loss_last")
                        ),
                        adjoint_cosine=fmt_scalar(
                            row.get("pc_adjoint_cosine_alignment_last")
                        ),
                        adjoint_norm_ratio=fmt_scalar(
                            row.get("pc_adjoint_prediction_teacher_norm_ratio_last")
                        ),
                        adjoint_update_rms=fmt_scalar(
                            row.get("pc_adjoint_update_rms_last")
                        ),
                        update_intents=row.get(
                            "pc_local_parameter_update_intents_total", ""
                        ),
                        updates=row.get("pc_parameter_updates_total", ""),
                        structured_steps=row.get("pc_structured_terminal_steps_total", ""),
                        structured_skips=row.get(
                            "pc_structured_terminal_skipped_steps_total", ""
                        ),
                        structured_groups=row.get(
                            "pc_structured_terminal_groups_total", ""
                        ),
                        structured_rows=row.get("pc_structured_terminal_rows_total", ""),
                        factors=row.get("pc_factors_last", ""),
                        grads=row.get("pc_gradient_tensors_last", ""),
                        clip=fmt_scalar(row.get("pc_clip_fraction_mean_last")),
                        constraint=fmt_scalar(row.get("pc_constraint_rms_last")),
                        dual=fmt_scalar(row.get("pc_dual_rms_last")),
                        signal=fmt_scalar(row.get("pc_composite_signal_rms_last")),
                    )
                )
            lines.append("")
        lines.append(
            "| Run | Cold valid | Stream warm | Objective | Best | Final-best | Regression | Val slope | Carry gain | Source difficulty | Verifier acc | Output entropy | Deploy gate | Promotions | Ineligible | Statistical regressions | Rollbacks | Source recoveries | Gates | Fatal gates |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for row in event_rows[:40]:
            lines.append(
                "| {run} | {valid} | {warm} | {objective} | {best} | {final_minus_best} | {regression} | {slope} | {carry} | {difficulty} | {verifier} | {entropy} | {deployment_gate} | {promotions} | {ineligible} | {statistical_regressions} | {rollbacks} | {source_recoveries} | {gates} | {fatal} |".format(
                    run=row.get("run", ""),
                    valid=fmt_scalar(row.get("valid_loss_mean")),
                    warm=fmt_scalar(row.get("stream_warm_loss_mean")),
                    objective=fmt_scalar(row.get("validation_objective_loss_last")),
                    best=fmt_scalar(row.get("validation_loss_best")),
                    final_minus_best=fmt_scalar(
                        row.get("validation_loss_final_minus_best")
                    ),
                    regression=fmt_scalar(
                        row.get("validation_loss_regression_fraction")
                    ),
                    slope=fmt_scalar(
                        row.get("validation_loss_slope_per_checkpoint")
                    ),
                    carry=fmt_scalar(row.get("stream_carry_nll_gain_last")),
                    difficulty=fmt_scalar(row.get("source_mean_difficulty_last")),
                    verifier=fmt_scalar(row.get("ruliad_verifier_accuracy_last")),
                    entropy=fmt_scalar(row.get("output_entropy_bits_last")),
                    deployment_gate=fmt_scalar(
                        row.get("ruliad_deployment_capability_gate_passed_last")
                    ),
                    promotions=row.get("checkpoint_promoted_count", ""),
                    ineligible=row.get("checkpoint_promotion_ineligible_count", ""),
                    statistical_regressions=row.get(
                        "capability_statistical_regression_count", ""
                    ),
                    rollbacks=row.get("rollback_recovery_count", ""),
                    source_recoveries=row.get(
                        "source_capability_recovery_count", ""
                    ),
                    gates=row.get("gate_count", ""),
                    fatal=row.get("fatal_gate_count", ""),
                )
            )
        lines.append("")

    if gpu_rows:
        lines.append("## GPU Telemetry")
        lines.append("")
        lines.append("| Arm | Seed | Samples | Util mean | Util p50 | Power mean | Power p50 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in gpu_rows:
            lines.append(
                "| {arm} | {seed} | {samples} | {util_mean} | {util_p50} | {power_mean} | {power_p50} |".format(
                    arm=row.get("arm", "") or Path(str(row.get("file", ""))).name,
                    seed=row.get("seed", ""),
                    samples=row.get("samples", ""),
                    util_mean=fmt_scalar(row.get("util_mean")),
                    util_p50=fmt_scalar(row.get("util_p50")),
                    power_mean=fmt_scalar(row.get("power_mean")),
                    power_p50=fmt_scalar(row.get("power_p50")),
                )
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n")


def fmt_mean_ci(row: dict[str, Any], metric: str) -> str:
    return fmt_value_ci(row.get(f"{metric}_mean"), row.get(f"{metric}_ci95"))


def fmt_value_ci(mean: Any, ci95: Any) -> str:
    mean_value = as_float(mean)
    ci_value = as_float(ci95)
    if mean_value is None:
        return ""
    if ci_value is None or ci_value == 0.0:
        return f"{mean_value:.4g}"
    return f"{mean_value:.4g} +/- {ci_value:.3g}"


def fmt_scalar(value: Any) -> str:
    parsed = as_float(value)
    return "" if parsed is None else f"{parsed:.4g}"


def run_analysis(inputs: list[str], out_dir: Path, baseline: str, compare: str) -> None:
    summary_csvs, gpu_csvs, event_jsonls, manifest_jsons = discover_inputs(inputs)
    manifest_rows = read_manifests(manifest_jsons)
    event_rows, bucket_rows, coverage_rows, capability_group_rows = collect_event_summaries(
        event_jsonls, manifest_rows
    )
    summary_rows = merge_summary_rows(
        read_summary_csvs(summary_csvs), normalize_event_summaries(event_rows)
    )
    gpu_rows = read_gpu_csvs(gpu_csvs, manifest_rows)
    grouped_rows = grouped_summary(summary_rows)
    paired_rows = paired_deltas(summary_rows, baseline, compare)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "normalized_summary.csv", SUMMARY_COLUMNS, summary_rows)
    write_csv(
        out_dir / "summary_by_arm.csv",
        [*EXPERIMENT_CONTEXT_FIELDS, "arm", "runs", "seeds"]
        + [
            f"{metric}_{suffix}"
            for metric in ANALYSIS_METRICS
            for suffix in ["mean", "ci95"]
        ],
        grouped_rows,
    )
    write_csv(
        out_dir / "paired_deltas.csv",
        [
            *EXPERIMENT_CONTEXT_FIELDS,
            "comparison",
            "metric",
            "pairs",
            "delta_mean",
            "delta_ci95",
        ],
        paired_rows,
    )
    write_csv(out_dir / "event_run_summary.csv", EVENT_SUMMARY_COLUMNS, event_rows)
    write_csv(out_dir / "source_bucket_summary.csv", BUCKET_COLUMNS, bucket_rows)
    write_csv(
        out_dir / "source_capability_coverage.csv",
        CAPABILITY_COVERAGE_COLUMNS,
        coverage_rows,
    )
    write_csv(
        out_dir / "capability_group_trajectory.csv",
        CAPABILITY_GROUP_COLUMNS,
        capability_group_rows,
    )
    write_csv(out_dir / "gpu_summary.csv", GPU_COLUMNS, gpu_rows)
    write_csv(out_dir / "manifest_summary.csv", MANIFEST_COLUMNS, manifest_rows)
    write_markdown(out_dir / "paper_tables.md", grouped_rows, paired_rows, event_rows, gpu_rows)

    print(
        "summary_csvs={} event_jsonls={} gpu_csvs={} manifests={}".format(
            len(summary_csvs), len(event_jsonls), len(gpu_csvs), len(manifest_jsons)
        )
    )
    print(f"wrote {out_dir}")


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        summary = root / "pc_ablation_all_summary.csv"
        summary.write_text(
            "run,iters,arm,seed,wall_s,tok_s,train_first,train_last,valid_last,lr_last,pc_ms_mean,pc_corrected_fraction,source_loss,source_mean_difficulty,source_norm_difficulty,verifier_failures\n"
            "r1,512,adamw,1,10,100,2,1,0.5,0.001,0,0,0.7,3,0.3,0\n"
            "r2,512,adamwpc,1,12,80,2,0.9,0.4,0.001,5,0.5,0.6,3.2,0.32,0\n"
            "r3,512,adamw,2,11,90,2,1.1,0.55,0.001,0,0,0.75,3.1,0.31,0\n"
            "r4,512,adamwpc,2,13,75,2,1.0,0.45,0.001,6,0.5,0.65,3.3,0.33,0\n"
        )
        events = root / "run-a" / "events"
        events.mkdir(parents=True)
        (events / "training_events.jsonl").write_text(
            json.dumps(
                {
                    "type": "metric",
                    "run_id": "run-a",
                    "split": "valid",
                    "epoch": 1,
                    "name": "Loss",
                    "value": 0.4,
                    "running_value": 0.45,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "metric",
                    "run_id": "run-a",
                    "split": "train",
                    "name": "Learning Rate",
                    "value": 0.00025,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "metric",
                    "run_id": "run-a",
                    "split": "valid",
                    "epoch": 1,
                    "name": "Stream Warm Loss",
                    "value": 0.6,
                    "running_value": 0.55,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "metric",
                    "run_id": "run-a",
                    "split": "valid",
                    "epoch": 1,
                    "name": "Source Weighted Loss",
                    "value": 0.52,
                    "running_value": 0.52,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "validation_finished",
                    "run_id": "run-a",
                    "epoch": 1,
                    "objective": "source_weighted",
                    "loss": 0.55,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "validation_finished",
                    "run_id": "run-a",
                    "epoch": 2,
                    "objective": "source_weighted",
                    "loss": 0.5,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "validation_finished",
                    "run_id": "run-a",
                    "epoch": 3,
                    "objective": "source_weighted",
                    "loss": 0.6,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "metric",
                    "run_id": "run-a",
                    "split": "valid",
                    "epoch": 3,
                    "name": "Ruliad Deployment Capability Gate Passed",
                    "value": 1.0,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "metric",
                    "run_id": "run-a",
                    "split": "valid",
                    "epoch": 2,
                    "name": "Ruliad Policy Rollout Solve Rate",
                    "value": 0.6,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "metric",
                    "run_id": "run-a",
                    "split": "valid",
                    "epoch": 2,
                    "name": "Ruliad Policy Rollout Goal Completion Rate",
                    "value": 0.7,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "metric",
                    "run_id": "run-a",
                    "split": "valid",
                    "epoch": 3,
                    "name": "Ruliad Policy Rollout Solve Rate",
                    "value": 0.5,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "metric",
                    "run_id": "run-a",
                    "split": "valid",
                    "epoch": 3,
                    "name": "Ruliad Policy Rollout Goal Completion Rate",
                    "value": 0.65,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "metric",
                    "run_id": "run-a",
                    "split": "valid",
                    "epoch": 3,
                    "name": "Ruliad Policy Model Top-1 Expert Rate",
                    "value": 0.8,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "checkpoint",
                    "run_id": "run-a",
                    "epoch": 2,
                    "absolute_step": 127,
                    "checkpoint_id": "model-2",
                    "promoted": False,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "checkpoint",
                    "run_id": "run-a",
                    "epoch": 3,
                    "absolute_step": 255,
                    "checkpoint_id": "model-3",
                    "promoted": True,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "gate",
                    "run_id": "run-a",
                    "gate": "continual_learning_checkpoint_promotion_ineligible",
                    "severity": "info",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "gate",
                    "run_id": "run-a",
                    "gate": "continual_learning_ruliad_capability_regression",
                    "severity": "warning",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "dynamics_control",
                    "run_id": "run-a",
                    "mode": "rollback_recovery",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "predictive_coding",
                    "run_id": "run-a",
                    "learning_contract": "local_factor_vjp_v1",
                    "execution_contract_version": 1,
                    "activity_derivative_contract": "analytic_local",
                    "parameter_derivative_contract": "analytic_local",
                    "global_autodiff_graph": False,
                    "global_backward_calls": 0,
                    "local_vjp_calls": 12,
                    "temporal_state_vjp_calls": 4,
                    "fused_temporal_vjp_calls": 4,
                    "temporal_credit_mode": "exact_window",
                    "temporal_window_chunks": 2,
                    "direct_forward_updates": 6,
                    "feedback_parameter_updates": 6,
                    "local_parameter_update_intents": 7,
                    "parameter_updates": 3,
                    "factors": 4,
                    "gradient_tensors": 9,
                    "clip_fraction_mean": 0.125,
                    "constraint_rms": 0.01,
                    "dual_rms": 0.02,
                    "composite_signal_rms": 0.03,
                    "elapsed_ms": 3.5,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "metric",
                    "run_id": "run-a",
                    "split": "valid",
                    "name": "Ruliad Verifier Accuracy",
                    "value": 0.25,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "metric",
                    "run_id": "run-a",
                    "split": "valid",
                    "name": "Ruliad Correctness Constrained Equivalent Top-1 Rate",
                    "value": 0.75,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "metric",
                    "run_id": "run-a",
                    "split": "valid",
                    "name": "Ruliad Mean Partial Progress",
                    "value": 0.5,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "metric",
                    "run_id": "run-a",
                    "split": "valid",
                    "name": "Ruliad Correctness Constrained Context-Swap Top-1 Change Rate",
                    "value": 0.375,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "metric",
                    "run_id": "run-a",
                    "split": "valid",
                    "name": "Ruliad Correctness Constrained Counterfactual-Target Equivalent Probability Gain",
                    "value": 0.125,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "capability_probe",
                    "run_id": "run-a",
                    "epoch": 3,
                    "absolute_step": 12,
                    "probe_name": "ruliad_correctness",
                    "group_buckets": [
                        {
                            "label": "difficulty:d2",
                            "item_count": 8,
                            "exact_rate": 0.125,
                            "semantic_rate": 0.25,
                            "verifier_rate": 0.375,
                            "partial_credit_rate": 0.5,
                            "schema_valid_wrong_rate": 0.25,
                            "malformed_rate": 0.0,
                            "missing_rate": 0.0,
                            "mean_partial_progress": 0.625,
                            "answer_field_accuracy": 0.75,
                            "answer_field_coverage": 0.875,
                            "answer_termination_rate": 1.0,
                        }
                    ],
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "source_selection",
                    "run_id": "run-a",
                    "absolute_step": 4,
                    "loss": 0.6,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "source_selection",
                    "run_id": "run-a",
                    "absolute_step": 3,
                    "loss": 0.4,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "source_selection",
                    "run_id": "run-a",
                    "absolute_step": 4,
                    "loss": 0.6,
                    "entropy_bits": 3.0,
                    "mean_difficulty_level": 5.0,
                    "active_max_difficulty_level": 6,
                    "curriculum_released_max_difficulty_level": 7,
                    "active_max_difficulty_probability": 0.2,
                    "normalized_difficulty_score": 0.5,
                    "mastered_probability": 0.2,
                    "capability_frontier_allowed_max_difficulty": 4,
                    "verifier_failures": 0,
                    "capability_frontier_coverage": [
                        {
                            "difficulty_level": 5,
                            "candidate_coverage": 0.75,
                            "family_coverage": 1.0,
                            "task_coverage": 0.5,
                            "contract_coverage": 0.25,
                            "observed_items": 96,
                            "mastered": False,
                        }
                    ],
                    "difficulty_buckets": [
                        {
                            "label": "d5",
                            "candidate_count": 3,
                            "probability": 0.4,
                            "mean_loss": 0.6,
                            "learning_progress": 0.1,
                            "mastered_probability": 0.2,
                            "mean_difficulty_level": 5.0,
                            "capability_feedback_probability": 0.75,
                            "capability_verifier_ema": 0.6,
                            "capability_completion_health_ema": 0.8,
                            "capability_schema_wrong_ema": 0.1,
                            "capability_malformed_ema": 0.05,
                            "capability_missing_ema": 0.0,
                            "capability_lagging_probability": 0.25,
                        }
                    ],
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "source_selection",
                    "run_id": "run-a",
                    "absolute_step": 5,
                    "loss": None,
                    "capability_frontier_allowed_max_difficulty": 5,
                    "capability_frontier_coverage": [],
                }
            )
            + "\n"
        )
        manifests = root / "manifests"
        manifests.mkdir()
        log_path = root / "run-a.log"
        log_path.write_text(
            "[stage-profile][training] total_ns=2000000000 train_tokens=1024 "
            "wall_tokens_per_second=512 model_tokens_per_second=640 "
            "model_duty_fraction=0.8 train_compute_fraction=0.7 "
            "optimizer_fraction=0.1 dataloader_foreground_wait_fraction=0.01 "
            "host_sync_points=0\n"
        )
        gpu_path = root / "run-a.gpu.csv"
        gpu_path.write_text(
            "timestamp,index,utilization_gpu,power_w\n"
            "2026/01/01 00:00:00,0,80,50\n"
            "2026/01/01 00:00:01,0,100,60\n"
        )
        (manifests / "run-a.json").write_text(
            json.dumps(
                {
                    "trial_key": "pc-smoke-run-a",
                    "matrix": "smoke",
                    "iters": 4,
                    "arm": "adamwpc",
                    "seed": 7,
                    "batch_size": 8,
                    "checkpoint_interval_iters": 2,
                    "wall_clock_seconds": 60,
                    "ruliad_policy_probe_every_epochs": 1,
                    "validation_objective": "source_weighted",
                    "validation_sampling": "fixed_holdout",
                    "backend": "cpu",
                    "features": "train",
                    "profile": "profile.toml",
                    "overlay": "overlay.toml",
                    "run_root": str(root),
                    "run_dir": str(root / "run-a"),
                    "log_path": str(log_path),
                    "gpu_path": str(gpu_path),
                    "status": "ok",
                    "elapsed_seconds": 1,
                    "peak_used_mb": 10,
                    "min_available_mb": 100,
                    "git_sha": "test",
                    "git_branch": "test",
                    "git_dirty": False,
                }
            )
        )
        out = root / "analysis"
        run_analysis([str(root)], out, "adamw", "adamwpc")
        paired = list(csv.DictReader((out / "paired_deltas.csv").open()))
        assert paired, "paired deltas should be written"
        buckets = list(csv.DictReader((out / "source_bucket_summary.csv").open()))
        assert buckets, "bucket summary should be written"
        event_rows = list(csv.DictReader((out / "event_run_summary.csv").open()))
        assert event_rows[0]["trial_key"] == "pc-smoke-run-a"
        assert event_rows[0]["arm"] == "adamwpc"
        assert event_rows[0]["wall_tokens_per_second"] == "512.0"
        assert event_rows[0]["model_tokens_per_second"] == "640.0"
        assert event_rows[0]["model_duty_fraction"] == "0.8"
        assert event_rows[0]["valid_loss_mean"] == "0.45"
        assert event_rows[0]["stream_warm_loss_mean"] == "0.55"
        assert event_rows[0]["validation_objective_loss_last"] == "0.6"
        assert event_rows[0]["validation_objective_kind_last"] == "source_weighted"
        assert event_rows[0]["fixed_holdout_loss_first_checkpoint"] == "0.45"
        assert event_rows[0]["source_weighted_loss_first_checkpoint"] == "0.52"
        assert event_rows[0]["stream_warm_loss_first_checkpoint"] == "0.55"
        assert event_rows[0]["validation_checkpoints"] == "3"
        assert event_rows[0]["validation_loss_first_checkpoint"] == "0.55"
        assert event_rows[0]["validation_loss_best"] == "0.5"
        assert event_rows[0]["validation_loss_best_epoch"] == "2"
        assert math.isclose(
            as_float(event_rows[0]["validation_loss_final_minus_best"]),
            0.1,
        )
        assert math.isclose(
            as_float(event_rows[0]["validation_loss_regression_fraction"]),
            0.2,
        )
        assert math.isclose(
            as_float(event_rows[0]["validation_loss_slope_per_checkpoint"]),
            0.025,
        )
        assert event_rows[0]["ruliad_context_swap_top1_change_rate_last"] == "0.375"
        assert (
            event_rows[0][
                "ruliad_counterfactual_target_equivalent_probability_gain_last"
            ]
            == "0.125"
        )
        assert event_rows[0]["source_loss_last"] == "0.6"
        assert event_rows[0]["source_loss_cadence_mean"] == "0.5"
        assert event_rows[0]["source_loss_observations"] == "2"
        assert event_rows[0]["source_active_max_difficulty_last"] == "6.0"
        assert (
            event_rows[0]["source_curriculum_released_max_difficulty_last"]
            == "7.0"
        )
        assert event_rows[0]["source_active_max_difficulty_probability_last"] == "0.2"
        assert event_rows[0]["source_capability_allowed_max_difficulty_last"] == "5.0"
        assert event_rows[0]["ruliad_deployment_capability_gate_passed_last"] == "1.0"
        assert event_rows[0]["checkpoint_count"] == "2"
        assert event_rows[0]["checkpoint_last_epoch"] == "3"
        assert event_rows[0]["checkpoint_last_absolute_step"] == "255"
        assert event_rows[0]["checkpoint_promoted_count"] == "1"
        assert event_rows[0]["checkpoint_last_promoted_epoch"] == "3"
        assert event_rows[0]["ruliad_policy_solve_rate_best"] == "0.6"
        assert math.isclose(
            as_float(event_rows[0]["ruliad_policy_solve_rate_final_minus_best"]),
            -0.1,
        )
        assert event_rows[0]["ruliad_policy_solve_rate_promoted"] == "0.5"
        assert event_rows[0]["ruliad_policy_goal_completion_rate_best"] == "0.7"
        assert event_rows[0]["ruliad_policy_goal_completion_rate_promoted"] == "0.65"
        assert event_rows[0]["ruliad_policy_top1_expert_rate_promoted"] == "0.8"
        assert event_rows[0]["checkpoint_promotion_ineligible_count"] == "1"
        assert event_rows[0]["capability_statistical_regression_count"] == "1"
        assert event_rows[0]["capability_quality_regression_count"] == "0"
        assert event_rows[0]["dynamics_control_count"] == "1"
        assert event_rows[0]["rollback_recovery_count"] == "1"
        assert event_rows[0]["source_capability_recovery_count"] == "0"
        assert event_rows[0]["validation_recovery_count"] == "0"
        assert event_rows[0]["pc_learning_contract_last"] == "local_factor_vjp_v1"
        assert event_rows[0]["pc_execution_contract_version_last"] == "1"
        assert event_rows[0]["pc_activity_derivative_contract_last"] == "analytic_local"
        assert event_rows[0]["pc_parameter_derivative_contract_last"] == "analytic_local"
        assert event_rows[0]["pc_global_autodiff_graph_last"] == "False"
        assert event_rows[0]["pc_global_backward_calls_total"] == "0"
        assert event_rows[0]["pc_local_vjp_calls_total"] == "12"
        assert event_rows[0]["pc_temporal_state_vjp_calls_total"] == "4"
        assert event_rows[0]["pc_fused_temporal_vjp_calls_total"] == "4"
        assert event_rows[0]["pc_temporal_credit_mode_last"] == "exact_window"
        assert event_rows[0]["pc_temporal_window_chunks_last"] == "2"
        assert event_rows[0]["pc_direct_forward_updates_total"] == "6"
        assert event_rows[0]["pc_feedback_parameter_updates_total"] == "6"
        assert event_rows[0]["pc_local_parameter_update_intents_total"] == "7"
        assert event_rows[0]["pc_parameter_updates_total"] == "3"
        assert event_rows[0]["pc_clip_fraction_mean_last"] == "0.125"
        assert event_rows[0]["pc_constraint_rms_last"] == "0.01"
        assert event_rows[0]["pc_dual_rms_last"] == "0.02"
        assert event_rows[0]["pc_composite_signal_rms_last"] == "0.03"
        normalized = list(csv.DictReader((out / "normalized_summary.csv").open()))
        event_normalized = next(row for row in normalized if row["run"] == "run-a")
        assert event_normalized["matrix"] == "smoke"
        assert event_normalized["backend"] == "cpu"
        assert event_normalized["profile"] == "profile.toml"
        assert event_normalized["batch_size"] == "8"
        assert event_normalized["checkpoint_interval_iters"] == "2"
        assert event_normalized["ruliad_policy_probe_every_epochs"] == "1"
        assert event_normalized["tok_s"] == "512.0"
        assert event_normalized["model_tok_s"] == "640.0"
        assert event_normalized["model_duty_fraction"] == "0.8"
        assert event_normalized["validation_objective"] == "source_weighted"
        assert event_normalized["validation_sampling"] == "fixed_holdout"
        assert event_normalized["fixed_holdout_loss_first_checkpoint"] == "0.45"
        assert event_normalized["lr_last"] == "0.00025"
        assert event_normalized["source_loss_cadence_mean"] == "0.5"
        assert event_normalized["ruliad_verifier_accuracy"] == "0.25"
        assert event_normalized["ruliad_constrained_equivalent_top1"] == "0.75"
        assert event_normalized["ruliad_constrained_free_accuracy_gap"] == "0.5"
        assert event_normalized["ruliad_partial_progress"] == "0.5"
        assert event_normalized["ruliad_deployment_capability_gate_passed"] == "1.0"
        assert event_normalized["checkpoint_promoted_count"] == "1.0"
        assert event_normalized["checkpoint_last_epoch"] == "3.0"
        assert event_normalized["checkpoint_last_absolute_step"] == "255.0"
        assert event_normalized["ruliad_policy_solve_rate_best"] == "0.6"
        assert event_normalized["ruliad_policy_solve_rate_promoted"] == "0.5"
        assert event_normalized["checkpoint_promotion_ineligible_count"] == "1.0"
        assert event_normalized["capability_statistical_regression_count"] == "1.0"
        assert event_normalized["rollback_recovery_count"] == "1.0"
        assert event_normalized["checkpoint_last_promoted_epoch"] == "3.0"
        assert event_normalized["ruliad_context_swap_top1_change_rate"] == "0.375"
        assert (
            event_normalized[
                "ruliad_counterfactual_target_equivalent_probability_gain"
            ]
            == "0.125"
        )
        coverage = list(csv.DictReader((out / "source_capability_coverage.csv").open()))
        assert coverage[0]["candidate_coverage"] == "0.75"
        capability_groups = list(
            csv.DictReader((out / "capability_group_trajectory.csv").open())
        )
        assert capability_groups[0]["kind"] == "difficulty"
        assert capability_groups[0]["label"] == "d2"
        assert capability_groups[0]["verifier_rate"] == "0.375"
        difficulty_bucket = next(row for row in buckets if row["kind"] == "difficulty")
        assert difficulty_bucket["capability_verifier_ema"] == "0.6"
        gpu = list(csv.DictReader((out / "gpu_summary.csv").open()))
        assert gpu[0]["arm"] == "adamwpc"
        assert gpu[0]["util_mean"] == "90.0"
        relative_gpu_path = Path(os.path.relpath(gpu_path, Path.cwd()))
        relative_gpu = read_gpu_csvs(
            [relative_gpu_path], read_manifests([manifests / "run-a.json"])
        )
        assert relative_gpu[0]["trial_key"] == "pc-smoke-run-a"
        assert relative_gpu[0]["arm"] == "adamwpc"
        assert relative_gpu[0]["seed"] == 7
        three_sample_stats = stats([1.0, 2.0, 3.0])
        assert math.isclose(three_sample_stats.ci95, 4.303 / math.sqrt(3.0))
        markdown = (out / "paper_tables.md").read_text()
        assert "Run Summary" in markdown
        assert "Local Learning Contract" in markdown
        assert (
            infer_proof_policy_mode("local_pc_fixed_verifier_paired_dagger")
            == "static_then_paired_dagger"
        )
        assert infer_proof_policy_mode("local_pc_fixed_verifier_dagger") == "dagger"
        assert infer_proof_policy_mode("local_pc_fixed_verifier_static") == "static_expert"
        assert (
            canonical_matrix("local-verifier-closed-loop", False)
            == "local-verifier-source-frozen"
        )
        assert (
            canonical_matrix("local-verifier-closed-loop", True)
            == "local-verifier-closed-loop"
        )
        assert experiment_context(
            {
                "matrix": "policy",
                "backend": "cuda",
                "profile": "one.toml",
                "iters": 4,
                "batch_size": 8,
                "source_selection_feedback_updates_enabled": False,
                "proof_policy_mode": "static_expert",
            }
        ) != experiment_context(
            {
                "matrix": "policy",
                "backend": "cuda",
                "profile": "one.toml",
                "iters": 4,
                "batch_size": 8,
                "source_selection_feedback_updates_enabled": False,
                "proof_policy_mode": "dagger",
            }
        )
        batch_rows = [
            {
                "iters": 4,
                "matrix": "batch-screen",
                "backend": "cuda",
                "profile": "one.toml",
                "arm": "adamw",
                "seed": 1,
                "batch_size": 8,
                "valid_mean": 1.0,
            },
            {
                "iters": 4,
                "matrix": "batch-screen",
                "backend": "cuda",
                "profile": "one.toml",
                "arm": "adamwpc",
                "seed": 1,
                "batch_size": 16,
                "valid_mean": 0.5,
            },
        ]
        grouped_batches = grouped_summary(batch_rows)
        assert len(grouped_batches) == 2, "batch sweeps must remain separate"
        assert not paired_deltas(
            batch_rows, "adamw", "adamwpc"
        ), "different batch sizes must never be paired"
        profile_rows = [
            {**batch_rows[0], "batch_size": 8},
            {**batch_rows[1], "batch_size": 8, "profile": "two.toml"},
        ]
        assert not paired_deltas(
            profile_rows, "adamw", "adamwpc"
        ), "different profiles must never be paired"
        cadence_rows = [
            {
                **batch_rows[0],
                "checkpoint_interval_iters": 2,
                "ruliad_policy_probe_every_epochs": 1,
            },
            {
                **batch_rows[1],
                "batch_size": 8,
                "checkpoint_interval_iters": 4,
                "ruliad_policy_probe_every_epochs": 4,
            },
        ]
        assert not paired_deltas(
            cadence_rows, "adamw", "adamwpc"
        ), "different checkpoint/probe cadences must never be paired"
        coincident_run_names = [
            {**batch_rows[0], "run": "same-name"},
            {**batch_rows[0], "run": "same-name", "matrix": "other-matrix"},
        ]
        assert len(merge_summary_rows(coincident_run_names, [])) == 2
        print("self-test ok")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    run_analysis(args.inputs, Path(args.out_dir), args.baseline, args.compare)


if __name__ == "__main__":
    main()
