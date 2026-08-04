#!/usr/bin/env python3
"""Analyze matched Ruliad structural-holdout policy-supervision experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import tomllib
from pathlib import Path
from typing import Any, Iterable


ROOT_DIR = Path(__file__).resolve().parents[1]
BASELINE_ARMS = ("seed_ce", "structural_ce")
CANDIDATE_ARMS = (
    "structural_energy_static025",
    "structural_energy_head_only025",
    "structural_energy_head_only_fullrate100",
    "structural_energy_fullrate100",
    "structural_semantic_value_binding",
    "structural_energy_value_binding025",
    "structural_semantic_ce",
    "structural_semantic_static025",
    "structural_semantic_language_head_only025",
    "structural_semantic_static_dense025",
    "structural_semantic_static_prefix025",
    "structural_semantic_static_marginal025",
    "structural_values",
    "structural_value_balanced",
    "structural_static025",
    "structural_static_marginal025",
    "structural_static_orbit_marginal025",
    "structural_static_orbit_worst_marginal025",
    "structural_dagger025",
    "structural_dagger_marginal025",
    "structural_bc_paired_dagger_marginal025",
    "structural_bc_paired_dagger_orbit_marginal025",
)
SEMANTIC_ENERGY_ARMS = {
    "structural_energy_static025",
    "structural_energy_head_only025",
    "structural_energy_head_only_fullrate100",
    "structural_energy_fullrate100",
    "structural_energy_value_binding025",
}
COUNTERFACTUAL_TARGET_ARMS = SEMANTIC_ENERGY_ARMS | {
    "structural_semantic_language_head_only025"
}
DEFAULT_CANDIDATE_ARM = "structural_dagger025"
ARM_CONTRACTS = {
    "seed_ce": ("seed_disjoint_v1", "mode:seed_disjoint_v1"),
    "structural_ce": ("structural_holdout_v1", "mode:structural_validation_v1"),
    "structural_energy_static025": (
        "structural_holdout_v1",
        "mode:structural_validation_v1",
    ),
    "structural_energy_head_only025": (
        "structural_holdout_v1",
        "mode:structural_validation_v1",
    ),
    "structural_energy_head_only_fullrate100": (
        "structural_holdout_v1",
        "mode:structural_validation_v1",
    ),
    "structural_energy_fullrate100": (
        "structural_holdout_v1",
        "mode:structural_validation_v1",
    ),
    "structural_semantic_value_binding": (
        "structural_holdout_v1",
        "mode:structural_validation_v1",
    ),
    "structural_energy_value_binding025": (
        "structural_holdout_v1",
        "mode:structural_validation_v1",
    ),
    "structural_semantic_ce": (
        "structural_holdout_v1",
        "mode:structural_validation_v1",
    ),
    "structural_semantic_static025": (
        "structural_holdout_v1",
        "mode:structural_validation_v1",
    ),
    "structural_semantic_language_head_only025": (
        "structural_holdout_v1",
        "mode:structural_validation_v1",
    ),
    "structural_semantic_static_dense025": (
        "structural_holdout_v1",
        "mode:structural_validation_v1",
    ),
    "structural_semantic_static_prefix025": (
        "structural_holdout_v1",
        "mode:structural_validation_v1",
    ),
    "structural_semantic_static_marginal025": (
        "structural_holdout_v1",
        "mode:structural_validation_v1",
    ),
    "structural_values": (
        "structural_holdout_v1",
        "mode:structural_validation_v1",
    ),
    "structural_value_balanced": (
        "structural_holdout_v1",
        "mode:structural_validation_v1",
    ),
    "structural_static025": (
        "structural_holdout_v1",
        "mode:structural_validation_v1",
    ),
    "structural_static_marginal025": (
        "structural_holdout_v1",
        "mode:structural_validation_v1",
    ),
    "structural_static_orbit_marginal025": (
        "structural_holdout_v1",
        "mode:structural_validation_v1",
    ),
    "structural_static_orbit_worst_marginal025": (
        "structural_holdout_v1",
        "mode:structural_validation_v1",
    ),
    "structural_dagger025": (
        "structural_holdout_v1",
        "mode:structural_validation_v1",
    ),
    "structural_dagger_marginal025": (
        "structural_holdout_v1",
        "mode:structural_validation_v1",
    ),
    "structural_bc_paired_dagger_marginal025": (
        "structural_holdout_v1",
        "mode:structural_validation_v1",
    ),
    "structural_bc_paired_dagger_orbit_marginal025": (
        "structural_holdout_v1",
        "mode:structural_validation_v1",
    ),
}
ARM_ANSWER_CONTRACTS = {
    arm: (
        "semantic_step"
        if arm.startswith("structural_semantic") or arm in SEMANTIC_ENERGY_ARMS
        else "presentation_index"
    )
    for arm in (*BASELINE_ARMS, *CANDIDATE_ARMS)
}
ARM_POLICY_SCORING = {
    arm: "semantic_energy" if arm in SEMANTIC_ENERGY_ARMS else "completion_likelihood"
    for arm in (*BASELINE_ARMS, *CANDIDATE_ARMS)
}
METRICS = (
    "valid_ce",
    "correctness_verifier",
    "correctness_partial",
    "correctness_constrained_items",
    "correctness_constrained_equivalent_top1",
    "correctness_constrained_preferred_top1",
    "correctness_constrained_equivalent_nll",
    "correctness_constrained_valid_invalid_margin",
    "correctness_constrained_canonical_equivalent_top1",
    "correctness_constrained_canonical_preferred_top1",
    "correctness_constrained_canonical_equivalent_nll",
    "correctness_constrained_canonical_valid_invalid_margin",
    "correctness_constrained_worst_presentation_equivalent_top1",
    "correctness_constrained_worst_presentation_equivalent_nll",
    "correctness_constrained_worst_presentation_valid_invalid_margin",
    "correctness_constrained_orbit_js_divergence",
    "correctness_constrained_orbit_top1_consensus",
    "correctness_constrained_complete_orbit_items",
    "correctness_constrained_presentation_rows",
    "correctness_constrained_presentation_equivalent_top1",
    "correctness_constrained_presentation_preferred_top1",
    "correctness_constrained_context_swap_items",
    "correctness_constrained_context_swap_equivalent_top1",
    "correctness_constrained_context_swap_equivalent_nll",
    "correctness_constrained_context_swap_top1_change",
    "correctness_constrained_context_swap_equivalent_probability_drop",
    "correctness_constrained_context_swap_js_divergence",
    "correctness_constrained_counterfactual_target_items",
    "correctness_constrained_counterfactual_target_equivalent_top1",
    "correctness_constrained_counterfactual_target_equivalent_nll",
    "correctness_constrained_counterfactual_target_top1_change",
    "correctness_constrained_counterfactual_target_equivalent_probability_gain",
    "correctness_constrained_counterfactual_target_js_divergence",
    "completion_health",
    "actual_answer_distinct_fraction",
    "expected_answer_distinct_fraction",
    "actual_answer_dominant_fraction",
    "schema_wrong",
    "malformed",
    "policy_solve",
    "policy_goal_completion",
    "policy_valid_action",
    "policy_top1_expert",
    "policy_runtime_gate_passed",
    "model_tokens_per_second",
    "main_model_tokens_per_second",
    "wall_tokens_per_second",
    "model_duty_fraction",
    "auxiliary_objective_fraction",
    "proof_policy_fraction",
    "train_compute_fraction",
    "optimizer_fraction",
    "metric_sync_fraction",
    "accounted_fraction",
    "validation_fraction",
    "dataloader_cpu_thread_fraction",
    "dataloader_foreground_wait_fraction",
    "gpu_util_mean",
    "gpu_power_mean",
    "gpu_active_util_mean",
    "gpu_active_power_mean",
    "gpu_active_sm_clock_mean_mhz",
    "gpu_active_sm_clock_min_mhz",
    "gpu_active_memory_util_mean",
    "gpu_active_temperature_max_c",
    "gpu_high_util_fraction",
    "gpu_low_util_fraction",
    "gpu_max_consecutive_sub80_samples",
    "gpu_max_consecutive_idle_samples",
    "gpu_active_power_cv",
    "elapsed_seconds",
    "peak_used_mb",
)
PAIRED_METRICS = (
    "correctness_verifier",
    "correctness_partial",
    "correctness_constrained_equivalent_top1",
    "correctness_constrained_equivalent_nll",
    "correctness_constrained_valid_invalid_margin",
    "correctness_constrained_canonical_equivalent_top1",
    "correctness_constrained_canonical_equivalent_nll",
    "correctness_constrained_worst_presentation_equivalent_top1",
    "correctness_constrained_worst_presentation_equivalent_nll",
    "correctness_constrained_orbit_js_divergence",
    "correctness_constrained_orbit_top1_consensus",
    "correctness_constrained_presentation_equivalent_top1",
    "correctness_constrained_context_swap_equivalent_top1",
    "correctness_constrained_context_swap_equivalent_probability_drop",
    "correctness_constrained_context_swap_js_divergence",
    "correctness_constrained_counterfactual_target_equivalent_top1",
    "correctness_constrained_counterfactual_target_equivalent_probability_gain",
    "correctness_constrained_counterfactual_target_js_divergence",
    "policy_solve",
    "policy_goal_completion",
    "policy_top1_expert",
    "valid_ce",
    "model_tokens_per_second",
    "dataloader_foreground_wait_fraction",
    "elapsed_seconds",
)
T_CRITICAL_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
SEMANTIC_PROMPT_MARKERS = (
    '"equational_monoid_normalization"',
    '"free_category_path_normalization"',
    '"propositional_normalization"',
    '"regular_language_normalization"',
    '"rho_process_structural_congruence"',
    '"metagraph_pattern_rewriting"',
    '"add_zero_left"',
    '"mul_one_right"',
    '"double_negation"',
    '"identity_left"',
    '"identity_right"',
    '"and_top"',
    '"or_bottom"',
    '"epsilon_prefix"',
    '"empty_union"',
    '"encode_decode"',
    '"nil_parallel"',
    '"nil_choice"',
    '"quote_eval"',
    '"empty_merge"',
    '"empty_overlay"',
    '"quote_unquote"',
    '"compose"',
    '"functor_map"',
    '"conjoin_diagrams"',
    '"under_assumption"',
    '"left_quotient"',
    '"grounded_scope"',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Matrix output root")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--expected-seeds", default="1337,2027,9001")
    parser.add_argument("--minimum-promotion-iters", type=int, default=1024)
    parser.add_argument(
        "--candidate-arm",
        choices=CANDIDATE_ARMS,
        default=DEFAULT_CANDIDATE_ARM,
    )
    parser.add_argument(
        "--comparison-arm",
        choices=(*BASELINE_ARMS, *CANDIDATE_ARMS),
        default="structural_ce",
        help="Matched causal baseline for candidate deltas and non-inferiority gates",
    )
    return parser.parse_args()


def finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def fmt(value: Any) -> str:
    parsed = finite(value)
    if parsed is None:
        return "n/a"
    if abs(parsed) >= 1000:
        return f"{parsed:.1f}"
    return f"{parsed:.4f}"


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {error}") from error
            if isinstance(value, dict):
                rows.append(value)
    return rows


def mean_ci95(values: Iterable[float | None]) -> tuple[float | None, float | None]:
    clean = [value for value in values if value is not None and math.isfinite(value)]
    if not clean:
        return None, None
    center = statistics.mean(clean)
    if len(clean) < 2:
        return center, None
    sem = statistics.stdev(clean) / math.sqrt(len(clean))
    t_value = T_CRITICAL_95.get(len(clean) - 1, 1.96)
    return center, t_value * sem


def metric_last(
    events: list[dict[str, Any]],
    name: str,
    *,
    split: str | None = None,
    running: bool = False,
) -> float | None:
    for event in reversed(events):
        if (
            event.get("type") == "metric"
            and event.get("name") == name
            and (split is None or event.get("split") == split)
        ):
            value = finite(event.get("running_value" if running else "value"))
            if value is not None:
                return value
    return None


def probe_rows(run_dir: Path, probe_name: str) -> list[dict[str, Any]]:
    return [
        row
        for row in read_jsonl(run_dir / "events" / "capability_probe.jsonl")
        if row.get("probe_name") == probe_name
    ]


def group_labels(probe: dict[str, Any]) -> set[str]:
    return {
        str(bucket.get("label"))
        for bucket in probe.get("group_buckets") or []
        if isinstance(bucket, dict) and bucket.get("label")
    }


def corpus_config(run_dir: Path) -> dict[str, Any] | None:
    snapshot_path = run_dir / "training_config.json"
    if not snapshot_path.exists():
        return None
    snapshot = read_json(snapshot_path)
    dataset = snapshot.get("dataset") or {}
    config_path = dataset.get("config") if isinstance(dataset, dict) else None
    if not config_path:
        return None
    path = Path(str(config_path))
    if not path.is_absolute():
        path = ROOT_DIR / path
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return tomllib.load(handle)


def corpus_contract(run_dir: Path) -> str | None:
    config = corpus_config(run_dir)
    return (
        str(config.get("formal_generalization") or "seed_disjoint_v1")
        if config is not None
        else None
    )


def proof_action_answer_contract(run_dir: Path) -> str | None:
    config = corpus_config(run_dir)
    if config is None:
        return None
    source_selection = config.get("source_selection") or {}
    formal_task_mix = source_selection.get("formal_task_mix") or {}
    return str(
        formal_task_mix.get("proof_action_answer_contract") or "presentation_index"
    )


def parse_number(value: str) -> float | None:
    token = value.strip().split()[0].rstrip("%,") if value.strip() else ""
    return finite(token)


def longest_matching_run(values: Iterable[float], predicate: Any) -> int:
    longest = 0
    current = 0
    for value in values:
        if predicate(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def gpu_stats(
    path: Path | None,
) -> tuple[
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
]:
    if path is None or not path.exists():
        return (None,) * 13
    utilization: list[float] = []
    power: list[float] = []
    active_utilization: list[float] = []
    active_power: list[float] = []
    active_sm_clocks: list[float] = []
    active_memory_utilization: list[float] = []
    active_temperatures: list[float] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            util_key = next((key for key in row if "utilization.gpu" in key), None)
            power_key = next((key for key in row if "power.draw" in key), None)
            sm_clock_key = next((key for key in row if "clocks.current.sm" in key), None)
            memory_util_key = next(
                (key for key in row if "utilization.memory" in key), None
            )
            temperature_key = next((key for key in row if "temperature.gpu" in key), None)
            util = parse_number(row.get(util_key) or "") if util_key else None
            row_power = parse_number(row.get(power_key) or "") if power_key else None
            if util is not None:
                utilization.append(util)
            if row_power is not None:
                power.append(row_power)
            if util is None or util < 50.0:
                continue
            active_utilization.append(util)
            if row_power is not None:
                active_power.append(row_power)
            for key, values in (
                (sm_clock_key, active_sm_clocks),
                (memory_util_key, active_memory_utilization),
                (temperature_key, active_temperatures),
            ):
                value = parse_number(row.get(key) or "") if key else None
                if value is not None:
                    values.append(value)
    active_power_mean = statistics.mean(active_power) if active_power else None
    high_indices = [index for index, value in enumerate(utilization) if value >= 80.0]
    active_window = (
        utilization[high_indices[0] : high_indices[-1] + 1]
        if high_indices
        else utilization
    )
    return (
        statistics.mean(utilization) if utilization else None,
        statistics.mean(power) if power else None,
        statistics.mean(active_utilization) if active_utilization else None,
        active_power_mean,
        statistics.mean(active_sm_clocks) if active_sm_clocks else None,
        min(active_sm_clocks) if active_sm_clocks else None,
        statistics.mean(active_memory_utilization)
        if active_memory_utilization
        else None,
        max(active_temperatures) if active_temperatures else None,
        sum(u >= 80.0 for u in utilization) / len(utilization) if utilization else None,
        sum(u <= 20.0 for u in utilization) / len(utilization) if utilization else None,
        float(longest_matching_run(active_window, lambda value: value < 80.0))
        if active_window
        else None,
        float(longest_matching_run(active_window, lambda value: value <= 20.0))
        if active_window
        else None,
        (
            statistics.pstdev(active_power) / active_power_mean
            if len(active_power) > 1 and active_power_mean and active_power_mean > 0.0
            else None
        ),
    )


def analysis_rows(arm_dir: Path) -> dict[str, dict[str, str]]:
    path = arm_dir / "analysis" / "latent_reasoning_steps_summary.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {
            str(row.get("trial_key")): row
            for row in csv.DictReader(handle)
            if row.get("trial_key")
        }


def histogram_summary(histogram: dict[str, Any]) -> tuple[float | None, float | None]:
    counts = [finite(value) or 0.0 for value in histogram.values()]
    total = sum(counts)
    if total <= 0.0:
        return None, None
    probabilities = [count / total for count in counts if count > 0.0]
    entropy = -sum(probability * math.log2(probability) for probability in probabilities)
    return entropy, max(probabilities)


def dagger_telemetry(run_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(run_dir / "events" / "ruliad_proof_policy_dagger.jsonl")
    active = [row for row in rows if not row.get("skip_reason")]
    selected: dict[str, float] = {}
    model: dict[str, float] = {}
    for row in active:
        for key, value in (row.get("expert_selected_index_histogram") or {}).items():
            selected[str(key)] = selected.get(str(key), 0.0) + (finite(value) or 0.0)
        for key, value in (row.get("model_selected_index_histogram") or {}).items():
            model[str(key)] = model.get(str(key), 0.0) + (finite(value) or 0.0)
    selected_entropy, selected_dominant = histogram_summary(selected)
    model_entropy, model_dominant = histogram_summary(model)
    expert_rows = sum(int(finite(row.get("expert_rows")) or 0) for row in active)
    semantic_state_rows = sum(
        int(finite(row.get("semantic_state_rows")) or finite(row.get("expert_rows")) or 0)
        for row in active
    )
    base_semantic_state_rows = sum(
        int(finite(row.get("base_semantic_state_rows")) or 0) for row in active
    )
    counterfactual_semantic_state_rows = sum(
        int(finite(row.get("counterfactual_semantic_state_rows")) or 0)
        for row in active
    )
    counterfactual_target_shortfall = sum(
        int(finite(row.get("counterfactual_target_shortfall")) or 0) for row in active
    )
    presentation_rows = sum(
        int(
            finite(row.get("supervised_presentation_rows"))
            or finite(row.get("expert_rows"))
            or 0
        )
        for row in active
    )
    presentation_rows_max = max(
        (
            int(
                finite(row.get("supervised_presentation_rows"))
                or finite(row.get("expert_rows"))
                or 0
            )
            for row in active
        ),
        default=0,
    )
    presentation_row_budget_min = min(
        (
            int(finite(row.get("max_presentation_rows_per_update")) or 0)
            for row in active
        ),
        default=0,
    )
    semantic_row_budget_min = min(
        (int(finite(row.get("semantic_row_budget")) or 0) for row in active),
        default=0,
    )
    static_expert_rows = sum(
        int(finite(row.get("static_expert_rows")) or 0) for row in active
    )
    on_policy_expert_rows = sum(
        int(finite(row.get("dagger_expert_rows")) or 0) for row in active
    )
    paired_rows = [row for row in active if row.get("mode") == "paired_dagger"]
    paired_static_expert_rows = sum(
        int(finite(row.get("static_expert_rows")) or 0) for row in paired_rows
    )
    paired_on_policy_expert_rows = sum(
        int(finite(row.get("dagger_expert_rows")) or 0) for row in paired_rows
    )
    model_visited_expert_rows = sum(
        int(finite(row.get("model_visited_expert_rows")) or 0) for row in active
    )
    candidate_tokens = sum(
        int(finite(row.get("candidate_target_tokens")) or 0) for row in active
    )
    equivalent_tokens = sum(
        int(finite(row.get("equivalent_target_tokens")) or 0) for row in active
    )
    prefix_branch_rows = sum(
        int(finite(row.get("prefix_branch_rows")) or 0) for row in active
    )
    prefix_candidate_tokens = sum(
        int(finite(row.get("prefix_candidate_tokens")) or 0) for row in active
    )
    prefix_equivalent_tokens = sum(
        int(finite(row.get("prefix_equivalent_tokens")) or 0) for row in active
    )
    model_scoring_padded_tokens = sum(
        int(finite(row.get("model_scoring_padded_tokens")) or 0) for row in active
    )
    model_scoring_ms = sum(finite(row.get("model_scoring_ms")) or 0.0 for row in active)
    objectives = sorted({str(row.get("objective") or "") for row in active})
    gradient_scopes = sorted({str(row.get("gradient_scope") or "") for row in active})
    answer_contracts = sorted({str(row.get("answer_contract") or "") for row in active})
    presentation_risks = sorted(
        {str(row.get("presentation_risk") or "") for row in active}
    )
    configured_modes = sorted({str(row.get("configured_mode") or "") for row in active})
    modes = sorted({str(row.get("mode") or "") for row in active})
    symmetries = sorted({str(row.get("candidate_symmetry") or "") for row in active})
    return {
        "dagger_calls": len(active),
        "dagger_expert_rows": expert_rows,
        "dagger_semantic_state_rows": semantic_state_rows,
        "dagger_base_semantic_state_rows": base_semantic_state_rows,
        "dagger_counterfactual_semantic_state_rows": counterfactual_semantic_state_rows,
        "dagger_counterfactual_target_shortfall": counterfactual_target_shortfall,
        "dagger_configured_counterfactual_targets_per_state": min(
            (
                int(finite(row.get("configured_counterfactual_targets_per_state")) or 0)
                for row in active
            ),
            default=0,
        ),
        "dagger_presentation_rows": presentation_rows,
        "dagger_presentation_rows_max": presentation_rows_max,
        "dagger_presentation_row_budget_min": presentation_row_budget_min,
        "dagger_semantic_row_budget_min": semantic_row_budget_min,
        "dagger_presentations_per_state": (
            presentation_rows / semantic_state_rows if semantic_state_rows else None
        ),
        "dagger_static_expert_rows": static_expert_rows,
        "dagger_on_policy_expert_rows": on_policy_expert_rows,
        "dagger_paired_static_expert_rows": paired_static_expert_rows,
        "dagger_paired_on_policy_expert_rows": paired_on_policy_expert_rows,
        "dagger_model_visited_expert_rows": model_visited_expert_rows,
        "dagger_rollout_depth_reached_max": max(
            (int(finite(row.get("rollout_depth_reached")) or 0) for row in active),
            default=0,
        ),
        "dagger_telemetry_version_min": min(
            (int(finite(row.get("version")) or 0) for row in active), default=0
        ),
        "dagger_objective": objectives[0] if len(objectives) == 1 else ",".join(objectives),
        "dagger_gradient_scope": (
            gradient_scopes[0]
            if len(gradient_scopes) == 1
            else ",".join(gradient_scopes)
        ),
        "dagger_answer_contract": (
            answer_contracts[0]
            if len(answer_contracts) == 1
            else ",".join(answer_contracts)
        ),
        "dagger_presentation_risk": (
            presentation_risks[0]
            if len(presentation_risks) == 1
            else ",".join(presentation_risks)
        ),
        "dagger_configured_mode": (
            configured_modes[0]
            if len(configured_modes) == 1
            else ",".join(configured_modes)
        ),
        "dagger_mode": modes[0] if len(modes) == 1 else ",".join(modes),
        "dagger_candidate_symmetry": (
            symmetries[0] if len(symmetries) == 1 else ",".join(symmetries)
        ),
        "dagger_candidate_targets_per_row": (
            candidate_tokens / presentation_rows if presentation_rows else None
        ),
        "dagger_equivalent_targets_per_row": (
            equivalent_tokens / presentation_rows if presentation_rows else None
        ),
        "dagger_prefix_branch_rows": prefix_branch_rows,
        "dagger_prefix_candidate_tokens": prefix_candidate_tokens,
        "dagger_prefix_equivalent_tokens": prefix_equivalent_tokens,
        "dagger_expert_index_entropy_bits": selected_entropy,
        "dagger_expert_index_dominant_fraction": selected_dominant,
        "dagger_model_index_entropy_bits": model_entropy,
        "dagger_model_index_dominant_fraction": model_dominant,
        "dagger_sampling_model_materialize_ms": sum(
            finite(row.get("sampling_model_materialize_ms")) or 0.0 for row in active
        ),
        "dagger_state_prepare_ms": sum(
            finite(row.get("state_prepare_ms")) or 0.0 for row in active
        ),
        "dagger_rollout_cpu_prepare_ms": sum(
            finite(row.get("rollout_cpu_prepare_ms")) or 0.0 for row in active
        ),
        "dagger_model_scoring_ms": model_scoring_ms,
        "dagger_model_scoring_padded_tokens": model_scoring_padded_tokens,
        "dagger_model_scoring_padded_tokens_per_second": (
            model_scoring_padded_tokens * 1_000.0 / model_scoring_ms
            if model_scoring_ms > 0.0
            else None
        ),
    }


def prompt_semantic_leakage(run_dir: Path) -> tuple[int, int]:
    rows = read_jsonl(run_dir / "events" / "ruliad_completion_samples.jsonl")
    prompts = [str(row.get("prompt") or "") for row in rows]
    leak_count = sum(
        1
        for prompt in prompts
        if any(marker in prompt for marker in SEMANTIC_PROMPT_MARKERS)
    )
    return len(prompts), leak_count


def summarize_trial(arm: str, manifest_path: Path, analysis: dict[str, dict[str, str]]) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    trial_key = str(manifest.get("trial_key") or manifest_path.stem)
    run_dir = Path(str(manifest.get("run_dir") or ""))
    events = read_jsonl(run_dir / "events" / "training_events.jsonl") if run_dir else []
    correctness = probe_rows(run_dir, "ruliad_correctness") if run_dir else []
    policy = probe_rows(run_dir, "ruliad_proof_policy_rollout") if run_dir else []
    latest_correctness = correctness[-1] if correctness else {}
    latest_policy = policy[-1] if policy else {}
    expected_contract, expected_mode = ARM_CONTRACTS[arm]
    labels = group_labels(latest_correctness)
    correctness_values = [finite(row.get("verifier_rate")) for row in correctness]
    policy_values = [finite(row.get("verifier_rate")) for row in policy]
    correctness_clean = [value for value in correctness_values if value is not None]
    policy_clean = [value for value in policy_values if value is not None]
    gpu_log = Path(str(manifest.get("gpu_log_path"))) if manifest.get("gpu_log_path") else None
    (
        gpu_util,
        gpu_power,
        gpu_active_util,
        gpu_active_power,
        gpu_active_sm_clock_mean,
        gpu_active_sm_clock_min,
        gpu_active_memory_util_mean,
        gpu_active_temperature_max,
        gpu_high_util,
        gpu_low_util,
        gpu_max_consecutive_sub80,
        gpu_max_consecutive_idle,
        gpu_active_power_cv,
    ) = gpu_stats(gpu_log)
    latent = analysis.get(trial_key, {})
    dagger = dagger_telemetry(run_dir) if run_dir else {}
    audited_prompts, semantic_leaks = (
        prompt_semantic_leakage(run_dir) if run_dir else (0, 0)
    )

    return {
        "arm": arm,
        "trial_key": trial_key,
        "seed": int(manifest.get("seed")),
        "backend": str(manifest.get("backend") or "unknown"),
        "status": manifest.get("status"),
        "max_iters": int(manifest.get("max_iters") or 0),
        "batch_size": int(manifest.get("batch_size") or 0),
        "block_size": int(manifest.get("block_size") or 0),
        "run_dir": str(run_dir),
        "generalization_contract": corpus_contract(run_dir) if run_dir else None,
        "proof_action_answer_contract": (
            proof_action_answer_contract(run_dir) if run_dir else None
        ),
        "expected_answer_contract": ARM_ANSWER_CONTRACTS[arm],
        "expected_contract": expected_contract,
        "expected_mode": expected_mode,
        "mode_telemetry_present": expected_mode in labels,
        "probe_count": len(correctness),
        "policy_probe_count": len(policy),
        "correctness_verifier": finite(latest_correctness.get("verifier_rate")),
        "correctness_partial": finite(latest_correctness.get("partial_credit_rate")),
        "correctness_constrained_items": metric_last(
            events, "Ruliad Correctness Constrained Items"
        ),
        "correctness_constrained_equivalent_top1": metric_last(
            events, "Ruliad Correctness Constrained Equivalent Top-1 Rate"
        ),
        "correctness_constrained_preferred_top1": metric_last(
            events, "Ruliad Correctness Constrained Preferred Top-1 Rate"
        ),
        "correctness_constrained_equivalent_nll": metric_last(
            events, "Ruliad Correctness Constrained Equivalent NLL"
        ),
        "correctness_constrained_valid_invalid_margin": metric_last(
            events, "Ruliad Correctness Constrained Valid-Invalid Margin"
        ),
        "correctness_constrained_canonical_equivalent_top1": metric_last(
            events,
            "Ruliad Correctness Constrained Canonical Equivalent Top-1 Rate",
        ),
        "correctness_constrained_canonical_preferred_top1": metric_last(
            events,
            "Ruliad Correctness Constrained Canonical Preferred Top-1 Rate",
        ),
        "correctness_constrained_canonical_equivalent_nll": metric_last(
            events, "Ruliad Correctness Constrained Canonical Equivalent NLL"
        ),
        "correctness_constrained_canonical_valid_invalid_margin": metric_last(
            events,
            "Ruliad Correctness Constrained Canonical Valid-Invalid Margin",
        ),
        "correctness_constrained_worst_presentation_equivalent_top1": metric_last(
            events,
            "Ruliad Correctness Constrained Worst-Presentation Equivalent Top-1 Rate",
        ),
        "correctness_constrained_worst_presentation_equivalent_nll": metric_last(
            events,
            "Ruliad Correctness Constrained Worst-Presentation Equivalent NLL",
        ),
        "correctness_constrained_worst_presentation_valid_invalid_margin": metric_last(
            events,
            "Ruliad Correctness Constrained Worst-Presentation Valid-Invalid Margin",
        ),
        "correctness_constrained_orbit_js_divergence": metric_last(
            events, "Ruliad Correctness Constrained Orbit JS Divergence"
        ),
        "correctness_constrained_orbit_top1_consensus": metric_last(
            events,
            "Ruliad Correctness Constrained Orbit Top-1 Consensus Fraction",
        ),
        "correctness_constrained_complete_orbit_items": metric_last(
            events, "Ruliad Correctness Constrained Complete Orbit Items"
        ),
        "correctness_constrained_presentation_rows": metric_last(
            events, "Ruliad Correctness Constrained Presentation Rows"
        ),
        "correctness_constrained_presentation_equivalent_top1": metric_last(
            events,
            "Ruliad Correctness Constrained Presentation Equivalent Top-1 Rate",
        ),
        "correctness_constrained_presentation_preferred_top1": metric_last(
            events,
            "Ruliad Correctness Constrained Presentation Preferred Top-1 Rate",
        ),
        "correctness_constrained_context_swap_items": metric_last(
            events, "Ruliad Correctness Constrained Context-Swap Items"
        ),
        "correctness_constrained_context_swap_equivalent_top1": metric_last(
            events,
            "Ruliad Correctness Constrained Context-Swap Equivalent Top-1 Rate",
        ),
        "correctness_constrained_context_swap_equivalent_nll": metric_last(
            events, "Ruliad Correctness Constrained Context-Swap Equivalent NLL"
        ),
        "correctness_constrained_context_swap_top1_change": metric_last(
            events, "Ruliad Correctness Constrained Context-Swap Top-1 Change Rate"
        ),
        "correctness_constrained_context_swap_equivalent_probability_drop": metric_last(
            events,
            "Ruliad Correctness Constrained Context-Swap Equivalent Probability Drop",
        ),
        "correctness_constrained_context_swap_js_divergence": metric_last(
            events, "Ruliad Correctness Constrained Context-Swap JS Divergence"
        ),
        "correctness_constrained_counterfactual_target_items": metric_last(
            events, "Ruliad Correctness Constrained Counterfactual-Target Items"
        ),
        "correctness_constrained_counterfactual_target_equivalent_top1": metric_last(
            events,
            "Ruliad Correctness Constrained Counterfactual-Target Equivalent Top-1 Rate",
        ),
        "correctness_constrained_counterfactual_target_equivalent_nll": metric_last(
            events,
            "Ruliad Correctness Constrained Counterfactual-Target Equivalent NLL",
        ),
        "correctness_constrained_counterfactual_target_top1_change": metric_last(
            events,
            "Ruliad Correctness Constrained Counterfactual-Target Top-1 Change Rate",
        ),
        "correctness_constrained_counterfactual_target_equivalent_probability_gain": metric_last(
            events,
            "Ruliad Correctness Constrained Counterfactual-Target Equivalent Probability Gain",
        ),
        "correctness_constrained_counterfactual_target_js_divergence": metric_last(
            events,
            "Ruliad Correctness Constrained Counterfactual-Target JS Divergence",
        ),
        "correctness_constrained_symmetry_balanced": metric_last(
            events, "Ruliad Correctness Constrained Symmetry Balanced"
        ),
        "correctness_constrained_symmetry_orbit_averaged": metric_last(
            events, "Ruliad Correctness Constrained Symmetry Orbit Averaged"
        ),
        "completion_health": finite(latest_correctness.get("completion_health_rate")),
        "actual_answer_distinct_fraction": finite(
            latest_correctness.get("actual_answer_distinct_fraction")
        ),
        "expected_answer_distinct_fraction": finite(
            latest_correctness.get("expected_answer_distinct_fraction")
        ),
        "actual_answer_dominant_fraction": finite(
            latest_correctness.get("actual_answer_dominant_fraction")
        ),
        "schema_wrong": finite(latest_correctness.get("schema_valid_wrong_rate")),
        "malformed": finite(latest_correctness.get("malformed_rate")),
        "correctness_best": max(correctness_clean) if correctness_clean else None,
        "correctness_drop_from_best": (
            max(correctness_clean) - correctness_clean[-1] if correctness_clean else None
        ),
        "policy_solve": finite(latest_policy.get("verifier_rate")),
        "policy_goal_completion": finite(latest_policy.get("partial_credit_rate")),
        "policy_best_solve": max(policy_clean) if policy_clean else None,
        "policy_solve_drop_from_best": (
            max(policy_clean) - policy_clean[-1] if policy_clean else None
        ),
        "policy_valid_action": metric_last(events, "Ruliad Policy Rollout Valid Action Rate"),
        "policy_top1_expert": metric_last(events, "Ruliad Policy Model Top-1 Expert Rate"),
        "policy_runtime_gate_passed": metric_last(
            events, "Ruliad Policy Promotion Gate Passed"
        ),
        "policy_candidate_symmetry_balanced": metric_last(
            events, "Ruliad Policy Candidate Symmetry Balanced"
        ),
        "policy_candidate_symmetry_orbit_averaged": metric_last(
            events, "Ruliad Policy Candidate Symmetry Orbit Averaged"
        ),
        "valid_ce": metric_last(
            events,
            "Teacher Forced Answer CE",
            split="valid",
            running=True,
        )
        or metric_last(events, "Loss", split="valid", running=True),
        "model_tokens_per_second": finite(latent.get("stage_model_tokens_per_sec")),
        "main_model_tokens_per_second": finite(
            latent.get("stage_main_model_tokens_per_sec")
        ),
        "wall_tokens_per_second": finite(latent.get("stage_wall_tokens_per_sec")),
        "model_duty_fraction": finite(latent.get("stage_model_duty_fraction")),
        "auxiliary_objective_fraction": finite(
            latent.get("stage_auxiliary_objective_fraction")
        ),
        "proof_policy_fraction": finite(latent.get("stage_proof_policy_fraction")),
        "train_compute_fraction": finite(latent.get("stage_train_compute_fraction")),
        "optimizer_fraction": finite(latent.get("stage_optimizer_fraction")),
        "metric_sync_fraction": finite(latent.get("stage_metric_sync_fraction")),
        "accounted_fraction": finite(latent.get("stage_accounted_fraction")),
        "validation_fraction": finite(latent.get("stage_validation_fraction")),
        "dataloader_cpu_thread_fraction": finite(
            latent.get("stage_dataloader_cpu_thread_fraction")
        ),
        "dataloader_foreground_wait_fraction": finite(
            latent.get("stage_dataloader_foreground_wait_fraction")
        ),
        "gpu_util_mean": gpu_util,
        "gpu_power_mean": gpu_power,
        "gpu_active_util_mean": gpu_active_util,
        "gpu_active_power_mean": gpu_active_power,
        "gpu_active_sm_clock_mean_mhz": gpu_active_sm_clock_mean,
        "gpu_active_sm_clock_min_mhz": gpu_active_sm_clock_min,
        "gpu_active_memory_util_mean": gpu_active_memory_util_mean,
        "gpu_active_temperature_max_c": gpu_active_temperature_max,
        "gpu_high_util_fraction": gpu_high_util,
        "gpu_low_util_fraction": gpu_low_util,
        "gpu_max_consecutive_sub80_samples": gpu_max_consecutive_sub80,
        "gpu_max_consecutive_idle_samples": gpu_max_consecutive_idle,
        "gpu_active_power_cv": gpu_active_power_cv,
        "elapsed_seconds": finite(manifest.get("elapsed_seconds")),
        "peak_used_mb": finite(manifest.get("peak_used_mb")),
        **dagger,
        "audited_prompt_count": audited_prompts,
        "semantic_prompt_leak_count": semantic_leaks,
    }


def collect_trials(root: Path, arms: tuple[str, ...]) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    for arm in arms:
        arm_dir = root / arm
        analyzed = analysis_rows(arm_dir)
        for manifest in sorted((arm_dir / "manifests").glob("*.json")):
            trials.append(summarize_trial(arm, manifest, analyzed))
    return trials


def arm_summaries(
    trials: list[dict[str, Any]], arms: tuple[str, ...]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in arms:
        arm_trials = [row for row in trials if row["arm"] == arm]
        summary: dict[str, Any] = {
            "arm": arm,
            "trials": len(arm_trials),
            "ok_trials": sum(row["status"] == "ok" for row in arm_trials),
            "seeds": ",".join(str(row["seed"]) for row in arm_trials),
        }
        for metric in METRICS:
            center, ci = mean_ci95(finite(row.get(metric)) for row in arm_trials)
            summary[f"{metric}_mean"] = center
            summary[f"{metric}_ci95"] = ci
        rows.append(summary)
    return rows


def paired_deltas(
    trials: list[dict[str, Any]], baseline_arm: str, candidate_arm: str
) -> list[dict[str, Any]]:
    baseline = {row["seed"]: row for row in trials if row["arm"] == baseline_arm}
    candidate = {row["seed"]: row for row in trials if row["arm"] == candidate_arm}
    rows: list[dict[str, Any]] = []
    for seed in sorted(baseline.keys() & candidate.keys()):
        row: dict[str, Any] = {"seed": seed}
        for metric in PAIRED_METRICS:
            left = finite(baseline[seed].get(metric))
            right = finite(candidate[seed].get(metric))
            row[f"{metric}_baseline"] = left
            row[f"{metric}_candidate"] = right
            row[f"{metric}_delta"] = right - left if left is not None and right is not None else None
        rows.append(row)
    return rows


def promotion_decision(
    trials: list[dict[str, Any]],
    expected_seeds: set[int],
    minimum_iters: int,
    candidate_arm: str = DEFAULT_CANDIDATE_ARM,
    comparison_arm: str = "structural_ce",
) -> dict[str, Any]:
    failures: list[str] = []
    required_arms = tuple(dict.fromkeys((*BASELINE_ARMS, comparison_arm, candidate_arm)))
    by_arm = {arm: [row for row in trials if row["arm"] == arm] for arm in required_arms}
    for arm, rows in by_arm.items():
        seeds = {int(row["seed"]) for row in rows}
        if seeds != expected_seeds:
            failures.append(f"{arm}:seed_set={sorted(seeds)} expected={sorted(expected_seeds)}")
        for row in rows:
            label = f"{arm}/seed{row['seed']}"
            if row.get("status") != "ok":
                failures.append(f"{label}:status={row.get('status')}")
            if int(row.get("max_iters") or 0) < minimum_iters:
                failures.append(f"{label}:immature_updates")
            if row.get("generalization_contract") != row.get("expected_contract"):
                failures.append(f"{label}:corpus_contract_mismatch")
            if row.get("proof_action_answer_contract") != row.get(
                "expected_answer_contract"
            ):
                failures.append(f"{label}:proof_action_answer_contract_mismatch")
            if not row.get("mode_telemetry_present"):
                failures.append(f"{label}:missing_partition_telemetry")
            if arm != "seed_ce" and int(row.get("audited_prompt_count") or 0) <= 0:
                failures.append(f"{label}:missing_prompt_leakage_audit")
            if arm != "seed_ce" and int(row.get("semantic_prompt_leak_count") or 0) > 0:
                failures.append(f"{label}:semantic_prompt_leakage")
            if int(row.get("probe_count") or 0) < 2:
                failures.append(f"{label}:insufficient_temporal_probes")
            constrained_items = finite(row.get("correctness_constrained_items")) or 0.0
            if constrained_items < 64.0:
                failures.append(f"{label}:insufficient_same_item_constrained_probe")
            if finite(row.get("correctness_constrained_symmetry_balanced")) != 1.0:
                failures.append(f"{label}:same_item_candidate_symmetry_contract")
            if finite(row.get("correctness_constrained_symmetry_orbit_averaged")) != 1.0:
                failures.append(f"{label}:same_item_exact_orbit_contract")
            if (
                finite(row.get("correctness_constrained_complete_orbit_items"))
                or 0.0
            ) < constrained_items:
                failures.append(f"{label}:incomplete_same_item_orbit")
            if (finite(row.get("correctness_constrained_presentation_rows")) or 0.0) < (
                2.0 * constrained_items
            ):
                failures.append(f"{label}:insufficient_same_item_presentations")
            if (finite(row.get("correctness_constrained_context_swap_items")) or 0.0) < constrained_items:
                failures.append(f"{label}:insufficient_context_swap_items")
            if (
                finite(row.get("correctness_constrained_counterfactual_target_items"))
                or 0.0
            ) < constrained_items:
                failures.append(f"{label}:insufficient_counterfactual_target_items")
            if finite(row.get("policy_candidate_symmetry_balanced")) != 1.0:
                failures.append(f"{label}:rollout_candidate_symmetry_contract")
            if finite(row.get("policy_candidate_symmetry_orbit_averaged")) != 1.0:
                failures.append(f"{label}:rollout_exact_orbit_contract")
            for diagnostic in (
                "correctness_constrained_canonical_equivalent_top1",
                "correctness_constrained_canonical_equivalent_nll",
                "correctness_constrained_worst_presentation_equivalent_top1",
                "correctness_constrained_worst_presentation_equivalent_nll",
                "correctness_constrained_orbit_js_divergence",
                "correctness_constrained_orbit_top1_consensus",
                "correctness_constrained_presentation_equivalent_top1",
                "correctness_constrained_context_swap_equivalent_top1",
                "correctness_constrained_context_swap_equivalent_nll",
                "correctness_constrained_context_swap_top1_change",
                "correctness_constrained_context_swap_equivalent_probability_drop",
                "correctness_constrained_context_swap_js_divergence",
                "correctness_constrained_counterfactual_target_equivalent_top1",
                "correctness_constrained_counterfactual_target_equivalent_nll",
                "correctness_constrained_counterfactual_target_top1_change",
                "correctness_constrained_counterfactual_target_equivalent_probability_gain",
                "correctness_constrained_counterfactual_target_js_divergence",
            ):
                if finite(row.get(diagnostic)) is None:
                    failures.append(f"{label}:missing_{diagnostic}")
            if (finite(row.get("malformed")) or 0.0) > 0.05:
                failures.append(f"{label}:malformed_rate")
            if (finite(row.get("correctness_drop_from_best")) or 0.0) > 0.125:
                failures.append(f"{label}:correctness_collapse")
            if str(row.get("backend") or "").startswith("cuda"):
                active_util = finite(row.get("gpu_active_util_mean"))
                high_util = finite(row.get("gpu_high_util_fraction"))
                low_util = finite(row.get("gpu_low_util_fraction"))
                sub80_streak = finite(row.get("gpu_max_consecutive_sub80_samples"))
                idle_streak = finite(row.get("gpu_max_consecutive_idle_samples"))
                loader_wait = finite(row.get("dataloader_foreground_wait_fraction"))
                if active_util is None or active_util < 85.0:
                    failures.append(f"{label}:gpu_active_util_below_85")
                if high_util is None or high_util < 0.80:
                    failures.append(f"{label}:gpu_high_util_fraction_below_0.80")
                if low_util is None or low_util > 0.10:
                    failures.append(f"{label}:gpu_low_util_fraction_above_0.10")
                if sub80_streak is None or sub80_streak > 10.0:
                    failures.append(f"{label}:gpu_sub80_streak_above_10_samples")
                if idle_streak is None or idle_streak > 5.0:
                    failures.append(f"{label}:gpu_idle_streak_above_5_samples")
                if loader_wait is None or loader_wait > 0.02:
                    failures.append(f"{label}:loader_foreground_wait_above_0.02")

    candidate_rows = by_arm[candidate_arm]
    semantic_action_contract = (
        ARM_ANSWER_CONTRACTS[candidate_arm] == "semantic_step"
        or ARM_POLICY_SCORING[candidate_arm] == "semantic_energy"
    )
    expected_policy_contract = {
        "structural_energy_static025": ("static_expert", "static_expert"),
        "structural_energy_head_only025": ("static_expert", "static_expert"),
        "structural_energy_head_only_fullrate100": ("static_expert", "static_expert"),
        "structural_energy_fullrate100": ("static_expert", "static_expert"),
        "structural_energy_value_binding025": ("static_expert", "static_expert"),
        "structural_static025": ("static_expert", "static_expert"),
        "structural_semantic_static025": ("static_expert", "static_expert"),
        "structural_semantic_language_head_only025": (
            "static_expert",
            "static_expert",
        ),
        "structural_semantic_static_dense025": ("static_expert", "static_expert"),
        "structural_semantic_static_prefix025": ("static_expert", "static_expert"),
        "structural_semantic_static_marginal025": ("static_expert", "static_expert"),
        "structural_static_marginal025": ("static_expert", "static_expert"),
        "structural_static_orbit_marginal025": ("static_expert", "static_expert"),
        "structural_static_orbit_worst_marginal025": (
            "static_expert",
            "static_expert",
        ),
        "structural_dagger025": ("dagger", "dagger"),
        "structural_dagger_marginal025": ("dagger", "dagger"),
        "structural_bc_paired_dagger_marginal025": (
            "static_then_paired_dagger",
            "paired_dagger,static_expert",
        ),
        "structural_bc_paired_dagger_orbit_marginal025": (
            "static_then_paired_dagger",
            "paired_dagger,static_expert",
        ),
    }.get(candidate_arm)
    for row in candidate_rows:
        label = f"{candidate_arm}/seed{row['seed']}"
        if expected_policy_contract is not None:
            expected_configured_mode, expected_effective_modes = expected_policy_contract
            if int(row.get("dagger_calls") or 0) <= 0 or int(row.get("dagger_expert_rows") or 0) <= 0:
                failures.append(f"{label}:policy_objective_not_exercised")
            if int(row.get("dagger_telemetry_version_min") or 0) < 6:
                failures.append(f"{label}:policy_objective_contract_before_v6")
            expected_objective = (
                "semantic_sequence_energy_counterfactual_v1"
                if candidate_arm in SEMANTIC_ENERGY_ARMS
                else "candidate_normalized_counterfactual_v1"
                if candidate_arm == "structural_semantic_language_head_only025"
                else "prefix_conditional_equivalent_v1"
                if candidate_arm == "structural_semantic_static_prefix025"
                else "vocabulary_marginal_equivalent_v1"
                if candidate_arm in {
                    "structural_static_marginal025",
                    "structural_semantic_static_marginal025",
                    "structural_static_orbit_marginal025",
                    "structural_static_orbit_worst_marginal025",
                    "structural_dagger_marginal025",
                    "structural_bc_paired_dagger_marginal025",
                    "structural_bc_paired_dagger_orbit_marginal025",
                }
                else "candidate_normalized_equivalent_v1"
            )
            if row.get("dagger_objective") != expected_objective:
                failures.append(f"{label}:policy_objective_contract")
            if candidate_arm == "structural_semantic_static_prefix025":
                if int(row.get("dagger_telemetry_version_min") or 0) < 16:
                    failures.append(f"{label}:policy_prefix_contract_before_v16")
                if int(row.get("dagger_prefix_branch_rows") or 0) <= 0:
                    failures.append(f"{label}:policy_prefix_branches_not_exercised")
                if int(row.get("dagger_prefix_candidate_tokens") or 0) <= int(
                    row.get("dagger_prefix_equivalent_tokens") or 0
                ):
                    failures.append(f"{label}:policy_prefix_contrast_missing")
            expected_policy_answer_contract = (
                "semantic_step"
                if candidate_arm in SEMANTIC_ENERGY_ARMS
                else row.get("expected_answer_contract")
            )
            if row.get("dagger_answer_contract") != expected_policy_answer_contract:
                failures.append(f"{label}:policy_answer_contract")
            if (
                candidate_arm in SEMANTIC_ENERGY_ARMS
                and int(row.get("dagger_telemetry_version_min") or 0) < 18
            ):
                failures.append(f"{label}:policy_energy_contract_before_v18")
            if candidate_arm in {
                "structural_energy_head_only025",
                "structural_energy_head_only_fullrate100",
            }:
                if int(row.get("dagger_telemetry_version_min") or 0) < 19:
                    failures.append(f"{label}:head_only_telemetry_version_below_19")
                if row.get("dagger_gradient_scope") != "score_head_only":
                    failures.append(f"{label}:head_only_gradient_scope_not_exercised")
            if candidate_arm == "structural_semantic_language_head_only025":
                if int(row.get("dagger_telemetry_version_min") or 0) < 19:
                    failures.append(
                        f"{label}:language_head_only_telemetry_version_below_19"
                    )
                if row.get("dagger_gradient_scope") != "language_head_only":
                    failures.append(
                        f"{label}:language_head_only_gradient_scope_not_exercised"
                    )
            if candidate_arm in COUNTERFACTUAL_TARGET_ARMS:
                if int(
                    row.get("dagger_configured_counterfactual_targets_per_state") or 0
                ) != 1:
                    failures.append(f"{label}:policy_counterfactual_target_contract")
                if int(row.get("dagger_counterfactual_target_shortfall") or 0) != 0:
                    failures.append(f"{label}:policy_counterfactual_target_shortfall")
                base_rows = int(row.get("dagger_base_semantic_state_rows") or 0)
                counterfactual_rows = int(
                    row.get("dagger_counterfactual_semantic_state_rows") or 0
                )
                if base_rows <= 0 or counterfactual_rows != base_rows:
                    failures.append(f"{label}:policy_counterfactual_target_pairing")
            if row.get("dagger_configured_mode") != expected_configured_mode:
                failures.append(f"{label}:policy_configured_mode_contract")
            if row.get("dagger_mode") != expected_effective_modes:
                failures.append(f"{label}:policy_mode_contract")
            if expected_configured_mode in {"dagger", "static_then_paired_dagger"}:
                if int(row.get("dagger_telemetry_version_min") or 0) < 8:
                    failures.append(f"{label}:policy_dagger_contract_before_v8")
                if int(row.get("dagger_model_visited_expert_rows") or 0) <= 0:
                    failures.append(f"{label}:policy_no_model_visited_supervision")
                if int(row.get("dagger_rollout_depth_reached_max") or 0) < 2:
                    failures.append(f"{label}:policy_rollout_depth_not_exercised")
            if expected_configured_mode == "static_then_paired_dagger" and int(
                row.get("dagger_telemetry_version_min") or 0
            ) < 10:
                failures.append(f"{label}:policy_schedule_contract_before_v10")
            if expected_configured_mode == "static_then_paired_dagger":
                static_rows = int(row.get("dagger_paired_static_expert_rows") or 0)
                on_policy_rows = int(row.get("dagger_paired_on_policy_expert_rows") or 0)
                if static_rows <= 0 or on_policy_rows <= 0:
                    failures.append(f"{label}:policy_paired_population_missing")
                pair_tolerance = max(2, int(0.05 * (static_rows + on_policy_rows)))
                if abs(static_rows - on_policy_rows) > pair_tolerance:
                    failures.append(f"{label}:policy_paired_population_imbalance")
            if candidate_arm == "structural_semantic_static_prefix025" or candidate_arm in COUNTERFACTUAL_TARGET_ARMS or candidate_arm in {
                "structural_static_marginal025",
                "structural_semantic_static_marginal025",
                "structural_dagger_marginal025",
                "structural_bc_paired_dagger_marginal025",
            }:
                if row.get("dagger_candidate_symmetry") != "balanced_rotation":
                    failures.append(f"{label}:policy_candidate_symmetry_contract")
            if candidate_arm in {
                "structural_static_orbit_marginal025",
                "structural_static_orbit_worst_marginal025",
                "structural_bc_paired_dagger_orbit_marginal025",
            }:
                if row.get("dagger_candidate_symmetry") != "cyclic_orbit_average":
                    failures.append(f"{label}:policy_candidate_symmetry_contract")
                if int(row.get("dagger_telemetry_version_min") or 0) < 13:
                    failures.append(f"{label}:policy_orbit_contract_before_v13")
                if (finite(row.get("dagger_presentations_per_state")) or 0.0) < 2.0:
                    failures.append(f"{label}:policy_orbit_not_materialized")
                presentation_budget = int(
                    finite(row.get("dagger_presentation_row_budget_min")) or 0
                )
                presentation_rows_max = int(
                    finite(row.get("dagger_presentation_rows_max")) or 0
                )
                if presentation_budget <= 0:
                    failures.append(f"{label}:policy_presentation_budget_missing")
                elif presentation_rows_max > presentation_budget:
                    failures.append(f"{label}:policy_presentation_budget_exceeded")
                if int(finite(row.get("dagger_semantic_row_budget_min")) or 0) <= 0:
                    failures.append(f"{label}:policy_semantic_budget_missing")
                if finite(row.get("correctness_constrained_symmetry_orbit_averaged")) != 1.0:
                    failures.append(f"{label}:same_item_orbit_average_missing")
                if finite(row.get("policy_candidate_symmetry_orbit_averaged")) != 1.0:
                    failures.append(f"{label}:rollout_orbit_average_missing")
            expected_presentation_risk = (
                "worst"
                if candidate_arm == "structural_static_orbit_worst_marginal025"
                else "mean"
            )
            if row.get("dagger_presentation_risk") != expected_presentation_risk:
                failures.append(f"{label}:policy_presentation_risk_contract")
            if expected_presentation_risk == "worst" and int(
                row.get("dagger_telemetry_version_min") or 0
            ) < 14:
                failures.append(f"{label}:policy_presentation_risk_contract_before_v14")
            expert_entropy = finite(row.get("dagger_expert_index_entropy_bits"))
            if expert_entropy is None or expert_entropy < 1.8:
                failures.append(f"{label}:policy_expert_index_entropy")
            expert_dominant = finite(row.get("dagger_expert_index_dominant_fraction"))
            if expert_dominant is None or expert_dominant > 0.40:
                failures.append(f"{label}:policy_expert_index_dominance")
        valid_action = finite(row.get("policy_valid_action"))
        if valid_action is None or valid_action < 0.95:
            failures.append(f"{label}:valid_action_floor")
        if finite(row.get("policy_runtime_gate_passed")) != 1.0:
            failures.append(f"{label}:typed_policy_runtime_gate_failed")
        counterfactual_top1 = finite(
            row.get("correctness_constrained_counterfactual_target_equivalent_top1")
        )
        counterfactual_change = finite(
            row.get("correctness_constrained_counterfactual_target_top1_change")
        )
        counterfactual_probability_gain = finite(
            row.get(
                "correctness_constrained_counterfactual_target_equivalent_probability_gain"
            )
        )
        counterfactual_js = finite(
            row.get("correctness_constrained_counterfactual_target_js_divergence")
        )
        if counterfactual_top1 is None or counterfactual_top1 < 0.60:
            failures.append(f"{label}:counterfactual_target_top1_below_0.60")
        if counterfactual_change is None or counterfactual_change < 0.50:
            failures.append(f"{label}:counterfactual_target_preference_change_below_0.50")
        if counterfactual_probability_gain is None or counterfactual_probability_gain < 0.05:
            failures.append(f"{label}:counterfactual_target_probability_gain_below_0.05")
        if counterfactual_js is None or counterfactual_js < 0.005:
            failures.append(f"{label}:counterfactual_target_js_below_0.005")
        if semantic_action_contract:
            # Four verifier-enumerated candidates imply 25% chance. A typed semantic policy must
            # reach at least three times chance on the orbit, canonical presentation, and every
            # presentation, while retaining high cross-presentation consensus. This replaces the
            # positional action-index gain gate; it does not relax free-generation requirements.
            for metric, minimum in (
                ("correctness_constrained_equivalent_top1", 0.75),
                ("correctness_constrained_canonical_equivalent_top1", 0.75),
                (
                    "correctness_constrained_worst_presentation_equivalent_top1",
                    0.75,
                ),
                ("correctness_constrained_orbit_top1_consensus", 0.90),
            ):
                value = finite(row.get(metric))
                if value is None or value < minimum:
                    failures.append(f"{label}:typed_policy_{metric}_floor")
        if (finite(row.get("policy_solve_drop_from_best")) or 0.0) > 0.25:
            failures.append(f"{label}:policy_solve_collapse")
        actual_distinct = finite(row.get("actual_answer_distinct_fraction"))
        expected_distinct = finite(row.get("expected_answer_distinct_fraction"))
        if (
            actual_distinct is None
            or expected_distinct is None
            or actual_distinct < expected_distinct * 0.75
        ):
            failures.append(f"{label}:free_action_answer_coverage")
        if (finite(row.get("actual_answer_dominant_fraction")) or 0.0) > 0.80:
            failures.append(f"{label}:free_action_answer_dominance")

    paired = paired_deltas(trials, comparison_arm, candidate_arm)
    if len(paired) != len(expected_seeds):
        failures.append("candidate:incomplete_matched_pairs")
    if paired:
        def delta_mean(metric: str) -> float | None:
            return mean_ci95(finite(row.get(f"{metric}_delta")) for row in paired)[0]

        solve_delta = delta_mean("policy_solve")
        goal_delta = delta_mean("policy_goal_completion")
        top1_delta = delta_mean("policy_top1_expert")
        same_item_top1_delta = delta_mean("correctness_constrained_equivalent_top1")
        same_item_nll_delta = delta_mean("correctness_constrained_equivalent_nll")
        canonical_top1_delta = delta_mean(
            "correctness_constrained_canonical_equivalent_top1"
        )
        canonical_nll_delta = delta_mean(
            "correctness_constrained_canonical_equivalent_nll"
        )
        worst_top1_delta = delta_mean(
            "correctness_constrained_worst_presentation_equivalent_top1"
        )
        worst_nll_delta = delta_mean(
            "correctness_constrained_worst_presentation_equivalent_nll"
        )
        orbit_js_delta = delta_mean("correctness_constrained_orbit_js_divergence")
        orbit_consensus_delta = delta_mean(
            "correctness_constrained_orbit_top1_consensus"
        )
        presentation_top1_delta = delta_mean(
            "correctness_constrained_presentation_equivalent_top1"
        )
        verifier_delta = delta_mean("correctness_verifier")
        if solve_delta is None or solve_delta < -0.03125:
            failures.append("candidate:policy_solve_noninferiority")
        if goal_delta is None or goal_delta < 0.0:
            failures.append("candidate:goal_completion_regression")
        if top1_delta is None or top1_delta < 0.05:
            failures.append("candidate:top1_gain_below_0.05")
        if semantic_action_contract:
            if same_item_top1_delta is None or same_item_top1_delta < -0.03125:
                failures.append("candidate:same_item_top1_noninferiority")
        elif same_item_top1_delta is None or same_item_top1_delta < 0.03:
            failures.append("candidate:same_item_top1_gain_below_0.03")
        if same_item_nll_delta is None or same_item_nll_delta > 0.0:
            failures.append("candidate:same_item_equivalent_nll_regression")
        if canonical_top1_delta is None or canonical_top1_delta < 0.0:
            failures.append("candidate:canonical_top1_regression")
        if canonical_nll_delta is None or canonical_nll_delta > 0.0:
            failures.append("candidate:canonical_equivalent_nll_regression")
        if worst_top1_delta is None or worst_top1_delta < 0.0:
            failures.append("candidate:worst_presentation_top1_regression")
        if worst_nll_delta is None or worst_nll_delta > 0.0:
            failures.append("candidate:worst_presentation_nll_regression")
        if orbit_js_delta is None or orbit_js_delta > 0.0:
            failures.append("candidate:presentation_js_divergence_regression")
        if orbit_consensus_delta is None or orbit_consensus_delta < 0.0:
            failures.append("candidate:presentation_top1_consensus_regression")
        if presentation_top1_delta is None or presentation_top1_delta < 0.0:
            failures.append("candidate:presentation_equivalent_top1_regression")
        if verifier_delta is None or verifier_delta < -0.03125:
            failures.append("candidate:correctness_noninferiority")
        top1_positive = sum((finite(row.get("policy_top1_expert_delta")) or 0.0) > 0.0 for row in paired)
        same_item_top1_positive = sum(
            (finite(row.get("correctness_constrained_equivalent_top1_delta")) or 0.0) > 0.0
            for row in paired
        )
        goal_positive = sum((finite(row.get("policy_goal_completion_delta")) or 0.0) > 0.0 for row in paired)
        if top1_positive < 2:
            failures.append("candidate:top1_gain_not_seed_robust")
        if not semantic_action_contract and same_item_top1_positive < 2:
            failures.append("candidate:same_item_top1_gain_not_seed_robust")
        if goal_positive < 2:
            failures.append("candidate:goal_gain_not_seed_robust")
        baseline_speed = mean_ci95(
            finite(row.get("model_tokens_per_second"))
            for row in by_arm[comparison_arm]
        )[0]
        candidate_speed = mean_ci95(
            finite(row.get("model_tokens_per_second")) for row in candidate_rows
        )[0]
        if baseline_speed and candidate_speed and candidate_speed / baseline_speed < 0.65:
            failures.append("candidate:model_throughput_ratio_below_0.65")

    structural_verifier = mean_ci95(
        finite(row.get("correctness_verifier")) for row in candidate_rows
    )[0]
    if structural_verifier is None or structural_verifier <= 0.30:
        failures.append("candidate:structural_verifier_not_above_chance_guard")

    typed_only_markers = (
        ":policy_",
        ":typed_policy_",
        ":rollout_",
        ":valid_action_",
        ":same_item_",
        ":incomplete_same_item_",
        ":insufficient_same_item_",
        ":insufficient_context_swap_",
        ":context_swap_",
        ":insufficient_counterfactual_target_",
        ":counterfactual_target_",
        "candidate:policy_",
        "candidate:goal_",
        "candidate:top1_",
        "candidate:same_item_",
        "candidate:canonical_",
        "candidate:worst_presentation_",
        "candidate:presentation_",
    )
    free_only_markers = (
        ":free_action_",
        ":malformed_rate",
        ":correctness_collapse",
        "candidate:correctness_noninferiority",
        "candidate:structural_verifier_not_above_chance_guard",
    )
    typed_policy_failures = [
        failure
        for failure in failures
        if not any(marker in failure for marker in free_only_markers)
    ]
    free_generation_failures = [
        failure
        for failure in failures
        if not any(marker in failure for marker in typed_only_markers)
    ]

    return {
        "evidence_class": "matched_multiseed" if len(expected_seeds) >= 3 and minimum_iters >= 1024 else "smoke",
        "typed_policy_promotion_passed": not typed_policy_failures,
        "typed_policy_failures": typed_policy_failures,
        "free_generation_promotion_passed": not free_generation_failures,
        "free_generation_failures": free_generation_failures,
        "directional_promotion_passed": not failures,
        "failures": failures,
        "paired_seed_count": len(paired),
        "candidate_arm": candidate_arm,
        "comparison_arm": comparison_arm,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    trials: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    decision: dict[str, Any],
    arms: tuple[str, ...],
    candidate_arm: str,
    comparison_arm: str,
) -> None:
    summary_by_arm = {row["arm"]: row for row in summaries}
    with path.open("w") as handle:
        handle.write("# Ruliad Structural-Generalization Matrix\n\n")
        handle.write(
            "This report separates the old seed-only control from the alpha-renamed, "
            "law-held-out, topology-held-out validation contract. No composite score is used.\n\n"
        )
        handle.write("| arm | ok | valid CE | verifier | partial | same-item top-1 | action distinct | action dominant | policy solve | goal completion | rollout top-1 | valid action | wall tok/s | objective tok/s | main model tok/s | model duty | aux stage | proof policy | validation wall | loader CPU | loader wait | GPU util | active util | active power W | SM MHz | min SM MHz | max C | power CV | >=80% samples | <=20% samples | longest <80% | longest <=20% | wall s |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for arm in arms:
            row = summary_by_arm[arm]
            cell = lambda metric: f"{fmt(row.get(metric + '_mean'))} +/- {fmt(row.get(metric + '_ci95'))}"
            handle.write(
                f"| {arm} | {row['ok_trials']}/{row['trials']} | {cell('valid_ce')} | "
                f"{cell('correctness_verifier')} | {cell('correctness_partial')} | "
                f"{cell('correctness_constrained_equivalent_top1')} | "
                f"{cell('actual_answer_distinct_fraction')} | {cell('actual_answer_dominant_fraction')} | "
                f"{cell('policy_solve')} | {cell('policy_goal_completion')} | "
                f"{cell('policy_top1_expert')} | {cell('policy_valid_action')} | "
                f"{cell('wall_tokens_per_second')} | {cell('model_tokens_per_second')} | "
                f"{cell('main_model_tokens_per_second')} | {cell('model_duty_fraction')} | "
                f"{cell('auxiliary_objective_fraction')} | {cell('proof_policy_fraction')} | {cell('validation_fraction')} | "
                f"{cell('dataloader_cpu_thread_fraction')} | {cell('dataloader_foreground_wait_fraction')} | {cell('gpu_util_mean')} | "
                f"{cell('gpu_active_util_mean')} | {cell('gpu_active_power_mean')} | "
                f"{cell('gpu_active_sm_clock_mean_mhz')} | {cell('gpu_active_sm_clock_min_mhz')} | "
                f"{cell('gpu_active_temperature_max_c')} | {cell('gpu_active_power_cv')} | "
                f"{cell('gpu_high_util_fraction')} | {cell('gpu_low_util_fraction')} | "
                f"{cell('gpu_max_consecutive_sub80_samples')} | {cell('gpu_max_consecutive_idle_samples')} | "
                f"{cell('elapsed_seconds')} |\n"
            )

        handle.write(
            "\n`objective tok/s` and `model duty` include the main language forward, all auxiliary-"
            "objective construction/forwards, and the combined backward. `main model tok/s` "
            "excludes auxiliary-objective forwards and is diagnostic only. GPU duty is represented "
            "by the sampled utilization columns.\n"
        )
        handle.write(
            "`longest <80%` and `longest <=20%` are consecutive one-second samples inside the "
            "active window from the first to last >=80% sample, excluding startup and teardown.\n"
        )

        handle.write("\n## Candidate-Presentation Robustness\n\n")
        handle.write(
            "Rows use each arm's configured policy scorer, shown explicitly below, over the same "
            "exact cyclic orbit. Canonical is rotation zero; "
            "worst requires robustness to every presentation; JS measures disagreement "
            "that the orbit average would otherwise hide.\n\n"
        )
        handle.write(
            "| arm | scorer | orbit top-1 | presentation top-1 | canonical top-1 | worst top-1 | orbit NLL | "
            "canonical NLL | worst NLL | orbit JS | top-1 consensus | complete items | "
            "presentation rows |\n"
        )
        handle.write(
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        )
        for arm in arms:
            row = summary_by_arm[arm]
            cell = lambda metric: fmt(row.get(metric + "_mean"))
            handle.write(
                f"| {arm} | {ARM_POLICY_SCORING[arm]} | {cell('correctness_constrained_equivalent_top1')} | "
                f"{cell('correctness_constrained_presentation_equivalent_top1')} | "
                f"{cell('correctness_constrained_canonical_equivalent_top1')} | "
                f"{cell('correctness_constrained_worst_presentation_equivalent_top1')} | "
                f"{cell('correctness_constrained_equivalent_nll')} | "
                f"{cell('correctness_constrained_canonical_equivalent_nll')} | "
                f"{cell('correctness_constrained_worst_presentation_equivalent_nll')} | "
                f"{cell('correctness_constrained_orbit_js_divergence')} | "
                f"{cell('correctness_constrained_orbit_top1_consensus')} | "
                f"{cell('correctness_constrained_complete_orbit_items')} | "
                f"{cell('correctness_constrained_presentation_rows')} |\n"
            )

        handle.write("\n## Context Dependence Control\n\n")
        handle.write(
            "The context-swap control holds the formal laws and candidate action menu fixed while "
            "replacing the current and target proof states with another held-out item. Because the "
            "borrowed state is not guaranteed to make the retained action menu applicable, this is "
            "an out-of-contract stress diagnostic, not a promotion gate. The verifier-valid exact "
            "counterfactual-target intervention below is the causal target-dependence gate.\n\n"
        )
        handle.write(
            "| arm | original top-1 | swapped top-1 | top-1 drop | probability drop | "
            "preference change | JS divergence | items |\n"
        )
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for arm in arms:
            row = summary_by_arm[arm]
            original = finite(
                row.get("correctness_constrained_equivalent_top1_mean")
            )
            swapped = finite(
                row.get("correctness_constrained_context_swap_equivalent_top1_mean")
            )
            top1_drop = (
                original - swapped
                if original is not None and swapped is not None
                else None
            )
            handle.write(
                f"| {arm} | {fmt(original)} | {fmt(swapped)} | {fmt(top1_drop)} | "
                f"{fmt(row.get('correctness_constrained_context_swap_equivalent_probability_drop_mean'))} | "
                f"{fmt(row.get('correctness_constrained_context_swap_top1_change_mean'))} | "
                f"{fmt(row.get('correctness_constrained_context_swap_js_divergence_mean'))} | "
                f"{fmt(row.get('correctness_constrained_context_swap_items_mean'))} |\n"
            )

        handle.write("\n## Exact Counterfactual Targets\n\n")
        handle.write(
            "This intervention preserves the formal laws, current proof state, and candidate "
            "actions while replacing only the goal with one verifier-valid candidate outcome. "
            "The correct equivalence class is recomputed exactly; success therefore requires the "
            "policy preference to follow the target rather than candidate surface form.\n\n"
        )
        handle.write(
            "| arm | alternate-target top-1 | alternate-target NLL | probability gain | "
            "preference change | JS divergence | items |\n"
        )
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for arm in arms:
            row = summary_by_arm[arm]
            handle.write(
                f"| {arm} | "
                f"{fmt(row.get('correctness_constrained_counterfactual_target_equivalent_top1_mean'))} | "
                f"{fmt(row.get('correctness_constrained_counterfactual_target_equivalent_nll_mean'))} | "
                f"{fmt(row.get('correctness_constrained_counterfactual_target_equivalent_probability_gain_mean'))} | "
                f"{fmt(row.get('correctness_constrained_counterfactual_target_top1_change_mean'))} | "
                f"{fmt(row.get('correctness_constrained_counterfactual_target_js_divergence_mean'))} | "
                f"{fmt(row.get('correctness_constrained_counterfactual_target_items_mean'))} |\n"
            )

        handle.write(f"\n## Matched {candidate_arm} Deltas\n\n")
        handle.write(
            f"Compared with `{comparison_arm}`. Positive means {candidate_arm} is higher; "
            "negative valid CE and wall time are favorable.\n\n"
        )
        handle.write("| metric | mean delta | 95% CI half-width | positive seeds |\n")
        handle.write("| --- | ---: | ---: | ---: |\n")
        for metric in PAIRED_METRICS:
            values = [finite(row.get(f"{metric}_delta")) for row in paired]
            center, ci = mean_ci95(values)
            positive = sum(value is not None and value > 0.0 for value in values)
            handle.write(f"| {metric} | {fmt(center)} | {fmt(ci)} | {positive}/{len(paired)} |\n")

        handle.write("\n## Contract Gates\n\n")
        handle.write(f"- evidence class: `{decision['evidence_class']}`\n")
        handle.write(
            "- typed verifier-guided policy promotion: "
            f"`{str(decision['typed_policy_promotion_passed']).lower()}`\n"
        )
        handle.write(
            "- unconstrained free-generation promotion: "
            f"`{str(decision['free_generation_promotion_passed']).lower()}`\n"
        )
        handle.write(
            "- combined directional promotion: "
            f"`{str(decision['directional_promotion_passed']).lower()}`\n"
        )
        handle.write(
            "The combined gate remains strict: a typed-policy pass does not waive free text "
            "coverage, correctness, or malformed-output requirements.\n"
        )
        if decision["failures"]:
            for failure in decision["failures"]:
                handle.write(f"- failure: `{failure}`\n")
        else:
            handle.write("- all structural, temporal, validity, non-inferiority, and throughput gates passed\n")

        seed_rows = {row["seed"]: row for row in trials if row["arm"] == "seed_ce"}
        structural_rows = {row["seed"]: row for row in trials if row["arm"] == "structural_ce"}
        gaps = []
        for seed in sorted(seed_rows.keys() & structural_rows.keys()):
            weak = finite(seed_rows[seed].get("correctness_verifier"))
            held = finite(structural_rows[seed].get("correctness_verifier"))
            if weak is not None and held is not None:
                gaps.append(held - weak)
        gap_mean, gap_ci = mean_ci95(gaps)
        handle.write("\n## Holdout Diagnostic\n\n")
        handle.write(
            f"Matched structural-minus-seed correctness verifier delta: {fmt(gap_mean)} "
            f"+/- {fmt(gap_ci)} (95% CI half-width). A negative value quantifies the difficulty "
            "hidden by seed-only validation.\n"
        )


def main() -> None:
    args = parse_args()
    root = Path(args.input).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_seeds = {int(seed.strip()) for seed in args.expected_seeds.split(",") if seed.strip()}
    arms = tuple(dict.fromkeys((*BASELINE_ARMS, args.comparison_arm, args.candidate_arm)))
    trials = collect_trials(root, arms)
    if not trials:
        raise SystemExit(f"no trial manifests found under {root}")
    summaries = arm_summaries(trials, arms)
    paired = paired_deltas(trials, args.comparison_arm, args.candidate_arm)
    decision = promotion_decision(
        trials,
        expected_seeds,
        args.minimum_promotion_iters,
        args.candidate_arm,
        args.comparison_arm,
    )
    write_csv(out_dir / "trials.csv", trials)
    write_csv(out_dir / "arms.csv", summaries)
    write_csv(out_dir / "paired_deltas.csv", paired)
    with (out_dir / "promotion_gate.json").open("w") as handle:
        json.dump(decision, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_markdown(
        out_dir / "report.md",
        trials,
        summaries,
        paired,
        decision,
        arms,
        args.candidate_arm,
        args.comparison_arm,
    )
    print(out_dir / "report.md")
    print(out_dir / "promotion_gate.json")


if __name__ == "__main__":
    main()
