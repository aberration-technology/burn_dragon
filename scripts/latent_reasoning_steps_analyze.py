#!/usr/bin/env python3
"""Summarize latent reasoning max_steps ablation runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


SUMMARY_COLUMNS = [
    "trial_key",
    "max_steps",
    "seed",
    "status",
    "elapsed_seconds",
    "tokens_per_sec",
    "examples_per_sec",
    "stage_wall_tokens_per_sec",
    "stage_model_tokens_per_sec",
    "stage_forward_ns",
    "stage_backward_ns",
    "stage_validation_ns",
    "stage_generation_ns",
    "stage_event_sink_ns",
    "gpu_util_mean",
    "gpu_util_p50",
    "gpu_power_mean",
    "peak_used_mb",
    "train_first",
    "train_last",
    "train_delta",
    "valid_teacher_ce_last",
    "latent_loss_calls_last",
    "latent_jepa_components_last",
    "latent_nextlat_components_last",
    "latent_energy_model_components_last",
    "latent_step_contract_components_last",
    "configured_steps_last",
    "latent_eval_final_ce_delta_last",
    "latent_eval_final_ce_violation_last",
    "latent_eval_final_entropy_bits_last",
    "latent_eval_final_delta_rms_last",
    "latent_eval_best_energy_step_last",
    "latent_eval_final_energy_mean_last",
    "latent_eval_final_energy_delta_last",
    "latent_eval_final_energy_violation_last",
    "latent_extra_eval_max_ce_delta_last",
    "latent_extra_eval_max_ce_violation_last",
    "latent_extra_eval_min_entropy_bits_last",
    "latent_extra_eval_max_delta_rms_last",
    "source_loss_last",
    "source_entropy_bits_last",
    "source_active_candidate_count_last",
    "source_active_max_entropy_bits_last",
    "source_normalized_entropy_last",
    "source_mean_difficulty_last",
    "source_active_max_difficulty_last",
    "source_mastered_max_difficulty_last",
    "source_max_difficulty_level_last",
    "source_materialized_frontier_edge_last",
    "source_max_difficulty_probability_last",
    "source_normalized_difficulty_score_last",
    "source_target_difficulty_score_last",
    "source_hash_noise_probability_last",
    "source_capability_lagging_probability_last",
    "ruliad_verifier_last",
    "ruliad_semantic_last",
    "ruliad_partial_last",
    "ruliad_schema_wrong_last",
    "ruliad_malformed_last",
    "ruliad_missing_last",
    "ruliad_answer_field_accuracy_last",
    "ruliad_answer_field_coverage_last",
    "ruliad_answer_termination_rate_last",
    "ruliad_completion_quality_last",
    "ruliad_expected_answer_distinct_last",
    "ruliad_answer_distinct_last",
    "ruliad_mean_completion_tokens_last",
    "completion_health_last",
    "completion_repetition_last",
    "completion_distinct_1_last",
    "completion_distinct_2_last",
    "completion_period_2_to_16_last",
    "completion_period_2_to_64_last",
    "completion_dominant_period_2_to_64_last",
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
    "capability_gate_passed_last",
    "capability_gate_failure_count_last",
    "capability_completion_health_metric_last",
    "capability_first_pass_epoch",
    "capability_first_pass_step",
    "capability_best_epoch",
    "capability_best_score",
    "capability_score_auc",
    "capability_verifier_auc",
    "capability_completion_auc",
    "capability_schema_wrong_auc",
    "capability_malformed_auc",
    "capability_score_first",
    "capability_score_last",
    "capability_score_best",
    "capability_score_drop_from_best",
    "capability_verifier_best",
    "capability_verifier_drop_from_best",
    "capability_completion_best",
    "capability_completion_drop_from_best",
    "capability_bucket_mastered_count",
    "capability_bucket_lagging_count",
    "capability_contract_mastered_count",
    "capability_contract_lagging_count",
    "dynamics_control_count",
    "stable_control_count",
    "recovery_control_count",
    "recovery_control_fraction",
    "rollback_control_count",
    "hard_recovery_control_count",
    "source_capability_recovery_control_count",
    "source_capability_recovery_control_fraction",
    "capability_quality_recovery_count",
    "capability_gate_failed_count",
    "eval_step_sweep",
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
    "gate_count",
    "fatal_gate_count",
    "healthy",
    "rank_score",
    "run_dir",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Output directory from latent_reasoning_steps_ablation.sh")
    parser.add_argument("--out-dir", default=None, help="Analysis output directory")
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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    clean = sorted(values)
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return clean[lo]
    frac = pos - lo
    return clean[lo] * (1.0 - frac) + clean[hi] * frac


def gpu_stats(path: Path | None) -> tuple[float | None, float | None, float | None]:
    if path is None or not path.exists():
        return None, None, None
    util: list[float] = []
    power: list[float] = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            util_value = finite(row.get(" utilization.gpu [%]") or row.get("utilization.gpu [%]"))
            power_value = finite(row.get(" power.draw [W]") or row.get("power.draw [W]"))
            if util_value is not None:
                util.append(util_value)
            if power_value is not None:
                power.append(power_value)
    util_mean = sum(util) / len(util) if util else None
    power_mean = sum(power) / len(power) if power else None
    return util_mean, percentile(util, 0.5), power_mean


def parse_stage_profile(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    profile: dict[str, float] = {}
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except OSError:
        return profile
    for line in lines:
        if "[stage-profile][training]" not in line:
            continue
        for field in line.split():
            if "=" not in field:
                continue
            key, value = field.split("=", 1)
            parsed = finite(value)
            if parsed is not None:
                profile[key] = parsed
    return profile


def metric_values(events: list[dict[str, Any]], split: str, name: str) -> list[float]:
    values: list[float] = []
    for event in events:
        if event.get("type") != "metric":
            continue
        if event.get("split") != split or event.get("name") != name:
            continue
        value = finite(event.get("value"))
        if value is not None:
            values.append(value)
    return values


def last_metric(events: list[dict[str, Any]], split: str, name: str) -> float | None:
    values = metric_values(events, split, name)
    return values[-1] if values else None


def latent_energy_eval_metrics(events: list[dict[str, Any]], max_steps: Any) -> dict[str, Any]:
    steps = finite(max_steps)
    if steps is None:
        return {}
    step_count = int(steps)
    prefix = f"Latent Eval Steps {step_count}"
    return {
        "latent_eval_final_ce_delta_last": last_metric(
            events, "valid", f"{prefix} Step {step_count} CE Delta"
        ),
        "latent_eval_final_ce_violation_last": last_metric(
            events, "valid", f"{prefix} Step {step_count} CE Monotonic Violation Rate"
        ),
        "latent_eval_final_entropy_bits_last": last_metric(
            events, "valid", f"{prefix} Step {step_count} Entropy Bits"
        ),
        "latent_eval_final_delta_rms_last": last_metric(
            events, "valid", f"{prefix} Step {step_count} Delta RMS"
        ),
        "latent_eval_best_energy_step_last": last_metric(
            events, "valid", f"{prefix} Best Energy Step"
        ),
        "latent_eval_final_energy_mean_last": last_metric(
            events, "valid", f"{prefix} Step {step_count} Energy Mean"
        ),
        "latent_eval_final_energy_delta_last": last_metric(
            events, "valid", f"{prefix} Step {step_count} Energy Delta"
        ),
        "latent_eval_final_energy_violation_last": last_metric(
            events, "valid", f"{prefix} Step {step_count} Energy Monotonic Violation Rate"
        ),
    }


def latent_extra_eval_trajectory_metrics(
    events: list[dict[str, Any]], max_steps: Any
) -> dict[str, Any]:
    trained_steps = finite(max_steps)
    if trained_steps is None:
        return {}
    trained_step_count = int(trained_steps)
    pattern = re.compile(
        r"^Latent Eval Steps (?P<total>\d+) Step (?P<step>\d+) "
        r"(?P<metric>CE Delta|CE Monotonic Violation Rate|Entropy Bits|Delta RMS)$"
    )
    latest: dict[tuple[int, int, str], float] = {}
    for event in events:
        if event.get("split") != "valid":
            continue
        name = str(event.get("name") or "")
        match = pattern.match(name)
        if match is None:
            continue
        total_steps = int(match.group("total"))
        step = int(match.group("step"))
        metric = match.group("metric")
        if total_steps <= trained_step_count or step <= trained_step_count:
            continue
        value = finite(event.get("value"))
        if value is not None:
            latest[(total_steps, step, metric)] = value

    def values(metric: str) -> list[float]:
        return [
            value
            for key, value in latest.items()
            if key[2] == metric
        ]

    ce_deltas = values("CE Delta")
    ce_violations = values("CE Monotonic Violation Rate")
    entropies = values("Entropy Bits")
    delta_rms = values("Delta RMS")
    return {
        "latent_extra_eval_max_ce_delta_last": max(ce_deltas) if ce_deltas else None,
        "latent_extra_eval_max_ce_violation_last": (
            max(ce_violations) if ce_violations else None
        ),
        "latent_extra_eval_min_entropy_bits_last": min(entropies) if entropies else None,
        "latent_extra_eval_max_delta_rms_last": max(delta_rms) if delta_rms else None,
    }


def count_gates(events: list[dict[str, Any]]) -> tuple[int, int]:
    gate_count = 0
    fatal_count = 0
    for event in events:
        if event.get("type") not in {"gate", "training_gate"}:
            continue
        gate_count += 1
        severity = str(event.get("severity") or event.get("level") or "").lower()
        if severity in {"fatal", "error", "hard"}:
            fatal_count += 1
    return gate_count, fatal_count


def dynamics_and_gate_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    recovery_modes = {
        "plasticity_recovery",
        "validation_recovery",
        "rollback_recovery",
        "hard_recovery",
        "hard_collapse",
    }
    control_events = [
        event for event in events if event.get("type") == "dynamics_control"
    ]
    controls = [str(event.get("mode") or "").lower() for event in control_events]
    recovery_count = sum(1 for mode in controls if mode in recovery_modes)
    control_count = len(controls)
    source_capability_recovery_count = sum(
        1
        for event, mode in zip(control_events, controls)
        if mode == "source_capability_recovery"
        or (
            "source-selection capability recovery"
            in str(event.get("reason") or "").lower()
        )
    )
    gate_names = [
        str(event.get("gate") or "")
        for event in events
        if event.get("type") in {"gate", "training_gate"}
    ]
    return {
        "dynamics_control_count": control_count,
        "stable_control_count": sum(1 for mode in controls if mode == "stable"),
        "recovery_control_count": recovery_count,
        "recovery_control_fraction": (
            recovery_count / control_count if control_count > 0 else 0.0
        ),
        "rollback_control_count": sum(1 for mode in controls if mode == "rollback_recovery"),
        "hard_recovery_control_count": sum(
            1 for mode in controls if mode in {"hard_recovery", "hard_collapse"}
        ),
        "source_capability_recovery_control_count": source_capability_recovery_count,
        "source_capability_recovery_control_fraction": (
            source_capability_recovery_count / control_count if control_count > 0 else 0.0
        ),
        "capability_quality_recovery_count": gate_names.count(
            "continual_learning_capability_quality_recovery"
        ),
        "capability_gate_failed_count": gate_names.count("ruliad_capability_gate_failed"),
    }


def last_source_selection(run_dir: Path) -> dict[str, Any] | None:
    source_events = read_jsonl(run_dir / "events" / "source_selection.jsonl")
    if source_events:
        return source_events[-1]
    training_events = read_jsonl(run_dir / "events" / "training_events.jsonl")
    for event in reversed(training_events):
        if event.get("type") == "source_selection":
            return event
    return None


def difficulty_level_from_bucket_label(label: Any) -> int | None:
    text = str(label or "")
    if not text.startswith("d"):
        return None
    try:
        return int(text[1:])
    except ValueError:
        return None


def source_difficulty_summary(source: dict[str, Any]) -> dict[str, Any]:
    active: list[int] = []
    mastered: list[int] = []
    for bucket in source.get("difficulty_buckets") or []:
        if not isinstance(bucket, dict):
            continue
        level = difficulty_level_from_bucket_label(bucket.get("label"))
        if level is None:
            continue
        probability = finite(bucket.get("probability")) or 0.0
        mastered_probability = finite(bucket.get("mastered_probability")) or 0.0
        if probability > 1e-6:
            active.append(level)
        if mastered_probability > 1e-6:
            mastered.append(level)
    return {
        "source_active_max_difficulty_last": max(active) if active else None,
        "source_mastered_max_difficulty_last": max(mastered) if mastered else None,
    }


def capability_probes(run_dir: Path) -> list[dict[str, Any]]:
    return read_jsonl(run_dir / "events" / "capability_probe.jsonl")


def last_capability_probe(run_dir: Path, probe_name: str = "ruliad_correctness") -> dict[str, Any] | None:
    rows = capability_probes(run_dir)
    named = [row for row in rows if row.get("probe_name") == probe_name]
    if named:
        return named[-1]
    non_eval = [
        row
        for row in rows
        if not str(row.get("probe_name") or "").startswith("ruliad_correctness_eval_steps_")
    ]
    return non_eval[-1] if non_eval else (rows[-1] if rows else None)


def eval_step_probe_summary(
    run_dir: Path, base: dict[str, Any], trained_max_steps: Any
) -> dict[str, Any]:
    latest_by_step: dict[int, dict[str, Any]] = {}
    for row in capability_probes(run_dir):
        name = str(row.get("probe_name") or "")
        prefix = "ruliad_correctness_eval_steps_"
        if not name.startswith(prefix):
            continue
        try:
            steps = int(name[len(prefix):])
        except ValueError:
            continue
        latest_by_step[steps] = row
    if not latest_by_step:
        return {}

    def key(item: tuple[int, dict[str, Any]]) -> tuple[float, float, float, float, float, float, int]:
        steps, row = item
        return (
            finite(row.get("verifier_rate")) or 0.0,
            finite(row.get("semantic_rate")) or 0.0,
            finite(row.get("partial_credit_rate")) or 0.0,
            finite(row.get("completion_health_rate")) or 0.0,
            -(finite(row.get("schema_valid_wrong_rate")) or 1.0),
            -(finite(row.get("malformed_rate")) or 1.0),
            -steps,
        )

    best_steps, best = max(latest_by_step.items(), key=key)
    base_verifier = finite(base.get("verifier_rate")) or 0.0
    base_schema = finite(base.get("schema_valid_wrong_rate")) or 0.0
    base_completion = finite(base.get("completion_health_rate")) or 0.0
    best_verifier = finite(best.get("verifier_rate")) or 0.0
    best_schema = finite(best.get("schema_valid_wrong_rate")) or 0.0
    best_completion = finite(best.get("completion_health_rate")) or 0.0
    summary = {
        "best_eval_steps": best_steps,
        "best_eval_verifier": best_verifier,
        "best_eval_semantic": best.get("semantic_rate"),
        "best_eval_partial": best.get("partial_credit_rate"),
        "best_eval_schema_wrong": best_schema,
        "best_eval_completion": best_completion,
        "best_eval_verifier_delta": best_verifier - base_verifier,
        "best_eval_schema_delta": best_schema - base_schema,
        "best_eval_completion_delta": best_completion - base_completion,
    }
    trained_steps = finite(trained_max_steps)
    if trained_steps is None:
        return summary
    extra = {
        steps: row
        for steps, row in latest_by_step.items()
        if steps > int(trained_steps)
    }
    if not extra:
        return summary
    worst_steps, _worst = min(extra.items(), key=key)
    base_malformed = finite(base.get("malformed_rate")) or 0.0
    extra_verifiers = [finite(row.get("verifier_rate")) or 0.0 for row in extra.values()]
    extra_completions = [
        finite(row.get("completion_health_rate")) or 0.0 for row in extra.values()
    ]
    extra_schema_deltas = [
        (finite(row.get("schema_valid_wrong_rate")) or 0.0) - base_schema
        for row in extra.values()
    ]
    extra_malformed_deltas = [
        (finite(row.get("malformed_rate")) or 0.0) - base_malformed
        for row in extra.values()
    ]
    extra_periods = [
        finite(row.get("completion_max_period_2_to_64_fraction")) or 0.0
        for row in extra.values()
    ]
    min_verifier = min(extra_verifiers)
    min_completion = min(extra_completions)
    summary.update(
        {
            "extra_eval_step_count": len(extra),
            "extra_eval_worst_steps": worst_steps,
            "extra_eval_min_verifier": min_verifier,
            "extra_eval_min_verifier_delta": min_verifier - base_verifier,
            "extra_eval_min_completion": min_completion,
            "extra_eval_min_completion_delta": min_completion - base_completion,
            "extra_eval_max_schema_delta": max(extra_schema_deltas),
            "extra_eval_max_malformed_delta": max(extra_malformed_deltas),
            "extra_eval_max_period_2_to_64": max(extra_periods),
        }
    )
    return summary


def contract_probe_summary(run_dir: Path, base: dict[str, Any]) -> dict[str, Any]:
    contract = last_capability_probe(run_dir, "ruliad_correctness_contract")
    if not contract:
        return {}
    base_verifier = finite(base.get("verifier_rate")) or 0.0
    base_completion = finite(base.get("completion_health_rate")) or 0.0
    base_field = finite(base.get("answer_field_accuracy")) or 0.0
    base_distinct = finite(base.get("actual_answer_distinct_fraction")) or 0.0
    contract_verifier = finite(contract.get("verifier_rate")) or 0.0
    contract_completion = finite(contract.get("completion_health_rate")) or 0.0
    contract_field = finite(contract.get("answer_field_accuracy")) or 0.0
    contract_distinct = finite(contract.get("actual_answer_distinct_fraction")) or 0.0
    return {
        "contract_probe_verifier_last": contract_verifier,
        "contract_probe_semantic_last": contract.get("semantic_rate"),
        "contract_probe_partial_last": contract.get("partial_credit_rate"),
        "contract_probe_schema_wrong_last": contract.get("schema_valid_wrong_rate"),
        "contract_probe_malformed_last": contract.get("malformed_rate"),
        "contract_probe_answer_field_accuracy_last": contract_field,
        "contract_probe_answer_field_coverage_last": contract.get("answer_field_coverage"),
        "contract_probe_completion_health_last": contract_completion,
        "contract_probe_answer_distinct_last": contract_distinct,
        "contract_probe_verifier_delta": contract_verifier - base_verifier,
        "contract_probe_completion_delta": contract_completion - base_completion,
        "contract_probe_answer_field_delta": contract_field - base_field,
        "contract_probe_answer_distinct_delta": contract_distinct - base_distinct,
    }


def capability_score_from_probe(row: dict[str, Any]) -> float:
    verifier = finite(row.get("verifier_rate")) or 0.0
    semantic = finite(row.get("semantic_rate")) or 0.0
    partial = finite(row.get("partial_credit_rate")) or 0.0
    completion = finite(row.get("completion_health_rate")) or 0.0
    schema_wrong = finite(row.get("schema_valid_wrong_rate")) or 0.0
    malformed = finite(row.get("malformed_rate")) or 0.0
    missing = finite(row.get("missing_rate")) or 0.0
    answer_field = finite(row.get("answer_field_accuracy")) or 0.0
    answer_termination = finite(row.get("answer_termination_rate")) or 0.0
    return (
        verifier * 6.0
        + semantic * 3.0
        + partial * 2.0
        + completion
        + answer_field
        + answer_termination * 0.5
        - schema_wrong * 2.0
        - malformed * 3.0
        - missing * 2.0
    )


def mean_auc(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [finite(row.get(field)) for row in rows]
    clean = [value for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def capability_first_pass(events: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    for event in events:
        if event.get("type") != "metric":
            continue
        if event.get("split") != "valid":
            continue
        if event.get("name") != "Ruliad Capability Gate Passed":
            continue
        value = finite(event.get("value"))
        if value is not None and value >= 1.0:
            epoch = event.get("epoch")
            step = event.get("absolute_step")
            return (
                int(epoch) if isinstance(epoch, int) else None,
                int(step) if isinstance(step, int) else None,
            )
    return None, None


def capability_probe_summary(run_dir: Path, events: list[dict[str, Any]]) -> dict[str, Any]:
    probes = [
        row
        for row in capability_probes(run_dir)
        if row.get("probe_name") == "ruliad_correctness"
    ]
    if not probes:
        return {}
    first_epoch, first_step = capability_first_pass(events)
    scored = [(capability_score_from_probe(row), row) for row in probes]
    best_score, best = max(scored, key=lambda item: item[0])
    first_score = scored[0][0]
    last_score = scored[-1][0]
    latest = probes[-1]
    verifier_values = [
        value for value in (finite(row.get("verifier_rate")) for row in probes) if value is not None
    ]
    completion_values = [
        value
        for value in (finite(row.get("completion_health_rate")) for row in probes)
        if value is not None
    ]
    verifier_last = verifier_values[-1] if verifier_values else None
    verifier_best = max(verifier_values) if verifier_values else None
    completion_last = completion_values[-1] if completion_values else None
    completion_best = max(completion_values) if completion_values else None
    buckets = latest.get("group_buckets") or []
    mastered = 0
    lagging = 0
    contract_mastered = 0
    contract_lagging = 0
    for bucket in buckets:
        completion = finite(bucket.get("completion_health_rate"))
        if completion is None:
            malformed = finite(bucket.get("malformed_rate")) or 0.0
            missing = finite(bucket.get("missing_rate")) or 0.0
            schema = finite(bucket.get("schema_valid_wrong_rate")) or 0.0
            completion = max(0.0, 1.0 - malformed - missing - schema)
        verifier = finite(bucket.get("verifier_rate")) or 0.0
        schema = finite(bucket.get("schema_valid_wrong_rate")) or 0.0
        malformed = finite(bucket.get("malformed_rate")) or 0.0
        missing = finite(bucket.get("missing_rate")) or 0.0
        bucket_mastered = capability_bucket_mastered(
            verifier, completion, schema, malformed, missing
        )
        bucket_lagging = capability_bucket_lagging(
            verifier, completion, schema, malformed, missing
        )
        if bucket_mastered:
            mastered += 1
        elif bucket_lagging:
            lagging += 1
        if str(bucket.get("label") or "").startswith("contract:"):
            if bucket_mastered:
                contract_mastered += 1
            elif bucket_lagging:
                contract_lagging += 1
    score_auc = sum(score for score, _ in scored) / len(scored)
    return {
        "capability_first_pass_epoch": first_epoch,
        "capability_first_pass_step": first_step,
        "capability_best_epoch": best.get("epoch"),
        "capability_best_score": best_score,
        "capability_score_auc": score_auc,
        "capability_verifier_auc": mean_auc(probes, "verifier_rate"),
        "capability_completion_auc": mean_auc(probes, "completion_health_rate"),
        "capability_schema_wrong_auc": mean_auc(probes, "schema_valid_wrong_rate"),
        "capability_malformed_auc": mean_auc(probes, "malformed_rate"),
        "capability_score_first": first_score,
        "capability_score_last": last_score,
        "capability_score_best": best_score,
        "capability_score_drop_from_best": max(0.0, best_score - last_score),
        "capability_verifier_best": verifier_best,
        "capability_verifier_drop_from_best": (
            max(0.0, verifier_best - verifier_last)
            if verifier_best is not None and verifier_last is not None
            else None
        ),
        "capability_completion_best": completion_best,
        "capability_completion_drop_from_best": (
            max(0.0, completion_best - completion_last)
            if completion_best is not None and completion_last is not None
            else None
        ),
        "capability_bucket_mastered_count": mastered,
        "capability_bucket_lagging_count": lagging,
        "capability_contract_mastered_count": contract_mastered,
        "capability_contract_lagging_count": contract_lagging,
    }


def capability_bucket_mastered(
    verifier: float,
    completion: float,
    schema: float,
    malformed: float,
    missing: float,
) -> bool:
    return verifier >= 0.50 and completion >= 0.80 and max(schema, malformed, missing) <= 0.10


def capability_bucket_lagging(
    verifier: float,
    completion: float,
    schema: float,
    malformed: float,
    missing: float,
) -> bool:
    if verifier > 0.05:
        return False
    structurally_parseable = malformed <= 0.25 and missing <= 0.25
    schema_valid_wrong = schema >= 0.25 and structurally_parseable
    answerable_but_unverified = completion >= 0.60 and structurally_parseable
    return schema_valid_wrong or answerable_but_unverified


def health_and_score(row: dict[str, Any]) -> tuple[bool, float]:
    score = 0.0
    status_ok = row.get("status") == "ok"
    train_delta = finite(row.get("train_delta")) or 0.0
    score += max(-2.0, min(3.0, train_delta))
    valid = finite(row.get("valid_teacher_ce_last"))
    if valid is not None:
        score += max(0.0, 3.0 - min(3.0, valid)) * 0.3
    verifier = finite(row.get("ruliad_verifier_last")) or 0.0
    semantic = finite(row.get("ruliad_semantic_last")) or 0.0
    partial = finite(row.get("ruliad_partial_last")) or 0.0
    schema_wrong = finite(row.get("ruliad_schema_wrong_last"))
    malformed = finite(row.get("ruliad_malformed_last"))
    missing = finite(row.get("ruliad_missing_last"))
    completion = finite(row.get("completion_health_last")) or 0.0
    answer_field = finite(row.get("ruliad_answer_field_accuracy_last")) or 0.0
    answer_termination = finite(row.get("ruliad_answer_termination_rate_last")) or 0.0
    completion_distinct2 = finite(row.get("completion_distinct_2_last"))
    completion_period = finite(row.get("completion_period_2_to_64_last"))
    completion_repetition = finite(row.get("completion_repetition_last"))
    capability_passed = finite(row.get("capability_gate_passed_last"))
    capability_failures = finite(row.get("capability_gate_failure_count_last")) or 0.0
    capability_score_drop = finite(row.get("capability_score_drop_from_best")) or 0.0
    capability_verifier_drop = finite(row.get("capability_verifier_drop_from_best")) or 0.0
    capability_completion_drop = finite(row.get("capability_completion_drop_from_best")) or 0.0
    capability_lagging_buckets = finite(row.get("capability_bucket_lagging_count")) or 0.0
    contract_verifier_delta = finite(row.get("contract_probe_verifier_delta")) or 0.0
    contract_field_delta = finite(row.get("contract_probe_answer_field_delta")) or 0.0
    contract_completion_delta = finite(row.get("contract_probe_completion_delta")) or 0.0
    recovery_fraction = finite(row.get("recovery_control_fraction")) or 0.0
    quality_recoveries = finite(row.get("capability_quality_recovery_count")) or 0.0
    score += (
        verifier * 6.0
        + semantic * 3.0
        + partial * 2.0
        + completion
        + answer_field
        + answer_termination * 0.5
    )
    if schema_wrong is not None:
        score -= schema_wrong * 2.0
    if malformed is not None:
        score -= malformed * 3.0
    if missing is not None:
        score -= missing * 2.0
    score -= capability_failures * 0.5
    if capability_passed == 0.0:
        score -= 2.0
    score -= min(3.0, capability_score_drop)
    score -= capability_verifier_drop * 4.0
    score -= capability_completion_drop * 2.0
    score -= min(3.0, max(0.0, capability_lagging_buckets - 8.0) * 0.10)
    score += min(1.0, max(0.0, contract_verifier_delta * 2.0))
    score += min(0.5, max(0.0, contract_field_delta))
    score += min(0.5, max(0.0, contract_completion_delta))
    score -= min(2.0, recovery_fraction * 2.0)
    score -= min(1.0, quality_recoveries * 0.25)
    entropy = finite(row.get("output_entropy_bits_last"))
    distinct2 = finite(row.get("output_distinct_2_last"))
    repetition = finite(row.get("output_repetition_last"))
    period = finite(row.get("output_period_2_to_64_last"))
    collapse_like = (
        entropy is not None
        and entropy < 0.5
        and (
            (repetition is not None and repetition > 0.5)
            or (period is not None and period > 0.5)
            or (distinct2 is not None and distinct2 < 0.05)
        )
    )
    if collapse_like:
        score -= 2.0
    if completion_distinct2 is not None and completion_distinct2 < 0.20:
        score -= 1.0
    if completion_period is not None and completion_period > 0.70:
        score -= 1.0
    if completion_repetition is not None and completion_repetition > 0.70:
        score -= 1.0
    fatal = int(row.get("fatal_gate_count") or 0)
    score -= fatal * 2.0
    healthy = (
        status_ok
        and fatal == 0
        and capability_passed != 0.0
        and not collapse_like
        and (repetition is None or repetition <= 0.85)
        and (period is None or period <= 0.85)
        and (completion_distinct2 is None or completion_distinct2 >= 0.20)
        and (completion_period is None or completion_period <= 0.70)
        and (completion_repetition is None or completion_repetition <= 0.70)
        and capability_score_drop <= 1.0
        and capability_verifier_drop <= 0.125
        and capability_completion_drop <= 0.30
        and capability_lagging_buckets <= 8.0
        and recovery_fraction <= 0.50
    )
    return healthy, score


def summarize_manifest(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    run_dir = Path(manifest.get("run_dir") or "")
    events = read_jsonl(run_dir / "events" / "training_events.jsonl")
    source = last_source_selection(run_dir) or {}
    source_difficulty = source_difficulty_summary(source)
    capability = last_capability_probe(run_dir) or {}
    eval_summary = eval_step_probe_summary(run_dir, capability, manifest.get("max_steps"))
    contract_summary = contract_probe_summary(run_dir, capability)
    gpu_log_path = Path(manifest["gpu_log_path"]) if manifest.get("gpu_log_path") else None
    log_path = Path(manifest["log_path"]) if manifest.get("log_path") else None
    gpu_util_mean, gpu_util_p50, gpu_power_mean = gpu_stats(gpu_log_path)
    stage_profile = parse_stage_profile(log_path)

    train_values = metric_values(events, "train", "Loss")
    train_first = train_values[0] if train_values else None
    train_last = train_values[-1] if train_values else None
    train_delta = (
        train_first - train_last
        if train_first is not None and train_last is not None
        else None
    )
    elapsed = finite(manifest.get("elapsed_seconds"))
    max_iters = finite(manifest.get("max_iters"))
    batch_size = finite(manifest.get("batch_size"))
    block_size = finite(manifest.get("block_size"))
    examples_per_sec = (
        max_iters * batch_size / elapsed
        if max_iters is not None and batch_size is not None and elapsed and elapsed > 0.0
        else None
    )
    tokens_per_sec = (
        max_iters * batch_size * block_size / elapsed
        if max_iters is not None
        and batch_size is not None
        and block_size is not None
        and elapsed
        and elapsed > 0.0
        else None
    )
    gate_count, fatal_gate_count = count_gates(events)
    row: dict[str, Any] = {
        "trial_key": manifest.get("trial_key"),
        "max_steps": manifest.get("max_steps"),
        "seed": manifest.get("seed"),
        "status": manifest.get("status"),
        "elapsed_seconds": elapsed,
        "tokens_per_sec": tokens_per_sec,
        "examples_per_sec": examples_per_sec,
        "stage_wall_tokens_per_sec": stage_profile.get("wall_tokens_per_second"),
        "stage_model_tokens_per_sec": stage_profile.get("model_tokens_per_second"),
        "stage_forward_ns": stage_profile.get("forward_ns"),
        "stage_backward_ns": stage_profile.get("loss_backward_ns"),
        "stage_validation_ns": stage_profile.get("validation_ns"),
        "stage_generation_ns": stage_profile.get("generation_ns"),
        "stage_event_sink_ns": stage_profile.get("event_sink_ns"),
        "gpu_util_mean": gpu_util_mean,
        "gpu_util_p50": gpu_util_p50,
        "gpu_power_mean": gpu_power_mean,
        "peak_used_mb": manifest.get("peak_used_mb"),
        "train_first": train_first,
        "train_last": train_last,
        "train_delta": train_delta,
        "valid_teacher_ce_last": last_metric(events, "valid", "Teacher Forced CE")
        or last_metric(events, "valid", "Loss"),
        "latent_loss_calls_last": last_metric(events, "train", "Latent Reasoning Loss Calls"),
        "latent_jepa_components_last": last_metric(
            events, "train", "Latent Reasoning JEPA Components"
        ),
        "latent_nextlat_components_last": last_metric(
            events, "train", "Latent Reasoning NextLat Components"
        ),
        "latent_energy_model_components_last": last_metric(
            events, "train", "Latent Reasoning Energy Model Components"
        ),
        "latent_step_contract_components_last": last_metric(
            events, "train", "Latent Reasoning Step Contract Components"
        ),
        "configured_steps_last": last_metric(
            events, "train", "Latent Reasoning Configured Steps"
        ),
        "source_loss_last": source.get("loss"),
        "source_entropy_bits_last": source.get("entropy_bits"),
        "source_active_candidate_count_last": source.get("active_candidate_count"),
        "source_active_max_entropy_bits_last": source.get("active_max_entropy_bits"),
        "source_normalized_entropy_last": source.get("normalized_entropy"),
        "source_mean_difficulty_last": source.get("mean_difficulty_level"),
        **source_difficulty,
        "source_max_difficulty_level_last": source.get("max_difficulty_level"),
        "source_materialized_frontier_edge_last": source.get("materialized_frontier_edge"),
        "source_max_difficulty_probability_last": source.get("max_difficulty_probability"),
        "source_normalized_difficulty_score_last": source.get("normalized_difficulty_score"),
        "source_target_difficulty_score_last": source.get("target_difficulty_score"),
        "source_hash_noise_probability_last": source.get("hash_noise_probability"),
        "source_capability_lagging_probability_last": source.get(
            "capability_lagging_probability"
        ),
        "ruliad_verifier_last": capability.get("verifier_rate"),
        "ruliad_semantic_last": capability.get("semantic_rate"),
        "ruliad_partial_last": capability.get("partial_credit_rate"),
        "ruliad_schema_wrong_last": capability.get("schema_valid_wrong_rate"),
        "ruliad_malformed_last": capability.get("malformed_rate"),
        "ruliad_missing_last": capability.get("missing_rate"),
        "ruliad_answer_field_accuracy_last": capability.get("answer_field_accuracy"),
        "ruliad_answer_field_coverage_last": capability.get("answer_field_coverage"),
        "ruliad_answer_termination_rate_last": capability.get("answer_termination_rate"),
        "ruliad_completion_quality_last": last_metric(
            events, "valid", "Ruliad Mean Completion Quality"
        ),
        "ruliad_expected_answer_distinct_last": last_metric(
            events, "valid", "Ruliad Expected Answer Distinct Fraction"
        ),
        "ruliad_answer_distinct_last": last_metric(
            events, "valid", "Ruliad Actual Answer Distinct Fraction"
        ),
        "ruliad_mean_completion_tokens_last": capability.get("mean_completion_tokens"),
        "completion_health_last": capability.get("completion_health_rate"),
        "completion_repetition_last": capability.get("completion_repetition_fraction"),
        "completion_distinct_1_last": capability.get("completion_distinct_1_fraction"),
        "completion_distinct_2_last": capability.get("completion_distinct_2_fraction"),
        "completion_period_2_to_16_last": capability.get(
            "completion_max_period_2_to_16_fraction"
        ),
        "completion_period_2_to_64_last": capability.get(
            "completion_max_period_2_to_64_fraction"
        ),
        "completion_dominant_period_2_to_64_last": capability.get(
            "completion_dominant_period_2_to_64"
        ),
        "capability_gate_passed_last": last_metric(
            events, "valid", "Ruliad Capability Gate Passed"
        ),
        "capability_gate_failure_count_last": last_metric(
            events, "valid", "Ruliad Capability Gate Failure Count"
        ),
        "capability_completion_health_metric_last": last_metric(
            events, "valid", "Ruliad Capability Completion Health Rate"
        ),
        "eval_step_sweep": manifest.get("eval_step_sweep"),
        "output_entropy_bits_last": last_metric(events, "valid", "Output Entropy Bits"),
        "output_distinct_2_last": last_metric(events, "valid", "Output Distinct-2 Fraction"),
        "output_repetition_last": last_metric(events, "valid", "Output Repetition Fraction"),
        "output_period_2_to_64_last": last_metric(
            events, "valid", "Output Max Period-2..64 Fraction"
        ),
        "gate_count": gate_count,
        "fatal_gate_count": fatal_gate_count,
        "run_dir": str(run_dir) if str(run_dir) else "",
    }
    row.update(latent_energy_eval_metrics(events, manifest.get("max_steps")))
    row.update(latent_extra_eval_trajectory_metrics(events, manifest.get("max_steps")))
    row.update(eval_summary)
    row.update(contract_summary)
    row.update(capability_probe_summary(run_dir, events))
    row.update(dynamics_and_gate_summary(events))
    healthy, score = health_and_score(row)
    row["healthy"] = healthy
    row["rank_score"] = score
    return row


def write_summary(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "latent_reasoning_steps_summary.csv"
    md_path = out_dir / "latent_reasoning_steps_summary.md"

    rows = sorted(
        rows,
        key=lambda row: (
            finite(row.get("max_steps")) if finite(row.get("max_steps")) is not None else 1e9,
            str(row.get("seed") or ""),
        ),
    )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in SUMMARY_COLUMNS})

    rank_rows = sorted(rows, key=lambda row: finite(row.get("rank_score")) or -1e9, reverse=True)
    with md_path.open("w") as handle:
        handle.write("# Latent Reasoning Max Steps Ablation\n\n")
        handle.write("| max_steps | status | wall tok/s | model tok/s | valid CE | active d | mastered d | source lag | verifier | semantic | partial | schema wrong | malformed | field | coverage | term | completion | contract verifier | contract field | contract completion | contract dver | contract dfield | comp d2 | comp period | cap gate | cap fails | cap first | cap auc | cap drop | rec frac | src rec | q rec | lag buckets | lag contracts | best eval steps | best eval verifier | extra steps | extra worst | extra dver | extra dcomp | extra dmal | extra lat CE | extra lat viol | extra lat H | extra lat rms | energy comps | step comps | out H | out d2 | gpu util | score |\n")
        handle.write("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rank_rows:
            handle.write(
                "| "
                + " | ".join(
                    [
                        fmt(row.get("max_steps")),
                        str(row.get("status") or ""),
                        fmt(row.get("tokens_per_sec")),
                        fmt(row.get("stage_model_tokens_per_sec")),
                        fmt(row.get("valid_teacher_ce_last")),
                        fmt(row.get("source_active_max_difficulty_last")),
                        fmt(row.get("source_mastered_max_difficulty_last")),
                        fmt(row.get("source_capability_lagging_probability_last")),
                        fmt(row.get("ruliad_verifier_last")),
                        fmt(row.get("ruliad_semantic_last")),
                        fmt(row.get("ruliad_partial_last")),
                        fmt(row.get("ruliad_schema_wrong_last")),
                        fmt(row.get("ruliad_malformed_last")),
                        fmt(row.get("ruliad_answer_field_accuracy_last")),
                        fmt(row.get("ruliad_answer_field_coverage_last")),
                        fmt(row.get("ruliad_answer_termination_rate_last")),
                        fmt(row.get("completion_health_last")),
                        fmt(row.get("contract_probe_verifier_last")),
                        fmt(row.get("contract_probe_answer_field_accuracy_last")),
                        fmt(row.get("contract_probe_completion_health_last")),
                        fmt(row.get("contract_probe_verifier_delta")),
                        fmt(row.get("contract_probe_answer_field_delta")),
                        fmt(row.get("completion_distinct_2_last")),
                        fmt(row.get("completion_period_2_to_64_last")),
                        fmt(row.get("capability_gate_passed_last")),
                        fmt(row.get("capability_gate_failure_count_last")),
                        fmt(row.get("capability_first_pass_epoch")),
                        fmt(row.get("capability_score_auc")),
                        fmt(row.get("capability_score_drop_from_best")),
                        fmt(row.get("recovery_control_fraction")),
                        fmt(row.get("source_capability_recovery_control_count")),
                        fmt(row.get("capability_quality_recovery_count")),
                        fmt(row.get("capability_bucket_lagging_count")),
                        fmt(row.get("capability_contract_lagging_count")),
                        fmt(row.get("best_eval_steps")),
                        fmt(row.get("best_eval_verifier")),
                        fmt(row.get("extra_eval_step_count")),
                        fmt(row.get("extra_eval_worst_steps")),
                        fmt(row.get("extra_eval_min_verifier_delta")),
                        fmt(row.get("extra_eval_min_completion_delta")),
                        fmt(row.get("extra_eval_max_malformed_delta")),
                        fmt(row.get("latent_extra_eval_max_ce_delta_last")),
                        fmt(row.get("latent_extra_eval_max_ce_violation_last")),
                        fmt(row.get("latent_extra_eval_min_entropy_bits_last")),
                        fmt(row.get("latent_extra_eval_max_delta_rms_last")),
                        fmt(row.get("latent_energy_model_components_last")),
                        fmt(row.get("latent_step_contract_components_last")),
                        fmt(row.get("output_entropy_bits_last")),
                        fmt(row.get("output_distinct_2_last")),
                        fmt(row.get("gpu_util_mean")),
                        fmt(row.get("rank_score")),
                    ]
                )
                + " |\n"
            )
        handle.write("\n")
        handle.write("Rank score is a triage helper only; use the adjacent metrics for conclusions.\n")

    print(md_path)
    print(csv_path)
    write_capability_bucket_summary(rows, out_dir)


def write_capability_bucket_summary(rows: list[dict[str, Any]], out_dir: Path) -> None:
    bucket_rows: list[dict[str, Any]] = []
    for row in rows:
        run_dir_text = row.get("run_dir")
        if not run_dir_text:
            continue
        run_dir = Path(str(run_dir_text))
        latest = last_capability_probe(run_dir) or {}
        for bucket in latest.get("group_buckets") or []:
            verifier = finite(bucket.get("verifier_rate")) or 0.0
            semantic = finite(bucket.get("semantic_rate")) or 0.0
            partial = finite(bucket.get("partial_credit_rate")) or 0.0
            schema = finite(bucket.get("schema_valid_wrong_rate")) or 0.0
            malformed = finite(bucket.get("malformed_rate")) or 0.0
            missing = finite(bucket.get("missing_rate")) or 0.0
            completion = finite(bucket.get("completion_health_rate"))
            if completion is None:
                completion = max(0.0, 1.0 - schema - malformed - missing)
            answer_field = finite(bucket.get("answer_field_accuracy"))
            answer_coverage = finite(bucket.get("answer_field_coverage"))
            answer_termination = finite(bucket.get("answer_termination_rate"))
            mastered = capability_bucket_mastered(verifier, completion, schema, malformed, missing)
            lagging = capability_bucket_lagging(verifier, completion, schema, malformed, missing)
            bucket_rows.append(
                {
                    "trial_key": row.get("trial_key"),
                    "max_steps": row.get("max_steps"),
                    "seed": row.get("seed"),
                    "bucket": bucket.get("label"),
                    "item_count": bucket.get("item_count"),
                    "verifier_rate": verifier,
                    "semantic_rate": semantic,
                    "partial_credit_rate": partial,
                    "completion_health_rate": completion,
                    "answer_field_accuracy": answer_field,
                    "answer_field_coverage": answer_coverage,
                    "answer_termination_rate": answer_termination,
                    "schema_valid_wrong_rate": schema,
                    "malformed_rate": malformed,
                    "missing_rate": missing,
                    "mastered": mastered,
                    "lagging": lagging,
                    "run_dir": row.get("run_dir"),
                }
            )
    if not bucket_rows:
        return
    csv_path = out_dir / "capability_bucket_summary.csv"
    md_path = out_dir / "capability_bucket_summary.md"
    fieldnames = [
        "trial_key",
        "max_steps",
        "seed",
        "bucket",
        "item_count",
        "verifier_rate",
        "semantic_rate",
        "partial_credit_rate",
        "completion_health_rate",
        "answer_field_accuracy",
        "answer_field_coverage",
        "answer_termination_rate",
        "schema_valid_wrong_rate",
        "malformed_rate",
        "missing_rate",
        "mastered",
        "lagging",
        "run_dir",
    ]
    bucket_rows.sort(
        key=lambda item: (
            str(item.get("trial_key") or ""),
            str(item.get("bucket") or ""),
        )
    )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(bucket_rows)
    with md_path.open("w") as handle:
        handle.write("# Capability Bucket Summary\n\n")
        handle.write("| trial | bucket | items | verifier | partial | field | coverage | terminated | completion | schema wrong | malformed | missing | mastered | lagging |\n")
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for item in bucket_rows:
            handle.write(
                "| "
                + " | ".join(
                    [
                        str(item.get("trial_key") or ""),
                        str(item.get("bucket") or ""),
                        fmt(item.get("item_count")),
                        fmt(item.get("verifier_rate")),
                        fmt(item.get("partial_credit_rate")),
                        fmt(item.get("answer_field_accuracy")),
                        fmt(item.get("answer_field_coverage")),
                        fmt(item.get("answer_termination_rate")),
                        fmt(item.get("completion_health_rate")),
                        fmt(item.get("schema_valid_wrong_rate")),
                        fmt(item.get("malformed_rate")),
                        fmt(item.get("missing_rate")),
                        str(item.get("mastered")),
                        str(item.get("lagging")),
                    ]
                )
                + " |\n"
            )
    print(md_path)
    print(csv_path)


def main() -> None:
    args = parse_args()
    root = Path(args.input)
    out_dir = Path(args.out_dir) if args.out_dir else root / "analysis"
    manifests = sorted((root / "manifests").glob("*.json"))
    if not manifests:
        raise SystemExit(f"no manifests found under {root / 'manifests'}")
    rows = [summarize_manifest(path) for path in manifests]
    write_summary(rows, out_dir)


if __name__ == "__main__":
    main()
