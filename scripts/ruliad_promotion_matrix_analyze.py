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
    "stage_model_tokens_per_sec",
    "gpu_util_mean",
    "elapsed_seconds",
    "peak_used_mb",
    "valid_teacher_ce_last",
    "source_mean_difficulty_last",
    "source_entropy_bits_last",
    "source_hash_noise_probability_last",
    "ruliad_verifier_last",
    "ruliad_semantic_last",
    "ruliad_partial_last",
    "ruliad_schema_wrong_last",
    "ruliad_malformed_last",
    "ruliad_answer_field_accuracy_last",
    "ruliad_answer_termination_rate_last",
    "completion_health_last",
    "completion_distinct_2_last",
    "completion_period_2_to_64_last",
    "completion_repetition_last",
    "capability_score_auc",
    "capability_verifier_auc",
    "capability_bucket_lagging_count",
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
    "policy_scalarization_count",
    "policy_reward_mean",
    "policy_reward_std",
    "policy_reward_min",
    "policy_reward_max",
    "policy_advantage_mean",
    "policy_advantage_std",
    "policy_advantage_clip_fraction",
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
    "policy_vpo_dominant_completion_health",
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
    parser.add_argument("--min-completion-distinct-2", type=float, default=0.20)
    parser.add_argument("--max-completion-period", type=float, default=0.70)
    parser.add_argument("--max-completion-repetition", type=float, default=0.70)
    parser.add_argument("--min-output-entropy", type=float, default=0.25)
    parser.add_argument("--min-output-distinct-2", type=float, default=0.10)
    parser.add_argument("--min-throughput-ratio", type=float, default=0.85)
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
        "policy_scalarization_count": last_sum("scalarization_count"),
        "policy_reward_mean": weighted_mean("reward_mean"),
        "policy_reward_std": weighted_mean("reward_std"),
        "policy_reward_min": min_value("reward_min"),
        "policy_reward_max": max_value("reward_max"),
        "policy_advantage_mean": weighted_mean("advantage_mean"),
        "policy_advantage_std": weighted_mean("advantage_std"),
        "policy_advantage_clip_fraction": weighted_mean("advantage_clip_fraction"),
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
        "policy_vpo_dominant_completion_health": last_sum("vpo_scalarization_dominant_completion_health"),
    }


def collect_trials(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in sorted(root.glob(f"*/analysis/{TRIAL_SUMMARY}")):
        arm = summary.parents[1].name
        for row in read_csv(summary):
            out: dict[str, Any] = {"arm": arm}
            out.update(row)
            out.update(read_policy_telemetry(row.get("run_dir")))
            rows.append(out)
    return rows


def summarize_by_arm(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for arm in sorted({str(row.get("arm") or "") for row in rows}):
        arm_rows = [row for row in rows if row.get("arm") == arm]
        ok_rows = [row for row in arm_rows if row.get("status") == "ok"]
        summary: dict[str, Any] = {
            "arm": arm,
            "trials": len(arm_rows),
            "ok_trials": len(ok_rows),
        }
        for column in METRIC_COLUMNS + POLICY_METRIC_COLUMNS:
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
    base_valid = value(baseline, "valid_teacher_ce_last_mean", 0.0)
    base_source = value(baseline, "source_mean_difficulty_last_mean", 0.0)
    base_verifier = value(baseline, "ruliad_verifier_last_mean", 0.0)
    base_partial = value(baseline, "ruliad_partial_last_mean", 0.0)
    base_schema = value(baseline, "ruliad_schema_wrong_last_mean", 0.0)
    base_malformed = value(baseline, "ruliad_malformed_last_mean", 0.0)
    base_answer_field = value(baseline, "ruliad_answer_field_accuracy_last_mean", 0.0)
    base_answer_termination = value(baseline, "ruliad_answer_termination_rate_last_mean", 0.0)
    base_completion = value(baseline, "completion_health_last_mean", 0.0)
    base_entropy = value(baseline, "output_entropy_bits_last_mean", 0.0)
    base_distinct2 = value(baseline, "output_distinct_2_last_mean", 0.0)

    out: list[dict[str, Any]] = []
    for row in rows:
        arm = str(row.get("arm") or "")
        model_tps = value(row, "stage_model_tokens_per_sec_mean", 0.0)
        valid = value(row, "valid_teacher_ce_last_mean", 0.0)
        source = value(row, "source_mean_difficulty_last_mean", 0.0)
        verifier = value(row, "ruliad_verifier_last_mean", 0.0)
        partial = value(row, "ruliad_partial_last_mean", 0.0)
        schema = value(row, "ruliad_schema_wrong_last_mean", 0.0)
        malformed = value(row, "ruliad_malformed_last_mean", 0.0)
        answer_field = value(row, "ruliad_answer_field_accuracy_last_mean", 0.0)
        answer_termination = value(row, "ruliad_answer_termination_rate_last_mean", 0.0)
        completion = value(row, "completion_health_last_mean", 0.0)
        completion_distinct2 = value(row, "completion_distinct_2_last_mean", 1.0)
        completion_period = value(row, "completion_period_2_to_64_last_mean", 0.0)
        completion_repetition = value(row, "completion_repetition_last_mean", 0.0)
        entropy = value(row, "output_entropy_bits_last_mean", 0.0)
        distinct2 = value(row, "output_distinct_2_last_mean", 0.0)
        throughput_ratio = model_tps / base_model_tps if base_model_tps > 0.0 else 0.0
        valid_delta = valid - base_valid
        source_delta = source - base_source
        verifier_delta = verifier - base_verifier
        partial_delta = partial - base_partial
        schema_delta = schema - base_schema
        malformed_delta = malformed - base_malformed
        answer_field_delta = answer_field - base_answer_field
        answer_termination_delta = answer_termination - base_answer_termination
        completion_delta = completion - base_completion
        entropy_delta = entropy - base_entropy
        distinct2_delta = distinct2 - base_distinct2
        fatal_gate_count = value(row, "fatal_gate_count_mean", 0.0)

        reasons: list[str] = []
        if int(row.get("ok_trials") or 0) < int(row.get("trials") or 0):
            reasons.append("failed_trials")
        if fatal_gate_count > 0.0:
            reasons.append("fatal_gates")
        if throughput_ratio < args.min_throughput_ratio:
            reasons.append("slow")
        if valid_delta > args.max_valid_ce_delta:
            reasons.append("valid_ce_regression")
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
        if completion_distinct2 < args.min_completion_distinct_2:
            reasons.append("completion_distinct2_collapse")
        if completion_period > args.max_completion_period:
            reasons.append("completion_period_collapse")
        if completion_repetition > args.max_completion_repetition:
            reasons.append("completion_repetition_collapse")
        if entropy < args.min_output_entropy:
            reasons.append("entropy_collapse")
        if distinct2 < args.min_output_distinct_2:
            reasons.append("output_distinct2_collapse")

        score_delta = (
            verifier_delta * 6.0
            + partial_delta * 2.0
            - schema_delta * 2.0
            + completion_delta * 1.5
            + answer_field_delta
            + answer_termination_delta * 0.5
            - valid_delta
            - max(source_delta, 0.0) * 0.20
            + (throughput_ratio - 1.0) * 0.50
            + entropy_delta * 0.05
            + distinct2_delta * 0.5
        )
        if arm == args.baseline_arm:
            decision = "control"
            reasons_text = ""
            score_delta = 0.0
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
                "promotion_score_delta": score_delta,
                "throughput_ratio": throughput_ratio,
                "valid_ce_delta": valid_delta,
                "source_difficulty_delta": source_delta,
                "verifier_delta": verifier_delta,
                "partial_delta": partial_delta,
                "schema_wrong_delta": schema_delta,
                "malformed_delta": malformed_delta,
                "completion_delta": completion_delta,
                "answer_field_delta": answer_field_delta,
                "answer_termination_delta": answer_termination_delta,
                "output_entropy_delta": entropy_delta,
                "output_distinct_2_delta": distinct2_delta,
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


def write_markdown(rows: list[dict[str, Any]], out_dir: Path, baseline_arm: str) -> None:
    path = out_dir / "ruliad_promotion_matrix_summary.md"
    with path.open("w") as handle:
        handle.write("# Ruliad Promotion Matrix\n\n")
        handle.write(f"Baseline arm: `{baseline_arm}`\n\n")
        handle.write("| arm | decision | ok/trials | seconds | peak MB | model tok/s | tput | valid CE | dCE | source diff | verifier | dver | partial | schema | dschema | field | dfield | term | dterm | completion | dcomp | comp d2 | comp period | out d2 | policy rstd | policy clip | policy comp | policy health | vpo compact | score d | reasons |\n")
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows:
            handle.write(
                "| "
                + " | ".join(
                    [
                        str(row.get("arm") or ""),
                        str(row.get("decision") or ""),
                        f"{row.get('ok_trials', 0)}/{row.get('trials', 0)}",
                        fmt(row.get("elapsed_seconds_mean")),
                        fmt(row.get("peak_used_mb_mean")),
                        fmt(row.get("stage_model_tokens_per_sec_mean")),
                        fmt(row.get("throughput_ratio")),
                        fmt(row.get("valid_teacher_ce_last_mean")),
                        fmt(row.get("valid_ce_delta")),
                        fmt(row.get("source_mean_difficulty_last_mean")),
                        fmt(row.get("ruliad_verifier_last_mean")),
                        fmt(row.get("verifier_delta")),
                        fmt(row.get("ruliad_partial_last_mean")),
                        fmt(row.get("ruliad_schema_wrong_last_mean")),
                        fmt(row.get("schema_wrong_delta")),
                        fmt(row.get("ruliad_answer_field_accuracy_last_mean")),
                        fmt(row.get("answer_field_delta")),
                        fmt(row.get("ruliad_answer_termination_rate_last_mean")),
                        fmt(row.get("answer_termination_delta")),
                        fmt(row.get("completion_health_last_mean")),
                        fmt(row.get("completion_delta")),
                        fmt(row.get("completion_distinct_2_last_mean")),
                        fmt(row.get("completion_period_2_to_64_last_mean")),
                        fmt(row.get("output_distinct_2_last_mean")),
                        fmt(row.get("policy_reward_std_mean")),
                        fmt(row.get("policy_advantage_clip_fraction_mean")),
                        fmt(row.get("policy_vector_compactness_mean_mean")),
                        fmt(row.get("policy_vector_completion_health_mean_mean")),
                        fmt(row.get("policy_vpo_dominant_compactness_mean")),
                        fmt(row.get("promotion_score_delta")),
                        str(row.get("fail_reasons") or ""),
                    ]
                )
                + " |\n"
            )
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
        ["arm", "decision", "fail_reasons", "trials", "ok_trials"]
        + [f"{column}_mean" for column in METRIC_COLUMNS + POLICY_METRIC_COLUMNS]
        + [
            "promotion_score_delta",
            "throughput_ratio",
            "valid_ce_delta",
            "source_difficulty_delta",
            "verifier_delta",
            "partial_delta",
            "schema_wrong_delta",
            "malformed_delta",
            "completion_delta",
            "answer_field_delta",
            "answer_termination_delta",
            "output_entropy_delta",
            "output_distinct_2_delta",
        ]
    )
    trial_fields = ["arm"] + [field for field in trials[0].keys() if field != "arm"]
    write_csv(out_dir / "ruliad_promotion_matrix_trials.csv", trials, trial_fields)
    write_csv(out_dir / "ruliad_promotion_matrix_arm_summary.csv", gated, arm_fields)
    write_markdown(gated, out_dir, args.baseline_arm)
    print(out_dir / "ruliad_promotion_matrix_arm_summary.csv")
    print(out_dir / "ruliad_promotion_matrix_trials.csv")


if __name__ == "__main__":
    main()
