#!/usr/bin/env python3
"""Aggregate paired Ruliad checkpoint-evaluation reports.

Input filenames should use ``<arm>-seed<seed>.json``. The evaluator's older
``<arm>-final.json`` form is also accepted when its checkpoint path contains a
``-seed<seed>-`` segment. Reports are rejected when they do not share the same
fixed-panel fingerprint, preventing accidental comparisons across different
generated evaluation sets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import tempfile
from pathlib import Path
from typing import Any, Iterable


REPORT_NAME = re.compile(r"^(?P<arm>.+)-seed(?P<seed>\d+)\.json$")
FINAL_REPORT_NAME = re.compile(r"^(?P<arm>.+)-final\.json$")
CHECKPOINT_SEED = re.compile(r"(?:^|[-_/])seed(?P<seed>\d+)(?:[-_/]|$)")
T_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


CORE_METRICS = [
    "free_verifier_accuracy",
    "free_semantic_accuracy",
    "free_exact_accuracy",
    "free_answer_field_accuracy",
    "free_answer_field_coverage",
    "free_answer_termination_rate",
    "free_malformed_completion_rate",
    "free_actual_answer_distinct_fraction",
    "free_actual_answer_dominant_fraction",
    "policy_equivalent_top1_rate",
    "policy_equivalent_nll",
    "policy_valid_invalid_margin",
    "policy_context_swap_top1_change_rate",
    "policy_context_swap_equivalent_probability_drop",
    "policy_counterfactual_target_top1_change_rate",
    "policy_counterfactual_target_equivalent_probability_gain",
    "policy_orbit_js_divergence",
    "policy_orbit_top1_consensus_fraction",
    "rollout_solve_rate",
    "rollout_goal_completion_rate",
    "rollout_valid_action_rate",
    "rollout_invalid_action_rate",
    "rollout_top1_expert_rate",
    "rollout_frontier_exhaustion_rate",
    "rollout_mean_steps",
]


def nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def report_identity(path: Path, document: dict[str, Any]) -> tuple[str, int]:
    match = REPORT_NAME.match(path.name)
    if match is not None:
        return match.group("arm"), int(match.group("seed"))
    final_match = FINAL_REPORT_NAME.match(path.name)
    checkpoint_match = CHECKPOINT_SEED.search(str(document.get("checkpoint", "")))
    if final_match is not None and checkpoint_match is not None:
        return final_match.group("arm"), int(checkpoint_match.group("seed"))
    raise ValueError(
        "report identity requires <arm>-seed<seed>.json or an evaluator "
        f"<arm>-final.json with a seeded checkpoint path: {path}"
    )


def report_row(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    arm, seed = report_identity(path, document)
    evaluation = nested(document, "evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError(f"missing evaluation object: {path}")
    free = nested(evaluation, "free_run", "report", default={})
    policy = nested(evaluation, "constrained_policy", default={})
    rollout = nested(evaluation, "closed_loop_rollout", default={})
    if not all(isinstance(section, dict) for section in (free, policy, rollout)):
        raise ValueError(f"malformed evaluation sections: {path}")

    scored = finite_number(free.get("scored_count")) or 0.0
    malformed = finite_number(free.get("malformed_completion_count")) or 0.0
    row: dict[str, Any] = {
        "arm": arm,
        "seed": seed,
        "path": str(path),
        "backend": document.get("backend"),
        "checkpoint_epoch": document.get("checkpoint_epoch"),
        "panel_fingerprint_sha256": evaluation.get("panel_fingerprint_sha256"),
        "free_verifier_accuracy": finite_number(free.get("verifier_accuracy")),
        "free_semantic_accuracy": finite_number(free.get("semantic_accuracy")),
        "free_exact_accuracy": finite_number(free.get("exact_accuracy")),
        "free_answer_field_accuracy": finite_number(free.get("answer_field_accuracy")),
        "free_answer_field_coverage": finite_number(free.get("answer_field_coverage")),
        "free_answer_termination_rate": finite_number(free.get("answer_termination_rate")),
        "free_malformed_completion_rate": malformed / scored if scored > 0.0 else None,
        "free_actual_answer_distinct_fraction": finite_number(
            free.get("actual_answer_distinct_fraction")
        ),
        "free_actual_answer_dominant_fraction": finite_number(
            free.get("actual_answer_dominant_fraction")
        ),
    }
    for name, source in (
        ("policy", policy),
        ("rollout", rollout),
    ):
        for key, value in source.items():
            number = finite_number(value)
            if number is not None and key != "items":
                row[f"{name}_{key}"] = number

    by_difficulty = evaluation.get("rollout_by_difficulty", {})
    if isinstance(by_difficulty, dict):
        for difficulty, metrics in sorted(by_difficulty.items(), key=lambda item: item[0]):
            if not isinstance(metrics, dict):
                continue
            for key, value in metrics.items():
                number = finite_number(value)
                if number is not None and key != "items":
                    row[f"rollout_d{difficulty}_{key}"] = number
    return row


def mean_ci(values: list[float]) -> tuple[float, float | None, float | None]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, None, None
    std = statistics.stdev(values)
    critical = T_95.get(len(values) - 1, 1.96)
    half_width = critical * std / math.sqrt(len(values))
    return mean, std, half_width


def numeric_metrics(rows: Iterable[dict[str, Any]]) -> list[str]:
    excluded = {
        "arm",
        "seed",
        "path",
        "backend",
        "checkpoint_epoch",
        "panel_fingerprint_sha256",
    }
    return sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in excluded and finite_number(value) is not None
        },
        key=lambda key: (key not in CORE_METRICS, CORE_METRICS.index(key) if key in CORE_METRICS else key),
    )


def summarize(rows: list[dict[str, Any]], metrics: list[str]) -> list[dict[str, Any]]:
    arms = sorted({str(row["arm"]) for row in rows})
    output = []
    for arm in arms:
        arm_rows = [row for row in rows if row["arm"] == arm]
        for metric in metrics:
            values = [value for row in arm_rows if (value := finite_number(row.get(metric))) is not None]
            if not values:
                continue
            mean, std, ci95 = mean_ci(values)
            output.append(
                {
                    "arm": arm,
                    "metric": metric,
                    "n": len(values),
                    "mean": mean,
                    "std": std,
                    "ci95_half_width": ci95,
                }
            )
    return output


def paired(
    rows: list[dict[str, Any]], metrics: list[str], reference_arm: str
) -> list[dict[str, Any]]:
    by_key = {(str(row["arm"]), int(row["seed"])): row for row in rows}
    arms = sorted({str(row["arm"]) for row in rows if row["arm"] != reference_arm})
    reference_seeds = {int(row["seed"]) for row in rows if row["arm"] == reference_arm}
    output = []
    for arm in arms:
        arm_seeds = {int(row["seed"]) for row in rows if row["arm"] == arm}
        if arm_seeds != reference_seeds:
            raise ValueError(
                f"paired arms have different seeds: {reference_arm}={sorted(reference_seeds)} "
                f"{arm}={sorted(arm_seeds)}"
            )
        for metric in metrics:
            differences = []
            for seed in sorted(reference_seeds):
                candidate = finite_number(by_key[(arm, seed)].get(metric))
                reference = finite_number(by_key[(reference_arm, seed)].get(metric))
                if candidate is not None and reference is not None:
                    differences.append(candidate - reference)
            if not differences:
                continue
            mean, std, ci95 = mean_ci(differences)
            output.append(
                {
                    "arm": arm,
                    "reference_arm": reference_arm,
                    "metric": metric,
                    "n": len(differences),
                    "mean_delta": mean,
                    "std_delta": std,
                    "ci95_half_width": ci95,
                    "ci95_excludes_zero": ci95 is not None and abs(mean) > ci95,
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    number = finite_number(value)
    if number is None:
        return "-"
    return f"{number:.6g}"


def markdown_report(
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    deltas: list[dict[str, Any]],
    reference_arm: str,
) -> str:
    fingerprints = sorted({str(row["panel_fingerprint_sha256"]) for row in rows})
    lines = [
        "# Ruliad Checkpoint Evaluation",
        "",
        f"- Reports: {len(rows)}",
        f"- Arms: {', '.join(sorted({str(row['arm']) for row in rows}))}",
        f"- Reference arm: `{reference_arm}`",
        f"- Fixed-panel fingerprint: `{fingerprints[0]}`",
        "",
        "## Core Metrics",
        "",
        "| Arm | Metric | n | Mean | Std | 95% CI half-width |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        if row["metric"] in CORE_METRICS or row["metric"].startswith("rollout_d"):
            lines.append(
                f"| {row['arm']} | {row['metric']} | {row['n']} | {fmt(row['mean'])} | "
                f"{fmt(row['std'])} | {fmt(row['ci95_half_width'])} |"
            )
    lines.extend(
        [
            "",
            "## Paired Deltas",
            "",
            "Positive deltas mean the candidate arm is numerically larger; interpret lower-is-better metrics such as NLL separately.",
            "",
            "| Arm - Reference | Metric | n | Mean delta | Std | 95% CI half-width | Excludes zero |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in deltas:
        if row["metric"] in CORE_METRICS or row["metric"].startswith("rollout_d"):
            lines.append(
                f"| {row['arm']} - {row['reference_arm']} | {row['metric']} | {row['n']} | "
                f"{fmt(row['mean_delta'])} | {fmt(row['std_delta'])} | "
                f"{fmt(row['ci95_half_width'])} | {row['ci95_excludes_zero']} |"
            )
    lines.append("")
    return "\n".join(lines)


def analyze(paths: list[Path], output_dir: Path, reference_arm: str) -> dict[str, Any]:
    rows = [report_row(path) for path in sorted(paths)]
    if not rows:
        raise ValueError("no checkpoint evaluation reports found")
    duplicates = [(row["arm"], row["seed"]) for row in rows]
    if len(duplicates) != len(set(duplicates)):
        raise ValueError("duplicate arm/seed checkpoint reports")
    fingerprints = {row["panel_fingerprint_sha256"] for row in rows}
    if None in fingerprints or len(fingerprints) != 1:
        raise ValueError(f"reports do not share one panel fingerprint: {sorted(map(str, fingerprints))}")
    arms = {str(row["arm"]) for row in rows}
    if reference_arm not in arms:
        raise ValueError(f"reference arm {reference_arm!r} not found in {sorted(arms)}")

    metrics = numeric_metrics(rows)
    summary_rows = summarize(rows, metrics)
    delta_rows = paired(rows, metrics, reference_arm)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_columns = [
        "arm",
        "seed",
        "path",
        "backend",
        "checkpoint_epoch",
        "panel_fingerprint_sha256",
        *metrics,
    ]
    write_csv(output_dir / "checkpoint_eval_rows.csv", rows, raw_columns)
    write_csv(
        output_dir / "checkpoint_eval_summary.csv",
        summary_rows,
        ["arm", "metric", "n", "mean", "std", "ci95_half_width"],
    )
    write_csv(
        output_dir / "checkpoint_eval_paired_deltas.csv",
        delta_rows,
        [
            "arm",
            "reference_arm",
            "metric",
            "n",
            "mean_delta",
            "std_delta",
            "ci95_half_width",
            "ci95_excludes_zero",
        ],
    )
    payload = {
        "version": 1,
        "reference_arm": reference_arm,
        "panel_fingerprint_sha256": next(iter(fingerprints)),
        "rows": rows,
        "summary": summary_rows,
        "paired_deltas": delta_rows,
    }
    (output_dir / "checkpoint_eval_analysis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "checkpoint_eval_report.md").write_text(
        markdown_report(rows, summary_rows, delta_rows, reference_arm)
    )
    return payload


def fixture_report(solve_rate: float, top1: float, fingerprint: str = "panel") -> dict[str, Any]:
    return {
        "backend": "cuda",
        "checkpoint_epoch": 3,
        "evaluation": {
            "panel_fingerprint_sha256": fingerprint,
            "free_run": {
                "report": {
                    "scored_count": 4,
                    "malformed_completion_count": 1,
                    "verifier_accuracy": 0.25,
                    "semantic_accuracy": 0.25,
                    "exact_accuracy": 0.0,
                    "answer_field_accuracy": 0.25,
                    "answer_field_coverage": 1.0,
                    "answer_termination_rate": 1.0,
                    "actual_answer_distinct_fraction": 0.5,
                    "actual_answer_dominant_fraction": 0.5,
                }
            },
            "constrained_policy": {"items": 4, "equivalent_top1_rate": top1, "equivalent_nll": 0.5},
            "closed_loop_rollout": {
                "items": 4,
                "solve_rate": solve_rate,
                "goal_completion_rate": solve_rate,
                "valid_action_rate": 1.0,
            },
            "rollout_by_difficulty": {
                "0": {"items": 2, "solve_rate": solve_rate, "goal_completion_rate": solve_rate}
            },
        },
    }


def self_test() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for arm, values in {
            "adam": [(0.25, 0.50), (0.50, 0.60)],
            "pc": [(0.50, 0.70), (0.75, 0.80)],
        }.items():
            for seed, (solve, top1) in enumerate(values, start=7):
                (root / f"{arm}-seed{seed}.json").write_text(
                    json.dumps(fixture_report(solve, top1))
                )
        payload = analyze(list(root.glob("*.json")), root / "analysis", "adam")
        solve_delta = next(
            row
            for row in payload["paired_deltas"]
            if row["arm"] == "pc" and row["metric"] == "rollout_solve_rate"
        )
        assert solve_delta["n"] == 2
        assert abs(solve_delta["mean_delta"] - 0.25) < 1.0e-12
        assert (root / "analysis/checkpoint_eval_report.md").is_file()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="report JSON files or directories")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoint-eval-analysis"))
    parser.add_argument("--reference-arm", default="adam")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def expand_paths(paths: list[Path]) -> list[Path]:
    expanded = []
    for path in paths:
        if path.is_dir():
            expanded.extend(path.glob("*.json"))
        else:
            expanded.append(path)
    return sorted(set(expanded))


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        print("ruliad checkpoint evaluator analyzer self-test passed")
        return
    analyze(expand_paths(args.paths), args.output_dir, args.reference_arm)
    print(args.output_dir / "checkpoint_eval_report.md")


if __name__ == "__main__":
    main()
