#!/usr/bin/env python3
"""Apply promotion gates to ruliad continual-learning candidate arms."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


TRIAL_SUMMARY = "latent_reasoning_steps_summary.csv"

METRIC_COLUMNS = [
    "stage_model_tokens_per_sec",
    "gpu_util_mean",
    "valid_teacher_ce_last",
    "source_mean_difficulty_last",
    "source_entropy_bits_last",
    "source_hash_noise_probability_last",
    "ruliad_verifier_last",
    "ruliad_semantic_last",
    "ruliad_partial_last",
    "ruliad_schema_wrong_last",
    "ruliad_malformed_last",
    "completion_health_last",
    "capability_score_auc",
    "capability_verifier_auc",
    "capability_bucket_lagging_count",
    "output_entropy_bits_last",
    "output_repetition_last",
    "output_period_2_to_64_last",
    "fatal_gate_count",
    "rank_score",
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
    parser.add_argument("--min-output-entropy", type=float, default=0.25)
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


def collect_trials(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in sorted(root.glob(f"*/analysis/{TRIAL_SUMMARY}")):
        arm = summary.parents[1].name
        for row in read_csv(summary):
            out: dict[str, Any] = {"arm": arm}
            out.update(row)
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
        for column in METRIC_COLUMNS:
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
    base_completion = value(baseline, "completion_health_last_mean", 0.0)
    base_entropy = value(baseline, "output_entropy_bits_last_mean", 0.0)

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
        completion = value(row, "completion_health_last_mean", 0.0)
        entropy = value(row, "output_entropy_bits_last_mean", 0.0)
        throughput_ratio = model_tps / base_model_tps if base_model_tps > 0.0 else 0.0
        valid_delta = valid - base_valid
        source_delta = source - base_source
        verifier_delta = verifier - base_verifier
        partial_delta = partial - base_partial
        schema_delta = schema - base_schema
        malformed_delta = malformed - base_malformed
        completion_delta = completion - base_completion
        entropy_delta = entropy - base_entropy
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
        if entropy < args.min_output_entropy:
            reasons.append("entropy_collapse")

        score_delta = (
            verifier_delta * 6.0
            + partial_delta * 2.0
            - schema_delta * 2.0
            + completion_delta * 1.5
            - valid_delta
            - max(source_delta, 0.0) * 0.20
            + (throughput_ratio - 1.0) * 0.50
            + entropy_delta * 0.05
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
                "output_entropy_delta": entropy_delta,
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
        handle.write("| arm | decision | ok/trials | model tok/s | tput | valid CE | dCE | source diff | dsource | verifier | dver | partial | schema | dschema | completion | dcomp | out H | score d | reasons |\n")
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows:
            handle.write(
                "| "
                + " | ".join(
                    [
                        str(row.get("arm") or ""),
                        str(row.get("decision") or ""),
                        f"{row.get('ok_trials', 0)}/{row.get('trials', 0)}",
                        fmt(row.get("stage_model_tokens_per_sec_mean")),
                        fmt(row.get("throughput_ratio")),
                        fmt(row.get("valid_teacher_ce_last_mean")),
                        fmt(row.get("valid_ce_delta")),
                        fmt(row.get("source_mean_difficulty_last_mean")),
                        fmt(row.get("source_difficulty_delta")),
                        fmt(row.get("ruliad_verifier_last_mean")),
                        fmt(row.get("verifier_delta")),
                        fmt(row.get("ruliad_partial_last_mean")),
                        fmt(row.get("ruliad_schema_wrong_last_mean")),
                        fmt(row.get("schema_wrong_delta")),
                        fmt(row.get("completion_health_last_mean")),
                        fmt(row.get("completion_delta")),
                        fmt(row.get("output_entropy_bits_last_mean")),
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
        + [f"{column}_mean" for column in METRIC_COLUMNS]
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
            "output_entropy_delta",
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
