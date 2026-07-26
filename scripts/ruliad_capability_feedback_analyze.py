#!/usr/bin/env python3
"""Aggregate ruliad capability-feedback ablation arms."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


TRIAL_SUMMARY = "latent_reasoning_steps_summary.csv"
BUCKET_SUMMARY = "capability_bucket_summary.csv"


METRIC_COLUMNS = [
    "tokens_per_sec",
    "stage_model_tokens_per_sec",
    "gpu_util_mean",
    "valid_teacher_ce_last",
    "source_entropy_bits_last",
    "source_mean_difficulty_last",
    "source_hash_noise_probability_last",
    "ruliad_verifier_last",
    "ruliad_semantic_last",
    "ruliad_partial_last",
    "ruliad_schema_wrong_last",
    "ruliad_malformed_last",
    "ruliad_missing_last",
    "completion_health_last",
    "capability_score_auc",
    "capability_verifier_auc",
    "capability_completion_auc",
    "capability_schema_wrong_auc",
    "capability_malformed_auc",
    "capability_bucket_lagging_count",
    "rank_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Output directory from ruliad_capability_feedback_ablation.sh")
    parser.add_argument("--out-dir", default=None, help="Analysis output directory")
    return parser.parse_args()


def finite(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.upper() in {"N/A", "NA", "NONE"}:
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
            out = {"arm": arm}
            out.update(row)
            rows.append(out)
    return rows


def collect_buckets(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in sorted(root.glob(f"*/analysis/{BUCKET_SUMMARY}")):
        arm = summary.parents[1].name
        for row in read_csv(summary):
            out = {"arm": arm}
            out.update(row)
            rows.append(out)
    return rows


def summarize_by_arm(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    arms = sorted({str(row.get("arm") or "") for row in rows})
    summaries: list[dict[str, Any]] = []
    for arm in arms:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        ok_rows = [row for row in arm_rows if row.get("status") == "ok"]
        summary: dict[str, Any] = {
            "arm": arm,
            "trials": len(arm_rows),
            "ok_trials": len(ok_rows),
        }
        for column in METRIC_COLUMNS:
            values = [finite(row.get(column)) for row in ok_rows]
            clean = [value for value in values if value is not None]
            summary[f"{column}_mean"] = mean(clean)
        summaries.append(summary)
    return sorted(
        summaries,
        key=lambda row: finite(row.get("rank_score_mean")) or -1e9,
        reverse=True,
    )


def lagging_bucket_summary(bucket_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in bucket_rows:
        if str(row.get("lagging")).lower() != "true":
            continue
        key = (str(row.get("arm") or ""), str(row.get("bucket") or ""))
        entry = grouped.setdefault(
            key,
            {
                "arm": key[0],
                "bucket": key[1],
                "trials": 0,
                "items": 0.0,
                "verifier_sum": 0.0,
                "completion_sum": 0.0,
                "schema_sum": 0.0,
                "malformed_sum": 0.0,
            },
        )
        entry["trials"] += 1
        entry["items"] += finite(row.get("item_count")) or 0.0
        entry["verifier_sum"] += finite(row.get("verifier_rate")) or 0.0
        entry["completion_sum"] += finite(row.get("completion_health_rate")) or 0.0
        entry["schema_sum"] += finite(row.get("schema_valid_wrong_rate")) or 0.0
        entry["malformed_sum"] += finite(row.get("malformed_rate")) or 0.0
    out = []
    for entry in grouped.values():
        trials = max(1, int(entry["trials"]))
        out.append(
            {
                "arm": entry["arm"],
                "bucket": entry["bucket"],
                "trials": entry["trials"],
                "items": entry["items"],
                "verifier_mean": entry["verifier_sum"] / trials,
                "completion_mean": entry["completion_sum"] / trials,
                "schema_wrong_mean": entry["schema_sum"] / trials,
                "malformed_mean": entry["malformed_sum"] / trials,
            }
        )
    return sorted(out, key=lambda row: (-int(row["trials"]), str(row["arm"]), str(row["bucket"])))


def write_markdown(arm_rows: list[dict[str, Any]], lagging: list[dict[str, Any]], out_dir: Path) -> None:
    path = out_dir / "ruliad_capability_feedback_summary.md"
    with path.open("w") as handle:
        handle.write("# Ruliad Capability Feedback Ablation\n\n")
        handle.write("| arm | ok/trials | model tok/s | gpu util | valid CE | source diff | source H | verifier | partial | schema wrong | malformed | completion | cap AUC | verifier AUC | lag buckets | score |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in arm_rows:
            handle.write(
                "| "
                + " | ".join(
                    [
                        str(row.get("arm") or ""),
                        f"{row.get('ok_trials', 0)}/{row.get('trials', 0)}",
                        fmt(row.get("stage_model_tokens_per_sec_mean")),
                        fmt(row.get("gpu_util_mean_mean")),
                        fmt(row.get("valid_teacher_ce_last_mean")),
                        fmt(row.get("source_mean_difficulty_last_mean")),
                        fmt(row.get("source_entropy_bits_last_mean")),
                        fmt(row.get("ruliad_verifier_last_mean")),
                        fmt(row.get("ruliad_partial_last_mean")),
                        fmt(row.get("ruliad_schema_wrong_last_mean")),
                        fmt(row.get("ruliad_malformed_last_mean")),
                        fmt(row.get("completion_health_last_mean")),
                        fmt(row.get("capability_score_auc_mean")),
                        fmt(row.get("capability_verifier_auc_mean")),
                        fmt(row.get("capability_bucket_lagging_count_mean")),
                        fmt(row.get("rank_score_mean")),
                    ]
                )
                + " |\n"
            )
        if lagging:
            handle.write("\n## Frequent Lagging Buckets\n\n")
            handle.write("| arm | bucket | trials | items | verifier | completion | schema wrong | malformed |\n")
            handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
            for row in lagging[:40]:
                handle.write(
                    "| "
                    + " | ".join(
                        [
                            str(row.get("arm") or ""),
                            str(row.get("bucket") or ""),
                            fmt(row.get("trials")),
                            fmt(row.get("items")),
                            fmt(row.get("verifier_mean")),
                            fmt(row.get("completion_mean")),
                            fmt(row.get("schema_wrong_mean")),
                            fmt(row.get("malformed_mean")),
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
        raise SystemExit(f"no per-arm summaries found under {root}")
    arms = summarize_by_arm(trials)
    buckets = collect_buckets(root)
    lagging = lagging_bucket_summary(buckets)
    trial_fields = ["arm"] + [field for field in trials[0].keys() if field != "arm"]
    arm_fields = ["arm", "trials", "ok_trials"] + [f"{column}_mean" for column in METRIC_COLUMNS]
    lagging_fields = [
        "arm",
        "bucket",
        "trials",
        "items",
        "verifier_mean",
        "completion_mean",
        "schema_wrong_mean",
        "malformed_mean",
    ]
    write_csv(out_dir / "ruliad_capability_feedback_trials.csv", trials, trial_fields)
    write_csv(out_dir / "ruliad_capability_feedback_arm_summary.csv", arms, arm_fields)
    write_csv(out_dir / "ruliad_capability_feedback_lagging_buckets.csv", lagging, lagging_fields)
    write_markdown(arms, lagging, out_dir)
    print(out_dir / "ruliad_capability_feedback_arm_summary.csv")
    print(out_dir / "ruliad_capability_feedback_trials.csv")


if __name__ == "__main__":
    main()
