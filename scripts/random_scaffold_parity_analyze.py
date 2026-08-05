#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

ACTIVE_GPU_UTILIZATION_FLOOR = 20.0
SELECTED_VARIANT = "rs-rank16"


def final_events(path: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("type") == "validation_finished":
                result["valid_loss"] = float(event["loss"])
            elif event.get("type") == "capability_probe":
                result.update(
                    verifier_rate=float(event["verifier_rate"]),
                    partial_credit_rate=float(event["partial_credit_rate"]),
                    completion_health_rate=float(event["completion_health_rate"]),
                    output_entropy_bits=float(event["output_entropy_bits"]),
                )
            elif (
                event.get("type") == "metric"
                and event.get("split") == "train"
                and event.get("name") == "Loss"
            ):
                result["train_loss"] = float(event["value"])
                result["last_train_step"] = float(event["absolute_step"])
    return result


def gpu_stats(path: Path) -> dict[str, float]:
    samples: list[tuple[float, float]] = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 3:
                continue
            try:
                power = float(row[1].strip())
                utilization = float(row[2].strip())
            except ValueError:
                continue
            if utilization >= ACTIVE_GPU_UTILIZATION_FLOOR:
                samples.append((power, utilization))
    if not samples:
        return {"active_gpu_samples": 0}
    return {
        "active_gpu_samples": len(samples),
        "active_power_w_mean": statistics.fmean(value[0] for value in samples),
        "active_power_w_max": max(value[0] for value in samples),
        "active_utilization_mean": statistics.fmean(value[1] for value in samples),
        "active_utilization_max": max(value[1] for value in samples),
    }


def peak_rss_kib(path: Path) -> int:
    for line in path.read_text(encoding="utf-8").splitlines():
        if "Maximum resident set size (kbytes)" in line:
            return int(line.rsplit(":", 1)[1].strip())
    raise ValueError(f"maximum RSS missing from {path}")


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the matched Dragon random-scaffold CUDA matrix."
    )
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        help="matrix manifest (defaults under target/experiments)",
    )
    parser.add_argument(
        "--require-gates",
        action="store_true",
        help="exit non-zero unless the selected scaffold clears quality and efficiency gates",
    )
    return parser.parse_args()


def variant_mean(
    summary: dict[str, dict[str, object]], variant: str, metric: str
) -> float:
    metric_summary = summary[variant][metric]
    assert isinstance(metric_summary, dict)
    return float(metric_summary["mean"])


def selected_comparison(
    records: list[dict[str, object]],
    summaries: dict[str, dict[str, object]],
) -> tuple[dict[str, object], dict[str, bool]]:
    dense = {int(row["seed"]): row for row in records if row["variant"] == "dense"}
    selected = {
        int(row["seed"]): row
        for row in records
        if row["variant"] == SELECTED_VARIANT
    }
    matched_seeds = sorted(set(dense).intersection(selected))
    dense_valid = variant_mean(summaries, "dense", "valid_loss")
    selected_valid = variant_mean(summaries, SELECTED_VARIANT, "valid_loss")
    dense_verifier = variant_mean(summaries, "dense", "verifier_rate")
    selected_verifier = variant_mean(summaries, SELECTED_VARIANT, "verifier_rate")
    dense_throughput = variant_mean(summaries, "dense", "steps_per_second")
    selected_throughput = variant_mean(
        summaries, SELECTED_VARIANT, "steps_per_second"
    )
    dense_utilization = variant_mean(
        summaries, "dense", "active_utilization_mean"
    )
    selected_utilization = variant_mean(
        summaries, SELECTED_VARIANT, "active_utilization_mean"
    )
    dense_completion = variant_mean(summaries, "dense", "completion_health_rate")
    selected_completion = variant_mean(
        summaries, SELECTED_VARIANT, "completion_health_rate"
    )
    per_seed_validation_ratios = {
        str(seed): float(selected[seed]["valid_loss"]) / float(dense[seed]["valid_loss"])
        for seed in matched_seeds
    }
    comparison: dict[str, object] = {
        "baseline": "dense",
        "selected": SELECTED_VARIANT,
        "matched_seeds": matched_seeds,
        "validation_loss_ratio": selected_valid / dense_valid,
        "validation_loss_change_fraction": selected_valid / dense_valid - 1.0,
        "verifier_rate_delta": selected_verifier - dense_verifier,
        "throughput_ratio": selected_throughput / dense_throughput,
        "active_gpu_utilization_ratio": selected_utilization / dense_utilization,
        "completion_health_rate_delta": selected_completion - dense_completion,
        "per_seed_validation_loss_ratios": per_seed_validation_ratios,
    }
    gates = {
        "three_matched_seeds": len(matched_seeds) >= 3
        and set(dense) == set(selected),
        "mean_validation_loss_within_2_percent": selected_valid
        <= dense_valid * 1.02,
        "every_seed_validation_loss_within_5_percent": bool(
            per_seed_validation_ratios
        )
        and max(per_seed_validation_ratios.values()) <= 1.05,
        "verifier_rate_not_worse_by_more_than_2_points": selected_verifier
        >= dense_verifier - 0.02,
        "throughput_at_least_90_percent": selected_throughput
        >= dense_throughput * 0.90,
        "active_gpu_utilization_at_least_90_percent": selected_utilization
        >= dense_utilization * 0.90,
        "completion_health_not_worse_by_more_than_5_points": selected_completion
        >= dense_completion - 0.05,
    }
    return comparison, gates


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = args.manifest or (
        root / "target/experiments/random-scaffold-parity/manifest.tsv"
    )
    rows: list[dict[str, object]] = []
    with manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            run_dir = root / "runs" / row["run_id"]
            metrics = final_events(run_dir / "events/training_events.jsonl")
            completed_steps = int(metrics.pop("last_train_step")) + 1
            elapsed_ms = int(row["elapsed_ms"])
            record: dict[str, object] = {
                "variant": row["variant"],
                "seed": int(row["seed"]),
                "run_id": row["run_id"],
                "elapsed_ms": elapsed_ms,
                "completed_steps": completed_steps,
                "steps_per_second": completed_steps * 1_000.0 / elapsed_ms,
                "peak_rss_kib": peak_rss_kib(Path(row["time_log"])),
                **metrics,
                **gpu_stats(Path(row["gpu_log"])),
            }
            rows.append(record)

    variants: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        variants.setdefault(str(row["variant"]), []).append(row)

    summary: dict[str, object] = {
        "manifest": str(manifest),
        "active_gpu_utilization_floor_percent": ACTIVE_GPU_UTILIZATION_FLOOR,
        "runs": rows,
        "variants": {},
    }
    for variant, records in variants.items():
        numeric_keys = [
            "valid_loss",
            "train_loss",
            "verifier_rate",
            "partial_credit_rate",
            "completion_health_rate",
            "output_entropy_bits",
            "steps_per_second",
            "peak_rss_kib",
            "active_power_w_mean",
            "active_utilization_mean",
        ]
        variant_summary: dict[str, object] = {"run_count": len(records)}
        for key in numeric_keys:
            values = [
                float(record[key])
                for record in records
                if key in record and math.isfinite(float(record[key]))
            ]
            if values:
                variant_summary[key] = mean_std(values)
        summary["variants"][variant] = variant_summary

    summaries = summary["variants"]
    assert isinstance(summaries, dict)
    comparison, gates = selected_comparison(rows, summaries)
    summary["selected_comparison"] = comparison
    summary["quality_efficiency_gates"] = gates
    summary["quality_efficiency_gate_passed"] = all(gates.values())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return int(args.require_gates and not all(gates.values())) * 2


if __name__ == "__main__":
    raise SystemExit(main())
