#!/usr/bin/env python3
"""Summarize ruliad regularization ablation runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


SUMMARY_COLUMNS = [
    "trial_key",
    "arm",
    "seed",
    "iters",
    "batch_size",
    "block_size",
    "status",
    "elapsed_seconds",
    "examples_per_sec",
    "tokens_per_sec",
    "peak_used_mb",
    "gpu_util_mean",
    "gpu_util_p50",
    "gpu_power_mean",
    "run_dir",
    "train_first",
    "train_last",
    "train_delta",
    "valid_last",
    "source_loss_last",
    "source_entropy_bits_last",
    "source_hash_noise_probability_last",
    "source_mean_difficulty_last",
    "source_norm_difficulty_last",
    "source_mastered_probability_last",
    "source_verifier_failures_last",
    "ruliad_verifier_accuracy_last",
    "ruliad_partial_progress_last",
    "output_entropy_bits_last",
    "output_distinct_2_fraction_last",
    "output_repetition_fraction_last",
    "output_period_2_to_64_fraction_last",
    "gate_count",
    "fatal_gate_count",
    "healthy",
    "heuristic_rank_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="A run output directory produced by ruliad_regularization_ablation.sh")
    parser.add_argument("--out-dir", default=None, help="Analysis output directory")
    return parser.parse_args()


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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


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


def last_event(events: list[dict[str, Any]], event_type: str, **filters: Any) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("type") != event_type:
            continue
        if any(event.get(key) != value for key, value in filters.items()):
            continue
        return event
    return None


def metric_values(events: list[dict[str, Any]], split: str, name: str) -> list[float]:
    values: list[float] = []
    for event in events:
        if event.get("type") != "metric" or event.get("split") != split or event.get("name") != name:
            continue
        value = finite(event.get("value"))
        if value is not None:
            values.append(value)
    return values


def event_value(event: dict[str, Any] | None, key: str) -> float | None:
    if event is None:
        return None
    return finite(event.get(key))


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


def health_and_heuristic_rank_score(row: dict[str, Any]) -> tuple[bool, float]:
    """Return a triage-only rank score.

    This scalar is intentionally not a capability metric. It exists only to sort
    ablation rows by "worth inspecting first" while preserving the underlying
    metrics in adjacent columns.
    """
    score = 0.0
    status_ok = row.get("status") == "ok"
    train_delta = finite(row.get("train_delta")) or 0.0
    if train_delta > 0.0:
        score += min(2.0, train_delta)
    valid_last = finite(row.get("valid_last"))
    if valid_last is not None:
        score += max(0.0, 3.0 - min(3.0, valid_last)) * 0.25
    entropy = finite(row.get("output_entropy_bits_last"))
    if entropy is not None:
        score += min(2.0, entropy) * 0.5
    distinct2 = finite(row.get("output_distinct_2_fraction_last"))
    if distinct2 is not None:
        score += min(1.0, distinct2 * 2.0)
    repetition = finite(row.get("output_repetition_fraction_last"))
    if repetition is not None:
        score -= max(0.0, repetition - 0.5) * 2.0
    period = finite(row.get("output_period_2_to_64_fraction_last"))
    if period is not None:
        score -= max(0.0, period - 0.5) * 2.0
    fatal_gate_count = int(row.get("fatal_gate_count") or 0)
    score -= fatal_gate_count * 2.0
    hash_noise = finite(row.get("source_hash_noise_probability_last"))
    if hash_noise is not None:
        score -= max(0.0, hash_noise - 0.02) * 20.0
    verifier_failures = finite(row.get("source_verifier_failures_last")) or 0.0
    score -= verifier_failures
    healthy = (
        status_ok
        and fatal_gate_count == 0
        and (entropy is None or entropy >= 1.5)
        and (repetition is None or repetition <= 0.85)
        and (period is None or period <= 0.85)
        and (hash_noise is None or hash_noise <= 0.03)
        and verifier_failures == 0.0
    )
    return healthy, score


def summarize_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    run_dir = Path(manifest.get("run_dir") or "")
    training_config = read_json(run_dir / "training_config.json")
    training_section = training_config.get("training") if isinstance(training_config.get("training"), dict) else {}
    block_size = finite(training_section.get("block_size"))
    iters = finite(manifest.get("iters"))
    batch_size = finite(manifest.get("batch_size"))
    elapsed_seconds = finite(manifest.get("elapsed_seconds"))
    examples_per_sec = (
        (iters * batch_size / elapsed_seconds)
        if iters is not None and batch_size is not None and elapsed_seconds is not None and elapsed_seconds > 0.0
        else None
    )
    tokens_per_sec = (
        (examples_per_sec * block_size)
        if examples_per_sec is not None and block_size is not None
        else None
    )
    gpu_log = Path(manifest["gpu_log_path"]) if manifest.get("gpu_log_path") else None
    gpu_util_mean, gpu_util_p50, gpu_power_mean = gpu_stats(gpu_log)
    events = read_jsonl(run_dir / "events" / "training_events.jsonl")
    source_events = [event for event in events if event.get("type") == "source_selection"]
    if not source_events:
        source_events = read_jsonl(run_dir / "events" / "source_selection.jsonl")
    train_losses = metric_values(events, "train", "Loss")
    valid_losses = metric_values(events, "valid", "Loss")
    if not valid_losses:
        validation = last_event(events, "validation_finished")
        value = event_value(validation, "loss")
        if value is not None:
            valid_losses = [value]
    output = last_event(events, "output_degeneracy", split="valid") or last_event(events, "output_degeneracy")
    source = source_events[-1] if source_events else None
    ruliad = last_event(events, "ruliad_correctness")
    gates = [event for event in events if event.get("type") == "gate"]
    fatal_gates = [
        event
        for event in gates
        if str(event.get("severity", "")).lower() in {"error", "fatal"}
        or str(event.get("action", "")).lower() in {"stop", "fail", "halt"}
    ]
    row: dict[str, Any] = {
        "trial_key": manifest.get("trial_key", path.stem),
        "arm": manifest.get("arm"),
        "seed": manifest.get("seed"),
        "iters": manifest.get("iters"),
        "batch_size": manifest.get("batch_size"),
        "block_size": block_size,
        "status": manifest.get("status"),
        "elapsed_seconds": manifest.get("elapsed_seconds"),
        "examples_per_sec": examples_per_sec,
        "tokens_per_sec": tokens_per_sec,
        "peak_used_mb": manifest.get("peak_used_mb"),
        "gpu_util_mean": gpu_util_mean,
        "gpu_util_p50": gpu_util_p50,
        "gpu_power_mean": gpu_power_mean,
        "run_dir": str(run_dir),
        "train_first": train_losses[0] if train_losses else None,
        "train_last": train_losses[-1] if train_losses else None,
        "train_delta": (train_losses[0] - train_losses[-1]) if len(train_losses) >= 2 else None,
        "valid_last": valid_losses[-1] if valid_losses else None,
        "source_loss_last": event_value(source, "loss"),
        "source_entropy_bits_last": event_value(source, "entropy_bits"),
        "source_hash_noise_probability_last": event_value(source, "hash_noise_probability"),
        "source_mean_difficulty_last": event_value(source, "mean_difficulty_level"),
        "source_norm_difficulty_last": event_value(source, "normalized_difficulty_score"),
        "source_mastered_probability_last": event_value(source, "mastered_probability"),
        "source_verifier_failures_last": event_value(source, "verifier_failures"),
        "ruliad_verifier_accuracy_last": event_value(ruliad, "verifier_accuracy"),
        "ruliad_partial_progress_last": event_value(ruliad, "partial_progress"),
        "output_entropy_bits_last": event_value(output, "entropy_bits"),
        "output_distinct_2_fraction_last": event_value(output, "distinct_2_fraction"),
        "output_repetition_fraction_last": event_value(output, "repetition_fraction"),
        "output_period_2_to_64_fraction_last": event_value(output, "max_period_2_to_64_fraction"),
        "gate_count": len(gates),
        "fatal_gate_count": len(fatal_gates),
    }
    healthy, heuristic_rank_score = health_and_heuristic_rank_score(row)
    row["healthy"] = healthy
    row["heuristic_rank_score"] = heuristic_rank_score
    return row


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key)) for key in SUMMARY_COLUMNS})


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    ranked = sorted(
        rows,
        key=lambda row: finite(row.get("heuristic_rank_score")) or -1e9,
        reverse=True,
    )
    health = Counter(str(row.get("healthy")) for row in rows)
    arms = Counter(str(row.get("arm")) for row in rows)
    lines = [
        "# Ruliad Regularization Ablation",
        "",
        f"runs: {len(rows)}",
        f"healthy: {health.get('True', 0)} / {len(rows)}",
        f"arms: {', '.join(f'{arm}={count}' for arm, count in sorted(arms.items()))}",
        "",
        "| rank | arm | seed | iters | status | healthy | heuristic rank | tok/s | train delta | valid | out entropy | gpu util mean | gpu util p50 | gpu W mean | d2 | rep | period | source diff | run |",
        "|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for idx, row in enumerate(ranked, 1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    fmt(row.get("arm")),
                    fmt(row.get("seed")),
                    fmt(row.get("iters")),
                    fmt(row.get("status")),
                    fmt(row.get("healthy")),
                    fmt(row.get("heuristic_rank_score")),
                    fmt(row.get("tokens_per_sec")),
                    fmt(row.get("train_delta")),
                    fmt(row.get("valid_last")),
                    fmt(row.get("output_entropy_bits_last")),
                    fmt(row.get("gpu_util_mean")),
                    fmt(row.get("gpu_util_p50")),
                    fmt(row.get("gpu_power_mean")),
                    fmt(row.get("output_distinct_2_fraction_last")),
                    fmt(row.get("output_repetition_fraction_last")),
                    fmt(row.get("output_period_2_to_64_fraction_last")),
                    fmt(row.get("source_mean_difficulty_last")),
                    fmt(row.get("run_dir")),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    root = Path(args.input)
    out_dir = Path(args.out_dir) if args.out_dir else root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifests = sorted((root / "manifests").glob("*.json"))
    if not manifests:
        raise SystemExit(f"no manifests found under {root / 'manifests'}")
    rows = [summarize_manifest(path) for path in manifests]
    write_csv(out_dir / "regularization_summary.csv", rows)
    write_markdown(out_dir / "regularization_summary.md", rows)
    best = max(rows, key=lambda row: finite(row.get("heuristic_rank_score")) or -1e9)
    print(f"analyzed {len(rows)} runs -> {out_dir}")
    print(
        "best:"
        f" arm={best.get('arm')}"
        f" heuristic_rank={fmt(best.get('heuristic_rank_score'))}"
        f" healthy={fmt(best.get('healthy'))}"
        f" train_delta={fmt(best.get('train_delta'))}"
        f" valid={fmt(best.get('valid_last'))}"
        f" output_entropy={fmt(best.get('output_entropy_bits_last'))}"
        f" run={best.get('run_dir')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
