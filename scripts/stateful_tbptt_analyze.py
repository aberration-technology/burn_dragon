#!/usr/bin/env python3
"""Analyze matched stateful-TBPTT training trials without third-party packages."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


METRICS = {
    "train_loss": ("train", ("Stream Warm Loss", "Loss")),
    "valid_loss": ("valid", ("Random Cold Loss", "Loss")),
    "stream_warm_loss": ("valid", ("Stream Warm Loss",)),
    "paired_warm_loss": ("valid", ("Stream Paired Warm Loss",)),
    "paired_cold_loss": ("valid", ("Stream Paired Cold Loss",)),
    "carry_nll_gain": ("valid", ("Stream Carry NLL Gain",)),
    "carry_relative_gain": ("valid", ("Stream Carry Relative Gain",)),
    "paired_batches": ("valid", ("Stream Carry Probe Batches",)),
    "rho_rms": ("valid", ("Sequence State Rho RMS",)),
    "rho_variance_ratio": ("valid", ("Sequence State Rho Slot Variance Ratio",)),
    "rho_redundancy": ("valid", ("Sequence State Rho Slot Redundancy",)),
    "canonical_verifier_accuracy": ("valid", ("Ruliad Verifier Accuracy",)),
    "training_serialization_verifier_accuracy": (
        "valid",
        ("Ruliad Training Serialization Ruliad Verifier Accuracy",),
    ),
    "partial_credit": ("valid", ("Ruliad Partial Credit Rate",)),
    "completion_quality": ("valid", ("Ruliad Mean Completion Quality",)),
    "output_entropy_bits": ("valid", ("Output Entropy Bits",)),
    "output_repetition": ("valid", ("Output Repetition Fraction",)),
}

CORE_ARMS = {
    "block512_reset",
    "block512_carry",
    "chunk128_reset",
    "chunk128_carry",
}


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def numeric_cell(value: str) -> float | None:
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", value)
    return finite(match.group(0)) if match else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {path}:{line_number}: {error}") from error
        if isinstance(row, dict):
            rows.append(row)
    return rows


def last_metric(events: Iterable[dict[str, Any]], split: str, name: str) -> float | None:
    result = None
    for event in events:
        if event.get("type") != "metric" or event.get("split") != split:
            continue
        if event.get("name") == name:
            result = finite(event.get("running_value", event.get("value")))
    return result


def last_metric_any(
    events: Iterable[dict[str, Any]], split: str, names: tuple[str, ...]
) -> float | None:
    for name in names:
        value = last_metric(events, split, name)
        if value is not None:
            return value
    return None


def metric_series(
    events: Iterable[dict[str, Any]], split: str, name: str
) -> list[float]:
    values = []
    for event in events:
        if (
            event.get("type") == "metric"
            and event.get("split") == split
            and event.get("name") == name
        ):
            value = finite(event.get("value"))
            if value is not None:
                values.append(value)
    return values


def trailing_nonzero_train_loss(events: Iterable[dict[str, Any]]) -> tuple[float | None, float | None]:
    values = []
    for name in METRICS["train_loss"][1]:
        values = metric_series(events, "train", name)
        if values:
            break
    if not values:
        return None, None
    nonzero = [value for value in values if abs(value) > 1.0e-12]
    trailing = nonzero[-8:]
    loss = statistics.fmean(trailing) if trailing else 0.0
    zero_fraction = sum(abs(value) <= 1.0e-12 for value in values) / len(values)
    return loss, zero_fraction


def parse_stage_profile(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    line = next(
        (
            value
            for value in reversed(path.read_text(errors="replace").splitlines())
            if "[stage-profile][training]" in value
        ),
        "",
    )
    fields: dict[str, float] = {}
    for key, raw in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", line):
        value = finite(raw)
        if value is not None:
            fields[key] = value
    return fields


def parse_gpu_log(path: Path) -> dict[str, float]:
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    with path.open(newline="", errors="replace") as handle:
        rows = list(csv.DictReader(handle, skipinitialspace=True))
    if not rows:
        return {}

    def values(prefix: str) -> list[float]:
        column = next((name for name in rows[0] if name.strip().startswith(prefix)), None)
        if column is None:
            return []
        parsed = [numeric_cell(row.get(column, "")) for row in rows]
        return [value for value in parsed if value is not None]

    result = {}
    for output, prefix in [
        ("gpu_util_mean", "utilization.gpu"),
        ("gpu_power_mean_w", "power.draw"),
        ("gpu_memory_peak_mb", "memory.used"),
    ]:
        samples = values(prefix)
        if samples:
            result[output] = max(samples) if output.endswith("peak_mb") else statistics.fmean(samples)
            if output == "gpu_util_mean":
                ordered = sorted(samples)
                result["gpu_util_p10"] = ordered[max(0, math.ceil(0.10 * len(ordered)) - 1)]
                result["gpu_util_p90"] = ordered[max(0, math.ceil(0.90 * len(ordered)) - 1)]
    return result


def parse_time_log(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    result = {}
    for line in path.read_text(errors="replace").splitlines():
        if "Maximum resident set size (kbytes):" in line:
            value = numeric_cell(line.rsplit(":", 1)[-1])
            if value is not None:
                result["process_peak_rss_mb"] = value / 1024.0
    return result


def load_trial(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    run_dir = Path(manifest.get("run_dir") or "")
    events = read_jsonl(run_dir / "events" / "training_events.jsonl") if run_dir else []
    row: dict[str, Any] = dict(manifest)
    row["manifest_path"] = str(manifest_path)
    for output, (split, names) in METRICS.items():
        row[output] = last_metric_any(events, split, names)
    row["train_loss"], row["zero_train_loss_sample_fraction"] = trailing_nonzero_train_loss(
        events
    )
    stage = parse_stage_profile(Path(manifest.get("log_path") or ""))
    row["wall_tokens_per_second"] = stage.get("wall_tokens_per_second")
    row["model_tokens_per_second"] = stage.get("model_tokens_per_second")
    row["model_duty_fraction"] = stage.get("model_duty_fraction")
    row["validation_fraction"] = stage.get("validation_fraction")
    row.update(parse_gpu_log(Path(manifest.get("gpu_log_path") or "")))
    row.update(parse_time_log(Path(manifest.get("time_log_path") or "")))
    return row


def mean(values: Iterable[Any]) -> float | None:
    valid = [value for item in values if (value := finite(item)) is not None]
    return statistics.fmean(valid) if valid else None


def sample_sd(values: Iterable[Any]) -> float | None:
    valid = [value for item in values if (value := finite(item)) is not None]
    return statistics.stdev(valid) if len(valid) >= 2 else None


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("arm"))].append(row)
    outputs = []
    numeric_fields = list(METRICS) + [
        "elapsed_seconds",
        "peak_used_mb",
        "wall_tokens_per_second",
        "model_tokens_per_second",
        "model_duty_fraction",
        "validation_fraction",
        "gpu_util_mean",
        "gpu_util_p10",
        "gpu_util_p90",
        "gpu_power_mean_w",
        "gpu_memory_peak_mb",
        "process_peak_rss_mb",
        "zero_train_loss_sample_fraction",
    ]
    for arm, trials in sorted(groups.items()):
        output: dict[str, Any] = {
            "arm": arm,
            "trials": len(trials),
            "ok_trials": sum(row.get("status") == "ok" for row in trials),
            "seeds": sorted({int(row["seed"]) for row in trials}),
        }
        for field in numeric_fields:
            output[field] = mean(row.get(field) for row in trials)
            output[f"{field}_sd"] = sample_sd(row.get(field) for row in trials)
        outputs.append(output)
    return outputs


def paired_comparison(
    rows: list[dict[str, Any]], reset_arm: str, carry_arm: str
) -> dict[str, Any]:
    reset = {int(row["seed"]): row for row in rows if row.get("arm") == reset_arm}
    carry = {int(row["seed"]): row for row in rows if row.get("arm") == carry_arm}
    seeds = sorted(reset.keys() & carry.keys())
    fields = [
        "paired_warm_loss",
        "paired_cold_loss",
        "carry_nll_gain",
        "carry_relative_gain",
        "valid_loss",
        "verifier_accuracy",
        "model_tokens_per_second",
    ]
    output: dict[str, Any] = {
        "reset_arm": reset_arm,
        "carry_arm": carry_arm,
        "seeds": seeds,
        "paired_trials": len(seeds),
    }
    for field in fields:
        deltas = []
        for seed in seeds:
            left = finite(reset[seed].get(field))
            right = finite(carry[seed].get(field))
            if left is not None and right is not None:
                deltas.append(right - left)
        output[f"{field}_delta"] = mean(deltas)
        output[f"{field}_delta_sd"] = sample_sd(deltas)
        if len(deltas) >= 2:
            output[f"{field}_delta_ci95"] = 1.96 * statistics.stdev(deltas) / math.sqrt(len(deltas))
        else:
            output[f"{field}_delta_ci95"] = None
    reset_throughput = mean(reset[seed].get("model_tokens_per_second") for seed in seeds)
    carry_throughput = mean(carry[seed].get("model_tokens_per_second") for seed in seeds)
    output["throughput_ratio"] = (
        carry_throughput / reset_throughput
        if reset_throughput not in (None, 0.0) and carry_throughput is not None
        else None
    )
    return output


def classify(
    rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    requested_arms: set[str],
) -> tuple[str, str, list[str], list[str]]:
    infrastructure_reasons = []
    mechanics_reasons = []
    quality_reasons = []
    present = {str(row.get("arm")) for row in rows}
    missing = sorted(requested_arms - present)
    if missing:
        infrastructure_reasons.append(f"missing requested arms: {', '.join(missing)}")
    if not CORE_ARMS <= requested_arms:
        infrastructure_reasons.append(
            "intentional partial arm screen; no state-carry promotion decision is possible"
        )
    failed = [f"{row.get('arm')}/seed{row.get('seed')}={row.get('status')}" for row in rows if row.get("status") != "ok"]
    if failed:
        infrastructure_reasons.append("failed trials: " + ", ".join(failed))
    missing_probe = [
        f"{row.get('arm')}/seed{row.get('seed')}"
        for row in rows
        if row.get("status") == "ok"
        and any(
            row.get(field) is None
            for field in (
                "stream_warm_loss",
                "paired_warm_loss",
                "paired_cold_loss",
                "carry_nll_gain",
                "rho_rms",
            )
        )
    ]
    if missing_probe:
        infrastructure_reasons.append("missing state-probe metrics: " + ", ".join(missing_probe))

    seeds_per_arm = defaultdict(set)
    for row in rows:
        seeds_per_arm[str(row.get("arm"))].add(row.get("seed"))
    max_iters = min((int(row.get("max_iters", 0)) for row in rows), default=0)
    promotion_scale = (
        CORE_ARMS <= requested_arms
        and CORE_ARMS <= present
        and all(len(seeds_per_arm[arm]) >= 3 for arm in CORE_ARMS)
        and max_iters >= 512
    )
    if not promotion_scale:
        infrastructure_reasons.append(
            "screening evidence only: promotion requires >=3 seeds per core arm and >=512 updates"
        )

    for comparison in comparisons:
        pair = f"{comparison['reset_arm']}->{comparison['carry_arm']}"
        warm_delta = finite(comparison.get("paired_warm_loss_delta"))
        valid_delta = finite(comparison.get("valid_loss_delta"))
        carry_gain_delta = finite(comparison.get("carry_nll_gain_delta"))
        carry_gain_ci95 = finite(comparison.get("carry_nll_gain_delta_ci95"))
        throughput_ratio = finite(comparison.get("throughput_ratio"))
        if warm_delta is None or warm_delta > 0.02:
            mechanics_reasons.append(
                f"{pair} fails the +0.02 paired warm-loss non-inferiority bound"
            )
        if valid_delta is None or valid_delta > 0.02:
            mechanics_reasons.append(
                f"{pair} fails the +0.02 cold-validation non-inferiority bound"
            )
        if (
            carry_gain_delta is None
            or carry_gain_ci95 is None
            or carry_gain_delta - carry_gain_ci95 <= 0.0
        ):
            mechanics_reasons.append(
                f"{pair} does not show a positive carry-gain effect with a 95% paired interval"
            )
        if throughput_ratio is None or throughput_ratio < 0.90:
            mechanics_reasons.append(
                f"{pair} retains less than 90% of reset-arm throughput"
            )

    requested_carry_arms = requested_arms & {"block512_carry", "chunk128_carry"}
    carry_rows = [
        row
        for row in rows
        if row.get("arm") in requested_carry_arms
        and row.get("status") == "ok"
    ]
    matched_verifier = mean(
        row.get("training_serialization_verifier_accuracy") for row in carry_rows
    )
    canonical_verifier = mean(row.get("canonical_verifier_accuracy") for row in carry_rows)
    if requested_carry_arms and matched_verifier is None:
        quality_reasons.append(
            "missing training-serialization verifier panel; teacher-forced loss is not a correctness result"
        )
    elif requested_carry_arms and matched_verifier <= 0.0:
        quality_reasons.append(
            "training-serialization verifier accuracy is zero on carry arms"
        )
    if requested_carry_arms and canonical_verifier is None:
        quality_reasons.append("missing canonical-transfer verifier panel")
    elif requested_carry_arms and canonical_verifier <= 0.0:
        quality_reasons.append("canonical-transfer verifier accuracy is zero on carry arms")

    reasons = infrastructure_reasons + mechanics_reasons + quality_reasons
    if missing or failed or missing_probe:
        return "invalid", "invalid", reasons, mechanics_reasons
    if not promotion_scale:
        return "screening_only", "screening_only", reasons, mechanics_reasons
    mechanics_decision = "passed" if not mechanics_reasons else "failed"
    decision = "promotable" if not reasons else "not_promoted"
    return decision, mechanics_decision, reasons, mechanics_reasons


def fmt(value: Any, digits: int = 4) -> str:
    number = finite(value)
    return "-" if number is None else f"{number:.{digits}f}"


def fmt_percent_fraction(value: Any, digits: int = 1) -> str:
    number = finite(value)
    return "-" if number is None else f"{number * 100.0:.{digits}f}"


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Stateful TBPTT Ablation",
        "",
        f"Decision: **{result['decision']}**",
        "",
        "This is a matched-data state-carry ablation. Negative carry-vs-reset loss deltas are better; positive throughput ratios should remain near one.",
        "",
        "## Aggregate Results",
        "",
        f"State-mechanics gate: **{result['mechanics_decision']}**",
        "",
        "| arm | n | trailing train loss | zero-loss samples | valid cold | paired warm | paired cold | carry NLL gain | carry rel. | rho var. | rho redund. | matched verifier | transfer verifier | wall tok/s | model tok/s | model duty % | validation % | GPU mean/p10/p90 % | power W | peak RAM MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["aggregates"]:
        lines.append(
            "| {arm} | {ok}/{trials} | {train} | {zero_loss} | {valid} | {warm} | {cold} | {gain} | {relative} | {variance} | {redundancy} | {matched_verifier} | {canonical_verifier} | {wall_throughput} | {model_throughput} | {model_duty} | {validation} | {gpu_mean}/{gpu_p10}/{gpu_p90} | {power} | {ram} |".format(
                arm=row["arm"], ok=row["ok_trials"], trials=row["trials"],
                train=fmt(row.get("train_loss")), valid=fmt(row.get("valid_loss")),
                zero_loss=fmt(row.get("zero_train_loss_sample_fraction")),
                warm=fmt(row.get("paired_warm_loss")), cold=fmt(row.get("paired_cold_loss")),
                gain=fmt(row.get("carry_nll_gain")), relative=fmt(row.get("carry_relative_gain")),
                variance=fmt(row.get("rho_variance_ratio")), redundancy=fmt(row.get("rho_redundancy")),
                matched_verifier=fmt(row.get("training_serialization_verifier_accuracy")),
                canonical_verifier=fmt(row.get("canonical_verifier_accuracy")),
                wall_throughput=fmt(row.get("wall_tokens_per_second"), 0),
                model_throughput=fmt(row.get("model_tokens_per_second"), 0),
                model_duty=fmt_percent_fraction(row.get("model_duty_fraction")),
                validation=fmt_percent_fraction(row.get("validation_fraction")),
                gpu_mean=fmt(row.get("gpu_util_mean"), 1),
                gpu_p10=fmt(row.get("gpu_util_p10"), 1),
                gpu_p90=fmt(row.get("gpu_util_p90"), 1),
                power=fmt(row.get("gpu_power_mean_w"), 1),
                ram=fmt(row.get("peak_used_mb"), 0),
            )
        )
    lines.extend(["", "## Paired Carry Effects", ""])
    if not result["comparisons"]:
        lines.append("- No paired carry comparison was requested.")
    for comparison in result["comparisons"]:
        lines.append(
            "- `{reset}` -> `{carry}` (n={n}): warm-loss delta {warm}, cold-loss delta {cold}, carry-gain delta {gain}, throughput ratio {ratio}.".format(
                reset=comparison["reset_arm"], carry=comparison["carry_arm"], n=comparison["paired_trials"],
                warm=fmt(comparison.get("paired_warm_loss_delta")), cold=fmt(comparison.get("valid_loss_delta")),
                gain=fmt(comparison.get("carry_nll_gain_delta")), ratio=fmt(comparison.get("throughput_ratio")),
            )
        )
    lines.extend(["", "## Gate Findings", ""])
    if result["reasons"]:
        lines.extend(f"- {reason}" for reason in result["reasons"])
    else:
        lines.append("- All configured mechanics, quality, and systems gates passed.")
    lines.extend([
        "",
        "GPU statistics cover the whole process. Model duty and validation fractions come from the training stage profiler and distinguish dense model work from synchronous evaluation.",
        "",
        "The matched verifier tests the exact training-document serialization. The transfer verifier tests the canonical root-task serialization. Neither is a proof of long-horizon memory; promotion additionally requires a carry-sensitive holdout and a longer convergence run.",
        "",
    ])
    return "\n".join(lines)


def analyze_matrix(root: Path) -> dict[str, Any]:
    manifests = sorted((root / "manifests").glob("*.json"))
    rows = [load_trial(path) for path in manifests]
    matrix_config_path = root / "matrix-config.json"
    if matrix_config_path.is_file():
        matrix_config = json.loads(matrix_config_path.read_text())
        requested_arms = set(matrix_config.get("requested_arms") or [])
    else:
        requested_arms = set(CORE_ARMS)
    pair_specs = [
        ("block512_reset", "block512_carry"),
        ("chunk128_reset", "chunk128_carry"),
    ]
    comparisons = [
        paired_comparison(rows, reset, carry)
        for reset, carry in pair_specs
        if {reset, carry} <= requested_arms
    ]
    decision, mechanics_decision, reasons, mechanics_reasons = classify(
        rows, comparisons, requested_arms
    )
    result = {
        "schema_version": 1,
        "root": str(root),
        "decision": decision,
        "mechanics_decision": mechanics_decision,
        "reasons": reasons,
        "mechanics_reasons": mechanics_reasons,
        "requested_arms": sorted(requested_arms),
        "trials": rows,
        "aggregates": aggregate(rows),
        "comparisons": comparisons,
    }
    (root / "stateful-tbptt-results.json").write_text(json.dumps(result, indent=2) + "\n")
    (root / "stateful-tbptt-report.md").write_text(render_report(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="matrix output directory")
    args = parser.parse_args()
    result = analyze_matrix(args.root.resolve())
    print(render_report(result))
    return 0 if result["decision"] != "invalid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
