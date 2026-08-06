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
import re
import statistics
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SUMMARY_COLUMNS = [
    "run",
    "iters",
    "arm",
    "seed",
    "wall_s",
    "tok_s",
    "train_first",
    "train_last",
    "valid_last",
    "valid_mean",
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
    "source_mean_difficulty",
    "source_norm_difficulty",
    "verifier_failures",
    "ruliad_verifier_accuracy",
    "ruliad_partial_progress",
    "capability_allowed_max_difficulty",
    "output_entropy_bits",
    "output_distinct_2_fraction",
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
    "training_wall_seconds",
    "wall_tokens_per_second",
    "model_tokens_per_second",
    "model_duty_fraction",
    "train_compute_fraction",
    "optimizer_fraction",
    "dataloader_foreground_wait_fraction",
    "host_sync_points",
    "train_loss_first",
    "train_loss_last",
    "valid_loss_last",
    "valid_loss_mean",
    "stream_warm_loss_mean",
    "validation_objective_loss_last",
    "stream_paired_warm_loss_last",
    "stream_paired_cold_loss_last",
    "stream_carry_nll_gain_last",
    "stream_carry_relative_gain_last",
    "source_loss_last",
    "source_entropy_bits_last",
    "source_mean_difficulty_last",
    "source_norm_difficulty_last",
    "source_mastered_probability_last",
    "source_capability_allowed_max_difficulty_last",
    "source_verifier_failures_last",
    "ruliad_verifier_accuracy_last",
    "ruliad_partial_progress_last",
    "output_entropy_bits_last",
    "output_mean_max_probability_last",
    "output_distinct_2_fraction_last",
    "output_repetition_fraction_last",
    "output_period_2_to_64_fraction_last",
    "gate_count",
    "fatal_gate_count",
    "capacity_scale_count",
    "pc_event_count",
    "pc_ms_mean",
    "pc_learning_contract_last",
    "pc_global_autodiff_graph_last",
    "pc_global_backward_calls_total",
    "pc_local_vjp_calls_total",
    "pc_factors_last",
    "pc_gradient_tensors_last",
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
    "seed",
    "batch_size",
    "local_learning_rate",
    "tbptt_chunk_size",
    "tbptt_persist_across_steps",
    "sequence_batching",
    "sequence_state_probe",
    "sequence_state_probe_paired_batches",
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
    normalized["iters"] = as_int(row.get("iters")) or infer_iters_from_run(normalized["run"])
    normalized["arm"] = row.get("arm", "")
    normalized["seed"] = as_int(row.get("seed"))
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


def read_summary_csvs(paths: Iterable[Path]) -> list[dict[str, Any]]:
    keyed: dict[tuple[Any, ...], dict[str, Any]] = {}
    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                normalized = normalize_summary_row(row)
                key = (
                    normalized.get("run"),
                    normalized.get("iters"),
                    normalized.get("arm"),
                    normalized.get("seed"),
                )
                existing = keyed.get(key)
                if existing is None or populated_count(normalized) > populated_count(existing):
                    keyed[key] = normalized
    return sorted(
        keyed.values(),
        key=lambda row: (
            row.get("iters") or -1,
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
    if split == "train" and name in {"Loss", "Stream Warm Loss"}:
        if summary.get("train_loss_first", "") == "":
            summary["train_loss_first"] = value
        summary["train_loss_last"] = value
    elif split == "valid" and name == "Loss":
        summary["valid_loss_last"] = value
        summary["valid_loss_mean"] = running_value if running_value is not None else value
    elif split == "valid" and name == "Random Cold Loss":
        summary["valid_loss_mean"] = value
    elif split == "valid" and name == "Stream Warm Loss":
        summary["stream_warm_loss_mean"] = (
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


def update_source(summary: dict[str, Any], event: dict[str, Any]) -> None:
    fields = {
        "source_loss_last": "loss",
        "source_entropy_bits_last": "entropy_bits",
        "source_mean_difficulty_last": "mean_difficulty_level",
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
    summary["pc_event_count"] = 0
    summary["pc_global_backward_calls_total"] = 0
    summary["pc_local_vjp_calls_total"] = 0
    summary["_pc_ms_values"] = []
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


def collect_event_summaries(
    paths: Iterable[Path], manifests: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: dict[str, dict[str, Any]] = {}
    latest_source_by_run: dict[str, dict[str, Any]] = {}

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
                summary["validation_objective_loss_last"] = as_float(event.get("loss"))
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
            elif event_type == "model_scale_applied":
                summary["capacity_scale_count"] += 1
            elif event_type == "predictive_coding":
                summary["pc_event_count"] += 1
                summary["pc_learning_contract_last"] = event.get(
                    "learning_contract", ""
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
                summary["pc_factors_last"] = event.get("factors", "")
                summary["pc_gradient_tensors_last"] = event.get(
                    "gradient_tensors", ""
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
            ):
                summary[key] = manifest.get(key, "")
            profile = read_stage_profile(manifest.get("log_path"))
            if "total_ns" in profile:
                summary["training_wall_seconds"] = profile["total_ns"] / 1_000_000_000.0
            for key in (
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
    return sorted(rows, key=lambda row: row["run"]), bucket_rows, coverage_rows


def normalize_event_summaries(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    for event in rows:
        if not event.get("arm"):
            continue
        row = {key: "" for key in SUMMARY_COLUMNS}
        row.update(
            {
                "run": event.get("run", ""),
                "iters": as_int(event.get("iters")),
                "arm": event.get("arm", ""),
                "seed": as_int(event.get("seed")),
                "wall_s": as_float(event.get("training_wall_seconds"))
                or as_float(event.get("elapsed_seconds")),
                "tok_s": as_float(event.get("wall_tokens_per_second")),
                "train_first": as_float(event.get("train_loss_first")),
                "train_last": as_float(event.get("train_loss_last")),
                "valid_last": as_float(event.get("valid_loss_last")),
                "valid_mean": as_float(event.get("valid_loss_mean")),
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
                "source_mean_difficulty": as_float(
                    event.get("source_mean_difficulty_last")
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
                "ruliad_partial_progress": as_float(
                    event.get("ruliad_partial_progress_last")
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
        normalized_rows.append(row)
    return normalized_rows


def merge_summary_rows(
    legacy_rows: Iterable[dict[str, Any]], event_rows: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged: dict[tuple[Any, Any, Any, Any], dict[str, Any]] = {}
    for row in [*legacy_rows, *event_rows]:
        key = (row.get("iters"), row.get("arm"), row.get("seed"), row.get("run"))
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
            row.get("iters") or -1,
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
        str(Path(str(manifest.get("gpu_path")))): manifest
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
        manifest = manifest_by_gpu_path.get(str(path))
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
    groups: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("iters"), row.get("arm"))].append(row)
    out: list[dict[str, Any]] = []
    metrics = [
        "wall_s",
        "tok_s",
        "train_last",
        "valid_last",
        "valid_mean",
        "stream_warm_loss",
        "validation_objective_loss",
        "stream_paired_warm_loss",
        "stream_paired_cold_loss",
        "stream_carry_nll_gain",
        "stream_carry_relative_gain",
        "source_loss",
        "source_mean_difficulty",
        "source_norm_difficulty",
        "ruliad_verifier_accuracy",
        "ruliad_partial_progress",
        "capability_allowed_max_difficulty",
        "output_entropy_bits",
        "output_distinct_2_fraction",
        "pc_ms_mean",
    ]
    for (iters, arm), group in sorted(groups.items(), key=lambda item: (item[0][0] or -1, str(item[0][1]))):
        row: dict[str, Any] = {
            "iters": iters,
            "arm": arm,
            "runs": len(group),
            "seeds": len({g.get("seed") for g in group}),
        }
        for metric in metrics:
            metric_stats = stats(as_float(g.get(metric)) for g in group)
            row[f"{metric}_mean"] = metric_stats.mean
            row[f"{metric}_ci95"] = metric_stats.ci95
        out.append(row)
    return out


def paired_deltas(rows: list[dict[str, Any]], baseline: str, compare: str) -> list[dict[str, Any]]:
    by_key: dict[tuple[Any, Any], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_key[(row.get("iters"), row.get("seed"))][row.get("arm")] = row

    metrics = [
        "valid_mean",
        "stream_warm_loss",
        "validation_objective_loss",
        "stream_paired_warm_loss",
        "stream_paired_cold_loss",
        "stream_carry_nll_gain",
        "stream_carry_relative_gain",
        "source_loss",
        "train_last",
        "wall_s",
        "tok_s",
        "ruliad_verifier_accuracy",
        "ruliad_partial_progress",
        "capability_allowed_max_difficulty",
        "output_entropy_bits",
        "output_distinct_2_fraction",
    ]
    deltas: dict[tuple[Any, str], list[float]] = defaultdict(list)
    for (iters, _seed), arms in by_key.items():
        if baseline not in arms or compare not in arms:
            continue
        for metric in metrics:
            base = as_float(arms[baseline].get(metric))
            comp = as_float(arms[compare].get(metric))
            if base is None or comp is None:
                continue
            deltas[(iters, metric)].append(comp - base)

    rows_out: list[dict[str, Any]] = []
    for (iters, metric), values in sorted(deltas.items(), key=lambda item: (item[0][0] or -1, item[0][1])):
        metric_stats = stats(values)
        rows_out.append(
            {
                "iters": iters,
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

    lines.append("## Fixed-Token Summary")
    lines.append("")
    lines.append(
        "| Iters | Arm | Runs | Seeds | Cold valid | Stream warm | Validation objective | Verifier acc | Partial progress | Tok/s | PC ms |"
    )
    lines.append(
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for row in summary_rows:
        lines.append(
            "| {iters} | {arm} | {runs} | {seeds} | {valid} | {warm} | {objective} | {verifier} | {partial} | {tok} | {pc} |".format(
                iters=row.get("iters", ""),
                arm=row.get("arm", ""),
                runs=row.get("runs", ""),
                seeds=row.get("seeds", ""),
                valid=fmt_mean_ci(row, "valid_mean"),
                warm=fmt_mean_ci(row, "stream_warm_loss"),
                objective=fmt_mean_ci(row, "validation_objective_loss"),
                verifier=fmt_mean_ci(row, "ruliad_verifier_accuracy"),
                partial=fmt_mean_ci(row, "ruliad_partial_progress"),
                tok=fmt_mean_ci(row, "tok_s"),
                pc=fmt_mean_ci(row, "pc_ms_mean"),
            )
        )
    lines.append("")

    lines.append("## Paired Deltas")
    lines.append("")
    lines.append("| Iters | Comparison | Metric | Pairs | Delta |")
    lines.append("| ---: | --- | --- | ---: | ---: |")
    for row in paired_rows:
        lines.append(
            "| {iters} | {comparison} | {metric} | {pairs} | {delta} |".format(
                iters=row.get("iters", ""),
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
                "| Run | Contract | Global graph | Global backwards | Local VJPs | Factors | Gradient tensors |"
            )
            lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
            for row in pc_rows[:40]:
                lines.append(
                    "| {run} | {contract} | {graph} | {backwards} | {vjps} | {factors} | {grads} |".format(
                        run=row.get("run", ""),
                        contract=row.get("pc_learning_contract_last", ""),
                        graph=row.get("pc_global_autodiff_graph_last", ""),
                        backwards=row.get("pc_global_backward_calls_total", ""),
                        vjps=row.get("pc_local_vjp_calls_total", ""),
                        factors=row.get("pc_factors_last", ""),
                        grads=row.get("pc_gradient_tensors_last", ""),
                    )
                )
            lines.append("")
        lines.append(
            "| Run | Cold valid | Stream warm | Objective | Carry gain | Source difficulty | Verifier acc | Output entropy | Gates | Fatal gates |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for row in event_rows[:40]:
            lines.append(
                "| {run} | {valid} | {warm} | {objective} | {carry} | {difficulty} | {verifier} | {entropy} | {gates} | {fatal} |".format(
                    run=row.get("run", ""),
                    valid=fmt_scalar(row.get("valid_loss_mean")),
                    warm=fmt_scalar(row.get("stream_warm_loss_mean")),
                    objective=fmt_scalar(row.get("validation_objective_loss_last")),
                    carry=fmt_scalar(row.get("stream_carry_nll_gain_last")),
                    difficulty=fmt_scalar(row.get("source_mean_difficulty_last")),
                    verifier=fmt_scalar(row.get("ruliad_verifier_accuracy_last")),
                    entropy=fmt_scalar(row.get("output_entropy_bits_last")),
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
    event_rows, bucket_rows, coverage_rows = collect_event_summaries(
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
        ["iters", "arm", "runs", "seeds"]
        + [
            f"{metric}_{suffix}"
            for metric in [
                "wall_s",
                "tok_s",
                "train_last",
                "valid_last",
                "valid_mean",
                "stream_warm_loss",
                "validation_objective_loss",
                "stream_paired_warm_loss",
                "stream_paired_cold_loss",
                "stream_carry_nll_gain",
                "stream_carry_relative_gain",
                "source_loss",
                "source_mean_difficulty",
                "source_norm_difficulty",
                "ruliad_verifier_accuracy",
                "ruliad_partial_progress",
                "capability_allowed_max_difficulty",
                "output_entropy_bits",
                "output_distinct_2_fraction",
                "pc_ms_mean",
            ]
            for suffix in ["mean", "ci95"]
        ],
        grouped_rows,
    )
    write_csv(
        out_dir / "paired_deltas.csv",
        ["iters", "comparison", "metric", "pairs", "delta_mean", "delta_ci95"],
        paired_rows,
    )
    write_csv(out_dir / "event_run_summary.csv", EVENT_SUMMARY_COLUMNS, event_rows)
    write_csv(out_dir / "source_bucket_summary.csv", BUCKET_COLUMNS, bucket_rows)
    write_csv(
        out_dir / "source_capability_coverage.csv",
        CAPABILITY_COVERAGE_COLUMNS,
        coverage_rows,
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
                    "split": "valid",
                    "name": "Stream Warm Loss",
                    "value": 0.6,
                    "running_value": 0.55,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "validation_finished",
                    "run_id": "run-a",
                    "loss": 0.55,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "predictive_coding",
                    "run_id": "run-a",
                    "learning_contract": "local_factor_vjp_v1",
                    "global_autodiff_graph": False,
                    "global_backward_calls": 0,
                    "local_vjp_calls": 12,
                    "factors": 4,
                    "gradient_tensors": 9,
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
                    "name": "Ruliad Mean Partial Progress",
                    "value": 0.5,
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
        assert event_rows[0]["valid_loss_mean"] == "0.45"
        assert event_rows[0]["stream_warm_loss_mean"] == "0.55"
        assert event_rows[0]["validation_objective_loss_last"] == "0.55"
        assert event_rows[0]["source_loss_last"] == "0.6"
        assert event_rows[0]["source_capability_allowed_max_difficulty_last"] == "5.0"
        assert event_rows[0]["pc_learning_contract_last"] == "local_factor_vjp_v1"
        assert event_rows[0]["pc_global_autodiff_graph_last"] == "False"
        assert event_rows[0]["pc_global_backward_calls_total"] == "0"
        assert event_rows[0]["pc_local_vjp_calls_total"] == "12"
        normalized = list(csv.DictReader((out / "normalized_summary.csv").open()))
        event_normalized = next(row for row in normalized if row["run"] == "run-a")
        assert event_normalized["tok_s"] == "512.0"
        assert event_normalized["ruliad_verifier_accuracy"] == "0.25"
        assert event_normalized["ruliad_partial_progress"] == "0.5"
        coverage = list(csv.DictReader((out / "source_capability_coverage.csv").open()))
        assert coverage[0]["candidate_coverage"] == "0.75"
        gpu = list(csv.DictReader((out / "gpu_summary.csv").open()))
        assert gpu[0]["arm"] == "adamwpc"
        assert gpu[0]["util_mean"] == "90.0"
        three_sample_stats = stats([1.0, 2.0, 3.0])
        assert math.isclose(three_sample_stats.ci95, 4.303 / math.sqrt(3.0))
        markdown = (out / "paper_tables.md").read_text()
        assert "Fixed-Token Summary" in markdown
        assert "Local Learning Contract" in markdown
        print("self-test ok")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    run_analysis(args.inputs, Path(args.out_dir), args.baseline, args.compare)


if __name__ == "__main__":
    main()
