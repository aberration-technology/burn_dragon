#!/usr/bin/env python3
"""Unit tests for latent_reasoning_steps_analyze.py."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("latent_reasoning_steps_analyze.py")
SPEC = importlib.util.spec_from_file_location("latent_reasoning_steps_analyze", SCRIPT)
assert SPEC is not None
ANALYZE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ANALYZE)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class ContinualLearningTrajectoryTests(unittest.TestCase):
    def test_summarize_manifest_reports_capability_drop_and_recovery_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            events_dir = run_dir / "events"
            events_dir.mkdir(parents=True)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "trial_key": "synthetic",
                        "max_steps": 1,
                        "seed": 7,
                        "status": "ok",
                        "elapsed_seconds": 10,
                        "max_iters": 10,
                        "batch_size": 2,
                        "block_size": 4,
                        "peak_used_mb": 128,
                        "run_dir": str(run_dir),
                    }
                )
            )
            write_jsonl(
                events_dir / "training_events.jsonl",
                [
                    {
                        "type": "metric",
                        "split": "train",
                        "name": "Loss",
                        "value": 2.0,
                    },
                    {
                        "type": "metric",
                        "split": "train",
                        "name": "Loss",
                        "value": 1.0,
                    },
                    {
                        "type": "metric",
                        "split": "valid",
                        "name": "Ruliad Capability Gate Passed",
                        "value": 1.0,
                        "epoch": 1,
                        "absolute_step": 8,
                    },
                    {
                        "type": "metric",
                        "split": "valid",
                        "name": "Ruliad Capability Gate Failure Count",
                        "value": 0.0,
                    },
                    {
                        "type": "dynamics_control",
                        "mode": "stable",
                    },
                    {
                        "type": "dynamics_control",
                        "mode": "plasticity_recovery",
                    },
                    {
                        "type": "gate",
                        "gate": "continual_learning_capability_quality_recovery",
                    },
                    {
                        "type": "gate",
                        "gate": "ruliad_capability_gate_failed",
                    },
                ],
            )
            write_jsonl(
                events_dir / "capability_probe.jsonl",
                [
                    {
                        "probe_name": "ruliad_correctness",
                        "epoch": 1,
                        "absolute_step": 0,
                        "verifier_rate": 0.0,
                        "semantic_rate": 0.0,
                        "partial_credit_rate": 0.0,
                        "completion_health_rate": 0.2,
                        "answer_field_accuracy": 0.0,
                        "answer_termination_rate": 0.0,
                    },
                    {
                        "probe_name": "ruliad_correctness",
                        "epoch": 2,
                        "absolute_step": 10,
                        "verifier_rate": 0.5,
                        "semantic_rate": 0.5,
                        "partial_credit_rate": 0.5,
                        "completion_health_rate": 0.9,
                        "answer_field_accuracy": 0.8,
                        "answer_termination_rate": 0.9,
                    },
                    {
                        "probe_name": "ruliad_correctness",
                        "epoch": 3,
                        "absolute_step": 20,
                        "verifier_rate": 0.1,
                        "semantic_rate": 0.1,
                        "partial_credit_rate": 0.1,
                        "completion_health_rate": 0.4,
                        "answer_field_accuracy": 0.4,
                        "answer_termination_rate": 0.5,
                    },
                ],
            )

            row = ANALYZE.summarize_manifest(manifest_path)

        self.assertEqual(row["dynamics_control_count"], 2)
        self.assertEqual(row["recovery_control_count"], 1)
        self.assertEqual(row["recovery_control_fraction"], 0.5)
        self.assertEqual(row["capability_quality_recovery_count"], 1)
        self.assertEqual(row["capability_gate_failed_count"], 1)
        self.assertAlmostEqual(row["capability_verifier_best"], 0.5)
        self.assertAlmostEqual(row["capability_verifier_drop_from_best"], 0.4)
        self.assertAlmostEqual(row["capability_completion_best"], 0.9)
        self.assertAlmostEqual(row["capability_completion_drop_from_best"], 0.5)
        self.assertGreater(row["capability_score_drop_from_best"], 1.0)
        self.assertFalse(row["healthy"])


if __name__ == "__main__":
    unittest.main()
