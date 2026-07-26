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
                        "type": "metric",
                        "split": "valid",
                        "name": "Latent Eval Steps 1 Step 1 CE Delta",
                        "value": 0.25,
                    },
                    {
                        "type": "metric",
                        "split": "valid",
                        "name": "Latent Eval Steps 1 Step 1 CE Monotonic Violation Rate",
                        "value": 0.5,
                    },
                    {
                        "type": "metric",
                        "split": "valid",
                        "name": "Latent Eval Steps 1 Step 1 Entropy Bits",
                        "value": 0.125,
                    },
                    {
                        "type": "metric",
                        "split": "valid",
                        "name": "Latent Eval Steps 1 Step 1 Delta RMS",
                        "value": 17.0,
                    },
                    {
                        "type": "metric",
                        "split": "valid",
                        "name": "Latent Eval Steps 4 Step 2 CE Delta",
                        "value": 8.0,
                    },
                    {
                        "type": "metric",
                        "split": "valid",
                        "name": "Latent Eval Steps 4 Step 2 CE Monotonic Violation Rate",
                        "value": 1.0,
                    },
                    {
                        "type": "metric",
                        "split": "valid",
                        "name": "Latent Eval Steps 4 Step 2 Entropy Bits",
                        "value": 0.01,
                    },
                    {
                        "type": "metric",
                        "split": "valid",
                        "name": "Latent Eval Steps 4 Step 2 Delta RMS",
                        "value": 64.0,
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
                        "type": "dynamics_control",
                        "mode": "source_capability_recovery",
                        "reason": "source-selection capability recovery: feedback_p=1.000 lagging_p=1.000",
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
                        "answer_field_coverage": 0.0,
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
                        "answer_field_coverage": 0.85,
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
                        "answer_field_coverage": 0.6,
                        "answer_termination_rate": 0.5,
                        "group_buckets": [
                            {
                                "label": "family:category",
                                "item_count": 8,
                                "verifier_rate": 0.0,
                                "partial_credit_rate": 0.0,
                                "completion_health_rate": 0.0,
                                "schema_valid_wrong_rate": 1.0,
                                "malformed_rate": 0.0,
                                "missing_rate": 0.0,
                                "answer_field_coverage": 0.0,
                            },
                            {
                                "label": "family:rewrite",
                                "item_count": 8,
                                "verifier_rate": 0.0,
                                "partial_credit_rate": 0.0,
                                "completion_health_rate": 0.0,
                                "schema_valid_wrong_rate": 0.0,
                                "malformed_rate": 1.0,
                                "missing_rate": 0.0,
                                "answer_field_coverage": 0.0,
                            },
                            {
                                "label": "contract:ok,l,r",
                                "item_count": 8,
                                "verifier_rate": 0.0,
                                "partial_credit_rate": 0.0,
                                "completion_health_rate": 0.0,
                                "schema_valid_wrong_rate": 1.0,
                                "malformed_rate": 0.0,
                                "missing_rate": 0.0,
                                "answer_field_coverage": 0.0,
                            },
                        ],
                    },
                    {
                        "probe_name": "ruliad_correctness_contract",
                        "epoch": 3,
                        "absolute_step": 20,
                        "verifier_rate": 0.3,
                        "semantic_rate": 0.3,
                        "partial_credit_rate": 0.6,
                        "schema_valid_wrong_rate": 0.2,
                        "malformed_rate": 0.1,
                        "completion_health_rate": 0.7,
                        "answer_field_accuracy": 0.5,
                        "answer_field_coverage": 0.8,
                        "answer_termination_rate": 0.9,
                        "actual_answer_distinct_fraction": 0.4,
                    },
                    {
                        "probe_name": "ruliad_correctness_eval_steps_2",
                        "epoch": 3,
                        "absolute_step": 20,
                        "verifier_rate": 0.0,
                        "semantic_rate": 0.0,
                        "partial_credit_rate": 0.2,
                        "schema_valid_wrong_rate": 0.7,
                        "malformed_rate": 0.4,
                        "completion_health_rate": 0.1,
                        "completion_max_period_2_to_64_fraction": 0.9,
                    },
                    {
                        "probe_name": "ruliad_correctness_eval_steps_4",
                        "epoch": 3,
                        "absolute_step": 20,
                        "verifier_rate": 0.2,
                        "semantic_rate": 0.2,
                        "partial_credit_rate": 0.4,
                        "schema_valid_wrong_rate": 0.3,
                        "malformed_rate": 0.2,
                        "completion_health_rate": 0.5,
                        "completion_max_period_2_to_64_fraction": 0.4,
                    },
                ],
            )
            write_jsonl(
                events_dir / "source_selection.jsonl",
                [
                    {
                        "absolute_step": 20,
                        "entropy_bits": 2.8,
                        "active_candidate_count": 9,
                        "active_max_entropy_bits": 3.169925,
                        "normalized_entropy": 0.883,
                        "mean_difficulty_level": 0.0,
                        "max_difficulty_level": 0,
                        "materialized_frontier_edge": 0,
                        "max_difficulty_probability": 1.0,
                        "normalized_difficulty_score": 0.0,
                        "target_difficulty_score": 0.5,
                        "hash_noise_probability": 0.0,
                        "capability_lagging_probability": 0.75,
                        "difficulty_buckets": [
                            {
                                "label": "d0",
                                "probability": 0.0,
                                "mastered_probability": 0.0,
                            },
                            {
                                "label": "d1",
                                "probability": 0.25,
                                "mastered_probability": 0.25,
                            },
                            {
                                "label": "d2",
                                "probability": 0.75,
                                "mastered_probability": 0.0,
                            },
                        ],
                    }
                ],
            )

            row = ANALYZE.summarize_manifest(manifest_path)

        self.assertEqual(row["dynamics_control_count"], 3)
        self.assertEqual(row["recovery_control_count"], 1)
        self.assertAlmostEqual(row["recovery_control_fraction"], 1 / 3)
        self.assertEqual(row["source_capability_recovery_control_count"], 1)
        self.assertAlmostEqual(row["source_capability_recovery_control_fraction"], 1 / 3)
        self.assertEqual(row["capability_quality_recovery_count"], 1)
        self.assertEqual(row["capability_gate_failed_count"], 1)
        self.assertAlmostEqual(row["capability_verifier_best"], 0.5)
        self.assertAlmostEqual(row["capability_verifier_drop_from_best"], 0.4)
        self.assertAlmostEqual(row["capability_completion_best"], 0.9)
        self.assertAlmostEqual(row["capability_completion_drop_from_best"], 0.5)
        self.assertAlmostEqual(row["ruliad_answer_field_coverage_last"], 0.6)
        self.assertAlmostEqual(row["source_entropy_bits_last"], 2.8)
        self.assertEqual(row["source_active_candidate_count_last"], 9)
        self.assertAlmostEqual(row["source_active_max_entropy_bits_last"], 3.169925)
        self.assertAlmostEqual(row["source_normalized_entropy_last"], 0.883)
        self.assertEqual(row["source_active_max_difficulty_last"], 2)
        self.assertEqual(row["source_mastered_max_difficulty_last"], 1)
        self.assertAlmostEqual(row["source_capability_lagging_probability_last"], 0.75)
        self.assertAlmostEqual(row["latent_eval_final_ce_delta_last"], 0.25)
        self.assertAlmostEqual(row["latent_eval_final_ce_violation_last"], 0.5)
        self.assertAlmostEqual(row["latent_eval_final_entropy_bits_last"], 0.125)
        self.assertAlmostEqual(row["latent_eval_final_delta_rms_last"], 17.0)
        self.assertAlmostEqual(row["latent_extra_eval_max_ce_delta_last"], 8.0)
        self.assertAlmostEqual(row["latent_extra_eval_max_ce_violation_last"], 1.0)
        self.assertAlmostEqual(row["latent_extra_eval_min_entropy_bits_last"], 0.01)
        self.assertAlmostEqual(row["latent_extra_eval_max_delta_rms_last"], 64.0)
        self.assertAlmostEqual(row["contract_probe_verifier_last"], 0.3)
        self.assertAlmostEqual(row["contract_probe_answer_field_accuracy_last"], 0.5)
        self.assertAlmostEqual(row["contract_probe_completion_health_last"], 0.7)
        self.assertAlmostEqual(row["contract_probe_verifier_delta"], 0.2)
        self.assertAlmostEqual(row["contract_probe_answer_field_delta"], 0.1)
        self.assertAlmostEqual(row["contract_probe_completion_delta"], 0.3)
        self.assertAlmostEqual(row["contract_probe_answer_distinct_delta"], 0.4)
        self.assertEqual(row["capability_bucket_lagging_count"], 2)
        self.assertEqual(row["capability_contract_lagging_count"], 1)
        self.assertEqual(row["extra_eval_step_count"], 2)
        self.assertEqual(row["extra_eval_worst_steps"], 2)
        self.assertAlmostEqual(row["extra_eval_min_verifier_delta"], -0.1)
        self.assertAlmostEqual(row["extra_eval_min_completion_delta"], -0.3)
        self.assertAlmostEqual(row["extra_eval_max_malformed_delta"], 0.4)
        self.assertAlmostEqual(row["extra_eval_max_period_2_to_64"], 0.9)
        self.assertGreater(row["capability_score_drop_from_best"], 1.0)
        self.assertFalse(row["healthy"])

    def test_schema_valid_wrong_bucket_counts_as_lagging_not_malformed(self) -> None:
        self.assertTrue(
            ANALYZE.capability_bucket_lagging(
                verifier=0.0,
                completion=0.0,
                schema=1.0,
                malformed=0.0,
                missing=0.0,
            )
        )
        self.assertFalse(
            ANALYZE.capability_bucket_lagging(
                verifier=0.0,
                completion=0.0,
                schema=0.0,
                malformed=1.0,
                missing=0.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
