#!/usr/bin/env python3
"""Unit tests for ruliad_promotion_matrix_analyze.py."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).with_name("ruliad_promotion_matrix_analyze.py")
SPEC = importlib.util.spec_from_file_location("ruliad_promotion_matrix_analyze", SCRIPT)
assert SPEC is not None
ANALYZE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ANALYZE)


def promotion_args() -> SimpleNamespace:
    return SimpleNamespace(
        baseline_arm="baseline",
        max_valid_ce_delta=0.15,
        max_source_difficulty_delta=0.75,
        max_verifier_regression=0.03125,
        max_schema_wrong_delta=0.10,
        max_malformed_delta=0.05,
        max_completion_regression=0.10,
        max_answer_field_regression=0.05,
        max_answer_termination_regression=0.10,
        min_completion_distinct_2=0.20,
        max_completion_period=0.70,
        max_completion_repetition=0.70,
        min_output_entropy=0.25,
        min_output_distinct_2=0.10,
        min_throughput_ratio=0.85,
        max_capability_score_drop=1.0,
        max_verifier_drop_from_best=0.125,
        max_completion_drop_from_best=0.30,
        max_recovery_control_fraction=0.50,
        max_policy_advantage_clip_fraction=0.95,
        min_raw_completion_quality=0.20,
        min_raw_completion_answer_distinct=0.20,
    )


def healthy_arm_row(arm: str) -> dict[str, float | int | str]:
    return {
        "arm": arm,
        "trials": 1,
        "ok_trials": 1,
        "healthy_trial_fraction": 1.0,
        "stage_model_tokens_per_sec_mean": 1000.0,
        "valid_teacher_ce_last_mean": 1.0,
        "source_mean_difficulty_last_mean": 5.0,
        "ruliad_verifier_last_mean": 0.25,
        "ruliad_partial_last_mean": 0.5,
        "ruliad_schema_wrong_last_mean": 0.0,
        "ruliad_malformed_last_mean": 0.0,
        "ruliad_answer_field_accuracy_last_mean": 0.8,
        "ruliad_answer_termination_rate_last_mean": 0.9,
        "completion_health_last_mean": 0.8,
        "completion_distinct_2_last_mean": 0.5,
        "completion_period_2_to_64_last_mean": 0.0,
        "completion_repetition_last_mean": 0.0,
        "raw_completion_quality_mean_mean": 0.8,
        "raw_completion_expected_answer_distinct_fraction_mean": 0.8,
        "raw_completion_actual_answer_distinct_fraction_mean": 0.8,
        "policy_completion_rows_mean": 0.0,
        "policy_advantage_clip_fraction_mean": 0.0,
        "policy_update_skipped_count_mean": 0.0,
        "output_entropy_bits_last_mean": 2.0,
        "output_distinct_2_last_mean": 0.5,
        "fatal_gate_count_mean": 0.0,
        "capability_score_drop_from_best_mean": 0.0,
        "capability_verifier_drop_from_best_mean": 0.0,
        "capability_completion_drop_from_best_mean": 0.0,
        "recovery_control_fraction_mean": 0.0,
    }


class RawCompletionSampleTests(unittest.TestCase):
    def test_latest_base_probe_records_are_aggregated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            events = run_dir / "events"
            events.mkdir()
            path = events / "ruliad_completion_samples.jsonl"
            rows = [
                {
                    "epoch": 1,
                    "absolute_step": 0,
                    "probe_name": "ruliad_correctness",
                    "status": "Malformed",
                    "verifier_match": False,
                    "semantic_match": False,
                    "partial_credit": False,
                    "correct_field_count": 0,
                    "expected_field_count": 2,
                    "answer_terminated": False,
                    "completion_quality_ppm": 0,
                    "generated_token_count": 32,
                    "hash_canary": False,
                    "expected_answer": "old",
                    "actual_answer": "old-loop",
                },
                {
                    "epoch": 2,
                    "absolute_step": 10,
                    "probe_name": "ruliad_correctness_eval_steps_2",
                    "status": "VerifierMatch",
                    "verifier_match": True,
                    "semantic_match": True,
                    "partial_credit": True,
                    "correct_field_count": 2,
                    "expected_field_count": 2,
                    "answer_terminated": True,
                    "completion_quality_ppm": 1_000_000,
                    "generated_token_count": 4,
                    "hash_canary": False,
                    "expected_answer": "eval",
                    "actual_answer": "eval-sweep",
                },
                {
                    "epoch": 2,
                    "absolute_step": 10,
                    "probe_name": "ruliad_correctness",
                    "status": "VerifierMatch",
                    "verifier_match": True,
                    "semantic_match": True,
                    "partial_credit": True,
                    "correct_field_count": 2,
                    "expected_field_count": 2,
                    "answer_terminated": True,
                    "completion_quality_ppm": 1_000_000,
                    "generated_token_count": 3,
                    "hash_canary": False,
                    "expected_answer": "ok=1;l=2;r=2",
                    "actual_answer": "ok=1;l=2;r=2",
                },
                {
                    "epoch": 2,
                    "absolute_step": 10,
                    "probe_name": "ruliad_correctness",
                    "status": "SchemaValidWrong",
                    "verifier_match": False,
                    "semantic_match": False,
                    "partial_credit": True,
                    "correct_field_count": 1,
                    "expected_field_count": 2,
                    "answer_terminated": False,
                    "completion_quality_ppm": 500_000,
                    "generated_token_count": 5,
                    "hash_canary": False,
                    "expected_answer": "ok=1;l=3;r=3",
                    "actual_answer": "ok=0;l=2;r=9",
                },
            ]
            with path.open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            summary = ANALYZE.read_raw_completion_samples(str(run_dir))

        self.assertEqual(summary["raw_completion_rows"], 2)
        self.assertEqual(summary["raw_completion_verifier_rate"], 0.5)
        self.assertEqual(summary["raw_completion_semantic_rate"], 0.5)
        self.assertEqual(summary["raw_completion_partial_rate"], 1.0)
        self.assertEqual(summary["raw_completion_schema_wrong_rate"], 0.5)
        self.assertEqual(summary["raw_completion_malformed_rate"], 0.0)
        self.assertEqual(summary["raw_completion_field_accuracy_mean"], 0.75)
        self.assertEqual(summary["raw_completion_termination_rate"], 0.5)
        self.assertEqual(summary["raw_completion_quality_mean"], 0.75)
        self.assertEqual(summary["raw_completion_generated_tokens_mean"], 4.0)
        self.assertEqual(summary["raw_completion_expected_answer_distinct_fraction"], 1.0)
        self.assertEqual(summary["raw_completion_actual_answer_distinct_fraction"], 1.0)
        self.assertEqual(summary["raw_completion_status_entropy_bits"], 1.0)
        self.assertEqual(summary["raw_completion_dominant_status_fraction"], 0.5)

    def test_missing_raw_completion_file_returns_empty_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = ANALYZE.read_raw_completion_samples(tmp)

        self.assertEqual(set(summary), set(ANALYZE.RAW_COMPLETION_METRIC_COLUMNS))
        self.assertTrue(all(value is None for value in summary.values()))


class PromotionGateTests(unittest.TestCase):
    def test_capability_drop_rejects_candidate(self) -> None:
        args = promotion_args()
        rows = [
            healthy_arm_row("baseline"),
            {
                **healthy_arm_row("candidate"),
                "capability_score_drop_from_best_mean": 1.5,
                "capability_verifier_drop_from_best_mean": 0.2,
            },
        ]

        gated = ANALYZE.add_gate_decisions(rows, args)
        candidate = next(row for row in gated if row["arm"] == "candidate")

        self.assertEqual(candidate["decision"], "reject")
        self.assertIn("capability_score_drop", candidate["fail_reasons"])
        self.assertIn("verifier_drop_from_best", candidate["fail_reasons"])

    def test_unhealthy_control_keeps_reasons_and_marks_matrix_unvalidated(self) -> None:
        args = promotion_args()
        baseline = {
            **healthy_arm_row("baseline"),
            "healthy_trial_fraction": 0.0,
            "recovery_control_fraction_mean": 1.0,
        }
        candidate = {
            **healthy_arm_row("candidate"),
            "ruliad_verifier_last_mean": 0.0,
            "completion_health_last_mean": 0.1,
            "raw_completion_quality_mean_mean": 0.1,
        }

        gated = ANALYZE.add_gate_decisions([baseline, candidate], args)
        control = next(row for row in gated if row["arm"] == "baseline")
        summary = ANALYZE.validation_summary(gated, "baseline")

        self.assertEqual(control["decision"], "control")
        self.assertIn("unhealthy_trials", control["fail_reasons"])
        self.assertIn("recovery_thrash", control["fail_reasons"])
        self.assertEqual(summary["status"], "no_validated_candidate")
        self.assertEqual(summary["unhealthy_control_arms"], ["baseline"])

    def test_validation_summary_reports_promoted_healthy_candidate(self) -> None:
        args = promotion_args()
        candidate = {
            **healthy_arm_row("candidate"),
            "ruliad_verifier_last_mean": 0.35,
            "ruliad_partial_last_mean": 0.6,
            "completion_health_last_mean": 0.9,
        }

        gated = ANALYZE.add_gate_decisions([healthy_arm_row("baseline"), candidate], args)
        summary = ANALYZE.validation_summary(gated, "baseline")

        self.assertEqual(summary["status"], "validated_candidate")
        self.assertEqual(summary["promoted_arms"], ["candidate"])

    def test_low_raw_answer_diversity_requires_diverse_expected_answers(self) -> None:
        args = promotion_args()
        candidate = {
            **healthy_arm_row("candidate"),
            "raw_completion_expected_answer_distinct_fraction_mean": 0.05,
            "raw_completion_actual_answer_distinct_fraction_mean": 0.05,
        }

        gated = ANALYZE.add_gate_decisions([healthy_arm_row("baseline"), candidate], args)
        candidate_row = next(row for row in gated if row["arm"] == "candidate")

        self.assertNotIn("raw_completion_answer_collapse", candidate_row["fail_reasons"])


if __name__ == "__main__":
    unittest.main()
