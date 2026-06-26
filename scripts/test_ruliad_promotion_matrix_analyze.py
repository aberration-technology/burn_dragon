#!/usr/bin/env python3
"""Unit tests for ruliad_promotion_matrix_analyze.py."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("ruliad_promotion_matrix_analyze.py")
SPEC = importlib.util.spec_from_file_location("ruliad_promotion_matrix_analyze", SCRIPT)
assert SPEC is not None
ANALYZE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ANALYZE)


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
        self.assertEqual(summary["raw_completion_actual_answer_distinct_fraction"], 1.0)
        self.assertEqual(summary["raw_completion_status_entropy_bits"], 1.0)
        self.assertEqual(summary["raw_completion_dominant_status_fraction"], 0.5)

    def test_missing_raw_completion_file_returns_empty_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = ANALYZE.read_raw_completion_samples(tmp)

        self.assertEqual(set(summary), set(ANALYZE.RAW_COMPLETION_METRIC_COLUMNS))
        self.assertTrue(all(value is None for value in summary.values()))


if __name__ == "__main__":
    unittest.main()
