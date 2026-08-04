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
        min_mature_verifier_rate=0.03125,
        min_mature_semantic_rate=0.03125,
        min_mature_partial_rate=0.05,
        min_mature_answer_field_rate=0.10,
        min_completion_distinct_2=0.20,
        max_completion_period=0.70,
        max_completion_repetition=0.70,
        min_output_entropy=0.25,
        min_output_distinct_2=0.10,
        min_throughput_ratio=0.85,
        max_peak_memory_ratio=1.50,
        min_raw_verifier_gain_for_memory_regression=0.03125,
        max_capability_score_drop=1.0,
        max_verifier_drop_from_best=0.125,
        max_completion_drop_from_best=0.30,
        max_extra_step_verifier_drop=0.125,
        max_extra_step_completion_drop=0.25,
        max_extra_step_malformed_increase=0.25,
        max_latent_eval_ce_delta=1.0,
        max_latent_eval_ce_violation=0.25,
        min_latent_eval_entropy=0.10,
        max_latent_eval_delta_rms=16.0,
        max_latent_extra_eval_ce_delta=1.0,
        max_latent_extra_eval_ce_violation=0.25,
        min_latent_extra_eval_entropy=0.10,
        max_latent_extra_eval_delta_rms=16.0,
        max_capability_lagging_buckets=8.0,
        max_recovery_control_fraction=0.50,
        max_policy_advantage_clip_fraction=0.95,
        min_raw_completion_quality=0.20,
        min_raw_completion_rows=16.0,
        min_raw_completion_verifier_rate=0.03125,
        min_raw_completion_semantic_rate=0.03125,
        min_raw_completion_partial_rate=0.05,
        max_raw_completion_schema_wrong_rate=0.75,
        max_raw_completion_malformed_rate=0.25,
        max_raw_completion_missing_rate=0.25,
        min_raw_completion_answer_distinct=0.20,
        min_raw_completion_field_value_distinct_ratio=0.35,
        max_raw_completion_field_value_dominance=0.85,
        min_raw_completion_family_count=4.0,
        min_raw_completion_family_rows=2.0,
        min_raw_completion_family_verifier_rate=0.03125,
        min_raw_completion_family_partial_rate=0.05,
        min_raw_completion_family_field_rate=0.05,
        min_raw_completion_family_quality=0.20,
        max_raw_completion_family_schema_wrong_rate=0.90,
        max_raw_completion_family_malformed_rate=0.75,
        max_raw_completion_family_schema_key_mismatch=0.50,
        max_raw_completion_family_answer_dominance=0.85,
        min_prompt_schema_completion_quality=0.20,
        min_prompt_schema_completion_rows=16.0,
        min_prompt_schema_completion_verifier_rate=0.03125,
        min_prompt_schema_completion_semantic_rate=0.03125,
        min_prompt_schema_completion_partial_rate=0.05,
        max_prompt_schema_completion_schema_wrong_rate=0.75,
        max_prompt_schema_completion_malformed_rate=0.25,
        max_prompt_schema_completion_missing_rate=0.25,
        min_prompt_schema_completion_answer_distinct=0.20,
        min_prompt_schema_completion_field_value_distinct_ratio=0.35,
        max_prompt_schema_completion_field_value_dominance=0.85,
        max_free_run_contract_verifier_gap=0.125,
        min_field_binding_positive_token_fraction=0.55,
        min_field_binding_exact_pair_rank_fraction=0.35,
        min_mature_iters=0,
    )


def healthy_arm_row(arm: str) -> dict[str, float | int | str]:
    return {
        "arm": arm,
        "trials": 1,
        "ok_trials": 1,
        "healthy_trial_fraction": 1.0,
        "stage_model_tokens_per_sec_mean": 1000.0,
        "peak_used_mb_mean": 1000.0,
        "valid_teacher_ce_last_mean": 1.0,
        "latent_eval_final_ce_delta_last_mean": 0.0,
        "latent_eval_final_ce_violation_last_mean": 0.0,
        "latent_eval_final_entropy_bits_last_mean": 1.0,
        "latent_eval_final_delta_rms_last_mean": 1.0,
        "latent_extra_eval_max_ce_delta_last_mean": 0.0,
        "latent_extra_eval_max_ce_violation_last_mean": 0.0,
        "latent_extra_eval_min_entropy_bits_last_mean": 1.0,
        "latent_extra_eval_max_delta_rms_last_mean": 1.0,
        "source_mean_difficulty_last_mean": 5.0,
        "ruliad_verifier_last_mean": 0.25,
        "ruliad_semantic_last_mean": 0.25,
        "ruliad_partial_last_mean": 0.5,
        "ruliad_schema_wrong_last_mean": 0.0,
        "ruliad_malformed_last_mean": 0.0,
        "ruliad_answer_field_accuracy_last_mean": 0.8,
        "ruliad_answer_field_coverage_last_mean": 0.85,
        "ruliad_answer_termination_rate_last_mean": 0.9,
        "completion_health_last_mean": 0.8,
        "completion_distinct_2_last_mean": 0.5,
        "completion_period_2_to_64_last_mean": 0.0,
        "completion_repetition_last_mean": 0.0,
        "raw_completion_quality_mean_mean": 0.8,
        "raw_completion_rows_mean": 32.0,
        "raw_completion_verifier_rate_mean": 0.25,
        "raw_completion_semantic_rate_mean": 0.25,
        "raw_completion_partial_rate_mean": 0.5,
        "raw_completion_schema_wrong_rate_mean": 0.0,
        "raw_completion_malformed_rate_mean": 0.0,
        "raw_completion_missing_rate_mean": 0.0,
        "raw_completion_expected_answer_distinct_fraction_mean": 0.8,
        "raw_completion_actual_answer_distinct_fraction_mean": 0.8,
        "raw_completion_field_value_distinct_ratio_mean": 1.0,
        "raw_completion_actual_field_value_dominant_fraction_mean": 0.2,
        "raw_completion_family_count_mean": 6.0,
        "raw_completion_min_family_rows_mean": 4.0,
        "raw_completion_worst_family_verifier_rate_mean": 0.25,
        "raw_completion_worst_family_partial_rate_mean": 0.5,
        "raw_completion_worst_family_field_accuracy_mean": 0.8,
        "raw_completion_worst_family_completion_quality_mean": 0.8,
        "raw_completion_max_family_schema_wrong_rate_mean": 0.0,
        "raw_completion_max_family_malformed_rate_mean": 0.0,
        "raw_completion_max_family_schema_key_mismatch_rate_mean": 0.0,
        "raw_completion_max_family_answer_dominant_fraction_mean": 0.2,
        "prompt_schema_completion_rows_mean": 32.0,
        "prompt_schema_completion_quality_mean_mean": 0.8,
        "prompt_schema_completion_verifier_rate_mean": 0.25,
        "prompt_schema_completion_semantic_rate_mean": 0.25,
        "prompt_schema_completion_partial_rate_mean": 0.5,
        "prompt_schema_completion_schema_wrong_rate_mean": 0.0,
        "prompt_schema_completion_malformed_rate_mean": 0.0,
        "prompt_schema_completion_missing_rate_mean": 0.0,
        "prompt_schema_completion_expected_answer_distinct_fraction_mean": 0.8,
        "prompt_schema_completion_actual_answer_distinct_fraction_mean": 0.8,
        "prompt_schema_completion_field_value_distinct_ratio_mean": 1.0,
        "prompt_schema_completion_actual_field_value_dominant_fraction_mean": 0.2,
        "policy_completion_rows_mean": 0.0,
        "policy_advantage_clip_fraction_mean": 0.0,
        "policy_update_skipped_count_mean": 0.0,
        "policy_config_weight_mean": 0.0,
        "policy_config_expected_update_steps_mean": 0.0,
        "answer_contract_config_weight_mean": 0.0,
        "answer_contract_config_premature_close_unlikelihood_weight_mean": 0.0,
        "answer_contract_config_expected_update_steps_mean": 0.0,
        "answer_contract_oracle_rows_mean": 0.0,
        "answer_contract_tokens_mean": 0.0,
        "answer_contract_premature_close_tokens_mean": 0.0,
        "answer_contract_missing_policy_batch_count_mean": 0.0,
        "answer_contract_policy_batch_present_fraction_mean": 1.0,
        "contrast_config_weight_mean": 0.0,
        "contrast_config_expected_update_steps_mean": 0.0,
        "contrast_pairs_mean": 0.0,
        "output_entropy_bits_last_mean": 2.0,
        "output_distinct_2_last_mean": 0.5,
        "fatal_gate_count_mean": 0.0,
        "capability_score_drop_from_best_mean": 0.0,
        "capability_bucket_lagging_count_mean": 0.0,
        "capability_verifier_drop_from_best_mean": 0.0,
        "capability_completion_drop_from_best_mean": 0.0,
        "recovery_control_fraction_mean": 0.0,
        "source_capability_recovery_control_count_mean": 0.0,
        "source_capability_recovery_control_fraction_mean": 0.0,
    }


class ArmSummaryTests(unittest.TestCase):
    def test_source_capability_recovery_columns_are_aggregated(self) -> None:
        rows = [
            {
                "arm": "candidate",
                "status": "ok",
                "healthy": "true",
                "source_capability_recovery_control_count": 1,
                "source_capability_recovery_control_fraction": 0.25,
            },
            {
                "arm": "candidate",
                "status": "ok",
                "healthy": "true",
                "source_capability_recovery_control_count": 3,
                "source_capability_recovery_control_fraction": 0.75,
            },
        ]

        summary = ANALYZE.summarize_by_arm(rows)[0]

        self.assertEqual(summary["source_capability_recovery_control_count_mean"], 2.0)
        self.assertEqual(
            summary["source_capability_recovery_control_fraction_mean"], 0.5
        )

    def test_answer_contract_columns_are_aggregated(self) -> None:
        rows = [
            {
                "arm": "candidate",
                "status": "ok",
                "healthy": "true",
                "answer_contract_config_weight": 0.25,
                "answer_contract_config_premature_close_unlikelihood_weight": 0.5,
                "answer_contract_config_prompt_schema_value_weight": 2.0,
                "answer_contract_config_prompt_schema_max_rows_per_step": 4,
                "answer_contract_config_expected_update_steps": 8,
                "answer_contract_prompt_schema_sample_groups": 3,
                "answer_contract_oracle_rows": 8,
                "answer_contract_prompt_schema_rows": 6,
                "answer_contract_tokens": 160,
                "answer_contract_prompt_schema_value_tokens": 48,
                "answer_contract_premature_close_tokens": 96,
                "answer_contract_policy_batch_present_fraction": 1.0,
            },
            {
                "arm": "candidate",
                "status": "ok",
                "healthy": "true",
                "answer_contract_config_weight": 0.25,
                "answer_contract_config_premature_close_unlikelihood_weight": 0.5,
                "answer_contract_config_prompt_schema_value_weight": 2.0,
                "answer_contract_config_prompt_schema_max_rows_per_step": 4,
                "answer_contract_config_expected_update_steps": 8,
                "answer_contract_prompt_schema_sample_groups": 1,
                "answer_contract_oracle_rows": 4,
                "answer_contract_prompt_schema_rows": 2,
                "answer_contract_tokens": 80,
                "answer_contract_prompt_schema_value_tokens": 16,
                "answer_contract_premature_close_tokens": 48,
                "answer_contract_policy_batch_present_fraction": 0.5,
            },
        ]

        summary = ANALYZE.summarize_by_arm(rows)[0]

        self.assertEqual(summary["answer_contract_config_weight_mean"], 0.25)
        self.assertEqual(
            summary[
                "answer_contract_config_premature_close_unlikelihood_weight_mean"
            ],
            0.5,
        )
        self.assertEqual(
            summary["answer_contract_config_expected_update_steps_mean"], 8.0
        )
        self.assertEqual(
            summary["answer_contract_config_prompt_schema_value_weight_mean"], 2.0
        )
        self.assertEqual(
            summary["answer_contract_config_prompt_schema_max_rows_per_step_mean"], 4.0
        )
        self.assertEqual(
            summary["answer_contract_prompt_schema_sample_groups_mean"], 2.0
        )
        self.assertEqual(summary["answer_contract_oracle_rows_mean"], 6.0)
        self.assertEqual(summary["answer_contract_prompt_schema_rows_mean"], 4.0)
        self.assertEqual(summary["answer_contract_tokens_mean"], 120.0)
        self.assertEqual(summary["answer_contract_prompt_schema_value_tokens_mean"], 32.0)
        self.assertEqual(summary["answer_contract_premature_close_tokens_mean"], 72.0)
        self.assertEqual(
            summary["answer_contract_policy_batch_present_fraction_mean"], 0.75
        )

    def test_best_eval_step_columns_are_aggregated(self) -> None:
        rows = [
            {
                "arm": "candidate",
                "status": "ok",
                "healthy": "true",
                "best_eval_steps": 2,
                "best_eval_verifier": 0.25,
                "best_eval_completion": 0.40,
                "best_eval_verifier_delta": 0.10,
                "best_eval_completion_delta": 0.05,
            },
            {
                "arm": "candidate",
                "status": "ok",
                "healthy": "true",
                "best_eval_steps": 4,
                "best_eval_verifier": 0.50,
                "best_eval_completion": 0.80,
                "best_eval_verifier_delta": 0.20,
                "best_eval_completion_delta": 0.15,
            },
        ]

        summary = ANALYZE.summarize_by_arm(rows)[0]

        self.assertEqual(summary["best_eval_steps_mean"], 3.0)
        self.assertEqual(summary["best_eval_verifier_mean"], 0.375)
        self.assertAlmostEqual(summary["best_eval_completion_mean"], 0.60)
        self.assertAlmostEqual(summary["best_eval_verifier_delta_mean"], 0.15)
        self.assertAlmostEqual(summary["best_eval_completion_delta_mean"], 0.10)


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
        self.assertEqual(summary["raw_completion_actual_answer_dominant_fraction"], 0.5)
        self.assertAlmostEqual(
            summary["raw_completion_expected_field_value_distinct_fraction"],
            5.0 / 6.0,
        )
        self.assertAlmostEqual(
            summary["raw_completion_actual_field_value_distinct_fraction"],
            5.0 / 6.0,
        )
        self.assertEqual(summary["raw_completion_field_value_distinct_ratio"], 1.0)
        self.assertAlmostEqual(
            summary["raw_completion_actual_field_value_dominant_fraction"],
            2.0 / 6.0,
        )
        self.assertAlmostEqual(
            summary["raw_completion_actual_field_value_entropy_bits"],
            2.2516291673878226,
        )
        self.assertEqual(summary["raw_completion_status_entropy_bits"], 1.0)
        self.assertEqual(summary["raw_completion_dominant_status_fraction"], 0.5)

    def test_repeated_field_values_surface_collapse_metrics(self) -> None:
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
                    "status": "SchemaValidWrong",
                    "verifier_match": False,
                    "semantic_match": False,
                    "partial_credit": True,
                    "correct_field_count": 1,
                    "expected_field_count": 3,
                    "answer_terminated": True,
                    "completion_quality_ppm": 500_000,
                    "expected_answer": f"ok=1;l={index};r={index}",
                    "actual_answer": "ok=1;l=1;r=1",
                }
                for index in range(1, 5)
            ]
            with path.open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            summary = ANALYZE.read_raw_completion_samples(str(run_dir))

        self.assertEqual(summary["raw_completion_rows"], 4)
        self.assertEqual(summary["raw_completion_expected_answer_distinct_fraction"], 1.0)
        self.assertEqual(summary["raw_completion_actual_answer_distinct_fraction"], 0.25)
        self.assertEqual(summary["raw_completion_actual_answer_dominant_fraction"], 1.0)
        self.assertLess(summary["raw_completion_field_value_distinct_ratio"], 0.5)
        self.assertGreater(summary["raw_completion_actual_field_value_dominant_fraction"], 0.3)

    def test_prompt_schema_probe_records_surface_value_collapse_metrics(self) -> None:
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
                    "status": "VerifierMatch",
                    "verifier_match": True,
                    "semantic_match": True,
                    "partial_credit": True,
                    "correct_field_count": 2,
                    "expected_field_count": 2,
                    "answer_terminated": True,
                    "completion_quality_ppm": 1_000_000,
                    "expected_answer": "ok=1;l=2",
                    "actual_answer": "ok=1;l=2",
                },
                *[
                    {
                        "epoch": 2,
                        "absolute_step": 16,
                        "probe_name": "ruliad_correctness_prompt_schema",
                        "status": "SchemaValidWrong",
                        "verifier_match": False,
                        "semantic_match": False,
                        "partial_credit": False,
                        "correct_field_count": 0,
                        "expected_field_count": 3,
                        "answer_terminated": True,
                        "completion_quality_ppm": 250_000,
                        "generated_token_count": 24,
                        "hash_canary": False,
                        "expected_answer": f"ok=1;l={index};r={index}",
                        "actual_answer": "ok=cccc;l=cccc;r=cccc",
                    }
                    for index in range(1, 5)
                ],
            ]
            with path.open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            summary = ANALYZE.read_prompt_schema_completion_samples(str(run_dir))

        self.assertEqual(summary["prompt_schema_completion_rows"], 4)
        self.assertEqual(summary["prompt_schema_completion_verifier_rate"], 0.0)
        self.assertEqual(summary["prompt_schema_completion_schema_wrong_rate"], 1.0)
        self.assertEqual(summary["prompt_schema_completion_malformed_rate"], 0.0)
        self.assertEqual(summary["prompt_schema_completion_quality_mean"], 0.25)
        self.assertLess(
            summary["prompt_schema_completion_field_value_distinct_ratio"],
            0.5,
        )
        self.assertGreater(
            summary["prompt_schema_completion_actual_field_value_dominant_fraction"],
            0.3,
        )

    def test_family_schema_leakage_surfaces_worst_slice_metrics(self) -> None:
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
                    "family": "eca",
                    "status": "SchemaValidWrong",
                    "verifier_match": False,
                    "semantic_match": False,
                    "partial_credit": False,
                    "correct_field_count": 0,
                    "expected_field_count": 4,
                    "answer_terminated": False,
                    "completion_quality_ppm": 500_000,
                    "expected_answer": "xlen=18;xalpha=01;xcounts=10,8;xedge=10",
                    "actual_answer": "nflen=6;nfalpha=abc;nfcounts=1,1",
                },
                {
                    "epoch": 1,
                    "absolute_step": 0,
                    "probe_name": "ruliad_correctness",
                    "family": "eca",
                    "status": "SchemaValidWrong",
                    "verifier_match": False,
                    "semantic_match": False,
                    "partial_credit": False,
                    "correct_field_count": 0,
                    "expected_field_count": 4,
                    "answer_terminated": False,
                    "completion_quality_ppm": 500_000,
                    "expected_answer": "xlen=12;xalpha=01;xcounts=7,5;xedge=01",
                    "actual_answer": "nflen=6;nfalpha=abc;nfcounts=1,1",
                },
                {
                    "epoch": 1,
                    "absolute_step": 0,
                    "probe_name": "ruliad_correctness",
                    "family": "category",
                    "status": "VerifierMatch",
                    "verifier_match": True,
                    "semantic_match": True,
                    "partial_credit": True,
                    "correct_field_count": 3,
                    "expected_field_count": 3,
                    "answer_terminated": True,
                    "completion_quality_ppm": 1_000_000,
                    "expected_answer": "ok=1;l=2;r=2",
                    "actual_answer": "ok=1;l=2;r=2",
                },
                {
                    "epoch": 1,
                    "absolute_step": 0,
                    "probe_name": "ruliad_correctness",
                    "family": "category",
                    "status": "Partial",
                    "verifier_match": False,
                    "semantic_match": False,
                    "partial_credit": True,
                    "correct_field_count": 1,
                    "expected_field_count": 3,
                    "answer_terminated": True,
                    "completion_quality_ppm": 1_000_000,
                    "expected_answer": "ok=1;l=3;r=3",
                    "actual_answer": "ok=1;l=2;r=2",
                },
            ]
            with path.open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            summary = ANALYZE.read_raw_completion_samples(str(run_dir))

        self.assertEqual(summary["raw_completion_family_count"], 2)
        self.assertEqual(summary["raw_completion_min_family_rows"], 2)
        self.assertEqual(summary["raw_completion_worst_family_verifier_rate"], 0.0)
        self.assertEqual(summary["raw_completion_worst_family_partial_rate"], 0.0)
        self.assertEqual(summary["raw_completion_worst_family_field_accuracy"], 0.0)
        self.assertEqual(summary["raw_completion_max_family_schema_key_mismatch_rate"], 1.0)
        self.assertEqual(summary["raw_completion_max_family_answer_dominant_fraction"], 1.0)

    def test_missing_raw_completion_file_returns_empty_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = ANALYZE.read_raw_completion_samples(tmp)

        self.assertEqual(set(summary), set(ANALYZE.RAW_COMPLETION_METRIC_COLUMNS))
        self.assertTrue(all(value is None for value in summary.values()))

    def test_missing_prompt_schema_completion_file_returns_empty_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = ANALYZE.read_prompt_schema_completion_samples(tmp)

        self.assertEqual(
            set(summary), set(ANALYZE.PROMPT_SCHEMA_COMPLETION_METRIC_COLUMNS)
        )
        self.assertTrue(all(value is None for value in summary.values()))


class PolicyConfigTests(unittest.TestCase):
    def test_policy_config_reports_expected_update_slots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "training_config.json").write_text(
                json.dumps(
                    {
                        "training": {
                            "max_iters": 256,
                            "ruliad_supervision": {
                                "verifier_reward": {
                                    "enabled": True,
                                    "weight": 0.01,
                                    "start_after_steps": 0,
                                    "every_steps": 32,
                                }
                            },
                        }
                    }
                )
                + "\n"
            )

            summary = ANALYZE.read_policy_config(str(run_dir))

        self.assertEqual(summary["policy_config_enabled"], 1.0)
        self.assertEqual(summary["policy_config_weight"], 0.01)
        self.assertEqual(summary["policy_config_start_after_steps"], 0)
        self.assertEqual(summary["policy_config_every_steps"], 32)
        self.assertEqual(summary["policy_config_expected_update_steps"], 8)

    def test_answer_contract_config_reports_expected_update_slots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "training_config.json").write_text(
                json.dumps(
                    {
                        "training": {
                            "max_iters": 256,
                            "ruliad_supervision": {
                                "answer_contract": {
                                    "enabled": True,
                                    "weight": 0.25,
                                    "premature_close_unlikelihood_weight": 0.5,
                                    "start_after_steps": 64,
                                    "every_steps": 16,
                                    "max_completion_tokens": 64,
                                    "max_rows_per_step": 8,
                                    "prompt_schema_max_rows_per_step": 4,
                                    "prompt_schema_value_weight": 2.0,
                                }
                            },
                        }
                    }
                )
                + "\n"
            )

            summary = ANALYZE.read_answer_contract_config(str(run_dir))

        self.assertEqual(summary["answer_contract_config_weight"], 0.25)
        self.assertEqual(
            summary["answer_contract_config_premature_close_unlikelihood_weight"],
            0.5,
        )
        self.assertEqual(summary["answer_contract_config_start_after_steps"], 64)
        self.assertEqual(summary["answer_contract_config_every_steps"], 16)
        self.assertEqual(summary["answer_contract_config_max_completion_tokens"], 64)
        self.assertEqual(summary["answer_contract_config_max_rows_per_step"], 8)
        self.assertEqual(
            summary["answer_contract_config_prompt_schema_max_rows_per_step"], 4
        )
        self.assertEqual(summary["answer_contract_config_prompt_schema_value_weight"], 2.0)
        self.assertEqual(summary["answer_contract_config_expected_update_steps"], 12)

    def test_answer_contract_telemetry_aggregates_oracle_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            events = run_dir / "events"
            events.mkdir()
            path = events / "ruliad_answer_contract.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "policy_batch_present": True,
                                "sample_groups": 2,
                                "prompt_schema_sample_groups": 1,
                                "oracle_rows": 2,
                                "prompt_schema_rows": 3,
                                "contract_tokens": 24,
                                "prompt_schema_value_tokens": 7,
                                "schema_tokens": 12,
                                "value_tokens": 8,
                                "other_tokens": 4,
                                "premature_close_tokens": 20,
                                "answer_contract_weight": 0.25,
                                "premature_close_unlikelihood_weight": 0.5,
                                "max_completion_tokens": 64,
                                "max_rows_per_step": 8,
                                "prompt_schema_max_rows_per_step": 4,
                            }
                        ),
                        json.dumps(
                            {
                                "policy_batch_present": False,
                                "skip_reason": "missing_policy_batch",
                                "sample_groups": 0,
                                "prompt_schema_sample_groups": 0,
                                "oracle_rows": 0,
                                "prompt_schema_rows": 0,
                                "contract_tokens": 0,
                                "prompt_schema_value_tokens": 0,
                                "schema_tokens": 0,
                                "value_tokens": 0,
                                "other_tokens": 0,
                                "premature_close_tokens": 0,
                                "answer_contract_weight": 0.25,
                                "premature_close_unlikelihood_weight": 0.5,
                                "max_completion_tokens": 64,
                                "max_rows_per_step": 8,
                                "prompt_schema_max_rows_per_step": 4,
                            }
                        ),
                    ]
                )
                + "\n"
            )

            summary = ANALYZE.read_answer_contract_telemetry(str(run_dir))

        self.assertEqual(summary["answer_contract_sample_groups"], 2)
        self.assertEqual(summary["answer_contract_prompt_schema_sample_groups"], 1)
        self.assertEqual(summary["answer_contract_oracle_rows"], 2)
        self.assertEqual(summary["answer_contract_prompt_schema_rows"], 3)
        self.assertEqual(summary["answer_contract_tokens"], 24)
        self.assertEqual(summary["answer_contract_prompt_schema_value_tokens"], 7)
        self.assertEqual(summary["answer_contract_schema_tokens"], 12)
        self.assertEqual(summary["answer_contract_value_tokens"], 8)
        self.assertEqual(summary["answer_contract_premature_close_tokens"], 20)
        self.assertEqual(summary["answer_contract_policy_batch_present_fraction"], 0.5)
        self.assertEqual(summary["answer_contract_missing_policy_batch_count"], 1)
        self.assertEqual(summary["answer_contract_prompt_schema_max_rows_per_step"], 4)

    def test_structured_contrast_config_reports_expected_update_slots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "training_config.json").write_text(
                json.dumps(
                    {
                        "training": {
                            "max_iters": 256,
                            "ruliad_supervision": {
                                "verifier_reward": {
                                    "enabled": True,
                                    "structured_contrast_weight": 0.01,
                                    "structured_contrast_start_after_steps": 128,
                                    "structured_contrast_every_steps": 16,
                                }
                            },
                        }
                    }
                )
                + "\n"
            )

            summary = ANALYZE.read_structured_contrast_config(str(run_dir))

        self.assertEqual(summary["contrast_config_weight"], 0.01)
        self.assertEqual(summary["contrast_config_start_after_steps"], 128)
        self.assertEqual(summary["contrast_config_every_steps"], 16)
        self.assertEqual(summary["contrast_config_expected_update_steps"], 8)

    def test_field_binding_contrast_config_reports_expected_update_slots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "training_config.json").write_text(
                json.dumps(
                    {
                        "training": {
                            "max_iters": 256,
                            "ruliad_supervision": {
                                "verifier_reward": {
                                    "enabled": True,
                                    "field_binding_contrast_weight": 0.01,
                                    "field_binding_contrast_start_after_steps": 128,
                                    "field_binding_contrast_every_steps": 16,
                                    "field_binding_contrast_rank_metric_every_steps": 32,
                                    "field_binding_contrast_pair_weight": 0.5,
                                    "field_binding_contrast_replay_capacity": 64,
                                }
                            },
                        }
                    }
                )
                + "\n"
            )

            summary = ANALYZE.read_field_binding_contrast_config(str(run_dir))

        self.assertEqual(summary["field_binding_config_weight"], 0.01)
        self.assertEqual(summary["field_binding_config_start_after_steps"], 128)
        self.assertEqual(summary["field_binding_config_every_steps"], 16)
        self.assertEqual(summary["field_binding_config_rank_metric_every_steps"], 32)
        self.assertEqual(summary["field_binding_config_pair_weight"], 0.5)
        self.assertEqual(summary["field_binding_config_replay_capacity"], 64)
        self.assertEqual(summary["field_binding_config_expected_update_steps"], 8)
        self.assertEqual(summary["field_binding_config_expected_rank_metric_steps"], 4)

    def test_generated_attractor_config_reports_replay_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "training_config.json").write_text(
                json.dumps(
                    {
                        "training": {
                            "ruliad_supervision": {
                                "verifier_reward": {
                                    "generated_attractor_replay_capacity": 128,
                                    "generated_attractor_replay_min_count": 2,
                                    "generated_attractor_replay_max_candidates": 4,
                                    "generated_attractor_replay_min_distinct_answers": 2,
                                    "generated_attractor_replay_max_dominant_fraction": 0.5,
                                }
                            }
                        }
                    }
                )
                + "\n"
            )

            summary = ANALYZE.read_generated_attractor_config(str(run_dir))

        self.assertEqual(summary["generated_attractor_config_capacity"], 128)
        self.assertEqual(summary["generated_attractor_config_min_count"], 2)
        self.assertEqual(summary["generated_attractor_config_max_candidates"], 4)
        self.assertEqual(summary["generated_attractor_config_min_distinct_answers"], 2)
        self.assertEqual(summary["generated_attractor_config_max_dominant_fraction"], 0.5)

    def test_generated_attractor_telemetry_aggregates_rows_and_dominance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            events = run_dir / "events"
            events.mkdir()
            path = events / "ruliad_generated_attractor_replay.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "source": "policy",
                                "observed_completion_rows": 8,
                                "recorded_attractor_rows": 3,
                                "selected_candidate_rows": 1,
                                "selected_field_binding_pairs": 0,
                                "replay_pool_size": 2,
                                "active_attractor_count": 1,
                                "active_observation_count": 2,
                                "distinct_answer_count": 1,
                                "dominant_answer_count": 2,
                                "dominant_answer_fraction": 1.0,
                            }
                        ),
                        json.dumps(
                            {
                                "source": "field_binding",
                                "observed_completion_rows": 0,
                                "recorded_attractor_rows": 0,
                                "selected_candidate_rows": 2,
                                "selected_field_binding_pairs": 2,
                                "replay_pool_size": 3,
                                "active_attractor_count": 2,
                                "active_observation_count": 4,
                                "distinct_answer_count": 2,
                                "dominant_answer_count": 3,
                                "dominant_answer_fraction": 0.75,
                            }
                        ),
                    ]
                )
                + "\n"
            )

            summary = ANALYZE.read_generated_attractor_telemetry(str(run_dir))

        self.assertEqual(summary["generated_attractor_observed_rows"], 8)
        self.assertEqual(summary["generated_attractor_recorded_rows"], 3)
        self.assertEqual(summary["generated_attractor_selected_candidate_rows"], 3)
        self.assertEqual(summary["generated_attractor_selected_field_binding_pairs"], 2)
        self.assertEqual(summary["generated_attractor_replay_pool_size"], 3)
        self.assertEqual(summary["generated_attractor_active_count"], 2)
        self.assertEqual(summary["generated_attractor_active_observation_count"], 4)
        self.assertEqual(summary["generated_attractor_distinct_answer_count"], 2)
        self.assertEqual(summary["generated_attractor_dominant_answer_count"], 3)
        self.assertEqual(summary["generated_attractor_dominant_answer_fraction"], 0.75)

    def test_structured_recovery_config_reports_expected_update_slots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "training_config.json").write_text(
                json.dumps(
                    {
                        "training": {
                            "max_iters": 256,
                            "ruliad_supervision": {
                                "answer_denoising": {
                                    "enabled": True,
                                    "structured_recovery_weight": 0.25,
                                    "structured_recovery_start_after_steps": 0,
                                    "structured_recovery_every_steps": 8,
                                    "structured_recovery_negative_count": 2,
                                    "structured_recovery_template_negative_count": 1,
                                    "structured_recovery_schema_negative_count": 3,
                                    "structured_recovery_max_completion_tokens": 64,
                                }
                            },
                        }
                    }
                )
                + "\n"
            )

            summary = ANALYZE.read_structured_recovery_config(str(run_dir))

        self.assertEqual(summary["recovery_config_weight"], 0.25)
        self.assertEqual(summary["recovery_config_start_after_steps"], 0)
        self.assertEqual(summary["recovery_config_every_steps"], 8)
        self.assertEqual(summary["recovery_config_negative_count"], 2)
        self.assertEqual(summary["recovery_config_template_negative_count"], 1)
        self.assertEqual(summary["recovery_config_schema_negative_count"], 3)
        self.assertEqual(summary["recovery_config_max_completion_tokens"], 64)
        self.assertEqual(summary["recovery_config_expected_update_steps"], 32)

    def test_structured_recovery_config_reports_unscheduled_short_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "training_config.json").write_text(
                json.dumps(
                    {
                        "training": {
                            "max_iters": 128,
                            "ruliad_supervision": {
                                "answer_denoising": {
                                    "enabled": True,
                                    "structured_recovery_weight": 0.25,
                                    "structured_recovery_start_after_steps": 128,
                                    "structured_recovery_every_steps": 8,
                                }
                            },
                        }
                    }
                )
                + "\n"
            )

            summary = ANALYZE.read_structured_recovery_config(str(run_dir))

        self.assertEqual(summary["recovery_config_weight"], 0.25)
        self.assertEqual(summary["recovery_config_expected_update_steps"], 0)


class StructuredContrastTelemetryTests(unittest.TestCase):
    def test_recovery_sidecar_sums_activity_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            events = run_dir / "events"
            events.mkdir()
            path = events / "ruliad_structured_recovery.jsonl"
            rows = [
                {
                    "policy_batch_present": True,
                    "skip_reason": None,
                    "sample_groups": 2,
                    "recovery_rows": 6,
                    "field_negative_recovery_rows": 4,
                    "template_negative_recovery_rows": 2,
                    "schema_negative_recovery_rows": 1,
                    "structured_recovery_weight": 0.1,
                    "structured_recovery_max_completion_tokens": 32,
                },
                {
                    "policy_batch_present": False,
                    "skip_reason": "missing_policy_batch",
                    "sample_groups": 0,
                    "recovery_rows": 0,
                    "field_negative_recovery_rows": 0,
                    "template_negative_recovery_rows": 0,
                    "schema_negative_recovery_rows": 0,
                    "structured_recovery_weight": 0.2,
                    "structured_recovery_max_completion_tokens": 64,
                },
            ]
            with path.open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            summary = ANALYZE.read_structured_recovery_telemetry(str(run_dir))

        self.assertEqual(summary["recovery_sample_groups"], 2)
        self.assertEqual(summary["recovery_rows"], 6)
        self.assertEqual(summary["recovery_field_negative_rows"], 4)
        self.assertEqual(summary["recovery_template_negative_rows"], 2)
        self.assertEqual(summary["recovery_schema_negative_rows"], 1)
        self.assertEqual(summary["recovery_policy_batch_present_fraction"], 0.5)
        self.assertEqual(summary["recovery_missing_policy_batch_count"], 1)
        self.assertEqual(summary["recovery_weight"], 0.2)
        self.assertEqual(summary["recovery_max_completion_tokens"], 64)

    def test_missing_recovery_sidecar_returns_empty_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = ANALYZE.read_structured_recovery_telemetry(tmp)

        self.assertEqual(set(summary), set(ANALYZE.STRUCTURED_RECOVERY_METRIC_COLUMNS))
        self.assertTrue(all(value is None for value in summary.values()))

    def test_contrast_sidecar_sums_activity_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            events = run_dir / "events"
            events.mkdir()
            path = events / "ruliad_structured_contrast.jsonl"
            rows = [
                {
                    "sample_groups": 2,
                    "oracle_completion_rows": 2,
                    "field_negative_completion_rows": 4,
                    "template_negative_completion_rows": 3,
                    "contrast_pairs": 7,
                    "contrast_discriminative_tokens": 19,
                    "structured_contrast_weight": 0.1,
                    "structured_contrast_margin": 0.25,
                },
                {
                    "sample_groups": 1,
                    "oracle_completion_rows": 1,
                    "field_negative_completion_rows": 2,
                    "template_negative_completion_rows": 2,
                    "contrast_pairs": 4,
                    "contrast_discriminative_tokens": 11,
                    "structured_contrast_weight": 0.2,
                    "structured_contrast_margin": 0.5,
                },
            ]
            with path.open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            summary = ANALYZE.read_structured_contrast_telemetry(str(run_dir))

        self.assertEqual(summary["contrast_sample_groups"], 3)
        self.assertEqual(summary["contrast_oracle_completion_rows"], 3)
        self.assertEqual(summary["contrast_field_negative_completion_rows"], 6)
        self.assertEqual(summary["contrast_template_negative_completion_rows"], 5)
        self.assertEqual(summary["contrast_pairs"], 11)
        self.assertEqual(summary["contrast_discriminative_tokens"], 30)
        self.assertEqual(summary["contrast_weight"], 0.2)
        self.assertEqual(summary["contrast_margin"], 0.5)

    def test_missing_contrast_sidecar_returns_empty_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = ANALYZE.read_structured_contrast_telemetry(tmp)

        self.assertEqual(set(summary), set(ANALYZE.STRUCTURED_CONTRAST_METRIC_COLUMNS))
        self.assertTrue(all(value is None for value in summary.values()))

    def test_field_binding_sidecar_sums_activity_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            events = run_dir / "events"
            events.mkdir()
            path = events / "ruliad_field_binding_contrast.jsonl"
            rows = [
                {
                    "sample_groups": 2,
                    "prompt_pairs": 4,
                    "contrast_pairs": 4,
                    "candidate_pairs": 7,
                    "filtered_presented_action_candidates": 3,
                    "contrast_discriminative_tokens": 19,
                    "negative_pool_size": 12,
                    "replay_pool_size": 8,
                    "replay_contrast_pairs": 3,
                    "rank_metric_pairs": 2,
                    "rank_metric_tokens": 10,
                    "logit_margin_mean": 0.5,
                    "positive_token_fraction": 0.8,
                    "margin_satisfied_token_fraction": 0.4,
                    "exact_pair_rank_fraction": 0.5,
                    "exact_pair_margin_fraction": 0.25,
                    "sequence_rank_metric_pairs": 2,
                    "sequence_log_probability_margin_mean": 0.75,
                    "positive_sequence_fraction": 0.5,
                    "sequence_margin_satisfied_fraction": 0.25,
                    "field_binding_contrast_weight": 0.1,
                    "field_binding_contrast_margin": 0.25,
                },
                {
                    "sample_groups": 3,
                    "prompt_pairs": 6,
                    "contrast_pairs": 6,
                    "candidate_pairs": 9,
                    "filtered_presented_action_candidates": 5,
                    "contrast_discriminative_tokens": 31,
                    "negative_pool_size": 16,
                    "replay_pool_size": 12,
                    "replay_contrast_pairs": 5,
                    "rank_metric_pairs": 4,
                    "rank_metric_tokens": 30,
                    "logit_margin_mean": 1.5,
                    "positive_token_fraction": 0.6,
                    "margin_satisfied_token_fraction": 0.2,
                    "exact_pair_rank_fraction": 0.25,
                    "exact_pair_margin_fraction": 0.0,
                    "sequence_rank_metric_pairs": 4,
                    "sequence_log_probability_margin_mean": 1.5,
                    "positive_sequence_fraction": 0.75,
                    "sequence_margin_satisfied_fraction": 0.5,
                    "field_binding_contrast_weight": 0.2,
                    "field_binding_contrast_margin": 0.5,
                },
            ]
            with path.open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            summary = ANALYZE.read_field_binding_contrast_telemetry(str(run_dir))

        self.assertEqual(summary["field_binding_sample_groups"], 5)
        self.assertEqual(summary["field_binding_prompt_pairs"], 10)
        self.assertEqual(summary["field_binding_contrast_pairs"], 10)
        self.assertEqual(summary["field_binding_candidate_pairs"], 16)
        self.assertEqual(
            summary["field_binding_filtered_presented_action_candidates"], 8
        )
        self.assertEqual(summary["field_binding_discriminative_tokens"], 50)
        self.assertEqual(summary["field_binding_negative_pool_size"], 16)
        self.assertEqual(summary["field_binding_replay_pool_size"], 12)
        self.assertEqual(summary["field_binding_replay_contrast_pairs"], 8)
        self.assertEqual(summary["field_binding_rank_metric_pairs"], 6)
        self.assertEqual(summary["field_binding_rank_metric_tokens"], 40)
        self.assertAlmostEqual(summary["field_binding_logit_margin_mean"], 1.25)
        self.assertAlmostEqual(summary["field_binding_positive_token_fraction"], 0.65)
        self.assertAlmostEqual(summary["field_binding_margin_satisfied_token_fraction"], 0.25)
        self.assertAlmostEqual(summary["field_binding_exact_pair_rank_fraction"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["field_binding_exact_pair_margin_fraction"], 1.0 / 12.0)
        self.assertEqual(summary["field_binding_sequence_rank_metric_pairs"], 6)
        self.assertAlmostEqual(
            summary["field_binding_sequence_log_probability_margin_mean"], 1.25
        )
        self.assertAlmostEqual(
            summary["field_binding_positive_sequence_fraction"], 2.0 / 3.0
        )
        self.assertAlmostEqual(
            summary["field_binding_sequence_margin_satisfied_fraction"], 5.0 / 12.0
        )
        self.assertEqual(summary["field_binding_weight"], 0.2)
        self.assertEqual(summary["field_binding_margin"], 0.5)

    def test_missing_field_binding_sidecar_returns_empty_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = ANALYZE.read_field_binding_contrast_telemetry(tmp)

        self.assertEqual(
            set(summary), set(ANALYZE.FIELD_BINDING_CONTRAST_METRIC_COLUMNS)
        )
        self.assertTrue(all(value is None for value in summary.values()))

    def test_verifier_rollout_sidecar_sums_generated_and_accepted_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            events = run_dir / "events"
            events.mkdir()
            path = events / "ruliad_verifier_rollout_imitation.jsonl"
            rows = [
                {
                    "sample_groups": 2,
                    "generated_completion_rows": 8,
                    "candidate_completion_rows": 1,
                    "accepted_completion_rows": 1,
                    "accepted_imitation_rows": 1,
                    "accepted_recovery_rows": 0,
                    "health_gate_passed": True,
                    "verifier_rate_ppm": 125_000,
                    "schema_wrong_rate_ppm": 375_000,
                    "malformed_rate_ppm": 250_000,
                    "verifier_match_rows": 0,
                    "semantic_match_rows": 1,
                    "partial_rows": 2,
                    "schema_wrong_rows": 3,
                    "malformed_rows": 2,
                    "missing_rows": 0,
                    "field_accuracy_mean": 0.25,
                    "partial_progress_mean": 0.2,
                    "completion_quality_mean": 0.75,
                    "rollout_imitation_weight": 0.02,
                    "rollout_recovery_weight": 0.0,
                    "max_completion_tokens": 64,
                },
                {
                    "sample_groups": 1,
                    "generated_completion_rows": 4,
                    "candidate_completion_rows": 2,
                    "accepted_completion_rows": 2,
                    "accepted_imitation_rows": 0,
                    "accepted_recovery_rows": 2,
                    "health_gate_passed": False,
                    "verifier_rate_ppm": 250_000,
                    "schema_wrong_rate_ppm": 250_000,
                    "malformed_rate_ppm": 250_000,
                    "verifier_match_rows": 1,
                    "semantic_match_rows": 0,
                    "partial_rows": 1,
                    "schema_wrong_rows": 1,
                    "malformed_rows": 1,
                    "missing_rows": 0,
                    "field_accuracy_mean": 0.5,
                    "partial_progress_mean": 0.4,
                    "completion_quality_mean": 1.0,
                    "rollout_imitation_weight": 0.03,
                    "rollout_recovery_weight": 0.05,
                    "max_completion_tokens": 32,
                },
            ]
            with path.open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            summary = ANALYZE.read_verifier_rollout_telemetry(str(run_dir))

        self.assertEqual(summary["rollout_imitation_sample_groups"], 3)
        self.assertEqual(summary["rollout_imitation_generated_rows"], 12)
        self.assertEqual(summary["rollout_imitation_candidate_rows"], 3)
        self.assertEqual(summary["rollout_imitation_accepted_rows"], 3)
        self.assertEqual(summary["rollout_imitation_accepted_imitation_rows"], 1)
        self.assertEqual(summary["rollout_imitation_accepted_recovery_rows"], 2)
        self.assertEqual(summary["rollout_imitation_health_gate_passed_fraction"], 0.5)
        self.assertAlmostEqual(summary["rollout_imitation_verifier_rate"], 1.0 / 6.0)
        self.assertAlmostEqual(summary["rollout_imitation_schema_wrong_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["rollout_imitation_malformed_rate"], 0.25)
        self.assertEqual(summary["rollout_imitation_verifier_rows"], 1)
        self.assertAlmostEqual(summary["rollout_imitation_field_accuracy_mean"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["rollout_imitation_partial_progress_mean"], 4.0 / 15.0)
        self.assertAlmostEqual(summary["rollout_imitation_completion_quality_mean"], 10.0 / 12.0)
        self.assertEqual(summary["rollout_imitation_weight"], 0.03)
        self.assertEqual(summary["rollout_recovery_weight"], 0.05)
        self.assertEqual(summary["rollout_imitation_max_completion_tokens"], 32)

    def test_missing_verifier_rollout_sidecar_returns_empty_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = ANALYZE.read_verifier_rollout_telemetry(tmp)

        self.assertEqual(set(summary), set(ANALYZE.VERIFIER_ROLLOUT_METRIC_COLUMNS))
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

    def test_excess_lagging_buckets_reject_candidate(self) -> None:
        args = promotion_args()
        rows = [
            healthy_arm_row("baseline"),
            {
                **healthy_arm_row("candidate"),
                "capability_bucket_lagging_count_mean": 12.0,
            },
        ]

        gated = ANALYZE.add_gate_decisions(rows, args)
        candidate = next(row for row in gated if row["arm"] == "candidate")

        self.assertEqual(candidate["decision"], "reject")
        self.assertIn("capability_lagging_buckets", candidate["fail_reasons"])

    def test_recovery_weight_without_rows_rejects_candidate(self) -> None:
        args = promotion_args()
        rows = [
            healthy_arm_row("baseline"),
            {
                **healthy_arm_row("candidate"),
                "recovery_weight_mean": 0.25,
                "recovery_rows_mean": 0.0,
                "recovery_missing_policy_batch_count_mean": 0.0,
                "recovery_policy_batch_present_fraction_mean": 1.0,
            },
        ]

        gated = ANALYZE.add_gate_decisions(rows, args)
        candidate = next(row for row in gated if row["arm"] == "candidate")

        self.assertEqual(candidate["decision"], "reject")
        self.assertIn("recovery_objective_inactive", candidate["fail_reasons"])

    def test_scheduled_recovery_without_rows_rejects_candidate(self) -> None:
        args = promotion_args()
        rows = [
            healthy_arm_row("baseline"),
            {
                **healthy_arm_row("candidate"),
                "recovery_config_weight_mean": 0.25,
                "recovery_config_every_steps_mean": 8.0,
                "recovery_config_expected_update_steps_mean": 16.0,
                "recovery_rows_mean": 0.0,
                "recovery_missing_policy_batch_count_mean": 0.0,
                "recovery_policy_batch_present_fraction_mean": 1.0,
            },
        ]

        gated = ANALYZE.add_gate_decisions(rows, args)
        candidate = next(row for row in gated if row["arm"] == "candidate")

        self.assertEqual(candidate["decision"], "reject")
        self.assertIn("recovery_objective_inactive", candidate["fail_reasons"])

    def test_configured_recovery_without_update_slots_rejects_candidate(self) -> None:
        args = promotion_args()
        rows = [
            healthy_arm_row("baseline"),
            {
                **healthy_arm_row("candidate"),
                "recovery_config_weight_mean": 0.25,
                "recovery_config_every_steps_mean": 8.0,
                "recovery_config_expected_update_steps_mean": 0.0,
                "recovery_rows_mean": 0.0,
                "recovery_missing_policy_batch_count_mean": 0.0,
                "recovery_policy_batch_present_fraction_mean": 1.0,
            },
        ]

        gated = ANALYZE.add_gate_decisions(rows, args)
        candidate = next(row for row in gated if row["arm"] == "candidate")

        self.assertEqual(candidate["decision"], "reject")
        self.assertIn("recovery_objective_unscheduled", candidate["fail_reasons"])

    def test_scheduled_policy_without_rows_rejects_candidate(self) -> None:
        args = promotion_args()
        rows = [
            healthy_arm_row("baseline"),
            {
                **healthy_arm_row("candidate"),
                "policy_config_weight_mean": 0.01,
                "policy_config_expected_update_steps_mean": 4.0,
                "policy_completion_rows_mean": 0.0,
            },
        ]

        gated = ANALYZE.add_gate_decisions(rows, args)
        candidate = next(row for row in gated if row["arm"] == "candidate")

        self.assertEqual(candidate["decision"], "reject")
        self.assertIn("policy_objective_inactive", candidate["fail_reasons"])

    def test_scheduled_answer_contract_without_rows_rejects_candidate(self) -> None:
        args = promotion_args()
        rows = [
            healthy_arm_row("baseline"),
            {
                **healthy_arm_row("candidate"),
                "answer_contract_config_weight_mean": 0.25,
                "answer_contract_config_expected_update_steps_mean": 4.0,
                "answer_contract_oracle_rows_mean": 0.0,
                "answer_contract_tokens_mean": 0.0,
                "answer_contract_missing_policy_batch_count_mean": 0.0,
                "answer_contract_policy_batch_present_fraction_mean": 1.0,
            },
        ]

        gated = ANALYZE.add_gate_decisions(rows, args)
        candidate = next(row for row in gated if row["arm"] == "candidate")

        self.assertEqual(candidate["decision"], "reject")
        self.assertIn(
            "answer_contract_objective_inactive", candidate["fail_reasons"]
        )
        self.assertIn("answer_contract_objective_no_tokens", candidate["fail_reasons"])

    def test_prompt_schema_answer_contract_without_rows_rejects_candidate(self) -> None:
        args = promotion_args()
        rows = [
            healthy_arm_row("baseline"),
            {
                **healthy_arm_row("candidate"),
                "answer_contract_config_weight_mean": 0.25,
                "answer_contract_config_prompt_schema_value_weight_mean": 4.0,
                "answer_contract_config_expected_update_steps_mean": 4.0,
                "answer_contract_oracle_rows_mean": 16.0,
                "answer_contract_prompt_schema_rows_mean": 0.0,
                "answer_contract_tokens_mean": 128.0,
                "answer_contract_prompt_schema_value_tokens_mean": 0.0,
                "answer_contract_missing_policy_batch_count_mean": 0.0,
                "answer_contract_policy_batch_present_fraction_mean": 1.0,
            },
        ]

        gated = ANALYZE.add_gate_decisions(rows, args)
        candidate = next(row for row in gated if row["arm"] == "candidate")

        self.assertEqual(candidate["decision"], "reject")
        self.assertIn(
            "answer_contract_prompt_schema_objective_inactive",
            candidate["fail_reasons"],
        )
        self.assertIn(
            "answer_contract_prompt_schema_no_tokens", candidate["fail_reasons"]
        )

    def test_scheduled_contrast_without_pairs_rejects_candidate(self) -> None:
        args = promotion_args()
        rows = [
            healthy_arm_row("baseline"),
            {
                **healthy_arm_row("candidate"),
                "contrast_config_weight_mean": 0.01,
                "contrast_config_expected_update_steps_mean": 8.0,
                "contrast_pairs_mean": 0.0,
            },
        ]

        gated = ANALYZE.add_gate_decisions(rows, args)
        candidate = next(row for row in gated if row["arm"] == "candidate")

        self.assertEqual(candidate["decision"], "reject")
        self.assertIn("contrast_objective_inactive", candidate["fail_reasons"])

    def test_answer_contract_missing_policy_batch_rejects_candidate(self) -> None:
        args = promotion_args()
        rows = [
            healthy_arm_row("baseline"),
            {
                **healthy_arm_row("candidate"),
                "answer_contract_config_weight_mean": 0.25,
                "answer_contract_config_expected_update_steps_mean": 4.0,
                "answer_contract_oracle_rows_mean": 16.0,
                "answer_contract_tokens_mean": 128.0,
                "answer_contract_missing_policy_batch_count_mean": 2.0,
                "answer_contract_policy_batch_present_fraction_mean": 0.5,
            },
        ]

        gated = ANALYZE.add_gate_decisions(rows, args)
        candidate = next(row for row in gated if row["arm"] == "candidate")

        self.assertEqual(candidate["decision"], "reject")
        self.assertIn(
            "answer_contract_policy_batch_missing", candidate["fail_reasons"]
        )
        self.assertIn("answer_contract_policy_batch_partial", candidate["fail_reasons"])

    def test_recovery_missing_policy_batch_rejects_candidate(self) -> None:
        args = promotion_args()
        rows = [
            healthy_arm_row("baseline"),
            {
                **healthy_arm_row("candidate"),
                "recovery_weight_mean": 0.25,
                "recovery_rows_mean": 16.0,
                "recovery_missing_policy_batch_count_mean": 2.0,
                "recovery_policy_batch_present_fraction_mean": 0.5,
            },
        ]

        gated = ANALYZE.add_gate_decisions(rows, args)
        candidate = next(row for row in gated if row["arm"] == "candidate")

        self.assertEqual(candidate["decision"], "reject")
        self.assertIn("recovery_policy_batch_missing", candidate["fail_reasons"])

    def test_rollout_imitation_weight_without_accepted_rows_rejects_candidate(self) -> None:
        args = promotion_args()
        rows = [
            healthy_arm_row("baseline"),
            {
                **healthy_arm_row("candidate"),
                "rollout_imitation_weight_mean": 0.02,
                "rollout_imitation_generated_rows_mean": 8.0,
                "rollout_imitation_candidate_rows_mean": 2.0,
                "rollout_imitation_accepted_rows_mean": 0.0,
                "rollout_imitation_health_gate_passed_fraction_mean": 0.0,
            },
        ]

        gated = ANALYZE.add_gate_decisions(rows, args)
        candidate = next(row for row in gated if row["arm"] == "candidate")

        self.assertEqual(candidate["decision"], "reject")
        self.assertIn("rollout_imitation_health_gate_blocked", candidate["fail_reasons"])
        self.assertIn("rollout_imitation_inactive", candidate["fail_reasons"])

    def test_immature_candidate_is_held_not_rejected(self) -> None:
        args = promotion_args()
        args.min_mature_iters = 1024
        rows = [
            {**healthy_arm_row("baseline"), "max_iters_mean": 256.0},
            {
                **healthy_arm_row("candidate"),
                "max_iters_mean": 256.0,
                "ruliad_schema_wrong_last_mean": 0.75,
                "completion_health_last_mean": 0.10,
            },
        ]

        gated = ANALYZE.add_gate_decisions(rows, args)
        candidate = next(row for row in gated if row["arm"] == "candidate")
        baseline = next(row for row in gated if row["arm"] == "baseline")
        summary = ANALYZE.validation_summary(gated, args.baseline_arm)

        self.assertEqual(candidate["decision"], "hold")
        self.assertEqual(candidate["mature_enough"], 0.0)
        self.assertIn("insufficient_mature_iters", candidate["fail_reasons"])
        self.assertEqual(baseline["decision"], "control")
        self.assertEqual(summary["status"], "insufficient_mature_evidence")
        self.assertEqual(summary["mature_arm_count"], 0)
        self.assertEqual(summary["unhealthy_control_count"], 0)

    def test_peak_memory_regression_without_raw_gain_rejects_candidate(self) -> None:
        args = promotion_args()
        rows = [
            {**healthy_arm_row("baseline"), "peak_used_mb_mean": 10_000.0},
            {
                **healthy_arm_row("candidate"),
                "peak_used_mb_mean": 25_000.0,
                "ruliad_verifier_last_mean": 0.25,
                "raw_completion_verifier_rate_mean": 0.25,
            },
        ]

        gated = ANALYZE.add_gate_decisions(rows, args)
        candidate = next(row for row in gated if row["arm"] == "candidate")

        self.assertEqual(candidate["decision"], "reject")
        self.assertGreater(candidate["peak_memory_ratio"], args.max_peak_memory_ratio)
        self.assertIn("memory_regression_without_raw_gain", candidate["fail_reasons"])

    def test_peak_memory_regression_allowed_when_raw_verifier_improves(self) -> None:
        args = promotion_args()
        rows = [
            {
                **healthy_arm_row("baseline"),
                "peak_used_mb_mean": 10_000.0,
                "ruliad_verifier_last_mean": 0.25,
                "raw_completion_verifier_rate_mean": 0.25,
            },
            {
                **healthy_arm_row("candidate"),
                "peak_used_mb_mean": 25_000.0,
                "ruliad_verifier_last_mean": 0.3125,
                "raw_completion_verifier_rate_mean": 0.3125,
            },
        ]

        gated = ANALYZE.add_gate_decisions(rows, args)
        candidate = next(row for row in gated if row["arm"] == "candidate")

        self.assertGreater(candidate["peak_memory_ratio"], args.max_peak_memory_ratio)
        self.assertNotIn("memory_regression_without_raw_gain", candidate["fail_reasons"])

    def test_mature_verifier_zero_candidate_rejected_even_without_regression(self) -> None:
        args = promotion_args()
        baseline = {
            **healthy_arm_row("baseline"),
            "ruliad_verifier_last_mean": 0.0,
            "ruliad_semantic_last_mean": 0.0,
            "ruliad_partial_last_mean": 0.0,
            "raw_completion_verifier_rate_mean": 0.0,
            "raw_completion_semantic_rate_mean": 0.0,
            "raw_completion_partial_rate_mean": 0.0,
        }
        candidate = {
            **healthy_arm_row("candidate"),
            "valid_teacher_ce_last_mean": 0.9,
            "ruliad_verifier_last_mean": 0.0,
            "ruliad_semantic_last_mean": 0.0,
            "ruliad_partial_last_mean": 0.0,
            "raw_completion_verifier_rate_mean": 0.0,
            "raw_completion_semantic_rate_mean": 0.0,
            "raw_completion_partial_rate_mean": 0.0,
            "raw_completion_quality_mean_mean": 0.9,
        }

        gated = ANALYZE.add_gate_decisions([baseline, candidate], args)
        candidate_row = next(row for row in gated if row["arm"] == "candidate")

        self.assertEqual(candidate_row["decision"], "reject")
        self.assertIn("verifier_floor", candidate_row["fail_reasons"])
        self.assertIn("semantic_floor", candidate_row["fail_reasons"])
        self.assertIn("partial_floor", candidate_row["fail_reasons"])
        self.assertIn("raw_completion_verifier_floor", candidate_row["fail_reasons"])

    def test_contract_probe_distinguishes_free_run_gap_from_value_failure(self) -> None:
        args = promotion_args()
        free_run_gap = {
            **healthy_arm_row("free-run-gap"),
            "ruliad_verifier_last_mean": 0.0,
            "ruliad_semantic_last_mean": 0.0,
            "raw_completion_verifier_rate_mean": 0.0,
            "raw_completion_semantic_rate_mean": 0.0,
            "contract_probe_verifier_last_mean": 0.30,
            "contract_probe_answer_field_accuracy_last_mean": 0.45,
            "contract_probe_completion_health_last_mean": 0.70,
            "contract_probe_verifier_delta_mean": 0.30,
            "contract_probe_answer_field_delta_mean": 0.25,
            "contract_probe_completion_delta_mean": 0.30,
        }
        value_failure = {
            **healthy_arm_row("value-failure"),
            "ruliad_verifier_last_mean": 0.0,
            "ruliad_semantic_last_mean": 0.0,
            "raw_completion_verifier_rate_mean": 0.0,
            "raw_completion_semantic_rate_mean": 0.0,
            "contract_probe_verifier_last_mean": 0.0,
            "contract_probe_answer_field_accuracy_last_mean": 0.0,
            "contract_probe_completion_health_last_mean": 0.05,
            "contract_probe_verifier_delta_mean": 0.0,
            "contract_probe_answer_field_delta_mean": -0.2,
            "contract_probe_completion_delta_mean": -0.2,
        }

        gated = ANALYZE.add_gate_decisions(
            [healthy_arm_row("baseline"), free_run_gap, value_failure],
            args,
        )
        gap_row = next(row for row in gated if row["arm"] == "free-run-gap")
        failure_row = next(row for row in gated if row["arm"] == "value-failure")

        self.assertEqual(gap_row["decision"], "reject")
        self.assertIn("free_run_contract_gap", gap_row["fail_reasons"])
        self.assertNotIn("contract_value_failure", gap_row["fail_reasons"])
        self.assertEqual(failure_row["decision"], "reject")
        self.assertIn("contract_probe_verifier_floor", failure_row["fail_reasons"])
        self.assertIn("contract_probe_field_floor", failure_row["fail_reasons"])
        self.assertIn("contract_value_failure", failure_row["fail_reasons"])

    def test_latent_step_selector_gap_is_reported(self) -> None:
        args = promotion_args()
        candidate = {
            **healthy_arm_row("candidate"),
            "ruliad_verifier_last_mean": 0.0,
            "ruliad_semantic_last_mean": 0.0,
            "raw_completion_verifier_rate_mean": 0.0,
            "raw_completion_semantic_rate_mean": 0.0,
            "best_eval_steps_mean": 4.0,
            "best_eval_verifier_mean": 0.25,
            "best_eval_completion_mean": 0.7,
            "best_eval_verifier_delta_mean": 0.25,
            "best_eval_completion_delta_mean": 0.2,
        }

        baseline = {
            **healthy_arm_row("baseline"),
            "best_eval_steps_mean": 2.0,
            "best_eval_verifier_mean": 0.10,
            "best_eval_completion_mean": 0.4,
        }

        gated = ANALYZE.add_gate_decisions([baseline, candidate], args)
        candidate_row = next(row for row in gated if row["arm"] == "candidate")

        self.assertEqual(candidate_row["decision"], "reject")
        self.assertIn("latent_step_selector_needed", candidate_row["fail_reasons"])
        self.assertGreater(candidate_row["best_eval_verifier_baseline_delta"], 0.0)

    def test_extra_latent_step_collapse_is_reported(self) -> None:
        args = promotion_args()
        candidate = {
            **healthy_arm_row("candidate"),
            "extra_eval_step_count_mean": 3.0,
            "extra_eval_min_verifier_delta_mean": -0.25,
            "extra_eval_min_completion_delta_mean": -0.40,
            "extra_eval_max_malformed_delta_mean": 0.50,
        }

        gated = ANALYZE.add_gate_decisions([healthy_arm_row("baseline"), candidate], args)
        candidate_row = next(row for row in gated if row["arm"] == "candidate")

        self.assertEqual(candidate_row["decision"], "reject")
        self.assertIn("extra_step_verifier_collapse", candidate_row["fail_reasons"])
        self.assertIn("extra_step_completion_collapse", candidate_row["fail_reasons"])
        self.assertIn("extra_step_malformed_collapse", candidate_row["fail_reasons"])

    def test_latent_eval_trajectory_instability_is_reported(self) -> None:
        args = promotion_args()
        candidate = {
            **healthy_arm_row("candidate"),
            "latent_eval_final_ce_delta_last_mean": 8.0,
            "latent_eval_final_ce_violation_last_mean": 1.0,
            "latent_eval_final_entropy_bits_last_mean": 0.01,
            "latent_eval_final_delta_rms_last_mean": 64.0,
        }

        gated = ANALYZE.add_gate_decisions([healthy_arm_row("baseline"), candidate], args)
        candidate_row = next(row for row in gated if row["arm"] == "candidate")

        self.assertEqual(candidate_row["decision"], "reject")
        self.assertIn("latent_eval_ce_explosion", candidate_row["fail_reasons"])
        self.assertIn("latent_eval_monotonic_violation", candidate_row["fail_reasons"])
        self.assertIn("latent_eval_entropy_collapse", candidate_row["fail_reasons"])
        self.assertIn("latent_eval_delta_explosion", candidate_row["fail_reasons"])

    def test_latent_extra_eval_trajectory_instability_is_reported(self) -> None:
        args = promotion_args()
        candidate = {
            **healthy_arm_row("candidate"),
            "latent_extra_eval_max_ce_delta_last_mean": 128.0,
            "latent_extra_eval_max_ce_violation_last_mean": 1.0,
            "latent_extra_eval_min_entropy_bits_last_mean": 0.0,
            "latent_extra_eval_max_delta_rms_last_mean": 256.0,
        }

        gated = ANALYZE.add_gate_decisions([healthy_arm_row("baseline"), candidate], args)
        candidate_row = next(row for row in gated if row["arm"] == "candidate")

        self.assertEqual(candidate_row["decision"], "reject")
        self.assertIn("latent_extra_eval_ce_explosion", candidate_row["fail_reasons"])
        self.assertIn(
            "latent_extra_eval_monotonic_violation", candidate_row["fail_reasons"]
        )
        self.assertIn("latent_extra_eval_entropy_collapse", candidate_row["fail_reasons"])
        self.assertIn("latent_extra_eval_delta_explosion", candidate_row["fail_reasons"])

    def test_mature_candidate_without_raw_probe_is_rejected(self) -> None:
        args = promotion_args()
        candidate = {
            **healthy_arm_row("candidate"),
            "raw_completion_rows_mean": 0.0,
            "raw_completion_quality_mean_mean": 1.0,
            "raw_completion_verifier_rate_mean": 0.25,
            "raw_completion_semantic_rate_mean": 0.25,
            "raw_completion_partial_rate_mean": 0.5,
        }

        gated = ANALYZE.add_gate_decisions([healthy_arm_row("baseline"), candidate], args)
        candidate_row = next(row for row in gated if row["arm"] == "candidate")

        self.assertEqual(candidate_row["decision"], "reject")
        self.assertIn("raw_completion_probe_too_small", candidate_row["fail_reasons"])

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

    def test_field_value_collapse_rejects_candidate(self) -> None:
        args = promotion_args()
        candidate = {
            **healthy_arm_row("candidate"),
            "raw_completion_expected_answer_distinct_fraction_mean": 0.8,
            "raw_completion_actual_answer_distinct_fraction_mean": 0.8,
            "raw_completion_field_value_distinct_ratio_mean": 0.2,
            "raw_completion_actual_field_value_dominant_fraction_mean": 0.9,
        }

        gated = ANALYZE.add_gate_decisions([healthy_arm_row("baseline"), candidate], args)
        candidate_row = next(row for row in gated if row["arm"] == "candidate")

        self.assertEqual(candidate_row["decision"], "reject")
        self.assertIn("raw_completion_field_value_collapse", candidate_row["fail_reasons"])
        self.assertIn("raw_completion_field_value_dominance", candidate_row["fail_reasons"])

    def test_prompt_schema_value_collapse_rejects_candidate(self) -> None:
        args = promotion_args()
        candidate = {
            **healthy_arm_row("candidate"),
            "prompt_schema_completion_rows_mean": 32.0,
            "prompt_schema_completion_quality_mean_mean": 0.1,
            "prompt_schema_completion_verifier_rate_mean": 0.0,
            "prompt_schema_completion_semantic_rate_mean": 0.0,
            "prompt_schema_completion_partial_rate_mean": 0.0,
            "prompt_schema_completion_schema_wrong_rate_mean": 1.0,
            "prompt_schema_completion_malformed_rate_mean": 0.0,
            "prompt_schema_completion_missing_rate_mean": 0.0,
            "prompt_schema_completion_expected_answer_distinct_fraction_mean": 0.8,
            "prompt_schema_completion_actual_answer_distinct_fraction_mean": 0.05,
            "prompt_schema_completion_field_value_distinct_ratio_mean": 0.1,
            "prompt_schema_completion_actual_field_value_dominant_fraction_mean": 0.95,
        }

        gated = ANALYZE.add_gate_decisions([healthy_arm_row("baseline"), candidate], args)
        candidate_row = next(row for row in gated if row["arm"] == "candidate")

        self.assertEqual(candidate_row["decision"], "reject")
        self.assertIn(
            "prompt_schema_completion_verifier_floor", candidate_row["fail_reasons"]
        )
        self.assertIn(
            "prompt_schema_completion_schema_wrong_high", candidate_row["fail_reasons"]
        )
        self.assertIn(
            "prompt_schema_completion_quality_collapse", candidate_row["fail_reasons"]
        )
        self.assertIn(
            "prompt_schema_completion_answer_collapse", candidate_row["fail_reasons"]
        )
        self.assertIn(
            "prompt_schema_completion_field_value_collapse",
            candidate_row["fail_reasons"],
        )
        self.assertIn(
            "prompt_schema_completion_field_value_dominance",
            candidate_row["fail_reasons"],
        )

    def test_family_schema_leakage_rejects_candidate(self) -> None:
        args = promotion_args()
        candidate = {
            **healthy_arm_row("candidate"),
            "raw_completion_worst_family_verifier_rate_mean": 0.0,
            "raw_completion_worst_family_partial_rate_mean": 0.0,
            "raw_completion_worst_family_field_accuracy_mean": 0.0,
            "raw_completion_max_family_schema_key_mismatch_rate_mean": 1.0,
            "raw_completion_max_family_answer_dominant_fraction_mean": 1.0,
        }

        gated = ANALYZE.add_gate_decisions([healthy_arm_row("baseline"), candidate], args)
        candidate_row = next(row for row in gated if row["arm"] == "candidate")

        self.assertEqual(candidate_row["decision"], "reject")
        self.assertIn("raw_completion_family_verifier_floor", candidate_row["fail_reasons"])
        self.assertIn("raw_completion_family_partial_floor", candidate_row["fail_reasons"])
        self.assertIn("raw_completion_family_field_floor", candidate_row["fail_reasons"])
        self.assertIn("raw_completion_schema_key_leakage", candidate_row["fail_reasons"])
        self.assertIn("raw_completion_family_answer_attractor", candidate_row["fail_reasons"])

    def test_field_binding_objective_inactive_rejects_candidate(self) -> None:
        args = promotion_args()
        candidate = {
            **healthy_arm_row("candidate"),
            "field_binding_config_weight_mean": 0.01,
            "field_binding_config_expected_update_steps_mean": 4,
            "field_binding_contrast_pairs_mean": 0,
        }

        gated = ANALYZE.add_gate_decisions([healthy_arm_row("baseline"), candidate], args)
        candidate_row = next(row for row in gated if row["arm"] == "candidate")

        self.assertEqual(candidate_row["decision"], "reject")
        self.assertIn("field_binding_objective_inactive", candidate_row["fail_reasons"])

    def test_field_binding_rank_metrics_missing_rejects_candidate(self) -> None:
        args = promotion_args()
        candidate = {
            **healthy_arm_row("candidate"),
            "field_binding_config_weight_mean": 0.01,
            "field_binding_config_expected_update_steps_mean": 4,
            "field_binding_config_expected_rank_metric_steps_mean": 2,
            "field_binding_contrast_pairs_mean": 8,
            "field_binding_rank_metric_tokens_mean": 0,
        }

        gated = ANALYZE.add_gate_decisions([healthy_arm_row("baseline"), candidate], args)
        candidate_row = next(row for row in gated if row["arm"] == "candidate")

        self.assertEqual(candidate_row["decision"], "reject")
        self.assertIn("field_binding_rank_metrics_missing", candidate_row["fail_reasons"])

    def test_field_binding_weak_rank_metrics_reject_candidate(self) -> None:
        args = promotion_args()
        candidate = {
            **healthy_arm_row("candidate"),
            "field_binding_config_weight_mean": 0.01,
            "field_binding_config_expected_update_steps_mean": 4,
            "field_binding_config_expected_rank_metric_steps_mean": 2,
            "field_binding_contrast_pairs_mean": 8,
            "field_binding_rank_metric_tokens_mean": 32,
            "field_binding_positive_token_fraction_mean": 0.50,
            "field_binding_exact_pair_rank_fraction_mean": 0.25,
        }

        gated = ANALYZE.add_gate_decisions([healthy_arm_row("baseline"), candidate], args)
        candidate_row = next(row for row in gated if row["arm"] == "candidate")

        self.assertEqual(candidate_row["decision"], "reject")
        self.assertIn("field_binding_positive_rank_weak", candidate_row["fail_reasons"])
        self.assertIn("field_binding_pair_rank_weak", candidate_row["fail_reasons"])

    def test_field_binding_sequence_rank_metrics_are_required_when_enabled(self) -> None:
        args = promotion_args()
        candidate = {
            **healthy_arm_row("candidate"),
            "field_binding_config_weight_mean": 0.01,
            "field_binding_config_pair_weight_mean": 1.0,
            "field_binding_config_expected_update_steps_mean": 4,
            "field_binding_config_expected_rank_metric_steps_mean": 2,
            "field_binding_contrast_pairs_mean": 8,
            "field_binding_rank_metric_tokens_mean": 8,
            "field_binding_positive_token_fraction_mean": 1.0,
            "field_binding_exact_pair_rank_fraction_mean": 1.0,
            "field_binding_sequence_rank_metric_pairs_mean": 0,
        }

        gated = ANALYZE.add_gate_decisions([healthy_arm_row("baseline"), candidate], args)
        candidate_row = next(row for row in gated if row["arm"] == "candidate")

        self.assertEqual(candidate_row["decision"], "reject")
        self.assertIn(
            "field_binding_sequence_rank_metrics_missing", candidate_row["fail_reasons"]
        )

    def test_field_binding_weak_sequence_rank_rejects_candidate(self) -> None:
        args = promotion_args()
        candidate = {
            **healthy_arm_row("candidate"),
            "field_binding_config_weight_mean": 0.01,
            "field_binding_config_pair_weight_mean": 1.0,
            "field_binding_config_expected_update_steps_mean": 4,
            "field_binding_config_expected_rank_metric_steps_mean": 2,
            "field_binding_contrast_pairs_mean": 8,
            "field_binding_rank_metric_tokens_mean": 8,
            "field_binding_positive_token_fraction_mean": 1.0,
            "field_binding_exact_pair_rank_fraction_mean": 1.0,
            "field_binding_sequence_rank_metric_pairs_mean": 8,
            "field_binding_positive_sequence_fraction_mean": 0.25,
        }

        gated = ANALYZE.add_gate_decisions([healthy_arm_row("baseline"), candidate], args)
        candidate_row = next(row for row in gated if row["arm"] == "candidate")

        self.assertEqual(candidate_row["decision"], "reject")
        self.assertIn("field_binding_sequence_rank_weak", candidate_row["fail_reasons"])

    def test_generated_attractor_replay_inactive_rejects_candidate(self) -> None:
        args = promotion_args()
        candidate = {
            **healthy_arm_row("candidate"),
            "generated_attractor_config_capacity_mean": 128,
            "generated_attractor_observed_rows_mean": 16,
            "generated_attractor_recorded_rows_mean": 4,
            "generated_attractor_active_count_mean": 0,
            "generated_attractor_selected_candidate_rows_mean": 0,
            "contrast_generated_attractor_negative_completion_rows_mean": 0,
        }

        gated = ANALYZE.add_gate_decisions([healthy_arm_row("baseline"), candidate], args)
        candidate_row = next(row for row in gated if row["arm"] == "candidate")

        self.assertEqual(candidate_row["decision"], "reject")
        self.assertIn("generated_attractor_replay_inactive", candidate_row["fail_reasons"])

    def test_generated_attractor_replay_disabled_does_not_gate_candidate(self) -> None:
        args = promotion_args()
        candidate = {
            **healthy_arm_row("candidate"),
            "generated_attractor_config_capacity_mean": 0,
            "generated_attractor_observed_rows_mean": 0,
            "generated_attractor_recorded_rows_mean": 0,
            "generated_attractor_active_count_mean": 0,
            "generated_attractor_selected_candidate_rows_mean": 0,
            "contrast_generated_attractor_negative_completion_rows_mean": 0,
        }

        gated = ANALYZE.add_gate_decisions([healthy_arm_row("baseline"), candidate], args)
        candidate_row = next(row for row in gated if row["arm"] == "candidate")

        self.assertEqual(candidate_row["decision"], "promote")
        self.assertNotIn("generated_attractor_replay_inactive", candidate_row["fail_reasons"])

    def test_generated_attractor_replay_consumed_allows_candidate(self) -> None:
        args = promotion_args()
        candidate = {
            **healthy_arm_row("candidate"),
            "generated_attractor_config_capacity_mean": 128,
            "generated_attractor_observed_rows_mean": 16,
            "generated_attractor_recorded_rows_mean": 4,
            "generated_attractor_active_count_mean": 2,
            "generated_attractor_distinct_answer_count_mean": 2,
            "generated_attractor_dominant_answer_fraction_mean": 0.5,
            "generated_attractor_selected_candidate_rows_mean": 2,
            "contrast_generated_attractor_negative_completion_rows_mean": 2,
        }

        gated = ANALYZE.add_gate_decisions([healthy_arm_row("baseline"), candidate], args)
        candidate_row = next(row for row in gated if row["arm"] == "candidate")

        self.assertEqual(candidate_row["decision"], "promote")
        self.assertNotIn("generated_attractor_replay_inactive", candidate_row["fail_reasons"])
        self.assertNotIn("generated_attractor_replay_unselected", candidate_row["fail_reasons"])
        self.assertNotIn("generated_attractor_replay_unconsumed", candidate_row["fail_reasons"])

    def test_generated_attractor_dominance_rejects_candidate(self) -> None:
        args = promotion_args()
        candidate = {
            **healthy_arm_row("candidate"),
            "generated_attractor_config_capacity_mean": 128,
            "generated_attractor_config_min_distinct_answers_mean": 2,
            "generated_attractor_config_max_dominant_fraction_mean": 0.5,
            "generated_attractor_observed_rows_mean": 32,
            "generated_attractor_recorded_rows_mean": 16,
            "generated_attractor_active_count_mean": 4,
            "generated_attractor_distinct_answer_count_mean": 4,
            "generated_attractor_dominant_answer_fraction_mean": 0.75,
            "generated_attractor_selected_candidate_rows_mean": 8,
            "contrast_generated_attractor_negative_completion_rows_mean": 8,
        }

        gated = ANALYZE.add_gate_decisions([healthy_arm_row("baseline"), candidate], args)
        candidate_row = next(row for row in gated if row["arm"] == "candidate")

        self.assertEqual(candidate_row["decision"], "reject")
        self.assertIn("generated_attractor_dominance_high", candidate_row["fail_reasons"])


if __name__ == "__main__":
    unittest.main()
