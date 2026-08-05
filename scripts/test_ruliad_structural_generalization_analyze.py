#!/usr/bin/env python3
"""Tests for ruliad_structural_generalization_analyze.py."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("ruliad_structural_generalization_analyze.py")
SPEC = importlib.util.spec_from_file_location("ruliad_structural_generalization_analyze", SCRIPT)
assert SPEC and SPEC.loader
ANALYZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZE)


def trial(arm: str, seed: int) -> dict[str, object]:
    contract, mode = ANALYZE.ARM_CONTRACTS[arm]
    candidate = arm in ANALYZE.CANDIDATE_ARMS
    orbit = "orbit" in arm
    hybrid = arm in {
        "structural_bc_paired_dagger_marginal025",
        "structural_bc_paired_dagger_orbit_marginal025",
    }
    static = arm in {
        "structural_energy_static025",
        "structural_energy_head_only025",
        "structural_energy_head_only_fullrate100",
        "structural_energy_fullrate100",
        "structural_energy_value_binding025",
        "structural_semantic_static025",
        "structural_semantic_language_head_only025",
        "structural_semantic_static_dense025",
        "structural_semantic_static_prefix025",
        "structural_semantic_static_marginal025",
        "structural_static025",
        "structural_static_marginal025",
        "structural_static_orbit_marginal025",
        "structural_static_orbit_worst_marginal025",
    }
    policy_mode = "paired_dagger,static_expert" if hybrid else ("static_expert" if static else "dagger")
    configured_mode = "static_then_paired_dagger" if hybrid else ("static_expert" if static else "dagger")
    marginal = arm in {
        "structural_semantic_static_marginal025",
        "structural_static_marginal025",
        "structural_static_orbit_marginal025",
        "structural_static_orbit_worst_marginal025",
        "structural_dagger_marginal025",
        "structural_bc_paired_dagger_marginal025",
        "structural_bc_paired_dagger_orbit_marginal025",
    }
    prefix = arm == "structural_semantic_static_prefix025"
    energy = arm in ANALYZE.SEMANTIC_ENERGY_ARMS
    counterfactual = arm in ANALYZE.COUNTERFACTUAL_TARGET_ARMS
    semantic = arm.startswith("structural_semantic") or energy
    return {
        "arm": arm,
        "seed": seed,
        "backend": "cuda",
        "status": "ok",
        "max_iters": 1024,
        "generalization_contract": contract,
        "proof_action_answer_contract": ANALYZE.ARM_ANSWER_CONTRACTS[arm],
        "expected_answer_contract": ANALYZE.ARM_ANSWER_CONTRACTS[arm],
        "expected_contract": contract,
        "expected_mode": mode,
        "mode_telemetry_present": True,
        "audited_prompt_count": 16,
        "semantic_prompt_leak_count": 0,
        "probe_count": 4,
        "policy_probe_count": 4,
        "correctness_verifier": 0.42,
        "correctness_partial": 0.72,
        "correctness_constrained_items": 128,
        "correctness_constrained_equivalent_top1": (
            0.80 if semantic else 0.62 + (0.06 if candidate else 0.0)
        ),
        "correctness_constrained_preferred_top1": 0.58 + (0.06 if candidate else 0.0),
        "correctness_constrained_equivalent_nll": 0.8 - (0.1 if candidate else 0.0),
        "correctness_constrained_valid_invalid_margin": 0.3 + (0.1 if candidate else 0.0),
        "correctness_constrained_canonical_equivalent_top1": (
            0.80 if semantic else 0.60 + (0.06 if candidate else 0.0)
        ),
        "correctness_constrained_canonical_preferred_top1": 0.56
        + (0.06 if candidate else 0.0),
        "correctness_constrained_canonical_equivalent_nll": 0.82
        - (0.10 if candidate else 0.0),
        "correctness_constrained_canonical_valid_invalid_margin": 0.28
        + (0.10 if candidate else 0.0),
        "correctness_constrained_worst_presentation_equivalent_top1": (
            0.78 if semantic else 0.52 + (0.05 if candidate else 0.0)
        ),
        "correctness_constrained_worst_presentation_equivalent_nll": 0.95
        - (0.08 if candidate else 0.0),
        "correctness_constrained_worst_presentation_valid_invalid_margin": 0.18
        + (0.08 if candidate else 0.0),
        "correctness_constrained_orbit_js_divergence": 0.05
        - (0.02 if candidate else 0.0),
        "correctness_constrained_orbit_top1_consensus": (
            0.95 if semantic else 0.85 + (0.05 if candidate else 0.0)
        ),
        "correctness_constrained_complete_orbit_items": 128,
        "correctness_constrained_presentation_rows": 512,
        "correctness_constrained_presentation_equivalent_top1": 0.70
        + (0.06 if candidate else 0.0),
        "correctness_constrained_presentation_preferred_top1": 0.64
        + (0.06 if candidate else 0.0),
        "correctness_constrained_context_swap_items": 128,
        "correctness_constrained_context_swap_equivalent_top1": 0.30,
        "correctness_constrained_context_swap_equivalent_nll": 1.35,
        "correctness_constrained_context_swap_top1_change": 0.65,
        "correctness_constrained_context_swap_equivalent_probability_drop": 0.20,
        "correctness_constrained_context_swap_js_divergence": 0.08,
        "correctness_constrained_counterfactual_target_items": 128,
        "correctness_constrained_counterfactual_target_equivalent_top1": 0.70,
        "correctness_constrained_counterfactual_target_equivalent_nll": 0.72,
        "correctness_constrained_counterfactual_target_top1_change": 0.68,
        "correctness_constrained_counterfactual_target_equivalent_probability_gain": 0.22,
        "correctness_constrained_counterfactual_target_js_divergence": 0.09,
        "correctness_constrained_symmetry_balanced": 1.0,
        "correctness_constrained_symmetry_orbit_averaged": 1.0,
        "completion_health": 0.75,
        "actual_answer_distinct_fraction": 0.03125,
        "expected_answer_distinct_fraction": 0.03125,
        "actual_answer_dominant_fraction": 0.30,
        "schema_wrong": 0.20,
        "malformed": 0.0,
        "correctness_drop_from_best": 0.02,
        "policy_solve": 0.20,
        "policy_goal_completion": 0.30 + (0.10 if candidate else 0.0),
        "policy_valid_action": 1.0,
        "policy_top1_expert": 0.45 + (0.10 if candidate else 0.0),
        "policy_runtime_gate_passed": 1.0,
        "policy_candidate_symmetry_balanced": 1.0,
        "policy_candidate_symmetry_orbit_averaged": 1.0,
        "policy_solve_drop_from_best": 0.02,
        "valid_ce": 0.4,
        "model_tokens_per_second": 800.0 if candidate else 1000.0,
        "wall_tokens_per_second": 680.0 if candidate else 850.0,
        "model_duty_fraction": 0.70,
        "dataloader_cpu_thread_fraction": 0.10,
        "dataloader_foreground_wait_fraction": 0.01,
        "gpu_active_util_mean": 93.0,
        "gpu_active_power_mean": 52.0,
        "gpu_high_util_fraction": 0.95,
        "gpu_low_util_fraction": 0.01,
        "gpu_max_consecutive_sub80_samples": 2.0,
        "gpu_max_consecutive_idle_samples": 0.0,
        "gpu_active_power_cv": 0.08,
        "elapsed_seconds": 12.0 if candidate else 10.0,
        "dagger_calls": 4 if candidate else 0,
        "dagger_expert_rows": 64 if candidate else 0,
        "dagger_static_expert_rows": 32 if hybrid else (64 if static else 0),
        "dagger_on_policy_expert_rows": 32 if hybrid else (0 if static else 64),
        "dagger_paired_static_expert_rows": 32 if hybrid else 0,
        "dagger_paired_on_policy_expert_rows": 32 if hybrid else 0,
        "dagger_telemetry_version_min": (
            19
            if arm
            in {
                "structural_energy_head_only025",
                "structural_energy_head_only_fullrate100",
                "structural_semantic_language_head_only025",
            }
            else 18
            if energy
            else 16
        ) if candidate else 0,
        "dagger_objective": (
            "semantic_sequence_energy_counterfactual_v1"
            if energy
            else "candidate_normalized_counterfactual_v1"
            if arm == "structural_semantic_language_head_only025"
            else "prefix_conditional_equivalent_v1"
            if prefix
            else "vocabulary_marginal_equivalent_v1"
            if marginal
            else "candidate_normalized_equivalent_v1"
        ) if candidate else "",
        "dagger_answer_contract": (
            "semantic_step"
            if energy
            else ANALYZE.ARM_ANSWER_CONTRACTS[arm]
            if candidate
            else ""
        ),
        "dagger_gradient_scope": (
            "score_head_only"
            if arm
            in {
                "structural_energy_head_only025",
                "structural_energy_head_only_fullrate100",
            }
            else "language_head_only"
            if arm == "structural_semantic_language_head_only025"
            else "full_model"
            if candidate
            else ""
        ),
        "dagger_presentation_risk": (
            "worst"
            if arm == "structural_static_orbit_worst_marginal025"
            else "mean"
        ) if candidate else "",
        "dagger_configured_mode": configured_mode if candidate else "",
        "dagger_mode": policy_mode if candidate else "",
        "dagger_candidate_symmetry": (
            "cyclic_orbit_average"
            if orbit
            else ("balanced_rotation" if marginal or counterfactual else "canonical")
        ) if candidate else "",
        "dagger_presentations_per_state": 4.0 if orbit else 1.0,
        "dagger_presentation_rows_max": 32 if orbit else (16 if candidate else 0),
        "dagger_presentation_row_budget_min": 32 if candidate else 0,
        "dagger_semantic_row_budget_min": 8 if orbit else (32 if candidate else 0),
        "dagger_base_semantic_state_rows": 32 if counterfactual else (64 if candidate else 0),
        "dagger_counterfactual_semantic_state_rows": 32 if counterfactual else 0,
        "dagger_counterfactual_target_shortfall": 0,
        "dagger_configured_counterfactual_targets_per_state": 1 if counterfactual else 0,
        "dagger_candidate_targets_per_row": 4.0 if candidate else None,
        "dagger_equivalent_targets_per_row": 1.25 if candidate else None,
        "dagger_prefix_branch_rows": 24 if prefix else 0,
        "dagger_prefix_candidate_tokens": 48 if prefix else 0,
        "dagger_prefix_equivalent_tokens": 24 if prefix else 0,
        "dagger_model_visited_expert_rows": (
            0 if static else 32
        ) if candidate else 0,
        "dagger_rollout_depth_reached_max": (
            1 if static else 8
        ) if candidate else 0,
        "dagger_expert_index_entropy_bits": 1.95 if candidate else None,
        "dagger_expert_index_dominant_fraction": 0.28 if candidate else None,
    }


class StructuralGeneralizationAnalyzeTests(unittest.TestCase):
    def test_answer_contract_reads_nested_corpus_task_mix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.toml"
            corpus.write_text(
                '[source_selection.formal_task_mix]\n'
                'proof_action_answer_contract = "semantic_step"\n'
            )
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "training_config.json").write_text(
                json.dumps({"dataset": {"config": str(corpus)}})
            )
            self.assertEqual(
                ANALYZE.proof_action_answer_contract(run_dir),
                "semantic_step",
            )

    def test_validation_ce_reads_the_running_mean_not_the_last_minibatch(self) -> None:
        events = [
            {
                "type": "metric",
                "split": "valid",
                "name": "Teacher Forced Answer CE",
                "value": 0.9,
                "running_value": 0.4,
            }
        ]
        self.assertEqual(
            ANALYZE.metric_last(
                events,
                "Teacher Forced Answer CE",
                split="valid",
                running=True,
            ),
            0.4,
        )

    def test_matched_healthy_candidate_passes_directional_gate(self) -> None:
        trials = [
            trial(arm, seed)
            for arm in (*ANALYZE.BASELINE_ARMS, ANALYZE.DEFAULT_CANDIDATE_ARM)
            for seed in (1337, 2027, 9001)
        ]
        decision = ANALYZE.promotion_decision(trials, {1337, 2027, 9001}, 1024)
        self.assertTrue(decision["directional_promotion_passed"], decision["failures"])
        self.assertEqual(decision["evidence_class"], "matched_multiseed")

    def test_semantic_energy_candidate_binds_a_separate_policy_contract(self) -> None:
        for candidate_arm in sorted(ANALYZE.SEMANTIC_ENERGY_ARMS):
            with self.subTest(candidate_arm=candidate_arm):
                trials = [
                    trial(arm, seed)
                    for arm in (*ANALYZE.BASELINE_ARMS, candidate_arm)
                    for seed in (1337, 2027, 9001)
                ]
                decision = ANALYZE.promotion_decision(
                    trials,
                    {1337, 2027, 9001},
                    1024,
                    candidate_arm,
                )
                self.assertTrue(
                    decision["directional_promotion_passed"], decision["failures"]
                )

                broken = [dict(row) for row in trials]
                for row in broken:
                    if row["arm"] == candidate_arm:
                        row["dagger_answer_contract"] = "presentation_index"
                decision = ANALYZE.promotion_decision(
                    broken,
                    {1337, 2027, 9001},
                    1024,
                    candidate_arm,
                )
                self.assertIn(
                    f"{candidate_arm}/seed1337:policy_answer_contract",
                    decision["failures"],
                )
                if candidate_arm in {
                    "structural_energy_head_only025",
                    "structural_energy_head_only_fullrate100",
                }:
                    broken = [dict(row) for row in trials]
                    for row in broken:
                        if row["arm"] == candidate_arm:
                            row["dagger_gradient_scope"] = "full_model"
                    decision = ANALYZE.promotion_decision(
                        broken,
                        {1337, 2027, 9001},
                        1024,
                        candidate_arm,
                    )
                    self.assertIn(
                        f"{candidate_arm}/seed1337:head_only_gradient_scope_not_exercised",
                        decision["failures"],
                    )
                if candidate_arm == "structural_semantic_language_head_only025":
                    broken = [dict(row) for row in trials]
                    for row in broken:
                        if row["arm"] == candidate_arm:
                            row["dagger_gradient_scope"] = "full_model"
                    decision = ANALYZE.promotion_decision(
                        broken,
                        {1337, 2027, 9001},
                        1024,
                        candidate_arm,
                    )
                    self.assertIn(
                        f"{candidate_arm}/seed1337:language_head_only_gradient_scope_not_exercised",
                        decision["failures"],
                    )

    def test_invalid_context_swap_is_diagnostic_only(self) -> None:
        candidate_arm = "structural_energy_static025"
        trials = [
            trial(arm, seed)
            for arm in (*ANALYZE.BASELINE_ARMS, candidate_arm)
            for seed in (1337, 2027, 9001)
        ]
        for row in trials:
            if row["arm"] != candidate_arm:
                continue
            row["correctness_constrained_context_swap_equivalent_top1"] = row[
                "correctness_constrained_equivalent_top1"
            ]
            row["correctness_constrained_context_swap_top1_change"] = 0.0
            row["correctness_constrained_context_swap_equivalent_probability_drop"] = 0.0
            row["correctness_constrained_context_swap_js_divergence"] = 0.0
        decision = ANALYZE.promotion_decision(
            trials,
            {1337, 2027, 9001},
            1024,
            candidate_arm,
        )
        self.assertTrue(
            decision["typed_policy_promotion_passed"],
            decision["typed_policy_failures"],
        )
        self.assertFalse(
            any("context_swap" in failure for failure in decision["failures"])
        )

    def test_target_independent_energy_blocks_promotion(self) -> None:
        candidate_arm = "structural_energy_static025"
        trials = [
            trial(arm, seed)
            for arm in (*ANALYZE.BASELINE_ARMS, candidate_arm)
            for seed in (1337, 2027, 9001)
        ]
        for row in trials:
            if row["arm"] != candidate_arm:
                continue
            row["correctness_constrained_counterfactual_target_equivalent_top1"] = 0.0
            row["correctness_constrained_counterfactual_target_top1_change"] = 0.0
            row[
                "correctness_constrained_counterfactual_target_equivalent_probability_gain"
            ] = 0.0
            row["correctness_constrained_counterfactual_target_js_divergence"] = 0.0
        decision = ANALYZE.promotion_decision(
            trials,
            {1337, 2027, 9001},
            1024,
            candidate_arm,
        )
        self.assertFalse(decision["typed_policy_promotion_passed"])
        self.assertTrue(
            any(
                "counterfactual_target_preference_change" in failure
                for failure in decision["failures"]
            )
        )
        self.assertTrue(
            any(
                "counterfactual_target_js" in failure
                for failure in decision["failures"]
            )
        )

    def test_gpu_stats_separates_low_duty_from_power_variation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpu.csv"
            path.write_text(
                "timestamp, utilization.gpu [%], utilization.memory [%], power.draw [W], clocks.current.sm [MHz], temperature.gpu\n"
                "t0, 95 %, 40 %, 50 W, 2400 MHz, 55\n"
                "t1, 96 %, 42 %, 70 W, 2380 MHz, 57\n"
                "t2, 10 %, 2 %, 18 W, 900 MHz, 48\n"
                "t3, 95 %, 41 %, 55 W, 2420 MHz, 56\n"
            )
            (
                util,
                power,
                active_util,
                active_power,
                active_sm_clock,
                active_sm_clock_min,
                active_memory_util,
                active_temperature_max,
                high_util,
                low_util,
                sub80_streak,
                idle_streak,
                active_power_cv,
            ) = ANALYZE.gpu_stats(path)

        self.assertAlmostEqual(util, 74.0)
        self.assertAlmostEqual(power, 193.0 / 4.0)
        self.assertAlmostEqual(active_util, 286.0 / 3.0)
        self.assertAlmostEqual(active_power, 175.0 / 3.0)
        self.assertAlmostEqual(active_sm_clock, 2400.0)
        self.assertAlmostEqual(active_sm_clock_min, 2380.0)
        self.assertAlmostEqual(active_memory_util, 41.0)
        self.assertAlmostEqual(active_temperature_max, 57.0)
        self.assertAlmostEqual(high_util, 3.0 / 4.0)
        self.assertAlmostEqual(low_util, 1.0 / 4.0)
        self.assertEqual(sub80_streak, 1.0)
        self.assertEqual(idle_streak, 1.0)
        self.assertGreater(active_power_cv, 0.10)

    def test_semantic_candidate_requires_semantic_answer_contract(self) -> None:
        candidate_arm = "structural_semantic_ce"
        trials = [
            trial(arm, seed)
            for arm in (*ANALYZE.BASELINE_ARMS, candidate_arm)
            for seed in (1337, 2027, 9001)
        ]
        decision = ANALYZE.promotion_decision(
            trials,
            {1337, 2027, 9001},
            1024,
            candidate_arm,
        )
        self.assertTrue(decision["directional_promotion_passed"], decision["failures"])
        trials[-1]["proof_action_answer_contract"] = "presentation_index"
        rejected = ANALYZE.promotion_decision(
            trials,
            {1337, 2027, 9001},
            1024,
            candidate_arm,
        )
        self.assertFalse(rejected["directional_promotion_passed"])
        self.assertTrue(
            any(
                "proof_action_answer_contract_mismatch" in failure
                for failure in rejected["failures"]
            )
        )

    def test_typed_policy_and_free_generation_have_independent_promotion_results(self) -> None:
        candidate_arm = "structural_semantic_static025"
        trials = [
            trial(arm, seed)
            for arm in (*ANALYZE.BASELINE_ARMS, candidate_arm)
            for seed in (1337, 2027, 9001)
        ]
        for row in trials:
            if row["arm"] == candidate_arm:
                row["correctness_verifier"] = 0.04
                row["actual_answer_distinct_fraction"] = 0.01
                row["expected_answer_distinct_fraction"] = 0.25

        decision = ANALYZE.promotion_decision(
            trials,
            {1337, 2027, 9001},
            1024,
            candidate_arm,
        )
        self.assertTrue(
            decision["typed_policy_promotion_passed"],
            decision["typed_policy_failures"],
        )
        self.assertFalse(decision["free_generation_promotion_passed"])
        self.assertFalse(decision["directional_promotion_passed"])
        self.assertTrue(
            any(
                "free_action_answer_coverage" in failure
                for failure in decision["free_generation_failures"]
            )
        )

    def test_runtime_policy_gate_blocks_only_typed_promotion(self) -> None:
        candidate_arm = "structural_semantic_static025"
        trials = [
            trial(arm, seed)
            for arm in (*ANALYZE.BASELINE_ARMS, candidate_arm)
            for seed in (1337, 2027, 9001)
        ]
        for row in trials:
            if row["arm"] == candidate_arm:
                row["policy_runtime_gate_passed"] = 0.0

        decision = ANALYZE.promotion_decision(
            trials,
            {1337, 2027, 9001},
            1024,
            candidate_arm,
        )
        self.assertFalse(decision["typed_policy_promotion_passed"])
        self.assertTrue(
            decision["free_generation_promotion_passed"],
            decision["free_generation_failures"],
        )

    def test_semantic_static_marginal_compares_against_semantic_ce(self) -> None:
        comparison_arm = "structural_semantic_ce"
        candidate_arm = "structural_semantic_static_marginal025"
        trials = [
            trial(arm, seed)
            for arm in (*ANALYZE.BASELINE_ARMS, comparison_arm, candidate_arm)
            for seed in (1337, 2027, 9001)
        ]
        for row in trials:
            if row["arm"] == candidate_arm:
                row["policy_top1_expert"] = float(row["policy_top1_expert"]) + 0.10
                row["policy_goal_completion"] = float(row["policy_goal_completion"]) + 0.10
                row["correctness_constrained_equivalent_top1"] = (
                    float(row["correctness_constrained_equivalent_top1"]) + 0.06
                )
        decision = ANALYZE.promotion_decision(
            trials,
            {1337, 2027, 9001},
            1024,
            candidate_arm,
            comparison_arm,
        )
        self.assertTrue(decision["directional_promotion_passed"], decision["failures"])
        self.assertEqual(decision["comparison_arm"], comparison_arm)

    def test_semantic_prefix_candidate_uses_trie_objective_contract(self) -> None:
        candidate_arm = "structural_semantic_static_prefix025"
        row = trial(candidate_arm, 1337)
        self.assertEqual(row["dagger_objective"], "prefix_conditional_equivalent_v1")
        self.assertEqual(row["proof_action_answer_contract"], "semantic_step")

    def test_static_expert_candidate_uses_its_explicit_mode_contract(self) -> None:
        candidate_arm = "structural_static025"
        trials = [
            trial(arm, seed)
            for arm in (*ANALYZE.BASELINE_ARMS, candidate_arm)
            for seed in (1337, 2027, 9001)
        ]
        decision = ANALYZE.promotion_decision(
            trials,
            {1337, 2027, 9001},
            1024,
            candidate_arm,
        )
        self.assertTrue(decision["directional_promotion_passed"], decision["failures"])
        self.assertEqual(decision["candidate_arm"], candidate_arm)

    def test_marginal_candidate_requires_v8_balanced_symmetry_contract(self) -> None:
        candidate_arm = "structural_dagger_marginal025"
        trials = [
            trial(arm, seed)
            for arm in (*ANALYZE.BASELINE_ARMS, candidate_arm)
            for seed in (1337, 2027, 9001)
        ]
        decision = ANALYZE.promotion_decision(
            trials,
            {1337, 2027, 9001},
            1024,
            candidate_arm,
        )
        self.assertTrue(decision["directional_promotion_passed"], decision["failures"])
        trials[-1]["dagger_candidate_symmetry"] = "canonical"
        rejected = ANALYZE.promotion_decision(
            trials,
            {1337, 2027, 9001},
            1024,
            candidate_arm,
        )
        self.assertFalse(rejected["directional_promotion_passed"])
        self.assertTrue(
            any(
                "policy_candidate_symmetry_contract" in failure
                for failure in rejected["failures"]
            )
        )

    def test_promotion_rejects_canonical_only_evaluator(self) -> None:
        trials = [
            trial(arm, seed)
            for arm in (*ANALYZE.BASELINE_ARMS, ANALYZE.DEFAULT_CANDIDATE_ARM)
            for seed in (1337, 2027, 9001)
        ]
        trials[-1]["correctness_constrained_symmetry_balanced"] = 0.0
        trials[-1]["correctness_constrained_symmetry_orbit_averaged"] = 0.0
        trials[-1]["policy_candidate_symmetry_balanced"] = 0.0
        trials[-1]["policy_candidate_symmetry_orbit_averaged"] = 0.0
        decision = ANALYZE.promotion_decision(trials, {1337, 2027, 9001}, 1024)
        self.assertFalse(decision["directional_promotion_passed"])
        self.assertTrue(
            any(
                "candidate_symmetry_contract" in failure
                for failure in decision["failures"]
            )
        )

    def test_promotion_rejects_incomplete_same_item_orbit(self) -> None:
        trials = [
            trial(arm, seed)
            for arm in (*ANALYZE.BASELINE_ARMS, ANALYZE.DEFAULT_CANDIDATE_ARM)
            for seed in (1337, 2027, 9001)
        ]
        trials[-1]["correctness_constrained_complete_orbit_items"] = 127
        decision = ANALYZE.promotion_decision(trials, {1337, 2027, 9001}, 1024)
        self.assertFalse(decision["directional_promotion_passed"])
        self.assertTrue(
            any("incomplete_same_item_orbit" in failure for failure in decision["failures"])
        )

    def test_promotion_rejects_hidden_worst_presentation_regression(self) -> None:
        trials = [
            trial(arm, seed)
            for arm in (*ANALYZE.BASELINE_ARMS, ANALYZE.DEFAULT_CANDIDATE_ARM)
            for seed in (1337, 2027, 9001)
        ]
        for row in trials:
            if row["arm"] == ANALYZE.DEFAULT_CANDIDATE_ARM:
                row["correctness_constrained_worst_presentation_equivalent_top1"] = 0.50
        decision = ANALYZE.promotion_decision(trials, {1337, 2027, 9001}, 1024)
        self.assertFalse(decision["directional_promotion_passed"])
        self.assertTrue(
            any(
                "worst_presentation_top1_regression" in failure
                for failure in decision["failures"]
            )
        )

    def test_static_marginal_candidate_uses_balanced_vocabulary_contract(self) -> None:
        candidate_arm = "structural_static_marginal025"
        trials = [
            trial(arm, seed)
            for arm in (*ANALYZE.BASELINE_ARMS, candidate_arm)
            for seed in (1337, 2027, 9001)
        ]
        decision = ANALYZE.promotion_decision(
            trials,
            {1337, 2027, 9001},
            1024,
            candidate_arm,
        )
        self.assertTrue(decision["directional_promotion_passed"], decision["failures"])

    def test_orbit_candidate_requires_tensorized_semantic_contract(self) -> None:
        candidate_arm = "structural_static_orbit_marginal025"
        trials = [
            trial(arm, seed)
            for arm in (*ANALYZE.BASELINE_ARMS, candidate_arm)
            for seed in (1337, 2027, 9001)
        ]
        decision = ANALYZE.promotion_decision(
            trials,
            {1337, 2027, 9001},
            1024,
            candidate_arm,
        )
        self.assertTrue(decision["directional_promotion_passed"], decision["failures"])

        trials[-1]["dagger_presentations_per_state"] = 1.0
        rejected = ANALYZE.promotion_decision(
            trials,
            {1337, 2027, 9001},
            1024,
            candidate_arm,
        )
        self.assertFalse(rejected["directional_promotion_passed"])
        self.assertTrue(
            any("policy_orbit_not_materialized" in failure for failure in rejected["failures"])
        )

        trials[-1] = trial(candidate_arm, 9001)
        trials[-1]["dagger_presentation_rows_max"] = 33
        rejected = ANALYZE.promotion_decision(
            trials,
            {1337, 2027, 9001},
            1024,
            candidate_arm,
        )
        self.assertFalse(rejected["directional_promotion_passed"])
        self.assertTrue(
            any(
                "policy_presentation_budget_exceeded" in failure
                for failure in rejected["failures"]
            )
        )

    def test_worst_orbit_candidate_requires_distributionally_robust_contract(self) -> None:
        candidate_arm = "structural_static_orbit_worst_marginal025"
        trials = [
            trial(arm, seed)
            for arm in (*ANALYZE.BASELINE_ARMS, candidate_arm)
            for seed in (1337, 2027, 9001)
        ]
        decision = ANALYZE.promotion_decision(
            trials,
            {1337, 2027, 9001},
            1024,
            candidate_arm,
        )
        self.assertTrue(decision["directional_promotion_passed"], decision["failures"])

        trials[-1]["dagger_presentation_risk"] = "mean"
        rejected = ANALYZE.promotion_decision(
            trials,
            {1337, 2027, 9001},
            1024,
            candidate_arm,
        )
        self.assertFalse(rejected["directional_promotion_passed"])
        self.assertTrue(
            any(
                "policy_presentation_risk_contract" in failure
                for failure in rejected["failures"]
            )
        )

    def test_bc_paired_dagger_candidate_requires_both_effective_modes(self) -> None:
        candidate_arm = "structural_bc_paired_dagger_marginal025"
        trials = [
            trial(arm, seed)
            for arm in (*ANALYZE.BASELINE_ARMS, candidate_arm)
            for seed in (1337, 2027, 9001)
        ]
        decision = ANALYZE.promotion_decision(
            trials,
            {1337, 2027, 9001},
            1024,
            candidate_arm,
        )
        self.assertTrue(decision["directional_promotion_passed"], decision["failures"])
        trials[-1]["dagger_mode"] = "paired_dagger"
        rejected = ANALYZE.promotion_decision(
            trials,
            {1337, 2027, 9001},
            1024,
            candidate_arm,
        )
        self.assertFalse(rejected["directional_promotion_passed"])
        self.assertTrue(
            any("policy_mode_contract" in failure for failure in rejected["failures"])
        )

    def test_bc_paired_dagger_candidate_requires_balanced_paired_rows(self) -> None:
        candidate_arm = "structural_bc_paired_dagger_marginal025"
        trials = [
            trial(arm, seed)
            for arm in (*ANALYZE.BASELINE_ARMS, candidate_arm)
            for seed in (1337, 2027, 9001)
        ]
        trials[-1]["dagger_paired_on_policy_expert_rows"] = 8
        decision = ANALYZE.promotion_decision(
            trials,
            {1337, 2027, 9001},
            1024,
            candidate_arm,
        )
        self.assertFalse(decision["directional_promotion_passed"])
        self.assertTrue(
            any(
                "policy_paired_population_imbalance" in failure
                for failure in decision["failures"]
            )
        )

    def test_missing_partition_telemetry_blocks_promotion(self) -> None:
        trials = [
            trial(arm, seed)
            for arm in (*ANALYZE.BASELINE_ARMS, ANALYZE.DEFAULT_CANDIDATE_ARM)
            for seed in (1337, 2027, 9001)
        ]
        trials[-1]["mode_telemetry_present"] = False
        decision = ANALYZE.promotion_decision(trials, {1337, 2027, 9001}, 1024)
        self.assertFalse(decision["directional_promotion_passed"])
        self.assertTrue(
            any("missing_partition_telemetry" in failure for failure in decision["failures"])
        )

    def test_semantic_prompt_leakage_blocks_structural_promotion(self) -> None:
        trials = [
            trial(arm, seed)
            for arm in (*ANALYZE.BASELINE_ARMS, ANALYZE.DEFAULT_CANDIDATE_ARM)
            for seed in (1337, 2027, 9001)
        ]
        structural = next(row for row in trials if row["arm"] == "structural_ce")
        structural["semantic_prompt_leak_count"] = 1
        decision = ANALYZE.promotion_decision(trials, {1337, 2027, 9001}, 1024)
        self.assertFalse(decision["directional_promotion_passed"])
        self.assertTrue(
            any("semantic_prompt_leakage" in failure for failure in decision["failures"])
        )

    def test_single_seed_is_only_smoke_evidence(self) -> None:
        trials = [
            trial(arm, 1337)
            for arm in (*ANALYZE.BASELINE_ARMS, ANALYZE.DEFAULT_CANDIDATE_ARM)
        ]
        for row in trials:
            row["max_iters"] = 64
        decision = ANALYZE.promotion_decision(trials, {1337}, 1024)
        self.assertEqual(decision["evidence_class"], "smoke")
        self.assertFalse(decision["directional_promotion_passed"])
        self.assertTrue(any("immature_updates" in failure for failure in decision["failures"]))

    def test_confidence_interval_is_reported_for_three_seeds(self) -> None:
        center, half_width = ANALYZE.mean_ci95([0.1, 0.2, 0.3])
        self.assertAlmostEqual(center, 0.2)
        self.assertIsNotNone(half_width)
        self.assertGreater(half_width, 0.0)

    def test_cuda_density_regression_blocks_promotion(self) -> None:
        trials = [
            trial(arm, seed)
            for arm in (*ANALYZE.BASELINE_ARMS, ANALYZE.DEFAULT_CANDIDATE_ARM)
            for seed in (1337, 2027, 9001)
        ]
        trials[-1]["gpu_active_util_mean"] = 72.0
        trials[-1]["gpu_high_util_fraction"] = 0.55
        trials[-1]["gpu_low_util_fraction"] = 0.25
        trials[-1]["gpu_max_consecutive_sub80_samples"] = 11.0
        trials[-1]["gpu_max_consecutive_idle_samples"] = 6.0
        decision = ANALYZE.promotion_decision(trials, {1337, 2027, 9001}, 1024)
        self.assertFalse(decision["directional_promotion_passed"])
        self.assertTrue(
            any("gpu_active_util_below_85" in failure for failure in decision["failures"])
        )
        self.assertTrue(
            any(
                "gpu_high_util_fraction_below_0.80" in failure
                for failure in decision["failures"]
            )
        )
        self.assertTrue(
            any(
                "gpu_low_util_fraction_above_0.10" in failure
                for failure in decision["failures"]
            )
        )
        self.assertTrue(
            any(
                "gpu_sub80_streak_above_10_samples" in failure
                for failure in decision["failures"]
            )
        )
        self.assertTrue(
            any(
                "gpu_idle_streak_above_5_samples" in failure
                for failure in decision["failures"]
            )
        )

    def test_host_stage_share_is_not_misclassified_as_gpu_duty(self) -> None:
        trials = [
            trial(arm, seed)
            for arm in (*ANALYZE.BASELINE_ARMS, ANALYZE.DEFAULT_CANDIDATE_ARM)
            for seed in (1337, 2027, 9001)
        ]
        for row in trials:
            row["model_duty_fraction"] = 0.10
        decision = ANALYZE.promotion_decision(trials, {1337, 2027, 9001}, 1024)
        self.assertFalse(
            any("model_duty" in failure for failure in decision["failures"]),
            decision["failures"],
        )


if __name__ == "__main__":
    unittest.main()
