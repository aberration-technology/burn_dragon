#!/usr/bin/env python3

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("stateful_tbptt_analyze.py")
SPEC = importlib.util.spec_from_file_location("stateful_tbptt_analyze", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class StatefulTbpttAnalyzeTests(unittest.TestCase):
    def write_trial(self, root: Path, arm: str, seed: int, warm: float, valid: float, throughput: float):
        trial = f"{arm}-seed{seed}"
        run_dir = root / "runs" / trial
        event_dir = run_dir / "events"
        event_dir.mkdir(parents=True)
        metrics = {
            "Loss": valid,
            "Stream Warm Loss": warm,
            "Stream Paired Warm Loss": warm,
            "Stream Paired Cold Loss": valid + 0.05,
            "Stream Carry NLL Gain": valid + 0.05 - warm,
            "Stream Carry Relative Gain": (valid + 0.05 - warm) / (valid + 0.05),
            "Stream Carry Probe Batches": 8.0,
            "Sequence State Rho RMS": 0.5,
            "Sequence State Rho Slot Variance Ratio": 0.4,
            "Sequence State Rho Slot Redundancy": 0.2,
            "Ruliad Verifier Accuracy": 0.25,
            "Ruliad Training Serialization Ruliad Verifier Accuracy": 0.5,
        }
        events = []
        for name, value in metrics.items():
            split = "train" if name == "Loss" and False else "valid"
            events.append({"type": "metric", "split": split, "name": name, "value": value, "running_value": value})
        events.append({"type": "metric", "split": "train", "name": "Loss", "value": 1.0, "running_value": 1.0})
        events.append({"type": "metric", "split": "train", "name": "Loss", "value": 0.0, "running_value": 0.0})
        (event_dir / "training_events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events))
        log_path = root / "logs" / f"{trial}.log"
        log_path.parent.mkdir(exist_ok=True)
        log_path.write_text(
            f"[stage-profile][training] wall_tokens_per_second={throughput} "
            f"model_tokens_per_second={throughput} model_duty_fraction=0.9 "
            "validation_fraction=0.1\n"
        )
        gpu_log_path = root / "logs" / f"{trial}.gpu.csv"
        gpu_log_path.write_text(
            "utilization.gpu [%],power.draw [W]\n"
            "10 %,20 W\n"
            "80 %,40 W\n"
            "90 %,50 W\n"
        )
        manifest = {
            "arm": arm,
            "seed": seed,
            "status": "ok",
            "max_iters": 512,
            "batch_size": 16,
            "elapsed_seconds": 10,
            "peak_used_mb": 4096,
            "run_dir": str(run_dir),
            "log_path": str(log_path),
            "gpu_log_path": str(gpu_log_path),
            "time_log_path": "",
        }
        manifest_dir = root / "manifests"
        manifest_dir.mkdir(exist_ok=True)
        (manifest_dir / f"{trial}.json").write_text(json.dumps(manifest))

    def test_matched_matrix_is_aggregated_and_promoted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed in (1, 2, 3):
                self.write_trial(root, "block512_reset", seed, 1.00, 1.00, 1000)
                self.write_trial(root, "block512_carry", seed, 0.95, 1.00, 980)
                self.write_trial(root, "chunk128_reset", seed, 1.10, 1.10, 900)
                self.write_trial(root, "chunk128_carry", seed, 1.00, 1.10, 850)
            result = MODULE.analyze_matrix(root)
            self.assertEqual(result["decision"], "promotable")
            self.assertEqual(result["mechanics_decision"], "passed")
            self.assertEqual(len(result["aggregates"]), 4)
            self.assertAlmostEqual(result["comparisons"][0]["paired_warm_loss_delta"], -0.05)
            self.assertAlmostEqual(result["aggregates"][0]["zero_train_loss_sample_fraction"], 0.5)
            self.assertAlmostEqual(result["aggregates"][0]["model_duty_fraction"], 0.9)
            self.assertAlmostEqual(result["aggregates"][0]["validation_fraction"], 0.1)
            self.assertAlmostEqual(result["aggregates"][0]["gpu_util_p10"], 10.0)
            self.assertAlmostEqual(result["aggregates"][0]["gpu_util_p90"], 90.0)
            report = (root / "stateful-tbptt-report.md").read_text()
            self.assertIn("model duty %", report)
            self.assertIn("90.0 | 10.0 | 60.0/10.0/90.0", report)
            self.assertTrue((root / "stateful-tbptt-report.md").is_file())

    def test_partial_smoke_is_not_a_promotion_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for arm in ("block512_reset", "block512_carry", "chunk128_reset", "chunk128_carry"):
                self.write_trial(root, arm, 1, 1.0 if "reset" in arm else 0.9, 1.0, 1000)
            result = MODULE.analyze_matrix(root)
            self.assertEqual(result["decision"], "screening_only")
            self.assertTrue(any("screening evidence only" in reason for reason in result["reasons"]))

    def test_persistent_arm_uses_stream_warm_training_loss(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_trial(root, "block512_carry", 1, 0.9, 1.0, 1000)
            events_path = (
                root
                / "runs"
                / "block512_carry-seed1"
                / "events"
                / "training_events.jsonl"
            )
            with events_path.open("a") as handle:
                for value in (2.0, 1.0):
                    handle.write(
                        json.dumps(
                            {
                                "type": "metric",
                                "split": "train",
                                "name": "Stream Warm Loss",
                                "value": value,
                                "running_value": value,
                            }
                        )
                        + "\n"
                    )

            trial = MODULE.load_trial(
                root / "manifests" / "block512_carry-seed1.json"
            )
            self.assertAlmostEqual(trial["train_loss"], 1.5)
            self.assertAlmostEqual(trial["zero_train_loss_sample_fraction"], 0.0)

    def test_intentional_single_arm_screen_is_not_invalid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_trial(root, "block512_reset", 1, 1.0, 1.0, 1000)
            (root / "matrix-config.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "requested_arms": ["block512_reset"],
                        "requested_seeds": [1],
                    }
                )
                + "\n"
            )

            result = MODULE.analyze_matrix(root)
            self.assertEqual(result["decision"], "screening_only")
            self.assertEqual(result["comparisons"], [])
            self.assertTrue(
                any("intentional partial arm screen" in reason for reason in result["reasons"])
            )


if __name__ == "__main__":
    unittest.main()
