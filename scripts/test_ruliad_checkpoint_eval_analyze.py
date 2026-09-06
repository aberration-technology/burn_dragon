import json
from pathlib import Path
import tempfile
import unittest

from scripts.ruliad_checkpoint_eval_analyze import analyze, fixture_report, report_row


class EvaluationContracts(unittest.TestCase):
    def test_v8_controls_reject_partial_or_inconsistent_evidence(self):
        report = fixture_report(0.0, 0.5)
        report["evaluation"]["version"] = 8
        report["options"] = {"policy_items": 4}
        report["evaluation"]["policy_controls"] = {
            "version": 1, "kernel_audited_candidates": 16,
            "summary": {"items": 4, "uniform_expected_accuracy": 0.5,
                        "model_accuracy": 0.5, "no_context_accuracy": 0.25,
                        "model_minus_no_context": 0.25},
            "items": [{"oracle_hash": str(index), "candidates": 4, "equivalent_candidates": 2,
                       "uniform_expected_accuracy": 0.5, "model_correct": index < 2,
                       "no_context_correct": index == 0} for index in range(4)],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pc-seed13.json"
            path.write_text(json.dumps(report))
            report_row(path)
            for corruption in ("drop", "aggregate", "chance", "duplicate", "kernel", "outcome", "missing"):
                changed = json.loads(json.dumps(report))
                controls = changed["evaluation"]["policy_controls"]
                if corruption == "drop":
                    controls["items"].pop()
                elif corruption == "aggregate":
                    controls["summary"]["model_accuracy"] = 1.0
                elif corruption == "chance":
                    controls["items"][0]["uniform_expected_accuracy"] = 0.25
                elif corruption == "duplicate":
                    controls["items"][0]["oracle_hash"] = "1"
                elif corruption == "kernel":
                    controls["kernel_audited_candidates"] = 3
                elif corruption == "outcome":
                    controls["items"][0]["model_correct"] = 1
                else:
                    del changed["evaluation"]["policy_controls"]
                path.write_text(json.dumps(changed))
                with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                    report_row(path)

    def test_policy_controls_remain_distinct_from_learned_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pc-seed13.json"
            report = fixture_report(0.0, 0.7)
            report["evaluation"]["policy_controls"] = {
                "summary": {"uniform_expected_accuracy": 0.5, "no_context_accuracy": 0.6,
                            "model_minus_no_context": 0.1},
                "by_difficulty": {"3": {"items": 16, "model_minus_chance": -0.2}},
            }
            path.write_text(json.dumps(report))
            row = report_row(path)
            self.assertEqual(row["control_uniform_expected_accuracy"], 0.5)
            self.assertEqual(row["control_no_context_accuracy"], 0.6)
            self.assertEqual(row["control_d3_model_minus_chance"], -0.2)

    def test_full_context_metrics_and_version_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adam = fixture_report(0.25, 0.5)
            adam["evaluation"]["version"] = 7
            adam["evaluation"]["free_run"]["teacher_forced"] = {"version": 2, "mean_nll": 0.3, "mean_sequence_nll": 12.0}
            adam_path = root / "adam-seed13.json"
            adam_path.write_text(json.dumps(adam))
            self.assertEqual(report_row(adam_path)["free_run_teacher_forced_mean_sequence_nll"], 12.0)
            pc_path = root / "pc-seed13.json"
            pc_path.write_text(json.dumps(adam))
            analyze([adam_path, pc_path], root / "valid", "adam")
            for contract in ("evaluation", "teacher", "corpus"):
                changed = json.loads(json.dumps(adam))
                if contract == "evaluation":
                    changed["evaluation"]["version"] = 6
                elif contract == "teacher":
                    changed["evaluation"]["free_run"]["teacher_forced"]["version"] = 1
                else:
                    changed["corpus_semantic_fingerprint"] = "changed"
                pc_path.write_text(json.dumps(changed))
                with self.subTest(contract=contract), self.assertRaisesRegex(ValueError, "incompatible"):
                    analyze([adam_path, pc_path], root / "invalid", "adam")


if __name__ == "__main__":
    unittest.main()
