from pathlib import Path
import json
import tempfile
import unittest

from scripts.experiments.training_summary import completion_trajectory, epoch_counter_total, find_launch, launch_configuration


class TrainingSummaryTests(unittest.TestCase):
    def test_launch_configuration_does_not_read_mutable_resume_config(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "training_config.json").write_text('{"training": {"batch_size": 999}}')
            settings = dict(batch_size=4, ruliad_supervision={"mode": "answer_completion"},
                            tbptt_chunk_size=256, tbptt_credit_window_chunks=4,
                            tbptt_persist_across_steps=True)
            snapshot = {"training": {"training": settings}, "model": {"fused_kernels": {
                "rotary_embedding": "alibi", "alibi_slopes": [0.25, 0.0625]}}}
            (run / "launch.json").write_text(json.dumps(snapshot))
            training, contract = launch_configuration(run, {"config_snapshot": "launch.json"})
            self.assertEqual(training["batch_size"], 4)
            self.assertEqual(contract["alibi_slopes"], [0.25, 0.0625])
            self.assertEqual(contract["supervision_mode"], "answer_completion")
            with self.assertRaises(FileNotFoundError):
                launch_configuration(run, {"config_snapshot": "missing.json"})

    def test_completion_panels_separate_difficulty_and_count_model_not_lexical_tokens(self):
        common = dict(epoch=1, absolute_step=512, probe_name="free", actual_answer="wrong",
                      status="SchemaValidWrong", generated_token_count=1,
                      generated_model_token_count=128, answer_terminated=False,
                      generation_hit_budget=True)
        rows = [dict(common, difficulty_level=0, verifier_match=True, semantic_match=True),
                dict(common, difficulty_level=1, verifier_match=False, semantic_match=False),
                dict(common, epoch=2, absolute_step=1024, difficulty_level=0,
                     verifier_match=False, semantic_match=False, generated_model_token_count=None)]
        panels = completion_trajectory(rows)
        self.assertEqual(len(panels), 2)
        self.assertEqual(panels[0]["aggregate"]["mean_model_tokens"], 128)
        self.assertEqual(panels[0]["aggregate"]["dominant_answer_fraction"], 1.0)
        self.assertEqual(panels[0]["difficulties"]["0"]["verified"], 1)
        self.assertEqual(panels[0]["difficulties"]["1"]["verified"], 0)
        self.assertIsNone(panels[1]["aggregate"]["mean_model_tokens"])
        self.assertEqual(panels[1]["aggregate"]["model_token_count_items"], 0)
        self.assertEqual(completion_trajectory([]), [])

    def test_epoch_counters_are_not_summed_repeatedly_at_each_log_interval(self):
        events = [{"type": "metric", "split": "train", "name": "tokens", "epoch": epoch, "value": value}
                  for epoch, value in [(1, 10), (1, 20), (2, 15), (2, 25)]]
        self.assertEqual(epoch_counter_total(events, "tokens"), 45)
        self.assertIsNone(epoch_counter_total(events, "missing"))

    def test_matches_complete_captured_command_not_just_run_name_or_binary(self):
        command = ["/matrix/binary/train", "--seed", "13"]
        manifests = [(Path("runs/a/experiment_manifest.json"), {"launches": [{"command": command}]}),
                     (Path("runs/b/experiment_manifest.json"), {"launches": [{"command": command[:-1] + ["29"]}]})]
        path, _, launch = find_launch({"execution_argv": command}, manifests)
        self.assertEqual(path, Path("runs/a"))
        self.assertEqual(launch["command"], command)

    def test_missing_or_ambiguous_launch_fails_closed(self):
        case = {"execution_argv": ["/matrix/binary/train"]}
        manifest = (Path("runs/a/experiment_manifest.json"), {"launches": [{"command": case["execution_argv"]}]})
        for matches in ([], [manifest, manifest]):
            with self.subTest(matches=len(matches)), self.assertRaises(ValueError):
                find_launch(case, matches)
