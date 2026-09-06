"""CPU-only safety tests; no tests allocate memory to trigger a guard."""

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.experiments.config import Case, Limits, Matrix, load, strict
from scripts.experiments.guard import Memory, gpu_read, gpu_violation, kill_group, memory_violation, watch
from scripts.experiments.runner import run


class ConfigTests(unittest.TestCase):
    def test_rejects_unsafe_limits(self):
        for value in (0, -1, 0.901, 1.0, float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Limits(system_fraction=value)
        for values in ({"headroom_mib": 0}, {"poll_seconds": 2}, {"shared_gpu_memory": "true"}):
            with self.assertRaises(ValueError):
                Limits(**values)

    def test_strict_schema_and_unique_case_ids(self):
        with self.assertRaises(ValueError):
            strict(Limits, {"system_fracion": 0.9})
        case = Case("case", ["true"], 1, 1)
        with self.assertRaises(ValueError):
            Matrix(1, "out", [case, case])
        with self.assertRaises(ValueError):
            Case("../escape", ["true"], 1, 1)
        with self.assertRaises(ValueError):
            Case("bad", "shell string", 1, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.toml"
            path.write_text('version=1\noutput="out"\n[[cases]]\nid="ok"\nargv=["true"]\nexpected_peak_mib=1\ntimeout_seconds=1\n')
            self.assertEqual(load(path).cases, [Case("ok", ["true"], 1, 1)])


class GuardTests(unittest.TestCase):
    def test_available_memory_not_rss_and_headroom(self):
        limits = Limits(headroom_mib=10)
        self.assertIsNone(memory_violation(Memory(1000, 900), limits, 700))
        self.assertIsNotNone(memory_violation(Memory(1000, 900), limits, 800))
        self.assertIsNotNone(memory_violation(Memory(1000, 100), limits))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meminfo"
            path.write_text("MemTotal: 1024000 kB\nMemAvailable: 512000 kB\nMemFree: 1024 kB\n")
            self.assertEqual(Memory.read(path), Memory(1000, 500))
            path.write_text("MemTotal: 1024 kB\n")
            with self.assertRaises(KeyError):
                Memory.read(path)

    def test_unified_counted_once_discrete_unknown_fails_closed(self):
        self.assertIsNone(gpu_violation(None, Limits()))
        discrete = Limits(shared_gpu_memory=False, headroom_mib=10)
        self.assertIsNotNone(gpu_violation(None, discrete))
        self.assertIsNotNone(gpu_violation({"used_mib": 850, "total_mib": 1000}, discrete, 50))
        self.assertIsNone(gpu_violation({"used_mib": 100, "total_mib": 1000}, discrete, 50))

    def test_nonfinite_gpu_telemetry_is_unavailable(self):
        response = subprocess.CompletedProcess([], 0, stdout="nan, 13, [N/A], inf\n")
        with patch("scripts.experiments.guard.subprocess.run", return_value=response):
            self.assertEqual(gpu_read(0), {"util_percent": None, "power_w": 13, "used_mib": None, "total_mib": None})

    def test_guard_and_timeout_cleanup_real_process_group(self):
        for reading, timeout, expected in ((Memory(1000, 900), 0.02, "timeout"), (Memory(1000, 5), 10, "host_memory")):
            process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
            try:
                result = watch(process, Limits(headroom_mib=10, poll_seconds=0.01), timeout, io.StringIO(), memory_read=lambda: reading)
                self.assertTrue(result["status"].startswith(expected))
            finally:
                kill_group(process)
            self.assertIsNotNone(process.poll())
            with self.assertRaises(ProcessLookupError):
                os.killpg(process.pid, 0)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        (self.root / ".gitignore").write_text("out/\n")
        subprocess.run(["git", "-C", str(self.root), "add", ".gitignore"], check=True)
        subprocess.run(["git", "-C", str(self.root), "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)

    def matrix(self, *cases):
        return Matrix(1, "out", list(cases), Limits(headroom_mib=1, poll_seconds=0.01))

    @patch("scripts.experiments.runner.Memory.read", return_value=Memory(100000, 90000))
    def test_success_provenance_and_no_overwrite(self, _):
        case = Case("first", [sys.executable, "-c", "print('complete')"], 1, 5)
        result = run(self.matrix(case), self.root)
        self.assertTrue(result["complete"])
        self.assertTrue((self.root / "out/source-0/tracked.patch").is_file())
        manifest = json.loads((self.root / "out/first/manifest.json").read_text())
        self.assertTrue(Path(manifest["execution_argv"][0]).is_file())
        self.assertNotEqual(manifest["execution_argv"][0], manifest["argv"][0])
        with self.assertRaises(FileExistsError):
            run(self.matrix(case), self.root)

    def test_admission_refuses_before_launch(self):
        case = Case("unsafe", [sys.executable, "-c", "raise Exception('launched')"], 10**12, 5)
        result = run(self.matrix(case), self.root)
        self.assertEqual(result["cases"][0]["status"], "admission_rejected")
        self.assertFalse((self.root / "out/unsafe/stdout.log").exists())

    def test_failure_stops_matrix(self):
        failed = Case("failed", [sys.executable, "-c", "raise SystemExit(2)"], 1, 5)
        second = Case("second", [sys.executable, "-c", "print('unexpected')"], 1, 5)
        result = run(self.matrix(failed, second), self.root)
        self.assertFalse(result["complete"])
        self.assertEqual(len(result["cases"]), 1)
        self.assertEqual(result["cases"][0]["exit_code"], 2)

    def test_interrupt_records_incomplete_and_cleans_up(self):
        case = Case("interrupt", [sys.executable, "-c", "import time; time.sleep(60)"], 1, 5)
        with patch("scripts.experiments.runner.watch", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                run(self.matrix(case), self.root)
        result = json.loads((self.root / "out/results.json").read_text())
        self.assertFalse(result["complete"])
        self.assertEqual(result["cases"][0]["status"], "interrupted")
        self.assertEqual(result["cases"][0]["exit_code"], -9)

    def test_inherited_burn_overrides_are_removed(self):
        case = Case("env", [sys.executable, "-c", "import os; assert not any(k.startswith(('BURN_', 'DragonModel_')) for k in os.environ)"], 1, 5)
        with patch.dict(os.environ, {"BURN_TEST_OVERRIDE": "should not propagate", "DragonModel_STAGE_PROFILE": "1"}):
            self.assertTrue(run(self.matrix(case), self.root)["complete"])

    def test_input_mutation_is_not_success(self):
        (self.root / "input.txt").write_text("original")
        case = Case("mutate", [sys.executable, "-c", "from pathlib import Path; Path('input.txt').write_text('changed')"], 1, 5, inputs=["input.txt"])
        result = run(self.matrix(case), self.root)
        self.assertFalse(result["complete"])
        self.assertEqual(result["cases"][0]["status"], "input_drift")


if __name__ == "__main__":
    unittest.main()
