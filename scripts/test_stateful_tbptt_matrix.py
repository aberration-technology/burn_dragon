"""Launch preflight tests; never compile or launch a training process."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/stateful_tbptt_matrix.sh"


class StatefulMatrixPreflight(unittest.TestCase):
    def test_archived_reset_arms_fail_before_creating_artifacts(self):
        for arm in ("block512_reset", "chunk128_reset"):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "not-created"
                result = subprocess.run(
                    ["bash", str(SCRIPT), "--arms", arm, "--out-dir", str(output)],
                    cwd=ROOT, capture_output=True, text=True, timeout=10,
                )
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("unsupported masked-stream reset", result.stderr)
                self.assertFalse(output.exists())

    def test_default_dry_run_contains_only_supported_carry_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                ["bash", str(SCRIPT), "--dry-run", "--seeds", "13", "--out-dir", directory],
                cwd=ROOT, capture_output=True, text=True, timeout=20,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            config = json.loads((Path(directory) / "matrix-config.json").read_text())
            self.assertEqual(config["requested_arms"], ["block512_carry", "chunk128_carry", "chunk64_carry"])
            manifests = list((Path(directory) / "manifests").glob("*.json"))
            self.assertEqual(len(manifests), 3)
            self.assertTrue(all(json.loads(path.read_text())["status"] == "dry_run" for path in manifests))


if __name__ == "__main__":
    unittest.main()
