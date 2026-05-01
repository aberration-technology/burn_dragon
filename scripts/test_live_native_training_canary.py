#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "live-native-training-canary.yml"
SCRIPT = REPO_ROOT / "scripts" / "live_native_training_canary.py"
DISPATCH_SCRIPT = REPO_ROOT / "scripts" / "dispatch_native_training_canary_and_wait.sh"


def main() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    on_config = workflow.get("on", workflow.get(True))
    dispatch_inputs = on_config["workflow_dispatch"]["inputs"]
    assert dispatch_inputs["environment"]["default"] == "production"
    assert dispatch_inputs["experiment_id"]["default"] == "nca-prepretraining"
    assert dispatch_inputs["backend"]["default"] == "cpu"
    assert "schedule" in on_config

    job = workflow["jobs"]["canary"]
    assert job["environment"] == "burn-dragon-p2p-${{ github.event.inputs.environment || 'production' }}"
    env = job["env"]
    for key, value in env.items():
        assert "runner." not in str(value), f"job env {key} uses unavailable runner context"
    assert (
        env["BURN_DRAGON_NATIVE_CANARY_CALLBACK_TOKEN"]
        == "${{ secrets.BURN_DRAGON_P2P_BROWSER_CANARY_CALLBACK_TOKEN }}"
    )
    assert env["BURN_DRAGON_NATIVE_CANARY_ARTIFACT_DIR"].startswith("/tmp/")
    assert env["BURN_DRAGON_NATIVE_CANARY_WINDOWS"] == "${{ github.event.inputs.windows || '2' }}"
    runs = "\n".join(step.get("run", "") for step in job["steps"])
    assert "scripts/ensure-burn-p2p-sibling.sh" in runs
    assert "cargo build --locked -p burn_dragon_p2p --bin burn_dragon_p2p_native" in runs
    assert "python3 scripts/live_native_training_canary.py" in runs

    script = SCRIPT.read_text()
    required = [
        "enroll-static-principal",
        "--principal-kind",
        "run-validator-daemon",
        "train-window-once",
        "--require-head-advanced",
        "/directory/signed",
        "/artifacts/heads/",
        "/metrics/catchup/",
        "canonical head did not advance",
        "canonical loss regressed",
        "canonical_loss_improved",
        "train_loss",
        "batch_count",
        "BURN_DRAGON_P2P_NATIVE_STORAGE_ROOT",
        "BURN_DRAGON_NATIVE_CANARY_VALIDATOR_PRINCIPAL_ID",
    ]
    for snippet in required:
        assert snippet in script, f"missing native canary script snippet: {snippet}"
    assert (
        '"--require-head-advanced",\n                    "true",'
        not in script
    ), "--require-head-advanced is a presence flag; do not pass a boolean value"

    dispatch_script = DISPATCH_SCRIPT.read_text()
    for snippet in [
        ".github/workflows/live-native-training-canary.yml",
        "gh workflow run",
        "gh run watch",
        "BURN_DRAGON_NATIVE_CANARY_EDGE_BASE_URL",
    ]:
        assert snippet in dispatch_script, f"missing native canary dispatch snippet: {snippet}"

    print("live-native-training-canary-ok")


if __name__ == "__main__":
    main()
