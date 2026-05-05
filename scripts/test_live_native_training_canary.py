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
    assert dispatch_inputs["training_max_iters"]["default"] == "4"
    assert dispatch_inputs["evaluation_max_batches"]["default"] == "1"
    assert dispatch_inputs["settle_diffusion"]["default"] == "true"
    assert dispatch_inputs["diffusion_settle_passes"]["default"] == "3"
    assert dispatch_inputs["serve_after_publish_secs"]["default"] == "120"
    assert dispatch_inputs["command_timeout_secs"]["default"] == "1800"
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
    assert env["BURN_DRAGON_NATIVE_CANARY_TRAINING_BATCH_SIZE"] == "1"
    assert (
        env["BURN_DRAGON_NATIVE_CANARY_TRAINING_MAX_ITERS"]
        == "${{ github.event.inputs.training_max_iters || '4' }}"
    )
    assert (
        env["BURN_DRAGON_NATIVE_CANARY_EVALUATION_MAX_BATCHES"]
        == "${{ github.event.inputs.evaluation_max_batches || '1' }}"
    )
    assert env["BURN_DRAGON_NATIVE_CANARY_HEAD_SYNC_TIMEOUT_SECS"] == "300"
    assert (
        env["BURN_DRAGON_NATIVE_CANARY_SETTLE_DIFFUSION"]
        == "${{ github.event.inputs.settle_diffusion || 'true' }}"
    )
    assert (
        env["BURN_DRAGON_NATIVE_CANARY_DIFFUSION_SETTLE_PASSES"]
        == "${{ github.event.inputs.diffusion_settle_passes || '3' }}"
    )
    assert (
        env["BURN_DRAGON_NATIVE_CANARY_SERVE_AFTER_PUBLISH_SECS"]
        == "${{ github.event.inputs.serve_after_publish_secs || '120' }}"
    )
    assert (
        env["BURN_DRAGON_NATIVE_CANARY_COMMAND_TIMEOUT_SECS"]
        == "${{ github.event.inputs.command_timeout_secs || '1800' }}"
    )
    assert env["BURN_DRAGON_NATIVE_CANARY_P2P_TIMEOUT_SECS"] == "300"
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
        "probe-swarm",
        "--fetch-snapshot",
        "p2p_bootstrap_addresses",
        "BURN_DRAGON_NATIVE_CANARY_P2P_BOOTSTRAP_ADDRS",
        "BURN_DRAGON_NATIVE_CANARY_HTTP_ATTEMPTS",
        "quic-v1",
        "p2p bootstrap snapshot did not advertise canonical head",
        "p2p_signal",
        "p2p_wait_secs",
        "assert_head_provider_signal",
        "head_provider_signal",
        "head_provider_before",
        "connected_provider_peer_ids",
        "available_profiles",
        "published_artifacts",
        "require_edge_provider=False",
        "trainer-window-",
        "trainer-enroll-storage",
        "update_announcements",
        "trainer_promotion_attestation_announcements",
        "diffusion_promotion_certificate_announcements",
        "canonical head did not advance",
        "canonical loss regressed",
        "canonical_loss_improved",
        "canonical_loss_metric",
        "comparable_loss_signal",
        "train_loss",
        "batch_count",
        "training_batch_size",
        "training_max_iters",
        "evaluation_max_batches",
        "head_sync_timeout_secs",
        "settle_diffusion",
        "diffusion_settle_passes",
        "serve_after_publish_secs",
        "--head-sync-timeout-secs",
        "--settle-diffusion",
        "--diffusion-settle-passes",
        "--serve-after-publish-secs",
        "diffusion_settlement",
        "passes_completed",
        "initialize_head_on_start",
        'not bool(head_before.get("head_id"))',
        "--training-batch-size",
        "--training-max-iters",
        "--evaluation-max-batches",
        "BURN_DRAGON_P2P_NATIVE_STORAGE_ROOT",
        "BURN_DRAGON_NATIVE_CANARY_VALIDATOR_PRINCIPAL_ID",
        "BURN_DRAGON_NATIVE_CANARY_SETTLE_DIFFUSION",
        "BURN_DRAGON_NATIVE_CANARY_DIFFUSION_SETTLE_PASSES",
        "BURN_DRAGON_NATIVE_CANARY_SERVE_AFTER_PUBLISH_SECS",
        "BURN_DRAGON_NATIVE_CANARY_EVALUATION_MAX_BATCHES",
    ]
    for snippet in required:
        assert snippet in script, f"missing native canary script snippet: {snippet}"
    assert (
        '"--require-head-advanced",\n                    "true",'
        not in script
    ), "--require-head-advanced is a presence flag; do not pass a boolean value"

    native_source = (
        REPO_ROOT
        / "crates"
        / "burn_dragon_p2p"
        / "src"
        / "bin"
        / "burn_dragon_p2p_native.rs"
    ).read_text()
    for snippet in [
        "DEFAULT_TRAIN_WINDOW_HEAD_SYNC_TIMEOUT_SECS",
        "DiffusionSettlementReport",
        "head_sync_timeout_secs",
        "settle_diffusion",
        "diffusion_settle_passes",
        "serve_after_publish_secs",
        "mirror_live_head_to_edge",
        "register_live_head_reference_with_edge",
        "edge_registered_head",
        "advance_diffusion_steady_state(",
        "serving published artifact",
        "wait_for_head_provider(",
        "{log_prefix}-head-waiting",
        "validator-head-sync-waiting",
        "served_head={}",
        "no experiment head became available within",
    ]:
        assert snippet in native_source, f"missing native head-sync readiness snippet: {snippet}"

    dispatch_script = DISPATCH_SCRIPT.read_text()
    for snippet in [
        ".github/workflows/live-native-training-canary.yml",
        "gh workflow run",
        "gh run watch",
        "BURN_DRAGON_NATIVE_CANARY_EDGE_BASE_URL",
        "BURN_DRAGON_NATIVE_CANARY_SETTLE_DIFFUSION",
        "BURN_DRAGON_NATIVE_CANARY_EVALUATION_MAX_BATCHES",
        "serve_after_publish_secs",
    ]:
        assert snippet in dispatch_script, f"missing native canary dispatch snippet: {snippet}"

    print("live-native-training-canary-ok")


if __name__ == "__main__":
    main()
