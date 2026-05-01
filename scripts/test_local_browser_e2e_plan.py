#!/usr/bin/env python3

from pathlib import Path


def main() -> None:
    xtask = Path("xtask/src/main.rs").read_text()
    native_tests = Path("crates/burn_dragon_p2p/tests/native_training.rs").read_text()
    canary = Path("scripts/live-browser-canary.mjs").read_text()
    workflow = Path(".github/workflows/live-browser-canary.yml").read_text()

    required_xtask_snippets = [
        "LocalBrowserE2e",
        "LocalProdE2e",
        "WasmTrainingSmoke",
        "fn local_browser_e2e() -> Result<()>",
        "fn local_prod_e2e() -> Result<()>",
        "fn local_browser_contract_e2e(build_site: bool) -> Result<()>",
        "deployment_script_checks()?;",
        "browser_site::build_browser_site_default()?;",
        'cargo_native_test(Some("local_browser_training_e2e"), false)?;',
        "wasm_training_smoke()",
        "fn wasm_training_smoke() -> Result<()>",
        'wasm_browser_test(Some("browser_training_smoke_generated_nca"))',
        "scripts/test_local_browser_e2e_plan.py",
    ]
    for snippet in required_xtask_snippets:
        assert snippet in xtask, f"xtask local browser e2e plan missing: {snippet}"

    local_browser_e2e_body = xtask.split("fn local_browser_e2e() -> Result<()>", 1)[1].split(
        "\n}\n",
        1,
    )[0]
    forbidden_xtask_snippets = [
        "cargo clean",
        "native_scale()",
        "native_large()",
        "mixed_fleet()",
        "edge_drill()",
    ]
    for snippet in forbidden_xtask_snippets:
        assert snippet not in local_browser_e2e_body, (
            f"local-browser-e2e should stay a short local loop, found {snippet}"
        )

    required_native_snippets = [
        "const TEST_WEBRTC_DIRECT_SEED",
        "/dns4/edge.example/udp/443/webrtc-direct/certhash/",
        "webrtc_direct: true,",
        "webtransport_gateway: false,",
        "wss_fallback: false,",
        "fn local_browser_training_e2e()",
        'run_edge_drill_for_prepared(&prepared, "local-browser-e2e");',
        "receipt_submission_batches >= 2",
        "native plus two distinct browser peers should enroll against the same edge",
    ]
    for snippet in required_native_snippets:
        assert snippet in native_tests, f"native local browser e2e missing: {snippet}"

    required_canary_snippets = [
        "training_p2p_checkpoint_ready: null",
        "report.training_p2p_checkpoint_ready = machineStateCheckpointReady(report);",
        "browser training canary did not sync the active head checkpoint over P2P before training",
        "productionBrowserTrainingConfig?.live_participant?.load_active_head_artifact === true",
        "Training P2P checkpoint ready",
    ]
    canary_and_summary = canary + "\n" + Path("scripts/summarize_live_browser_canary.py").read_text()
    for snippet in required_canary_snippets:
        assert snippet in canary_and_summary, f"browser training canary parity missing: {snippet}"

    assert "chromium-webrtc-direct-training" in workflow
    assert "chromium-webrtc-direct-checkpoint" in workflow

    print("local-browser-e2e-plan-ok")


if __name__ == "__main__":
    main()
