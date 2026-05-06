from pathlib import Path

import yaml


WORKFLOW_PATHS = [
    Path(".github/workflows/deploy-burn-dragon-p2p-aws.yml"),
    Path(".github/workflows/restore-burn-dragon-p2p-aws.yml"),
    Path(".github/workflows/live-browser-canary.yml"),
]
BOOTSTRAP_SETTINGS_SCRIPT = Path("scripts/resolve_bootstrap_stack_settings.sh")


def main() -> None:
    resolver_text = BOOTSTRAP_SETTINGS_SCRIPT.read_text()
    for workflow_path in WORKFLOW_PATHS:
        workflow_text = workflow_path.read_text()
        workflow = yaml.safe_load(workflow_text)
        jobs = workflow.get("jobs", {})
        assert jobs, f"{workflow_path} missing jobs"

        if workflow_path.name == "live-browser-canary.yml":
            required_snippets = [
                "workflow_call:",
                "cargo run -p xtask -- install-playwright-chromium",
                "cargo run -p xtask -- run-live-browser-canary",
                'cargo run -p xtask -- summarize-live-browser-canary "$report_path" >>"$GITHUB_STEP_SUMMARY"',
                "burn-dragon-live-browser-canary",
                "default: browser-canary-mainnet-nca",
                'BURN_DRAGON_BROWSER_CANARY_TRAIN_TIMEOUT_MS: "300000"',
                "chromium-webrtc-direct-connect",
                "chromium-webrtc-direct-checkpoint",
                "chromium-webrtc-direct-training",
                "firefox-auto-connect",
                "firefox-webrtc-direct-connect",
                "continue-on-error: ${{ matrix.required == '0' }}",
                "BURN_DRAGON_BROWSER_CANARY_BROWSER: ${{ matrix.browser }}",
                "BURN_DRAGON_BROWSER_CANARY_TRANSPORT_MODE: ${{ matrix.transport_mode }}",
                "BURN_DRAGON_BROWSER_CANARY_EXPECT_TRAINING: ${{ matrix.expect_training }}",
                "BURN_DRAGON_BROWSER_CANARY_EXPECT_CHECKPOINT_SYNC: ${{ matrix.expect_checkpoint_sync }}",
                "BURN_DRAGON_BROWSER_CANARY_MIN_ACCEPTED_RECEIPTS: ${{ matrix.min_accepted_receipts }}",
                'BURN_DRAGON_P2P_BROWSER_CANARY_CALLBACK_TOKEN:',
            ]
            for snippet in required_snippets:
                assert snippet in workflow_text, (
                    f"{workflow_path} missing required snippet: {snippet}"
                )
            assert (
                "description: GitHub environment suffix to run against" in workflow_text
            ), f"{workflow_path} missing canary environment selector"
            assert (
                'environment: burn-dragon-p2p-${{ inputs.environment || github.event.inputs.environment || \'production\' }}'
                in workflow_text
            ), f"{workflow_path} missing reusable environment resolution"
            lanes = {
                item["lane"]: item
                for item in jobs["canary"]["strategy"]["matrix"]["include"]
            }
            expected_required = {
                "chromium-auto-connect": "1",
                "chromium-webrtc-direct-connect": "1",
                "chromium-webrtc-direct-checkpoint": "1",
                "chromium-webrtc-direct-training": "1",
                "firefox-auto-connect": "1",
                "firefox-webrtc-direct-connect": "1",
            }
            assert set(lanes) == set(expected_required), (
                f"{workflow_path} live canary lanes drifted: {sorted(lanes)}"
            )
            for lane, required in expected_required.items():
                assert lanes[lane]["required"] == required, (
                    f"{workflow_path} lane {lane} required={lanes[lane]['required']} expected {required}"
                )
            for lane in expected_required:
                expected_training = (
                    "1" if lane == "chromium-webrtc-direct-training" else "0"
                )
                assert lanes[lane]["expect_training"] == expected_training
                expected_checkpoint = (
                    "1" if lane == "chromium-webrtc-direct-checkpoint" else "0"
                )
                assert lanes[lane]["expect_checkpoint_sync"] == expected_checkpoint
                expected_min_receipts = (
                    "2" if lane == "chromium-webrtc-direct-training" else "0"
                )
                assert lanes[lane]["min_accepted_receipts"] == expected_min_receipts
        else:
            resolver_snippets = [
                'browser_canary_principal_id="browser-canary-${TF_WORKSPACE_NAME}-nca"',
                'native_canary_principal_id="native-canary-${TF_WORKSPACE_NAME}-nca"',
                'native_canary_validator_principal_id="${native_canary_principal_id}-validator"',
                """auth_principals_json="$(python3 -c '''import json, sys; principals = json.loads(sys.argv[1] or "[]"); principal_id = sys.argv[2]; principals = [item for item in principals if item.get("principal_id") != principal_id]; print(json.dumps(principals))''' "$auth_principals_json" "$browser_canary_principal_id")" """.strip(),
                """auth_principals_json="$(python3 -c '''import json, sys; principals = json.loads(sys.argv[1] or "[]"); principal_ids = set(sys.argv[2:]); principals = [item for item in principals if item.get("principal_id") not in principal_ids]; print(json.dumps(principals))''' "$auth_principals_json" "$native_canary_principal_id" "$native_canary_validator_principal_id")" """.strip(),
                'echo "BROWSER_CANARY_PRINCIPAL_ID=$browser_canary_principal_id"',
                'echo "NATIVE_CANARY_PRINCIPAL_ID=$native_canary_principal_id"',
                'echo "NATIVE_CANARY_VALIDATOR_PRINCIPAL_ID=$native_canary_validator_principal_id"',
                'echo "TF_VAR_github_browser_canary_principal_id=$browser_canary_principal_id"',
                'echo "TF_VAR_github_browser_canary_callback_token=$BROWSER_CANARY_CALLBACK_TOKEN"',
                'echo "TF_VAR_github_native_canary_principal_id=$native_canary_principal_id"',
                'echo "TF_VAR_github_native_canary_validator_principal_id=$native_canary_validator_principal_id"',
            ]
            for snippet in resolver_snippets:
                assert snippet in resolver_text, (
                    f"{BOOTSTRAP_SETTINGS_SCRIPT} missing required resolver snippet: {snippet}"
                )

            workflow_snippets = [
                'cargo run -p xtask -- dispatch-pages-deploy-and-wait',
                'BURN_DRAGON_DEPLOY_PAGES_ENVIRONMENT: ${{ env.DEPLOY_ENVIRONMENT }}',
                'BURN_DRAGON_DEPLOY_PAGES_EDGE_BASE_URL: ${{ steps.outputs.outputs.edge_url }}',
                'BURN_DRAGON_DEPLOY_PAGES_EXPERIMENT_ID: ${{ env.BROWSER_CANARY_EXPERIMENT_ID }}',
                'BURN_DRAGON_DEPLOY_PAGES_REVISION_ID: ${{ env.BROWSER_CANARY_REVISION_ID }}',
            ]
            for snippet in workflow_snippets:
                assert snippet in workflow_text, (
                    f"{workflow_path} missing required deploy snippet: {snippet}"
                )
            forbidden_snippets = [
                'gh workflow run .github/workflows/deploy-pages.yml',
                'gh run watch "$pages_run_id"',
                '--json databaseId,createdAt,headBranch',
                'run.get("headBranch") == branch',
                'uses: ./.github/workflows/live-browser-canary.yml',
                'secrets: inherit',
                'bash scripts/install_playwright_chromium.sh',
                'bash scripts/run_live_browser_canary.sh',
                'python3 scripts/summarize_live_browser_canary.py "$report_path" >>"$GITHUB_STEP_SUMMARY"',
                'upload live browser canary artifact',
            ]
            for snippet in forbidden_snippets:
                assert snippet not in workflow_text, (
                    f"{workflow_path} should use shared deploy-pages dispatch helper: {snippet}"
                )
            assert (
                "${{ runner.temp }}/bootstrap-install" not in workflow_text
            ), f"{workflow_path} should not cache bootstrap-install"

    deploy_pages_text = Path(".github/workflows/deploy-pages.yml").read_text()
    assert "agent_task_id:" in deploy_pages_text
    assert "run-name: deploy github pages" in deploy_pages_text

    xtask_text = Path("xtask/src/agent_task.rs").read_text()
    for snippet in [
        'workflow: ".github/workflows/deploy-pages.yml"',
        'input_env("environment", "BURN_DRAGON_DEPLOY_PAGES_ENVIRONMENT")',
        "wait: true",
        "exit_status: true",
    ]:
        assert snippet in xtask_text, (
            f"xtask pages dispatch missing agent task snippet: {snippet}"
        )

    print("live-browser-canary-workflows-ok")


if __name__ == "__main__":
    main()
