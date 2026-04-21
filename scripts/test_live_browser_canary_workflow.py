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
                "bash scripts/install_playwright_chromium.sh",
                "bash scripts/run_live_browser_canary.sh",
                'python3 scripts/summarize_live_browser_canary.py "$report_path" >>"$GITHUB_STEP_SUMMARY"',
                "burn-dragon-live-browser-canary",
                "default: browser-canary-mainnet-nca",
                'BURN_DRAGON_BROWSER_CANARY_TRAIN_TIMEOUT_MS: "300000"',
                "chromium-webrtc-direct-training",
                "firefox-auto-connect",
                "firefox-webrtc-direct-connect",
                "chromium-wss-connect",
                "firefox-wss-connect",
                "continue-on-error: ${{ matrix.required == '0' }}",
                "BURN_DRAGON_BROWSER_CANARY_BROWSER: ${{ matrix.browser }}",
                "BURN_DRAGON_BROWSER_CANARY_TRANSPORT_MODE: ${{ matrix.transport_mode }}",
                "BURN_DRAGON_BROWSER_CANARY_EXPECT_TRAINING: ${{ matrix.expect_training }}",
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
        else:
            resolver_snippets = [
                'browser_canary_principal_id="browser-canary-${TF_WORKSPACE_NAME}-nca"',
                """auth_principals_json="$(python3 -c '''import json, sys; principals = json.loads(sys.argv[1] or "[]"); principal_id = sys.argv[2]; principals = [item for item in principals if item.get("principal_id") != principal_id]; print(json.dumps(principals))''' "$auth_principals_json" "$browser_canary_principal_id")" """.strip(),
                'echo "BROWSER_CANARY_PRINCIPAL_ID=$browser_canary_principal_id"',
                'echo "TF_VAR_github_browser_canary_principal_id=$browser_canary_principal_id"',
                'echo "TF_VAR_github_browser_canary_callback_token=$BROWSER_CANARY_CALLBACK_TOKEN"',
            ]
            for snippet in resolver_snippets:
                assert snippet in resolver_text, (
                    f"{BOOTSTRAP_SETTINGS_SCRIPT} missing required resolver snippet: {snippet}"
                )

            workflow_snippets = [
                'bash scripts/dispatch_pages_deploy_and_wait.sh',
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

    print("live-browser-canary-workflows-ok")


if __name__ == "__main__":
    main()
