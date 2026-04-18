from pathlib import Path

import yaml


WORKFLOW_PATHS = [
    Path(".github/workflows/deploy-burn-dragon-p2p-aws.yml"),
    Path(".github/workflows/restore-burn-dragon-p2p-aws.yml"),
    Path(".github/workflows/live-browser-canary.yml"),
]


def main() -> None:
    for workflow_path in WORKFLOW_PATHS:
        workflow_text = workflow_path.read_text()
        workflow = yaml.safe_load(workflow_text)
        jobs = workflow.get("jobs", {})
        assert jobs, f"{workflow_path} missing jobs"

        required_snippets = [
            'node scripts/live-browser-canary.mjs',
            "npx --yes playwright install --with-deps chromium",
            "## live browser canary",
            "burn-dragon-live-browser-canary",
        ]
        for snippet in required_snippets:
            assert snippet in workflow_text, f"{workflow_path} missing required snippet: {snippet}"

        if workflow_path.name != "live-browser-canary.yml":
            deploy_specific_snippets = [
                'browser_canary_principal_id="browser-canary-${TF_WORKSPACE_NAME}-nca"',
                """auth_principals_json="$(python3 -c '''import json, sys; principals = json.loads(sys.argv[1] or "[]"); principal_id = sys.argv[2]; principals = [item for item in principals if item.get("principal_id") != principal_id]; print(json.dumps(principals))''' "$auth_principals_json" "$browser_canary_principal_id")" """.strip(),
                'echo "BROWSER_CANARY_PRINCIPAL_ID=$browser_canary_principal_id"',
                'echo "TF_VAR_github_browser_canary_principal_id=$browser_canary_principal_id"',
                'echo "TF_VAR_github_browser_canary_callback_token=$BROWSER_CANARY_CALLBACK_TOKEN"',
                'BROWSER_CANARY_CALLBACK_TOKEN: ${{ secrets.BURN_DRAGON_P2P_BROWSER_CANARY_CALLBACK_TOKEN }}',
                'trusted_callback_args+=(--trusted-callback-token "$BROWSER_CANARY_CALLBACK_TOKEN")',
                'rm -rf "$bootstrap_root"',
                '--force \\',
                'gh workflow run .github/workflows/deploy-pages.yml',
                'gh run watch "$pages_run_id"',
                '--json databaseId,createdAt,headBranch',
                'GITHUB_REF_NAME="$GITHUB_REF_NAME"',
                'run.get("headBranch") == branch',
            ]
            for snippet in deploy_specific_snippets:
                assert snippet in workflow_text, (
                    f"{workflow_path} missing required deploy snippet: {snippet}"
                )
            assert (
                "${{ runner.temp }}/bootstrap-install" not in workflow_text
            ), f"{workflow_path} should not cache bootstrap-install"
        else:
            assert (
                'environment:' in workflow_text
                and 'burn-dragon-p2p-${{ github.event.inputs.environment || \'production\' }}'
                in workflow_text
            ), f"{workflow_path} missing environment-scoped canary execution"
            assert (
                "description: GitHub environment suffix to run against" in workflow_text
            ), f"{workflow_path} missing canary environment selector"
            assert (
                'BURN_DRAGON_BROWSER_CANARY_CALLBACK_TOKEN: ${{ secrets.BURN_DRAGON_P2P_BROWSER_CANARY_CALLBACK_TOKEN }}'
                in workflow_text
            ), f"{workflow_path} missing browser canary callback token secret"
            assert (
                "default: browser-canary-mainnet-nca" in workflow_text
            ), f"{workflow_path} missing browser canary mainnet principal default"
            assert (
                'BURN_DRAGON_BROWSER_CANARY_TRAIN_TIMEOUT_MS: "300000"' in workflow_text
            ), f"{workflow_path} missing extended browser canary train timeout"

    print("live-browser-canary-workflows-ok")


if __name__ == "__main__":
    main()
