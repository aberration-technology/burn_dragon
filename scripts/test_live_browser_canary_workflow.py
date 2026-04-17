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
                '"display_name": "burn_dragon live browser canary"',
                '"BrowserTrainerWgpu", "BrowserObserver"',
                '"browser_canary": "true"',
                'echo "BROWSER_CANARY_PRINCIPAL_ID=$browser_canary_principal_id"',
            ]
            for snippet in deploy_specific_snippets:
                assert snippet in workflow_text, (
                    f"{workflow_path} missing required deploy snippet: {snippet}"
                )

    print("live-browser-canary-workflows-ok")


if __name__ == "__main__":
    main()
