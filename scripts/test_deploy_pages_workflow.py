from pathlib import Path

import yaml


def main() -> None:
    workflow_path = Path(".github/workflows/deploy-pages.yml")
    workflow = yaml.safe_load(workflow_path.read_text())

    on_clause = workflow.get("on") or workflow.get(True) or {}
    assert "workflow_dispatch" in on_clause, "deploy-pages.yml missing workflow_dispatch"

    workflow_text = workflow_path.read_text()
    required_snippets = [
        "cargo run -q -p xtask -- resolve-pages-deploy-settings",
        '--environment "${{ github.event.inputs.environment }}"',
        '--edge-base-url-input "${{ github.event.inputs.edge_base_url }}"',
        '--seed-node-urls-from-env "${{ vars.BURN_DRAGON_P2P_PAGES_SEED_NODE_URLS }}"',
        '--selected-revision-id-from-env "${{ vars.BURN_DRAGON_P2P_PAGES_SELECTED_REVISION_ID }}"',
        'python3 scripts/write_pages_deploy_settings_outputs.py "$settings_path"',
        'canary_principal_id:',
        'selected_revision_id: ${{ steps.resolve_browser_shell_settings.outputs.selected_revision_id }}',
        'needs:\n      - build\n      - deploy',
        'BURN_DRAGON_BROWSER_CANARY_SITE_BASE_URL: ${{ needs.build.outputs.site_base_url }}',
        'BURN_DRAGON_BROWSER_CANARY_EDGE_BASE_URL: ${{ needs.build.outputs.edge_base_url }}',
        'BURN_DRAGON_BROWSER_CANARY_PRINCIPAL_ID: ${{ needs.build.outputs.canary_principal_id }}',
        'BURN_DRAGON_BROWSER_CANARY_CALLBACK_TOKEN: ${{ secrets.BURN_DRAGON_P2P_BROWSER_CANARY_CALLBACK_TOKEN }}',
        'bash scripts/install_playwright_chromium.sh',
        'bash scripts/run_live_browser_canary.sh',
        'python3 scripts/summarize_live_browser_canary.py "$report_path" >>"$GITHUB_STEP_SUMMARY"',
        'burn-dragon-live-browser-canary',
    ]
    for snippet in required_snippets:
        assert snippet in workflow_text, f"deploy-pages.yml missing required snippet: {snippet}"

    forbidden_snippets = [
        'with urllib.request.urlopen(f"{edge_base_url}/browser/seeds/signed", timeout=10) as response:',
        'with urllib.request.urlopen(f"{edge_base_url}/portal/snapshot", timeout=10) as response:',
        'seed_urls.append(f"/dns4/{host}/tcp/443/wss")',
        'node scripts/live-browser-canary.mjs',
    ]
    for snippet in forbidden_snippets:
        assert snippet not in workflow_text, (
            f"deploy-pages.yml should not inline duplicated deploy logic: {snippet}"
        )

    print("deploy-pages-workflow-ok")


if __name__ == "__main__":
    main()
