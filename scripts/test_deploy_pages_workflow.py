from pathlib import Path

import yaml


def main() -> None:
    workflow_path = Path(".github/workflows/deploy-pages.yml")
    workflow = yaml.safe_load(workflow_path.read_text())

    on_clause = workflow.get("on") or workflow.get(True) or {}
    assert "workflow_dispatch" in on_clause, "deploy-pages.yml missing workflow_dispatch"

    workflow_text = workflow_path.read_text()
    required_snippets = [
        "python3 scripts/resolve_pages_deploy_settings.py",
        '--environment "${{ github.event.inputs.environment }}"',
        '--edge-base-url-input "${{ github.event.inputs.edge_base_url }}"',
        '--seed-node-urls-from-env "${{ vars.BURN_DRAGON_P2P_PAGES_SEED_NODE_URLS }}"',
        '--selected-revision-id-from-env "${{ vars.BURN_DRAGON_P2P_PAGES_SELECTED_REVISION_ID }}"',
        'python3 scripts/write_pages_deploy_settings_outputs.py "$settings_path"',
        'canary_principal_id:',
        'selected_revision_id: ${{ steps.resolve_browser_shell_settings.outputs.selected_revision_id }}',
        'needs:\n      - build\n      - deploy',
        'uses: ./.github/workflows/live-browser-canary.yml',
        'environment: ${{ github.event.inputs.environment }}',
        'site_base_url: ${{ needs.build.outputs.site_base_url }}',
        "edge_base_url: ${{ needs.build.outputs.edge_base_url }}",
        'principal_id: ${{ needs.build.outputs.canary_principal_id }}',
        'experiment_id: ${{ needs.build.outputs.selected_experiment_id }}',
        'secrets: inherit',
        'run: bash scripts/run_pages_predeploy_canary.sh',
        "BURN_DRAGON_BROWSER_CANARY_EDGE_BASE_URL: ${{ steps.resolve_browser_shell_settings.outputs.edge_base_url }}",
        'BURN_DRAGON_PAGES_PREDEPLOY_SITE_DIR',
        'BURN_DRAGON_BROWSER_CANARY_MIN_ACCEPTED_RECEIPTS',
        'name: burn-dragon-pages-predeploy-canary',
    ]
    for snippet in required_snippets:
        assert snippet in workflow_text, f"deploy-pages.yml missing required snippet: {snippet}"

    forbidden_snippets = [
        'with urllib.request.urlopen(f"{edge_base_url}/browser/seeds/signed", timeout=10) as response:',
        'with urllib.request.urlopen(f"{edge_base_url}/portal/snapshot", timeout=10) as response:',
        'seed_urls.append(f"/dns4/{host}/tcp/443/wss")',
        'node scripts/live-browser-canary.mjs',
        'bash scripts/install_playwright_chromium.sh',
        'bash scripts/run_live_browser_canary.sh',
        "BURN_DRAGON_BROWSER_CANARY_EDGE_BASE_URL: ${{ vars.BURN_DRAGON_P2P_PAGES_EDGE_BASE_URL || format('https://{0}', vars.BURN_DRAGON_P2P_EDGE_DOMAIN_NAME) || 'https://edge.dragon.aberration.technology' }}",
        "edge_base_url: ${{ vars.BURN_DRAGON_P2P_PAGES_EDGE_BASE_URL || format('https://{0}', vars.BURN_DRAGON_P2P_EDGE_DOMAIN_NAME) || 'https://edge.dragon.aberration.technology' }}",
    ]
    for snippet in forbidden_snippets:
        assert snippet not in workflow_text, (
            f"deploy-pages.yml should not inline duplicated deploy logic: {snippet}"
        )

    print("deploy-pages-workflow-ok")


if __name__ == "__main__":
    main()
