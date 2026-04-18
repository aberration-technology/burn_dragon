from pathlib import Path

import yaml


def main() -> None:
    workflow_path = Path(".github/workflows/deploy-pages.yml")
    workflow = yaml.safe_load(workflow_path.read_text())

    on_clause = workflow.get("on") or workflow.get(True) or {}
    assert "workflow_dispatch" in on_clause, "deploy-pages.yml missing workflow_dispatch"

    workflow_text = workflow_path.read_text()
    required_snippets = [
        'with urllib.request.urlopen(f"{edge_base_url}/browser/seeds/signed", timeout=10) as response:',
        'for attempt in range(3):',
        'time.sleep(2)',
        'with urllib.request.urlopen(f"{edge_base_url}/portal/snapshot", timeout=10) as response:',
        'def is_dialable_browser_seed(value: str) -> bool:',
        'def is_direct_browser_seed(value: str) -> bool:',
        'def snapshot_advertises_direct_transports(transports: dict[str, object]) -> bool:',
        'if "webrtc-direct" in segments:',
        'segments[0] in {"ip4", "ip6"} and "certhash" in segments',
        'if "webtransport" in segments:',
        'return "quic-v1" in segments and "certhash" in segments',
        'browser pages deploy refusing to publish degraded WSS-only config while direct browser transports are advertised',
        'seed_urls.append(f"/dns4/{host}/tcp/443/wss")',
        'browser pages deploy requires at least one browser-capable seed multiaddr',
        'canary_principal_id:',
        'needs:\n      - build\n      - deploy',
        'BURN_DRAGON_BROWSER_CANARY_SITE_BASE_URL: ${{ needs.build.outputs.site_base_url }}',
        'BURN_DRAGON_BROWSER_CANARY_EDGE_BASE_URL: ${{ needs.build.outputs.edge_base_url }}',
        'BURN_DRAGON_BROWSER_CANARY_PRINCIPAL_ID: ${{ needs.build.outputs.canary_principal_id }}',
        'BURN_DRAGON_BROWSER_CANARY_CALLBACK_TOKEN: ${{ secrets.BURN_DRAGON_P2P_BROWSER_CANARY_CALLBACK_TOKEN }}',
        'node scripts/live-browser-canary.mjs',
        'burn-dragon-live-browser-canary',
    ]
    for snippet in required_snippets:
        assert snippet in workflow_text, f"deploy-pages.yml missing required snippet: {snippet}"

    print("deploy-pages-workflow-ok")


if __name__ == "__main__":
    main()
