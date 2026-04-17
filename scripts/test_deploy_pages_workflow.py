from pathlib import Path

import yaml


def main() -> None:
    workflow_path = Path(".github/workflows/deploy-pages.yml")
    workflow = yaml.safe_load(workflow_path.read_text())

    on_clause = workflow.get("on") or workflow.get(True) or {}
    assert "workflow_dispatch" in on_clause, "deploy-pages.yml missing workflow_dispatch"

    workflow_text = workflow_path.read_text()
    required_snippets = [
        'with urllib.request.urlopen(f"{edge_base_url}/portal/snapshot", timeout=10) as response:',
        'seed_urls.append(f"/dns4/{host}/tcp/443/wss")',
        'browser pages deploy requires at least one browser-capable seed multiaddr',
    ]
    for snippet in required_snippets:
        assert snippet in workflow_text, f"deploy-pages.yml missing required snippet: {snippet}"

    print("deploy-pages-workflow-ok")


if __name__ == "__main__":
    main()
