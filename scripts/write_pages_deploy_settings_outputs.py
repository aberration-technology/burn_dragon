#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def write_lines(path: str | None, lines: list[str]) -> None:
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: write_pages_deploy_settings_outputs.py <settings_json_path>")

    settings = json.loads(Path(sys.argv[1]).read_text())
    seed_node_urls = ",".join(settings["seed_node_urls"])

    env_lines = [
        f'EDGE_BASE_URL={settings["edge_base_url"]}',
        f"SEED_NODE_URLS={seed_node_urls}",
        f'SELECTED_EXPERIMENT_ID={settings["selected_experiment_id"]}',
        f'SELECTED_REVISION_ID={settings["selected_revision_id"]}',
        f'BROWSER_CANARY_PRINCIPAL_ID={settings["canary_principal_id"]}',
        f'SITE_HOST={settings["site_host"]}',
    ]
    output_lines = [
        f'edge_base_url={settings["edge_base_url"]}',
        f'browser_app_base_url={settings["browser_app_base_url"]}',
        f'selected_experiment_id={settings["selected_experiment_id"]}',
        f'selected_revision_id={settings["selected_revision_id"]}',
        f'canary_principal_id={settings["canary_principal_id"]}',
    ]

    write_lines(os.environ.get("GITHUB_ENV"), env_lines)
    write_lines(os.environ.get("GITHUB_OUTPUT"), output_lines)


if __name__ == "__main__":
    main()
