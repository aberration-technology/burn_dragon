#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_module():
    path = Path("scripts/summarize_github_run.py")
    spec = importlib.util.spec_from_file_location("summarize_github_run", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_module()
    active_job = {
        "name": "deploy",
        "status": "in_progress",
        "steps": [
            {"name": "terraform apply", "status": "completed", "conclusion": "success"},
            {"name": "edge health", "status": "in_progress", "conclusion": None},
        ],
    }
    assert module.active_step(active_job) == "edge health"

    failed_job = {
        "name": "canary",
        "status": "completed",
        "conclusion": "failure",
        "steps": [
            {"name": "build", "status": "completed", "conclusion": "success"},
            {"name": "run live canary", "status": "completed", "conclusion": "failure"},
        ],
    }
    assert module.failed_step(failed_job) == "run live canary"
    assert module.interesting_log_lines(
        "ok\nwarning\nError: canonical head did not advance\nmore\n",
        2,
    ) == ["Error: canonical head did not advance"]

    dispatch_pages = Path("scripts/dispatch_pages_deploy_and_wait.sh").read_text()
    dispatch_native = Path("scripts/dispatch_native_training_canary_and_wait.sh").read_text()
    for text in (dispatch_pages, dispatch_native):
        assert "scripts/summarize_github_run.py" in text
        assert "gh run watch" not in text
        assert "--watch" in text
        assert "--exit-status" in text

    print("summarize-github-run-ok")


if __name__ == "__main__":
    main()
