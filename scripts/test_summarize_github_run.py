#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


def main() -> None:
    xtask_source = Path("xtask/src/agent_task.rs").read_text()
    for snippet in [
        "pub enum AgentTaskCommand",
        "GhDispatch(GhDispatchArgs)",
        "GhWait(GhWaitArgs)",
        "fn summarize_github_run(",
        "fn interesting_log_lines(",
        "fn active_step(",
        "fn failed_step(",
        "pub fn dispatch_pages_deploy_and_wait()",
        "pub fn dispatch_native_training_canary_and_wait()",
    ]:
        assert snippet in xtask_source, f"xtask agent task source missing {snippet}"

    dispatch_pages = Path("scripts/dispatch_pages_deploy_and_wait.sh").read_text()
    dispatch_native = Path("scripts/dispatch_native_training_canary_and_wait.sh").read_text()
    assert '"${CARGO:-cargo}" run -p xtask -- dispatch-pages-deploy-and-wait' in dispatch_pages
    assert (
        '"${CARGO:-cargo}" run -p xtask -- dispatch-native-training-canary-and-wait'
        in dispatch_native
    )
    for text in (dispatch_pages, dispatch_native):
        assert "scripts/summarize_github_run.py" not in text
        assert "scripts/agent_task.py" not in text
        assert "gh run watch" not in text
        assert "gh workflow run" not in text

    wrapper = Path("scripts/summarize_github_run.py").read_text()
    assert '"agent-task",' in wrapper
    assert '"gh-wait",' in wrapper
    assert "BURN_DRAGON_XTASK_BIN" in wrapper

    print("summarize-github-run-ok")


if __name__ == "__main__":
    main()
