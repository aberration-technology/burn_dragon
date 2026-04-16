#!/usr/bin/env python3
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_TF = REPO_ROOT / "crates" / "burn_dragon_p2p" / "deploy" / "terraform" / "aws" / "main.tf"
DEPLOY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "deploy-burn-dragon-p2p-aws.yml"
RESTORE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "restore-burn-dragon-p2p-aws.yml"


def assert_contains(text: str, needle: str, context: str) -> None:
    if needle not in text:
        raise SystemExit(f"missing {needle!r} in {context}")


def main() -> None:
    terraform = MAIN_TF.read_text()
    assert_contains(terraform, '"RegisterLiveHead"', str(MAIN_TF))

    deploy_workflow = DEPLOY_WORKFLOW.read_text()
    restore_workflow = RESTORE_WORKFLOW.read_text()
    for path, text in (
        (DEPLOY_WORKFLOW, deploy_workflow),
        (RESTORE_WORKFLOW, restore_workflow),
    ):
        assert_contains(text, '"bootstrap_head_mirror": "true"', str(path))
        assert_contains(text, '"admin_capabilities": "register_live_head"', str(path))

    print("head-mirror-admin-capability-ok")


if __name__ == "__main__":
    main()
