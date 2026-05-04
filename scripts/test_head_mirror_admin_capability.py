#!/usr/bin/env python3
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_TF = REPO_ROOT / "crates" / "burn_dragon_p2p" / "deploy" / "terraform" / "aws" / "main.tf"
DEPLOY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "deploy-burn-dragon-p2p-aws.yml"
RESTORE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "restore-burn-dragon-p2p-aws.yml"
BOOTSTRAP_SETTINGS_SCRIPT = REPO_ROOT / "scripts" / "resolve_bootstrap_stack_settings.sh"


def assert_contains(text: str, needle: str, context: str) -> None:
    if needle not in text:
        raise SystemExit(f"missing {needle!r} in {context}")


def main() -> None:
    terraform = MAIN_TF.read_text()
    assert_contains(terraform, '"RegisterLiveHead"', str(MAIN_TF))
    assert_contains(
        terraform,
        'canary             = "native-trainer"\n          admin_capabilities = "register_live_head"',
        str(MAIN_TF),
    )

    resolver_text = BOOTSTRAP_SETTINGS_SCRIPT.read_text()
    assert_contains(
        resolver_text,
        '"bootstrap_head_mirror": "true"',
        str(BOOTSTRAP_SETTINGS_SCRIPT),
    )
    assert_contains(
        resolver_text,
        '"admin_capabilities": "register_live_head,rollout_auth_policy"',
        str(BOOTSTRAP_SETTINGS_SCRIPT),
    )

    deploy_workflow = DEPLOY_WORKFLOW.read_text()
    restore_workflow = RESTORE_WORKFLOW.read_text()
    for path, text in (
        (DEPLOY_WORKFLOW, deploy_workflow),
        (RESTORE_WORKFLOW, restore_workflow),
    ):
        assert_contains(text, "admin-rollout-profile", str(path))
        assert_contains(text, "--recover-current-head-from-visible-root", str(path))
        assert_contains(text, "--require-directory-entry-published", str(path))

    print("head-mirror-admin-capability-ok")


if __name__ == "__main__":
    main()
