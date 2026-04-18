#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "deploy-burn-dragon-p2p-aws.yml"
RESTORE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "restore-burn-dragon-p2p-aws.yml"
README = REPO_ROOT / "crates" / "burn_dragon_p2p" / "deploy" / "README.md"
MAIN_TF = REPO_ROOT / "crates" / "burn_dragon_p2p" / "deploy" / "terraform" / "aws" / "main.tf"


def workflow_inputs(path: Path) -> dict[str, object]:
    workflow = yaml.safe_load(path.read_text())
    on_config = workflow.get("on", workflow.get(True))
    assert on_config is not None, f"{path} missing workflow_dispatch block"
    return on_config["workflow_dispatch"]["inputs"]


def main() -> None:
    for path in (DEPLOY_WORKFLOW, RESTORE_WORKFLOW):
        inputs = workflow_inputs(path)
        assert inputs["bootstrap_install_source"]["default"] == "crate", path.name
        assert (
            "Use `git` only when validating an unpublished burn_p2p revision"
            in inputs["bootstrap_install_source"]["description"]
        ), path.name

    deploy_inputs = workflow_inputs(DEPLOY_WORKFLOW)
    restore_inputs = workflow_inputs(RESTORE_WORKFLOW)
    assert deploy_inputs["managed_trainer_desired_capacity"]["default"] == "0"
    assert deploy_inputs["managed_trainer_backend"]["default"] == "cpu"
    assert restore_inputs["plan_only"]["default"] is True

    readme = README.read_text()
    required_snippets = [
        "keep `bootstrap_install_source=crate` for production deploys and restores",
        "use `bootstrap_install_source=git` only when validating an unpublished `burn_p2p` revision before release",
        "leave the managed trainer pool at `0` until the control plane and browser path are stable under the intended traffic pattern",
        "keep restore drills on `plan_only=true` until you are intentionally executing a failover",
        "The supported production bootstrap path is the published `burn_p2p_bootstrap` crate.",
        "use `git` only when validating an unpublished upstream `burn_p2p` revision.",
        "deploy-pages.yml` now runs the live browser canary after the Pages publish completes",
        "keep the Route53 edge health check on `https://${BURN_DRAGON_P2P_EDGE_DOMAIN_NAME}/portal/snapshot`, not a raw TCP 443 probe",
        "keep the post-deploy Pages browser canary green before treating a browser publish as complete",
    ]
    for snippet in required_snippets:
        assert snippet in readme, f"README missing required strategy snippet: {snippet}"

    main_tf = MAIN_TF.read_text()
    assert 'resource "aws_route53_health_check" "edge_primary"' in main_tf
    assert 'type              = "HTTPS"' in main_tf
    assert 'resource_path     = "/portal/snapshot"' in main_tf
    assert 'enable_sni        = true' in main_tf

    print("deployment-strategy-ok")


if __name__ == "__main__":
    main()
