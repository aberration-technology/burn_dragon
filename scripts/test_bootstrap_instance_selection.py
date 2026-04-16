#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEPLOY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "deploy-burn-dragon-p2p-aws.yml"
RESTORE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "restore-burn-dragon-p2p-aws.yml"


class BootstrapInstanceSelectionTests(unittest.TestCase):
    def test_workflows_prefer_terraform_bootstrap_instance_output(self) -> None:
        terraform_probe = 'instance_id="$(terraform -chdir="$TF_ROOT" output -raw bootstrap_instance_id'
        public_ip_probe = 'Name=ip-address,Values=${public_ip}'

        for workflow in (DEPLOY_WORKFLOW, RESTORE_WORKFLOW):
            text = workflow.read_text(encoding="utf-8")
            self.assertIn(terraform_probe, text, workflow.name)
            self.assertIn(public_ip_probe, text, workflow.name)
            self.assertLess(
                text.index(terraform_probe),
                text.index(public_ip_probe),
                workflow.name,
            )


if __name__ == "__main__":
    unittest.main()
