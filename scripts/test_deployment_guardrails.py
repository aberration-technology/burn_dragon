#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import unittest

import yaml


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEPLOY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "deploy-burn-dragon-p2p-aws.yml"
RESTORE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "restore-burn-dragon-p2p-aws.yml"
CLEANUP_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "cleanup-burn-dragon-p2p-aws.yml"
PUBLISH_DATASET_WORKFLOW = (
    REPO_ROOT / ".github" / "workflows" / "publish-burn-dragon-p2p-dataset.yml"
)
README = REPO_ROOT / "crates" / "burn_dragon_p2p" / "deploy" / "README.md"
IAM_POLICY_DOC = REPO_ROOT / "crates" / "burn_dragon_p2p" / "deploy" / "aws" / "github-actions-iam.md"


def workflow(path: pathlib.Path) -> dict:
    raw = yaml.safe_load(path.read_text())
    return raw


def guardrail_report(env: dict[str, str]) -> dict:
    xtask = pathlib.Path(os.environ["BURN_DRAGON_XTASK_BIN"])
    completed = subprocess.run(
        [str(xtask), "deployment-guardrail-report"],
        cwd=REPO_ROOT,
        env={**os.environ, **env},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


class DeploymentGuardrailTests(unittest.TestCase):
    def test_default_production_profile_stays_under_cap(self) -> None:
        report = guardrail_report(
            {
                "DEPLOY_ENVIRONMENT": "production",
                "DEPLOYMENT_OPERATION": "deploy",
                "TF_WORKSPACE_NAME": "mainnet",
                "BOOTSTRAP_INSTALL_SOURCE": "crate",
                "TF_VAR_alarm_sns_topic_arn": "arn:aws:sns:us-east-2:123456789012:burn-dragon-p2p-alerts",
            }
        )
        self.assertEqual(report["errors"], [])
        self.assertLess(report["fixed_monthly_cost_usd"], 100.0)

    def test_missing_production_alarm_route_is_rejected(self) -> None:
        report = guardrail_report(
            {
                "DEPLOY_ENVIRONMENT": "production",
                "DEPLOYMENT_OPERATION": "deploy",
                "TF_WORKSPACE_NAME": "mainnet",
                "BOOTSTRAP_INSTALL_SOURCE": "crate",
                "TF_VAR_alarm_sns_topic_arn": "",
            }
        )
        self.assertIn(
            "production deployment must set BURN_DRAGON_P2P_ALARM_SNS_TOPIC_ARN so CloudWatch alarms route somewhere actionable",
            report["errors"],
        )

    def test_named_alarm_sns_env_also_satisfies_guardrail(self) -> None:
        report = guardrail_report(
            {
                "DEPLOY_ENVIRONMENT": "production",
                "DEPLOYMENT_OPERATION": "deploy",
                "TF_WORKSPACE_NAME": "mainnet",
                "BOOTSTRAP_INSTALL_SOURCE": "crate",
                "BURN_DRAGON_P2P_ALARM_SNS_TOPIC_ARN": "arn:aws:sns:us-east-2:123456789012:burn-dragon-p2p-alerts",
            }
        )
        self.assertEqual(report["errors"], [])

    def test_production_git_bootstrap_is_rejected(self) -> None:
        report = guardrail_report(
            {
                "DEPLOY_ENVIRONMENT": "production",
                "DEPLOYMENT_OPERATION": "deploy",
                "TF_WORKSPACE_NAME": "mainnet",
                "BOOTSTRAP_INSTALL_SOURCE": "git",
                "TF_VAR_alarm_sns_topic_arn": "arn:aws:sns:us-east-2:123456789012:burn-dragon-p2p-alerts",
            }
        )
        self.assertIn(
            "production deployment must use bootstrap_install_source=crate, got `git`",
            report["errors"],
        )

    def test_managed_trainer_pool_exceeding_cap_is_rejected(self) -> None:
        report = guardrail_report(
            {
                "DEPLOY_ENVIRONMENT": "production",
                "DEPLOYMENT_OPERATION": "deploy",
                "TF_WORKSPACE_NAME": "mainnet",
                "BOOTSTRAP_INSTALL_SOURCE": "crate",
                "TF_VAR_alarm_sns_topic_arn": "arn:aws:sns:us-east-2:123456789012:burn-dragon-p2p-alerts",
                "TF_VAR_managed_trainer_desired_capacity": "1",
            }
        )
        self.assertTrue(
            any("exceeds the hard cap" in error for error in report["errors"]),
            report["errors"],
        )

    def test_workflows_run_guardrail_script_and_cleanup_role_is_split(self) -> None:
        for path in (DEPLOY_WORKFLOW, RESTORE_WORKFLOW):
            text = path.read_text()
            self.assertIn("cargo run -p xtask -- check-deployment-guardrails", text, path.name)
            wf = workflow(path)
            self.assertEqual(
                wf["permissions"],
                {"id-token": "write", "contents": "read", "actions": "write"},
                path.name,
            )
            self.assertIn("mask-aws-account-id: true", text, path.name)
            self.assertIn(
                'account_id="$(aws sts get-caller-identity --query Account --output text)"',
                text,
                path.name,
            )
            self.assertIn(
                'if [ "$account_id" != "$AWS_ACCOUNT_ID" ]; then',
                text,
                path.name,
            )

        cleanup_text = CLEANUP_WORKFLOW.read_text()
        self.assertIn("BURN_DRAGON_P2P_AWS_CLEANUP_ROLE_ARN", cleanup_text)

    def test_publish_dataset_workflow_uses_remote_backend(self) -> None:
        text = PUBLISH_DATASET_WORKFLOW.read_text()
        self.assertIn("prepare terraform backend", text)
        self.assertNotIn("init -backend=false", text)
        self.assertIn("TF_BACKEND_CONFIG", text)

    def test_docs_cover_budget_alarming_and_iam_policy(self) -> None:
        readme = README.read_text()
        self.assertIn("BURN_DRAGON_P2P_ALARM_SNS_TOPIC_ARN", readme)
        self.assertIn("BURN_DRAGON_P2P_AWS_CLEANUP_ROLE_ARN", readme)
        self.assertIn("under `$100`", readme)
        self.assertIn("github-actions-iam.md", readme)

        policy_doc = IAM_POLICY_DOC.read_text()
        self.assertIn("AssumeRoleWithWebIdentity", policy_doc)
        self.assertIn("burn-dragon-p2p-production", policy_doc)
        self.assertIn("BURN_DRAGON_P2P_AWS_CLEANUP_ROLE_ARN", policy_doc)


if __name__ == "__main__":
    unittest.main()
