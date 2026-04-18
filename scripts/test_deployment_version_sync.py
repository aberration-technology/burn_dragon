#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import re
import tomllib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEPLOY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "deploy-burn-dragon-p2p-aws.yml"
RESTORE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "restore-burn-dragon-p2p-aws.yml"
README = REPO_ROOT / "crates" / "burn_dragon_p2p" / "deploy" / "README.md"
NATIVE_PEER_EXAMPLE = (
    REPO_ROOT / "crates" / "burn_dragon_p2p" / "deploy" / "native-peer.toml.example"
)
TERRAFORM_TFVARS_EXAMPLE = (
    REPO_ROOT / "crates" / "burn_dragon_p2p" / "deploy" / "terraform" / "aws" / "terraform.tfvars.example"
)
TERRAFORM_VARIABLES = (
    REPO_ROOT / "crates" / "burn_dragon_p2p" / "deploy" / "terraform" / "aws" / "variables.tf"
)


def workspace_manifest() -> dict:
    with (REPO_ROOT / "Cargo.toml").open("rb") as handle:
        return tomllib.load(handle)


def workspace_version() -> str:
    return workspace_manifest()["workspace"]["package"]["version"]


def workspace_dependency_version(crate_name: str) -> str:
    raw = workspace_manifest()["workspace"]["dependencies"][crate_name]["version"]
    return raw.lstrip("=")


def terraform_variable_default(text: str, variable_name: str) -> str:
    pattern = re.compile(
        rf'variable "{re.escape(variable_name)}" \{{.*?default\s+=\s+"([^"]+)"',
        re.DOTALL,
    )
    match = pattern.search(text)
    if match is None:
        raise AssertionError(f"missing terraform default for {variable_name}")
    return match.group(1)


class DeploymentVersionSyncTests(unittest.TestCase):
    def test_workflow_bootstrap_defaults_track_workspace_burn_p2p_version(self) -> None:
        expected = workspace_dependency_version("burn_p2p_bootstrap")
        for path in (DEPLOY_WORKFLOW, RESTORE_WORKFLOW):
            text = path.read_text()
            self.assertIn(
                f"Example: `{expected}`.",
                text,
                path.name,
            )
            self.assertIn(
                f"default: {expected}",
                text,
                path.name,
            )
            self.assertIn(
                f'bootstrap_version="{expected}"',
                text,
                path.name,
            )

    def test_workflow_managed_trainer_default_follows_workspace_version(self) -> None:
        for path in (DEPLOY_WORKFLOW, RESTORE_WORKFLOW):
            text = path.read_text()
            self.assertIn(
                'managed_trainer_crate_version="$dragon_crate_version"',
                text,
                path.name,
            )

    def test_terraform_defaults_match_current_versions(self) -> None:
        text = TERRAFORM_VARIABLES.read_text()
        self.assertEqual(
            terraform_variable_default(text, "bootstrap_crate_version"),
            workspace_dependency_version("burn_p2p_bootstrap"),
        )
        for variable_name in (
            "dragon_crate_version",
            "managed_trainer_crate_version",
            "managed_validator_crate_version",
        ):
            self.assertEqual(
                terraform_variable_default(text, variable_name),
                workspace_version(),
                variable_name,
            )

    def test_examples_and_docs_match_current_versions(self) -> None:
        dragon_version = workspace_version()
        burn_p2p_version = workspace_dependency_version("burn_p2p_bootstrap")
        self.assertIn(
            f'app_semver = "{dragon_version}"',
            NATIVE_PEER_EXAMPLE.read_text(),
        )
        self.assertIn(
            f'# managed_trainer_crate_version = "{dragon_version}"',
            TERRAFORM_TFVARS_EXAMPLE.read_text(),
        )
        readme = README.read_text()
        self.assertIn(
            f"Defaults to `{burn_p2p_version}`.",
            readme,
        )
        self.assertIn(
            f"currently `{dragon_version}`",
            readme,
        )


if __name__ == "__main__":
    unittest.main()
