#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
HELPER_PATH = REPO_ROOT / "scripts" / "render_bootstrap_runtime_sync_commands.py"
DEPLOY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "deploy-burn-dragon-p2p-aws.yml"
RESTORE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "restore-burn-dragon-p2p-aws.yml"

spec = importlib.util.spec_from_file_location(
    "render_bootstrap_runtime_sync_commands", HELPER_PATH
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def base_env() -> dict[str, str]:
    return {
        "BOOTSTRAP_OBJECT_URI": "s3://bucket/runtime/bootstrap.json",
        "CADDY_OBJECT_URI": "s3://bucket/runtime/Caddyfile",
        "HEAD_MIRROR_CONFIG_OBJECT_URI": "s3://bucket/runtime/bootstrap-head-mirror.toml",
        "HEAD_MIRROR_AUTH_SCRIPT_OBJECT_URI": "s3://bucket/runtime/fetch-head-mirror-auth",
        "HEAD_MIRROR_SERVICE_OBJECT_URI": "s3://bucket/runtime/head-mirror.service",
        "BOOTSTRAP_INSTALL_SOURCE": "crate",
        "BOOTSTRAP_CRATE_VERSION": "0.21.0-pre.15",
        "BOOTSTRAP_GIT_REPOSITORY": "https://github.com/aberration-technology/burn_p2p.git",
        "BOOTSTRAP_GIT_REF": "deadbeef",
        "BOOTSTRAP_FEATURES": "admin-http,metrics,browser-edge,auth-github",
        "DRAGON_GIT_REPOSITORY": "https://github.com/aberration-technology/burn_dragon.git",
        "DRAGON_GIT_REF": "main",
        "HEAD_MIRROR_REINSTALL": "false",
        "BOOTSTRAP_REINSTALL": "false",
        "BOOTSTRAP_BINARY_OBJECT_URI": "",
        "HEAD_MIRROR_BINARY_OBJECT_URI": "s3://bucket/runtime/burn_dragon_p2p_native",
    }


class BootstrapRuntimeSyncTests(unittest.TestCase):
    def test_remote_sync_waits_for_runtime_prereqs_and_aws_before_s3_ops(self) -> None:
        commands = module.generate_commands(base_env())
        prereq_index = next(
            index
            for index, command in enumerate(commands)
            if "timed out waiting for bootstrap runtime sync prerequisites" in command
        )
        prereq_command = commands[prereq_index]
        ensure_aws_index = next(
            index for index, command in enumerate(commands) if "awscli-exe-linux-x86_64.zip" in command
        )
        first_s3_index = next(
            index for index, command in enumerate(commands) if command.startswith("aws s3 cp ")
        )
        self.assertLess(prereq_index, ensure_aws_index)
        self.assertLess(ensure_aws_index, first_s3_index)
        self.assertIn("runtime_sync_ready=1; break;", prereq_command)
        self.assertNotIn("exit 0;", prereq_command)

    def test_crate_path_installs_bootstrap_when_binary_missing(self) -> None:
        commands = module.generate_commands(base_env())
        joined = "\n".join(commands)
        self.assertIn("cargo install --locked burn_p2p_bootstrap", joined)
        self.assertIn("--version '0.21.0-pre.15'", joined)
        self.assertIn(
            "ln -sf /root/.cargo/bin/burn-p2p-bootstrap /usr/local/bin/burn-p2p-bootstrap",
            joined,
        )
        self.assertLess(
            commands.index("systemctl enable burn-p2p-bootstrap"),
            commands.index("systemctl restart burn-p2p-bootstrap"),
        )
        self.assertIn("systemctl enable burn-dragon-p2p-head-mirror", joined)
        self.assertNotIn("systemctl restart burn-dragon-p2p-head-mirror", joined)

    def test_git_path_requires_and_uses_bootstrap_git_ref(self) -> None:
        env = base_env()
        env["BOOTSTRAP_INSTALL_SOURCE"] = "git"
        env["BOOTSTRAP_GIT_REF"] = "b14acc12"
        env["BOOTSTRAP_REINSTALL"] = "true"
        commands = module.generate_commands(env)
        joined = "\n".join(commands)
        self.assertIn(
            "cargo install --locked --git 'https://github.com/aberration-technology/burn_p2p.git' --rev 'b14acc12' burn_p2p_bootstrap",
            joined,
        )

    def test_prebuilt_bootstrap_binary_skips_bootstrap_cargo_install(self) -> None:
        env = base_env()
        env["BOOTSTRAP_BINARY_OBJECT_URI"] = "s3://bucket/runtime/burn-p2p-bootstrap"
        commands = module.generate_commands(env)
        joined = "\n".join(commands)
        self.assertIn(
            "aws s3 cp 's3://bucket/runtime/burn-p2p-bootstrap' /usr/local/bin/burn-p2p-bootstrap",
            joined,
        )
        self.assertNotIn("cargo install --locked burn_p2p_bootstrap", joined)

    def test_missing_prebuilt_bootstrap_binary_is_rejected_for_git_sync_without_reinstall(self) -> None:
        env = base_env()
        env["BOOTSTRAP_INSTALL_SOURCE"] = "git"
        env["BOOTSTRAP_REINSTALL"] = "false"
        with self.assertRaises(SystemExit):
            module.generate_commands(env)

    def test_missing_prebuilt_head_mirror_binary_is_rejected_without_reinstall(self) -> None:
        env = base_env()
        env["HEAD_MIRROR_REINSTALL"] = "false"
        env["HEAD_MIRROR_BINARY_OBJECT_URI"] = ""
        with self.assertRaises(SystemExit):
            module.generate_commands(env)

    def test_head_mirror_reinstall_path_is_guarded(self) -> None:
        env = base_env()
        env["HEAD_MIRROR_REINSTALL"] = "true"
        env["HEAD_MIRROR_BINARY_OBJECT_URI"] = ""
        commands = module.generate_commands(env)
        joined = "\n".join(commands)
        self.assertIn("cargo install --locked --git", joined)
        self.assertIn("burn_dragon_p2p --bin burn_dragon_p2p_native", joined)

    def test_missing_git_ref_is_rejected(self) -> None:
        env = base_env()
        env["BOOTSTRAP_INSTALL_SOURCE"] = "git"
        env["BOOTSTRAP_GIT_REF"] = ""
        with self.assertRaises(SystemExit):
            module.generate_commands(env)

    def test_deploy_and_restore_workflows_do_not_require_host_replacement_for_git_sync(self) -> None:
        for path in (DEPLOY_WORKFLOW, RESTORE_WORKFLOW):
            text = path.read_text()
            self.assertNotIn(
                "git-based burn_p2p bootstrap deploys must replace the bootstrap host.",
                text,
                path.name,
            )
            self.assertIn(
                "scripts/sync-bootstrap-runtime-config.sh",
                text,
                path.name,
            )
            self.assertIn(
                "timed out waiting for bootstrap edge enable/start",
                text,
                path.name,
            )
            self.assertIn(
                '/usr/local/bin/burn-dragon-p2p-sync-secrets',
                text,
                path.name,
            )
            self.assertIn(
                'missing /usr/local/bin/burn-p2p-bootstrap',
                text,
                path.name,
            )

    def test_sync_script_times_out_when_ssm_command_never_succeeds(self) -> None:
        text = (REPO_ROOT / "scripts" / "sync-bootstrap-runtime-config.sh").read_text()
        self.assertIn(
            'if [ "${invocation_status:-}" != "Success" ]; then',
            text,
        )
        self.assertIn(
            "timed out waiting for bootstrap runtime config sync to finish",
            text,
        )


if __name__ == "__main__":
    unittest.main()
