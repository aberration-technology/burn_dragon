#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from typing import Mapping


def truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def require(env: Mapping[str, str], key: str) -> str:
    value = env.get(key, "")
    if not value:
        raise SystemExit(f"{key} is required")
    return value


def bootstrap_install_command(env: Mapping[str, str]) -> str:
    install_command = (
        "export HOME=/root CARGO_HOME=/root/.cargo RUSTUP_HOME=/root/.rustup; "
        "if [ ! -x /root/.cargo/bin/cargo ]; then curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal; fi; "
        ". /root/.cargo/env; "
    )
    bootstrap_install_source = require(env, "BOOTSTRAP_INSTALL_SOURCE")
    bootstrap_features = require(env, "BOOTSTRAP_FEATURES")
    if bootstrap_install_source == "crate":
        bootstrap_crate_version = require(env, "BOOTSTRAP_CRATE_VERSION")
        install_command += (
            "cargo install --locked burn_p2p_bootstrap "
            f"--version '{bootstrap_crate_version}' "
            "--bin burn-p2p-bootstrap --no-default-features "
            f"--features '{bootstrap_features}'"
        )
    else:
        bootstrap_git_repository = require(env, "BOOTSTRAP_GIT_REPOSITORY")
        bootstrap_git_ref = require(env, "BOOTSTRAP_GIT_REF")
        install_command += (
            "cargo install --locked "
            f"--git '{bootstrap_git_repository}' "
            f"--rev '{bootstrap_git_ref}' "
            "burn_p2p_bootstrap "
            "--bin burn-p2p-bootstrap --no-default-features "
            f"--features '{bootstrap_features}'"
        )
    return install_command


def head_mirror_install_command(env: Mapping[str, str]) -> str:
    dragon_git_repository = require(env, "DRAGON_GIT_REPOSITORY")
    dragon_git_ref = require(env, "DRAGON_GIT_REF")
    return (
        "if [ ! -x /usr/local/bin/burn_dragon_p2p_native ] && [ ! -x /root/.cargo/bin/burn_dragon_p2p_native ]; then "
        "export HOME=/root CARGO_HOME=/root/.cargo RUSTUP_HOME=/root/.rustup; "
        "if [ ! -x /root/.cargo/bin/cargo ]; then curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal; fi; "
        ". /root/.cargo/env; "
        "cargo install --locked "
        f"--git '{dragon_git_repository}' "
        f"--rev '{dragon_git_ref}' "
        "burn_dragon_p2p --bin burn_dragon_p2p_native --no-default-features --features native; fi"
    )


def ensure_aws_cli_command() -> str:
    return (
        "if ! command -v aws >/dev/null 2>&1; then "
        "tmpdir=$(mktemp -d); "
        "curl -fsSL 'https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip' -o \"$tmpdir/awscliv2.zip\"; "
        "unzip -q \"$tmpdir/awscliv2.zip\" -d \"$tmpdir\"; "
        "\"$tmpdir/aws/install\" --bin-dir /usr/local/bin --install-dir /usr/local/aws-cli --update; "
        "rm -rf \"$tmpdir\"; "
        "fi"
    )


def wait_for_runtime_sync_prereqs_command() -> str:
    return (
        "runtime_sync_ready=; "
        "for attempt in $(seq 1 60); do "
        "if [ -x /usr/local/bin/burn-dragon-p2p-sync-secrets ] && command -v aws >/dev/null 2>&1; then "
        "runtime_sync_ready=1; "
        "break; "
        "fi; "
        "sleep 5; "
        "done; "
        "if [ -z \"$runtime_sync_ready\" ]; then "
        "echo 'timed out waiting for bootstrap runtime sync prerequisites' >&2; "
        "exit 1; "
        "fi"
    )


def generate_commands(env: Mapping[str, str]) -> list[str]:
    preamble = [
        "set -eu",
        "export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH",
        wait_for_runtime_sync_prereqs_command(),
        ensure_aws_cli_command(),
        "install -d -m 0755 /etc/burn-dragon-p2p /etc/caddy /etc/burn_dragon_p2p /usr/local/bin",
        "install -d -m 0700 /var/lib/burn_dragon_p2p",
    ]
    bootstrap_setup: list[str] = []
    head_mirror_setup: list[str] = []
    commands = [
        "systemctl reset-failed burn-p2p-bootstrap || true",
        "systemctl reset-failed burn-dragon-p2p-head-mirror || true",
        "aws s3 cp '{}' /etc/burn-dragon-p2p/bootstrap.json".format(
            require(env, "BOOTSTRAP_OBJECT_URI")
        ),
        "aws s3 cp '{}' /etc/caddy/Caddyfile".format(require(env, "CADDY_OBJECT_URI")),
        "aws s3 cp '{}' /etc/burn_dragon_p2p/bootstrap-head-mirror.toml".format(
            require(env, "HEAD_MIRROR_CONFIG_OBJECT_URI")
        ),
        "aws s3 cp '{}' /usr/local/bin/burn-dragon-p2p-fetch-head-mirror-auth-bundle".format(
            require(env, "HEAD_MIRROR_AUTH_SCRIPT_OBJECT_URI")
        ),
        "aws s3 cp '{}' /etc/systemd/system/burn-dragon-p2p-head-mirror.service".format(
            require(env, "HEAD_MIRROR_SERVICE_OBJECT_URI")
        ),
        "chmod 0644 /etc/burn-dragon-p2p/bootstrap.json /etc/caddy/Caddyfile /etc/burn_dragon_p2p/bootstrap-head-mirror.toml /etc/systemd/system/burn-dragon-p2p-head-mirror.service",
        "chmod 0755 /usr/local/bin/burn-dragon-p2p-fetch-head-mirror-auth-bundle",
        "/usr/local/bin/burn-dragon-p2p-sync-secrets",
        "systemctl daemon-reload",
        "systemctl restart caddy",
        "systemctl enable burn-dragon-p2p-head-mirror",
        "systemctl enable burn-p2p-bootstrap",
        "systemctl restart burn-p2p-bootstrap",
        "systemctl is-active caddy",
        "systemctl is-active burn-p2p-bootstrap",
        "journalctl -u caddy -u burn-p2p-bootstrap -u burn-dragon-p2p-head-mirror --no-pager -n 60 || true",
    ]

    bootstrap_binary_object_uri = env.get("BOOTSTRAP_BINARY_OBJECT_URI", "")
    if bootstrap_binary_object_uri:
        bootstrap_setup = [
            f"aws s3 cp '{bootstrap_binary_object_uri}' /usr/local/bin/burn-p2p-bootstrap",
            "chmod 0755 /usr/local/bin/burn-p2p-bootstrap",
        ]
    elif env.get("BOOTSTRAP_INSTALL_SOURCE") == "git" and not truthy(env.get("BOOTSTRAP_REINSTALL")):
        raise SystemExit(
            "BOOTSTRAP_BINARY_OBJECT_URI is required for git bootstrap sync when BOOTSTRAP_REINSTALL is false"
        )
    else:
        bootstrap_setup = [
            "if [ ! -x /usr/local/bin/burn-p2p-bootstrap ] && [ ! -x /root/.cargo/bin/burn-p2p-bootstrap ]; then "
            + bootstrap_install_command(env)
            + "; fi",
            "if [ -x /root/.cargo/bin/burn-p2p-bootstrap ]; then ln -sf /root/.cargo/bin/burn-p2p-bootstrap /usr/local/bin/burn-p2p-bootstrap; fi",
        ]

    head_mirror_binary_object_uri = env.get("HEAD_MIRROR_BINARY_OBJECT_URI", "")
    if head_mirror_binary_object_uri:
        head_mirror_setup = [
            f"aws s3 cp '{head_mirror_binary_object_uri}' /usr/local/bin/burn_dragon_p2p_native",
            "chmod 0755 /usr/local/bin/burn_dragon_p2p_native",
        ]
    elif not truthy(env.get("HEAD_MIRROR_REINSTALL")):
        raise SystemExit(
            "HEAD_MIRROR_BINARY_OBJECT_URI is required when HEAD_MIRROR_REINSTALL is false"
        )
    elif truthy(env.get("HEAD_MIRROR_REINSTALL")):
        head_mirror_setup = [
            head_mirror_install_command(env),
            "if [ -x /root/.cargo/bin/burn_dragon_p2p_native ]; then ln -sf /root/.cargo/bin/burn_dragon_p2p_native /usr/local/bin/burn_dragon_p2p_native; fi",
        ]

    return preamble + head_mirror_setup + bootstrap_setup + commands


def main() -> None:
    print(json.dumps({"commands": generate_commands(os.environ)}))


if __name__ == "__main__":
    main()
