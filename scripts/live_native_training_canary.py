#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None or value == "":
        raise SystemExit(f"missing required environment variable {name}")
    return value


def fetch_json(url: str, timeout: int = 30) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def metric_number(metrics: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def require_metric_number(metrics: dict[str, Any], *keys: str) -> float:
    value = metric_number(metrics, *keys)
    if value is None:
        raise RuntimeError(f"missing finite metric; expected one of {', '.join(keys)}")
    return value


def fetch_head_artifact(edge_base_url: str, head_id: str) -> dict[str, Any]:
    quoted_head_id = urllib.parse.quote(head_id, safe="")
    artifact = fetch_json(f"{edge_base_url.rstrip('/')}/artifacts/heads/{quoted_head_id}")
    head = artifact.get("head") or {}
    if head.get("head_id") != head_id:
        raise RuntimeError(
            f"head artifact mismatch: expected {head_id!r}, got {head.get('head_id')!r}"
        )
    return {
        "head_id": head.get("head_id"),
        "parent_head_id": head.get("parent_head_id"),
        "artifact_id": head.get("artifact_id"),
        "global_step": head.get("global_step"),
        "metrics": head.get("metrics") or {},
        "provider_peer_ids": artifact.get("provider_peer_ids") or [],
    }


def current_directory_head(edge_base_url: str, experiment_id: str) -> dict[str, Any]:
    signed = fetch_json(f"{edge_base_url.rstrip('/')}/directory/signed")
    entries = (((signed.get("payload") or {}).get("payload") or {}).get("entries") or [])
    for entry in entries:
        if entry.get("experiment_id") == experiment_id:
            head_id = entry.get("current_head_id")
            artifact = fetch_head_artifact(edge_base_url, head_id) if head_id else {}
            return {
                "head_id": head_id,
                "revision_id": entry.get("current_revision_id"),
                "workload_id": entry.get("workload_id"),
                "generated_at": ((signed.get("payload") or {}).get("payload") or {}).get(
                    "generated_at"
                ),
                "global_step": artifact.get("global_step"),
                "artifact_id": artifact.get("artifact_id"),
                "parent_head_id": artifact.get("parent_head_id"),
                "metrics": artifact.get("metrics") or {},
                "provider_peer_ids": artifact.get("provider_peer_ids") or [],
            }
    raise RuntimeError(f"directory has no entry for experiment {experiment_id!r}")


def run_native(
    command: list[str],
    *,
    storage_root: Path,
    timeout_secs: int,
    stdout_path: Path,
) -> None:
    proc_env = os.environ.copy()
    proc_env["BURN_DRAGON_P2P_NATIVE_STORAGE_ROOT"] = str(storage_root)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w") as stdout:
        result = subprocess.run(
            command,
            env=proc_env,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_secs,
            check=False,
        )
    if result.returncode != 0:
        tail = stdout_path.read_text(errors="replace")[-6000:]
        raise RuntimeError(
            f"command failed with exit {result.returncode}: {' '.join(command)}\n{tail}"
        )


def start_validator(
    binary: str,
    *,
    edge_base_url: str,
    experiment_kind: str,
    auth_bundle: Path,
    storage_root: Path,
    log_path: Path,
) -> subprocess.Popen[str]:
    proc_env = os.environ.copy()
    proc_env["BURN_DRAGON_P2P_NATIVE_STORAGE_ROOT"] = str(storage_root)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w")
    process = subprocess.Popen(
        [
            binary,
            "run-validator-daemon",
            "--experiment-kind",
            experiment_kind,
            "--backend",
            "cpu",
            "--edge-url",
            edge_base_url,
            "--auth-bundle",
            str(auth_bundle),
            "--status-interval-secs",
            "10",
            "--validation-interval-millis",
            "500",
            "--initialize-head-on-start",
            "true",
            "--restore-head-on-start",
            "true",
        ],
        env=proc_env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(5)
    if process.poll() is not None:
        tail = log_path.read_text(errors="replace")[-6000:]
        raise RuntimeError(f"validator exited early with {process.returncode}\n{tail}")
    return process


def stop_validator(process: subprocess.Popen[str], timeout_secs: int = 20) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout_secs)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout_secs)


def wait_for_head_advance(
    edge_base_url: str,
    experiment_id: str,
    previous_head_id: str | None,
    timeout_secs: int,
) -> tuple[dict[str, Any], float]:
    started = time.monotonic()
    last_head: dict[str, Any] | None = None
    last_error: str | None = None
    while time.monotonic() - started < timeout_secs:
        try:
            last_head = current_directory_head(edge_base_url, experiment_id)
            last_error = None
        except Exception as error:
            last_error = str(error)
            time.sleep(5)
            continue
        head_id = last_head.get("head_id")
        if head_id and head_id != previous_head_id:
            return last_head, time.monotonic() - started
        time.sleep(5)
    raise RuntimeError(
        "canonical head did not advance from "
        f"{previous_head_id} within {timeout_secs}s; last={last_head}; last_error={last_error}"
    )


def assert_train_report(report: dict[str, Any]) -> dict[str, float]:
    if not report.get("can_train"):
        raise RuntimeError(f"native trainer was not train-capable: {report}")
    base_global_step = int(report.get("base_global_step") or 0)
    published_global_step = int(report.get("published_global_step") or 0)
    if published_global_step <= base_global_step:
        raise RuntimeError(
            "native trainer did not advance its local head: "
            f"base={base_global_step} published={published_global_step}"
        )
    metrics = report.get("metrics") or {}
    train_loss = require_metric_number(metrics, "train_loss", "loss")
    train_steps = require_metric_number(metrics, "train_steps", "batch_count")
    batch_count = metric_number(metrics, "batch_count") or train_steps
    if train_steps <= 0 or batch_count <= 0:
        raise RuntimeError(f"native trainer reported no work: metrics={metrics}")
    return {
        "train_loss": train_loss,
        "train_steps": train_steps,
        "batch_count": batch_count,
    }


def assert_canonical_signal(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, float | bool | None]:
    before_step = int(before.get("global_step") or 0)
    after_step = int(after.get("global_step") or 0)
    if after_step <= before_step:
        raise RuntimeError(
            "canonical head did not advance global step: "
            f"before={before_step} after={after_step}"
        )
    before_loss = metric_number(before.get("metrics") or {}, "loss", "train_loss")
    after_loss = require_metric_number(after.get("metrics") or {}, "loss", "train_loss")
    if before_loss is not None and after_loss > before_loss + 1e-6:
        raise RuntimeError(
            "canonical loss regressed after native training window: "
            f"before={before_loss} after={after_loss}"
        )
    return {
        "canonical_loss_before": before_loss,
        "canonical_loss_after": after_loss,
        "canonical_loss_delta": None if before_loss is None else after_loss - before_loss,
        "canonical_loss_improved": None if before_loss is None else after_loss <= before_loss,
    }


def enroll_static_principal(
    binary: str,
    *,
    edge_base_url: str,
    experiment_kind: str,
    backend: str,
    principal_id: str,
    principal_kind: str,
    trusted_callback_token: str,
    auth_bundle: Path,
    storage_root: Path,
    log_path: Path,
    timeout_secs: int,
) -> None:
    run_native(
        [
            binary,
            "enroll-static-principal",
            "--experiment-kind",
            experiment_kind,
            "--backend",
            backend,
            "--edge-url",
            edge_base_url,
            "--principal-id",
            principal_id,
            "--principal-hint",
            principal_id,
            "--principal-kind",
            principal_kind,
            "--trusted-callback-token",
            trusted_callback_token,
            "--auth-bundle-out",
            str(auth_bundle),
            "--output-format",
            "json",
        ],
        storage_root=storage_root,
        timeout_secs=timeout_secs,
        stdout_path=log_path,
    )


def main() -> int:
    binary = env("BURN_DRAGON_NATIVE_CANARY_BINARY", "target/debug/burn_dragon_p2p_native")
    edge_base_url = env(
        "BURN_DRAGON_NATIVE_CANARY_EDGE_BASE_URL",
        "https://edge.dragon.aberration.technology",
    ).rstrip("/")
    experiment_kind = env("BURN_DRAGON_NATIVE_CANARY_EXPERIMENT_KIND", "nca")
    experiment_id = env("BURN_DRAGON_NATIVE_CANARY_EXPERIMENT_ID", "nca-prepretraining")
    backend = env("BURN_DRAGON_NATIVE_CANARY_BACKEND", "cpu")
    principal_id = env("BURN_DRAGON_NATIVE_CANARY_PRINCIPAL_ID", "native-canary-mainnet-nca")
    validator_principal_id = env(
        "BURN_DRAGON_NATIVE_CANARY_VALIDATOR_PRINCIPAL_ID",
        f"{principal_id}-validator",
    )
    trusted_callback_token = env("BURN_DRAGON_NATIVE_CANARY_CALLBACK_TOKEN")
    windows = int(env("BURN_DRAGON_NATIVE_CANARY_WINDOWS", "2"))
    command_timeout_secs = int(env("BURN_DRAGON_NATIVE_CANARY_COMMAND_TIMEOUT_SECS", "900"))
    canonical_timeout_secs = int(env("BURN_DRAGON_NATIVE_CANARY_CANONICAL_TIMEOUT_SECS", "900"))
    artifact_dir = Path(
        env("BURN_DRAGON_NATIVE_CANARY_ARTIFACT_DIR", "/tmp/burn-dragon-native-canary")
    )
    output_json = Path(
        env(
            "BURN_DRAGON_NATIVE_CANARY_OUTPUT_JSON",
            str(artifact_dir / "native-canary-summary.json"),
        )
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    trainer_storage = artifact_dir / "trainer-storage"
    validator_storage = artifact_dir / "validator-storage"
    trainer_bundle = artifact_dir / "trainer-auth-bundle.json"
    validator_bundle = artifact_dir / "validator-auth-bundle.json"

    head_before = current_directory_head(edge_base_url, experiment_id)
    enroll_static_principal(
        binary,
        edge_base_url=edge_base_url,
        experiment_kind=experiment_kind,
        backend=backend,
        principal_id=principal_id,
        principal_kind="trainer",
        trusted_callback_token=trusted_callback_token,
        auth_bundle=trainer_bundle,
        storage_root=trainer_storage,
        log_path=artifact_dir / "enroll-trainer.log",
        timeout_secs=command_timeout_secs,
    )
    enroll_static_principal(
        binary,
        edge_base_url=edge_base_url,
        experiment_kind=experiment_kind,
        backend="cpu",
        principal_id=validator_principal_id,
        principal_kind="validator",
        trusted_callback_token=trusted_callback_token,
        auth_bundle=validator_bundle,
        storage_root=validator_storage,
        log_path=artifact_dir / "enroll-validator.log",
        timeout_secs=command_timeout_secs,
    )

    validator = start_validator(
        binary,
        edge_base_url=edge_base_url,
        experiment_kind=experiment_kind,
        auth_bundle=validator_bundle,
        storage_root=validator_storage,
        log_path=artifact_dir / "validator.log",
    )
    window_reports: list[dict[str, Any]] = []
    try:
        previous_head = head_before
        for window_index in range(windows):
            report_path = artifact_dir / f"train-window-{window_index + 1}.json"
            run_native(
                [
                    binary,
                    "train-window-once",
                    "--experiment-kind",
                    experiment_kind,
                    "--backend",
                    backend,
                    "--edge-url",
                    edge_base_url,
                    "--auth-bundle",
                    str(trainer_bundle),
                    "--initialize-head-on-start",
                    "true",
                    "--restore-head-on-start",
                    "true",
                    "--require-head-advanced",
                    "--output",
                    str(report_path),
                    "--output-format",
                    "json",
                ],
                storage_root=trainer_storage,
                timeout_secs=command_timeout_secs,
                stdout_path=artifact_dir / f"train-window-{window_index + 1}.log",
            )
            train_report = json.loads(report_path.read_text())
            train_signal = assert_train_report(train_report)
            advanced_head, wait_secs = wait_for_head_advance(
                edge_base_url,
                experiment_id,
                previous_head.get("head_id"),
                canonical_timeout_secs,
            )
            canonical_signal = assert_canonical_signal(previous_head, advanced_head)
            window_reports.append(
                {
                    "window_index": window_index + 1,
                    "head_before": previous_head,
                    "train_report": train_report,
                    "train_signal": train_signal,
                    "head_after": advanced_head,
                    "canonical_wait_secs": wait_secs,
                    "canonical_signal": canonical_signal,
                }
            )
            previous_head = advanced_head
    finally:
        stop_validator(validator)

    summary = {
        "success": True,
        "edge_base_url": edge_base_url,
        "experiment_kind": experiment_kind,
        "experiment_id": experiment_id,
        "backend": backend,
        "principal_id": principal_id,
        "validator_principal_id": validator_principal_id,
        "head_before": head_before,
        "windows": window_reports,
        "head_after": current_directory_head(edge_base_url, experiment_id),
        "catchup": fetch_json(f"{edge_base_url}/metrics/catchup/{experiment_id}"),
        "live_latest": fetch_json(f"{edge_base_url}/metrics/live/latest"),
        "leaderboard": fetch_json(f"{edge_base_url}/leaderboard/signed"),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"live native training canary failed: {error}", file=sys.stderr)
        raise
