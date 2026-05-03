#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import selectors
import socket
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


def env_bool(name: str, default: str) -> bool:
    value = env(name, default).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise SystemExit(f"environment variable {name} must be boolean, got {value!r}")


def fetch_json(url: str, timeout: int = 30) -> Any:
    attempts = int(os.environ.get("BURN_DRAGON_NATIVE_CANARY_HTTP_ATTEMPTS", "5"))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return json.loads(response.read())
        except Exception as error:
            last_error = error
            if attempt >= attempts:
                break
            time.sleep(min(2 * attempt, 10))
    raise RuntimeError(f"failed to fetch {url} after {attempts} attempts: {last_error}")


def metric_number(metrics: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def comparable_loss_signal(
    before_metrics: dict[str, Any],
    after_metrics: dict[str, Any],
) -> tuple[str | None, float | None, float | None]:
    for key in ("train_loss", "loss"):
        before = metric_number(before_metrics, key)
        after = metric_number(after_metrics, key)
        if before is not None and after is not None:
            return key, before, after
    return None, None, None


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
        "connected_provider_peer_ids": artifact.get("connected_provider_peer_ids") or [],
        "available_profiles": artifact.get("available_profiles") or [],
        "published_artifacts": artifact.get("published_artifacts") or [],
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
                "connected_provider_peer_ids": artifact.get("connected_provider_peer_ids") or [],
                "available_profiles": artifact.get("available_profiles") or [],
                "published_artifacts": artifact.get("published_artifacts") or [],
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
        process = subprocess.Popen(
            command,
            env=proc_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        selector = selectors.DefaultSelector()
        assert process.stdout is not None
        selector.register(process.stdout, selectors.EVENT_READ)
        started = time.monotonic()
        timed_out = False
        while process.poll() is None:
            remaining = timeout_secs - (time.monotonic() - started)
            if remaining <= 0:
                timed_out = True
                process.kill()
                break
            for key, _ in selector.select(timeout=min(1.0, remaining)):
                line = key.fileobj.readline()
                if not line:
                    continue
                stdout.write(line)
                stdout.flush()
                print(line, end="", flush=True)
        tail = process.stdout.read()
        if tail:
            stdout.write(tail)
            stdout.flush()
            print(tail, end="", flush=True)
        selector.close()
        returncode = process.wait()
    if timed_out:
        tail = stdout_path.read_text(errors="replace")[-6000:]
        raise RuntimeError(
            f"command timed out after {timeout_secs}s: {' '.join(command)}\n{tail}"
        )
    if returncode != 0:
        tail = stdout_path.read_text(errors="replace")[-6000:]
        raise RuntimeError(
            f"command failed with exit {returncode}: {' '.join(command)}\n{tail}"
        )


def parse_json_stdout(path: Path) -> Any:
    text = path.read_text(errors="replace")
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError(f"command output did not contain a JSON object: {text[-2000:]}")
    return json.loads(text[start : end + 1])


def p2p_bootstrap_addresses(edge_base_url: str) -> list[str]:
    override = os.environ.get("BURN_DRAGON_NATIVE_CANARY_P2P_BOOTSTRAP_ADDRS", "").strip()
    if override:
        return [address.strip() for address in override.split(",") if address.strip()]

    parsed = urllib.parse.urlparse(edge_base_url)
    if not parsed.hostname:
        raise RuntimeError(f"edge URL has no hostname: {edge_base_url!r}")
    host = parsed.hostname
    addresses = [
        f"/dns4/{host}/tcp/4001",
        f"/dns4/{host}/udp/4001/quic-v1",
    ]
    try:
        resolved = socket.getaddrinfo(host, 4001, family=socket.AF_INET, type=socket.SOCK_STREAM)
    except OSError:
        resolved = []
    for entry in resolved:
        ip = entry[4][0]
        addresses.extend(
            [
                f"/ip4/{ip}/tcp/4001",
                f"/ip4/{ip}/udp/4001/quic-v1",
            ]
        )

    deduped = []
    for address in addresses:
        if address not in deduped:
            deduped.append(address)
    return deduped


def p2p_probe_summary(probe: dict[str, Any]) -> dict[str, Any]:
    snapshot = probe.get("snapshot") or {}
    heads = snapshot.get("heads") or []
    return {
        "connected": probe.get("connected"),
        "connected_peer_id": probe.get("connected_peer_id"),
        "address": probe.get("address"),
        "elapsed_millis": probe.get("elapsed_millis"),
        "snapshot_error": probe.get("snapshot_error"),
        "head_announcements": snapshot.get("head_announcements"),
        "directory_announcements": snapshot.get("directory_announcements"),
        "peer_directory_announcements": snapshot.get("peer_directory_announcements"),
        "merge_announcements": snapshot.get("merge_announcements"),
        "merge_window_announcements": snapshot.get("merge_window_announcements"),
        "update_announcements": snapshot.get("update_announcements"),
        "aggregate_proposal_announcements": snapshot.get("aggregate_proposal_announcements"),
        "reduction_certificate_announcements": snapshot.get("reduction_certificate_announcements"),
        "validation_quorum_announcements": snapshot.get("validation_quorum_announcements"),
        "trainer_promotion_attestation_announcements": snapshot.get(
            "trainer_promotion_attestation_announcements"
        ),
        "diffusion_promotion_certificate_announcements": snapshot.get(
            "diffusion_promotion_certificate_announcements"
        ),
        "head_ids": [head.get("head_id") for head in heads if head.get("head_id")],
    }


def probe_p2p_snapshot(
    binary: str,
    *,
    edge_base_url: str,
    storage_root: Path,
    log_path: Path,
    timeout_secs: int,
) -> dict[str, Any]:
    errors = []
    addresses = p2p_bootstrap_addresses(edge_base_url)
    for index, address in enumerate(addresses, start=1):
        candidate_log_path = log_path
        if len(addresses) > 1:
            candidate_log_path = log_path.with_name(
                f"{log_path.stem}-addr{index}{log_path.suffix}"
            )
        try:
            return probe_p2p_snapshot_address(
                binary,
                address=address,
                storage_root=storage_root,
                log_path=candidate_log_path,
                timeout_secs=timeout_secs,
            )
        except Exception as error:
            errors.append(f"{address}: {error}")
    joined = "\n".join(errors[-4:])
    raise RuntimeError(f"p2p bootstrap probes failed across {len(addresses)} addresses:\n{joined}")


def probe_p2p_snapshot_address(
    binary: str,
    *,
    address: str,
    storage_root: Path,
    log_path: Path,
    timeout_secs: int,
) -> dict[str, Any]:
    run_native(
        [
            binary,
            "probe-swarm",
            "--address",
            address,
            "--timeout-secs",
            "30",
            "--max-events",
            "96",
            "--fetch-snapshot",
            "--snapshot-timeout-secs",
            "15",
            "--output-format",
            "json",
        ],
        storage_root=storage_root,
        timeout_secs=timeout_secs,
        stdout_path=log_path,
    )
    probe = parse_json_stdout(log_path)
    if not probe.get("connected"):
        raise RuntimeError(f"p2p bootstrap probe did not connect: {p2p_probe_summary(probe)}")
    if probe.get("snapshot_error"):
        raise RuntimeError(f"p2p bootstrap snapshot fetch failed: {p2p_probe_summary(probe)}")
    if not probe.get("snapshot"):
        raise RuntimeError(f"p2p bootstrap probe did not return a snapshot: {probe}")
    return probe


def wait_for_p2p_head(
    binary: str,
    *,
    edge_base_url: str,
    head_id: str,
    storage_root: Path,
    log_dir: Path,
    timeout_secs: int,
) -> tuple[dict[str, Any], float]:
    started = time.monotonic()
    attempt = 1
    last_summary: dict[str, Any] | None = None
    last_error: str | None = None
    while time.monotonic() - started < timeout_secs:
        try:
            probe = probe_p2p_snapshot(
                binary,
                edge_base_url=edge_base_url,
                storage_root=storage_root,
                log_path=log_dir / f"p2p-probe-{attempt}.log",
                timeout_secs=60,
            )
            summary = p2p_probe_summary(probe)
            last_summary = summary
            last_error = None
            if head_id in set(summary.get("head_ids") or []):
                return summary, time.monotonic() - started
        except Exception as error:
            last_error = str(error)
        attempt += 1
        time.sleep(5)
    raise RuntimeError(
        "p2p bootstrap snapshot did not advertise canonical head "
        f"{head_id} within {timeout_secs}s; last={last_summary}; last_error={last_error}"
    )


def assert_head_provider_signal(
    head: dict[str, Any],
    p2p_signal: dict[str, Any],
    *,
    require_edge_provider: bool,
) -> dict[str, Any]:
    head_id = head.get("head_id")
    provider_peer_ids = [
        provider for provider in (head.get("provider_peer_ids") or []) if provider
    ]
    if not provider_peer_ids:
        raise RuntimeError(f"head {head_id} has no artifact provider peers: {head}")
    edge_peer_id = p2p_signal.get("connected_peer_id")
    edge_provider = bool(edge_peer_id and edge_peer_id in set(provider_peer_ids))
    if require_edge_provider and not edge_provider:
        raise RuntimeError(
            "canonical head is visible over p2p but is not edge-backed for fresh restores: "
            f"head={head_id} edge_peer_id={edge_peer_id} providers={provider_peer_ids}"
        )
    return {
        "head_id": head_id,
        "edge_peer_id": edge_peer_id,
        "provider_peer_ids": provider_peer_ids,
        "connected_provider_peer_ids": head.get("connected_provider_peer_ids") or [],
        "available_profiles": head.get("available_profiles") or [],
        "published_artifacts": head.get("published_artifacts") or [],
        "edge_provider": edge_provider,
        "non_edge_provider_count": len(
            [provider for provider in provider_peer_ids if provider != edge_peer_id]
        ),
    }


def start_validator(
    binary: str,
    *,
    edge_base_url: str,
    experiment_kind: str,
    auth_bundle: Path,
    storage_root: Path,
    log_path: Path,
    training_batch_size: int,
    training_max_iters: int,
    evaluation_max_batches: int,
    initialize_head_on_start: bool,
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
            "--training-batch-size",
            str(training_batch_size),
            "--training-max-iters",
            str(training_max_iters),
            "--evaluation-max-batches",
            str(evaluation_max_batches),
            "--initialize-head-on-start",
            str(initialize_head_on_start).lower(),
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
    settlement = report.get("diffusion_settlement")
    if settlement:
        if not settlement.get("enabled"):
            raise RuntimeError(f"diffusion settlement was requested but not enabled: {settlement}")
        if int(settlement.get("passes_completed") or 0) <= 0:
            raise RuntimeError(f"diffusion settlement did not run: {settlement}")
        if int(settlement.get("update_announcements") or settlement.get("updates") or 0) <= 0:
            raise RuntimeError(f"diffusion settlement saw no trainer updates: {settlement}")
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
    before_metrics = before.get("metrics") or {}
    after_metrics = after.get("metrics") or {}
    before_loss = metric_number(before_metrics, "train_loss", "loss")
    after_loss = require_metric_number(after_metrics, "train_loss", "loss")
    comparable_loss_key, comparable_before_loss, comparable_after_loss = comparable_loss_signal(
        before_metrics,
        after_metrics,
    )
    if (
        comparable_before_loss is not None
        and comparable_after_loss is not None
        and comparable_after_loss > comparable_before_loss + 1e-6
    ):
        raise RuntimeError(
            "canonical loss regressed after native training window: "
            f"metric={comparable_loss_key} before={comparable_before_loss} "
            f"after={comparable_after_loss}"
        )
    return {
        "canonical_loss_before": before_loss,
        "canonical_loss_after": after_loss,
        "canonical_loss_delta": None if before_loss is None else after_loss - before_loss,
        "canonical_loss_improved": None
        if comparable_before_loss is None or comparable_after_loss is None
        else comparable_after_loss <= comparable_before_loss,
        "canonical_loss_metric": comparable_loss_key,
        "comparable_loss_before": comparable_before_loss,
        "comparable_loss_after": comparable_after_loss,
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
    training_batch_size = int(env("BURN_DRAGON_NATIVE_CANARY_TRAINING_BATCH_SIZE", "1"))
    training_max_iters = int(env("BURN_DRAGON_NATIVE_CANARY_TRAINING_MAX_ITERS", "4"))
    evaluation_max_batches = int(env("BURN_DRAGON_NATIVE_CANARY_EVALUATION_MAX_BATCHES", "1"))
    head_sync_timeout_secs = int(env("BURN_DRAGON_NATIVE_CANARY_HEAD_SYNC_TIMEOUT_SECS", "300"))
    settle_diffusion = env_bool("BURN_DRAGON_NATIVE_CANARY_SETTLE_DIFFUSION", "1")
    diffusion_settle_passes = int(env("BURN_DRAGON_NATIVE_CANARY_DIFFUSION_SETTLE_PASSES", "3"))
    serve_after_publish_secs = int(
        env("BURN_DRAGON_NATIVE_CANARY_SERVE_AFTER_PUBLISH_SECS", "120")
    )
    command_timeout_secs = int(env("BURN_DRAGON_NATIVE_CANARY_COMMAND_TIMEOUT_SECS", "1800"))
    canonical_timeout_secs = int(env("BURN_DRAGON_NATIVE_CANARY_CANONICAL_TIMEOUT_SECS", "900"))
    p2p_timeout_secs = int(env("BURN_DRAGON_NATIVE_CANARY_P2P_TIMEOUT_SECS", "300"))
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

    trainer_storage = artifact_dir / "trainer-enroll-storage"
    validator_storage = artifact_dir / "validator-storage"
    probe_storage = artifact_dir / "probe-storage"
    trainer_bundle = artifact_dir / "trainer-auth-bundle.json"
    validator_bundle = artifact_dir / "validator-auth-bundle.json"

    head_before = current_directory_head(edge_base_url, experiment_id)
    p2p_before = p2p_probe_summary(
        probe_p2p_snapshot(
            binary,
            edge_base_url=edge_base_url,
            storage_root=probe_storage,
            log_path=artifact_dir / "p2p-probe-before.log",
            timeout_secs=60,
        )
    )
    head_provider_before = None
    if head_before.get("head_id"):
        head_provider_before = assert_head_provider_signal(
            head_before,
            p2p_before,
            require_edge_provider=True,
        )
    initialize_head_on_start = not bool(head_before.get("head_id"))
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
        training_batch_size=training_batch_size,
        training_max_iters=training_max_iters,
        evaluation_max_batches=evaluation_max_batches,
        initialize_head_on_start=initialize_head_on_start,
    )
    window_reports: list[dict[str, Any]] = []
    try:
        previous_head = head_before
        for window_index in range(windows):
            report_path = artifact_dir / f"train-window-{window_index + 1}.json"
            window_trainer_storage = artifact_dir / f"trainer-window-{window_index + 1}-storage"
            train_command = [
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
                str(initialize_head_on_start).lower(),
                "--restore-head-on-start",
                "true",
                "--training-batch-size",
                str(training_batch_size),
                "--training-max-iters",
                str(training_max_iters),
                "--evaluation-max-batches",
                str(evaluation_max_batches),
                "--head-sync-timeout-secs",
                str(head_sync_timeout_secs),
                "--serve-after-publish-secs",
                str(serve_after_publish_secs),
                "--require-head-advanced",
                "--output",
                str(report_path),
                "--output-format",
                "json",
            ]
            if settle_diffusion:
                train_command.extend(
                    [
                        "--settle-diffusion",
                        "--diffusion-settle-passes",
                        str(diffusion_settle_passes),
                    ]
                )
            run_native(
                train_command,
                storage_root=window_trainer_storage,
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
            p2p_signal, p2p_wait_secs = wait_for_p2p_head(
                binary,
                edge_base_url=edge_base_url,
                head_id=advanced_head["head_id"],
                storage_root=probe_storage,
                log_dir=artifact_dir / f"p2p-window-{window_index + 1}",
                timeout_secs=p2p_timeout_secs,
            )
            head_provider_signal = assert_head_provider_signal(
                advanced_head,
                p2p_signal,
                require_edge_provider=True,
            )
            window_reports.append(
                {
                    "window_index": window_index + 1,
                    "head_before": previous_head,
                    "train_report": train_report,
                    "train_signal": train_signal,
                    "head_after": advanced_head,
                    "canonical_wait_secs": wait_secs,
                    "canonical_signal": canonical_signal,
                    "p2p_wait_secs": p2p_wait_secs,
                    "p2p_signal": p2p_signal,
                    "head_provider_signal": head_provider_signal,
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
        "training_batch_size": training_batch_size,
        "training_max_iters": training_max_iters,
        "evaluation_max_batches": evaluation_max_batches,
        "head_sync_timeout_secs": head_sync_timeout_secs,
        "settle_diffusion": settle_diffusion,
        "diffusion_settle_passes": diffusion_settle_passes,
        "serve_after_publish_secs": serve_after_publish_secs,
        "p2p_timeout_secs": p2p_timeout_secs,
        "initialize_head_on_start": initialize_head_on_start,
        "principal_id": principal_id,
        "validator_principal_id": validator_principal_id,
        "head_before": head_before,
        "p2p_before": p2p_before,
        "head_provider_before": head_provider_before,
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
