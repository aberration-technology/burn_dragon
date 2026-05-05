#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    workflow_name: str
    display_title: str
    status: str
    conclusion: str
    url: str
    active_job: str
    active_step: str
    failed_job: str
    failed_step: str
    failed_log_tail: list[str]

    def signature(self) -> tuple[str, str, str, str, str, str]:
        return (
            self.status,
            self.conclusion,
            self.active_job,
            self.active_step,
            self.failed_job,
            self.failed_step,
        )


def gh_json(args: list[str]) -> Any:
    completed = subprocess.run(
        ["gh", *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(completed.stdout)


def gh_text(args: list[str]) -> str:
    completed = subprocess.run(
        ["gh", *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.stdout


def step_name(step: dict[str, Any]) -> str:
    return str(step.get("name") or step.get("number") or "unknown")


def active_step(job: dict[str, Any]) -> str:
    for step in job.get("steps") or []:
        if step.get("status") in {"queued", "in_progress"}:
            return step_name(step)
    return ""


def failed_step(job: dict[str, Any]) -> str:
    for step in job.get("steps") or []:
        if step.get("conclusion") not in {None, "", "success", "skipped"}:
            return step_name(step)
    return ""


def interesting_log_lines(text: str, limit: int) -> list[str]:
    markers = (
        "error",
        "failed",
        "failure",
        "timed out",
        "timeout",
        "panic",
        "exception",
        "##[error]",
    )
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    interesting = [
        line for line in lines if any(marker in line.lower() for marker in markers)
    ]
    return (interesting or lines)[-limit:]


def summarize(repo: str, run_id: str, failed_log_lines: int) -> RunSummary:
    run = gh_json(
        [
            "run",
            "view",
            run_id,
            "--repo",
            repo,
            "--json",
            "conclusion,displayTitle,jobs,status,url,workflowName",
        ]
    )
    jobs = run.get("jobs") or []
    active = next(
        (job for job in jobs if job.get("status") in {"queued", "in_progress"}),
        None,
    )
    failed = next(
        (
            job
            for job in jobs
            if job.get("conclusion") not in {None, "", "success", "skipped"}
        ),
        None,
    )
    failed_log_tail: list[str] = []
    if failed is not None:
        failed_log_tail = interesting_log_lines(
            gh_text(["run", "view", run_id, "--repo", repo, "--log-failed"]),
            failed_log_lines,
        )
    return RunSummary(
        run_id=run_id,
        workflow_name=str(run.get("workflowName") or "unknown"),
        display_title=str(run.get("displayTitle") or ""),
        status=str(run.get("status") or "unknown"),
        conclusion=str(run.get("conclusion") or ""),
        url=str(run.get("url") or ""),
        active_job=str((active or {}).get("name") or ""),
        active_step=active_step(active or {}),
        failed_job=str((failed or {}).get("name") or ""),
        failed_step=failed_step(failed or {}),
        failed_log_tail=failed_log_tail,
    )


def print_summary(summary: RunSummary) -> None:
    status = summary.status
    if summary.conclusion:
        status = f"{status}/{summary.conclusion}"
    print(f"github run {summary.run_id}: {summary.workflow_name} {status}", flush=True)
    if summary.active_job:
        detail = summary.active_job
        if summary.active_step:
            detail = f"{detail} / {summary.active_step}"
        print(f"active: {detail}", flush=True)
    if summary.failed_job:
        detail = summary.failed_job
        if summary.failed_step:
            detail = f"{detail} / {summary.failed_step}"
        print(f"failed: {detail}", flush=True)
    if summary.url:
        print(f"url: {summary.url}", flush=True)
    if summary.failed_log_tail:
        print("failure log tail:", flush=True)
        for line in summary.failed_log_tail:
            print(line, flush=True)


def append_step_summary(summary: RunSummary) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"## GitHub run {summary.run_id}\n\n")
        handle.write(f"- Workflow: `{summary.workflow_name}`\n")
        handle.write(f"- Status: `{summary.status}`\n")
        handle.write(f"- Conclusion: `{summary.conclusion or 'n/a'}`\n")
        if summary.active_job:
            handle.write(f"- Active: `{summary.active_job}`")
            if summary.active_step:
                handle.write(f" / `{summary.active_step}`")
            handle.write("\n")
        if summary.failed_job:
            handle.write(f"- Failed: `{summary.failed_job}`")
            if summary.failed_step:
                handle.write(f" / `{summary.failed_step}`")
            handle.write("\n")
        if summary.url:
            handle.write(f"- URL: {summary.url}\n")
        if summary.failed_log_tail:
            handle.write("\n```text\n")
            handle.write("\n".join(summary.failed_log_tail[-20:]))
            handle.write("\n```\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a sparse, agent-friendly summary for a GitHub Actions run."
    )
    parser.add_argument("--repo", required=True, help="Repository in owner/name form.")
    parser.add_argument("--run-id", required=True, help="GitHub Actions run id.")
    parser.add_argument("--watch", action="store_true", help="Poll until completion.")
    parser.add_argument(
        "--interval-secs",
        type=int,
        default=180,
        help="Polling interval while --watch is active.",
    )
    parser.add_argument(
        "--timeout-secs",
        type=int,
        default=0,
        help="Optional overall watch timeout. Zero means no timeout.",
    )
    parser.add_argument(
        "--failed-log-lines",
        type=int,
        default=40,
        help="Number of failure lines to print when the run fails.",
    )
    parser.add_argument(
        "--exit-status",
        action="store_true",
        help="Exit nonzero when the completed run conclusion is not success.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.interval_secs <= 0:
        raise SystemExit("--interval-secs must be positive")
    started = time.monotonic()
    last_signature: tuple[str, str, str, str, str, str] | None = None
    while True:
        summary = summarize(args.repo, args.run_id, args.failed_log_lines)
        if summary.signature() != last_signature or summary.status == "completed":
            print_summary(summary)
            last_signature = summary.signature()
        if summary.status == "completed":
            append_step_summary(summary)
            if args.exit_status and summary.conclusion != "success":
                return 1
            return 0
        if not args.watch:
            return 0
        if args.timeout_secs and time.monotonic() - started > args.timeout_secs:
            print(f"github run {args.run_id}: timed out waiting for completion", file=sys.stderr)
            return 124
        time.sleep(args.interval_secs)


if __name__ == "__main__":
    raise SystemExit(main())
