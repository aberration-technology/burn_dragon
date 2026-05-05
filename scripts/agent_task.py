#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_INTERVAL_SECS = 180
DEFAULT_DISCOVER_TIMEOUT_SECS = 150
DEFAULT_FAILED_LOG_LINES = 40
DEFAULT_TAIL_LINES = 40


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


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip().lower()).strip("-")
    return slug[:48] or "task"


def new_task_id(label: str) -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{slugify(label)}-{stamp}-{uuid.uuid4().hex[:8]}"


def repo_root_for(cwd: Path) -> Path:
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if completed.returncode == 0 and completed.stdout.strip():
        return Path(completed.stdout.strip())
    return cwd


def default_state_root(cwd: Path) -> Path:
    return repo_root_for(cwd).joinpath("target", "agent-tasks")


def task_dir_for(state_root: Path, task_id: str) -> Path:
    return state_root.joinpath(task_id)


def rel(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def task_path(task_dir: Path) -> Path:
    return task_dir.joinpath("task.json")


def read_task(task_dir: Path) -> dict[str, Any]:
    return load_json(task_path(task_dir))


def write_task(task_dir: Path, task: dict[str, Any]) -> None:
    task["updated_at"] = utc_now()
    write_json(task_path(task_dir), task)


def tail_lines(path: Path, limit: int) -> list[str]:
    if limit <= 0 or not path.exists():
        return []
    lines = [line.rstrip() for line in path.read_text(errors="replace").splitlines()]
    return lines[-limit:]


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


def run_command(
    args: list[str],
    *,
    check: bool,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=check,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def gh_json(args: list[str], retries: int = 3) -> Any:
    last: subprocess.CalledProcessError | None = None
    for attempt in range(retries):
        try:
            completed = run_command(["gh", *args], check=True)
            return json.loads(completed.stdout)
        except subprocess.CalledProcessError as error:
            last = error
            if attempt + 1 < retries:
                time.sleep(2 * (attempt + 1))
    assert last is not None
    raise last


def gh_text(args: list[str]) -> str:
    completed = subprocess.run(
        ["gh", *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.stdout


def summarize_github_run(repo: str, run_id: str, failed_log_lines: int) -> RunSummary:
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


def run_summary_lines(summary: RunSummary) -> list[str]:
    status = summary.status
    if summary.conclusion:
        status = f"{status}/{summary.conclusion}"
    lines = [f"github run {summary.run_id}: {summary.workflow_name} {status}"]
    if summary.active_job:
        detail = summary.active_job
        if summary.active_step:
            detail = f"{detail} / {summary.active_step}"
        lines.append(f"active: {detail}")
    if summary.failed_job:
        detail = summary.failed_job
        if summary.failed_step:
            detail = f"{detail} / {summary.failed_step}"
        lines.append(f"failed: {detail}")
    if summary.url:
        lines.append(f"url: {summary.url}")
    if summary.failed_log_tail:
        lines.append("failure log tail:")
        lines.extend(summary.failed_log_tail)
    return lines


def update_github_output(task: dict[str, Any]) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        if task.get("run_id"):
            handle.write(f"run_id={task['run_id']}\n")
        handle.write(f"agent_task_id={task['id']}\n")


def append_step_summary(task: dict[str, Any], summary_md: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(summary_md)
        if not summary_md.endswith("\n"):
            handle.write("\n")


def create_task(
    *,
    args: argparse.Namespace,
    kind: str,
    label: str,
    cwd: Path,
    extra: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    state_root = Path(args.state_root).resolve() if args.state_root else default_state_root(cwd)
    task_id = args.task_id or new_task_id(label)
    task_dir = task_dir_for(state_root, task_id)
    task_dir.mkdir(parents=True, exist_ok=True)
    task = {
        "id": task_id,
        "kind": kind,
        "label": label,
        "status": "queued",
        "conclusion": "",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "cwd": rel(cwd),
        "task_dir": rel(task_dir),
        "summary_path": rel(task_dir.joinpath("summary.md")),
        "stdout_path": rel(task_dir.joinpath("stdout.log")),
        "stderr_path": rel(task_dir.joinpath("stderr.log")),
        **extra,
    }
    write_task(task_dir, task)
    return task_dir, task


def render_task_summary(task: dict[str, Any]) -> str:
    lines = [
        f"# agent task {task['id']}",
        "",
        f"- Label: `{task.get('label', '')}`",
        f"- Kind: `{task.get('kind', '')}`",
        f"- Status: `{task.get('status', '')}`",
        f"- Conclusion: `{task.get('conclusion') or 'n/a'}`",
        f"- Task dir: `{task.get('task_dir', '')}`",
    ]
    if task.get("duration_secs") is not None:
        lines.append(f"- Duration: `{task['duration_secs']:.1f}s`")
    if task.get("exit_code") is not None:
        lines.append(f"- Exit code: `{task['exit_code']}`")
    if task.get("run_id"):
        lines.append(f"- GitHub run id: `{task['run_id']}`")
    if task.get("run_url"):
        lines.append(f"- GitHub run URL: {task['run_url']}")
    if task.get("command"):
        command = " ".join(shlex.quote(part) for part in task["command"])
        lines.append(f"- Command: `{command}`")
    if task.get("workflow"):
        lines.append(f"- Workflow: `{task['workflow']}`")
    if task.get("repo"):
        lines.append(f"- Repository: `{task['repo']}`")
    if task.get("failed_job"):
        failed = task["failed_job"]
        if task.get("failed_step"):
            failed = f"{failed} / {task['failed_step']}"
        lines.append(f"- Failed: `{failed}`")

    stderr_tail = task.get("stderr_tail") or []
    failed_log_tail = task.get("failed_log_tail") or []
    if failed_log_tail:
        lines.extend(["", "```text", *failed_log_tail[-DEFAULT_TAIL_LINES:], "```"])
    elif stderr_tail and task.get("conclusion") not in {"success", ""}:
        lines.extend(["", "```text", *stderr_tail[-DEFAULT_TAIL_LINES:], "```"])
    lines.append("")
    return "\n".join(lines)


def finalize_task(task_dir: Path, task: dict[str, Any]) -> int:
    summary_md = render_task_summary(task)
    Path(task["summary_path"]).write_text(summary_md, encoding="utf-8")
    write_task(task_dir, task)
    append_step_summary(task, summary_md)
    print(summary_md.rstrip(), flush=True)
    conclusion = task.get("conclusion")
    if conclusion == "success":
        return 0
    if conclusion in {"timeout", "stale"}:
        return 124
    return int(task.get("exit_code") or 1)


def terminate_process(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()


def current_log_activity(paths: list[Path]) -> float:
    latest = 0.0
    for path in paths:
        if path.exists():
            latest = max(latest, path.stat().st_mtime)
    return latest


def run_local_worker(task_dir: Path) -> int:
    task = read_task(task_dir)
    stdout_path = Path(task["stdout_path"])
    stderr_path = Path(task["stderr_path"])
    cwd = Path(task["cwd"])
    command = [str(part) for part in task["command"]]
    timeout_secs = int(task.get("timeout_secs") or 0)
    stale_secs = int(task.get("stale_secs") or 0)
    started = time.monotonic()
    last_activity = time.time()

    task.update({"status": "running", "started_at": utc_now()})
    write_task(task_dir, task)

    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        task["pid"] = process.pid
        write_task(task_dir, task)

        while process.poll() is None:
            now = time.monotonic()
            newest_activity = current_log_activity([stdout_path, stderr_path])
            if newest_activity > last_activity:
                last_activity = newest_activity
            if timeout_secs and now - started > timeout_secs:
                terminate_process(process)
                task.update({"conclusion": "timeout", "exit_code": 124})
                break
            if stale_secs and time.time() - last_activity > stale_secs:
                terminate_process(process)
                task.update({"conclusion": "stale", "exit_code": 124})
                break
            time.sleep(1)

        if not task.get("conclusion"):
            exit_code = process.wait()
            task.update(
                {
                    "conclusion": "success" if exit_code == 0 else "failure",
                    "exit_code": exit_code,
                }
            )

    task.update(
        {
            "status": "completed",
            "completed_at": utc_now(),
            "duration_secs": time.monotonic() - started,
            "stdout_tail": tail_lines(stdout_path, int(task.get("tail_lines") or DEFAULT_TAIL_LINES)),
            "stderr_tail": tail_lines(stderr_path, int(task.get("tail_lines") or DEFAULT_TAIL_LINES)),
        }
    )
    return finalize_task(task_dir, task)


def spawn_worker(task_dir: Path, worker: str) -> int:
    broker_log = task_dir.joinpath("broker.log")
    with broker_log.open("ab") as log:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), worker, "--task-dir", str(task_dir)],
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    task = read_task(task_dir)
    task["broker_pid"] = process.pid
    task["broker_log_path"] = rel(broker_log)
    write_task(task_dir, task)
    print(f"agent task {task['id']} detached")
    print(f"summary: {task['summary_path']}")
    return 0


def parse_command(command: list[str]) -> list[str]:
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("missing command after --")
    return command


def command_run(args: argparse.Namespace) -> int:
    cwd = Path(args.cwd or os.getcwd()).resolve()
    command = parse_command(args.command)
    label = args.label or Path(command[0]).name
    task_dir, task = create_task(
        args=args,
        kind="local",
        label=label,
        cwd=cwd,
        extra={
            "command": command,
            "timeout_secs": args.timeout_secs,
            "stale_secs": args.stale_secs,
            "tail_lines": args.tail_lines,
        },
    )
    print(f"agent task {task['id']} started: {label}", flush=True)
    print(f"task dir: {task['task_dir']}", flush=True)
    if args.detach:
        return spawn_worker(task_dir, "_run-worker")
    return run_local_worker(task_dir)


def parse_inputs(values: list[str]) -> list[tuple[str, str]]:
    parsed = []
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--input must use key=value form: {value}")
        key, raw = value.split("=", 1)
        if not key:
            raise SystemExit(f"--input has empty key: {value}")
        parsed.append((key, raw))
    return parsed


def discover_github_run(
    *,
    repo: str,
    workflow: str,
    task_id: str,
    branch: str,
    dispatch_started_at: int,
    timeout_secs: int,
) -> dict[str, Any] | None:
    deadline = time.monotonic() + timeout_secs
    while time.monotonic() < deadline:
        runs = gh_json(
            [
                "run",
                "list",
                "--repo",
                repo,
                "--workflow",
                workflow,
                "--limit",
                "20",
                "--json",
                "databaseId,createdAt,displayTitle,headBranch,status,url",
            ]
        )
        by_task_id = [
            run for run in runs if task_id and task_id in str(run.get("displayTitle") or "")
        ]
        if by_task_id:
            by_task_id.sort(key=lambda run: int(run.get("databaseId") or 0), reverse=True)
            return by_task_id[0]

        fallback = []
        for run in runs:
            if run.get("headBranch") != branch:
                continue
            created_at = run.get("createdAt")
            if not created_at:
                continue
            if int(parse_utc(created_at).timestamp()) >= dispatch_started_at:
                fallback.append(run)
        if fallback:
            fallback.sort(key=lambda run: int(run.get("databaseId") or 0), reverse=True)
            return fallback[0]
        time.sleep(5)
    return None


def wait_github_worker(task_dir: Path) -> int:
    task = read_task(task_dir)
    repo = task["repo"]
    run_id = str(task["run_id"])
    interval_secs = int(task.get("interval_secs") or DEFAULT_INTERVAL_SECS)
    timeout_secs = int(task.get("timeout_secs") or 0)
    stale_secs = int(task.get("stale_secs") or 0)
    failed_log_lines = int(task.get("failed_log_lines") or DEFAULT_FAILED_LOG_LINES)
    started = time.monotonic()
    last_signature_change = started
    last_signature: tuple[str, str, str, str, str, str] | None = None

    task.update({"status": "running", "started_at": task.get("started_at") or utc_now()})
    write_task(task_dir, task)

    while True:
        summary = summarize_github_run(repo, run_id, failed_log_lines)
        signature = summary.signature()
        if signature != last_signature:
            last_signature = signature
            last_signature_change = time.monotonic()
        task.update(
            {
                "github_status": summary.status,
                "github_conclusion": summary.conclusion,
                "workflow_name": summary.workflow_name,
                "display_title": summary.display_title,
                "run_url": summary.url,
                "active_job": summary.active_job,
                "active_step": summary.active_step,
                "failed_job": summary.failed_job,
                "failed_step": summary.failed_step,
                "failed_log_tail": summary.failed_log_tail,
            }
        )
        write_task(task_dir, task)

        if summary.status == "completed":
            task.update(
                {
                    "status": "completed",
                    "conclusion": summary.conclusion or "failure",
                    "exit_code": 0 if summary.conclusion == "success" else 1,
                    "completed_at": utc_now(),
                    "duration_secs": time.monotonic() - started,
                }
            )
            code = finalize_task(task_dir, task)
            if not task.get("exit_status") and task.get("conclusion") != "success":
                return 0
            return code

        if timeout_secs and time.monotonic() - started > timeout_secs:
            task.update(
                {
                    "status": "completed",
                    "conclusion": "timeout",
                    "exit_code": 124,
                    "completed_at": utc_now(),
                    "duration_secs": time.monotonic() - started,
                }
            )
            return finalize_task(task_dir, task)

        if stale_secs and time.monotonic() - last_signature_change > stale_secs:
            task.update(
                {
                    "status": "completed",
                    "conclusion": "stale",
                    "exit_code": 124,
                    "completed_at": utc_now(),
                    "duration_secs": time.monotonic() - started,
                }
            )
            return finalize_task(task_dir, task)

        time.sleep(interval_secs)


def command_gh_dispatch(args: argparse.Namespace) -> int:
    cwd = Path(args.cwd or os.getcwd()).resolve()
    label = args.label or f"gh-{Path(args.workflow).stem}"
    task_dir, task = create_task(
        args=args,
        kind="github-workflow",
        label=label,
        cwd=cwd,
        extra={
            "repo": args.repo,
            "workflow": args.workflow,
            "ref": args.ref,
            "interval_secs": args.interval_secs,
            "timeout_secs": args.timeout_secs,
            "stale_secs": args.stale_secs,
            "failed_log_lines": args.failed_log_lines,
            "exit_status": args.exit_status,
        },
    )
    inputs = parse_inputs(args.input)
    if args.agent_task_input:
        input_keys = {key for key, _ in inputs}
        if args.agent_task_input not in input_keys:
            inputs.append((args.agent_task_input, task["id"]))

    dispatch_started_at = int(time.time())
    command = [
        "gh",
        "workflow",
        "run",
        args.workflow,
        "--repo",
        args.repo,
        "--ref",
        args.ref,
    ]
    for key, value in inputs:
        command.extend(["-f", f"{key}={value}"])
    task["dispatch_command"] = command
    task["inputs"] = dict(inputs)
    write_task(task_dir, task)

    print(f"agent task {task['id']} dispatching: {args.workflow}", flush=True)
    dispatched = run_command(command, check=False, cwd=cwd)
    Path(task["stdout_path"]).write_text(dispatched.stdout, encoding="utf-8")
    Path(task["stderr_path"]).write_text(dispatched.stderr, encoding="utf-8")
    if dispatched.returncode != 0:
        task.update(
            {
                "status": "completed",
                "conclusion": "failure",
                "exit_code": dispatched.returncode,
                "stderr_tail": tail_lines(Path(task["stderr_path"]), args.tail_lines),
            }
        )
        return finalize_task(task_dir, task)

    run = discover_github_run(
        repo=args.repo,
        workflow=args.workflow,
        task_id=task["id"],
        branch=args.ref,
        dispatch_started_at=dispatch_started_at,
        timeout_secs=args.discover_timeout_secs,
    )
    if not run:
        task.update(
            {
                "status": "completed",
                "conclusion": "failure",
                "exit_code": 1,
                "stderr_tail": [
                    f"failed to discover {args.workflow} run for ref {args.ref}"
                ],
            }
        )
        return finalize_task(task_dir, task)

    task.update(
        {
            "run_id": str(run.get("databaseId")),
            "run_url": str(run.get("url") or ""),
            "github_status": str(run.get("status") or ""),
        }
    )
    write_task(task_dir, task)
    update_github_output(task)
    print(f"github run {task['run_id']} dispatched for task {task['id']}", flush=True)
    if args.detach:
        return spawn_worker(task_dir, "_gh-wait-worker")
    if args.wait:
        return wait_github_worker(task_dir)
    summary_md = render_task_summary(task)
    Path(task["summary_path"]).write_text(summary_md, encoding="utf-8")
    return 0


def command_gh_wait(args: argparse.Namespace) -> int:
    cwd = Path(args.cwd or os.getcwd()).resolve()
    label = args.label or f"gh-run-{args.run_id}"
    task_dir, task = create_task(
        args=args,
        kind="github-run",
        label=label,
        cwd=cwd,
        extra={
            "repo": args.repo,
            "run_id": args.run_id,
            "interval_secs": args.interval_secs,
            "timeout_secs": args.timeout_secs,
            "stale_secs": args.stale_secs,
            "failed_log_lines": args.failed_log_lines,
            "exit_status": args.exit_status,
        },
    )
    print(f"agent task {task['id']} waiting for github run {args.run_id}", flush=True)
    if args.detach:
        return spawn_worker(task_dir, "_gh-wait-worker")
    return wait_github_worker(task_dir)


def resolve_task_dir(state_root: Path, task_id: str | None) -> Path:
    if task_id:
        candidate = Path(task_id)
        if candidate.exists() and candidate.is_dir():
            return candidate
        return state_root.joinpath(task_id)
    tasks = sorted(
        [path for path in state_root.glob("*") if path.joinpath("task.json").exists()],
        key=lambda path: path.joinpath("task.json").stat().st_mtime,
        reverse=True,
    )
    if not tasks:
        raise SystemExit(f"no agent tasks found under {state_root}")
    return tasks[0]


def command_status(args: argparse.Namespace) -> int:
    cwd = Path(args.cwd or os.getcwd()).resolve()
    state_root = Path(args.state_root).resolve() if args.state_root else default_state_root(cwd)
    tasks = []
    for path in sorted(state_root.glob("*"), key=lambda item: item.stat().st_mtime, reverse=True):
        if path.joinpath("task.json").exists():
            tasks.append(read_task(path))
        if len(tasks) >= args.limit:
            break
    if args.json:
        print(json.dumps(tasks, indent=2, sort_keys=True))
        return 0
    for task in tasks:
        detail = task.get("run_id") or " ".join(task.get("command") or [])
        print(
            f"{task['id']}\t{task.get('status', '')}\t{task.get('conclusion') or 'n/a'}\t{detail}"
        )
    return 0


def command_summarize(args: argparse.Namespace) -> int:
    cwd = Path(args.cwd or os.getcwd()).resolve()
    state_root = Path(args.state_root).resolve() if args.state_root else default_state_root(cwd)
    task_dir = resolve_task_dir(state_root, args.task_id)
    task = read_task(task_dir)
    summary_path = Path(task["summary_path"])
    if not summary_path.exists():
        summary_path.write_text(render_task_summary(task), encoding="utf-8")
    print(summary_path.read_text(encoding="utf-8").rstrip())
    return 0


def add_common_task_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-root", default="", help="Directory for task state.")
    parser.add_argument("--task-id", default="", help="Explicit task id.")
    parser.add_argument("--cwd", default="", help="Working directory.")


def add_wait_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--interval-secs", type=int, default=DEFAULT_INTERVAL_SECS)
    parser.add_argument("--timeout-secs", type=int, default=0)
    parser.add_argument("--stale-secs", type=int, default=0)
    parser.add_argument("--failed-log-lines", type=int, default=DEFAULT_FAILED_LOG_LINES)
    parser.add_argument("--exit-status", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--watch", action="store_true", help=argparse.SUPPRESS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local broker for long agent tasks and GitHub workflow waits."
    )
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    run_parser = subparsers.add_parser("run", help="Run a local command quietly.")
    add_common_task_args(run_parser)
    run_parser.add_argument("--label", default="")
    run_parser.add_argument("--timeout-secs", type=int, default=0)
    run_parser.add_argument("--stale-secs", type=int, default=0)
    run_parser.add_argument("--tail-lines", type=int, default=DEFAULT_TAIL_LINES)
    run_parser.add_argument("--detach", action="store_true")
    run_parser.add_argument("--wait", action="store_true")
    run_parser.add_argument("command", nargs=argparse.REMAINDER)
    run_parser.set_defaults(func=command_run)

    dispatch_parser = subparsers.add_parser(
        "gh-dispatch", help="Dispatch a GitHub workflow and optionally wait quietly."
    )
    add_common_task_args(dispatch_parser)
    add_wait_args(dispatch_parser)
    dispatch_parser.add_argument("--repo", required=True)
    dispatch_parser.add_argument("--workflow", required=True)
    dispatch_parser.add_argument("--ref", required=True)
    dispatch_parser.add_argument("--label", default="")
    dispatch_parser.add_argument("--input", action="append", default=[])
    dispatch_parser.add_argument("--agent-task-input", default="agent_task_id")
    dispatch_parser.add_argument("--discover-timeout-secs", type=int, default=DEFAULT_DISCOVER_TIMEOUT_SECS)
    dispatch_parser.add_argument("--tail-lines", type=int, default=DEFAULT_TAIL_LINES)
    dispatch_parser.set_defaults(func=command_gh_dispatch)

    wait_parser = subparsers.add_parser("gh-wait", help="Wait for a GitHub run quietly.")
    add_common_task_args(wait_parser)
    add_wait_args(wait_parser)
    wait_parser.add_argument("--repo", required=True)
    wait_parser.add_argument("--run-id", required=True)
    wait_parser.add_argument("--label", default="")
    wait_parser.set_defaults(func=command_gh_wait)

    status_parser = subparsers.add_parser("status", help="List recent agent tasks.")
    add_common_task_args(status_parser)
    status_parser.add_argument("--limit", type=int, default=20)
    status_parser.add_argument("--json", action="store_true")
    status_parser.set_defaults(func=command_status)

    summarize_parser = subparsers.add_parser("summarize", help="Print a task summary.")
    add_common_task_args(summarize_parser)
    summarize_parser.add_argument("task_id", nargs="?")
    summarize_parser.set_defaults(func=command_summarize)

    run_worker = subparsers.add_parser("_run-worker")
    run_worker.add_argument("--task-dir", required=True)
    run_worker.set_defaults(func=lambda args: run_local_worker(Path(args.task_dir)))

    gh_wait_worker = subparsers.add_parser("_gh-wait-worker")
    gh_wait_worker.add_argument("--task-dir", required=True)
    gh_wait_worker.set_defaults(func=lambda args: wait_github_worker(Path(args.task_dir)))

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(args, "interval_secs") and args.interval_secs <= 0:
        raise SystemExit("--interval-secs must be positive")
    if getattr(args, "watch", False):
        args.wait = True
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
