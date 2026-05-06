#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
XTASK_TARGET = tempfile.TemporaryDirectory(prefix="burn-dragon-agent-task-")
XTASK_BIN: Path | None = None


def cargo_bin() -> str:
    direct = Path("/home/mosure/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin/cargo")
    if direct.exists():
        return str(direct)
    return os.environ.get("CARGO", "cargo")


def ensure_xtask_bin() -> Path:
    global XTASK_BIN
    if XTASK_BIN is not None:
        return XTASK_BIN
    prebuilt = os.environ.get("BURN_DRAGON_XTASK_BIN")
    if prebuilt:
        XTASK_BIN = Path(prebuilt)
        assert XTASK_BIN.exists(), f"BURN_DRAGON_XTASK_BIN does not exist: {XTASK_BIN}"
        return XTASK_BIN
    target_dir = Path(XTASK_TARGET.name)
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(target_dir)
    completed = subprocess.run(
        [
            cargo_bin(),
            "build",
            "--manifest-path",
            str(REPO_ROOT / "Cargo.toml"),
            "-p",
            "xtask",
        ],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    XTASK_BIN = target_dir / "debug" / "xtask"
    assert XTASK_BIN.exists()
    return XTASK_BIN


def run_agent(args: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(ensure_xtask_bin()), "agent-task", *args],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def task_files(state_root: Path) -> list[Path]:
    return sorted(state_root.glob("*/task.json"))


def test_local_quiet_run() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        state_root = Path(tmp) / "agent-tasks"
        completed = run_agent(
            [
                "run",
                "--state-root",
                str(state_root),
                "--label",
                "quiet-ok",
                "--",
                sys.executable,
                "-c",
                "print('hidden command output')",
            ]
        )
        assert completed.returncode == 0, completed.stderr
        assert "\nhidden command output\n" not in completed.stdout
        assert "Conclusion: `success`" in completed.stdout
        task = json.loads(task_files(state_root)[0].read_text())
        assert Path(task["stdout_path"]).read_text().strip() == "hidden command output"


def test_detached_local_run() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        state_root = Path(tmp) / "agent-tasks"
        completed = run_agent(
            [
                "run",
                "--state-root",
                str(state_root),
                "--label",
                "detached-ok",
                "--detach",
                "--",
                sys.executable,
                "-c",
                "import time; time.sleep(0.2); print('done')",
            ]
        )
        assert completed.returncode == 0, completed.stderr
        assert "detached" in completed.stdout
        deadline = time.time() + 5
        while time.time() < deadline:
            status = run_agent(["status", "--state-root", str(state_root), "--json"])
            tasks = json.loads(status.stdout)
            if tasks and tasks[0].get("status") == "completed":
                assert tasks[0].get("conclusion") == "success"
                return
            time.sleep(0.1)
        raise AssertionError("detached task did not complete")


def write_fake_gh(bin_dir: Path) -> Path:
    gh = bin_dir / "gh"
    gh.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json
            import os
            import sys
            from pathlib import Path

            args = sys.argv[1:]
            state_path = Path(os.environ["GH_FAKE_STATE"])
            state = json.loads(state_path.read_text()) if state_path.exists() else {}

            if args[:3] == ["workflow", "run", ".github/workflows/deploy-pages.yml"]:
                for idx, value in enumerate(args):
                    if value == "-f" and idx + 1 < len(args):
                        key, _, raw = args[idx + 1].partition("=")
                        if key == "agent_task_id":
                            state["agent_task_id"] = raw
                state_path.write_text(json.dumps(state))
                sys.exit(0)

            if args[:2] == ["run", "list"]:
                task_id = state.get("agent_task_id", "missing-task")
                print(json.dumps([
                    {
                        "databaseId": 12345,
                        "createdAt": "2026-05-05T00:00:00Z",
                        "displayTitle": f"deploy github pages {task_id}",
                        "headBranch": "main",
                        "status": "completed",
                        "url": "https://github.test/runs/12345",
                    }
                ]))
                sys.exit(0)

            if args[:2] == ["run", "view"] and "--json" in args:
                print(json.dumps({
                    "conclusion": "success",
                    "displayTitle": "deploy github pages",
                    "jobs": [],
                    "status": "completed",
                    "url": "https://github.test/runs/12345",
                    "workflowName": "deploy github pages",
                }))
                sys.exit(0)

            if args[:2] == ["run", "view"] and "--log-failed" in args:
                sys.exit(0)

            print(f"unexpected gh args: {args}", file=sys.stderr)
            sys.exit(2)
            """
        ),
        encoding="utf-8",
    )
    gh.chmod(0o755)
    return gh


def test_github_dispatch_wait_uses_agent_task_id() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bin_dir = root / "bin"
        bin_dir.mkdir()
        state_file = root / "fake-gh-state.json"
        write_fake_gh(bin_dir)
        env = os.environ.copy()
        env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
        env["GH_FAKE_STATE"] = str(state_file)
        state_root = root / "agent-tasks"
        completed = run_agent(
            [
                "gh-dispatch",
                "--state-root",
                str(state_root),
                "--repo",
                "owner/repo",
                "--workflow",
                ".github/workflows/deploy-pages.yml",
                "--ref",
                "main",
                "--input",
                "environment=production",
                "--wait",
                "--interval-secs",
                "1",
                "--exit-status",
            ],
            env=env,
        )
        assert completed.returncode == 0, completed.stderr
        assert "Conclusion: `success`" in completed.stdout
        fake_state = json.loads(state_file.read_text())
        assert fake_state["agent_task_id"].startswith("gh-deploy-pages")
        task = json.loads(task_files(state_root)[0].read_text())
        assert task["run_id"] == "12345"
        assert task["inputs"]["agent_task_id"] == fake_state["agent_task_id"]


def test_xtask_binary_is_the_agent_entrypoint() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        state_root = Path(tmp) / "agent-tasks"
        completed = subprocess.run(
            [
                str(ensure_xtask_bin()),
                "agent-task",
                "run",
                "--state-root",
                str(state_root),
                "--label",
                "wrapper-ok",
                "--",
                sys.executable,
                "-c",
                "print('wrapped command output')",
            ],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert "Conclusion: `success`" in completed.stdout
        assert "\nwrapped command output\n" not in completed.stdout
        task = json.loads(task_files(state_root)[0].read_text())
        assert Path(task["stdout_path"]).read_text().strip() == "wrapped command output"


def main() -> None:
    test_local_quiet_run()
    test_detached_local_run()
    test_github_dispatch_wait_uses_agent_task_id()
    test_xtask_binary_is_the_agent_entrypoint()
    print("agent-task-ok")


if __name__ == "__main__":
    main()
