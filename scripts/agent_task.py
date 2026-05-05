#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    xtask_bin = os.environ.get("BURN_DRAGON_XTASK_BIN")
    if xtask_bin:
        command = [xtask_bin, "agent-task", *sys.argv[1:]]
    else:
        command = [
            os.environ.get("CARGO", "cargo"),
            "run",
            "-p",
            "xtask",
            "--",
            "agent-task",
            *sys.argv[1:],
        ]
    return subprocess.call(command, cwd=repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
