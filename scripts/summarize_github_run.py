#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_task import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(["gh-wait", *sys.argv[1:]]))
