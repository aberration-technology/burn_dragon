#!/usr/bin/env python3

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: extract_deployment_diagnostics_from_ssm_output.py <ssm-output.json> <out.json>")

    source = Path(sys.argv[1])
    target = Path(sys.argv[2])
    if not source.exists():
        return 0

    payload = json.loads(source.read_text())
    stdout = payload.get("StandardOutputContent") or ""
    begin = "--- deployment diagnostics json begin ---"
    end = "--- deployment diagnostics json end ---"
    if begin not in stdout or end not in stdout:
        return 0

    start = stdout.index(begin) + len(begin)
    finish = stdout.index(end, start)
    body = stdout[start:finish].strip()
    if not body:
        return 0

    json.loads(body)
    target.write_text(body + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
