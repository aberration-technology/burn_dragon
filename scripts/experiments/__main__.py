"""Run from the workspace root: python3 -m scripts.experiments matrix.toml."""

import argparse
from pathlib import Path
import signal

from .config import load
from .runner import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    args = parser.parse_args()
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupt)
    try:
        matrix = load(args.manifest)
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))
    try:
        result = run(matrix, args.workspace)
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    raise SystemExit(0 if result["complete"] else 1)


if __name__ == "__main__":
    main()
