#!/usr/bin/env python3

import json
import os
import sys


def main() -> int:
    private_ip = os.environ.get("BOOTSTRAP_PRIVATE_IP", "").strip()
    if not private_ip:
        raise SystemExit("BOOTSTRAP_PRIVATE_IP is required")

    replacement = f'seed_node_urls = [\n  "/ip4/{private_ip}/tcp/4001",\n]\n'
    commands = [
        "set -eu",
        (
            "python3 - <<'PY2'\n"
            "from pathlib import Path\n"
            "import re\n"
            "p = Path('/etc/burn_dragon_p2p/bootstrap-head-mirror.toml')\n"
            "text = p.read_text()\n"
            f"replacement = {replacement!r}\n"
            "text, count = re.subn(r'seed_node_urls\\s*=\\s*\\[(?:.|\\n)*?\\]\\n', replacement, text, count=1, flags=re.MULTILINE)\n"
            "if count != 1:\n"
            "    raise SystemExit('failed to rewrite seed_node_urls in bootstrap-head-mirror.toml')\n"
            "p.write_text(text)\n"
            "print(text)\n"
            "PY2"
        ),
        "/usr/local/bin/burn-dragon-p2p-fetch-head-mirror-auth-bundle || true",
        "systemctl reset-failed burn-dragon-p2p-head-mirror || true",
        "systemctl restart burn-dragon-p2p-head-mirror",
        "systemctl status burn-dragon-p2p-head-mirror --no-pager || true",
        "journalctl -u burn-dragon-p2p-head-mirror --no-pager -n 120 || true",
        "curl -fsS https://127.0.0.1/heads || true",
        "curl -fsS https://127.0.0.1/portal/snapshot || true",
    ]
    json.dump({"commands": commands}, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
