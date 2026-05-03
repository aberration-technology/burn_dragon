#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.request
from typing import Any


def head_sort_key(head: dict[str, Any]) -> tuple[str, str]:
    return (str(head.get("created_at") or ""), str(head.get("head_id") or ""))


def entry_key(entry: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(entry.get("study_id") or ""),
        str(entry.get("experiment_id") or ""),
        str(entry.get("current_revision_id") or entry.get("revision_id") or ""),
    )


def snapshot_directory_entries(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    directory = snapshot.get("directory")
    if not isinstance(directory, dict):
        return []
    entries = directory.get("entries")
    return entries if isinstance(entries, list) else []


def snapshot_heads(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    heads = snapshot.get("heads")
    return heads if isinstance(heads, list) else []


def recover_visible_root(
    entry: dict[str, Any],
    heads: list[dict[str, Any]],
) -> str | None:
    study_id, experiment_id, revision_id = entry_key(entry)
    candidates = [
        head
        for head in heads
        if str(head.get("study_id") or "") == study_id
        and str(head.get("experiment_id") or "") == experiment_id
        and str(head.get("revision_id") or "") == revision_id
        and head.get("parent_head_id") is None
        and head.get("head_id")
    ]
    if not candidates:
        return None
    return str(max(candidates, key=head_sort_key)["head_id"])


def preserve_current_heads(
    config: dict[str, Any],
    snapshot: dict[str, Any],
    recover_roots: bool,
) -> dict[str, int]:
    auth = config.get("auth")
    if not isinstance(auth, dict):
        return {"preserved": 0, "recovered": 0}
    directory_entries = auth.get("directory_entries")
    if not isinstance(directory_entries, list):
        return {"preserved": 0, "recovered": 0}

    current_by_key = {
        entry_key(entry): entry.get("current_head_id")
        for entry in snapshot_directory_entries(snapshot)
        if entry.get("current_head_id")
    }
    heads = snapshot_heads(snapshot)
    preserved = 0
    recovered = 0
    for entry in directory_entries:
        if not isinstance(entry, dict) or entry.get("current_head_id"):
            continue
        current_head_id = current_by_key.get(entry_key(entry))
        if current_head_id:
            entry["current_head_id"] = current_head_id
            preserved += 1
            continue
        if recover_roots:
            recovered_head_id = recover_visible_root(entry, heads)
            if recovered_head_id:
                entry["current_head_id"] = recovered_head_id
                recovered += 1
    return {"preserved": preserved, "recovered": recovered}


def load_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    if args.snapshot_file:
        return json.loads(pathlib.Path(args.snapshot_file).read_text())
    with urllib.request.urlopen(args.snapshot_url, timeout=args.timeout_secs) as response:
        return json.loads(response.read())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshot-url")
    source.add_argument("--snapshot-file")
    parser.add_argument("--timeout-secs", type=float, default=8.0)
    parser.add_argument("--recover-visible-root", action="store_true")
    args = parser.parse_args()

    config_path = pathlib.Path(args.config)
    config = json.loads(config_path.read_text())
    snapshot = load_snapshot(args)
    report = preserve_current_heads(config, snapshot, args.recover_visible_root)
    config_path.write_text(json.dumps(config, separators=(",", ":"), sort_keys=True))
    print(json.dumps(report, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
