#!/usr/bin/env python3

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: write_inspection_summary.py <artifact-dir>")

    artifact_dir = Path(sys.argv[1])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "instance_id": None,
        "ssm_status": None,
        "deployment_ready": None,
        "profile_source": None,
        "matching_head_present": None,
        "matching_head_id": None,
        "blocking_issues": [],
        "observed_warnings": [],
    }

    describe_path = artifact_dir / "instance-describe.json"
    if describe_path.exists():
        describe = json.loads(describe_path.read_text())
        reservations = describe.get("Reservations") or []
        if reservations and reservations[0].get("Instances"):
            summary["instance_id"] = reservations[0]["Instances"][0].get("InstanceId")

    diagnostics_path = artifact_dir / "bootstrap-deployment-diagnostics.json"
    if diagnostics_path.exists():
        diagnostics = json.loads(diagnostics_path.read_text())
        readiness = diagnostics.get("readiness") or {}
        edge_snapshot = (diagnostics.get("edge_snapshot") or {}).get("value") or {}
        profile_resolution = (diagnostics.get("profile_resolution") or {}).get("value") or {}
        summary["deployment_ready"] = readiness.get("ready")
        summary["profile_source"] = profile_resolution.get("source")
        summary["matching_head_present"] = edge_snapshot.get("matching_head_present")
        summary["matching_head_id"] = edge_snapshot.get("matching_head_id")
        summary["blocking_issues"] = readiness.get("blocking_issues") or []
        summary["observed_warnings"] = readiness.get("observed_warnings") or []

    ssm_status_path = artifact_dir / "ssm-status.txt"
    if ssm_status_path.exists():
        summary["ssm_status"] = ssm_status_path.read_text().strip()

    (artifact_dir / "inspection-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
