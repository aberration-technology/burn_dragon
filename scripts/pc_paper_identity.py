#!/usr/bin/env python3
"""Validate paired Ruliad source streams and realized verifier objectives."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import tempfile
from typing import Any


class IdentityFailure(RuntimeError):
    pass


OBJECTIVE_CONTRACT_FIELDS = (
    "proof_policy_mode",
    "proof_policy_scoring",
    "proof_policy_normalization",
    "proof_policy_target",
    "proof_policy_gradient_scope",
    "proof_policy_candidate_symmetry",
    "proof_policy_presentation_risk",
    "proof_policy_counterfactual_objective",
    "proof_policy_counterfactual_targets",
    "proof_policy_max_rows_per_update",
    "proof_policy_max_presentation_rows_per_update",
    "proof_policy_decoder_calibration_steps",
)


def stream_sha(rows: list[list[int]]) -> str:
    return hashlib.sha256(
        json.dumps(rows, separators=(",", ":")).encode()
    ).hexdigest()


def first_panel_divergence(
    reference: list[list[int]], candidate: list[list[int]]
) -> dict[str, Any] | None:
    for left, right in zip(reference, candidate):
        if left != right:
            return {"reference": left, "candidate": right}
    if len(reference) != len(candidate):
        return {
            "reference_length": len(reference),
            "candidate_length": len(candidate),
        }
    return None


def input_group_key(manifest: dict[str, Any]) -> tuple[Any, ...]:
    return (
        manifest.get("seed"),
        manifest.get("iters"),
        manifest.get("batch_size"),
        manifest.get("block_size"),
        manifest.get("verifier_every_steps"),
        manifest.get("proof_policy_start_after_steps", 0),
        manifest.get("ruliad_dagger_start_after_steps"),
        manifest.get("proof_policy_prompt_context"),
        manifest.get("profile"),
        manifest.get("source_selection_feedback_updates_enabled"),
        manifest.get("ruliad_source_selection_cold_start_enabled"),
        manifest.get("ruliad_source_selection_documents_per_step"),
    )


def load_groups(root: pathlib.Path) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for manifest_path in sorted((root / "manifests").glob("*.json")):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "ok" or "verifier" not in str(
            manifest.get("arm", "")
        ):
            continue
        run_dir = pathlib.Path(manifest.get("run_dir", ""))
        telemetry = run_dir / "events" / "ruliad_proof_policy_dagger.jsonl"
        audit_path = run_dir / "ruliad_supervision_audit.json"
        experiment_manifest_path = run_dir / "experiment_manifest.json"
        if (
            not telemetry.is_file()
            or not audit_path.is_file()
            or not experiment_manifest_path.is_file()
        ):
            raise IdentityFailure(
                "paired identity lacks telemetry, supervision audit, or experiment "
                f"manifest: manifest={manifest_path}"
            )
        events = [
            json.loads(line) for line in telemetry.read_text().splitlines() if line.strip()
        ]
        if not events:
            raise IdentityFailure(f"proof-policy telemetry is empty: {telemetry}")
        missing = [
            event.get("step_index")
            for event in events
            if event.get("policy_batch_fingerprint") is None
            or event.get("objective_panel_fingerprint") is None
            or int(event.get("objective_panel_fingerprint", 0)) == 0
        ]
        if missing:
            raise IdentityFailure(
                f"proof-policy identity fields are missing at steps={missing}: {telemetry}"
            )
        input_rows = [
            [int(event["step_index"]), int(event["policy_batch_fingerprint"])]
            for event in events
        ]
        panel_rows = [
            [int(event["step_index"]), int(event["objective_panel_fingerprint"])]
            for event in events
        ]
        audit = json.loads(audit_path.read_text())
        experiment_manifest = json.loads(experiment_manifest_path.read_text())
        launches = experiment_manifest.get("launches", [])
        if not launches:
            raise IdentityFailure(
                f"experiment manifest has no launch identity: {experiment_manifest_path}"
            )
        initial_model_sha256 = launches[0].get("initial_model_sha256")
        initial_model_schema = launches[0].get(
            "initial_model_tensor_fingerprint_schema"
        )
        if not initial_model_sha256 or not initial_model_schema:
            raise IdentityFailure(
                "paired experiment lacks an initial-model tensor fingerprint: "
                f"{experiment_manifest_path}"
            )
        groups[input_group_key(manifest)].append(
            {
                "arm": manifest.get("arm"),
                "initial_model_sha256": initial_model_sha256,
                "initial_model_schema": initial_model_schema,
                "input_sha256": stream_sha(input_rows),
                "panel_sha256": stream_sha(panel_rows),
                "panel_rows": panel_rows,
                "audit_sha256": audit.get("fingerprint_sha256"),
                "on_policy": any(
                    int(event.get("model_scoring_batches", 0)) > 0 for event in events
                ),
                "objective_contract": {
                    name: manifest.get(name) for name in OBJECTIVE_CONTRACT_FIELDS
                },
            }
        )
    return groups


def validate(root: pathlib.Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": 1,
        "input_compared_groups": 0,
        "input_identity": True,
        "initial_model_compared_groups": 0,
        "initial_model_identity": True,
        "static_panel_compared_groups": 0,
        "static_panel_identity": True,
        "on_policy_panel_compared_groups": 0,
        "on_policy_panel_identical_groups": 0,
        "on_policy_panel_divergent_groups": 0,
        "groups": [],
        "failures": [],
    }
    for key, rows in load_groups(root).items():
        if len(rows) < 2:
            continue
        report["input_compared_groups"] += 1
        report["initial_model_compared_groups"] += 1
        initial_hashes = {row["initial_model_sha256"] for row in rows}
        initial_schemas = {row["initial_model_schema"] for row in rows}
        if len(initial_hashes) != 1 or len(initial_schemas) != 1:
            report["initial_model_identity"] = False
            report["failures"].append(
                {
                    "kind": "initial_model_identity",
                    "key": list(key),
                    "rows": [
                        {
                            "arm": row["arm"],
                            "schema": row["initial_model_schema"],
                            "sha256": row["initial_model_sha256"],
                        }
                        for row in rows
                    ],
                }
            )
            continue
        input_hashes = {row["input_sha256"] for row in rows}
        audit_hashes = {row["audit_sha256"] for row in rows}
        if len(input_hashes) != 1 or len(audit_hashes) != 1:
            report["input_identity"] = False
            report["failures"].append(
                {
                    "kind": "input_identity",
                    "key": list(key),
                    "rows": [
                        {
                            "arm": row["arm"],
                            "input_sha256": row["input_sha256"],
                            "audit_sha256": row["audit_sha256"],
                        }
                        for row in rows
                    ],
                }
            )
            continue
        objective_contracts = {
            json.dumps(row["objective_contract"], sort_keys=True, separators=(",", ":"))
            for row in rows
        }
        if len(objective_contracts) != 1:
            report["failures"].append(
                {
                    "kind": "objective_contract",
                    "key": list(key),
                    "rows": [
                        {"arm": row["arm"], "contract": row["objective_contract"]}
                        for row in rows
                    ],
                }
            )
            continue
        on_policy_flags = {row["on_policy"] for row in rows}
        if len(on_policy_flags) != 1:
            report["failures"].append(
                {
                    "kind": "dagger_execution",
                    "key": list(key),
                    "rows": [
                        {"arm": row["arm"], "on_policy": row["on_policy"]}
                        for row in rows
                    ],
                }
            )
            continue
        panel_hashes = {row["panel_sha256"] for row in rows}
        group_report: dict[str, Any] = {
            "key": list(key),
            "on_policy": next(iter(on_policy_flags)),
            "initial_model_schema": next(iter(initial_schemas)),
            "initial_model_sha256": next(iter(initial_hashes)),
            "input_sha256": next(iter(input_hashes)),
            "audit_sha256": next(iter(audit_hashes)),
            "objective_contract": rows[0]["objective_contract"],
            "panels": [
                {"arm": row["arm"], "sha256": row["panel_sha256"]} for row in rows
            ],
            "panel_identity": len(panel_hashes) == 1,
            "first_panel_divergences": [],
        }
        reference = rows[0]
        for candidate in rows[1:]:
            divergence = first_panel_divergence(
                reference["panel_rows"], candidate["panel_rows"]
            )
            if divergence is not None:
                group_report["first_panel_divergences"].append(
                    {
                        "reference_arm": reference["arm"],
                        "candidate_arm": candidate["arm"],
                        **divergence,
                    }
                )
        if group_report["on_policy"]:
            report["on_policy_panel_compared_groups"] += 1
            counter = (
                "on_policy_panel_identical_groups"
                if group_report["panel_identity"]
                else "on_policy_panel_divergent_groups"
            )
            report[counter] += 1
        else:
            report["static_panel_compared_groups"] += 1
            if not group_report["panel_identity"]:
                report["static_panel_identity"] = False
                report["failures"].append(
                    {
                        "kind": "static_panel_identity",
                        "key": list(key),
                        "divergences": group_report["first_panel_divergences"],
                    }
                )
        report["groups"].append(group_report)

    output = root / "matrix-identity.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["failures"]:
        raise IdentityFailure(
            "proof-policy identity failed: "
            + json.dumps(report["failures"], separators=(",", ":"))
        )
    return report


def write_fixture(root: pathlib.Path, panel_b: int, on_policy: bool) -> None:
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    for arm, panel in (("adamw_verifier", 31), ("pc_verifier", panel_b)):
        run_dir = root / f"run-{arm}"
        (run_dir / "events").mkdir(parents=True, exist_ok=True)
        events = [
            {
                "step_index": step,
                "policy_batch_fingerprint": 11 + step,
                "objective_panel_fingerprint": panel + step,
                "model_scoring_batches": int(on_policy),
            }
            for step in range(2)
        ]
        (run_dir / "events" / "ruliad_proof_policy_dagger.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in events)
        )
        (run_dir / "ruliad_supervision_audit.json").write_text(
            json.dumps({"fingerprint_sha256": "audit"})
        )
        (run_dir / "experiment_manifest.json").write_text(
            json.dumps(
                {
                    "launches": [
                        {
                            "initial_model_tensor_fingerprint_schema": "test-v1",
                            "initial_model_sha256": "initial",
                        }
                    ]
                }
            )
        )
        manifest = {
            "status": "ok",
            "arm": arm,
            "run_dir": str(run_dir),
            "seed": 7,
            "iters": 2,
            "batch_size": 1,
            "block_size": 64,
            "verifier_every_steps": 1,
            "proof_policy_start_after_steps": 0,
            "ruliad_dagger_start_after_steps": 8,
            "proof_policy_prompt_context": "local_action_state",
            "profile": "fixture.toml",
            "proof_policy_mode": "static_then_paired_dagger",
            "proof_policy_scoring": "residual_energy",
        }
        (root / "manifests" / f"{arm}.json").write_text(json.dumps(manifest))


def self_test() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        write_fixture(root, panel_b=31, on_policy=False)
        static = validate(root)
        assert static["static_panel_identity"]
        changed_manifest = root / "run-pc_verifier" / "experiment_manifest.json"
        changed = json.loads(changed_manifest.read_text())
        changed["launches"][0]["initial_model_sha256"] = "different"
        changed_manifest.write_text(json.dumps(changed))
        try:
            validate(root)
        except IdentityFailure:
            pass
        else:
            raise AssertionError("initial-model divergence must fail")
        write_fixture(root, panel_b=31, on_policy=False)
        write_fixture(root, panel_b=41, on_policy=False)
        try:
            validate(root)
        except IdentityFailure:
            pass
        else:
            raise AssertionError("static panel divergence must fail")
        write_fixture(root, panel_b=41, on_policy=True)
        dynamic = validate(root)
        assert dynamic["on_policy_panel_divergent_groups"] == 1
    print("self-test ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.root is None:
        parser.error("root is required unless --self-test is used")
    report = validate(args.root)
    if report["input_compared_groups"] == 0:
        print("proof-policy identity report: no paired groups")
    else:
        print(
            "proof-policy identity passed: "
            f"initial_models={report['initial_model_compared_groups']} "
            f"input_groups={report['input_compared_groups']} "
            f"static_panels={report['static_panel_compared_groups']} "
            f"on_policy_panels={report['on_policy_panel_compared_groups']} "
            f"on_policy_divergent={report['on_policy_panel_divergent_groups']}"
        )


if __name__ == "__main__":
    main()
