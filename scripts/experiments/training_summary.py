"""Summarize guarded training cases using their exact launch identities.

Usage: python3 -m scripts.experiments.training_summary MATRIX_OUTPUT
No learning quality is inferred from a successful process exit.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

from scripts.stateful_tbptt_analyze import METRICS as STATEFUL_METRICS, last_metric_any, read_jsonl
from .runner import file_identity, write_json

METRICS = {
    **STATEFUL_METRICS,
    "answer_nll": ("valid", ("Ruliad Teacher Forced NLL",)),
    "answer_sequence_nll": ("valid", ("Ruliad Teacher Forced Sequence NLL",)),
    "answer_token_accuracy": ("valid", ("Ruliad Teacher Forced Token Accuracy",)),
    "answer_sequence_accuracy": ("valid", ("Ruliad Teacher Forced Sequence Accuracy",)),
    "answer_context_binding_nll_gain": ("valid", ("Ruliad Teacher Forced Context-Binding NLL Gain",)),
}


def read_json(path):
    return json.loads(path.read_text())


def find_launch(case_manifest, run_manifests):
    matches = []
    for path, manifest in run_manifests:
        for launch in manifest["launches"]:
            if launch["command"] == case_manifest["execution_argv"]:
                matches.append((path.parent, manifest, launch))
    if len(matches) != 1:
        raise ValueError(f"expected one exact training launch, found {len(matches)}")
    return matches[0]


def epoch_counter_total(events, name):
    epochs = {}
    for event in events:
        if event.get("type") == "metric" and event.get("split") == "train" and event.get("name") == name:
            epochs[event["epoch"]] = event["value"]
    return sum(epochs.values()) if epochs else None


def launch_configuration(run, launch):
    """Read the immutable launch snapshot, not a later resume's current config."""
    path = run / launch["config_snapshot"]
    snapshot = read_json(path)
    training = snapshot["training"]["training"]
    attention = snapshot["model"]["fused_kernels"]
    return training, dict(config_identity=file_identity(path),
                          rotary_embedding=attention["rotary_embedding"],
                          alibi_slopes=attention["alibi_slopes"],
                          supervision_mode=training["ruliad_supervision"]["mode"],
                          tbptt_chunk_size=training["tbptt_chunk_size"],
                          tbptt_credit_window_chunks=training["tbptt_credit_window_chunks"],
                          tbptt_persist_across_steps=training["tbptt_persist_across_steps"])


def completion_trajectory(records):
    """Keep difficulty strata and model-token costs separate from lexical counts."""
    panels = defaultdict(list)
    for record in records:
        panels[(record["epoch"], record["absolute_step"], record["probe_name"])].append(record)

    def summarize_rows(rows):
        lengths = [row["generated_model_token_count"] for row in rows
                   if row.get("generated_model_token_count") is not None]
        answers = Counter(row["actual_answer"] for row in rows)
        return dict(items=len(rows),
                    verified=sum(row.get("verifier_match") is True for row in rows),
                    semantic=sum(row.get("semantic_match") is True for row in rows),
                    malformed=sum(row.get("status") == "Malformed" for row in rows),
                    terminated=sum(row.get("answer_terminated") is True for row in rows),
                    budget_hits=sum(row.get("generation_hit_budget") is True for row in rows),
                    dominant_answer_fraction=max(answers.values()) / len(rows),
                    mean_model_tokens=statistics.fmean(lengths) if lengths else None,
                    model_token_count_items=len(lengths))

    result = []
    for (epoch, step, probe), rows in sorted(panels.items()):
        strata = defaultdict(list)
        for row in rows:
            strata[row["difficulty_level"]].append(row)
        result.append(dict(epoch=epoch, absolute_step=step, probe_name=probe,
                           aggregate=summarize_rows(rows),
                           difficulties={str(level): summarize_rows(items)
                                         for level, items in sorted(strata.items())}))
    return result


def summarize(root, runs):
    result = read_json(root / "results.json")
    run_manifests = [(path, read_json(path)) for path in sorted(runs.glob("*/experiment_manifest.json"))]
    cases = []
    for case in result["cases"]:
        if case["status"] != "ok":
            cases.append(dict(case))
            continue
        directory = root / case["id"]
        run, manifest, launch = find_launch(read_json(directory / "manifest.json"), run_manifests)
        events_path = run / "events/training_events.jsonl"
        events = read_jsonl(events_path)
        training, contract = launch_configuration(run, launch)
        gpu = read_json(directory / "gpu.json")["samples"]
        memory = read_jsonl(directory / "memory.jsonl")
        row = dict(case, run_dir=str(run.resolve()), initial_model_sha256=launch.get("initial_model_sha256"),
                   next_latent_contract=launch.get("next_latent_objective_contract_version"),
                   model=manifest["model_spec"], batch_size=training["batch_size"],
                   block_size=training["block_size"], planned_updates=training["max_iters"],
                   launch_contract=contract,
                   events_identity=file_identity(events_path))
        row["metrics"] = {name: last_metric_any(events, split, names)
                          for name, (split, names) in METRICS.items()}
        row["stream_validation_contract"] = last_metric_any(events, "valid", ("Stream Validation Contract Version",)) or 1
        row["source_supervised_tokens"] = epoch_counter_total(events, "Epoch Source Supervised Tokens")
        row["source_scheduled_tokens"] = epoch_counter_total(events, "Epoch Scheduled Tokens")
        row["unknown_supervision_batches"] = epoch_counter_total(events, "Epoch Unknown Supervision Batches")
        row["source_supervised_batches"] = epoch_counter_total(events, "Epoch Source Supervised Batches")
        row["zero_supervision_batches"] = epoch_counter_total(events, "Epoch Zero Supervision Batches")
        completions = run / "events/ruliad_completion_samples.jsonl"
        row["completion_trajectory"] = completion_trajectory(read_jsonl(completions)) if completions.exists() else []
        metric_steps = [item["absolute_step"] for item in events if item.get("type") == "metric" and item.get("split") == "train"]
        row["last_logged_training_step"] = max(metric_steps, default=-1) + 1
        for key in ("util_percent", "power_w"):
            values = [item[key] for item in gpu if item.get(key) is not None]
            row[f"sampled_gpu_{key}_mean"] = statistics.fmean(values) if values else None
            row[f"sampled_gpu_{key}_max"] = max(values) if values else None
        row["sampled_host_used_mib_max"] = max(
            (item["total_mib"] - item["available_mib"] for item in memory), default=None,
        )
        cases.append(row)
    identities = [case.get("initial_model_sha256") for case in cases]
    return dict(complete=result["complete"], source_unchanged=result.get("source_unchanged"),
                same_initial_weights=bool(identities) and None not in identities and len(set(identities)) == 1,
                cases=cases,
                caveats=["GPU samples include startup, training, validation and checkpointing.",
                         "Train loss may include auxiliary terms. Validation CE follows each arm's supervision mask; use common answer-panel NLL across masks.",
                         "Null ALiBi slopes preserve historical defaults, not the reference ALiBi schedule.",
                         "Stream validation v1 is batch-averaged; v2 is token-weighted. Do not pool versions.",
                         "Source exposure does not count independent structured terminals or latent targets.",
                         "Scheduled steps include zero-supervision state advances; they are not necessarily parameter updates.",
                         "Repeated development panels are not independent confirmation or training seeds.",
                         "Kernel verification is not an independently implemented mathematical checker."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    args = parser.parse_args()
    summary = summarize(args.output, args.runs)
    write_json(args.output / "training-summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
