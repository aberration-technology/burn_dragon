"""Summarize checkpoint matrices without pooling different evaluation contracts."""

import argparse
import json
from pathlib import Path

from scripts.ruliad_checkpoint_eval_analyze import validate_policy_controls
from .runner import file_identity, write_json


def validate_identities(rows):
    models, panels = {}, {}
    for row in rows:
        if row["status"] != "ok":
            continue
        checkpoint = (row["checkpoint"], row["epoch"])
        fingerprint = row["model_fingerprint"]
        if models.setdefault(checkpoint, fingerprint) != fingerprint:
            raise ValueError("one checkpoint produced different model identities across panels")
        options = row["panel_options"]
        panel = (row["corpus_fingerprint"], options["panel_seed"], options["free_run_items"],
                 options["policy_items"], options["difficulty_levels"])
        fingerprint = row["panel_fingerprint"]
        if panels.setdefault(panel, fingerprint) != fingerprint:
            raise ValueError("matched corpus/panel requests produced different item identities")


def summarize(root):
    results = json.loads((root / "results.json").read_text())
    rows = []
    for case in results["cases"]:
        if case["status"] != "ok":
            rows.append(dict(case))
            continue
        path = root / case["id"] / "evaluation.json"
        document = json.loads(path.read_text())
        validate_policy_controls(document, path)
        evaluation = document["evaluation"]
        override = document.get("corpus_override")
        if override and override["evaluation_corpus_fingerprint"] != document["corpus_semantic_fingerprint"]:
            raise ValueError("prepared corpus differs from the declared evaluation override")
        free = evaluation["free_run"]
        report = free["report"]
        teacher = free["teacher_forced"]
        controls = evaluation["policy_controls"]
        rows.append(dict(case, checkpoint=document["checkpoint"], epoch=document["checkpoint_epoch"],
                         corpus_fingerprint=document["corpus_semantic_fingerprint"],
                         corpus_override=override, model_fingerprint=evaluation["model_tensor_fingerprint_sha256"],
                         panel_fingerprint=evaluation["panel_fingerprint_sha256"], panel_options=document["options"],
                         report_identity=file_identity(path),
                         free_items=free["item_count"], free_verifier_accuracy=report["verifier_accuracy"],
                         answer_nll=teacher["mean_nll"], answer_token_accuracy=teacher["token_accuracy"],
                         answer_sequence_accuracy=teacher["sequence_accuracy"],
                         context_binding_nll_gain=teacher["context_binding_nll_gain"],
                         termination_rate=report["answer_termination_rate"],
                         dominant_answer_fraction=report["actual_answer_dominant_fraction"],
                         mean_model_tokens=free["mean_generated_model_tokens"],
                         typed_policy=evaluation["constrained_policy"], policy_controls=controls))
    validate_identities(rows)
    return dict(complete=results["complete"], source_unchanged=results.get("source_unchanged"), cases=rows,
                caveats=["Do not pool training-grammar and structural/new-law holdouts.",
                         "One training seed and one evaluation panel do not establish seed robustness.",
                         "Typed policy accuracy uses an oracle action menu, unlike free generation."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = summarize(args.output)
    write_json(args.output / "evaluation-summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
