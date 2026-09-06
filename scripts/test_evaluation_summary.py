import copy
import unittest

from scripts.experiments.evaluation_summary import validate_identities


class EvaluationSummaryTests(unittest.TestCase):
    def row(self):
        return dict(status="ok", checkpoint="run/a", epoch=4, model_fingerprint="weights",
                    corpus_fingerprint="training_grammar", panel_fingerprint="panel",
                    panel_options=dict(panel_seed=71, free_run_items=64, policy_items=16, difficulty_levels=1))

    def test_allows_separate_structural_and_training_grammar_panels(self):
        original = self.row()
        structural = dict(original, corpus_fingerprint="structural", panel_fingerprint="other_panel")
        validate_identities([original, structural, dict(status="process_failed")])

    def test_rejects_a_checkpoint_that_changes_across_evaluation_panels(self):
        original = self.row()
        changed = dict(original, model_fingerprint="different_weights", corpus_fingerprint="structural")
        with self.assertRaisesRegex(ValueError, "model identities"):
            validate_identities([original, changed])

    def test_rejects_mismatched_items_for_the_same_panel_request(self):
        original = self.row()
        changed = copy.deepcopy(original)
        changed.update(checkpoint="run/b", model_fingerprint="other_model", panel_fingerprint="other_items")
        with self.assertRaisesRegex(ValueError, "item identities"):
            validate_identities([original, changed])
