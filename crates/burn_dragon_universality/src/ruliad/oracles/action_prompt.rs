//! Proof-action observations. Exact observations omit target-derived focus hints.

use super::document::{
    bounded_transition_pattern, formal_proof_source_equality, transition_pattern_focus,
};
use super::*;

#[derive(Clone, Copy)]
pub(super) enum ActionPromptDetail {
    Focused,
    Exact,
}

pub(super) fn candidate_menu(
    problem: &RuliadProofProblem,
    actions: &crate::ruliad::policy::RuliadProofActionSet,
    detail: ActionPromptDetail,
) -> Result<String> {
    actions
        .candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| {
            let source = match &candidate.step.source {
                RuliadProofSource::Axiom { id } => format!("a:{id}"),
                RuliadProofSource::Lemma { goal } => format!("l:{goal}"),
            };
            let direction = match candidate.step.direction {
                RuliadRewriteDirection::Forward => "f",
                RuliadRewriteDirection::Reverse => "r",
            };
            let path = if candidate.step.path.is_empty() {
                "-".to_string()
            } else {
                candidate
                    .step
                    .path
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(".")
            };
            let (lhs, rhs) = formal_proof_source_equality(problem, &candidate.step.source)?;
            let (before, after) = match candidate.step.direction {
                RuliadRewriteDirection::Forward => (lhs, rhs),
                RuliadRewriteDirection::Reverse => (rhs, lhs),
            };
            let (before, after) = match detail {
                ActionPromptDetail::Focused => {
                    let (before, after) = transition_pattern_focus(before, after);
                    (
                        bounded_transition_pattern(before),
                        bounded_transition_pattern(after),
                    )
                }
                ActionPromptDetail::Exact => (before.canonical_text(), after.canonical_text()),
            };
            Ok(format!(
                "c{index}={source}|{direction}|{path}|{before}>{after}"
            ))
        })
        .collect::<Result<Vec<_>>>()
        .map(|candidates| candidates.join(","))
}

/// Complete current/target terms and oriented candidate rules, without clipping or focus hints.
///
/// This is still a verifier-enumerated menu, not a deployable proposal mechanism. It exposes
/// neither reference-certificate labels nor cached candidate outcomes/distances. Callers must
/// preserve the full observation or reject it explicitly when a context budget is insufficient.
pub fn ruliad_proof_action_exact_prompt(
    problem: &RuliadProofProblem,
    actions: &crate::ruliad::policy::RuliadProofActionSet,
) -> Result<String> {
    Ok(format!(
        "?:select;g={};cur={};{};dst={}\n!:",
        actions.goal,
        actions.current.canonical_text(),
        candidate_menu(problem, actions, ActionPromptDetail::Exact)?,
        actions.target.canonical_text(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_observation_preserves_full_terms_and_rules_without_labels_or_hints() -> Result<()> {
        let bundle = generate_formal_bundle(
            61,
            RuliadFormalGeneratorConfig {
                rewrite_depth: 4,
                leaf_count: 8,
                context_depth: 3,
                distractor_axioms: 4,
                ..Default::default()
            },
        )?;
        let actions = crate::ruliad::policy::oracle_proof_action_set(
            &bundle.problem,
            &bundle.certificate,
            0,
            4,
        )?;
        let prompt = ruliad_proof_action_exact_prompt(&bundle.problem, &actions)?;
        assert!(prompt.contains(&format!(";cur={};", actions.current.canonical_text())));
        assert!(prompt.contains(&format!(";dst={}\n!:", actions.target.canonical_text())));
        assert!(!prompt.contains(";at="));
        assert!(!prompt.contains("[R3 "));
        assert!(!prompt.contains("..."));
        for candidate in &actions.candidates {
            let (lhs, rhs) = formal_proof_source_equality(&bundle.problem, &candidate.step.source)?;
            let (before, after) = match candidate.step.direction {
                RuliadRewriteDirection::Forward => (lhs, rhs),
                RuliadRewriteDirection::Reverse => (rhs, lhs),
            };
            assert!(prompt.contains(&format!(
                "|{}>{}",
                before.canonical_text(),
                after.canonical_text()
            )));
        }
        let mut labels_changed = actions.clone();
        labels_changed.selected_index = (actions.selected_index + 1) % actions.candidates.len();
        labels_changed.equivalent_indices = vec![labels_changed.selected_index];
        for candidate in &mut labels_changed.candidates {
            candidate.outcome = None;
            candidate.distance_to_goal = None;
        }
        assert_eq!(
            prompt,
            ruliad_proof_action_exact_prompt(&bundle.problem, &labels_changed)?
        );
        for rotation in 0..actions.candidates.len() {
            let rotated = actions.rotate_left(rotation)?;
            let text = ruliad_proof_action_exact_prompt(&bundle.problem, &rotated)?;
            assert!(text.contains(&format!(";cur={};", actions.current.canonical_text())));
            assert!(text.contains(&format!(";dst={}\n!:", actions.target.canonical_text())));
            assert!(text.contains(&candidate_menu(
                &bundle.problem,
                &rotated,
                ActionPromptDetail::Exact
            )?));
        }
        Ok(())
    }

    #[test]
    fn exact_observation_does_not_alias_states_hidden_by_focused_rendering() -> Result<()> {
        let bundle = generate_formal_bundle(91, RuliadFormalGeneratorConfig::default())?;
        let mut actions = crate::ruliad::policy::oracle_proof_action_set(
            &bundle.problem,
            &bundle.certificate,
            0,
            4,
        )?;
        let atom = |symbol: &str| RuliadTerm::Atom {
            symbol: symbol.into(),
        };
        let wrap = |first, second| RuliadTerm::Apply {
            operator: "pair".into(),
            arguments: vec![first, second],
        };
        actions.current = wrap(atom("x"), atom("a"));
        actions.target = wrap(atom("y"), atom("a"));
        let mut changed = actions.clone();
        changed.current = wrap(atom("x"), atom("b"));
        changed.target = wrap(atom("y"), atom("b"));
        assert_eq!(
            ruliad_proof_action_local_prompt(&bundle.problem, &actions)?,
            ruliad_proof_action_local_prompt(&bundle.problem, &changed)?,
        );
        assert_ne!(
            ruliad_proof_action_exact_prompt(&bundle.problem, &actions)?,
            ruliad_proof_action_exact_prompt(&bundle.problem, &changed)?,
        );
        Ok(())
    }
}
