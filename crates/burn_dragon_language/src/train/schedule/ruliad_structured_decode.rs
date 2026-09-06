//! Verifier-backed rendering for typed Ruliad proof-policy decisions.

use super::*;

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct RuliadStructuredPolicyEvaluation {
    pub report: burn_dragon_universality::RuliadEvalReport,
    pub item_count: usize,
}

pub(super) struct RuliadStructuredPolicyArtifacts {
    pub(super) evaluation: RuliadStructuredPolicyEvaluation,
    pub(super) items: Vec<burn_dragon_universality::RuliadEvalItem>,
    pub(super) completions: Vec<burn_dragon_universality::RuliadCompletionRecord>,
}

fn structured_policy_completion(
    item: &burn_dragon_universality::RuliadEvalItem,
    actions: &burn_dragon_universality::ruliad::RuliadProofActionSet,
    selected_semantic_index: usize,
) -> Result<burn_dragon_universality::RuliadCompletionRecord> {
    let Some(burn_dragon_universality::RuliadSampleSpec::FormalProof {
        action_presentation_rotation,
        action_answer_contract,
        task: burn_dragon_universality::ruliad::RuliadTaskKind::SelectProofAction,
        ..
    }) = item.spec.as_ref()
    else {
        return Err(anyhow!(
            "structured proof-policy decoding requires a select-proof-action item"
        ));
    };
    let answer = burn_dragon_universality::ruliad::proof_action_answer_for_semantic_index(
        actions,
        selected_semantic_index,
        action_presentation_rotation.unwrap_or_default(),
        *action_answer_contract,
    )?;
    Ok(burn_dragon_universality::RuliadCompletionRecord {
        oracle_hash: item.oracle_hash.clone(),
        completion: format!("!:{answer}\n{}", item.document_close_marker()),
    })
}

pub(super) fn evaluate_ruliad_structured_policy_decisions(
    dataset_name: &str,
    items: &[burn_dragon_universality::RuliadEvalItem],
    jobs: &[RuliadCorrectnessConstrainedPolicyJob],
    decisions: &[crate::train::ruliad_policy::RuliadProofActionDecision],
) -> Result<RuliadStructuredPolicyArtifacts> {
    if items.len() != jobs.len() || jobs.len() != decisions.len() {
        return Err(anyhow!(
            "structured proof-policy decode alignment mismatch: items={} jobs={} decisions={}",
            items.len(),
            jobs.len(),
            decisions.len()
        ));
    }

    let completions = items
        .iter()
        .zip(jobs)
        .zip(decisions)
        .map(|((item, job), decision)| {
            let context = job
                .base_context
                .as_ref()
                .ok_or_else(|| anyhow!("structured proof-policy job has no canonical context"))?;
            structured_policy_completion(item, &context.actions, decision.selected_semantic_index)
        })
        .collect::<Result<Vec<_>>>()?;
    let items = items.to_vec();
    let report = burn_dragon_universality::evaluate_completions(
        format!("{dataset_name}_structured_policy_decode"),
        &items,
        &completions,
    );
    Ok(RuliadStructuredPolicyArtifacts {
        evaluation: RuliadStructuredPolicyEvaluation {
            item_count: items.len(),
            report,
        },
        items,
        completions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structured_decode_preserves_semantics_across_presentation_rotation() {
        use burn_dragon_universality::ruliad::{
            RuliadFamilyKind, RuliadProofActionAnswerContract, RuliadSampleSpec, RuliadTaskKind,
            formal::RuliadFormalGeneratorConfig, formal::generate_formal_bundle,
        };

        let bundle = generate_formal_bundle(91, RuliadFormalGeneratorConfig::default())
            .expect("formal bundle");
        let actions = burn_dragon_universality::ruliad::oracle_proof_action_set(
            &bundle.problem,
            &bundle.certificate,
            0,
            3,
        )
        .expect("proof actions");
        assert_eq!(actions.candidates.len(), 3);
        let rotation = 1;
        let presented = actions.rotate_left(rotation).expect("rotated actions");
        let expected = burn_dragon_universality::ruliad::proof_action_answer(
            &presented,
            presented.selected_index,
            RuliadProofActionAnswerContract::PresentationIndex,
        )
        .expect("expected answer");
        let item = burn_dragon_universality::RuliadEvalItem {
            oracle_hash: "structured-policy-test".into(),
            sample_index: 0,
            split: burn_dragon_universality::SampleSplit::Validation,
            family: RuliadFamilyKind::FormalProof.label().into(),
            task_kind: RuliadTaskKind::SelectProofAction.label().into(),
            math_domains: Vec::new(),
            reasoning_modes: Vec::new(),
            prompt: "!:".into(),
            expected_answer: expected,
            difficulty_level: Some(0),
            spec: Some(RuliadSampleSpec::FormalProof {
                problem: bundle.problem,
                certificate: bundle.certificate,
                candidate: None,
                proof_step_index: Some(0),
                action_presentation_rotation: Some(rotation),
                action_candidate_count: Some(actions.candidates.len()),
                action_answer_contract: RuliadProofActionAnswerContract::PresentationIndex,
                task: RuliadTaskKind::SelectProofAction,
            }),
        };

        let completion = structured_policy_completion(&item, &actions, actions.selected_index)
            .expect("structured completion");
        let report = burn_dragon_universality::evaluate_completions(
            "structured-policy-test",
            std::slice::from_ref(&item),
            &[completion],
        );
        assert_eq!(report.verifier_match_count, 1);

        let wrong_index = actions
            .candidates
            .iter()
            .enumerate()
            .find_map(|(index, _)| (!actions.is_equivalent_index(index)).then_some(index))
            .expect("non-equivalent action");
        let wrong = structured_policy_completion(&item, &actions, wrong_index)
            .expect("structured distractor");
        let wrong_report = burn_dragon_universality::evaluate_completions(
            "structured-policy-test-wrong",
            &[item],
            &[wrong],
        );
        assert_eq!(wrong_report.verifier_match_count, 0);
    }
}
