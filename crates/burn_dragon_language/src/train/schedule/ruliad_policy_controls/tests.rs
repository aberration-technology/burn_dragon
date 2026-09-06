use super::*;
use crate::train::ruliad_policy::{
    EncodedRuliadProofActionPresentation, SemanticActionOrbitSummary,
};
use burn_dragon_universality::ruliad::{
    RuliadProofActionAnswerContract, RuliadTaskKind,
    formal::{RuliadFormalGeneratorConfig, generate_formal_bundle},
};

fn fixture() -> (
    burn_dragon_universality::RuliadEvalItem,
    RuliadPolicyActionPromptContext,
) {
    let bundle = generate_formal_bundle(91, RuliadFormalGeneratorConfig::default()).unwrap();
    let actions = burn_dragon_universality::ruliad::oracle_proof_action_set(
        &bundle.problem,
        &bundle.certificate,
        0,
        4,
    )
    .unwrap();
    let item = burn_dragon_universality::RuliadEvalItem {
        oracle_hash: "control-test".into(),
        sample_index: 0,
        split: burn_dragon_universality::SampleSplit::Validation,
        family: "formal_proof".into(),
        task_kind: RuliadTaskKind::SelectProofAction.label().into(),
        math_domains: Vec::new(),
        reasoning_modes: Vec::new(),
        prompt: "!:".into(),
        expected_answer: String::new(),
        difficulty_level: Some(0),
        spec: Some(burn_dragon_universality::RuliadSampleSpec::FormalProof {
            problem: bundle.problem.clone(),
            certificate: bundle.certificate,
            candidate: None,
            proof_step_index: Some(0),
            action_presentation_rotation: Some(0),
            action_candidate_count: Some(actions.candidates.len()),
            action_answer_contract: RuliadProofActionAnswerContract::SemanticStep,
            task: RuliadTaskKind::SelectProofAction,
        }),
    };
    (
        item,
        RuliadPolicyActionPromptContext {
            problem: bundle.problem,
            actions,
        },
    )
}

#[test]
fn controls_replay_and_reject_cache_or_label_tampering() {
    let (item, context) = fixture();
    let distances = audit_transitions(&item, &context).unwrap();
    assert_eq!(distances.len(), context.actions.candidates.len());
    for corruption in 0..4 {
        let mut changed = context.clone();
        match corruption {
            0 => changed.actions.candidates[0].distance_to_goal = Some(usize::MAX),
            1 => changed.actions.candidates[0].outcome = None,
            2 => changed
                .actions
                .equivalent_indices
                .push(changed.actions.selected_index),
            _ => changed.actions.target = changed.actions.current.clone(),
        }
        assert!(
            audit_transitions(&item, &changed).is_err(),
            "corruption {corruption}"
        );
    }
}

#[test]
fn controls_use_equivalent_mass_and_unbiased_ties() {
    let (_, mut context) = fixture();
    context.actions.equivalent_indices = vec![0, 1];
    assert_eq!(expected_accuracy(&[0, 1, 2, 3], &context.actions), 0.5);
    assert_eq!(expected_accuracy(&[1, 2], &context.actions), 0.5);
    assert_eq!(minimum_indices(&[7, 2, 2, 9]), vec![1, 2]);
    assert!(minimum_indices(&[]).is_empty());
}

#[test]
fn controls_remove_every_prompt_without_changing_actions_or_orbit() {
    let request = EncodedRuliadProofActionRequest {
        answer_contract: RuliadProofActionAnswerContract::SemanticStep,
        presentations: (0..3)
            .map(|rotation| EncodedRuliadProofActionPresentation {
                rotation,
                original_prompt_token_count: 900,
                prompt_tokens: vec![23, 91, 33],
                candidate_tokens: vec![vec![7], vec![8], vec![9]],
            })
            .collect(),
    };
    let changed = replace_action_context(&request, &[1]);
    for (before, after) in request.presentations.iter().zip(&changed.presentations) {
        assert_eq!(after.prompt_tokens, vec![1]);
        assert_eq!(after.original_prompt_token_count, 1);
        assert_eq!(before.rotation, after.rotation);
        assert_eq!(before.candidate_tokens, after.candidate_tokens);
    }
    assert_eq!(request.presentations[0].prompt_tokens, vec![23, 91, 33]);
    assert_eq!(request.answer_contract, changed.answer_contract);
}

#[test]
fn controls_preserve_replayed_heuristic_and_chance_under_rotation() {
    let (item, context) = fixture();
    let distances = audit_transitions(&item, &context).unwrap();
    let expected = expected_accuracy(&minimum_indices(&distances), &context.actions);
    for rotation in 0..context.actions.candidates.len() {
        let rotated = RuliadPolicyActionPromptContext {
            problem: context.problem.clone(),
            actions: context.actions.rotate_left(rotation).unwrap(),
        };
        let distances = audit_transitions(&item, &rotated).unwrap();
        assert_eq!(
            expected_accuracy(&minimum_indices(&distances), &rotated.actions),
            expected
        );
    }
}

#[test]
fn controls_report_paired_outcomes_and_reject_misalignment() {
    let (item, context) = fixture();
    let count = context.actions.candidates.len();
    let decision = RuliadProofActionDecision {
        selected_semantic_index: context.actions.selected_index,
        selected_completion_tokens: Vec::new(),
        orbit: SemanticActionOrbitSummary {
            averaged_log_probs: vec![-(count as f32).ln(); count],
            presentation_log_probs: Vec::new(),
            js_divergence: 0.0,
            top1_consensus_fraction: 1.0,
            complete_cyclic_orbit: false,
        },
    };
    let mut no_context = decision.clone();
    no_context.selected_semantic_index = (0..count)
        .find(|index| !context.actions.is_equivalent_index(*index))
        .unwrap();
    let job = RuliadCorrectnessConstrainedPolicyJob {
        difficulty_level: 0,
        source_label: "test".into(),
        presentations: vec![RuliadPolicyActionPresentation {
            rotation: 0,
            prompt_tokens: vec![1],
            candidate_tokens: vec![vec![2]; count],
            answer_contract: RuliadProofActionAnswerContract::SemanticStep,
        }],
        prompt_contexts: vec![context.clone()],
        selected_index: context.actions.selected_index,
        equivalent_indices: context.actions.equivalent_indices.clone(),
        base_context: Some(context),
    };
    assert!(evaluate_ruliad_policy_controls(&[], &[], &[], &[decision.clone()]).is_err());
    let report =
        evaluate_ruliad_policy_controls(&[item], &[job], &[decision], &[no_context]).unwrap();
    assert_eq!(report.summary.model_accuracy, 1.0);
    assert_eq!(report.summary.no_context_accuracy, 0.0);
    assert_eq!(report.summary.context_helped_items, 1);
    assert_eq!(report.summary.context_harmed_items, 0);
    assert_eq!(report.summary.context_equivalent_probability_gain, 0.0);
    assert_eq!(report.summary, report.by_difficulty[&0]);
    assert_eq!(report.summary, report.by_source["test"]);
    assert_eq!(report.kernel_audited_candidates, count);
}

#[test]
fn empty_controls_are_explicit_and_finite() {
    let report = evaluate_ruliad_policy_controls(&[], &[], &[], &[]).unwrap();
    assert_eq!(report.summary, RuliadPolicyControlSummary::default());
    assert_eq!(report.kernel_audited_candidates, 0);
    assert!(
        serde_json::to_string(&report)
            .unwrap()
            .contains("reference_certificate_oracle_menu")
    );
}
